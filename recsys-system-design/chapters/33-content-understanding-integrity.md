# 33. Content Understanding and Integrity in Recommender Systems: Multimodal Item Embeddings, Integrity Filters, Borderline Demotion, Spam and the Policy Layer

Most of this Atlas is about predicting what a person will engage with. This chapter covers the two layers that decide what an item *is* and whether it *should* be recommended at all.

1. **Content understanding.** Models turn raw pixels, text, audio and video into dense vectors or discrete codes. Those representations feed retrieval (cold start), ranking features, deduplication, and the integrity classifiers themselves.
2. **Integrity and policy.** Classifiers, rule engines and value-model terms remove violating content, demote borderline content, suppress spam and manipulation, and apply legal and product policy. These layers are kept explicit and separate from the engagement model.

The two layers share infrastructure. The same image or post embedding that places a new Pin near similar Pins also lets a safety classifier spot a near-duplicate of known violating media. The same semantic ID that lets a generative recommender decode a cold item is checked against visibility filtering before the item enters an index [13].

A note on sources. The live web was not reachable for this chapter's research pass. Facts are taken from primary sources that other chapters of this Atlas fetched and verified (Meta, Instagram, X/Twitter open-source code, YouTube, Pinterest, Kuaishou, Spotify, Netflix, Etsy, Google). A small number of well-known primary sources are cited from prior knowledge and are marked "(not re-fetched)" in the references. Numbers from those sources are given only where they are widely reported, and are marked approximate.

---

## 33.1 Where content understanding and integrity sit in the stack

Every public design in this Atlas keeps an explicit, non-learned layer for eligibility and policy alongside the learned funnel (Chapter 15). Google's "Rules of Machine Learning" recommends keeping heuristics and filtering outside the model when they express policy rather than prediction [40]. The composite picture below is drawn from Meta Feed [20], Instagram Explore [26], X's 2023 and 2026 open-source releases [13][16][17] and YouTube's public descriptions [30][33].

```
 ITEM INGEST (async, per upload)                         SERVING (per request)
 +-------------------------------------------+
 | raw item: image, video frames, audio,     |
 | text, OCR, ASR, author, metadata          |
 +--------------------+----------------------+
                      |
          +-----------v-----------+
          | content encoders      |  CLIP-like image/text, video MLLM,
          | (multimodal)          |  audio CNN, text LLM
          +-----+-----------+-----+
                |           |
     dense embedding     semantic ID (RQ codes)
                |           |
   +------------v--+   +----v---------------+     +-----------------------------+
   | integrity     |   | ANN / SID index    |---->| retrieval (cold + warm)     |
   | classifiers,  |   | (gated by labels)  |     +--------------+--------------+
   | LLM labelers, |   +--------------------+                    |
   | hash/near-dup |                                             v
   +------+--------+                                 eligibility pre-filter
          |  labels: violating / borderline /        (blocks, mutes, hard labels)
          |  spam / adult / quality score                        |
          v                                                      v
   +----------------+                               ranking (MTL heads incl.
   | label store    |------------------------------> negative-feedback heads)
   +----------------+                                            |
          |                                                      v
          +-----------------------------------------> value model: weights,
                                                      demotion multipliers
                                                                 |
                                                                 v
                                                      re-rank + visibility filter
                                                      (drop / interstitial / label)
                                                                 |
                                                                 v
                                                             final page
```

Three properties recur across companies:

- **Label once, enforce many times.** Content understanding runs asynchronously at upload or first impression. Labels go to a store that retrieval, ranking and serving-time filters all read. X's 2026 release shows this explicitly: content-understanding services write labels to storage, and a separate visibility-filtering service reads them per (post, viewer) [13].
- **Enforce at several points.** Cheap eligibility checks run before the expensive models to save compute. Meta Feed runs "integrity processes" before the first ranking pass [20]. Stricter or costlier checks run again at the end, because blending and backfill can reintroduce items [13][26].
- **Prefer demotion to deletion for grey areas.** Content that violates policy is removed. Content near the line is kept but not amplified [29][30][31]. That distinction needs a soft lever (multipliers or value-model terms) as well as a hard filter.

---

## 33.2 Multimodal item embeddings

Content embeddings serve three purposes in recommendation:

1. **Cold start.** A new item can be scored before it has any interactions.
2. **Similarity and deduplication.** Near-duplicate detection, "more like this", and diversity kernels all need a content distance.
3. **Integrity reuse.** Safety classifiers can sit on top of general-purpose embeddings rather than each training its own backbone.

### 33.2.1 CLIP-style contrastive pretraining

**Design.** OpenAI's CLIP trains an image encoder and a text encoder jointly with a contrastive objective. Matching (image, caption) pairs in a batch are pulled together and all other pairs are pushed apart. The model was trained on roughly 400 million image-text pairs collected from the web, and it showed strong zero-shot transfer to many classification tasks (approximate; not re-fetched) [1].

**Why recommenders care.** A CLIP-like model gives one space for images and text. A text query, a product title and a photo can be compared directly. Classifiers can be bootstrapped from text prompts before labeled data exists. In recommendation stacks it most often appears as:

- a frozen feature extractor whose embedding is fed to rankers;
- a clustering basis, as in X's Home Mixer parameters, which list "CLIP image clusters and multimodal embeddings as features" [15];
- an input to safety services. X's 2026 release lists `clip/` and `pnsfwmedia/` (a predicted-NSFW media model) among the content-understanding services that write safety labels [13].

**Lesson.** A general vision-language embedding is a platform asset. Engagement ranking, near-duplicate detection and safety classification can all reuse it, which justifies centralizing its training and serving.

### 33.2.2 Pinterest: unified visual embeddings

**Problem.** Pinterest ran several visual products (Lens camera search, Flashlight in-image search, Shop the Look), each with its own specialized visual embedding. Maintaining many embeddings multiplies training, storage and backfill costs, and each improvement had to be repeated per product.

**Design.** Zhai et al. (KDD 2019) describe a single **unified visual embedding** trained with **multi-task metric learning**: one shared convolutional backbone, several task-specific training datasets, and proxy-based classification losses, so that one embedding serves all the visual search applications. The paper also reports binarizing the embedding to cut storage and serving cost for billions of images (details approximate; not re-fetched) [2].

```
  training sets (per product task)     shared CNN backbone      unified embedding
  +----------------------------+
  | Lens (camera-to-Pin)       |--+
  | Shop-the-Look (product)    |--+-->  image -> CNN -> FC -> emb --> proxy softmax
  | Flashlight (in-image crops)|--+                              per task (training)
  +----------------------------+                    |
                                                     v
                                     serving: one (binarized) embedding per image
                                     used by every visual search product
```

**Lessons.** (1) Multi-task training with one embedding beat maintaining specialized embeddings, operationally and in quality, according to the paper [2]. (2) Compression (binarization) is part of the design at billion-image scale, not an afterthought.

### 33.2.3 Pinterest: from PinSage to ItemSage and OmniSearchSage

Pinterest's representation stack layers content on top of graph signals (Chapter 22).

- **PinSage (KDD 2018).** A GraphSAGE-style inductive GCN that aggregates **visual and text features** of each Pin with those of its random-walk neighbours. It was trained on 7.5 billion examples over a graph of 3 billion nodes and 18 billion edges [3]. Because the input is content features, new Pins can be embedded inductively. PinSage embeddings became Pinterest's shared coordinate system: PinnerSage clusters users in PinSage space [6], and PinnerFormer uses 256-dimensional PinSage embeddings as frozen per-action features [7].
- **ItemSage (KDD 2022).** Baltescu et al. describe product (shopping) embeddings that combine image and text features with a transformer, are trained on several engagement tasks at once, and are made compatible with PinSage space so that product and Pin embeddings can be compared across surfaces (details approximate; not re-fetched) [5].
- **OmniSearchSage (2024).** Multi-task, multi-entity embeddings for Pinterest Search: queries, Pins and products in one space trained on several tasks [4].

**Lesson.** The trend at Pinterest is from one embedding per product, to one embedding per entity type, to one shared multi-entity space. Content features keep that space usable for cold items, and graph signals keep it aligned with engagement.

### 33.2.4 Kuaishou OneRec: multimodal LLM item tokenizer

Kuaishou's OneRec (2025) replaces the retrieval-and-ranking cascade with one generative model, and its item representation is entirely content-derived [8][9]:

- Inputs per video: **title, tags, ASR, OCR, cover image and sampled frames**.
- A multimodal LLM (**miniCPM-V-8B**) encodes them into **1,280 tokens**. A **QFormer** compresses these into **4 query vectors**.
- An **item-pair contrastive loss** injects collaborative signal, with title generation as an auxiliary task.
- **RQ-KMeans** over 3 residual levels turns each video into a **3-token semantic ID** (about 8K codes per level, approximate).

```
  title, tags, ASR, OCR, cover, frames
            |
     miniCPM-V-8B  ->  1,280 tokens  -> QFormer -> 4 vectors
            |                                   |
   item-pair contrastive (CF signal)     RQ-KMeans, 3 levels
   + title generation (aux)                     |
                                       semantic ID (c1, c2, c3)
                                                |
                         generative recommender decodes SIDs per session
```

**Lesson.** Once the generator only emits item codes, the **tokenizer is the content-understanding system**. Its quality, collision rate and refresh policy determine cold-start behaviour. It is also the natural place to attach integrity labels: an item without a vetted code cannot be generated.

### 33.2.5 YouTube and Google: semantic IDs from content

- **TIGER (NeurIPS 2023)** quantizes content embeddings into hierarchical semantic IDs with an RQ-VAE, and a seq2seq transformer decodes the next item's IDs. Similar items share prefixes, which helps cold items [12].
- **Semantic IDs in ranking (RecSys 2024)** showed that replacing random item-ID hashing with content-derived semantic IDs improves generalization at a fixed embedding-table size [11].
- **PLUM (2025)** builds "Semantic IDs v2" from **multimodal content embeddings plus a co-engagement signal**, continues pre-training a Gemini-family LLM on watch sequences written as SIDs, and serves generative retrieval as a source feeding YouTube's ranking stack (approximate) [10].

### 33.2.6 X Phoenix: semantic IDs over multimodal post embeddings

X's 2026 Phoenix retrieval model represents posts by **semantic IDs: residual-quantized codes, 6 levels x 256 codes, derived from each post's multimodal embedding**, plus hashed author IDs. Posts on the same topic share SID prefixes, which helps the model generalize to unseen posts [14]. The production index holds 10.24M posts (28.67M in the combined configuration) [14]. Integrity is coupled to indexing: the `phoenix-rankall-strato` service decides which index a post belongs in by "consulting visibility filtering first" [13].

### 33.2.7 Audio and text: Spotify and Xiaohongshu

- **Spotify audio CNN.** A convolutional network over mel-spectrograms of 30-second clips is trained to **regress the collaborative-filtering latent vector** of tracks that have enough plays. A brand-new track is then embedded where the CF model would have placed it, in the same ANN index [36][37]. This "regress into the CF space" trick generalizes to any modality.
- **NoteLLM (Xiaohongshu, WWW 2024)** compresses each note into a special token from an LLM's last layer. That embedding is trained contrastively on co-occurrence pairs for item-to-item retrieval, with hashtag and category generation as auxiliary tasks [38].

### 33.2.8 Comparison of content-embedding designs

| System | Year | Modalities | Training signal | Output | Main use in recs | Ref |
|---|---|---|---|---|---|---|
| CLIP | 2021 | image + text | image-caption contrastive (~400M pairs, approx.) | dense vectors | features, clusters, zero-shot labels | [1] |
| Pinterest unified visual | 2019 | image | multi-task proxy metric learning | one (binarized) vector | all visual search products | [2] |
| PinSage | 2018 | image + text + graph | co-engagement, hard negatives | 256-d (as used downstream) | retrieval, user models | [3][7] |
| ItemSage | 2022 | image + text (products) | multi-task engagement | vector compatible with PinSage | shopping surfaces | [5] |
| OmniSearchSage | 2024 | query, Pin, product | multi-task, multi-entity | shared vector space | search retrieval | [4] |
| Spotify audio CNN | 2014 | audio | regress CF factors | vector in CF space | cold-start tracks | [36] |
| NoteLLM | 2024 | text (notes) | co-occurrence contrastive + generation | LLM token embedding | I2I retrieval | [38] |
| OneRec tokenizer | 2025 | video, image, text, ASR, OCR | MLLM + item-pair contrastive | 3-level semantic ID | generative rec | [8][9] |
| YouTube PLUM SIDs v2 | 2025 | multimodal + co-engagement | RQ-style quantization | multi-level SID | LLM generative retrieval | [10] |
| X Phoenix SIDs | 2026 | multimodal post embedding | RQ, 6 x 256 codes | SID + hashed author | retrieval and ranking | [14] |

---

## 33.3 Integrity filters inside the funnel

### 33.3.1 Meta Feed: integrity first

Meta's 2021 Feed ranking post describes the funnel as passes over an inventory of more than 1,000 eligible posts per viewer, run in parallel on serving machines called predictors [20]:

```
  inventory (> 1,000 posts)
     |
  pass 0: integrity processes + eligibility   <- violating / ineligible content removed
     |
  pass 1: lightweight model -> ~500 posts
     |
  pass 2: multitask NNs -> P(like), P(comment), P(hide), P("worth your time"), ...
          -> one value per post
     |
  pass 3: contextual re-rank (diversity, avoid runs of similar posts)
```

Running integrity before the heavy pass means the expensive models are only spent on content that can be shown [20]. Meta also publishes a "system card" for Feed that lists the main prediction signals and stages [23].

### 33.3.2 Instagram Explore: negative heads and final integrity filters

Instagram Explore shows three integrity mechanisms at different stages.

1. **Final re-ranking with integrity filters.** The 2019 funnel goes from about 500 candidates through three ranking passes to about 25 shown "after integrity and diversity rules" [24]. In the 2023 four-stage design, final re-ranking applies "integrity filters plus business rules" such as author-repetition penalties [26].
2. **Negative-feedback heads in the value model.** The 2020 design predicts "see less" and "not interested" alongside likes and saves, and combines them **log-linearly** [25]:

   ```
   Value(post) = P(like) ^ w_like * (1 - P(not interested)) ^ w_see_less
   ```

   In a multiplicative form, a high predicted negative probability can veto an item however likely a like is. The 2023 design uses a linear value model with a negative term `- W_see_less * P(see less)` [26].
3. **Guardrail-constrained tuning.** Value-model weights are tuned with Bayesian optimization in which engineers specify the parameters, the goal metric and regression thresholds on guardrail metrics [26].

### 33.3.3 X / Twitter: visibility filtering as a separate rule engine

X has published the most detailed integrity layer of any recommender. It is useful as a reference design.

**2023: `visibilitylib`.** The README calls it "a centralized rule engine that instructs clients how to alter the display of certain Twitter content on read time" [16]. Its model is:

- **SafetyLevel**: the product context (Timeline, Profile, and so on). Each SafetyLevel has one policy.
- **Features**: safety labels, user flags, user-user relationships, user-content relationships.
- **Conditions -> Rules -> Policy.** Conditions combine into rules. Rules are evaluated in priority order.
- **Actions**: **Drop** (hard filter), **Labels and Interstitials** (soft filter), and **coarse-grained downranking**.

Part of the code was withheld in 2023 as "being reviewed and rebuilt" [16].

**2026: the full labeling path.** The `xai-org/x-algorithm` release added the content-understanding and labeling services [13]:

```
 CONTENT UNDERSTANDING (async)          LABELING RULES              SERVE
 grox/  LLM + classifiers: spam,        scarecrow/ event rules,     visibility-filtering/
        adult, violent media,           botmaker rule engine        per (post, viewer):
        text/image embeddings           abuse-enforcement-svc/        ALLOW | NOTICE |
 media-model-proxy/, clip/,              (+ abuse-ledger appeals)     INTERSTITIAL | DROP
   pnsfwmedia/                          safety-label-user-agg/            |
 agatha/ (blocks+reports vs favs)                |                    VFFilter,
 bdsm/   (behaviour sequences)                   |                    AncillaryVFFilter
 user-cred-v2/ (PageRank credibility)            |
              \___________ labels -> storage ___/
```

Design points from the release [13]:

- **Ranking and visibility are separate systems** with "different services, different inputs, different rules". Ranking sets order; VF decides whether a post can be shown at all.
- **VF runs after selection.** `VFCandidateHydrator`, `VFFilter` and `AncillaryVFFilter` run after top-K is fixed. This avoids per-candidate VF calls on the whole pool. Cheap pre-scoring filters already handle blocks, mutes and muted keywords. `AncillaryVFFilter` also drops a post when its parent, quoted post or reposted post was dropped.
- **First DROP wins.** Evaluation ends at the first rule that answers DROP.
- **Stricter rules for out-of-network recommendations.** An additional set of out-of-network-only rules can only drop. High-recall spam classifiers suppress a post *as a recommendation* while still showing it to followers.
- **Withheld rules.** Some rules and the Grox LLM prompts are deliberately unpublished "to reduce the risk of gaming". An "Under the Hood" tool gives each account a report of the visibility-limiting labels on its account and posts, and since September 2026 also shows legal withholding by country.
- **Jurisdiction filters** appear in code, for example `Brazil2026ElectionFilter`.
- **Index gating.** As noted in 33.2.6, retrieval indexing consults VF before a post enters an index.

**Lessons.** (1) Model visibility as `(item, viewer, surface) -> action`, not as a property of the item alone. (2) Separate recall-oriented recommendation rules from precision-oriented removal rules: a false positive in "do not recommend" costs much less than a false positive in "remove". (3) Run expensive checks on the final top-K, and cheap ones early.

### 33.3.4 Health signals inside ranking and notifications

Integrity does not live only in filters. X's 2023 heavy ranker consumed realtime **pToxicity and pBlock** health-model scores as features [17]. The For You value model put large negative weights on reports, blocks and mutes, calibrated to base rates. The 2026 parameters note that a report is more than 1,000x rarer than a like (Chapter 21) [13].

Notifications use the same idea with stricter gating, because a push interrupts the user. X's `pushservice` predicate package includes `BqmlHealthModelPredicates`, `HealthPredicates`, `OutOfNetworkCandidatesQualityPredicates`, `JointDauAndQualityModelPredicate` and `PNegMultimodalPredicates` (a predicted-negative model), plus gates for withheld content and negative-feedback-based out-of-network filtering [19].

---

## 33.4 Borderline content demotion

### 33.4.1 The engagement curve argument

In his November 2018 note "A Blueprint for Content Governance and Enforcement", Mark Zuckerberg argued that, left alone, people engage more with content the closer it gets to the policy line, even when they later say they dislike it. Meta's stated response was to **reduce the distribution of borderline content**, so that engagement falls as content approaches the line instead of rising (approximate; not re-fetched) [29]. Meta's Transparency Center now documents the classes of content whose distribution it reduces as "Content Distribution Guidelines"; the exact page was not verified for this chapter.

This argument explains why a pure engagement optimizer drifts toward borderline content, and why a separate counter-term is needed. Reporting on Meta's 2018 **Meaningful Social Interactions (MSI)** value model gives an empirical example. MSI gave point values to likes, comments, re-shares, reactions and RSVPs, with weights derived from surveys (one cited survey covered more than 69,000 people in five countries) and data analysis. Heavy weights on comments, re-shares and reaction emojis were reported to amplify outrage-provoking content [22]. Meta later added "Is this post worth your time?" surveys and gave more weight to survey-predicted value [21].

### 33.4.2 YouTube: raters, borderline classifiers and authoritative boosts

YouTube's January 2019 announcement said it would begin reducing recommendations of borderline content: content that comes close to violating Community Guidelines, and content that could misinform users in harmful ways. It estimated such content at well under 1% of what is watched on YouTube (approximate; not re-fetched) [31]. The signals come from **human evaluators** trained on public guidelines, whose ratings train classifiers that score videos at scale [31][32].

In December 2019 YouTube reported that, after these changes, watch time of borderline content coming from recommendations to non-subscribed channels fell by about 70% on average in the US (approximate; not re-fetched) [32]. Cristos Goodrow's 2021 post repeats the roughly 70% figure and describes the companion lever: **raising authoritative sources** for news and information queries and recommendations [30]. The same post places this in the evolution of YouTube's objective: clicks (which encouraged clickbait), then watch time, then "valued watch time" predicted from user satisfaction surveys [30]. The "How YouTube Works" site presents the same system from the policy side [33].

```
  human raters (public guidelines)        survey responses (1-5 stars)
            |                                        |
   borderline / authoritativeness             satisfaction model
   classifiers (score every video)             (predict for every watch)
            |                                        |
            v                                        v
  candidate scoring:  value = f(engagement preds, satisfaction pred)
                      x demotion for borderline  x boost for authoritative
            |
   recommendation surfaces (Home, Up Next)  -- search and subscriptions treated separately
```

**Lessons.**

- **Borderline is a ranking problem, not a removal problem.** The item stays available, especially to subscribers who seek it out, but is not amplified to people who did not ask for it. This matches X's out-of-network-only rules [13].
- **Rater programs are infrastructure.** Guidelines, rater training, agreement measurement and classifier retraining form a pipeline that needs its own owners.
- **Pair demotion with promotion.** Demoting harmful content in a topic area leaves a vacuum that the ranker fills with whatever is next. Boosting authoritative sources fills it deliberately [30].

### 33.4.3 Remove, reduce, inform

Across Meta, YouTube and X, the public designs converge on three graded responses:

| Response | Typical trigger | Mechanism | Example implementations |
|---|---|---|---|
| Remove | clear policy violation | hard filter at ingest or serving; takedown | Meta pass-0 integrity [20]; X DROP action [13][16] |
| Reduce | borderline, low quality, spam-like, unoriginal | demotion multiplier, value-model penalty, out-of-network ineligibility | YouTube borderline demotion [30][32]; X OON-only rules [13]; Meta borderline reduction [29] |
| Inform / interstitial | sensitive but allowed; disputed | label, warning screen, context panel | X Labels and Interstitials, NOTICE / INTERSTITIAL [13][16] |
| Raise | authoritative or high-quality | boost multiplier | YouTube authoritative sources [30] |

---

## 33.5 Harmful-content classifiers that serve recommendation

### 33.5.1 Meta Few-Shot Learner

Policies change faster than labeled datasets can be collected. Meta's December 2021 post on its **Few-Shot Learner (FSL)** describes a large multilingual model pre-trained on very large amounts of general and integrity-related text, then adapted with zero or few labeled examples. Policies are expressed as natural-language descriptions, and classification is framed as entailment between the post and the policy text. Meta reported that it covered more than 100 languages, worked on text and images, and cut the time to deploy a new classifier from months to weeks (approximate; not re-fetched) [28].

**Why it matters for recsys.** Borderline and emerging-harm categories are exactly the ones where labels are scarce and policy definitions shift. A few-shot classifier lets the "reduce" lever track new policy quickly, without waiting for a supervised dataset.

### 33.5.2 Shared embeddings as integrity features

Twitter's TwHIN paper reports that the same heterogeneous-graph embeddings are reused for candidate generation, as heavy-ranker features, in ads, and for **harmful-content detection** [18]. Behavioural embeddings catch coordinated accounts and spam rings that content-only models miss, because those accounts look alike in who they interact with rather than in what they post.

### 33.5.3 LLM labelers

X's 2026 `grox/` service combines an LLM with classifiers for spam, adult content and violent media, using text and image embeddings, and writes labels consumed by visibility filtering [13]. The prompts are withheld to limit gaming [13]. The pattern of an LLM labeling offline, with classifiers or distilled models serving at scale, mirrors how LLMs enter recommendation stacks generally (Chapter 29): put the LLM on the asynchronous side of the cache line.

---

## 33.6 Spam, fraud and manipulation in recommendations

Recommenders create an incentive to manipulate whatever signal they reward. Public designs show defenses at several layers.

### 33.6.1 Reward design against clickbait

- **YouTube** moved from clicks to watch time in 2012 because clicks encouraged clickbait, and later to survey-predicted "valued watch time" [30]. The 2016 deep ranking model predicts expected watch time with weighted logistic regression for the same reason [34].
- **Netflix** artwork personalization rewards a **quality play** (a play followed by enough viewing), not a raw click, so a misleading thumbnail that draws abandoned plays is penalized [35].
- **Twitter/X** uses dwell-qualified "good click" labels and large negative weights on report, block and mute (Chapters 11 and 21) [17].

### 33.6.2 Credibility and behaviour signals

X's labeling path includes services aimed at manipulation rather than content [13]:

- `user-cred-v2/`: PageRank-based user credibility, a descendant of the earlier Tweepcred reputation score.
- `agatha/`: compares blocks and reports with favorites, a ratio that separates content people dislike from content that is merely unpopular.
- `bdsm/`: models behaviour sequences, which catches automated or coordinated activity.
- `safety-label-user-agg/`: aggregates post-level labels to the author level, so repeat offenders lose distribution across all their posts.

### 33.6.3 Quality-aware retrieval

Etsy blends a product-quality signal into ANN retrieval, so the nearest-neighbour stage does not surface low-quality listings that the ranker would only demote later [39]. X gates its retrieval index on visibility filtering for the same reason [13]. Filtering inside retrieval protects recall: if a selective policy filter runs only after ANN, most of the retrieved budget can be wasted on ineligible items (Chapter 2).

### 33.6.4 Platform eligibility rules

TikTok's 2020 explainer said that some content, such as spam, duplicated content and content that may be shocking to a general audience, may not be eligible for the For You feed, even when it remains on the platform (approximate; not re-fetched) [41]. This is the same "allowed but not recommended" tier as YouTube's borderline policy and X's out-of-network rules.

### 33.6.5 Exploration under adversaries

Integrity filters also bias the training data. X's notification stack includes `BigFilteringEpsilonGreedyExplorationPredicate`, which lets a small share of candidates through filters that would have blocked them, so the logs contain outcomes for filtered items [19]. Without such exploration, a filter's mistakes cannot be measured from the data it shapes. X's 2026 For You pipeline uses deterministic per-(post, viewer) holdout filters to measure inventory effects [13].

---

## 33.7 Responsible recommendation and the policy layer

### 33.7.1 Separate prediction from policy

The most consistent pattern in this Atlas is that engagement models predict calibrated probabilities of concrete actions, and a small, versioned **value function** turns them into a score (Chapter 4). Meta Feed [20], Instagram Explore [25][26] and X [13][17] all keep this split. For integrity it has three benefits:

1. **Auditable trade-offs.** A weight on `P(report)` or a borderline multiplier is a reviewable line of configuration.
2. **Fast policy changes.** New demotions ship as configuration, with no retraining.
3. **Clean attribution.** When harmful content rises, one can tell whether a predictor drifted or a policy weight changed.

The cost is calibration. A value model that adds `w_i * p_i` is only meaningful if each `p_i` is calibrated; a mis-calibrated `P(report)` silently changes policy (Chapter 4).

### 33.7.2 A reference policy layer

```
                 +-----------------------------------------------------------+
  scored         |  POLICY LAYER (config + rules, versioned, owned by policy) |
  candidates --> |                                                           |
  (p_engage,     |  1. hard eligibility: blocks, mutes, age gates, legal,    |
   p_negative,   |     jurisdiction (e.g. election filters)                  |
   labels)       |  2. value model: sum w_i * p_i, large negative weights    |
                 |     on report / block / see-less; survey-predicted value  |
                 |  3. demotions: borderline x a, low quality x b, OON spam  |
                 |     ineligible; boosts: authoritative x c                 |
                 |  4. diversity and fatigue: author decay, DPP              |
                 |  5. final visibility filter on top-K (+ ancillary items)  |
                 |  6. logging: every drop/demotion with reason, holdouts,   |
                 |     exploration share, per-account transparency report    |
                 +-----------------------------+-----------------------------+
                                               |
                                               v
                                          final page
```

Every stage maps to a public example: jurisdiction filters and post-selection VF [13], value-model negative weights [13][26], borderline demotion and authoritative boosts [30][32], log-linear vetoes [25], filter exploration and holdouts [13][19], and per-account label reports [13].

### 33.7.3 Transparency as a design requirement

Transparency artifacts are becoming part of the architecture rather than PR:

- **Meta** publishes system cards for Facebook Feed and Instagram Explore that list stages and main prediction signals [23][27].
- **YouTube** maintains "How YouTube Works" pages for recommendations [33].
- **X** publishes production ranking code and parameters, and gives each account an "Under the Hood" report of labels limiting its visibility, including legal withholding by country [13].
- **Bluesky** moves ranking out of the platform into third-party feed generators that return only post URIs, which the platform hydrates. Moderation and spam filtering then sit in shared hydration and labeling layers rather than inside each feed [42] (Chapter 28).

The tension is gaming. X publishes code but withholds some rules and LLM prompts "to reduce the risk of gaming", and compensates with per-account reports [13]. Designers should decide explicitly which parts of the integrity layer are public, which are disclosed per account, and which stay private.

### 33.7.4 Objectives beyond engagement

Responsible-recommendation work in production is mostly about objectives. YouTube's valued watch time [30], Meta's survey-weighted Feed value [21] and Netflix's quality-play reward [35] all replace a raw engagement label with a measure of satisfaction or value. The MSI history [22] is the cautionary counterexample: weights chosen from surveys still amplified divisive content when they rewarded reactive engagement. Objective changes need long-term holdouts and integrity metrics (prevalence of borderline content in recommendations), not only engagement metrics.

---

## 33.8 Side-by-side: integrity mechanisms by company

| Company / system | Where integrity runs | Hard actions | Soft actions | Signals | Transparency | Ref |
|---|---|---|---|---|---|---|
| Meta Facebook Feed | pass 0 before ranking; value model | remove ineligible | borderline reduction; survey-weighted value | integrity classifiers, surveys, few-shot classifiers | Feed system card | [20][21][23][28][29] |
| Instagram Explore | value model; final re-rank | integrity filters | log-linear veto on P(not interested); see-less penalty | negative-feedback heads | Explore system card | [24][25][26][27] |
| YouTube | candidate scoring | removal of violations | borderline demotion; authoritative boost | rater-trained classifiers, survey models | How YouTube Works | [30][31][32][33] |
| X 2023 | features; post-selection VF | Drop | labels, interstitials, coarse downranking; pToxicity/pBlock | safety labels, health models | open code (partial VF) | [16][17] |
| X 2026 | async labeling; index gating; VF on top-K | DROP (first wins), jurisdiction filters | NOTICE, INTERSTITIAL; OON-only spam rules | Grox LLM, CLIP, pNSFW, cred, behaviour | code + Under the Hood | [13][14] |
| X notifications | predicates in take step | withheld-content gates | health / quality / pNeg predicates; OON spread control | BQML health and quality models | open code | [19] |
| Etsy search | inside ANN retrieval | n/a | quality-aware retrieval | product quality score | n/a | [39] |
| TikTok For You | eligibility | not eligible for FYF | n/a (details not public) | spam, duplicate, shocking content | newsroom explainer | [41] |

---

## Key takeaways for system designers

1. **Build one content-understanding platform and reuse it.** A shared multimodal embedding (CLIP-like, Pinterest unified visual, PinSage) serves cold-start retrieval, ranking features, deduplication and safety classifiers. Pinterest's move from per-product embeddings to one unified embedding shows the operational payoff [2][3].
2. **Regress content into the collaborative space.** Spotify's audio CNN and OneRec's item-pair contrastive loss both align content representations with engagement, so cold items land where warm ones would [8][36].
3. **In generative recommenders, the tokenizer is the content and integrity boundary.** Semantic IDs from multimodal embeddings (YouTube, Kuaishou, X) decide what can be generated. Gate index and codebook membership on integrity labels, as X does by consulting VF before indexing [10][13][14].
4. **Label asynchronously, enforce at several points.** Write labels once to a store. Apply cheap filters before ranking, demotions in the value model, and a final visibility filter on top-K, including dependent items such as quoted or parent posts [13][20].
5. **Model visibility as (item, viewer, surface) -> action.** X's SafetyLevel, rule priority and action set (drop, label, interstitial, downrank) is a reusable schema [16].
6. **Use graded responses.** Remove violations, reduce borderline and spammy content, inform with labels, and raise authoritative sources. Apply stricter rules to recommendations of unfollowed content than to content users chose to follow [13][30][32].
7. **Expect engagement optimizers to drift toward the line.** Borderline content attracts engagement. Counter it with explicit demotion and with satisfaction-based objectives (surveys, quality plays), and monitor prevalence, not only engagement [22][29][30][35].
8. **Give negative signals real weight.** Negative-feedback heads, large report and block weights calibrated to base rates, and multiplicative vetoes are the cheapest safety mechanisms in the stack [13][25].
9. **Plan for policy churn.** Few-shot and LLM-based classifiers let integrity levers follow new policy in weeks rather than months [13][28].
10. **Defend against manipulation with behaviour, not just content.** Credibility scores, block-versus-favorite ratios, behaviour-sequence models and author-level label aggregation catch coordinated abuse that content classifiers miss [13][18].
11. **Measure the filters.** Log every drop and demotion with its reason, keep holdouts, and explore a small share of filtered candidates so filter errors are observable [13][19].
12. **Decide transparency deliberately.** System cards, policy explainer sites, open code and per-account label reports each disclose a different layer. Withhold only what enables gaming, and compensate with per-account disclosure [13][23][33].

---

## References

Sources marked "(not re-fetched)" are well-known primary sources cited from prior knowledge because the live web was unreachable during this chapter's research pass. Their figures are marked approximate in the text. All other sources were fetched and verified by other chapters of this Atlas.

1. Radford, A., Kim, J. W., Hallacy, C., et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP). OpenAI, ICML 2021. https://arxiv.org/abs/2103.00020 (not re-fetched)
2. Zhai, A., Wu, H.-Y., Tzeng, E., Park, D. H., Rosenberg, C. "Learning a Unified Embedding for Visual Search at Pinterest." Pinterest, KDD 2019. https://arxiv.org/abs/1908.01707 (not re-fetched)
3. Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W. L., Leskovec, J. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage). Pinterest / Stanford, KDD 2018. https://arxiv.org/abs/1806.01973
4. Pinterest. "OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search." 2024. https://arxiv.org/abs/2404.16260
5. Baltescu, P., Chen, H., Pancha, N., Zhai, A., Leskovec, J., Rosenberg, C. "ItemSage: Learning Product Embeddings for Shopping Recommendations at Pinterest." Pinterest, KDD 2022. https://arxiv.org/abs/2205.11728 (not re-fetched)
6. Pal, A., Eksombatchai, C., Zhou, Y., Zhao, B., Rosenberg, C., Leskovec, J. "PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest." Pinterest, KDD 2020. https://arxiv.org/abs/2007.03634
7. Pancha, N., Zhai, A., Leskovec, J., Rosenberg, C. "PinnerFormer: Sequence Modeling for User Representation at Pinterest." Pinterest, KDD 2022. https://arxiv.org/abs/2205.04507
8. Kuaishou OneRec Team. "OneRec Technical Report." Kuaishou, 2025. https://arxiv.org/abs/2506.13695
9. Xing, H. "End-to-End Generative Recommendation" (OneRec / OneRec-V2 technical notes, fun-rec-mdbook; secondary). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part8-e2e/e2e-recommendation.md
10. Google / YouTube. "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations." 2025. https://arxiv.org/abs/2510.07784
11. Singh, A., Vu, T., Mehta, N., et al. "Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations." Google / YouTube, RecSys 2024. https://arxiv.org/abs/2306.08121
12. Rajput, S., Mehta, N., Singh, A., et al. "Recommender Systems with Generative Retrieval" (TIGER). Google, NeurIPS 2023. https://arxiv.org/abs/2305.05065
13. xAI / X. "X For You Feed Algorithm" (x-algorithm README, code and history; initial release 2026-01-20, updates through 2026-09). 2026. https://github.com/xai-org/x-algorithm
14. xAI / X. "Phoenix: Recommendation System" README. 2026. https://github.com/xai-org/x-algorithm/blob/main/phoenix/README.md
15. Twitter / X. "ScoredTweetsParam.scala" (Home Mixer scored-tweets parameters). 2023-2025. https://github.com/twitter/the-algorithm/blob/main/home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/param/ScoredTweetsParam.scala
16. Twitter / X. "Visibility Filtering" README (visibilitylib). 2023. https://github.com/twitter/the-algorithm/blob/main/visibilitylib/README.md
17. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter Engineering Blog, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
18. El-Kishky, A., Markovich, T., Park, S., et al. "TwHIN: Embedding the Twitter Heterogeneous Information Network for Personalized Recommendation." Twitter, KDD 2022. https://arxiv.org/abs/2202.05387
19. Twitter / X. pushservice Scala `predicate` package. 2023. https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/scala/com/twitter/frigate/pushservice/predicate
20. Lada, A., Wang, M., Yan, T. "How machine learning powers Facebook's News Feed ranking algorithm." Engineering at Meta, 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
21. Meta. "Incorporating More Feedback Into News Feed Ranking." Meta Newsroom, 2021. https://about.fb.com/news/2021/04/incorporating-more-feedback-into-news-feed-ranking/
22. CNN Business (syndicated). "Likes, anger emojis and RSVPs: the math behind Facebook's News Feed and how it backfired." 2021. https://abc17news.com/news/2021/10/27/likes-anger-emojis-and-rsvps-the-math-behind-facebooks-news-feed-and-how-it-backfired/
23. Meta Transparency Center. "Facebook Feed AI system" (system card). Accessed 2026. https://transparency.meta.com/features/explaining-ranking/fb-feed
24. Medvedev, I., Wu, H., Gordon, T. "Powered by AI: Instagram's Explore recommender system." Instagram Engineering, 2019. https://instagram-engineering.com/powered-by-ai-instagrams-explore-recommender-system-7ca901d2a882
25. Meta / Instagram. "How Instagram suggests new content." Engineering at Meta, 2020. https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/
26. Vorotilov, V., Shugaepov, I. "Scaling the Instagram Explore recommendations system." Engineering at Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
27. Meta Transparency Center. "Instagram Explore AI system" (system card). Accessed 2026. https://transparency.meta.com/features/explaining-ranking/ig-explore/
28. Meta AI. "Harmful content can evolve quickly. Our new AI system adapts to tackle it" (Few-Shot Learner). Meta, 2021. https://ai.meta.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it/ (not re-fetched)
29. Zuckerberg, M. "A Blueprint for Content Governance and Enforcement." Facebook, 2018. https://www.facebook.com/notes/mark-zuckerberg/a-blueprint-for-content-governance-and-enforcement/10156443129621634/ (not re-fetched)
30. Goodrow, C. "On YouTube's recommendation system." YouTube Official Blog, 2021. https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/
31. YouTube. "Continuing our work to improve recommendations on YouTube." YouTube Official Blog, January 2019. https://blog.youtube/news-and-events/continuing-our-work-to-improve/ (not re-fetched)
32. YouTube. "The Four Rs of Responsibility, Part 2: Raising authoritative content and reducing borderline content and harmful misinformation." YouTube Official Blog, December 2019. https://blog.youtube/inside-youtube/the-four-rs-of-responsibility-raise-and-reduce/ (not re-fetched)
33. YouTube. "How YouTube Works: Recommendations." Accessed 2026. https://www.youtube.com/howyoutubeworks/product-features/recommendations/
34. Covington, P., Adams, J., Sargin, E. "Deep Neural Networks for YouTube Recommendations." Google / YouTube, RecSys 2016. https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
35. Chandrashekar, A., Amat, F., Basilico, J., Jebara, T. "Artwork Personalization at Netflix." Netflix Technology Blog, 2017. https://netflixtechblog.com/artwork-personalization-c589f074ad76
36. Dieleman, S. "Recommending music on Spotify with deep learning." Personal blog, 2014. https://benanne.github.io/2014/08/05/spotify-cnns.html
37. van den Oord, A., Dieleman, S., Schrauwen, B. "Deep content-based music recommendation." NeurIPS 2013. https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation
38. Zhang, C., et al. "NoteLLM: A Retrievable Large Language Model for Note Recommendation." Xiaohongshu, WWW 2024. https://arxiv.org/abs/2403.01744
39. Jha, R., Subramaniyam, S., Benjamin, E., Taula, T. "Unified Embedding Based Personalized Retrieval in Etsy Search." Etsy, KDD 2023. https://arxiv.org/abs/2306.04833
40. Zinkevich, M. "Rules of Machine Learning: Best Practices for ML Engineering." Google for Developers. https://developers.google.com/machine-learning/guides/rules-of-ml
41. TikTok. "How TikTok recommends videos #ForYou." TikTok Newsroom, 2020. https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you (not re-fetched)
42. Bluesky Social. "ATProto Feed Generator" starter kit README. https://github.com/bluesky-social/feed-generator
