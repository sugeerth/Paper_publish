# 15. Re-ranking, Blending and Whole-Page Optimization

The last stage of almost every industrial recommender gets the least attention in papers and holds the most product policy in code. After retrieval, pre-ranking and a heavy multi-task ranker have assigned each candidate a score, something still has to turn a scored set into the thing a person sees. That might be a feed of posts mixed with ads and modules, a Netflix homepage of rows, a Taobao result page or a Kuaishou session of six videos. This chapter covers that "last mile":

- **Listwise and slate re-ranking.** Models that score an item in the context of the other items on the list. Examples are DLCM, PRM (Alibaba), SetRank and PEAR, plus generator-evaluator and generative rerankers such as Seq2Slate, GRN/PRS, PIER and NAR4Rec.
- **Diversification.** MMR, determinantal point processes (Hulu, YouTube, X), author decay, calibration.
- **Heterogeneous feed blending.** Mixing content types, sources and ads into one stream (LinkedIn, Pinterest, X/Twitter, Meta).
- **Page-level optimization.** Choosing and ordering rows and shelves (Netflix, Spotify, Deezer, Amazon).
- **The business-rules and constraints layer**, which every published design keeps even as models get larger.

A note on sourcing. The live web-search and fetch budget for this chapter was used up before research could start, and outbound access to arXiv and company blogs was blocked from the writing environment. The chapter therefore relies on two kinds of source. The first is primary sources already fetched and verified by other chapters of this Atlas, such as Meta, X, Pinterest, LinkedIn, Netflix and Kuaishou posts and repositories. The second is a small set of canonical, widely cited papers whose identifiers are well established. Where a number could not be checked again, it is marked approximate. Readers should treat specific online-lift figures as indicative only.

---

## 1. Why a separate re-ranking stage exists

A pointwise ranker estimates something like P(click | user, item) for each candidate on its own. Then it sorts. That rests on three assumptions that fail in real products:

1. **Independence.** An item's utility does not depend on its neighbours. In reality, the fifth phone case in a row is worth less than the first. A cheap item next to an expensive one changes how both are perceived. Alibaba's work on "mutual influence aware ranking" in Taobao search was motivated directly by this [10].
2. **Single objective.** One score captures everything. In practice the page must also satisfy diversity, freshness, creator exposure, ad revenue, integrity and legal constraints [1][14][15].
3. **Uniform positions.** Every slot is equal apart from a position-bias discount. Real surfaces are two-dimensional (Netflix rows, Spotify shelves) or have strong cascade behaviour (carousels where only the first few cards are seen) [23][24][25].

Google's public recommender course names the three-box pipeline candidate generation, scoring and **re-ranking**. It assigns freshness, diversity and fairness to the last box [2]. Eugene Yan's system-design synthesis likewise ends the online path with a ranking step followed by business logic and filtering [3]. Every public funnel in Chapter 1 has some final pass: Instagram's "final reranking" with integrity, diversity and business rules [14], Facebook Feed's final pass with integrity demotions and diversity [16], X's Home Mixer heuristics [17][18], and Pinterest's Multi-Objective Blender [20].

### 1.1 Canonical last-mile architecture

```
 heavy ranker output: ~10^1-10^3 candidates, each with p_click, p_like, p_hide, ...
        |
        v
 +--------------------+   value model: score = sum w_i * p_i (+ offsets)
 | 1. score fusion    |   (Ch. 4: Meta value model, X weights, Kuaishou ensemble sort)
 +---------+----------+
           v
 +--------------------+   hard filters: integrity, blocked/muted, seen, legal,
 | 2. eligibility     |   inventory, geo, age, dedup by item / cluster
 +---------+----------+
           v
 +--------------------+   context-aware rescoring: DLCM / PRM / SetRank / PEAR
 | 3. listwise rerank |   or generator + evaluator (Seq2Slate, GRN, PIER, NAR4Rec)
 +---------+----------+
           v
 +--------------------+   MMR / DPP / author decay / calibration / cold-start lift
 | 4. diversification |
 +---------+----------+
           v
 +--------------------+   interleave content types and sources, insert ads at
 | 5. blending / slots|   allowed positions, modules (WTF, stories, carousels)
 +---------+----------+
           v
 +--------------------+   rows / shelves / modules chosen and ordered;
 | 6. page assembly   |   cross-row dedup; device templates
 +---------+----------+
           v
     response + impression log (slate, positions, propensities)
```

In real systems these boxes are often merged. X folds diversity, out-of-network discounts and cold-start lifts into one scoring pass before a DPP reranker [18][19]. Netflix folds row selection and title placement into "page generation" [23][24]. The logical split is still useful when designing one.

---

## 2. Rule-based and heuristic re-ranking

### 2.1 Maximal Marginal Relevance (MMR)

MMR (Carbonell and Goldstein, SIGIR 1998) is the ancestor of every greedy diversifier. Items are picked one at a time. Each step takes the candidate that maximizes lambda * rel(i) - (1 - lambda) * max_{j in S} sim(i, j), where S is the set already selected [4]. It costs O(N*K) similarity evaluations, needs no training and has one knob. Its weakness is that it only penalizes the nearest already-picked item, not the set as a whole.

### 2.2 Determinantal point processes (Hulu, YouTube, X)

A DPP scores a set S by det(L_S), where the kernel L_ij = q_i * sim(i, j) * q_j combines per-item quality q_i with pairwise similarity. The determinant is large when items are both good and mutually dissimilar.

- **Hulu (NeurIPS 2018).** Chen, Zhang and Zhou made greedy MAP inference for DPPs practical. They update a Cholesky factor incrementally, so selecting K items from N costs O(N*K^2) instead of a fresh determinant at every step. They also add a sliding-window variant, so diversity is enforced only among nearby positions in a long feed [5]. That window idea matters in feeds, where a user cares that adjacent items differ, not that item 3 differs from item 300.
- **YouTube (CIKM 2018).** Wilhelm et al. describe deploying a DPP on YouTube's mobile home feed. They learn the kernel from user interactions, parametrizing quality from the existing ranker and similarity from embeddings plus a few learned parameters. They apply it in windows over the feed. The paper reports gains in user satisfaction metrics from the deployment. Exact lift figures are not repeated here because they could not be checked again [6].
- **X (2026).** In xAI's open-sourced x-algorithm, a separate `vm-ranker` step reorders the scored candidates with a DPP over embeddings after value-model scoring. It gives up a little score for less similarity between neighbouring posts [18][19].

### 2.3 Multiplicative decays and lifts (X / Twitter)

The X value-model code is the most detailed public example of a heuristic reranking layer [18][19]:

```
  weighted sum of ~25 heads  -> negative-score offset (keeps scores > 0)
     -> + exploration bonus
     -> x author-diversity multiplier: (1 - floor) * decay^k + floor
            (k = number of higher-ranked posts already chosen from that author)
     -> x out-of-network factor (< 1 for unfollowed accounts)
     -> cold-start lift (new author raised to at least a target rank)
     -> DPP reorder (vm-ranker) -> visibility filters -> timeline
```

Two details are worth copying. The **negative-score offset** maps negative raw scores into a small positive band [0, OFFSET) before any multiplier runs. A multiplier below 1 therefore cannot "promote" a negatively scored item by pulling it towards zero. And the **author decay is sequential**: it depends on how many posts by that author already sit above the candidate, so it has to be applied greedily in rank order, not as a single vectorized pass. The 2023 Twitter Home Mixer README lists the same family of heuristics: author diversity, content balance between in-network and out-of-network, feedback fatigue, de-duplication and social proof. Ads, Who-to-Follow and other modules are mixed in afterwards [17].

### 2.4 Calibration as a reranking objective (Netflix)

Steck's "Calibrated Recommendations" (Netflix, RecSys 2018) re-ranks a list so that its genre distribution q(g) matches the user's historical distribution p(g). It measures miscalibration with a KL divergence C_KL(p, q) and greedily maximizes (1 - lambda) * relevance - lambda * C_KL [7]. The motivation is that accuracy-optimized lists crowd out a user's minority interests. Someone who watches 70% romance and 30% action can get a list that is 100% romance. Calibration is a *distributional* constraint, not a pairwise similarity penalty, and it is cheap (greedy, submodular-style).

### 2.5 Round-robin blending (Pinterest)

Pinterest's early home feed ranked each content type with its own specialized model, then "blended with a ratio-based round robin" to produce the final feed [21]. A round robin with fixed ratios is the simplest heterogeneous blender. It guarantees a share of each type, but it cannot trade a mediocre item of type A against an excellent item of type B. That limit motivated Pinterest's later Multi-Objective Blender inside the ranking service [20].

---

## 3. Listwise contextual re-rankers

These models take the top-K list from the ranker (K is typically tens to low hundreds) and re-score every item *given the whole list*. Output is still a score per item, so serving stays "sort by score". This keeps them compatible with existing infrastructure.

### 3.1 DLCM: a recurrent list encoder (2018)

Ai et al.'s Deep Listwise Context Model (SIGIR 2018) feeds the top-K documents from a base ranker, in reverse rank order, into a GRU. The final hidden state is a "local context" vector. An attention-style scoring function then re-scores each item against that context. Training uses a listwise attention-rank loss [8]. It established the pattern: **base ranker, then list encoder, then re-scored list**. Its weakness is that the RNN is order-sensitive and processes items sequentially, so distant items interact only weakly.

### 3.2 Groupwise scoring functions (GSF)

Ai et al. (ICTIR 2019) score small groups of documents jointly with a DNN and aggregate the group scores per document. This makes "multivariate scoring" explicit without a full sequence model [9]. It shows that cross-item features help, but the number of groups grows combinatorially, which pushed later work towards attention.

### 3.3 Mutual-influence-aware ranking at Taobao (MIDNN)

Zhuang et al. (Alibaba, IJCAI 2018) formalize "globally optimized mutual influence aware ranking" for Taobao search. They add list-level features to each item, such as how an item's price compares with the rest of the list, and train a DNN on them. The final permutation is chosen with beam search under an evaluator [10]. This is an early industrial instance of the **generator-evaluator** pattern described in Section 4.

### 3.4 PRM: Personalized Re-ranking Model (Alibaba, RecSys 2019)

PRM, by Pei et al., is the most-cited industrial reranker and the template for most later work [11].

**Problem.** The initial list in Taobao recommendation is ranked pointwise. Item-item interactions on the list, and how those interactions differ per user, are ignored. A user who cares about price reacts differently to a list of similar items than a user who cares about brand.

**Architecture.**

```
  initial list from ranker: items i_1 .. i_n  (n = list length, tens of items)
        |
        v
  per item input  x_i = [ item feature vector  ||  PV_i ]  +  PE_i
        |               PV_i = personalized vector: hidden layer of a pre-trained
        |                      CTR model taking (user, item) -> user-specific item rep
        |               PE_i = learnable position embedding (initial rank)
        v
  linear projection to d_model
        v
  N x Transformer encoder blocks (multi-head self-attention + FFN)
        |   every item attends to every other item -> O(n^2) but n is small
        v
  linear + softmax over the list  -> re-ranked scores
        v
  sort -> final list
```

**Key ideas.**
- *Self-attention instead of RNN.* Any two items interact directly, whatever their distance in the list. Attention is also parallel across positions, which matters for serving latency.
- *Personalized vector (PV).* Rather than feed raw user features into the reranker, PRM pre-trains a pointwise CTR model and uses its hidden-layer output for each (user, item) pair as a per-item personalized embedding. This puts user-specific item representations into the reranker cheaply.
- *Position embedding of the initial rank.* The ranker's ordering is itself a strong feature. Encoding it lets PRM learn when to keep and when to override the ranker.
- *Training objective.* A list-level softmax cross-entropy over clicked items in the displayed list.

**Results and deployment.** The paper reports offline gains over DLCM and other baselines on a public learning-to-rank dataset and on a Taobao e-commerce recommendation dataset. It reports online A/B gains in page views, clicks and GMV on Taobao's recommendation surface. The exact percentages are not repeated here because they could not be checked again [11]. The authors also released the e-commerce re-ranking dataset.

**Lessons.** PRM established three conventions most later rerankers keep: (1) re-rank a short list (tens of items) so quadratic attention is affordable; (2) inject personalization through embeddings from upstream models instead of retraining a big user tower; and (3) keep the original rank as an input feature.

### 3.5 SetRank and permutation invariance

Pang et al.'s SetRank (SIGIR 2020) argues that a reranker should be *permutation-invariant* with respect to its input. The result should not depend on the order in which the base ranker happened to present items. It uses stacked induced multi-head self-attention blocks without positional encodings (or with optional rank encodings) [12]. In practice industrial systems still include the initial rank as a feature, as PRM does, because it carries useful information. SetRank shows that it should be a *feature*, not a structural assumption.

### 3.6 PEAR: contextualized transformer with history (Huawei, WWW 2022)

PEAR (Li et al.) extends PRM-style models in two ways. It models both feature-level and item-level interactions inside the list. It also lets the list items attend to the user's *historical* interactions, so the reranker sees context from the user's past as well as from the current list. It adds an auxiliary list-level classification task that predicts whether the user engages with the list as a whole [13]. This makes the reranker partly a sequence model. The cost is more attention compute per request.

### 3.7 The evaluator problem

Every listwise model in this section scores items given the *initial* ordering. But once the reranker moves items, the context it scored them in no longer exists. Re-scoring after re-ranking is circular. This "evaluation-before-reranking" gap is what motivated the generator-evaluator and generative approaches in Section 4.

---

## 4. Generative and generator-evaluator re-ranking

### 4.1 Seq2Slate (Google, 2018/2019)

Bello et al.'s Seq2Slate frames re-ranking as sequence generation with a **pointer network**. An encoder (RNN in the original) reads the candidate set. A decoder emits one candidate per step, conditioned on what it has already placed, by attending over the inputs [27]. Because each step sees the items already chosen, the model captures "what is good *next* given what is above", which a scoring model cannot do. Training uses click logs. The paper proposes supervised listwise losses that treat clicked items as targets to be placed early, as well as a REINFORCE-style policy-gradient variant. It reports offline gains on learning-to-rank benchmarks and an evaluation on data from a large-scale commercial recommender at Google [27].

Serving cost is K sequential decoding steps over N candidates. That is acceptable for K of around 10-20 and N in the low hundreds, but it is the main obstacle to generative reranking at feed scale.

### 4.2 Generator-evaluator: Exact-K, PRS, GRN

Several Alibaba and Meituan papers split the problem in two:

```
            candidates (N)                      evaluator (list-wise CTR / utility model)
                |                                       ^
                v                                       | score each candidate permutation
   +---------------------------+     M candidate    +---+----------------------------+
   | generator: beam search /  | --> permutations-->| list-aware model: sees full   |
   | sampling / pointer net /  |     (M ~ tens-     | slate incl. later positions;  |
   | heuristic permutations    |      hundreds)     | predicts list reward          |
   +---------------------------+                    +---+----------------------------+
                                                        v
                                         pick argmax permutation -> serve
```

- **Exact-K (Alibaba, KDD 2019).** Gong et al. pose "exact-K recommendation", choosing a set of exactly K items that is jointly good, as maximal-clique optimization on a graph of candidate items. They use a graph-attention encoder plus a pointer-network decoder trained with RL, with a reward from a learned evaluator [28].
- **PRS: "Revisit Recommender System in the Permutation Prospective" (2021).** This paper separates a Permutation-Matching stage, which generates candidate permutations efficiently with beam search, from a Permutation-Ranking stage. The second stage scores each whole permutation with a list-wise evaluator. It reports online gains in a large production feed. The authors are at Alibaba/Taobao; the exact attribution and numbers were not re-checked [29].
- **GRN: Generative Rerank Network (2021).** Feng et al. train a context-wise *evaluator* first, then train a pointer-network *generator* with policy gradient, using the evaluator's predicted list reward as the reward signal [30].

The evaluator is trained on logged slates, so it sees each item in the context it was actually shown in. It is therefore a more faithful judge of a permutation than any pointwise score. The generator only has to propose good candidates.

### 4.3 PIER (Meituan, WSDM 2023)

PIER (Permutation-level Interest-based End-to-end Re-ranking) addresses the cost of the generator-evaluator pattern at Meituan's scale. A Fine-grained Permutation Selection Module uses SimHash-style hashing of user interest to pick the top permutations cheaply from a very large permutation space. An Omnidirectional Context-aware Prediction Module then scores those permutations. The two modules are trained end to end, instead of with a frozen evaluator [31]. The lesson is that permutation search has to be approximated aggressively. Hashing and retrieval-style tricks from candidate generation carry over to the permutation space.

### 4.4 NAR4Rec: non-autoregressive generation (Kuaishou, KDD 2024)

Autoregressive generators decode one position at a time, which adds latency in proportion to slate length. Kuaishou's NAR4Rec generates all positions in parallel with a non-autoregressive model. It adds techniques for the sparse-data and position-dependency problems that come with that choice: a matching model for training and contrastive decoding to avoid repeated items. The paper reports deployment in Kuaishou's short-video app, which has hundreds of millions of daily users, with gains in engagement metrics. Specific figures are approximate and were not re-checked [32].

### 4.5 The generative endpoint: session-wise generation (Kuaishou OneRec)

Kuaishou's OneRec (2025) goes furthest. A single generative model produces a whole *session* of videos, removing the retrieve/rank/rerank cascade for that path. It is then aligned with a reward model via preference optimization [33][34]. In effect the reranker's job of choosing a jointly good list moves into the generator, and the reranker's evaluator becomes a reward model. Chapter 6 covers OneRec in depth. The relevant point here is that even OneRec's reports keep filtering and business-rule logic outside the model.

---

## 5. Slate RL and long-horizon reranking

Pointwise and listwise rerankers optimize immediate list reward. Two lines of work extend this over time.

- **SlateQ (Google, IJCAI 2019).** Under a single-choice assumption (the user picks at most one item from the slate) and the assumption that rewards and transitions depend only on the item picked, the long-term value of a slate decomposes into item-level Q-values weighted by a user choice model. The slate can then be optimized exactly with a linear program for the conditional-logit choice model, or approximately with greedy top-k. SlateQ was tested in live YouTube experiments by replacing the myopic per-item score with the decomposed long-term value [35].
- **Page-wise RL (JD.com, RecSys 2018).** Zhao et al. treat a whole 2-D page of items as the action. An actor-critic uses a deconvolutional actor to output the page layout, and a critic scores the page given user state and feedback on earlier pages [36].
- **Slate off-policy evaluation (Microsoft, NeurIPS 2017).** Swaminathan et al.'s pseudo-inverse estimator evaluates new slate policies from logged data, assuming the slate reward is additive over (item, position) pairs. Its sample complexity is far lower than naive inverse propensity scoring over the combinatorial slate space [37]. It is the standard tool for deciding whether a new reranker is worth an A/B slot.

Chapter 13 discusses these methods at length. For reranking, the main lesson is that **logging propensities for whole slates** is an architectural prerequisite. Without it, neither OPE nor slate RL works.

---

## 6. Heterogeneous feed blending

A single surface often shows several kinds of thing at once. Feed posts, jobs, ads, people-you-may-know cards, stories trays and video carousels come from separate teams with separately trained, separately calibrated models. Blending is the problem of putting them on one comparable scale and one layout.

### 6.1 LinkedIn: first-pass rankers and a second-pass ranker

LinkedIn's 2019 feed architecture is federated. Multiple **first-pass rankers (FPRs)** each select and score candidates from their own inventory (network updates, jobs, recommended content and so on). A **second-pass ranker (SPR)** merges their outputs into one ranked list. FollowFeed, which serves updates from a member's network, was the dominant FPR, supplying more than 80% of feed updates [22].

```
   FollowFeed FPR ----+   (network updates; >80% of feed items in 2019 [22])
   Jobs FPR ----------+
   Recommended FPR ---+--> SPR: unified feature space, multi-objective score
   PYMK / modules ----+        (e.g. viral actions, contributions, creator side)
   Ads (separate auction) ------------+
                                      v
                          slotting: ads at allowed positions -> feed
```

The design matches how the organization is split, since each inventory team owns its FPR. But it pushes **cross-source calibration** onto the SPR. If one FPR's scores drift, its share of the feed drifts too. LinkedIn tuned the SPR's multi-objective weights with automated online search instead of by hand (secondary coverage) [38]. The 2019 post's "community-focused" work also changed the objective to account for value to creators and the poster's community, not only to the viewer [22].

### 6.2 Pinterest: from round-robin to a multi-objective blender

As noted in Section 2.5, Pinterest first blended type-specific rankers with a fixed-ratio round robin [21]. Its WWW 2022 home-feed funnel puts a **Multi-Objective Blender** after the Utility Prediction Scorer inside the Ranking Service [20]. Later work personalizes the utility weights per request: PRL-PUTS, a one-step value-based RL policy that picks a weight vector per request and exposes an empirical Pareto frontier for governance. It reports +0.13% successful sessions online with no added serving latency [39]. Moving from ratio rules, to a single learned utility, to per-request weights is the typical path for a blender.

### 6.3 X / Twitter: Home Mixer and modules

X's Home Mixer is built on Product Mixer, a framework of pipelines. A product pipeline selects mixer pipelines, which run candidate pipelines, and the mixer interleaves ranked tweets with ads and modules such as Who-to-Follow [17][40]. The 2026 x-algorithm describes the same orchestration as reusable Rust traits named Source, Hydrator, Filter, Scorer, Selector and SideEffect [18]. That vocabulary maps neatly onto the boxes in Section 1.1. The in-network/out-of-network balance, about 50/50 on average in 2023, is a blending decision made through an out-of-network multiplier, not a fixed quota [17][19].

### 6.4 Meta: Facebook Feed and Instagram

Meta's 2021 Feed post describes ranking in passes. Integrity processes run first. A lightweight model narrows more than 1,000 candidate posts per user to about 500. Multitask neural networks predict several actions. Those predictions are combined into a single value. A final pass applies contextual rules such as diversity, so that a user does not see several posts from the same source in a row [16]. Instagram Explore's 2023 architecture likewise ends with a final reranking stage applying integrity, diversity and business rules after a linear value model over MTML predictions [14]. The 2020 Instagram post explains why some rules live as *constraints*: suggested content in Explore had to pass integrity and quality bars before being eligible. A log-linear value model lets a negative-feedback prediction act as a near-veto [15].

Meta's public posts do not describe in detail how ads are slotted into organic feeds. The architectural pattern visible across the industry is that ads go through their own retrieval, ranking and auction (Chapter 1 summarizes Andromeda and GEM) and are then **inserted** into the organic stream at allowed positions. Insertion is governed by ad-load and spacing constraints, not by direct score comparison with organic items.

### 6.5 Ads and organic: the allocation problem

A pattern consistent with public designs and with academic literature on feed ad allocation:

```
 organic ranked list O = [o1, o2, ...]       ad list A = [a1, a2, ...] (auctioned)
                \                                   /
                 v                                 v
          +---------------------------------------------------+
          | allocator                                         |
          |  - candidate slots: fixed (e.g. every k-th) or    |
          |    dynamic                                        |
          |  - constraints: min gap between ads, max ad load  |
          |    per session, no ad in slot 1, category rules   |
          |  - objective: revenue(a, pos) + lambda *          |
          |    engagement_loss(pos)                           |
          |  - solver: greedy / DP over slots / RL policy     |
          +---------------------------------------------------+
                                  v
                       final feed + ad-load telemetry
```

The key design decisions are:

1. **Shared currency.** Organic value (predicted engagement) and ad value (expected revenue) are in different units. Exchange rates (lambda) are set as policy and tuned online against long-term retention, not learned end to end.
2. **Fixed vs dynamic slots.** Fixed slots are predictable and easy to audit. Dynamic slots let a high-value ad move up, or an irrelevant one move down or out, per user and session.
3. **Externalities.** An ad's cost is lost engagement on the organic items it displaces and any session shortening it causes. That makes this a page-level problem, not an item-level one.

Several Chinese platforms (Meituan, ByteDance, Kuaishou) have published RL-based ad-allocation papers with this structure. Because none of them could be checked again for this chapter, they are not cited individually.

---

## 7. Whole-page optimization

### 7.1 Netflix: page generation

Netflix's homepage is the canonical two-dimensional recommendation surface. Gomez-Uribe and Hunt (ACM TMIS 2015) describe it as a grid of rows. A typical page had about 40 rows, each with up to about 75 titles (device-dependent). Each row is produced by a different algorithm: the Personalized Video Ranker (PVR) for genre rows, a Top-N ranker for "Top Picks", trending, continue-watching, and video-video similarity for "Because you watched" rows. A **page generation** algorithm selects and orders the rows [23]. The paper notes that a member typically loses interest after about 60-90 seconds of choosing, having looked at perhaps 10-20 titles, which is why page composition matters so much. It also estimates that recommendation and personalization saved Netflix more than 1 billion US dollars a year in reduced churn (approximate, as stated in the paper) [23].

Netflix's 2015 tech-blog post "Learning a Personalized Homepage" (Alvino and Basilico) describes the page-generation problem in more detail [24]:

```
  candidate rows: genre rows (PVR), Top Picks, Because-You-Watched, Trending,
                  Continue Watching, New Releases, ...  (each row ranks its own titles)
        |
        v
  row scoring  (relevance of row to member, given context/device)
        |
        v
  stage-wise greedy page construction:
     for slot in 1..R:
        pick row maximizing  score(row | rows already placed)
          - penalize duplicate titles across rows (dedup / diversity)
          - enforce constraints (e.g. Continue Watching near top, device limits,
            min/max of row types)
        |
        v
  render: rows top-to-bottom, titles left-to-right (2-D attention)
```

Points from that post:

- **Template vs row-scoring approaches.** A fixed template ("row type X at position Y") is simple but not personal. Scoring rows independently and sorting gives relevance but not diversity. Rows end up redundant and repeat the same titles.
- **Stage-wise (greedy) construction.** The page is built one row at a time. Each candidate row is scored given the rows already chosen, which naturally captures diversity and dedup. It is the row-level analogue of MMR.
- **Page-level metrics.** Because the user scans in two dimensions, Netflix evaluates pages with 2-D extensions of ranking metrics, for example a recall over the top-m rows by top-n columns that a user is likely to see.
- **Balance.** The page has to balance accuracy, diversity, discovery and freshness against constraints that differ by device.

Netflix's later posts add more page-level signals. Artwork is personalized per member with contextual bandits, and the choice of image interacts with the rest of the page [25]. Ranking has been consolidated into fewer, multi-task models serving many rows and canvases [41]. Objectives have moved towards long-term member satisfaction [42]. Netflix also uses interleaving to compare *row* rankers. It cannot measure page-composition effects, so those still need A/B tests [43].

### 7.2 Spotify home: shelves with explanations

Spotify's BaRT (RecSys 2018) is the bandit system behind the home page, a grid of **shelves** of playlists and albums. Each recommendation can carry an explanation ("Because you listened to X"). BaRT treats the (item, explanation) pair as the action and learns which combination to show in each shelf position. It uses counterfactual training from logged propensities [26]. It is whole-page optimization at the shelf level. The reranker chooses *how a recommendation is framed* as well as which item to show.

### 7.3 Deezer: carousels and the visibility model

Deezer models its playlist carousel as a contextual **cascade bandit**. Only the first L cards are visible without swiping, and cards beyond the last one the user would have seen get no feedback, so they are neither penalized nor rewarded. Users are clustered so that bandits learn per cluster (semi-personalization) [44]. The lesson for reranking: the **click model of the surface** has to be part of the objective, otherwise the system learns to avoid positions that were never seen.

### 7.4 Amazon: combinatorial page layouts

Hill et al. (Amazon, KDD 2017) optimize a page of several widgets with many possible values each, giving a combinatorial space of layouts. They use a Bayesian linear model with pairwise interaction terms, Thompson sampling, and greedy hill climbing to approximately maximize the sampled model in real time. They report about a 21% conversion improvement over the median layout in a live experiment [45]. The pattern is to **sample a model, then use a cheap combinatorial optimizer**. It carries over directly to row ordering and module placement.

### 7.5 Airbnb: diversity in search ranking

Airbnb's "Learning To Rank Diversely" (2022) addresses a whole-page effect in search. Listings ranked purely by booking probability can become homogeneous, for example all at the same price point or in the same neighbourhood. Users who would book something different then find nothing. The paper changes the training and inference of the neural ranker so that each listing's score depends on what was ranked above it, which is a contextual "similarity to listings above" mechanism. It reports a booking gain in online experiments [46].

---

## 8. The business-rules and constraints layer

Every public design keeps an explicit, non-learned layer: Netflix, Meta, X, Pinterest, LinkedIn, Kuaishou, even OneRec. Google's "Rules of Machine Learning" recommends keeping heuristics and filtering explicit and outside the model where they are policy rather than prediction [47].

### 8.1 What lives there

| Category | Examples | Hard / soft | Typical implementation |
|---|---|---|---|
| Eligibility | blocked/muted, already seen, out of stock, geo/licensing, age gates | hard | filter before or after rerank [16][17] |
| Integrity | policy-violating, borderline, low-quality | hard or demotion | integrity models run first; demotion multipliers [14][16] |
| Dedup | same item, near-duplicate media, same cluster, cross-row duplicates | hard | hash / cluster-id sets; page-level dedup [17][24] |
| Diversity | author/source caps, category spread, content balance | soft | decay multipliers, DPP, MMR [5][18][19] |
| Freshness / fatigue | recency boosts, "seen too often" penalties | soft | multipliers, impression-count features [17] |
| Exposure | cold-start creators, new items, exploration | soft | minimum rank, bonuses, reserved slots [18][19] |
| Mix / quotas | in-network vs out-of-network, content-type ratios | soft or hard | multipliers or ratio round robin [19][21] |
| Monetization | ad load, ad spacing, no ad in slot 1 | hard | allocator constraints (Section 6.5) |
| Layout | row count, row-type limits, device templates, pinned rows | hard | page-generation constraints [23][24] |

### 8.2 Design patterns

```
 config store (versioned, A/B-able)
      |  rules as data: {id, scope(surface, country), type(filter|cap|boost|slot),
      |                  params, priority, experiment_id}
      v
 +-------------------------------------------------------------------------+
 | rule engine in the mixer                                                |
 |  1. hard filters (set operations; cheap; run early AND re-run at end)   |
 |  2. score adjustments (multipliers/offsets; order-preserving sign care) |
 |  3. sequential constraints (greedy fill: author cap, gap rules, DPP)    |
 |  4. slot constraints (ads, modules, pinned rows)                        |
 |  5. backfill if constraints leave holes (fallback sources)              |
 +-------------------------------------------------------------------------+
      |
      v  log: which rule touched which item (for debugging and OPE)
```

Practical lessons from public sources:

- **Separate prediction from policy.** X ships its weights and multipliers as configuration (`param.rs`) separate from the model, so policy changes need no retraining [18][19]. Meta tunes value-model weights to trade off online metrics [14].
- **Watch the sign and scale of scores.** Multiplicative rules assume positive, calibrated scores. X's negative-score offset exists because a multiplier below 1 can raise a negative score [19]. Chapter 4 explains why calibration is the contract between the model and this layer.
- **Run filters twice.** Cheap eligibility filters run before expensive models to save compute (Meta runs integrity first [16]). They run again at the end, because later stages such as blending and backfill can reintroduce items.
- **Make rules visible to learning.** Rules change the logged distribution. If the reranker or ranker is trained on post-rule impressions without knowing which rule acted, it learns the rules as if they were user preferences. Logging rule decisions and propensities fixes this [37].
- **Rules accumulate.** The usual failure is hundreds of interacting boosts that nobody can explain. Treat rules as code: give each one an owner and an experiment, and expire it.

---

## 9. Comparison of re-ranking designs

| Design | Org / year | Input size (approx.) | Mechanism | Captures | Serving cost | Production evidence |
|---|---|---|---|---|---|---|
| MMR [4] | CMU, 1998 | any | greedy relevance - max sim | pairwise redundancy | O(NK) sims | widely used heuristic |
| Fast greedy DPP [5] | Hulu, 2018 | 10^2-10^3 | incremental Cholesky MAP, windowed | set diversity | O(NK^2) | Hulu experiments |
| DPP on feed [6] | YouTube, 2018 | feed windows | learned kernel, windowed | set diversity | low | live YouTube home |
| X value model + DPP [18][19] | X, 2026 | ~10^3 | multipliers, author decay, DPP | policy, diversity | low | open-sourced prod code |
| Calibration [7] | Netflix, 2018 | 10^2 | greedy KL to user genre mix | interest proportions | low | Netflix research |
| DLCM [8] | 2018 | top-K ~10-40 | GRU list encoder | local list context | sequential RNN | LTR benchmarks |
| MIDNN + beam [10] | Alibaba, 2018 | tens | list features + beam search | mutual influence | beam x DNN | Taobao search |
| PRM [11] | Alibaba, 2019 | tens | Transformer + PV + rank PE | item-item, personal | O(n^2 d) parallel | Taobao online A/B |
| SetRank [12] | 2020 | tens | permutation-invariant attention | set context | O(n^2 d) | LTR benchmarks |
| PEAR [13] | Huawei, 2022 | tens | list + history attention | list + history | higher | reported online |
| Seq2Slate [27] | Google, 2019 | 10^2 -> K | pointer-net decoding | sequential context | K decode steps | Google recsys data |
| Exact-K [28] | Alibaba, 2019 | tens -> K | graph attn + pointer, RL | joint set | decode + evaluator | reported online |
| PRS / GRN [29][30] | 2021 | tens | generator + list evaluator | whole permutation | M evaluator calls | reported online |
| PIER [31] | Meituan, 2023 | large perm space | hashed perm selection + eval | permutations | sub-linear search | Meituan online |
| NAR4Rec [32] | Kuaishou, 2024 | tens -> K | non-autoregressive generation | joint slate | 1 parallel pass | Kuaishou online |
| SlateQ [35] | Google, 2019 | slate | decomposed long-term Q + LP | long-term value | LP / greedy | live YouTube tests |
| Page generation [23][24] | Netflix, 2015 | ~10^2 rows | stage-wise greedy rows | 2-D page, dedup | low | Netflix homepage |
| BaRT [26] | Spotify, 2018 | shelves | bandit over item + explanation | framing, explore | low | Spotify home |
| OneRec [33][34] | Kuaishou, 2025 | full corpus | generative session + reward | end-to-end list | heavy generation | Kuaishou online |

---

## 10. Evaluating re-rankers

Re-rankers are hard to evaluate offline because they change the *context* in which every item is seen. Logged labels were collected under a different ordering.

1. **Pointwise metrics on re-ranked lists** (NDCG@K on logged clicks) are biased. An item moved to a position where it was never shown has no label there.
2. **Evaluator models**, the list-wise CTR models from the generator-evaluator work [29][30][31], are themselves learned and can be exploited by the generator. They should be validated against online results.
3. **Slate OPE** (the pseudo-inverse estimator [37]) gives unbiased estimates under additivity assumptions, but it needs logged propensities for the production slate policy.
4. **Interleaving** works for comparing two row-level rankers quickly, as at Netflix, but it cannot measure page composition [43].
5. **A/B tests with page-level and long-term metrics** (sessions, retention, diversity of consumption, creator-side metrics) remain the final arbiter. Netflix's shift towards long-term satisfaction [42] and Pinterest's Pareto-frontier governance [39] show that the operating point is a product decision.

---

## 11. Key takeaways for system designers

1. **Budget a separate last-mile stage.** It turns scores into a page. Even if you start with sort-by-score plus filters, give it its own service or pipeline step, with versioned config and per-rule logging [16][17][18].
2. **Re-rank short lists with context-aware models.** PRM-style Transformers over the top tens of items, with personalized vectors from upstream models and the original rank as a feature, are the proven first step beyond heuristics [11]. Quadratic attention is cheap at n of about 50.
3. **Mind the evaluation-before-reranking gap.** If list context matters, scoring items in the *initial* context is inconsistent. Generator-evaluator designs (PRS, GRN, PIER) or parallel generative models (NAR4Rec) close the gap at higher cost [29][30][31][32].
4. **Use DPP or MMR for diversity, windowed for feeds.** Hulu's fast greedy DPP and YouTube's windowed feed DPP show it scales. X runs it in production after its value model [5][6][18].
5. **Calibrate distributions, not just pairs.** Steck's calibration protects users' minority interests cheaply [7].
6. **Blend on a common, calibrated scale, then apply quotas as soft multipliers.** Fixed ratio round robin is a starting point (early Pinterest) [21]. Federated FPR-to-SPR designs need explicit cross-source calibration (LinkedIn) [22]. Per-request learned weights are where mature blenders end up (Pinterest PRL-PUTS) [39].
7. **Treat ads as an allocation problem with externalities.** Use separate auctions, insertion under ad-load and spacing constraints, and an explicit exchange rate between revenue and engagement, tuned against long-term metrics.
8. **For 2-D surfaces, optimize the page.** Netflix's stage-wise greedy row selection with cross-row dedup and page-level 2-D metrics is the template [23][24]. Spotify and Deezer show that framing (explanations) and visibility (cascade models) belong in the objective [26][44].
9. **Keep business rules explicit, auditable and logged.** Separate hard filters from soft adjustments. Protect score signs before multiplying (X's offset) [19]. Re-run filters at the end. Give every rule an owner and an experiment [47].
10. **Log the slate, not just the click.** Slate-level propensities and rule decisions are what make off-policy evaluation, slate RL and unbiased retraining possible [35][37].
11. **Expect generative models to absorb listwise reranking, not rules.** OneRec-style session generation moves set selection into the model, but published designs still keep integrity, eligibility and policy outside it [33][34].

---

## References

1. Covington, P., Adams, J., Sargin, E. "Deep Neural Networks for YouTube Recommendations." Google, RecSys 2016. https://research.google/pubs/pub45530
2. Google for Developers. "Recommendation systems overview" (Machine Learning: Recommendation Systems course; candidate generation, scoring, re-ranking). Google. https://developers.google.com/machine-learning/recommendation/overview/types
3. Yan, E. "System Design for Recommendations and Search." eugeneyan.com, 2021. https://eugeneyan.com/writing/system-design-for-discovery/
4. Carbonell, J., Goldstein, J. "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." CMU, SIGIR 1998. https://dl.acm.org/doi/10.1145/290941.291025
5. Chen, L., Zhang, G., Zhou, H. "Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity." Hulu, NeurIPS 2018. https://arxiv.org/abs/1709.05135
6. Wilhelm, M., Ramanathan, A., Bonomo, A., Jain, S., Chi, E. H., Gillenwater, J. "Practical Diversified Recommendations on YouTube with Determinantal Point Processes." Google/YouTube, CIKM 2018. https://dl.acm.org/doi/10.1145/3269206.3272018
7. Steck, H. "Calibrated Recommendations." Netflix, RecSys 2018. https://dl.acm.org/doi/10.1145/3240323.3240372
8. Ai, Q., Bi, K., Guo, J., Croft, W. B. "Learning a Deep Listwise Context Model for Ranking Refinement" (DLCM). SIGIR 2018. https://arxiv.org/abs/1804.05936
9. Ai, Q., Wang, X., Bruch, S., Golbandi, N., Bendersky, M., Najork, M. "Learning Groupwise Multivariate Scoring Functions Using Deep Neural Networks." Google, ICTIR 2019. https://arxiv.org/abs/1811.04415
10. Zhuang, T., Ou, W., Wang, Z. "Globally Optimized Mutual Influence Aware Ranking in E-Commerce Search." Alibaba, IJCAI 2018. https://arxiv.org/abs/1805.08524
11. Pei, C., Zhang, Y., Zhang, Y., Sun, F., Lin, X., Sun, H., Wu, J., Jiang, P., Ou, W. "Personalized Re-ranking for Recommendation" (PRM). Alibaba, RecSys 2019. https://arxiv.org/abs/1904.06813
12. Pang, L., Xu, J., Ai, Q., Lan, Y., Cheng, X., Wen, J. "SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval." SIGIR 2020. https://arxiv.org/abs/1912.05891
13. Li, Y., et al. "PEAR: Personalized Re-ranking with Contextualized Transformer for Recommendation." Huawei, WWW 2022 Companion. https://arxiv.org/abs/2203.12267
14. Vorotilov, V., Shugaepov, I. "Scaling the Instagram Explore recommendations system." Meta, Engineering at Meta, 2023. https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/
15. Meta / Instagram. "How Instagram suggests new content" (constrained exploration system). Engineering at Meta, 2020. https://engineering.fb.com/2020/12/10/web/how-instagram-suggests-new-content/
16. Lada, A., Wang, M., Yan, T. "How machine learning powers Facebook's News Feed ranking algorithm." Meta, Engineering at Meta, 2021. https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/
17. Twitter. "Home Mixer README" (the-algorithm). Twitter, 2023. https://github.com/twitter/the-algorithm/blob/main/home-mixer/README.md
18. xAI / X. "x-algorithm" repository README (Home Mixer, Phoenix, Thunder, candidate pipeline traits). xAI / X, 2026. https://github.com/xai-org/x-algorithm
19. xAI / X. "xai-value-model/scoring.rs" and "home-mixer/params/param.rs." xAI / X, 2026. https://github.com/xai-org/x-algorithm/blob/main/xai-value-model/scoring.rs ; https://github.com/xai-org/x-algorithm/blob/main/home-mixer/params/param.rs
20. Pinterest. "Pinterest at WWW 2022" (sponsor slides on the home-feed funnel, incl. Multi-Objective Blender). Pinterest, 2022. https://archives.iw3c2.org/www2022/wp-content/uploads/Sponsors/WWW2022-Pinterest.pdf
21. MLconf. Pinterest Homefeed talk abstract (ratio-based round-robin blending). MLconf, 2016. https://mlconf.com/?p=8909
22. LinkedIn Engineering. "Community-focused Feed optimization." LinkedIn, 2019. https://engineering.linkedin.com/blog/2019/06/community-focused-feed-optimization
23. Gomez-Uribe, C. A., Hunt, N. "The Netflix Recommender System: Algorithms, Business Value, and Innovation." Netflix, ACM TMIS 6(4), 2015. https://dl.acm.org/doi/10.1145/2843948
24. Alvino, C., Basilico, J. "Learning a Personalized Homepage." Netflix Technology Blog, 2015. https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a
25. Chandrashekar, A., Amat, F., Basilico, J., Jebara, T. "Artwork Personalization at Netflix." Netflix Technology Blog, 2017. https://netflixtechblog.com/artwork-personalization-c589f074ad76
26. McInerney, J., Lacker, B., Hansen, S., Higley, K., Bouchard, H., Gruson, A., Mehrotra, R. "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits" (BaRT). Spotify, RecSys 2018. https://dl.acm.org/doi/10.1145/3240323.3240354
27. Bello, I., Kulkarni, S., Jain, S., Boutilier, C., Chi, E., Eban, E., Luo, X., Mackey, A., Meshi, O. "Seq2Slate: Re-ranking and Slate Optimization with RNNs." Google, 2018/2019. https://arxiv.org/abs/1810.02019
28. Gong, Y., et al. "Exact-K Recommendation via Maximal Clique Optimization." Alibaba, KDD 2019. https://arxiv.org/abs/1905.07089
29. Feng, Y., et al. "Revisit Recommender System in the Permutation Prospective" (PRS). 2021. https://arxiv.org/abs/2102.12057
30. Feng, Y., et al. "GRN: Generative Rerank Network for Context-wise Recommendation." 2021. https://arxiv.org/abs/2104.00860
31. Shi, X., et al. "PIER: Permutation-Level Interest-Based End-to-End Re-ranking Framework in E-commerce." Meituan, WSDM 2023. https://arxiv.org/abs/2302.03487
32. Ren, Y., et al. "Non-autoregressive Generative Models for Reranking Recommendation" (NAR4Rec). Kuaishou, KDD 2024. https://arxiv.org/abs/2402.06871
33. Deng, J., et al. "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment." Kuaishou, 2025. https://arxiv.org/abs/2502.18965
34. Kuaishou OneRec Team. "OneRec Technical Report." Kuaishou, 2025. https://arxiv.org/abs/2506.13695
35. Ie, E., Jain, V., Wang, J., et al. "SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets." Google, IJCAI 2019. https://arxiv.org/abs/1905.12767
36. Zhao, X., Xia, L., Zhang, L., Ding, Z., Yin, D., Tang, J. "Deep Reinforcement Learning for Page-wise Recommendations." JD.com / Michigan State, RecSys 2018. https://arxiv.org/abs/1805.02343
37. Swaminathan, A., Krishnamurthy, A., Agarwal, A., Dudik, M., Langford, J., Jose, D., Zitouni, I. "Off-policy Evaluation for Slate Recommendation." Microsoft Research, NeurIPS 2017. https://arxiv.org/abs/1605.04812
38. KDnuggets (LinkedIn). "Autotuning for Multi-Objective Optimization on LinkedIn's Feed Ranking." 2020. https://www.kdnuggets.com/2020/08/autotuning-multi-objective-optimization-linkedin-feed-ranking.html
39. Pinterest. "A Production-Ready RL Framework for Personalized Utility Tuning with Pareto Sweeping in Pinterest Recommender Systems (PRL-PUTS)." Pinterest, 2026. https://arxiv.org/abs/2605.16344
40. Twitter Engineering. "Twitter's Recommendation Algorithm." Twitter, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm
41. Netflix Technology Blog. "Lessons learnt from consolidating ML models in a large scale recommendation system." Netflix, 2023. https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a
42. Netflix Technology Blog. "Recommending for Long-Term Member Satisfaction at Netflix." Netflix, 2024. https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef
43. Parks, J., Aurisset, J., Ramm, M. "Innovating Faster on Personalization Algorithms at Netflix Using Interleaving." Netflix Technology Blog, 2017. https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
44. Bendada, W., Salha, G., Bontempelli, T. "Carousel Personalization in Music Streaming Apps with Contextual Bandits." Deezer, RecSys 2020. https://arxiv.org/abs/2009.06546
45. Hill, D. N., Nassif, H., Liu, Y., Iyer, A., Vishwanathan, S. V. N. "An Efficient Bandit Algorithm for Realtime Multivariate Optimization." Amazon, KDD 2017. https://arxiv.org/abs/1810.09558
46. Haldar, M., et al. "Learning To Rank Diversely At Airbnb." Airbnb, 2022 (CIKM 2023). https://arxiv.org/abs/2210.07774
47. Zinkevich, M. "Rules of Machine Learning: Best Practices for ML Engineering." Google. https://developers.google.com/machine-learning/guides/rules-of-ml
