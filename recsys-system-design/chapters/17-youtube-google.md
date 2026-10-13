# 17. Google & YouTube Case Studies: From Co-Visitation Graphs to Semantic IDs and LLM-Native Recommenders

> Research note: this chapter was written when the shared live web-search budget for the run had been used up and direct page fetches were failing. It relies on the primary papers and posts listed in the References, using facts already checked against sources for earlier chapters of this Atlas (Chapters 1-4, 6, 11-15) plus the published content of the papers themselves. Numbers that could not be re-checked for this chapter are marked "approx.".

## Overview

Google has published more about its production recommenders than almost any other company. The papers form a fairly continuous line, and many techniques that are now standard industry practice started there:

- **2007-2010: graph and counting systems.** Google News used MinHash, PLSI and co-visitation counts [19][20]. YouTube's 2010 recommender built a related-video graph from co-visitation and expanded it from a user's seed videos [1].
- **2016: deep learning arrives.** Wide & Deep for Google Play [15] and the two-stage "Deep Neural Networks for YouTube Recommendations" paper [2] set the retrieval-plus-ranking template that most of the industry still uses.
- **2018-2019: multi-objective, debiased, RL-aware systems.** MMoE [6], the YouTube multitask Watch Next ranker with a shallow position-bias tower [5], two-tower retrieval with streaming logQ correction [4], top-K off-policy REINFORCE [8], SlateQ [9], and DPP diversification [12].
- **2020-2022: feature interaction and representation work.** DCN-V2 [18], mixed negative sampling [16], self-supervised item towers [17], survey-based "valued watch time" objectives [14], and reward shaping [10].
- **2023-2026: tokenized items and LLMs.** TIGER generative retrieval with semantic IDs [23], semantic IDs inside the YouTube ranker [24], LLM-based user-journey modeling [26], and PLUM, which adapts Gemini-family LLMs into industrial generative retrievers for YouTube [25].

The rest of the chapter takes each system in turn: problem, architecture, models, public scale numbers, key tricks and lessons. It closes with a comparison table and takeaways.

```
 Google / YouTube recommender lineage (simplified)

 2007 Google News  --- MinHash + PLSI + co-visitation (MapReduce/Bigtable)
 2010 YouTube      --- co-visitation related graph, seed expansion, heuristic rank
 2016 Google Play  --- Wide & Deep (memorization + generalization)
 2016 YouTube      --- DNN candidate gen (softmax -> ANN) + DNN ranker (watch time)
 2018 MMoE / DPP   --- multi-gate experts; windowed DPP feed diversification
 2019 YouTube      --- two-tower + logQ | MMoE + shallow tower | REINFORCE | SlateQ
 2020 Google       --- DCN-V2, mixed negative sampling, ScaNN
 2021 YouTube      --- "valued watch time" from surveys; self-supervised towers
 2023 Google       --- TIGER: semantic IDs + seq2seq generative retrieval
 2024 YouTube      --- semantic IDs replace video IDs in ranking embeddings
 2025 YouTube      --- PLUM: LLM continued pre-training on SIDs, generative retrieval
```

## 17.1 Google News personalization (2007, 2010)

### Problem

Google News had millions of users and a corpus that turned over within hours. Stories have very short lifetimes, so purely item-based models trained offline go stale, and new stories have no click history at all.

### Architecture (Das et al., WWW 2007)

Das, Datar, Garg and Rajaram described a **collaborative filtering system that is a blend of three signals**, each computed with Google's batch and serving infrastructure (MapReduce and Bigtable) [19]:

1. **MinHash clustering.** Users are clustered by the Jaccard similarity of their click histories, approximated with MinHash and locality-sensitive hashing. A story is recommended if users in the same cluster clicked it.
2. **PLSI.** Probabilistic latent semantic indexing over user-story clicks gives soft user-to-latent-class membership. The paper had to build a MapReduce version of the EM algorithm to fit it at Google scale.
3. **Co-visitation.** Two stories are related if the same user clicked both within a time window. These counts can be updated online as clicks arrive, which handles freshness much better than the batch models.

The final score is a **linear combination** of the three. Clusters and PLSI models are recomputed offline, while co-visitation counts and per-cluster click counts update in near-real time [19].

```
 Google News CF (2007)

 click logs --MapReduce--> MinHash user clusters ---+
            --MapReduce--> PLSI (EM) user classes ---+--> Bigtable stats
 live clicks --> co-visitation counts (online) ------+         |
                                                               v
 request(user) --> candidate stories --> score = w1*minhash + w2*plsi + w3*covisit
```

### Content-plus-trend model (Liu, Dolan and Pedersen, IUI 2010)

A follow-up added a **Bayesian model of each user's news interests by topic**. It separates a user's "genuine" topic interest from the local news trend, which is estimated from what everyone in the same country is clicking at that moment. This content-based score is combined with the existing CF score. In a live experiment the hybrid reportedly improved click-through on the recommended section by about 30% (approx.) and increased visits [20].

### Lessons

- Fast-decaying corpora need at least one signal that updates online (co-visitation, trending counts).
- Blending several cheap scorers linearly was a workable "ensemble" a decade before learned multi-source rankers.
- Modeling the population trend explicitly keeps personalization from fighting the news cycle.

## 17.2 The YouTube video recommendation system (2010)

### Problem

In 2010, YouTube's goal was to show personalized recommendations on the home page. Its corpus was huge, its metadata poor, most videos were short, and interactions were noisy and implicit [1].

### Architecture (Davidson et al., RecSys 2010)

```
 YouTube 2010 (batch-computed, served from Bigtable)

 watch logs (24h sessions) --MapReduce--> co-visitation counts c_ij
                                        --> related(v_i) = top-N by r(v_i, v_j)
                                                  r = c_ij / f(v_i, v_j)
 user seeds S = {watched, liked, favorited, playlisted videos}
          |
          v
 candidates C_n(S) = union of related-graph neighbours up to n hops
          |
          v
 rank by linear mix of: video quality | user specificity | diversification
          |
          v
 top ~4-60 recs (pre-computed several times a day; served with explanation)
```

- **Related videos.** For each video, co-visitation counts c_ij are computed within a session window (24 hours in the paper, approx.). The relatedness score divides c_ij by a normalization function f(v_i, v_j); the simplest choice is the product of the two videos' global popularity, which keeps very popular videos from being related to everything [1].
- **Candidate generation.** Start from a seed set of videos the user engaged with: watched, liked, favorited, added to playlists. Take the union of their related videos, then expand a limited number of hops through the related-video graph. Candidates further out add breadth (approx. description) [1].
- **Ranking.** Candidates are scored by a combination of video quality (view count, ratings, comments, favorites, shares, upload time), user specificity (how strongly the seed is tied to the user's interests) and diversification (for example limiting recommendations per seed or per channel) [1].
- **Serving.** Recommendations were pre-generated in batch through a series of MapReduce jobs over Bigtable, and refreshed several times a day. Each recommendation came with an explanation naming the seed video ("because you watched ...").

### Reported results (approx.)

The paper reported that recommendations accounted for about 60% of video clicks from the home page, and that co-visitation recommendations had a click-through rate around twice the "most viewed" baseline (about 207% of baseline, approx.) [1].

### Lessons

- Item-to-item co-occurrence over sessions is a strong, cheap first recommender. It is still a standard candidate source.
- Normalizing by popularity is required; raw co-counts collapse onto the head.
- Explanations come almost for free when recommendations are derived from seeds.
- Batch precomputation was enough at the time, but it is the opposite of the online freshness the next generation needed.

## 17.3 Wide & Deep Learning for Google Play (2016)

### Problem

Google Play needed app recommendations for over one billion active users and over one million apps [15]. Generalized linear models with hand-crafted cross-product features **memorize** well (for example "user installed Netflix AND impression is Pandora") but generalize poorly to unseen pairs. Embedding-based deep models **generalize** but over-generalize for niche users, recommending irrelevant apps when the user-item matrix is sparse and high-rank.

### Architecture

```
      wide (memorization)                     deep (generalization)
 sparse ids + cross products         sparse ids -> embeddings (~32-d each)
            |                                  + dense features
            |                                     |
            |                               MLP (ReLU layers, e.g. 1024-512-256)
            |                                     |
            +----------- sum of logits -----------+
                               |
                            sigmoid -> P(app acquisition)

 serving funnel: query -> retrieval (ML + rules, ~O(100) apps) -> Wide & Deep rank
```

- The wide part is a linear model on raw sparse features plus cross-product transformations, trained with **FTRL with L1 regularization**. The deep part maps categorical features to dense embeddings, concatenates them with continuous features, and passes them through a ReLU MLP, trained with **AdaGrad** [15].
- The two are **jointly trained**: their logits are summed before one sigmoid and one loss. That is different from an ensemble, in which the models are trained separately and combined only at inference.
- The label is **app acquisition** (install) for an impression.
- Retrieval first returns a short list of apps using a mix of learned models and rules. Wide & Deep only ranks that list [15].

### Scale and serving

The models were trained on over 500 billion examples. Each new model is **warm-started** from the previous one's embeddings and linear weights, so daily retraining does not start from scratch, and a sanity check runs before a model is pushed to serving. Scoring all candidates single-threaded took about 31 ms; splitting the batch across threads (batch of 50, 4 threads) brought client-side latency down to about 14 ms, against a budget of roughly 10 ms [15].

### Results

A three-week live A/B test showed **+3.9% app acquisitions** over a highly optimized wide-only control and about **+1%** over deep-only [15]. The implementation was open-sourced as a TensorFlow Estimator.

### Follow-on Google Play work

- **Mixed Negative Sampling (WWW 2020).** Two-tower retrievers trained with in-batch negatives never see items that were never engaged with. MNS adds a stream of negatives sampled uniformly from the whole corpus, shared across the batch. It was evaluated on Google Play app recommendations and beat batch-only sampling [16].
- **Self-supervised item towers (CIKM 2021).** Yao et al. add a contrastive auxiliary task. Two augmented views of an item's features are created with correlated feature masking and dropout, and pushed together in embedding space. This regularizes the item tower for long-tail items. The paper reports gains on Google-scale app recommendation [17].
- **DCN-V2 (WWW 2021, arXiv 2020).** DCN-V2 replaces Wide & Deep's hand-designed crosses with learned explicit crosses: x_{l+1} = x0 * (W_l x_l + b_l) + x_l with full-rank W_l. It then factorizes W into low-rank mixtures for cost. The paper reports offline and online gains "across many web-scale learning to rank systems at Google" [18].

### Lessons

Wide & Deep made "memorize with a wide path, generalize with a deep path" the default ranking template. Its weak point was the hand-crafted crosses in the wide part, which DeepFM, DCN and DCN-V2 then learned automatically.

## 17.4 Deep Neural Networks for YouTube Recommendations (2016)

### Problem

Covington, Adams and Sargin describe YouTube as "one of the largest scale and most sophisticated industrial recommendation systems in existence". They name three challenges: **scale** (existing algorithms that work on small problems fail), **freshness** (many hours of video uploaded per second, and new content should be balanced against well-established videos) and **noise** (sparse, implicit feedback with almost no ground-truth satisfaction labels) [2].

### Architecture

The system has two neural networks, following the "classic two-stage information retrieval dichotomy" [2]:

```
 YouTube 2016

 corpus (millions) ---> CANDIDATE GENERATION ---> hundreds ---> RANKING ---> dozens
                        user history + context       |          rich features,
 other candidate  ----> -> MLP tower -> user vec u    |          impression data,
 sources (earlier       train: sampled softmax        |          weighted LR on
 work, co-vis, etc.)    serve: ANN top-N by <u, v_j>  +--------> watch time
                                                                     |
                                                       A/B test on watch time/user
```

### Candidate generation

- **Framing.** Recommendation is **extreme multiclass classification**: predict which video, out of millions, the user watches next at time t, given the user U and context C [2].
- **Inputs.** The user's watch history is a variable-length bag of video IDs, each mapped to a learned embedding and **averaged**. Search history is handled the same way, tokenized into unigrams and bigrams. Demographic, geographic, device and gender features are added as dense or embedded inputs. The paper reports vocabularies of about 1M videos and 1M search tokens, embedded in 256 floats each, with up to 50 recent watches and 50 recent searches (approx.) [2][3].
- **Tower.** A "tower" of fully connected ReLU layers whose width halves at each step. Depths up to four layers (for example 2048 -> 1024 -> 512 -> 256) kept improving holdout MAP (approx.) [2].
- **Training.** **Sampled softmax** with several thousand negatives and importance weighting. Secondary summaries report a speedup of over 100x versus the full softmax [2][3].
- **Serving.** The softmax output layer is replaced by **nearest-neighbor search in dot-product space** over the learned video vectors, using the user vector as the query. The authors note that the choice of ANN algorithm did not much affect A/B results [2]. This "train as a classifier, serve as a MIPS lookup" pattern became the template for later two-tower retrieval (see Chapter 2).

### Key tricks in candidate generation

- **Example age.** Models trained on historical data learn the *average* watch likelihood over the training window and under-predict fresh viral videos. Adding the age of the training example as a feature, and setting it to zero (or slightly negative) at serving, lets the model learn the time-dependence of popularity and reproduce the sharp spike after upload [2].
- **Training on all watches**, including watches of YouTube videos embedded on other sites, not only on watches that came from recommendations. That limits feedback-loop exploitation and lets new content surface [2].
- **A fixed number of examples per user**, so a small group of highly active users does not dominate the loss [2].
- **Withholding information on purpose.** Search queries are fed as an unordered bag of tokens. Otherwise the model learns to recommend the results page of the user's last search, which users find annoying [2].
- **Predict the future, not a random held-out watch.** Consumption is asymmetric (episodes are watched in order; users explore broadly and then narrow). The training label is therefore the *next* watch, using only earlier history as input. Randomly held-out watches leak future information and did much worse in live tests [2].

### Ranking

- **Features.** Hundreds of features, most of them categorical and continuous signals about the user's past interaction with the video and similar items: when the user last watched the channel, how many videos from the channel they watched, and how often the video was already shown to them without being watched ("churn"). Impression-level data is available because only hundreds of candidates are scored [2].
- **Embedding sizes.** Each categorical feature gets its own embedding. Dimension grows roughly with the logarithm of the vocabulary size, and very large vocabularies are truncated to the top-N IDs by click frequency. Embeddings of the same ID space are shared across features (for example "impression video ID" and "last watched video ID") [2].
- **Continuous features** are normalized to [0, 1) using their cumulative distribution (quantiles). Powers such as x^2 and sqrt(x) are also fed in, so the network can model super- and sub-linear functions easily [2].
- **Objective: expected watch time, via weighted logistic regression.** Positive (clicked) impressions are weighted by their watch time and negatives get unit weight. The learned odds then approximate E[T](1 + P), which is close to E[T] when CTR P is small. At serving, exp(logit) is used as the predicted watch time [2]. Ranking by click probability promotes clickbait that users do not finish, so watch time is a better proxy for satisfaction.
- **Evaluation.** Offline, the paper uses "weighted, per-user loss" (the fraction of watch time mis-predicted when a watched and an unwatched impression are mis-ordered). Wider and deeper networks lowered this loss, with the best configuration in the paper's table reaching about 34.6% (approx.) [2]. Final decisions were made by live A/B tests on watch time per user.

### Scale

The paper and secondary summaries describe a model of roughly one billion parameters trained on hundreds of billions of examples (approx.) [2][3]. It was trained with Google Brain's distributed system (the successor DistBelief infrastructure and TensorFlow).

### Lessons

1. The two-stage funnel separates **recall at low cost** from **precision with rich features**, and each stage can have its own objective.
2. **Label and example engineering** (example age, per-user caps, future-watch labels, withholding the search sequence) mattered as much as architecture.
3. Choose the **objective** carefully. Moving from clicks to watch time was the first of several steps toward modeling satisfaction (see 17.8).

## 17.5 Two-tower retrieval with sampling-bias correction (2019)

### Problem

The 2016 candidate generator learned one output vector per video ID. That makes content features and cold start awkward, and the vocabulary must be fixed. Streaming training makes a fixed vocabulary even harder [4].

### Design

Yi et al. moved to a **two-tower** model. A query tower encodes user and context features, and a candidate tower encodes video features (ID plus content). Both produce embeddings scored by inner product [4].

```
  query tower  q_i = f(user_i, context_i)          batch i = 1..B
  item tower   v_j = g(video_j features)
  logits       s_ij = <q_i, v_j> / tau  -  log p_j        (logQ correction)
  loss         weighted softmax CE over in-batch candidates (weight = reward)
  p_j          streaming estimate: hashed arrays of "last step seen" and
               moving average of the gap between occurrences; p_j ~ 1 / gap
```

- **In-batch negatives** are cheap, but popular items appear as negatives more often, so they get over-penalized. Subtracting log p_j from each logit turns the in-batch softmax into an approximately unbiased estimate of the full softmax [4].
- **Streaming frequency estimation** without a fixed vocabulary: hashed arrays on parameter servers store the last global step at which each item was seen and a moving average of the step gap. The estimate of p_j is roughly the inverse gap [4].
- **Reward-weighted loss.** Each training pair can be weighted by a reward, for example watch completion, so that clicks with short watches count less [4].
- **Embedding normalization and temperature** were reported to improve training quality.

### Deployment

The model was deployed for YouTube as "Neural Deep Retrieval", retrieving from a corpus of tens of millions of videos, with live A/B gains [4]. Retrieval serving uses Google's quantization-based MIPS (the line of work later released as ScaNN [27]).

## 17.6 Recommending What Video to Watch Next: the multitask ranker (2019)

### Problem

The Watch Next ranker scores a few hundred candidates retrieved for a video the user is currently watching. It faces two problems [5]:

1. **Multiple competing objectives.** Engagement (clicks, watch time) and satisfaction (likes, dismissals, survey-style signals) are correlated but in tension.
2. **Implicit selection bias.** Users click on top-ranked items partly because they are on top. Training on that feedback reinforces the current ranker.

### Architecture

The model extends Wide & Deep with two ideas [5][6][7]:

```
 YouTube Watch Next ranker (2019)

 query video + candidate + user + context features
                 |
          shared hidden layer (cheaper than feeding raw inputs to every expert)
                 |
      +-------- MMoE: N experts --------+
      |   gate_k(x) = softmax(W_k x)    |    (one gate per objective)
      +--+--------+--------+-----------+
         |        |        |
   engagement  engagement  satisfaction ... task towers
     (click)   (watch)     (like, dismiss, rating)
         |        |        |
         +--------+--------+--> logits
                               + shallow-tower bias logit (training only)
 position, device ---> SHALLOW TOWER ------^   (position dropout ~10%, approx.)
                 |
 serving: position feature missing -> debiased per-task predictions
          -> weighted combination (weights tuned by online experiments) -> rank
```

- **MMoE.** Multi-gate Mixture-of-Experts [6] gives each objective its own softmax gate over a shared set of experts, so each task can choose how much to share. The experts sit on top of a shared hidden layer to keep serving cost down [5]. At the same model complexity, MMoE improved both engagement and satisfaction metrics over shared-bottom, in configurations with 4 and 8 experts (from the paper's tables, approx.) [5].
- **Shallow tower for position bias.** A small network takes features that explain selection bias, such as position and device, and outputs a bias logit that is **added** to each engagement task's logit during training. At serving the position feature is treated as missing, so the main model's prediction is the debiased utility. Position is dropped out during training (about 10%, per a secondary summary) so the model does not over-rely on it [5][7].
- **Fusion.** Per-objective predictions are combined with a weighted function whose weights are tuned manually for the best online engagement-satisfaction trade-off [5][7].
- **Training** is continuous on fresh logs ("training in sequence"), so the model follows the data distribution.

### Lessons

- Put **bias into the model structure**: a separate additive tower absorbs presentation effects that would otherwise contaminate the utility heads.
- Soft parameter sharing (MMoE) is the low-cost upgrade over shared-bottom when objectives conflict. Later work (PLE and others, Chapter 4) adds task-specific experts.
- The paper notes that MMoE gating can become very imbalanced (expert polarization), and that dropout on experts helps.

## 17.7 Reinforcement learning at YouTube: REINFORCE and SlateQ (2019)

### Top-K Off-Policy Correction for a REINFORCE Recommender (WSDM 2019)

**Problem.** Supervised candidate generators imitate the logs of earlier recommenders, which reinforces their biases. Chen et al. recast YouTube's candidate generator as a **policy** pi(a | s) over millions of videos, trained with REINFORCE to maximize cumulative reward over the user's trajectory [8].

```
 user event sequence (watches, context) --> RNN state s_t
                                              |
               +------------------------------+-------------------+
               v                                                  v
      pi_theta(a|s): softmax over               beta_hat(a|s): behavior-policy head
      millions of videos (deployed)             (gradient blocked into shared state)
               |                                                  |
               +---------- importance weight  pi / beta_hat ------+
                                     |
                lambda_K = K * (1 - pi(a|s))^(K-1)     (top-K correction)
                                     |
          REINFORCE update: sum_t  w_t * lambda_K * R_t * grad log pi(a_t|s_t)
          + weight capping, normalized IS, TRPO-style constraint
```

Key tricks [8]:

1. **Learned behavior policy.** Logs come from a mixture of past systems whose propensities were not logged. A second softmax head estimates beta_hat(a | s) from the same state, with gradients blocked so it does not distort the main encoder.
2. **Top-K correction.** YouTube shows K items at once. The probability that item a is in the slate is 1 - (1 - pi(a))^K, which adds the multiplier lambda_K = K (1 - pi(a))^(K-1) to the gradient. Unlikely items get pushed up strongly; items that are already likely get little extra push, which spreads probability mass across a good set rather than one item.
3. **Variance control.** Weight capping, self-normalized importance sampling, and a TRPO-like trust region.
4. **Exploration.** Serving returns the top K' items and samples the remaining K - K' from the policy (Boltzmann-style exploration), which keeps the logs informative.

**Results.** Live experiments showed statistically significant gains in viewing time, around 0.8-0.9% (approx.) for the top-K correction, which the authors call one of the biggest single launches in recent years. More long-tail items were also surfaced [8].

**Follow-ups.** Reward shaping imputes sparse satisfaction signals, such as survey responses, onto unlabeled interactions and uses them as REINFORCE rewards [10]. RecSim is an open-source simulator for testing slate-based RL recommenders before they reach users [11].

### SlateQ (IJCAI 2019)

Value-based RL over slates is combinatorial. Ie et al. show that under two assumptions, that the user consumes at most one item and that reward and transition depend only on the consumed item, the slate's Q-value decomposes as Q(s, A) = sum over i in A of P(i | s, A) * Q_bar(s, i). Here P is a user choice model, such as a conditional logit, and Q_bar is an item-level long-term value learned with SARSA/TD. Slate optimization becomes a linear program, or a greedy top-k approximation that works almost as well [9]. SlateQ was tested in live YouTube experiments in which item-level long-term values replaced the myopic scores in the existing ranking, with reported gains in user engagement (approx.) [9].

### Lessons

- RL for recommendation became practical by **keeping the deployed architecture** (softmax over items, served by ANN) and changing only the learning signal and corrections.
- Off-policy correction and slate-aware gradients matter more than the choice of RL algorithm.
- Long-term rewards need careful design; reward shaping from surveys is how YouTube tied RL to satisfaction.

## 17.8 Diversity, fairness and the "valued watch time" objective

### DPP diversification in the home feed (CIKM 2018)

Pointwise rankers fill the feed with near-duplicates. Wilhelm et al. re-rank the top candidates with a **determinantal point process** whose kernel is L_ii = q_i^2 and L_ij = alpha * q_i * q_j * exp(-D_ij / (2 sigma^2)), with q the ranker's quality score and D_ij a distance between videos (from embeddings or topics). Greedy MAP inference is run **window by window**, where a window is roughly the items visible on a phone screen, to keep latency bounded. Alpha and sigma were first grid-searched against online metrics; later the paper learns a deep kernel from feed interactions. The DPP beat earlier heuristic rules such as per-channel caps on long-term satisfaction metrics [12].

### Pairwise fairness (KDD 2019)

Beutel et al. define a pairwise fairness metric for rankers (does an item from group A that the user engaged with outrank an unengaged item from group B as often as the reverse?), estimated from randomized experiment data. They add a pairwise regularizer to training and report improved fairness with little loss of accuracy in a large-scale production recommender at Google [13].

### From clicks to watch time to valued watch time (2021)

YouTube's public explanation of its recommender, written by Cristos Goodrow, VP of Engineering, describes three objective eras [14]:

1. **Clicks.** These encouraged clickbait.
2. **Watch time** (from 2012). A better proxy, but not every minute watched is a good minute.
3. **"Valued watch time"** measured with **user surveys** ("How satisfied were you with this video?", rated 1-5 stars). Only a fraction of views get a survey, so a model is trained to predict survey answers for all watches, and that prediction becomes part of the objective. Likes, dislikes, shares and "not interested" signals are also used.

The same post describes **authoritative-content boosting** for news and information and **demotion of borderline content**, using human raters following published guidelines. It reports that consumption of borderline content from recommendations dropped by about 70% in the US after the 2019 changes (approx.) [14]. The YouTube "How YouTube Works" pages present this as the policy view of the same system [29].

### Lesson

Objective design is iterative and partly a policy decision. YouTube treats it as a product of models (survey predictors, quality classifiers) and explicit policy layers, not as a single label.

## 17.9 TIGER: generative retrieval with semantic IDs (NeurIPS 2023)

### Problem

Two-tower retrieval needs one embedding per item and an ANN index. Atomic IDs carry no meaning, cold-start items have no trained vector, and the item-embedding table grows with the corpus [23].

### Architecture

```
 TIGER (Rajput et al., Google, 2023)

 item text/metadata -> Sentence-T5 (768-d, approx.) -> RQ-VAE
     level 1 code c1 (256 options)  -> residual r1
     level 2 code c2 (256 options)  -> residual r2
     level 3 code c3 (256 options)  [+ collision token c4]
 Semantic ID(item) = (c1, c2, c3, c4)

 user history: SID(i1) SID(i2) ... SID(in)  (+ user id token, hashed)
          |
   Transformer encoder-decoder (T5-style, small)
          |
   beam search decodes the next item's SID tuple -> lookup -> items
```

- **Semantic IDs** are coarse-to-fine: items that share a prefix are semantically similar, so parameters are shared across related items through the shared code embeddings. Collisions (identical tuples) are resolved by an extra disambiguation token [23].
- **Generative retrieval.** A sequence-to-sequence Transformer reads the user's history as a token sequence and **autoregressively generates the next item's semantic ID**. The model itself acts as the index; no ANN is needed at serving [23].
- **Results.** On Amazon Beauty, Sports and Toys, TIGER beat SASRec, S3-Rec, P5 and other baselines on Recall@K and NDCG@K. The paper reports gains in cold-start retrieval and controllable diversity by sampling at the first code level [23].
- **Caveats.** Beam search can generate invalid IDs (rare in practice). Decoding cost grows with beam width and ID length. Later work identified codebook imbalance and "hourglass" collapse in intermediate levels (Chapter 6).

### Semantic IDs inside the YouTube ranker (RecSys 2024)

Singh et al. ask whether semantic IDs can replace video-ID embeddings in a production ranking model. They derive semantic IDs from a frozen content encoder (a VideoBERT-style audio-visual model, approx.) with an RQ-VAE, then compare ways to map them to embeddings: n-gram lookups over code prefixes versus **SentencePiece-style learned subword tokenization** of the codes. SentencePiece-based semantic IDs matched the memorization ability of random-hashed video IDs while generalizing better to new and long-tail videos, with a fixed embedding-table budget [24].

### Lessons

- Treat the item tokenizer as infrastructure: content encoder, quantizer, collision handling, and a refresh policy.
- Semantic IDs help even if you never adopt generative retrieval. They are a better feature for ranking embeddings at a fixed table size.

## 17.10 The LLM / Gemini era: PLUM and user-journey modeling (2023-2026)

### LLM user-interest journeys (2023)

Christakopoulou et al. used LLMs to extract and **name user interest journeys** (for example "learning to bake sourdough") from a user's activity, using prompt tuning and fine-tuning on a small set of human-labeled journeys. The paper argues that journey-level interest representations could make recommendations more explainable and better aligned with users' long-term goals. It is research rather than a reported production launch [26].

### PLUM: Pre-trained Language Models for industrial-scale generative recommendation (2025)

PLUM is Google's most detailed public description of LLM-based recommendation at YouTube [25]. As described in the paper (details approx.):

```
 PLUM (Google / YouTube, 2025)

 1. Item tokenization: Semantic IDs v2
    multimodal content embeddings (+ co-engagement signal) -> RQ-style
    multi-level codes; tuned for coverage and low collision at YouTube scale
 2. Continued pre-training (CPT) of a pre-trained Gemini-family LLM
    on a mixture of: user watch sequences as SID tokens, SIDs aligned with
    text (titles, topics), so the model "speaks" SIDs and language
 3. Task fine-tuning: generative retrieval
    input: user context + history (SIDs + text) -> decode SIDs of next videos
 4. Serving: beam decoding of SIDs -> videos, as a retrieval source
    feeding the existing ranking stack
```

- The item vocabulary is semantic IDs rather than atomic video IDs, so a frozen-size vocabulary can cover a corpus of billions of videos and new uploads.
- Continued pre-training aligns the LLM's language knowledge with the SID "language" before fine-tuning for retrieval.
- The paper reports that the generative retriever beat a heavily optimized production large-embedding retrieval model in live experiments on YouTube surfaces, and studies scaling with model size and compute (approx.) [25].
- It also argues that this design uses far fewer item-specific parameters than embedding-table retrieval, at the price of much more compute per request [25].

### What changed versus 2016

| Dimension | 2016 DNN candidate gen [2] | 2019 two-tower [4] | 2025 PLUM [25] |
|---|---|---|---|
| Item representation | learned per-ID output vector | tower over ID + content features | semantic-ID tokens from content and co-engagement |
| User representation | averaged watch/search embeddings | query tower over features | LLM context over SID and text tokens |
| Training | sampled softmax | in-batch softmax + logQ | LLM continued pre-training + fine-tuning |
| Serving | ANN over item vectors | ANN (quantized MIPS) | beam decoding of SIDs |
| Cold start | weak (needs ID history) | content features help | content-derived SIDs; shared prefixes |
| Cost profile | memory-heavy tables, cheap compute | moderate | small tables, heavy compute (TPU) |

### YouTube Shorts

Public technical detail about the Shorts recommender is limited. YouTube's own help and "How YouTube Works" pages say that the same principles apply: personalization from watch and engagement history, satisfaction signals (likes, skips, "not interested", surveys) and quality and responsibility policies [14][29]. Short-form feeds differ from long-form in ways covered elsewhere in this Atlas: watch time must be normalized by duration (Chapter 11, Kuaishou D2Q), skips are a dense negative label, and the swipe feed is a sequential decision process, which favors the REINFORCE- and slate-style learning in 17.7. The PLUM paper's experiments reportedly include short-form surfaces (approx.) [25]. Beyond that, treat any specific Shorts architecture claim found in secondary sources as unverified.

### Google Discover

Google Discover, the feed in the Google app and on mobile google.com, was announced in 2018 as a rebrand of the Google Feed. At that time Google said it reached more than 800 million monthly users (approx.) [21]. Its design as described publicly [21][22]:

- It is a **query-less** feed built from the user's Search and app activity and their declared follows, and it uses the **Knowledge Graph "Topic Layer"** to model interests and how a user's expertise in a topic develops over time.
- It includes evergreen content as well as fresh news, with controls such as "follow topic" and "not interested", which act as explicit feedback.
- The publisher-side documentation says content becomes eligible automatically if it is indexed and meets content policies, and that large, high-quality images (at least 1200 px wide, with max-image-preview:large) increase engagement [22].

No public paper describes Discover's model stack. Its building blocks are presumably those in this chapter, but that is an inference, not a documented fact.

## 17.11 Infrastructure threads

Several infrastructure papers sit underneath these systems:

- **Large online learners.** Google's 2013 ad-click paper popularized FTRL-Proximal, per-coordinate learning rates and probabilistic feature inclusion for sparse models at very large scale. These ideas are visible in the wide part of Wide & Deep [31][15].
- **Vector search.** ScaNN's anisotropic quantization targets inner-product accuracy for top-k MIPS, the retrieval primitive for every two-tower model above [27].
- **Hardware.** TPU v4 includes SparseCores, dedicated hardware for embedding lookups. Google reports that embeddings are a major share of recommendation workloads and that SparseCores accelerate embedding-heavy models by several times relative to running them on the TensorCores alone (approx.) [28].
- **Engineering practice.** Google's "Rules of Machine Learning" codifies lessons such as logging serving features to avoid training-serving skew and starting with simple heuristics and models [30].

## 17.12 Comparison table

| System (year) | Surface | Stage | Core model | Objective / label | Public scale (approx.) | Signature trick |
|---|---|---|---|---|---|---|
| Google News CF (2007) [19] | News | retrieval + score | MinHash, PLSI, co-visitation | clicks | millions of users | online co-visitation for freshness |
| YouTube 2010 [1] | Home | full stack | co-visitation graph | clicks | ~60% of home clicks from recs | seed-set graph expansion |
| Wide & Deep (2016) [15] | Play Store | ranking | linear + MLP, joint | app install | 1B+ users, 1M+ apps, 500B+ examples | memorization + generalization |
| YouTube DNN (2016) [2] | Home | retrieval + ranking | MLP tower + sampled softmax; MLP ranker | next watch; watch time | ~1B params, 100s of billions of examples | example age; weighted LR |
| Two-tower NDR (2019) [4] | YouTube | retrieval | two-tower | reward-weighted softmax | tens of millions of videos | streaming logQ correction |
| Watch Next MTL (2019) [5] | Watch Next | ranking | MMoE + shallow tower | engagement + satisfaction | hundreds of candidates | additive position-bias tower |
| REINFORCE (2019) [8] | YouTube | retrieval | RNN softmax policy | long-term reward | millions of actions | top-K off-policy correction |
| SlateQ (2019) [9] | YouTube | ranking | choice model x item Q | long-term engagement | live experiments | slate value decomposition |
| DPP feed (2018) [12] | Home (mobile) | re-ranking | DPP greedy MAP | long-term satisfaction | windowed feed | learned quality/diversity kernel |
| DCN-V2 (2020) [18] | many Google LTR systems | ranking | cross network + DNN | task-specific | high QPS | low-rank mixture of crosses |
| TIGER (2023) [23] | research | retrieval | RQ-VAE + seq2seq | next item | public datasets | semantic IDs |
| SID ranking (2024) [24] | YouTube | ranking | SID embeddings | ranking tasks | production-scale | SentencePiece over codes |
| PLUM (2025) [25] | YouTube | retrieval | Gemini-family LLM | next watch (generative) | billions of videos | CPT on SIDs + text |

## Key takeaways for system designers

1. **The funnel is durable; the components are not.** Google's recommenders kept the same shape from 2016 to 2025: retrieve hundreds from millions or billions, rank with richer features, re-rank for diversity. What changed is each stage's model: co-counts, then DNNs, then two-towers, then RL policies, then LLM-based generative retrieval.
2. **Example and label engineering pays off before architecture.** Example age, per-user caps, predicting the future watch, withholding search order, and reward-weighted losses gave large wins with simple models [2][4].
3. **Choose objectives deliberately and keep revisiting them.** Clicks, then watch time, then survey-predicted valued watch time, plus explicit quality and responsibility policies [14]. Build survey-prediction models so that sparse satisfaction labels can cover all traffic.
4. **Put bias handling in the model.** Use logQ for sampling bias [4], an additive shallow tower for position bias [5], off-policy and top-K corrections for logging-policy bias [8], and uniform negatives for selection bias [16].
5. **Soft sharing for conflicting objectives.** MMoE with per-task gates plus a tunable fusion layer is the baseline multi-objective ranker; watch for gate polarization [5][6].
6. **Make RL compatible with existing serving.** REINFORCE kept the softmax-plus-ANN serving path; SlateQ reused the ranker with long-term values. Both relied on careful variance control and live experimentation [8][9].
7. **Diversity is a set property.** Model it with a set-level re-ranker (DPP), tuned or learned against long-term metrics, not with ad hoc caps [12].
8. **Tokenize items.** Semantic IDs improve generalization within ranking [24], allow generative retrieval [23], and are the bridge that lets pre-trained LLMs work over a catalog of billions of items [25]. Plan for collisions, codebook balance and refresh.
9. **Trade memory for compute knowingly.** Embedding tables (2016-2020) were memory-bound and needed special hardware (SparseCore [28]). LLM-based generative retrieval (2025) is compute-bound. Budget serving cost, latency and caching accordingly.
10. **Ship through A/B tests on long-term metrics.** Nearly every result here was decided online, and Google's long-term experiment methodology warns that short tests can mislead [32].

## References

1. James Davidson, Benjamin Liebald, Junning Liu, Palash Nandy, Taylor Van Vleet, Ullas Gargi, Sujoy Gupta, Yu He, Mike Lambert, Blake Livingston, Dasarathi Sampath (Google/YouTube). "The YouTube Video Recommendation System." RecSys 2010. https://dl.acm.org/doi/10.1145/1864708.1864770
2. Paul Covington, Jay Adams, Emre Sargin (Google/YouTube). "Deep Neural Networks for YouTube Recommendations." RecSys 2016. https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/ (PDF: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
3. Adrian Colyer. "Deep neural networks for YouTube recommendations" (paper summary). The Morning Paper, 2016. https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/
4. Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, Ed Chi (Google/YouTube). "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." RecSys 2019. https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
5. Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, Ed Chi (Google/YouTube). "Recommending What Video to Watch Next: A Multitask Ranking System." RecSys 2019. https://dl.acm.org/doi/10.1145/3298689.3346997
6. Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed H. Chi (Google). "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." KDD 2018. https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
7. KDnuggets. "How YouTube is Recommending Your Next Video." 2019 (secondary summary of [5]). https://www.kdnuggets.com/2019/10/youtube-recommending-next-video.html
8. Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, Ed H. Chi (Google/YouTube). "Top-K Off-Policy Correction for a REINFORCE Recommender System." WSDM 2019. https://arxiv.org/abs/1812.02353
9. Eugene Ie, Vihan Jain, Jing Wang, Sanmit Narvekar, Ritesh Agarwal, Rui Wu, Heng-Tze Cheng, Tushar Chandra, Craig Boutilier (Google). "SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets." IJCAI 2019. https://arxiv.org/abs/1905.12767
10. Konstantina Christakopoulou, Can Xu, Sai Zhang, et al. (Google). "Reward Shaping for User Satisfaction in a REINFORCE Recommender." 2022. https://arxiv.org/abs/2209.15166
11. Eugene Ie, Chih-wei Hsu, Martin Mladenov, et al. (Google Research). "RecSim: A Configurable Simulation Platform for Recommender Systems." 2019; repository. https://github.com/google-research/recsim
12. Mark Wilhelm, Ajith Ramanathan, Alexander Bonomo, Sagar Jain, Ed H. Chi, Jennifer Gillenwater (Google/YouTube). "Practical Diversified Recommendations on YouTube with Determinantal Point Processes." CIKM 2018. https://dl.acm.org/doi/10.1145/3269206.3272018
13. Alex Beutel, Jilin Chen, Tulsee Doshi, Hai Qian, Li Wei, Yi Wu, Lukasz Heldt, Zhe Zhao, Lichan Hong, Ed H. Chi, Cristos Goodrow (Google). "Fairness in Recommendation Ranking through Pairwise Comparisons." KDD 2019. https://arxiv.org/abs/1903.00780
14. Cristos Goodrow (YouTube). "On YouTube's recommendation system." YouTube Official Blog, 2021. https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/
15. Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, et al. (Google). "Wide & Deep Learning for Recommender Systems." DLRS 2016. https://arxiv.org/abs/1606.07792
16. Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, et al. (Google). "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations." WWW 2020 Companion. https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/
17. Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, et al. (Google). "Self-supervised Learning for Large-scale Item Recommendations." CIKM 2021. https://arxiv.org/abs/2007.12865
18. Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi (Google). "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems." WWW 2021. https://arxiv.org/abs/2008.13535
19. Abhinandan Das, Mayur Datar, Ashutosh Garg, Shyam Rajaram (Google). "Google News Personalization: Scalable Online Collaborative Filtering." WWW 2007. https://dl.acm.org/doi/10.1145/1242572.1242610
20. Jiahui Liu, Peter Dolan, Elin Ronby Pedersen (Google). "Personalized News Recommendation Based on Click Behavior." IUI 2010. https://dl.acm.org/doi/10.1145/1719970.1719976
21. Google (Karen Corbin). "Introducing Google Discover." The Keyword (Google blog), 2018. https://blog.google/products/search/introducing-google-discover/
22. Google Search Central. "Google Discover and your website" (documentation). Google, accessed 2026. https://developers.google.com/search/docs/appearance/google-discover
23. Shashank Rajput, Nikhil Mehta, Anima Singh, et al. (Google). "Recommender Systems with Generative Retrieval" (TIGER). NeurIPS 2023. https://arxiv.org/abs/2305.05065
24. Anima Singh, Trung Vu, Nikhil Mehta, et al. (Google/YouTube). "Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations." RecSys 2024. https://arxiv.org/abs/2306.08121
25. Google / YouTube. "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations." 2025. https://arxiv.org/abs/2510.07784
26. Konstantina Christakopoulou, Alberto Lalama, Cj Adams, et al. (Google). "Large Language Models for User Interest Journeys." 2023. https://arxiv.org/abs/2305.15498
27. Google Research. "Announcing ScaNN: Efficient Vector Similarity Search." Google Research Blog, 2020. https://research.google/blog/announcing-scann-efficient-vector-similarity-search/
28. Norman P. Jouppi, et al. (Google). "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings." ISCA 2023. https://arxiv.org/abs/2304.01433
29. YouTube. "How YouTube Works: Recommendations." YouTube, accessed 2026. https://www.youtube.com/howyoutubeworks/product-features/recommendations/
30. Martin Zinkevich (Google). "Rules of Machine Learning: Best Practices for ML Engineering." Google for Developers. https://developers.google.com/machine-learning/guides/rules-of-ml
31. H. Brendan McMahan, Gary Holt, D. Sculley, et al. (Google). "Ad Click Prediction: a View from the Trenches." KDD 2013. https://dl.acm.org/doi/10.1145/2487575.2488200
32. Henning Hohnhold, Deirdre O'Brien, Diane Tang (Google). "Focusing on the Long-term: It's Good for Users and Business." KDD 2015. https://dl.acm.org/doi/10.1145/2783258.2788583
