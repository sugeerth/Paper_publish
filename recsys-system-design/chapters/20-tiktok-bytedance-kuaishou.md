# 20. TikTok/ByteDance and Kuaishou: Real-Time Short-Video Recommendation, from Monolith to OneRec

Short-video feeds are the hardest everyday recommendation workload. A user may swipe through dozens of items in a few minutes. Every swipe is an implicit label: watch time, completion, replay, skip, like, comment, share, follow. The corpus churns by the hour, and the product has a single full-screen slot, so every exposure has to count. The two companies that defined this category, ByteDance (Douyin in China, TikTok elsewhere) and Kuaishou (Kuaishou and Kuaishou Lite), have published more system detail than almost any other consumer platform. ByteDance's work is mostly about infrastructure and ranking-model scaling. Kuaishou's is mostly about long-sequence modelling, the multi-stage cascade, and, since 2025, replacing that cascade with one generative model.

This chapter covers both companies:

- **ByteDance / TikTok.** What the company has said publicly about the For You feed, the reported "TikTok Algo 101" internal document, Monolith real-time training, Deep Retrieval, LONGER, RankMixer and TokenMixer-Large, OneTrans and HLLM.
- **Kuaishou.** The six-stage cascade (RecFlow), PERSIA training, TWIN and TWIN V2 lifelong sequences, KuaiFormer retrieval, multi-task and fusion work (HoME, Pantheon, RLUR), watch-time debiasing (D2Q), and the OneRec family (V1, technical report, V2, Think, OpenOneRec).

A note on sourcing. The live web-search budget for this run was used up before research could start, and outbound access to arXiv and most company sites failed from the writing environment. This chapter therefore relies on three kinds of source. The first is primary sources that could be fetched (GitHub repositories of ByteDance and Kuaishou). The second is technical notes that summarise the papers in detail (the fun-rec-mdbook chapters). The third is primary-source facts and arXiv identifiers already checked by earlier chapters of this Atlas (Chapters 1, 3, 5, 6, 9, 11, 13, 14 and 15). The two TikTok public disclosures (the 2020 newsroom post and the 2021 New York Times report) could not be re-fetched. They are described from the public record and marked as such. Numbers are as reported, and any that come from secondary notes are marked approximate.

---

## 20.1 The short-video recommendation problem

Short-video feeds differ from long-form video (Chapter 17) and photo feeds (Chapter 19) in several ways that drive the designs below:

1. **One item at a time, very many items per session.** Each request returns a small batch, and the feed then prefetches more. Kuaishou's funnel ends with roughly 6 to 10 exposed items per request [23][24]. So request QPS is high relative to the number of users, and the ranking decision is close to a sequential decision problem.
2. **Watch time is the main label, and it is biased.** Longer videos mechanically collect more watch time. Kuaishou's D2Q paper predicts watch-time quantiles within duration buckets to remove that confound [30]. OneRec-V2 log-buckets watch time by duration for the same reason [19][22].
3. **Extreme freshness.** New videos must be learnable within minutes, and interests drift within a session. ByteDance's Monolith exists because daily batch retraining and hashed embedding tables could not keep up [1][2].
4. **Very long histories.** Heavy users produce 10^4 to 10^6 behaviours over their lifetime. Kuaishou reports lifelong sequences beyond 10^5 behaviours, which is what motivates TWIN V2 [14] and OneRec's 100,000-record lifelong pathway [22].
5. **Retention is the business objective.** Both companies optimise immediate engagement as a proxy for long-term return. The reported TikTok internal document names retention and time spent as the ultimate goals [12]. Kuaishou has published RL work that targets return time directly [28].

---

## 20.2 ByteDance / TikTok: what the company has disclosed

### 20.2.1 The 2020 "How TikTok recommends videos #ForYou" post

In June 2020 TikTok published a newsroom post describing the For You feed [11]. (The post could not be re-fetched for this chapter; this summary follows the widely quoted public text.) It describes the feed as a recommendation system that ranks videos on a combination of factors:

- **User interactions.** Videos liked or shared, accounts followed, comments posted, and content created.
- **Video information.** Captions, sounds and hashtags.
- **Device and account settings.** Language preference, country and device type. The post says these get lower weight than the other signals, because users do not actively express them.
- **Signal strength.** Each factor is weighted by its strength as an indicator of interest. The post gives the example that finishing a longer video is a stronger signal than whether the viewer and creator are in the same country.
- **Not direct factors.** Follower count and a creator's previous high-performing videos are not direct ranking factors, though accounts with more followers naturally collect more of the other signals.
- **Diversity rules.** The feed is designed not to show two videos in a row from the same creator or with the same sound. It also inserts videos that do not obviously match a user's stated interests, so that the feed does not become a filter bubble.
- **Cold start.** New users can select interest categories at sign-up, and early interactions quickly replace those defaults.

The design lesson is that TikTok publicly frames the system as a weighted multi-signal ranker with an explicit diversity and exploration layer. This matches the generic funnel of Chapter 1 and the value-model pattern of Chapter 4.

### 20.2.2 The reported "TikTok Algo 101" document (2021)

In December 2021 the New York Times reported on an internal TikTok document called "TikTok Algo 101", which it said was produced by the Beijing engineering team for non-technical staff [12]. (The article could not be re-fetched; details below follow the widely cited reporting and should be treated as reported, not confirmed by ByteDance.) Key points as reported:

- The company's stated "ultimate goal" is adding daily active users, optimised through two closely related metrics: **retention** and **time spent**.
- The ranking score is described as a simplified weighted sum of predicted engagement:

```
  score = P_like * V_like + P_comment * V_comment
        + E_playtime * V_playtime + P_play * V_play
  P_*: predicted probability of the action      E_playtime: expected play time
  V_*: value weight for that action (set by product policy)
```

- The document reportedly also discussed suppressing repetitive content and preserving creator diversity, since repetitive feeds can bore users.

This is exactly the "value model" pattern documented at Meta, X and Kuaishou in Chapter 4: a multi-task model predicts several engagement probabilities, and hand-set (later learned) weights fuse them into one score. Kuaishou's papers describe the same stage as "ensemble sort" [27].

### 20.2.3 Regulatory context (2025-2026)

Because of US national-security concerns, TikTok's US recommendation algorithm became a subject of legislation and of the US joint-venture arrangement in 2025-2026. Under that arrangement the algorithm was reported to be retrained on US user data under the oversight of a US partner. No technical design detail about the retrained system had been published in sources available to this chapter, so it is mentioned only as context. Architecturally, the published ByteDance papers below are the best public evidence of how the stack works.

---

## 20.3 Monolith: real-time training with collisionless embeddings (ByteDance, 2022)

### Problem

ByteDance's feed and ads products have three properties that clash with standard TensorFlow training [1][2]:

- The sparse ID space (users, videos, creators, crosses) is **non-stationary and unbounded**.
- Feature frequency is **long-tailed**.
- **Concept drift** happens within hours.

TensorFlow's built-in variables are fixed-size dense tensors. Fitting unbounded IDs into them requires hashing, and hash collisions degrade quality. The paper also argues that conventional frameworks keep batch training and serving "completely separated", so models cannot react to user feedback in real time [1].

### Architecture

**Collisionless embedding table.** Monolith replaces `tf.Variable` tables with a key-value hash table on the parameter servers, built on **Cuckoo hashing** [1]. Cuckoo hashing uses two hash functions and two arrays. On insert, an occupant is evicted to its alternate slot until every key has a place. Lookups are O(1), and distinct IDs never share a row. The repository describes the goal as giving each ID feature "its own unique embedding representation" [2]. Two policies bound memory [1]:

- **Frequency filtering (admission).** An ID gets an embedding only after it has appeared a threshold number of times. A probabilistic filter keeps the counting cost low.
- **Expirable embeddings (eviction).** IDs inactive past an expiry time, such as stale users or deleted videos, are removed.

**Online training loop.**

```
 client actions (play, like, skip...) --> Kafka: action log --+
 serving-time feature log ------------> Kafka: feature log --+
                                                              |
                         Flink online joiner (join by request key;
                         in-memory cache + on-disk KV for late actions;
                         negative sampling)
                                                              |
                      +---------------------------------------+-----------+
                      v                                                   v
           Kafka: training examples                          HDFS (batch dumps)
                      |                                                   |
         online training workers <---------- same model code -----> batch training
                      |
         training PS (Cuckoo hash tables, admission + expiry)
                      |  sparse params: incremental sync of touched IDs (minutes)
                      |  dense params: synced far less often
                      v
         serving PS --> online inference for the feed
```

How it works [1][2]:

1. **One framework, two modes.** Batch training on historical HDFS data bootstraps the model. Online training then consumes the streaming example queue indefinitely.
2. **Online joiner.** A Flink job joins user actions with the features logged at serving time. Actions can arrive minutes after the impression, so features wait in a memory cache backed by on-disk storage. Negatives are down-sampled, and a log-odds correction at serving time restores calibration.
3. **Asymmetric parameter sync.** Pushing terabytes of parameters to serving every minute is impractical. Sparse embeddings are synced **incrementally**: only IDs touched since the last sync are sent, at minute-level intervals. Dense weights change slowly under momentum-style optimisers, so they are synced much less often.
4. **Cheap fault tolerance.** The paper finds that daily parameter-server snapshots lose very little quality on a PS failure. Each shard holds a small fraction of IDs, and online training recovers quickly, so frequent snapshots are not worth their I/O cost.

### Results and lessons

- Collisionless tables beat hashed (colliding) tables on public and internal data [1].
- Shorter sync intervals give consistently higher online AUC, and real-time training beats batch training in the ads experiments [1].
- The open-source repository is TensorFlow plus Bazel, with demos for distributed asynchronous training on Kubernetes via Primus. It was archived (read-only) by ByteDance in October 2025 [2].

The main lessons: **freshness is a model-quality feature**, the sparse/dense asymmetric sync is what makes minute-level updates affordable, and admission and eviction policy matter as much as the table data structure.

---

## 20.4 ByteDance retrieval: Deep Retrieval and LLM-based item/user models

### Deep Retrieval (2020)

Deep Retrieval replaces ANN over a fixed embedding space with a **learned discrete structure** [3]. Each item maps to one or more paths through a D-layer lattice with K nodes per layer, so every item gets a code of D tokens from K values. Path assignments are model parameters, learned jointly with the user-to-path model by expectation-maximisation. Serving runs beam search over paths, then scores the items on the retrieved paths. ByteDance reports near brute-force accuracy on public data at sub-linear cost, and online engagement gains over a tuned ANN baseline [3]. This is an early, production-tested ancestor of the semantic-ID generative retrieval that Kuaishou later scaled in OneRec.

```
  user features --> layer-1 softmax over K --> layer-2 (cond.) --> ... --> layer-D
                         beam search keeps top-B partial paths at each layer
  retrieved paths --> items assigned to those paths --> rerank with a scorer
```

### HLLM (2024)

HLLM (Hierarchical Large Language Model) is ByteDance research code that uses two LLMs [8]:

- An **item LLM** reads item text (title, tags, description) and produces an item embedding.
- A **user LLM** consumes the sequence of item embeddings to predict the next item.

Released checkpoints are HLLM-1B (from TinyLlama-1.1B) and HLLM-7B (from Baichuan2-7B). On the Books benchmark with 28,672 negatives, HLLM-1B reaches R@10 9.28 and HLLM-7B 9.39, against 6.50 for HSTU-large. On Pixel8M, HLLM-1B reaches R@10 6.13 against 5.08 (SASRec) and 4.83 (HSTU) [8]. A follow-up, HLLM-Creator (2025), extends the approach to personalised creative generation [8]. The repository is also archived [8]. Lesson: text-derived item representations help on sparse and cold items, at the cost of LLM-scale compute in the item tower.

---

## 20.5 ByteDance ranking: long sequences and hardware-efficient scaling

### 20.5.1 LONGER: end-to-end long sequences (RecSys 2025)

**Problem.** Two-stage behaviour retrieval (SIM, TWIN) is lossy: the cheap first stage can drop behaviours that the final model would value. ByteDance wanted end-to-end Transformer modelling of sequences up to about 10,000 behaviours inside the ranker [4].

**Design** [4][5]:

- **Global tokens** (target item, user summary tokens) attend over the full context and stabilise attention.
- **Token merge.** Adjacent behaviours are grouped into fewer tokens, with a light "InnerTransformer" preserving detail within each group.
- **Hybrid attention.** A small set of query tokens cross-attends over the long sequence, then self-attention runs on the short output, which cuts the quadratic cost.
- **Engineering.** Mixed precision, activation recomputation, and **KV-cache serving**: the user-side sequence is encoded once per request and shared across all candidates.

**Results.** Consistent offline and online gains in advertising and e-commerce. The paper says LONGER is fully deployed in dozens of ByteDance scenarios serving billions of users [4].

### 20.5.2 RankMixer (2025) and TokenMixer-Large (2026)

**Problem.** Years of hand-designed feature-interaction modules left ByteDance's rankers with very poor hardware efficiency. RankMixer reports about **4.5% model FLOPs utilisation (MFU)** on GPUs before the redesign [6]. Technical notes attribute the low MFU to memory-bound embedding lookups (about 40% of time), many small kernels (about 35%, each under 10% MFU) and launch overhead (about 20%) [9] (approximate, from secondary notes).

**Architecture** [6][9]:

- Features are grouped into T **tokens** of uniform dimension D. Recommendation models have hundreds to thousands of tokens.
- Each RankMixer block has two parts:
  - **Multi-head token mixing.** Each token is split into heads and the heads are recombined across tokens. This is parameter-free, costs O(TD), uses no softmax, and replaces O(T^2 D) self-attention over heterogeneous feature spaces.
  - **Per-token FFN.** Each token (feature subspace) gets its own FFN parameters. This gives T times the parameters of a shared FFN at the same FLOPs.
- A **sparse MoE** variant uses ReLU routing with an L1-style regulariser, and **DTSI-MoE** (dense-training, sparse-inference routers) to scale parameters while keeping per-sample compute fixed.
- Typical depth is 6 to 12 blocks, and the output is mean-pooled into the logits.

```
  raw features --> group/project --> T tokens x D
     [ token mixing (heads shuffled across tokens) -> add & norm
       per-token FFN / sparse MoE (separate weights per token) -> add & norm ] x L
  --> mean pool --> multi-task heads (finish, like, play time, ...)
```

**Results.** MFU rose from 4.5% to 45%, and ranking-model parameters grew about 100x at roughly constant inference latency. Full-traffic deployment in **Douyin feed ranking** gave +0.2% active days and +0.5% app duration [6]. The successor **TokenMixer-Large** (2026) reports offline scaling to 7B and 15B parameters for Douyin ads and e-commerce [7].

### 20.5.3 OneTrans: one Transformer for sequence and features

OneTrans, described in technical notes as a successor in the same line [9][10], folds sequence modelling and feature interaction into a single Transformer [10]:

- **S-tokens** (behaviour events, projected by behaviour-specific MLPs) and **NS-tokens** (non-sequential features, split from one MLP by an "auto-split tokenizer") form one input.
- **Mixed parameterisation.** S-tokens share Q/K/V and FFN weights, while each NS-token has its own.
- **Pyramid stack.** Queries come only from the last L' tokens at each layer, so the sequence shrinks, for example 1190 -> 595 -> 297 -> ... -> 12 tokens.
- **Two-level KV caching.** The S-side keys and values are computed once per request and reused across candidates. Across requests, only newly appended events need new K/V.

The notes describe a model of hundreds of millions of parameters [10]. They do not report deployment numbers, so treat OneTrans's production status as unconfirmed here.

### 20.5.4 The ByteDance pattern

```
       +-------------------- Monolith-style online training ------------------+
       | Kafka actions + feature logs -> Flink joiner -> streaming trainer    |
       | Cuckoo-hash sparse PS  --(minute-level incremental sync)--> serving  |
       +----------------------------------------------------------------------+
  request --> retrieval (two-tower / Deep Retrieval paths / multi-channel)
          --> pre-rank --> rank: LONGER (10k seq, KV cache) + RankMixer tokens
          --> value-model fusion (P_like*V_like + E_playtime*V_playtime + ...)
          --> diversity rules (no same creator / same sound back to back)
          --> For You feed
           (composite, synthesised from [1][4][6][11][12]; not an official diagram)
```

---

## 20.6 Kuaishou: the cascade, quantified

### 20.6.1 RecFlow and the six-stage funnel

Kuaishou has published the most complete stage-level data of any short-video platform. **RecFlow** (ICLR 2025) logs six stages of the production funnel. It contains about 9.3 million requests and 1.9 billion stage samples, so researchers can study inter-stage effects such as the exposure-only training bias [23]. A 2026 cross-stage-fusion paper on the same cascade gives candidate sizes falling from about 8,000 to about 500 after pre-ranking, and finally to about 6 to 10 exposed items [24]. KuaiFormer describes retrieval feeding pre-rank, then cascade rank, then fine-grained full rank [16].

```
  corpus (10^8+ videos)
     | multi-channel retrieval: i2i, u2i two-tower, KuaiFormer, ...
     v
  ~8,000 merged candidates
     | pre-ranking (lightweight)
     v
  ~500
     | cascade rank -> full rank (multi-task: pLTR, pWTR, pFTR, watch time, ...)
     v
  tens
     | ensemble sort (pXTR fusion) + re-rank (listwise, diversity, business rules)
     v
  ~6-10 exposed                                   (after [16][23][24]; approximate)
```

### 20.6.2 PERSIA: training 100-trillion-parameter models

PERSIA (Kuaishou with ETH Zurich, KDD 2022) is Kuaishou's answer to the same sparse-dense split that Monolith addresses [25]. Embedding layers are updated **asynchronously** on CPU parameter servers, because embedding communication is huge and sparse updates tolerate staleness. The dense network is trained **synchronously** on GPUs, because its accuracy suffers under asynchrony. PERSIA reports models of up to **100 trillion parameters** and production use at Kuaishou for applications with hundreds of millions of DAU. It reports several-fold throughput gains (approximately up to 7x) over contemporary systems [25].

### 20.6.3 Multi-task ranking and ensemble sort

- **HoME** (KDD 2025) fixes MMoE pathologies seen in production: expert collapse (ReLU experts producing mostly zeros), shared experts degrading into single-task experts, and expert underfitting. Its answer is hierarchical, normalised expert groups. It is deployed on short-video services serving about 400 million daily users [26].
- **Ensemble sort.** The ranker emits many pXTRs, and a fusion formula merges them, which is the same value-model idea as the TikTok formula in 20.2.2. **Pantheon** (2025) learns personalised fusion weights jointly with the real-time ranker using Iterative Pareto Policy Optimisation, deployed for about 400 million daily users [27]. **GRADE** (2025) uses GRPO-style group-relative RL over weight vectors with Dirichlet exploration [37].
- **RLUR** (WWW 2023) optimises **user retention**, measured as the time until the next visit, as an infinite-horizon MDP. The actor outputs fusion weights, not item scores, and a TD-trained retention critic is shaped with immediate watch-time rewards [28]. The companion **TSCAC** maximises watch time subject to constraints on likes, follows and shares [29].
- **Re-ranking.** NAR4Rec (KDD 2024) generates the final slate with a non-autoregressive model, filling all positions in one parallel pass instead of decoding position by position. This keeps listwise re-ranking within the latency budget [32].

### 20.6.4 Label and cold-start engineering

- **D2Q** (KDD 2022) splits videos into duration buckets and predicts watch-time **quantiles** within each bucket. This removes the length bias of raw watch-time regression [30].
- **POSO** (2021) gates model sub-modules by user cohort, such as new versus regular users. Signals from new users are otherwise drowned out by heavy users in shared parameters [31].
- **KuaiRec** is a public dataset with an almost fully observed user-item matrix. The small matrix has 1,411 users, 3,327 items and 99.6% density, and the big matrix has 7,176 users, 10,728 items and 16.3% density. It is useful for unbiased offline evaluation [33].

---

## 20.7 Kuaishou lifelong user modelling: TWIN and TWIN V2

### 20.7.1 TWIN (KDD 2023)

**Problem.** SIM-style two-stage models (Chapter 5) use a cheap General Search Unit (GSU) to pick relevant behaviours and an Exact Search Unit (ESU) to attend over them. The GSU's relevance metric differs from the ESU's attention, so the GSU drops behaviours that the ESU would rank highly [13].

**Key idea.** Make the two stages "twins". The **Consistency-Preserved GSU (CP-GSU)** uses the same target-attention metric, architecture and parameters as the ESU. It scores all ~10^4 long-term behaviours for every candidate and keeps the top 100. The ESU then runs multi-head target attention over those 100 [13].

**Making it affordable.** TWIN splits behaviour features into two kinds:

- **Inherent item features** (video ID, author, tags, duration) are independent of the user. Their key projections are **precomputed and cached** per item, shared across all users.
- **User-item cross features** (when the user watched it, how long they played it) are compressed into a scalar bias on the attention logit.

```
  per behaviour j:  K_j = W * inherent_j        (precomputed, cached per item)
                    b_j = w . cross_j            (cheap, user-specific)
  per candidate c:  s_j = Q(c) . K_j / sqrt(d) + b_j
  CP-GSU: top-100 by s_j over ~10^4 behaviours -> ESU: multi-head target attention
```

**Scale.** Offline experiments use about 46 billion production samples. Optimised serving infrastructure gave a **99.3% reduction** in the compute bottleneck, which allowed deployment on Kuaishou's main traffic [13]. Secondary summaries report watch-time gains of roughly 1.3% to 6.2% across tabs (approximate) [15].

### 20.7.2 TWIN V2 (CIKM 2024)

**Problem.** Full life-cycle histories now exceed 10^5 to 10^6 behaviours, more than even TWIN can scan online [14].

**Design** [14][15]:

1. **Offline hierarchical clustering** of each user's behaviours into clusters of similar items. Secondary notes describe this as recursive k-means that splits clusters until each leaf holds at most gamma items.
2. Each cluster becomes one **virtual behaviour** with aggregated features.
3. **Cluster-aware target attention** in the GSU and ESU, with attention adjusted by cluster size so that strong interests keep their weight.

**Results.** Deployed on primary traffic serving hundreds of millions of DAU. Watch-time gains of +0.672%, +0.800% and +0.728% on the Featured-Video, Discovery and Slide tabs. The authors note that a 0.1% gain is significant at Kuaishou [14].

---

## 20.8 KuaiFormer: Transformer retrieval (2024)

**Problem.** Single-vector two-tower retrieval squeezes all of a user's interests into one embedding and handles long sequences poorly [16].

**Design** [16]:

- **Multiple learnable query tokens** (like BERT's [CLS]) are appended to the behaviour sequence. Each produces one interest vector, which runs its own ANN query, for multi-interest retrieval.
- **Adaptive sequence compression.** The oldest behaviours are merged in groups of 64 and middle-aged ones in groups of 16, each group aggregated by a one-layer bidirectional Transformer. The most recent behaviours are kept as-is. For example, 256 behaviours become 55 tokens.
- A **causal Transformer** over the compressed sequence, trained with smoothed in-batch softmax and logQ correction for popularity bias.

```
  [ old 64-groups | mid 16-groups | recent raw ] + [Q1 .. Qk] query tokens
              -> causal Transformer -> k interest vectors -> k ANN lookups -> merge
```

**Scale.** In production since May 2024 in Kuaishou's short-video recommendation, serving more than 400 million DAU [16]. Lesson: "compress the old, keep the recent exact" is a cheap, robust way to fit long histories into a retrieval latency budget. LONGER's token merge applies the same idea on the ranking side [4].

---

## 20.9 OneRec: collapsing the cascade into one generative model

OneRec is the most complete public case of replacing an entire short-video cascade with one generative model at hundreds-of-millions-DAU scale.

### 20.9.1 OneRec V1 (February 2025 paper, June 2025 technical report)

**Problem.** In the cascade each stage is optimised separately, compute is fragmented across many small models with low MFU, and earlier stages cap the quality of later ones [17][18]. OneRec generates a **session** (a short list of videos) end to end and aligns it with user preferences through RL.

**Item tokenizer** [22]:

- Multimodal inputs (title, tags, ASR, OCR, cover image, sampled frames) are encoded by a multimodal LLM (miniCPM-V-8B) into 1,280 tokens. A QFormer compresses these into 4 query vectors.
- An item-pair contrastive loss injects collaborative signal, with title generation as an auxiliary task.
- **RQ-KMeans** over 3 residual levels gives each video a 3-token semantic ID. Chapter 6 reports about 8K codes per level (approximate).

**Encoder (four pathways)** [22]:

- Static user features through two dense layers.
- Short-term behaviour: the last 20 interactions.
- Positive-feedback behaviour: the last 256 high-engagement interactions.
- Lifelong behaviour: up to 100,000 records. These are compressed by hierarchical k-means into a length-2,000 sequence, then by a QFormer into 128 tokens.
- Total context: 1 + 20 + 256 + 128 = **405 tokens**.

**Decoder.** A [BOS] token plus semantic-ID tokens, with causal self-attention, cross-attention to the encoder, and top-k **MoE** feed-forward layers, trained with next-token prediction [22].

```
 static feats -----+
 short seq (20) ---+--> Encoder (Transformer, RMSNorm) --> 405 context tokens
 pos-feedback(256)-+                                            | cross-attn
 lifelong (1e5) -> hier. k-means (2000) -> QFormer (128) -------+
                                                                v
                [BOS] s1 s2 s3 --> Decoder (causal self-attn + MoE FFN)
                                                                | beam search
                                                                v
                candidate sessions --> reward system:
                    P-Score (SIM-based, one tower per objective: CTR, LTR, VTR, ...)
                    + format reward (legal SIDs only) + SIR (viral down-weighting)
                                                                |
                IPA / DPO (V1 paper)  ->  ECPO = early-clipped GRPO (tech report)
```

**Preference alignment.**

- The original paper used **Iterative Preference Alignment (IPA)**: a reward model scores sampled sessions, and the best and worst sessions form DPO pairs [17].
- The technical report moved to **ECPO**, an early-clipped GRPO. It pre-clips the policy ratio on negative-advantage samples so their gradients cannot explode [18][22].
- A **format reward** keeps only legal semantic IDs (advantage 1 for legal samples; illegal ones are discarded).
- **SIR** industrial alignment down-weights viral content once its share passes a threshold. It cut viral-content exposure by 9.59% with core metrics stable [22].

**Scale and economics (as reported, approximate)** [18]:

- Training MFU about 24% and inference MFU about 29%, against single-digit MFU for the legacy rankers.
- Operating expense about 10% of the previous cascade.
- Deployed on the main Kuaishou and Kuaishou Lite feeds, handling about 25% of their traffic.
- App stay-time gains in the 0.5% to 1.2% range.

### 20.9.2 OneRec-V2 (August 2025)

**Problem.** In V1 most compute went into the encoder, and only a small fraction trained the generator. RL against a learned reward model was sample-inefficient and risked reward hacking [19][22].

**Lazy decoder-only architecture** [22]:

- A **context processor** computes key/value pairs from the user context once. They are shared across decoder layers with GQA, and there is no per-layer context projection.
- During training the decoder sees only 3 tokens: [BOS] plus the first two semantic IDs.
- Each layer has lazy cross-attention, causal self-attention, and an FFN, with MoE in the deeper layers.

| Architecture (1B params) | GFLOPs | Converged loss |
|---|---|---|
| Encoder-decoder (1:1) | 296.36 | 3.28 |
| Lazy decoder-only | 18.89 | 3.27 |

That is about 94% less compute and about 90% less training resource for the same loss [22].

**Scaling law** [22]:

- Dense models from 0.1B to 8B fit L(N) = 3.13 + 3660 / N^0.489.
- Loss falls 3.57 (0.1B) -> 3.33 (0.5B) -> 3.27 (1B) -> 3.23 (2B) -> 3.20 (4B) -> 3.19 (8B).
- A 4B-total / 0.5B-active MoE reaches 3.22, better than the 2B dense model.

**RL from real user feedback.** Rewards come directly from logged watch time, not from a learned reward model [22]:

- Watch time is bucketed logarithmically by duration. The top 25% within a bucket get a positive label, explicit negative feedback gets a negative label, and the rest are ignored.
- **GBPO** (Gradient-Bounded Policy Optimisation) keeps gradients for all samples but bounds the RL gradient by a BCE-style gradient. This avoids the clipping instability of PPO, GRPO and ECPO.

Reported online stay-time gains are again sub-1% (approximate) [19].

### 20.9.3 OneRec-Think, OpenOneRec and the "One" family

- **OneRec-Think (October 2025)** puts reasoning in the generator. It starts from Qwen3 with a vocabulary extended by itemic tokens and trains in three stages: itemic alignment, reasoning activation (chain-of-thought SFT), and reasoning enhancement with a reward that accepts multiple valid answers. It reports a further +0.159% app stay time online [20].
- **OpenOneRec (December 2025)** is Kuaishou's open release [21]:
  - **RecIF-Bench**: about 100M interactions from 200K users across short video, ads and products, with 8 tasks in a four-layer capability hierarchy.
  - **Checkpoints**: OneRec-1.7B and OneRec-8B on Qwen3 backbones, trained on about 33B tokens, and "-Pro" variants trained on about 130B tokens, including an industrial Kuaishou corpus.
  - **Training pipeline**: itemic-text alignment and co-pretraining, then multi-task SFT, on-policy distillation and RL.
  - **Results**: OneRec-8B-Pro scores short-video Recall@32 of 0.0369, against 0.0119 for SASRec and 0.0180 for LC-Rec. The README reports a 26.8% average Recall@10 gain over the second-best method on 10 Amazon datasets.
- **MiniOneRec** (October 2025) is a community reproduction with RQ-KMeans semantic IDs, SFT and GRPO with constrained beam search [36].
- **The "One" recipe beyond the feed.** Kuaishou has applied the same recipe to e-commerce search (OneSearch) and other surfaces [35]. Long-sequence generative work continues with GEMs (2026), which uses a multi-stream decoder [34].

---

## 20.10 Comparison

| System | Org / year | Stage | Core idea | Scale / reported result |
|---|---|---|---|---|
| For You feed disclosure [11] | TikTok 2020 | whole feed | Weighted multi-signal ranking + diversity rules | Public description only |
| "Algo 101" formula [12] | TikTok (reported) 2021 | ranking fusion | Sum of P_action * V_action | Goals: retention, time spent |
| Monolith [1][2] | ByteDance 2022 | training infra | Cuckoo-hash PS, admission/expiry, online training | Minute-level sparse sync |
| Deep Retrieval [3] | ByteDance 2020 | retrieval | Learned D x K path codes, beam search | ~Brute-force accuracy |
| LONGER [4] | ByteDance 2025 | ranking | End-to-end 10k sequences, token merge, KV cache | Dozens of scenarios |
| RankMixer [6] | ByteDance 2025 | ranking | Token mixing + per-token FFN/MoE | MFU 4.5% -> 45%; Douyin +0.5% duration |
| TokenMixer-Large [7] | ByteDance 2026 | ranking | RankMixer successor | 7B / 15B offline |
| HLLM [8] | ByteDance 2024 | seq. rec | Item LLM + user LLM | 1B / 7B checkpoints |
| PERSIA [25] | Kuaishou+ETH 2022 | training infra | Async embeddings, sync dense | Up to 100T params |
| TWIN [13] | Kuaishou 2023 | ranking | Consistent GSU/ESU, cached item keys | 99.3% less bottleneck |
| TWIN V2 [14] | Kuaishou 2024 | ranking | Hierarchical clustering of 10^5+ behaviours | +0.67% to +0.80% watch time |
| KuaiFormer [16] | Kuaishou 2024 | retrieval | Multi-query causal Transformer, compression | 400M+ DAU |
| HoME / Pantheon [26][27] | Kuaishou 2024-25 | MTL / fusion | Expert hierarchy; learned ensemble sort | ~400M daily users |
| RLUR [28] | Kuaishou 2023 | fusion policy | RL on fusion weights for return time | Retention gains |
| OneRec V1 [17][18] | Kuaishou 2025 | whole cascade | Enc-dec + MoE + SIDs + IPA/ECPO | ~25% traffic; OPEX ~10% |
| OneRec-V2 [19] | Kuaishou 2025 | whole cascade | Lazy decoder-only, RL on real watch time | 94% less compute at equal loss |
| OpenOneRec [21] | Kuaishou 2025 | foundation | Qwen3 + itemic tokens, open weights | 1.7B / 8B |

---

## 20.11 Cross-cutting design patterns

1. **Freshness through streaming, not just features.** Monolith (online training with incremental sparse sync) and PERSIA (async sparse updates) both separate the fast-moving sparse part of the model from the slow dense part. Kuaishou's Pantheon trains its fusion policy jointly with a real-time ranker [27]. In short video, a model that is a day old is a worse model.
2. **The value model is the product.** TikTok's reported formula, Kuaishou's ensemble sort and RLUR's learned fusion weights all show that the final fusion layer carries the business objective. Over time that layer has become personalised and learned [12][27][28].
3. **Compute where the user signal is.** TWIN caches item-side projections, KuaiFormer and LONGER merge old tokens, TWIN V2 clusters, and OneRec compresses 100,000 records to 128 tokens. Every system splits the cost into a user-only part computed once per request and a per-candidate part that must be cheap [4][13][14][16][22].
4. **MFU as a first-class metric.** RankMixer (4.5% -> 45%) and OneRec (about 24%/29% training/inference) show that redesigning models for GPU efficiency is a prerequisite for scaling parameters [6][18].
5. **From cascade to generation.** Deep Retrieval's path codes (2020) anticipate semantic-ID generation. OneRec takes it to the whole cascade, keeping a reward model or real-feedback RL in place of the old multi-stage objectives [3][17][19].
6. **Debias the label before you scale.** Duration bias (D2Q, OneRec-V2's bucketed watch time) and viral-content concentration (OneRec SIR) are both handled explicitly in labels and rewards [22][30].

---

## Key takeaways for system designers

- **Design for minute-level freshness from day one.** Use a streaming joiner keyed by a serving-time request ID, an online trainer, and incremental sync of only the touched sparse rows. Snapshot rarely, because online training heals PS losses quickly [1].
- **Use collisionless, dynamic embedding tables with admission and eviction** for unbounded, churning ID spaces such as videos and creators. Hash collisions are a silent quality tax [1][2].
- **Make the cheap stage agree with the expensive one.** TWIN's lesson is that the first-stage selection metric should match the final model's attention. Inconsistent stages waste the expensive model's capacity [13].
- **Compress old history, keep recent history exact.** Hierarchical clustering, group merging and QFormer compression all reach 10^4 to 10^5 behaviours within latency budgets [14][16][22].
- **Treat watch time carefully.** Normalise for duration (quantiles or log-buckets) or the system will learn to promote long videos [22][30].
- **Separate prediction from policy, then learn the policy.** Start with an explicit P_action * V_action fusion that product owners can read, then move to personalised or RL-learned weights. RLUR and Pantheon do this with no change to the ranker [12][27][28].
- **Measure and fix MFU before scaling parameters.** Token-based rankers (RankMixer, OneTrans) and decoder-only generators (OneRec-V2) earn their size by being GPU-friendly [6][10][22].
- **If you go generative, budget compute for the decoder and use real feedback.** OneRec-V2 matched encoder-decoder quality with about 94% less compute and replaced the learned reward model with bucketed real watch-time signals, which reduces reward hacking [22].
- **Keep an explicit rule layer.** TikTok's public description keeps diversity rules (no back-to-back same creator or sound) outside the model, and OneRec adds explicit viral-content control [11][22].
- **Expect small percentages to matter.** At Kuaishou scale a 0.1% watch-time gain is significant. Plan experiments with enough power to detect sub-1% effects [14].

---

## References

1. Zhuoran Liu et al. (ByteDance). "Monolith: Real Time Recommendation System With Collisionless Embedding Table." arXiv 2209.07663, 2022. https://arxiv.org/abs/2209.07663
2. ByteDance. "Monolith: A Lightweight Recommendation System" (GitHub repository, archived October 2025). Accessed 2026. https://github.com/bytedance/monolith
3. Weihao Gao et al. (ByteDance). "Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations." arXiv 2007.07203, 2020. https://arxiv.org/abs/2007.07203v1
4. ByteDance. "LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders." RecSys 2025, arXiv 2505.04421. https://arxiv.org/abs/2505.04421
5. Machine Learning at Scale. "[RecSys] Part 3: LONGER, scaling up long sequence modelling in industrial recommenders." Substack, 2025. https://machinelearningatscale.substack.com/p/recsys-part-3-longer-scaling-up-long
6. ByteDance. "RankMixer: Scaling Up Ranking Models in Industrial Recommenders." arXiv 2507.15551, 2025. https://arxiv.org/abs/2507.15551
7. ByteDance. "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders." arXiv 2602.06563, 2026. https://arxiv.org/abs/2602.06563
8. ByteDance. "HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling." arXiv 2409.12740, 2024; code and checkpoints (archived): https://github.com/bytedance/HLLM ; paper: https://arxiv.org/abs/2409.12740
9. Haozhe Xing. "RankMixer: A Hardware-Efficient Architecture" (fun-rec-mdbook technical notes). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/rankmixer.md
10. Haozhe Xing. "OneTrans: A Unified Transformer" (fun-rec-mdbook technical notes). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/onetrans.md
11. TikTok. "How TikTok recommends videos #ForYou." TikTok Newsroom, June 2020 (not re-fetched for this chapter). https://newsroom.tiktok.com/en-us/how-tiktok-recommends-videos-for-you
12. Ben Smith. "How TikTok Reads Your Mind." The New York Times, 5 December 2021 (reporting on the internal "TikTok Algo 101" document; not re-fetched for this chapter). https://www.nytimes.com/2021/12/05/business/media/tiktok-algorithm.html
13. Jianxin Chang et al. (Kuaishou). "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou." KDD 2023, arXiv 2302.02352. https://arxiv.org/abs/2302.02352
14. Zihua Si et al. (Kuaishou). "TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling for Enhanced CTR Prediction at Kuaishou." CIKM 2024, arXiv 2407.16357. https://arxiv.org/abs/2407.16357
15. Machine Learning Frontiers (S. Bhattacharya). "Target Attention Is All You Need: Modeling Extremely Long User Action Sequences in Recommender Systems." Substack, 2024. https://mlfrontiers.substack.com/p/target-attention-is-all-you-need
16. Kuaishou. "KuaiFormer: Transformer-Based Retrieval at Kuaishou." arXiv 2411.10057, 2024. https://arxiv.org/abs/2411.10057
17. Jiaxin Deng et al. (Kuaishou). "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment." arXiv 2502.18965, February 2025. https://arxiv.org/abs/2502.18965
18. Kuaishou OneRec Team. "OneRec Technical Report." arXiv 2506.13695, June 2025. https://arxiv.org/abs/2506.13695
19. Kuaishou OneRec Team. "OneRec-V2 Technical Report." arXiv 2508.20900, August 2025. https://arxiv.org/abs/2508.20900
20. Kuaishou OneRec Team. "OneRec-Think: In-Text Reasoning for Generative Recommendation." arXiv 2510.11639, October 2025. https://arxiv.org/abs/2510.11639
21. Kuaishou OneRec Team. "OpenOneRec Technical Report" (arXiv 2512.24762) and OpenOneRec repository. December 2025. https://arxiv.org/abs/2512.24762 ; https://github.com/Kuaishou-OneRec/OpenOneRec
22. Haozhe Xing. "End-to-End Generative Recommendation" (OneRec / OneRec-V2 technical notes, fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part8-e2e/e2e-recommendation.md
23. Kuaishou. "RecFlow: An Industrial Full Flow Recommendation Dataset." ICLR 2025, arXiv 2410.20868. https://arxiv.org/html/2410.20868v1
24. "UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems." arXiv 2609.11052, 2026. https://arxiv.org/pdf/2609.11052
25. Xiangru Lian et al. (Kuaishou and ETH Zurich). "Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters." KDD 2022, arXiv 2111.05897. https://arxiv.org/abs/2111.05897
26. Xu Wang et al. (Kuaishou). "HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou." arXiv 2408.05430, 2024 (KDD 2025). https://arxiv.org/abs/2408.05430
27. Cao et al. (Kuaishou). "Pantheon: Personalized Multi-objective Ensemble Sort via Iterative Pareto Policy Optimization." arXiv 2505.13894, 2025. https://arxiv.org/abs/2505.13894
28. Qingpeng Cai et al. (Kuaishou). "Reinforcing User Retention in a Billion Scale Short Video Recommender System." WWW 2023 Companion, arXiv 2302.01724. https://arxiv.org/abs/2302.01724
29. Qingpeng Cai et al. (Kuaishou). "Two-Stage Constrained Actor-Critic for Short Video Recommendation." WWW 2023, arXiv 2302.01680. https://arxiv.org/abs/2302.01680
30. Ruohan Zhan et al. (Kuaishou). "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation." KDD 2022, arXiv 2206.06003. https://arxiv.org/abs/2206.06003
31. Shangfeng Dai et al. (Kuaishou). "POSO: Personalized Cold Start Modules for Large-scale Recommender Systems." arXiv 2108.04690, 2021. https://arxiv.org/abs/2108.04690
32. Yuxin Ren et al. (Kuaishou). "Non-autoregressive Generative Models for Reranking Recommendation" (NAR4Rec). KDD 2024, arXiv 2402.06871. https://arxiv.org/abs/2402.06871
33. Chongming Gao et al. "KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems." CIKM 2022, arXiv 2202.10842; dataset repository. https://arxiv.org/abs/2202.10842 ; https://github.com/chongminggao/KuaiRec
34. Kuaishou. "GEMs: Long-sequence generative recommendation with a multi-stream decoder." arXiv 2602.13631, 2026. https://arxiv.org/abs/2602.13631
35. Kuaishou. "OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search." arXiv 2509.03236, 2025. https://arxiv.org/abs/2509.03236
36. Kong et al. "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation." arXiv 2510.24431, 2025; code: https://github.com/AkaliKong/MiniOneRec ; paper: https://arxiv.org/abs/2510.24431
37. Kuaishou. "GRADE: Personalized Multi-Task Fusion via Group-relative Reinforcement Learning with Adaptive Dirichlet Exploration." arXiv 2510.07919, 2025. https://arxiv.org/abs/2510.07919
