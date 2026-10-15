# 06. Generative Recommenders (2023-2026): HSTU, Semantic IDs, OneRec, MTGR and the Foundation-Model Turn

Between 2023 and 2026 the industrial recommender stack went through its largest architectural change since deep learning recommendation models (DLRMs) replaced matrix factorization. Three ideas drove it:

1. **Recommendation as sequence transduction.** Meta's HSTU paper ("Actions Speak Louder than Words") recast retrieval and ranking as generative modeling over a user's chronological stream of (item, action) tokens. It reported power-law quality scaling with compute and production models of about 1.5 trillion parameters [1][2].
2. **Items as token sequences (semantic IDs).** Google's TIGER replaced atomic item IDs with short tuples of discrete codes learned by residual quantization (RQ-VAE). Retrieval then becomes autoregressive decoding of the next item's code tuple [4][5].
3. **One model instead of a cascade.** Kuaishou's OneRec, Meituan's MTGR and EGA, Tencent's GPR, and others replaced parts or all of the retrieve -> pre-rank -> rank -> re-rank cascade with a single generative model, often post-trained with reinforcement learning against a reward model [10][11][12][16][41].

At the same time, Netflix, LinkedIn and Pinterest published "foundation model" approaches. Each pre-trains one large user-sequence model and feeds many downstream recommenders from it [25][27][29]. X (formerly Twitter) open-sourced a For You pipeline whose main ranker, Phoenix, is a transformer over the viewer's engagement sequence with candidate-isolated attention [31].

This chapter covers each design: the problem it targets, its architecture and data flow, model and scale numbers where public, the key tricks, and the lessons learned. It ends with a comparison table and takeaways for system designers.

> Source note: this chapter was compiled when only some primary hosts could be fetched directly. Facts marked "(approx.)" or "as reported" come from abstracts or secondary technical notes, not from a full read of the paper. Check them against the cited primary source before you depend on an exact figure.

---

## 1. The conceptual shift: from DLRM to generative recommenders

A classical DLRM ranker scores each (user, candidate) pair independently. It uses hundreds of hand-built features, very large sparse embedding tables, and a feature-interaction network (DCN, DeepFM, and similar). Three structural problems limited further scaling:

- **Hardware inefficiency.** Embedding tables often hold more than 90% of DLRM parameters, and those parameters are sparse and memory-bound [22][21]. Secondary analyses put DLRM model FLOPs utilization (MFU) at roughly 4-5%, against 40-60% for LLM training [23].
- **Cascade information loss.** Items dropped by an early stage are never seen by later, stronger models. Each stage also optimizes a different objective [22].
- **Per-impression training cost.** Each impression is a separate training example, so a user with 500 interactions gets 500 separate forward passes over almost the same history [18].

Generative recommenders (GRs) address all three. They model the user as a single token sequence, run one causal forward pass that produces supervision at every position, and use dense, GEMM-heavy transformer compute that scales the way LLMs do.

```
 Classical cascade (DLRM era)            Generative recommender (GR era)
 ----------------------------            --------------------------------
 user feats ─┐                           [item0, act0, item1, act1, ... , itemN]
 item feats ─┼─> Retrieval (two-tower)              │ causal transformer (HSTU,
 cross feats ┘      │ ~10^3-10^4                     │  decoder-only, enc-dec)
                    v                                v
             Pre-rank (light DNN)        retrieval head: p(next item | history)
                    │ ~10^2-10^3         ranking head:  p(action | history, item)
                    v                    or: decode semantic-ID tokens directly
             Rank (DLRM, MTL)                        │
                    │ ~10^1-10^2                     v
                    v                    (optional) RL post-training vs reward model
             Re-rank / blend
```

---

## 2. Meta HSTU: "Actions Speak Louder than Words"

### 2.1 Problem

Meta wanted to know whether recommendation quality scales with compute the way language modeling does, and what architecture would allow it on high-cardinality, non-stationary streaming data with billions of users [1]. Standard transformers did not fit well. Softmax attention normalizes away the intensity of user interest, and the FFN-heavy blocks used too much activation memory for the very large batches that industrial training needs [18].

### 2.2 Problem formulation

HSTU interleaves content tokens (Φ) and action tokens (a) into one chronological sequence [Φ0, a0, Φ1, a1, ..., Φn-1, an-1] of length 2n_c. Other categorical features are merged onto this timeline by timestamp, and runs of identical values are compressed [18]. Both main tasks then become sequence transduction:

- **Ranking:** p(a_i | Φ0, a0, ..., Φi). This is target-aware, because the candidate item is in the context when the action is predicted.
- **Retrieval:** p(Φ_i | Φ0, a0, ..., a_{i-1}). This predicts the next item.

Under a causal mask, one forward pass yields n_c supervision signals. For a user with about 500 interactions, the secondary analysis estimates roughly a 500x reduction in training forward passes compared with impression-level DLRM training [18].

### 2.3 The HSTU block

Each layer has three parts [1][18]:

1. **Pointwise projection:** U, V, Q, K = Split(SiLU(f1(X))).
2. **Pointwise aggregated attention:** A(X)V(X) = SiLU(Q K^T + rab_{p,t}) ⊙ V. There is no softmax, so the weights do not sum to 1 and the absolute strength of interest is preserved. rab_{p,t} is a relative attention bias over both position and time differences.
3. **Pointwise transformation:** Y = f2(Norm(A(X)V(X)) ⊙ U(X)). The gate U replaces the usual 4x-wide FFN.

```
   X ──> Linear ──> SiLU ──> split ──> U, V, Q, K
                                      │
            SiLU(Q·K^T + rab(pos,time)) ⊙ V      <- no softmax; rel. pos+time bias
                                      │
                       LayerNorm(·) ⊙ U          <- gating replaces the FFN
                                      │
                            Linear ──> + X (residual)
```

The secondary analysis reports per-layer activation memory falling from about 33x hidden dimension (standard transformer) to about 14x [18]. This matters because industrial batch sizes run from tens of thousands to hundreds of thousands, and activation memory is usually the limit [18].

### 2.4 Efficiency tricks

- **Stochastic Length (SL).** Long user histories are sparsely subsampled during training, so attention cost grows sub-quadratically. The secondary notes report 80%+ sparsity with almost no loss in quality [18].
- **Efficient attention kernels.** The paper reports HSTU running 5.3x to 15.2x faster than FlashAttention2-based transformers on 8192-length sequences [1]. The open-source repo now includes Triton kernels, an HSTU attention kernel based on FlashAttention-3 for H100 GPUs, and row-wise FP8 GEMMs [2].
- **M-FALCON (inference).** To score m candidates against one history, the candidates are appended to the sequence with an attention mask so that candidates see only the history and not each other. They are processed in micro-batches, and the history's K/V cache is shared across micro-batches. The secondary analysis attributes speedups of tens to hundreds of times to the combination [18].

### 2.5 Scale and results

- Up to 65.8% NDCG improvement over baselines on public and synthetic datasets [1].
- 1.5-trillion-parameter models deployed on multiple surfaces of a billion-user platform, with a reported +12.4% on online A/B metrics [1].
- Quality scales as a power law of training compute across three orders of magnitude, up to roughly GPT-3/LLaMa-2 training compute [1]. Secondary notes give the fitted laws as HR@100 = 0.15 + 0.0195 ln C for retrieval and NE = 0.549 - 0.0053 ln C for ranking. The sweep covered sequence lengths 512-8192, hidden dimensions 256-1024, and up to 24 layers [18].
- Public benchmark reproduction from the repo: on ML-20M, HSTU-large reaches HR@10 0.3556 against 0.2889 for SASRec (+23.1%). On Amazon Books it reaches NDCG@10 0.0262 against 0.0164 (+60.7%) [2].

### 2.6 Follow-ups

The repo also contains a "DLRM-v3" ranker built on HSTU with an MLCommons LoadGen inference harness, and "HSTU Ultra", which adds semi-local and target-aware attention and FP8 projections [2]. In 2026 Meta published "Bending the Scaling Law Curve in Large-Scale Recommendation Systems" (listed under the ULTRA-HSTU name), and an analysis of how well generative recommenders generalize [35][45].

**Lessons.** (1) Softmax is not sacred: for recommendation, unnormalized SiLU attention plus a relative time bias works better. (2) Most of the value comes from restructuring training (one sequence per user, causal supervision), not only from model size. (3) Inference needs its own algorithm (M-FALCON) to make target-aware ranking over many candidates affordable.

---

## 3. Semantic IDs and TIGER (Google)

### 3.1 Problem

Atomic item IDs carry no meaning: item 81723 and item 81724 are unrelated unless behavior links them. Very large ID vocabularies also make softmax retrieval expensive, and brand-new items have no trained embedding at all. TIGER ("Recommender Systems with Generative Retrieval", NeurIPS 2023) proposed describing every item with a short tuple of discrete semantic codes and training a sequence model to generate the next item's code tuple [4][5].

### 3.2 Pipeline

```
 item text/metadata ──> frozen text encoder (e.g., Sentence-T5) ──> dense emb e
                                                                    │
                       RQ-VAE: encoder -> z
                       level 1: c1 = argmin ||z - C1[k]||, r1 = z - C1[c1]
                       level 2: c2 = argmin ||r1 - C2[k]||, r2 = r1 - C2[c2]
                       level 3: c3 = ...
                                                                    │
                    Semantic ID = (c1, c2, c3 [, dedup token])  e.g. (10, 5, 42)
                                                                    │
 user history = [SID(item1), SID(item2), ...] ──> Transformer seq2seq (T5-style)
                                                ──> beam-decode SID of next item
                                                ──> map SID -> item (lookup)
```

- **RQ-VAE loss.** Reconstruction error, plus a codebook term and a commitment term (weight β) at each level. Gradients pass through the quantizer with the straight-through estimator [21].
- **Hierarchy.** On the Beauty dataset, the first codeword tracks broad categories and later codewords refine them. Items that share a prefix are semantically similar, so knowledge transfers between related items [4].
- **Cold start.** A new item gets its semantic ID from its content embedding with no interaction history. The authors report better behavior on cold items [4].
- **Collisions.** When two items get the same code tuple, an extra disambiguation token is appended. Decoding can also be restricted to valid prefixes so that every output maps to a real item [6].
- Commonly reported TIGER configuration (approx.): a 768-d Sentence-T5 embedding, 3 RQ levels with 256 codes each, plus the collision token. The open-source reproduction uses smaller toy settings in its quick-start (codebook 16, 3 levels) [6].

### 3.3 Variations on semantic IDs

| Variant | Idea | Where used |
|---|---|---|
| RQ-VAE | Learned encoder + residual codebooks, end to end | TIGER [4]; EGA (3 levels x 1024) [24] |
| RQ-KMeans | Residual k-means on frozen embeddings; decoupled from representation learning; new items assigned by nearest-centroid search | OneRec (3 levels, about 8K codes per level) [20][21] |
| RQ-KMeans+ | Improved init/utilization; codebook utilization reported 65% -> 92%, reconstruction error -15% | Tencent GPR [24] |
| RQ + OPQ suffix | 3 RQ layers (4096/1024/512) plus 2 OPQ codes (2 x 256) to separate near-duplicates | Kuaishou OneSearch [21][43] |
| Constrained RQ-KMeans | Balancing and dedup layer for conflicting items | MiniOneRec [15] |
| Hybrid sparse + dense | Generate SID, then refine with a dense vector | Baidu COBRA [40], Meta LIGER [39] |

Known pathologies and practical guidance:

- **The "hourglass" effect.** Intermediate RQ levels can collapse onto a few codes, which limits the ceiling of generative retrieval. A 2024 analysis proposes fixes [9].
- **Codebook collapse and imbalance** cause collisions. Uniform allocation, capacity limits, and k-means-based assignment all reduce it [21].
- Snap's GRID framework and its "Practitioner's Handbook" (CIKM 2025) benchmark RQ-VAE, RVQ and RQ-KMeans tokenizers inside a TIGER-style model on a common pipeline [8]. A companion study looks at how semantic-ID generative recommenders scale with model size [36].
- Google's PLUM (2025) adapts pre-trained LLMs to industrial generative recommendation with semantic IDs [38].

**Lessons.** The semantic-ID tokenizer is the "vocabulary design" of a generative recommender, and it is a first-class system component. Its collision rate, codebook utilization, refresh cadence, and how well it reflects collaborative signal, not only content, matter as much as the sequence model.

---

## 4. Kuaishou OneRec family

OneRec is the most complete public example of replacing an entire recommendation cascade with a single generative model in a short-video app serving hundreds of millions of users.

### 4.1 OneRec (V1) - February 2025 paper and June 2025 technical report

**Problem.** Remove the retrieve/pre-rank/rank cascade. Generate a session of recommended videos end to end, and align the generator with user preference through RL [10][11].

**Tokenizer.**

- Inputs (title, tags, ASR, OCR, cover, sampled frames) are compressed by a multimodal LLM (miniCPM-V-8B) to 1280 tokens, then by a QFormer to 4 query vectors.
- An item-pair contrastive loss injects collaborative signal, and title generation is used as an auxiliary task.
- RQ-KMeans then produces 3 code levels [20].

**Encoder.** Four pathways [20]:

- Static user features, through two dense layers.
- Short-term behavior: the last 20 interactions.
- Positive-feedback behavior: the last 256 high-engagement interactions.
- Lifelong history: up to 100,000 records. These are compressed by hierarchical k-means to 2,000 items, then by a QFormer to 128 tokens.

The encoder context is 1 + 20 + 256 + 128 = 405 tokens.

**Decoder.** A [BOS] token plus semantic-ID tokens, with causal self-attention, cross-attention to the encoder, and MoE feed-forward layers with top-k routing [20].

```
 user static ─┐
 short seq 20 ├─> Encoder (Transformer, RMSNorm) ──> 405 ctx tokens
 pos-fb 256  ─┤                                          │ cross-attn
 lifelong 1e5 ─> hier-kmeans 2000 ─> QFormer 128 ─┘      v
                         [BOS] s1 s2 s3 ──> Decoder (causal + MoE FFN)
                                                         │ beam search over SIDs
                                                         v
                       candidate sessions ──> Reward model (P-Score: CTR/LTR/VTR..)
                                                         │
                         preference alignment (DPO-style IPA in V1 paper;
                         ECPO = early-clipped GRPO in technical report)
```

**Alignment and RL.**

- The original paper used Iterative Preference Alignment (IPA). A reward model scores sampled sessions, and chosen/rejected pairs drive DPO-style updates [10].
- The technical report moved to ECPO, an early-clipped variant of GRPO that prevents exploding gradients on negative-advantage samples [20].
- It also added format rewards so the model generates only legal semantic IDs.
- "SIR" industrial alignment down-weights viral content. It is reported to cut viral-content exposure by 9.59% while keeping core metrics stable [20].

**Scale (as reported, approx.).** The technical report claims:

- Training MFU of about 24% and inference MFU of about 29%.
- Operating cost of about 10% of the previous cascade.
- Deployment on the main Kuaishou and Kuaishou Lite feeds, handling about a quarter of their traffic.
- App stay-time gains of about 0.5-1.2%.

These figures come from the report's abstract and should be verified there [11].

### 4.2 OneRec-V2 (August 2025)

**Problem.** In V1 most compute went into the encoder. Only a small fraction trained the generator, and RL from a learned reward model risked reward hacking [20].

**Lazy decoder-only architecture.**

- A context processor computes key/value pairs once and shares them across decoder layers. There is no per-layer KV projection, and the shared KV uses GQA.
- The decoder input is only [BOS] plus the first two semantic-ID tokens.
- Each layer has lazy cross-attention, causal self-attention, and an FFN, with MoE in the deeper layers [20].

At 1B parameters, the encoder-decoder used 296.36 GFLOPs and reached a converged loss of 3.28. The lazy decoder-only model used 18.89 GFLOPs and reached 3.27. That is about 94% less compute and about 90% less training resource for the same quality [20]. Secondary notes put OneRec-V2 MFU at 20%+, against roughly 5-10% for encoder-decoder setups [22].

**Scaling law.**

- Dense models from 0.1B to 8B parameters fit L(N) = E + A / N^α, with E = 3.13, A = 3660, α = 0.489.
- Converged loss falls from 3.57 (0.1B) to 3.19 (8B).
- A 4B-total / 0.5B-active MoE reaches 3.22, better than a 2B dense model [20].

**RL from real user feedback.**

- Rewards come directly from logged watch time, which is log-bucketed. The top 25% within a bucket get advantage +1, explicit negative feedback gets -1, and the rest get 0.
- GBPO (Gradient-Bounded Policy Optimization) keeps gradients for all samples but bounds the RL gradient by the BCE gradient [20].

Reported online stay-time gains are again in the sub-1% range (approx.) [12].

### 4.3 OneRec-Think, OpenOneRec, MiniOneRec

- **OneRec-Think (Oct 2025).** Built on Qwen3 with a vocabulary extended by semantic-ID tokens. Training runs in three stages: itemic alignment (mapping item tokens into text space), reasoning activation (chain-of-thought SFT), and reasoning enhancement with a reward that allows multiple valid answers. It reports a further +0.159% app stay time online [13].
- **OpenOneRec (Dec 2025).** Kuaishou's open release.
  - Model checkpoints: OneRec-1.7B and 8B, trained on about 33B tokens, and "-Pro" variants trained on about 130B tokens, including industrial data. All use Qwen3 backbones.
  - Items are represented as three-level "itemic tokens".
  - Post-training pipeline: multi-task SFT, then on-policy distillation to restore general reasoning, then RL.
  - RecIF-Bench: 8 tasks over 100M interactions from 200K users in short video, ads, and products [14].
- **MiniOneRec (Oct 2025).** A community open-source reproduction. It builds text-embedding semantic IDs (RQ-VAE / RQ-KMeans / constrained RQ-KMeans) and SFT co-trained with language alignment. It then applies GRPO with a rank-aware reward and a KL penalty, and decodes with constrained beam search through a logits processor [15].

Kuaishou extended the same recipe to e-commerce search (OneSearch), query suggestion (OneSug), live streaming (OneLive), and local services (OneLoc) [43][44].

**Lessons.**

- (1) In a generative recommender the encoder is mostly a context compressor, and compute belongs in the generating decoder. This is V2's main lesson.
- (2) RL post-training needs guardrails: format rewards, clipping and bounding of policy gradients, and anti-virality shaping.
- (3) Lifelong histories of about 10^5 events are compressed (clustering, then QFormer) instead of being attended to directly.

---

## 5. Meituan MTGR (and MTGRBoost, DFGR, EGA)

### 5.1 Problem

Meituan's food-delivery ranking depends heavily on cross features: user x item statistics such as the user's historical CTR on a category, or their preference by hour of day. Pure HSTU-style generative ranking drops these features. Meituan found that this hurt so much that even the largest generative model without cross features performed worse than a mid-sized DLRM [19].

### 5.2 Architecture

MTGR is a hybrid: a transformer encoder with HSTU-style user-level sequence compression, trained with a discriminative (CTR/CTCVR) objective [16][19].

- **Token groups:**
  - User tokens: static attributes, one token per attribute.
  - Sequence tokens: long-term history, one per item.
  - Real-time tokens: recent events.
  - Candidate tokens: MLP(concat(item embedding, cross-feature embedding)), plus position and exposure time.
- **Cross features stay on candidate tokens** and never enter the history, which avoids causal leakage.
- **Group LayerNorm (GLN):** separate normalization statistics and affine parameters per token group.
- **Dynamic masking:** static tokens are visible to all. Real-time tokens are causal, and a candidate can see them only if their timestamp is before that candidate's exposure time. Candidates cannot see each other (a diagonal block), so all candidates for a user are scored in one pass.
- **Loss** is computed only at candidate positions. Encoding the history once and scoring m candidates in parallel costs O((n+m)^2), against O(m·n^2) when candidates are scored separately [19].

```
 [U1..Uk | S1 ... Sn | R1 ... Rr | C1  C2  ...  Cm]
   static   long seq   real-time   candidates (item ⊕ cross feats)
 mask: U,S -> visible to all; R causal & time-gated per candidate; Ci ⟂ Cj
 heads: CTR / CTCVR on C positions only
```

**Scale (as reported, approx.).** The paper reports per-sample inference compute about 65x that of the DLRM baseline. It describes this as the largest offline and online gain in about two years, deployed on Meituan's delivery recommendation [16]. Training was supported by MTGRBoost, a companion system paper on large-scale generative-model training infrastructure [17].

Related Meituan work:

- **DFGR** ("Action is All You Need: Dual-Flow Generative Ranking Network") restructures action and item flows.
- **EGA-V1/V2** build fully generative advertising. EGA uses RQ-VAE semantic IDs (3 levels x 1024 codes) and decomposes generation into P(POI) x P(creative | POI). It puts auction mechanics into the model with token-level bids and a payment network that guarantees individual rationality (payment never exceeds bid). Incentive compatibility is enforced approximately through a Lagrangian on ex-post regret [24][42].

**Lessons.** Generative architecture and discriminative objectives can coexist. Keep the scaling benefits of user-level sequence encoding and parallel candidate scoring, and keep the cross features that a pure next-token model throws away.

---

## 6. Other generative ranking systems worth knowing

- **Xiaohongshu GenRank ("Towards Large-scale Generative Ranking", 2025).**
  - Trained on hundreds of billions of exposure logs.
  - An action-oriented sequence makes actions the main tokens and folds items in as attributes. This halves the sequence (2n_c -> n_c), cuts attention compute by 75%, and speeds up training by 78.7%.
  - ALiBi-style parameter-free position and time biases raise the total speedup to 94.8%.
  - Loss at history positions hurt AUC, and so did removing the causal mask. Real-time features stayed important while most historical aggregate features lost value [22][32].
- **ByteDance RankMixer (2025).**
  - Not a generative model, but part of the same scaling trend.
  - Self-attention is replaced by parameter-free token mixing, plus per-token FFNs and a dense-train / sparse-infer MoE.
  - Raises ranking MFU from about 4-5% to about 45% [23][33].
  - Successors: OneTrans and TokenMixer-Large [44].
- **Tencent GPR (2025, WeChat Channels ads).**
  - Four token types (user, organic, environment, ad), RQ-KMeans+ semantic IDs, and a heterogeneous hierarchical decoder with "thinking tokens".
  - Value-guided trie beam search.
  - Reported results: latency from 200ms+ to about 80ms, and 5 models collapsed into 1 [24][41].
- **Alibaba LUM (2025).** A three-step paradigm: pre-train a large user model generatively, query it, then use its outputs in discriminative models. It aims to unlock scaling laws in industrial recommendation [37].

---

## 7. Netflix: Foundation Model for Personalized Recommendation

### 7.1 Problem

Netflix ran many specialized models (rows, ranking, search, notifications). Each was trained on overlapping data with its own feature engineering, which made improvements hard to share. In a March 2025 tech-blog post, Netflix described consolidating user-preference learning into one large, centrally trained foundation model in the spirit of LLMs. Downstream systems consume it [25].

### 7.2 Design elements (as described in the blog; summarized via a public reimplementation's mapping)

- **Interaction tokenization.** User interaction events are tokenized, and adjacent related actions are merged, much like BPE in language models. Sequences stay informative but shorter [25][26].
- **Long-context handling.** Sparse or local attention, plus sliding-window sampling over long histories during training [25][26].
- **Multi-token / multi-attribute prediction.** Heads predict several future items and attributes such as genre, not only the next item ID. This reduces myopia [25][26].
- **Cold start.** Metadata-based embeddings are mixed with learned ID embeddings, with a gate weighted by item "age". New titles lean on metadata, and the model is incrementally retrained to add new IDs [25][26].
- **Embedding stability.** A projection, described as an orthogonal transformation, keeps embedding spaces comparable across retrains, so downstream consumers do not break [25][26].
- **Downstream use.** Three modes: (a) the model directly as a predictive model, (b) exported user and item embeddings, and (c) fine-tuning for specific surfaces [25].

```
 member events ──> tokenizer (merge adjacent actions) ──> FM transformer
                                                         │   (sparse attn,
                                                         │    multi-token heads)
                    ┌────────────────────┬───────────────┴──────────┐
                    v                    v                          v
          direct prediction      user/item embeddings        fine-tuned task models
          (next-title scores)    (stable via projection)     (rows, search, notifs)
```

Netflix reported that performance kept improving as it scaled the model and data, consistent with scaling-law behavior (approx.; see [25]).

**Lessons.** For companies with many surfaces, the main gain is organizational: one representation learner, with stable embedding contracts, feeding many consumers. The engineering work is concentrated in tokenization, cold-start mixing, and keeping embeddings stable across retrains.

---

## 8. LinkedIn 360Brew

360Brew ("A Decoder-only Foundation Model for Personalized Ranking and Recommendation", 2025) takes the most radical "LLM-as-recommender" route of the systems here [27][28].

- **Problem.** Replace many task-specific ranking models and their hand-engineered features with one model that reasons over textual member profiles and histories.
- **Interface.** Each task is written as a natural-language prompt. For example: given the member's profile and past job interactions (applied, viewed, dismissed), predict whether they will apply to, view, or dismiss a given job. No manual feature engineering is needed [28].
- **Scaling observations.** Performance improved with (a) more training tokens across five releases over nine months, (b) longer context, meaning more interaction history, and (c) larger model size [28].
- **Scale (approx.).** The paper abstract describes a decoder-only model of about 150B parameters covering 30+ ranking tasks across LinkedIn surfaces, matching or beating production models without task-specific fine-tuning. Verify the figures in [27].

**Lessons.** Text-native foundation rankers trade very high serving cost for much less feature and model sprawl, plus zero-shot transfer to new surfaces. In practice they are often distilled or used for some surfaces rather than as the single high-QPS scorer.

---

## 9. Pinterest PinFM (and PinRec, UniPinRec)

PinFM ("Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform", 2025) pre-trains a large transformer on user activity sequences and then fine-tunes it inside multiple downstream ranking models [29].

- **Pre-train / fine-tune split.** Large-scale pre-training on lifelong activity sequences. The module is then fine-tuned inside each downstream ranker, for example home-feed or related-pins ranking, and not served as a separate system [29].
- **Serving efficiency (approx.).** The abstract describes a deduplicated cross-attention design (DCAT). It computes the user-sequence part once per request and shares it across all candidates, and is reported to improve throughput substantially (approx. 600%). It also reports gains in engagement with new items. Verify the figures in [29].
- **Model size (approx.).** The parameter count is reported in the tens of billions (approx. 20B+) [29].
- **Related work.** PinRec (2025) is an outcome-conditioned, multi-token generative retrieval model [30]. UniPinRec (2026) unifies generative retrieval and ranking at Pinterest scale [46].

**Lessons.** The "user-sequence foundation model plus downstream fine-tune" pattern deploys incrementally. It slots into an existing ranker as a learned feature extractor. Dedup and KV-sharing tricks like DCAT (similar to M-FALCON and MTGR's masking) are what make it affordable.

---

## 10. X (Twitter) For You: Phoenix transformer ranker

X's open-sourced `x-algorithm` repository describes the 2025-2026 For You pipeline [31]. (Public commentary often calls Phoenix "Grok-based". The repository itself says only that it is a transformer. "Grox" is a separate set of LLM content classifiers [31].)

- **Home-mixer pipeline:** query hydration, then candidate sources, then hydration, then filters, then scoring, then selection, then post-selection visibility filtering, then blending.
- **Candidate sources:**
  - Thunder: an in-memory store of recent posts from followed accounts.
  - Phoenix retrieval: embedding nearest-neighbor search for out-of-network posts.
  - SimClusters.
  - Popular posts.
- **Phoenix ranker.** Its main input is the viewer's recent engagement sequence. It predicts per-action probabilities for engagement (favorite, reply, repost, quote, share), clicks, attention (dwell, video view), follow, and negative actions (not interested, mute, block, report). Training and serving are written in JAX with a Rust serving layer [31].
- **Candidate isolation.** "During transformer inference, candidates cannot attend to each other - only to the viewer context." Each post's score is therefore independent of how candidates are batched. This is the same masking idea as M-FALCON and MTGR [31].
- **Hash embeddings.** Retrieval and ranking both use multiple hash functions for embedding lookup, so there is no ID vocabulary to maintain [31].
- **Value model.** Final score = Σ w_i · P(action_i). It is followed by an author-diversity decay, an out-of-network discount, a new-author boost, and a DPP-based re-ranker (VMRanker) that trades score for diversity [31].

```
 viewer engagement seq ─┐
 Thunder (in-network)  ─┼─> candidates ─> filters ─> Phoenix transformer
 Phoenix ANN (OON)     ─┤                         (cands attend only to viewer ctx)
 SimClusters / popular ─┘                                 │ P(fav), P(reply), ...
                                                          v
                      Σ w_i P(a_i) ─> diversity/OON adj ─> DPP rerank ─> VF ─> blend
```

**Lessons.** A transformer over the engagement sequence can replace a hand-built feature ranker without becoming end-to-end generative. Multi-action prediction combined with an explicit, tunable value model keeps product control (weights, negative signals) outside the network.

---

## 11. Scaling laws for recommendation

| Work | Model family | Scaling claim (as reported) |
|---|---|---|
| HSTU (Meta, 2024) [1][18] | Sequential transducer | Power law in compute over 3 orders of magnitude; HR@100 and NE linear in ln(compute) |
| Wukong (Meta, 2024) [34] | Stacked factorization machines (DLRM-style) | Scaling law for dense interaction stacks across two orders of magnitude in complexity (approx.) |
| OneRec-V2 (Kuaishou, 2025) [20] | Lazy decoder-only + MoE | L(N) = 3.13 + 3660 / N^0.489, 0.1B-8B |
| 360Brew (LinkedIn, 2025) [28] | Text decoder LLM | Gains with tokens, context length, model size |
| LUM (Alibaba, 2025) [37] | Generative pre-trained user model | Scaling unlocked by three-step generative -> discriminative paradigm |
| SID scaling study (Snap, 2025) [36] | Semantic-ID GR | Studies model-size scaling limits of SID-based GR |
| ULTRA-HSTU (Meta, 2026) [35] | HSTU successor | "Bending" the scaling curve via architecture/efficiency |

Common findings:

- (a) Gains are roughly logarithmic in compute, so each 10x buys a fixed increment. For HSTU this is about 4.5 points of HR@100 and about 1.2 points of NE per 10x, per secondary analysis [18].
- (b) Sequence length is a scaling axis on equal footing with depth and width.
- (c) MoE gives better loss per active FLOP [20].
- (d) Embedding-table size alone does not scale. Dense compute does.

---

## 12. Comparison table

| System | Org / year | Paradigm | Item representation | Stage replaced | Notable scale / result (as reported) |
|---|---|---|---|---|---|
| HSTU / GR | Meta 2024 | Causal sequential transducer | Atomic IDs + actions interleaved | Retrieval & ranking models | 1.5T params; +12.4% online; 5.3-15.2x faster attention [1] |
| TIGER | Google 2023 | Seq2seq generative retrieval | Semantic IDs (RQ-VAE) | Retrieval | SOTA on Amazon datasets; cold-start gains [4] |
| OneRec V1 | Kuaishou 2025 | Enc-dec + MoE + RL (IPA/ECPO) | 3-level RQ-KMeans SIDs | Whole cascade | MFU about 24%/29%; OPEX about 10% (approx.) [11] |
| OneRec-V2 | Kuaishou 2025 | Lazy decoder-only + RL from real feedback | SIDs | Whole cascade | 94% less compute at equal loss; 8B scaling [20] |
| MTGR | Meituan 2025 | Hybrid generative-encoder, discriminative loss | Atomic IDs + cross features | Ranking | about 65x FLOPs/sample vs DLRM (approx.) [16] |
| GenRank | Xiaohongshu 2025 | Action-oriented generative ranking | Actions as tokens | Ranking | 94.8% training speedup [22] |
| GPR | Tencent 2025 | Decoder + value-guided trie search | RQ-KMeans+ SIDs | Ads cascade | 5 models -> 1; latency to about 80ms [24] |
| Netflix FM | Netflix 2025 | Central foundation model | Merged interaction tokens + metadata | Feeds many models | Scaling with data/params (approx.) [25] |
| 360Brew | LinkedIn 2025 | Text decoder LLM ranker | Natural-language prompts | Many task rankers | about 150B params, 30+ tasks (approx.) [27] |
| PinFM | Pinterest 2025 | Pre-trained user-seq FM, fine-tuned downstream | IDs + content | Feature extractor in rankers | about 20B+ params; DCAT about 600% throughput (approx.) [29] |
| Phoenix | X 2025-26 | Transformer ranker + ANN retrieval | Hash embeddings | Main ranker | Candidate-isolated attention; multi-action heads [31] |

---

## Key takeaways for system designers

1. **Restructure training before you scale.** Moving from impression-level samples to one causal sequence per user (HSTU, GenRank, MTGR) gives order-of-magnitude efficiency gains before any parameters are added.
2. **Candidate-isolated attention is the universal serving trick.** M-FALCON (Meta), dynamic masking (Meituan), DCAT (Pinterest) and Phoenix (X) all encode the user context once, share its KV, and let candidates attend only to the context. This makes target-aware transformer ranking linear in the number of candidates.
3. **Treat the tokenizer as infrastructure.** Semantic IDs need a pipeline: embedding model, RQ-VAE or RQ-KMeans, collision handling, valid-prefix tries for constrained decoding, and a refresh policy. Monitor codebook utilization, collision rate, and the "hourglass" effect [9][21].
4. **Keep the features that carry signal.** MTGR shows that dropping cross features to be "purely generative" can lose more than scaling gains back. Hybrid designs are legitimate.
5. **Put compute in the generator.** OneRec-V2's lazy decoder-only design matched encoder-decoder quality with about 94% less compute. Compress long context (clustering, QFormer) instead of attending over 10^5 raw events.
6. **RL post-training needs stabilization.** Use format rewards, clipped or bounded policy gradients (ECPO, GBPO), rewards grounded in real user feedback, and anti-virality or diversity shaping. Learned reward models alone invite reward hacking.
7. **MFU is the new north-star infrastructure metric.** Going from about 5% (DLRM) to 20-45% (OneRec, RankMixer) changes what model sizes are affordable.
8. **Foundation models win organizationally.** Netflix, PinFM and 360Brew show that one user-sequence model feeding many consumers, with stable embeddings, reduces duplicated feature work. Plan for embedding-versioning contracts.
9. **Keep business control explicit.** Even end-to-end systems keep value models, auction or payment modules, diversity re-rankers and visibility filters outside or on top of the generator (X value model, EGA payment network).
10. **Expect logarithmic returns.** Budget for about 10x compute per fixed metric increment, and use MoE and sequence-length scaling to stretch the curve.

---

## References

1. Zhai, J. et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." Meta, ICML 2024. https://arxiv.org/abs/2402.17152
2. meta-recsys. "generative-recommenders" (HSTU, M-FALCON, DLRM-v3, HSTU Ultra code). Meta, 2024-2026. https://github.com/meta-recsys/generative-recommenders
3. Zhai, J. et al. ICML 2024 slides, "Actions Speak Louder than Words." Meta, 2024. https://icml.cc/media/icml-2024/Slides/32684.pdf
4. Rajput, S. et al. "Recommender Systems with Generative Retrieval" (TIGER). Google, NeurIPS 2023. https://arxiv.org/abs/2305.05065
5. Rajput, S. et al. TIGER, NeurIPS proceedings PDF. Google, 2023. https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf
6. Botta, E. "RQ-VAE-Recommender" (PyTorch implementation of generative retrieval with semantic IDs). 2024-2025. https://github.com/EdoardoBotta/RQ-VAE-Recommender
7. Lee, D. et al. "Autoregressive Image Generation using Residual Quantization" (RQ-VAE). POSTECH/Kakao, 2022. https://arxiv.org/abs/2203.01941
8. Ju, C. et al. "Generative Recommendation with Semantic IDs: A Practitioner's Handbook" (GRID). Snap, CIKM 2025. https://arxiv.org/abs/2507.22224 ; code: https://github.com/snap-research/GRID
9. "The Hourglass phenomenon in residual-quantized semantic IDs for generative search" (arXiv 2407.21488). 2024. https://arxiv.org/html/2407.21488v1
10. Deng, J. et al. "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment." Kuaishou, 2025. https://arxiv.org/abs/2502.18965
11. Kuaishou OneRec Team. "OneRec Technical Report." Kuaishou, 2025. https://arxiv.org/abs/2506.13695
12. Kuaishou OneRec Team. "OneRec-V2 Technical Report." Kuaishou, 2025. https://arxiv.org/abs/2508.20900
13. Kuaishou OneRec Team. "OneRec-Think: In-Text Reasoning for Generative Recommendation." Kuaishou, 2025. https://arxiv.org/abs/2510.11639 ; code: https://github.com/wangshy31/OneRec-Think
14. Kuaishou OneRec Team. "OpenOneRec Technical Report" and repository. Kuaishou, 2025. https://arxiv.org/abs/2512.24762 ; https://github.com/Kuaishou-OneRec/OpenOneRec
15. Kong et al. "MiniOneRec" (open-source generative recommendation framework). 2025. https://arxiv.org/abs/2510.24431 ; https://github.com/AkaliKong/MiniOneRec
16. Han, R. et al. "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan." Meituan, 2025. https://arxiv.org/abs/2505.18654
17. "MTGRBoost: Boosting Large-scale Generative Recommendation Models in Meituan." Meituan, 2025. https://arxiv.org/abs/2505.12663
18. Xing, H. "HSTU: The Cornerstone of Scaling Laws" (fun-rec-mdbook, technical notes). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/hstu.md
19. Xing, H. "MTGR: A Hybrid Paradigm" (fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/mtgr.md
20. Xing, H. "End-to-End Generative Recommendation" (OneRec / OneRec-V2 notes, fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part8-e2e/e2e-recommendation.md
21. Xing, H. "Codebook Quantization & Semantic IDs" (fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part6-gr-basic/codebook-quantization.md
22. Xing, H. "The Generative Ranking Paradigm" and "Architectural Foundations of Generative Recommendation" (fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/generative-ranking.md ; https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part6-gr-basic/gr-architecture.md
23. Xing, H. "RankMixer: A Hardware-Efficient Architecture" (fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part7-scaling/rankmixer.md
24. Xing, H. "End-to-End Generative Advertising" (EGA, GPR notes, fun-rec-mdbook). 2026. https://github.com/Haozhe-Xing/fun-rec-mdbook/blob/main/src/en/part8-e2e/e2e-advertising.md
25. Hsiao, K.-J., Feng, Y., Lamkhede, S. "Foundation Model for Personalized Recommendation." Netflix Technology Blog, 2025. https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39
26. Venkatesan, M. "Foundation Model for Personalized Recommendation (H&M Dataset)" (open reimplementation mapping the Netflix design). 2026. https://github.com/madanmohan-venkatesan/Recommender-systems-on-HNM-fashion-dataset
27. Firooz, H. et al. "360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation." LinkedIn, 2025. https://arxiv.org/abs/2501.16450
28. LLMPersonalization. "360Brew-V1" repository README. LinkedIn, 2025. https://github.com/LLMPersonalization/360Brew-V1
29. "PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform." Pinterest, 2025. https://arxiv.org/abs/2507.12704
30. "PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems." Pinterest, 2025. https://arxiv.org/abs/2504.10507
31. xAI / X. "x-algorithm" (For You feed: home-mixer, Thunder, Phoenix). 2025-2026. https://github.com/xai-org/x-algorithm
32. "Towards Large-scale Generative Ranking" (GenRank). Xiaohongshu, 2025. https://arxiv.org/abs/2505.04180
33. "RankMixer: Scaling Up Ranking Models in Industrial Recommenders." ByteDance, 2025. https://arxiv.org/abs/2507.15551
34. Zhang, B. et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation." Meta, ICML 2024. https://arxiv.org/abs/2403.02545
35. "Bending the Scaling Law Curve in Large-Scale Recommendation Systems" (ULTRA-HSTU). Meta, 2026. https://arxiv.org/abs/2602.16986
36. "Understanding Generative Recommendation with Semantic IDs from a Model-scaling View." Snap, 2025. https://arxiv.org/abs/2509.25522
37. "Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model" (LUM). Alibaba, 2025. https://arxiv.org/abs/2502.08309
38. "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations." Google/YouTube, 2025. https://arxiv.org/abs/2510.07784
39. "Unifying Generative and Dense Retrieval for Sequential Recommendation" (LIGER). Meta, 2024. https://arxiv.org/abs/2411.18814
40. "Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations" (COBRA). Baidu, 2025. https://arxiv.org/abs/2503.02453
41. "GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation." Tencent, 2025. https://arxiv.org/abs/2511.10138
42. "EGA-V2: An End-to-end Generative Framework for Industrial Advertising." Meituan, 2025. https://arxiv.org/abs/2505.17549
43. "OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search." Kuaishou, 2025. https://arxiv.org/abs/2509.03236
44. Huang, Y. et al. "Awesome-Generative-Recommendation" (curated industrial paper list). 2025-2026. https://github.com/uestc-huangyw/Awesome-Generative-Recommendation
45. "How Well Does Generative Recommendation Generalize?" Meta, 2026. https://arxiv.org/abs/2603.19809
46. "UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale." Pinterest, 2026. https://arxiv.org/abs/2606.00422
