# 05. Sequential and User-History Modeling: From GRU4Rec to Lifelong Sequences and Foundation User Models

## Overview

Most modern recommender systems are, at heart, systems for turning a user's history into a
prediction. A user's ordered stream of actions (clicks, views, saves, purchases, hides,
searches) is the richest signal a platform owns. Over the last decade the industry has moved
through four broad generations of models for that stream:

1. **Session RNNs (2015 to 2017).** GRU4Rec showed that a recurrent network over a session
   could beat item-to-item similarity for next-click prediction [1], although later
   benchmarks showed tuned nearest-neighbour baselines remained strong [2].
2. **Self-attention for next-item prediction (2018 to 2023).** SASRec [3] and BERT4Rec [4]
   brought Transformers to sequential recommendation. A long replicability debate followed
   [5][6].
3. **Target-attention and Transformers inside industrial CTR rankers (2018 onward).** Alibaba's
   DIN/DIEN line [8], BST [7], and the long-sequence family MIMN [9], SIM [10], ETA [11],
   SDIM [13], TWIN [14] and TWIN V2 [15] put user histories of 10^2 to 10^5 behaviours into
   the ranking stage under tight latency budgets. Pinterest's TransAct [20] and TransAct V2
   [21] do the same for real-time and lifelong action sequences.
4. **User representations and foundation user models (2020 onward).** PinnerSage [18] and
   PinnerFormer [19] produce reusable user embeddings. PinFM [23], Meta's HSTU generative
   recommenders [24], Netflix's foundation model [25][26] and ByteDance's LONGER [17] pretrain
   large sequence models on billions of interactions and share them across many surfaces.

This chapter walks through each design: what problem it solved, how it is built, the public
scale numbers, the key tricks, and what the industry learned. The last sections give a
comparison table and design takeaways.

A note on sources: every number below comes from the cited paper or blog. Figures that only
appear in secondary summaries are marked "reported" or "approximately".

```
                     The user-history design space
  +-------------------------------------------------------------------------+
  |                 short (10^1-10^2)    long (10^3-10^4)    lifelong (10^5+) |
  | retrieval       GRU4Rec, SASRec,     PinnerFormer,       PinFM, HSTU,     |
  | (user tower)    BERT4Rec, KuaiFormer Netflix FM          Netflix FM       |
  | ranking         DIN, BST, TransAct   MIMN, LONGER,       SIM, ETA, SDIM,  |
  | (target-aware)                       TransAct V2         TWIN, TWIN V2    |
  +-------------------------------------------------------------------------+
     cost grows with length x candidates x model depth -> most tricks attack this
```

## 1. Academic foundations: next-item prediction

### 1.1 GRU4Rec (Hidasi et al., 2015)

**Problem.** In e-commerce and media sites, many sessions are anonymous and short. The
standard approach before 2015 was item-to-item similarity (for example "people who viewed X
also viewed Y"), which uses only the last item and ignores the rest of the session [1].

**Architecture.** GRU4Rec feeds the session's item sequence, one item per step, into a GRU.
The hidden state after each step is scored against all items to predict the next click. The
paper made RNNs practical for this task with several changes [1]:

- **Session-parallel mini-batches.** Sessions have very different lengths, so the batch is
  built from many sessions in parallel. When one session ends, the next one takes its slot
  and the hidden state is reset.
- **Sampled output.** Scoring every item is too expensive, so the loss uses other items in
  the mini-batch as negatives (an early form of in-batch negatives).
- **Ranking losses.** The paper introduced pairwise ranking losses (BPR and TOP1) in place of
  plain cross-entropy.

**Lessons.** Jannach and Ludewig (RecSys 2017) showed that a simple session-based
nearest-neighbour method beat GRU4Rec in most of their tested configurations, and the best
results often came from combining the two [2]. Hidasi and Karatzoglou later published a
follow-up ("Recurrent Neural Networks with Top-k Gains", 2017) on loss functions and
sampling for RNN session recommenders. The lasting lesson: sequence models need careful loss and negative-sampling
choices, and strong simple baselines should always be reported.

### 1.2 SASRec (Kang and McAuley, 2018)

**Problem.** Markov-chain models use only the last few actions and work well on sparse data.
RNNs can use the full history but are slow and need dense data. SASRec aimed to get both:
capture long-range dependencies, but make predictions that depend mostly on a few relevant
actions [3].

**Architecture.** A causal (left-to-right) Transformer encoder over item embeddings plus
learned position embeddings. At each position the output is matched by dot product to the
next item's embedding. The original training used binary cross-entropy with one sampled
negative per positive.

**Results.** The authors report SASRec beat MC, CNN and RNN baselines on both sparse and dense
datasets, and was about an order of magnitude more efficient than comparable CNN/RNN models
[3]. SASRec became the default academic baseline and, later, the conceptual ancestor of most
industrial "user tower" Transformers.

### 1.3 BERT4Rec (Sun et al., 2019)

**Problem.** Left-to-right models only use past context. BERT4Rec argued that this
unidirectional design limits how well hidden representations capture the history [4].

**Architecture.** A bidirectional Transformer encoder trained with a Cloze (masked item)
task: a share of items in each sequence is replaced by a [mask] token and predicted from both
left and right context. At inference a [mask] token is appended to the end of the sequence
and its final hidden state is used to predict the next item [4]. The paper notes that masking
k items in a sequence of length n gives C(n, k) possible training samples, versus n for
next-item training.

### 1.4 The replicability debate and gSASRec

Petrov and Macdonald (RecSys 2022) reviewed published BERT4Rec versus SASRec comparisons
[5]. They found:

- BERT4Rec was not consistently better than SASRec across the literature.
- They could not reproduce the original BERT4Rec results with the default configuration.
- The original results could be reproduced if training ran much longer, up to about 30x.
- Their Hugging Face Transformers based implementation reproduced the results on 3 of 4
  datasets with up to 95% less training time.

Their follow-up, gSASRec (RecSys 2023), pins down the cause [6]. Negative sampling raises the
share of positives the model sees during training, so the model becomes **overconfident**: it
overestimates the probability of positive interactions. gSASRec keeps the SASRec architecture
but uses more negatives and a generalized BCE (gBCE) loss that corrects for the sampling rate.
They report gSASRec can beat BERT4Rec (for example +9.47% NDCG on MovieLens-1M) while training
about 73% faster, and that it scales to catalogues with over 1 million items, where BERT4Rec's
full softmax does not [6].

**Lesson for system designers.** The training objective and negative-sampling scheme often
matter more than uni- versus bidirectional attention. This is why most industrial user towers
use a causal model with sampled softmax plus logQ correction (see PinnerFormer, KuaiFormer and
PinFM below).

## 2. Target attention in industrial rankers: DIN, DIEN and BST (Alibaba)

### 2.1 DIN and DIEN

The Alibaba advertising team introduced **target attention** for CTR models. In DIN, a local
activation unit weights each historical behaviour by its relevance to the candidate ad, so
the user representation changes with each candidate (as summarised in the DIEN paper [8]). DIN does not
model relationships between items within the sequence.

DIEN (AAAI 2019) adds two layers [8]:

- An **interest extractor** GRU over the behaviour sequence, with an auxiliary loss that
  supervises each hidden state with the next behaviour.
- An **interest evolving** layer, AUGRU, which scales the GRU update gate by the attention
  score against the target item, so behaviours unrelated to the target have less effect on
  the hidden state.

The DIEN paper reports deployment in Taobao display advertising with a 20.7% CTR improvement
[8]. However, sequential GRUs are slow to serve, which limited DIEN in production to about 50
behaviours according to the MIMN paper [9].

### 2.2 BST: Behavior Sequence Transformer (Taobao, 2019)

**Problem.** Embedding-and-MLP models such as WDL and DIN mostly concatenate or pool features
and ignore the order of user behaviours [7].

**Architecture.**

```
  other features (user profile, context, cross)        target item
          |                                                 |
     embeddings                     behaviour seq: item_1 .. item_n
          |                                 |  (item_id, category_id,
          |                                 |   position = t_now - t_click)
          |                         +-------v-------------------+
          |                         | Transformer layer          |
          |                         | (self-attn over seq+target)|
          |                         +-------+-------------------+
          +-------------+-------------------+
                        |  concat
                   MLP (3 layers) -> sigmoid -> pCTR
```

Key choices reported for BST [7]:

- Each sequence token is built from item_id and category_id embeddings plus a **position
  feature defined as the time gap** between the recommendation time and the click time, not
  sinusoidal positions.
- The target item is appended to the sequence so self-attention can mix candidate and history.
- It runs in the **ranking stage** of Taobao recommendation.

**Results.** In the paper's table BST reached offline AUC 0.7894 versus 0.7734 for WDL and
0.7866 for DIN, and an online CTR gain of +7.57% versus +4.55% for DIN [7]. Secondary summaries
report online latency rising from about 13 ms to 20 ms (approximate). The paper found that a
single Transformer block worked best on its data; stacking more blocks did not help.

**Lesson.** BST is the template that TransAct and many later rankers follow: a short recent
sequence, a light Transformer, candidate-aware fusion, and the result concatenated into a
standard DNN ranker.

## 3. Long and lifelong sequences in ranking

The obvious next step is to use much longer histories. Lifelong histories on Taobao, Kuaishou
or Pinterest contain 10^4 to 10^5 behaviours. Full target attention costs O(L x d) per
candidate, and a ranker scores hundreds to thousands of candidates per request, so naive
attention is not affordable. Each design below cuts this cost in a different way.

### 3.1 MIMN and the User Interest Center (Alibaba, KDD 2019)

**Problem.** Production DIEN used about 50 behaviours due to serving pressure, while longer
histories clearly helped offline [9].

**Design.** A **system and algorithm co-design**:

- **UIC (User Interest Center)** is a separate serving module that keeps per-user interest
  state and updates it **incrementally** whenever a behaviour event arrives, not at request
  time.
- **MIMN (Multi-channel user Interest Memory Network)** is a memory-network model (based on
  Neural Turing Machine style memory with multiple channels) whose memory is exactly what UIC
  stores and updates.

At ranking time the model reads the precomputed memory, so latency is constant regardless of
history length. The authors claim the design could in principle handle unlimited length, and
it handled sequences in the thousands [9].

**Results.** On an Alibaba dataset MIMN improved AUC over DIEN by about 0.01. In an online A/B
test from 2019-03-30 to 2019-05-10, MIMN with UIC gave +7.5% CTR and +6% RPM over DIEN [9].

**Lesson.** The memory is user-only (not candidate-aware), so it compresses away
target-specific detail. The SIM paper later reported that MIMN accuracy drops when sequences
grow 10x or more [10].

### 3.2 SIM: Search-based Interest Model (Alibaba, CIKM 2020)

**Problem.** Model lifelong histories (up to 54,000 behaviours) in CTR prediction while
staying candidate-aware [10].

**Architecture.** Two cascaded search units:

```
 lifelong behaviours (10^4 - 10^5)
          |
 +--------v-------------------------------+
 | GSU: General Search Unit                |
 |  hard search: behaviours with the same  |
 |   category as the candidate             |   offline: User Behavior Tree
 |  soft search: MIPS on embeddings        |<--  key1 = user_id
 +--------+-------------------------------+     key2 = category_id
          | top-K sub-sequence (SBS)            value = behaviour items
 +--------v-------------------------------+     (distributed, ~22 TB)
 | ESU: Exact Search Unit                  |
 |  multi-head target attention on SBS     |
 |  + time-interval embeddings             |
 +--------+-------------------------------+
          v
   concat with short-term DIN features -> MLP -> pCTR
```

- **Soft search** embeds behaviours and uses maximum inner product search to get the top-K.
- **Hard search** simply keeps behaviours in the same category as the candidate.
- The authors found top-K lists from the two methods were very similar, and hard search was
  much cheaper. They deployed **hard search** online [10].
- The hard-search index is a two-level **User Behavior Tree (UBT)**: first key user ID, second
  key category ID, value the behaviour items. It is built offline, is distributed, and reaches
  about 22 TB [10].

**Results.** Deployed in Alibaba display advertising since 2019, with +7.1% CTR and +4.4% RPM,
and a maximum modelled length of 54,000, about 54x MIMN [10]. Secondary write-ups report the
added serving latency as only a few milliseconds (approximate) [28].

**Lesson.** SIM established the **two-stage GSU/ESU pattern** that nearly every later
lifelong-sequence model adopts. Its weakness is that the GSU relevance metric (category match
or a separately trained embedding) differs from the ESU's attention metric.

### 3.3 ETA and SDIM: hashing the GSU

**ETA (Alibaba, 2021).** ETA replaces the separate GSU with **locality-sensitive hashing**
inside the model so retrieval is end-to-end trainable [11]. Each item embedding is turned into
a binary signature with SimHash (random projections, keep the sign). The top-k behaviours are
chosen by Hamming distance to the candidate's signature, which needs only cheap bit
operations. Multi-head target attention then runs on those k items. Because the hash uses the
same embeddings the attention uses, the GSU and ESU stay roughly consistent. The paper reports
a 3.1% GMV gain [11]; the extended ETA-Net version reports +1.8% CTR and +3.1% GMV on Taobao
over two-stage methods [12].

**SDIM (Meituan, CIKM 2022).** SDIM goes further and drops the attention step [13]. It hashes
the candidate and every behaviour with several hash functions, then gathers (sums) the
behaviours that share the candidate's signature. The authors prove this sampling approximates
target attention's pattern. Behaviour hashing is moved into a separate **Behavior Sequence
Encoding (BSE)** service, so the CTR model only does a lookup. Meituan reports +2.98% CTR and
+2.69% VBR in its search system [13].

### 3.4 TWIN: TWo-stage Interest Network (Kuaishou, KDD 2023)

**Problem.** In SIM-like systems, the GSU's relevance metric differs from the ESU's target
attention. The GSU therefore misses behaviours the ESU would rate highly and keeps ones the ESU
would ignore [14].

**Key idea: consistency.** TWIN makes the GSU and ESU "twins". The **Consistency-Preserved
GSU (CP-GSU)** uses the same target-attention relevance as the ESU, with the same
architecture and parameters [14]. CP-GSU scores all ~10^4 long-term behaviours and keeps the
**top 100**. The ESU then does weighted pooling (multi-head target attention) over those 100
finalists [14].

**Making full attention affordable.** Running real target attention on 10^4 to 10^5
behaviours per candidate would be too slow. TWIN **splits behaviour features**:

- **Inherent item features** (video ID, author, tags, duration and so on) do not depend on the
  user. Their linear projections (the keys) can be **precomputed and cached** per item, and
  shared across all users.
- **User-item cross features** (for example when the user watched it and how long they played
  it) are user-specific. TWIN compresses them to a cheap scalar bias added to the attention
  logit, avoiding a full projection.

```
  per behaviour j:  K_j = W * inherent_j     (cached offline per item)
                    b_j = w . cross_j         (cheap, per user)
  per candidate c:  score_j = (Q(c) . K_j) / sqrt(d) + b_j
  CP-GSU: top-100 by score_j  ->  ESU: multi-head attention over top-100
```

**Scale.** Offline experiments use a production dataset of about 46 billion samples. With
optimized online infrastructure the authors report a **99.3% reduction** in the compute
bottleneck, which allowed deployment on Kuaishou's main traffic [14]. Secondary summaries
report watch-time gains of about 1.3% to 6.2% across the Featured-Video, Discovery and Slide
tabs (approximate; check the paper's tables) [29].

### 3.5 TWIN V2 (Kuaishou, CIKM 2024)

**Problem.** Users' full life-cycle histories now exceed 10^5 to 10^6 behaviours, beyond what
even TWIN can scan online [15].

**Design: divide and conquer.**

1. **Offline hierarchical clustering.** Each user's behaviours are grouped into clusters of
   similar items. Secondary descriptions say this is recursive k-means that splits clusters
   until each leaf has at most a set number of items (gamma) [29]. Limiting cluster size
   compresses sequences well beyond 10^5 to a length manageable online [15].
2. **Cluster representation.** Each cluster becomes one virtual behaviour (aggregated features).
3. **Cluster-aware target attention.** The GSU/ESU run on clusters, and attention scores are
   adjusted by cluster size so large clusters (strong interests) carry their weight [15].

**Results.** Deployed on primary traffic serving hundreds of millions of daily active users.
The paper's Table 5 lists watch-time gains of +0.672%, +0.800% and +0.728% on the
Featured-Video, Discovery and Slide tabs; the authors note that a 0.1% gain is significant at
Kuaishou [15].

### 3.6 LONGER (ByteDance, RecSys 2025)

**Problem.** Two-stage retrieval (SIM, TWIN) is lossy. ByteDance wanted **end-to-end**
Transformer modelling of sequences up to 10,000 items in ranking [17].

**Design** [17][33]:

- **Global tokens** (for example the target item and user tokens) that attend across the
  whole context to stabilise attention over long sequences.
- **Token merge**: adjacent behaviours are merged into fewer tokens, with lightweight
  **InnerTransformers** that keep detail inside each merged group.
- **Hybrid attention**: cross-attention from a small set of query tokens over the long
  sequence, followed by self-attention layers on the short output, to cut quadratic cost.
- Engineering: mixed-precision training, activation recomputation, and **KV-cache serving**
  so the user-only part is computed once per request and shared across candidates.

**Results.** Consistent offline and online gains in advertising and e-commerce. The v2 paper
says LONGER is fully deployed in dozens of ByteDance scenarios serving billions of users [17].

## 4. Pinterest: real-time, batch and lifelong user sequences

Pinterest has published one of the most complete public lines of user-history work, so it is
worth following in order.

### 4.1 PinnerSage (KDD 2020): multi-embedding users

**Problem.** One user vector cannot represent users with many unrelated interests (for
example recipes, tattoos and home decor) [18].

**Design.** Take the user's action pins from the last 90 days, cluster their PinSage
embeddings with Ward hierarchical clustering, and represent each cluster by its **medoid**
(a real pin from the user's history, which keeps clusters coherent and robust to outliers).
Clusters get an importance score; at retrieval time the system samples a few medoids and uses
each as an ANN query [18]. A/B tests showed it beat single-embedding methods.

### 4.2 PinnerFormer (KDD 2022): a batch user embedding trained for the long term

**Problem.** Pinterest wanted a single user embedding, computed in a **daily batch job**, that
works nearly as well as a real-time embedding, and that predicts long-term engagement rather
than only the next action [19].

**Architecture.**

```
 last N actions (pin, action type, surface, time features)
      |  each pin: 256-d PinSage embedding (frozen) + learned feature embeddings
 +----v------------------------------+
 | causal Transformer encoder         |
 +----+------------------------------+
      v
 user embedding (L2-normalised), computed daily for every user
      |
      v  ANN retrieval / ranking feature, matched against pin embeddings
```

**Dense all-action loss.** Instead of predicting only the next action (SASRec style), each
position's embedding is trained to predict **all positive engagements (repins, close-ups,
clicks) in a future window of up to 28 days** [19]. Training uses sampled softmax with a mix of
in-batch and random negatives and logQ correction [19].

**Why it matters.** The objective makes the embedding robust to staleness. Moving from
real-time to daily batch inference cut Recall@10 by 13.9% with the SASRec-style next-action
objective, but only by 8.3% with the dense all-action objective [19]. A/B tests showed better
engagement and retention than the previous user representation, and it has been in production
since fall 2021 [19]. A secondary summary reports the team did not scale beyond about 256
actions (approximate).

**Lesson.** For an embedding that will be computed offline and reused, train it to forecast a
window of future behaviour, not one next click.

### 4.3 TransAct (KDD 2023): real-time actions inside the ranker

**Problem.** PinnerFormer is updated daily, so Homefeed ranking could not react to what a user
did seconds ago. A real-time sequence model is needed, but running a full user model on every
action is expensive [20].

**Hybrid design.** TransAct is a module inside Pinterest's Homefeed ranking model (Pinnability)
that encodes the most recent actions in real time, while PinnerFormer and other batch features
provide long-term context [20].

```
   real-time action stream (Kafka-like)                  candidate pin
          |                                                    |
   latest ~100 actions: PinSage emb + action type + time       |
          |                                                    |
   early fusion: concat each action with candidate embedding <-+
          |
   Transformer encoder (small: ~2 layers, reported)
          |
   output compression: first K tokens + max pooling -> flatten
          |
   concat with PinnerFormer user emb + other features -> DCN-style ranker
          |
   multi-head outputs (repin, click, hide, ...) -> utility score
```

Key tricks reported in the paper [20]:

- **Early fusion** of the candidate with every sequence item (concatenation), which made the
  Transformer candidate-aware.
- **Action type embeddings** were important; **positional encodings added little**.
- **Random time-window masking** during training: hide actions from a random recent window so
  the model does not over-rely on the last few actions and recommendations stay diverse.
- Serving on GPUs, since the per-candidate Transformer is expensive.

**Results and challenges.** A third-party summary of the paper reports +11% Homefeed repin
volume and -10% hides in online A/B tests (approximate). TransAct was deployed in Homefeed,
Related Pins, Notifications and Search [20]. The paper discusses three production problems:
high serving complexity, reduced diversity, and **engagement decay** (gains shrink if the
model is not retrained often), which led to frequent retraining.

### 4.4 TransAct V2 (2025): lifelong sequences in ranking

**Problem.** TransAct used only recent actions (about 100). TransAct V2 extends Homefeed
ranking to lifelong histories of about 16,000 actions, roughly 160x more [21][22][31].

**Architecture.**

```
  lifelong seq (~16k, int8 PinSage, offline store)  real-time seq   impression seq
          \                                           |             /
           +------- nearest-neighbour selection by candidate -------+
                   (dot product in PinSage space; always keep
                    the most recent real-time actions)
                              |
                      short selected sequence
                              |
                Transformer (fused single-kernel)
                              |
              features -> ranker heads  +  Next Action Loss (aux)
```

- Each token holds timestamp, a multi-hot action type, surface, and a **32-dimensional PinSage
  embedding**; lifelong actions are stored losslessly with **int8 quantization** [21][22].
- **Candidate-anchored nearest-neighbour (NN) selection** picks a short subsequence from the
  lifelong, real-time and impression sequences. It is a SIM-style GSU, with the most recent
  actions always kept [21].
- **Next Action Loss**: an auxiliary loss that asks the model to predict what the user does
  next, not only engagement probability [21][31].
- Serving: a fused Triton kernel combines dequantization to float16, normalisation and NN
  search, cutting model latency by 20% versus native PyTorch. A single-kernel Transformer
  (SKUT) is reported 6.6x faster than PyTorch in the forward pass. A broadcast-free NN search
  reportedly cut PCIe transfers for sequences by 8x. End-to-end inference latency improved by
  103x (p50), 338x (p90) and 250x (p99) versus the unoptimised baseline [21].

**Results.** Pinterest reports +6.35% repins, +1.41% time spent, a 12.8% drop in hides
[31], and a 13.31% increase in top-3 repin hit rate [22].

### 4.5 PinFM (RecSys 2025): a foundation model for user activity

**Problem.** Many Pinterest surfaces each trained their own sequence modules. Pinterest wanted
one large pretrained model of user activity to share across surfaces [23].

**Design.**

- **Pretraining.** A Transformer with more than 20 billion parameters (most in ID embedding
  tables) is pretrained on about two years of user actions, with contrastive next-token,
  multi-token and future-token losses (in the PinnerFormer tradition) [23].
- **Fine-tuning.** The pretrained Transformer and embedding tables are plugged into ranking
  models for each application and fine-tuned with the ranking losses; the pretraining losses
  can optionally be kept [23]. An ablation found that switching to bidirectional attention in
  fine-tuning cost 1.4% in evaluation metrics, so causal attention is kept [23].
- **DCAT (Deduplicated Cross-Attention Transformer).** At serving, there are about 1,000
  candidates per unique user sequence (about 10 in training). DCAT splits the model into a
  **context part** that runs the Transformer once on the user history and caches its KV, and a
  **crossing part** that scores each candidate against the cached history [23].
  It raised throughput by 600% on internal data. Embeddings are int4-quantized [23].
- **Cold start.** A conference talk describes ID noise, age-dependent dropout and a class
  token for fresh pins (secondary).

**Results.** +20% engagement on new items, deployed to Home Feed and Related Items for more
than half a billion users at neutral cost and marginal latency increase [23].

### 4.6 PinRec (2025): generative retrieval from the same sequences

PinRec uses a user-sequence Transformer as a generative retriever, with outcome
conditioning (for example generating for saves versus clicks) and multi-token generation
[32]. It shows the same history encoder moving from ranking into retrieval.

## 5. Kuaishou: sequence models for retrieval

### KuaiFormer (2024)

**Problem.** Single-vector two-tower retrieval squeezes all user interests into one vector
and handles long sequences poorly [16].

**Design** [16]:

- **Multiple learnable query tokens** (like BERT's [CLS]) appended to the sequence, each
  producing one interest vector for multi-interest ANN retrieval.
- **Adaptive sequence compression.** Older behaviours are merged in groups of 64, middle
  behaviours in groups of 16, each group aggregated by a single-layer bidirectional
  Transformer; recent items are kept as-is. For example, 256 behaviours are fed as 55 tokens.
- A causal Transformer over the compressed sequence, trained with a smoothed in-batch softmax
  with logQ correction.

**Scale.** In production since May 2024 in Kuaishou's short-video recommendation, serving more
than 400 million daily active users [16]. Kuaishou has since published generative
recommenders for long sequences, such as GEMs (2026) with a multi-stream decoder [34].

## 6. Generative recommenders and foundation user models

### 6.1 Meta HSTU and Generative Recommenders (ICML 2024)

**Problem.** Classic deep learning recommenders (DLRMs) did not keep improving with more
compute. Meta asked whether a sequence model of user actions, scaled like an LLM, would [24].

**Reformulation.** Meta rewrote ranking and retrieval as **sequential transduction**. The
user's history becomes one chronological stream mixing content and actions. Retrieval predicts
the next content item; ranking predicts the action on an inserted candidate. Most hand-built
features are replaced by the sequence itself [24].

**HSTU (Hierarchical Sequential Transduction Unit)** [24][30]:

- **Pointwise aggregated attention.** Scores go through SiLU instead of row-wise softmax and are
  scaled by sequence length. This keeps the intensity of user preferences (how many related
  actions there are) and suits non-stationary, streaming vocabularies.
- **Relative attention bias** from both position and time gaps.
- A fused, simplified layer (fewer linear layers than a standard Transformer block) that
  uses less memory, enabling deeper stacks.
- Efficiency: HSTU is 5.3x to 15.2x faster than FlashAttention2 Transformers at length 8,192
  [24]. The paper also describes stochastic length (sampling sub-sequences of long histories) to cut cost.
  At inference, **M-FALCON** batches many candidates into one pass with attention masks so
  the user history cost is shared.

**Results.** Up to 65.8% better NDCG than baselines on public and synthetic data; a
**1.5-trillion-parameter** generative recommender improved online A/B metrics by 12.4% across
multiple surfaces of a platform with billions of users. Model quality followed a power law of
training compute across three orders of magnitude, reaching GPT-3/LLaMa-2 scale [24].
Attention dominates inference time (over 90% in one characterisation study), which drove
fused kernels such as NVIDIA's cuDNN HSTU attention [30].

### 6.2 Netflix foundation model for personalization (2025)

**Problem.** Netflix ran many specialised models ("Continue Watching", "Today's Top Picks" and
others), each trained separately on similar data. Maintenance was expensive and improvements
did not transfer between models [25][35].

**Design** (from the Netflix tech blog [25] and secondary summaries [35]):

- An LLM-style, autoregressive Transformer trained on **hundreds of billions of interactions
  from more than 300 million users**, roughly the token scale of an LLM.
- **Interaction tokenization.** Raw events are merged into meaningful tokens (for example
  consecutive related events grouped), balancing sequence length and detail, similar to BPE
  for text.
- Very long histories are handled with **sparse attention and sliding-window sampling** in
  training (Netflix published a separate sliding-window training paper [27]), plus KV caching
  at inference.
- **Objectives.** Next-item prediction, with auxiliary targets (for example genre or original
  language) and **multi-token prediction** of several future items, which also reduces
  myopia.
- **Cold start.** Item embeddings combine learnable ID embeddings with metadata embeddings, so
  new titles get useful vectors before anyone watches them. Training supports incremental
  updates with unseen entities.
- Reported scaling from millions to billions of parameters, with gains following scaling
  laws (secondary).

**Integration** [26][35]. Netflix describes three ways downstream teams consume the model:

| Pattern | How it works | Trade-off |
|---|---|---|
| Embeddings | Precomputed user/item embeddings published to an Embedding Store | Cheap and easy; can be stale |
| Subgraph | FM layers embedded in the downstream model and trained with it | Fresh, adaptable; downstream must build the FM's inputs |
| Fine-tuning | FM fine-tuned directly for a task | Best fit; most compute and ops cost |

### 6.3 LinkedIn and others

LinkedIn has publicly described a large foundation ranking model (360Brew, reported at about
150B parameters) that reads member history as part of its input, but public details on its
user-history component come mainly from secondary sources and are not covered here.

## 7. Comparison table

| System (org, year) | Stage | History length | Selection / compression | Model | Public results |
|---|---|---|---|---|---|
| GRU4Rec (academic, 2015) [1] | Next-item | Session | None | GRU, in-batch negatives | Beat item-kNN in paper; tuned kNN competitive [2] |
| SASRec (UCSD, 2018) [3] | Next-item | ~10^2 | None | Causal Transformer | ~10x faster than CNN/RNN |
| BERT4Rec (Alibaba, 2019) [4] | Next-item | ~10^2 | None | Bidirectional, Cloze | Strong only with long training [5] |
| gSASRec (Glasgow, 2023) [6] | Next-item | ~10^2 | None | SASRec + gBCE | +9.47% NDCG vs BERT4Rec (ML-1M) |
| DIEN (Alibaba, 2019) [8] | Ranking | ~50 online | None | GRU + AUGRU | +20.7% CTR |
| BST (Alibaba, 2019) [7] | Ranking | Short (recent clicks) | None | 1 Transformer block | +7.57% CTR, AUC 0.7894 |
| MIMN + UIC (Alibaba, 2019) [9] | Ranking | ~10^3 | Incremental memory | Memory network | +7.5% CTR, +6% RPM |
| SIM (Alibaba, 2020) [10] | Ranking | Up to 54,000 | Category hard search (UBT) | GSU + ESU attention | +7.1% CTR, +4.4% RPM |
| ETA (Alibaba, 2021) [11] | Ranking | ~10^4 | SimHash + Hamming top-k | End-to-end attention | +3.1% GMV |
| SDIM (Meituan, 2022) [13] | Ranking | ~10^4 | Multi-hash sampling | No attention | +2.98% CTR |
| TWIN (Kuaishou, 2023) [14] | Ranking | 10^4 - 10^5 | CP-GSU, top-100 | Shared GSU/ESU attention | 99.3% less compute bottleneck |
| TWIN V2 (Kuaishou, 2024) [15] | Ranking | 10^5+ | Hierarchical clustering | Cluster-aware attention | +0.67% to +0.80% watch time |
| LONGER (ByteDance, 2025) [17] | Ranking | Up to 10,000 | Token merge, global tokens | End-to-end Transformer | Dozens of scenarios |
| PinnerSage (Pinterest, 2020) [18] | Retrieval | 90 days | Ward clustering, medoids | Non-parametric | Beat single embedding |
| PinnerFormer (Pinterest, 2022) [19] | Retrieval/feature | ~256 (approx.) | None, daily batch | Causal Transformer, dense loss | Batch drop 8.3% vs 13.9% |
| TransAct (Pinterest, 2023) [20] | Ranking | ~100 real-time | Random time-window mask | Small Transformer, early fusion | ~+11% repins (reported) |
| TransAct V2 (Pinterest, 2025) [21] | Ranking | ~16,000 | Candidate NN search | Transformer + next-action loss | +6.35% repins, -12.8% hides |
| KuaiFormer (Kuaishou, 2024) [16] | Retrieval | 256 -> 55 tokens | Group merging (64/16) | Causal + multi-query | 400M+ DAU |
| PinFM (Pinterest, 2025) [23] | Ranking (FM) | Lifelong, 2 yrs | DCAT KV reuse | 20B+ param pretrained | +20% new-item engagement |
| HSTU GR (Meta, 2024) [24] | Retrieval + ranking | Up to 8,192+ | Stochastic length, M-FALCON | HSTU, 1.5T params | +12.4% online |
| Netflix FM (2025) [25] | Shared FM | Lifetime | Tokenization, sliding window | Autoregressive Transformer | Replaces many models |

## 8. Recurring design patterns

### 8.1 Split the user part from the candidate part

The single most important serving trick is to compute the **user-only part once per request**
and reuse it across all candidates:

```
  request -> [ user history encoder ]  ---- once per request (or cached / batch)
                    |   KV cache / embedding
          +---------+----------+----------+
          v         v          v          v
       cand_1    cand_2  ...  cand_1000      cheap cross-attention or dot product
```

Examples: MIMN/UIC (precompute memory), SDIM's BSE service, TWIN's cached item keys, LONGER's
KV-cache serving, PinFM's DCAT, HSTU's M-FALCON, PinnerFormer's daily embeddings.

### 8.2 Retrieve, then attend

For histories past about 10^3, every production system first selects a short relevant
subsequence and then runs exact attention: category search (SIM), LSH (ETA, SDIM), shared
attention scores (TWIN), clustering (TWIN V2, PinnerSage), and embedding NN search (TransAct
V2). TWIN's lesson is that the **selection metric should match the final model's metric**.

### 8.3 Choose the objective for how the representation is used

- Batch embeddings: forecast a future window (PinnerFormer's dense all-action loss).
- Real-time ranking: candidate-aware target attention with a multi-task ranking head
  (TransAct), plus an auxiliary next-action loss (TransAct V2).
- Foundation models: next-token, multi-token and future-token losses (PinFM, Netflix).
- With sampled negatives, correct for sampling (logQ, gBCE) or the model becomes overconfident
  [6].

### 8.4 Real-time versus batch

Pinterest's hybrid (daily PinnerFormer + real-time TransAct) is the clearest public example of
splitting long-term and short-term interest across infrastructure tiers. Real-time models
react within seconds but cost more and can reduce diversity (hence random time-window masks);
batch models are cheap but stale (hence objectives robust to staleness).

### 8.5 Scale like an LLM

HSTU, PinFM, Netflix's model and LONGER all report that larger sequence models trained on more
interactions keep improving, which DLRM-style models did not [24][25]. The cost is dominated by
attention on long sequences, so these systems invest in custom kernels (Triton, cuDNN HSTU
attention), quantization (int8/int4 embeddings), and KV reuse.

## 9. Key takeaways for system designers

1. **History is the main feature.** Moving from ~50 to 10^4 or more behaviours has repeatedly
   given multi-percent online gains (MIMN, SIM, TWIN, TransAct V2). Plan storage and serving
   for lifelong histories from the start.
2. **Never pay O(L) per candidate if you can pay it once per request.** Split user-only
   computation from candidate crossing and cache it (KV cache, precomputed keys, embeddings).
3. **Use two stages for very long sequences, and keep them consistent.** A cheap GSU picks
   about 100 behaviours; an exact ESU attends. Share parameters or metrics between them
   (TWIN), or make retrieval differentiable (ETA).
4. **Compress the old, keep the recent exact.** Clustering (TWIN V2), group merging
   (KuaiFormer, LONGER) and always-keep-recent rules (TransAct V2) are cheap and effective.
5. **Pick the training target for the consumer.** Daily embeddings should forecast a window of
   future behaviour; rankers should be candidate-aware and multi-task.
6. **Mind negative sampling.** In-batch and sampled negatives need logQ correction or a
   calibrated loss such as gBCE; otherwise models are overconfident and benchmarks mislead.
7. **Do not trust academic leaderboards blindly.** BERT4Rec versus SASRec showed that training
   time and loss choices can flip conclusions; also compare against strong kNN baselines.
8. **Real-time features cause decay and diversity problems.** Retrain often, use random time
   window masks or similar regularisation, and monitor hides and diversity, not just clicks.
9. **Positional encodings matter less than action types and timestamps.** BST uses time gaps;
   TransAct found action-type embeddings and early fusion mattered more than position.
10. **Foundation user models are now standard at the largest platforms.** Pretrain once on
    billions of interactions, then serve through embeddings, subgraphs or fine-tuning (Netflix)
    or as a module inside rankers (PinFM). Budget for custom kernels and quantization.
11. **Cold start needs explicit handling in ID-heavy sequence models.** Combine ID embeddings
    with content/metadata embeddings (Netflix) and regularise new IDs (PinFM).

## References

1. Hidasi, B., Karatzoglou, A., Baltrunas, L., Tikk, D. "Session-based Recommendations with
   Recurrent Neural Networks." arXiv / ICLR 2016, 2015. https://arxiv.org/abs/1511.06939
2. Jannach, D., Ludewig, M. "When Recurrent Neural Networks meet the Neighborhood for
   Session-Based Recommendation." RecSys 2017.
   https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys_2017.pdf
3. Kang, W.-C., McAuley, J. "Self-Attentive Sequential Recommendation." UC San Diego, ICDM
   2018. https://arxiv.org/abs/1808.09781
4. Sun, F. et al. "BERT4Rec: Sequential Recommendation with Bidirectional Encoder
   Representations from Transformer." Alibaba, CIKM 2019. https://arxiv.org/abs/1904.06690
5. Petrov, A., Macdonald, C. "A Systematic Review and Replicability Study of BERT4Rec for
   Sequential Recommendation." University of Glasgow, RecSys 2022.
   https://arxiv.org/abs/2207.07483
6. Petrov, A., Macdonald, C. "gSASRec: Reducing Overconfidence in Sequential Recommendation
   Trained with Negative Sampling." University of Glasgow, RecSys 2023.
   https://arxiv.org/abs/2308.07192
7. Chen, Q., Zhao, H., Li, W., Huang, P., Ou, W. "Behavior Sequence Transformer for
   E-commerce Recommendation in Alibaba." Alibaba, DLP-KDD 2019.
   https://arxiv.org/abs/1905.06874
8. Zhou, G. et al. "Deep Interest Evolution Network for Click-Through Rate Prediction."
   Alibaba, AAAI 2019. https://arxiv.org/abs/1809.03672
9. Pi, Q., Bian, W., Zhou, G. et al. "Practice on Long Sequential User Behavior Modeling for
   Click-Through Rate Prediction." Alibaba, KDD 2019. https://arxiv.org/abs/1905.09248
10. Pi, Q., Zhu, X., Zhou, G. et al. "Search-based User Interest Modeling with Lifelong
    Sequential Behavior Data for Click-Through Rate Prediction." Alibaba, CIKM 2020.
    https://arxiv.org/abs/2006.05639
11. Chen, Q., Pei, C. et al. "End-to-End User Behavior Retrieval in Click-Through Rate
    Prediction Model." Alibaba, 2021. https://arxiv.org/abs/2108.04468
12. "Efficient Long Sequential User Data Modeling for Click-Through Rate Prediction"
    (ETA-Net). Alibaba, 2022. https://arxiv.org/abs/2209.12212
13. Cao, Y. et al. "Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR
    Prediction." Meituan, CIKM 2022. https://arxiv.org/abs/2205.10249
14. Chang, J. et al. "TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in
    CTR Prediction at Kuaishou." Kuaishou, KDD 2023. https://arxiv.org/abs/2302.02352
15. Si, Z. et al. "TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling for Enhanced
    CTR Prediction at Kuaishou." Kuaishou, CIKM 2024. https://arxiv.org/abs/2407.16357
16. "KuaiFormer: Transformer-Based Retrieval at Kuaishou." Kuaishou, 2024.
    https://arxiv.org/abs/2411.10057
17. "LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders." ByteDance,
    RecSys 2025. https://arxiv.org/abs/2505.04421
18. Pal, A., Eksombatchai, C. et al. "PinnerSage: Multi-Modal User Embedding Framework for
    Recommendations at Pinterest." Pinterest, KDD 2020. https://arxiv.org/abs/2007.03634
19. Pancha, N., Zhai, A., Leskovec, J., Rosenberg, C. "PinnerFormer: Sequence Modeling for
    User Representation at Pinterest." Pinterest, KDD 2022. https://arxiv.org/abs/2205.04507
20. Xia, X. et al. "TransAct: Transformer-based Realtime User Action Model for Recommendation
    at Pinterest." Pinterest, KDD 2023. https://arxiv.org/abs/2306.00248
21. "TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation."
    Pinterest, 2025. https://arxiv.org/abs/2506.02267
22. Pinterest Engineering. "Next-Level Personalization: How 16k+ Lifelong User Actions
    Supercharge Pinterest's Recommendations." Medium, 2025.
    https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d3
23. "PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery
    Platform." Pinterest, RecSys 2025. https://arxiv.org/abs/2507.12704
24. Zhai, J. et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential
    Transducers for Generative Recommendations." Meta, ICML 2024.
    https://arxiv.org/abs/2402.17152
25. Netflix Technology Blog. "Foundation Model for Personalized Recommendation." Netflix,
    2025. https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39
26. Netflix Technology Blog. "Integrating Netflix's Foundation Model into Personalization
    Applications." Netflix, 2025.
    https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb
27. "Sliding Window Training: Utilizing Historical Recommender Systems Data for Foundation
    Models." Netflix, RecSys 2024. https://arxiv.org/abs/2409.14517
28. Machine Learning Frontiers (S. Bhattacharya). "Towards Life-Long User History Modeling in
    Recommender Systems." Substack, 2024.
    https://mlfrontiers.substack.com/p/towards-life-long-user-history-modeling
29. Machine Learning Frontiers. "Target Attention Is All You Need: Modeling Extremely Long
    User Action Sequences in Recommender Systems." Substack, 2024.
    https://mlfrontiers.substack.com/p/target-attention-is-all-you-need
30. NVIDIA. "HSTU Attention." cuDNN frontend documentation, 2025.
    https://docs.nvidia.com/cudnn/fe-oss-apis/hstu/hstu_attention
31. Social Media Today. "Pinterest Updates Ranking Model With More Data." 2025.
    https://www.socialmediatoday.com/news/pinterest-updates-ranking-model-with-more-data/750122/
32. "PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale
    Recommendation Systems." Pinterest, 2025. https://arxiv.org/abs/2504.10507
33. Machine Learning at Scale. "[RecSys] Part 3: LONGER, scaling up long sequence modelling in
    industrial recommenders." Substack, 2025.
    https://machinelearningatscale.substack.com/p/recsys-part-3-longer-scaling-up-long
34. "GEMs: Long-sequence generative recommendation with a multi-stream decoder." Kuaishou,
    2026. https://arxiv.org/abs/2602.13631
35. ZenML LLMOps Database. "Netflix: Foundation Model for Large-Scale Personalized
    Recommendation." 2025.
    https://www.zenml.io/llmops-database/foundation-model-for-large-scale-personalized-recommendation
