# 22. Pinterest Case Study: From Pixie Random Walks to PinFM Foundation Models

Pinterest has published one of the most complete public records of how a large recommender evolved. The same company documented a real-time graph-walk retriever (Pixie), the first web-scale graph convolutional network (PinSage), several generations of user representations (PinnerSage, PinnerFormer), real-time and lifelong sequence rankers (TransAct, TransAct V2), a pretrained user-activity foundation model (PinFM), generative retrieval (PinRec, UniPinRec), its own ANN platform (Manas), GPU serving for ranking, and an ads stack built on DCN-style crossing networks. Most of these papers say clearly which production surface they serve and what they replaced. That makes Pinterest a good longitudinal case study: you can watch each layer of the funnel get rebuilt while the others stay in place.

This chapter follows that history layer by layer. It covers the home feed funnel, retrieval (graph, embedding, conditional and generative), item and user representations, ranking (Pinnability, TransAct, TransAct V2, PinFM), utility blending, GPU serving, ads ranking, and Related Pins.

A note on sourcing. The live web-search budget for this run was used up, and arXiv and Medium could not be fetched directly from the writing environment. The facts here come from primary-source abstracts and excerpts, and Pinterest engineering-blog excerpts, that were retrieved earlier in this research effort and cross-checked across several summaries. Figures marked "approximate" or "secondary" come from third-party summaries and should be checked against the cited paper before you quote them. OmniSage, named in this chapter's scope, could not be verified against a primary source in this pass, so it gets only a short pointer (Section 4.4).

---

## 1. The home feed funnel at a glance

Pinterest's WWW 2022 sponsor slides draw the home feed as a classic multi-stage funnel [25]:

- **Candidate generators.** Token-based indexes, HNSW embedding indexes, graph-based random walks (Pixie) and explore/exploit sources.
- **Light-Weight Scorer (LWS).** A pre-ranking stage. Later blog posts describe it as a unified two-tower lightweight scorer [29][30].
- **Ranking Service.** A Utility Prediction Scorer (the multi-task ranker, Pinnability) followed by a Multi-Objective Blender.

An older engineering post describes the same split in terms of services. **Polaris** retrieves, filters and applies lightweight scoring. It returns a list of Pins to the **Feed Generator**, which sends that list with real-time user signals to **Scorpion** "for a second pass of full scoring" [26]. In the early days, each content type had its own specialised ranking model, and the outputs were "blended with a ratio-based round robin" [27].

```
  user request
      |
      v
 +---------------------- CANDIDATE GENERATION (recall) ----------------------+
 | Pixie walks | learned two-tower EBR (Manas) | conditional / multi-emb EBR  |
 | token indexes | PinRec generative retrieval | explore/exploit sources      |
 +-------------------------------------+------------------------------------+
                                       | merge + dedup (thousands)
                                       v
                    Light-Weight Scorer (two-tower pre-ranking)
                                       | hundreds
                                       v
       Pinnability ranker (wide&deep + DCN-style crossing + MTL heads)
         <- TransAct / TransAct V2 sequence modules, PinFM, PinnerFormer
                                       |
                                       v
         Multi-Objective Blender (utility weights; PRL-PUTS per request)
                                       |
                                       v
                          home feed grid  (after [25][26][33])
```

Pinterest's Related Pins paper describes how the role of the early stages changed. Once machine-learned ranking arrived, candidate generation was asked to optimise recall rather than precision [5]. That shift is now the norm across the industry, and the rest of this chapter follows it.

---

## 2. Pixie: real-time random walks on a pruned bipartite graph (2017-2018)

### Problem

Pinterest needed to recommend from more than 3 billion Pins to more than 200 million users in real time. The recommendations had to react to what the user had just done. The previous Hadoop-based system precomputed results offline and could not react within a session [1].

### Architecture

Pixie loads a pruned bipartite **Pin-Board graph** into the RAM of a single machine. A save of a Pin into a board is an edge. The pruned graph has about 3 billion nodes and 17 billion edges and fits into about 120 GB of memory. The unpruned graph had about 7 billion nodes and more than 100 billion edges [1]. The deployment used AWS r3.8xlarge machines with 244 GB of RAM. Each server holds a full replica, so a walk never crosses machines and the service scales by adding replicas [1].

```
 user actions (saves, clicks)        offline graph build (daily)
         |                                   |
         v                                   v
 query pins Q = {q1..qk}, weights      pruned Pin<->Board graph (~120 GB)
         |                                   |
 +------ Pixie server (full replica, everything in RAM) ---------------+
 | per q: step budget ~ weight(q) scaled by degree(q)                  |
 |   walk pin -> board -> pin ... biased by user features,             |
 |   restart with prob alpha; count visits V_q[p]                      |
 |   early-stop when n_p pins have >= n_v visits                       |
 | combine: V[p] = (sum_q sqrt(V_q[p]))^2   (multi-hit boost)          |
 +-------------------------------+-------------------------------------+
                                 v
                    top-K pins -> ranking -> surface
```

### Algorithm and tricks

- **Random walk with restarts.** Many short walks run from the query Pins, and the visit counts approximate personalised PageRank. The walk runs in constant time, independent of graph size [1].
- **Biased walks.** Edge selection is biased toward boards and Pins that match user features such as language and topic, which personalises the walk without per-user graphs [1].
- **Weighted multi-pin queries.** Each query Pin gets a step budget proportional to its weight and adjusted for its degree. The per-pin counts are combined with a sum of square roots, then squared, which rewards candidates reached from several query Pins [1].
- **Boosting small boards.** In a plain walk, low-degree nodes would contribute too little signal, so Pixie boosts the impact of smaller boards [1].
- **Early stopping.** A walk stops once enough Pins have reached a minimum visit count, which cuts latency with little loss in quality [1].
- **Graph pruning as modelling.** Topically diverse boards (high entropy) are removed and very popular Pins keep edges only to their most relevant boards. Pruning improved quality as well as memory [1].

### Scale and impact

- 99th-percentile latency under 60 ms [1].
- About 1,200 requests per second per server, and about 100,000 requests per second across the cluster [1].
- Up to 50% higher user engagement than the previous Hadoop-based system [1].
- A summary of the paper reports that Pixie-backed systems contributed more than 80% of all user engagement on Pinterest at the time [2].

### Lessons

If the working graph fits in RAM, replicate it rather than partition it, because walks are pointer chasing and network hops would ruin them. Pruning hub nodes and noisy boards is a modelling decision. A walk produces a candidate distribution, and a learned ranker still has to sit on top.

---

## 3. PinSage: graph convolution at 3 billion nodes (2018)

### Problem

Pixie needs graph connectivity at query time and cannot use content features, so new Pins with few saves get weak recommendations. Pinterest wanted one embedding per Pin that combined visual and text features with graph structure, computed for billions of items and served by nearest-neighbour search [3][4].

### Architecture and scale

PinSage is a GraphSAGE-style inductive GCN. It was trained on 7.5 billion examples over a graph of 3 billion nodes (Pins and boards) and 18 billion edges [3]. The engineering blog describes the graph as about 10,000x larger than typical GCN applications of the time [4].

```
  TRAINING (producer-consumer)
   CPU producers: sample (query, positive) pairs from co-engagement
                  -> random-walk neighbourhoods (top-T by visit count)
                  -> gather visual + text features            --+
   GPU consumer:  2 conv layers, importance pooling,             |
                  max-margin loss, random + curriculum negatives <+
  INFERENCE (MapReduce, no redundant computation)
   layer-1 embeddings for all nodes -> join -> layer-2 -> Pin embeddings
                  -> ANN index -> Related Pins / Homefeed candidates
```

### Key tricks

1. **Random-walk neighbourhoods.** Short random walks from each node pick the top-T most-visited nodes (L1-normalised counts) as the neighbourhood. This keeps the memory footprint per node fixed even though Pinterest degrees can be huge [3].
2. **Importance pooling.** The visit counts become aggregation weights. The paper credits this with a 46% gain in offline metrics [3].
3. **Curriculum training with hard negatives.** Later epochs add harder negatives, Pins ranked moderately high but not top by personalised PageRank. The paper reports a 12% gain from curriculum training [3]. Another chapter of this Atlas notes that the hard negatives roughly doubled the epochs to convergence.
4. **Producer-consumer minibatching** keeps the GPUs fed while CPUs sample and fetch features [3].
5. **MapReduce inference** computes each layer once for all nodes instead of recomputing overlapping neighbourhoods [3].

A secondary summary reports 10-30% repin-rate improvements over earlier embedding systems in Homefeed (approximate) [4].

### Why PinSage mattered beyond retrieval

PinSage became Pinterest's shared coordinate system for Pins. PinnerSage clusters users in PinSage space [6]. PinnerFormer uses 256-dimensional PinSage embeddings as frozen per-action features [7]. TransAct V2 stores a 32-dimensional PinSage embedding per lifelong action token and runs its nearest-neighbour selection in that space [10]. Manas benchmarks use a 100-million-vector GraphSage dataset [23]. Graph embeddings that many teams reuse are a platform, not a feature.

---

## 4. Multi-graph and multi-entity successors

### 4.1 MultiBiSage

MultiBiSage extends the PinSage idea from one Pin-Board graph to **multiple bipartite graphs** at Pinterest [41]. The detailed design was not retrieved in this research pass. The title alone shows the direction: one item representation fed by several interaction graphs instead of one.

### 4.2 OmniSearchSage

OmniSearchSage learns multi-task, multi-entity embeddings for Pinterest Search [42]. Together with the PinSage line, it shows a trend toward one embedding space shared by Pins, queries and products, trained on several tasks at once.

### 4.3 PinnerSage: multi-modal user embeddings (KDD 2020)

**Problem.** One user vector averages over unrelated interests such as recipes, tattoos and home decor [6].

**Design.** Take the user's action Pins from the last 90 days. Cluster their PinSage embeddings with Ward hierarchical clustering. Represent each cluster by its **medoid**, a real Pin from the user's history, which keeps the cluster topically coherent and robust to outliers. Score clusters by importance and sample a few medoids per request as ANN queries [6]. Offline experiments and A/B tests showed it beat single-embedding methods [6]. Because the PinSage item space stays fixed, PinnerSage never retrains the item model.

PinnerSage also became a ranking feature. The PinnerFormer paper says the previous production ranker used a weighted average of a user's top-k PinnerSage embeddings, and that replacing it with one PinnerFormer embedding significantly improved Homefeed repins [7].

### 4.4 OmniSage (pointer only)

OmniSage, a later Pinterest representation-learning effort over a multi-entity heterogeneous graph, is listed in this chapter's scope. Its primary source could not be retrieved or verified in this pass, so no architecture or numbers are given here. Readers should treat it as the natural successor of PinSage, MultiBiSage and OmniSearchSage, and check the original paper for details.

---

## 5. PinnerFormer: a daily batch user embedding trained for the long term (KDD 2022)

### Problem

Pinterest wanted a single user embedding, computed in a **daily batch job**, that would work nearly as well as a real-time embedding and predict long-term engagement, not just the next action [7].

### Architecture

```
 user's recent actions (pin, action type, surface, time features)
     |  each pin: 256-d PinSage embedding (frozen) + learned feature embs
     v
 causal Transformer encoder
     v
 L2-normalised user embedding, recomputed daily for every user
     |
     +--> ANN retrieval against pin embeddings
     +--> feature in Homefeed ranking and in ads ranking models
```

### The dense all-action loss

Instead of predicting only the next action (SASRec style), the embedding is trained to predict **all positive engagements (repins, close-ups, clicks) in a future window of up to 28 days** [7]. Training uses sampled softmax with in-batch and random negatives. A secondary summary reports logQ correction as well [7]. The paper's evaluation uses a 14-day window.

The objective makes the embedding robust to staleness. Moving from real-time to daily batch inference cut Recall@10 by **13.9%** with the next-action objective, but only by **8.3%** with the dense all-action objective [7]. A/B tests showed better engagement and retention than the previous user representation, and the model has been in production since fall 2021 [7]. A secondary source says the team did not scale sequences beyond about 256 actions (approximate).

### Ads usage

The paper reports that each primary surface (Homefeed, Related Pins, Search) has a separate ads ordering model, and that adding PinnerFormer as a feature gave significant gains in CTR and long CTR on each surface [7].

### Lesson

A user embedding that will be computed offline and reused should forecast a window of future behaviour. The objective, not only the refresh rate, decides how much freshness you need.

---

## 6. Learned retrieval and the Manas ANN platform (2022-2026)

### 6.1 Two-tower learned retrieval

**Problem.** The older candidate generators were heuristic or graph-based. A learned retriever optimises engagement directly [16].

**Architecture.** A standard two-tower model [16][17]:

- The **Pin tower** encodes visual, text and statistical features. It runs offline, and its embeddings are pushed into Manas.
- The **user tower** encodes engagement sequences and context. It is computed once per Homefeed request.
- Training is contrastive with in-batch negatives (a popularity correction is reported by secondary summaries [44]).

**The version-sync problem.** Each two-tower model is split into **two artifacts deployed to separate services**. When a model is retrained, the serving versions of both services must be kept in sync [16]. Otherwise user vectors from version k get scored against Pin vectors from version k-1, and retrieval silently degrades.

**Results.** A secondary summary says the learned retriever became one of the top candidate generators, with the **highest user coverage** and a top-three save rate. It was deployed for Homefeed and notifications [44].

### 6.2 Advancements: crossing inside towers, ID embeddings, multi-embedding

The 2025 Homefeed EBR post describes several upgrades [17]:

- **Feature crossing within each tower** with MaskNet and DHEN blocks, to win back some of the expressiveness lost to the dot product. A secondary summary reports +0.1-0.2% engagement and more than 1% more Homefeed saves from DHEN (approximate) [44].
- **Pre-trained ID embeddings.** User and Pin ID embeddings are pre-trained contrastively on a cross-surface dataset to reduce overfitting [17].
- **Multi-embedding retrieval.** Several user embeddings per request. Implicit interests come from a differentiable clustering module, and explicit interests from conditional retrieval [18]. Results from each embedding are merged by **round-robin with de-duplication**. Post-filtered conditional retrieval was the best variant, and p90 retrieval latency rose from **150 ms to 205 ms**, which was still within budget [18].
- Reported gains across sessions, saves and clicks of roughly 0.1% to 1.2% (secondary, approximate).

### 6.3 Conditional learned retrieval

The user tower takes an explicit condition, such as a topic, so one user can issue several conditioned queries grounded in the same behaviour representation [19][20]. A topic-based notification feed built this way gave **+0.26% weekly active users** and beat two-tower models with filters [19]. A 2026 post describes redesigning serving to support many conditions per Home Feed request [20].

### 6.4 Manas: from in-RAM HNSW to quantized SPANN

- **Realtime HNSW.** Pins are inserted incrementally into the HNSW graph for real-time EBR. Filtering was first done by over-fetching K neighbours and post-filtering [21].
- **Streaming filters.** The inverted index was reused as the filter store, with bitmap and array-backed posting lists replacing skiplists for faster lookups [22].
- **Quantization benchmark (2026)** on a 100-million-embedding GraphSage dataset [23][24]:

| Index | Memory | Recall@100 |
|---|---|---|
| HNSW (baseline) | 121 GB | 93.72% |
| HNSW + scalar quantization | 50 GB | 92.92% |
| IVF (baseline) | 97 GB | 91.69% |
| IVF + scalar quantization | 25 GB | 95.71% |

Product quantization compressed more but lost much more recall [23]. Manas also implemented **SPANN** for SSD-resident indexes, with an in-memory HNSW over the cluster centroids [23]. Pinterest Ads separately describes offline ANN for ad retrieval [40].

**Lesson.** Scalar quantization is often the best first compression step for recommendation embeddings. Production ANN platforms tend to move from all-in-RAM HNSW toward IVF or SPANN once cost and update rates matter.

### 6.5 Generative retrieval: PinRec and UniPinRec

**PinRec** (2025) uses a user-sequence Transformer as a generative retriever. It adds **outcome conditioning**, for example generating for saves versus clicks, and **multi-token generation** [14]. A teardown reports A/B lifts of up to +0.55% time spent and +4.01% Homefeed grid clicks (secondary) [43]. **UniPinRec** (2026) proposes unifying generative retrieval and ranking at Pinterest scale [15]. The same sequence encoder that powers ranking is moving into retrieval.

---

## 7. Pre-ranking: the Light-Weight Scorer

The LWS sits between thousands of merged candidates and the expensive ranker [25]. Pinterest has written about a **unified two-tower lightweight scorer** for Home Feed [29] and about **modernizing the pre-ranking stage** [30]. On the ads side, **MTMD** is a multi-task, multi-domain framework for unified ad lightweight ranking [39]. The design point is the same in each case. One shared, cheap, decomposable model replaces per-source heuristics, and it is trained to agree with the full ranker.

---

## 8. Ranking: Pinnability and its sequence modules

### 8.1 Pinnability

Pinnability was introduced on the engineering blog in 2015 to estimate how likely a Pinner is to interact with a Pin, which turned the home feed from chronological to relevance-ranked [28]. In the TransAct era it is a **point-wise multi-task** model. A wide-and-deep stack feeds feature-interaction layers and MLPs that output per-action probabilities for click, repin, hide and more, with hide as a negative signal [8]. The "Rethinking Personalized Ranking" paper calls the transformer-learned user sequence embedding one of the most important features in Homefeed ranking [31]. A 2024 workshop paper explains how Pinterest picked a new feature-interaction architecture for Homefeed under latency, GPU-memory and reproducibility constraints rather than purely on public benchmarks [32].

### 8.2 TransAct: real-time actions inside the ranker (KDD 2023)

**Problem.** PinnerFormer refreshes daily, so the ranker could not react to what the user did seconds ago [8].

**Hybrid design.** TransAct encodes the most recent actions in real time inside Pinnability, while PinnerFormer and other batch features provide long-term context. The authors argue this combines real-time responsiveness with the cost-efficiency of batch representations [8].

```
 real-time action stream                                candidate pin
         |                                                    |
 latest ~100 actions: PinSage emb + action type + timestamp   |
         |                                                    |
 early fusion: concat each action with candidate embedding <--+
         v
 small Transformer encoder
         v
 output compression (first K tokens + max pooling) -> flatten
         v
 concat with PinnerFormer + other features -> DCN v2 crossing -> MLP
         v
 multi-task heads (repin, click, hide, ...) -> utility
```

**Findings** [8]:

- **Positional encodings added little.** Better early fusion and **action-type embeddings** mattered more.
- **Random time-window masking** in training hides recent actions so the model does not over-rely on the last few clicks, which protects diversity (reported by summaries).
- Production problems the paper discusses: high serving complexity, reduced diversity and **engagement decay**, so the model needs frequent retraining.

**Serving.** The TransAct blog post says that because the Transformer output became much smaller, an explicit DCN v2 crossing layer on the full feature set became affordable. The new model raised CPU latency by more than 20x, so the team moved to **GPU serving** to stay latency-neutral at the same cost [9].

**Results.** A summary reports +11% Homefeed repin volume and -10% hide volume online (secondary, approximate). TransAct was deployed in Homefeed, Related Pins, Notifications and Search [8].

### 8.3 TransAct V2: lifelong sequences (2025)

**Problem.** TransAct saw about 100 recent actions. TransAct V2 extends Homefeed ranking to lifelong histories of up to about **16,000 actions**, roughly 160x more [10][11][12].

```
 lifelong seq (~16k, int8)    real-time seq     impression seq
            \                      |                 /
             +-- candidate-anchored NN search (dot product in PinSage
                 space); most recent real-time actions always kept --+
                                   |
                     short selected subsequence
                                   v
                 single-kernel Transformer (Triton)
                                   v
       ranker heads  +  auxiliary Next Action Loss
```

- **Tokens.** Each token holds a timestamp, a multi-hot action type, the surface and a **32-d PinSage embedding**. Lifelong actions are stored losslessly with **int8 quantization** [10][11].
- **Candidate-anchored NN selection** over lifelong, real-time and impression sequences. It is a SIM-style search stage that always keeps the most recent actions [10].
- **Next Action Loss.** An auxiliary task that asks what the user will do next, not only the engagement probability [10][12].
- **Serving kernels** [10]:
  - fusing float16 dequantization and normalisation with NN search cut model latency by 20% versus native PyTorch;
  - a single-kernel Transformer (SKUT) ran 6.6x faster than PyTorch on the forward pass;
  - a broadcast-free NN search cut PCIe transfers for sequences by 8x (secondary);
  - end-to-end inference latency improved **103x (p50), 338x (p90) and 250x (p99)** over the unoptimised baseline.
- **Results.** +6.35% repins, +1.41% time spent and -12.8% hides [12]. Offline, HIT@3/repin improved by 13.31% and HIT@3/hide fell by 11.25% [10][11].

### 8.4 PinFM: a foundation model for user activity (RecSys 2025)

**Problem.** Each surface trained its own sequence modules. Pinterest wanted one large pretrained user-activity model shared across applications [13].

**Pretraining.** A Transformer with **more than 20 billion parameters**, most of them in ID embedding tables, pretrained on about two years of user actions with contrastive next-token, multi-token and future-token losses [13]. A talk summary describes InfoNCE next-Pin prediction for pretraining and BCE fine-tuning on surface-specific tasks (secondary).

**Fine-tuning.** The pretrained Transformer and embedding tables are plugged into each application's ranking model and fine-tuned with ranking losses. The pretraining losses can be kept to speed adaptation. Switching to bidirectional attention during fine-tuning cost 1.4% in evaluation metrics, so causal attention is kept [13].

**DCAT (Deduplicated Cross-Attention Transformer).** There are about 1,000 candidates per unique user sequence at serving time, and about 10 in training. DCAT splits the model into a **context part** that runs the Transformer once over the user history and caches its keys and values, and a **crossing part** that scores each candidate against the cached history [13].

```
 user history --> context Transformer (once per request) --> cached K,V
                                                              |
 candidates c1..c1000 --> crossing attention over cached K,V -+--> features
                                                                 -> ranker
```

DCAT raised throughput by **600%** on internal data, and embeddings are **int4-quantized** [13]. Cold-start handling for fresh Pins reportedly uses ID noise, age-dependent dropout and a class token (secondary).

**Results.** +20% engagement with new items. Deployed in Home Feed and Related Items for more than half a billion users at neutral cost and with a marginal latency increase [13].

### 8.5 Comparison of Pinterest's user-sequence models

| Model (year) | Stage | History length | Refresh | Key mechanism | Reported impact |
|---|---|---|---|---|---|
| PinnerSage (2020) [6] | Retrieval | 90 days of actions | Daily + real time | Ward clustering, medoids | Beat single embedding |
| PinnerFormer (2022) [7] | Retrieval + feature | ~256 actions (approx.) | Daily batch | Causal Transformer, dense all-action loss | Batch drop 8.3% vs 13.9% |
| TransAct (2023) [8] | Ranking | ~100 real-time actions | Per request | Early fusion, time-window mask | ~+11% repins (secondary) |
| TransAct V2 (2025) [10] | Ranking | up to ~16k actions | Per request | Candidate NN selection, next-action loss | +6.35% repins, -12.8% hides |
| PinFM (2025) [13] | Ranking (pretrained) | ~2 years | Pretrain + fine-tune | 20B+ params, DCAT KV reuse | +20% new-item engagement |
| PinRec (2025) [14] | Retrieval (generative) | User sequence | Per request | Outcome-conditioned, multi-token | +4.01% grid clicks (secondary) |

---

## 9. Utility blending and long-term objectives

The ranker outputs a vector of action probabilities. A **utility-weight vector** turns it into one score, a weighted sum of the predicted objectives [33]. Historically the weights were set by offline analysis and online experiments, applied globally to all requests, and left in place for long periods [33].

- **Round robin to blender.** The 2016 home feed blended per-type rankers with fixed ratios [27]. By 2022 a Multi-Objective Blender sat inside the ranking service [25].
- **Lifecycle-dependent weights.** The Pinner Progression post makes utility weights depend on the user's lifecycle stage and shifts the feed toward retention rather than pure engagement (secondary summary) [35].
- **PRL-PUTS (2026).** Weight selection becomes a one-step, value-based RL problem: for each request an agent picks the weight vector with the highest expected request-level reward. An inference-time scalarization parameter yields a family of policies and an **empirical Pareto frontier**, used to govern the choice between, for example, repins and other actions. It runs alongside ranking with no added latency and no ranker change, is trained on unbiased exploration logs, and gave **+0.13% successful sessions** on Homefeed [33].
- **Downstream rewards (2026).** A model-agnostic method learns longer-horizon downstream rewards on top of immediate engagement. It launched on Homefeed and was extended to other surfaces [34].

---

## 10. GPU serving for ranking

Pinterest's 2022 post "GPU-accelerated ML Inference at Pinterest" describes moving ranking models from CPU to GPU serving [36]. In outline, per the summary in Chapter 10 of this Atlas (approximate):

- **Motivation.** CPU cost grew roughly linearly with model size, which blocked larger rankers such as transformer sequence modules and bigger embeddings.
- **Outcome.** GPU serving made models with on the order of 100x more compute affordable at comparable cost and latency.
- **Techniques.** Batching candidates per request, CUDA Graphs to cut kernel-launch overhead, merging many small embedding lookups into fewer larger operations, mixed precision, and moving feature preprocessing onto the GPU.

TransAct shows the economic logic in miniature. A 20x CPU latency increase became latency-neutral at the same cost on GPUs [9]. TransAct V2's Triton kernels and PinFM's DCAT and int4 embeddings are the next steps in the same direction: once ranking runs on GPUs, kernel fusion, KV reuse and quantization become first-order design tools [10][13].

```
 CPU era                  GPU era (2022+)            kernel era (2025+)
 tree/linear + MLP  -->   batched DNN, CUDA graphs -->  fused Triton kernels,
 per-candidate CPU        fused embeddings,            int8/int4 sequences,
 cost ~ model size        GPU preprocessing            KV dedup (DCAT)
```

---

## 11. Pinterest ads ranking

Ads run in the same surfaces as organic content (Homefeed, Related Pins, Search), each with its own ads ordering model [7].

- **User sequences in ads engagement models.** A Pinterest post describes adding a Transformer over user action sequences to ads engagement models, with a **DCNv2** production baseline [37]. The team started with a single-layer, single-head Transformer and used position embeddings based on **time deltas**. Adding layers helped, but adding heads did not. Earlier feature interaction between the sequence and user/Pin representations also helped. The post warns that the DCNv2 cross layer sometimes **overflowed in float16 training** [37].
- **PinnerFormer as an ads feature** gave significant CTR and long-CTR gains on each surface [7].
- **ML-DCN (2026).** Scaling DCNv2 and MaskNet, both deployed in Pinterest ads, gave offline AUC gains that flattened quickly at affordable compute. ML-DCN adds an **instance-conditioned mask to a low-rank crossing layer**, reaches higher AUC than DCNv2 and MaskNet at matched FLOPs, and has a better AUC-FLOPs curve as compute grows. Online it gave significant CTR and click-quality gains and was deployed at neutral serving cost [38].
- **MTMD** unifies ad lightweight ranking across tasks and domains [39].
- **Offline ANN for ad retrieval** is used where online ANN is unnecessary [40].

```
 ad candidates (offline/online ANN, targeting)
          v
 MTMD lightweight ranking (multi-task, multi-domain)
          v
 engagement model: user seq Transformer + PinnerFormer + features
                   -> DCNv2 / ML-DCN crossing -> pCTR, long-CTR heads
          v
 auction + blending with organic feed
```

The ads story mirrors organic ranking. Crossing networks scale poorly when made naively bigger, mixed-precision training exposes numeric problems, and sequence features carry most of the new signal.

---

## 12. Related Pins

Related Pins ("more like this" under a Pin closeup) is Pinterest's oldest recommendation surface. The 2017 paper "Related Pins at Pinterest: The Evolution of a Real-World Recommender System" traces it from heuristic board co-occurrence to a multi-source candidate set followed by learned ranking. It notes that candidate generation moved from optimising precision to optimising recall once machine-learned ranking was introduced [5]. Pixie walks [1], PinSage nearest neighbours [3], TransAct [8] and PinFM [13] were all deployed to Related Pins (Related Items) as well as Homefeed. This shows a reusable-component strategy: one retriever or sequence module serving many surfaces.

---

## 13. System-level comparison

| System | Year | Layer | Data | Scale (public) | Serving |
|---|---|---|---|---|---|
| Pixie [1] | 2018 | Retrieval | Pin-Board graph | 3B nodes, 17B edges, 120 GB | RAM replica; p99 < 60 ms; ~100k QPS |
| PinSage [3] | 2018 | Item embedding | Graph + content | 3B nodes, 18B edges, 7.5B examples | MapReduce inference -> ANN |
| PinnerSage [6] | 2020 | User embedding | 90-day actions | Per-user medoids | Medoid ANN queries |
| PinnerFormer [7] | 2022 | User embedding | Action sequence | 256-d PinSage inputs | Daily batch |
| Learned EBR [16][17] | 2023-25 | Retrieval | Two-tower | Billions of Pins | Manas; p90 150 -> 205 ms multi-emb |
| Manas [23] | 2019-26 | ANN | Embeddings | 100M-vector benchmark | HNSW -> IVF/SPANN + SQ |
| TransAct [8] | 2023 | Ranking | Real-time seq | ~100 actions | GPU |
| TransAct V2 [10] | 2025 | Ranking | Lifelong seq | ~16k actions | Triton kernels, int8 |
| PinFM [13] | 2025 | Ranking FM | 2 yrs actions | 20B+ params | DCAT, int4; 500M+ users |
| PRL-PUTS [33] | 2026 | Blending | Exploration logs | Per-request weights | No added latency |
| ML-DCN [38] | 2026 | Ads ranking | Ads features | Matched-FLOP scaling | Neutral cost |

---

## Key takeaways for system designers

1. **Rebuild one layer at a time.** Pinterest replaced retrieval, representations, ranking modules and blending at different times while keeping a stable funnel (candidate generators, LWS, ranker, blender) [25]. A stable interface between stages is what makes this incremental evolution possible.
2. **Start with walks, then compile the graph into vectors.** Pixie gave real-time, explainable candidates with no training loop [1]. PinSage turned the same graph plus content into embeddings that any ANN system could serve and that cover cold items [3].
3. **Use random walks for neighbourhood selection.** Importance pooling bounds fan-out and weights neighbours by relevance in one step (+46% offline) [3].
4. **Make a shared embedding space a platform.** PinSage embeddings feed PinnerSage, PinnerFormer, TransAct V2 and Manas benchmarks [6][7][10][23].
5. **Pair batch and real-time user models.** A daily embedding trained on a long-horizon dense loss (PinnerFormer) plus a real-time sequence module in the ranker (TransAct) gives most of the freshness at a fraction of the cost [7][8].
6. **Train for the window you will be stale for.** The dense all-action loss cut the batch-staleness penalty from 13.9% to 8.3% [7].
7. **Action type and early fusion beat positional encoding** for engagement sequences [8]. Time-delta positions worked in ads [37].
8. **Lifelong sequences need retrieval inside the ranker.** Candidate-anchored nearest-neighbour selection with an always-keep-recent rule, plus int8 storage, makes 16k-action histories affordable [10].
9. **Serving engineering is the model.** Moving to GPUs, fused Triton kernels (103-338x latency gains) and DCAT KV reuse (+600% throughput) are what made TransAct, TransAct V2 and PinFM deployable [9][10][13][36].
10. **Pretrain once, fine-tune inside each ranker.** PinFM plugs into existing rankers rather than replacing them, keeps causal attention, and helps new items most [13].
11. **Keep two-tower artifacts version-locked.** Separate user and item services must roll forward together [16].
12. **Multiple user embeddings raise recall at a latency price.** Round-robin merging with dedup kept p90 at 205 ms [18].
13. **Quantize before you shard.** Scalar quantization halved or quartered ANN memory with little or no recall loss [23].
14. **Separate prediction from policy, then personalise the policy.** Global utility weights became per-request RL weights with a Pareto frontier for governance [33].
15. **Do not scale crossing layers naively.** DCNv2 and MaskNet saturate. Low-rank, instance-masked crossing gives a better AUC-FLOPs curve, and float16 training can overflow cross layers [37][38].

---

## References

1. Eksombatchai, C., Jindal, P., Liu, J. Z., Liu, Y., Sharma, R., Sugnet, C., Ulrich, M., Leskovec, J. "Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time." Pinterest / Stanford, WWW 2018. https://arxiv.org/abs/1711.07601
2. Colyer, A. "Pixie: a system for recommending 3+ billion items to 200+ million users in real-time." The Morning Paper, 2018. https://blog.acolyer.org/2018/05/23/pixie-a-system-for-recommending-3-billion-items-to-200-million-users-in-real-time/
3. Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W. L., Leskovec, J. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage). Pinterest / Stanford, KDD 2018. https://arxiv.org/abs/1806.01973
4. Pinterest Engineering. "PinSage: A new graph convolutional neural network for web-scale recommender systems." Pinterest Engineering Blog, 2018. https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48
5. Liu, D. C., et al. "Related Pins at Pinterest: The Evolution of a Real-World Recommender System." Pinterest, WWW 2017 companion. https://arxiv.org/abs/1702.07969
6. Pal, A., Eksombatchai, C., Zhou, Y., Zhao, B., Rosenberg, C., Leskovec, J. "PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest." Pinterest, KDD 2020. https://arxiv.org/abs/2007.03634
7. Pancha, N., Zhai, A., Leskovec, J., Rosenberg, C. "PinnerFormer: Sequence Modeling for User Representation at Pinterest." Pinterest, KDD 2022. https://arxiv.org/abs/2205.04507
8. Xia, X., et al. "TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest." Pinterest, KDD 2023. https://arxiv.org/abs/2306.00248
9. Pinterest Engineering. "How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume." Pinterest Engineering Blog, 2023. https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8
10. Pinterest. "TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation." 2025. https://arxiv.org/abs/2506.02267
11. Pinterest Engineering. "Next-Level Personalization: How 16k+ Lifelong User Actions Supercharge Pinterest's Recommendations." Pinterest Engineering Blog, 2025. https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d3
12. Social Media Today. "Pinterest Updates Ranking Model With More Data." 2025. https://www.socialmediatoday.com/news/pinterest-updates-ranking-model-with-more-data/750122/
13. Pinterest. "PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform." RecSys 2025. https://arxiv.org/abs/2507.12704
14. Pinterest. "PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems." 2025. https://arxiv.org/abs/2504.10507
15. Pinterest. "UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale." 2026. https://arxiv.org/abs/2606.00422
16. Pinterest Engineering. "Establishing a Large Scale Learned Retrieval System at Pinterest." Pinterest Engineering Blog, 2025. https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5
17. Pinterest Engineering. "Advancements in Embedding-Based Retrieval at Pinterest Homefeed." Pinterest Engineering Blog, 2025. https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e
18. Pinterest. "Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest." KDD 2025. https://arxiv.org/abs/2506.23060
19. Pinterest. "Bootstrapping Conditional Retrieval for User-to-Item Recommendations." RecSys 2024. https://arxiv.org/abs/2508.16793
20. Pinterest Engineering. "Scaling Conditional Learned Retrieval for Pinterest Home Feed." Pinterest Engineering Blog, 2026. https://medium.com/pinterest-engineering/scaling-conditional-learned-retrieval-for-pinterest-home-feed-ecfba7e5a426
21. Pinterest Engineering. "Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/manas-hnsw-realtime-powering-realtime-embedding-based-retrieval-dc71dfd6afdd
22. Koh, T. "Manas HNSW Streaming Filters." Pinterest Engineering Blog, 2022. https://medium.com/pinterest-engineering/manas-hnsw-streaming-filters-351adf9ac1c4
23. Pinterest Engineering. "Evolving Pinterest's Embedding Retrieval Platform." Pinterest Engineering Blog, 2026. https://medium.com/pinterest-engineering/evolving-pinterests-embedding-retrieval-platform-aede4e831e01
24. InfoQ. "From Memory-Hungry HNSW to Quantized SPANN: The Technical Evolution of Pinterest's Manas Platform." 2026. https://www.infoq.com/news/2026/09/pinterest-search/
25. Pinterest. "Pinterest at WWW 2022" (sponsor slides on the home feed funnel). 2022. https://archives.iw3c2.org/www2022/wp-content/uploads/Sponsors/WWW2022-Pinterest.pdf
26. Pinterest Engineering. "Building a dynamic and responsive Pinterest." Pinterest Engineering Blog. https://medium.com/@Pinterest_Engineering/building-a-dynamic-and-responsive-pinterest-7d410e99f0a9
27. MLconf. Pinterest Homefeed talk abstract (ratio-based round-robin blending). 2016. https://mlconf.com/?p=8909
28. Pinterest Engineering. "Pinnability: Machine learning in the home feed." Pinterest Engineering Blog, 2015. https://medium.com/@Pinterest_Engineering/pinnability-machine-learning-in-the-home-feed-64be2074bf60
29. Pinterest Engineering. "Pinterest Home Feed Unified Lightweight Scoring: A Two-tower Approach." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/pinterest-home-feed-unified-lightweight-scoring-a-two-tower-approach-b3143ac70b55
30. Pinterest Engineering. "Modernizing Home Feed Pre-Ranking Stage." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/modernizing-home-feed-pre-ranking-stage-e636c9cdc36b
31. Pinterest. "Rethinking Personalized Ranking at Pinterest: An End-to-End Approach." 2022. https://arxiv.org/abs/2209.08435
32. Pinterest. "Improving feature interactions at Pinterest under industry constraints." CIKM 2024 workshop (IRS). https://arxiv.org/abs/2412.01985
33. Pinterest. "A Production-Ready RL Framework for Personalized Utility Tuning with Pareto Sweeping in Pinterest Recommender Systems (PRL-PUTS)." 2026. https://arxiv.org/abs/2605.16344
34. Pinterest. "Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning." 2026. https://arxiv.org/abs/2607.14192
35. Pinterest Engineering. "Pinner Progression: Better Use-Case Representation Driving Weekly Active User Growth at Pinterest." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/pinner-progression-better-use-case-representation-driving-weekly-active-user-growth-at-pinterest-bd2131ab238a
36. Pinterest Engineering. "GPU-accelerated ML Inference at Pinterest." Pinterest Engineering Blog, 2022. https://medium.com/@Pinterest_Engineering/gpu-accelerated-ml-inference-at-pinterest-ad1b6a03a16d
37. Pinterest Engineering. "User Action Sequence Modeling for Pinterest Ads Engagement Modeling." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/user-action-sequence-modeling-for-pinterest-ads-engagement-modeling-21139cab8f4e
38. Li, J., et al. "ML-DCN: Masked Low-Rank Deep Crossing Network Towards Scalable Ads Click-through Rate Prediction at Pinterest." Pinterest, 2026. https://arxiv.org/abs/2602.09194
39. Pinterest. "MTMD: A Multi-Task Multi-Domain Framework for Unified Ad Lightweight Ranking at Pinterest." 2025. https://arxiv.org/abs/2510.09857
40. Pinterest Engineering. "Unlocking Efficient Ad Retrieval: Offline Approximate Nearest Neighbors in Pinterest Ads." Pinterest Engineering Blog. https://medium.com/pinterest-engineering/unlocking-efficient-ad-retrieval-offline-approximate-nearest-neighbors-in-pinterest-ads-6fccc131ac14
41. Pinterest. "MultiBiSage: A Web-Scale Recommendation System Using Multiple Bipartite Graphs at Pinterest." 2022. https://arxiv.org/abs/2205.10666
42. Pinterest. "OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search." 2024. https://arxiv.org/abs/2404.16260
43. Shaped.ai. "Inside PinRec - Pinterest's Production-Ready Generative Retrieval Model." 2025. https://www.shaped.ai/blog/pinrec-teardown-inside-pinterests-production-ready-generative-retrieval-model
44. ZenML LLMOps Database. "Pinterest: Large-Scale Learned Retrieval System with Two-Tower Architecture." https://www.zenml.io/llmops-database/large-scale-learned-retrieval-system-with-two-tower-architecture
