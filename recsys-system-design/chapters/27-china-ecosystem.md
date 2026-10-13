# 27. The Chinese Ecosystem: Alibaba, Tencent, Meituan, Baidu, Huawei, JD, Xiaohongshu and the Architecture of Hyperscale Commerce and Content

## Overview

A large share of the industrial recommender-system literature since 2017 comes from Chinese companies. Several reasons explain this. Alibaba, Tencent, Meituan, Kuaishou, ByteDance, Baidu, JD.com and Huawei each run several surfaces with hundreds of millions of daily users. Their advertising teams (Alimama at Alibaba, Tencent Ads, Baidu Phoenix Nest) publish heavily at KDD, SIGIR, CIKM, RecSys and WWW. Many of them also open-source their training stacks: XDL, DeepRec, EasyRec, Graph-Learn, Euler, Angel, embedx and PaddleRec.

This chapter maps that body of work by company. It focuses on the designs that became industry templates:

- **Alibaba / Taobao / Alimama.** Target attention (DIN), interest evolution (DIEN), lifelong sequences (MIMN/UIC, SIM, ETA), entire-space multi-task learning (ESMM), graph item embeddings (EGES), learned tree indexes for retrieval (TDM, JTM), multi-interest retrieval (MIND), cost-aware pre-ranking (COLD), listwise re-ranking (PRM), and the XDL, DeepRec and Graph-Learn platforms.
- **Tencent.** Multi-task learning with PLE, STEM and embedding-collapse research, WeChat look-alike retrieval (RALM), the embedx and Angel platforms, and the generative ads model GPR for WeChat Channels.
- **Meituan.** Long-sequence sampling (SDIM), multi-step conversion modeling (AITM), permutation re-ranking (PIER), and generative ranking (MTGR, EGA).
- **Huawei Noah's Ark.** Feature-interaction models (DeepFM, PNN family, FGCNN, EDCN, FinalMLP), position-bias modeling (PAL), re-ranking (PEAR), and the FuxiCTR/BARS benchmarks.
- **Baidu, JD.com, Xiaohongshu, Sina Weibo** and a note on **Pinduoduo**.

Kuaishou and ByteDance are covered in Chapter 20 and appear here only for comparison.

A note on sourcing. The live web-search budget for this run was used up before research began, and outbound access to arXiv, ACM and most company blogs failed from the writing environment. Facts here come from three kinds of source. The first is primary documents that could be fetched directly: the Alibaba XDL wiki pages for DIN, DIEN, ESMM, TDM, TDM Serving, CrossMedia and Rocket Training; the READMEs of XDL, Blaze, DeepRec, EasyRec, HybridBackend, Euler, Tencent embedx and Angel, PaddleRec, FuxiCTR and DeepCTR; and the DIN and DIEN reference-code READMEs. The second is paper results recorded in other chapters of this Atlas, which were researched from the papers themselves. The third, used rarely and always flagged, is papers that could not be re-fetched. Treat any number marked approximate as approximate.

```
           THE CHINESE "FULL-LINK" RECOMMENDATION / ADS FUNNEL
   (Alimama's own description: recall -> pre-rank -> rank -> exposure
                 -> click -> conversion -> repurchase [12])

 corpus 10^7 - 10^9 items
   |  multi-channel recall: item-CF, EGES i2i, MIND multi-interest ANN,
   |  TDM tree beam search, RALM look-alike, graph walks (Graph-Learn / embedx)
   v  ~10^3 - 10^4 candidates
 pre-rank ("cu pai")  COLD: small DNN with cross features, compute-aware
   v  ~10^2 - 10^3
 rank ("jing pai")    DIN / DIEN / SIM / ETA / BST target attention;
   |                  ESMM / PLE / AITM multi-task heads (CTR, CVR, CTCVR, ...)
   v  ~10^1 - 10^2
 re-rank              PRM / PEAR / PIER / generator-evaluator permutations
   v
 exposure -> click -> conversion -> repurchase (labels flow back via streaming)
```

## 1. Alibaba: the Alimama and Taobao line

Alibaba's most influential recommender papers came from the Alimama targeted-advertising algorithm team. The XDL wiki credits DIN, DIEN, ESMM, TDM, CrossMedia and Rocket Training to that team [3][4][12][18][24][25]. All six were open-sourced together as "XDL algorithm solutions" on top of the XDL training engine [22].

### 1.1 DIN: Deep Interest Network (KDD 2018)

**Problem.** The standard Embedding&MLP CTR model of 2016 pooled a user's behaviors into one fixed-length vector. The XDL documentation states the problem directly. A Taobao user has many concurrent interests. One fixed vector V_u limits the rank of the model's solution space, and its dimension cannot grow without limit because of compute and generalization. When predicting a click on a down jacket, interest in home appliances is noise [3].

**Architecture.** An "activation unit" uses the candidate ad to activate the user's historical clicked items. Each behavior gets a relevance weight, and the weighted sum is the user's interest representation for this candidate. That vector is concatenated with user profile, context and ad features and fed to an MLP [3]. The weights are not softmax-normalized, so the magnitude also carries interest intensity [1].

```
 candidate ad emb --------------------------+
 behavior_1 emb --[act unit(b1, ad)]-- w1 --+
 behavior_2 emb --[act unit(b2, ad)]-- w2 --+--> sum_i w_i * b_i  (candidate-specific
 ...                                        |                       user interest)
 behavior_T emb --[act unit(bT, ad)]-- wT --+
        concat(user interest, profile, ad, context) --> MLP (Dice) --> pCTR
```

**Training tricks.** Two tricks are reported [1][3]:

- **Mini-batch-aware regularization.** L2 is applied only to embedding rows of features present in the mini-batch, because full L2 over billions of parameters per step is infeasible. It added about 0.0031 AUC over dropout on Alibaba data.
- **Dice.** An activation whose rectification point adapts to the mean and variance of its input, instead of the fixed step at zero used by ReLU or PReLU. It added about 0.0015 AUC over PReLU.

**Scale and results.** The production dataset had more than 2 billion samples. An online A/B test (May to June 2017) showed up to +10.0% CTR and +3.8% RPM, and DIN served the main traffic of Alibaba display advertising [1]. The authors later flagged a BatchNorm bug in the original public demo. Their corrected Amazon-Electronics GAUC table gives PNN 0.8679, DeepFM 0.8683, DIN 0.8698 and DIN with Dice 0.8711, and recommends the refactored DIEN/XDL code instead [5]. That is a useful reminder that public-benchmark gaps between these models are small. BARS found the same thing on a much larger scale (Section 5.4).

**Lesson.** Make the user representation a function of the candidate. Nearly every later sequence model at Alibaba, Kuaishou, Meituan, Pinterest and Meta inherits this "target attention" idea.

### 1.2 DIEN: Deep Interest Evolution Network (AAAI 2019)

**Problem.** DIN treats each behavior as an interest. The DIEN authors argue that interest is latent and behaviors are only its manifestations. Supervision from ad clicks alone cannot teach a sequence model what the user wanted at each step [4].

**Architecture.** DIEN has two layers [2][4]:

- An **interest-extractor** GRU over the behavior sequence. An **auxiliary loss** asks each hidden state to predict the next clicked item against a sampled non-clicked item. The XDL docs note three benefits of this loss: it adds feedback signal, eases gradient flow on long sequences, and gives the embeddings more semantic supervision.
- An **interest-evolving** layer, **AUGRU**. The GRU update gate is scaled by the attention score between the candidate ad and each extracted interest. Each candidate therefore gets its own evolution path, and interests unrelated to it are suppressed.

**Serving.** A GRU per candidate is expensive. The paper reports cutting latency from 38.2 ms to 6.6 ms with element-parallel GRU and kernel fusion, batching adjacent requests from different users on GPU, and model compression. The system served more than a million users per second at peak, and the paper reports a 20.7% CTR improvement in Taobao display ads [2]. The XDL docs state that DIEN was deployed across Alimama's targeted-ads products [4]. Serving pressure still limited production DIEN to about 50 behaviors [6].

### 1.3 Lifelong sequences: MIMN/UIC, SIM, ETA

This line is the clearest example of Chinese system and algorithm co-design.

- **MIMN + UIC (KDD 2019).** A **User Interest Center** keeps a per-user memory state on the serving side. The state is updated *incrementally whenever a behavior event arrives*, not at request time. MIMN is a multi-channel, NTM-style memory network whose memory is exactly the UIC state. Ranking latency is therefore constant in history length, and sequences in the thousands were handled. Online (2019-03-30 to 2019-05-10), MIMN with UIC gave +7.5% CTR and +6% RPM over DIEN [6].
- **SIM (CIKM 2020).** MIMN's user-only memory degraded as histories grew 10x or more. SIM splits the problem into two units [7]:
  - A **General Search Unit (GSU)** picks the top-K relevant behaviors from up to 54,000. Hard search keeps behaviors in the candidate's category. Soft search uses MIPS over embeddings.
  - An **Exact Search Unit (ESU)** runs multi-head target attention on those K behaviors.

  The two searches returned similar lists, so the cheaper hard search went online. Its index is a two-level **User Behavior Tree** (user_id -> category_id -> behaviors), built offline, distributed, about 22 TB. Results: +7.1% CTR and +4.4% RPM, in production since 2019 [7].
- **ETA (2021-2022).** ETA replaces the separate GSU with **SimHash** signatures of the same embeddings the attention uses. Top-k selection by Hamming distance needs only bit operations, and retrieval becomes end-to-end trainable. Reported gains are 3.1% GMV for ETA [8], and +1.8% CTR and +3.1% GMV for ETA-Net on Taobao [9].

```
 UIC (event-driven)                      SIM (request-driven, two-stage)
 behavior event --> update memory        request(user, candidate c)
                    M_u in UIC             |  GSU: UBT lookup [user][cat(c)]
 request --> read M_u (O(1)) --> MIMN      |       (22 TB index, hard search)
                                           v  top-K behaviors
                                         ESU: multi-head target attn(c, top-K)
                                           v
                                         concat with short-term DIN/DIEN -> MLP
```

**BST (DLP-KDD 2019)** applied a single Transformer layer to the short behavior sequence in Taobao *recommendation* ranking. Its position feature is the time gap between the recommendation and the click. Offline AUC was 0.7894 against 0.7866 for DIN, and online CTR rose +7.57% against +4.55% for DIN [10].

### 1.4 ESMM: Entire Space Multi-Task Model (SIGIR 2018)

**Problem.** Alimama's docs decompose GMV as traffic x CTR x CVR x price per order, which makes CVR a direct lever [12]. CVR models trained only on clicked impressions suffer from **sample selection bias (SSB)**, since they score all impressions but train on clicks. They also suffer from **data sparsity (DS)**: in Taobao data CVR samples were about 4% of CTR samples [11]. Delayed feedback is a third challenge [12].

**Design.** pCVR is never supervised directly. The model predicts pCTR and pCTCVR = pCTR x pCVR over *all impressions* and supervises those two [11][12]:

```
 loss = sum_i CE(y_i, pCTR(x_i)) + sum_i CE(y_i AND z_i, pCTR(x_i) * pCVR(x_i))

  impression x --> [shared embedding layer] --+--> CTR tower --> pCTR ----+
                                              |                           (x)--> pCTCVR
                                              +--> CVR tower --> pCVR ----+
```

The product avoids dividing by small pCTR values, which is numerically unstable. The CVR tower shares embeddings with the CTR task, which has orders of magnitude more samples. That transfer is what addresses the sparsity problem [12]. Each tower can be any base network, so ESMM is a framework rather than a model [12]. The 8.9-billion-sample Taobao dataset was released as a 1% public sample, Ali-CCP [11][12].

**Successors.** ESM2 decomposes the post-click path into cart and wishlist actions [13]. ESCM2 adds a counterfactual regularizer because ESMM's CVR is itself biased [14]. Meituan's AITM generalizes the idea to learned information transfer across longer funnels (Section 3.2) [60]. Alibaba's delayed-feedback work, ES-DFM and DEFER, handles the third challenge [35].

### 1.5 EGES: billion-scale commodity embedding (KDD 2018)

Taobao's matching stage connects about a billion users to about two billion items [15]. EGES works in four steps:

1. Build a weighted directed item graph from sessions of about one hour, filtering very short clicks and over-active users.
2. Run DeepWalk-style weighted random walks and train skip-gram on them (BGE).
3. Add side-information embeddings (category, shop, brand, price level), which helps cold-start items (GES).
4. In EGES proper, learn per-item attention weights over those side embeddings.

Embedding nearest neighbors become each item's i2i candidate list, and the paper reports CTR gains over the prior CF-based matching [15]. The lesson: side-information embeddings close the cold-start gap at nearly zero serving cost.

### 1.6 TDM and JTM: learned tree indexes for retrieval

**Problem.** Item-CF and inner-product vector retrieval cannot use an expressive deep model over a corpus of tens of millions of items. Scoring every item is O(n) [18].

**Design.** All items are indexed as leaves of a tree, and each internal node represents a set of items, from coarse to fine. The tree need not be binary. A deep model acts as a per-node "interest discriminator". Retrieval runs **layer-wise beam search**: keep the top K nodes per level and expand their children. The cost of whole-corpus retrieval drops from O(n) to O(log n), and any advanced model can be used because the beam prunes aggressively [16][18]. The open-source model uses only user behaviors, split into time windows, plus node embeddings, with attention over behaviors in the style of DIN [18].

**Tree and model jointly.** The tree supplies training samples: positives are a positive leaf's ancestors, and negatives come from the same level. It also decides pruning at serving time. TDM therefore alternates two steps. It trains the model, then re-clusters the learned leaf embeddings hierarchically (k-means) to build a new tree, and repeats until both are stable [18]. JTM later learns the tree and model under one objective [17]. The XDL docs claim TDM was applied in video, product and news recommendation as well as Alimama ads [18].

**Serving.** TDM Serving is a C++ library with three layers [19]:

- an **API** layer for loading data and real-time retrieval;
- an **Index** layer for tree beam search, index management and business-logic filtering;
- a **Model** layer for per-level DNN scoring, executed by the **Blaze** inference engine.

Blaze is Alimama's high-performance inference engine for ads, search and recommendation. It uses its own ULF model format, converted from XDL output [23]. Baidu's PaddleRec also ships a TDM implementation [65].

```
 offline (XDL)                                  online (TDM Serving)
 item embs --k-means--> tree T_k                request(user behaviors)
     ^                     |                       |
     |      samples: ancestors(+), same-level(-)   v
     +---- train DNN M_k <-+                    level 1: score children of root (Blaze)
          (repeat until stable)                  keep top-K --> expand children
                                                 ... level L: top-K leaves = items
 export: tree.pb + dense/sparse/item_emb  --->   + business filters in Index layer
```

### 1.7 MIND: multi-interest retrieval at Tmall

MIND uses capsule dynamic routing to turn a user's behaviors into K interest vectors, trained with label-aware attention. Each vector issues its own ANN query. It served major traffic on the Mobile Tmall homepage [20]. It is the inverse of DIN: DIN builds one user vector per candidate at ranking time, while MIND builds several candidate-independent user vectors so retrieval stays factorized.

### 1.8 COLD and pre-ranking; Rocket Training and distillation

COLD (2020) rejected the two-tower pre-ranker because it cannot use user-item cross features. COLD uses a small fully connected network with cross features. Feature selection is made under a compute budget, and the model is co-designed with serving (low precision, parallelism) and trained online [21].

Rocket Training is Alimama's co-training compression method. A large "booster" net and a small "light" net share bottom layers such as embeddings and are trained together. A hint loss pulls the light net's logits toward the booster's. A **gradient block** freezes the booster-only parameters on that hint term so the small net cannot drag the large one down. Only the light net serves [25].

### 1.9 CrossMedia / DICM and the Advanced Model Server

DICM (CIKM 2018) adds images to CTR prediction: the ad image, plus the images of items the user clicked in the past 14 days [24]. Image vectors come from a fixed VGG16 plus a trainable fully connected layer, and user-behavior images are aggregated by sum or attention pooling. Shipping images through a parameter server for billions of samples is too costly. XDL therefore built the **Advanced Model Server (AMS)**, where *servers* run the image-embedding forward and backward passes. Images are stored once on the servers, and duplicate embeddings of the same image within a batch are removed [24]. This is an early form of the "compute near the parameters" pattern.

### 1.10 Re-ranking, RL and simulators

Taobao contributed several re-ranking and RL designs:

- **MIDNN (IJCAI 2018)**: list-level features plus beam search under an evaluator, for mutual-influence-aware ranking in Taobao search [75].
- **PRM (RecSys 2019)**: a Transformer re-ranker over the top tens of items, with pre-trained "personalized vectors" and the initial rank as an input [31].
- **Exact-K (KDD 2019)**: chooses an exact set of K items jointly [74].
- **Virtual-Taobao (AAAI 2019)**: trains a search-ranking policy in a GAN and imitation-learning simulator, reporting more than 2% online revenue gain [32].
- **PE-LTR (RecSys 2019)**: a Pareto-efficient algorithm for jointly optimizing GMV and CTR [34].

### 1.11 Platforms: XDL, DeepRec, EasyRec, Graph-Learn, Euler

- **XDL** is Alimama's framework for high-dimensional sparse data. Version 1.2 added 50-100% speedups for large-batch, low-concurrency jobs, automatic global parameter placement with request merging to remove parameter-server hotspots, and full streaming-training features: feature admission, feature eviction, incremental model export and feature counting [22].
- **DeepRec** is a heavily modified TensorFlow 1.15 developed since 2016. It supports Taobao search, recommendation and ads, and dozens of outside companies use it. It claims training at "trillion samples and over ten trillion parameters" [26]. Key features: EmbeddingVariable (dynamic, adaptive and multi-hash), multi-tier HBM/DRAM/PMem/SSD storage, asynchronous parameter-server training (grpc+seastar, StarServer) or synchronous collective training (HybridBackend, SOK), and delta checkpoints [26][73].
- **EasyRec / TorchEasyRec** is the configuration-driven model zoo on Alibaba Cloud (MaxCompute, PAI, EAS). It supports large embeddings and online learning [72].
- **Graph-Learn (formerly AliGraph)** retrains GNNs incrementally every hour [27][30]. Its **Dynamic Graph Service** pre-samples the graph update stream against pre-registered queries. One node (64 cores, 256 GB) answers 2-hop queries at 20 ms P99 and 20K QPS while ingesting 110 MB/s of updates [28]. **Euler** is the separate Alimama graph framework, with GQL and message passing [29].

### 1.12 Generative turn: LUM

Alibaba's LUM (2025) pre-trains a large user model generatively, queries it, and feeds its outputs into existing discriminative models. This "three-step paradigm" aims to unlock scaling laws without discarding the existing cascade [33].

## 2. Tencent: WeChat, Tencent Video and Tencent Ads

### 2.1 PLE: Progressive Layered Extraction (RecSys 2020 best long paper)

**Problem.** Tencent's video recommender saw a **seesaw**: MMoE improved one task at the expense of another when task correlations were complex [36].

**Design.** **CGC** gives each task its own experts plus a pool of shared experts. Task k's gate mixes only its own experts and the shared ones. **PLE** stacks CGC levels, with a shared gate feeding shared experts at the next level, so specific and shared knowledge separate progressively [36].

```
 level 1:  [VTR experts]   [shared experts]   [VCR experts]
              gate_A          gate_shared         gate_B
 level 2:  [VTR experts]   [shared experts]   [VCR experts]
              gate_A                              gate_B
           tower_A -> VTR (effective view, AUC)   tower_B -> VCR (completion, MSE)
```

**Data and results.** The dataset had about 1 billion samples. MMoE barely moved VCR (about +0.0001 MSE improvement), while PLE improved both tasks significantly. Online, view count rose 2.23% and watch time 1.84% over the previous multi-task model [36]. PLE is now a stock component: DeepCTR, FuxiCTR and EasyRec all ship it [53][71][72].

### 2.2 STEM and embedding collapse

Two Tsinghua/Tencent papers moved the multi-task and scaling debate down to the embedding layer:

- **STEM (AAAI 2024)** keeps one shared embedding table plus one per task. An "All Forward Task-specific Backward" gate lets every expert contribute forward, while stop-gradient keeps a task's gradients out of other tasks' tables [37].
- **"On the Embedding Collapse" (ICML 2024)** shows that embedding matrices collapse into low-dimensional subspaces as they are scaled. Its multi-embedding design keeps improving up to 10x embedding size [38].

### 2.3 WeChat: RALM look-alike retrieval and embedx

WeChat's article feed ("Kan Yi Kan") published **RALM** (KDD 2019), a real-time attention-based look-alike model for recommending fresh or long-tail content. The idea is to represent an item by the users who engaged with it ("seeds") and match new users against them. PaddleRec ships an implementation [39][65]. (The full paper could not be re-fetched. Treat this description as a summary of its title and framing, not its numbers.)

**embedx** is a C++ distributed embedding training and inference system. It covers graph models with billions of nodes and hundreds of billions of edges, and deep ranking and retrieval models with tens of billions of samples and features [40]. Its README lists the products it has served:

- recommendation, search and payments at WeChat: Kan Yi Kan, WeChat Channels, WeChat Search and WeChat Pay;
- security: WeChat Security and Tencent's anti-fraud work;
- content and apps: Tencent News, the App Store (Yingyongbao), QQ Music, JOOX and Tencent Classroom.

Its design notes describe four things [40]:

- **Joint graph and ranking models** (graphdeepfm, graphdssm). Training them together avoids the objective mismatch of the usual two-stage use of graph embeddings.
- **A minute-level online-learning loop.** Recall and ranking both train and serve from the real-time stream, and changes take effect end to end within minutes.
- **Incremental parameter export and asynchronous parameter updates** in serving.
- **Importance-sampling correction** for the distribution shift of real-time samples.

The README claims 30-60% machine savings over open-source frameworks [40].

**Angel**, built by Tencent and Peking University, is the older parameter-server platform in Java/Scala on YARN, with Spark-on-Angel for ML and graph algorithms [41]. Tencent is also a listed contributor to TensorFlow Recommenders-Addons dynamic embeddings [45].

### 2.4 Tencent Ads: FLEN, meta-embeddings, GPR

- **FLEN (DLP-KDD 2020)** is a field-aware CTR model designed for scalable serving [44].
- **Meta-Embedding (SIGIR 2019)** learns a generator from ad attributes to an initial ID embedding with a MAML-style two-phase objective, which warms up cold-start ads [43].
- **GPR (2025)** serves WeChat Channels ads. It is a decoder over four token types (user, organic, environment, ad). Items get RQ-KMeans+ semantic IDs, and decoding uses value-guided trie beam search. It reportedly collapsed five cascade models into one and cut latency from more than 200 ms to about 80 ms (approximate) [42].

## 3. Meituan: local services, food delivery and ads

Meituan's recommendation problems are shaped by location, time of day and multi-step conversion: impression, click, order, then delivery.

### 3.1 SDIM: sampling instead of attention (CIKM 2022)

SDIM hashes the candidate and every behavior with several hash functions and sums the behaviors that share the candidate's signature. The authors prove that this approximates target attention. Hashing runs in a separate **Behavior Sequence Encoding** service, so the CTR model only does a lookup. Meituan search reported +2.98% CTR and +2.69% VBR [59].

### 3.2 AITM: multi-step conversion (KDD 2021)

AITM targets credit-card customer acquisition, where the funnel is impression, click, application, approval, activation. An Adaptive Information Transfer module learns what to pass from earlier steps to later ones. A Behavioral Expectation Calibrator keeps later-step probabilities consistent with earlier ones. It was deployed in the Meituan app [60].

### 3.3 PIER: permutation re-ranking (WSDM 2023)

PIER picks top permutations cheaply with SimHash-style hashing of user interest, then scores them with an omnidirectional context-aware module, trained end to end. It makes generator-evaluator re-ranking affordable at Meituan's scale [61].

### 3.4 MTGR, MTGRBoost and EGA (2025)

**Problem.** Meituan found that a pure HSTU-style generative ranker, which drops user x item cross features, did worse than a mid-sized DLRM even at the largest size tried [57].

**Design.** MTGR is a Transformer encoder with HSTU-style user-level sequence compression, trained on discriminative CTR/CTCVR losses [57]:

- **Token groups**: user, long sequence, real-time and candidate. Cross features are kept *only on candidate tokens*.
- **Group LayerNorm**: separate normalization per token group.
- **Dynamic masking**: real-time tokens are causal and time-gated per candidate, and candidates cannot see each other. All m candidates of a user are scored in one pass at O((n+m)^2), against O(m n^2) when scored separately.

Per-sample inference compute was about 65x that of the DLRM baseline (approximate). Meituan called it the largest offline and online gain in about two years. Training infrastructure is described in MTGRBoost [57][58].

**EGA-V2** generates ads end to end. It uses RQ-VAE semantic IDs (3 levels x 1024 codes), factors generation into P(POI) x P(creative | POI), and adds a payment network that guarantees individual rationality, with approximate incentive compatibility enforced through a regret Lagrangian [62].

## 4. Huawei Noah's Ark: feature interactions, bias and benchmarks

Huawei's recommender research is anchored in the **Huawei App Store** (AppGallery) and in open benchmarks.

### 4.1 DeepFM (IJCAI 2017) and the PNN family

**DeepFM** replaces Wide&Deep's hand-crafted wide part with an FM component that *shares the embedding layer* with the DNN. It therefore needs no feature engineering beyond raw features. It improved AUC and logloss on Criteo and on a commercial Huawei app-store dataset [46].

**PNN (ICDM 2016)** inserts a product layer (inner or outer products of field embeddings) before the MLP [47]. It is one of the baselines DIN compared against [5]. A journal extension (the "PIN" product-network-in-network variant) generalizes the pairwise product to small per-pair subnetworks. It could not be re-fetched for this chapter, so its details are not stated here.

Later Huawei CTR models listed in FuxiCTR [53] include:

- **FGCNN (WWW 2019)**: CNN-generated new features [49];
- **EDCN (DLP-KDD 2021)**: information sharing between the parallel explicit and implicit towers [50];
- **FinalMLP (AAAI 2023)**: two MLP streams fused by bilinear interaction, plus FINAL and EulerNet (SIGIR 2023) [51];
- **MIRRN (KDD 2025)**: multi-granularity retrieval and refinement for long behavior sequences [56].

### 4.2 PAL: position bias as a probability (RecSys 2019)

PAL factorizes pCTR(click | x, pos) = pSeen(pos) x pCTR(click | x, seen). The two modules are trained jointly, and only the CTR module ranks at serving time, so position never has to be known before it is chosen. Huawei App Store A/B tests reported CTR and conversion gains over a "position as feature" baseline [48].

### 4.3 PEAR and cross-batch negatives

- **PEAR (WWW 2022)** extends PRM-style re-ranking with list-item attention over the user's *history* and an auxiliary list-level engagement task [54].
- **Cross-Batch Negative Sampling (SIGIR 2021)** keeps a FIFO memory of recent item embeddings and reuses them as extra negatives for two-tower training, at close to in-batch cost [55].

### 4.4 FuxiCTR and BARS: honesty about benchmarks

FuxiCTR is a configurable, reproducible CTR library. Its model zoo labels each model by company, and it is the easiest index of Chinese CTR work [53]. The BARS-CTR benchmark re-ran 24 models in more than 7,000 experiments and more than 12,000 GPU hours. With proper tuning, many deep models differ far less than their papers claim [52]. Read every offline AUC gain in this chapter with that result in mind.

## 5. Baidu, JD.com, Xiaohongshu, Weibo and Pinduoduo

### 5.1 Baidu

- **MOBIUS (KDD 2019)** targets Baidu's sponsored-search query-ad matching. It is described as moving the matching layer from pure relevance toward the downstream business objective (CPM), so that matching and ranking are no longer optimized for different targets. The paper uses a relevance "teacher" to keep low-relevance, high-CTR pairs out, plus approximate nearest-neighbor search. (Not re-fetched for this chapter. This summary follows the published abstract as recalled; treat the details as approximate) [66].
- **PaddleRec** is Baidu's open model zoo on PaddlePaddle, with parameter-server and multi-GPU training. It covers matching (DSSM, MultiView-SimNet), recall (TDM, MIND, RALM, YouTube-DNN, DeepWalk, GRU4Rec) and ranking models [65].
- **COBRA (2025)** is Baidu's cascaded sparse-dense generative recommender. It generates a semantic ID and then refines it with a dense vector [64].

### 5.2 JD.com

JD's best-known public recommender work is **page-wise RL** (RecSys 2018). An actor-critic recommends a whole 2-D page at once, with a deconvolutional actor and a critic that values the page. It is trained first against an offline simulator built from logs [63]. Few JD production-architecture details (QPS, model sizes) appear in the sources gathered for this chapter.

### 5.3 Xiaohongshu (RedNote)

- **GenRank (2025)** is generative ranking trained on hundreds of billions of exposure logs [67]:
  - An *action-oriented* sequence makes actions the tokens and folds items in as attributes. This halves sequence length, cuts attention compute by 75% and speeds training by 78.7%.
  - Parameter-free ALiBi-style position and time biases raise the total speedup to 94.8%.
  - Real-time features stayed important, while most historical aggregate features lost value.
- **NoteLLM (WWW 2024)** compresses each note into a special LLM token for note-to-note (item-to-item) recommendation. It trains that token with contrastive learning on co-occurrence signals, alongside hashtag and category generation. (Not re-fetched; summarized from the paper as recalled) [68].

### 5.4 Sina Weibo

Weibo contributed two widely reused interaction modules [69]:

- **FiBiNET (RecSys 2019)**: SENET-style feature importance plus bilinear interactions.
- **MaskNet (2021)**: instance-guided multiplicative masks. MaskNet later became the backbone of Twitter's heavy ranker.

### 5.5 Pinduoduo

None of the sources reachable for this chapter documents Pinduoduo's recommender architecture in a primary engineering paper or blog. Pinduoduo is far less public than its peers. Designers should not assume its stack matches the Alibaba templates.

## 6. Comparison of the major designs

| Design | Company / year | Stage | Core idea | Reported scale / result |
|---|---|---|---|---|
| DIN [1] | Alibaba 2018 | Rank | Candidate-aware attention over behaviors | 2B+ samples; up to +10% CTR, +3.8% RPM |
| DIEN [2] | Alibaba 2019 | Rank | GRU + aux loss + AUGRU | +20.7% CTR; 38.2 -> 6.6 ms; 1M+ users/s |
| MIMN+UIC [6] | Alibaba 2019 | Rank | Event-driven user memory | +7.5% CTR, +6% RPM over DIEN |
| SIM [7] | Alibaba 2020 | Rank | GSU (category search) + ESU attention | 54K behaviors; 22 TB UBT; +7.1% CTR |
| ESMM [11] | Alibaba 2018 | Rank (CVR) | pCTCVR = pCTR x pCVR over all impressions | 8.9B samples; Ali-CCP released |
| EGES [15] | Alibaba 2018 | Recall (i2i) | Session-graph walks + side-info attention | ~1B users, ~2B items |
| TDM/JTM [16][17] | Alibaba 2018-19 | Recall | Tree index + deep model, beam search | O(log n); 10^7+ items |
| MIND [20] | Alibaba 2019 | Recall | K interest capsules, K ANN queries | Tmall homepage main traffic |
| COLD [21] | Alibaba 2020 | Pre-rank | Small cross-feature DNN under compute budget | online-trained |
| PLE [36] | Tencent 2020 | Rank (MTL) | Task-specific + shared experts, stacked | ~1B samples; +2.23% views |
| embedx [40] | Tencent (WeChat) | Platform | Graph + sparse DNN, minute-level online learning | 10^9 nodes, 10^11 edges |
| GPR [42] | Tencent 2025 | Ads cascade | Generative decoder, semantic IDs | 5 models -> 1; ~80 ms |
| SDIM [59] | Meituan 2022 | Rank | Hash sampling replaces attention | +2.98% CTR |
| AITM [60] | Meituan 2021 | Rank (MTL) | Learned transfer across funnel steps | deployed in Meituan app |
| MTGR [57] | Meituan 2025 | Rank | Generative encoder + cross features | ~65x DLRM FLOPs/sample |
| DeepFM [46] | Huawei 2017 | Rank | FM + DNN with shared embeddings | Huawei app store data |
| PAL [48] | Huawei 2019 | Rank | pSeen(pos) x pCTR | App Store A/B gains |
| GenRank [67] | Xiaohongshu 2025 | Rank | Action-oriented generative sequence | 94.8% training speedup |
| Page-wise RL [63] | JD 2018 | Page | Actor-critic over 2-D pages | simulator then online |

## 7. Architecture patterns characteristic of Chinese hyperscale

Many of these ideas exist elsewhere. Their combination and early adoption are distinctive.

1. **Ads teams drive the frontier.** DIN, DIEN, MIMN, SIM, ESMM, TDM, CrossMedia, Rocket and COLD all came from Alimama's targeted-ads team [3][4][12][18][24][25]. Revenue per point of CTR or CVR justifies GPU serving and custom infrastructure earlier than organic feeds do.
2. **Model the funnel explicitly.** Chinese papers treat recall -> pre-rank -> rank -> exposure -> click -> conversion -> repurchase as one chain [12]. ESMM, ESM2, ESCM2 and AITM encode that chain in the loss. COLD and Rocket make each stage as expressive as its compute budget allows.
3. **Algorithm-system co-design is the norm, not the exception.** UIC moves computation to event time. SIM's 22 TB behavior tree, ETA's SimHash and SDIM's BSE service turn attention into indexing. AMS moves image embedding onto parameter servers. TDM's tree is both a training-sample generator and a serving index. DGS pre-samples graphs. MTGR's masking scores all candidates in one pass. In every case the model is shaped by what the serving system can do cheaply [6][7][8][24][18][28][57][59].
4. **Streaming everything.** Feature admission and eviction, incremental export, delta checkpoints and minute-level online learning appear in XDL, DeepRec and embedx, as in ByteDance's Monolith [22][26][40].
5. **Open-source platforms as recruiting and ecosystem tools.** XDL, DeepRec, EasyRec, Graph-Learn, Euler, embedx, Angel and PaddleRec are released with Chinese-first documentation and cloud integration (Alibaba Cloud PAI/MaxCompute, Baidu AI Studio) [22][26][27][29][40][41][65][72].
6. **Lifelong behavior as a first-class signal.** E-commerce and short-video users accumulate 10^4 to 10^5 behaviors. The SIM -> ETA -> SDIM -> TWIN line (Kuaishou, Chapter 20) exists because Chinese platforms had both the histories and the latency pressure [7][9][59][70].
7. **Fast embrace of generative ranking with pragmatic hybrids.** Kuaishou OneRec, Meituan MTGR and EGA, Tencent GPR, Xiaohongshu GenRank, Baidu COBRA and Alibaba LUM all shipped or reported production results in 2025. Several deliberately keep cross features, discriminative losses or the existing cascade [33][42][57][62][64][67][70].

```
         ALGORITHM-SYSTEM CO-DESIGN: WHERE THE WORK MOVES

  request-time compute  -->  moved to ...                 example
  ----------------------     ---------------------------  ----------------------
  attend over history        event-time memory update      UIC / MIMN [6]
  attend over 54K items      offline category index        SIM UBT, 22 TB [7]
  pick top-k behaviors       bit ops on SimHash codes      ETA [8], SDIM [59]
  embed images per sample    parameter-server compute      AMS / DICM [24]
  traverse graph n hops      write-time pre-sampling       DGS, 20 ms P99 [28]
  score 10^7 items           beam search on learned tree   TDM Serving [19]
  score m candidates         one pass, masked attention    MTGR [57]
```

## Key takeaways for system designers

1. **Make the user representation candidate-aware in ranking and candidate-independent in retrieval.** DIN, DIEN and SIM attend per candidate at ranking time. MIND, EGES and TDM keep retrieval factorized or indexable.
2. **Supervise in the space you serve in.** ESMM's pCTCVR trick removes sample-selection bias and borrows density from the CTR task. Extend it to multi-step funnels (ESM2, AITM) and correct its residual bias (ESCM2).
3. **When attention is too expensive, turn it into retrieval.** Use a two-stage GSU/ESU (SIM), hashing (ETA, SDIM) or event-time memory (UIC). Choose by whether you need candidate-awareness and end-to-end training.
4. **Learned indexes let deep models do retrieval.** TDM's tree, rebuilt from learned embeddings, gives O(log n) whole-corpus scoring with arbitrary models. The price is a serving engine (beam search plus per-level scoring) and a training loop that alternates tree and model.
5. **Separate task-specific from shared capacity when you see a seesaw.** PLE and CGC are the default answer. STEM pushes the separation into embeddings. Watch for embedding collapse when scaling.
6. **Budget compute per stage, and spend it where crosses pay.** COLD's cross-feature pre-ranker and Rocket's booster-light co-training show how to buy quality under a latency cap.
7. **Invest in streaming infrastructure before bigger models.** Feature admission and eviction, incremental export and minute-level online learning (XDL, DeepRec, embedx) are table stakes for fast-churning catalogs.
8. **Do not discard signal to be "purely generative".** MTGR and GenRank show that cross features and real-time features remain decisive. Hybrid generative-discriminative designs are the production norm in 2025-2026.
9. **Discount offline gains.** DIN's corrected GAUC table and the BARS benchmark both show that tuned baselines close most published gaps. Trust online A/B results and production deployment claims more than offline AUC.

## References

1. Zhou, G., Song, C., Zhu, X., et al. "Deep Interest Network for Click-Through Rate Prediction." Alibaba, KDD 2018. https://arxiv.org/abs/1706.06978
2. Zhou, G., Mou, N., et al. "Deep Interest Evolution Network for Click-Through Rate Prediction." Alibaba, AAAI 2019. https://arxiv.org/abs/1809.03672
3. Alibaba Alimama. "Deep Interest Network (DIN)" (XDL wiki page). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/wiki/%E6%B7%B1%E5%BA%A6%E5%85%B4%E8%B6%A3%E7%BD%91%E7%BB%9C(DIN)
4. Alibaba Alimama. "Deep Interest Evolution Network (DIEN)" (XDL wiki page). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E5%85%B4%E8%B6%A3%E6%BC%94%E5%8C%96%E6%A8%A1%E5%9E%8B(DIEN)
5. Zhou, G. "DeepInterestNetwork" reference implementation README (corrected GAUC results). GitHub, 2018. https://github.com/zhougr1993/DeepInterestNetwork
6. Pi, Q., Bian, W., Zhou, G., et al. "Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction" (MIMN/UIC). Alibaba, KDD 2019. https://arxiv.org/abs/1905.09248
7. Pi, Q., Zhu, X., Zhou, G., et al. "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction" (SIM). Alibaba, CIKM 2020. https://arxiv.org/abs/2006.05639
8. Chen, Q., Pei, C., et al. "End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model" (ETA). Alibaba, 2021. https://arxiv.org/abs/2108.04468
9. "Efficient Long Sequential User Data Modeling for Click-Through Rate Prediction" (ETA-Net). Alibaba, DLP-KDD 2022. https://arxiv.org/abs/2209.12212
10. Chen, Q., Zhao, H., Li, W., Huang, P., Ou, W. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba." Alibaba, DLP-KDD 2019. https://arxiv.org/abs/1905.06874
11. Ma, X., Zhao, L., et al. "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." Alibaba, SIGIR 2018. https://arxiv.org/abs/1804.07931
12. Alibaba Alimama. "Entire Space Multi-task Model (ESMM)" (XDL wiki page). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/wiki/%E5%85%A8%E7%A9%BA%E9%97%B4%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%A8%A1%E5%9E%8B(ESMM)
13. Wen, H., et al. "Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction" (ESM2). Alibaba, 2019. https://arxiv.org/abs/1910.07099
14. Wang, H., et al. "ESCM2: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation." SIGIR 2022. https://arxiv.org/abs/2204.05125
15. Wang, J., Huang, P., Zhao, H., Zhang, Z., Zhao, B., Lee, D. L. "Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba" (EGES). Alibaba, KDD 2018. https://arxiv.org/abs/1803.02349
16. Zhu, H., et al. "Learning Tree-based Deep Model for Recommender Systems" (TDM). Alibaba, KDD 2018. https://arxiv.org/abs/1801.02294
17. Zhu, H., et al. "Joint Optimization of Tree-based Index and Deep Model for Recommender Systems" (JTM). Alibaba, NeurIPS 2019. https://arxiv.org/abs/1902.07565
18. Alibaba Alimama. "Tree-based Deep Match (TDM)" (XDL wiki page). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/wiki/%E6%B7%B1%E5%BA%A6%E6%A0%91%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%9E%8B(TDM)
19. Alibaba Alimama. "TDM Serving" (XDL wiki page). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/wiki/TDMServing
20. Li, C., et al. "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall" (MIND). Alibaba, CIKM 2019. https://arxiv.org/abs/1904.08030
21. Wang, Z., Zhao, L., Jiang, B., et al. "COLD: Towards the Next Generation of Pre-Ranking System." Alibaba, DLP-KDD 2020. https://arxiv.org/abs/2007.16122
22. Alibaba Alimama. "X-DeepLearning (XDL)" repository README (v1.2 features). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning
23. Alibaba Alimama. "Blaze" inference engine README. Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/blob/master/blaze/README.md
24. Ge, T., et al. "Image Matters: Visually Modeling User Behaviors Using Advanced Model Server" (DICM / CrossMedia). Alibaba, CIKM 2018; and XDL wiki page "CrossMedia". https://arxiv.org/abs/1711.06505 ; https://github.com/alibaba/x-deeplearning/wiki/%E5%9B%BE%E5%83%8F%E8%81%94%E5%90%88%E8%AE%AD%E7%BB%83%E7%AE%97%E6%B3%95(CrossMedia)
25. Alibaba Alimama. "Rocket Training: lightweight general model compression" (XDL wiki page). Alibaba, 2018-2019. https://github.com/alibaba/x-deeplearning/wiki/%E8%BD%BB%E9%87%8F%E7%BA%A7%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E7%AE%97%E6%B3%95(Rocket-Training)
26. Alibaba. "DeepRec" repository README. Alibaba / LF AI & Data, accessed 2026. https://github.com/alibaba/DeepRec
27. Alibaba. "Graph-Learn (formerly AliGraph)" repository README. Alibaba, 2020-2023. https://github.com/alibaba/graph-learn
28. Alibaba. "Dynamic Graph Service (DGS) Introduction." Graph-Learn documentation. https://github.com/alibaba/graph-learn/blob/master/docs/en/dgs/intro.md
29. Alibaba Alimama. "Euler" graph learning framework repository README. Alibaba, 2019-2020. https://github.com/alibaba/euler
30. Zhu, R., Zhao, K., Yang, H., et al. "AliGraph: A Comprehensive Graph Neural Network Platform." Alibaba, PVLDB 12(12), 2019. https://arxiv.org/abs/1902.08730
31. Pei, C., Zhang, Y., et al. "Personalized Re-ranking for Recommendation" (PRM). Alibaba, RecSys 2019. https://arxiv.org/abs/1904.06813
32. Shi, J.-C., Yu, Y., Da, Q., Chen, S.-Y., Zeng, A.-X. "Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning." Nanjing University / Alibaba, AAAI 2019. https://arxiv.org/abs/1805.10000
33. "Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model" (LUM). Alibaba, 2025. https://arxiv.org/abs/2502.08309
34. Lin, X., et al. "A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation" (PE-LTR). Alibaba, RecSys 2019. https://www.doi.org/10.1145/3298689.3346998
35. Yang, J.-Q., et al. "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling" (ES-DFM), Alibaba, AAAI 2021, https://arxiv.org/abs/2012.03245 ; Gu, S., et al. "Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling" (DEFER), Alibaba, KDD 2021, https://arxiv.org/abs/2104.14121
36. Tang, H., Liu, J., Zhao, M., Gong, X. "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations." Tencent, RecSys 2020. https://dl.acm.org/doi/10.1145/3383313.3412236
37. Su, L., Pan, J., et al. "STEM: Unleashing the Power of Embeddings for Multi-task Recommendation." Tsinghua / Tencent, AAAI 2024. https://arxiv.org/abs/2308.13537
38. Guo, X., et al. "On the Embedding Collapse When Scaling up Recommendation Models." Tsinghua / Tencent, ICML 2024. https://arxiv.org/abs/2310.04400
39. Liu, Y., et al. "Real-time Attention Based Look-alike Model for Recommender System" (RALM). Tencent WeChat, KDD 2019 (not re-fetched; cited via PaddleRec). https://arxiv.org/abs/1906.05022
40. Tencent. "embedx" repository README and "embedx detailed introduction" (docs/intro_embedx.md); Zou, Y., et al. "EmbedX: A Versatile, Efficient and Scalable Platform to Embed Both Graphs and High-Dimensional Sparse Data," PVLDB 16(12), 2023. https://github.com/Tencent/embedx ; https://doi.org/10.14778/3611540.3611546
41. Tencent / Peking University. "Angel" repository README. Angel-ML, accessed 2026. https://github.com/Angel-ML/angel
42. "GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation." Tencent, 2025. https://arxiv.org/abs/2511.10138
43. Pan, F., Li, S., Ao, X., Tang, P., He, Q. "Warm Up Cold-start Advertisements: Improving CTR Predictions via Learning to Learn ID Embeddings." SIGIR 2019. https://arxiv.org/abs/1904.11547
44. Chen, W., et al. "FLEN: Leveraging Field for Scalable CTR Prediction." Tencent, DLP-KDD 2020. https://arxiv.org/abs/1911.04690
45. TensorFlow SIG Recommenders. "TensorFlow Recommenders-Addons" repository README. Accessed 2026. https://github.com/tensorflow/recommenders-addons
46. Guo, H., Tang, R., Ye, Y., Li, Z., He, X. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." Huawei Noah's Ark, IJCAI 2017. https://arxiv.org/abs/1703.04247
47. Qu, Y., et al. "Product-based Neural Networks for User Response Prediction" (PNN). ICDM 2016. https://arxiv.org/abs/1611.00144
48. Guo, H., Yu, J., Liu, Q., Tang, R., Zhang, Y. "PAL: A Position-bias Aware Learning Framework for CTR Prediction in Live Recommender Systems." Huawei, RecSys 2019. https://doi.org/10.1145/3298689.3347033
49. Liu, B., et al. "Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction" (FGCNN). Huawei, WWW 2019. https://arxiv.org/abs/1904.04447
50. Chen, B., et al. "Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models" (EDCN). Huawei, DLP-KDD 2021. https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf
51. Mao, K., et al. "FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction." Huawei, AAAI 2023. https://arxiv.org/abs/2304.00902
52. Zhu, J., Liu, J., Yang, S., Zhang, Q., He, X. "BARS-CTR: Open Benchmarking for Click-Through Rate Prediction." Huawei, CIKM 2021. https://arxiv.org/abs/2009.05794
53. RecZoo. "FuxiCTR" repository README and model zoo. Accessed 2026. https://github.com/reczoo/FuxiCTR
54. Li, Y., et al. "PEAR: Personalized Re-ranking with Contextualized Transformer for Recommendation." Huawei, WWW 2022 Companion. https://arxiv.org/abs/2203.12267
55. Wang, J., Zhu, J., He, X. "Cross-Batch Negative Sampling for Training Two-Tower Recommenders." Huawei Noah's Ark, SIGIR 2021. https://arxiv.org/abs/2110.15154
56. "Multi-granularity Interest Retrieval and Refinement Network for Long-Term User Behavior Modeling in CTR Prediction" (MIRRN). Huawei, KDD 2025. https://arxiv.org/abs/2411.15005
57. Han, R., et al. "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan." Meituan, 2025. https://arxiv.org/abs/2505.18654
58. "MTGRBoost: Boosting Large-scale Generative Recommendation Models in Meituan." Meituan, 2025. https://arxiv.org/abs/2505.12663
59. Cao, Y., et al. "Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction" (SDIM). Meituan, CIKM 2022. https://arxiv.org/abs/2205.10249
60. Xi, D., Chen, Z., et al. "Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising" (AITM). Meituan, KDD 2021. https://arxiv.org/abs/2105.08489
61. Shi, X., et al. "PIER: Permutation-Level Interest-Based End-to-End Re-ranking Framework in E-commerce." Meituan, WSDM 2023. https://arxiv.org/abs/2302.03487
62. "EGA-V2: An End-to-end Generative Framework for Industrial Advertising." Meituan, 2025. https://arxiv.org/abs/2505.17549
63. Zhao, X., Xia, L., Zhang, L., Ding, Z., Yin, D., Tang, J. "Deep Reinforcement Learning for Page-wise Recommendations." JD.com / Michigan State, RecSys 2018. https://arxiv.org/abs/1805.02343
64. "Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations" (COBRA). Baidu, 2025. https://arxiv.org/abs/2503.02453
65. Baidu PaddlePaddle. "PaddleRec" repository README (model list). Accessed 2026. https://github.com/PaddlePaddle/PaddleRec
66. Fan, M., Guo, J., Zhu, S., et al. "MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu's Search Engine." Baidu, KDD 2019 (not re-fetched). https://doi.org/10.1145/3292500.3330651
67. "Towards Large-scale Generative Ranking" (GenRank). Xiaohongshu, 2025. https://arxiv.org/abs/2505.04180
68. Zhang, C., et al. "NoteLLM: A Retrievable Large Language Model for Note Recommendation." Xiaohongshu, WWW 2024 (not re-fetched). https://arxiv.org/abs/2403.01744
69. Huang, T., Zhang, Z., Zhang, J. "FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction," Sina Weibo, RecSys 2019, https://arxiv.org/abs/1905.09433 ; Wang, Z., She, Q., Zhang, J. "MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask," Sina Weibo, 2021, https://arxiv.org/abs/2102.07619
70. Kuaishou cross-references (see Chapter 20): Chang, J., et al. "TWIN," KDD 2023, https://arxiv.org/abs/2302.02352 ; Lian, X., et al. "Persia," KDD 2022, https://arxiv.org/abs/2111.05897 ; Deng, J., et al. "OneRec," 2025, https://arxiv.org/abs/2502.18965
71. Shen, W. "DeepCTR" repository README (model list). GitHub, accessed 2026. https://github.com/shenweichen/DeepCTR
72. Alibaba. "EasyRec" repository README (and TorchEasyRec). Alibaba Cloud PAI, accessed 2026. https://github.com/alibaba/EasyRec
73. Alibaba. "HybridBackend" repository README. Alibaba, accessed 2026. https://github.com/alibaba/HybridBackend
74. Gong, Y., et al. "Exact-K Recommendation via Maximal Clique Optimization." Alibaba, KDD 2019. https://arxiv.org/abs/1905.07089
75. Zhuang, T., Ou, W., Wang, Z. "Globally Optimized Mutual Influence Aware Ranking in E-Commerce Search." Alibaba, IJCAI 2018. https://arxiv.org/abs/1805.08524
