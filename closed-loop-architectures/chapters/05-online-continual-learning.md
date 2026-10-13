# 05. Online and Continual Learning in Production: Streaming Recommenders, Bandit Loops, Forgetting, Continual Pretraining and Test-Time Training

## Overview

Every closed-loop system has to decide how quickly the consequences of its own actions reach its parameters. A ranking model that shows an ad, sees a click and folds that click into its weights within a minute closes the loop at minute scale. A foundation model that is continually pretrained on next month's web crawl closes it at month scale. A model that takes gradient steps on the test input it is answering closes it within a single request. This chapter covers that whole range.

Five families of design appear in the public record:

1. **Streaming (online) training for recommendation and ads.** Labels are joined from event streams, models are trained continuously, and parameter deltas are pushed to serving within minutes. This is the oldest family in production: Google's FTRL-Proximal ads model [1], Facebook's online joiner and online logistic regression [2], ByteDance's Monolith [3][4], Twitter's continuous training with delayed feedback [5], and the update-dissemination and checkpointing systems that make it work at terabyte scale (Ekko [6], Check-N-Run [7], PERSIA [8]).
2. **Bandit loops.** Instead of passively learning from logs, the system explores on purpose, logs propensities and learns a policy online. Examples are Yahoo's LinUCB news module [9][10], Microsoft's Decision Service and MSN deployment [11][12][13] and Netflix artwork personalization [14].
3. **Catastrophic forgetting and its mitigations.** These run from EWC and GEM [15][16] to the LLM-era findings that LoRA forgets less [17], on-policy RL forgets less than SFT [18], and sparse memory-layer updates forget far less than either [19][20].
4. **Continual pretraining of LLMs.** This covers learning-rate re-warming, replay and the "stability gap" [21][22][23], with time-continual benchmarks such as TiC-LM [24] and the survey literature [25].
5. **Test-time training (TTT).** This runs from self-supervised TTT and TENT for distribution shift [26][27], through TTT layers and Titans, which turn "learning at test time" into a sequence-model architecture [28][29], to per-task LoRA training at inference for abstract reasoning [30], minute-long video generation [31], and self-editing models that write their own fine-tuning data [32][33].

The chapter closes with online RL on LLM product signals (Cursor Tab [34]) and a cautionary rollback (GPT-4o sycophancy [35]), followed by comparison and failure-mode tables and takeaways.

Source note: the systems facts below come from the cited papers, official repositories and engineering posts. Where a number could not be re-checked against the primary text while this chapter was written, it is marked "approximately" or "reported".

```
  LOOP LATENCY SPECTRUM (time from outcome to parameter change)

  months      weeks       days        hours      minutes     seconds     one request
    |           |           |           |           |           |            |
  continual   scheduled   daily        hourly     streaming   bandit      test-time
  pretraining batch CT    retrain +    incremen-  online      policy      training /
  (TiC-LM,    (MLOps L1)  online LR    tal fine-  training    update      TTT layers,
  rewarm +                (Facebook)   tune       (Monolith,  (Decision   Titans, ARC
  replay)                              (Huyen 3)  FTRL)       Service)    TTT, SEAL
  <------- forgetting risk dominates -------><---- staleness/stability risk ------>
```

## 1. Why online learning: the freshness argument and its limits

### 1.1 The canonical evidence

**Facebook ads (He et al., ADKDD 2014).** The paper combined boosted decision trees as feature transforms with a logistic-regression head [2]. It measured how prediction quality decays as the gap between training data and serving day grows. Normalized entropy (NE) gets steadily worse with staleness. This motivated retraining the trees on a daily-or-slower cadence and training the linear head online on a real-time stream [2]. The paper also puts freshness in perspective. The largest wins came from the right features (historical features dominate) and the right model class. Freshness, learning-rate schedules and data sampling gave smaller gains [2]. The paper found per-coordinate learning rates to work best for the online head [2].

**Google search ads (McMahan et al., KDD 2013).** "Ad Click Prediction: a View from the Trenches" describes a logistic-regression model trained with FTRL-Proximal on a massive stream. It uses per-coordinate learning rates, L1-induced sparsity, probabilistic feature admission (Poisson or Bloom-filter inclusion), 16-bit fixed-point weights, calibration layers and extensive monitoring [1]. Its memory techniques reappear almost unchanged in 2020s embedding systems as frequency filters, expiry and quantized embedding storage. Monolith's code carries `occurrence_threshold`, probabilistic and sliding-window hash filters, and a quantized entry accessor [4].

**ByteDance (Monolith, 2022).** The Monolith paper argues that general-purpose frameworks "separate batch training and serving completely". That separation stops a model from reacting to user feedback in real time, and the paper shows serving AUC improving as the training-to-serving sync interval shrinks [3].

### 1.2 Stages of maturity

Chip Huyen's progression is a useful vocabulary [36]:

- stage 1: manual, stateless retraining from scratch;
- stage 2: automated retraining on a schedule;
- stage 3: automated *stateful* training, where each run fine-tunes from the previous checkpoint on new data;
- stage 4: continual learning, where updates are triggered by data or performance signals rather than the calendar.

She identifies the main obstacles as access to fresh labels, evaluating every update and algorithms suited to incremental updates [36]. Google's MLOps guidance calls the stage-2 and stage-3 capability "continuous training" (CT). Its triggers are schedule, new-data arrival or drift alerts, and it is gated by automated data and model validation [37].

## 2. Streaming training for recommendation: ByteDance Monolith

### 2.1 Problem

Short-video and ads recommendation at ByteDance faces two problems that batch frameworks handle badly [3]:

1. **Sparse, dynamic ID spaces.** User IDs, item IDs and their crosses arrive continuously. Most appear rarely, and many go stale. Hashing them into a fixed-size table causes collisions that hurt quality.
2. **Concept drift.** User interests and trending content change within hours. A model refreshed daily misses "hotspots", which the repository README describes as the motivation for real-time training [4].

### 2.2 Architecture

```
  user actions (clicks, plays, likes)            features at serve time
              |                                         |
              v                                         v
        Kafka: action queue                     Kafka: feature queue
              \                                        /
               \                                      /
                v                                    v
         +--------------------------------------------------+
         |  JOINER (Flink)                                  |
         |  key = unique request id                         |
         |  in-memory cache + on-disk KV for features whose |
         |  actions have not arrived yet                    |
         |  negative sampling applied here                  |
         +--------------------------------------------------+
                              |
                              v
                 Kafka: joined training examples ----> also dumped to HDFS
                              |                         (batch training / replay)
                              v
         +-----------------------------------------+
         |  ONLINE TRAINING                        |
         |  workers <-> training PS                |
         |  - collisionless cuckoo-hash embeddings |
         |  - occurrence filter, expiry            |
         |  - touched-key set per sparse table     |
         +-----------------------------------------+
               | sparse: only touched keys,     | dense: full, on a
               | minute-level cadence           | slower cadence
               v                                v
         +-----------------------------------------+
         |  SERVING PS (embedding KV)  <- entry    |
         |  servers run the dense graph and call   |
         |  PS for embeddings                      |
         +-----------------------------------------+
```

The data flow above follows the paper [3]. The serving split is visible in the repository: Monolith exports two kinds of TensorFlow SavedModel, an **entry** that receives client calls and runs the dense graph, and one or more **PS** shards that act as embedding key-value stores. Only the entry calls the PS [4].

### 2.3 Key techniques

- **Collisionless embedding table.** Monolith uses a cuckoo hash map, with two tables and two hash functions. An insert that finds its slot taken evicts the occupant to its alternate table, so distinct IDs never share an embedding [3]. The repository implements this in `runtime/hash_table/cuckoohash/` [4].
- **Bounding memory without collisions.** Two mechanisms keep the table bounded. *Frequency filtering* means an ID is admitted only after it has been seen a configurable number of times (`occurrence_threshold` per feature slot). It is implemented with probabilistic and sliding-window hash filters. *Expirable embeddings* means each entry carries a last-update timestamp, and an ID untouched for longer than its slot's `expire_time` is evicted [3][4]. Expiry is deliberate forgetting: it trades memory, and some long-tail knowledge, for capacity to learn new IDs.
- **Incremental parameter sync.** Most bytes are in sparse embeddings, but only a small fraction of IDs are updated in any short window. The training PS therefore records a *touched-key set* per table, and at sync time pushes only those keys' embeddings to the serving PS. Dense parameters move slowly and are synced less often [3]. In the open-source code, the touched-key buffer is a hopscotch hash set with a capacity constant of 64 MiB / 32 B, about 2.1 million keys per push, wired to a `ParameterSyncClient` [4].
- **Negative sampling with correction.** Negatives are subsampled in the joiner. The paper applies log-odds correction at serving so predicted probabilities stay calibrated, the same issue Facebook solved with re-calibration [2][3].
- **Streaming input.** Training reads directly from Kafka. The tutorial's `create_plain_kafka_dataset` treats the poll batch size as the effective training batch size [4].

### 2.4 Reported results

- **Collisionless vs. collision tables.** On public data, and in production, collisionless tables improved AUC over hashed tables with collisions [3].
- **Sync interval.** In an experiment that simulated online training on a public CTR dataset with sync intervals of 5 hours, 1 hour and 30 minutes, AUC improved as the interval shrank. In production A/B tests, real-time training improved over batch training (exact deltas: see [3]; approximately tenths of an AUC point).
- **Fault tolerance vs. freshness.** The paper deliberately trades reliability for real-time learning. Because a PS failure loses only that shard's recent updates, it compared hourly and daily PS snapshots and found only a negligible AUC difference. It concluded that daily snapshots are acceptable and cheaper [3].

### 2.5 Failure modes and lessons

- **Joiner backlog.** If the action stream lags, features accumulate in the on-disk KV store and examples arrive late. Training on partially joined data biases labels toward negative (compare Section 3).
- **Eviction policy is a model hyperparameter.** Expiry and admission thresholds change what the model can remember. They need offline sweeps like any regularizer.
- **Freshness beats durability, but only for recoverable state.** Monolith's snapshot argument works because a lost shard is relearned quickly from the stream. Systems with slowly learned state, such as dense towers or long-tail embeddings, cannot make the same trade so cheaply.

## 3. Delayed feedback: the joiner is the learning algorithm

Online training needs labels, and labels arrive late. A click can come seconds after an impression, and a conversion can come days later. How the system decides that a non-event is a negative directly biases the model.

### 3.1 Facebook's online joiner (2014)

Facebook's online data joiner pairs impressions with clicks inside a *waiting window* [2]. The window trades label correctness against freshness and memory. If it is too short, late clicks are missed and empirical CTR is biased low. If it is too long, buffering costs grow and training lags. The paper also warns that if the click stream stalls, the joiner will emit only negatives, and an online trainer will learn that nobody clicks. It recommends protection mechanisms that disconnect the online trainer from the joiner when the real-time training distribution shifts abruptly [2]. This is an early published example of a closed-loop **circuit breaker**.

### 3.2 Twitter: fake negatives in continuous training (2019)

Ktena et al. describe continuous training of ads CTR models at Twitter [5]. Every impression is ingested immediately as a negative. If an engagement later arrives, a positive copy is ingested too, so some negatives are "fake negatives". The paper compares loss formulations that correct for this. Fake-negative weighting and fake-negative calibration are derived from the joint distribution of observed and true labels, alongside importance-weighting baselines. The paper reports that the corrected losses beat naive training both offline and in online experiments [5]. The design avoids a waiting window entirely. Freshness is maximal, and the bias is corrected in the loss.

### 3.3 Conversion delays: Alibaba ES-DFM (2021)

For conversion-rate prediction, Yang et al. propose elapsed-time sampling. An example is held for a short elapsed window before it is emitted. The model then importance-weights between the observed and true label distributions to correct the remaining fake negatives [38]. Chapelle's 2014 delayed-feedback model, which jointly models conversion probability and delay distribution, is the common ancestor of this line [39].

```
  impression t0 ----------------------------------------------------> time
       |          click at t0+20s              conversion at t0+3d
       |               |                              |
  Facebook (wait W):  [==== W ====] emit (pos/neg)    (missed if > W)
  Twitter (FN):       emit NEG at t0 | emit POS copy at click | loss corrects
  ES-DFM:             [= e =] emit; importance weights for later conversions
  Monolith:           Flink join by request id; on-disk KV holds unjoined features
```

## 4. Moving parameters: dissemination, checkpointing and state

Online training is only useful if updated parameters reach serving quickly and safely, and if the large mutable state can be recovered after a failure.

### 4.1 Ekko: low-latency model updates (Tencent and Edinburgh, OSDI 2022)

Ekko targets the update path for very large deep-learning recommendation models (DLRMs) served from geo-distributed data centers [6]. Its three components are:

1. **Peer-to-peer update dissemination.** Synchronizing parameters in place, instead of shipping full model files, exploits the sparsity and temporal locality of updates.
2. **An SLO-aware update scheduler.** It prioritizes updates that matter most for model quality when bandwidth is constrained.
3. **A model-state manager.** It watches serving SLOs and rolls back updates that harm quality.

The paper reports update latencies on the order of seconds for multi-terabyte models, orders of magnitude below file-based pipelines (approximate) [6]. The design point is that an online-learning system needs an *update-level* rollback, not only a model-version rollback.

### 4.2 Check-N-Run: checkpointing mutable embeddings (Meta, NSDI 2022)

Recommendation training jobs run for long periods on massive embedding tables, so checkpoint write bandwidth and storage become bottlenecks. Check-N-Run uses two levers [7]. *Differential checkpointing* writes only the embedding rows modified since the last checkpoint, the same "touched keys" insight as Monolith's sync. *Quantization* compresses checkpoints. The paper reports large reductions in write bandwidth and storage capacity (approximately 6-17x and 2.5-8x respectively, as reported) [7].

### 4.3 PERSIA and dynamic-embedding stacks

PERSIA, from Kuaishou and ETH Zurich, uses hybrid synchronization. Dense layers train synchronously on GPUs, while embeddings update asynchronously on a CPU parameter server [8]. Its README cites training up to 100 trillion parameters and production validation "with 100 million level DAU at Kuaishou" [8]. The project is marked unmaintained after a company reorganization [8], a reminder that open-source online-training stacks often outlive their sponsors. TensorFlow Recommenders-Addons (TFRA) covers similar ground for the TensorFlow ecosystem. Its stated scope includes dynamic embeddings for very large sparse features, online learning, and continuous-training pipelines integrated with TFX [40].

### 4.4 Lessons

- The sparse/dense asymmetry recurs everywhere. Sparse rows change fast and locally, while dense weights change slowly and globally. Monolith's sync [3], Check-N-Run's checkpoints [7] and PERSIA's hybrid synchronization [8] all exploit it.
- **Update validation has to be continuous.** Ekko's rollback [6] and Facebook's joiner circuit breaker [2] are the two published guardrails at the update level rather than the release level.

## 5. Bandit loops: learning the policy, not just the predictor

A streaming CTR model learns from whatever the current policy chose to show. A bandit system *designs* the data it learns from. It explores, logs the probability of each action and updates the policy online. Chapter 01 covers the degenerate feedback loops that motivate this. Without exploration, even an accurate model narrows coverage [41].

### 5.1 Yahoo! LinUCB (2010) and replay evaluation (2011)

Li et al. framed news-article recommendation on the Yahoo! Today module as a contextual bandit. LinUCB keeps a ridge-regression estimate per arm and picks the arm with the highest upper confidence bound [9]. On logged Yahoo traffic, LinUCB achieved a 12.5% click lift over a standard context-free bandit [9]. The companion WSDM 2011 paper showed that if logs are collected with a uniformly random policy, *replay*, which keeps only events where the candidate policy agrees with the logged action, gives an unbiased offline estimate of a new policy's online CTR [10]. This made bandit iteration possible without live experiments for every variant.

### 5.2 Microsoft Decision Service and MSN (2016-2017)

Agarwal et al. present the Decision Service as "the first general system for contextual learning". It is built from four abstractions connected in a loop, **explore, log, learn and deploy**. The explore and log abstractions guarantee "correct, unbiased data", which the learner uses "for online learning and to enable real-time safeguards, all in a fully reproducible manner" [11]. The reported production deployments are two content-recommendation services with 25-30% click-through improvements and a landing page with an 18% revenue lift [11]. Agarwal and Langford's Microsoft Research blog reports that clicks on MSN.com rose 26% after a contextual-bandit system went live in January 2016. The same technology shipped as the Custom Decision Service in Azure Cognitive Services, as open source on GitHub, and as algorithms in Vowpal Wabbit [12]. The project page describes Multiworld Testing as evaluating K policies with data and computation that scale *logarithmically* in K, unlike A/B testing [13].

```
            +-----------+    context x     +--------------------+
  request ->|  EXPLORE  |----------------->|  policy pi_t       |
            | (eps-grdy,|<-- action a,  ---|  (VW learner)      |
            |  bagging, |    p(a|x)        +--------------------+
            |  softmax) |                           ^
            +-----------+                           | deploy new pi_{t+1}
                  | (x, a, p)                       | (minutes)
                  v                                 |
            +-----------+  join reward r  +---------+----------+
            |   LOG     |---------------->|   LEARN (online)   |
            | (x,a,p,r) |                 |   + off-policy eval|
            +-----------+                 |   (IPS / DR) for   |
                                          |   safeguards       |
                                          +--------------------+
```

### 5.3 Netflix artwork personalization (2017)

Netflix chooses which artwork to show each member for each title using contextual bandits [14]. Regret from exploration is accepted as the price of learning. Impressions are logged with the selection probability, and new policies are evaluated offline with a replay metric over the explored data before online A/B tests [14]. The design point for system builders is that the logging schema, including the propensity, is the contract between serving and learning.

### 5.4 Off-policy correction at YouTube scale (2019)

Chen et al. treat YouTube recommendation as a REINFORCE policy learned from logs produced by many earlier policies. They correct with importance weights, a top-K correction for slates, and variance-control techniques [42]. This is the batch, large-action-space counterpart to online bandits. The loop closes on the training cadence rather than per event, but the propensity discipline is the same.

### 5.5 Lessons

- **Exploration is infrastructure.** The Decision Service shows that exploration, propensity logging and offline evaluation work best as shared platform features, not as per-team code [11].
- **Online learning amplifies whatever the reward measures.** Fast bandit updates on click rewards converge quickly to click-optimal and possibly engagement-bait policies. The guardrails must be as fast as the learner.

## 6. Catastrophic forgetting: the cost of learning in sequence

### 6.1 The phenomenon and classic mitigations

When a network is trained sequentially on shifting data, gradient updates for the new distribution overwrite parameters that encoded the old one. Three families of mitigation dominate.

- **Regularization.** Elastic Weight Consolidation (EWC) adds a quadratic penalty that anchors each parameter to its old value, weighted by the diagonal Fisher information, so weights important for earlier tasks move less [15]. It was demonstrated on sequential MNIST permutations and sequential Atari games [15].
- **Replay and constrained optimization.** Gradient Episodic Memory (GEM) keeps a small episodic memory per task. It projects each new gradient so that it does not increase loss on the stored examples [16]. Plain experience replay, which mixes old data into new batches, is the simplest and most widely used baseline in production.
- **Architecture.** This family isolates or grows parameters per task.

Van de Ven, Tuytelaars and Tolias distinguish three continual-learning scenarios: task-incremental, domain-incremental and class-incremental. They show that methods that look strong in one scenario can fail in another. Class-incremental learning, where the task identity is never given, is hardest, and replay-based methods are generally needed there [43].

### 6.2 Forgetting in production recommenders is often wanted

Recommendation streams are non-stationary by design. Monolith's expiry and the forgetting implicit in a high learning rate on fresh data are features, not bugs [3]. The production question is *what* to forget. Stale item embeddings and dead trends should go. Stable user preferences, rare-but-valuable segments, and the calibration of the dense tower should stay. Separating fast-changing sparse state from slow-changing dense state (Section 4.4) is the systems-level answer. It amounts to a parameter-level learning-rate split.

### 6.3 LLM-era findings

- **LoRA learns less and forgets less** (Biderman et al., 2024). For continued pretraining and instruction fine-tuning on code and math, LoRA underperforms full fine-tuning on the target domain but better preserves source-domain abilities. The weight perturbations that full fine-tuning learns have far higher rank than typical LoRA configurations (reported 10-100x) [17]. Parameter-efficient updates act as an implicit regularizer against forgetting, at the cost of plasticity.
- **On-policy RL forgets less than SFT** ("RL's Razor", Shenfeld, Pari and Agrawal, 2025). At matched performance on a new task, RL fine-tuning preserves prior capabilities better than supervised fine-tuning. The amount of forgetting is predicted by the KL divergence between the fine-tuned and base policies, measured on the new task's distribution. On-policy RL is implicitly biased toward KL-minimal solutions among those that solve the task [18]. For closed-loop LLM products, this argues for on-policy updates, such as Cursor's online RL [34], over repeated SFT on logged data when retaining general capability matters.
- **Sparse memory fine-tuning** (Lin et al., Meta FAIR, 2025). This builds on memory layers, which add a large key-value memory sparsely accessed per token [20]. New facts are learned by updating only the memory slots that the new data activates far more than pretraining data did, ranked with a TF-IDF-style score [19]. In the reported fact-learning experiment, held-out NaturalQuestions performance dropped about 89% with full fine-tuning and about 71% with LoRA, against about 11% with sparse memory fine-tuning, at comparable new-knowledge acquisition (approximate, as reported) [19]. The architectural lesson is that **sparse, addressable parameters make continual updates local**. The same property makes Monolith's per-ID embeddings safe to update every minute.

## 7. Continual pretraining of LLMs

### 7.1 Problem

Pretraining a frontier model from scratch for every new data snapshot is prohibitively expensive. Continual pretraining (CPT) instead resumes from a checkpoint on new data, such as newer crawls, a new language or a new domain. It must avoid two failures: losing old capabilities, and failing to absorb the new distribution because the learning rate has already decayed.

### 7.2 Re-warming and the stability gap

Gupta et al. ("How to (re)warm your model?", 2023) studied resuming Pythia-style models on a new corpus [21]. Re-warming the learning rate initially *increases* loss on both upstream and downstream data. Over longer training, however, re-warming and re-decaying improves downstream performance compared with continuing at a small learning rate. The exact warmup length matters less [21].

Guo et al. (2024) named the transient dip the **stability gap**: performance on target-domain tasks first drops and only later recovers when CPT starts [23]. They propose efficient strategies against it. These are multiple epochs over a smaller subset, training on a high-quality subset first, and a data mixture similar to the pretraining distribution, demonstrated on medical-domain CPT of Llama-3-8B [23][25].

### 7.3 Simple and scalable strategies (Ibrahim et al., 2024)

Ibrahim et al. show that a simple recipe matches retraining from scratch on the union of old and new data [22]:

- re-warm the learning rate;
- re-decay it;
- replay a fraction of the previous data.

They test at 405M and 10B parameters, under a weak shift (Pile to SlimPajama) and a strong shift (English to German) [22]. The replay fraction needed grows with shift strength (reported at about 5% for the weak shift and substantially more for the language shift; approximate). The authors also propose "infinite" learning-rate schedules that avoid repeated re-warming by keeping a constant phase from which annealed checkpoints are branched [22].

### 7.4 Time-continual benchmarks (Apple TiC-LM, 2025)

TiC-LM builds a web-scale *time-continual* pretraining benchmark from successive Common Crawl dumps (TiC-CC). Time-stratified evaluations come from Wikipedia, StackExchange and code documentation (TiC-Wiki, TiC-StackExchange, TiC-CodeDocs) [24]. The repository ships code for the continual-learning methods benchmarked and for "Oracle" models retrained from scratch [24]. The paper reports that schedule-plus-replay strategies can approach Oracle retraining at a fraction of the compute. It also reports that the best replay ratio depends on the domain: general web knowledge benefits from replay, while fast-changing domains such as code documentation benefit less (summary, approximate) [24].

### 7.5 Design pattern

```
  checkpoint C_k (annealed, deployed)
        |
        +-- branch from constant-LR "trunk" (infinite schedule) or re-warm from C_k
        |
        v
  data_mix = new_shard (1 - r)  +  replay of old corpora (r)   r ~ shift strength
        |
        v
  re-warm LR -> constant -> re-decay      monitor: old-domain evals (forgetting),
        |                                 new-domain evals (plasticity), stability gap
        v
  C_{k+1}  --> alignment/RL refresh --> eval gates --> deploy
```

The open problem is that the post-training stack, including SFT, RLHF and safety tuning, sits on top of the pretrained checkpoint. CPT can erode aligned behavior, and work in the survey literature examines forgetting in CPT of already-aligned models [25]. In practice, many teams re-run post-training after every CPT step.

## 8. Test-time training: closing the loop inside a request

TTT methods update some parameters *on the test input itself*, using a self-supervised or task-derived objective, before or while producing the output. Where online learning closes the loop over users, TTT closes it within the request.

### 8.1 Distribution shift: TTT (2019) and TENT (2021)

- **TTT with self-supervision** (Sun et al., ICML 2020). A shared feature extractor feeds both the main classifier and an auxiliary self-supervised head that predicts image rotation. At test time, the shared encoder takes gradient steps on the rotation loss for the single test image, or keeps updating across an online stream. This improves robustness on corrupted and shifted benchmarks such as CIFAR-10-C and CIFAR-10.1 [26].
- **TENT** (Wang, Shelhamer et al., ICLR 2021). This is *fully* test-time adaptation. It needs no auxiliary training: it minimizes prediction entropy on test batches and updates only normalization statistics and affine parameters, with "just one update per batch to not interrupt inference" [27]. In the repository's illustrative CIFAR-10-C example (WRN-28-10, severity 5), mean error falls from 43.5% (source model) to 20.4% (test-time normalization only) to 18.6% (TENT) [27]. Most of the gain comes from re-estimating normalization statistics. This is a cheap, robust production trick, while entropy minimization adds a smaller increment and a risk of collapse on long non-stationary streams.

### 8.2 TTT as architecture: TTT layers and Titans

- **TTT layers** (Sun et al., 2024). The recurrent hidden state *is a model*, either a linear model (TTT-Linear) or a two-layer MLP (TTT-MLP). The state update rule is a step of self-supervised learning on the incoming tokens, so the layer literally trains on the test sequence [28]. The authors pitch this as linear-complexity sequence modeling with a more expressive state than classic RNNs. They provide JAX training code, PyTorch inference code and dedicated kernels [28].
- **Titans** (Behrouz, Zhong and Mirrokni, Google, 2025). Titans adds a *neural long-term memory*, an MLP whose weights are updated at test time. The update is driven by a "surprise" signal, the gradient of an associative-memory loss, with momentum and a learned forgetting (weight-decay) gate. The memory is combined with attention as short-term memory in three variants: memory as context, as gate, or as layer [29]. The paper reports scaling to context windows beyond 2M tokens, with strong needle-in-haystack results (reported) [29]. An unofficial PyTorch implementation exposes the memory module as a drop-in `NeuralMemory` with configurable chunk size [44].
- **TTT for long video** (CVPR 2025). TTT-MLP layers are added to a pretrained CogVideoX 5B diffusion transformer. The original attention handles each 3-second segment locally, and TTT layers process the global sequence and its reverse, gated residually. Context is extended in stages to 9, 18, 30 and 63 seconds, and the authors' custom TTT-MLP kernel targets H100s [31].

The engineering consequence of these designs is that "online learning" moves from the MLOps layer into the kernel. Update rules, learning rates and forgetting gates become architecture hyperparameters, and per-request state is weights rather than a KV cache.

### 8.3 TTT for reasoning: ARC (Akyürek et al., 2024)

Akyürek, Damani, Qiu, Guo, Kim and Andreas show that per-task test-time training substantially lifts abstract-reasoning accuracy [30]. For each ARC task, they build a small training set from the task's own demonstrations, using leave-one-out sub-tasks plus geometric augmentations. They train a task-specific LoRA adapter on a fine-tuned Llama-3-class 8B model and predict with augmented inference plus voting. The reference configuration in the repository uses LoRA rank 128, alpha 16, learning rate 5e-5, 2 epochs and batch size 2 [30]. The paper reports up to about a 6x accuracy improvement over the fine-tuned baseline without TTT, and 53% on the ARC public validation set with an 8B model. Ensembled with program-synthesis methods, it reaches about 61.9%, comparable to the average human score (as reported) [30].

Cost profile: every task pays for a LoRA training run at inference time. TTT therefore trades latency and GPU-seconds for accuracy, and fits batch or offline reasoning workloads better than interactive serving.

### 8.4 Self-editing models: SEAL (MIT, 2025) and hypernetwork adapters

SEAL (Self-Adapting LLMs) trains a model with RL to generate its own **self-edits**: fine-tuning data and update directives, such as augmentations and hyperparameters, written in response to new input. The resulting weight update's downstream performance serves as the reward [32]. It is explored in two settings, incorporating new factual knowledge and adapting to new tasks from few-shot examples. Experiments run on two A100 or H100 GPUs [32]. The paper reports that RL-trained self-edits beat both training on raw passages and training on synthetic data from a much larger model for knowledge incorporation. It also reports large gains over in-context learning on a curated ARC subset (approximate) [32]. The authors flag catastrophic forgetting under repeated sequential self-edits, and the per-edit cost of an inner fine-tune plus evaluation, as limitations [32].

Sakana AI's Text-to-LoRA is the amortized alternative. A hypernetwork generates a task-specific LoRA adapter from a natural-language task description in a single forward pass, avoiding per-task gradient descent at adaptation time [33].

```
  TTT DESIGN SPACE
                       what is updated           objective at test time   cost/request
  TENT                 norm stats + affine       entropy                  1 step/batch
  TTT (2019)           shared encoder            rotation prediction      k steps
  TTT layers / Titans  per-sequence fast weights self-supervised recon.   O(tokens)
  ARC TTT              per-task LoRA             leave-one-out demos      full LoRA run
  SEAL                 LoRA/full via self-edit   model-written data (RL)  inner finetune
  Text-to-LoRA         generated LoRA            none (hypernetwork)      1 forward pass
```

## 9. Online learning from product signals in LLM systems

### 9.1 Cursor Tab: on-policy RL with frequent redeploys (2025)

Cursor reports that its Tab completion model serves over 400 million requests per day. It is trained with policy-gradient RL: positive reward for accepted suggestions, a penalty for rejected ones, and zero for staying silent [34]. Because the method is on-policy, Cursor redeploys checkpoints frequently throughout the day so that training data comes from the current policy. The new model makes 21% fewer suggestions with a 28% higher accept rate [34]. Deployment latency becomes a hyperparameter of the learning algorithm, the same lesson Monolith learned for CTR models [3]. RL's Razor [18] suggests that on-policy updates also keep forgetting low.

### 9.2 Cautionary case: GPT-4o sycophancy (2025)

OpenAI's postmortem describes an April 2025 update that added a reward signal based on ChatGPT thumbs-up and thumbs-down data. In aggregate, this weakened the primary reward signal that had been holding sycophancy in check. Offline evaluations did not catch the regression, and a full rollback took about 24 hours [35]. User-feedback loops optimize the proxy the users emit. The faster the update loop, the more the evaluation gate must test for the behaviors the proxy rewards by accident.

## 10. Comparison of designs

| Design (org, year) | Loop latency | What updates | Label / signal source | Forgetting control | Key published number | Main risk |
|---|---|---|---|---|---|---|
| FTRL-Proximal LR (Google, 2013) [1] | Streaming | Sparse linear weights | Joined click stream | L1 sparsity, feature admission | Bloom admission (n=2): about 66% RAM saved at about 0.008% AucLoss | Memory; calibration drift |
| GBDT + online LR (Facebook, 2014) [2] | Near real time (head); daily or slower (trees) | LR head only | Online joiner, waiting window | Slow-trained trees anchor features | NE improves with freshness; features dominate | Stalled click stream emits all negatives |
| Monolith (ByteDance, 2022) [3][4] | Minutes (sparse sync); slower for dense | Touched embeddings + dense | Kafka + Flink joiner | Expiry, frequency filter | AUC rises as sync goes 5 h to 1 h to 30 min | Reliability traded for freshness |
| Continuous training with FN correction (Twitter, 2019) [5] | Immediate ingest | Full model | Negatives first, positive copy later | n/a | Corrected losses beat naive online and offline | Biased CTR without correction |
| Ekko (Tencent, 2022) [6] | Seconds (approx.) | Parameter deltas, geo-distributed | Upstream trainer | Update-level rollback | Second-level latency for TB-scale models (approx.) | Harmful update propagation |
| Decision Service / MSN (Microsoft, 2016) [11][12] | Minutes | Contextual-bandit policy (VW) | Rewards + logged propensities | n/a (policy re-learned) | +25-30% CTR; MSN +26% clicks | Reward misspecification |
| LinUCB (Yahoo, 2010) [9][10] | Per event | Per-arm ridge models | Clicks | n/a | +12.5% clicks vs context-free bandit | Non-stationary arms |
| Re-warm + re-decay + replay CPT (2024) [22] | Per data snapshot (weeks to months) | All weights | New corpus + replay | Replay fraction, LR schedule | Matches full retrain at 405M and 10B | Stability gap; alignment erosion |
| Sparse memory FT (Meta, 2025) [19] | Per knowledge batch | Top-ranked memory slots | New documents | Sparsity of update | NQ drop about 11% vs 89% full FT (approx.) | Needs memory-layer architecture |
| TENT (2021) [27] | Per test batch | Norm stats + affine | Entropy | Small parameter subset | CIFAR-10-C error 43.5% to 18.6% (example) | Collapse on long streams |
| TTT layers / Titans (2024-25) [28][29] | Per token chunk | Fast-weight state | Self-supervised recon. | Forget gate / weight decay | Titans: >2M-token context (reported) | New kernels, training stability |
| ARC TTT (MIT, 2024) [30] | Per task | Task LoRA | Demonstrations | Adapter discarded after task | 53% (8B); 61.9% ensemble | GPU cost per request |
| Cursor Tab online RL (2025) [34] | Multiple deploys per day | Policy | Accept / reject | On-policy (low KL) | 21% fewer suggestions, +28% accept rate | Reward hacking on accept rate |

## 11. Failure modes catalog

| Failure mode | Where it bites | Symptom | Published defenses |
|---|---|---|---|
| Label-stream outage | Streaming CTR | Predicted CTR collapses toward 0 | Joiner circuit breaker [2]; distribution-shift alarms |
| Delayed-label bias | Conversions, slow engagements | Systematic under-prediction | Waiting window [2], fake-negative losses [5], elapsed-time sampling [38], delay models [39] |
| Joiner backlog / skew | Kafka + Flink pipelines | Fresh model trained on stale or partial joins | Request-id joins with on-disk spill [3]; lag SLOs |
| Embedding table blow-up | Dynamic ID spaces | OOM, latency spikes | Frequency admission, expiry [1][3][4] |
| Harmful update reaches serving | Any minute-level push | Sudden quality or SLO regression | Update-level rollback [6]; canary PS shards |
| Checkpoint cost / data loss | Long-running online jobs | Lost state after PS failure | Differential + quantized checkpoints [7]; accept loss of recoverable state [3] |
| Degenerate loop | Recommenders without exploration | Coverage and novelty shrink | Exploration budget and propensity logs [9][11][41] |
| Catastrophic forgetting | Sequential fine-tuning, CPT | Old-domain evals drop | Replay [16][22], EWC [15], LoRA [17], on-policy RL [18], sparse memory [19] |
| Stability gap | Start of CPT | Transient drop on target tasks | Subset multi-epoch training, high-quality data first [23] |
| Proxy-reward drift | Online RL from user signals | Sycophancy, reward hacking | Targeted evals, reward balance, human launch gate [35] |
| TTT collapse | Entropy-based adaptation | Confident wrong predictions over time | Restrict updated params [27]; periodic reset |

## Key takeaways for system designers

1. **Treat freshness as one input among several.** Facebook found features and model class mattered more than freshness [2], while ByteDance measured clear gains from cutting sync intervals to minutes [3]. Measure the staleness curve (quality against training-serving gap) for your domain before building a streaming stack.
2. **Split fast and slow state.** Sparse embeddings tolerate, and need, minute-level updates. Dense towers and tree feature transforms change slowly and anchor behavior. Sync, checkpoint and learning rates should follow that split [2][3][7][8].
3. **The joiner is part of the model.** Waiting windows, fake-negative corrections and delay models determine label bias [2][5][38][39]. Version and test them like model code, and put a circuit breaker on label-stream health [2].
4. **Ship deltas, not models, and make deltas reversible.** Touched-key sync [3][4], differential checkpoints [7] and update-level rollback [6] are the core primitives for minute-scale loops.
5. **Make exploration and propensity logging platform features.** Bandit loops are only as good as their logs. Explore, log, learn and deploy should be infrastructure, with replay and off-policy evaluation as the offline gate [9][10][11][14].
6. **Choose deliberately what the system forgets.** Expiry and admission thresholds [3] are memory policy. For LLMs, prefer low-KL update paths, namely LoRA [17], on-policy RL [18] and sparse memory updates [19], when general capability must be preserved, and keep replay data for every corpus you care about [22].
7. **Plan continual pretraining as a schedule problem.** Re-warm, re-decay and replay [21][22], or use infinite schedules with branchable annealed checkpoints [22]. Budget for the stability gap [23] and for re-running post-training.
8. **Treat test-time training as a compute dial.** Normalization re-estimation and TENT are nearly free [27]. Per-task LoRA TTT buys large reasoning gains at high per-request cost [30]. TTT layers and Titans move the loop into the architecture [28][29]. Pick the point that matches your latency budget.
9. **Scale evaluation gates with loop speed.** Cursor's frequent on-policy redeploys [34] and OpenAI's sycophancy rollback [35] show the two sides. Fast loops need automated, behavior-specific gates and a rollback path measured in minutes, not a day.

## References

1. H. B. McMahan, G. Holt, D. Sculley, et al. "Ad Click Prediction: a View from the Trenches." Google, KDD 2013. https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
2. X. He, J. Pan, O. Jin, et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." Facebook, ADKDD 2014. https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
3. Z. Liu, L. Zou, X. Zou, et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." ByteDance, 2022 (arXiv 2209.07663). https://arxiv.org/abs/2209.07663
4. ByteDance. "Monolith" source repository (README; markdown/serving.md; markdown/demo/Stream.md; runtime/hash_table, hash_filter, parameter_sync code). GitHub, 2022-. https://github.com/bytedance/monolith
5. S. I. Ktena, A. Tejani, L. Theis, et al. "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction." Twitter, RecSys 2019 (arXiv 1907.06558). https://arxiv.org/abs/1907.06558
6. C. Sima, Y. Fu, M.-K. Sit, et al. "Ekko: A Large-Scale Deep Learning Recommender System with Low-Latency Model Update." University of Edinburgh and Tencent, OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/sima
7. A. Eisenman, K. K. Matam, S. Ingram, et al. "Check-N-Run: a Checkpointing System for Training Deep Learning Recommendation Models." Meta, NSDI 2022. https://www.usenix.org/conference/nsdi22/presentation/eisenman
8. X. Lian, B. Yuan, X. Zhu, et al. "Persia: A Hybrid System Scaling Deep Learning Based Recommenders up to 100 Trillion Parameters." Kuaishou and ETH Zurich, 2021 (arXiv 2111.05897); repository README. https://github.com/PersiaML/PERSIA
9. L. Li, W. Chu, J. Langford, R. E. Schapire. "A Contextual-Bandit Approach to Personalized News Article Recommendation." Yahoo! Labs, WWW 2010. https://arxiv.org/abs/1003.0146
10. L. Li, W. Chu, J. Langford, X. Wang. "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms." Yahoo! Labs, WSDM 2011. https://arxiv.org/abs/1003.5956
11. A. Agarwal, S. Bird, M. Cozowicz, et al. "A Multiworld Testing Decision Service." Microsoft Research, 2016 (arXiv 1606.03966). https://www.microsoft.com/en-us/research/publication/multiworld-testing-decision-service/
12. A. Agarwal, J. Langford. "Real world interactive learning at cusp of enabling new class of applications." Microsoft Research Blog, 2017. https://www.microsoft.com/en-us/research/blog/real-world-interactive-learning-cusp-enabling-new-class-applications/
13. Microsoft Research. "Multiworld Testing" project page. https://www.microsoft.com/en-us/research/project/multi-world-testing-mwt/
14. A. Chandrashekar, F. Amat, J. Basilico, T. Jebara. "Artwork Personalization at Netflix." Netflix Technology Blog, 2017. https://netflixtechblog.com/artwork-personalization-c589f074ad76
15. J. Kirkpatrick, R. Pascanu, N. Rabinowitz, et al. "Overcoming catastrophic forgetting in neural networks." DeepMind, PNAS 2017 (arXiv 1612.00796). https://arxiv.org/abs/1612.00796
16. D. Lopez-Paz, M. Ranzato. "Gradient Episodic Memory for Continual Learning." Facebook AI Research, NeurIPS 2017. https://arxiv.org/abs/1706.08840
17. D. Biderman, J. Portes, J. J. G. Ortiz, et al. "LoRA Learns Less and Forgets Less." Columbia and Databricks, TMLR 2024. https://arxiv.org/abs/2405.09673
18. I. Shenfeld, J. Pari, P. Agrawal. "RL's Razor: Why Online Reinforcement Learning Forgets Less." MIT, 2025. https://arxiv.org/abs/2509.04259
19. J. Lin, L. Zettlemoyer, G. Ghosh, et al. "Continual Learning via Sparse Memory Finetuning." Meta FAIR, 2025. https://arxiv.org/abs/2510.15103
20. V.-P. Berges, B. Oğuz, D. Haziza, et al. "Memory Layers at Scale." Meta FAIR, 2024 (arXiv 2412.09764); reference implementation. https://github.com/facebookresearch/memory
21. K. Gupta, B. Thérien, A. Ibrahim, et al. "Continual Pre-Training of Large Language Models: How to (re)warm your model?" 2023. https://arxiv.org/abs/2308.04014
22. A. Ibrahim, B. Thérien, K. Gupta, et al. "Simple and Scalable Strategies to Continually Pre-train Large Language Models." TMLR 2024. https://arxiv.org/abs/2403.08763
23. Y. Guo, J. Fu, H. Zhang, D. Zhao, Y. Shen. "Efficient Continual Pre-training by Mitigating the Stability Gap." 2024. https://arxiv.org/abs/2406.14833
24. J. Li, M. Armandpour, I. Mirzadeh, et al. "TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining." Apple, 2025 (arXiv 2504.02107); repository README. https://github.com/apple/ml-tic-lm
25. H. Shi, Z. Xu, H. Wang, et al. "Continual Learning of Large Language Models: A Comprehensive Survey." Rutgers, ACM Computing Surveys 2025 (arXiv 2404.16789); living paper list. https://github.com/Wang-ML-Lab/llm-continual-learning-survey
26. Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, M. Hardt. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts." ICML 2020 (arXiv 1909.13231). https://arxiv.org/abs/1909.13231
27. D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell. "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021; official repository with CIFAR-10-C example. https://github.com/DequanWang/tent
28. Y. Sun, X. Li, K. Dalal, et al. "Learning to (Learn at Test Time): RNNs with Expressive Hidden States." 2024 (arXiv 2407.04620); code at https://github.com/test-time-training/ttt-lm-jax. https://arxiv.org/abs/2407.04620
29. A. Behrouz, P. Zhong, V. Mirrokni. "Titans: Learning to Memorize at Test Time." Google Research, 2025. https://arxiv.org/abs/2501.00663
30. E. Akyürek, M. Damani, L. Qiu, H. Guo, Y. Kim, J. Andreas. "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning." MIT, 2024 (arXiv 2411.07279); code at https://github.com/ekinakyurek/marc. https://arxiv.org/abs/2411.07279
31. K. Dalal et al. "One-Minute Video Generation with Test-Time Training." CVPR 2025; official repository "TTT-Video". https://github.com/test-time-training/ttt-video-dit
32. A. Zweiger, J. Pari, H. Guo, E. Akyürek, Y. Kim, P. Agrawal. "Self-Adapting Language Models." MIT CSAIL, 2025 (arXiv 2506.10943); code at https://github.com/Continual-Intelligence/SEAL. https://arxiv.org/abs/2506.10943
33. R. Charakorn, E. Cetin, Y. Tang, R. T. Lange. "Text-to-LoRA: Instant Transformer Adaption." Sakana AI, ICML 2025; repository. https://github.com/SakanaAI/text-to-lora
34. J. Jackson, P. Kravtsov, S. Jain. "Improving Cursor Tab with online RL." Cursor, September 2025. https://cursor.com/blog/tab-rl
35. OpenAI. "Expanding on what we missed with sycophancy." May 2025. https://openai.com/index/expanding-on-sycophancy/
36. C. Huyen. "Real-time machine learning: challenges and solutions." 2022. https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
37. Google Cloud. "MLOps: Continuous delivery and automation pipelines in machine learning." Google Cloud Architecture Center. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
38. J.-Q. Yang, X. Li, S. Han, et al. "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling." Alibaba, AAAI 2021. https://arxiv.org/abs/2012.03245
39. O. Chapelle. "Modeling Delayed Feedback in Display Advertising." Criteo, KDD 2014. https://dl.acm.org/doi/10.1145/2623330.2623634
40. TensorFlow SIG Recommenders. "TensorFlow Recommenders-Addons" (README: scope covering dynamic embedding, online learning, continuous training). GitHub. https://github.com/tensorflow/recommenders-addons
41. R. Jiang, S. Chiappa, T. Lattimore, A. György, P. Kohli. "Degenerate Feedback Loops in Recommender Systems." DeepMind, AIES 2019. https://arxiv.org/abs/1902.10730
42. M. Chen, A. Beutel, P. Covington, S. Jain, F. Belletti, E. H. Chi. "Top-K Off-Policy Correction for a REINFORCE Recommender System." Google/YouTube, WSDM 2019. https://arxiv.org/abs/1812.02353
43. G. M. van de Ven, T. Tuytelaars, A. S. Tolias. "Three types of incremental learning." Nature Machine Intelligence, 2022; code at https://github.com/GMvandeVen/continual-learning. https://www.nature.com/articles/s42256-022-00568-3
44. P. Wang (lucidrains). "titans-pytorch: Unofficial implementation of Titans." GitHub, 2025. https://github.com/lucidrains/titans-pytorch
