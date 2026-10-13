# 29. LLMs in Recommender Systems (2023-2026): Rankers, Features, Agents, Distillation and Production Deployments

Between 2023 and 2026, large language models (LLMs) entered almost every stage of the recommendation stack. The integration did not take a single form, and the mainstream industrial pattern is not "ask ChatGPT what to recommend". Public work splits into five families, each with different costs:

1. **LLM as the recommender.** Recommendation is posed as text-to-text generation, as in P5, TALLRec, LLaRA, LlamaRec and LinkedIn's 360Brew.
2. **LLM as a feature or embedding producer.** The LLM runs offline or nearline to enrich item and user representations, which a conventional model consumes. Examples are KAR, LLMRec, RLMRec, NoteLLM, HLLM, LinkedIn's LLaMA-3 retriever and Spotify's narratives.
3. **LLM-native generative retrieval over item tokens.** Items become semantic IDs and a pre-trained LLM is continued-pre-trained to "speak" them. Examples are TIGER, PLUM at YouTube, Text2Tracks at Spotify, and OneRec-Think and OpenOneRec at Kuaishou.
4. **LLM as the conversational or agentic front end.** The LLM parses intent, plans and calls conventional recommender tools. Examples are InteRecAgent, Chat-REC, RecMind, Spotify AI DJ and AI Playlist.
5. **LLM as teacher.** A large model's knowledge or rankings are distilled into small, servable students. Examples are DLLM2Rec, SLMRec, SLIM and instruction distillation, and the same pattern appears in "teacher foundation model plus distilled students" at Meta and Netflix.

The economic tension is the theme of the chapter. A production ranker scores thousands of candidates per request within tens of milliseconds at tens of thousands of QPS, while a 7B-parameter LLM needs on the order of tens to hundreds of milliseconds for one forward pass over a long prompt. Almost every successful industrial design is therefore an answer to one question: where can the LLM's cost be amortized?

> **Sourcing note.** For this chapter the live web-search budget was exhausted and outbound
> HTTP to arxiv.org, company blogs and aws.amazon.com was blocked from the research
> environment. Primary sources read directly in this session were the GitHub repositories
> (READMEs, code and prompt templates) of P5, OpenP5, TALLRec, LLMRank, LLaRA, LC-Rec,
> CLLM4Rec, LLMRec and Microsoft RecAI (InteRecAgent, Knowledge Plugin, RecLM-emb), and two
> curated paper indexes (the TOIS survey list and the LLM4Rec survey list). Industry facts
> for LinkedIn, YouTube, Spotify, Netflix and Kuaishou come from the companion chapters of
> this Atlas (Chapters 6, 17, 18, 20, 23 and 25), whose authors read the primary sources.
> Some paper summaries are marked "(as recalled)". For those, the paper's existence, title and
> URL were verified from a paper index but the full text was not re-read. Amazon Rufus,
> Instacart and YouTube's Gemini consumer features could not be verified from any primary
> source in this session, and Section 29.9 treats them accordingly.

---

## 29.1 A taxonomy: where and how the LLM is used

The TOIS survey by Lin et al. organises the field along two axes [40]:

- **Where** the LLM sits in the pipeline: feature engineering, feature encoder, scoring or ranking function, user interaction, or pipeline controller.
- **How** it is adapted: whether the LLM is tuned (frozen or fine-tuned), and whether a conventional recommendation model (CRM) is kept at inference time.

The companion index by Wu et al. ("A Survey on Large Language Models for Recommendation") splits the field into discriminative uses (LLM as encoder) and generative uses (LLM as generator) [41]. A third survey focuses on LLM-based generative recommendation [42]. The CHIANGEL index alone lists several hundred papers by 2024 [40].

```
                       WHERE THE LLM SITS (2023-2026)
 +----------------------------------------------------------------------------+
 | offline / nearline                          | online (per request)         |
 |                                             |                              |
 |  [item text, reviews, images]               |  user query / dialogue       |
 |        |                                    |        |                     |
 |        v                                    |        v                     |
 |  LLM knowledge gen (KAR, LLMRec)            |  LLM agent / intent parser   |
 |  LLM embeddings (RLMRec, NoteLLM, HLLM,     |  (InteRecAgent, AI Playlist) |
 |   LinkedIn LLaMA-3 dual encoder)            |        |  tool calls         |
 |  semantic-ID tokenizer (TIGER, PLUM)        |        v                     |
 |  LLM teacher labels (DLLM2Rec, SLMRec)      |  retrieval -> rank -> rerank |
 |        |                                    |   ^          ^        ^      |
 |        +--> features/IDs/students ----------+---+          |        |      |
 |                                             |  LLM ranker (360Brew, LlamaRec)|
 |                                             |  LLM rerank / explain (DJ)   |
 +----------------------------------------------------------------------------+
   cost amortised over items or users          cost paid per request x candidate
```

The single most useful design question is which side of the vertical line the LLM lives on. On the left, cost scales with catalog size and refresh rate. On the right, it scales with traffic multiplied by candidates.

---

## 29.2 LLM as the recommender: the 2022-2024 academic foundations

### 29.2.1 P5: "Recommendation as Language Processing" (RecSys 2022)

**Problem.** Every recommendation task (rating, sequential next-item, explanation, review summarisation, direct top-k) had its own architecture and loss. P5 asked whether one text-to-text model could learn all of them with a single language-modeling objective [1][2].

**Architecture.** P5 converts "user-item interactions, item metadata, and user reviews" into natural-language sequences and pre-trains a T5 encoder-decoder (Small and Base checkpoints are released for Amazon Beauty, Toys, Sports and Yelp) on a multi-task mixture of **personalized prompts** [2]. The released template file defines five pre-training prompt families plus a zero-shot family [2]:

| Family | Task | Example source template (from the repo) | Metrics |
|---|---|---|---|
| 1 | Rating | "Which star rating will user_{} give item_{} ? (1 being lowest and 5 being highest)" | RMSE/MAE |
| 2 | Sequential | "I find the purchase history list of user_{} : {} I wonder what is the next item to recommend to the user ." | HR, NDCG, MRR |
| 3 | Explanation | generate review-style explanation for (user, item) | BLEU/ROUGE |
| 4 | Review | summarise or predict rating from review text | ROUGE / RMSE |
| 5 | Direct ("traditional") | pick or generate the item for a user, possibly from candidates | HR, NDCG |
| Z | Zero-shot | unseen prompts / cold items | varies |

Users and items appear as tokens such as `user_23` and `item_7391`. Those IDs are split by the SentencePiece tokenizer into sub-tokens, and P5 adds whole-word embeddings so the model knows which sub-tokens form one ID [2].

**Lessons.**

- P5 established the "instruction plus personalised prompt" interface that almost all later work reuses.
- It also exposed the **item-ID problem**. Arbitrary numeric IDs tokenised as text leak spurious structure: `item_7391` and `item_7392` share sub-tokens without being related. The follow-up "How to Index Item IDs for Recommendation Foundation Models" compared random, sequential, collaborative and semantic indexing [3][61]. OpenP5 turned the setup into a benchmark with 10 datasets and 3 item-indexing methods, on T5 and LLaMA-2 backbones [4].
- P5's scale is tiny by industrial standards. The OpenP5 datasets range from 943 users (ML-100K) to 277,631 users and 112,394 items (Yelp) [4].

### 29.2.2 TALLRec: aligning an LLM with recommendation via LoRA (RecSys 2023)

**Problem.** In-context ChatGPT recommendation performed close to random on binary "will the user like this item?" tasks, because a general LLM has never been trained on recommendation data. TALLRec asks how cheaply an LLM can be aligned [5].

**Architecture.**

- Base model: **LLaMA-7B**, with LoRA adapters. The code builds on alpaca-lora [6].
- Two tuning stages:
  1. **Alpaca instruction tuning**, the general instruction-following LoRA.
  2. **Rec-tuning**, a second LoRA on recommendation instructions of the form "user liked these items and disliked these; will the user enjoy target item? Answer Yes/No" [5][6].
- Scoring uses the probability of the "Yes" token, which gives an AUC-ready score.

**Results.** Few-shot AUC with only 16, 64 or 256 training samples, from the official repository [6]:

| Model | Movie 16 | Movie 64 | Movie 256 | Book 16 | Book 64 | Book 256 |
|---|---|---|---|---|---|---|
| GRU4Rec | 49.07 | 49.87 | 52.89 | 48.95 | 49.64 | 49.86 |
| SASRec | 50.43 | 50.48 | 52.25 | 49.48 | 50.06 | 50.20 |
| DROS-BERT | 50.21 | 51.71 | 53.94 | 50.07 | 48.98 | 50.20 |
| **TALLRec** | **67.24** | **67.48** | **71.98** | **56.36** | **60.39** | **64.38** |

**Lessons.** The LLM's world knowledge yields strong **data efficiency**: ID models are near-random at 256 samples, while TALLRec is well above. A 7B model can be aligned on a single GPU. However, the evaluation is pointwise (binary) with few candidates, so it says nothing about full-corpus retrieval. That gap motivated grounding work such as BIGRec, LlamaRec and LC-Rec.

### 29.2.3 LLMs as zero-shot rankers (LLMRank, ECIR 2024)

Hou et al. formulated ranking as instruction following. The prompt is built from two natural-language patterns, the user's **sequential history** and the **retrieved candidates**, and the LLM (ChatGPT in the paper) outputs a ranked list [7]. The repository states three findings [7]:

1. **Order blindness.** LLMs "struggle to perceive the order" of interaction histories. Recency-focused prompting and in-context examples trigger order awareness.
2. **Biases.** LLM rankers show **position bias**, where the candidate's slot in the prompt changes its rank, and **popularity bias**. Prompt design and **bootstrapping** reduce both. Bootstrapping means shuffling candidate order across several calls and aggregating.
3. **Zero-shot ability is real but conditional.** It is strongest on candidates retrieved by multiple candidate generators with different strategies.

Early evaluations of ChatGPT as a recommender, "Is ChatGPT a Good Recommender?" [14] and "Uncovering ChatGPT's Capabilities in Recommender Systems" [15], reached similar conclusions. InstructRec instead fine-tuned FLAN-T5-XL (3B) on recommendation instructions [13]. Zero-shot quality trails tuned ID models on accuracy, while explanation and dialogue are strengths. FaiRLLM showed that LLM recommendations vary with sensitive user attributes stated in the prompt, which makes fairness evaluation necessary [16].

**Design implication.** Bootstrapping multiplies cost by the number of permutations. Position bias also means that a listwise LLM reranker is **not permutation-invariant**, which complicates logging and counterfactual evaluation (Chapter 12).

### 29.2.4 Grounding: from "generate a title" to "pick a real item"

A generative LLM can name items that do not exist. Four solutions recur:

| Approach | Mechanism | Example |
|---|---|---|
| Post-hoc grounding | Generate text, then map it to the nearest real item embedding | BIGRec (LLaMA-7B, LoRA) [11] (as recalled) |
| Candidate verbalizer | Retrieve about 20 candidates with a sequential model, label them with letters, read the LLM's next-token distribution over the letter tokens in one forward pass | LlamaRec (LLaMA-2 7B, QLoRA) [10] (as recalled) |
| Collaborative tokens in vocabulary | Add user and item ID tokens and train their embeddings while freezing the LM | CLLM4Rec (LinkedIn + UVA, WWW 2024) [9] |
| Learned semantic indices plus alignment tasks | Vector-quantize item embeddings into non-conflicting indices, then teach the LLM via alignment tasks | LC-Rec (LLaMA-7B) [8] |
| Hybrid prompts with CF embeddings | Project SASRec/GRU4Rec item embeddings into the LLM token space and mix them with text tokens; curriculum learning | LLaRA (LLaMA-2 7B, SIGIR 2024) [12] |

The CLLM4Rec repository describes its tokenizer extension concretely. With original vocabulary size N, the token `user_i` gets ID N+i and `item_j` gets ID N+I+j, with user and item tokens treated as atomic words. Only the user and item embeddings are updated during pre-training, while the transformer and the original token embeddings stay frozen. A recommendation-oriented fine-tuning stage then generates multiple items over the **whole item space** "without hallucination" [9].

LLaRA's updated numbers show the effect of the injected CF signal [12]:

| LLaRA variant (CF encoder) | MovieLens HR@1 | Steam HR@1 | LastFM HR@1 |
|---|---|---|---|
| LLaRA (GRU4Rec) | 0.4000 | 0.4916 | 0.4918 |
| LLaRA (Caser) | 0.4211 | 0.4621 | 0.4836 |
| LLaRA (SASRec) | 0.4526 | 0.5051 | 0.5246 |

LLaRA also reports a **ValidRatio** of about 0.95-0.996, the fraction of outputs that name a valid candidate [12]. Production designs need this metric: every LLM ranker should log the share of malformed or out-of-catalog outputs.

LC-Rec's README states the two key ideas directly [8]. The first is a "learning-based vector quantization method with uniform semantic mapping, which can assign meaningful and non-conflicting IDs". The second is tuning tasks that make the LLM "deeply integrate language and collaborative semantics". This is the academic precursor of the semantic-ID tokenizers used at YouTube and Kuaishou (Section 29.4).

---

## 29.3 LLM-generated features, knowledge and embeddings (the most deployed pattern)

The pattern that ships most easily keeps the production model and adds LLM outputs as features. Cost scales with the number of items (or user-refresh events), not with traffic.

```
  OFFLINE / NEARLINE (batch, cacheable)                     ONLINE (unchanged latency)
  item text, reviews, images, user history summaries
          |
          v
  +-----------------------+    +------------------+    +----------------------------+
  | LLM (frozen, API or   |--->| text encoder /   |--->| feature store: dense vecs, |
  | fine-tuned): generate |    | adapter (small)  |    | tags, semantic IDs         |
  | reasoning, attributes |    +------------------+    +-------------+--------------+
  | profiles, labels      |                                          |
  +-----------------------+                                          v
                                                     conventional CTR / two-tower model
                                                     (DIN, DCN, SASRec ...) at full QPS
```

### 29.3.1 KAR: open-world knowledge augmentation (Huawei, 2023)

KAR ("Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models") prompts a frozen LLM (ChatGPT in the paper listing [17]) for two kinds of knowledge. **Reasoning knowledge** covers user preferences, for example "what factors does this user care about in movies?". **Factual knowledge** covers items. KAR uses "factorization prompting", which first enumerates the scenario-specific factors that matter and then asks per factor. A small **hybrid-expert adaptor** compresses the generated text encodings into augmentation vectors that any CTR model can consume. Because the knowledge is pre-computed and stored, online inference cost is unchanged (as recalled). The follow-up "Efficient and Deployable Knowledge Infusion for Open-World Recommendations via LLMs" [37] and a decoding-acceleration paper for "industrial deployable LLM-based recommender systems" [36] focus on making that offline generation cheap enough to refresh. Rec-GPT4V extends knowledge generation to item images with large vision-language models [39]. Both titles appear in [40]. The acceleration work speeds up long-form knowledge generation for very large user and item populations (as recalled).

### 29.3.2 LLMRec: LLM-based graph augmentation (WSDM 2024)

LLMRec (HKU with Baidu) uses an LLM to augment the user-item interaction graph in three ways, as the repository states [18]:

1. **Reinforcing user-item edges**, where the LLM samples plausible positive and negative items from candidates.
2. **Enhancing item node attributes**, where the LLM generates missing attributes.
3. **Profiling user nodes**, where the LLM writes user profiles from history.

The released artifacts show the cost pattern. Augmented text was generated once with `gpt-3.5-turbo-0613` and embedded with `text-embedding-ada-002`, and the recommender was then trained on the augmented data in a separate stage [18]. A denoising step prunes unreliable augmented edges.

### 29.3.3 Representation alignment: RLMRec, ONCE, LLM-CF

- **RLMRec** (WWW 2024) aligns CF embeddings with LLM-generated user and item profile representations through contrastive and generative alignment, as a plug-in for existing recommenders [19] (as recalled).
- **ONCE** (WSDM 2024) combines an open-source LLM fine-tuned as a content encoder with a closed-source LLM used as a prompt-based data enricher, for content-based (news) recommendation [20] (as recalled).
- **LLM-CF**, "Large Language Models Enhanced Collaborative Filtering" [60], and "Breaking the Barrier", an LLM-built inferential knowledge graph for industrial recommendation [38], follow the same "generate offline, consume online" structure.

### 29.3.4 LLM item and user encoders in production-scale research

- **NoteLLM (Xiaohongshu, WWW 2024)** compresses each note into a special token from the LLM's last layer. That embedding is trained contrastively on co-occurrence pairs and used for item-to-item retrieval, with hashtag and category generation as auxiliary tasks [56].
- **HLLM (ByteDance, 2024)** uses an **item LLM** to read item text and produce an item embedding, and a **user LLM** to model the sequence of item embeddings and predict the next one. The released HLLM-1B is initialised from TinyLlama-1.1B and HLLM-7B from Baichuan2-7B. On the Books benchmark, HLLM-1B reaches R@10 9.28 and HLLM-7B 9.39, versus 6.50 for HSTU-large [55] (Chapter 20).
- **Microsoft RecLM-emb** fine-tunes LMs for "versatile text-based item retrieval". It trains on ten query types: user history, item-to-item, user summary, full or sparse attributes, misspelled names, vague conditions, negative attributes, and history plus explicit query. The point is that general text-embedding models "often obtain suboptimal performance" for item retrieval [21][23].
- **Kuaishou LEARN** ("Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application") [35] freezes an LLM as an item content encoder and trains a twin-tower adaptor on collaborative signals. This avoids catastrophic forgetting from fine-tuning the LLM on ID data (as recalled).

### 29.3.5 LinkedIn: LLaMA-3 dual-encoder retrieval for the Feed (2025)

This is one of the clearest public descriptions of an LLM inside a high-QPS production retrieval stage [44] (Chapter 23):

- **Model.** A fine-tuned **LLaMA 3** decoder used as a **dual encoder**. Member text (profile plus activity) and post text are encoded separately and pooled into embeddings.
- **Scale.** About **2,000 candidates from a pool of hundreds of millions of posts**, in a budget of a few milliseconds at several thousand QPS (approximate).
- **Trick 1: numbers in prompts.** Numeric features, such as counts and recency, worked better when **quantized into discrete tokens** than when written as raw digits.
- **Trick 2: who benefits.** The biggest online gains went to **new members** with sparse networks. This is the cold-start regime, where text understanding beats ID and graph signals.

```
 member profile + recent activity text --> LLaMA-3 (fine-tuned) --pool--> member vec
                                                                                |
 post text (nearline, on create/update) --> LLaMA-3 (shared or twin) --pool--> post vec
                                                                                |
                         ANN / GPU exhaustive search over hundreds of millions  v
                         -----------------------------------------> ~2,000 candidates
                                                                    -> Feed ranker
```

The cost works because post embeddings are computed once per post, nearline, while member embeddings are refreshed asynchronously. The online path is a vector search.

---

## 29.4 LLM-native generative retrieval: semantic IDs meet pre-trained LLMs

Chapter 6 covers semantic IDs in depth. Here the focus is the step where a **pre-trained language model** is adapted to an item-token vocabulary.

### 29.4.1 YouTube PLUM (2025)

PLUM ("Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations") is Google's most detailed public account of LLM-based recommendation at YouTube [46] (Chapter 17):

```
 1. Item tokenization: Semantic IDs v2 (multimodal content emb + co-engagement)
       -> RQ-style multi-level codes, tuned for coverage and low collision
 2. Continued pre-training (CPT) of a Gemini-family LLM on:
       user watch sequences as SID tokens + SID<->text alignment (titles, topics)
 3. Task fine-tuning for generative retrieval:
       user context + history (SIDs + text) -> decode SIDs of next videos
 4. Serving: beam decoding of SIDs -> videos, as one retrieval source
       feeding the existing ranking stack
```

- The paper reports that the generative retriever beat a heavily optimised production large-embedding retrieval model in live YouTube experiments. It also studies scaling with model size and compute (approx.) [46].
- **Cost profile shift.** PLUM uses far fewer item-specific parameters than embedding-table retrieval, but much more compute per request. Memory-bound becomes compute-bound (TPU) [46].
- PLUM builds on TIGER's RQ-VAE semantic IDs [47] and on YouTube's finding that semantic IDs improve ranking generalisation at a fixed table size (Chapter 17).

Earlier Google work used LLMs to **name user interest journeys** (for example "learning to bake sourdough") from activity, with prompt tuning on a small set of human-labelled journeys. This was framed as research toward explainable, goal-aligned recommendations, not as a launch [48].

### 29.4.2 Spotify Text2Tracks (2025)

Text2Tracks fine-tunes an encoder-decoder LLM to **generate track identifiers directly from a text prompt**, such as "upbeat 80s synthpop for running" [50]. The key finding is about the **ID scheme**. Random integer IDs, or titles used as IDs, perform worse than **semantic IDs built by hierarchical clustering of collaborative-filtering embeddings**. Short IDs are also far cheaper to decode than titles, and they avoid hallucinated tracks (Chapter 25).

### 29.4.3 Kuaishou OneRec-Think and OpenOneRec (2025)

- **OneRec-Think** is built on **Qwen3**, with the vocabulary extended by semantic-ID tokens. Training has three stages: **itemic alignment** (mapping item tokens into text space), **reasoning activation** (chain-of-thought SFT), and **reasoning enhancement** (RL with a reward that allows multiple valid answers). It reports a further **+0.159% app stay time** online [52] (Chapter 6).
- **OpenOneRec** released OneRec-1.7B and 8B on Qwen3 backbones. These were trained on about 33B tokens, and "-Pro" variants on about 130B tokens including industrial data. Post-training runs multi-task SFT, then **on-policy distillation to restore general reasoning**, then RL. The release also includes **RecIF-Bench**, 8 tasks over 100M interactions from 200K users [53].
- **MiniOneRec** is a community reproduction. It builds RQ-VAE / RQ-KMeans semantic IDs from text embeddings, runs SFT with language alignment and GRPO with a rank-aware reward, and decodes with **constrained beam search** through a logits processor so that only valid IDs are emitted [54].
- **Snap GRID** is an open framework that quantizes LLM text embeddings of items into hierarchical semantic IDs, using residual k-means, RVQ or RQ-VAE, for TIGER-style generation [57].

**Lesson.** Kuaishou's "restore general reasoning" step is the giveaway. Fine-tuning an LLM heavily on item tokens erodes its language ability, which defeats the reason for starting from an LLM. Alignment (SID-to-text) and anti-forgetting steps are now standard.

---

## 29.5 LLM as the ranker in production: LinkedIn 360Brew (2025)

360Brew is the most aggressive public "LLM is the ranker" system [43] (Chapter 23).

- **Problem.** Remove "hundreds of thousands of handcrafted features" and many per-surface rankers. Fix cold start, hand-crafted interactions and poor domain transfer.
- **Formulation.** Each member's profile and history are verbalized as text. Ranking becomes **many-shot in-context learning**: P(interaction on new entity | instruction, profile, history). For comparison with binary rankers, the model answers yes or no, and the **logits of the answer tokens** are the score.
- **Model.** **150B-parameter decoder-only**, based on the **Mixtral 8x22B** MoE, further trained on LinkedIn entity and interaction data across 5+ surfaces. It is evaluated on **30+ tasks across 8+ surfaces**, built by a small team over **9 months**, and labelled "pre-production" (V1.0).
- **Results.** Comparable to or better than production models offline **without task-specific fine-tuning**. Performance improved with 3x more data over five releases, with larger models, and with longer context (normalised to an 8K-token baseline). The largest gains are in cold start (history cut from 100 to 5 interactions) and on unseen surfaces. Quality also degrades more slowly as test data ages.
- **Prompt steering.** Instructions can include steering such as "30% weight to profile match, 70% to historical activity". This form of product control is impossible in a feature-based ranker.

```
 Instruction: predict apply/view/dismiss for the Question job (optional steering)
 Member Profile: position, company, location
 Past job interactions:  applied: [title, location, company, description] ...
                         viewed:  [...]   dismissed: [...]
 Question: Will the member apply to: [job text]
 Answer:   "The member will apply"   <-- score = logit(answer tokens)
```

**Cost.** The V1.0 report does not discuss serving a 150B MoE at Feed QPS. LinkedIn's parallel retrieval work uses a much smaller LLaMA-3 dual encoder [44]. A reasonable inference, not a published fact, is that 360Brew acts as a teacher or as a ranker for high-value, lower-QPS surfaces. LinkedIn open-sourced **Liger-Kernel** (Triton RMSNorm, RoPE, SwiGLU, fused linear cross-entropy). It reports about +20% multi-GPU training throughput and about 60% memory reduction, which helps make LLM-scale training affordable [45].

---

## 29.6 Conversational and agentic recommenders

### 29.6.1 Why conversational recommendation changed in 2023

Pre-LLM conversational recommender systems (CRS) such as UniCRS (KDD 2022, DialoGPT-small with knowledge-enhanced prompts) [29] were hard to build and evaluated on small dialogue datasets. LLMs made the dialogue layer nearly free. "Rethinking the Evaluation for Conversational Recommendation in the Era of LLMs" (EMNLP 2023) argued that static turn-level metrics undercount LLM ability. It proposed **interactive evaluation with LLM-simulated users** (iEvaLM) [28] (as recalled). LLM-simulated users became a standard evaluation tool.

### 29.6.2 InteRecAgent (Microsoft RecAI): LLM brain, recommender tools

InteRecAgent is the reference design for "LLM orchestrates, recommender executes" [21][22]. From the repository [59]:

- **Components.**
  1. An LLM as the conversational agent.
  2. An **item profile table** (id, title, tags, description, price, release date, visit counts, and so on).
  3. A **query module**, a SQL tool over the item table.
  4. A **retrieval module** with at least two tools: a **SQL tool** for hard conditions ("popular sports games") and an **item-similarity tool** for implicit intent ("games similar to Call of Duty").
  5. A **ranking module**, which can be a traditional recommender, that orders candidates by a schema (popularity, similarity or preference) using the user profile of `prefer` and `unwanted` items.
- **Principle.** "Neither the LLM nor the recommendation tools will be updated or modified." Integration is purely through tool interfaces [22].
- Paper details (as recalled): a **candidate memory bus** passes candidate sets between tools without putting thousands of item IDs in the prompt, a **plan-first execution** strategy makes one plan call rather than many ReAct rounds, and **RecLlama** is a 7B model fine-tuned to imitate GPT-4 tool plans for cheaper serving [22].

```
 user: "something like Hollow Knight but co-op, under $20"
   |
   v
 LLM planner --(1 plan call)--> [SQL: price<20 AND tag~co-op] -> candidate bus
                                [item-sim: Hollow Knight]     -> candidate bus
                                [rank: preference(user profile)] -> top-k
   |                                                                 |
   +<------------------------ results (IDs -> titles) ---------------+
   v
 LLM writes the answer + explanation; profile memory updated (prefer/unwanted)
```

The RecAI repository also covers **Knowledge Plugins** (DOKE), which inject collaborative item-to-item and user-to-item knowledge extracted from MF and SASRec embeddings into prompts without fine-tuning [58]. It further includes **RecExplainer**, which uses LLMs as surrogate models to explain an embedding-based recommender, and **RecLM-eval**, an evaluation service [21].

### 29.6.3 Other agentic designs

| System | Year | Role of LLM | Key idea | Source |
|---|---|---|---|---|
| Chat-REC | 2023 | ChatGPT with recommender candidates injected into the prompt | Interactive, explainable rec; conversation turns become queries for a CRM | [24] |
| RecMind | 2023 (NAACL 2024) | LLM agent with planning over tools | "Self-inspiring" planning that keeps alternative reasoning paths | [25] (as recalled) |
| InteRecAgent | 2023 | Planner and dialogue; tools do retrieval and rank | Candidate bus, plan-first, RecLlama | [22] |
| Agent4Rec | 2023 | LLM agents *simulate users* | Generative user agents with profile, memory and action modules for offline evaluation | [26] (as recalled) |
| AgentCF | 2023 (WWW 2024) | Both users and items as agents | Agents "collaboratively learn" by reflecting on mismatched choices | [27] |
| Spotify AI DJ | 2023 | Narration only; recommender selects | LLM scripts grounded by editors, plus TTS | [49] |
| Spotify AI Playlist | 2024 | Prompt -> intents -> retrieval | Refinable via follow-up prompts | [49][50] |

### 29.6.4 Spotify AI DJ and AI Playlist: LLM at the edge of a mature stack

Spotify's DJ (beta from February 2023, US and Canada) is a useful counterexample to "LLM as recommender" (Chapter 25):

1. The **existing personalization stack** chooses segments of tracks: recent favourites, rediscoveries and new music.
2. **Generative AI (OpenAI LLM technology)** writes short commentary. Spotify's music editors supply facts and style through prompts, which grounds the output.
3. A **dynamic AI voice** (from the Sonantic acquisition) speaks the script [49].

The LLM explains and frames, while the recommender selects. This keeps quality controllable and latency predictable, and it adds a new feedback signal: tapping DJ to "switch the vibe" is a segment-level negative. AI Playlist (2024) lets the user type a prompt. Spotify Research's Text2Tracks [50] shows one way to map prompts to tracks generatively, though Spotify has not published the production system's internals.

---

## 29.7 Knowledge distillation from LLMs

Distillation is the bridge from LLM quality to recommender latency. Three flavours appear:

1. **Ranking distillation into conventional sequential models.**
   - "Distillation Matters: Empowering Sequential Recommenders to Match the Performance of LLMs" (DLLM2Rec) [30] distills an LLM recommender's rankings and embeddings into SASRec/GRU-style students, using importance-aware ranking distillation and collaborative embedding distillation. The student keeps conventional inference cost (as recalled).
   - "Can Small Language Models be Good Reasoners for Sequential Recommendation?" (SLIM, WWW 2024) [32] distills step-by-step reasoning from a large teacher into a small LM, whose rationales are then encoded as features (as recalled).
2. **Model compression of the LLM recommender itself.**
   - SLMRec [31] studies how much of a large LLM-based sequential recommender's depth is needed, and distills into a much smaller model (as recalled).
   - "Instruction Distillation Makes LLMs Efficient Zero-shot Rankers" [33] distills a pairwise or listwise ranking teacher into a pointwise FLAN-T5-XL (3B) student.
   - POD ("Prompt Distillation for Efficient LLM-based Recommendation", CIKM 2023) distills discrete prompts into continuous prompt vectors on a T5-small (60M) backbone, which shortens inputs [34].
3. **Foundation teacher, many students (industry).** Meta's GEM is described as trained at LLM scale and is used offline as a teacher whose knowledge is transferred to serving models (Chapters 16 and 19) [62]. Netflix's foundation model is a large, centrally trained model whose embeddings and fine-tuned variants feed many downstream systems [51]. Neither is a text LLM. Both borrow the LLM operating model of pre-training once and distilling or fine-tuning many times.

```
                    TEACHER (offline, expensive)
     LLM ranker / 360Brew-like / LLM-scale FM      <- trained on text + interactions
         |  soft labels, rankings, rationales, embeddings
         v
  +-------------+   +-------------------+   +----------------------+
  | small LM    |   | SASRec / two-tower|   | CTR model features   |
  | (1-3B)      |   | student           |   | (LLM-derived dense)  |
  +-------------+   +-------------------+   +----------------------+
         \___________________ online serving at full QPS ______________/
```

---

## 29.8 Industry deployments: what is publicly documented

| Company | System | LLM role | Stage | Model / scale (public) | Cost strategy | Ref |
|---|---|---|---|---|---|---|
| LinkedIn | Feed LLM retrieval (2025) | Dual-encoder embeddings | Retrieval | LLaMA 3; ~2k of hundreds of millions; few ms; thousands QPS (approx.) | Nearline item embedding, ANN | [44] |
| LinkedIn | 360Brew (2025) | Text ranker / foundation | Ranking | 150B MoE (Mixtral 8x22B); 30+ tasks | Pre-production; likely teacher (inference) | [43] |
| LinkedIn + UVA | CLLM4Rec (2024) | Collaborative LLM | Gen. rec | GPT-2 with user and item tokens | Freeze LM, train ID tokens | [9] |
| YouTube | PLUM (2025) | SID-speaking LLM | Retrieval | Gemini-family LLM, CPT + FT | Beam decode SIDs; small item tables | [46] |
| Spotify | AI DJ (2023) | Narration | Presentation | OpenAI LLM + Sonantic TTS | Recommender selects; LLM frames | [49] |
| Spotify | Text2Tracks (2025) | Prompt -> track IDs | Retrieval (research) | Fine-tuned enc-dec LLM; CF semantic IDs | Short IDs, no title resolution | [50] |
| Netflix | Foundation model (2025) | LLM-style FM (not text LLM) | Representation | Hundreds of billions of interactions; 300M+ members | Central training, downstream reuse | [51] |
| Kuaishou | OneRec-Think / OpenOneRec (2025) | LLM backbone (Qwen3) + SIDs | End-to-end gen. | 1.7B / 8B; ~33B-130B tokens | RL post-training; reasoning restore | [52][53] |
| ByteDance | HLLM (2024) | Item and user LLMs | Seq. rec | 1B / 7B | Item embeddings cacheable | [55] |
| Xiaohongshu | NoteLLM (2024) | Item embedding LLM | I2I retrieval | LLM compressed-token embedding | Offline item embeddings | [56] |
| Huawei (as recalled) | KAR (2023) | Knowledge generation | Features | ChatGPT-class LLM + adaptor | Precompute and cache knowledge | [17] |
| Microsoft | RecAI / InteRecAgent | Agent and tools | Conversational | GPT-4 class or RecLlama 7B (as recalled) | Plan-first; candidate bus | [21][22] |

### 29.8.1 Lessons visible across deployments

- **LLM compute lives where it can be cached.** LinkedIn embeds posts nearline, Spotify keeps the LLM at narration, KAR precomputes knowledge, and PLUM spends compute on retrieval but replaces large embedding tables. The most expensive design (360Brew) is the one not yet described as serving at Feed scale.
- **Cold start is the most consistent win.** It shows up in LinkedIn's retrieval (new members), 360Brew (5-interaction histories), TALLRec (16-256 samples) and Netflix (metadata gating for new titles).
- **IDs remain essential.** Systems that drop collaborative signals entirely underperform. Successful designs inject CF through semantic IDs (PLUM, Text2Tracks, LC-Rec), CF embeddings (LLaRA, CLLM4Rec) or CF knowledge in prompts (Knowledge Plugin).

---

## 29.9 Deployments we could not verify in this session

The topic brief lists **Amazon Rufus** (Amazon's generative shopping assistant), **Instacart** (LLM use in search and discovery), and **YouTube's Gemini-powered consumer features**. Primary sources for these could not be fetched from the research environment, and no companion chapter of this Atlas documents them. To avoid stating unverified numbers, this chapter gives no architecture or scale figures for them. Structurally, public descriptions of shopping assistants match the agentic pattern of Section 29.6. A conversational LLM grounds its answers in catalog, review and Q&A data through retrieval, and product suggestions come from the existing search and recommendation stack. The serving challenge is LLM inference at peak shopping traffic, not model quality alone. Readers should consult the companies' engineering blogs directly. For YouTube, PLUM [46] is the verified primary source on Gemini-family models in recommendation.

---

## 29.10 Cost and latency trade-offs

### 29.10.1 Where the cost lands

```
 cost per day  ~=  (#requests) x (#candidates scored by LLM) x (tokens per candidate)
                   x (cost per token)                         <- per-request LLM ranker
               vs. (#items + #users refreshed) x (tokens) x (cost per token)
                                                              <- offline feature/embedding
```

| Pattern | Scales with | Typical latency impact | Freshness | Main risk |
|---|---|---|---|---|
| Offline LLM features (KAR, LLMRec) | catalog size x refresh rate | none online | hours to days | stale or hallucinated knowledge |
| Nearline LLM embeddings (LinkedIn, NoteLLM) | new items + user updates | ANN only | minutes | embedding drift across model versions |
| LLM generative retrieval (PLUM, OneRec) | requests x beam x decode steps | high; needs TPU/GPU serving | per request | invalid IDs; compute cost |
| LLM pointwise ranker (360Brew, TALLRec) | requests x candidates x prompt length | very high | per request | QPS infeasible beyond small candidate sets |
| LLM listwise reranker (LLMRank, LlamaRec) | requests x (prompt with ~20 candidates) | high; one call | per request | position bias; non-determinism |
| LLM agent (InteRecAgent, DJ) | conversational turns | seconds acceptable | per turn | tool errors; prompt injection |
| Distilled student (DLLM2Rec, SLIM) | training only | none extra | retrain cadence | teacher bias inherited |

### 29.10.2 Engineering tricks that recur

1. **Score with logits, not generation.** TALLRec ("Yes" probability), 360Brew (answer-token logits) and LlamaRec (letter-index verbalizer) all turn ranking into one forward pass with no autoregressive decoding [6][43][10].
2. **Shrink the candidate set first.** LLM rankers run on about 10-100 candidates from a conventional retriever. LLMRank found zero-shot ability strongest on candidates from multiple diverse retrievers [7].
3. **Share the user prefix.** When every candidate shares the user history prefix, KV caching of the prefix amortizes it. This is the same idea as the DCAT and M-FALCON-style tricks in Chapter 6, and as KV-cache serving of user sequences at ByteDance (Chapter 20).
4. **Constrain decoding.** Prefix tries or logits processors over valid semantic IDs prevent invalid items and reduce beam waste [54]. Track a ValidRatio-style metric [12].
5. **Quantize numbers into tokens.** Raw digits are poorly modelled, and bucketed tokens work better [44].
6. **Parameter-efficient tuning.** LoRA (TALLRec, LLaRA) or QLoRA (LlamaRec) keeps per-domain adapters cheap. CLLM4Rec freezes the LM and trains only ID embeddings [9].
7. **Plan once.** Agents that plan once and execute tools avoid multi-round ReAct latency. Fine-tuned small planners (RecLlama-style) replace GPT-4-class planners [22].
8. **Restore general ability after domain tuning.** OpenOneRec's on-policy distillation step [53] keeps the LLM useful for explanation and dialogue.
9. **Precompute explanations.** Generate LLM text per (item, segment) offline, filter it with quality and safety classifiers, and serve it from cache (Spotify pattern, Chapter 25).
10. **Keep a guardrail layer.** LLM outputs pass through the same business-rule, diversity and safety re-rankers as any model (Chapter 15). FaiRLLM-style audits check sensitivity to user attributes [16].

---

## 29.11 Key takeaways for system designers

1. **Decide which side of the cache line the LLM lives on.** Offline and nearline LLM use (features, embeddings, semantic IDs, teacher labels) is cheap to adopt and leaves online latency unchanged. Per-request LLM ranking needs a very small candidate set or a high-value, low-QPS surface.
2. **Start with LLM features and embeddings, not LLM ranking.** KAR, LLMRec, NoteLLM and LinkedIn's LLaMA-3 retriever show measurable gains with no change to the serving stack.
3. **Inject collaborative signal.** Pure-text LLMs underperform ID models on warm items. Use semantic IDs, CF embeddings in prompts, or added ID tokens.
4. **Invest in the item tokenizer.** PLUM, Text2Tracks, LC-Rec and OneRec all depend on good, collision-free, refreshable item codes. Treat the tokenizer as infrastructure with versioning.
5. **Exploit cold start.** The most consistent LLM wins are for new users, new items and new surfaces. Route traffic accordingly, for example LLM-heavy paths for sparse-history users.
6. **Use the LLM's strengths at the edges.** Intent parsing, conversation, explanation and narration (AI DJ, InteRecAgent) add product value while the recommender keeps control of selection.
7. **Distill.** Train the expensive model once and serve students: sequential models (DLLM2Rec), small LMs (SLIM, SLMRec) or feature producers. This is the same pattern as industrial foundation models at Meta and Netflix.
8. **Measure LLM-specific failure modes.** Track position bias, popularity bias, invalid-output rate, attribute sensitivity (fairness) and non-determinism across calls. Bootstrapped permutations cost money, so budget for them.
9. **Plan for evaluation with simulated users, but do not trust it alone.** LLM-simulated users (iEvaLM, Agent4Rec) are good for conversational iteration. Online A/B tests remain the arbiter (Chapter 12).
10. **Expect convergence.** By 2025-2026, generative recommenders (Chapter 6) and LLMs are merging: Qwen3-backbone OneRec, Gemini-backbone PLUM and Mixtral-backbone 360Brew. The open problems are serving economics and freshness, not whether LLMs can recommend.

---

## References

1. Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang (Rutgers). "Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)." RecSys 2022. https://arxiv.org/abs/2203.13366
2. Shijie Geng et al. "P5" official repository (README, prompt templates `src/all_amazon_templates.py`, checkpoints). GitHub, 2022. https://github.com/jeykigung/P5
3. Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang. "How to Index Item IDs for Recommendation Foundation Models." 2023. https://arxiv.org/pdf/2305.06569.pdf
4. Shuyuan Xu, Wenyue Hua, Yongfeng Zhang (agiresearch). "OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems." Paper and repository, 2023-2024. https://arxiv.org/pdf/2306.11134.pdf ; https://github.com/agiresearch/OpenP5
5. Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He. "TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation." RecSys 2023. https://arxiv.org/abs/2305.00447
6. TALLRec official repository (README, main results table, training scripts). GitHub, 2023. https://github.com/SAI990323/TALLRec
7. Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, Wayne Xin Zhao. "Large Language Models are Zero-Shot Rankers for Recommender Systems." ECIR 2024. https://arxiv.org/abs/2305.08845 ; repository: https://github.com/RUCAIBox/LLMRank
8. Bowen Zheng et al. (RUC AI Box). "Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation" (LC-Rec). ICDE 2024. https://ieeexplore.ieee.org/abstract/document/10597986/ ; repository: https://github.com/RUCAIBox/LC-Rec
9. Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, Jundong Li (UVA and LinkedIn). "Collaborative Large Language Model for Recommender Systems" (CLLM4Rec). WWW 2024. https://arxiv.org/abs/2311.01343 ; repository: https://github.com/yaochenzhu/LLM4Rec
10. Zhenrui Yue et al. "LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking." PGAI@CIKM 2023. https://arxiv.org/abs/2311.02089
11. Keqin Bao et al. "A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems" (BIGRec). 2023. https://arxiv.org/abs/2308.08434
12. Jiayi Liao et al. "LLaRA: Large Language-Recommendation Assistant." SIGIR 2024. https://arxiv.org/abs/2312.02445 ; repository (updated results, 2024.7): https://github.com/ljy0ustc/LLaRA
13. Junjie Zhang et al. "Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach" (InstructRec). 2023. https://arxiv.org/abs/2305.07001
14. Junling Liu et al. "Is ChatGPT a Good Recommender? A Preliminary Study." 2023. https://arxiv.org/abs/2304.10149
15. Sunhao Dai et al. "Uncovering ChatGPT's Capabilities in Recommender Systems." RecSys 2023. https://arxiv.org/abs/2305.02182
16. Jizhi Zhang et al. "Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation" (FaiRLLM). RecSys 2023. https://arxiv.org/abs/2305.07609
17. Yunjia Xi et al. (SJTU and Huawei). "Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models" (KAR). 2023. https://arxiv.org/abs/2306.10933
18. Wei Wei, Xubin Ren, Jiabin Tang, Qingyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang (HKU and Baidu). "LLMRec: Large Language Models with Graph Augmentation for Recommendation." WSDM 2024. https://arxiv.org/abs/2311.00423 ; repository: https://github.com/HKUDS/LLMRec
19. Xubin Ren et al. "Representation Learning with Large Language Models for Recommendation" (RLMRec). WWW 2024. https://arxiv.org/abs/2310.15950
20. Qijiong Liu et al. "ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models." WSDM 2024. https://arxiv.org/abs/2305.06566
21. Microsoft. "RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems" (InteRecAgent, Knowledge Plugin, RecLM-emb, RecLM-gen, RecExplainer, RecLM-eval). GitHub, 2023-2024. https://github.com/microsoft/RecAI
22. Xu Huang, Jianxun Lian, et al. (Microsoft). "Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations" (InteRecAgent). 2023. https://arxiv.org/abs/2308.16505
23. Yuxuan Lei, Jianxun Lian, et al. (Microsoft). "Aligning Language Models for Versatile Text-based Item Retrieval" (RecLM-emb). 2024. https://arxiv.org/abs/2402.18899
24. Yunfan Gao et al. "Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System." 2023. https://arxiv.org/abs/2303.14524
25. Yancheng Wang et al. "RecMind: Large Language Model Powered Agent For Recommendation." NAACL 2024 Findings. https://arxiv.org/abs/2308.14296
26. An Zhang et al. "On Generative Agents in Recommendation" (Agent4Rec). 2023. https://arxiv.org/abs/2310.10108
27. Junjie Zhang et al. "AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems." WWW 2024. https://arxiv.org/abs/2310.09233
28. Xiaolei Wang et al. "Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models." EMNLP 2023. https://arxiv.org/abs/2305.13112
29. Xiaolei Wang et al. "Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning" (UniCRS). KDD 2022. https://arxiv.org/abs/2206.09363
30. Yu Cui et al. "Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Model" (DLLM2Rec). 2024. https://arxiv.org/abs/2405.00338
31. Wujiang Xu et al. "SLMRec: Empowering Small Language Models for Sequential Recommendation." 2024. https://arxiv.org/abs/2405.17890
32. Yuling Wang et al. "Can Small Language Models be Good Reasoners for Sequential Recommendation?" (SLIM). WWW 2024. https://arxiv.org/abs/2403.04260
33. Weiwei Sun et al. "Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers." 2023. https://arxiv.org/abs/2311.01555
34. Lei Li, Yongfeng Zhang, Li Chen. "Prompt Distillation for Efficient LLM-based Recommendation" (POD). CIKM 2023. https://lileipisces.github.io/files/CIKM23-POD-paper.pdf
35. Jian Jia et al. (Kuaishou). "Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application" (LEARN). 2024. https://arxiv.org/abs/2405.03988
36. "A Decoding Acceleration Framework for Industrial Deployable LLM-based Recommender Systems." 2024. https://arxiv.org/abs/2408.05676
37. "Efficient and Deployable Knowledge Infusion for Open-World Recommendations via Large Language Models." 2024. https://www.arxiv.org/abs/2408.10520
38. "Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph." 2024. https://arxiv.org/abs/2402.13750
39. "Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models." 2024. https://arxiv.org/abs/2402.08670
40. Jianghao Lin et al. "How Can Recommender Systems Benefit from Large Language Models: A Survey." ACM TOIS, 2024; paper index repository. https://dl.acm.org/doi/10.1145/3678004 ; https://github.com/CHIANGEL/Awesome-LLM-for-RecSys
41. Likang Wu et al. "A Survey on Large Language Models for Recommendation." 2023; paper index repository. https://arxiv.org/abs/2305.19860 ; https://github.com/WLiK/LLM4Rec-Awesome-Papers
42. Lei Li et al. "Large Language Models for Generative Recommendation: A Survey and Visionary Discussions." 2023. https://arxiv.org/abs/2309.01157
43. 360Brew Team (H. Firooz, M. Sanjabi, et al.), LinkedIn. "360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation." 2025. https://arxiv.org/abs/2501.16450 ; report and README: https://github.com/LLMPersonalization/360Brew-V1
44. LinkedIn. "Large Scale Retrieval for the LinkedIn Feed using Causal Language Models." arXiv 2510.14223, 2025 (AAAI 2026). https://arxiv.org/abs/2510.14223
45. LinkedIn. "Liger Kernel: Efficient Triton Kernels for LLM Training." GitHub, 2024-2025. https://github.com/linkedin/Liger-Kernel
46. Google / YouTube. "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations." 2025. https://arxiv.org/abs/2510.07784
47. Shashank Rajput, Nikhil Mehta, Anima Singh, et al. (Google). "Recommender Systems with Generative Retrieval" (TIGER). NeurIPS 2023. https://arxiv.org/abs/2305.05065
48. Konstantina Christakopoulou, Alberto Lalama, Cj Adams, et al. (Google). "Large Language Models for User Interest Journeys." 2023. https://arxiv.org/abs/2305.15498
49. Spotify Newsroom. "Spotify Debuts a New AI DJ, Right in Your Pocket." Spotify, 2023. https://newsroom.spotify.com/2023-02-22/spotify-debuts-a-new-ai-dj-right-in-your-pocket/ (AI Playlist announcements: https://newsroom.spotify.com/)
50. Enrico Palumbo, Gustavo Penha, Andreas Damianou, et al. (Spotify). "Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval." 2025. https://arxiv.org/abs/2503.24193
51. Ko-Jen Hsiao, Yesu Feng, Sudarshan Lamkhede (Netflix). "Foundation Model for Personalized Recommendation." Netflix Technology Blog, 2025. https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39
52. Kuaishou OneRec Team. "OneRec-Think: In-Text Reasoning for Generative Recommendation." 2025. https://arxiv.org/abs/2510.11639
53. Kuaishou OneRec Team. "OpenOneRec Technical Report" and repository. 2025. https://arxiv.org/abs/2512.24762 ; https://github.com/Kuaishou-OneRec/OpenOneRec
54. Kong et al. "MiniOneRec" (open-source generative recommendation framework). 2025. https://arxiv.org/abs/2510.24431 ; https://github.com/AkaliKong/MiniOneRec
55. ByteDance. "HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling." 2024. https://arxiv.org/abs/2409.12740 ; https://github.com/bytedance/HLLM
56. Chao Zhang et al. (Xiaohongshu). "NoteLLM: A Retrievable Large Language Model for Note Recommendation." WWW 2024. https://arxiv.org/abs/2403.01744
57. Ju, C. et al. (Snap). "Generative Recommendation with Semantic IDs: A Practitioner's Handbook" (GRID). CIKM 2025. https://arxiv.org/abs/2507.22224 ; https://github.com/snap-research/GRID
58. Microsoft RecAI. "Knowledge Plugins: Enhancing Large Language Models for Domain-Specific Recommendations" (DOKE) repository. GitHub, 2023-2024. https://github.com/microsoft/RecAI/tree/main/Knowledge_Plugin
59. Microsoft RecAI. "InteRecAgent" repository README. GitHub, 2023-2024. https://github.com/microsoft/RecAI/tree/main/InteRecAgent
60. "Large Language Models Enhanced Collaborative Filtering" (LLM-CF). 2024. https://arxiv.org/abs/2403.17688
61. Wenyue Hua et al. "LLM-RecSys-ID" repository (code for item-ID indexing in recommendation foundation models). GitHub, 2023. https://github.com/Wenyueh/LLM-RecSys-ID
62. Engineering at Meta. "GEM Training: How Meta Doubled the Efficiency of Its LLM-Scale Ads Foundation Model." Meta, August 2026. https://engineering.fb.com/2026/08/03/ml-applications/training-gem-at-llm-scale-meta-ads-recommendation-foundation-model/
