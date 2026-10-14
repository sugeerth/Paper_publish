# 12. Design Harmful Content Detection / Content Moderation (Facebook/Instagram Integrity)

> Prompt as usually given: "Users post text, images and videos on a social platform. Design
> a system that detects harmful posts (violence, nudity, hate speech, self-harm, spam,
> misinformation) and removes or demotes them." Variants: comment toxicity, hateful memes
> (image + text), message moderation in DMs or dating apps, LLM guardrails, spam/abusive
> accounts (actor-level integrity).
>
> Sourcing note: at review time (Oct 2026) web search was exhausted and ai.meta.com,
> about.fb.com, arxiv.org and bytebytego.com did not resolve from the build sandbox; only
> GitHub was reachable. Tags: (fetched) = read for this chapter; (repo-link) = URL appears
> verbatim in a fetched index (ByteByteGo's ml-bytebytego ch. 5 reference list [2] or the
> Engineer1999 list [11]); (ch-ref) = verified in another chapter; (known) = canonical
> paper by arXiv ID. Only "built it" company tags with a source are kept.

## Company tags

| Company | Evidence | Notes | Source |
|---|---|---|---|
| Meta (Facebook, Instagram) | Built it: WPIE, XLM-R + Linformer hate-speech models, Few-Shot Learner, TIES | The canonical "house problem" for Meta MLE / integrity loops; ByteByteGo ch. 5 frames it on a Facebook-like platform | [1][3][12][13][14][15][16][18] |
| Pinterest | Built it: ML for misinformation, hate speech and self-harm; comment-health models | Pin-level multimodal classifiers using PinSage/visual/text embeddings | [20][21] |
| LinkedIn | Built it: viral spam detection; isolation forest for abuse | Spam / abusive actor framing | [22][23] |
| Bumble, Nextdoor, Whatnot, Slack | Built it: rude-message detection (multilingual DMs); harmful-comment prediction; GenAI marketplace-spam T&S; invite-spam blocking | Variant systems for DM / comment / marketplace prompts (as indexed in [11]) | [24][25][26][27] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| ByteByteGo (Aminian & Xu, MLSDI), ch. 5 "Harmful Content Detection" | Fullest treatment: multimodal early fusion, multi-task heads, BCE per task, prevalence / proactive rate, demotion vs violation services | [1][2] |
| alirezadir Machine-Learning-Interviews, "mlsd-harmful-content" | Condensed notes of the book chapter: 500M posts/day (~10K annotated), DistilmBERT + CLIP + VideoMoCo, gradient blending, focal loss; listed under "Harmful content / Spam detection" in the MLSD index | [3][4] |
| Educative "Grokking the ML Interview" / Khang Pham MLSD | Generic classification + ranking templates (no dedicated chapter verified); reuse the classifier pattern | [5][6] |
| Exponent, Hello Interview, IGotAnOffer, Patrick Halina | MLSD delivery frameworks and Meta MLE loop guides; no dedicated harmful-content chapter verified | [7][8][9][10] |
| Engineer1999 curated case studies | Index of real T&S / spam / fraud systems from 20+ company blogs | [11] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Input: a post (text, image(s), video, links, author). Output: per-category harm probabilities, action (remove / demote / send to review / allow), and a user-visible reason | [1][3] |
| Harm categories | Violence, nudity, hate speech (+ self-harm, spam, misinformation, bullying in variants) | [1][3][20] |
| Out of scope (book) | Harmful actors (fake / inauthentic accounts) and harmful acts; added as follow-ups | [3][37] |
| Explainability | Must tell the user which policy was violated; appeals flow | [1][3] |
| Volume | ~500M new posts/day (~6K posts/s avg, ~15-20K/s peak) | [3] (book) |
| Labels | ~10K posts hand-annotated (eval set); user reports ("natural labels") for the training set | [3] (book) |
| Languages | 100+ languages; multilingual encoder required | [14][15] |
| Latency | Varies by category: blocking classes (CSAM, terror) pre-publish p99 < ~200-500 ms; most classes near-real-time (seconds to minutes) after publish; re-scoring on virality | [3] + typical assumption |
| Precision / recall | Auto-remove needs very high precision (~95%+); demotion and review queues trade precision for recall | typical assumption |
| Adversarial | Evasion (text in image, leetspeak, re-encoded media), fast drift of new harm types | [16][17] |
| Known-bad media | Re-uploads of known CSAM / terror / NCII media matched by perceptual hash before any model | [36] |

## Diagram 1: end-to-end serving architecture

**Diagram 1. Online moderation path (post -> hash match -> cheap filters -> multimodal
classifier -> policy engine -> remove / demote / review / allow)**

```
 +-----------+ create post +-----------------+ publish event +-----------------+
 |  Client   |------------>|  Post service   |-------------->|  Event bus      |
 | (FB / IG) |             | (write to store)|               |  (Kafka-like)   |
 +-----------+             +--------+--------+               +--------+--------+
                                    |                                 |
     SYNC pre-publish check         |              ASYNC post-publish |
     (CSAM, terror, banned media)   |              (all other classes)|
                                    +----------------+----------------+
                                                     |
                                                     v
 +---------------------------------------------------------------------------------+
 |  STAGE 0: HASH MATCH  (PDQ / TMK+PDQF / vPDQ, HMA-style banned-media banks)     |
 |  exact / near-dup of known violating media  --> auto-remove, no model needed    |
 +---------------------------------------+-----------------------------------------+
                                         | no match
                                         v
 +---------------------------------------------------------------------------------+
 |  STAGE 1: CHEAP FILTERS  keyword/regex lists, URL blocklists, spam heuristics,  |
 |  author-risk score lookup  --> drop obvious spam, flag high-risk authors        |
 +---------------------------------------+-----------------------------------------+
                                         |
          +------------------------------+------------------------------+
          | fetch features                                              | raw media
          v                                                             v
 +--------------------+                                     +------------------------+
 |  Feature store     |  author history, report counts,     |  Content encoders      |
 |  (online KV)       |  engagement velocity, device, ctx   |  text+OCR / image /    |
 +---------+----------+                                     |  video+ASR (GPU,       |
           |                                                |  cached by media hash) |
           |                                                +-----------+------------+
           |                features + embeddings                       |
           +------------------------------+-----------------------------+
                                          v
 +---------------------------------------------------------------------------------+
 |  STAGE 2: MULTIMODAL MULTI-TASK CLASSIFIER  (model store: versioned, shadowed)  |
 |  outputs P(violence), P(nudity), P(hate), P(self-harm), P(spam), ...            |
 +---------------------------------------+-----------------------------------------+
                                         | calibrated scores per category
                                         v
 +---------------------------------------------------------------------------------+
 |  STAGE 3: POLICY / DECISION ENGINE  per-category thresholds, market rules,      |
 |  severity x predicted reach (expected views) prioritisation                     |
 +--------+-------------------+-------------------+-------------------+------------+
          |                   |                   |                   |
    p >= t_high     t_mid <= p < t_high   uncertain / reach   p < t_low
          |                   |                   |                   |
          v                   v                   v                   v
 +-----------------+ +-----------------+ +-----------------+ +-----------------+
 | Violation svc   | | Demotion svc    | | Human review    | | Allow           |
 | remove, notify, | | down-rank in    | | queue, ranked   | | (still logged   |
 | reason + appeal | | feed / no recs  | | severity x views| |  for audits)    |
 +--------+--------+ +--------+--------+ +--------+--------+ +--------+--------+
          |                   |                   | labels            |
          +-------------------+---------+---------+-------------------+
                                        v
                     +-------------------------------------+
                     | Logging: scores, actions, reports,  |
                     | appeals, reviewer labels, views     |--> offline pipeline (D2)
                     +-------------------------------------+
   re-score trigger: new user reports / view-velocity spike --> back to Stage 2
```

## Diagram 2: offline training & data pipeline

**Diagram 2. Labels -> features -> multi-task training -> evaluation -> registry ->
shadow/canary deploy, with reviewer and appeal feedback loops**

```
 +----------------+   +----------------+   +----------------+   +-----------------+
 | Post store     |   | User reports   |   | Reviewer       |   | Appeals         |
 | (text, media,  |   | (noisy natural |   | decisions      |   | outcomes        |
 |  author, ctx)  |   |  labels)       |   | (gold labels)  |   | (label fixes)   |
 +-------+--------+   +-------+--------+   +-------+--------+   +--------+--------+
         |                    |                    |                     |
         +--------------------+---------+----------+---------------------+
                                        v
               +----------------------------------------------------+
               | Label join + cleaning: dedup by media hash,        |
               | reviewer agreement, label smoothing, per-category  |
               | multi-label targets, time-split train/val/test     |
               +-------------------------+--------------------------+
                                         |
          +------------------------------+------------------------------+
          | active learning: uncertain / | hard-negative sampling,      | eval set
          | high-reach posts             | class reweighting            | (hand labels)
          v                              v                              v
 +------------------+      +----------------------------------+  +----------------+
 | Annotation tool  |----->| Feature pipeline (batch+stream)  |  | Held-out gold  |
 | (policy guide,   |labels| text+OCR tokens, image/video     |  | eval set, per  |
 |  multi-rater)    |      | frames+ASR, author + engagement  |  | category/lang  |
 +------------------+      +----------------+-----------------+  +-------+--------+
                                            |   --> Feature store        |
                                            |   (online/offline parity)  |
                                            v                            |
               +----------------------------------------------------+    |
               | Training: pretrain encoders (self-supervised,      |    |
               | XLM-R / CLIP-like) -> fine-tune multi-task fusion  |    |
               | heads (sum of BCE / focal losses)                  |    |
               +-------------------------+--------------------------+    |
                                         v                               |
               +----------------------------------------------------+    |
               | Offline eval: PR-AUC, recall @ precision >= P0     |<---+
               | per category / language / market; slice fairness;  |
               | adversarial + red-team set; calibration            |
               +-------------------------+--------------------------+
                                         | pass gates
                                         v
 +------------------+   +---------------------+   +----------------------------------+
 | Model registry   |-->| Shadow mode: score  |-->| Canary / A-B on % of traffic,    |
 | (+ thresholds)   |   | live, no actions    |   | then full rollout; thresholds    |
 +------------------+   +---------------------+   | re-tuned per market              |
                                                  +-----------------+----------------+
                                                                    |
       feedback loop: new actions -> new reports / reviews / appeals|
       <------------------------------------------------------------+
       retrain cadence: daily-weekly; emergency few-shot update for new harm types
```

## Diagram 3: model architecture

**Diagram 3. Early-fusion multimodal multi-task classifier (ByteByteGo / book design,
with WPIE-style whole-post embedding) [1][3][13][29][31][33][34]**

```
  INPUTS
 +-------------+  +---------------+  +---------------+  +------------+  +-----------+
 | Post text   |  | Image(s)      |  | Video         |  | Author     |  | Engagement|
 | + OCR text  |  |               |  | (frames+audio)|  | features   |  | + context |
 +------+------+  +-------+-------+  +-------+-------+  +-----+------+  +-----+-----+
        |                 |                  |                |              |
        v                 v                  v                v              v
 +-------------+  +---------------+  +---------------+  +------------+  +-----------+
 | Multilingual|  | Image encoder |  | Video encoder |  | Embeddings |  | Normalise,|
 | text enc.   |  | (CLIP-ViT /   |  | (VideoMoCo /  |  | + numeric  |  | bucketize |
 | DistilmBERT/|  |  SimCLR)      |  |  frame pool)  |  | (age, prior|  | (reports, |
 | XLM-R       |  |               |  |               |  | violations)|  | shares)   |
 +------+------+  +-------+-------+  +-------+-------+  +-----+------+  +-----+-----+
        | 768-d           | 512-d            | 512-d          |              |
        +-----------------+--------+---------+----------------+--------------+
                                   v
                   +----------------------------------+
                   |  EARLY FUSION: concat / cross-   |
                   |  attention transformer -> whole- |
                   |  post embedding (WPIE-style)     |
                   +----------------+-----------------+
                                    v
                   +----------------------------------+
                   |  Shared MLP layers               |
                   +---+--------+--------+--------+---+
                       |        |        |        |
                       v        v        v        v
                  +-------+ +-------+ +-------+ +-------+
                  |Violen.| |Nudity | | Hate  | | Spam  |   task-specific heads
                  | head  | | head  | | head  | | head  |   (sigmoid each)
                  +---+---+ +---+---+ +---+---+ +---+---+
                      v         v         v         v
                  P(viol.)  P(nude)   P(hate)   P(spam)   --> policy engine (D1)

  LOSS = sum_k w_k * BCE_k  (or focal loss); gradient blending to stop one
  modality (usually image) from dominating / overfitting
```

## Diagram 4: real company system (Meta integrity stack)

**Diagram 4. Meta-style layered integrity system: WPIE + XLM-R/Linformer + Few-Shot
Learner + TIES (actor/interaction signals); borderline content is demoted, not removed
[13][14][16][17][18][19][28]**

```
 +-------------------+     +--------------------------------------------------------+
 | Policy team writes|     | Few-Shot Learner (FSL): large multilingual multimodal  |
 | new rule (natural |---->| model; pretrain on billions of generic text/images ->  |
 | language + a few  |     | policy-specific training -> few-shot/zero-shot entail- |
 | examples)         |     | ment of "post vs policy text"; weeks vs months to ship |
 +-------------------+     +---------------------------+----------------------------+
                                                       | new-policy classifier
                                                       v
 +-----------------+   +----------------------+   +--------------------------------+
 | Post: text,     |-->| WPIE: whole-post     |-->| Integrity classifiers per      |
 | image, video,   |   | integrity embedding  |   | policy (hate, violence, nudity,|
 | comments        |   | (multimodal, pre-    |   | misinfo...) on top of shared   |
 +-----------------+   | trained once, reused)|   | embeddings                     |
                       +----------------------+   +---------------+----------------+
 +-----------------+   +----------------------+                   |
 | Text in 100+    |-->| XLM-R (cross-lingual)|------------------>|
 | languages       |   | + Linformer (linear  |                   |
 +-----------------+   |   attention, cheap)  |                   |
                       +----------------------+                   |
 +-----------------+   +----------------------+                   |
 | Account + inter-|-->| TIES: temporal       |------------------>|
 | action sequences|   | interaction embedding|   actor-risk      |
 +-----------------+   +----------------------+                   v
                                                  +--------------------------------+
                                                  | Enforcement: remove / reduce   |
                                                  | distribution (borderline) /    |
                                                  | review; RL-tuned (RIO) to      |
                                                  | optimise prevalence directly   |
                                                  +--------------------------------+
```

## Data & labels

| Source | Label type | Pros | Cons |
|---|---|---|---|
| Human reviewers (contractors) | Gold, per-policy multi-label | Accurate, policy-aware; book uses them for the eval set [3] | Slow, costly (~10K labelled), reviewer wellbeing, disagreement; review tooling at scale [38] |
| User reports | Natural, noisy positive | Huge volume, fast | Brigading, bias, many false reports; few negatives |
| Appeals overturned | Label correction | Fixes false positives | Sparse, delayed |
| Hash banks (known violating media) | Exact / near-dup positive (PDQ, TMK+PDQF, vPDQ; shared via ThreatExchange / HMA) [36] | Zero-shot removal of re-uploads; cross-company sharing | Only known content |
| Synthetic / LLM-generated | Augmented positives | New harm types, rare languages | Distribution shift |
| Split strategy | Time-based split; dedup by media hash across splits | Avoids leakage of near-dup memes | - |

## Features

| Group | Examples | Encoding |
|---|---|---|
| Text | Caption, hashtags, OCR from images, ASR from video, comments | Multilingual transformer (DistilmBERT / XLM-R) [15][34][39]; comments aggregated |
| Image | Pixels, detected objects, faces | CLIP / SimCLR embedding [31][32] |
| Video | Frames, audio | VideoMoCo / frame-pooled image encoder + audio model [33] |
| Author | Account age, followers, prior violations, report rate, profanity rate | Numeric, bucketized; embedding of author |
| Interactions | Likes, shares, comments, reports, view velocity | Counts, ratios, time-decayed |
| Context | Time of day, device, country/language, group/page type | One-hot / embedding |
| Graph / actor | Interaction sequences, cluster membership | TIES-style temporal embeddings [18] |

## Model choices (baseline -> advanced)

| Option | Description | Pros | Cons |
|---|---|---|---|
| Rules + keyword lists | Regex, blocklists, hash match | Instant, explainable | Low recall, easy to evade |
| Late fusion | One model per modality, combine scores | Simple, modular, debuggable | Misses harm that only appears in combination (hateful memes [35][40]) |
| Early fusion (book choice) | Concatenate/attend modality embeddings, one classifier | Captures cross-modal harm | Harder to train, needs more data |
| Single binary classifier | "harmful vs not" | Simple | No category -> no explanation; hard to set per-class thresholds |
| One model per category | Separate binary models | Independent tuning | N models to train/serve/maintain |
| Multi-label single head | One output layer, K sigmoids | One model | Shared features may not fit all classes |
| Multi-task (book choice) | Shared trunk + per-task heads | One model, data sharing helps rare classes, explainable | Task interference; loss weighting |
| Foundation / few-shot (FSL) | Policy-text entailment, LLM classifier [16][17] | Fast adaptation to new policies | Expensive; needs distillation for serving |
| Efficient transformer | XLM-R with Linformer linear attention [14][28] | Serve large multilingual model on every post | Approximation; engineering cost |
| LLM-as-reviewer | LLM judges borderline items with policy prompt | Explainable, flexible | Cost, latency, jailbreak / prompt-injection |

## Loss

| Loss | When |
|---|---|
| Sum of per-task BCE | Default multi-task multi-label |
| Weighted BCE / class weights | Severe imbalance (prevalence often < 0.1%) |
| Focal loss [30] | Focus on hard, rare positives |
| Gradient blending [29] | One modality dominating or overfitting in multimodal training |
| Label smoothing / noise-robust loss | Noisy user-report labels |

## Offline metrics

| Metric | Why |
|---|---|
| PR-AUC per category | Highly imbalanced data; ROC-AUC over-optimistic |
| Recall @ fixed precision (e.g. P >= 95%) | Matches auto-remove threshold |
| F1 | Single threshold summary |
| ROC-AUC | Ranking quality, threshold-free |
| Per-language / per-market slices | Fairness and coverage across 100+ languages |
| Calibration (ECE) | Thresholds and severity x reach depend on calibrated probabilities |

## Online metrics / A-B

| Metric | Definition |
|---|---|
| Prevalence | Harmful posts (or views) not prevented / all posts (views) [3][13] |
| Harmful impressions | Views of harmful content before action |
| Proactive rate | Detected by system / (system + user reports) [3][14] |
| Valid appeals rate | Successful (reversed) appeals / removals (false positive proxy) [3] |
| User reports per 1K views | Leading indicator of misses |
| Reviewer throughput / queue latency | Ops health |
| Guardrails | Engagement / creator retention of non-violating content (over-enforcement) |
| A-B caveat | Holdouts limited by ethics; rely on shadow scoring + enforcement on small % + interleaved reviewer audits |

## Serving & latency budget

| Stage | Budget (typical) | Notes |
|---|---|---|
| Hash match | < 10 ms | Perceptual hash lookup |
| Rules / cheap filters | < 5 ms | In-process |
| Feature fetch | 10-20 ms | Online KV store |
| Text encoder | 10-30 ms | Distilled / quantized, GPU or CPU |
| Image encoder | 20-50 ms | GPU batch, cached by media hash |
| Video | seconds (async) | Sample frames; score after publish |
| Fusion + heads | < 5 ms | Small MLP |
| Policy engine | < 5 ms | Thresholds per market |
| Total pre-publish (blocking classes) | ~100-200 ms p99 | Else async post-publish within seconds |

## Monitoring

| Signal | Action |
|---|---|
| Score distribution drift per category / language | Alert, re-calibrate thresholds |
| Spike in user reports or appeals | Trigger re-scoring, retraining |
| Reviewer disagreement with model | Error analysis, label audit |
| New adversarial patterns (text-in-image, leetspeak) | Add to red-team set, retrain |
| Latency / GPU saturation | Fall back to cheaper text-only model |
| Prevalence dashboard (sampled reviewer audit) | North-star tracking |

## Failure modes

| Failure | Mitigation |
|---|---|
| Adversarial evasion | OCR/ASR, augmentation, hash near-dup, fast few-shot updates |
| Label noise and reviewer bias | Multi-rater consensus, gold sets, noise-robust losses |
| Over-enforcement on dialects / minority languages | Per-slice metrics, fairness thresholds |
| One modality dominates | Gradient blending, modality dropout |
| New harm type (e.g. new scam, crisis event) | FSL / LLM classifier, rules as stop-gap |
| Viral harmful post before review | Prioritise by severity x predicted reach; re-score on velocity |
| Cold-start languages | Cross-lingual transfer (XLM-R) |
| Feedback loop bias (only reported posts labelled) | Random sampling for audits, active learning |

## Often-missed components (portal / real-system gaps)

| Component | What to add | Source |
|---|---|---|
| Hash-sharing / hasher-matcher-actioner | Maintain banks of PDQ / TMK+PDQF / vPDQ hashes, sync with industry exchanges, act on matches before the classifier | [36] |
| Borderline-content demotion | Content close to (but not violating) policy gets reduced distribution, not removal; separate threshold band | [19] |
| Human review system at scale | Queue routing by policy + language, reviewer tooling, wellbeing (blur, time limits), QA on reviewer accuracy | [38] |
| Inauthentic-behaviour / actor layer | Account-level classifiers (fake accounts, coordinated behaviour) feed author-risk into post scoring | [18][37] |
| Benchmarks for multimodal hate | Hateful Memes dataset and MMF baselines to test early vs late fusion | [35][40] |

## Interviewer follow-ups

- How do you handle severe class imbalance (prevalence well under 1%)?
- Early vs late fusion: when does late fusion fail? (hateful memes)
- Why multi-task over one model per category? How do you weight task losses?
- How do you pick thresholds for remove vs demote vs human review?
- How do you prioritise the human-review queue (severity x reach)?
- How do you measure prevalence without labelling everything?
- How do you handle 100+ languages and low-resource languages?
- A new harm type appears today: how fast can you ship a classifier? (few-shot, rules, LLM)
- How do you defend against adversarial users (text in images, re-encoded video)?
- How do you deal with noisy, biased user-report labels and reviewer bias?
- Pre-publish blocking vs post-publish detection: what goes where and why?
- How would you detect harmful actors (fake accounts) rather than content? (TIES, graph)
- How do you explain decisions to users and handle appeals?
- How do you serve video cheaply at 500M posts/day?
- How would you use an LLM in this system without blowing latency and cost budgets?
- How would you share hashes of known violating media with other platforms, and how do you keep false matches low? [36]
- How do you treat borderline content differently from violating content? [19]
- How do you scale and quality-check the human review operation? [38]

## Portal comparison

| Portal | Framing | Model | Metrics emphasis | Distinctive points |
|---|---|---|---|---|
| ByteByteGo ch. 5 [1][2] | Multimodal post classifier on social media | Early fusion + multi-task heads, BCE sum | PR-AUC offline; prevalence, proactive rate, appeals online | Demotion vs violation services; gradient blending; focal loss |
| alirezadir notes [3] | Same as book, condensed | DistilmBERT + CLIP/SimCLR + VideoMoCo | F1, PR-AUC, ROC-AUC | Lists follow-ups: labeling bias, temporal signals, fake accounts, linear transformers |
| Meta engineering (real system) [13]-[17] | Layered integrity stack | WPIE, XLM-R + Linformer, FSL, TIES | Prevalence, proactive detection rate | Few-shot policy adaptation; RL-tuned enforcement |
| Pinterest engineering [20][21] | Pin / comment health | Embedding-based multi-head classifiers | Precision at action thresholds | Uses existing Pin embeddings; batch + online scoring |
| Generic MLSD guides (Exponent, Hello Interview, IGotAnOffer, Halina) [7]-[10] | Framework-level | Classifier template | Precision/recall tradeoff | Emphasis on clarifying requirements and delivery structure |

## References

1. A. Aminian, A. Xu, "Machine Learning System Design Interview", ch. 5 "Harmful Content Detection", ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/harmful-content-detection (book)
2. ByteByteGo, "ml-bytebytego: references for Machine Learning System Design Interview", GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
3. alirezadir, "Harmful content detection" (mlsd-harmful-content.md), GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-harmful-content.md (fetched)
4. alirezadir, "ML system design" (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
5. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
6. Educative / Khang Pham, "Machine Learning System Design", educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
7. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
8. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
9. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
10. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
11. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
12. A. Halevy et al., "Preserving Integrity in Online Social Networks", arXiv, 2020. https://arxiv.org/pdf/2009.10311.pdf (repo-link)
13. Meta AI, "Community Standards Enforcement Report" (WPIE), ai.facebook.com, 2020. https://ai.facebook.com/blog/community-standards-report/ (repo-link)
14. Meta AI, "How Facebook uses super-efficient AI models to detect hate speech", ai.facebook.com, 2020. https://ai.facebook.com/blog/how-facebook-uses-super-efficient-ai-models-to-detect-hate-speech/ (repo-link)
15. A. Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R), 2019. https://arxiv.org/abs/1911.02116 (known)
16. Meta AI, "Harmful content can evolve quickly. Our new AI system adapts to tackle it" (Few-Shot Learner), ai.facebook.com, 2021. https://ai.facebook.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it/ (repo-link)
17. Meta, "Meta's new AI system to help tackle harmful content", about.fb.com, 2021. https://about.fb.com/news/2021/12/metas-new-ai-system-tackles-harmful-content/ (repo-link)
18. N. Noorshams, S. Verma, A. Hofleitner, "TIES: Temporal Interaction Embeddings for Enhancing Social Media Integrity at Facebook", 2020. https://arxiv.org/pdf/2002.07917.pdf (repo-link)
19. Meta Transparency Center, "Content borderline to the Community Standards", transparency.fb.com. https://transparency.fb.com/features/approach-to-ranking/content-distribution-guidelines/content-borderline-to-the-community-standards/ (repo-link)
20. Pinterest Engineering, "How Pinterest fights misinformation, hate speech, and self-harm content with machine learning", Medium, 2021. https://medium.com/pinterest-engineering/how-pinterest-fights-misinformation-hate-speech-and-self-harm-content-with-machine-learning-1806b73b40ef (repo-link)
21. Pinterest Engineering, "How Pinterest powers a healthy comment ecosystem with machine learning", Medium, 2021. https://medium.com/pinterest-engineering/how-pinterest-powers-a-healthy-comment-ecosystem-with-machine-learning-9e5c3414c8ad (repo-link)
22. LinkedIn Engineering, "Viral spam content detection at LinkedIn", 2023. https://engineering.linkedin.com/blog/2023/viral-spam-content-detection-at-linkedin (repo-link)
23. LinkedIn Engineering, "Detecting and preventing abuse on LinkedIn using isolation forests", 2019. https://engineering.linkedin.com/blog/2019/isolation-forest (repo-link)
24. Bumble Tech, "Multilingual message content moderation at scale (part 1)", Medium, 2021. https://medium.com/bumble-tech/multilingual-message-content-moderation-at-scale-ddd0da1e23ed (repo-link)
25. Nextdoor Engineering, "Using predictive technology to foster constructive conversations", 2022. https://engblog.nextdoor.com/using-predictive-technology-to-foster-constructive-conversations-4af437942bd4 (repo-link)
26. Whatnot Engineering, "How Whatnot Utilizes Generative AI to Enhance Trust and Safety", Medium, 2023. https://medium.com/whatnot-engineering/how-whatnot-utilizes-generative-ai-to-enhance-trust-and-safety-c7968eb6315e (repo-link)
27. Slack Engineering, "Blocking Slack Invite Spam With Machine Learning", 2021. https://slack.engineering/blocking-slack-invite-spam-with-machine-learning/ (repo-link)
28. S. Wang et al., "Linformer: Self-Attention with Linear Complexity", 2020. https://arxiv.org/pdf/2006.04768.pdf (repo-link)
29. W. Wang, D. Tran, M. Feiszli, "What Makes Training Multi-modal Classification Networks Hard?" (gradient blending), 2019. https://arxiv.org/pdf/1905.12681.pdf (repo-link)
30. T.-Y. Lin et al., "Focal Loss for Dense Object Detection", 2017. https://arxiv.org/abs/1708.02002 (known)
31. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021. https://arxiv.org/abs/2103.00020 (ch-ref)
32. T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), 2020. https://arxiv.org/abs/2002.05709 (ch-ref)
33. T. Pan et al., "VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples", 2021. https://arxiv.org/pdf/2103.05905.pdf (repo-link)
34. V. Sanh et al., "DistilBERT, a distilled version of BERT", 2019. https://arxiv.org/abs/1910.01108 (known)
35. D. Kiela et al., "The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes", 2020. https://arxiv.org/abs/2005.04790 (known)
36. Meta / Facebook, "ThreatExchange" (PDQ, TMK+PDQF, vPDQ hashing; HMA platform), GitHub. https://github.com/facebook/ThreatExchange (fetched)
37. Meta Transparency Center, "Inauthentic Behavior" (Community Standards), transparency.fb.com. https://transparency.fb.com/policies/community-standards/inauthentic-behavior/ (repo-link)
38. Meta @Scale, "Building and scaling human review system" (talk), facebook.com. https://www.facebook.com/atscaleevents/videos/1201751883328695/ (repo-link)
39. Hugging Face, "distilbert-base-multilingual-cased" model card. https://huggingface.co/distilbert-base-multilingual-cased (repo-link)
40. Meta AI Research, "MMF: Hateful Memes project (configs and baselines)", GitHub. https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes (fetched)
