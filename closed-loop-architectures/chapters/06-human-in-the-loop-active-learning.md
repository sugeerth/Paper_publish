# 06. Human-in-the-Loop and Active Learning: Labeling Platforms, Weak Supervision, Annotation Workflows, and Expert Feedback Pipelines

## Overview

Every closed-loop learning system in this atlas eventually has to answer the same question: where does the next unit of supervision come from, and who or what produces it? Chapters 02 and 03 cover two answers: fleet telemetry mined by a data engine, and pairwise preferences turned into a reward signal. This chapter covers the machinery between those two. It is about the human labor, the selection algorithms that decide what humans look at, and the platforms that carry that work at industrial scale.

Five families of design are covered:

1. **Active learning (AL).** The model picks which unlabeled examples a human should label next, using uncertainty, disagreement or diversity. The classical theory dates to the 1990s and 2000s [1]. Deep variants appeared from 2017 [2][3][4][5][6], and there is a mature open-source tool layer [7][8][9][11][15].
2. **Programmatic weak supervision.** Humans write *labeling functions* instead of labels, and a generative *label model* denoises them. This is the Snorkel line [16][17][18][19][20], which Google deployed as Snorkel DryBell [20][21].
3. **Labeling platforms.** Products such as Scale AI, Labelbox, Label Studio, Prodigy and Argilla turn the loop into APIs for tasks, batches, callbacks, model-assisted pre-labeling, review queues and agreement metrics [11][12][13][14][15][26][27][28]. Model-in-the-loop data engines such as Segment Anything push the same idea until the model does most of the labeling [65][66].
4. **Quality control and aggregation.** This covers gold questions, screening, consensus, reviewer tiers, the statistical aggregation of noisy annotators [29][30][31], model-based detection of label errors [32][33][34], and statistically valid estimation from a few human labels plus many model labels [67][68].
5. **Human feedback for LLMs.** This runs from crowdworker preference collection with vendors (OpenAI, Anthropic, Meta, NVIDIA, OpenAssistant) [37][39][41][44][46][48] and public preference arenas [69] to *expert* and *model-assisted* feedback: process supervision with active selection [52][53], AI critics that help human raters [54][56], expert-written frontier evaluations [57], expert rubrics used as rewards and evaluation specifications [71][72][73][74], and adversarial human red-teaming programs [42][58].

The common thread is that **human attention is the scarcest resource in the loop**. Each design in this chapter is a way of spending it better. Some choose better examples (AL). Some move humans up a level of abstraction (weak supervision, guidelines, constitutions). Some build better tools (pre-labeling, critics). Some measure and correct human error (aggregation, confident learning).

A note on sourcing: this chapter relies on primary papers, official repositories, model cards and vendor SDK documentation. Numbers the sources report as approximate, or that are drawn from paper abstracts, are flagged as such.

---

## 1. Reference architecture of a human-in-the-loop (HITL) labeling loop

Published systems differ in emphasis, but most of them instantiate the same set of components:

```
            +-------------------------------------------------------------+
            |                  UNLABELED POOL / STREAM                    |
            |   (logs, fleet data, API prompts, model samples, documents) |
            +------------------------------+------------------------------+
                                           |
                     +---------------------v---------------------+
                     | SELECTOR: random | uncertainty | diversity |
                     | QBC | error-mining | policy samples (RLHF)  |
                     +---------------------+---------------------+
                                           |  batch of tasks
          +--------------------------------v-------------------------------+
          | ROUTER: crowd | vendor workforce | in-house experts | LLM judge |
          |  (skills, qualification, cost, confidentiality, overlap k)     |
          +-------+--------------------------------------------+-----------+
                  |                                            |
       +----------v-----------+                     +----------v-----------+
       | ANNOTATION UI        |  pre-labels from    | PROGRAMMATIC SOURCES |
       | (+ model pre-labels, |<--- current model --| labeling functions,  |
       |  interactive tools)  |                     | heuristics, KBs, LLMs|
       +----------+-----------+                     +----------+-----------+
                  |  raw labels                                | noisy votes
       +----------v--------------------------------------------v-----------+
       | QA + AGGREGATION: gold checks, agreement, review/reject,          |
       | Dawid-Skene / label model, confident-learning error flags         |
       +----------------------------------+--------------------------------+
                                          |  labels + provenance + confidence
                        +-----------------v------------------+
                        | LABEL STORE (versioned, auditable) |
                        +-----------------+------------------+
                                          |
                        +-----------------v------------------+
                        | TRAIN / EVAL -> new model version  |----> back to SELECTOR
                        +------------------------------------+       and pre-labeler
```

Three properties separate mature loops from ad-hoc ones:

- **Provenance per label.** This means worker id, batch, guideline version, model version used for pre-labeling, and confidence. OpenAI's released summarization comparisons carry exactly these fields (`worker`, `batch`, `split`, a 1 to 9 `confidence`, and a free-text `note`) [38]. PRM800K labels record `labeler`, `timestamp`, `generation`, and whether the item was a quality-control or screening question [53].
- **Closed QA sub-loops.** A rejected annotation must go somewhere. Label Studio's reviewer can reject with "No Rework", "Return to Annotator" or "Pass to Another Annotator" [14]. Scale's task API exposes an `audit_task(accept|reject, comment)` call [26].
- **Coupling between model and selector.** The current model both pre-labels and chooses what to label next. That coupling is the source of AL's gains and of its main failure modes (Section 2.4).

---

## 2. Active learning: selecting what humans label

### 2.1 Problem and classical taxonomy

Labels are expensive, and most unlabeled examples are redundant. Settles' survey [1] remains the standard map of the field. It defines three **scenarios**:

- **Pool-based.** Score a large unlabeled pool and pick the best examples. This is the common industrial case.
- **Stream-based selective sampling.** Decide label or skip as each example arrives. This fits online systems and fleet triggers.
- **Membership query synthesis.** The learner generates its own query points. It is rarely used with humans because synthesized inputs can be meaningless to annotators.

It also groups **query strategies** into these families [1]:

| Strategy | Score for candidate x | Strengths | Known weaknesses |
|---|---|---|---|
| Least confidence | 1 - P(y_hat given x) | Trivial to compute | Ignores the rest of the distribution |
| Margin | gap between top-2 class probabilities | Good default for multiclass | Still myopic, outlier-prone |
| Entropy | H(P(y given x)) | Uses full distribution | Favors noisy or ambiguous items |
| Query-by-committee (QBC) | vote entropy or KL disagreement among an ensemble | Model-agnostic, captures epistemic uncertainty | Cost of K models |
| Expected model change | expected gradient length | Targets impactful items | Expensive; scale-sensitive |
| Expected error / variance reduction | expected future loss on the pool | Directly optimizes objective | Usually intractable at scale |
| Density-weighted | informativeness x representativeness | Avoids outliers | Needs good similarity metric |
| BALD (Bayesian) [2] | mutual information between y and model params | Separates epistemic from aleatoric | Needs approximate posterior (MC dropout, ensembles) |
| Core-set [4] | k-center coverage in embedding space | Batch diversity, no uncertainty needed | Embedding quality, high-dim distances |
| BADGE [5] | k-means++ over gradient embeddings | Mixes uncertainty and diversity | Gradient embedding cost |

Settles also lists the practical issues that decide whether AL works in production [1]:

- **Batch-mode querying.** Real annotation is batched, so the top-k most uncertain items are often near-duplicates.
- **Noisy oracles.** Annotators make mistakes, and hard items are the ones they get wrong most often.
- **Variable labeling costs.** Uncertain items often take longer to label.
- **Stopping criteria.** Knowing when further labeling stops paying off.

### 2.2 Deep active learning

Softmax confidence from deep networks is poorly calibrated, so the deep-AL literature substitutes better uncertainty estimates or diversity objectives:

- **Deep Bayesian AL (Gal, Islam, Ghahramani, 2017)** [2] uses Monte Carlo dropout as an approximate posterior. It computes BALD, entropy and variation ratios from T stochastic forward passes, and showed these acquisition functions beat random selection on image tasks.
- **BatchBALD (2019)** [3] scores the *joint* mutual information of a batch, so it avoids picking items that are each informative on their own but redundant with one another.
- **Core-set (Sener and Savarese, 2017)** [4] reframes batch selection as k-center coverage in the network's embedding space. It is a diversity-first approach that does not depend on calibrated uncertainty.
- **BADGE (Ash et al., 2019)** [5] embeds each candidate as the gradient of the loss with respect to the last layer under the model's predicted label. Gradient magnitude encodes uncertainty and direction encodes diversity. k-means++ seeding then picks a batch.
- **Stochastic batch acquisition (Kirsch et al., 2021)** [6] shows that sampling from a softmax or power distribution over acquisition scores, instead of taking the top-k, is a cheap way to get batch diversity. In the authors' experiments it often matches more expensive joint or diversity methods such as BatchBALD and BADGE at a small fraction of their compute.
- **Baal (ElementAI, later independent)** [7] packages MC dropout, MC DropConnect, deep ensembles and semi-supervised variants. Its accompanying paper is explicitly about "tips and tricks to make active learning usable in production".

### 2.3 Production AL loops in tools

Several tools implement the loop directly:

- **Label Studio** implements the loop with an ML backend SDK exposing `predict()` and `fit()` [11][12]. In the Enterprise edition:
  - Each created or updated annotation fires a webhook that retrains the backend.
  - When the annotator moves on, Label Studio fetches fresh predictions for the next task.
  - "Uncertainty task sampling" orders the queue so annotators always see the tasks with the lowest prediction scores from the latest model version [11].
  - The open-source Community edition cannot do this live. It degrades to *batched* AL: fetch predictions, sort the data manager by prediction score, label for a while, retrain, and repeat [11].
  - This split is a useful reminder that AL needs orchestration infrastructure, not just a scoring function.
- **Label Studio's backend catalog** distinguishes three capabilities per model [12]: pre-annotation, interactive mode (for example SAM 2 click-to-segment, Grounding DINO prompts, GLiNER), and training on submitted annotations. Interactive models move the human from drawing to correcting, which is a different lever from selection.
- **Prodigy** (Explosion) builds its `ner.teach` and `textcat.teach` recipes around binary accept/reject decisions on model suggestions, with "the model in the loop" deciding what to ask next [15]. The recipe source exposes the sorter. Swapping `prefer_uncertain()` for `prefer_high_scores()` turns uncertainty sampling into high-score mining. Companion recipes (`ner.correct`, `ner.silver-to-gold`) turn binary "silver" annotations into complete "gold" ones [15].
- **modAL** [8] (scikit-learn based) and **small-text** [9] (text classification with transformers) expose AL as composable query strategy, classifier and stopping-criterion components. small-text's v2.0 line (still an alpha development release as of v2.0.0.dev4, May 2026) adds refined interfaces, vector indices and new query strategies [9].
- **cleanlab** [34] includes ActiveLab, "active learning with data re-labeling", for *multi-annotator* settings. It decides whether the next label should go to a new example or to re-labeling an existing one.

### 2.4 Failure modes and lessons

- **Cold start.** Uncertainty from a model trained on a few dozen labels is mostly noise, so practitioners seed with random or diversity-based samples first.
- **Sampling bias and model coupling.** The labeled set is no longer i.i.d. It is shaped by one model's blind spots. A different future architecture may benefit less from that data than from a random sample of equal size.
- **Batch redundancy.** Naive top-k selection picks near-duplicates. Joint (BatchBALD), diversity (core-set, BADGE) or stochastic (power sampling) acquisition fixes this [3][4][5][6].
- **Outliers and noisy oracles.** Entropy maximization surfaces garbage and genuinely ambiguous items. Those are exactly the items annotators disagree on, so the information gain on paper becomes label noise in practice.
- **Evaluation contamination.** If the test set is drawn from AL-selected data, metrics are biased. Keep a random, independently labeled evaluation slice.
- **Adoption reality.** A community survey published at EACL 2026, co-authored by the small-text maintainers, re-examined how AL is actually used in contemporary NLP [10]. Its existence is a sign that AL's role is being renegotiated now that LLM zero-shot labeling and synthetic data compete with it for the same budget.

The durable lesson is that AL pays off most when **labels are truly expensive** (experts, medical, legal), **the pool is large and redundant**, and **the selector is coupled to a review loop** that can absorb the extra ambiguity it surfaces.

---

## 3. Programmatic weak supervision: the Snorkel lineage

### 3.1 Problem

Hand labeling hits a wall when domain experts are scarce (clinicians, lawyers, analysts), when label definitions change often, or when privacy forbids sending data out to a crowd. The Snorkel project started at Stanford in 2015 on the bet that "it would increasingly be the training data, not the models, algorithms, or infrastructure, that decided whether a machine learning project succeeded or failed" [22].

### 3.2 Data programming (NeurIPS 2016)

Ratner et al. [16] proposed that users write **labeling functions (LFs)**: small programs that vote a label or abstain, such as regexes, knowledge-base lookups, heuristics or other models. The LFs' accuracies and correlations are never observed directly. Instead, a **generative label model** learns them from the agreement and disagreement structure of the LF outputs alone. The resulting probabilistic labels then train a discriminative **end model** with a noise-aware loss. Because the end model sees features the LFs do not use, it generalizes beyond LF coverage.

### 3.3 Snorkel (VLDB 2018) architecture

```
  subject-matter expert
          |
          v  writes / edits
  +-----------------------+    apply over     +-----------------------------+
  | Labeling functions    |------------------>| Label matrix L (n x m)      |
  | (patterns, KB lookups,|   unlabeled data  | entries in {-1 abstain, 0,  |
  |  heuristics, models)  |                   |  ..., k-1}                  |
  +-----------------------+                   +--------------+--------------+
          ^                                                  |
          |  LF analysis: coverage, overlap, conflict,       v
          |  empirical accuracy on a small dev set   +----------------------+
          +------------------------------------------| Label model          |
                                                     | (learns LF accuracies|
                                                     |  w/o ground truth)   |
                                                     +----------+-----------+
                                                                | P(y given L)
                                                     +----------v-----------+
                                                     | End model (deep net),|
                                                     | noise-aware loss     |
                                                     +----------------------+
```

Snorkel [17] made this an end-to-end system: an LF authoring interface, a label model, and LF development analytics such as coverage, overlaps, conflicts and empirical accuracy on a small development set. As reported in its abstract, a user study found that subject-matter experts built models **about 2.8x faster** and improved predictive performance by **about 45.5% on average** compared with seven hours of hand labeling. Across applications, Snorkel models came within a few percentage points of models trained on large hand-curated training sets [17].

Later components extended the paradigm:

- **Snorkel MeTaL** [18] extended the label model to *multi-task* weak supervision, with tasks at different granularities and a matrix-completion-style estimator. The motivating vision was "massively multitask learning" with "tens or hundreds of tasks with supervision of widely varying quality". MeTaL was merged into Snorkel v0.9 in August 2019 [18].
- **Snorkel v0.9** added two more operators besides LFs. *Transformation functions* do data augmentation. *Slicing functions* monitor and up-weight critical subsets [23].
- **FlyingSquid** [19] replaced iterative SGD label-model fitting with a closed-form *triplet* method. It solves for LF accuracies from triplets of conditionally independent LFs, and averaging over all triplets proved more stable. This makes the label model fast enough for interactive and online use.

### 3.4 Snorkel DryBell at Google (SIGMOD Industry 2019)

DryBell [20][21] is the canonical industrial deployment. Google adapted Snorkel to its infrastructure and used it on content and event classification tasks. The paper's running example is classifying documents for celebrity mentions [20][23]. Key design points:

- **Organizational knowledge as LFs.** LFs wrapped existing internal resources: legacy models, knowledge graphs, entity taggers, aggregate statistics and heuristics. Many of these resources were *non-servable*. They were too slow, too expensive, or depended on features unavailable at serving time. Weak supervision **distilled non-servable signals into a servable end model** that uses only production features.
- **Execution at scale.** The label model was re-implemented to run over millions of examples in a distributed setting. The open-source tutorials mirror this with Dask and Spark versions of the pipeline and an `NLPLabelingFunction` template that caches expensive NLP preprocessing [23].
- **Reported results (abstract-level, approximate).** Classifiers of quality comparable to ones trained on tens of thousands of hand-labeled examples; about a 52% average improvement from converting non-servable resources into servable models; and labeling of millions of data points in tens of minutes [20].

### 3.5 Ecosystem and evolution

- **Snorkel Flow.** The academic team shifted to a commercial platform, Snorkel Flow, extending the ideas to data slicing, monitoring, augmentation and multi-task modeling. The open-source repository now points users there [22].
- **WRENCH** [24] standardized evaluation across many weak supervision datasets and label models. Its companion survey [25] organizes the field: label models, joint label and end models, and interactive LF development.
- **Overlap with crowdsourcing.** The same machinery handles crowdworkers as LFs. Snorkel's `crowdsourcing` tutorial treats each worker as a labeling function [23]. This makes the label model a generalization of Dawid-Skene aggregation (Section 5.2).

### 3.6 Failure modes and lessons

- **Correlated LFs** (for example, two regexes derived from one keyword list) violate independence assumptions and double-count evidence. Structure learning or careful LF design is required.
- **Coverage gaps.** Examples where all LFs abstain get no signal. End-model generalization helps, but it cannot fix a systematically uncovered region.
- **A dev set is still needed.** LF iteration relies on a small hand-labeled development set. Weak supervision reduces hand labeling; it does not eliminate it.
- **The human moves up a level.** The highest-leverage effect is organizational. Experts encode *rules and resources once*, and the system re-labels everything when definitions change. That makes labeling a versioned code artifact. This same idea reappears in LLM alignment as constitutions and rubrics (Section 7.4 and Chapter 03).

---

## 4. Labeling platforms as loop infrastructure

### 4.1 Scale AI: the task API and the vendor workforce

Scale AI's public Python client documents the abstractions an external labeling loop needs [26]:

- **Tasks** are created per task type with a `callback_url` (push notification on completion) and an idempotency key `unique_id`. Creating a duplicate raises an error, which prevents double-paying on client retries. Status moves through pending, completed, error and canceled.
- **Batches** group tasks. A batch is *finalized* to release it to the workforce, and progress is polled through `tasks_pending` and `tasks_completed`. Tasks can be filtered by `review_status`, for example `Accepted`.
- **Audits.** `audit_task(id, accept, comment)` mirrors Scale's Audit UI, so the customer's own QA decision feeds back to the vendor.
- **Projects** come in two documented modes, *Rapid* and *Studio* (set by `rapid=true` or `studio=true`). Assignments and labeler groups are Studio-only features, which matches Studio's role as the product for customer-managed labeling teams. Rapid projects support **evaluation tasks**: tasks with a known `expected_response` that are "used to measure workers' performance internally to ensure the quality". They also support **training tasks**, used "to onboard taskers onto your project" [26].

Scale's role extends upstream (curation and error mining with Nucleus, covered in Chapter 02) and into LLM post-training data. Three public records show Scale as an RLHF and evaluation vendor:

- InstructGPT's contractors were hired "through Upwork and ScaleAI" [40].
- NVIDIA's HelpSteer2 preference data was collected with Scale AI (Section 6.5) [46].
- Humanity's Last Exam was a joint effort of the Center for AI Safety and Scale AI [57].

The market's strategic weight showed in June 2025, when Meta took a non-voting 49% stake in Scale (reported at about $14 billion) and CEO Alexandr Wang moved to Meta [51]. That transaction raised the obvious vendor-neutrality question for frontier labs that also bought data from Scale. This is a structural risk for any loop that depends on one external workforce.

### 4.2 Labelbox: catalog, model-assisted labeling and LLM feedback projects

Labelbox positions itself as a platform to "curate and annotate data, generate high-quality human feedback data for computer vision and language models, evaluate and improve model performance, and automate tasks by seamlessly combining AI and human-centric workflows" [27]. Its official SDK notebooks map the loop components one-to-one [27]:

- **Curation:** data rows, metadata, custom embeddings for similarity search (Catalog).
- **Pre-labeling:** "Model predictions to project" and per-modality *prediction upload* notebooks, plus annotation import. This is model-assisted labeling, where humans correct imported predictions instead of labeling from scratch. Meta SAM integrations cover interactive segmentation.
- **Evaluation:** model experiments, model slices and custom metrics. These compare model runs against ground truth to find error clusters worth relabeling.
- **Workflow:** batches, queue management, webhooks, ontologies, user management.
- **LLM feedback:** dedicated project types for *multimodal chat evaluation*, *offline multimodal chat evaluation*, *prompt-response* creation and *LLM data generation* [27]. These are the human-feedback collection UIs used for SFT demonstrations and preference ranking.

### 4.3 Label Studio: open-source core, agreement and review

Label Studio (HumanSignal) is the most widely documented open-source platform. Beyond the AL loop in Section 2.3, its Enterprise documentation is unusually explicit about QA mechanics [13][14]:

- **Agreement is computed per control tag.** A task with bounding boxes, a choice field and a rating has three agreement scores. The overall score is a weighted mean, so critical fields can dominate.
- **Agreement metrics depend on data type.** Categorical tags use exact match (or numeric difference with a threshold for ratings). Spatial and structural tags use IoU for boxes and polygons, and span overlap for text.
- **Two aggregation methodologies:** pairwise average across annotator pairs, and consensus.
- **Review ordering.** Reviewers can order their queue by annotator (to audit individuals), by agreement (low-agreement first), or by **model confidence** (least confident first) [14]. This turns the review queue into a second AL loop.
- **Reject routing:** No Rework, Return to Annotator, or Pass to Another Annotator [14].

### 4.4 Argilla, Prodigy and LLM-native tooling

- **Argilla** describes itself as "a collaboration tool for AI engineers and domain experts to build high-quality datasets". It is deployable on the Hugging Face Hub [28] and is used for preference and SFT data review.
- **Prodigy** keeps annotation *binary and fast*. Accept/reject on a model suggestion is far cheaper per decision than full manual annotation [15].
- **LLM pre-labeling with confidence routing.** Refuel's Autolabel [36] labels text datasets with any LLM from a JSON config of guidelines and few-shot examples. It includes cost estimation and caching, and attaches a confidence estimate to every label. The recommended pattern is to "calibrate a confidence threshold for your labeling task, and then route less confident labels to humans" [36]. The README's toy run reports roughly $0.0027 per example. This is the 2023 to 2026 successor to classical AL: *the model labels everything, and humans see only what the model is unsure about.*

### 4.5 Model-in-the-loop data engines: Segment Anything and SAM 2

Meta's Segment Anything projects are the clearest published example of a labeling loop where the model being trained is also the annotation tool, and its share of the labeling grows stage by stage.

- **SAM (2023)** [65] built SA-1B, about 11 million images and 1.1 billion masks, with a three-stage data engine (counts below are as reported in the paper and approximate):
  1. *Assisted-manual.* Professional annotators clicked foreground and background points in a browser tool powered by an early SAM and refined the masks. The model was retrained several times on the growing data, and the reported time per mask fell from about 34 to about 14 seconds.
  2. *Semi-automatic.* The model pre-filled confident masks, and annotators labeled only the objects it had missed. This is a diversity-targeted selector: human effort goes to the less prominent objects.
  3. *Fully automatic.* The model was prompted with a regular point grid. Masks were kept by predicted IoU and stability filters, and the released dataset is almost entirely (about 99%) model-generated. Human effort shifted to quality audits of samples.
- **SAM 2 (2024)** [66] repeated the pattern for video with a "model-in-the-loop data engine, which improves model and data via user interaction" [66]. The resulting SA-V dataset has about 51K videos and 643K masklets. Of the training masklets, about 190K are manual and about 452K are automatic. The validation and test masklets are manually annotated in every frame, at 6 fps [66]. The paper reports that annotating with SAM 2 in the loop was several times faster than per-frame annotation with the original SAM (approximate).

Two design points carry over to any domain. First, the *human-to-model labor ratio is a schedule*, not a constant, and each stage needs a different UI and QA regime. Second, the evaluation splits stay fully human-annotated even when the training data is mostly automatic, which is the same discipline as the random evaluation slice in Section 2.4.

### 4.6 Platform comparison

| Platform | Deployment | AL / selection | Pre-labeling | QA primitives | LLM feedback UIs |
|---|---|---|---|---|---|
| Scale AI [26] | Managed workforce + API | Customer-driven; Nucleus mining (Ch. 02) | Studio / model-assisted | Evaluation (gold) tasks, training tasks, audits, review status | RLHF / evaluation vendor (InstructGPT, HelpSteer2, HLE) |
| Labelbox [27] | SaaS + SDK | Catalog search, model slices, metrics | Prediction import (MAL), SAM | Queues, review, webhooks | Chat evaluation, prompt-response, LLM data generation |
| Label Studio [11][12][13][14] | Open source + Enterprise | Uncertainty task sampling (Enterprise); manual sort (OSS) | ML backend `predict()`, interactive SAM 2 | Per-tag agreement, IoU/span metrics, reviewer reject routing | LLM backends for pre-annotation |
| Prodigy [15] | Local, scriptable | `*.teach` recipes with `prefer_uncertain()` | `*.correct` recipes | Silver-to-gold merge | Custom recipes |
| Argilla [28] | Open source, HF Hub | Via SDK | Via SDK | Multi-user review | Preference / SFT review |
| Snorkel [17][22] | Library -> Snorkel Flow | LF development analytics | Labeling functions | Label model estimates LF accuracy | (Flow: commercial) |
| Autolabel [36] | Library | Confidence-thresholded routing | LLM labels everything | Confidence calibration | N/A |

---

## 5. Annotation workflow design and quality control

### 5.1 The workforce funnel

Every documented high-quality pipeline uses a funnel of **screen, onboard, calibrate, monitor and review**:

```
 applicants --> screening test --> onboarding / training tasks --> production queue
                (agreement with     (guidelines, worked          | overlap k, hidden gold
                 researchers,        examples, shared chat)      | (QC items) per batch
                 sensitivity tests)                              v
                                                     reviewer tier (fix/accept/reject)
                                                                 |
                       per-worker accuracy, agreement, speed <---+--> aggregate label
                       -> retrain, re-instruct, or remove           with confidence
```

Concrete instances:

- **InstructGPT** selected about 40 contractors with a screening test. It measured their ability to identify and respond to sensitive prompts and their **agreement rate with researchers** on a task with detailed instructions. OpenAI ran an onboarding process, "wrote detailed instructions for each task, and answered labeler questions in a shared chat room" [40]. The team was deliberately kept small "because it's easier to have high-bandwidth communication with a smaller set of contractors who are doing the task full-time" [40].
- **PRM800K** served a fixed set of **30 screening questions** to every labeler ("generation -1") and reserved **quality-control solutions in every generation**, served to every labeler and checked against gold labels [53].
- **Scale Rapid** formalizes the same controls as evaluation tasks and training tasks [26].

### 5.2 Aggregating noisy annotators

When k annotators label the same item, majority vote ignores that annotators differ in skill and bias. Better options:

- **Dawid and Skene (1979)** [30] estimate a per-annotator confusion matrix and the true labels jointly with EM. It remains the baseline aggregation model.
- **Snow et al. (2008)** [31], "Cheap and Fast, But is it Good?", showed on several NLP tasks that a small number of non-expert crowd labels, properly aggregated, can approach expert quality. That result justified the crowdsourcing economy that followed.
- **Crowd-Kit** (Toloka) [29] packages these methods for production. Categorical methods include majority vote, one-coin and full Dawid-Skene, gold majority vote, GLAD, MACE, M-MSR, Wawa, Zero-Based Skill and KOS. It also covers text aggregation (ROVER, RASA, HRRASA), segmentation (majority vote, EM, RASA) and **pairwise comparisons (Bradley-Terry, noisy Bradley-Terry)**. The pairwise case is the same statistical object as an RLHF reward model's training target.

### 5.3 Finding the errors that survive QA

- **Confident learning** (Northcutt, Jiang, Chuang) [32] estimates the joint distribution of noisy and true labels from out-of-sample predicted probabilities. It then flags likely mislabeled examples without needing clean labels. The `cleanlab` library implements it and extends it to outliers, duplicates, multi-label, token classification, segmentation, object detection and multi-annotator consensus [34]. Its recommended loop is explicitly closed: train, diagnose data, fix, retrain, and evaluate on *cleaned* data [34].
- **Benchmark test sets are noisy too.** Northcutt, Athalye and Mueller estimated an average of at least 3.3% label errors across ten widely used test sets [33]. Rankings of models can flip once errors are corrected.
- **Prediction versus label disagreement, sorted by confidence,** is the cheapest error signal in practice. It is the same signal Scale Nucleus and Tesla's data engine exploit (Chapter 02), and the signal Label Studio's "review by model confidence" ordering exposes [14].

### 5.4 Guidelines as versioned artifacts

Published guidelines are rare but instructive:

- **InstructGPT** published its API-distribution labeling instructions and its toxicity labeling instructions as public documents [40].
- **PRM800K** released its phase 1 and phase 2 labeler instructions alongside the data [53].
- **OpenAssistant** published its guidelines openly and asked for pull requests. They include per-task do and don't lists (for example, "Present only information that has been verified by credible sources", and for opinion questions, "bring up at least 2 common viewpoints") plus a thumbs-up/down signal for flagging quality [49].

The lesson is that guidelines function as **the specification of the reward**. Changing a guideline mid-collection changes the label distribution, so the guideline version belongs in each label's provenance.

### 5.5 LLMs as annotators

Gilardi, Alizadeh and Kubli [35] found ChatGPT's zero-shot accuracy exceeded that of MTurk crowdworkers on most of the text-annotation tasks they tested (four of five in the published version), with higher intercoder agreement than both crowdworkers and trained annotators. They reported a per-annotation cost under about $0.003, an order of magnitude or more cheaper than MTurk (the multiple differs between the preprint and the PNAS version; approximate). The study covered a handful of tweet and news-article classification tasks with a single model, so the result is a cost benchmark, not a general claim that LLMs replace trained annotators. This and similar results drove the "LLM pre-labels, human verifies" designs in Section 4.4.

AI-labeled data needs its own audit loop. UltraFeedback, a GPT-4-annotated preference dataset of about 64k prompts, 256k responses and about 340k constructible comparison pairs [62], later found that **2,628 completions had been assigned an overall score of 10 when many should have been 1**. The maintainers fixed this by rule-based adjustment and re-annotation [62]. AlpacaFarm [63] goes further and *simulates* human annotators with a pool of LLM judges plus injected noise to mimic inter-annotator variance, so feedback methods can be developed cheaply before paying for human data.

### 5.6 Few human labels, many model labels: statistically valid estimates

When an LLM or the current model labels the bulk of the data, a separate question arises: how can *aggregate* quantities (an error rate, a prevalence, a win rate) be estimated without inheriting the labeler's bias? Prediction-powered inference (PPI) [67] answers it. A small set of gold human labels is used to measure the model labeler's bias (the "rectifier"). That correction is then applied to estimates computed over the large model-labeled set. The resulting confidence intervals stay valid whatever the model's quality, and they are tighter than intervals from the human labels alone whenever the model is reasonably accurate. The `ppi_py` package [67] implements means, quantiles and regression coefficients, along with follow-ups such as PPI++, cross-prediction-powered inference and a prediction-powered bootstrap.

Active statistical inference (Zrnic and Candès, 2024) [68] closes the loop with Section 2. The model's own uncertainty decides which items get a human label, and those items are reweighted by inverse selection probability so the final estimate stays unbiased. The result is an active learning scheme whose goal is a valid *estimate*, not a better classifier. It reaches a target interval width with fewer human labels than uniform sampling.

For loop designers this provides a principled middle ground between "trust the LLM judge" and "label everything by hand". Keep a randomly or probabilistically sampled human-labeled slice, log the selection probabilities, and report model-assisted metrics with corrected intervals.

---

## 6. Human feedback for LLMs: crowd and vendor pipelines

Chapter 03 covers the algorithms (reward models, PPO, DPO, iterated RLHF). This section covers the *human data supply chain* behind them.

### 6.1 OpenAI: from summarization comparisons to InstructGPT

- **Learning to summarize (2020)** [37][38] released 64,832 TL;DR summary comparisons, plus Likert axis evaluations (overall, accuracy, coverage, coherence) on TL;DR and CNN/DM. The schema stores the worker id, batch, the policy that produced each summary, the labeler's confidence, and a "naive interpretation note" written before reading the post. Training data for the supervised baseline came from specific batches (batch3 to batch10), while later batches were primarily evaluation. Validation splits were partitioned so that labels used for model selection (`valid1`) were never used for final evaluation (`valid2`) [38]. That is a clean design for avoiding selection leakage in human evaluation.
- **InstructGPT (2022)** [39][40]:
  - About 40 contractors, hired through Upwork and Scale AI, after a screening test.
  - Prompts came from the OpenAI API Playground, where customers were notified that data could be used for training, and were filtered for PII.
  - Labelers also wrote bootstrap prompts, because instruction-like prompts were rare on the earlier API.
  - The model card is candid that labeling tasks "rely on value judgments that may be impacted by the identity of our contractors", and that the group "is clearly not representative of the full spectrum of people who will use and be affected by our deployed models" [40].
  - Headline result: outputs of the 1.3B InstructGPT were preferred to those of the 175B GPT-3 [39].

### 6.2 Anthropic: crowd platforms, online iteration, red-teaming at scale

- **Workforce.** Anthropic's HH-RLHF and red-team data came from crowdworkers on Upwork and Amazon Mechanical Turk. The red-team release flags each record with `is_upworker` [43].
- **Online loop.** Preference models and RL policies were refreshed "on a weekly cadence" with fresh human feedback [41]. The released helpfulness data is split into tranches: base-model samples, rejection-sampled (mostly best-of-16) samples against an early preference model, and samples from the iterated "online" process [43]. This is a direct record of the selector changing as the model improves.
- **Red-teaming as a human loop.** Ganguli et al. released **38,961 red-team attacks** across 2.7B, 13B and 52B models and four model types. RLHF models "are increasingly difficult to red team as they scale" [42]. Each transcript includes the attacker's self-rated success, a short task description and a preference-model harmlessness score. Post-hoc crowdworker tags (up to six per transcript) were collected for a random sample of 1,000 attempts drawn from two of the four model types [43]. Each record also carries a `red_team_member_id`, so per-attacker productivity can be analyzed.

### 6.3 Meta Llama 2: vendor-scale collection and vendor variance

Llama 2-Chat [44][45] is the most detailed public account of a vendor-scale human preference loop:

- **Volume.** Over one million new human-annotated examples [45], including about 1.4 million binary comparisons collected in weekly batches (details in Chapter 03).
- **SFT data volume.** The SFT stage used a comparatively small set of high-quality vendor annotations, about 27,540 (approximate) [44]. Meta's stated finding was that a limited set of clean examples beat millions of noisier third-party ones.
- **Vendor variance.** Meta observed that **different annotation platforms and vendors can produce markedly different downstream model performance**. Its takeaway was that data checks matter even when vendors source the annotations [44].
- **Collection protocol.** Each comparison carried a 4-level preference strength and a separate safety label. The batches were sampled from the latest model, so the annotation stream is itself an on-policy selector [44].

### 6.4 OpenAssistant: a volunteer pipeline

OpenAssistant (LAION) crowdsourced an assistant-style corpus from volunteers through a gamified web app. The pipeline included prompt writing, assistant-reply writing, ranking, and labeling of quality and toxicity attributes, all governed by the public guidelines [49]. The accompanying paper reports roughly 161k messages in 35 languages with about 461k quality ratings, contributed by more than 13,000 volunteers (approximate) [48]. The project has since been completed, with a final oasst2 dataset [48]. Lesson: volunteer crowds can produce large multilingual data at near-zero cash cost. But quality control, abuse filtering and guideline adherence must be built into the app, because there is no contractual lever.

### 6.5 NVIDIA HelpSteer2: multi-attribute ratings via a vendor

HelpSteer2 [46] collected about 10k response pairs through Scale AI. Instead of a single preference bit, each response was rated on five 0 to 4 Likert attributes (helpfulness, correctness, coherence, complexity, verbosity) by multiple annotators. The data is permissively licensed (CC-BY-4.0). Reward models trained on it, including the Nemotron-4 340B reward model built with NeMo-Aligner [47], reached top RewardBench scores at release (approximate). (NeMo-Aligner itself was archived in May 2025, with NVIDIA directing users to NeMo RL [47].) Multi-attribute labels let downstream users re-weight objectives such as verbosity without re-collecting data. NVIDIA reported that more than 98% of Nemotron-4's alignment data was synthetic, with only about 20k human-annotated examples (Chapter 03). This is the clearest example of humans acting as *seed and calibrator* rather than *bulk producer*.

### 6.6 Labor conditions as a system property

The human layer has failure modes that do not show up in loss curves. A 2023 TIME investigation reported that workers in Kenya, employed through the outsourcing firm Sama to label toxic and graphic content for OpenAI's safety filters, were paid less than about $2 per hour and described psychological harm [50]. For system designers, wellness protocols, exposure limits and fair pay are not just ethical extras. They determine annotator retention, and therefore label consistency, which is a quality property of the dataset.

### 6.7 Public preference crowdsourcing: Chatbot Arena (LMArena)

Chatbot Arena [69] turned pairwise human preference collection into an open, always-on public service:

- **Protocol.** A user submits any prompt, receives answers from two anonymous models side by side, and votes for one (or a tie or "both bad"). Model identities are revealed only after the vote.
- **Aggregation.** The paper fits a Bradley-Terry model to the votes, with confidence intervals and anomalous-user detection [69]. This is the same statistical object as Crowd-Kit's pairwise aggregators and an RLHF reward model's target (Section 5.2). The paper also proposes adaptive sampling of model pairs so that rankings converge with fewer votes, which is active learning applied to evaluation.
- **Scale.** The paper reported about 240K votes at publication (March 2024). The project's FastChat repository later reported over 1.5 million human votes [69]. The project is now operated as LMArena.

The loop has the failure modes this chapter predicts. *The Leaderboard Illusion* (Singh et al., 2025) [70] argued that undisclosed private testing of many model variants before public release, unequal sampling rates across providers, and asymmetric access to Arena data let some providers fit the leaderboard rather than general preference. The lesson matches Section 9: when a human-preference signal is also a target that vendors optimize, the selection policy, data access and disclosure rules of the collection loop become part of its validity.

---

## 7. Expert feedback and scalable oversight (2022 to 2026)

As models approach expert performance, two problems appear. Crowd raters can no longer judge correctness, and the remaining errors become subtle. Published designs respond with active selection of the hardest items, AI assistance for raters, and expert-only pipelines.

### 7.1 Process supervision with active learning: PRM800K

OpenAI's "Let's Verify Step by Step" [52] collected about 800,000 **step-level** correctness labels (positive, neutral or negative) on model-generated solutions to MATH problems [53]. The released data documents the pipeline:

```
 Phase 1: labelers rate steps of sampled solutions; may label alternative
          completions; may write their own correct step if all candidates fail
                                     |
 Phase 2: generation g = 1..G
   best PRM so far --scores--> large pool of pre-generated solutions
          |                      select solutions to label (active learning;
          |                      the paper targets convincing wrong-answer solutions)
          v
   labelers rate steps; STOP at first error ("found_error") -> cost saving
   + QC solutions served to every labeler, checked against gold
          |
          v
   retrain PRM on all labels -> next generation
```

Key techniques [52][53]:

- **Model-in-the-loop selection.** In phase 2, collection was "split into generations, using our best PRM so far to pick which solutions to score in the next generation" [53]. The paper reports that this active learning made process supervision substantially more data-efficient than uniform selection, by roughly a factor of 2.6 (approximate) [52].
- **Early termination.** In phase 2, labeling stops at the first incorrect step (`finish_reason: found_error`), which spends human effort only on the informative prefix [53]. Labelers can also flag broken problems (`bad_problem`) or give up (`give_up`), and those signals are kept, not discarded.
- **Built-in QC.** Each generation has QC questions served to all labelers, plus a 30-question screening set [53].

Lesson: for expert-cost labels, **targeting the policy's convincing errors** (high model score, wrong answer) is the AL acquisition function that matters.

### 7.2 AI-assisted human evaluation

- **Self-critiquing models (Saunders et al., 2022)** [54] trained models to write critiques of summaries. Showing those critiques to human raters helped them find substantially more flaws than unassisted raters (about 50% more, approximate).
- **Measuring progress on scalable oversight (Bowman et al., 2022)** [55] proposed experiments on tasks that specialists can do but unaided non-experts and current models cannot (MMLU, time-limited QuALITY). Humans chatting with an unreliable LLM assistant "substantially outperform both the model alone and their own unaided performance" [55].
- **CriticGPT (McAleese et al., 2024)** [56] trained a GPT-4-based critic with RLHF to point out bugs in ChatGPT code answers, so that AI trainers could catch errors they would otherwise miss. As reported in the abstract, on code with naturally occurring LLM errors, model-written critiques were preferred over human-written ones in 63% of cases. Human-plus-critic teams wrote more comprehensive critiques than humans alone, while producing fewer hallucinated nitpicks than the critic alone [56]. The critics also flagged hundreds of errors in ChatGPT training data that had previously been rated "flawless" [56], which turns the critic into a label-error detector for the RLHF dataset itself (compare Section 5.3). Architecturally, the critic is a **second model whose job is to raise the human rater's recall**, inserted between policy output and human judgment.
- **Weak-to-strong generalization** [64] studies the limit case: what happens when a weaker supervisor (standing in for humans) labels data for a stronger model. It is the theoretical backdrop for why rater-assistance tools matter.

```
  policy output ---> critic model ---> annotated output ---> human rater ---> label
        |              (flags bugs,       (claims + quotes)     (accepts/rejects
        |               quotes spans)                            critiques, rates)
        +-------------------------------------------------------------------> RM data
```

### 7.3 Expert-authored evaluation sets: Humanity's Last Exam

HLE [57] is a 2,500-question multimodal benchmark of multiple-choice and short-answer questions, written by subject-matter experts worldwide and organized by CAIS and Scale AI. Its pipeline is an adversarial human-AI loop. Candidate questions were tested against frontier models, and questions the models answered correctly were rejected. The remaining questions went through rounds of expert review before inclusion. Contributors were paid from a prize pool reported at $500,000 (approximate). The repository includes a canary string to keep the data out of training corpora, a public form for flagging bad questions, and a rolling change log of revisions to the question set [57]. That form and change log are the post-release QA loop, which matters because errors in expert-written benchmarks are expensive to detect.

### 7.4 Adversarial human loops and public input

- **Constitutional Classifiers (Anthropic, 2025)** [58] combined synthetic training data with large human red-teaming loops:
  - A bug bounty on a prototype guarded Claude 3.5 Sonnet drew **183 active participants** spending **more than 3,000 hours** over about two months, with up to $15,000 offered for a universal jailbreak. No universal jailbreak was found, but the prototype over-refused and was expensive to run.
  - The production-oriented version was evaluated on 10,000 synthetic jailbreak prompts. Jailbreak success fell from 86% to 4.4%, with a 0.38% (not statistically significant) refusal increase and 23.7% compute overhead.
  - A public demo (February 3 to 10, 2025) drew 13,960 users overall. Of these, 339 jailbreakers passed at least one level, over more than 300,000 chats and about 3,700 hours. Four participants cleared all eight levels, one jailbreak was confirmed as universal (plus one borderline case), and $55K was paid out [58].
  - The design pattern is to **use paid human adversaries as the acquisition function for the hardest negatives**, then fold the discovered attacks back into synthetic data generation.
- **Collective Constitutional AI (Anthropic with the Collective Intelligence Project, 2023)** [59] collected *principles* instead of labels. About 1,000 U.S. adults used Polis to submit 1,127 statements and cast 38,252 votes (about 34 per person). After moderation, about 275 statements remained. Statements with high agreement were selected, then de-duplicated and merged to roughly match the size of the 58-principle standard constitution, and rewritten into "Choose the response that..." format. The resulting "Public" model performed comparably to the standard model on MMLU and GSM8K and showed less bias on BBQ [59]. This is weak supervision at the values level: humans write the rules, and AI feedback applies them.

### 7.5 Active selection for preference data

Classical AL ideas transfer directly to RLHF:

- **Efficient Exploration for LLMs (Dwaracherla et al., Google DeepMind, 2024)** [60] showed that choosing which response pairs to send to (simulated) raters with an epistemic neural network and **double Thompson sampling** reaches a given performance level with far fewer queries than passive pair selection.
- **Active Preference Learning (Muldrew et al., 2024)** [61] applied predictive-entropy and preference-certainty acquisition to DPO, selecting prompt and completion pairs for labeling during training.

Both results restate Section 2 for the preference setting. The information in a comparison depends on the current reward model's uncertainty, so on-policy, uncertainty-targeted pairs are worth more than random ones.

### 7.6 Experts as rubric writers and specialist raters (2025 to 2026)

Since 2025 the published record shows a shift in what experts are asked to produce. Instead of single preference bits, they write *rubrics*, *reference tasks* and *domain-specific judgments* that a model grader can then apply at scale. This is the Snorkel move from Section 3.6, applied to frontier LLMs.

- **HealthBench (OpenAI, 2025)** [71] has 5,000 multi-turn health conversations, each with a conversation-specific rubric written by physicians: 262 physicians from 60 countries wrote 48,562 unique rubric criteria. Each criterion carries positive or negative points. A model grader decides whether each criterion is met, and the score is the share of positive points earned. The grader is itself validated against physician grading in a meta-evaluation [71]. The reference implementation keeps the physician-written completions as baselines [71].
- **HelpSteer3-Preference (NVIDIA, 2025)** [72] succeeded HelpSteer2 with over 40,000 human-annotated preference samples, released under CC-BY-4.0. It covers general, STEM, code and multilingual tasks. Annotation was routed to *specialist* pools, for example annotators with relevant domain degrees or industry experience for STEM and coding, and fluent speakers for multilingual data. This is the router box of Section 1 made explicit.
- **Rubrics as Rewards (Scale AI, 2025)** [73] uses instance-specific, checklist-style rubrics (built from expert guidance and reference answers) as the reward signal for on-policy RL in domains without verifiable answers, such as medicine and science. An LLM judge scores each rollout against the rubric. Human expertise thus enters RL as a structured specification rather than as per-sample labels.
- **GDPval (OpenAI, 2025)** [74] commissioned tasks from industry professionals (reported average experience of about 14 years) across 44 occupations in the nine sectors contributing most to U.S. GDP. Each task has real deliverables and reference files. Model outputs are graded by blinded expert pairwise comparison against human expert deliverables. A 220-task gold subset was open-sourced from a full set of 1,320 tasks (approximate) [74].

Design implications:

1. **The unit of expert labor is moving from the label to the specification.** One rubric can grade unlimited rollouts. A single bad criterion is therefore replicated everywhere, so rubrics need the same versioning and review as guidelines (Section 5.4).
2. **Model graders need a human calibration slice.** HealthBench's physician meta-evaluation and GDPval's expert pairwise grading are both instances of the gold slice of Section 5.6.
3. **Routing by expertise becomes a first-class platform feature.** Specialist pools, credential checks and per-domain QA replace the generic crowd for these tasks.

---

## 8. Cross-design comparison

| Design | Human role | Selection mechanism | QA mechanism | Public scale (approx.) | Main failure mode |
|---|---|---|---|---|---|
| Classical / deep AL [1]-[6] | Label chosen items | Uncertainty, QBC, BALD, core-set, BADGE | External | Task-dependent | Sampling bias, batch redundancy, outliers |
| Label Studio AL loop [11] | Label lowest-confidence tasks | Uncertainty task sampling via ML backend | Agreement + review [13][14] | N/A | Needs Enterprise orchestration for live loop |
| Snorkel / data programming [16][17] | Write labeling functions | LF analytics on dev set | Label model learns LF accuracy | 2.8x faster than hand labeling (user study) | Correlated LFs, coverage gaps |
| Snorkel DryBell [20] | Wrap org resources as LFs | N/A (programmatic) | Label model at scale | Millions of points in tens of minutes | Non-servable feature leakage |
| Scale Rapid [26] | Vendor workforce | Customer batches | Evaluation (gold) tasks, audits | N/A | Vendor dependence |
| LLM pre-label + route [35][36] | Verify low-confidence labels | Confidence threshold | Calibration on gold | ~$0.003 per LLM label | Miscalibrated confidence, LLM bias |
| OpenAI TL;DR [38] | Pairwise comparisons | Policy samples | Confidence field, split hygiene | 64,832 comparisons | Labeler-researcher disagreement |
| InstructGPT [39][40] | Demos + rankings | API prompts | Screening, shared chat, small team | ~40 contractors | Non-representative raters |
| Anthropic HH / red team [41][42][43] | Chat A/B, adversarial attacks | Weekly online policy samples | Crowd platform selection | 38,961 red-team attacks | Helpful vs harmless tension |
| Llama 2 [44][45] | Vendor preferences + SFT | Weekly on-policy batches | Data checks across vendors | >1M annotations | Vendor variance |
| HelpSteer2 [46] | 5-attribute Likert | Prompt sampling | Multiple annotators | ~10k pairs | Attribute definitions drift |
| PRM800K [52][53] | Step-level expert labels | Best-PRM active selection | QC gold per generation, screening | ~800k step labels | Label cost per step |
| CriticGPT [56] | Rate with AI critic | N/A | Critic raises recall | N/A | Hallucinated critiques |
| HLE [57] | Expert question authors | Adversarial filter vs models | Multi-round expert review | 2,500 questions | Answer errors in hard items |
| Constitutional Classifiers [58] | Paid red-teamers | Human search for jailbreaks | Auto-grader + manual review | 3,000+ h; 339 demo jailbreakers | Over-refusal, compute cost |
| SAM / SAM 2 data engines [65][66] | Click-correct, then audit | Model fills confident masks; humans cover misses | Human-only eval splits | 1.1B masks; ~643K video masklets | Automatic-mask errors at scale |
| PPI / active inference [67][68] | Label a small gold slice | Uncertainty-weighted sampling with known probabilities | Bias rectifier, valid intervals | N/A | Breaks if selection probabilities are not logged |
| Chatbot Arena [69][70] | Public pairwise votes | User prompts; adaptive model-pair sampling | Bradley-Terry, anomalous-user detection | >1.5M votes (reported) | Leaderboard gaming via private variants |
| HealthBench / GDPval [71][74] | Experts write rubrics and tasks | Expert-authored scenarios | Model grader validated against experts | 48,562 rubric criteria; 1,320 tasks | Rubric errors replicate across all grades |

---

## 9. Cross-cutting failure modes

1. **The selector shapes the data distribution.** AL, on-policy RLHF batches, and adversarial filtering all produce non-i.i.d. labeled sets. A random, independently labeled evaluation slice is the only reliable guard (compare the `valid1`/`valid2` split discipline in [38]).
2. **Hard items are where annotators disagree.** Uncertainty-targeted selection concentrates genuinely ambiguous items. Without overlap, adjudication or an "unsure" option (Llama 2's "negligibly better/unsure" grade, PRM800K's neutral step label and `give_up` reason), the loop converts ambiguity into noise.
3. **Rater populations encode values.** InstructGPT's model card and Collective CAI both treat *who labels* as a design parameter [40][59].
4. **Vendors are not interchangeable.** Llama 2 observed markedly different downstream quality across vendors [44]. Ownership changes such as Meta's Scale stake [51] can force supplier changes mid-program.
5. **AI labels fail silently.** UltraFeedback's mis-scored completions [62] show that LLM annotation needs the same audit loop as human annotation.
6. **Expert-level errors need assistance to find.** CriticGPT and the scalable-oversight results [55][56] imply that, past a capability threshold, unassisted human labels stop being ground truth.
7. **Labor conditions affect quality.** Exposure to harmful content without protection [50] increases turnover, which undermines guideline consistency.
8. **Human-preference signals that become targets get gamed.** Public arenas are optimized against by the vendors they rank [70]. Selection policy, disclosure rules and data access are part of the loop's validity, not administrative details.
9. **Specifications amplify expert errors.** One wrong rubric criterion or reference answer is applied to every rollout it grades [71][73], so expert-written specifications need review and versioning at least as strict as labels.

---

## 10. Key takeaways for system designers

1. **Budget human attention like compute.** Decide explicitly which items get human eyes: uncertainty, disagreement, policy errors, or random audit. Log why each item was selected.
2. **Use diversity-aware batch acquisition.** Top-k uncertainty with real batch sizes wastes labels on near-duplicates. Power sampling [6] is the cheapest fix; BADGE or core-set [4][5] are the principled ones.
3. **Always keep a random evaluation slice.** Label it independently of the selector and keep it out of training. Never evaluate on AL-selected data alone.
4. **Move humans up a level of abstraction where possible.** Labeling functions [17], wrapped organizational resources [20], guidelines and constitutions [59] get re-applied automatically when definitions change. Hand labels do not.
5. **Treat guidelines as versioned code.** Record guideline version, pre-labeling model version, worker id, batch and confidence with every label [38][53].
6. **Instrument QA as a closed sub-loop.** Use hidden gold items per batch [26][53], screening sets, per-control-tag agreement [13], reviewer reject routing [14], and statistical aggregation instead of majority vote [29][30].
7. **Run label-error detection after QA.** Confident learning and prediction-versus-label disagreement sorted by confidence [32][34] catch errors that consensus misses, including in test sets [33].
8. **Pre-label with models and route by calibrated confidence.** Let LLMs or the current model label the bulk, and send low-confidence items to humans [36]. Re-validate the calibration whenever the labeling model changes.
9. **For LLM feedback, sample on-policy and target convincing errors.** Weekly refreshes [41][44], best-PRM selection of wrong-but-plausible solutions [52][53], and uncertainty-driven pair selection [60][61] all extract more value per human judgment.
10. **Give expert raters AI assistance, and measure its effect.** Critics raise human recall [54][56]. Track both the bugs found and the false critiques accepted.
11. **Diversify and audit vendors.** Run identical calibration batches across suppliers and compare downstream model quality, not just agreement [44].
12. **Report model-labeled metrics with corrected intervals.** Combine a small, probabilistically sampled human slice with model labels through prediction-powered or active inference [67][68], instead of trusting an LLM judge's raw numbers.
13. **Schedule the human-to-model labor ratio.** Start with model-assisted manual labeling, move to model-prefilled plus human gap-filling, then to automatic labeling with human audits, as in the SAM data engines [65][66]. Keep evaluation splits fully human-labeled throughout.
14. **Ask experts for specifications, not just labels.** Rubrics, reference tasks and specialist routing [71][72][73][74] let scarce expert time scale. Calibrate the model grader against expert judgments before trusting it.
15. **Design for the people in the loop.** Small, high-bandwidth teams [40], clear escalation channels, fair pay and exposure limits [50] produce more consistent labels than large anonymous crowds on hard, value-laden tasks.

---

## References

1. Active Learning Literature Survey (Computer Sciences Technical Report 1648). B. Settles, University of Wisconsin-Madison, 2009 (rev. 2010). https://burrsettles.com/pub/settles.activelearning.pdf
2. Deep Bayesian Active Learning with Image Data. Y. Gal, R. Islam, Z. Ghahramani (University of Cambridge), 2017. https://arxiv.org/abs/1703.02910
3. BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning. A. Kirsch, J. van Amersfoort, Y. Gal (University of Oxford), 2019. https://arxiv.org/abs/1906.08158
4. Active Learning for Convolutional Neural Networks: A Core-Set Approach. O. Sener, S. Savarese (Stanford), 2017. https://arxiv.org/abs/1708.00489
5. Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (BADGE). J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, A. Agarwal, 2019. https://arxiv.org/abs/1906.03671
6. Stochastic Batch Acquisition for Deep Active Learning. A. Kirsch, S. Farquhar, P. Atighehchian, A. Jesson, F. Branchaud-Charron, Y. Gal, 2021. https://arxiv.org/abs/2106.12059
7. Bayesian Active Learning for Production, a Systematic Study and a Reusable Library (Baal). P. Atighehchian, F. Branchaud-Charron, A. Lacoste (ElementAI), 2020; library at https://github.com/baal-org/baal. https://arxiv.org/abs/2006.09916
8. modAL: A Modular Active Learning Framework for Python 3. T. Danka et al., 2018 to present. https://github.com/modAL-python/modAL
9. small-text: Active Learning for Text Classification in Python. C. Schröder et al. (Webis), 2021 to 2026. https://github.com/webis-de/small-text
10. Reassessing Active Learning Adoption in Contemporary NLP: A Community Survey (EACL 2026). Small-text authors (Webis et al.), 2025 to 2026. https://arxiv.org/abs/2503.09701
11. Set up active learning with Label Studio (documentation source). HumanSignal, 2021 to 2026. https://github.com/HumanSignal/label-studio/blob/develop/docs/source/guide/active_learning.md
12. Label Studio ML backend (SDK and model catalog). HumanSignal, 2020 to 2026. https://github.com/HumanSignal/label-studio-ml-backend
13. Task agreement in Label Studio Enterprise (documentation source). HumanSignal, 2026. https://github.com/HumanSignal/label-studio/blob/develop/docs/source/guide/stats.md
14. Review annotations in Label Studio (documentation source). HumanSignal, 2026. https://github.com/HumanSignal/label-studio/blob/develop/docs/source/guide/quality.md
15. Prodigy Recipes. Explosion AI, 2018 to present. https://github.com/explosion/prodigy-recipes
16. Data Programming: Creating Large Training Sets, Quickly. A. Ratner, C. De Sa, S. Wu, D. Selsam, C. Ré (Stanford), NeurIPS 2016. https://arxiv.org/abs/1605.07723
17. Snorkel: Rapid Training Data Creation with Weak Supervision. A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, C. Ré (Stanford), VLDB 2018. https://arxiv.org/abs/1711.10160
18. Training Complex Models with Multi-Task Weak Supervision (Snorkel MeTaL). A. Ratner, B. Hancock, J. Dunnmon, F. Sala, S. Pandey, C. Ré, AAAI 2019; repository at https://github.com/HazyResearch/metal. https://arxiv.org/abs/1810.02840
19. Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods (FlyingSquid). D. Y. Fu, M. F. Chen, F. Sala, S. M. Hooper, K. Fatahalian, C. Ré, ICML 2020; code at https://github.com/HazyResearch/flyingsquid. https://arxiv.org/abs/2002.11955
20. Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale. S. H. Bach, D. Rodriguez, Y. Liu, et al. (Google, Stanford), SIGMOD Industry 2019. https://arxiv.org/abs/1812.00417
21. Harnessing Organizational Knowledge for Machine Learning. Google AI Blog (now Google Research Blog), 2019. https://research.google/blog/harnessing-organizational-knowledge-for-machine-learning/
22. Snorkel (repository README and project announcement). Snorkel team, 2019 to present. https://github.com/snorkel-team/snorkel
23. Snorkel Tutorials (including drybell and crowdsourcing examples). Snorkel team, 2019 to present. https://github.com/snorkel-team/snorkel-tutorials
24. WRENCH: A Comprehensive Benchmark for Weak Supervision. J. Zhang, Y. Yu, Y. Li, Y. Wang, Y. Yang, M. Yang, A. Ratner, NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2109.11377
25. A Survey on Programmatic Weak Supervision. J. Zhang, C.-Y. Hsieh, Y. Yu, C. Zhang, A. Ratner, 2022. https://arxiv.org/abs/2202.05433
26. Scale AI Python SDK (tasks, batches, audits, evaluation and training tasks). Scale AI, 2020 to present. https://github.com/scaleapi/scaleapi-python-client
27. Labelbox Python SDK and official SDK notebooks. Labelbox, 2019 to present. https://github.com/Labelbox/labelbox-python and https://github.com/Labelbox/labelbox-notebooks
28. Argilla: collaboration tool for AI engineers and domain experts. Argilla / Hugging Face, 2021 to present. https://github.com/argilla-io/argilla
29. Crowd-Kit: Computational Quality Control for Crowdsourcing. D. Ustalov et al. (Toloka), JOSS 2024. https://github.com/Toloka/crowd-kit
30. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. A. P. Dawid, A. M. Skene, Journal of the Royal Statistical Society Series C, 1979. https://doi.org/10.2307/2346806
31. Cheap and Fast, But is it Good? Evaluating Non-Expert Annotations for Natural Language Tasks. R. Snow, B. O'Connor, D. Jurafsky, A. Ng (Stanford), EMNLP 2008. https://aclanthology.org/D08-1027/
32. Confident Learning: Estimating Uncertainty in Dataset Labels. C. G. Northcutt, L. Jiang, I. L. Chuang (MIT), JAIR 2021. https://arxiv.org/abs/1911.00068
33. Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. C. G. Northcutt, A. Athalye, J. Mueller, NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2103.14749
34. cleanlab: data-centric AI package for data and label issues. Cleanlab, 2018 to present. https://github.com/cleanlab/cleanlab
35. ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks. F. Gilardi, M. Alizadeh, M. Kubli (University of Zurich), PNAS 2023. https://arxiv.org/abs/2303.15056
36. Autolabel: label, clean and enrich text datasets with LLMs. Refuel AI, 2023 to present. https://github.com/refuel-ai/autolabel
37. Learning to Summarize from Human Feedback. N. Stiennon, L. Ouyang, J. Wu, et al. (OpenAI), 2020. https://arxiv.org/abs/2009.01325
38. summarize-from-feedback (code and human feedback dataset documentation). OpenAI, 2020. https://github.com/openai/summarize-from-feedback
39. Training Language Models to Follow Instructions with Human Feedback (InstructGPT). L. Ouyang, J. Wu, X. Jiang, et al. (OpenAI), 2022. https://arxiv.org/abs/2203.02155
40. following-instructions-human-feedback: InstructGPT model card and labeling instructions. OpenAI, 2022. https://github.com/openai/following-instructions-human-feedback
41. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. Y. Bai, A. Jones, K. Ndousse, et al. (Anthropic), 2022. https://www.anthropic.com/research/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback
42. Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned. D. Ganguli, L. Lovitt, J. Kernion, et al. (Anthropic), 2022. https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned
43. hh-rlhf: Human preference data about helpfulness and harmlessness, and red teaming data (repository archived; mirrored as the Anthropic/hh-rlhf Hugging Face dataset). Anthropic, 2022. https://github.com/anthropics/hh-rlhf
44. Llama 2: Open Foundation and Fine-Tuned Chat Models. H. Touvron, L. Martin, K. Stone, et al. (Meta), 2023. https://arxiv.org/abs/2307.09288
45. Llama 2 Model Card. Meta, 2023. https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md
46. HelpSteer2: Open-source Dataset for Training Top-performing Reward Models. Z. Wang, Y. Dong, O. Delalleau, et al. (NVIDIA), 2024. https://arxiv.org/abs/2406.08673
47. NeMo-Aligner: scalable toolkit for model alignment (SteerLM, DPO, RLHF; Nemotron-4 340B Reward). NVIDIA, 2023 to 2025 (archived May 2025; successor NeMo RL). https://github.com/NVIDIA/NeMo-Aligner
48. OpenAssistant Conversations: Democratizing Large Language Model Alignment. A. Köpf, Y. Kilcher, D. von Rütte, et al. (LAION), 2023; project repository at https://github.com/LAION-AI/Open-Assistant. https://arxiv.org/abs/2304.07327
49. Open-Assistant Guidelines (data collection guidelines). LAION, 2023. https://github.com/LAION-AI/Open-Assistant/blob/main/docs/docs/guides/guidelines.md
50. Exclusive: OpenAI Used Kenyan Workers on Less Than $2 Per Hour to Make ChatGPT Less Toxic. B. Perrigo, TIME, 2023. https://time.com/6247678/openai-chatgpt-kenya-workers/
51. Meta invests $14B in Scale AI, acquires 49% stake and superintelligence lab leader. The Decoder, 2025. https://the-decoder.com/meta-invests-14b-in-scale-ai-acquires-49-stake-and-superintelligence-lab-leader/
52. Let's Verify Step by Step. H. Lightman, V. Kosaraju, Y. Burda, et al. (OpenAI), 2023. https://arxiv.org/abs/2305.20050
53. PRM800K: A Process Supervision Dataset (data schema, labeler instructions, QC). OpenAI, 2023. https://github.com/openai/prm800k
54. Self-critiquing Models for Assisting Human Evaluators. W. Saunders, C. Yeh, J. Wu, S. Bills, L. Ouyang, J. Ward, J. Leike (OpenAI), 2022. https://arxiv.org/abs/2206.05802
55. Measuring Progress on Scalable Oversight for Large Language Models. S. R. Bowman, J. Hyun, E. Perez, et al. (Anthropic), 2022. https://www.anthropic.com/research/measuring-progress-on-scalable-oversight-for-large-language-models
56. LLM Critics Help Catch LLM Bugs (CriticGPT). N. McAleese, R. Pokorny, J. F. Cerón Uribe, E. Nitishinskaya, M. Trebacz, J. Leike (OpenAI), 2024. https://arxiv.org/abs/2407.00215
57. Humanity's Last Exam. L. Phan, A. Gatti, Z. Han, N. Li, et al. (Center for AI Safety and Scale AI), 2025; repository at https://github.com/centerforaisafety/hle. https://arxiv.org/abs/2501.14249
58. Constitutional Classifiers: Defending against Universal Jailbreaks. Anthropic, 2025. https://www.anthropic.com/research/constitutional-classifiers
59. Collective Constitutional AI: Aligning a Language Model with Public Input. Anthropic and the Collective Intelligence Project, 2023. https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input
60. Efficient Exploration for LLMs. V. Dwaracherla, S. M. Asghari, B. Hao, B. Van Roy (Google DeepMind), 2024. https://arxiv.org/abs/2402.00396
61. Active Preference Learning for Large Language Models. W. Muldrew, P. Hayes, M. Zhang, D. Barber (UCL), 2024. https://arxiv.org/abs/2402.08114
62. UltraFeedback: Boosting Language Models with Scaled AI Feedback (paper and dataset README including re-annotation notice). G. Cui, L. Yuan, N. Ding, et al. (Tsinghua / OpenBMB), 2023; repository at https://github.com/OpenBMB/UltraFeedback. https://arxiv.org/abs/2310.01377
63. AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. Y. Dubois, X. Li, R. Taori, et al. (Stanford), 2023; code at https://github.com/tatsu-lab/alpaca_farm. https://arxiv.org/abs/2305.14387
64. Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision. C. Burns, P. Izmailov, J. H. Kirchner, et al. (OpenAI), 2023. https://arxiv.org/abs/2312.09390
65. Segment Anything. A. Kirillov, E. Mintun, N. Ravi, et al. (Meta AI), ICCV 2023; repository at https://github.com/facebookresearch/segment-anything. https://arxiv.org/abs/2304.02643
66. SAM 2: Segment Anything in Images and Videos. N. Ravi, V. Gabeur, Y.-T. Hu, et al. (Meta), 2024; repository and SA-V dataset documentation at https://github.com/facebookresearch/sam2 and https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md. https://arxiv.org/abs/2408.00714
67. Prediction-Powered Inference. A. N. Angelopoulos, S. Bates, C. Fannjiang, M. I. Jordan, T. Zrnic, Science, 2023; library at https://github.com/aangelopoulos/ppi_py. https://arxiv.org/abs/2301.09633
68. Active Statistical Inference. T. Zrnic, E. J. Candès, ICML 2024. https://arxiv.org/abs/2403.03208
69. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. W.-L. Chiang, L. Zheng, Y. Sheng, et al. (LMSYS), ICML 2024; vote counts from https://github.com/lm-sys/FastChat. https://arxiv.org/abs/2403.04132
70. The Leaderboard Illusion. S. Singh, Y. Nan, A. Wang, et al. (Cohere Labs and collaborators), 2025. https://arxiv.org/abs/2504.20879
71. HealthBench: Evaluating Large Language Models Towards Improved Human Health. R. K. Arora, J. Wei, R. Soskin Hicks, et al. (OpenAI), 2025; reference implementation at https://github.com/openai/simple-evals/blob/main/healthbench_eval.py. https://arxiv.org/abs/2505.08775
72. HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages. Z. Wang, J. Zeng, O. Delalleau, et al. (NVIDIA), 2025. https://arxiv.org/abs/2505.11475
73. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. A. Gunjal, A. Wang, E. Lau, et al. (Scale AI), 2025. https://arxiv.org/abs/2507.17746
74. GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks. T. Patwardhan, R. Dias, E. Proehl, et al. (OpenAI), 2025. https://arxiv.org/abs/2510.04374
