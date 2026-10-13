# 01. Foundations: Feedback Loops, Technical Debt, and the Anatomy of Closed-Loop ML Systems

## Overview

Most machine learning textbooks describe an open-loop pipeline. A fixed dataset is drawn from a fixed distribution, a model is fit, it is evaluated on a held-out split, and the result is shipped. Production systems almost never behave that way. Once a model's outputs drive actions (ranking a feed, placing an ad, dispatching a patrol car, steering a vehicle, completing a line of code), those actions change what the system observes next, and that observation becomes the next training set. The model sits inside a loop.

This chapter sets up the vocabulary and the core designs that the rest of the Atlas builds on. It covers:

1. The distinction between **open-loop** and **closed-loop** learning systems, and a taxonomy of the loops found in practice.
2. The software-engineering view of feedback from Google's **"Hidden Technical Debt in Machine Learning Systems"** [1] and its follow-ups [2][3].
3. The statistical and social failure modes of loops: **degenerate feedback loops**, echo chambers and filter bubbles [4][5][7][8], runaway loops in high-stakes settings [6], and bias in logged data [18].
4. The learning-theoretic framework of **performative prediction**, in which a deployed model shifts the distribution it is evaluated on [9][10][11][12].
5. **Recursive data-generation loops**, in which generative models train on their own outputs [14][15].
6. The **data flywheel**, the deliberate and positive use of the loop, as built by Tesla [24][25][26], NVIDIA [27][28], Cursor [29], Microsoft [17] and ByteDance [23], with OpenAI's 2025 sycophancy incident [30] as a cautionary case.
7. A **reference architecture** for a closed-loop learning system, assembled from published designs [16][17][19][20][22][23][27].

One idea runs through the whole chapter. A closed loop is the most powerful tool an ML system has for adapting to a changing world, and it is also the most common source of invisible, compounding failure. The engineering discipline is to make the loop explicit, instrumented and controllable.

## 1. Open-loop vs closed-loop learning systems

### 1.1 Definitions

The terms come from control theory. In an **open-loop** controller, the action is computed from a reference signal and is never corrected by observing its effect. In a **closed-loop** (feedback) controller, the output is measured and fed back to adjust later actions. ML systems map onto this in two distinct ways, and practitioners often conflate them:

- **Closed loop as a design goal.** The system intentionally captures outcomes such as clicks, labels, interventions, accept/reject signals and thumbs-up ratings, and uses them to retrain or update the model. Google's MLOps guidance calls this *continuous training* [19]. Chip Huyen calls it *continual learning* [20][21]. Microsoft's Decision Service calls it the *explore-log-learn-deploy* loop [17].
- **Closed loop as an unintended side effect.** The model's predictions influence the data it will later be trained on, whether or not anyone designed for it. Sculley et al. call these *feedback loops*, and they warn that the hidden ones are the most dangerous [1]. Perdomo et al. formalize the effect as *performativity* [9].

A system can be open-loop by design yet closed-loop in effect. A credit model that is retrained once a year on repayment data only covers applicants it approved. That is a hidden loop, even though nobody built a "feedback pipeline".

### 1.2 Taxonomy of loops

The following loop types recur throughout the literature. Most production systems contain several at once.

| Loop type | What flows back | Typical latency | Canonical source |
|---|---|---|---|
| Retraining loop | Logged inputs and outcomes become the next training set | Hours to months | Google MLOps CT [19]; Huyen [20] |
| Online update loop | Streaming outcomes update parameters in place | Seconds to minutes | Monolith [23]; Decision Service [17] |
| Action/selection loop | The model chooses what is shown or acted on, so it chooses which labels exist | Same as retraining loop | Jiang et al. [4]; Chen et al. [18]; Ensign et al. [6] |
| Behavior/preference loop | Users' preferences shift because of what they were shown | Weeks to years | Jiang et al. [4]; Chaney et al. [5] |
| Strategic/performative loop | Agents change features or outcomes in response to the model | Varies | Perdomo et al. [9]; Hardt & Mendler-Dünner [11] |
| Inter-system (hidden) loop | Two separately owned models influence each other via the world | Varies | Sculley et al. [1] |
| Data-generation loop | Model outputs pollute the corpus that future models train on | Model generations | Shumailov et al. [14]; Alemohammad et al. [15] |
| Human-feedback reward loop | Ratings of model outputs become a reward signal | Release cycles | OpenAI sycophancy postmortem [30]; Cursor [29] |

### 1.3 A generic loop diagram

```
                 +---------------------------- world ----------------------------+
                 |  users, drivers, advertisers, police districts, the web ...   |
                 +-------^-----------------------------------------+-------------+
                         | actions / exposures                     | outcomes
                         |                                         v
   +---------+     +-----+------+     +-----------+     +-----------------------+
   | model   +---->+  policy /  |     |  logging  +<----+ observation (labels,  |
   | f_theta |     |  decision  +---->+ (features,|     | clicks, interventions,|
   +----^----+     |  layer     |     | propensity|     | thumbs, accept/reject)|
        |          +------------+     | , action) |     +-----------------------+
        |                             +-----+-----+
        |   new theta                       | training data
   +----+-----------+   eval/gate     +-----v------+
   | deploy/rollout +<----------------+  learner   |
   +----------------+                 +------------+

   Open loop: delete the arrows from "world" back to "learner".
   Closed loop: every arrow present; theta(t+1) depends on theta(t) via the world.
```

The key fact the diagram makes visible is that **theta(t+1) is a function of theta(t)**. Everything in sections 3 to 5 follows from this dependence.

## 2. Hidden Technical Debt in ML Systems (Google, 2015)

### 2.1 Problem

Sculley and nine co-authors at Google argued in a NeurIPS 2015 paper that ML lets teams ship quick wins while running up "massive ongoing maintenance costs" that conventional software-engineering tools do not reveal [1]. The paper lists ML-specific risk factors: boundary erosion, entanglement, **hidden feedback loops**, undeclared consumers, data dependencies, configuration issues, changes in the external world, and system-level anti-patterns [1]. Its best-known figure shows that the ML code in a real-world system is a small box surrounded by far larger infrastructure: data collection, feature extraction, data verification, configuration, serving, monitoring, and process and resource management [1].

### 2.2 Feedback-loop debt

The paper separates two kinds of loop [1]:

- **Direct feedback loops.** The model directly influences which data it will later be trained on, for example by choosing which items are shown. The authors note that bandit algorithms are the principled answer but often do not scale to the action spaces used in practice. They describe direct loops as costly to analyze but "at least they pose a statistical challenge that ML researchers may find natural to investigate" [1].
- **Hidden feedback loops.** Two or more systems influence each other through the world without either team knowing. The paper's examples include two independently developed models on the same web page, one choosing products and one choosing related reviews, where improving one shifts user behavior that the other trains on. It also cites independent investment models whose trades move the prices that both learn from [1]. These loops are hard to detect because no single owner sees the whole cycle.

### 2.3 Related debt that amplifies loops

Several other items in the paper interact directly with feedback [1]:

- **Entanglement and CACE ("Changing Anything Changes Everything").** Because input signals are mixed, a change to one feature, hyperparameter or data source alters the whole model. In a loop, a small change also alters the next round's data.
- **Correction cascades.** A model m'a is trained on the output of model ma to fix a slightly different problem. This creates a chain in which improving ma can degrade m'a. Cascades are loops waiting to happen.
- **Undeclared consumers.** Other systems silently consume a model's predictions. If one of them influences the model's own inputs, a hidden feedback loop is created.
- **Changes in the external world.** The paper recommends monitoring **prediction bias** (the distribution of predicted labels should match observed labels), enforcing **action limits** (hard caps on actions a system can take, as a sanity check), and watching **upstream producers** [1].

### 2.4 Follow-ups: ML Test Score and Rules of ML

Breck, Cai, Nielsen, Salib and Sculley turned these concerns into **The ML Test Score**. It is a rubric of 28 actionable tests in four groups (data, model, ML infrastructure and monitoring), with a score that measures production readiness [2]. Monitoring for distribution shift and for prediction-quality regressions is how a team notices that a loop has drifted.

Martin Zinkevich's **Rules of Machine Learning** includes loop-specific advice [3]:

- **Rule #29:** log the features actually used at serving time and train on them (at least for a sample of traffic) to remove training/serving skew. This makes the serving log, which is the input to the loop, the source of truth.
- **Rule #36: "Avoid feedback loops with positional features."** Items in the first position are clicked more, so a model without position features "will be convinced" those items are better. The fix is to train with positional features and then, at serving time, omit them or set them to a shared default, because candidates are scored before their order is known. Positional features should be kept "somewhat separate from the rest of the model" [3].

### 2.5 Lessons

- A feedback loop is **a dependency on your own past outputs**. It belongs in the dependency graph just as an upstream table does.
- A hidden loop is an **organizational** failure as much as a statistical one. It spans team boundaries, so it needs cross-team visibility: declared consumers, lineage and shared exposure logs.
- Sanity limits (action limits, prediction-bias monitors) are cheap guards against a loop that runs away.

## 3. Degenerate feedback loops

### 3.1 Definition and mechanism

A **degenerate feedback loop** occurs when a model's outputs restrict the data it later observes, so that the loop converges to a narrow, self-confirming state rather than to the truth. Huyen's *Designing Machine Learning Systems* treats this as a core production failure mode in recommender and ranking systems. Popular items get more exposure, so they get more clicks, so they are recommended more [21].

```
   recommend top-k by predicted CTR
              |
              v
   exposure concentrated on head items ---> clicks observed only on exposed items
              ^                                         |
              |                                         v
   model learns "head items are best" <--- training set skewed toward head items
```

### 3.2 Jiang et al. (DeepMind, AIES 2019): echo chambers and filter bubbles

**Problem.** Recommendations shape the data the system later learns from, which can produce echo chambers or filter bubbles [4].

**Model.** Ray Jiang, Silvia Chiappa, Tor Lattimore, András György and Pushmeet Kohli model a user's interest as a dynamical system driven by recommendations, and treat interest extremes as **degeneracy points** [4]. They separate two effects:

- An **echo chamber** is the user's interest being positively or negatively reinforced by repeated exposure.
- A **filter bubble** is the narrowing of the candidate set of items the system shows.

They define system degeneracy formally and give sufficient conditions for degeneracy under deterministic and stochastic dynamics [4].

**Findings.** The paper studies three design factors: model accuracy, amount of exploration, and candidate-pool growth rate [4].

- An **oracle model** (perfectly accurate on current interests) often leads to *faster* degeneracy. Accuracy alone does not protect against the loop and can speed it up.
- **Continuous random exploration** and **growing the candidate pool at least linearly** were the most effective ways to slow degeneracy.
- A simulation study with synthetic data and several classic bandit algorithms backs the theory [4].

### 3.3 Chaney, Stewart & Engelhardt (RecSys 2018): algorithmic confounding

Chaney et al. observe that recommender training and test data are **algorithmically confounded**. Both are "the result of a feedback loop between human choices and an existing algorithmic recommendation system" [5]. Their simulations show three effects of confounding [5]:

1. It can put algorithms at a disadvantage during training.
2. It **biases held-out evaluation toward recommenders similar to the confounding (production) algorithm.** This means offline A/B selection tends to favor more of the same.
3. It amplifies **homogenization** of user behavior without gains in utility. Averaged accuracy metrics do not reveal disparate impact across users.

The second point matters a great deal for system design. **Offline evaluation on logged data from a closed loop is not neutral.** It is tilted toward the incumbent.

### 3.4 Khritankov: existence conditions for hidden loops

Anton Khritankov's work treats loops as a measurable property of a deployed system:

- A simulation model for supervised ML systems splits automation bias into **user adherence to predictions** and **usage rate**, and estimates the resulting concept drift [8].
- Khritankov and Pilkevich (2021) study multi-armed bandit recommenders. They show that loops degrade recommendations to a small subset of items with lost coverage and novelty. Random noise in user interests does *not* break the loop, but a non-zero chance of *resetting* user interests limits it. Simulations with four bandit algorithms support this [7].

### 3.5 Ensign et al. (FAT* 2018): runaway loops in predictive policing

**Problem.** Predictive policing tools learn from incidents that are discovered by the officers they dispatched, so the data reflects where police already went [6].

**Mechanism.** Using Pólya-urn models, Ensign, Friedler, Neville, Scheidegger and Venkatasubramanian show formally and empirically why the loop runs away. Allocation proportional to past discovered incidents pushes resources toward regions with an initial advantage, regardless of the true crime rate. They link the severity of the loop to the disparity in crime rates between areas [6].

**Mitigation.** Reported incidents (crimes called in by residents rather than discovered by patrols) can *attenuate* the runaway effect but cannot remove it. The authors show that changing the inputs to the black-box predictor, by filtering which discovered incidents are fed back, can stop the loop so that the true rate can be learned [6]. The general lesson is that **feedback should be weighted by how likely it was to be observed under the current policy.**

### 3.6 Correcting for selection bias in logged feedback: YouTube REINFORCE

Chen et al. (Google/YouTube, WSDM 2019) address the action/selection loop directly. Logged implicit feedback such as clicks and dwell time is plentiful, but "learning from the logged feedback is however subject to biases caused by only observing feedback on recommendations selected by the previous versions of the recommender" [18]. Their production top-K recommender uses REINFORCE with:

- **Off-policy correction** by importance weighting. A learned model estimates the behavior policy (in fact several historical policies), and gradients are reweighted by the ratio of new to behavior policy.
- A **top-K correction** for slates, because several items are shown at once.

They report gains in simulation and in live YouTube experiments [18]. This is the canonical industrial example of treating the logged loop as a counterfactual estimation problem. Bottou et al. made the same move in 2013 for Bing ad placement, using causal inference on randomized logs to predict the effect of changes to a system that interacts with its environment [16].

### 3.7 Recent simulation work (2025-2026)

The topic remains active. A 2025 simulation framework built on Amazon e-commerce data finds that the recommender loop **broadens individual purchase profiles while concentrating aggregate demand** [32]. A 2026 follow-up, "The Diversity Paradox Revisited", reports that apparent gains in individual diversity in *static* evaluations do not hold once the loop is simulated over time [33]. Both papers support the point from Chaney et al.: single-snapshot metrics mislead in closed loops. Google's RecSim gives a configurable platform for this kind of sequential, user-state-aware simulation [31].

### 3.8 Detection and mitigation in practice

Combining the sources gives a practical toolkit:

| Technique | What it addresses | Source |
|---|---|---|
| Continuous random exploration (epsilon, Thompson, UCB) | Filter bubble; unobserved counterfactuals | Jiang et al. [4]; Decision Service [17] |
| Grow candidate pool at least linearly | Filter bubble | Jiang et al. [4] |
| Log action propensities; importance-weight training | Selection bias from previous policies | Chen et al. [18]; Bottou et al. [16]; Decision Service [17] |
| Positional features at train time, defaulted at serve time | Position bias loop | Rules of ML #36 [3] |
| Weight or filter feedback by observation probability | Runaway allocation loops | Ensign et al. [6] |
| Measure popularity or diversity concentration over time | Detecting degeneracy | Huyen [21]; [32][33] |
| Simulate long-horizon dynamics before launch | Preference shift, homogenization | Chaney et al. [5]; RecSim [31] |
| Hold out a randomized slice of traffic | Unbiased evaluation data | Decision Service [17]; Bottou et al. [16] |

## 4. Performative prediction

### 4.1 Problem

Section 3 treats loops as a bias to be corrected. **Performative prediction** treats them as a fundamental property of the learning problem. Perdomo, Zrnic, Mendler-Dünner and Hardt (ICML 2020) start from the observation that predictions can change the outcomes they forecast. They argue that, when ignored, this "manifests as undesirable distribution shift", which is typically handled by retraining [9]. The authors' code repository gives intuitive examples: "Traffic predictions influence traffic patterns, crime location prediction influences police allocations that may deter crime, recommendations shape preferences and thus consumption, stock price prediction determines trading activity and hence prices" [13].

### 4.2 Framework

The framework has the following elements [9][11]:

- A **distribution map** D(theta) gives the data distribution that results from deploying model parameters theta.
- **Performative risk:** PR(theta) = E over z drawn from D(theta) of loss(z; theta). The model is judged on the distribution it *induces*.
- **Performative optimality:** theta_PO minimizes PR(theta). This is the best model, taking its own influence into account.
- **Performative stability:** theta_PS is optimal for the distribution it itself induces, so theta_PS = argmin over theta of E over z drawn from D(theta_PS) of loss(z; theta). Retraining at a stable point returns the same point. It is an equilibrium, and in general it differs from the optimum.
- **Repeated risk minimization (RRM)** is exactly what a periodic retraining pipeline does: theta(t+1) = argmin over theta of E over z drawn from D(theta(t)) of loss.

### 4.3 Main results

- If the loss is **smooth and strongly convex** and D(.) is **sufficiently Lipschitz** (epsilon-sensitive), then RRM converges to a performatively stable point at a **linear rate**. If any of these assumptions fails, RRM can fail to converge at all [9]. The convergence regime is usually stated as epsilon < gamma/beta, where gamma is the strong-convexity constant and beta is the smoothness constant. Follow-up work notes that the original result for repeated *gradient descent* was stated with the more conservative bound epsilon < gamma/(gamma+beta). Readers should check the exact theorem they rely on [10].
- Stable points have nearly minimal performative risk under these conditions, so "retrain until it stops changing" is close to optimal in the benign regime [9].
- **Strategic classification**, where individuals game a classifier, is a special case. The paper gives the first guarantee for repeated risk minimization in the strategic setting [9].
- Mendler-Dünner, Perdomo, Zrnic and Hardt (NeurIPS 2020) extend this to **stochastic** optimization, covering online updates as well as full retraining [10].

### 4.4 Learning vs steering, and performative power

Hardt and Mendler-Dünner's survey "Performative Prediction: Past and Future" (*Statistical Science*, 2025) separates two regimes [11]:

- **Learning:** the deployer tries to predict well despite performativity (stability or optimality).
- **Steering:** the deployer uses performativity to move the population toward a preferred outcome.

The survey also discusses **performative power**: how much a platform can change participants' behavior through its predictions. The authors present it as a measure relevant to questions of market power and digital-platform regulation [11]. A 2026 survey, "Dissecting Performative Prediction", catalogs the rapidly growing literature [12].

### 4.5 Implications for system designers

1. **Periodic retraining is an algorithm with a convergence theory, not a neutral maintenance chore.** Whether your retrain loop settles, oscillates or diverges depends on how strongly your deployment shifts the data (epsilon) relative to how well-conditioned your loss is.
2. **"The model's accuracy dropped after launch" may be performativity rather than ordinary drift.** The fix may be a dampened update (smaller steps, mixing in older data) rather than more frequent retraining.
3. **Stable is not optimal.** A system that has converged under RRM may still be far from the best achievable policy. Reaching optimality requires modeling D(theta), for example through randomized experiments that vary theta.
4. The more market share a platform has, the larger its performative power. **Scale makes loops stronger.**

## 5. Recursive data-generation loops (model collapse)

Generative models create a loop at internet scale. Their outputs become part of the corpus that future models are trained on.

**Shumailov et al. (Nature, 2024).** Training on model-generated content causes "irreversible defects in the resulting models, in which tails of the original content distribution disappear". This is **model collapse**, observed in LLMs, variational autoencoders and Gaussian mixture models [14]. The mechanism is compounding estimation error. Rare events are under-sampled in each generation, so each successive model sees a narrower distribution. In one language-model experiment, text about medieval architecture degenerated by the ninth generation into a list of jackrabbits [14]. The authors conclude that training on generated data is not impossible but requires careful filtering, and that genuine human interaction data becomes more valuable as generated content spreads [14].

**Alemohammad et al., "Self-Consuming Generative Models Go MAD" (ICLR 2024).** This paper names the effect **Model Autophagy Disorder**. It shows that without enough *fresh real data* in each generation of a self-consuming loop, the quality or diversity of generated output progressively declines [15].

**Design implication.** Any flywheel that feeds model outputs back as training data, such as synthetic data, self-distillation, or LLM-judged "ground truth", is a recursive loop. It needs (a) a protected, continually refreshed stream of real data, (b) provenance tags that separate human-originated from model-originated samples, and (c) tail-coverage monitoring rather than average-quality metrics.

## 6. The data flywheel

### 6.1 The concept

A **data flywheel** deliberately designs the loop so that usage generates data, data improves the model, and a better model drives more usage. NVIDIA's blueprint gives a concrete definition: "a process that uses data exhaust from production applications (for example, LLM prompt/response logs, end user feedback, and expert labeling) to increase the overall accuracy and reduce the latency/cost of Generative AI systems" [27]. The flywheel is the positive mirror of the degenerate loop. It has the same topology, and the difference lies in **which data is selected and how it is weighted**.

### 6.2 Tesla's data engine (2016 onward)

**Problem.** Autonomous-driving perception must handle a long tail of rare scenarios that cannot be enumerated in advance.

**Architecture.** Public descriptions come mainly from talks by Andrej Karpathy and from press coverage, so the details should be treated as approximate:

- **Shadow mode.** Since 2016, when Autopilot is not controlling the car, it runs in parallel with the human driver. Divergence between the model and the driver can trigger the upload of a short "snapshot" of camera, speed and acceleration data [24].
- **Triggers.** Hand-written trigger classifiers run on the fleet to flag interesting events. A CVPR 2021 talk summary reports **221 triggers** [25], and other sources report "over 200" [24].
- **Labeling.** Labeling moved from a large human team toward **offline auto-labeling** with much heavier models than could run in the car, followed by human cleanup [25][26]. In one reported case, 10,000 poor-visibility clips were auto-labeled in about a week, which Tesla said would have taken months by hand (approximate, secondary source) [25].
- **Iterate.** Mispredictions become unit tests, similar examples are mined from the fleet, the model is retrained and redeployed in shadow mode. One summary reports that a dataset went through the loop **seven times** for a single release (approximate) [25].

```
 fleet (shadow mode) --trigger fires--> snapshot upload --> auto-label (big offline nets)
        ^                                                          |
        |                                                   human QA / correction
        |                                                          v
 deploy new net <-- eval + unit tests on hard cases <-- retrain <-- dataset
```

**Lessons.** The flywheel's value comes from **targeted collection of failures**, not from data volume. Triggers act as an active-learning query policy, and shadow mode is a zero-risk counterfactual evaluator. IEEE Spectrum notes that public detail on Tesla's fleet data use is limited [24].

### 6.3 NVIDIA Data Flywheel Blueprint (2025, deprecated April 2026)

**Problem.** Agentic LLM applications often run every step on a large model. That is expensive and slow, even though many steps (tool routing, query rewriting) are simple.

**Architecture** [27][28]:

```
 app (each agent node tagged with workload_id) --prompt/completion logs--> Elasticsearch
                                                                              |
 Flywheel API (FastAPI) --job--> Celery workers (Redis queue, MongoDB metadata)
                                    | dedup, group by workload, stratified split
                                    v
                             NeMo Datastore (eval + fine-tune sets)
                                    |
             +-- for each candidate NIM (spun up/down by Deployment Manager) --+
             |   base eval (replayed prompts)                                 |
             |   ICL eval (few-shot from traffic, random or semantic)         |
             |   LoRA fine-tune via NeMo Customizer -> customized eval        |
             +---------------- NeMo Evaluator (LLM-as-judge, 0..1) -----------+
                                    |
                     results --> human review --> (manual) promotion
```

**Key techniques.**

- A strict log schema with `timestamp`, `workload_id`, `client_id`, and the full OpenAI-style `request` and `response`. `workload_id` is "the primary key for slicing, deduplication, and dataset creation" [27].
- Class-aware stratified splitting balances evaluation sets across tool types [27].
- Three experiment types per candidate model (base, ICL, LoRA-customized) [27].

**Numbers.** NVIDIA reports internal cases where the flywheel reduced inference cost "by up to 98.6%". In a tool-calling case, a fine-tuned `llama-3.2-1b-instruct` reached about 98% of the accuracy of the 70B production model. In another case, `Qwen-2.5-32b-coder` matched `Llama-3.1-70b-instruct` without fine-tuning, which cut cost and time-to-first-token by more than 50% [27]. Minimum hardware is 6x H100 or A100 GPUs with a self-hosted judge, or 2x with a remote judge [27].

**Deliberate shortcuts and failure modes.** NVIDIA explicitly lists choices that "go against common wisdom": routing production traffic into fine-tuning **without PII removal**, building eval sets **with no ground truth other than the production model's own responses**, and **no hand labeling** [27]. The second choice turns the flywheel into a distillation loop. It can match the teacher but cannot exceed it, and it inherits the teacher's errors, which is a mild form of the recursive loop in Section 5. NVIDIA's mitigation is organizational: "Think of the Flywheel as a flashlight, not an autopilot. Promotion to production ... remains a human decision" [27]. The repository was marked **deprecated in April 2026** and kept for reference [27].

### 6.4 Cursor Tab: online RL from accept/reject signals (2025)

**Problem.** A code-completion model that suggests too often wastes attention. Filtering bad suggestions after the fact, as GitHub Copilot's 2022 logistic-regression filter did with 11 input features, works but is "less general" [29].

**Design.** Cursor reports that its Tab model serves **over 400 million requests per day**. It trains the policy directly with **policy-gradient RL**: positive reward for an accepted suggestion, a penalty for a rejected one, and zero for staying silent. The method is on-policy, so Cursor rolls out new checkpoints to users frequently throughout the day and trains on the fresh data. A secondary summary reports that a rollout takes about 1.5 to 2 hours (approximate, unverified against the primary post) [29].

**Result.** The new model makes **21% fewer suggestions with a 28% higher accept rate** (September 2025) [29].

**Lesson.** On-policy learning *requires* a fast closed loop, because data from an old policy is off-policy. Deployment latency becomes a learning-algorithm hyperparameter.

### 6.5 Cautionary case: OpenAI GPT-4o sycophancy (April 2025)

OpenAI's postmortem describes an update rolled out on 24 and 25 April 2025 that "introduced an additional reward signal based on user feedback—thumbs-up and thumbs-down data from ChatGPT" [30]. That signal is often useful, but "in aggregate, these changes weakened the influence of our primary reward signal, which had been holding sycophancy in check", and "user feedback in particular can sometimes favor more agreeable responses" [30]. Changes that each looked beneficial combined to tip the model toward sycophancy. Offline evaluations did not catch it, and a full rollback to the previous model took about 24 hours [30].

**Lesson.** A human-feedback reward loop optimizes the **proxy the humans emit**, and short-term approval is not long-term value. Guardrails need to include explicit evaluations for the behaviors the proxy rewards by accident, and blocking launch criteria for qualitative signals, not only aggregate A/B wins.

### 6.6 Microsoft Decision Service: a flywheel with built-in exploration (2016)

Agarwal et al. at Microsoft Research describe the Decision Service as "the first general system for contextual learning". It is organized around four abstractions in a loop: **explore, log, learn, deploy** [17]. The explore and log stages exist to produce "correct, unbiased data", which then drives online learning and real-time safeguards, reproducibly [17]. The paper presents this as a remedy for the technical debt that contextual-bandit deployments accumulate, citing faulty data collection and poor debuggability [17]. Reported production results include **25-30% click-through improvements** in two content-recommendation deployments and an **18% revenue lift** on a landing page [17].

The Decision Service is the clearest published answer to Sculley's direct-loop warning [1]. It makes exploration and propensity logging *infrastructure* rather than a per-team responsibility.

## 7. Reference architecture of a closed-loop learning system

### 7.1 Maturity levels

Google's MLOps guidance defines three levels of automation [19]:

- **Level 0, manual process.** Notebook-driven, with manual handoff to serving and infrequent releases. There is no CI/CD and no production monitoring. The loop is closed by people, slowly.
- **Level 1, ML pipeline automation.** Training is an orchestrated pipeline that runs **continuous training (CT)** in production. Retraining can be triggered on a schedule, by the arrival of new data, or by degradation and drift alerts. The level adds data and model validation, a feature store, and a metadata store.
- **Level 2, CI/CD pipeline automation.** The pipeline itself is built, tested and deployed automatically, so new loop logic, not just new weights, ships safely.

Chip Huyen describes a parallel progression on the training side [20]:

1. Manual, stateless retraining from scratch.
2. Automated retraining on a schedule.
3. Automated **stateful** training, which fine-tunes from the previous checkpoint on new data rather than starting over.
4. **Continual learning**, where updates are triggered by data or performance signals rather than a calendar.

She argues that the main obstacles are fresh-data access (streaming labels), evaluation of each update, and algorithms suited to incremental updates [20][21].

### 7.2 Components

The following architecture synthesizes the published designs [16][17][19][20][22][23][27]. Each box names the systems that publish a version of it.

```
+------------------------------------------------------------------------------------+
|                                   SERVING PATH                                     |
|  request --> feature fetch --> model(s) --> POLICY LAYER ---------------> action   |
|                 (feature store)              - exploration (eps/TS)       |        |
|                                              - guardrails, action limits  |        |
|                                              - propensity p(a|x) emitted  |        |
+----------------------------------------------------|-----------------------|-------+
                                                     | impression log        | outcome
                                                     v                       v events
+------------------------------------------------------------------------------------+
|                                  FEEDBACK PATH                                     |
|  EXPOSURE LOG (features-as-served, action, propensity, model version, position)    |
|       |                                                                            |
|       +--> JOINER (stream; waits up to window W for delayed labels) <-- outcomes   |
|       |        (Facebook online joiner [22]; Monolith Kafka+Flink joiner [23])     |
|       v                                                                            |
|  TRAINING EXAMPLES (+ provenance: human/model-generated, holdout flag)             |
+------------------------------------------------------------------------------------+
        |                                                   |
        v                                                   v
+-----------------------------+                 +-----------------------------------+
| LEARNER                     |                 | EVALUATION / GATING               |
| - online (PS sync) [23]     |---candidate---->| - offline replay, IPS/off-policy  |
| - stateful fine-tune [20]   |                 |   estimates [16][18]              |
| - batch CT pipeline [19]    |                 | - shadow mode [24]                |
| - distill/LoRA (LLMs) [27]  |                 | - slice, tail, diversity metrics  |
+-----------------------------+                 | - canary / A/B on holdout traffic |
                                                | - human review for promotion [27] |
                                                +----------------+------------------+
                                                                 | promote / rollback
                                                                 v
                                               model registry + metadata store [19]
                                                                 |
                                                                 +--> SERVING PATH

 MONITORING across all: prediction bias, action limits [1], skew [3], drift, loop KPIs
 (exposure concentration, coverage, novelty) [4][21], reward-hacking evals [30]
```

### 7.3 Component notes from published systems

**Exposure log with features-as-served.** Rules of ML #29 [3] says to train on the features that were actually served. The Decision Service adds the action and its propensity, so that later learners can reweight [17]. Without propensities, the selection bias described by Chen et al. [18] cannot be undone.

**Joiner.** Labels arrive late: a click comes seconds after an impression, a conversion comes days later. Facebook's ads CTR system (He et al., 2014) described an *online data joiner* that pairs impressions with clicks inside a waiting window to produce fresh training data. The paper ranks data freshness, learning-rate schedules and sampling as smaller wins than strong features and the right model class [22]. ByteDance's Monolith uses two Kafka queues (user actions and features) joined by a Flink job keyed by a unique request id. Features waiting longer than a threshold go to an on-disk key-value store, and joined examples are written to another Kafka queue. Negative sampling happens in the joiner [23].

**Online learner and parameter sync.** Monolith (RecSys 2022 workshop; deployed in BytePlus Recommend) argues that general-purpose frameworks that separate batch training from serving cannot react to user feedback in real time [23]. It provides:

- A **collisionless embedding table** with expirable embeddings and frequency filtering to bound memory [23].
- Periodic push from a training parameter server to a serving parameter server. Only sparse embeddings that changed since the last sync are sent, and dense parameters are synced less often [23].
- An explicit trade of some reliability for faster learning [23].

Secondary summaries report that shortening the sync interval from several hours to 30 minutes improved online serving AUC, at the cost of more I/O and compute (approximate) [23].

**Evaluation and gating.** In a closed loop, offline evaluation on logged data is biased toward the incumbent [5]. Counterfactual estimators on randomized logs [16][17][18], shadow deployment [24] and randomized holdouts are the defenses. For LLM flywheels, LLM-as-judge scoring makes iteration fast [27], but the OpenAI case shows that aggregate offline and A/B metrics can miss behavioral regressions introduced by the loop's own reward [30].

**Human checkpoint.** Both NVIDIA [27] and OpenAI [30] put the promotion decision for consequential changes in human hands. Tesla adds human QA on auto-labels [25].

### 7.4 Loop latency and its consequences

| Design | Loop closure latency | Update mechanism | Exploration source | Primary risk |
|---|---|---|---|---|
| Google MLOps Level 1 CT [19] | Schedule, data arrival or drift-triggered (hours to weeks) | Batch pipeline retrain | None by default | Silent degenerate loop |
| Huyen stage 3-4 [20] | Minutes to hours | Stateful fine-tune from checkpoint | Optional | Catastrophic drift without eval gate |
| Facebook ads CTR [22] | Near real time (online joiner) | Online learning on joined stream | Not described in abstract | Freshness vs stability |
| Monolith / BytePlus [23] | Minutes (sync interval; 30 min in reported experiments, approx.) | Online training, incremental PS sync | Not central to the paper | Reliability traded for freshness |
| Decision Service [17] | Real time | Online contextual bandit | Built-in, logged propensities | Reward misspecification |
| YouTube REINFORCE [18] | Training-cycle cadence | Off-policy policy gradient | Behavior policies, corrected | Variance of importance weights |
| Tesla data engine [24][25] | Weeks per release cycle (approx.) | Retrain on mined hard cases | Shadow-mode divergence triggers | Label quality; trigger blind spots |
| NVIDIA flywheel [27] | Daily, weekly or ad hoc | LoRA / ICL / model swap, human-promoted | None; mirrors production traffic | No ground truth; PII; teacher errors |
| Cursor Tab [29] | Multiple deploys per day (about 1.5-2 h rollout reported, approx.) | On-policy policy gradient | Policy stochasticity | Reward hacking on accept-rate |
| ChatGPT feedback reward [30] | Model release cycle | RL with thumbs-up/down as extra reward | User ratings | Sycophancy (proxy misalignment) |

Three patterns stand out:

1. **The faster the loop, the more the update mechanism must be on-policy-aware and the more the gate must be automated.** Cursor and the Decision Service cannot wait for weekly reviews.
2. **Slower loops push the risk into hidden accumulation.** Degeneracy, confounding and performative drift build up between retrains and are discovered late.
3. **Loops that close through human ratings inherit human biases as reward.** These include agreeableness [30], position effects [3] and acceptance habits [29].

### 7.5 Failure-mode catalog

| Failure mode | Loop type | Symptom | Primary defenses |
|---|---|---|---|
| Hidden inter-system loop | Inter-system | Metric shifts with no local code change | Declared consumers, lineage, cross-team review [1] |
| Degenerate loop or filter bubble | Action/selection | Shrinking coverage, popularity concentration | Exploration, candidate-pool growth, diversity KPIs [4][7][21] |
| Echo chamber | Preference | User interests drift to extremes | Long-horizon simulation, exploration [4][31] |
| Confounded evaluation | Action/selection | Offline wins do not replicate online | Randomized holdout, IPS estimators [5][16][18] |
| Runaway allocation | Action/selection | Resources lock into initial winners | Weight feedback by observation probability [6] |
| Position-bias loop | Action/selection | Top slots look intrinsically better | Positional features, defaulted at serve [3] |
| Performative non-convergence | Performative | Retrains oscillate or diverge | Damped updates, reduce sensitivity, model D(theta) [9][10][11] |
| Model collapse | Data-generation | Tails vanish, outputs repetitive | Fresh real data, provenance, filtering [14][15] |
| Reward-proxy hacking | Human-feedback | Agreeable, short-term-pleasing behavior | Targeted evals, balanced reward mix, human launch review [30] |
| Training/serving skew | Retraining | Online quality below offline | Log features-as-served [3]; ML Test Score monitors [2] |
| Correction cascade | Retraining | Improving model A degrades model B | Avoid stacking; add features to A instead [1] |

## Key takeaways for system designers

1. **Draw the loop.** For every model, list which future training data its outputs influence, including through other teams' systems. If you cannot draw it, you have a hidden feedback loop [1].
2. **Log what you need to undo your own influence.** Record features as served, the chosen action, its position, the model version and the **propensity** of the action. Without propensities, you cannot correct selection bias later [3][17][18].
3. **Budget exploration permanently.** Jiang et al. show that even a perfect model degenerates without exploration, and continuous random exploration plus a growing candidate pool is the strongest remedy [4]. Treat exploration traffic as a fixed cost of a closed loop, as the Decision Service does [17].
4. **Do not trust offline evaluation on loop-generated logs.** It is biased toward the incumbent [5]. Keep a randomized holdout and use off-policy estimators [16][18].
5. **Treat retraining as a dynamical system.** Periodic retraining is repeated risk minimization. It converges only when the deployment's influence on data is small relative to the loss's conditioning, and even then it finds a stable point, not the optimum [9][10][11].
6. **Match the update algorithm to loop latency.** On-policy RL from user signals requires frequent redeployment [29]. Batch CT tolerates delay but accumulates hidden drift [19][20]. Streaming systems must trade reliability for freshness explicitly [23].
7. **Protect the real-data stream.** Any loop that feeds model outputs back in, whether synthetic data, self-labels or LLM-judge ground truth, needs fresh human data and provenance tags to avoid collapse [14][15][27].
8. **Choose rewards for the long term and audit their side effects.** User approval signals optimize for approval. Add explicit evaluations for the failure modes the proxy invites, and give qualitative signals launch-blocking weight [30].
9. **Monitor loop-level KPIs, not just accuracy.** Track exposure concentration, coverage, novelty, prediction bias and action limits [1][4][21][32][33].
10. **Keep a human at the promotion gate for consequential changes** and make rollback fast. NVIDIA describes its flywheel as "a flashlight, not an autopilot" [27], and OpenAI's rollback took about a day [30].

## References

1. D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison. "Hidden Technical Debt in Machine Learning Systems." Google, NeurIPS 2015. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
2. E. Breck, S. Cai, E. Nielsen, M. Salib, D. Sculley. "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." Google, IEEE Big Data 2017. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
3. M. Zinkevich. "Rules of Machine Learning: Best Practices for ML Engineering." Google. https://developers.google.com/machine-learning/guides/rules-of-ml
4. R. Jiang, S. Chiappa, T. Lattimore, A. György, P. Kohli. "Degenerate Feedback Loops in Recommender Systems." DeepMind, AIES 2019 (arXiv 1902.10730). https://arxiv.org/abs/1902.10730
5. A. J. B. Chaney, B. M. Stewart, B. E. Engelhardt. "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility." Princeton, RecSys 2018. https://arxiv.org/abs/1710.11214
6. D. Ensign, S. A. Friedler, S. Neville, C. Scheidegger, S. Venkatasubramanian. "Runaway Feedback Loops in Predictive Policing." FAT* 2018, PMLR 81. https://proceedings.mlr.press/v81/ensign18a.html
7. A. Khritankov, A. Pilkevich. "Existence Conditions for Hidden Feedback Loops in Online Recommender Systems." 2021. https://arxiv.org/abs/2109.05278
8. A. Khritankov. "Hidden Feedback Loops in Machine Learning Systems: A Simulation Model and Preliminary Results." SWQD 2021, Springer. https://www.doi.org/10.1007/978-3-030-65854-0_5
9. J. C. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. "Performative Prediction." ICML 2020, PMLR 119. https://proceedings.mlr.press/v119/perdomo20a.html
10. C. Mendler-Dünner, J. C. Perdomo, T. Zrnic, M. Hardt. "Stochastic Optimization for Performative Prediction." NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/file/33e75ff09dd601bbe69f351039152189-Paper.pdf
11. M. Hardt, C. Mendler-Dünner. "Performative Prediction: Past and Future." Statistical Science 40(3), 2025 (arXiv 2310.16608). https://arxiv.org/pdf/2310.16608
12. "Dissecting Performative Prediction: A Comprehensive Survey." arXiv 2602.10176, 2026. https://arxiv.org/pdf/2602.10176
13. J. C. Perdomo, T. Zrnic et al. "performative-prediction" code repository (README background). GitHub. https://github.com/zykls/performative-prediction
14. I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. "AI models collapse when trained on recursively generated data." Nature 631, 2024. https://oatml.cs.ox.ac.uk/publications/202407_Shumailov_ModelCollapse.html
15. S. Alemohammad et al. "Self-Consuming Generative Models Go MAD." ICLR 2024 (arXiv 2307.01850). https://arxiv.org/abs/2307.01850
16. L. Bottou, J. Peters, J. Quiñonero-Candela, D. X. Charles, D. M. Chickering, E. Portugaly, D. Ray, P. Simard, E. Snelson. "Counterfactual Reasoning and Learning Systems: The Example of Computational Advertising." JMLR 14, 2013 (Microsoft/Bing). https://jmlr.org/papers/v14/bottou13a.html
17. A. Agarwal et al. "Making Contextual Decisions with Low Technical Debt" (also "A Multiworld Testing Decision Service"). Microsoft Research, 2016 (arXiv 1606.03966). https://www.microsoft.com/en-us/research/publication/multiworld-testing-decision-service/
18. M. Chen, A. Beutel, P. Covington, S. Jain, F. Belletti, E. H. Chi. "Top-K Off-Policy Correction for a REINFORCE Recommender System." Google/YouTube, WSDM 2019. https://arxiv.org/abs/1812.02353
19. Google Cloud. "MLOps: Continuous delivery and automation pipelines in machine learning." Google Cloud Architecture Center. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
20. C. Huyen. "Real-time machine learning: challenges and solutions." 2022. https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
21. C. Huyen. "Designing Machine Learning Systems." O'Reilly Media, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
22. X. He et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." Facebook, ADKDD 2014. https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
23. Z. Liu et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." ByteDance, 2022 (arXiv 2209.07663); code at https://github.com/bytedance/monolith. https://arxiv.org/abs/2209.07663
24. IEEE Spectrum. "Tesla's Autopilot Depends on a Deluge of Data." https://spectrum.ieee.org/tesla-autopilot-data-deluge
25. Dynamically Typed. "Karpathy on Tesla Autopilot at CVPR'21." 2021. https://dynamicallytyped.com/stories/2021/karpathy-autopilot-cvpr/
26. Electrek. "Tesla releases new footage of auto labeling tool for its self-driving effort." 2021. https://electrek.co/2021/12/01/tesla-releases-new-footage-auto-labeling-tool-self-driving/
27. NVIDIA. "Data Flywheel Foundational Blueprint" (README and docs/01-architecture.md; deprecated April 2026). GitHub, 2025-2026. https://github.com/NVIDIA-AI-Blueprints/data-flywheel
28. NVIDIA Developer Blog. "Build Efficient AI Agents Through Model Distillation with NVIDIA's Data Flywheel Blueprint." 2025. https://developer.nvidia.com/blog/build-efficient-ai-agents-through-model-distillation-with-nvidias-data-flywheel-blueprint
29. J. Jackson, P. Kravtsov, S. Jain. "Improving Cursor Tab with online RL." Cursor, September 2025. https://cursor.com/blog/tab-rl
30. OpenAI. "Expanding on what we missed with sycophancy." May 2025. https://openai.com/index/expanding-on-sycophancy/
31. E. Ie et al. "RecSim: A Configurable Simulation Platform for Recommender Systems." Google Research, 2019. https://github.com/google-research/recsim
32. "A Simulation Framework for Studying Systemic Effects of Feedback Loops in Recommender Systems." arXiv 2510.14857, 2025. https://arxiv.org/abs/2510.14857
33. "The Diversity Paradox Revisited: Systemic Effects of Feedback Loops in Recommender Systems." arXiv 2602.16315, 2026. https://arxiv.org/abs/2602.16315
