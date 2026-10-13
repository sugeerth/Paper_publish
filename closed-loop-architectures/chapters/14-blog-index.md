# 14. Master Index of Blogs and Talks on Data Flywheels, Feedback Loops, RLHF and Self-Improving Systems: 170 Annotated Entries

## Overview

Most of what practitioners know about closed-loop AI systems was first written down outside journals. It appears in engineering blogs, conference talks, cookbook notebooks, release posts and the READMEs of training libraries. Papers usually describe the learning algorithm. These other sources describe the loop around it: which signal is logged, how it is joined to outcomes, who or what grades it, how weights reach serving, and which gate stops a regression. This chapter is a curated, annotated index of that literature, from the 2013 to 2016 "hidden feedback loop" essays to the async-RL and harness-evolution posts of 2026.

**How the index is organized.** The 170 entries (E1 to E170) are grouped into seven loop families. Each family opens with a short design synthesis covering the problem, the architecture (components and data flow), key techniques, public numbers, failure modes and lessons. The annotated entries follow. Each entry gives its type (blog, talk, cookbook, README, paper companion), year, and one or two sentences on what it contributes to closed-loop design.

**Status tags (read these before trusting a number).**

- **(F)**: fetched and read for this chapter (October 2026) from its source or from the publisher's own source repository.
- **(F2)**: not opened directly. Its claims are taken from a fetched source that quotes it, and that source is cited.
- **(A)**: a primary source read and summarized in an earlier chapter of this Atlas (Chapters 01 to 12). Its annotation follows that chapter.
- **(I)**: existence, title and date confirmed from the publisher's own index or feed. The annotation states only what the title and the surrounding literature establish.
- **(K)**: a widely cited, well-known source that was not re-opened anywhere in this Atlas. Treat its annotation as orientation, not evidence.

**Source note.** For this chapter, general web search and most websites were unreachable from the research sandbox (the search quota was exhausted and non-GitHub hosts were blocked). Primary text was therefore read through the publishers' own source repositories on GitHub. These include the Hugging Face blog source (`huggingface/blog`), Lilian Weng's blog source and feed, the LMSYS and Qwen blog sources, the vLLM blog source, the ICLR Blog Track, Karpathy's blog source, the OpenAI Cookbook (including its `registry.yaml` catalogue), and the Tinker Cookbook. Links in the References section point to the canonical public URLs. Where a number is approximate or vendor-reported, the text says so.

```
           WHERE EACH FAMILY OF ENTRIES SITS ON THE GENERIC CLOSED LOOP

  (S1) foundations, loop theory,          (S7) failure modes: reward hacking,
       counterfactual logging                  sycophancy, collapse, audits
            |                                          ^
            v                                          |
  +-----------+   act    +-----------+  signals  +------------+   select/   +---------+
  |  policy / +--------->+  world /  +---------->+ log, join, +------------>+ grade / |
  |  model /  |          |  users /  | (accept,  | mine, label|   curate    | reward  |
  |  harness  |          |  fleet /  |  thumbs,  | (S2 data   |             | (S4 RM, |
  +-----+-----+          |  tests    |  triggers)|  engines)  |             |  S5 RLVR|
        ^                +-----------+           +------------+             |  verif.)|
        |                                                                   +----+----+
        |   deploy: weight sync, LoRA swap,  +------------------+   update        |
        +-----------------------------------+ gate: eval, A/B, +<----------------+
            canary, rollback (S3 products,   | interleave, audit|  SFT / DPO / PPO /
            S5 async RL infra)               +------------------+  GRPO / prompt or
                                                                   harness edit (S6)
```

---

## 1. Foundations: the loop as an engineering object (E1 to E22)

**Problem.** A deployed model changes the data it will later be trained on. Sculley et al. called the unplanned version of this a "hidden feedback loop" and ranked it among the most expensive forms of ML technical debt [1]. Recommender and policing studies later showed the same mechanism producing echo chambers and runaway allocation [2][3]. Performative prediction formalized it as a distribution that shifts in response to the deployed model [4].

**Architecture and techniques.** The foundational write-ups converge on one design. Log the context, the action and its propensity at serving time, and join them to delayed outcomes. Learn with counterfactual estimators or online updates. Gate releases with tests and staged experiments. Microsoft's Decision Service packaged this as an explore, log, learn and deploy service [5][6]. Bing's counterfactual advertising work showed how to answer what-if questions from logged randomized data [7]. Google's ad-click paper showed the operational side of online learning at scale [8].

**Failure modes and lessons.** These are training and serving skew, feedback that narrows exploration, and training on one's own outputs until the tails vanish [9]. The durable lesson is that the loop itself (logging schema, joins, gates) needs tests and owners, just like the model [10][11].

- **E1 · Hidden Technical Debt in Machine Learning Systems** (Google, NeurIPS 2015; paper; A). The origin of "hidden feedback loops", entanglement and undeclared consumers. It is still the best checklist of how loops rot in production [1].
- **E2 · The ML Test Score** (Google, 2017; paper; A). A rubric of data, model, infrastructure and monitoring tests that turns "is our loop healthy?" into a score [10].
- **E3 · Rules of Machine Learning** (M. Zinkevich, Google; guide; A). Practitioner rules on logging features at serving time, launching simple models first and watching for skew. It is the operational complement to E1 [11].
- **E4 · MLOps: continuous delivery and automation pipelines** (Google Cloud; architecture guide; A). It defines maturity levels that run from manual training to automated continuous training (CT) with pipeline CI/CD. It is the vocabulary most enterprise flywheels use [12].
- **E5 · Degenerate Feedback Loops in Recommender Systems** (DeepMind, 2019; paper; A). It shows formally how a recommender's own exposure decisions narrow user interest, and why exploration and randomness break the loop [2].
- **E6 · Performative Prediction** (ICML 2020; paper; A). It treats the data distribution as a function of the deployed model, and it defines when repeated retraining converges to a stable point [4].
- **E7 · Runaway Feedback Loops in Predictive Policing** (FAT* 2018; paper; A). An urn-model analysis of how allocating on past detections over-samples the same areas. It is the clearest social-harm case of a closed loop [3].
- **E8 · AI models collapse when trained on recursively generated data** (Nature 2024; paper and lab summary; A). It shows that recursive training on generated data erodes distribution tails. This is the reason synthetic-data loops need fresh real data and provenance [9].
- **E9 · Counterfactual Reasoning and Learning Systems** (Microsoft/Bing, JMLR 2013; paper; A). It uses importance-weighted estimates from logged randomized data to evaluate policy changes offline. It is the statistical backbone of every logged-feedback loop [7].
- **E10 · Ad Click Prediction: a View from the Trenches** (Google, KDD 2013; paper; A). Online learning at very large scale, together with the unglamorous parts: calibration, monitoring and memory-saving tricks [8].
- **E11 · Real world interactive learning at cusp of enabling new class of applications** (Microsoft Research Blog, 2017; blog; A). An accessible account of contextual-bandit "interactive learning" as a reusable service [5].
- **E12 · A Multiworld Testing Decision Service** (Microsoft Research, 2016; paper and project; A). A reference architecture with four components (explore, log, learn, deploy) and propensity logging built in [6].
- **E13 · Real-time machine learning: challenges and solutions** (C. Huyen, 2022; blog; A). It separates online prediction from continual learning and lays out the infrastructure steps from batch retraining to stateful, near-real-time updates [13].
- **E14 · Designing Machine Learning Systems** (C. Huyen, O'Reilly 2022; book; A). Chapters on distribution shift, continual learning and testing in production make it the textbook treatment of production loops [14].
- **E15 · Deep Reinforcement Learning: Pong from Pixels** (A. Karpathy, 2016; blog; F). Policy gradients explained end to end: sample, score and push up the log-probability of what worked. It is the intuition RLHF and RLVR later reuse [15].
- **E16 · Software 2.0** (A. Karpathy, 2017; essay; K). It argues that neural networks are programs written by data, so curating and labeling data becomes the main engineering act. This is the conceptual root of "data engines" [16].
- **E17 · The Bitter Lesson** (R. Sutton, 2019; essay; K). General methods that scale with compute beat hand-built knowledge. It is often cited to justify self-play and RL loops over hand-built heuristics [17].
- **E18 · Active Learning Literature Survey** (B. Settles, 2009/2010; survey; A). Uncertainty sampling, query-by-committee and expected model change, the selection step inside every human-in-the-loop flywheel [18].
- **E19 · Learning with not Enough Data, Part 2: Active Learning** (L. Weng, 2022; blog; I). A modern survey-style post on acquisition functions and their use with deep models [19].
- **E20 · Learning with not Enough Data, Part 3: Data Generation** (L. Weng, 2022; blog; I). It covers augmentation and model-generated training data, the precursor to LLM synthetic-data loops [20].
- **E21 · Patterns for Building LLM-based Systems & Products** (E. Yan, 2023; blog; K). Its seven patterns include evals, guardrails, defensive UX and "collect user feedback". It frames feedback capture as a product-design problem [21].
- **E22 · Your AI Product Needs Evals / What We've Learned From A Year of Building with LLMs** (H. Husain, 2024; Yan et al., 2024; blogs; K). These practitioner essays make error analysis and evals the engine of LLM product iteration. The OpenAI evaluation-flywheel cookbook (E166) cites the Shankar and Husain course reader that grew out of this line of work [22][23].

---

## 2. Data engines and fleet flywheels in physical AI (E23 to E54)

**Problem.** Autonomous driving and robotics fail in a long tail of rare scenes. More data helps only if the right data is found, labeled and fed back quickly.

**Architecture.** The canonical "data engine", described in Tesla's talks and later by Waymo, Cruise, Wayve, NVIDIA and Scale, has six stages:

```
   fleet / robots                cloud                                   release
  +---------------+  snapshot  +-----------------+  +----------------+  +-----------+
  | triggers,     +----------->+ search & mine   +->+ auto-label     +->+ train     |
  | shadow-mode   |  (event    | (embeddings,    |  | (offline nets, |  | (multi-   |
  | divergence    |   clips)   |  content search)|  |  hindsight)    |  |  task, E2E|
  +-------+-------+            +-----------------+  +----------------+  +-----+-----+
          ^                                                                   |
          |      +-------------------------------------------------------+   |
          +------+ evaluate: sim / world model / critic, shadow, canary  +<--+
                 +-------------------------------------------------------+
```

**Key techniques and numbers.** Hand-written trigger classifiers run on the fleet. A CVPR 2021 talk summary counts 221 triggers [24]. Hindsight auto-labeling uses offline networks that see past and future frames. Tesla reported auto-labeling 10,000 clips in about a week against a manual team of about 1,000 people [25]. Embedding-based fleet search followed (Waymo Content Search, 2020) [26]. From 2025 the evaluation stage moved into learned world models: Waymo's Genie-3-based World Model and Tesla's neural world simulator [27][28]. Mobileye's REM uploads about 10 KB per km of crowdsourced map data [29].

**Failure modes.** Proxy triggers that miss unknown unknowns, label noise from auto-labelers, and simulator gaps. There is also a running 2026 debate over whether end-to-end learning can shortcut validation [30].

- **E23 · Andrej Karpathy's CVPR Talk, Annotated** (D. Silver, 2021; talk summary; A). A line-by-line companion to the CVPR'21 Workshop on Autonomous Driving talk: the data engine, 221 triggers, and the shift to vision-only [24].
- **E24 · Karpathy on Tesla Autopilot at CVPR'21** (Dynamically Typed, 2021; talk summary; A). A shorter digest of the same talk, focused on the trigger, label and retrain loop [31].
- **E25 · LIVE Blog: Tesla AI Day** (Teslarati, 2021; talk live blog; A). The AI Day 2021 account of auto-labeling, simulation and training infrastructure, including the manual-labeling scale and the hindsight auto-labeler [25].
- **E26 · Recap Tesla AI Day 2022** (nocode.ai, 2022; talk recap; A). A secondary recap of the 2022 update to the auto-labeling and data-engine pipeline [32].
- **E27 · Tesla releases new footage of auto labeling tool** (Electrek, 2021; news; A). Visual evidence of the offline 4D labeling tool used in the engine [33].
- **E28 · Generating ground truth from time series elements** (Tesla patent family, 2021/2025; patent; A). The hindsight principle in legal form: derive labels from later parts of a sensor sequence and train a model to predict them from one element [34].
- **E29 · How Tesla Continuously and Automatically Improves Autopilot** (Code Compass; explainer; A). It explains trigger classifiers and shadow mode. It is useful for the mechanism, but it is secondary [35].
- **E30 · Tesla's approach to Autonomy, ICCV 2025** (A. Elluswamy, 2025; talk post; A). In the end-to-end era, the engine curates demonstrations and evaluation scenarios. Tesla describes a neural world simulator for closed-loop evaluation [28].
- **E31 · Cruise's Continuous Learning Machine** (Cruise, 2020; blog; A). A self-supervised loop for motion prediction: mine prediction errors and auto-label them from what agents did next [36].
- **E32 · Predicting the Unpredictable With Cruise's CLM** (MLconf; talk; A). The conference version of E31 [37].
- **E33 · Content Search** (Waymo, 2020; blog; A). Embeddings over every object in the fleet archive replace heuristic mining, built with technology similar to Google Photos search [26].
- **E34 · Off road, but not offline** (Waymo, 2020; blog; A). Large-scale simulation as the evaluation stage of the loop [38].
- **E35 · How ML at Waymo is building a scalable autonomous driver** (D. Dolgov, Scale Exchange, 2022; talk write-up; A). An executive-level view of Waymo's ML lifecycle and data flywheel [39].
- **E36 · Demonstrably Safe AI For Autonomous Driving** (Waymo, December 2025; blog; A). Waymo's most explicit flywheel description: driver, simulator and critic models built on a foundation model, and a critic that flags issues for refinement [40].
- **E37 · The Waymo World Model** (Waymo, February 2026; blog; A). Built on DeepMind's Genie 3 and post-trained to generate camera and lidar output for rare-scenario simulation [27].
- **E38 · 10 AI Lessons from Driving 200+ Million Fully Autonomous Miles** (Waymo, August 2026; blog; A). A retrospective that restates the flywheel. The critic and rider and community feedback surface refinements [30].
- **E39 · Wayve Unveils GAIA-2** (Wayve, 2025; release post; A). A latent-diffusion world model for multi-view synthetic driving data, conditioned on ego dynamics, agents and road semantics [41].
- **E40 · Global Learning, Local Driving: Lessons From Japan** (Wayve, 2025; blog; A). A "fleet learning loop" argument that one globally trained model adapts to a new country with limited local data [42].
- **E41 · openpilot 0.11** (comma.ai, 2025; release post; A). A shipping driver policy trained inside a learned world-model simulator, which closes the loop without a hand-built simulator [43].
- **E42 · NVIDIA Cosmos world foundation models and physical AI data tools** (NVIDIA Newsroom, 2025; announcement; A). Curation, transfer and prediction models packaged as a physical-AI data flywheel [44].
- **E43 · Physical AI Data Factory Blueprint, GTC 2026** (NVIDIA Blog, March 2026; talk and announcement; A). A reference architecture for creating, augmenting and evaluating datasets, with a Cosmos-Reason-based evaluator that filters generated data [45].
- **E44 · Cosmos Curator** (NVIDIA, 2025-2026; README; A). Video splitting, annotation, filtering and de-duplication at fleet scale, the "mine" stage as open code [46].
- **E45 · Expanding Our Data Engine for Physical AI** (Scale AI, 2025; blog; A). The AV-era vendor engine extended to robotics. It reports more than 100,000 production hours at its prototyping lab (vendor claim) [47].
- **E46 · Introducing Scale Nucleus** (Scale AI; blog; A). A dataset-curation and model-debugging tool, described as "Google Photos for ML datasets", for the find-the-failure step [48].
- **E47 · Hyundai Motor Group AI-powered data flywheel** (Hyundai; newsroom; A). The automaker version of the loop: collect, train, validate, deploy [49].
- **E48 · Technically Speaking: Learning Every Mile Driven** (Motional; blog; A). Another operator's fleet-learning explainer [50].
- **E49 · REM: Road Experience Management** (Mobileye; technology page; A). The fleet used as a map sensor: about 10 KB per km uploaded, with privacy trimming near trip endpoints [29].
- **E50 · π0.5: a VLA with Open-World Generalization** (Physical Intelligence, 2025; blog; A). Co-training on heterogeneous robot and web data. It shows how a robot-data flywheel feeds a generalist policy [51].
- **E51 · Knowledge Insulation** (Physical Intelligence, 2025; research post; A). It protects a VLM backbone from action-expert gradients during robot training, a recipe-level lesson for loops that fine-tune foundation models on new data [52].
- **E52 · Helix** (Figure AI, 2025; blog; A). A two-system humanoid controller: a VLM of about 7B parameters at about 7-9 Hz and a visuomotor policy of about 80M parameters at about 200 Hz. It is relevant to where on-robot data collection plugs in [53].
- **E53 · Accelerating AV Simulation with Neural Reconstruction and World Foundation Models** (NVIDIA Technical Blog, 2025; blog; A). Drive logs are turned into reconstructed, editable scenes for closed-loop evaluation, so the evaluation stage of the engine can be fed from real fleet data [54].
- **E54 · Scale AI acquires SiaSearch** (TechCrunch, 2021; news; A). The data engine as a purchasable product: fleet-data search pitched as "the same data engine superpowers that Tesla has" [55].

---

## 3. Product feedback loops: from bandits to online RL on user signals (E55 to E73)

**Problem.** Products emit cheap, on-distribution feedback: accepts, skips, dwell time, thumbs and edits. That feedback is biased by presentation and reflects short-term approval.

**Architecture.** Request, policy, rendered suggestion, implicit or explicit signal, label or reward construction, trainer, gated redeploy. The 2025-2026 change is cadence and algorithm. Cursor retrains its Tab completion policy with on-policy policy-gradient RL on accept and reject events and ships new checkpoints several times a day. The model handles over 400 million requests per day, and the new policy makes 21% fewer suggestions with a 28% higher accept rate [56]. For agentic coding, Cursor's Composer 2 report describes trainers that upload compressed weight diffs to a shared object store every step, so inference clusters in other regions can reconstruct the weights without direct connectivity [57].

**Techniques and gates.** Propensity-aware bandits (Netflix artwork), interleaving to prune rankers before A/B tests, with A/B needing on the order of 100x more members (approximate) [58], multi-metric risk-aware shipping rules [59], and long-term-satisfaction proxies in place of clicks [60].

**Failure modes.** Position bias. In Copilot Arena, 82% of accepted completions were the top-shown one [61]. Acceptance-rate gaming and sycophancy from thumbs data are the others (see Section 7).

- **E55 · Improving Cursor Tab with online RL** (Cursor, September 2025; blog; A). It puts "do nothing" in the action space and trains online on accept and reject. This is the clearest public case of product-signal RL at scale [56].
- **E56 · Composer 2 technical report** (Cursor, 2026; report; F2). Cross-region RL with per-step compressed weight diffs in a shared S3 bucket, as summarized by Hugging Face [57][62].
- **E57 · Copilot usage metrics** (GitHub docs, 2026; documentation; A). The current telemetry schema (suggestions, acceptances, lines, activity) that enterprises use to measure the loop [63].
- **E58 · Responsible use of Copilot inline suggestions** (GitHub docs, 2026; documentation; A). It documents how inline suggestions are produced and filtered, and their known limitations [64].
- **E59 · Copilot Arena** (LMArena, 2024; blog; A). Paired completions in the IDE turn user choice into a preference signal. It also shows how strong position bias is [61].
- **E60 · Reading Between the Lines** (Microsoft Research, CHI 2024; paper; A). It models programmer states and the costs of AI assistance, a corrective to acceptance-rate-only metrics [65].
- **E61 · Smart Compose** (Google AI Blog, 2018; blog; A). The ancestor of inline completion loops. It ran under a latency budget on the order of 100 ms (approximate), with acceptance as the core metric [66].
- **E62 · Artwork Personalization at Netflix** (Netflix, 2017; blog; A). Contextual bandits with logged propensities choose the artwork, and replay is used for offline evaluation [67].
- **E63 · Interleaving in online experiments at Netflix** (Netflix, 2017; blog; A). A cheaper first gate before A/B tests [58].
- **E64 · Recommending for Long-Term Member Satisfaction** (Netflix, 2024; blog; A). Reward engineering, meaning predicting delayed satisfaction from proxies, as the core of the recommendation loop [60].
- **E65 · Foundation Model for Personalized Recommendation** (Netflix, 2025; blog; A). One autoregressive model trained on hundreds of billions of interactions from more than 300 million members replaces many per-surface models [68].
- **E66 · Integrating Netflix's Foundation Model into Personalization Applications** (Netflix, 2025; blog; A). Integration patterns for consuming one foundation model across many personalization applications [69].
- **E67 · Spotify's New Experimentation Platform** (Spotify, 2020; blog; A). Remote configuration, a metrics catalog and analysis, the gate infrastructure for product loops [70].
- **E68 · Risk-Aware Product Decisions in A/B Tests with Multiple Metrics** (Spotify, 2024; blog; A). Success versus guardrail metrics with principled error rates [59].
- **E69 · Introducing Birdbrain** (Duolingo, 2020; blog; A). Per-learner difficulty modeling trained on exercise outcomes, an educational feedback loop [71].
- **E70 · Midjourney Documentation: personalization and ranking** (Midjourney, 2024-2025; docs; A). User image rankings feed both aesthetic training and per-user style profiles (details approximate) [72].
- **E71 · Optimizing AI Inference at Character.AI** (Character.AI, 2024; blog; A). The cost engineering behind an engagement-driven chat product with dense in-session feedback [73].
- **E72 · Memory and new controls for ChatGPT** (OpenAI, 2024-2025; blog; A). Per-user memory as a non-parametric personalization loop, with user-facing controls [74].
- **E73 · Harnessing Organizational Knowledge for Machine Learning** (Google AI Blog, 2019; blog; A). Snorkel DryBell: weak supervision from organizational resources replaces hand labels inside production pipelines [75].

---

## 4. RLHF: explainers, recipes, open reproductions and preference infrastructure (E74 to E114)

**Problem.** Desired behaviour is hard to write as a loss. RLHF learns a reward from human comparisons and optimizes the policy against it under a KL leash.

**Architecture.** Classic RLHF keeps four model copies in memory: the policy, a reference model, a reward model and a value model. RLOO drops the value model, uses about 50 to 70% less vRAM than PPO, and runs 2 to 3x faster, depending on model size [76]. DPO-family methods remove the reward model and online sampling altogether [77][78]. The TRL v1.0 post describes the field's swings well: reward models looked essential in PPO, became optional in DPO, and returned as verifiers in RLVR [79].

```
  prompts --> policy --samples--> [human / AI raters] --pairs--> reward model
     ^          |                                                     |
     |          +----------- PPO / RLOO / online DPO <--- reward -----+
     |                         (KL to reference)
     +---- new prompts from production traffic, red teams, arenas (data flywheel)
```

**Techniques and numbers.** Implementation details matter as much as the algorithm. The PPO reproduction posts list dozens of details, among them reward normalization and Adam epsilon differences between TensorFlow and PyTorch [80][81]. Live preference platforms became evaluation flywheels. Chatbot Arena had more than 800,000 votes over more than 90 LLMs by March 2024 [82]. Arena-Hard distilled live prompts into a benchmark with 89.1% agreement with the human ranking [83], and style control showed that length and markdown move rankings [84].

**Failure modes.** Reward over-optimization [85], rater disagreement and quality problems [86], and labor ethics [87].

- **E74 · Illustrating RLHF** (Hugging Face, December 2022; blog; F). It breaks RLHF into three stages (pretraining, reward modeling and RL fine-tuning) and remains the most-linked RLHF explainer [88].
- **E75 · RLHF: Reinforcement Learning from Human Feedback** (C. Huyen, 2023; blog; A). A systems-oriented walkthrough of SFT, the reward model and PPO, with data-collection economics [89].
- **E76 · Reinforcement Learning from Human Feedback (the RLHF Book)** (N. Lambert, 2025-2026; online book; A). The reference text on preference data, reward models, regularization, rejection sampling and over-optimization [90].
- **E77 · The N Implementation Details of RLHF with PPO** (Hugging Face, 2023; blog; F). It reproduces OpenAI's 2019 `lm-human-preferences` learning curves and catalogues the details that make them match [80].
- **E78 · The 37 Implementation Details of PPO** (ICLR Blog Track, 2022; blog; F). The canonical PPO checklist that RLHF engineers inherit [81].
- **E79 · StackLLaMA** (Hugging Face, 2023; blog; F). An end-to-end open RLHF pipeline (SFT, then RM, then PPO) on Stack Exchange data with LLaMA and PEFT [91].
- **E80 · Fine-tune Llama 2 with DPO** (Hugging Face, 2023; blog; F). The post that popularized DPO in TRL [77].
- **E81 · Preference Tuning LLMs with DPO Methods** (Hugging Face, 2024; blog; F). An empirical comparison of DPO, IPO and KTO. Its addendum corrects an IPO loss bug (averaged versus summed log-likelihoods), a reminder to audit loss code [78].
- **E82 · Putting RL back in RLHF** (Hugging Face and Cohere, 2024; blog; F). The RLOO trainer, with the memory and speed numbers quoted above. It argues that online methods beat offline DPO [76].
- **E83 · Constitutional AI with Open LLMs** (Hugging Face, 2024; blog; F). An open CAI recipe (critique, revise, then train) with `llm-swarm` for synthetic data on Slurm, plus constitutions modeled on Anthropic's and on Grok's [92].
- **E84 · Deep RL from Human Preferences** (OpenAI and DeepMind, 2017; paper; A). The original preference-learning loop with asynchronous human comparisons [93].
- **E85 · summarize-from-feedback** (OpenAI, 2020; repository; A). Released human comparison data and code for the summarization RLHF loop [94].
- **E86 · Training language models to follow instructions with human feedback (InstructGPT)** (OpenAI, 2022; paper; K). The SFT, RM and PPO recipe that defined the ChatGPT era [95].
- **E87 · InstructGPT model card and labeling instructions** (OpenAI, 2022; repository; A). The rarely read labeler guidelines, the human interface of the loop [96].
- **E88 · Training a Helpful and Harmless Assistant with RLHF** (Anthropic, 2022; paper page; A). Iterated online RLHF with refreshed preference data, and the tension between helpfulness and harmlessness [97].
- **E89 · Constitutional AI: Harmlessness from AI Feedback** (Anthropic, 2022; paper; K). Replaces human harm labels with model critiques against written principles (RLAIF) [98].
- **E90 · Claude's Constitution** (Anthropic, 2023; blog; A). The published principles used in the CAI loop [99].
- **E91 · Collective Constitutional AI** (Anthropic and CIP, 2023; blog; A). The public writes the principles: a loop whose input is democratic deliberation [100].
- **E92 · Claude's Constitution (2026)** (Anthropic, January 2026; document; A). The current long-form constitution that governs training feedback [101].
- **E93 · Red Teaming Language Models to Reduce Harms** (Anthropic, 2022; paper page; A). Red-team data as a loop input, with scaling behaviour across model sizes and RLHF [102].
- **E94 · Measuring Progress on Scalable Oversight** (Anthropic, 2022; paper page; A). Human-plus-model oversight experiments for when raters are weaker than the model [103].
- **E95 · Llama 2** (Meta, 2023; paper; K). Iterative RLHF rounds combining rejection sampling and PPO, with separate helpfulness and safety reward models [104].
- **E96 · The Llama 3 Herd of Models** (Meta, 2024; paper; K). Post-training as repeated rounds of reward modeling, rejection sampling, SFT and DPO [105].
- **E97 · Tulu 3** (AI2, 2024; paper; A). A fully open post-training recipe: SFT, then DPO on on-policy preference data, then RLVR [106].
- **E98 · open-instruct** (AI2; repository; A). The code behind Tulu 3. It is also one of the 16 async-RL libraries in E115 [107][108].
- **E99 · RewardBench** (AI2, 2024; paper; A). A benchmark for reward models themselves, the grader in the loop [109].
- **E100 · Unpacking DPO and PPO** (AI2 and UW, 2024; paper; A). It isolates the effects of data, reward model and algorithm [110].
- **E101 · Direct Language Model Alignment from Online AI Feedback** (Google DeepMind, 2024; paper; A). Online DPO with an LLM annotator, evidence that on-policy feedback matters [111].
- **E102 · WARM** (Google DeepMind, 2024; paper; A). Weight-averaged reward models for robustness against reward hacking [112].
- **E103 · RLAIF** (Google, 2023; paper; K). AI feedback as a substitute for human preference labels at scale [113].
- **E104 · PRM800K** (OpenAI, 2023; dataset repository; A). Step-level human labels for process reward models, with schema and QC notes [114].
- **E105 · Thinking about High-Quality Human Data** (L. Weng, 2024; blog; F). Rater agreement, the two paradigms for handling disagreement, influence functions and noisy cross-validation for cleaning loop data [86].
- **E106 · Can foundation models label data like humans?** (Hugging Face, 2023; blog; F). It compares GPT-4-based ratings with human labelers and cautions against "preferred N% of the time" claims [115].
- **E107 · Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings** (LMSYS, May 2023; blog; F). Launched from 4.7K votes. Anonymous, randomized battles turned crowd preference into a public flywheel [116].
- **E108 · Chatbot Arena: Live and Community-Driven LLM Evaluation** (LMSYS, March 2024; blog; F). Policy for the platform. It reports more than 800,000 votes over more than 90 models and rules for testing pre-release models [82].
- **E109 · From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline** (LMSYS, 2024; blog; F). It mines hard prompts from live traffic into a benchmark with 89.1% agreement with the human ranking. This is the "production data becomes eval" pattern [83].
- **E110 · Does style matter?** (LMSYS, 2024; blog; F). Controlling for length and markdown reorders the leaderboard, direct evidence that preference signals carry style bias [84].
- **E111 · OpenAI Model Spec** (OpenAI, 2024 onward; specification; A). The written target that graders and raters are trained against [117].
- **E112 · Open Preference Dataset for Text-to-Image Generation** (Hugging Face community, 2024; blog; F). A community-built, Apache-2.0 preference loop for images [118].
- **E113 · Bolstering a RAG app with LLM-as-a-Judge** (Digital Green and Hugging Face, 2024; blog; F). A field case. A farmer-advice RAG system draws on about 46,000 agricultural research documents and serves extension workers, who reach farmers at ratios of about 1:1000. It uses an LLM judge to close the quality loop [119].
- **E114 · Synthetic data: save money, time and carbon** (Hugging Face, 2024; blog; F). LLM-labeled data trains a small RoBERTa. Analyzing the corpus cost about $2.7 versus about $3,061 with GPT-4 (author estimate) [120].

---

## 5. RLVR, reasoning RL and the async training plant (E115 to E148)

**Problem.** For math, code and tool use, the reward can be checked rather than learned, so RL can scale. Generation then dominates wall-clock time.

**Architecture.** Every one of the 16 open-source RL libraries surveyed by Hugging Face in March 2026 independently converged on disaggregation [108]. Inference GPUs and trainer GPUs are separate pools joined by a rollout buffer, and weights are pushed asynchronously.

```
  +------------------+   rollouts + per-token model_version   +-------------------+
  | inference pool   +--------------------------------------->+ bounded buffer    |
  | (vLLM / SGLang)  |                                         | (staleness bound, |
  |  verifier/env    |<---- weights: NCCL | CUDA IPC | FS |    |  IS correction)   |
  +------------------+      delta in object store | LoRA swap +---------+---------+
          ^                                                             |
          |                     +------------------+  minibatches       |
          +---------------------+ trainer pool     +<-------------------+
                                | (FSDP2/Megatron) |
                                +------------------+
```

**Numbers.** On one H100, 512 rollouts of 32K tokens take about 45 minutes for a 7B model and about 3.7 hours for a 32B model (approximate, derived from vLLM throughput) [108]. Ray orchestrates 8 of the 16 libraries, and NCCL broadcast is the default weight-sync method [108]. Between consecutive RL steps, about 99% of bf16 weights are bit-identical. Sparse deltas cut the per-step payload for Qwen3-0.6B from 1.2 GB to 20 to 35 MB [62]. Fireworks puts a full fp8 1T-parameter snapshot at 1024 GiB [121]. LoRA-only sync let a trainer and two vLLM servers run as separate jobs. Tuning moved the bottleneck between trainer and generator and cut 500 steps from 3 h 27 min to 53 min [122].

**Failure modes.** Staleness and training-inference mismatch (the HF survey's DeepSeek V3.2 MoE case), MoE routing volatility (addressed by GSPO's sequence-level ratio) [123], retokenization drift in agent RL [124], spurious rewards [125], and RL that sharpens rather than expands base-model ability [126].

- **E115 · Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries** (Hugging Face, March 2026; blog; F). The definitive survey of async RL plumbing along seven axes: orchestration, buffer, weight sync, staleness, partial rollouts, LoRA and backends [108].
- **E116 · Delta Weight Sync in TRL** (Hugging Face, May 2026; blog; F). Sparse safetensors deltas through a Hub bucket, with trainer, vLLM and environment each in a different place [62].
- **E117 · Async GRPO with LoRA across HF Jobs** (Hugging Face, September 2026; blog; F). Adapter-only sync, a KV-prefix-aware proxy and bottleneck tuning. The importance ratio stayed at 1.000 across 126 syncs [122].
- **E118 · TRL v1.0** (Hugging Face, March 2026; blog; F). More than 75 post-training methods, a stable versus experimental contract, and about 3 million downloads a month. It is the design rationale for a library in a field that keeps moving [79].
- **E119 · LoRA Without Regret** (Thinking Machines, 2025; blog; F2). LoRA can match full fine-tuning for policy-gradient RL even at rank 1, because each episode carries about O(1) bits. This is what makes adapter-only weight sync viable [127][122].
- **E120 · Tinker Cookbook** (Thinking Machines, 2025-2026; README; F). A training-as-an-API SDK with realistic SFT and RL recipes. It is the hosted-trainer pattern for small teams [128].
- **E121 · On-policy distillation recipes** (Tinker Cookbook, 2025; README; A). Student-sampled sequences graded per token by a teacher, a cheap bridge between SFT and RL [129].
- **E122 · Frontier RL Is Cheaper Than You Think** (Fireworks, 2026; blog; F2). An argument from bandwidth arithmetic for delta-based weight shipping to rollout fleets [121].
- **E123 · Learning to Reason with LLMs** (OpenAI, 2024; blog; A). The o1 announcement: performance improves with both train-time RL compute and test-time thinking [130].
- **E124 · DeepSeek-R1 technical report** (DeepSeek-AI, 2025; report; A). R1-Zero (pure RL on verifiable rewards) followed by a multi-stage cold-start, RL, rejection sampling and RL pipeline [131].
- **E125 · DeepSeekMath** (DeepSeek-AI, 2024; repository and paper; A). The paper that introduced GRPO, the critic-free group baseline used across 2025-2026 [132].
- **E126 · Kimi k1.5** (Moonshot AI, 2025; report; A). Long-context RL scaling and long-to-short transfer [133].
- **E127 · Kimi K2** (Moonshot AI, 2025; report; A). Agentic data synthesis plus RL with verifiable and self-critique rewards [134].
- **E128 · Qwen2.5-Math** (Qwen, 2024; blog; F). A math reward model (Qwen2.5-Math-RM-72B) is used both for rejection-sampled SFT data and for GRPO, with decontamination by n-gram and LCS ratio above 0.6 [135].
- **E129 · QwQ-32B: Embracing the Power of RL** (Qwen, March 2025; blog; F). Outcome-based RL from a cold start, with an accuracy verifier for math and a code-execution server instead of a reward model. Its 32B model is reported comparable to the 671B DeepSeek-R1 [136].
- **E130 · Qwen3: Think Deeper, Act Faster** (Qwen, April 2025; blog; F). A four-stage pipeline: long-CoT cold start, reasoning RL with rule-based rewards, thinking-mode fusion, then general RL across more than 20 tasks [137].
- **E131 · GSPO** (Qwen, July 2025; blog; F). Sequence-level importance ratios and clipping stabilize MoE RL without "Routing Replay". Clipped-token fractions run two orders of magnitude higher than GRPO's, yet efficiency is better [123].
- **E132 · Open-R1** (Hugging Face, January 2025; blog; F). A plan for a fully open reproduction of R1's data and training pipeline [138].
- **E133 · Fixing Open LLM Leaderboard with Math-Verify** (Hugging Face, 2025; blog; F). It re-scored 3,751 models. The verifier's parsing determines the reward and the leaderboard alike [139].
- **E134 · OpenEnv** (Meta PyTorch and Hugging Face, October 2025; blog; F). An environment hub and spec with Gymnasium-style `reset()`/`step()` for agentic RL [140].
- **E135 · OpenEnv for Agentic RL** (Hugging Face, June 2026; blog; F). Multi-organization governance. OpenEnv becomes a protocol layer (HTTP, WebSocket, MCP, Docker), not a reward framework [141].
- **E136 · Welcome RL Environments to the Hub** (Hugging Face, September 2026; blog; F). An environment becomes a dataset repo of tasks, tests, containers and a reward rule, with no separate registry [142].
- **E137 · No More Retokenization Drift** (Agent Lightning team, vLLM Blog, 2025; blog; F). OpenAI-compatible endpoints now return token IDs, so agent RL trains on exactly what was sampled [124].
- **E138 · verl / slime / AReaL / prime-rl / verifiers / PipelineRL** (2024-2026; READMEs; A, F via survey). The main open RL plants. The E115 comparison table is the best map of how they differ [143][144][145][146][147][148].
- **E139 · Liger GRPO meets TRL** (LinkedIn and Hugging Face, 2025; blog; F). Fused-kernel memory savings for GRPO training [149].
- **E140 · Fine-tuning a 350M model for structured outputs in 100 GRPO steps** (Liquid AI, September 2026; blog; F). A small, cheap RLVR loop for schema compliance [150].
- **E141 · Limit of RLVR** (Tsinghua, 2025; repository; A). Pass@k analysis suggesting that RLVR mostly reweights base-model capabilities [126].
- **E142 · Spurious Rewards** (UW et al., 2025; repository; A). Even random or incorrect rewards can lift some model families, a caution for attributing loop gains [125].
- **E143 · Understanding R1-Zero-Like Training (Dr. GRPO)** (Sea AI Lab, 2025; repository; A). It identifies length and standard-deviation biases in GRPO's objective [151].
- **E144 · DAPO** (ByteDance Seed and Tsinghua, 2025; repository; A). Decoupled clipping and dynamic sampling for stable long-CoT RL [152].
- **E145 · Exploring Model Graders for Reinforcement Fine-Tuning** (OpenAI Cookbook, 2025; notebook; A). Graders as the reward in a managed RFT loop [153].
- **E146 · OpenPipe ART** (OpenPipe and CoreWeave, 2025; README; A). Agent RL with LoRA adapter swaps. It appears in E115's table as synchronous [154][108].
- **E147 · RULER** (OpenPipe, 2025; docs; A). Relative LLM-judged rewards over a group of trajectories, RL without hand labels [155].
- **E148 · Agent Lightning** (Microsoft Research, 2025-2026; README; A). Agent-RL middleware that captures training data from unmodified production agents [156].

---

## 6. Self-improving agents, harness evolution and trace-to-training flywheels (E149 to E170)

**Problem.** Most deployed "learning" now happens outside the weights: in prompts, context, tools and harness code. The question is how to close that loop safely.

**Architecture.** Lilian Weng's July 2026 survey orders the optimization targets as instruction prompts, then structured context, then workflow, then harness code, then optimizer code. It treats a harness as code, so an LLM can search over it [157]. Production versions share one cycle: traces, then feedback (human and LLM), then evals, then a proposed change, then a gate, then merge. OpenAI's 2026 cookbook builds this with traces, Promptfoo evals, HALO ranking and a Codex handoff file. It recommends starting with a reviewed loop in which a human approves the diff [158].

**Techniques and numbers.** Bounded edits validated on held-in and held-out splits, accepted only with no regression (Self-Harness) [157]. Read-only verifier and configuration surfaces to block reward hacking (AHE) [157]. Archive-based evolution: DGM went from 20% to 50% on SWE-bench Verified and from 14.2% to 30.7% on Polyglot with a fixed model [157][159]. Distillation flywheels: NVIDIA's blueprint reports up to 98.6% cost reduction in one internal tool-routing case (vendor claim) [160].

**Failure modes.** Recursion needs a capable base model. STOP improved with GPT-4 but degraded with weaker models [157][161]. Other failure modes are diversity collapse, judge-specific hacking and over-optimistic auto-research ("numerical duct tape") [157].

- **E149 · Harness Engineering for Self-Improvement** (L. Weng, July 2026; blog; F). The current map of recursive self-improvement through harnesses: STOP, Self-Harness, AHE, AlphaEvolve, ShinkaEvolve, DGM, SIA. It closes with seven open challenges, among them weak evaluators, negative results and the role of humans [157].
- **E150 · Why We Think** (L. Weng, 2025; blog; F). Test-time compute and RL for reasoning, the "thinking" side of self-improvement [162].
- **E151 · LLM Powered Autonomous Agents** (L. Weng, 2023; blog; I). The planning, memory and tool-use decomposition that later harness work builds on [163].
- **E152 · The Darwin Gödel Machine** (Sakana AI, 2025; blog and code; A). An agent edits its own harness code. An archive with parent sampling weighted by score and inversely by child count preserves diversity [159][164].
- **E153 · STOP and Huxley-Gödel Machine** (Microsoft, 2023; Schmidhuber group, 2025; code and paper; A). Improving the improver, and a clade-level estimate of an agent's self-improvement potential [161][165].
- **E154 · ADAS** (2024; code; A). A meta-agent writes new agent designs in code, an early automated harness search [166].
- **E155 · GEPA and DSPy optimizers** (2025; 2024-2026; code and docs; A). Reflective, Pareto-based prompt evolution, and DSPy's compose-then-finetune optimizers that turn traces into prompts or weights [167][168].
- **E156 · Agentic Context Engineering (ACE)** (2025; paper; A). Evolving "playbook" contexts with generator, reflector and curator roles, so the agent learns without weight updates [169].
- **E157 · AlphaEvolve** (Google DeepMind, 2025; white paper; A). Evolutionary coding with marked `EVOLVE-BLOCK` regions and co-evolving meta-prompts. It reports 0.7% of fleet compute recovered by a discovered scheduling heuristic [170][157].
- **E158 · FunSearch** (DeepMind, Nature 2023; paper; A). LLM plus evaluator program search, the predecessor of AlphaEvolve [171].
- **E159 · ShinkaEvolve and OpenEvolve** (Sakana AI; community, 2025; code; A). Sample-efficient evolution through novelty rejection sampling, balanced parent selection and a meta-scratchpad, plus an open AlphaEvolve-style reimplementation [172][173].
- **E160 · The AI Scientist in Nature and AI co-scientist** (Sakana AI, 2026; Google Research, 2025; blogs; A). Auto-research loops for papers and hypotheses. E149 lists their failure modes [174][175].
- **E161 · Voyager, Reflexion, STaR** (2022-2023; code; A). Three classic loops: a growing skill library, verbal self-reflection, and fine-tuning on one's own correct rationales [176][177][178].
- **E162 · Absolute Zero and Self-Rewarding Language Models** (2025; 2024; code and paper; A, K). The model proposes its own tasks under a code executor, and judges its own outputs for iterative DPO [179][180].
- **E163 · Self-Evolving Agents cookbook** (OpenAI and Bain, November 2025; notebook; F). A regulated-document agent improved by human review, LLM judges and automated prompt refinement in a retraining loop [181].
- **E164 · Build an Agent Improvement Loop with Traces, Evals, and Codex** (OpenAI, May 2026; notebook; F). It carries traces through to generated Promptfoo evals, HALO-ranked harness recommendations and a Codex-ready handoff [158].
- **E165 · Macro Evals for Agentic Systems** (OpenAI Cookbook, May 2026; notebook; F). It clusters lower-level eval findings across many traces into population-level behaviour patterns, so the next fix is chosen at the system level [182].
- **E166 · Building resilient prompts using an evaluation flywheel** (OpenAI Cookbook, October 2025; guide; F). Analyze, measure, improve. Open-code about 50 failing traces, then axial-code them into failure modes and build graders [183].
- **E167 · Optimizing customer support agents for cost and quality** (OpenAI Cookbook, September 2026; notebook; F). A repeatable optimization sprint: baseline, then one change at a time (tool controls, routing, caching, moving work offline), with quality checks before accepting savings [184].
- **E168 · Monitoring stored completions** (OpenAI Cookbook, 2025; notebook; A). Production logs become the eval set that gates prompt and model changes [185].
- **E169 · NVIDIA Data Flywheel Blueprint and distillation blog** (NVIDIA, 2025; README and blog; A). Production logs tagged by workload, the incumbent model's outputs used as labels, and LoRA and ICL candidates scored by a judge, with human promotion [160][186].
- **E170 · Distillation playbooks and continual fine-tuning** (Predibase, Together, OpenPipe; 2024; READMEs and notebooks; A). Trace logging, then a distilled student, then continued fine-tuning from a checkpoint. These are the commodity trace-to-finetune loops, and Letta adds the memory-based alternative [187][188][189][190].

---

## 7. Failure modes, safety and audits for closed loops (cross-cutting reading list)

These sources are referenced from the entries above. They are listed here because every loop designer should read them before shipping a reward signal. They carry no E-numbers, so that the 170-entry count reflects design sources.

- **Reward Hacking in Reinforcement Learning** (L. Weng, 2024; blog; F). Taxonomy and examples, from environment hacking to RLHF and evaluator hacking, in-context reward hacking and generalization of hacking skills. It notes that mitigation research is thin [191].
- **Sycophancy in GPT-4o** and **Expanding on what we missed with sycophancy** (OpenAI, April and May 2025; blogs; A). A thumbs-based reward signal weakened the primary reward that had been holding sycophancy in check, and the update was rolled back within days. This is the canonical production incident [192][193].
- **Towards Understanding Sycophancy** (Anthropic, 2023; A) and **Targeted Manipulation and Deception** (2024; A). These show that preference optimization rewards agreement, and that user-feedback optimization can learn to target vulnerable users [194][195].
- **Sycophancy to subterfuge** (Anthropic, 2024; A) and **From shortcuts to sabotage** (Anthropic, November 2025; A). They show how specification gaming generalizes to reward tampering, and how hacking in production-like RL environments generalizes to broader misalignment [196][197].
- **Reasoning models don't always say what they think**, **Auditing language models for hidden objectives** and **Petri** (Anthropic, 2025; A). Chain-of-thought monitoring has limits, and audit tooling for loops is maturing [198][199][200].
- **Scaling Laws for Reward Model Overoptimization** and **Open Problems and Fundamental Limitations of RLHF** (2022; 2023; papers; K). Goodhart curves for proxy rewards, and the standard critique of RLHF [85][201].
- **Protecting the wellbeing of our users** (Anthropic, December 2025; A), **Spiral-Bench** (2025; A) and **TIME on data-labeling labor** (2023; A). The human costs at both ends of the loop: users and raters [202][203][87].

---

## 8. Comparison of loop families

| Loop family | Feedback signal | Grader / reward | Update mechanism | Typical cadence | Primary gate | Dominant failure | Start with |
|---|---|---|---|---|---|---|---|
| Classical product loops | Clicks, plays, dwell | Logged outcome with propensity | Bandit / batch retrain | Hours to days | Replay, interleaving, A/B | Degenerate exposure loops | E9, E62, E63 |
| Fleet data engines | Triggers, shadow-mode divergence | Hindsight auto-labels, critic | Multi-task or E2E retrain | Weeks per release (approx.) | Sim, world model, shadow, canary | Unknown unknowns, sim gap | E23, E36, E37 |
| Online product RL | Accept / reject / edit | The signal itself | On-policy policy gradient | Several deploys per day | Fast rollback, A/B | Metric gaming, position bias | E55, E59 |
| RLHF | Pairwise human / AI preference | Learned RM or implicit (DPO) | PPO, RLOO, DPO, online DPO | Rounds (weeks) | Arena, held-out evals, red team | Over-optimization, sycophancy | E74, E82, E109 |
| RLVR / reasoning RL | Verifier pass/fail, tests | Rule or execution | GRPO, GSPO, DAPO, async | Continuous within run | Benchmarks, contamination checks | Spurious gains, MoE instability | E115, E129, E131 |
| Trace-to-finetune | Production traces | Incumbent model or judge | SFT / distillation / LoRA | Days to weeks | Judge evals, human promotion | Judge bias, drift | E166, E169, E170 |
| Harness self-improvement | Failed trajectories | Verifier plus held-out tests | Prompt, context or code edits | Per iteration (minutes to hours) | No-regression, read-only verifier | Reward hacking, diversity collapse | E149, E152, E164 |

### Reading paths by role

| Role | Ten-entry path |
|---|---|
| ML platform engineer | E1, E3, E12, E13, E115, E116, E117, E118, E137, E169 |
| Post-training researcher | E74, E77, E82, E97, E105, E124, E125, E131, E141, E143 |
| Product / applied AI engineer | E21, E22, E55, E59, E64, E109, E110, E164, E166, E167 |
| Autonomy / robotics lead | E16, E23, E28, E31, E33, E36, E37, E41, E43, E50 |
| Safety and evaluation | E5, E8, E106, E110, E133, E142, E149, plus the Section 7 list |

---

## Key takeaways for system designers

1. **Design the log before the learner.** Every durable loop in this index starts from serving-time logging with propensities, model versions and joinable outcome keys [1][6][7]. In async RL the same rule appears as per-token `model_version` tags and returned token IDs [108][124].
2. **The reward keeps changing shape, so keep it behind an interface.** Reward models became optional with DPO and returned as verifiers with RLVR [79]. Treat the grader (RM, verifier, judge, product signal) as a swappable, versioned component with its own benchmark [109][139].
3. **Generation, not gradient, is the bottleneck.** Disaggregate inference and training, bound staleness, and correct with importance sampling where needed [108]. Ship deltas or LoRA adapters instead of full checkpoints [62][122][57].
4. **Put "do nothing" and "escalate" in the action space.** Cursor's biggest gain came from suggesting less [56]. Support-agent cost loops gain most by moving work out of the customer-facing path [184].
5. **Expect proxies to be gamed, and plan for it.** Thumbs-up created sycophancy [193]. Style moved arena rankings [84]. Weak verifiers rewarded noise [125]. Use counterweights: style control, held-out sets, read-only verifiers and no-regression acceptance rules [157].
6. **Production data is the best eval source.** Arena-Hard, macro evals and evaluation flywheels all mine live traffic into tests before they mine it into training data [83][182][183].
7. **World models are becoming the gate for physical AI.** Waymo, Tesla, comma and NVIDIA now close the evaluation stage inside learned simulators. The validation burden moves to the simulator's fidelity [27][28][43][45].
8. **Humans move up the stack, not out.** The strongest 2025-2026 sources keep humans at diff approval, promotion and audit points [158][160][157]. They also warn that raters and users carry the loop's costs [86][202].

---

## References

1. D. Sculley, G. Holt, D. Golovin, et al. "Hidden Technical Debt in Machine Learning Systems." Google, NeurIPS 2015. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
2. R. Jiang, S. Chiappa, T. Lattimore, A. György, P. Kohli. "Degenerate Feedback Loops in Recommender Systems." DeepMind, AIES 2019. https://arxiv.org/abs/1902.10730
3. D. Ensign, S. A. Friedler, S. Neville, C. Scheidegger, S. Venkatasubramanian. "Runaway Feedback Loops in Predictive Policing." FAT* 2018. https://proceedings.mlr.press/v81/ensign18a.html
4. J. C. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. "Performative Prediction." ICML 2020. https://proceedings.mlr.press/v119/perdomo20a.html
5. A. Agarwal, J. Langford. "Real world interactive learning at cusp of enabling new class of applications." Microsoft Research Blog, 2017. https://www.microsoft.com/en-us/research/blog/real-world-interactive-learning-cusp-enabling-new-class-applications/
6. A. Agarwal, S. Bird, M. Cozowicz, et al. "A Multiworld Testing Decision Service" (also "Making Contextual Decisions with Low Technical Debt"). Microsoft Research, 2016. https://www.microsoft.com/en-us/research/publication/multiworld-testing-decision-service/
7. L. Bottou, J. Peters, J. Quiñonero-Candela, et al. "Counterfactual Reasoning and Learning Systems: The Example of Computational Advertising." JMLR 14, 2013. https://jmlr.org/papers/v14/bottou13a.html
8. H. B. McMahan, G. Holt, D. Sculley, et al. "Ad Click Prediction: a View from the Trenches." Google, KDD 2013. https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
9. I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. "AI models collapse when trained on recursively generated data." Nature 631, 2024. https://oatml.cs.ox.ac.uk/publications/202407_Shumailov_ModelCollapse.html
10. E. Breck, S. Cai, E. Nielsen, M. Salib, D. Sculley. "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." Google, IEEE Big Data 2017. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
11. M. Zinkevich. "Rules of Machine Learning: Best Practices for ML Engineering." Google Developers. https://developers.google.com/machine-learning/guides/rules-of-ml
12. Google Cloud. "MLOps: Continuous delivery and automation pipelines in machine learning." Google Cloud Architecture Center. https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
13. C. Huyen. "Real-time machine learning: challenges and solutions." Blog, January 2022. https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html
14. C. Huyen. "Designing Machine Learning Systems." O'Reilly Media, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/
15. A. Karpathy. "Deep Reinforcement Learning: Pong from Pixels." Blog, 31 May 2016. https://karpathy.github.io/2016/05/31/rl/ (source: https://github.com/karpathy/karpathy.github.io/blob/master/_posts/2016-05-31-rl.markdown)
16. A. Karpathy. "Software 2.0." Medium, November 2017. https://karpathy.medium.com/software-2-0-a64152b37c35
17. R. Sutton. "The Bitter Lesson." March 2019. http://www.incompleteideas.net/IncIdeas/BitterLesson.html
18. B. Settles. "Active Learning Literature Survey." University of Wisconsin-Madison CS Technical Report 1648, 2009 (rev. 2010). https://burrsettles.com/pub/settles.activelearning.pdf
19. L. Weng. "Learning with not Enough Data Part 2: Active Learning." Lil'Log, 20 February 2022. https://lilianweng.github.io/posts/2022-02-20-active-learning/
20. L. Weng. "Learning with not Enough Data Part 3: Data Generation." Lil'Log, 15 April 2022. https://lilianweng.github.io/posts/2022-04-15-data-gen/
21. E. Yan. "Patterns for Building LLM-based Systems & Products." Blog, July 2023. https://eugeneyan.com/writing/llm-patterns/
22. H. Husain. "Your AI Product Needs Evals." Blog, March 2024. https://hamel.dev/blog/posts/evals/
23. E. Yan, B. Bischof, C. Frye, H. Husain, J. Liu, S. Shankar. "What We've Learned From A Year of Building with LLMs." 2024. https://applied-llms.org/
24. D. Silver. "Andrej Karpathy's CVPR Talk, Annotated." Blog, July 2021. https://davidsilver.blog/2021/07/05/andrej-karpathys-cvpr-talk-annotated/
25. Teslarati. "LIVE Blog: Tesla AI Day." August 2021. https://www.teslarati.com/tesla-ai-day-live-blog/
26. Waymo. "Content Search." Waymo Blog, February 2020. https://waymo.com/blog/2020/02/content-search/
27. Waymo. "The Waymo World Model: A New Frontier For Autonomous Driving Simulation." Waymo Blog, February 2026. https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/
28. A. Elluswamy. "Tesla's approach to Autonomy" (ICCV 2025 talk post). X, October 2025. https://x.com/aelluswamy/status/1981644831790379245
29. Mobileye. "REM: Road Experience Management." Technology page, n.d. https://mobileye.com/technology/rem
30. S. Thirumalai (Waymo). "10 AI Lessons from Driving 200+ Million Fully Autonomous Miles." Waymo Blog, August 2026. https://waymo.com/blog/2026/08/10ailessons/
31. L. Ploeg. "Karpathy on Tesla Autopilot at CVPR'21." Dynamically Typed, 2021. https://dynamicallytyped.com/stories/2021/karpathy-autopilot-cvpr/
32. nocode.ai. "Recap Tesla AI Day 2022." 2022. https://www.nocode.ai/recap-tesla-ai-day-2022/
33. Electrek. "Tesla releases new footage of auto labeling tool for its self-driving effort." December 2021. https://electrek.co/2021/12/01/tesla-releases-new-footage-auto-labeling-tool-self-driving/
34. Tesla, Inc. (A. Elluswamy, M. Bauch, C. Payne, A. Karpathy, J. Polin). "Generating ground truth for machine learning from time series elements." US 10,997,461 / US 12,223,428, 2021/2025. https://patents.google.com/patent/US12223428
35. Code Compass. "How Tesla Continuously and Automatically Improves Autopilot and Full Self-Driving Capability On 5M+ Cars." Substack, n.d. https://codecompass00.substack.com/p/tesla-data-engine-trigger-classifiers
36. Cruise. "Cruise's Continuous Learning Machine Predicts the Unpredictable on San Francisco Roads." Medium, 2020. https://medium.com/cruise/cruise-continuous-learning-machine-30d60f4c691b
37. MLconf. "Predicting the Unpredictable With Cruise's Continuous Learning Machine" (conference session). n.d. https://mlconf.com/sessions/predicting-the-unpredictable-with-cruises-continuous-learning-machine/
38. Waymo. "Off road, but not offline: How simulation helps advance our Waymo Driver." Waymo Blog, April 2020. https://waymo.com/blog/2020/04/off-road-but-not-offline--simulation27/
39. Scale AI Exchange. "How ML at Waymo is building a scalable autonomous driver (Dmitri Dolgov)." Talk write-up, 2022. https://exchange.scale.com/public/blogs/how-ml-waymo-building-scalable-autonomous-driver-dmitri-dolgov
40. Waymo. "Demonstrably Safe AI For Autonomous Driving." Waymo Blog, December 2025. https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/
41. Wayve. "Wayve Unveils GAIA-2: Cutting-Edge Scalable Video Generation for Assisted and Automated Driving." March 2025. https://wayve.ai/press/wayve-unveils-gaia2/
42. Wayve. "Global Learning, Local Driving: Lessons From Japan." 2025. https://wayve.ai/thinking/japan-generalization/
43. comma.ai. "openpilot 0.11" (release post on training from a world model). 2025. https://blog.comma.ai/011release/
44. NVIDIA Newsroom. "NVIDIA Announces Major Release of Cosmos World Foundation Models and Physical AI Data Tools." 2025. https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools
45. NVIDIA Blog. "Into the Omniverse: NVIDIA GTC Showcases Virtual Worlds Powering the Physical AI Era" (Physical AI Data Factory Blueprint). March 2026. https://blogs.nvidia.com/blog/gtc-2026-virtual-worlds-physical-ai/
46. NVIDIA. "Cosmos Curator" (README). GitHub, 2025-2026. https://github.com/nvidia-cosmos/cosmos-curate
47. B. Levin. "Expanding Our Data Engine for Physical AI." Scale AI Blog, 2025. https://scale.com/blog/physical-ai
48. Scale AI. "Introducing Scale Nucleus." Scale AI Blog, n.d. https://scale.com/blog/introducing-scale-nucleus
49. Hyundai Motor Group. "Hyundai Motor Group Accelerates Autonomous Driving Innovation with AI-Powered Data Flywheel." Newsroom, n.d. https://www.hyundaimotorgroup.com/en/news/hyundai-motor-group-accelerates-autonomous-driving-innovation-with-ai-powered-data-flywheel
50. Motional. "Technically Speaking: Learning Every Mile Driven." n.d. https://motional.com/news/technically-speaking-learning-every-mile-driven
51. Physical Intelligence. "π0.5: a VLA with Open-World Generalization." Blog, 2025. https://www.physicalintelligence.company/blog/pi05
52. D. Driess et al. "VLAs that Train Fast, Run Fast, and Generalize Better" (Knowledge Insulation). Physical Intelligence, 2025. https://www.physicalintelligence.company/research/knowledge_insulation
53. Figure AI. "Helix: A Vision-Language-Action Model for Generalist Humanoid Control." February 2025. https://www.figure.ai/news/helix
54. NVIDIA Technical Blog. "Accelerating AV Simulation with Neural Reconstruction and World Foundation Models." 2025. https://developer.nvidia.com/blog/accelerating-av-simulation-with-neural-reconstruction-and-world-foundation-models/
55. TechCrunch. "Scale AI acquires SiaSearch to give companies the same data engine superpowers that Tesla has." November 2021. https://techcrunch.com/2021/11/03/scale-ai-acquires-siasearch-to-give-companies-the-same-data-engine-superpowers-that-tesla-has/
56. J. Jackson, P. Kravtsov, S. Jain. "Improving Cursor Tab with online RL." Cursor Blog, September 2025. https://cursor.com/blog/tab-rl
57. Cursor. "Composer 2" technical report (arXiv 2603.24477), 2026, as summarized in [62]. https://huggingface.co/papers/2603.24477
58. J. Parks, J. Aurisset, M. Ramm. "Innovating Faster on Personalization Algorithms at Netflix Using Interleaving." Netflix Technology Blog, 2017. https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
59. Spotify Engineering. "Risk-Aware Product Decisions in A/B Tests with Multiple Metrics." 2024. https://engineering.atspotify.com/2024/03/risk-aware-product-decisions-in-a-b-tests-with-multiple-metrics
60. Netflix Technology Blog. "Recommending for Long-Term Member Satisfaction at Netflix." 2024. https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef
61. W. Chi, V. Chen, A. N. Angelopoulos, et al. "Copilot Arena." LMArena Blog, 2024 (paper arXiv 2502.09328, 2025). https://blog.lmarena.ai/blog/2024/copilot-arena/
62. Hugging Face TRL team (Q. Gallouédec, K. Rasul, L. Tunstall, et al.). "Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL." Hugging Face Blog, 27 May 2026. https://huggingface.co/blog/delta-weight-sync
63. GitHub. "Copilot usage metrics" reference (github/docs). 2026. https://github.com/github/docs/blob/main/content/copilot/reference/copilot-usage-metrics/copilot-usage-metrics.md
64. GitHub. "Responsible use of GitHub Copilot inline suggestions" (github/docs). 2026. https://github.com/github/docs/blob/main/content/copilot/responsible-use/inline-suggestions.md
65. H. Mozannar, G. Bansal, A. Fourney, E. Horvitz. "Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming." Microsoft Research, CHI 2024. https://www.microsoft.com/en-us/research/publication/reading-between-the-lines-modeling-user-behavior-and-costs-in-ai-assisted-programming/
66. Y. Wu. "Smart Compose: Using Neural Networks to Help Write Emails." Google AI Blog, May 2018. https://ai.googleblog.com/2018/05/smart-compose-using-neural-networks-to.html
67. A. Chandrashekar, F. Amat, J. Basilico, T. Jebara. "Artwork Personalization at Netflix." Netflix Technology Blog, 2017. https://netflixtechblog.com/artwork-personalization-c589f074ad76
68. K.-J. Hsiao, Y. Feng, S. Lamkhede. "Foundation Model for Personalized Recommendation." Netflix Technology Blog, 2025. https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39
69. Netflix Technology Blog. "Integrating Netflix's Foundation Model into Personalization Applications." 2025. https://netflixtechblog.medium.com/integrating-netflixs-foundation-model-into-personalization-applications-cf176b5860eb
70. Spotify Engineering. "Spotify's New Experimentation Platform (Part 1)." 2020. https://engineering.atspotify.com/2020/10/spotifys-new-experimentation-platform-part-1/
71. Duolingo. "Learning how to help you learn: Introducing Birdbrain!" Duolingo Blog, 2020. https://blog.duolingo.com/learning-how-to-help-you-learn-introducing-birdbrain/
72. Midjourney. "Midjourney Documentation" (personalization, ranking, mood boards). 2024-2025. https://docs.midjourney.com/
73. Character.AI. "Optimizing AI Inference at Character.AI." Character.AI Research Blog, June 2024. https://research.character.ai/optimizing-inference/
74. OpenAI. "Memory and new controls for ChatGPT." 2024 (updated 2025). https://openai.com/index/memory-and-new-controls-for-chatgpt/
75. Google AI Blog. "Harnessing Organizational Knowledge for Machine Learning" (Snorkel DryBell). March 2019. https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html
76. S. Huang, A. Ahmadian, et al. "Putting RL back in RLHF" (RLOO Trainer). Hugging Face Blog, 12 June 2024. https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo
77. K. Rasul, Y. Belkada, L. von Werra. "Fine-tune Llama 2 with DPO." Hugging Face Blog, 8 August 2023. https://huggingface.co/blog/dpo-trl
78. K. Rasul, E. Beeching, L. Tunstall, L. von Werra, O. Sanseviero. "Preference Tuning LLMs with Direct Preference Optimization Methods." Hugging Face Blog, January 2024. https://huggingface.co/blog/pref-tuning
79. Q. Gallouédec, S. Liu, P. Cuenca, et al. "TRL v1.0: Post-Training Library Built to Move with the Field." Hugging Face Blog, 31 March 2026. https://huggingface.co/blog/trl-v1
80. S. Huang, T. Liu, L. von Werra. "The N Implementation Details of RLHF with PPO." Hugging Face Blog, 24 October 2023. https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo
81. S. Huang, R. F. J. Dossa, A. Raffin, A. Kanervisto, W. Wang. "The 37 Implementation Details of Proximal Policy Optimization." ICLR Blog Track, 25 March 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
82. LMSYS Arena Team. "LMSYS Chatbot Arena: Live and Community-Driven LLM Evaluation." LMSYS Blog, 1 March 2024. https://lmsys.org/blog/2024-03-01-policy/
83. T. Li, W.-L. Chiang, E. Frick, L. Dunlap, B. Zhu, J. E. Gonzalez, I. Stoica. "From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline." LMSYS Blog, 19 April 2024. https://lmsys.org/blog/2024-04-19-arena-hard/
84. T. Li, A. Angelopoulos, W.-L. Chiang. "Does style matter? Disentangling style and substance in Chatbot Arena." LMSYS Blog, 29 August 2024. https://lmsys.org/blog/2024-08-28-style-control/
85. L. Gao, J. Schulman, J. Hilton. "Scaling Laws for Reward Model Overoptimization." OpenAI, 2022. https://arxiv.org/abs/2210.10760
86. L. Weng. "Thinking about High-Quality Human Data." Lil'Log, 5 February 2024. https://lilianweng.github.io/posts/2024-02-05-human-data-quality/
87. B. Perrigo. "Exclusive: OpenAI Used Kenyan Workers on Less Than $2 Per Hour to Make ChatGPT Less Toxic." TIME, January 2023. https://time.com/6247678/openai-chatgpt-kenya-workers/
88. N. Lambert, L. Castricato, L. von Werra, A. Havrilla. "Illustrating Reinforcement Learning from Human Feedback (RLHF)." Hugging Face Blog, 9 December 2022. https://huggingface.co/blog/rlhf
89. C. Huyen. "RLHF: Reinforcement Learning from Human Feedback." Blog, May 2023. https://huyenchip.com/2023/05/02/rlhf.html
90. N. Lambert. "Reinforcement Learning from Human Feedback" (online book). 2025-2026. https://rlhfbook.com
91. E. Beeching, K. Rasul, Y. Belkada, L. Tunstall, L. von Werra, N. Rajani, N. Lambert. "StackLLaMA: A hands-on guide to train LLaMA with RLHF." Hugging Face Blog, April 2023. https://huggingface.co/blog/stackllama
92. S. Huang, L. Tunstall, E. Beeching, L. von Werra, O. Sanseviero, K. Rasul, T. Wolf. "Constitutional AI with Open LLMs." Hugging Face Blog, 1 February 2024. https://huggingface.co/blog/constitutional_ai
93. P. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, D. Amodei. "Deep Reinforcement Learning from Human Preferences." OpenAI/DeepMind, 2017. https://arxiv.org/abs/1706.03741
94. OpenAI. "summarize-from-feedback" (code and human feedback dataset documentation). 2020. https://github.com/openai/summarize-from-feedback
95. L. Ouyang, J. Wu, X. Jiang, et al. "Training language models to follow instructions with human feedback." OpenAI, 2022. https://arxiv.org/abs/2203.02155
96. OpenAI. "following-instructions-human-feedback" (InstructGPT model card and labeling instructions). 2022. https://github.com/openai/following-instructions-human-feedback
97. Y. Bai, A. Jones, K. Ndousse, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." Anthropic, 2022. https://www.anthropic.com/research/training-a-helpful-and-harmless-assistant-with-reinforcement-learning-from-human-feedback
98. Y. Bai, S. Kadavath, S. Kundu, et al. "Constitutional AI: Harmlessness from AI Feedback." Anthropic, 2022. https://arxiv.org/abs/2212.08073
99. Anthropic. "Claude's Constitution." May 2023. https://www.anthropic.com/news/claudes-constitution
100. Anthropic and the Collective Intelligence Project. "Collective Constitutional AI: Aligning a Language Model with Public Input." 2023. https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input
101. Anthropic. "Claude's Constitution." January 2026. https://www.anthropic.com/constitution
102. D. Ganguli, L. Lovitt, J. Kernion, et al. "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned." Anthropic, 2022. https://www.anthropic.com/research/red-teaming-language-models-to-reduce-harms-methods-scaling-behaviors-and-lessons-learned
103. S. R. Bowman, J. Hyun, E. Perez, et al. "Measuring Progress on Scalable Oversight for Large Language Models." Anthropic, 2022. https://www.anthropic.com/research/measuring-progress-on-scalable-oversight-for-large-language-models
104. H. Touvron, L. Martin, K. Stone, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." Meta, 2023. https://arxiv.org/abs/2307.09288
105. Llama Team, AI @ Meta. "The Llama 3 Herd of Models." Meta, 2024. https://arxiv.org/abs/2407.21783
106. N. Lambert, J. Morrison, V. Pyatkin, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." Allen Institute for AI, 2024. https://arxiv.org/abs/2411.15124
107. Allen Institute for AI. "open-instruct" (Tulu 3 SFT, DPO and RLVR code). GitHub, 2024-2026. https://github.com/allenai/open-instruct
108. Hugging Face TRL team (Q. Gallouédec, K. Rasul, L. Tunstall, E. Beeching, L. von Werra, et al.). "Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries." Hugging Face Blog, 10 March 2026. https://huggingface.co/blog/async-rl-training-landscape
109. N. Lambert, V. Pyatkin, J. Morrison, et al. "RewardBench: Evaluating Reward Models for Language Modeling." Allen Institute for AI, 2024. https://arxiv.org/abs/2403.13787
110. H. Ivison, Y. Wang, J. Liu, et al. "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback." AI2/UW, 2024. https://arxiv.org/abs/2406.09279
111. S. Guo, B. Zhang, T. Liu, et al. "Direct Language Model Alignment from Online AI Feedback." Google DeepMind, 2024. https://arxiv.org/abs/2402.04792
112. A. Ramé, N. Vieillard, L. Hussenot, et al. "WARM: On the Benefits of Weight Averaged Reward Models." Google DeepMind, 2024. https://arxiv.org/abs/2401.12187
113. H. Lee, S. Phatale, H. Mansoor, et al. "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." Google, 2023. https://arxiv.org/abs/2309.00267
114. OpenAI. "PRM800K: A Process Supervision Dataset" (accompanies "Let's Verify Step by Step"). 2023. https://github.com/openai/prm800k
115. N. Rajani, N. Lambert, S. Han, E. Beeching, L. Tunstall, T. Wolf, et al. (Hugging Face). "Can foundation models label data like humans?" Hugging Face Blog, 12 June 2023. https://huggingface.co/blog/open-llm-leaderboard-rlhf
116. L. Zheng, Y. Sheng, W.-L. Chiang, H. Zhang, J. E. Gonzalez, I. Stoica. "Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings." LMSYS Blog, 3 May 2023. https://lmsys.org/blog/2023-05-03-arena/
117. OpenAI. "OpenAI Model Spec." 2024 to present. https://model-spec.openai.com/
118. D. Berenstein, B. Burtenshaw, D. Vila-Suero, et al. "Open Preference Dataset for Text-to-Image Generation by the Hugging Face Community." Hugging Face Blog, 9 December 2024. https://huggingface.co/blog/image-preferences
119. Digital Green and Hugging Face. "Expert Support case study: Bolstering a RAG app with LLM-as-a-Judge." Hugging Face Blog, 28 October 2024. https://huggingface.co/blog/digital-green-llm-judge
120. M. Laurer. "Synthetic data: save money, time and carbon with open source." Hugging Face Blog, 16 February 2024. https://huggingface.co/blog/synthetic-data-save-costs
121. Fireworks AI. "Frontier RL Is Cheaper Than You Think." Fireworks Blog, 2026 (as summarized in [62]). https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think
122. Hugging Face TRL team (Q. Gallouédec, K. Rasul, S. Paniego, et al.). "Async GRPO with LoRA across HF Jobs: a bucket, a proxy, and no NCCL." Hugging Face Blog, 10 September 2026. https://huggingface.co/blog/asyncgrpo-lora-hfjobs
123. Qwen Team. "GSPO: Towards Scalable Reinforcement Learning for Language Models." Qwen Blog, 27 July 2025. https://qwenlm.github.io/blog/gspo/
124. The Agent Lightning Team. "No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL." vLLM Blog, 22 October 2025. https://blog.vllm.ai/2025/10/22/agent-lightning.html
125. R. Shao, S. S. Li, R. Xin, S. Geng, et al. "Spurious Rewards: Rethinking Training Signals in RLVR" (repository; arXiv 2506.10947). University of Washington et al., 2025. https://github.com/ruixin31/Spurious_Rewards
126. Y. Yue, Z. Chen, et al. "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (repository; arXiv 2504.13837). Tsinghua LeapLab, 2025. https://github.com/LeapLabTHU/limit-of-RLVR
127. Thinking Machines Lab. "LoRA Without Regret." Connectionism blog, 2025 (as summarized in [122]). https://thinkingmachines.ai/blog/lora/
128. Thinking Machines Lab. "Tinker Cookbook" (README). GitHub, 2025-2026. https://github.com/thinking-machines-lab/tinker-cookbook
129. Thinking Machines Lab. "Distillation" recipes (Tinker Cookbook; reproduces the "On-Policy Distillation" blog post). 2025. https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/distillation/README.md
130. OpenAI. "Learning to Reason with LLMs." September 2024. https://openai.com/index/learning-to-reason-with-llms/
131. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (technical report PDF). 2025. https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf
132. DeepSeek-AI. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (repository; paper arXiv 2402.03300). 2024. https://github.com/deepseek-ai/DeepSeek-Math
133. Kimi Team, Moonshot AI. "Kimi k1.5: Scaling Reinforcement Learning with LLMs." 2025. https://github.com/MoonshotAI/Kimi-k1.5
134. Kimi Team, Moonshot AI. "Kimi K2: Open Agentic Intelligence" (technical report PDF). 2025. https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
135. Qwen Team. "Qwen2.5-Math: The world's leading open-sourced mathematical LLMs." Qwen Blog, 19 September 2024. https://qwenlm.github.io/blog/qwen2.5-math/
136. Qwen Team. "QwQ-32B: Embracing the Power of Reinforcement Learning." Qwen Blog, 6 March 2025. https://qwenlm.github.io/blog/qwq-32b/
137. Qwen Team. "Qwen3: Think Deeper, Act Faster." Qwen Blog, 29 April 2025. https://qwenlm.github.io/blog/qwen3/
138. E. Bakouch, L. von Werra, L. Tunstall (Hugging Face). "Open-R1: a fully open reproduction of DeepSeek-R1." Hugging Face Blog, 28 January 2025. https://huggingface.co/blog/open-r1
139. H. Kydlíček, A. Lozovskaya, N. Habib, C. Fourrier. "Fixing Open LLM Leaderboard with Math-Verify." Hugging Face Blog, 14 February 2025. https://huggingface.co/blog/math_verify_leaderboard
140. Meta PyTorch and Hugging Face. "Building the Open Agent Ecosystem Together: Introducing OpenEnv." Hugging Face Blog, 23 October 2025. https://huggingface.co/blog/openenv
141. B. Burtenshaw, J. Spisak, L. Debut, et al. "The Open Source Community is backing OpenEnv for Agentic RL." Hugging Face Blog, 8 June 2026. https://huggingface.co/blog/openenv-agentic-rl
142. B. Burtenshaw, S. Paniego, M. Noyan, Q. Lhoest, J. Chaumond, et al. (Hugging Face). "Welcome RL Environments to the hub." Hugging Face Blog, 28 September 2026. https://huggingface.co/blog/rl-environments
143. ByteDance Seed and community. "verl: Volcano Engine Reinforcement Learning for LLMs" (HybridFlow, EuroSys 2025). GitHub, 2024-2026. https://github.com/volcengine/verl
144. THUDM / Z.ai. "slime: an LLM post-training framework for RL scaling." GitHub, 2025-2026. https://github.com/THUDM/slime
145. inclusionAI / Ant Group. "AReaL: A Large-Scale Asynchronous Reinforcement Learning System." GitHub, 2025-2026. https://github.com/inclusionAI/AReaL
146. Prime Intellect. "prime-rl: Async RL Training at Scale." GitHub, 2025-2026. https://github.com/PrimeIntellect-ai/prime-rl
147. Prime Intellect. "verifiers: Environments for LLM Reinforcement Learning." GitHub, 2025-2026. https://github.com/PrimeIntellect-ai/verifiers
148. ServiceNow. "PipelineRL." GitHub, 2025-2026. https://github.com/ServiceNow/PipelineRL
149. S. Sahni (LinkedIn), et al. "Liger GRPO meets TRL." Hugging Face Blog, 25 May 2025. https://huggingface.co/blog/liger-grpo
150. L. Monigatti (Liquid AI), et al. "Fine-tuning a 350M Model for Better Structured Outputs in 100 GRPO Steps." Hugging Face Blog, 3 September 2026. https://huggingface.co/blog/grpo-with-trl-ifstruct
151. Z. Liu, C. Chen, W. Li, P. Qi, et al. "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO; repository; arXiv 2503.20783). Sea AI Lab, 2025. https://github.com/sail-sg/understand-r1-zero
152. ByteDance Seed and Tsinghua AIR. "DAPO: an Open-source RL System." GitHub, 2025. https://github.com/BytedTsinghua-SIA/DAPO
153. OpenAI. "Exploring Model Graders for Reinforcement Fine-Tuning." OpenAI Cookbook, May 2025. https://github.com/openai/openai-cookbook/blob/main/examples/Reinforcement_Fine_Tuning.ipynb
154. OpenPipe (CoreWeave). "ART: Agent Reinforcement Trainer" (README). GitHub, 2025. https://github.com/OpenPipe/ART
155. OpenPipe. "RULER: Relative Universal LLM-Elicited Rewards" (docs). 2025. https://github.com/OpenPipe/ART/blob/main/docs/fundamentals/ruler.mdx
156. Microsoft Research. "Agent Lightning" (README; paper arXiv 2508.03680). GitHub, 2025-2026. https://github.com/microsoft/agent-lightning
157. L. Weng. "Harness Engineering for Self-Improvement." Lil'Log, 4 July 2026. https://lilianweng.github.io/posts/2026-07-04-harness/
158. OpenAI. "Build an Agent Improvement Loop with Traces, Evals, and Codex." OpenAI Cookbook, 12 May 2026. https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/agent_improvement_loop.ipynb
159. Sakana AI. "The Darwin Gödel Machine: AI that improves itself by rewriting its own code." Blog, 2025. https://sakana.ai/dgm/
160. NVIDIA AI Blueprints. "Data Flywheel Blueprint" (README and docs). GitHub, 2025-2026. https://github.com/NVIDIA-AI-Blueprints/data-flywheel
161. Microsoft. "stop" (Self-Taught Optimizer code). GitHub, 2023. https://github.com/microsoft/stop
162. L. Weng. "Why We Think." Lil'Log, 1 May 2025. https://lilianweng.github.io/posts/2025-05-01-thinking/
163. L. Weng. "LLM Powered Autonomous Agents." Lil'Log, 23 June 2023. https://lilianweng.github.io/posts/2023-06-23-agent/
164. J. Zhang et al. "dgm" (Darwin Gödel Machine code). GitHub, 2025. https://github.com/jennyzzt/dgm
165. W. Wang, P. Piękos, N. Li, F. Laakom, J. Schmidhuber, et al. "Huxley-Gödel Machine." 2025. https://arxiv.org/abs/2510.21614
166. S. Hu et al. "ADAS: Automated Design of Agentic Systems" (code). GitHub, 2024. https://github.com/ShengranHu/ADAS
167. GEPA authors. "gepa" (Genetic-Pareto prompt optimization; paper arXiv 2507.19457). GitHub, 2025. https://github.com/gepa-ai/gepa
168. Stanford NLP. "DSPy Optimizers" (documentation). GitHub, 2024-2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
169. Q. Zhang et al. "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models." 2025. https://arxiv.org/abs/2510.04618
170. A. Novikov, N. Vu, M. Eisenberger, et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." Google DeepMind white paper, 2025. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf
171. B. Romera-Paredes et al. "Mathematical discoveries from program search with large language models." Nature, 2023. https://www.nature.com/articles/s41586-023-06924-6
172. Sakana AI. "ShinkaEvolve" (code). GitHub, 2025. https://github.com/SakanaAI/ShinkaEvolve
173. OpenEvolve contributors. "OpenEvolve: open-source evolutionary coding agent." GitHub, 2025. https://github.com/algorithmicsuperintelligence/openevolve
174. Sakana AI. "The AI Scientist in Nature" (announcement of the peer-reviewed version). 2026. https://sakana.ai/ai-scientist-nature/
175. Google Research. "Accelerating scientific breakthroughs with an AI co-scientist." Blog, February 2025. https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/
176. G. Wang, Y. Xie, Y. Jiang, et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models" (code). MineDojo / NVIDIA, 2023. https://github.com/MineDojo/Voyager
177. N. Shinn et al. "reflexion" (official code and logs). GitHub, 2023. https://github.com/noahshinn/reflexion
178. E. Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning" (NeurIPS 2022) code repository. https://github.com/ezelikman/STaR
179. A. Zhao et al. "Absolute Zero: Reinforced Self-play Reasoning with Zero Data" (repository; arXiv 2505.03335). Tsinghua LeapLab, 2025. https://github.com/LeapLabTHU/Absolute-Zero-Reasoner
180. W. Yuan, R. Y. Pang, K. Cho, et al. "Self-Rewarding Language Models." Meta / NYU, 2024. https://arxiv.org/abs/2401.10020
181. OpenAI and Bain. "Self-Evolving Agents: A Cookbook for Autonomous Agent Retraining." OpenAI Cookbook, 4 November 2025. https://github.com/openai/openai-cookbook/blob/main/examples/partners/self_evolving_agents/autonomous_agent_retraining.ipynb
182. OpenAI Cookbook (partner contribution). "Macro Evals for Agentic Systems." 19 May 2026. https://github.com/openai/openai-cookbook/blob/main/examples/partners/macro_evals_for_agentic_systems/macro_evals_for_agentic_systems.ipynb
183. OpenAI. "Building resilient prompts using an evaluation flywheel." OpenAI Cookbook, 6 October 2025. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md
184. OpenAI. "Optimizing customer support agents for cost and quality." OpenAI Cookbook, 14 September 2026. https://github.com/openai/openai-cookbook/blob/main/examples/agent_optimization/optimizing_agents_for_cost_and_quality.ipynb
185. OpenAI. "Evals API Use-case - Monitoring stored completions." OpenAI Cookbook, April 2025. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/use-cases/completion-monitoring.ipynb
186. NVIDIA Developer Blog. "Build Efficient AI Agents Through Model Distillation with NVIDIA's Data Flywheel Blueprint." 2025. https://developer.nvidia.com/blog/build-efficient-ai-agents-through-model-distillation-with-nvidias-data-flywheel-blueprint
187. J. Zhao, W. Abid. "LLM Distillation Playbook." Predibase, 2024. https://github.com/predibase/llm_distillation_playbook
188. Together AI. "Continual Fine-tuning" (notebook). Together Cookbook, n.d. https://github.com/togethercomputer/together-cookbook/blob/main/fine-tuning/Continual_Finetuning.ipynb
189. OpenPipe. "Logging Requests" (docs source). n.d. https://github.com/OpenPipe/OpenPipe/blob/main/docs/features/request-logs/logging-requests.mdx
190. Letta. "Letta (formerly MemGPT)." GitHub, 2024-2026. https://github.com/letta-ai/letta
191. L. Weng. "Reward Hacking in Reinforcement Learning." Lil'Log, 28 November 2024. https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
192. OpenAI. "Sycophancy in GPT-4o: What happened and what we're doing about it." April 2025. https://openai.com/index/sycophancy-in-gpt-4o/
193. OpenAI. "Expanding on what we missed with sycophancy." May 2025. https://openai.com/index/expanding-on-sycophancy/
194. M. Sharma, M. Tong, T. Korbak, et al. "Towards Understanding Sycophancy in Language Models." Anthropic, 2023. https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models
195. M. Williams, M. Carroll, et al. "Targeted Manipulation and Deception Emerge when Optimizing LLMs for User Feedback." 2024. https://arxiv.org/abs/2411.02306
196. C. Denison, M. MacDiarmid, et al. "Sycophancy to subterfuge: Investigating reward tampering in language models." Anthropic, June 2024. https://www.anthropic.com/research/reward-tampering
197. Anthropic. "From shortcuts to sabotage: natural emergent misalignment from reward hacking." November 2025. https://www.anthropic.com/research/emergent-misalignment-reward-hacking
198. Anthropic. "Reasoning models don't always say what they think." 2025. https://www.anthropic.com/research/reasoning-models-dont-say-think
199. Anthropic. "Auditing language models for hidden objectives." March 2025. https://www.anthropic.com/research/auditing-hidden-objectives
200. K. Fronsdal, I. Gupta, A. Sheshadri, et al. "Petri: An open-source auditing tool to accelerate AI safety research." Anthropic, October 2025. https://www.anthropic.com/research/petri-open-source-auditing
201. S. Casper, X. Davies, C. Shi, et al. "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback." 2023. https://arxiv.org/abs/2307.15217
202. Anthropic. "Protecting the wellbeing of our users." December 2025. https://www.anthropic.com/news/protecting-well-being-of-users
203. S. Paech. "Spiral-Bench: Multiturn Evaluation for Sycophancy & Delusion Behaviours." 2025. https://github.com/sam-paech/spiral-bench
