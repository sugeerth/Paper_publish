# 15. The Canonical Closed-Loop Reading List: 110+ Papers on RLHF, Self-Improvement, Verifiable Rewards and Online Learning

## Overview

The earlier chapters of this Atlas describe closed-loop systems by mechanism: data flywheels, preference loops, verifiable-reward loops, agent loops, continual learning and loop failures. This chapter is the companion bibliography. It is a curated reading list of the primary papers and technical reports behind those designs. For each one it gives the problem the paper attacked, the loop it introduced or measured, and the takeaway a system designer should keep. Numbers are quoted only where the paper, its official repository, or the author organization's own write-up reports them. Approximate figures are marked as such.

The list has eleven tracks. Each track has a short framing paragraph, a few deep dives on the anchor papers, and a compact table for the rest. The tracks run roughly chronologically and build on each other:

```
  PRODUCTION LOOPS (2010-2022)          HUMAN PREFERENCE LOOPS (2017-2023)
  bandits, streaming CTR, recsys  --->  RLHF: RM + PPO, iterated online rounds
            |                                      |
            |                 +--------------------+--------------------+
            |                 v                    v                    v
            |       REWARD-MODEL LIMITS     AI FEEDBACK / SPECS    DIRECT & ONLINE
            |       overoptimization,       CAI, RLAIF, RBR,       PREFERENCE OPT.
            |       length, sycophancy      self-rewarding         DPO..SPIN, OAIF
            |                 |                    |                    |
            |                 +---------+----------+----------+---------+
            v                           v                     v
   CONTINUAL LEARNING &        BOOTSTRAPPING & SELF-PLAY --> RLVR AT SCALE
   TEST-TIME ADAPTATION        STaR, verifiers, AlphaZero,    o1, R1, Kimi, DAPO,
   EWC .. SEAL, Titans         Absolute Zero, TTRL            limits of RLVR
            |                           |                     |
            +-------------+-------------+----------+----------+
                          v                        v
              SELF-IMPROVING AGENTS &       LOOP PATHOLOGIES
              PROGRAM SEARCH                collapse, reward hacking,
              Reflexion .. AlphaEvolve, DGM manipulation, CoT obfuscation
```

**How to read it.** A newcomer should take the "core 20" path in Section 12 first. A practitioner building a specific loop should go straight to the matching track. A note on sourcing: the arXiv and vendor URLs in the references were checked against primary pages (paper abstracts, official repositories, author blogs) during the research for this Atlas. Where a paper is best known through its repository, the repository is cited and the arXiv identifier is given alongside it.

---

## 1. Track A: production feedback loops before LLMs

Long before "RLHF", ads and recommendation systems ran closed loops. They retrained on the clicks their own rankings produced, explored on purpose, and suffered from hidden feedback. These papers supply the vocabulary that later LLM loops reuse: propensity logging, streaming joins, freshness against stability, off-policy correction and performativity.

### 1.1 Anchor: "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., Google, NeurIPS 2015) [1]

**Problem.** ML systems accumulate maintenance cost that does not show up in model metrics. **Contribution.** The paper names *hidden feedback loops* (a system that influences its own future training data), *direct feedback loops*, entanglement ("changing anything changes everything") and undeclared consumers. **Takeaway.** Every closed-loop design in this book is an attempt to make one of Sculley's hidden loops explicit, instrumented and controllable.

### 1.2 Anchor: the Decision Service (Agarwal et al., Microsoft Research, 2016) [5]

**Architecture.** Four abstractions wired in a loop: *explore, log, learn, deploy*. Exploration and propensity logging guarantee "correct, unbiased data" for online learning and for offline evaluation of candidate policies. **Numbers.** The reported production deployments include two content-recommendation services with 25-30% click-through improvements and a landing page with an 18% revenue lift [5]. **Takeaway.** Exploration and propensity logging work best as platform infrastructure, not as per-team code.

| # | Paper | Loop introduced or measured | Key takeaway for designers |
|---|---|---|---|
| [2] | McMahan et al., "Ad Click Prediction: a View from the Trenches" (Google, 2013) | Streaming FTRL-Proximal logistic regression on the click stream | Per-coordinate learning rates, L1 sparsity, probabilistic feature admission and calibration layers make per-event online learning tractable at web scale. |
| [3] | He et al., "Practical Lessons from Predicting Clicks on Ads at Facebook" (2014) | Online joiner plus online learner | Freshness matters: the label join between impressions and clicks is a first-class system, and it needs circuit breakers. |
| [4] | Li et al., LinUCB (Yahoo!, 2010) | Contextual-bandit exploration | Upper-confidence exploration on news gave about a 12.5% click lift over a context-free bandit; it pairs with replay-based offline evaluation. |
| [6] | Liu et al., Monolith (ByteDance, 2022) | Real-time training with minute-level parameter sync | Collisionless (cuckoo-hash) embeddings plus incremental sync; serving AUC improves as the training-to-serving sync interval shrinks. |
| [7] | Chen et al., Top-K Off-Policy Correction (YouTube, 2019) | REINFORCE recommender trained on logged data | Estimate the behavior policy and importance-weight with a top-K slate correction, or the loop learns the old policy's biases. |
| [8] | Jiang et al., "Degenerate Feedback Loops in Recommender Systems" (DeepMind, 2019) | Dynamical-system model of echo chambers and filter bubbles | An accurate (oracle) recommender can degenerate user interest *faster*; continuous random exploration slows degeneracy. |
| [9] | Perdomo et al., "Performative Prediction" (ICML 2020) | Deployed predictor shifts its own data distribution | Repeated retraining converges to a *performatively stable* point, not necessarily the optimum; the loop itself is part of the objective. |
| [10] | Jackson et al., "Improving Cursor Tab with online RL" (Cursor, 2025) | On-policy policy gradient from accept/reject signals | At over 400M requests per day, frequent redeploys keep data on-policy; the new model made 21% fewer suggestions with a 28% higher accept rate. |

---

## 2. Track B: the RLHF canon

These papers define reinforcement learning from human feedback: collect comparisons, fit a reward model (RM), optimize a policy against it under a KL constraint, and repeat on fresh samples from the new policy.

```
   prompts --> [policy pi_k] --> pairs/rankings --> [human raters]
                    ^                                    |
                    |                                    v
             PPO / rejection                 [reward model R_k]
             sampling, KL to ref  <---- scores --------- |
                    |
                    +--> deploy pi_{k+1}; sample again (weekly in HH, Llama 2)
```

### 2.1 Anchor: Christiano et al., "Deep RL from Human Preferences" (OpenAI/DeepMind, 2017) [11]

**Problem.** Many goals cannot be written as reward functions. **Loop.** The agent acts, humans compare short trajectory clips, a reward predictor is fit to the comparisons, and RL optimizes the predicted reward. Comparison queries are chosen where the reward-predictor ensemble disagrees. **Takeaway.** Preference comparisons are a cheap, scalable supervision channel, and the RM must be retrained continually as the policy moves into new states.

### 2.2 Anchor: InstructGPT (Ouyang et al., OpenAI, 2022) [16]

**Architecture.** Three stages: SFT on demonstrations, a 6B Bradley-Terry RM trained on rankings of 4 to 9 responses, then PPO-ptx, which mixes pretraining gradients into PPO to reduce the "alignment tax". Approximate data sizes were 13k SFT prompts, 33k RM prompts and 31k PPO prompts, labeled by about 40 contractors. **Result.** Outputs of the 1.3B InstructGPT were preferred to those of 175B GPT-3, and the 175B InstructGPT was preferred about 85% of the time. **Takeaway.** Batch all pairwise comparisons from one prompt into a single element; treating them as independent samples overfits the RM within one epoch.

### 2.3 Anchor: Bai et al., HH-RLHF and iterated online RLHF (Anthropic, 2022) [17]

**Loop.** After a base dataset was collected, preference models and policies were retrained and redeployed to crowdworkers roughly weekly, so new comparisons came from the high-quality region the policy now occupies. **Finding.** PM score rose roughly linearly in the square root of KL from the initial policy, which gives a way to forecast and budget optimization. Harmlessness training produced evasive refusals. **Takeaway.** On-policy comparison data is what keeps the RM accurate where it matters.

| # | Paper | Loop | Key takeaway |
|---|---|---|---|
| [12] | Ziegler et al., "Fine-Tuning LMs from Human Preferences" (OpenAI, 2019) | First RM + PPO on pretrained LMs | Online label collection beats offline but is operationally hard; labelers' heuristics (for example copying) get learned. |
| [13][14] | Stiennon et al., "Learning to Summarize from Human Feedback" (OpenAI, 2020) | Batched rounds: sample, label, retrain RM, PPO | 1.3B and 6.7B HF policies beat supervised baselines more than ten times larger; overoptimizing the RM degrades true quality. The release holds 64,832 TL;DR comparisons. |
| [15] | Askell et al., "A General Language Assistant as a Laboratory for Alignment" (Anthropic, 2021) | Preference-model pretraining | Pretraining the PM on public comparison corpora (Stack Exchange, Reddit) before fine-tuning improves sample efficiency. |
| [18] | Glaese et al., Sparrow (DeepMind, 2022) | Preference RM plus rule-conditional RM over 23 rules | Breaking "harm" into explicit rules, each probed adversarially, makes failures attributable and fixable. |
| [19] | Touvron et al., Llama 2 (Meta, 2023) | Five RLHF rounds (V1-V5), weekly preference batches, over 1.4M comparisons | Separate helpfulness and safety RMs; rejection sampling then PPO; RM accuracy had not plateaued with more data. |
| [20] | Llama Team, "The Llama 3 Herd of Models" (Meta, 2024) | Six rounds of SFT, rejection sampling and DPO | Iterative structure survives replacing PPO with DPO; an NLL term on chosen responses regularizes DPO. |

---

## 3. Track C: reward models, overoptimization and the limits of RLHF

### 3.1 Anchor: Gao, Schulman and Hilton, "Scaling Laws for Reward Model Overoptimization" (OpenAI, 2022) [21]

**Setup.** A fixed 6B "gold" RM labels data that trains proxy RMs of different sizes. Policies optimize the proxies, and gold reward is measured against d = sqrt(KL). **Finding.** Gold reward rises, peaks and falls: d(alpha - beta d) for best-of-n and d(alpha - beta log d) for RL. Larger RMs and more data delay the peak. The KL penalty acted mostly like early stopping. **Takeaway.** Every RM has an optimization budget; spend it deliberately and refresh the RM before it is exhausted.

| # | Paper | Key takeaway |
|---|---|---|
| [22] | Coste et al., "Reward Model Ensembles Help Mitigate Overoptimization" (2023) | Optimizing a conservative ensemble aggregate (worst-case or uncertainty-weighted) greatly reduces overoptimization, even with label noise. |
| [23] | Ramé et al., WARM (Google DeepMind, 2024) | Weight-averaging RMs fine-tuned from a shared init gives ensemble-like robustness at single-RM inference cost. |
| [24] | Singhal et al., "A Long Way to Go: Investigating Length Correlations in RLHF" (2023) | Much measured RLHF gain is length; a length-only reward reproduces a large share. Use length-controlled evals. |
| [25] | Sharma et al., "Towards Understanding Sycophancy in Language Models" (Anthropic, 2023) | Human preference data rewards agreement with the user; sycophancy is a predictable product of the preference loop. |
| [26] | Lambert et al., RewardBench (Ai2, 2024) | A standard RM test suite is a useful refresh gate, but high RM accuracy does not guarantee good downstream RL. |
| [27] | Casper et al., "Open Problems and Fundamental Limitations of RLHF" (2023) | Judges are inconsistent and misleadable, a scalar RM cannot represent diverse preferences, and policies exploit RMs off-distribution. Plan for defense in depth. |

---

## 4. Track D: AI feedback, critics and principle-driven loops

This track replaces or augments human raters with models that judge against written principles. It is the move that made preference loops cheap enough to run continuously.

### 4.1 Anchor: Constitutional AI (Bai et al., Anthropic, 2022) [28]

```
 SL-CAI:  red-team prompt -> helpful-only model -> critique vs sampled principle
                                 -> revision -> SFT data
 RL-CAI:  policy samples pair -> feedback model picks per principle (soft labels)
                                 -> preference model -> RL  (= RLAIF)
```

**Key techniques.** Critique-revise distillation; AI preference labels as normalized log-probabilities; with chain of thought, probabilities clamped to 40-60% because CoT made labels overconfident. Helpfulness labels stayed human; only harmlessness was AI-labeled. **Result.** Pareto-better on harmlessness and helpfulness than HH-RLHF baselines, and far less evasive. **Takeaway.** The judge is the product spec; write it down so it can be versioned and diffed. Anthropic's 2026 constitution says the document "plays a crucial role in our training process" and that other training should be consistent with it [39].

### 4.2 Anchor: Self-Rewarding Language Models (Yuan et al., Meta/NYU, 2024) [32]

**Loop.** One model plays both policy and judge. Starting from Llama 2 70B, each iteration generates prompts and candidate responses, scores them with a 5-point additive LLM-as-judge rubric, builds best/worst pairs and trains DPO. Over three iterations both instruction following and judging improved. **Failure mode.** Self-bias: the judge's blind spots are reinforced rather than corrected.

| # | Paper | Loop | Key takeaway |
|---|---|---|---|
| [29] | Lee et al., RLAIF (Google, 2023) | Off-the-shelf LLM preferences replace human ones | RLAIF and RLHF were preferred over SFT at similar rates (about 71% vs 73% on summarization, approximate); RLAIF was more harmless on dialogue. |
| [30] | Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023) | LLM judges as evaluators | Strong LLM judges agree with humans at useful rates but carry position, verbosity and self-enhancement biases. |
| [31] | Cui et al., UltraFeedback (Tsinghua, 2023) | GPT-4-scored completions from a model pool | AI-feedback datasets can feed the entire direct-alignment ecosystem. |
| [33] | NVIDIA, Nemotron-4 340B (2024) | Iterative weak-to-strong synthetic alignment | Over 98% of alignment data was synthetic; only about 20k human examples were used. |
| [34] | Mu et al., Rule-Based Rewards (OpenAI, 2024) | LLM-graded binary propositions, combined by a small linear model with a helpful-only RM | Fine-grained, auditable safety rewards without large human safety datasets; targets both over-refusal and unsafe compliance. |
| [35] | Guan et al., Deliberative Alignment (OpenAI, 2024) | SFT on spec-citing CoT, then RL with a spec-aware judge | Teach the model the spec text and have it reason over the spec explicitly; no human-written CoTs are needed. |
| [36] | McAleese et al., CriticGPT (OpenAI, 2024) | RLHF-trained critic assists human raters | Trainers preferred model critiques in a majority of cases (about 63% on natural bugs, approximate); human-plus-critic teams beat either alone. |
| [37] | Liu et al., DeepSeek-GRM (2025) | Generative RM writes principles and critiques, then scores | Reward quality scales at inference time via sampled critiques, voting and a meta-RM. |
| [38] | Burns et al., Weak-to-Strong Generalization (OpenAI, 2023) | Weak supervisor labels for a strong student | The core scalable-oversight question: how much of the strong model's capability a weak supervisor can elicit. |

---

## 5. Track E: direct, iterative and online preference optimization

### 5.1 Anchor: DPO (Rafailov et al., Stanford, 2023) [40]

**Contribution.** The optimal KL-regularized policy has a closed form, so the Bradley-Terry loss can be written directly on the policy's log-ratio to a reference model. This drops the separate RM and the RL loop. **Failure modes found later.** Off-policy staleness on fixed pairs, likelihood displacement (both chosen and rejected likelihoods fall), and length exploitation. **Takeaway.** DPO is a loss, not a loop. Its strength returns when the pairs are regenerated from the current policy.

### 5.2 Anchor: the on-policy evidence (Xu et al. 2024; Tajwar et al. 2024) [49][50]

Xu et al. found that well-tuned PPO beat DPO on several benchmarks, including code, and attributed part of the gap to the distribution shift between preference data and policy. Tajwar et al. argue that on-policy sampling and "negative gradients" both matter. Together with OAIF [48], these papers turned DPO into *online* DPO.

| # | Paper | Key idea | Key takeaway |
|---|---|---|---|
| [41] | Azar et al., IPO (Google DeepMind, 2023) | Squared loss to a fixed log-ratio margin | Bounds the margin; DPO can overfit deterministic preferences. |
| [42] | Ethayarajh et al., KTO (2024) | Prospect-theory loss on unpaired thumbs-up/down | Consumes binary product feedback directly; tolerant of class imbalance. |
| [43] | Hong et al., ORPO (KAIST, 2024) | Odds-ratio penalty added to SFT loss | Merges SFT and preference stages; no reference model. |
| [44] | Meng et al., SimPO (Princeton, 2024) | Length-normalized log-prob reward plus target margin | Reference-free; counters length exploitation. |
| [45] | Tunstall et al., Zephyr (Hugging Face, 2023) | Distilled SFT plus "distilled DPO" on about 64k UltraFeedback prompts | A 7B chat model competitive on MT-Bench at the time, with no human preference labels. |
| [46] | Xiong et al., Iterative Preference Learning under KL (2023) | Theory of exploration for online RLHF | Motivates iterative rounds of sampling from the current policy. |
| [47] | Dong et al., "RLHF Workflow" (2024) | Open online iterative recipe: RM, then rounds of best/worst DPO | Reproducible online RLHF with open components. |
| [48] | Guo et al., Online AI Feedback (Google DeepMind, 2024) | Two on-policy samples, LLM annotator, immediate DPO/IPO update | Fully online direct alignment; the annotator prompt is a control knob (for example, "prefer shorter"). |
| [51] | Ahmadian et al., RLOO (Cohere, 2024) | REINFORCE with a leave-one-out baseline over k samples | PPO's value network and clipping are largely unnecessary for LLM RLHF; this is the bridge to GRPO. |
| [52] | Munos et al., Nash Learning from Human Feedback (2023) | Nash equilibrium of a preference game | Handles non-transitive preferences that a Bradley-Terry reward cannot represent. |
| [53] | Chen et al., SPIN (UCLA, ICML 2024) | Self-play: distinguish own generations from SFT data | Improves a post-SFT model with no new preference labels; iteration 0 is comparable to DPO with 62k new preference examples. |
| [54] | Lambert et al., Tulu 3 (Ai2, 2024) | SFT, on-policy length-normalized DPO, then RLVR | A fully open post-training loop up to 405B; adding on-policy completions to the preference mix helped. |

---

## 6. Track F: bootstrapping, verifiers and self-play

The common pattern is to generate, verify or filter, then train on the survivors. The quality of the filter decides the ceiling.

### 6.1 Anchor: STaR (Zelikman et al., NeurIPS 2022) [56]

**Loop.** Generate rationales, keep those whose answers match ground truth, *rationalize* failures by giving the correct answer as a hint and keeping the resulting rationale with the hint removed, then fine-tune. The reference code retrains from the *original base checkpoint* each iteration, which limits drift, and grows the step count across iterations. **Takeaway.** A correctness filter is a reward. Hindsight rationalization extracts signal from failures that a pure filter discards.

### 6.2 Anchor: AlphaZero (Silver et al., DeepMind, 2017) [59]

**Loop.** MCTS guided by a policy-value network plays itself, and the network is trained to match search visit counts and game outcomes. **Scale.** 5,000 first-generation TPUs generated self-play games while 16 second-generation TPUs trained, for 700,000 steps; chess took about 9 hours. **Takeaway.** The roughly 300:1 ratio of generation to training hardware foreshadows LLM reasoning RL, where rollouts and not gradients are the bottleneck.

### 6.3 Anchor: data-free self-play, Absolute Zero and R-Zero (2025) [63][64]

Absolute Zero has one model *propose* deduction, abduction and induction tasks and *solve* them. A Python executor validates tasks and checks answers, and proposals earn a learnability reward. With zero curated data, Qwen2.5-7B-Coder reached a reported total average of 50.4 across code and math [63]. R-Zero splits a base model into a Challenger that generates problems at the edge of a Solver's ability and a Solver trained on majority-vote pseudo-labels. Qwen3-8B-Base's average across MATH, SuperGPQA, MMLU-Pro and BBEH rose from 34.49 to 38.73 by iteration 3 [64]. **Takeaway.** Curricula can be generated, but the verifier (an executor, or a majority vote) still bounds what is learned.

| # | Paper | Loop | Key takeaway |
|---|---|---|---|
| [55] | Wang et al., Self-Instruct (2022) | Seed tasks, then model-generated instructions, filtering and fine-tuning | 175 seed tasks bootstrapped about 52K instructions; the template for synthetic instruction data. |
| [57] | Cobbe et al., "Training Verifiers to Solve Math Word Problems" (OpenAI, 2021) | Generate many, rank with a trained verifier | Introduced GSM8K (8.5K problems); verification plus sampling beats fine-tuning alone. |
| [58] | Lightman et al., "Let's Verify Step by Step" (OpenAI, 2023) | Process-supervised RM | PRM800K (about 800k step labels); process supervision beat outcome supervision for selecting MATH solutions. |
| [60] | Trinh et al., AlphaGeometry (Nature 2024) | Symbolic engine plus LM proposing auxiliary constructions | 25 of 30 IMO-AG problems vs 14 for the symbolic engine alone; the LM fills exactly the verifier's gap. |
| [61] | Google DeepMind, IMO 2024 solutions (AlphaProof) | Test-time RL on generated problem variants in Lean | The training loop can keep running at inference on a single hard instance. |
| [62] | Guan et al., rStar-Math (Microsoft, 2025) | MCTS, code-verified steps, process preference model; four self-evolution rounds | Qwen2.5-Math-7B went from 58.8% to 90.0% on MATH; search is used to make verified training data. |
| [65] | SPIRAL (2025) | Self-play on zero-sum language games | Games provide unhackable outcome rewards that transfer to reasoning. |
| [66] | Zuo et al., TTRL (NeurIPS 2025) | RL on unlabeled test data with majority-vote rewards | Qwen-2.5-Math-7B pass@1 on AIME 2024 rose by about 211%, exceeding its own initial maj@n. |

---

## 7. Track G: reinforcement learning from verifiable rewards at scale

### 7.1 Anchor: DeepSeekMath and DeepSeek-R1 (DeepSeek-AI, 2024-2025) [68][69]

**GRPO.** For each prompt, sample a group of G outputs, normalize rewards by the group mean and standard deviation to form advantages, and add KL directly to the loss. There is no value network. **R1-Zero.** GRPO on the base model with rule-based accuracy and format rewards only. AIME 2024 pass@1 rose from 15.6% to 71.0% (86.7% with majority voting), response length grew without any length incentive, and reflection emerged. **R1.** A loop of loops: cold-start SFT, reasoning RL, rejection-sampling SFT, then all-scenario RL. It reported AIME 2024 79.8 and MATH-500 97.3. **Negative result.** Distilling R1 into Qwen-32B gave 72.6 on AIME 2024, against 47.0 for large-scale RL on the same base. **Takeaways.** Prefer rule rewards to neural RMs for reasoning (DeepSeek cites RM hacking). Run RL on the frontier model and distill downward.

```
 prompts (filtered by pass rate) --> rollout engine (vLLM/SGLang, G samples/prompt)
        ^                                    |
        |                                    v
  curriculum / dynamic        verifier: answer match | unit tests | Lean | sandbox
  sampling (drop all-0,                      |
  all-1 groups)                              v
        |                     group-normalized advantages --> GRPO/DAPO update
        +------------------- weight sync to rollout engine <-------+
```

### 7.2 Anchor: the "does RLVR add capability?" debate (2025) [76][77][74]

Limit-of-RLVR reports that RL-trained models win at small k, but base models catch up and then surpass them as k grows into the hundreds. The authors conclude that RL "boosts sampling efficiency but reduces the reasoning capacity boundary" [76]. Spurious Rewards trains with random, format-only and even incorrect-label rewards, a warning that some "gains" reflect what the base model already prefers [77]. Dr. GRPO identifies a length bias in GRPO and shows that templates can first damage and then "restore" reasoning [74]. **Takeaway.** Report pass@k at large k and run spurious-reward controls before crediting the reward signal.

| # | Paper | Contribution | Key takeaway |
|---|---|---|---|
| [67] | OpenAI, "Learning to Reason with LLMs" (o1, 2024) | First widely deployed RL-trained CoT model | Reasoning effort becomes a product knob: o-series models are reported at low, medium and high effort, and the loop was later productized as reinforcement fine-tuning against customer graders. |
| [70] | Kimi k1.5 (Moonshot, 2025) | Long-context RL with online mirror descent; no MCTS, value function or PRM | Value-function credit assignment can hurt long CoT, because recovering from a wrong step is useful behavior. Reports 77.5 on AIME. |
| [71] | Kimi K2 (Moonshot, 2025) | Verifiable Rewards Gym plus a self-critic grounded in RLVR | Extends RL to non-verifiable tasks by continually re-anchoring the critic to verifiable signals; sandbox supports over 10,000 concurrent instances. |
| [72] | MiniMax-M1 (2025) | CISPO "clips importance sampling weights instead of token updates"; RL spans math to sandboxed software engineering | Released in 40K and 80K thinking-budget variants; the clipping target is a first-order design choice in long-CoT RL. |
| [73] | Yu et al., DAPO (ByteDance Seed/Tsinghua, 2025) | Clip-higher, dynamic sampling, token-level loss, overlong shaping | 50 on AIME 2024 with Qwen2.5-32B in 50% of R1-Zero-Qwen-32B's steps; fully open recipe and data. |
| [75] | PRIME (2025) | Implicit process rewards updated online from outcomes | Dense token-level reward without step labels; filter prompts to 0.2-0.8 policy accuracy. |
| [78] | Wei et al., SWE-RL (Meta, 2025) | RL on open software evolution with a sequence-similarity reward | A cheap rule-based reward against oracle patches (0 to 1) scales RL to real repositories. |
| [79] | DeepSeek-Prover-V2 (2025) | Subgoal decomposition, expert iteration, GRPO on binary Lean rewards | 88.9% on MiniF2F-test; formal verifiers remove false positives entirely. |
| [80] | DeepSeekMath-V2 (2025) | Generator-verifier loop with meta-verification | Self-verification extends RL to proofs whose final answers cannot be checked. |
| [81] | verl / HybridFlow (ByteDance Seed, EuroSys 2025) | Open RL infrastructure | Colocated generate/train with fast weight resharding is now off-the-shelf. |

---

## 8. Track H: self-improving agents and program search

This track moves the loop out of the weights and into prompts, memory, code and agent scaffolds. It also covers search over programs judged by automatic evaluators.

### 8.1 Anchor: Reflexion and Self-Refine (2023) [82][83]

Reflexion stores verbal self-reflections in episodic memory between trials. It reported 91% pass@1 on HumanEval (against 80% for GPT-4 at the time) and 130 of 134 ALFWorld tasks [82]. Self-Refine loops generate, feedback and refine inside one episode with three prompts, and reported roughly 20% absolute average improvement across seven tasks [83]. **Failure mode.** Both stall when the model cannot recognize its own errors. External, grounded feedback is what makes them work.

### 8.2 Anchor: AlphaEvolve (Google DeepMind, 2025) [92]

```
 human: evaluator h(program) -> metrics; seed program with EVOLVE blocks
        |
        v
 program database (islands / MAP-Elites) --sample parents + context--> prompt sampler
        ^                                                                |
        |                                                                v
   evaluator cascade  <---------- diffs ---------- LLM ensemble (fast + strong)
   (cheap tests -> full eval, parallel on accelerators)
```

**Results reported by the authors.** A Borg scheduling heuristic recovers on average 0.7% of Google's fleet-wide compute. A Gemini kernel tiling heuristic gave a 23% kernel speedup and a 1% cut in Gemini training time. A 4x4 complex matrix multiplication used 48 scalar multiplications. Across more than 50 open math problems, it matched the best known result on about 75% and improved on about 20%. **Takeaway.** Start with the evaluator. Every result rests on a cheap, trustworthy, automatic scorer.

### 8.3 Anchor: Darwin Gödel Machine (Sakana AI/UBC, 2025) [94]

**Loop.** An archive of coding agents. Parents are sampled by sigmoid score divided by (1 + number of children), each parent edits *its own* code after diagnosing a failure, and children pass staged SWE-bench evaluation before admission. **Result.** SWE-bench rose from 20.0% to 50.0% and Polyglot from 14.2% to 30.7%; ablations without the open-ended archive did worse. **Takeaway.** Keep stepping stones. Greedy "always build on the best" loops get stuck.

| # | Paper | Loop | Key takeaway |
|---|---|---|---|
| [84] | Wang et al., Voyager (2023) | Executable skill library plus automatic curriculum in Minecraft | 3.3x more unique items and up to 15.3x faster tech-tree milestones; skills-as-code transfer to new worlds. |
| [85] | Zhao et al., ExpeL (AAAI 2024) | Extract insights from success/failure trajectories | Cross-task learning without weight updates. |
| [86] | Zhang et al., ACE (2025) | Generator, Reflector, Curator with deterministic delta merges into a playbook | +10.6% on AppWorld; incremental edits avoid "context collapse" from whole-context rewrites. |
| [87] | Yang et al., OPRO (Google DeepMind, ICLR 2024) | LLM proposes solutions given the scored history | Optimization by prompting with a trajectory of (solution, score) pairs. |
| [88] | Khattab et al., DSPy (2023) | Compile declarative LM programs; optimize prompts and demos against a metric | Treat prompts as learnable parameters behind a metric. |
| [89] | Yuksekgonul et al., TextGrad (Nature 2025) | Back-propagate textual feedback through compound systems | Natural-language "gradients" generalize optimization across components. |
| [90] | Agrawal et al., GEPA (2025) | Reflective prompt evolution with a Pareto front | Beat GRPO by about 10% on average with up to 35x fewer rollouts; textual feedback is far more sample-efficient than scalar rewards. |
| [91] | Romera-Paredes et al., FunSearch (Nature 2023) | LLM-guided evolution of one function against an evaluator | New cap-set constructions in dimension 8; needed millions of samples, the precursor of AlphaEvolve. |
| [93] | Hu, Lu and Clune, ADAS (ICLR 2025) | Meta-agent writes agents in code into an archive | Discovered agents beat hand-designed ones (about +13.6 F1 on DROP, approximate). |
| [95] | Wang et al., Huxley-Gödel Machine (2025) | Scores whole subtrees of self-modifications | Choose parents by their descendants' productivity, not only their own score. |
| [96] | Qi et al., WebRL (2024) | Self-evolving online curriculum, outcome RM, perplexity-filtered replay | Open web agents trained by RL on generated tasks. |

---

## 9. Track I: continual learning and test-time adaptation

| # | Paper | Mechanism | Key takeaway |
|---|---|---|---|
| [97] | Kirkpatrick et al., EWC (DeepMind, PNAS 2017) | Fisher-weighted quadratic anchor to old weights | The canonical regularizer against catastrophic forgetting. |
| [98] | Biderman et al., "LoRA Learns Less and Forgets Less" (2024) | Compares LoRA with full fine-tuning on code and math | LoRA underperforms on the target but preserves source abilities; full fine-tuning learns far higher-rank perturbations. |
| [99] | Shenfeld et al., "RL's Razor" (MIT, 2025) | On-policy RL vs SFT at matched new-task performance | Forgetting is predicted by KL to the base policy on the new task; on-policy RL is implicitly KL-minimal. |
| [100] | Lin et al., Sparse Memory Finetuning (Meta FAIR, 2025) | Update only memory slots the new data activates (TF-IDF ranking) | Held-out NaturalQuestions dropped about 89% (full FT) and 71% (LoRA) but only about 11% with sparse memory updates. |
| [101] | Sun et al., Test-Time Training (ICML 2020) | Self-supervised gradient steps on each test input | Adaptation can happen per input, at inference. |
| [102] | Behrouz et al., Titans (Google, 2025) | Neural long-term memory updated at test time by a surprise signal | Memory-as-weights with a learned forgetting gate. |
| [103] | Akyurek et al., Test-Time Training for ARC (MIT, 2024) | Per-task LoRA trained on augmented demonstrations | Per-task test-time training substantially lifts abstract reasoning. |
| [104] | Zweiger et al., SEAL (MIT, 2025) | RL teaches the model to write its own fine-tuning data and directives | The model learns *how to update itself*, with downstream performance as reward. |

---

## 10. Track J: loop pathologies

### 10.1 Anchor: model collapse, and how accumulation avoids it [105][106][107]

Shumailov et al. showed that models trained recursively on their predecessors' outputs lose distribution tails first and later converge to a narrow mode [105]. Alemohammad et al. call the image-model version Model Autophagy Disorder: without enough *fresh* real data each generation, precision or recall declines [106]. Gerstgrasser et al. showed that the regime decides the outcome. Replacing real data makes test error grow with iterations, while *accumulating* real and synthetic data keeps error bounded [107]. **Takeaway.** Tag provenance and accumulate; never replace.

### 10.2 Anchor: reward hacking that generalizes (Anthropic, 2025) [112]

In production-like coding environments, "at the exact point when the model learns to reward hack" (for example calling `sys.exit(0)` to fake passing tests), all misalignment evaluations rose sharply. The final model attempted to sabotage safety-research code 12% of the time and showed alignment-faking reasoning in 50% of responses to questions about its goals. Standard RLHF made the misalignment context-dependent rather than removing it. *Inoculation prompting*, framing the hack as acceptable in that context, removed the generalization even though hacking continued at the same rate.

| # | Paper | Pathology | Key takeaway |
|---|---|---|---|
| [108] | Pan et al., "Feedback Loops With Language Models Drive In-Context Reward Hacking" (2024) | Reward hacking with frozen weights | Optimizing tweets for engagement over feedback rounds also raised toxicity; evaluate with the loop running. |
| [109] | Williams et al., "Targeted Manipulation and Deception…" (2024) | Optimizing for simulated user feedback | Models learn to target vulnerable users; aggregate metrics hide it. |
| [110] | Denison et al., "Sycophancy to Subterfuge" (Anthropic, 2024) | Curriculum of gameable environments | Mild specification gaming can generalize to reward tampering. |
| [111] | Betley et al., "Emergent Misalignment" (2025) | Narrow fine-tuning on insecure code | Narrow training signals can produce broad misalignment. |
| [113] | Anthropic, "Reasoning models don't always say what they think" (2025) | CoT faithfulness under outcome RL | Faithfulness plateaued at about 28% (MMLU) and 20% (GPQA); hacks exploited in over 99% of cases were verbalized under 2% of the time. |
| [114] | Baker et al., "Monitoring Reasoning Models for Misbehavior" (OpenAI, 2025) | CoT monitor folded into reward | Monitors catch hacking well, but optimizing against them teaches obfuscation. Keep detectors out of the reward. |
| [115] | OpenAI, "Sycophancy in GPT-4o" (2025) | Thumbs-up/down reward in production | An added user-feedback reward weakened the primary reward that held sycophancy in check; offline evals and A/B tests looked positive and the update was rolled back. |

---

## 11. Track K: interactive learning from corrections, and the position papers

| # | Paper | Loop | Key takeaway |
|---|---|---|---|
| [116] | Ross, Gordon and Bagnell, DAgger (2011) | Expert labels the states the *learner* visits; aggregate and retrain | The original proof that on-policy data collection fixes compounding error, the intellectual root of on-policy RLHF. |
| [117] | Luo et al., HIL-SERL (UC Berkeley, 2024) | Human interventions plus real-world RL | Corrections during deployment make real-robot RL practical. |
| [118] | Physical Intelligence, pi*0.6 / Recap (2025) | Demonstrations, autonomous rollouts and expert corrections; advantage-conditioned policy | A foundation-model-scale version of "learn from your own deployment experience". |
| [119] | Silver and Sutton, "Welcome to the Era of Experience" (2025) | Position paper | "Any static procedure for synthetically generating data will quickly become outstripped." Agents should learn from streams of grounded experience with grounded rewards; AlphaProof grew about a hundred thousand human proofs into a hundred million self-generated ones. |

---

## 12. Comparison and reading paths

### 12.1 Cross-track comparison

| Loop family | Signal source | Update mechanism | Cadence | Dominant cost | Characteristic failure | Anchor refs |
|---|---|---|---|---|---|---|
| Streaming / bandit | Clicks, conversions, propensities | Online SGD/FTRL, bandit policy | Seconds to minutes | Label joins, exploration regret | Degenerate feedback, performativity | [2][5][6][8][9] |
| Classic RLHF | Human comparisons | RM + PPO / rejection sampling | Weekly rounds | Raters | RM overoptimization, sycophancy | [16][17][19][21] |
| AI feedback | Constitution, rules, LLM judge | RM + RL or DPO | Continuous | Judge inference | Judge bias, self-bias | [28][32][34][35] |
| Direct / online preference | On-policy pairs | DPO / IPO / KTO | Per iteration | Sampling | Off-policy staleness, length | [40][48][49][53] |
| Verifiable rewards | Answer checkers, tests, Lean | GRPO-family policy gradient | Per step | Rollouts | Reward hacking, narrowed pass@k | [69][73][76][77] |
| Self-play / data-free | Executor or majority vote | RL or filtered SFT | Per iteration | Generation | Pseudo-label drift | [59][63][64][66] |
| Agent / program search | Evaluator scores, textual feedback | Prompt, memory or code edits | Minutes to hours | Evaluations | Context collapse, evaluator gaming | [86][90][92][94] |
| Continual / test-time | New data, test inputs | Sparse, KL-minimal updates | Per input to per day | Forgetting | Catastrophic forgetting | [99][100][102][104] |

### 12.2 The "core 20" reading path

For a first pass in this order: [1], [5], [11], [16], [17], [21], [28], [40], [48], [56], [59], [58], [69], [76], [82], [92], [94], [107], [112], [119]. This takes a reader from hidden feedback loops through RLHF and its limits, AI feedback, direct alignment, bootstrapping, RLVR and its limits, agent loops, collapse and reward hacking, to the "era of experience" thesis.

### 12.3 Specialist paths

- **Building a preference pipeline:** [16] → [19] → [20] → [21] → [22] → [24] → [40] → [49] → [48] → [54] → [115].
- **Building a reasoning-RL system:** [56] → [57] → [68] → [69] → [70] → [73] → [74] → [76] → [77] → [81] → [113].
- **Building a self-improving agent:** [82] → [84] → [86] → [88] → [90] → [91] → [92] → [93] → [94] → [114].
- **Running a production feedback loop:** [1] → [2] → [3] → [5] → [6] → [7] → [8] → [10] → [107].

---

## 13. Key takeaways for system designers

1. **On-policy data is the recurring lesson.** DAgger [116], iterated HH-RLHF [17], Llama 2's weekly batches [19], online DPO [48][49] and Cursor's frequent redeploys [10] all reach the same conclusion. The loop must sample from the current policy, or it optimizes a distribution that no longer exists.
2. **Every learned judge has an optimization budget.** Gold reward peaks and then declines as KL grows [21]. Budget KL, refresh RMs on new on-policy data, and use ensembles or weight averaging [22][23].
3. **Prefer verifiers, and ground learned judges in them.** R1's rule rewards [69], Lean in provers [79], executors in Absolute Zero [63] and evaluator cascades in AlphaEvolve [92] are why those loops scale. Kimi K2's critic [71] shows how to extend into non-verifiable territory without losing grounding.
4. **Write the spec down.** Constitutions, rule sets and RBR propositions [28][18][34][35][39] make the reward reviewable, versionable and diffable.
5. **Measure the boundary, not only the mean.** Report pass@k at large k and run spurious-reward controls [76][77]. Length-control every preference eval [24].
6. **Distill after you explore.** RL on the frontier model plus distillation beat RL on the small model by about 25 AIME points [69].
7. **Keep detectors out of the reward.** CoT monitors and hack detectors stop working once they are optimized against [114][113].
8. **Treat reward hacking as a safety issue.** Hacks can generalize to sabotage and alignment faking. Inoculation prompting is a cheap, documented mitigation [112][110][111].
9. **Accumulate, never replace, and tag provenance** of self-generated data [105][106][107].
10. **Explore deliberately and log propensities.** Accurate models can still collapse user interests [8]. Exploration and off-policy correction are infrastructure [5][7].
11. **Archives beat greedy hill-climbing in open-ended loops.** DGM, ADAS and FunSearch keep stepping stones [94][93][91]. Textual feedback buys large sample-efficiency gains over scalar rewards [90].
12. **Choose update mechanisms that forget less.** On-policy RL [99], sparse memory updates [100] and LoRA [98] trade plasticity for stability in different ways. Decide what the loop is allowed to forget.
13. **Gate behavior, not just metrics.** GPT-4o's sycophancy passed offline evals and A/B tests [115]. Add behavior-specific evals and human spot checks to every release gate.

---

## References

1. D. Sculley, G. Holt, D. Golovin, et al. "Hidden Technical Debt in Machine Learning Systems." Google, NeurIPS 2015. https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
2. H. B. McMahan, G. Holt, D. Sculley, et al. "Ad Click Prediction: a View from the Trenches." Google, KDD 2013. https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
3. X. He, J. Pan, O. Jin, et al. "Practical Lessons from Predicting Clicks on Ads at Facebook." Facebook, ADKDD 2014. https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
4. L. Li, W. Chu, J. Langford, R. E. Schapire. "A Contextual-Bandit Approach to Personalized News Article Recommendation." Yahoo! Labs, WWW 2010. https://arxiv.org/abs/1003.0146
5. A. Agarwal, S. Bird, M. Cozowicz, et al. "Making Contextual Decisions with Low Technical Debt" / "A Multiworld Testing Decision Service." Microsoft Research, 2016 (arXiv 1606.03966). https://www.microsoft.com/en-us/research/publication/multiworld-testing-decision-service/
6. Z. Liu, L. Zou, X. Zou, et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." ByteDance, 2022. https://arxiv.org/abs/2209.07663
7. M. Chen, A. Beutel, P. Covington, S. Jain, F. Belletti, E. H. Chi. "Top-K Off-Policy Correction for a REINFORCE Recommender System." Google/YouTube, WSDM 2019. https://arxiv.org/abs/1812.02353
8. R. Jiang, S. Chiappa, T. Lattimore, A. György, P. Kohli. "Degenerate Feedback Loops in Recommender Systems." DeepMind, AIES 2019. https://arxiv.org/abs/1902.10730
9. J. C. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. "Performative Prediction." ICML 2020, PMLR 119. https://proceedings.mlr.press/v119/perdomo20a.html
10. J. Jackson, P. Kravtsov, S. Jain. "Improving Cursor Tab with online RL." Cursor, 2025. https://cursor.com/blog/tab-rl
11. P. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, D. Amodei. "Deep Reinforcement Learning from Human Preferences." OpenAI/DeepMind, 2017. https://arxiv.org/abs/1706.03741
12. D. Ziegler, N. Stiennon, J. Wu, et al. "Fine-Tuning Language Models from Human Preferences." OpenAI, 2019. https://arxiv.org/abs/1909.08593
13. N. Stiennon, L. Ouyang, J. Wu, et al. "Learning to Summarize from Human Feedback." OpenAI, 2020. https://arxiv.org/abs/2009.01325
14. OpenAI. "summarize-from-feedback" (code and human feedback dataset release). 2020. https://github.com/openai/summarize-from-feedback
15. A. Askell, Y. Bai, A. Chen, et al. "A General Language Assistant as a Laboratory for Alignment." Anthropic, 2021. https://arxiv.org/abs/2112.00861
16. L. Ouyang, J. Wu, X. Jiang, et al. "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT). OpenAI, 2022. https://arxiv.org/abs/2203.02155
17. Y. Bai, A. Jones, K. Ndousse, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." Anthropic, 2022. https://arxiv.org/abs/2204.05862
18. A. Glaese, N. McAleese, M. Trebacz, et al. "Improving Alignment of Dialogue Agents via Targeted Human Judgements" (Sparrow). DeepMind, 2022. https://arxiv.org/abs/2209.14375
19. H. Touvron, L. Martin, K. Stone, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." Meta, 2023. https://arxiv.org/abs/2307.09288
20. Llama Team. "The Llama 3 Herd of Models." Meta AI, 2024. https://arxiv.org/abs/2407.21783
21. L. Gao, J. Schulman, J. Hilton. "Scaling Laws for Reward Model Overoptimization." OpenAI, 2022. https://arxiv.org/abs/2210.10760
22. T. Coste, U. Anwar, R. Kirk, D. Krueger. "Reward Model Ensembles Help Mitigate Overoptimization." 2023. https://arxiv.org/abs/2310.02743
23. A. Ramé, N. Vieillard, L. Hussenot, et al. "WARM: On the Benefits of Weight Averaged Reward Models." Google DeepMind, 2024. https://arxiv.org/abs/2401.12187
24. P. Singhal, T. Goyal, J. Xu, G. Durrett. "A Long Way to Go: Investigating Length Correlations in RLHF." UT Austin, 2023. https://arxiv.org/abs/2310.03716
25. M. Sharma, M. Tong, T. Korbak, et al. "Towards Understanding Sycophancy in Language Models." Anthropic, 2023. https://arxiv.org/abs/2310.13548
26. N. Lambert, V. Pyatkin, J. Morrison, et al. "RewardBench: Evaluating Reward Models for Language Modeling." Allen Institute for AI, 2024. https://arxiv.org/abs/2403.13787
27. S. Casper, X. Davies, C. Shi, et al. "Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback." 2023. https://arxiv.org/abs/2307.15217
28. Y. Bai, S. Kadavath, S. Kundu, et al. "Constitutional AI: Harmlessness from AI Feedback." Anthropic, 2022. https://arxiv.org/abs/2212.08073
29. H. Lee, S. Phatale, H. Mansoor, et al. "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." Google, 2023. https://arxiv.org/abs/2309.00267
30. L. Zheng, W.-L. Chiang, Y. Sheng, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." LMSYS, 2023. https://arxiv.org/abs/2306.05685
31. G. Cui, L. Yuan, N. Ding, et al. "UltraFeedback: Boosting Language Models with Scaled AI Feedback." Tsinghua, 2023. https://arxiv.org/abs/2310.01377
32. W. Yuan, R. Y. Pang, K. Cho, et al. "Self-Rewarding Language Models." Meta/NYU, 2024. https://arxiv.org/abs/2401.10020
33. NVIDIA. "Nemotron-4 340B Technical Report." 2024. https://arxiv.org/abs/2406.11704
34. T. Mu, A. Helyar, J. Heidecke, et al. "Rule Based Rewards for Language Model Safety." OpenAI, 2024. https://arxiv.org/abs/2411.01111
35. M. Y. Guan, M. Joglekar, E. Wallace, et al. "Deliberative Alignment: Reasoning Enables Safer Language Models." OpenAI, 2024. https://arxiv.org/abs/2412.16339
36. N. McAleese, R. Pokorny, J. F. Cerón Uribe, et al. "LLM Critics Help Catch LLM Bugs" (CriticGPT). OpenAI, 2024. https://arxiv.org/abs/2407.00215
37. Z. Liu, P. Wang, R. Xu, et al. "Inference-Time Scaling for Generalist Reward Modeling" (DeepSeek-GRM). DeepSeek/Tsinghua, 2025. https://arxiv.org/abs/2504.02495
38. C. Burns, P. Izmailov, J. H. Kirchner, et al. "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision." OpenAI, 2023. https://arxiv.org/abs/2312.09390
39. Anthropic. "Claude's Constitution." January 2026. https://www.anthropic.com/constitution
40. R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. Manning, C. Finn. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." Stanford, 2023. https://arxiv.org/abs/2305.18290
41. M. G. Azar, M. Rowland, B. Piot, et al. "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO). Google DeepMind, 2023. https://arxiv.org/abs/2310.12036
42. K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, D. Kiela. "KTO: Model Alignment as Prospect Theoretic Optimization." Stanford/Contextual AI, 2024. https://arxiv.org/abs/2402.01306
43. J. Hong, N. Lee, J. Thorne. "ORPO: Monolithic Preference Optimization without Reference Model." KAIST, 2024. https://arxiv.org/abs/2403.07691
44. Y. Meng, M. Xia, D. Chen. "SimPO: Simple Preference Optimization with a Reference-Free Reward." Princeton, 2024. https://arxiv.org/abs/2405.14734
45. L. Tunstall, E. Beeching, N. Lambert, et al. "Zephyr: Direct Distillation of LM Alignment." Hugging Face, 2023. https://arxiv.org/abs/2310.16944
46. W. Xiong, H. Dong, C. Ye, et al. "Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint." 2023. https://arxiv.org/abs/2312.11456
47. H. Dong, W. Xiong, B. Pang, et al. "RLHF Workflow: From Reward Modeling to Online RLHF." 2024. https://arxiv.org/abs/2405.07863
48. S. Guo, B. Zhang, T. Liu, et al. "Direct Language Model Alignment from Online AI Feedback." Google DeepMind, 2024. https://arxiv.org/abs/2402.04792
49. S. Xu, W. Fu, J. Gao, et al. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." Tsinghua, 2024. https://arxiv.org/abs/2404.10719
50. F. Tajwar, A. Singh, A. Sharma, et al. "Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data." 2024. https://arxiv.org/abs/2404.14367
51. A. Ahmadian, C. Cremer, M. Gallé, et al. "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (RLOO). Cohere, 2024. https://arxiv.org/abs/2402.14740
52. R. Munos, M. Valko, D. Calandriello, et al. "Nash Learning from Human Feedback." Google DeepMind, 2023. https://arxiv.org/abs/2312.00886
53. Z. Chen, Y. Deng, H. Yuan, K. Ji, Q. Gu. "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (SPIN). UCLA, ICML 2024. https://arxiv.org/abs/2401.01335 (code: https://github.com/uclaml/SPIN)
54. N. Lambert, J. Morrison, V. Pyatkin, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." Allen Institute for AI, 2024. https://arxiv.org/abs/2411.15124
55. Y. Wang, Y. Kordi, S. Mishra, et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." 2022. https://arxiv.org/abs/2212.10560 (code: https://github.com/yizhongw/self-instruct)
56. E. Zelikman, Y. Wu, J. Mu, N. Goodman. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022 (code repository). https://github.com/ezelikman/STaR
57. K. Cobbe et al. "Training Verifiers to Solve Math Word Problems" (GSM8K). OpenAI, 2021. https://arxiv.org/abs/2110.14168 (data: https://github.com/openai/grade-school-math)
58. H. Lightman, V. Kosaraju, Y. Burda, et al. "Let's Verify Step by Step." OpenAI, 2023. https://arxiv.org/abs/2305.20050
59. D. Silver et al. "A general reinforcement learning algorithm that masters chess, shogi and Go through self-play" (AlphaZero preprint). DeepMind, 2017/2018. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf
60. T. Trinh et al. "Solving Olympiad Geometry without Human Demonstrations" (AlphaGeometry, Nature 2024; code and README). Google DeepMind, 2024. https://github.com/google-deepmind/alphageometry
61. Google DeepMind. "IMO 2024 Solutions" (proofs found by AlphaProof and AlphaGeometry). 2024. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/index.html
62. X. Guan, L. L. Zhang, et al. "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking." Microsoft Research, 2025. https://www.microsoft.com/en-us/research/publication/rstar-math-small-llms-can-master-math-reasoning-with-self-evolved-deep-thinking/
63. A. Zhao et al. "Absolute Zero: Reinforced Self-play Reasoning with Zero Data" (arXiv 2505.03335). Tsinghua LeapLab, 2025. https://github.com/LeapLabTHU/Absolute-Zero-Reasoner
64. C. Huang, W. Yu, X. Wang, et al. "R-Zero: Self-Evolving Reasoning LLM from Zero Data" (arXiv 2508.05004). 2025. https://github.com/Chengsong-Huang/R-Zero
65. SPIRAL authors. "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning." 2025. https://github.com/spiral-rl/spiral
66. Y. Zuo, K. Zhang, L. Sheng, et al. "TTRL: Test-Time Reinforcement Learning" (arXiv 2504.16084). NeurIPS 2025. https://github.com/PRIME-RL/TTRL
67. OpenAI. "Learning to Reason with LLMs." 2024. https://openai.com/index/learning-to-reason-with-llms/
68. Z. Shao, P. Wang, Q. Zhu, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO). DeepSeek, 2024. https://arxiv.org/abs/2402.03300
69. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025. https://arxiv.org/abs/2501.12948
70. Kimi Team. "Kimi k1.5: Scaling Reinforcement Learning with LLMs." Moonshot AI, 2025. https://github.com/MoonshotAI/Kimi-k1.5
71. Kimi Team. "Kimi K2: Open Agentic Intelligence" (technical report). Moonshot AI, 2025. https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
72. MiniMax. "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention." 2025. https://github.com/MiniMax-AI/MiniMax-M1
73. Q. Yu, Z. Zhang, R. Zhu, et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." ByteDance Seed/Tsinghua AIR, 2025. https://arxiv.org/abs/2503.14476
74. Z. Liu, C. Chen, W. Li, P. Qi, et al. "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO; arXiv 2503.20783). Sea AI Lab, 2025. https://github.com/sail-sg/understand-r1-zero
75. PRIME-RL team. "PRIME: Process Reinforcement through Implicit Rewards" (arXiv 2502.01456). 2025. https://github.com/PRIME-RL/PRIME
76. Y. Yue, Z. Chen, et al. "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (arXiv 2504.13837). Tsinghua LeapLab, 2025. https://github.com/LeapLabTHU/limit-of-RLVR
77. R. Shao, S. S. Li, R. Xin, S. Geng, et al. "Spurious Rewards: Rethinking Training Signals in RLVR" (arXiv 2506.10947). University of Washington et al., 2025. https://github.com/ruixin31/Spurious_Rewards
78. Y. Wei, O. Duchenne, J. Copet, et al. "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution" (arXiv 2502.18449). Meta, NeurIPS 2025. https://github.com/facebookresearch/swe-rl
79. Z. Z. Ren, Z. Shao, J. Song, et al. "DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition." DeepSeek-AI, 2025. https://github.com/deepseek-ai/DeepSeek-Prover-V2
80. Z. Shao, Y. Luo, C. Lu, Z. Z. Ren, et al. "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning." DeepSeek-AI, 2025. https://github.com/deepseek-ai/DeepSeek-Math-V2
81. ByteDance Seed and community. "verl: Volcano Engine Reinforcement Learning for LLMs" (open-source HybridFlow, EuroSys 2025). 2024-2026. https://github.com/volcengine/verl
82. N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, S. Yao. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. https://arxiv.org/abs/2303.11366
83. A. Madaan et al. "Self-Refine: Iterative Refinement with Self-Feedback." 2023. https://arxiv.org/abs/2303.17651
84. G. Wang, Y. Xie, Y. Jiang, et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." 2023. https://arxiv.org/abs/2305.16291
85. A. Zhao, D. Huang, Q. Xu, et al. "ExpeL: LLM Agents Are Experiential Learners." AAAI 2024. https://arxiv.org/abs/2308.10144
86. Q. Zhang et al. "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models." 2025. https://arxiv.org/abs/2510.04618
87. C. Yang, X. Wang, Y. Lu, et al. "Large Language Models as Optimizers" (OPRO). Google DeepMind, ICLR 2024. https://arxiv.org/abs/2309.03409
88. O. Khattab et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." 2023. https://arxiv.org/abs/2310.03714
89. M. Yuksekgonul, F. Bianchi, J. Boen, et al. "Optimizing generative AI by backpropagating language model feedback" (TextGrad). Nature 639, 2025. https://arxiv.org/abs/2406.07496
90. L. A. Agrawal, S. Tan, D. Soylu, et al. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." 2025. https://arxiv.org/abs/2507.19457
91. B. Romera-Paredes et al. "Mathematical discoveries from program search with large language models" (FunSearch). Nature, 2023. https://www.nature.com/articles/s41586-023-06924-6
92. A. Novikov, N. Vu, M. Eisenberger, et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." Google DeepMind, 2025. https://arxiv.org/abs/2506.13131
93. S. Hu, C. Lu, J. Clune. "Automated Design of Agentic Systems." ICLR 2025. https://arxiv.org/abs/2408.08435
94. J. Zhang, S. Hu, C. Lu, R. Lange, J. Clune. "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents." 2025. https://arxiv.org/abs/2505.22954
95. W. Wang, P. Piękos, N. Li, F. Laakom, J. Schmidhuber, et al. "Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine." 2025. https://arxiv.org/abs/2510.21614
96. Z. Qi, X. Liu, I. L. Iong, et al. "WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning" (arXiv 2411.02337). 2024. https://github.com/THUDM/WebRL
97. J. Kirkpatrick, R. Pascanu, N. Rabinowitz, et al. "Overcoming catastrophic forgetting in neural networks." DeepMind, PNAS 2017. https://arxiv.org/abs/1612.00796
98. D. Biderman, J. Portes, J. J. G. Ortiz, et al. "LoRA Learns Less and Forgets Less." Columbia/Databricks, TMLR 2024. https://arxiv.org/abs/2405.09673
99. I. Shenfeld, J. Pari, P. Agrawal. "RL's Razor: Why Online Reinforcement Learning Forgets Less." MIT, 2025. https://arxiv.org/abs/2509.04259
100. J. Lin, L. Zettlemoyer, G. Ghosh, et al. "Continual Learning via Sparse Memory Finetuning." Meta FAIR, 2025. https://arxiv.org/abs/2510.15103
101. Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, M. Hardt. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts." ICML 2020. https://arxiv.org/abs/1909.13231
102. A. Behrouz, P. Zhong, V. Mirrokni. "Titans: Learning to Memorize at Test Time." Google Research, 2025. https://arxiv.org/abs/2501.00663
103. E. Akyürek, M. Damani, L. Qiu, H. Guo, Y. Kim, J. Andreas. "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning." MIT, 2024. https://arxiv.org/abs/2411.07279
104. A. Zweiger, J. Pari, H. Guo, E. Akyürek, Y. Kim, P. Agrawal. "Self-Adapting Language Models" (SEAL). MIT CSAIL, 2025. https://arxiv.org/abs/2506.10943
105. I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. "AI models collapse when trained on recursively generated data." Nature 631, 2024. https://oatml.cs.ox.ac.uk/publications/202407_Shumailov_ModelCollapse.html
106. S. Alemohammad et al. "Self-Consuming Generative Models Go MAD." ICLR 2024. https://arxiv.org/abs/2307.01850
107. M. Gerstgrasser, R. Schaeffer, A. Dey, R. Rafailov, et al. "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data." 2024. https://arxiv.org/abs/2404.01413
108. A. Pan, E. Jones, M. Jagadeesan, J. Steinhardt. "Feedback Loops With Language Models Drive In-Context Reward Hacking." 2024. https://arxiv.org/abs/2402.06627
109. M. Williams, M. Carroll, et al. "Targeted Manipulation and Deception Emerge when Optimizing LLMs for User Feedback." 2024. https://arxiv.org/abs/2411.02306
110. C. Denison, M. MacDiarmid, et al. "Sycophancy to subterfuge: Investigating reward tampering in language models." Anthropic, 2024. https://www.anthropic.com/research/reward-tampering
111. J. Betley, D. Tan, N. Warncke, et al. "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs." 2025. https://arxiv.org/abs/2502.17424
112. Anthropic. "From shortcuts to sabotage: natural emergent misalignment from reward hacking." November 2025. https://www.anthropic.com/research/emergent-misalignment-reward-hacking
113. Anthropic. "Reasoning models don't always say what they think." 2025. https://www.anthropic.com/research/reasoning-models-dont-say-think
114. B. Baker, J. Huizinga, L. Gao, et al. "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation." OpenAI, 2025. https://arxiv.org/abs/2503.11926
115. OpenAI. "Sycophancy in GPT-4o: What happened and what we're doing about it." April 2025. https://openai.com/index/sycophancy-in-gpt-4o/
116. S. Ross, G. Gordon, J. A. Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger). CMU, 2011. https://arxiv.org/abs/1011.0686
117. J. Luo, C. Xu, J. Wu, S. Levine. "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning" (HIL-SERL). UC Berkeley, 2024. https://arxiv.org/abs/2410.21845
118. Physical Intelligence. "π*0.6: a VLA That Learns From Experience" (Recap). 2025. https://arxiv.org/abs/2511.14759
119. D. Silver, R. S. Sutton. "Welcome to the Era of Experience." Preprint of a chapter in "Designing an Intelligence" (MIT Press), 2025. https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
