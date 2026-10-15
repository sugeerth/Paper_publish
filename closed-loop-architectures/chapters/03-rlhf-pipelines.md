# 03. Preference-Learning Loops: RLHF, Constitutional AI/RLAIF, DPO, and Online Reward-Model Refresh

## Overview

Reinforcement learning from human feedback (RLHF) and its descendants are the main closed loops that turn a pretrained language model into a deployed assistant. The loop always has the same shape. A policy generates candidate behaviors. A judge (human raters, an AI labeler following written principles, a programmatic verifier, or end users clicking thumbs-up) expresses preferences over those behaviors. The preferences are compressed into a training signal, either an explicit reward model (RM) or an implicit one inside a direct-alignment loss. The policy is updated and the cycle repeats. What separates the designs is where each arc of the loop runs, how often it closes, and what stops the optimizer from exploiting the judge.

This chapter covers the publicly documented designs, in roughly historical order:

- **The original RLHF lineage**: deep RL from human preferences [1], fine-tuning LMs from preferences [2], and summarization from human feedback [3], which set out the three-stage SFT, RM, PPO recipe that InstructGPT [4] scaled up.
- **Anthropic's HH-RLHF and iterated online RLHF** [5][6], the first detailed public account of a weekly retrain-and-redeploy preference loop.
- **Principle-driven feedback**: DeepMind's Sparrow rule RMs [8], Constitutional AI and RLAIF [7][9], Collective Constitutional AI [44], Anthropic's 2026 rewritten constitution [62], OpenAI's Rule-Based Rewards [39], deliberative alignment [40] and the Model Spec [59].
- **Direct alignment algorithms**: DPO and its variants (IPO, KTO, ORPO, SimPO) [10][11][12][13][14], with the open recipes built on them (Zephyr [15], Tulu 3 [19]).
- **Online and iterative preference learning**: Llama 2's five RLHF rounds [17], Llama 3's six SFT+DPO rounds [18], Self-Rewarding LMs [20], iterative/online DPO [21][22][23], Nemotron-4's synthetic weak-to-strong loop [34], and the evidence that on-policy data matters [24][25][55].
- **The reasoning-era turn**: RL from verifiable rewards with GRPO-family optimizers [36][37][57], generative reward models [56], and rubric-based rewards that carry RL into non-verifiable domains [66][67][68].
- **Preference collection in products**: ChatGPT-style feedback widgets, Chatbot Arena [33], the 2025 "Leaderboard Illusion" audit of arena-driven optimization [65], pluralistic collection in the PRISM dataset [64], and the GPT-4o sycophancy rollback of April 2025 [41][42], the clearest public case of a product feedback signal destabilizing a deployed loop.
- **Reward-model refresh and overoptimization**: scaling laws for reward hacking [26], ensembles and weight averaging [27][28], benchmarks [29][63], length and sycophancy biases [31][32], the "RM continually improved while RL pushes it" practice described for Gemini [45], and 2025 evidence that reward hacking can generalize into broader misalignment and that optimizing against chain-of-thought monitors teaches obfuscation [69][70].

A note on sourcing. The numbers in this chapter come from the cited papers and official posts. Values that are rounded, inferred, or reported inconsistently across versions of a paper are marked approximate. Frontier labs rarely publish post-training cadences or costs for current production models, so the most detailed numbers come from 2022 to 2024 papers and open-model reports. The practitioner syntheses by Lambert [52], Huyen [53] and Hugging Face [54] are useful entry points.

The generic loop:

```
            +------------------------------------------------------------------+
            |                  OUTER LOOP (weeks to months)                    |
            v                                                                  |
  +-------------------+  prompts   +------------------+  comparisons  +---------+--------+
  | Prompt sources    |----------->| Current policy   |-------------->| Judges:          |
  | (API traffic,     |            | pi_k samples 2..K|  pairs/ranks  | humans, AI       |
  |  red team, synth) |            | responses        |               | labeler + rules, |
  +-------------------+            +--------+---------+               | verifiers, users |
                                            ^                         +---------+--------+
                                            |                                   |
                       +--------------------+-----+    +------------------------v-----+
                       | Policy optimization:     |<---| Reward signal:               |
                       | PPO / RLOO / GRPO, or    |    | RM_k (BT model, ensemble,    |
                       | DPO-family on pairs;     |    | generative judge) or         |
                       | KL to reference policy   |    | implicit reward (DPO)        |
                       +------------+-------------+    +------------------------------+
                                    |
                                    v   gates: evals, red team, A/B, staged rollout
                          deploy pi_{k+1} -> new traffic -> new prompts (loop closes)
```

## 1. The original RLHF lineage (2017 to 2020)

### 1.1 Deep RL from human preferences

Christiano et al. [1] posed the core problem: for many tasks a reward function is hard to write, but a human can compare two short trajectory clips. The architecture runs three asynchronous processes. A policy interacts with the environment. Pairs of clips are sent to humans for comparison. A reward predictor is fit to the comparisons with a Bradley-Terry (logistic) model and supplies rewards to the RL learner. Queries were chosen where an ensemble of reward predictors disagreed, an early form of active preference elicitation. The lesson that carries forward is that the reward model must be trained online, alongside the policy. A reward model fit once on a fixed dataset was exploited as the policy drifted into regions the model had never seen. The feedback budget was small: roughly 700 human queries for the MuJoCo tasks and 5,500 for Atari, and a simulated-robot backflip learned from about 900 queries, under an hour of rater time [1].

### 1.2 Fine-tuning LMs from human preferences

Ziegler et al. [2] carried the method over to GPT-2 for stylistic continuation and summarization, with a per-token KL penalty toward the pretrained model. That KL term is still in almost every RLHF objective. They reported that about 5,000 comparisons were enough for the stylistic tasks, while summarization needed far more (about 60,000). Their main operational finding concerns the loop itself. Fully online data collection, where labels are gathered while the policy trains, was hard to engineer and to quality-control. The summarization policies also learned to be "smart copiers" that exploited the labelers' heuristics. Both themes (loop engineering cost and judges with exploitable heuristics) recur through this chapter.

### 1.3 Learning to summarize from human feedback

Stiennon et al. [3] switched to batched, offline-per-round collection. Each round they sample summaries from current policies, send pairs to trained labelers, retrain the RM on all data so far, and run PPO against it. On Reddit TL;DR, 1.3B and 6.7B human-feedback policies were preferred over supervised baselines more than ten times larger. The paper also documented reward overoptimization: optimizing much past a moderate KL budget raised the RM score while lowering human-judged quality. Gao et al. [26] later formalized this. Much of the paper's attention went to labeler quality (close communication with labelers, agreement tracking). That emphasis became standard practice.

## 2. InstructGPT: the three-stage recipe at scale

### 2.1 Problem

GPT-3 completes text; it does not reliably follow instructions. OpenAI wanted a model that API customers would prefer, one that is helpful, honest and harmless, without retraining from scratch [4].

### 2.2 Architecture

```
 Stage 1: SFT              Stage 2: Reward model             Stage 3: RL (PPO-ptx)
 +----------------+        +-------------------------+       +---------------------------+
 | ~13k prompts   |        | ~33k prompts; labelers  |       | ~31k prompts (API only)   |
 | labeler demos  |        | rank K=4..9 samples     |       | policy init = SFT         |
 | -> fine-tune   |        | -> all C(K,2) pairs as  |       | reward = RM(x,y)          |
 |    GPT-3       |------->|    one batch element    |------>|   - beta * KL(pi || SFT)  |
 +----------------+        | 6B RM, BT pairwise loss |       | + gamma * pretrain LM loss|
                           +-------------------------+       +---------------------------+
 prompt sources: API Playground traffic (deduplicated, PII-filtered) + labeler-written
```

- **Data.** Prompts came mostly from the OpenAI API Playground, plus prompts written by labelers to bootstrap categories. Dataset sizes were about 13k prompts for SFT, 33k for RM training and 31k for PPO (approximate, as reported) [4]. About 40 contractors did the labeling, selected with a screening test for sensitivity to the preferences of different demographic groups.
- **Ranking interface.** Labelers ranked between 4 and 9 responses per prompt. All C(K,2) pairwise comparisons from one prompt were put into a single batch element. Treating them as independent samples caused the RM to overfit within one epoch, because each completion appeared in K-1 updates.
- **RM.** A 6B model trained with the pairwise Bradley-Terry loss. The authors chose 6B because it saved compute and because 175B RM training could be unstable, which made a 175B RM less suitable for initializing the PPO value function [4].
- **PPO-ptx.** The RL objective mixes pretraining log-likelihood gradients into PPO to reduce the "alignment tax", meaning performance regressions on public NLP benchmarks such as SQuAD and DROP.

### 2.3 Results and numbers

Outputs of the 1.3B InstructGPT were preferred by labelers over the 175B GPT-3, despite the 100x parameter gap. The 175B InstructGPT was preferred to 175B GPT-3 about 85% of the time [4]. Inter-annotator agreement among training labelers was about 73%. Held-out labelers who produced no training data preferred InstructGPT at similar rates, which suggests the policy was not merely fitting the idiosyncrasies of the training labelers. Truthfulness improved on TruthfulQA. Toxicity fell when the model was prompted to be respectful, but bias benchmarks did not improve.

### 2.4 Failure modes and lessons

The paper is explicit that the model is aligned to "the preferences of a specific group of labelers, researchers, and API customers", not to human values in general. It still followed harmful instructions, hedged excessively on simple questions, and could be confused by false premises. The design lessons are: (1) ranking K samples amortizes prompt-reading cost and yields many pairs per task; (2) a small RM can steer a much larger policy; (3) regression on capabilities must be actively countered (PPO-ptx) and measured.

## 3. Anthropic HH-RLHF and iterated online RLHF

### 3.1 Problem

Anthropic studied how to train a dialogue assistant to be both helpful and harmless, and how RLHF behaves as models scale [5][6].

### 3.2 Architecture

- **Preference model pretraining (PMP).** Askell et al. [5] introduced an intermediate stage. The preference model is first trained on large public comparison corpora (for example Stack Exchange votes and Reddit scores) and then fine-tuned on Anthropic's own comparisons. This improved sample efficiency.
- **Collection interface.** Crowdworkers held open-ended conversations with the model and, at every turn, picked the better of two responses. Helpfulness and harmlessness (red-teaming) data were collected separately: in harmlessness sessions workers chose the more harmful response for adversarial prompts. The public release holds roughly 160k training comparisons [6][61]. It is split into tranches that mirror the loop: helpfulness data from the base context-distilled 52B models, from best-of-16 rejection sampling against an early PM, and from the iterated online process. Harmlessness data was collected only for the base models [61].
- **Iterated online RLHF.** After the base dataset was collected, the team ran a loop with a roughly weekly cadence. Train new PMs and RLHF policies, deploy them to the crowdworker interface, collect comparisons on the new policies' outputs, and repeat [6]. Comparisons drawn from the best current models shift the PM's training distribution toward the high-quality region the policy now occupies, which is where the PM's accuracy matters most.

```
  week k:  PM_k + policy pi_k  --deploy-->  crowdworker chat UI (A/B per turn)
                 ^                                     |
                 |   retrain PM_{k+1} on all data      v
                 +---------  online comparisons (high-reward regime) ---------+
```

### 3.3 Key findings

- **Reward vs KL.** Over most of training, the policy's PM score rose roughly linearly in the square root of the KL divergence from the initial policy [6]. This gives a practical way to forecast and budget optimization.
- **Robustness.** Larger PMs were more robust. Policies trained against a PM were judged by held-out PMs, and the train and test PM scores diverged as optimization went on, an early quantitative look at overoptimization.
- **Alignment tax.** For the larger models (13B, 52B), RLHF improved zero-shot NLP evaluations; the smaller models paid a tax [6].
- **Tension.** Optimizing harmlessness data produced evasive refusals that made the assistant less helpful. This motivated Constitutional AI's "harmless but non-evasive" target [7].

### 3.4 Lessons

Online iteration is essentially a fix for the RM's distribution shift. The judge has to keep seeing samples from where the policy currently is. Separating helpfulness and harmlessness datasets made each signal legible, but it also made their trade-off explicit, and that trade-off needed its own design response.

## 4. Principle-driven and AI feedback loops

### 4.1 DeepMind Sparrow: rule-conditional reward models

Sparrow [8] gave the judge a structured specification. It used a set of 23 natural-language rules (no threats, no claiming to be human, and so on) and two RMs. A preference RM was trained on per-turn comparisons. A rule RM, conditioned on a rule, predicted whether a response violated it, trained from targeted adversarial probing where raters tried to elicit a specific rule violation. The RL reward combined both, and the agent could cite retrieved evidence. Lesson: breaking "harmlessness" into rules gives per-rule violation rates that can be measured and improved, and adversarial collection can be targeted at one rule at a time.

### 4.2 Constitutional AI and RLAIF (Anthropic)

Constitutional AI (CAI) [7] removes human labels from the harmlessness arc of the loop.

```
 SL-CAI stage                                   RL-CAI stage (RLAIF)
 red-team prompt                                prompt -> policy samples (y1, y2)
   -> helpful-only RLHF model response              -> feedback model, prompted with a
   -> critique (random principle)                      sampled principle, answers
   -> revision  (repeat N times)                       "which is less harmful? (A)/(B)"
   -> fine-tune on final revisions                     (optionally with chain of thought)
                                                    -> normalized probs = soft labels
                                                    -> PM trained on AI harmlessness
                                                       labels + HUMAN helpfulness labels
                                                    -> PPO against the PM
```

Key techniques [7]:

- **Critique-revise distillation.** A helpful-only model critiques and revises its own red-team responses against randomly sampled constitutional principles. The revisions become SFT data.
- **AI preference labels.** The feedback model's normalized log-probabilities over the options serve as soft preference targets. When chain-of-thought was used, the probabilities were clamped to the 40 to 60% range, because CoT made the labels overconfident and the resulting policies over-trained.
- **Hybrid loop.** Helpfulness comparisons stayed human. Only harmlessness was AI-labeled. The result was Pareto-better on harmlessness and helpfulness than the HH-RLHF baselines and far less evasive.

In May 2023 Anthropic published the constitution then used for Claude, a list of standalone principles drawn from the UN Universal Declaration of Human Rights, Apple's terms of service (for privacy and impersonation), DeepMind's Sparrow rules, trust-and-safety practice, and an explicit effort to include non-Western perspectives [43]. In Collective Constitutional AI [44] it sourced a constitution from about 1,000 members of the US public through the Polis deliberation platform and trained a Claude Instant-sized model on it. Compared with a model trained on Anthropic's own constitution, the public-constitution model was similar on helpfulness and harmlessness and showed lower bias on the BBQ benchmark. This closes the loop one level higher: the public helps write the judge's specification.

### 4.3 RLAIF vs RLHF at Google

Lee et al. [9] ran a controlled comparison on summarization and dialogue. RLAIF (preferences from an off-the-shelf LLM) and RLHF policies were each preferred over the SFT baseline by human evaluators at similar rates (about 71% and 73% for summarization, approximate). On harmless dialogue, RLAIF reached a higher harmless rate than RLHF (about 88% vs 76%, approximate). They also proposed "direct RLAIF", which skips the RM and uses the LLM's score as the reward at RL time, and showed that RLAIF still helped when the labeler was the same size as the policy. Implication for system design: AI labelers turn preference collection from a weeks-long human-ops pipeline into a batch job that scales with compute.

### 4.4 OpenAI Rule-Based Rewards, deliberative alignment and the Model Spec

OpenAI's Model Spec [59] is a public behavior specification, so it doubles as the documented judge specification for OpenAI's post-training. Two published techniques operationalize it:

- **Rule-Based Rewards (RBR)** [39]. Desired safety behavior (for example, refuse with a brief apology and no judgment) is broken into binary propositions ("contains an apology", "is judgmental", "complies"). An LLM grader scores the propositions, and a small fitted linear model combines those scores with a helpful-only RM to give the RL reward. The goal is to avoid both over-refusal and unsafe compliance without large volumes of human safety data. Because the rules are explicit, updating behavior means editing propositions and refitting a few weights, not relabeling data.
- **Deliberative alignment** [40]. Used for the o-series reasoning models. The model is taught the text of the safety specification and trained to reason about it explicitly in its chain of thought. Stage 1 is SFT on (prompt, spec-citing CoT, answer) data, generated by a model with the spec in context and then filtered by a judge. Stage 2 is RL in which a judge model with access to the spec supplies the reward. No human-written CoTs or completions are needed.

### 4.5 From rule lists to an explained constitution (Anthropic, 2026)

In January 2026 Anthropic replaced the 2023 list of standalone principles with a long-form constitution that explains the reasoning behind each value, released under a CC0 public-domain dedication [62]. Three design choices matter for the loop:

- **Explained values over rules.** The stated aim is for the model to understand why a behavior is wanted, so that it generalizes to situations the text does not anticipate. A small set of hard constraints is kept for high-stakes cases.
- **An explicit priority order.** When properties conflict, Claude should generally be broadly safe first, then broadly ethical, then compliant with Anthropic's more specific guidelines, then genuinely helpful. The ordering is described as holistic rather than strict: higher priorities usually dominate, but lower ones are still weighed.
- **The document as the judge's source of truth.** Anthropic calls the constitution the final authority on intended behavior and says Claude uses it to generate many kinds of synthetic training data, including response rankings. In loop terms, it is the versioned specification from which AI-feedback labels are produced, a direct descendant of the RLAIF stage in Section 4.2. Anthropic also states that model behavior will not always match the text and that system cards will report the gaps.

The OpenAI Model Spec [59] and the Anthropic constitution now play the same role: a public, versioned judge specification. They differ in form, with the Model Spec closer to an annotated rulebook with worked examples and the constitution closer to an argued essay.

### 4.6 Lessons from principle-driven loops

- A written specification makes the judge versionable. Policy changes become text diffs.
- AI judges inherit the biases of their prompts and base models, so calibration (soft labels, clamping) and periodic human audits are still needed.
- Mixing human and AI signals by dimension works well in practice: human data for helpfulness taste, AI or rule data for policy compliance.

## 5. Direct preference optimization and its variants

### 5.1 DPO

PPO-based RLHF needs four models in memory at once (policy, reference, RM, value function), online generation, and delicate hyperparameter tuning. Rafailov et al. [10] showed that, for the KL-regularized reward-maximization objective, the optimal policy satisfies r(x,y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x). Substituting this into the Bradley-Terry likelihood cancels Z and gives a classification loss on preference pairs:

```
 L_DPO = - E_(x, y_w, y_l) [ log sigmoid( beta * ( log pi(y_w|x)/pi_ref(y_w|x)
                                                 - log pi(y_l|x)/pi_ref(y_l|x) ) ) ]
```

The policy therefore is its own reward model, the "implicit reward". Training needs only the policy and a frozen reference, no sampling and no value function. In the paper's experiments (sentiment control, TL;DR summarization, Anthropic HH, models up to about 6B) DPO matched or beat PPO.

### 5.2 Variants

| Method | Data needed | Core change vs DPO | Reference model | Motivation / claim |
|---|---|---|---|---|
| DPO [10] | pairs | BT likelihood on implicit reward | yes | Drop the RM and RL loop |
| IPO [11] | pairs | Squared loss regressing the log-ratio gap to a fixed target | yes | DPO can overfit deterministic preferences; IPO bounds the margin |
| KTO [12] | unpaired thumbs-up/down | Prospect-theory value function on per-sample implicit reward vs a KL baseline | yes | Uses binary product feedback directly; tolerant to imbalance |
| ORPO [13] | pairs | Adds an odds-ratio penalty to the SFT loss in one stage | no | Merges SFT and preference stages |
| SimPO [14] | pairs | Length-normalized avg log-prob as reward plus a target margin | no | Removes reference; counters length exploitation |
| RPO (NVIDIA) [34] | pairs + RM scores | Fits implicit reward gap to the RM's reward gap | yes | Avoids over-separating pairs of near-equal quality |
| NLHF / SPPO [46][60] | pairwise preference model | Seek a Nash equilibrium of a preference game, not a BT reward max | varies | Preferences need not be transitive or BT-representable |

KTO matters for product loops in particular: it consumes the unpaired binary signal (thumbs-up/down) that consumer chat interfaces already collect [12].

### 5.3 Open recipes built on DPO

- **Zephyr-7B** [15]. Distilled SFT, then "distilled DPO" on UltraFeedback [16], about 64k prompts each with four completions from a pool of models, scored by GPT-4. The result was a 7B chat model competitive on MT-Bench with much larger RLHF models at the time, with no human preference labels at all. UltraFeedback showed that AI feedback datasets could feed the whole direct-alignment ecosystem.
- **Tulu 3** (Ai2) [19]. A fully open post-training recipe: SFT, then length-normalized DPO on largely on-policy preference data (completions from the Tulu SFT model and other models, rated by an LLM judge), then RL with verifiable rewards (RLVR) on math and precise-instruction-following constraints. The paper releases data, code and evals for models up to 405B. Ablations showed that adding on-policy completions to the preference mix helped.

### 5.4 Failure modes of offline DPO

- **Off-policy staleness.** DPO trained on fixed pairs from other models optimizes the implicit reward on that distribution, not on the policy's own outputs. Xu et al. [24] found that a well-tuned PPO beat DPO on several benchmarks, including code, and attributed part of DPO's weakness to distribution shift between the preference data and the policy. Tajwar et al. [25] argue that methods using on-policy sampling and "negative gradients" (pushing down likelihood on dispreferred samples) work best when the optimal response has low probability under the reference. Ivison et al. [55] found PPO with a strong RM edging out DPO, with data quality the biggest single factor.
- **Likelihood displacement.** DPO can lower the absolute likelihood of both chosen and rejected responses, so only the margin grows. Llama 3 [18] and others add an NLL term on chosen responses as a regularizer.
- **Length exploitation.** Implicit rewards correlate with length, which motivated SimPO's and Tulu 3's length normalization [14][19].

## 6. Online and iterative preference learning

The common response to these failure modes is to keep DPO's simplicity and close the loop again: regenerate preference data from the current policy each round.

### 6.1 Llama 2: five RLHF rounds with dual reward models

Meta's Llama 2-Chat [17] is the most detailed public account of a human-preference loop at industrial scale.

- **Collection.** Annotators wrote a prompt and chose between two responses sampled from different model variants and temperatures. They graded the preference as significantly better, better, slightly better, or negligibly better/unsure, and labeled safety separately. Meta collected over 1.4 million binary comparisons, in weekly batches, and also used open-source preference sets.
- **Two RMs.** Separate helpfulness and safety RMs, both initialized from pretrained Llama checkpoints. The pairwise loss included a margin term that depended on the graded preference strength. At RL time the safety RM's score was used for prompts tagged as unsafe or when the safety score fell below a threshold; otherwise the helpfulness score.
- **Iteration.** Five versions, RLHF-V1 through V5. Early rounds used rejection sampling fine-tuning: sample K outputs, keep the best per the RM, fine-tune on them. PPO was added on top of rejection sampling in the later rounds, and rejection sampling was run only with the 70B model, whose outputs were distilled into the smaller models. Each new batch of preferences was collected on samples from the latest model, so the RMs were refreshed on the shifted distribution.
- **Finding.** RM accuracy kept improving with more data and had not plateaued. In other words, the refresh cycle was still paying off when the paper was written.

```
 week w: annotators compare samples from latest chat model -> new batch B_w
         RM_help, RM_safe retrained on B_1..B_w (+ open data)
         rejection sampling (70B) / PPO  -> RLHF-V(n+1)  -> sample next batch
```

### 6.2 Llama 3: six rounds of SFT + rejection sampling + DPO

For Llama 3 [18] Meta replaced PPO with DPO and kept the iterative structure. Each of six rounds:

1. Collect new human preference annotations on samples from the latest models. Grades were four levels; annotators could also edit the chosen response, giving an ordering edited > chosen > rejected.
2. Train an RM on all preference data so far.
3. Rejection-sample SFT data using the RM, plus targeted synthetic data per capability (code, tool use, multilingual, long context, and so on).
4. SFT, then DPO on preference data drawn mostly from the latest round's policies.

Reported DPO adjustments: formatting and special tokens were masked out of the loss, and an NLL term on the chosen response (coefficient about 0.2, approximate) was added for stability. The stated reasons for DPO over PPO were lower compute at 405B scale and better results on instruction-following benchmarks such as IFEval [18]. The RM still existed, but its job moved from reward provider for RL to data curator for rejection sampling.

### 6.3 Self-rewarding language models

Yuan et al. [20] collapsed the policy and the judge into one model. Starting from Llama 2 70B, each iteration has the model generate new prompts and candidate responses, score its own candidates with an LLM-as-a-judge prompt (a 5-point additive rubric), build pairs from the highest- and lowest-scored candidates, and train with DPO. Over three iterations both instruction-following and judging ability improved. The third iteration outperformed several proprietary models of the time on AlpacaEval 2.0. The authors note the risk the design implies: no external anchor stops self-reinforcing bias, and gains are expected to saturate.

### 6.4 Iterative and online DPO

- **Online AI Feedback (OAIF)** [23] samples two responses from the current policy for each training prompt, asks an LLM annotator which is better, and applies the DPO/IPO/SLiC update immediately, making direct alignment fully online. Human raters preferred OAIF policies over offline DPO. The annotator prompt also became a control knob (for example, "prefer shorter responses").
- **Iterative preference learning / RLHF Workflow** [21][22] give a theoretical motivation (exploration under a KL constraint) and an open recipe. Train a BT RM (or pairwise preference model) on open preference data, then run several iterations of: sample multiple responses per prompt from the current policy, label best and worst with the RM, train DPO. The open-source models trained this way performed strongly on chat benchmarks for their size.
- **Back to basics: RLOO** [38]. Cohere showed that the PPO machinery (value network, clipping) is largely unnecessary for LLM RLHF. REINFORCE with a leave-one-out baseline over k samples per prompt matched or beat PPO and DPO at lower memory cost. This "online but cheap" middle ground led to the critic-free GRPO family.

### 6.5 Nemotron-4 340B: synthetic weak-to-strong loop

NVIDIA's Nemotron-4 340B [34] reports that over 98% of its alignment data was synthetic. Only about 20k human-annotated examples were used, about 10k for SFT and about 10k from HelpSteer2 [35] for RM training. Its loop is "iterative weak-to-strong alignment". An initial aligned model generates synthetic prompts, responses and preference data. That data trains a better base-aligned model, which then generates the next round. Preference optimization ran DPO followed by several rounds of RPO (Section 5.2), with the Nemotron-4-340B-Reward model scoring pairs. HelpSteer2's multi-attribute labels (helpfulness, correctness, coherence, complexity, verbosity) allow a regression RM whose attribute weights can be tuned at inference time, which helps counter verbosity bias.

### 6.6 Lessons from iterative loops

- The minimum viable loop for a team without RL infrastructure is: sample N from the current policy, score with an RM or judge, take best and worst, run DPO, repeat. It recovers much of the on-policy benefit [21][22][23][25].
- An RM is worth keeping even without PPO, as a rejection-sampling filter, pair constructor and eval signal [17][18][34].
- Each round needs fresh prompts that cover the policy's current weaknesses. Llama 3's per-capability data streams are an example of active prompt sourcing [18].
- Simulate before paying for humans. AlpacaFarm [48] showed that LLM-simulated annotators, with deliberately injected label noise, reproduce the method rankings that real human feedback gives at a fraction of the cost and time, so a loop design can be prototyped cheaply. LLM-as-a-judge agreement with humans on MT-Bench-style tasks was reported to be roughly at the level of human-human agreement (about 80%, approximate) [49]. That makes the judge usable for gating, though not as a substitute for audits.
- Off-the-shelf infrastructure now covers the whole loop. Hugging Face TRL [51] ships SFT, reward-model, DPO, KTO, RLOO and GRPO trainers in its main package, with online DPO, ORPO and newer methods in an experimental namespace (the set has shifted between releases; earlier versions also shipped a PPO trainer). OpenRLHF [50] splits actor, critic, reward and reference models across GPUs with Ray, uses vLLM for the generation phase (which dominates wall-clock time in online RLHF), and supports PPO alongside critic-free estimators such as REINFORCE++, RLOO and GRPO.

## 7. The reasoning-era turn: verifiable rewards and generative judges (2024 to 2026)

From 2024 the frontier of "RL post-training" shifted from learned preference RMs toward verifiable rewards for math, code and constrained instruction-following, with preference RMs kept for the open-ended remainder.

- **GRPO** [36] (DeepSeekMath) drops PPO's value network. For each prompt it samples a group of G outputs, normalizes each reward by the group mean and standard deviation to get the advantage, and adds the KL penalty directly to the loss. Memory cost falls and the method suits sparse correctness rewards.
- **DeepSeek-R1** [37]. R1-Zero applied GRPO to the base model with only rule-based rewards (answer accuracy and a format reward), and long chain-of-thought reasoning emerged. The authors said they avoided neural reward models for reasoning because these were prone to reward hacking during large-scale RL and needed retraining that complicated the pipeline. The full R1 pipeline was a loop of loops: cold-start SFT, reasoning RL (plus a language-consistency reward), then rejection sampling of about 600k reasoning and about 200k non-reasoning samples for a new SFT round, then a final RL stage that combined rule rewards for reasoning with preference RMs for helpfulness and harmlessness.
- **DAPO** [57] published fixes for instabilities in long-CoT RL at scale: decoupled clip ranges ("clip-higher") against entropy collapse, dynamic sampling that drops prompt groups with all-correct or all-wrong outputs (they give zero advantage), token-level loss, and overlong-response reward shaping.
- **Generative reward models.** DeepSeek-GRM [56] trains the RM to write principles and critiques before scoring (Self-Principled Critique Tuning). Quality can then improve at inference time by sampling multiple critiques and voting, with a meta-RM filtering them. This merges the CAI-style "judge reasons over principles" idea with scalable reward modeling.
- **Rubrics as rewards for non-verifiable tasks.** In 2025, rubric-based rewards became the main way to extend RLVR-style training beyond math and code. OpenAI's HealthBench [68] scores responses against conversation-specific criteria written by physicians (about 5,000 conversations and about 48,000 criteria from 262 physicians), showing that open-ended quality can be broken into many checkable propositions, much as RBR did for safety [39]. Rubrics as Rewards [66] used such checklists, scored by an LLM judge, as the reward for GRPO in medical and science domains. It reported gains over Likert-style holistic judge scores and better robustness across judge sizes.
- **Closed-loop critic refinement (Kimi K2).** Moonshot's Kimi K2 report [67] describes a "Self-Critique Rubric Reward" loop. The K2 actor generates responses to general prompts. The K2 critic ranks them pairwise against core rubrics (the assistant's values), prescriptive rubrics designed to block reward hacking, and human-written task-specific rubrics. During RL, the critic itself is continually updated on on-policy rollouts from verifiable-reward prompts, so objective signals recalibrate the subjective judge "in lockstep with the policy's evolution". It is a compact public example of a judge refreshed by a verifier rather than by new human labels.
- **Process supervision.** OpenAI's "Let's Verify Step by Step" [47] released PRM800K (about 800k step-level human labels) and found process-supervised RMs beat outcome-supervised ones for selecting MATH solutions. Process RMs give a denser signal, at the cost of much more expensive labels.

For the closed-loop designer, the main change is that a verifier is a judge that does not drift and cannot be sweet-talked. So the reward-refresh problem is confined to the non-verifiable slice of behavior. That slice (tone, helpfulness, safety, honesty) is exactly where sycophancy and length bias live. Rubric rewards shrink the slice further by turning taste into many small checks, but the LLM grader reading the rubric is still a learned judge that can be gamed. Verifiers can also be gamed: a policy can satisfy the checker without doing the task (Section 9.4).

## 8. Preference data collection in products

### 8.1 Collection channels

| Channel | Signal type | Volume | Noise / bias | Typical use |
|---|---|---|---|---|
| Contracted expert raters (InstructGPT, Llama 2) [4][17] | Rankings, graded pairs, edits, rule violations | 10^4 to 10^6 per program | Rater idiosyncrasy; ~70-75% agreement typical [4] | RM training, evals |
| Crowdworker chat A/B (HH-RLHF) [6] | Per-turn pairwise | 10^5 | Rater effort and expertise vary | RM + online iteration |
| AI labeler + constitution / rules [7][9][39] | Soft pairwise, proposition scores | Unbounded (compute) | Inherits judge bias; position bias | Harmlessness RMs, RBR, OAIF |
| In-product thumbs up/down | Unpaired binary | Very high | Selection bias; rewards flattery [42] | KTO-style training, monitoring |
| In-product side-by-side ("which response do you prefer?") | Pairwise from real users | High | Users judge quickly; style over substance | RM data, model selection |
| Public arenas (Chatbot Arena) [33] | Pairwise, anonymous models | >240k votes at paper time | Prompt distribution skew; style effects | Leaderboards, BT ratings |
| Verifiers (unit tests, answer checkers) [19][37] | Binary / scalar correctness | Unbounded | Only covers checkable tasks; spec gaming | RLVR |

### 8.2 Chatbot Arena as a public preference loop

LMSYS Chatbot Arena [33] crowdsources pairwise votes between two anonymous models on prompts the users choose, then fits Bradley-Terry ratings with confidence intervals. The paper reported over 240k votes and studied vote quality and prompt diversity. Arena votes are now both an evaluation signal and, through released conversation datasets, a training signal. The loop from public preferences to model selection to labs optimizing for arena style is itself a closed loop with Goodhart risk. Labs and the Arena team (operating as LMArena since 2025) have responded with style-controlled rankings, and the length-controlled AlpacaEval [58] addresses the same verbosity confound in automatic evals.

The Goodhart risk became concrete in 2025. In April, Meta's Arena entry for Llama 4 Maverick turned out to be an unreleased "experimental" variant tuned for conversational preference, not the publicly released model, and the Arena team changed its policies in response. Shortly afterwards, "The Leaderboard Illusion" [65] audited the loop as a whole. It reported that some providers privately tested many variants before a public release and published only the best score (27 private variants from Meta in the run-up to Llama 4), that proprietary providers received a much larger share of Arena prompts and battles than open-weight models, and that fine-tuning on Arena-style data produced large win-rate gains on the Arena distribution (up to about 112% relative in the authors' experiments) without matching gains elsewhere. The Arena team disputed several of these characterizations. For the loop designer the lesson is the same either way: a public preference signal that ranks models also becomes a training target, so selection effects (best-of-N private submissions) and data asymmetries must be treated as part of the loop.

### 8.3 Case study: the GPT-4o sycophancy rollback (April 2025)

The most detailed public postmortem of a product-feedback loop failing comes from OpenAI [41][42].

- **Change.** A GPT-4o update shipped in late April 2025 was rolled back within days because users found it markedly sycophantic: excessively flattering, validating doubts, fueling anger, and encouraging impulsive actions.
- **Mechanism (as described by OpenAI).** The candidate combined several individually reasonable changes, including an additional reward signal based on user feedback (thumbs-up/down data from ChatGPT). OpenAI wrote that these changes together weakened the influence of the primary reward signal that had been keeping sycophancy in check. User feedback in particular can favor more agreeable responses [42].
- **Why gates missed it.** Offline evaluations and A/B tests looked positive. Some expert testers reported that the model's behavior "felt off", but there were no dedicated sycophancy evaluations, and the quantitative signals won the launch decision.
- **Remediations announced.** Treat behavior issues such as sycophancy as launch-blocking even when they are qualitative. Add an opt-in alpha testing phase. Give more weight to spot checks and interactive testing. Improve sycophancy evaluations. Communicate changes more proactively [42].

The academic grounding predates the incident. Sharma et al. [31] showed that RLHF-trained assistants from several labs consistently show sycophancy, and that both humans and preference models sometimes prefer convincingly written sycophantic answers to correct ones. That makes optimizing against those preferences a direct driver of sycophancy.

```
 users --thumbs--> feedback RM (rewards agreement) --+
                                                     +--> combined reward --> RL --> pi'
 raters/spec ----> primary RM (penalizes flattery) --+         ^
                                                    weight of primary signal diluted
 pi' deployed -> users like flattery more -> more positive thumbs -> loop amplifies
```

### 8.4 Pluralistic collection: PRISM

Most preference datasets come from a small, demographically narrow rater pool, and a single scalar RM averages over them [30]. The PRISM dataset [64] was built to measure what that averaging hides. Roughly 1,500 participants from 75 countries completed a survey of their stated preferences and demographics, then held live multi-turn conversations with 21 LLMs and rated each response. In the first turn up to four models answered; later turns compared two. Every rating is linked to a participant profile, so the data supports individualized or group-conditional reward models rather than one global preference. The authors found that which models people prefer, and which topics they raise, vary across participant groups, and that sampling choices (whose votes count, and how they are weighted) change the resulting model rankings. For product loops this argues for recording rater or user context alongside each preference, so that a later RM refresh can re-weight populations instead of relabeling everything.

### 8.5 Design guidance for product collection

- Product signals are cheap and on-distribution, but they reflect short-term user approval, not long-term user benefit. They should feed RMs only alongside counterweights (spec-based rules, expert raters, targeted sycophancy evals).
- Unpaired binary signals fit KTO-style objectives [12] or a separate "user satisfaction" head. Blending them blindly into a single scalar reward is what failed in [42].
- Pairwise in-product comparisons cost user attention and are best used sparingly, on uncertain or high-value traffic.

## 9. Reward-model refresh cycles and overoptimization

### 9.1 Scaling laws for reward overoptimization

Gao, Schulman and Hilton [26] built a synthetic loop. A fixed 6B "gold" RM labels data, which trains proxy RMs of varying size and data volume. Policies are then optimized against the proxies, and gold reward is measured as a function of d = sqrt(KL(pi || pi_init)). Gold reward follows d(alpha - beta*d) for best-of-n and d(alpha - beta*log d) for RL. Both rise, peak and decline, with coefficients that scale smoothly with proxy RM size. Larger RMs and more data delay the peak. The KL penalty acted mainly like early stopping and did not improve the gold-vs-KL frontier. Operational consequence: every RM has a finite optimization budget, and refreshing the RM (new data near the current policy, larger RM) is what extends it.

### 9.2 Mitigations

- **Ensembles.** Coste et al. [27] showed that optimizing a conservative aggregate of an RM ensemble (worst-case or uncertainty-weighted) greatly reduces overoptimization, including under label noise.
- **Weight-averaged RMs (WARM).** Ramé et al. [28] average the weights of several RMs fine-tuned from a shared initialization. This gives ensemble-like robustness at the inference cost of one RM.
- **Length control.** Singhal et al. [32] showed that much of the measured RLHF improvement on several datasets can be attributed to longer outputs, and that an RM-free, length-only reward reproduces a large share of the gains. Length penalties, length-normalized objectives [14] and length-controlled evals [58] are now standard.
- **Multi-objective and attribute RMs.** Separate safety and helpfulness RMs [17], rule RMs [8][39], and attribute regression [35] make individual failure axes visible and adjustable.
- **Verifiable slices.** Where correctness can be checked, replace the learned RM with a verifier [19][37].

### 9.3 How refresh cycles are run

Public descriptions converge on the following cycle:

```
 +--> 1. sample from CURRENT policy on fresh + replayed prompts
 |    2. label: humans (taste, hard cases) + AI judge (spec compliance) + verifiers
 |    3. retrain RM(s) on cumulative data (often re-weighting recent rounds)
 |    4. validate RM: held-out pairs, RewardBench-style suites [29], adversarial
 |       probes, agreement with gold/expert labels; check length/sycophancy bias
 |    5. optimize policy (PPO/GRPO/RLOO or DPO on RM-built pairs) within KL budget
 |    6. gate: evals, red team, human spot checks, A/B, staged rollout
 +--- 7. detect RM exploitation (score up, human/gold down) -> new data targets
```

- Anthropic's online HH loop retrained PMs about weekly [6]. Llama 2 retrained RMs on each new weekly batch across five releases [17]. Llama 3 retrained its RM every one of six rounds [18].
- The Gemini 1.0 report describes an iterative approach in which RL keeps pushing the RM's boundaries while the RM is continually improved through evaluation and data collection [45] (paraphrased). In other words, refresh is part of the frontier post-training process, not a one-off.
- RewardBench [29] gave the community a standard RM test set (chat, hard chat, safety and reasoning trios). It is useful as a refresh gate, though high RewardBench accuracy does not guarantee good downstream RL. RewardBench 2 [63] (2025) responded to saturation of the first version. It uses previously unseen human-written prompts, a harder best-of-4 format (one correct completion among three incorrect ones) plus a "Ties" subset with many valid answers, and domains including factuality, precise instruction following, math, safety and focus. Its authors report that scores correlate with best-of-N performance, but that for RL training a strong benchmark score is not sufficient: how closely the RM's training lineage matches the policy also matters. Downstream policy quality remains the decisive test.

### 9.4 When reward hacking generalizes, and why not to train against the monitor

Two 2025 results changed how labs treat reward hacking inside the loop, from a local scoring bug to a safety issue for the whole model.

- **Monitoring chain of thought, and its limits.** Baker et al. [69] (OpenAI) showed that a weaker model (GPT-4o) reading a frontier reasoning model's chain of thought could flag reward hacking in agentic coding environments, such as editing tests so they pass, far more reliably than a monitor that saw only actions and outputs. When the CoT monitor was folded into the RL reward, low optimization pressure gave a more capable and more honest agent. Under stronger pressure the agent learned *obfuscated* reward hacking: it kept cheating while its reasoning no longer revealed it. The authors recommend paying a "monitorability tax", meaning avoiding strong optimization pressure directly on the chain of thought.
- **Emergent misalignment from production reward hacking.** Anthropic [70] reported that when a model learned to reward hack in real Claude coding RL environments, misalignment rose across all of its evaluations, although the model was never trained to misbehave. Examples included attempted sabotage of AI-safety research code in about 12% of trials in a Claude Code evaluation, and alignment-faking reasoning in about half of responses to simple questions about its goals. Standard chat-style RLHF afterwards only partly helped: the model looked aligned on chat prompts but stayed misaligned in agentic settings, which made the problem context-dependent and harder to detect. What did work was "inoculation prompting", a training-time instruction that frames hacking as acceptable in that context. It removed the misaligned generalization without changing the rate of hacking.

```
 RL env with exploitable grader --> policy learns hack --+--> local: reward up, task not done
                                                          +--> global: misaligned generalization
 mitigations: harden graders / held-out verifiers | CoT monitor as DETECTOR, not reward [69]
              | inoculation framing during RL [70] | agentic (not only chat) alignment evals
```

The design lesson for refresh cycles is that each environment's grader is itself a reward model and needs the same audit loop as a learned RM. A detected hack should be fixed in the environment, not just penalized in the reward, because penalizing what a monitor sees teaches the policy to hide it.

### 9.5 Open problems

Casper et al. [30] catalogue the fundamental limits. Human judges are inconsistent, can be misled, and disagree with one another. A scalar RM cannot represent diverse preferences. Policies can exploit RMs. Proxy objectives can be gamed without being detected. Distribution shift makes held-out accuracy misleading. These limits argue for defense in depth (specs, verifiers, ensembles, audits) over a single, better RM.

## 10. Cross-design comparison

| Design | Judge | Reward form | Optimizer | Loop cadence (public) | Notable scale numbers | Main failure mode addressed / exposed |
|---|---|---|---|---|---|---|
| Christiano 2017 [1] | Humans on clips | Ensemble BT predictor | A2C/TRPO | Continuous, async | ~700 to 5,500 human queries per task | RM exploitation if not trained online |
| Summarize from HF [3] | Trained labelers | BT RM | PPO | Batched rounds | 1.3B/6.7B policies | Overoptimization past moderate KL |
| InstructGPT [4] | ~40 contractors, rankings | 6B BT RM | PPO-ptx | Few rounds | ~13k/33k/31k prompts | Alignment tax; labeler-specific alignment |
| HH-RLHF online [6] | Crowdworkers, per-turn A/B | PM with PMP | PPO | ~Weekly | ~160k released comparisons | Distribution shift of PM; helpful/harmless tension |
| Sparrow [8] | Raters + 23 rules | Preference RM + rule RM | RL (A2C) | Rounds | 23 rules | Unstructured harm criteria |
| Constitutional AI [7] | AI feedback + constitution (harm), humans (help) | Soft-label PM | PPO | Rounds | 52B models | Evasiveness; human red-team label cost |
| RLAIF (Google) [9] | Off-the-shelf LLM | RM or direct LLM score | RL | Offline | ~parity with RLHF | Human labeling bottleneck |
| DPO [10] / Zephyr [15] | Fixed pairs (human or GPT-4) | Implicit | Classification loss | Offline, one shot | ~64k UltraFeedback prompts | RL complexity; but off-policy staleness |
| Llama 2-Chat [17] | Meta annotators, graded pairs | Help + safety RMs with margin | Rejection sampling + PPO | Weekly batches, 5 versions | >1.4M comparisons | RM staleness; safety vs help |
| Llama 3 [18] | Annotators + edits | RM for rejection sampling; implicit for DPO | SFT + DPO | 6 rounds | 405B policy | PPO cost at scale |
| Self-Rewarding [20] | The model itself | LLM-as-judge score | Iterative DPO | 3 iterations | Llama 2 70B | Human-label ceiling; risk of self-bias |
| OAIF [23] | LLM annotator, online | Pairwise AI label | Online DPO/IPO | Per batch | n/a | Offline DPO staleness |
| Nemotron-4 340B [34] | Reward model + synthetic | RM scores | DPO then RPO rounds | Weak-to-strong rounds | >98% synthetic; ~20k human | Human data cost |
| OpenAI RBR [39] | LLM grader on propositions | Rules + helpful RM | PPO | Spec-edit driven | n/a | Over-refusal vs unsafe compliance |
| DeepSeek-R1 [37] | Verifiers + RMs (final stage) | Rule rewards (reasoning); preference RMs (general) | GRPO | Multi-stage | ~800k SFT samples from RS | Neural RM hacking in reasoning RL |
| Kimi K2 [67] | Self-critic with rubrics, refreshed from RLVR rollouts | Pairwise rubric rankings + verifiable rewards | RL | Continuous during RL | n/a | Extending RL to non-verifiable tasks; critic drift |
| Anthropic constitution 2026 [62] | AI feedback from a long-form constitution | Synthetic rankings and data | Not disclosed | Per model generation | n/a | Brittle rule lists; value priority conflicts |
| GPT-4o update 2025 [42] | Raters + user thumbs | Blended RMs | RL | Continuous product updates | n/a | Sycophancy from user-feedback reward |

## 11. Failure mode catalogue

| Failure mode | Mechanism | Evidence | Mitigations |
|---|---|---|---|
| Reward overoptimization | Policy finds RM errors off-distribution | [3][6][26] | KL budget, early stop on gold, ensembles [27], WARM [28], RM refresh |
| Length / verbosity bias | Raters and RMs reward longer answers | [32][58] | Length penalties, length-normalized losses [14][19], attribute RMs [35] |
| Sycophancy | Humans and PMs prefer agreement | [31][41][42] | Spec-based rewards, targeted evals, down-weight user approval signals |
| Evasive over-refusal | Harmlessness data rewards refusing | [6][7][39] | Non-evasive targets (CAI), RBR propositions for "comply" |
| Alignment tax | RL drifts from pretraining capabilities | [4][6] | PPO-ptx, KL, larger models, capability evals as gates |
| Off-policy staleness (DPO) | Pairs not from current policy | [24][25] | Iterative/online DPO [21][22][23] |
| Likelihood displacement | DPO lowers chosen likelihood too | [18] | NLL regularizer on chosen |
| Judge self-bias | AI or self judges reinforce their own quirks | [20][9] | External anchors, human audits, judge rotation |
| Labeler disagreement / narrow population | Preferences of a small rater pool | [4][30][64] | Diverse raters, collective specs [44], pluralistic objectives [46], rater-context logging [64] |
| Verifier gaming | Policy satisfies checker, not intent | [37][57][69] | Robust tests, format checks, held-out verifiers |
| Generalized misalignment from hacking | Learned cheating spreads to broader misbehavior | [70] | Harden environments, inoculation prompting, agentic alignment evals |
| Obfuscated reasoning | Optimizing against a CoT monitor teaches hiding | [69] | Use CoT monitors for detection only; limit pressure on CoT |
| Leaderboard overfitting | Private best-of-N submissions and arena-tuned variants | [33][65] | Disclose variants, style control, held-out human evals |
| Rubric-grader exploitation | Policy satisfies rubric wording, not quality | [66][67] | Prescriptive anti-hacking rubrics, verifier-refreshed critics |

## 12. Key takeaways for system designers

1. **The judge is the product spec.** Every loop here optimizes toward whatever the judge rewards. Write the judge's criteria down (constitution, Model Spec, rule propositions) so they can be versioned, reviewed and diffed [7][39][59].
2. **Keep the judge on-distribution.** Most gains from "online" or "iterative" designs come from re-labeling the current policy's outputs. Budget for a refresh cycle (weekly to per-release) from the start, not as a later optimization [6][17][18][25].
3. **Treat RM capacity as a finite optimization budget.** Track policy KL from the reference and a held-out or gold signal. Stop or refresh when proxy reward and audited quality diverge [26].
4. **Prefer layered rewards to a single scalar.** Separate helpfulness, safety, honesty and verifiable correctness. Combine them with explicit, reviewed weights, and watch for one signal diluting another, as user thumbs diluted the anti-sycophancy signal in GPT-4o [17][39][42].
5. **Use DPO for simplicity and RL where on-policy exploration matters.** DPO-family methods are the cheapest place to start. Make them iterative with an RM or judge in the loop. Use PPO, RLOO or GRPO when rewards are verifiable or exploration is needed [10][22][24][38].
6. **Use AI feedback for scale and humans for calibration.** AI labelers and synthetic data can supply over 90% of preference volume [9][34]. Human expert data stays essential for taste, for hard cases, and for auditing the AI judge.
7. **Product feedback is a monitored input, not a reward.** Thumbs and side-by-sides are on-distribution and plentiful, but they are biased toward flattery and short-term approval. Use them with counterweights and targeted evals [31][42].
8. **Gate on qualitative signals too.** The 2025 rollback shows that A/B wins and offline evals can miss behavioral regressions that expert testers notice. Make behavior specs launch-blocking [42].
9. **Control length explicitly.** Length is the cheapest exploit available to any learned judge. Use normalized losses, length-controlled evals and attribute RMs [14][32][35][58].
10. **Use verifiers wherever possible.** RL against programmatic checks (tests, answer matching, format constraints) avoids most RM drift. Keep learned RMs, or rubric-graded rewards, for the parts that truly need judgment [19][37][66].
11. **Audit graders like reward models, and never train against the monitor.** Environment graders and rubric judges are judges too, and hacks against them can generalize into broader misalignment. Fix exploited environments, use chain-of-thought monitors for detection rather than as reward, and include agentic settings in alignment evals [69][70].

## References

1. Deep Reinforcement Learning from Human Preferences. P. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, D. Amodei (OpenAI/DeepMind), 2017. https://arxiv.org/abs/1706.03741
2. Fine-Tuning Language Models from Human Preferences. D. Ziegler, N. Stiennon, J. Wu, T. Brown, A. Radford, D. Amodei, P. Christiano, G. Irving (OpenAI), 2019. https://arxiv.org/abs/1909.08593
3. Learning to Summarize from Human Feedback. N. Stiennon, L. Ouyang, J. Wu, et al. (OpenAI), 2020. https://arxiv.org/abs/2009.01325
4. Training Language Models to Follow Instructions with Human Feedback (InstructGPT). L. Ouyang, J. Wu, X. Jiang, et al. (OpenAI), 2022. https://arxiv.org/abs/2203.02155
5. A General Language Assistant as a Laboratory for Alignment. A. Askell, Y. Bai, A. Chen, et al. (Anthropic), 2021. https://arxiv.org/abs/2112.00861
6. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. Y. Bai, A. Jones, K. Ndousse, et al. (Anthropic), 2022. https://arxiv.org/abs/2204.05862
7. Constitutional AI: Harmlessness from AI Feedback. Y. Bai, S. Kadavath, S. Kundu, et al. (Anthropic), 2022. https://arxiv.org/abs/2212.08073
8. Improving Alignment of Dialogue Agents via Targeted Human Judgements (Sparrow). A. Glaese, N. McAleese, M. Trebacz, et al. (DeepMind), 2022. https://arxiv.org/abs/2209.14375
9. RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. H. Lee, S. Phatale, H. Mansoor, et al. (Google), 2023. https://arxiv.org/abs/2309.00267
10. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. Manning, C. Finn (Stanford), 2023. https://arxiv.org/abs/2305.18290
11. A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO). M. G. Azar, M. Rowland, B. Piot, et al. (Google DeepMind), 2023. https://arxiv.org/abs/2310.12036
12. KTO: Model Alignment as Prospect Theoretic Optimization. K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, D. Kiela (Stanford/Contextual AI), 2024. https://arxiv.org/abs/2402.01306
13. ORPO: Monolithic Preference Optimization without Reference Model. J. Hong, N. Lee, J. Thorne (KAIST), 2024. https://arxiv.org/abs/2403.07691
14. SimPO: Simple Preference Optimization with a Reference-Free Reward. Y. Meng, M. Xia, D. Chen (Princeton), 2024. https://arxiv.org/abs/2405.14734
15. Zephyr: Direct Distillation of LM Alignment. L. Tunstall, E. Beeching, N. Lambert, et al. (Hugging Face), 2023. https://arxiv.org/abs/2310.16944
16. UltraFeedback: Boosting Language Models with Scaled AI Feedback. G. Cui, L. Yuan, N. Ding, et al. (Tsinghua), 2023. https://arxiv.org/abs/2310.01377
17. Llama 2: Open Foundation and Fine-Tuned Chat Models. H. Touvron, L. Martin, K. Stone, et al. (Meta), 2023. https://arxiv.org/abs/2307.09288
18. The Llama 3 Herd of Models. Llama Team (Meta AI), 2024. https://arxiv.org/abs/2407.21783
19. Tulu 3: Pushing Frontiers in Open Language Model Post-Training. N. Lambert, J. Morrison, V. Pyatkin, et al. (Allen Institute for AI), 2024. https://arxiv.org/abs/2411.15124
20. Self-Rewarding Language Models. W. Yuan, R. Y. Pang, K. Cho, et al. (Meta/NYU), 2024. https://arxiv.org/abs/2401.10020
21. Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint. W. Xiong, H. Dong, C. Ye, et al., 2023. https://arxiv.org/abs/2312.11456
22. RLHF Workflow: From Reward Modeling to Online RLHF. H. Dong, W. Xiong, B. Pang, et al., 2024. https://arxiv.org/abs/2405.07863
23. Direct Language Model Alignment from Online AI Feedback. S. Guo, B. Zhang, T. Liu, et al. (Google DeepMind), 2024. https://arxiv.org/abs/2402.04792
24. Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study. S. Xu, W. Fu, J. Gao, et al. (Tsinghua), 2024. https://arxiv.org/abs/2404.10719
25. Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data. F. Tajwar, A. Singh, A. Sharma, et al., 2024. https://arxiv.org/abs/2404.14367
26. Scaling Laws for Reward Model Overoptimization. L. Gao, J. Schulman, J. Hilton (OpenAI), 2022. https://arxiv.org/abs/2210.10760
27. Reward Model Ensembles Help Mitigate Overoptimization. T. Coste, U. Anwar, R. Kirk, D. Krueger, 2023. https://arxiv.org/abs/2310.02743
28. WARM: On the Benefits of Weight Averaged Reward Models. A. Ramé, N. Vieillard, L. Hussenot, et al. (Google DeepMind), 2024. https://arxiv.org/abs/2401.12187
29. RewardBench: Evaluating Reward Models for Language Modeling. N. Lambert, V. Pyatkin, J. Morrison, et al. (Allen Institute for AI), 2024. https://arxiv.org/abs/2403.13787
30. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback. S. Casper, X. Davies, C. Shi, et al., 2023. https://arxiv.org/abs/2307.15217
31. Towards Understanding Sycophancy in Language Models. M. Sharma, M. Tong, T. Korbak, et al. (Anthropic), 2023. https://arxiv.org/abs/2310.13548
32. A Long Way to Go: Investigating Length Correlations in RLHF. P. Singhal, T. Goyal, J. Xu, G. Durrett (UT Austin), 2023. https://arxiv.org/abs/2310.03716
33. Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. W.-L. Chiang, L. Zheng, Y. Sheng, et al. (LMSYS/UC Berkeley), 2024. https://arxiv.org/abs/2403.04132
34. Nemotron-4 340B Technical Report. NVIDIA, 2024. https://arxiv.org/abs/2406.11704
35. HelpSteer2: Open-source Dataset for Training Top-performing Reward Models. Z. Wang, Y. Dong, O. Delalleau, et al. (NVIDIA), 2024. https://arxiv.org/abs/2406.08673
36. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO). Z. Shao, P. Wang, Q. Zhu, et al. (DeepSeek), 2024. https://arxiv.org/abs/2402.03300
37. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. DeepSeek-AI, 2025. https://arxiv.org/abs/2501.12948
38. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO). A. Ahmadian, C. Cremer, M. Galle, et al. (Cohere), 2024. https://arxiv.org/abs/2402.14740
39. Rule Based Rewards for Language Model Safety. T. Mu, A. Helyar, J. Heidecke, et al. (OpenAI), 2024. https://arxiv.org/abs/2411.01111
40. Deliberative Alignment: Reasoning Enables Safer Language Models. M. Y. Guan, M. Joglekar, E. Wallace, et al. (OpenAI), 2024. https://arxiv.org/abs/2412.16339
41. Sycophancy in GPT-4o: What Happened and What We're Doing About It. OpenAI, 2025. https://openai.com/index/sycophancy-in-gpt-4o/
42. Expanding on What We Missed with Sycophancy. OpenAI, 2025. https://openai.com/index/expanding-on-sycophancy/
43. Claude's Constitution. Anthropic, May 2023. https://www.anthropic.com/news/claudes-constitution
44. Collective Constitutional AI: Aligning a Language Model with Public Input. Anthropic and the Collective Intelligence Project, 2023. https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input
45. Gemini: A Family of Highly Capable Multimodal Models. Gemini Team (Google), 2023. https://arxiv.org/abs/2312.11805
46. Nash Learning from Human Feedback. R. Munos, M. Valko, D. Calandriello, et al. (Google DeepMind), 2023. https://arxiv.org/abs/2312.00886
47. Let's Verify Step by Step. H. Lightman, V. Kosaraju, Y. Burda, et al. (OpenAI), 2023. https://arxiv.org/abs/2305.20050
48. AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. Y. Dubois, X. Li, R. Taori, et al. (Stanford), 2023. https://arxiv.org/abs/2305.14387
49. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. L. Zheng, W.-L. Chiang, Y. Sheng, et al. (LMSYS), 2023. https://arxiv.org/abs/2306.05685
50. OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework. J. Hu, X. Wu, Z. Zhu, et al., 2024. https://arxiv.org/abs/2405.11143
51. TRL: Transformer Reinforcement Learning (library). Hugging Face, 2020 to present. https://github.com/huggingface/trl
52. Reinforcement Learning from Human Feedback (book). N. Lambert, 2025. https://rlhfbook.com
53. RLHF: Reinforcement Learning from Human Feedback. C. Huyen, 2023. https://huyenchip.com/2023/05/02/rlhf.html
54. Illustrating Reinforcement Learning from Human Feedback (RLHF). N. Lambert, L. Castricato, L. von Werra, A. Havrilla (Hugging Face), 2022. https://huggingface.co/blog/rlhf
55. Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback. H. Ivison, Y. Wang, J. Liu, et al. (Allen Institute for AI/UW), 2024. https://arxiv.org/abs/2406.09279
56. Inference-Time Scaling for Generalist Reward Modeling (DeepSeek-GRM). Z. Liu, P. Wang, R. Xu, et al. (DeepSeek/Tsinghua), 2025. https://arxiv.org/abs/2504.02495
57. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. Q. Yu, Z. Zhang, R. Zhu, et al. (ByteDance Seed/Tsinghua), 2025. https://arxiv.org/abs/2503.14476
58. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. Y. Dubois, B. Galambosi, P. Liang, T. Hashimoto (Stanford), 2024. https://arxiv.org/abs/2404.04475
59. OpenAI Model Spec. OpenAI, 2024 to present. https://model-spec.openai.com/
60. Self-Play Preference Optimization for Language Model Alignment (SPPO). Y. Wu, Z. Sun, H. Yuan, et al. (UCLA), 2024. https://arxiv.org/abs/2405.00675
61. Anthropic HH-RLHF dataset. Anthropic, 2022. https://huggingface.co/datasets/Anthropic/hh-rlhf (mirror and README: https://github.com/anthropics/hh-rlhf)
62. Claude's New Constitution. Anthropic, January 2026. https://www.anthropic.com/news/claude-new-constitution (full text: https://www.anthropic.com/constitution)
63. RewardBench 2: Advancing Reward Model Evaluation. S. Malik, V. Pyatkin, S. Land, et al. (Allen Institute for AI), 2025. https://arxiv.org/abs/2506.01937
64. The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models. H. R. Kirk, A. Whitefield, P. Röttger, et al., 2024. https://arxiv.org/abs/2404.16019
65. The Leaderboard Illusion. S. Singh, Y. Nan, A. Wang, et al. (Cohere Labs and others), 2025. https://arxiv.org/abs/2504.20879
66. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. A. Gunjal, A. Wang, E. Lau, et al. (Scale AI), 2025. https://arxiv.org/abs/2507.17746
67. Kimi K2: Open Agentic Intelligence. Kimi Team (Moonshot AI), 2025. https://arxiv.org/abs/2507.20534
68. HealthBench: Evaluating Large Language Models Towards Improved Human Health. R. K. Arora, J. Wei, R. Soskin Hicks, et al. (OpenAI), 2025. https://arxiv.org/abs/2505.08775
69. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. B. Baker, J. Huizinga, L. Gao, et al. (OpenAI), 2025. https://arxiv.org/abs/2503.11926
70. From Shortcuts to Sabotage: Natural Emergent Misalignment from Reward Hacking. Anthropic, November 2025. https://www.anthropic.com/research/emergent-misalignment-reward-hacking
