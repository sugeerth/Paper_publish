# 04. LLM-as-a-Judge: Protocols, Biases, Calibration, Ensembles, Fine-Tuned Judges and Judge Alignment

## Overview

An LLM judge (also called a model grader, autorater or LLM-evaluator) is a language model that is prompted, or trained, to score or compare the outputs of another system. Teams adopted it for three reasons. Open-ended outputs such as chat answers, summaries, agent transcripts and medical advice cannot be graded by exact match. Human raters are slow and expensive: AlpacaEval's own accounting puts human annotation at about $300 and roughly 10 hours per 1,000 comparisons, against roughly $0.4 to $15 and minutes for LLM annotators [8]. And since 2023 strong models have agreed with humans about as often as humans agree with each other on general chat preference tasks [1][2].

A judge is still a measuring instrument. Like any instrument it has bias, variance and drift, and it can be gamed. Most of the published work since 2023 tries to characterise those errors and design around them. Surveys now catalogue hundreds of papers in this area [54]. This chapter organises that work around the design decisions a system builder faces:

1. **Protocol**: pointwise scoring, pairwise comparison, reference-guided grading, or rubric and checklist decomposition.
2. **Prompt mechanics**: chain-of-thought before the verdict, form-filling, probability-weighted scores, and few-shot anchors.
3. **Bias control**: position swapping, length and style regression, avoiding self-judging, and adversarial meta-evaluation.
4. **Calibration**: agreement metrics against humans, confidence intervals, selective abstention, and the statistical limits of debiasing.
5. **Aggregation**: single judge, jury or panel, or a cascade.
6. **Judge model**: a prompted frontier model or a fine-tuned open evaluator (Prometheus, JudgeLM, Auto-J, FLAMe, Self-Taught Evaluator, J1, Selene).
7. **Alignment workflow**: how criteria and judge prompts are iterated against expert labels (EvalGen, critique shadowing, MemAlign, SIMBA and GEPA optimisers).

A note on sourcing: where a number comes from an authors' repository or blog rather than the paper itself, the reference points to that artefact. Numbers recalled from paper abstracts that could not be re-fetched while writing are marked "approx."

## 1. The canonical judge loop

Every LLM-judge system reduces to the same data flow. Implementations differ in which boxes they include and how they aggregate.

```
             +-------------------+      +------------------------------+
 inputs ---->| System under test |----->| candidate output(s)          |
 (prompt,    +-------------------+      | A (and B for pairwise)       |
  context)                              +--------------+---------------+
                                                       |
   +--------------------+   +----------------------+   |
   | criteria / rubric  |   | reference answer     |   |
   | (from spec, expert,|   | (optional; gold or   |   |
   |  or EvalGen-style  |   |  judge-generated)    |   |
   |  elicitation)      |   +----------+-----------+   |
   +---------+----------+              |               |
             v                         v               v
        +--------------------------------------------------------+
        | Judge prompt assembly: task, criteria, few-shot        |
        | anchors, order randomisation / swap, output schema     |
        +---------------------------+----------------------------+
                                    v
        +--------------------------------------------------------+
        | Judge model(s): CoT -> verdict (score / label / A|B)   |
        | optional: logprobs, n samples, multiple judges         |
        +---------------------------+----------------------------+
                                    v
        +--------------------------------------------------------+
        | Parse + aggregate: swap-consistency, vote/average,     |
        | Bradley-Terry fit, length/style regression, bootstrap  |
        +---------------------------+----------------------------+
                                    v
        +--------------------------------------------------------+
        | Meta-evaluation: agreement with human labels (kappa,   |
        | MCC, F1, failure coverage), separability, CIs, drift   |
        +--------------------------------------------------------+
```

The OpenAI Evals "model-graded" template, one of the earliest public versions, already had most of these parts. It wraps the completion in an evaluation prompt, restricts the judge to a fixed set of `choice_strings` such as "ABCDE" or Yes/No/Unsure (any other output is parsed as `__invalid__`), maps choices to scores, and offers three output orders: `cot_classify` (reason then answer), `classify_cot`, and `classify`. The documentation recommends `cot_classify` as the default "as it typically provides the most accurate model-graded evaluations" [46]. Its `fact.yaml` template is an early reference-based design. It asks whether the submission is a subset, superset, equal to, inconsistent with, or immaterially different from an expert answer [46].

Current harnesses expose the same parts as configuration. UK AISI's Inspect has `model_graded_qa()` and `model_graded_fact()` scorers. They take a template, instructions, a `grade_pattern` regex (default `GRADE: C` or `GRADE: I`), optional `partial_credit` (scored 0.5), and `include_history` for multi-turn context. The `model` argument can be a list: "each model grades independently and the grades are combined by `reducer`" [47]. If no grader is bound, Inspect falls back to the model being evaluated, which invites self-preference (Section 4.3). It therefore offers `ModelRole("grader", required=True)` to fail closed [47]. LangChain's openevals uses a similar interface with `continuous` scores, discrete `choices` such as `[0.0, 0.5, 1.0]`, `few_shot_examples`, and a `use_reasoning` toggle [48].

## 2. The founding study: MT-Bench, Chatbot Arena and judge agreement

### 2.1 Problem

By early 2023, chat models were being compared on multiple-choice benchmarks such as MMLU that "do not reflect the typical use cases of LLM-based chat assistants" [2]. LMSYS had used GPT-4 as an ad hoc judge for Vicuna. The open question was whether such judgments could be trusted [2].

### 2.2 Design

Zheng et al. ("Judging LLM-as-a-judge with MT-Bench and Chatbot Arena") built two complementary instruments [1]:

- **MT-Bench**: 80 hand-written two-turn questions (160 turns), 10 in each of 8 categories: Writing, Roleplay, Extraction, Reasoning, Math, Coding, Knowledge I (STEM) and Knowledge II (humanities and social science). The categories were derived from two months of Chatbot Arena prompts [2].
- **Chatbot Arena**: crowdsourced, anonymous, side-by-side battles. The June 2023 leaderboard was built from about 42K votes [2]; the platform is described in [50].

The FastChat reference implementation supports three grading modes [3]:

| Mode | Mechanism | Cost scaling |
|---|---|---|
| `single` (default) | Judge scores each answer on a 1-10 scale per turn; turns are averaged | O(models x questions) |
| `pairwise-baseline` | Each model compared against a fixed baseline (default gpt-3.5-turbo) | O(models x questions) |
| `pairwise-all` | All model pairs on all questions | O(models^2 x questions); "could be more expensive" |

The authors released 3.3K expert pairwise human judgments over six models' MT-Bench answers so the judge could be validated against them [3].

### 2.3 Findings

- **Agreement.** "Strong LLM judges like GPT-4 can align impressively well with both controlled and crowdsourced human preferences, achieving over 80% agreement," which is "comparable to the agreement between two different human judges" [2]. Single-answer grading without pairwise comparison "can also rank models effectively" [2].
- **Four named limitations** [2]: *position bias* (favouring the first answer), *verbosity bias* (favouring longer answers "regardless of their quality"), *self-enhancement bias* (favouring its own responses), and *limited reasoning ability* when grading math and reasoning.
- **Mitigations studied** [1][2]: few-shot judges, chain-of-thought judges, reference-guided judges (the judge first solves the problem or is given a reference, then grades), and fine-tuned judges. The paper reports that guiding GPT-4 with a reference answer cut its failure rate on math grading questions from about 70% to about 15% (approx.). It also reports that a "repetitive list" attack, which pads an answer with rephrased items, fooled GPT-3.5 and Claude-v1 in most cases but GPT-4 far less often (approx. 9%) [1].

### 2.4 Lessons

MT-Bench established three ideas the field still uses: agreement with human preference is the validity criterion, biases have to be measured and controlled explicitly, and single-answer scores scale linearly while pairwise designs do not. Its weakness is resolution. An LMSYS follow-up found MT-Bench separated only 22.6% of top model pairs with 95% confidence, and agreed with the Arena on only 26.1% of pairs once confidence intervals were taken into account, despite a 91.3% Spearman correlation [9]. Section 3.3 returns to this.

## 3. Grading protocols: pointwise, pairwise, reference-based and rubric

### 3.1 Comparison

| Protocol | Judge sees | Output | Strengths | Known failure modes | Representative systems |
|---|---|---|---|---|---|
| Pointwise / direct assessment | One output (+ criteria) | Likert or 1-10 score, or pass/fail | Linear cost; absolute tracking over time; works for monitoring | Score compression and drift; scale ambiguity; weak separability | MT-Bench single [3], G-Eval [4][5], Prometheus absolute [29] |
| Pairwise | Two outputs | A / B / tie, sometimes graded (A>>B) | Relative judgments are easier and more stable; maps to Bradley-Terry | Position bias; needs a baseline or O(n^2) pairs; ties | AlpacaEval [8], Arena-Hard [9], PandaLM [31] |
| Listwise / best-of-n | k outputs | Ranking or best index | Amortises context; used in reward-model benchmarks | Longer context; order effects grow with k | RewardBench 2 rankings prompt [21] |
| Reference-guided | Output + gold or reference | Correct/incorrect, or consistency class | Turns open-ended grading closer to verification; helps on math | Penalises valid answers that differ from the reference; needs references | MT-Bench math [1], OpenAI `fact.yaml` [46], Inspect `model_graded_fact` [47] |
| Rubric / checklist | Output + per-item criteria | Per-criterion booleans, weighted | Decomposes judgment; auditable; per-tag analysis | Rubric authoring cost; criterion-level leniency; gameable by keyword coverage | HealthBench [45], Prometheus rubrics [28], CheckEval [53], FLASK skill sets [52], DeepEval DAG [6] |

### 3.2 G-Eval: form-filling with auto-CoT and probability-weighted scores

**Problem.** Classical NLG metrics (BLEU, ROUGE, BERTScore) correlate poorly with human judgments of summary or dialogue quality. Early prompt-based LLM scorers produced coarse integer scores with many ties.

**Architecture.** G-Eval (Liu et al., Microsoft, 2023) [4] runs in two steps:

```
 criteria text --> [LLM: generate evaluation steps (auto-CoT)] --> steps
                                                                    |
 (source, output) + steps --> [LLM: fill form, score 1..N] --> token probs
                                                                    |
                         score = sum_i  p(score = i) * i   <--------+
```

1. **Auto chain-of-thought.** Given a short criterion description (for example "Coherence: the collective quality of all sentences"), the LLM writes its own numbered evaluation steps.
2. **Form-filling.** The steps, the source and the candidate go into a form-style prompt that asks for a score on a small integer scale.
3. **Probability weighting.** Instead of taking the sampled integer, G-Eval computes the expected score over output-token probabilities. This yields continuous, fine-grained scores and fewer ties.

DeepEval's widely used implementation describes the same flow. It generates `evaluation_steps` from `criteria` if none are supplied, generates a 1-10 score, normalises it "using token probabilities," and divides by 10 to land in [0,1]. It notes that the weighted summation "was introduced in the paper because it minimizes bias in LLM scoring" [5]. Its practical advice is to let G-Eval generate steps once, then freeze them as explicit `evaluation_steps`, because supplying steps "make[s] it more reliable across different runs" [5].

**Results and caveats.** On SummEval, GPT-4-based G-Eval reported a Spearman correlation with human judgments of about 0.51 (approx.), well above the reference-based metrics of the time [4]. The paper also warned that LLM evaluators may prefer LLM-generated text, an early form of the self-preference concern [4].

**Descendant: deterministic decision graphs.** Holistic G-Eval scores are hard to control. DeepEval later added a `DAGMetric`, an LLM-powered decision tree in which each node is a focused binary or categorical judgment and the author assigns scores to the terminal outcomes. Its documentation positions it for "conditional rules, gates, and known scoring paths," with G-Eval for "subjective quality that is difficult to enumerate" [6].

### 3.3 Pairwise against an anchor: AlpacaEval and Arena-Hard

**AlpacaEval** compares each model with a fixed baseline on a fixed instruction set, using a GPT-4-class annotator. The repository publishes a meta-evaluation of candidate annotators against human preference labels. It reports human agreement of 69.2% for the `alpaca_eval_gpt4` annotator against 65.7% for a held-out human (humans agreeing with other humans), at $13.6 per 1,000 examples against $300. It also reports each annotator's probability of preferring the longer output: 0.68 for `alpaca_eval_gpt4`, 0.64 for humans, and 1.00 for a trivial "always pick longer" baseline that still reaches 62.2% human agreement [8]. That last row is a useful warning: a judge that only looks at length can look fairly good on raw agreement.

**Arena-Hard-Auto** (LMSYS, April 2024) [9][10] targets separability. Its pipeline:

```
 Chatbot Arena prompts --> embed + cluster into topics
         |
         v
 LLM annotator scores each prompt on 7 criteria
 (specificity, domain knowledge, complexity, problem-solving,
  creativity, technical accuracy, real-world application)
         |
         v
 keep clusters with mean score >= 6/7 (250 clusters) --> sample 2 each
         |
         v
 500 prompts x (model vs GPT-4-0314 baseline) x 2 orders = 1,000 judgments
         |
         v
 judge (GPT-4-Turbo): CoT, writes own answer first, then 5-way label
 A>>B, A>B, A~=B, B>A, B>>A
         |
         v
 Bradley-Terry fit over all comparisons, 100x bootstrap -> score + 95% CI
```

Design choices worth copying [9]:

- **Graded pairwise labels** (five classes) so that "a model will be penalized more in big losses than small losses," which helped separability.
- **Two-game swap**: every query is judged in both orders to cancel position bias.
- **Bradley-Terry with bootstrap** instead of raw win rate against the baseline, which proved "statistically stable."
- **New meta-metrics.** *Agreement with confidence* scores +1 when the benchmark confidently orders a pair the same way as the human reference, -1 when it confidently disagrees, and 0 otherwise. *Separability* is the share of model pairs with non-overlapping CIs. The *Brier score* treats the benchmark as a forecaster of pairwise human outcomes.

Reported results (top-20 Arena models, April 2024) [9]:

| Benchmark | Prompts per model | Agreement w/ Arena (95% CI) | Spearman | Separability | Cost |
|---|---|---|---|---|---|
| MT-Bench | 160 | 26.1% | 91.3% | 22.6% | - |
| AlpacaEval 2.0 LC | 800 | 81.2% | 90.8% | 83.2% | - |
| Arena-Hard-Auto v0.1 | 1,000 | 89.1% | 94.1% | 87.4% | about $25 |

The authors argue that Spearman "may be an unreliable metric for ranking correlation as it does not consider variance of the rankings" [9]. Practitioners who report only rank correlation between their judge and human labels should take note.

Arena-Hard-v2.0 (April 2025) added 500 new hard prompts and 250 creative-writing prompts. The official judge is Gemini-2.5, with GPT-4.1 as an alternative, and the recommended configuration for creative writing is an ensemble of both judges [11]. Style control became available in October 2024 (Section 4.2) [11].

### 3.4 Rubric grading at scale: HealthBench

**Problem.** Holistic scores in high-stakes domains hide which requirement failed, and a single "how good is this?" judgment mixes safety, accuracy, completeness and communication.

**Architecture.** OpenAI's HealthBench (2025) [45] attaches a conversation-specific rubric written by physicians to each multi-turn health conversation. The reference grader in `simple-evals` makes one judge call per rubric item, not one per response [45]:

```
 conversation + rubric items [(criterion_i, points_i, tags_i)]
          |
          +--> for each item i:  grader(conversation, criterion_i)
          |        -> {"explanation": ..., "criteria_met": true|false}
          v
 score = sum(points_i for met items) / sum(points_i where points_i > 0)
          |
          v
 example scores -> mean clipped to [0,1] -> bootstrap std (1,000 resamples)
          -> per-tag (theme / axis) breakdowns
```

Key techniques visible in the code [45]:

- **Binary per-criterion verdicts with negative points.** Undesirable criteria such as "Is overly verbose" or "fails to give the user accurate information" have negative points. The grader is told to report whether the criterion is *met*, not whether the response is good, so a correct response "does not meet" a negative criterion. This keeps the judge's task a factual check and moves valence into the scoring.
- **Conjunctive criteria with an exception for examples**: every sentence of an item must hold, except that "such as" or "for example" lists need not be exhaustive.
- **Strict JSON output** with an explanation field before the boolean.
- **A length-adjusted score helper** that subtracts a per-500-character penalty around a centre length, a direct guard against verbosity gaming.
- **Grader model**: `gpt-4.1-2025-04-14` by default. The repository also lets physicians' own ideal completions be scored against the rubrics, which provides a human reference ceiling.

**Lessons.** Rubric decomposition turns one subjective call into many narrower ones that are easier to audit and calibrate, and per-tag aggregation exposes regressions that a mean hides. The cost is roughly (number of rubric items) x (cost of one judge call) per example, so long rubrics multiply cost by an order of magnitude. Anthropic's 2026 guidance on agent evals reaches the same conclusion from the agent side: "an isolated LLM-as-judge to grade each dimension" rather than one judge for everything [26].

### 3.5 Rubrics as training inputs: Prometheus-style score descriptions

Prometheus (KAIST and collaborators) [28] standardised a rubric format that many fine-tuned judges now accept: an instruction, the response, a reference answer that "gets a score of 5," and a score rubric with a criterion question plus `score1_description` to `score5_description`. The output must be "Feedback: ... [RESULT] (an integer number between 1 and 5)" [28][30]. Making the per-level descriptions explicit reduces scale ambiguity, the main weakness of pointwise grading. Section 7 covers it as a fine-tuned judge.

## 4. Biases: catalogue, measurement and mitigation

### 4.1 Position bias

**Evidence.** Wang et al. ("Large Language Models are not Fair Evaluators," 2023) showed "severe positional bias" in ChatGPT and GPT-4 judges. In their headline example, Vicuna-13B could be made to beat ChatGPT on most of 80 Vicuna-bench questions (about 66 of 80, approx.) simply by changing the presentation order [13]. Shi et al. (2024) studied position bias systematically across judges and tasks. They measured repetition stability, position consistency and preference fairness, and found the bias is strongest when the two answers are close in quality [16].

**Mitigations.**

- **Balanced Position Calibration (BPC)**: evaluate both orders and average the scores [13].
- **Multiple Evidence Calibration (MEC)**: sample k chain-of-thought evaluations (the repo default is k=3) and aggregate [13].
- **Two-game swap with conflict handling**: Arena-Hard runs both orders and counts both games [9]. JudgeBench runs each pair "twice with different response orderings" by default, and offers `--single_game` only for order-independent reward models [19].
- **Swap augmentation during fine-tuning**: JudgeLM trains on order-swapped copies [32]. Self-Taught Evaluator generates synthetic judgments in both orders and keeps label-balanced samples [35].
- **Report a position-consistency metric**: LLMBar's tooling logs "positional agreement" next to accuracy [18], and Auto-J reports a swap-consistency score of 83.41 against GPT-4's 86.28 [33].

### 4.2 Verbosity and style bias

**Evidence.** Most LLM annotators in AlpacaEval prefer the longer output 60-79% of the time [8]. In Arena-Hard's ablation, telling GPT-3.5-Turbo to answer "with as much detail as possible" raised its win rate against GPT-4-0314 from 23.2% to 29.9% under a GPT-4-Turbo judge, and from 28.0% to 40.8% under a Claude-3-Opus judge. A "chatty" prompt that also increased length barely moved either judge. The authors concluded that "output length is not the only factor," since detail may be preferred because it is more helpful [9].

**Mitigation 1: regression-based debiasing (LC AlpacaEval).** Fit a logistic regression that predicts the judge's preference from the model identity, the instruction, and the length difference against the baseline. Then set the length term to zero and average to get a *length-controlled win rate*. Constraints keep the win-rate semantics: win_rate(m1,m2) = 1 - win_rate(m2,m1) and win_rate(m,m) = 0.5 [7][8]. Reported effects: Spearman correlation with Chatbot Arena rose from about 0.94 to 0.98, and "length gameability" fell from about 21% to about 6% [8]. A side note from the repo: switching to single-token outputs with logprob-based continuous preferences reduced length bias on its own [8].

**Mitigation 2: style control in Bradley-Terry.** LMSYS added style covariates to the Bradley-Terry logistic regression used for Arena scores, so each feature gets its own coefficient alongside the model strengths [12]. Each feature is a normalised relative difference, normalize((f_A - f_B) / (f_A + f_B)), computed for token length and the counts of markdown headers, bold elements and list items [12]. On human votes this reshuffled the leaderboard. GPT-4o-mini and Grok-2-mini dropped, and Claude 3.5 Sonnet, Opus and Llama-3.1-405B rose [12]. Arena-Hard-Auto applies the same `--control-features markdown length` to judge outputs [11]. Humans have style bias too, so style control corrects the reference signal as well as the judge.

**Mitigation 3: explicit penalties in the rubric**, such as HealthBench's negative-point "overly verbose" criteria and its length-adjusted score helper [45].

### 4.3 Self-preference (self-enhancement) bias

**Evidence.** Panickssery, Bowman and Feng (2024) showed that GPT-4 and Llama 2 can recognise their own generations at better-than-chance rates and also rate them higher. When they fine-tuned models to change self-recognition ability, self-preference strength changed with it, which suggests a causal link [14]. The Arena-Hard comparison of GPT-4-Turbo and Claude-3-Opus as judges shows the effect in a deployed system. Under the Claude judge, Claude-3 Opus, Sonnet and Haiku scored 11.4 to 16.8 points higher, and GPT-4-0613 scored 7.3 points lower. LMSYS also flagged that "GPT-4-Turbo's high score can be due to the GPT-4 judge favoring GPT-4 outputs" [9].

**Mitigations.** Never let the judge family grade itself on comparative leaderboards. Inspect's `required=True` grader role enforces this [47]. Use panels drawn from several model families (Section 6). Use benchmarks whose responses come from several generators: JudgeBench ships pairs generated separately by GPT-4o and by Claude-3.5-Sonnet so self-bias can be measured [19].

### 4.4 Leniency, judge "personality" and differing perspectives

When the GPT-4-Turbo and Claude-3-Opus judges disagreed on Arena-Hard, LMSYS traced most disagreements to two causes [9]. The first was **conservative scoring**: Claude-3-Opus was "much less likely to give harsh scores" and treated small coding and math errors leniently. The second was **differing perspectives on the prompt**, for example preferring educational value over practicality. The two judges still had an 80% "soft agreement" rate, defined as verdicts at most one label apart [9]. Ragas reports the same leniency failure in a production-style alignment exercise: all 39 baseline misalignments were false passes [42]. Leniency is the default failure of prompted judges.

### 4.5 Broader bias catalogues and adversarial meta-evaluation

- **CALM** ("Justice or Prejudice?", 2024) proposes an automated framework that quantifies 12 bias types, among them position, verbosity, self-enhancement, authority, bandwagon, sentiment, distraction, and chain-of-thought effects. It perturbs inputs and measures how often verdicts flip [15].
- **EvalBiasBench and OffsetBias** (NCSOFT, 2024) build a meta-evaluation set of bias-inducing pairs and a debiased training set ("OffsetBias data") for tuning judges [17].
- **LLMBar** (Princeton, ICLR 2024) has 419 instances with an objectively better output. It includes a Natural set and Adversarial subsets (Neighbor, GPTInst, GPTOut, Manual) built so that the worse output looks better on the surface, and it ablates prompting strategies such as Rules, Metrics, Swap and CoT [18].
- **JudgeBench** (2024) labels pairs by objective correctness on knowledge, reasoning, math and coding questions, for example from MMLU-Pro. The authors report that strong prompted judges such as GPT-4o do only slightly better than random guessing on it [19]. This is the "limited reasoning ability" limitation from MT-Bench, at frontier difficulty.

The lesson from these benchmarks is that agreement on natural chat data overstates judge quality. Every new judge or prompt should also be tested on adversarial and objectively labelled pairs.

## 5. Calibrating judges against humans

### 5.1 What to measure

| Metric | When to use | Source / example |
|---|---|---|
| Raw % agreement (with or without ties) | Pairwise preference; compare with the human-human ceiling | MT-Bench >80% [2]; AlpacaEval 69.2% vs 65.7% human [8] |
| Cohen's kappa | Binary or categorical labels with class imbalance | EvalGen reports kappa [40] |
| MCC | Binary pass/fail; robust to imbalance | EvalGen ranks candidate evaluators by MCC [40] |
| Failure coverage (recall on human fails) | When missing a real failure is costly | EvalGen [40] |
| False failure rate | When false alarms erode trust | EvalGen [40] |
| Spearman / Pearson over systems | System-level ranking validity | AlpacaEval, Prometheus [8][30] |
| Agreement / separability with CIs; Brier | Leaderboards where top-model ordering matters | Arena-Hard [9] |
| Swap-consistency | Pairwise judges | Auto-J, LLMBar [33][18] |

Instance-level agreement and system-level correlation answer different questions. A judge can rank systems correctly (high Spearman) while being wrong on many individual items. The opposite also happens: high item accuracy that comes mostly from easy items can still fail to separate close systems [9].

### 5.2 Error bars and variance

Judges are stochastic even at temperature 0. LMSYS found that GPT-4-Turbo "may still generate slightly different judgments" and recommends confidence intervals over point scores [9]. Anthropic's statistical guidance applies directly to judge-scored evals [25]:

- Report the standard error of the mean and 95% CIs. **Cluster** standard errors when items share a source; clustered SEs on popular evals were "over three times larger" than naive ones.
- Reduce within-item variance by **resampling** the judgment several times and averaging per item, or by using **token probabilities** instead of a sampled label. This is the same idea as G-Eval's weighting.
- Compare models with **paired differences** on the same items. Per-item score correlations between frontier models of about 0.3 to 0.7 make this a free variance reduction.
- Use **power analysis** to choose item counts and resample counts before running.

HealthBench follows this pattern, reporting a bootstrap standard deviation over 1,000 resamples for every score [45].

### 5.3 Selective judging with guarantees: Trust or Escalate

Jung, Brahman and Choi (2024) treat judge reliability as a selective-prediction problem [23]. The judge estimates its own confidence, here via "simulated annotators," that is, few-shot prompts imitating several distinct human annotators. A threshold is calibrated on a held-out human-labelled split so that, on the cases the judge does not abstain on, agreement with humans meets a target with high probability. **Cascaded Selective Evaluation** chains judges from cheap to expensive, escalating only when the cheaper judge abstains [23]:

```
 item --> [Mistral-7B judge] --conf >= t1--> accept verdict
              | abstain
              v
          [GPT-3.5 judge] ----conf >= t2--> accept verdict
              | abstain
              v
          [GPT-4-Turbo judge] -conf >= t3--> accept verdict
              | abstain
              v
          escalate to human
 thresholds t_i calibrated on a held-out human-labelled split
 (reference config: calibration set 500, alpha = 0.15, delta = 0.1)
```

The reference implementation calibrates on a 500-item split with risk level alpha = 0.15 and error tolerance delta = 0.1, and ships pre-generated judgments for all three judges [23]. Design lesson: a judge that may say "I'm not sure" is more useful than one that must always answer. Anthropic's agent-eval guidance gives the same advice in simpler form: give the judge "a way out," such as an "Unknown" option, to reduce hallucinated verdicts [26].

### 5.4 The statistical ceiling: judges do not replace labels

Dorner et al. ("Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data," 2024) analyse debiasing methods that combine many judge labels with a few human labels, such as prediction-powered inference [24]. Their result, as the title states, is that when the judge is no more accurate than the model being evaluated, these methods can at best save about a factor of two in human labels [24]. At the frontier, where the model under test may be stronger than the judge, human labels remain the binding constraint. Judges extend human labels; they do not replace them.

## 6. Judge ensembles, juries and multi-judge designs

### 6.1 PoLL: replacing judges with juries

**Problem.** A single large judge is expensive and brings its own family's biases, including self-preference.

**Design.** Verga et al. (Cohere, 2024) proposed a **Panel of LLM evaluators (PoLL)**: several smaller judges from *different model families*. The reported panel was Command R, Claude 3 Haiku and GPT-3.5 (approx.). Their verdicts are pooled by voting for categorical outputs or averaging for scores [22]. Across single-hop QA, multi-hop QA and Chatbot Arena-style settings on six datasets, the panel correlated better with human judgments than a single GPT-4 judge, showed less intra-model bias, and cost about seven times less (approx.) [22].

```
             +--> judge_1 (family X, small) --+
 item -------+--> judge_2 (family Y, small) --+--> vote / mean --> verdict
             +--> judge_3 (family Z, small) --+
```

**Why it works.** Errors that are uncorrelated across families partly cancel. Self-preference is diluted when no single family dominates the panel. Small models are cheap enough that three calls cost less than one frontier call.

### 6.2 Other ensemble patterns in production tooling

- **Cross-vendor pairs for subjective tasks**: Arena-Hard-v2.0 recommends GPT-4.1 + Gemini-2.5 as the best configuration for creative writing. Each judge runs independently and results are aggregated at display time [11].
- **Harness-level reducers**: Inspect lets a scorer take a list of grader models whose grades are combined by a reducer [47]. Anthropic lists "multi-judge consensus" as a standard model-based grading method [26].
- **Self-consistency within one judge**: MEC (k CoT samples) [13], and resample-and-average per item [25].
- **Cascades rather than panels**, when cost matters more than bias dilution (Section 5.3) [23].

**Failure modes.** Ensembles of correlated judges, for example several models distilled from the same teacher, add cost without diversity. Majority vote also hides minority judges that are right on hard items. JudgeBench-style objective pairs are the best test of whether a panel actually beats its best member [19].

## 7. Fine-tuned and specialised judges

Prompted frontier judges are strong but expensive, closed, version-unstable and hard to reproduce. Fine-tuned judges trade some generality for cost, privacy and reproducibility. The main lineage:

| Judge | Org / year | Base, sizes | Training signal | Protocols | Reported results (from source) |
|---|---|---|---|---|---|
| PandaLM | WeOpenML, 2023 | 7B | ~300K GPT-3.5-labelled pairs from Alpaca 52K | Pairwise + reason | F1 at 93.75% of GPT-3.5's and 88.28% of GPT-4's on ~1K human-labelled test set [31] |
| JudgeLM | BAAI, 2023 | 7B/13B/33B | 100K GPT-4 judgments; swap aug., reference support, reference drop | Pairwise, single, with or without reference | Agreement with GPT-4: 81.1-89.0 (no ref), 84.1-89.3 (ref); 7B judges 5K samples in 3 min on 8 A100 [32] |
| Auto-J | GAIR, ICLR 2024 | 13B (+6B bilingual) | 58 real-world scenarios; 3,436 pairwise + 960 single samples | Pairwise + single critique | Pairwise human agreement 54.96 vs GPT-4 62.28; critique win 73.7% vs ChatGPT refs [33] |
| Prometheus | KAIST et al., 2023 | 7B/13B | Feedback Collection: GPT-4 feedback over custom rubrics | Absolute (rubric 1-5) | Pearson with humans on par with GPT-4 on custom rubrics (approx. 0.9) [28] |
| Prometheus 2 | KAIST et al., 2024 | 7B, 8x7B | Feedback + Preference Collections; weight merging of absolute and pairwise models | Absolute + relative | Pearson ~0.6-0.7 with GPT-4-1106 on Likert; 72-85% human agreement on pairwise sets; 7B needs ~16GB VRAM [30] |
| M-Prometheus | 2025 | 3B/7B/14B | Multilingual | Absolute + relative | 7B/14B surpass Prometheus 2 7B/8x7B on RewardBench [30] |
| FLAMe | Google DeepMind, 2024 | PaLM-2-24B family | 100+ quality-assessment tasks, millions of human judgments (approx. 5M+) | Multi-task | Reported to outperform proprietary judges on many held-out tasks (approx.) [34] |
| Self-Taught Evaluator | Meta FAIR, 2024 | Llama-3.1-70B | Synthetic only: corrupted "worse response" pairs, both orders, rejection sampling, SFT + DPO | Pairwise | RewardBench 90.0 (Chat 96.9, Chat-Hard 85.1, Safety 89.6, Reasoning 88.4) [35] |
| EvalPlanner | Meta FAIR, 2025 | - | Plan-then-execute evaluation CoTs | Pairwise | "SOTA on RewardBench" per authors [37][35] |
| J1 | Meta FAIR, 2025 | 8B, 70B | RL (GRPO) on judgment tasks with verifiable rewards | Pairwise/pointwise | Outperforms EvalPlanner and distilled-R1 judges at 8B/70B per authors [36] |
| OffsetBias | NCSOFT, 2024 | Llama-3-8B | Debiased pairwise data | Pairwise | Evaluated on EvalBiasBench [17] |
| Selene Mini | Atla, 2025 | Llama-3.1-8B | Multi-task post-training | Absolute, classification, pairwise | Beats GPT-4o on RewardBench, EvalBiasBench and Auto-J per authors; "#1 8B generative model on RewardBench" [38] |

### 7.1 Design patterns across fine-tuned judges

1. **Distil a strong teacher's judgments** (PandaLM, JudgeLM, Prometheus 1): cheap and effective, but the student inherits the teacher's biases, and agreement is often reported against the teacher rather than against humans (JudgeLM's headline metric is agreement with GPT-4 [32]).
2. **Train in both orders, with and without references** (JudgeLM's swap augmentation and reference drop [32]) so the judge is robust to position and works whether or not a reference is available.
3. **Unify protocols**: Prometheus 2 merges the weights of separately trained absolute and pairwise evaluators into one model, so one judge supports both modes, selected by prompt format [29][30].
4. **Remove humans and teachers from the label loop**: Self-Taught Evaluator gets labels for free by constructing a degraded "worse response" for each instruction. It generates evaluation plans in both orders, keeps only plans that prefer the original, balances A/B labels, and iterates [35]. One caveat from its README: the released model "was trained to judge a pair of responses using the specific prompt format from the RewardBench benchmark," so prompt-format coupling is a real deployment risk [35].
5. **Reward the reasoning, not just the label**: EvalPlanner and J1 train judges to produce evaluation plans or chains of thought, using preference optimisation or reinforcement learning against verifiable correctness signals [36][37]. This responds to JudgeBench-style failures on hard reasoning pairs.
6. **Train against known biases**: OffsetBias builds training pairs that specifically neutralise length and surface-feature shortcuts [17].

### 7.2 Meta-evaluation benchmarks for judges

- **RewardBench** (AI2, 2024) evaluates reward models and generative judges on chosen/rejected pairs in Chat, Chat-Hard, Safety and Reasoning [20]. **RewardBench 2** (2025) moves to best-of-4 ranking and a "Ties" subset with up to 20+ completions per prompt. Generative judges can rank all four in one prompt or rate each separately and then recombine [21].
- **JudgeBench**, **LLMBar**, **EvalBiasBench** and **MT-Bench human judgments** (Section 4.5, [3][17][18][19]).

Judges that top chat-preference benchmarks can still fail on objective-correctness pairs. Choose the meta-benchmark that matches the deployment domain.

## 8. Aligning judges with human intent

A judge prompt is a specification. Most production failures come not from the judge model but from criteria that do not match what the domain expert actually cares about. Since 2024 several workflows have treated judge construction as an iterative, human-in-the-loop alignment process.

### 8.1 EvalGen: "Who validates the validators?"

**Problem.** Shankar, Zamfirescu-Pereira, Hartmann, Parameswaran and Arawjo (UIST 2024) studied how developers build LLM-based evaluators for LLM pipelines [39]. Their central finding is **criteria drift**: users need criteria to grade outputs, but grading outputs is how users discover and refine their criteria. Some criteria cannot be specified before seeing outputs. A judge written up front will therefore be misaligned. The paper's qualitative study involved a small group of practitioners (approx. nine) [39].

**Architecture** (from the open-source implementation in ChainForge, which credits Shankar for the module) [40]:

```
 developer prompt + sample (vars, prompt, response) triples
          |
          v
 [LLM] suggest evaluation criteria  --> developer edits / confirms
          |
          v
 per criterion: [LLM AssertionEmitter] generate multiple candidate
 evaluators (code assertions or LLM-judge prompts), streamed
          |
          v
 Executor (background): run every candidate on every example;
 update each example's grading priority from candidates' pass rates
          |
          v
 developer grades examples thumbs up/down, choosing next example by
 "priority" policy (or random) to get a mix of good and bad outputs
          |
          v
 per candidate: confusion matrix vs. developer grades ->
   F1, MCC, Cohen's kappa, failure coverage, false-failure rate
          |
          v
 select best candidate per criterion (sorted by MCC, then lower FFR)
 -> report card: coverage of failures, false failures, chosen set
```

**Key techniques** [40]:

- **Generate many candidate evaluators per criterion and select empirically** rather than trusting one judge prompt.
- **Active selection of what the human grades**, so a few labels cover both failing and passing examples.
- **Asymmetric metrics**: *failure coverage* = true fails / (true fails + false passes), and *false failure rate* = false fails / (true passes + false fails). These make the trade-off between missing bugs and raising false alarms explicit.

**Lessons.** Evaluator alignment is a loop, not a one-off step. Criteria should be versioned, because they will change once people look at outputs. The predecessor work SPADE [51] synthesises assertion sets from prompt-version histories under similar coverage and false-failure constraints.

### 8.2 Critique shadowing and the "principal domain expert"

Practitioner Hamel Husain's widely cited 2024 guide [41] describes a workflow often called **critique shadowing**:

1. Identify one **principal domain expert** whose judgment defines quality.
2. Have that expert label a diverse sample with **binary pass/fail** plus a written critique explaining why.
3. Build the judge prompt using those critiques as few-shot examples.
4. Measure agreement with the expert and iterate on the prompt until it plateaus.
5. Treat the judge as a way to scale the expert's error analysis, not as an end in itself.

Pass/fail plus critique replaces Likert scales because binary decisions force clarity and critiques carry the reasoning a judge can learn from. Ragas's 2025 alignment guide follows the same approach, telling teams to "involve a principal domain expert" and warning that "the true business value comes from systematically analyzing your data" [42].

**Worked numbers (Ragas, EvalsBench, gpt-4o-mini judge)** [42]. A baseline judge that checks responses against expert grading notes agreed with expert pass/fail labels on 121 of 160 items (75.6%). All 39 disagreements were false passes, meaning the judge was too lenient. After error analysis, a v2 prompt that counts the required concepts in the grading notes, with explicit tolerance for approximate numbers, reached 139 of 160 (86.9%). The guide's stopping rule: iterate until alignment "plateaus across 2-3 iterations or meets your business threshold" [42].

### 8.3 Automated judge-prompt optimisation: MLflow align()

MLflow (2025-2026) turns judge alignment into an API. A judge created with `make_judge` is aligned to human assessments attached to traces under the same name, and several optimisers are available [43]:

- **MemAlign** (default, experimental) uses a dual memory. *Semantic memory* holds distilled guidelines extracted from expert rationales, such as "Always evaluate safety based on intent, not just language". *Episodic memory* holds specific edge cases where the judge was wrong. At inference it builds context from all principles plus the most relevant retrieved examples, "similar to how human judges reference both a rulebook and case history." MLflow claims it is up to 100x faster than SIMBA and shows improvement with only a handful of examples [43].
- **SIMBA** (DSPy's Simplified Multi-Bootstrap Aggregation) and **GEPA** (DSPy's Genetic-Pareto reflective optimiser) rewrite judge instructions from feedback. Both require at least 10 human-assessed traces and recommend a mix of positive and negative labels plus natural-language rationales [43].
- MLflow's documentation claims aligned judges show a "30-50% reduction in false positives/negatives compared to generic evaluation prompts." This is a vendor claim without a published protocol, so treat it as approximate [43].

Eugene Yan's ALIGN Eval prototype packages the same loop (label, evaluate the judge prompt against labels, semi-automatically optimise) as a game for building binary LLM evaluators [44].

### 8.4 Lab guidance on judge alignment

Anthropic's January 2026 engineering guidance on agent evals [26]:

- Use code-based graders where possible and model graders where deterministic checks fall short. Model graders are "non-deterministic," costlier than code, and require "calibration with human graders."
- Write structured rubrics and use an isolated judge per dimension. Vague rubrics produce inconsistent judgments.
- Grade outcomes rather than the exact path an agent took, and allow partial credit.
- Read transcripts. A 0% pass rate across many trials usually means a broken task or grader. In one case, fixing grading and task bugs moved a model's CORE-Bench score from 42% to about 95%.
- Recalibrate rubrics against experts often for subjective tasks such as research agents.

Earlier Anthropic writing on model-generated evaluations warns that model-written or model-graded evals "can contain social biases and can fabricate information" and still need human verification [27].

## 9. Beyond single responses: agent-as-a-judge

When the artefact being judged is an agent's workspace or trajectory, a single-pass judge reading the final answer misses intermediate failures. **Agent-as-a-Judge** (ICML 2025) gives the judge agentic tools to inspect a workspace, locate relevant files and check each requirement of a development task (the DevAI benchmark). The authors report about 97.7% time savings and 97.6% cost savings against human expert evaluation, and position its step-level feedback as a reward signal for agent training [49]. openevals ships "trajectory LLM-as-judge" prompts for the lighter case of judging a tool-call sequence [48]. Chapter-level coverage of agent evaluation belongs elsewhere in this atlas. The point here is that the judge architecture grows from one prompt to an agent loop as the evidence to examine grows.

## 10. A reference architecture for a production judge service

Putting the patterns together, a production-grade judge pipeline looks like this:

```
             +------------------- offline (alignment) --------------------+
             |  expert labels + critiques (binary, principal expert)       |
             |        |                                                    |
             |        v                                                    |
             |  criteria elicitation (EvalGen-style) <--- criteria drift   |
             |        |                                                    |
             |        v                                                    |
             |  candidate judges (prompt variants / models / fine-tunes)   |
             |        |                                                    |
             |        v                                                    |
             |  meta-eval: MCC, kappa, failure coverage, FFR, swap-        |
             |  consistency, adversarial set (LLMBar/JudgeBench-like)      |
             |        |                                                    |
             |        v                                                    |
             |  optimise (few-shot, MemAlign/GEPA) -> freeze + version     |
             +--------+---------------------------------------------------+
                      |
 online / CI          v
 outputs --> per-criterion judges (isolated; rubric items; CoT -> JSON)
               |  pairwise? -> both orders; cross-family panel if comparative
               v
          aggregate: rubric points, vote/mean, Bradley-Terry + style
          covariates, clustered SEs / bootstrap CIs
               |
               +--> low confidence / abstain --> escalate (bigger judge, human)
               v
          dashboards + sampled human audit --> drift alarms --> back to offline
```

Cost and latency rules of thumb from the sources:

- Pairwise designs double judge calls per item when swapping orders (Arena-Hard: 500 prompts become 1,000 judgments per model, about $25 per evaluation with GPT-4-Turbo in 2024) [9].
- Rubric grading multiplies calls by the number of rubric items [45].
- Small fine-tuned judges cut marginal cost to GPU time: JudgeLM-7B judged 5K samples in about 3 minutes on 8 A100s [32], and Prometheus 2 7B fits in about 16GB of VRAM [30].
- Panels of small models can cost less than one frontier judge [22].
- AlpacaEval 2.0 with LC runs for under $10 in under 3 minutes [8].

## Key takeaways for system designers

1. **Validate the judge before trusting it.** Collect a few hundred expert labels and report agreement against the human-human ceiling. Use kappa or MCC for binary labels, plus failure coverage and false-failure rate. Spearman over systems alone is not enough [9][40].
2. **Prefer binary, per-criterion judgments over holistic Likert scores.** Decompose into rubric items with explicit points (negative points for undesirable behaviour) and an isolated judge per dimension. This is easier to calibrate, audit and slice by tag [26][45][41].
3. **Always run pairwise judgments in both orders** and log swap-consistency. Treat inconsistent pairs as ties or abstentions [9][13][19].
4. **Control length and style statistically, not just in the prompt.** Add length and markdown covariates to the preference model (LC win rate, style-controlled Bradley-Terry), or add explicit length penalties to rubrics [7][12][45].
5. **Never let a model family grade itself on comparative leaderboards.** Use a cross-family judge or a panel, and enforce a separate grader role in the harness [9][14][22][47].
6. **Use references wherever correctness is checkable.** Reference-guided grading sharply reduces judge failures on math and reasoning, and objective-correctness benchmarks show unguided judges near chance on hard pairs [1][19].
7. **Report uncertainty.** Use bootstrap or clustered CIs, paired differences between models, and repeated judge sampling or logprob-weighted scores to cut judge variance [25][4].
8. **Let the judge abstain, and cascade.** Calibrated abstention with escalation to stronger judges or humans gives guaranteed agreement on the cases it accepts, at lower cost [23][26].
9. **Expect criteria drift.** Build the judge through an EvalGen or critique-shadowing loop with a principal domain expert, version the criteria, and re-align when the product or user base changes [39][41][42][43].
10. **Fine-tuned judges are viable for scale, privacy and reproducibility**, but check prompt-format coupling, teacher-inherited biases, and performance on your domain rather than only on RewardBench [32][35][30][38].
11. **Judges extend human labels; they do not replace them.** When the judge is no stronger than the model under test, judge-plus-debiasing saves at most about half the human labels. Keep a sampled human audit in the loop [24].
12. **Read the transcripts.** Many apparent model failures are grader bugs, and a judge that has not been spot-checked by a human is not yet a measurement [26].

## References

1. Zheng, L., Chiang, W.-L., Sheng, Y., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." LMSYS / UC Berkeley, 2023. https://arxiv.org/abs/2306.05685
2. Zheng, L., Chiang, W.-L., Sheng, Y., Zhang, H. "Chatbot Arena Leaderboard Week 8: Introducing MT-Bench and Vicuna-33B." LMSYS blog, 2023. https://lmsys.org/blog/2023-06-22-leaderboard/
3. LMSYS. "LLM Judge (MT-Bench) README," FastChat repository, 2023. https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md
4. Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., Zhu, C. "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment." Microsoft, 2023. https://arxiv.org/abs/2303.16634 (code: https://github.com/nlpyang/geval)
5. Confident AI. "G-Eval" metric documentation, DeepEval repository, 2025-2026. https://github.com/confident-ai/deepeval/blob/main/docs/content/docs/(custom)/metrics-llm-evals.mdx
6. Confident AI. "DAG (Deep Acyclic Graph)" metric documentation, DeepEval repository, 2025-2026. https://github.com/confident-ai/deepeval/blob/main/docs/content/docs/(custom)/metrics-dag.mdx
7. Dubois, Y., Galambosi, B., Liang, P., Hashimoto, T. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." Stanford, 2024. https://arxiv.org/abs/2404.04475
8. Tatsu Lab. "AlpacaEval" repository (annotator meta-evaluation table, LC win rates), 2023-2025. https://github.com/tatsu-lab/alpaca_eval
9. Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Zhu, B., Gonzalez, J., Stoica, I. "From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline." LMSYS blog, 2024. https://lmsys.org/blog/2024-04-19-arena-hard/
10. Li, T., et al. "From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline." 2024. https://arxiv.org/abs/2406.11939
11. LMArena. "Arena-Hard-Auto" repository (v2.0, style control, ensemble judges), 2024-2025. https://github.com/lmarena/arena-hard-auto
12. Li, T., Angelopoulos, A., Chiang, W.-L. "Does Style Matter? Disentangling Style and Substance in Chatbot Arena." LMSYS blog, 2024. https://lmsys.org/blog/2024-08-28-style-control/
13. Wang, P., Li, L., Chen, L., et al. "Large Language Models are not Fair Evaluators." Peking University et al., 2023. https://arxiv.org/abs/2305.17926 (code: https://github.com/i-Eval/FairEval)
14. Panickssery, A., Bowman, S. R., Feng, S. "LLM Evaluators Recognize and Favor Their Own Generations." 2024. https://arxiv.org/abs/2404.13076
15. Ye, J., et al. "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge." 2024. https://arxiv.org/abs/2410.02736
16. Shi, L., et al. "Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs." 2024. https://arxiv.org/abs/2406.07791
17. Park, J., Jwa, S., Ren, M., Kim, D., Choi, S. "OffsetBias: Leveraging Debiased Data for Tuning Evaluators." NCSOFT, 2024. https://arxiv.org/abs/2407.06551 (code: https://github.com/ncsoft/offsetbias)
18. Zeng, Z., et al. "Evaluating Large Language Models at Evaluating Instruction Following" (LLMBar). Princeton, ICLR 2024. https://arxiv.org/abs/2310.07641 (code: https://github.com/princeton-nlp/LLMBar)
19. Tan, S., et al. "JudgeBench: A Benchmark for Evaluating LLM-Based Judges." UC Berkeley, 2024. https://arxiv.org/abs/2410.12784 (code: https://github.com/ScalerLab/JudgeBench)
20. Lambert, N., et al. "RewardBench: Evaluating Reward Models for Language Modeling." AI2, 2024. https://arxiv.org/abs/2403.13787
21. AI2. "RewardBench 2" paper and repository, 2025. https://arxiv.org/abs/2506.01937 ; https://github.com/allenai/reward-bench
22. Verga, P., et al. "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models." Cohere, 2024. https://arxiv.org/abs/2404.18796
23. Jung, J., Brahman, F., Choi, Y. "Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement." 2024. https://arxiv.org/abs/2407.18370 (code: https://github.com/jaehunjung1/cascaded-selective-evaluation)
24. Dorner, F. E., et al. "Limits to Scalable Evaluation at the Frontier: LLM as Judge Won't Beat Twice the Data." 2024. https://arxiv.org/abs/2410.13341
25. Anthropic. "A Statistical Approach to Model Evaluations." 2024. https://www.anthropic.com/research/statistical-approach-to-model-evals
26. Grace, M., Hadfield, J., Olivares, R., De Jonghe, J. "Demystifying Evals for AI Agents." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
27. Ganguli, D., Schiefer, N., Favaro, M., Clark, J. "Challenges in Evaluating AI Systems." Anthropic, 2023. https://www.anthropic.com/research/evaluating-ai-systems
28. Kim, S., Shin, J., Cho, Y., et al. "Prometheus: Inducing Fine-grained Evaluation Capability in Language Models." KAIST et al., 2023. https://arxiv.org/abs/2310.08491 (code: https://github.com/kaistAI/prometheus)
29. Kim, S., Suk, J., Longpre, S., et al. "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." 2024. https://arxiv.org/abs/2405.01535
30. Prometheus-Eval. "prometheus-eval" repository (Prometheus 2, BGB, M-Prometheus), 2024-2025. https://github.com/prometheus-eval/prometheus-eval
31. Wang, Y., et al. "PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization." ICLR 2024. https://arxiv.org/abs/2306.05087 (code: https://github.com/WeOpenML/PandaLM)
32. Zhu, L., Wang, X., Wang, X. "JudgeLM: Fine-tuned Large Language Models are Scalable Judges." BAAI, 2023. https://arxiv.org/abs/2310.17631 (code: https://github.com/baaivision/JudgeLM)
33. Li, J., et al. "Generative Judge for Evaluating Alignment" (Auto-J). GAIR, ICLR 2024. https://arxiv.org/abs/2310.05470 (code: https://github.com/GAIR-NLP/auto-j)
34. Vu, T., et al. "Foundational Autoraters: Taming Large Language Models for Better Automatic Evaluation" (FLAMe). Google DeepMind, 2024. https://arxiv.org/abs/2407.10817
35. Wang, T., et al. "Self-Taught Evaluators." Meta FAIR, 2024. https://arxiv.org/abs/2408.02666 (release notes: https://github.com/facebookresearch/RAM/tree/main/projects/self_taught_evaluator)
36. Whitehouse, C., et al. "J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning." Meta FAIR, 2025. https://arxiv.org/abs/2505.10320
37. Saha, S., et al. "Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge" (EvalPlanner). Meta FAIR, 2025. https://arxiv.org/abs/2501.18099
38. Atla AI. "Selene Mini" repository and technical report, 2025. https://github.com/atla-ai/selene-mini
39. Shankar, S., Zamfirescu-Pereira, J. D., Hartmann, B., Parameswaran, A., Arawjo, I. "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences." UIST 2024. https://arxiv.org/abs/2404.12272 (ACM: https://dl.acm.org/doi/abs/10.1145/3654777.3676450)
40. Shankar, S., Arawjo, I. "evalgen" module (executor, assertion emitter, alignment metrics), ChainForge repository, 2024. https://github.com/ianarawjo/ChainForge/tree/main/chainforge/react-server/src/backend/evalgen
41. Husain, H. "Creating a LLM-as-a-Judge That Drives Business Results." hamel.dev, 2024. https://hamel.dev/blog/posts/llm-judge/
42. Ragas. "How to Align an LLM as a Judge" guide, Ragas repository, 2025. https://github.com/explodinggradients/ragas/blob/main/docs/howtos/applications/align-llm-as-judge.md
43. MLflow. "Judge Alignment," "MemAlign," "SIMBA," and "GEPA" documentation, MLflow repository, 2025-2026. https://github.com/mlflow/mlflow/tree/master/docs/docs/genai/eval-monitor/scorers/llm-judge
44. Yan, E. "ALIGN Eval" (align-app) repository, 2024; and "Evaluating the Effectiveness of LLM-Evaluators (aka LLM-as-Judge)," 2024. https://github.com/eugeneyan/align-app ; https://eugeneyan.com/writing/llm-evaluators/
45. Arora, R. K., et al. (OpenAI). "HealthBench: Evaluating Large Language Models Towards Improved Human Health," 2025, and reference grader `healthbench_eval.py`. https://arxiv.org/abs/2505.08775 ; https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
46. OpenAI. "Eval Templates" (model-graded evals), openai/evals repository, 2023. https://github.com/openai/evals/blob/main/docs/eval-templates.md
47. UK AI Security Institute. "Model Grading," Inspect AI documentation, 2025-2026. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/model-graded.qmd
48. LangChain. "openevals" repository (LLM-as-judge, trajectory judges), 2025. https://github.com/langchain-ai/openevals
49. Zhuge, M., et al. "Agent-as-a-Judge: Evaluate Agents with Agents." ICML 2025. https://arxiv.org/abs/2410.10934 (code: https://github.com/metauto-ai/agent-as-a-judge)
50. Chiang, W.-L., et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." 2024. https://arxiv.org/abs/2403.04132
51. Shankar, S., et al. "SPADE: Synthesizing Data Quality Assertions for Large Language Model Pipelines." 2024. https://arxiv.org/abs/2401.03038
52. Ye, S., et al. "FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets." KAIST, 2023. https://arxiv.org/abs/2307.10928
53. Lee, Y., et al. "CheckEval: A Reliable LLM-as-a-Judge Framework for Evaluating Text Generation Using Checklists." 2024. https://arxiv.org/abs/2403.18771
54. Li, H., et al. "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods." 2024. https://arxiv.org/abs/2412.05579
