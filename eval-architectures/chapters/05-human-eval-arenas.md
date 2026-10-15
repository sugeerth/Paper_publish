# 05. Human Evaluation and Arenas: Pairwise Preference Platforms, Their Statistics, Their Critics, and Expert Annotation Pipelines

## Overview

Chapters 02 and 03 dealt with benchmarks that have a ground truth: a multiple-choice key, a unit test, an exact-match string. This chapter covers what is left when no such key exists. Open-ended chat, creative writing, coding help inside an IDE, search answers with citations, and agent behavior all need human judgment somewhere in the measurement loop. The field has settled on two main architectures for collecting that judgment at scale:

1. **Crowdsourced arenas.** Anonymous users submit their own prompts, see two anonymous model responses side by side, and vote. A paired-comparison model (Elo, later Bradley-Terry) turns the votes into a leaderboard. Chatbot Arena, now LMArena, is the reference design. It has many offshoots: Copilot Arena, WebDev Arena, Search Arena, RedTeam Arena, and Prompt-to-Leaderboard (P2L).
2. **Curated expert pipelines.** Screened and trained annotators (contractors, graduate students, domain experts) label model outputs against written guidelines, often on private or held-out prompt sets. Examples include the InstructGPT and Summarize-from-Feedback labeling programs at OpenAI, Anthropic's HH-RLHF crowdworker pipeline, the MT-Bench expert judgments, and Scale AI's SEAL leaderboards.

Between these two sit **preference datasets designed for research**, such as AlpacaFarm, PRISM and UltraFeedback. They make explicit choices about who rates, on what scale, with what redundancy, and how agreement is measured. Around all of it sits a growing body of **critique**: style bias, sampling and disclosure asymmetries ("The Leaderboard Illusion"), vote rigging, and low inter-annotator agreement.

The chapter is organized as follows:

- the Chatbot Arena pipeline, end to end;
- its statistical core (Bradley-Terry, bootstrap confidence intervals, anchoring, reweighting);
- confounder control (style and sentiment control) and prompt-distribution engineering (deduplication, categories, freshness);
- the Leaderboard Illusion critique and LMArena's response;
- vertical arenas and the derived artifacts (Arena-Hard, PPE, P2L);
- expert annotation pipelines and Scale SEAL;
- inter-annotator agreement as a design quantity;
- a design checklist for preference data collection.

A note on sourcing: during the research for this chapter, live web search and most hosts were unavailable. Primary sources were read from the canonical GitHub-hosted source of the LMArena blog, the FastChat rating code, and official READMEs of the cited projects. Figures that come from papers which could not be re-fetched (marked "reported") are given as the papers state them, with approximate qualifiers where appropriate.

---

## 1. Chatbot Arena / LMArena: the reference arena architecture

### 1.1 Problem

In spring 2023 the open-source community was releasing a new instruction-tuned LLaMA derivative almost every week. Static benchmarks such as HELM and lm-evaluation-harness measured close-ended tasks and could not rank open-ended assistants. The LMSYS launch post set out three properties a pairwise benchmark system should have [2]:

- **Scalability:** it works with many models when full pairwise coverage is infeasible.
- **Incrementality:** a new model can be rated from relatively few trials.
- **Unique order:** every pair of models is either ordered or tied.

The answer was a crowdsourced "battle" platform built on FastChat, LMSYS's multi-model serving system. The initial leaderboard (April 24 to May 1, 2023) rested on 4.7K valid anonymous votes over nine open models, with Vicuna-13B on top at 1169 Elo [2].

### 1.2 Architecture and data flow

```
 user prompt (any language, any task)
        |
        v
+--------------------+    sampling weights     +----------------------------+
|  Battle scheduler  |<------------------------|  Policy: new models w=5    |
|  picks (A, B) pair |                         |  until 3k votes; top-10 w=3|
+---------+----------+                         |  best of top-10 providers  |
          |                                    +----------------------------+
          v
+--------------------+   two anonymous, position-randomized responses
|  Serving layer     |---------------------------------------------------+
|  (FastChat / APIs) |                                                   |
+--------------------+                                                   v
                                                           +---------------------+
                                                           | Vote UI: A / B /    |
                                                           | tie / both bad      |
                                                           +----------+----------+
                                                                      | identities
                                                                      | revealed
                                                                      v  after vote
+--------------------+   +------------------+   +-----------------------------------+
| Abuse defenses     |-->| Battle log       |-->| Offline analytics                 |
| CAPTCHA, IP vote   |   | (anony flag,     |   | dedup -> category labels ->       |
| caps, outlier user |   |  style metadata) |   | BT fit (+style) -> bootstrap CI   |
| tests, Cloudflare  |   +------------------+   | -> anchored scale -> leaderboard  |
+--------------------+                          +-----------------------------------+
```

The key mechanics, from the launch post and the policy page [2][5]:

- **Blindness.** Model names are hidden until the vote is cast. Only votes cast while names were hidden enter the leaderboard; the FastChat analysis code filters on an `anony` flag [7].
- **Vote options.** A wins, B wins, tie, or tie (both bad). In the Bradley-Terry fit, ties count as half a win and half a loss [3][6].
- **Sampling policy (2024 policy, updated 2025).** A new public model gets sampling weight 5 until it has 3,000 votes, then weight 1. The top-10 models by Arena Score get weight 3 "to ensure the best community experience." The best model from each of the top-10 providers gets weight 1 for diversity. A model is listed once its ranking converges, which means after 3,000 votes or earlier if its confidence interval is narrow enough. Models can be retired after 3,000 votes if two newer models exist in the same series, or if more than three providers offer cheaper-or-equal and strictly better models [5].
- **Pre-release (anonymous) testing.** Providers may test unreleased models under anonymous labels. They receive the rating privately along with "samples of up to 20% of the votes," and only publicly available models are listed [5].
- **Security.** Defenses include malicious-user detection (Section 5.1 of the Arena paper), Cloudflare bot protection, Google reCAPTCHA v3 on every chat and vote, per-IP daily vote limits, deduplication of the most frequent prompts (top 0.1%), and a delayed leaderboard release to allow manual review [5].

### 1.3 Scale

The scale has grown steadily:

| Date | Votes / battles | Models |
|------|-----------------|--------|
| May 2023 | 4.7K votes | 9 |
| Late May 2023 | 27K votes | — |
| Dec 2023 | 130K valid votes | 45+ |
| Mar 2024 | 800K votes | 90+ |

Sources for the table: [2][27][3][5]. The Chatbot Arena paper (March 2024) reports about 240K votes in its analysis snapshot, from a large and multilingual user base [1].

By April 2025, LMArena reported more than 400 public models evaluated across modalities (text, vision, text-to-image, WebDev), 300+ pre-release evaluations, 3M+ votes, "tens of millions" of battle pairings served, 1.5 million prompts released for open research, and roughly 41% of battles involving an open model [13]. In April 2025 the team announced it was starting a company to support the platform, with a commitment to stay "neutral, open, and accessible" [16].

### 1.4 Lessons from the first year

The week-4 update already stated the arena's central limitation. Small-scale user studies "often cannot generate many hard or medium prompts" that separate models on long-tail capability, and "for difficult questions, it is also very hard for regular Arena users to judge which LLM has generated a better answer" [27]. Most of what follows in this chapter (hard-prompt categories, Arena-Hard, expert pipelines, SEAL) responds to that observation.

---

## 2. The statistical core: from online Elo to Bradley-Terry

### 2.1 Why online Elo was abandoned

The launch leaderboard used the classic online Elo update, in which ratings change after each game in proportion to the surprise of the outcome [2]. LMSYS found "considerable variability in the ratings" depending on the K-factor and on game order. Their first fix was to compute Elo over 1,000 random permutations of the battle history. This stabilized the point estimates but produced confidence intervals that were too wide, since they were "effectively CI's for the original online Elo estimates" [3].

Two facts about LLMs undercut the reasons chess uses online Elo [3]:

1. The full game history is available centrally, so there is no need for a decentralized algorithm.
2. Most models are static. Hosted APIs are the exception: their behavior can drift.

### 2.2 Bradley-Terry as logistic regression

In December 2023 the leaderboard switched to the **Bradley-Terry (BT) model**, which is the maximum-likelihood estimate of the underlying Elo model under a fixed but unknown win rate [3]. The style-control post writes it as a logistic regression [4]:

```
For battle i, X_i in R^M has +1 at the left model, -1 at the right model.
Y_i = 1 if the left model wins, else 0.

beta_hat = argmin_beta  (1/n) * sum_i BCE( sigmoid(X_i . beta), Y_i )

Reported Arena Score = 400 * beta (natural-log scale converted via log(10))
                       + 1000, then shifted so an anchor model has a fixed value.
```

The production code in FastChat's `rating_systems.py` shows the details [6]:

- The loss is binary cross-entropy weighted by how often each unique (matchup, outcome) occurs. "This form naturally counts a draw as half a win and half a loss."
- Optimization uses L-BFGS-B with `alpha = ln(10)`, so coefficients are on the base-10 Elo scale.
- `scale_and_offset` multiplies by 400, adds 1000, and anchors `mixtral-8x7b-instruct-v0.1` at 1114. The anchor makes scores comparable across leaderboard snapshots. The offset itself is cosmetic.
- **Bootstrap confidence intervals** are computed efficiently. Rather than resampling battles, the code draws multinomial counts over the unique (matchup, outcome) rows, refits BT per round in parallel, and reports the 2.5% and 97.5% quantiles [6][7].
- The style-controlled variant ("contextual BT") adds feature coefficients and an L2 penalty (`reg=0.5`). A code comment explains that this matches scikit-learn's LogisticRegression default after removing an earlier row duplication [6].

The style-control post also notes that "in practice, we also reweight this objective to handle non-uniform model sampling" [4]. This matters because the sampling policy deliberately oversamples new and top models, so the battle distribution is not uniform over pairs.

### 2.3 Extensions

- **Extended BT / Extended Online Arena Score.** For compound systems, each "player" is a set of active subsystems (for example, model plus tool plus prompt). The feature vector carries a -1 or +1 per active subsystem, and the fit is regularized logistic regression with an L1 or L2 penalty, or online logistic regression when data arrives sequentially [12].
- **RedTeam Arena.** Each round of the "Bad Words" jailbreak game is treated as a 1v1 between a player and a (prompt, model) pair. Players, models and prompts each get one-hot blocks in a single regression, because naive Elo "would result in slow convergence" [23].
- **Prompt-to-Leaderboard (P2L).** An LLM is trained to map a prompt to a vector of BT coefficients, so every prompt gets its own leaderboard. This supports per-task evaluation, routing and personalization. The P2L authors report that a router built this way reached #1 on Chatbot Arena in January 2025, and that P2L's prompt-specific accuracy follows a power-law scaling [18].

### 2.4 Anomalous-user detection

`elo_analysis.py` implements a sequential test per user [7]:

1. For each of up to 100 votes by a user with at least 5 votes, it computes the empirical p-value of that vote under the pooled win/loss distribution for the same model pair.
2. It forms the running product `M = prod(1 / (2 p))` for both tails.
3. It flags the user once `M > 1/alpha`, with `alpha = 0.05`.

This is a likelihood-ratio or e-value style test that identifies users whose votes are consistently unlikely. A separate `limit_user_votes` helper keeps only the first N votes per judge per day [7].

### 2.5 Failure modes of the statistics

- **Non-stationary endpoints.** When GPT-4 versions were separated in December 2023, `gpt-4-0314` scored 1201 and `gpt-4-0613` scored 1152. LMSYS listed three competing hypotheses: a shift in the user distribution over time, missing comparisons against newer models, or a genuine preference difference [3]. The BT fit assumes stationarity, so pooling a drifting API under one name violates the model.
- **Overlapping CIs read as ranks.** Leaderboards display ordinal ranks, but adjacent models often have overlapping intervals. LMArena's response to the Leaderboard Illusion leans on exactly this point: scores of 1069 (plus or minus 27) and 1054 (plus or minus 18 to 22) are "within the expected statistical noise" [14].
- **Independence assumptions.** Bootstrap over battles treats votes as i.i.d. Repeat users, prompt templates and sample prompts on the landing page all violate this. WebDev Arena found that deduplicating votes caused by sample prompts cut the dataset from 103,096 to 61,473 votes [21].

---

## 3. Confounder control: style and sentiment

### 3.1 Style control (August 2024)

**Problem.** In mid-2024, GPT-4o-mini ranked surprisingly high and Claude models surprisingly low compared with practitioners' anecdotal experience. The hypothesis was that verbosity and markdown formatting were inflating scores [4].

**Technique.** Each battle gets a style feature vector Z_i, and the regression becomes `sigmoid(X_i . beta + Z_i . gamma)`. The four version-1 features are answer token length, markdown header count, bold count, and list count [4][6]. Each feature is a normalized relative difference:

```
feature_i = normalize( (f_A - f_B) / (f_A + f_B) )     # z-scored across battles
```

The ratio form makes 500 vs 520 tokens a small difference and 20 vs 40 tokens a large one. AlpacaEval's length-controlled variant instead uses `tanh((f_A - f_B) / sigma(f_A - f_B))` [4][28].

**Results.** Length was the dominant style coefficient: 0.249 when controlling for both length and markdown, against 0.031 for lists, 0.024 for headers and 0.019 for bold. With markdown controlled alone, the markdown coefficients rose (lists 0.111), which suggests markdown partly proxies for length [4]. Under style control, GPT-4o-mini and Grok-2-mini fell below most frontier models, while Claude 3.5 Sonnet, Claude 3 Opus and Llama-3.1-405B rose. On Hard Prompts, Claude 3.5 Sonnet tied for #1 [4].

**Caveat stated by the authors.** The analysis is observational. An unobserved confounder, such as chain-of-thought explanations that are both longer and more correct, can make style control penalize real quality. The authors call for causal inference and prospective randomized trials [4].

Style control was later made the default view in LMArena and was added to Arena-Hard-Auto in October 2024 [17].

### 3.2 Sentiment and emoji control (April 2025)

The feature set was extended with emoji count and a five-level tone label (very negative to very positive). The tone label is produced by Gemini-2.0-flash-001 using a classifier prompt that tells it to ignore the user's message and judge only the assistant's tone [11]. Controlling for style, sentiment and emoji usage moved models "known for strong stylistic and emotional appeal" such as Grok-3 and Llama-4-Maverick-Experimental down, and moved more neutral models such as Claude-3.7 up [11].

An ablation that kept only the emoji and sentiment features found that positive sentiment keeps a strong positive effect while neutral and negative tones are penalized. In raw conditional win rates, a "very positive" response beat a "negative" one 78.3% of the time [11].

### 3.3 Search Arena: style randomization as an experiment

Search Arena takes a step toward the randomized trials called for in [4]. Each provider has a distinctive inline citation format, which threatens anonymity. Responses are therefore randomly shown either in a standardized citation style or in the provider's original style, which allows a direct estimate of the effect of citation style on votes [22]. Findings on roughly 7K filtered battles (from more than 11K collected) [22]:

- Response length, citation count, and citations to specific source types (YouTube, forums and blogs) correlate positively with preference.
- Standardizing citation style had minimal effect on the rankings.

The ICLR 2026 paper released a 24K-conversation dataset. It reports that "models do not always cite retrieved sources accurately and humans do not always check citations" [22].

**Lesson for designers.** Confounders that can be randomized at serving time (formatting, position, citation style) should be randomized rather than regressed out after the fact.

---

## 4. Engineering the prompt distribution

An arena measures preference over whatever prompts users happen to bring. LMArena has added a stack of controls on top of that distribution.

| Mechanism | What it does | Public numbers |
|---|---|---|
| Deduplication | Down-samples the top 0.1% most common prompts (about 1,000 prompts, mostly greetings) to the 99.9th-percentile frequency (25 occurrences) | Removes about 8.6% of votes [8] |
| Hard-prompt category | Llama-3-70B-Instruct labels prompts on 7 criteria (specificity, domain knowledge, complexity, problem-solving, creativity, technical accuracy, real-world application); a score of 6 or more counts as "hard" | About 20% of more than 1M prompts score 6 or higher [8] |
| Category pipeline ("Label Bench") | Checklist definition, then 2-5K battles labeled by a strong LLM, then a distilled classifier without CoT, then precision/recall check, then deployment over all battles | Creative-writing classifier: 96.1% accuracy (precision 66.7%, recall 96.6%) against GPT-4o-mini CoT labels on 2M prompts [9] |
| Freshness audit | Embedding similarity of 0.7 or more counts as a duplicate | About 75% of each day's prompts are fresh; under 1% appear in popular benchmarks (355,575 battles, May-Dec 2024) [10] |
| Topic explorer | Hierarchical BERTopic-style clustering of prompts | 100K-conversation preference dataset released [41] |

The category analysis also found that the share of hard, instruction-following, coding and math prompts has risen since launch. In other words, "people [are] asking harder questions as LLMs become better" [9].

The pipeline uses LLMs only to classify prompts, never to judge answers: "we do not use LLM as judges to evaluate model answers" [8].

---

## 5. The Leaderboard Illusion critique and the response

### 5.1 The critique

"The Leaderboard Illusion" (Singh et al., Cohere Labs with academic co-authors, April 2025) argued that undisclosed or unevenly used platform policies systematically distort Chatbot Arena [15]. Its principal claims, as reported in the paper and restated in LMArena's rebuttal [14][15], are:

1. **Best-of-N private testing.** A few providers test many anonymous variants before release and publish only the best. The paper reports, for example, that Meta tested 27 private variants in the run-up to Llama 4. It argues that choosing the maximum of N noisy estimates biases the published score upward, and illustrates this with a simulation.
2. **Data access asymmetry.** Through sampling rates and data sharing, proprietary providers receive much more arena data than open-weight developers. The paper reports Google and OpenAI each receiving roughly a fifth of all arena data (approximate figures), and states that open-source models represented 8.8% of the leaderboard.
3. **Silent deprecation.** Many models were removed from battle mode without announcement. Because BT assumes a connected, stationary comparison graph, the paper argues that removals can make rankings unreliable.
4. **Overfitting to the arena distribution.** Training on arena-style data produced large gains: a relative improvement of up to 112% on Arena-Hard.
5. **Same checkpoint, different scores.** Identical checkpoints submitted under different names received different scores.

The paper recommends:

- prohibiting score retraction after submission;
- limiting the number of private variants per provider;
- applying deprecation rules equally;
- making sampling fair (for example, uncertainty-based active sampling);
- full transparency about which models are tested and removed.

### 5.2 LMArena's response (May 2025)

LMArena accepted several recommendations and disputed several facts [14].

**Policy changes:**

- The policy now states explicitly that every provider may test multiple pre-release variants, subject to capacity.
- Retired models are marked. A public list of retired models has been kept since June 15, 2025.
- When more than 10 models were pre-release tested in parallel, scores are "provisional" until 2,000 fresh post-release votes arrive.
- From July 1, 2025, 100% of the vote data behind the public leaderboard (model identities and votes) is shared so the leaderboard can be reproduced [5][14].

**Factual disputes:**

- Open models account for 40.9% of battles, not 8.8%. LMArena says the paper missed open-weight families such as Llama and Gemma.
- The "100+ point" boost came from a simulation of Gaussians with an arbitrary variance. LMArena's own analysis puts the selection effect at about +11 Elo after 50 tests and 3,000 votes, decaying to zero as fresh votes accrue.
- The same-checkpoint examples have overlapping CIs.
- The 112% figure is on Arena-Hard (500 prompts, LLM judge), not on human-voted Chatbot Arena.
- Cohere itself received 9 pre-release tests since January 2025, which LMArena says is 2-3x more than xAI or OpenAI [14].

### 5.3 What a system designer should take from the exchange

Both sides agree on the mechanism. Both agree that selecting the maximum over N noisy scores introduces positive bias, and that the bias shrinks as fresh, post-selection votes accumulate. They disagree about its magnitude, which depends on per-variant vote counts and on whether provisional scores are re-estimated.

The architectural fixes are clear:

- (a) Separate the data used to select a variant from the data used to report its score. This is the holdout principle, and the 2,000-fresh-vote rule is a version of it.
- (b) Publish the full comparison graph, including retired nodes.
- (c) Make sampling policy auditable.
- (d) Report intervals, not ranks.

An earlier Cohere paper, "Elo Uncovered" [40], had already shown that online Elo is sensitive to ordering and hyperparameters. LMSYS cited it when moving to BT [3].

Separate academic work has studied adversarial vote manipulation on voting-based leaderboards. That is why de-anonymization (for example, recognizing a model by its style) together with targeted voting is part of the threat model. It is also part of the reason Search Arena standardizes citation formats [22].

---

## 6. Vertical arenas: adapting the battle to the workflow

The arena pattern has been rebuilt for settings where a chat-box vote is the wrong unit of measurement.

### 6.1 Copilot Arena (code completion inside VS Code)

Copilot Arena is a free VS Code extension that shows two completions from different models, one stacked above the other. Accepting one counts as the vote [20].

**Fairness controls:**

- top/bottom position and the model pairing are randomized;
- both completions are shown at the same moment, so the faster model waits;
- generation parameters are matched;
- chat models are prompted to re-emit part of the prefix and suffix, and the output is post-processed into a fill-in-the-middle completion, which greatly reduces formatting errors [20].

**Scale (Nov 2024):** 2.5K downloads, more than 100K completions, more than 10K battles, 833 unique voters, and 200-250 daily active users. Mean context was 1,002 tokens (median 560), far longer than HumanEval's median of about 100 [20].

**Position bias:** 82% of accepted completions were the top one. The bias also differed by model: shown in the bottom slot, Sonnet-3.5 was accepted 23.4% of the time and Gemini Flash 12.8%. The median time before selecting was 7 seconds, which suggests users do read the completions [20].

**Design note:** there is no "tie" option, which the authors list as a limitation.

### 6.2 WebDev Arena

Two models each build a web application from the same prompt, and users vote on the rendered apps. More than 80K votes were collected after the December 2024 launch [21]. Two observations stand out:

- **Ties:** the tie rate (26%) is lower than Chatbot Arena's (35%), but a much larger share of ties are "both bad," caused by uncompilable code or hallucinated imports.
- **Deduplication:** a large share of votes came from the sample prompts on the landing page. Deduplicating these changed the leaderboard only slightly [21].

### 6.3 Other derived products

- **RedTeam Arena.** A jailbreak game in which players are rated jointly with prompts and models [23].
- **Kaggle preference-prediction competition.** 55K training conversations, a 25K hidden test set and a $100K prize pool. It turned arena votes into a reward-modeling benchmark [24].
- **Preference Proxy Evaluations (PPE).** 16,038 human-labeled arena pairs (20 models, 121+ languages) plus 2,555 prompts with 32 sampled responses each, labeled for verifiable correctness [19]. PPE was validated end to end: nine reward models each produced DPO training sets for Llama-3.1-8B-Instruct, and the resulting models were deployed back to the arena for 12,190 votes. **Pairwise accuracy on the human-preference set was the best predictor** of downstream Arena Score, with correlation peaking around 0.80 at low-quantile aggregation. Spearman and Kendall correlations "have nearly zero correlation" with downstream outcomes [19].
- **Arena-Hard-Auto.** BenchBuilder selects hard, diverse prompts from arena logs, and an LLM judge (GPT-4.1 or Gemini-2.5 in v2.0) approximates the human vote [17]. Its metrics are:
  - agreement: confident pairwise agreement with Chatbot Arena;
  - separability: the share of model pairs whose bootstrap CIs do not overlap.

  v0.1 reported 87.4% separability and 89.1% agreement for about $25 per run [17]. v2.0 (April 2025) has 500 hard prompts and 250 creative-writing prompts, with style control as the default [17].

```
   live arena votes ----------------------------------+
        |                                             |
        | prompt logs                                 | human labels
        v                                             v
 BenchBuilder (7 criteria,   ---> Arena-Hard-Auto   PPE (reward-model
 topic clusters)                  (LLM judge, CI,   accuracy / downstream
        |                         separability)     RLHF correlation)
        v
 P2L (prompt -> BT vector) ---> routing, per-prompt leaderboards
```

---

## 7. Expert annotation pipelines

Arenas collect breadth: many people, many prompts, one vote each. Expert pipelines trade breadth for control over who judges, under which written rubric, and with what calibration. The public record shows a common architecture.

```
+------------+   +-------------+   +--------------+   +-------------------+
| Recruit    |-->| Screen      |-->| Onboard +    |-->| Production        |
| (vendor:   |   | (test vs.   |   | guidelines,  |   | labeling: rank /  |
|  Upwork,   |   |  researcher |   | shared chat, |   | pairwise / Likert |
|  Scale,    |   |  gold,      |   | FAQ updates  |   | + free-text notes |
|  MTurk,    |   |  sensitivity|   +--------------+   +---------+---------+
|  grads)    |   |  tests)     |                                |
+------------+   +-------------+                                v
                                                      +-------------------+
        +---------------------------------------------| QA: overlap items,|
        | feedback: retrain, re-screen, guideline     | gold checks, IAA, |
        | revisions, remove low-agreement labelers    | researcher audits |
        +-------------------------------------------->+-------------------+
```

### 7.1 OpenAI: Summarize-from-Feedback and InstructGPT

The Summarize-from-Feedback release contains 64,832 summary comparisons on TL;DR, plus Likert-scale axis evaluations on TL;DR and CNN/DM [31]. The paper is known for its emphasis on a close relationship with labelers and on measuring labeler-researcher agreement. It reports that labelers agreed with researchers at about the same rate that researchers agreed with each other, around 73-77% (approximate, reported in [31]).

InstructGPT's model card describes the labeling team: "about 40 contractors whom we hired through Upwork and ScaleAI." They were chosen with a screening test "designed to measure labeler performance" on sensitivity to the preferences of different demographic groups and on identifying harmful outputs, and by "their agreement rate with researchers on a labeling task with detailed instructions" [30]. The team was kept small on purpose because "it's easier to have high-bandwidth communication with a smaller set of contractors who are doing the task full-time." The model card also acknowledges the cost: the group "is clearly not representative of the full spectrum of people who will use and be affected by our deployed models," being primarily English-speaking [30].

The InstructGPT paper reports inter-annotator agreement of roughly 73% among training labelers and roughly 77% among held-out labelers (approximate, reported in [30]). Held-out labelers who never trained the reward model preferred InstructGPT outputs at similar rates, which is a validity check against overfitting to one rater pool.

### 7.2 Anthropic: HH-RLHF

The HH-RLHF data are chosen/rejected pairs collected from crowdworkers [32]:

- **Helpfulness** data come in three tranches: base context-distilled 52B models, rejection sampling (mostly best-of-16) against an early preference model, and an iterated "online" process.
- **Harmlessness** data come from base models only.
- The red-teaming release records whether each red-teamer came from Upwork or MTurk, a self-rated Likert success score, and preference-model harmlessness scores.

Anthropic's paper describes the open-ended, worker-chosen-topic interface and the relatively modest agreement between researchers and crowdworkers (reported in [32]). The difficulty of these judgments is part of why the dataset is used as a noisy-preference benchmark.

### 7.3 MT-Bench expert judgments

LMSYS released 3.3K expert pairwise preferences over 6 models' answers to 80 MT-Bench questions. The annotators were "mostly graduate students with expertise in the topic areas of each of the questions" [26]. The headline result is that "humans and GPT-4 judge achieve over 80% agreement, the same level of agreement between humans" [25][26]. This set the template for validating LLM judges (Chapter 06): measure judge-human agreement against the human-human ceiling, not against 100%.

### 7.4 Scale AI SEAL leaderboards

**Problem.** Public benchmarks get contaminated and gamed, and arena votes come from non-experts on self-selected prompts.

**Design pattern.** SEAL (Safety, Evaluations and Alignment Lab) leaderboards are built on curated prompt sets that are kept private or held out, and on evaluation by Scale's vetted expert contributor network, with public leaderboards at scale.com/leaderboard [36]. Only part of the methodology could be verified in source code during research for this chapter:

- **Public plus private splits of the same benchmark.** SWE-Bench Pro publishes a public leaderboard and a separate "Commercial (Private) Leaderboard" on labs.scale.com [37]. The private split exists to detect contamination and overfitting to the public set.
- **Locked evaluation protocols.** The SWE-Bench Pro V2 release (642 validated tasks, a HARD-51 subset) specifies an offline agent phase with web tools disabled, captures the patch, and re-grades it in a pristine sandbox with the unchanged verifier. Users are told to "report the re-graded number; publish both" [37]. Grading is moved out of the system under test.
- **Expert-authored content.** Humanity's Last Exam, a Scale and CAIS collaboration, has 2,500 questions "developed globally by subject-matter experts" and carries a canary string to help filter it from training data [38]. MASK, a CAIS and Scale honesty benchmark, separates honesty from accuracy [39].

**Failure modes.**

- Private sets cannot be audited by outsiders, so trust moves to the operator. That is the opposite trade-off from LMArena's "publish 100% of votes."
- The evaluator, the data vendor and labs' training-data suppliers can be the same organization, a structural conflict that should be disclosed.
- Even the public SWE-Bench Pro README records leaderboard issues that had to be corrected (May 2025 note) and outdated tests that had to be removed [37]. Expert pipelines also need ongoing maintenance.

### 7.5 Pluralistic and research datasets

- **PRISM.** PRISM links a survey of each participant (stated preferences, familiarity with LLMs, self-description, demographics) to that person's ratings of live multi-turn conversations. In the first turn, up to four models respond and each response is scored. Later turns continue with the chosen model and two responses [33]. PRISM's design question is *whose* preference is measured, not only *which response* is preferred. Its release records include the user's ID, so analyses can be conditioned on the rater.
- **AlpacaFarm and AlpacaEval human data.** 20K human preferences, including a 2.5K cross-annotation subset in which 4 humans labeled each of about 650 examples. MTurk workers were paid $21/hour [28][29].
- **UltraFeedback.** About 64K prompts, 4 responses each from a pool of models (256K samples), with GPT-4 annotations on four aspects: instruction-following, truthfulness, honesty and helpfulness. This yields about 340K comparison pairs [34]. It is the AI-feedback counterpoint: annotation cost falls by orders of magnitude, but the label source is a model and inherits that model's biases.

---

## 8. Inter-annotator agreement as a design quantity

### 8.1 How to measure it

AlpacaEval's documentation gives a precise, reusable protocol [28]:

- **Human agreement (leave-one-out).** For each of about 650 items with 4 human labels, take one label and test whether it predicts the mode of the other 3. Average over labels and items. If the mode is not unique, break the tie at random.
- **Bias.** Does the mode of 4 judge samples match the mode of 4 human labels?
- **Variance.** Leave-one-out agreement of a judge with itself across resampling (order swaps, seeds).
- **Price and time per 1,000 examples.**

The `humans` row in AlpacaEval's evaluator table reads [28]:

| Evaluator | Human agreement | Price ($/1000) | Time (s/1000) | P(prefer longer) |
|---|---|---|---|---|
| Humans (MTurk) | 65.7% | 300.0 | 36,800 | 0.64 |
| GPT-4 annotator | 66.9% | 12.5 | 1,037 | 0.65 |
| Claude annotator | 65.3% | 3.3 | 173 | 0.66 |
| lmsys_gpt4 | 65.3% | 13.9 | 17,982 | 0.74 |

The table makes three points:

- A single crowd rater agrees with the majority of other raters only about two-thirds of the time on open-ended instruction following.
- Several LLM judges match or exceed that ceiling at 1/25 to 1/100 of the cost.
- Humans also prefer longer outputs (0.64), so length bias is not purely an artifact of LLM judges.

### 8.2 Observed agreement levels across programs

| Program | Raters | Unit | Reported agreement |
|---|---|---|---|
| AlpacaFarm cross-annotations [28] | MTurk, 4 per item | pairwise | 65.7% (LOO vs majority) |
| MT-Bench expert set [25][26] | Domain grad students | pairwise | Human-human about the same as GPT-4-human (>80% without ties, reported) |
| Chatbot Arena paper [1] | Crowd vs. expert re-labels | pairwise | Crowd-expert agreement in the 70-80% range, comparable to expert-expert (reported, approximate) |
| InstructGPT [30] | About 40 screened contractors | ranking | About 73% training labelers, about 77% held-out (reported, approximate) |
| Llama 2 human eval [35] | Vendor raters, 3 per item | 7-point Likert | Gwet's AC2 about 0.37-0.55 depending on model pair (reported, approximate) |

### 8.3 Lessons

1. **Treat agreement as a ceiling, not a defect.** If humans agree 65-80% of the time, a judge or reward model that reaches that level has essentially saturated what the labels can tell you. PPE shows that pairwise accuracy against crowd labels still predicts downstream RLHF quality [19].
2. **Choose the statistic to suit the decision.** Raw percent agreement inflates when one option dominates. Chance-corrected statistics (Cohen's or Fleiss' kappa, Krippendorff's alpha, Gwet's AC) handle imbalanced labels and Likert scales. Arena-style rankings need *model-level* metrics with uncertainty, such as Arena-Hard's confident agreement and separability [17], rather than Spearman, which "does not consider variance of the rankings" [17].
3. **Low agreement can be signal.** PRISM's design and the sentiment-control results suggest that much "disagreement" reflects real heterogeneity in preferences across people and tones, not rater error [11][33]. Averaging it away into one leaderboard (the critique P2L answers [18]) hides the information.
4. **Ties are informative.** Copilot Arena's lack of a tie option forced choices. WebDev Arena's "both bad" option exposed failure modes such as uncompilable apps [20][21].

---

## 9. Comparison of human-evaluation architectures

| Dimension | Chatbot Arena / LMArena | Vertical arenas (Copilot, WebDev, Search) | Expert pipelines (InstructGPT, HH, MT-Bench) | Scale SEAL-style private evals | Research preference sets (PRISM, AlpacaFarm, UltraFeedback) |
|---|---|---|---|---|---|
| Who judges | Anonymous public users | Practitioners in their own workflow | Screened contractors or grad students | Vetted domain experts plus verifiers | Demographically profiled crowd, MTurk, or GPT-4 |
| Prompt source | User-supplied, live | User's real code, app or search task | Researcher, API, or labeler-written | Curated, private or held-out | Mixed; PRISM is user-led |
| Unit | Pairwise vote + tie / both-bad | Implicit acceptance or vote | Ranking, pairwise, Likert | Rubric score, verifier, pairwise | Pairwise or Likert, multi-aspect |
| Aggregation | BT logistic regression + bootstrap, style covariates | Same BT machinery | Win rate, reward-model training | Benchmark score with CIs | Dataset release |
| Scale | Tens of millions of pairings; 400+ models [13] | 10K-80K+ votes [20][21] | 3.3K-65K labels [26][31] | Hundreds to thousands of items [37][38] | 8K-340K items [33][34] |
| Contamination resistance | High (about 75% daily fresh prompts) [10] | High | Medium | High while private | Low once released |
| Auditability | High (vote data public since July 2025) [5] | Medium-high (datasets released) | Medium (datasets, partial guidelines) | Low for private splits | High |
| Main failure modes | Style bias, best-of-N selection, sampling asymmetry, non-expert judging | Position bias, sample-prompt duplication | Unrepresentative raters, small scale, cost | Opaqueness, vendor conflicts | Rater-pool bias; AI-feedback bias |
| Approximate cost | Free labor, high serving cost | Free labor, IDE engineering | About $21/hr MTurk to full-time contractors [28][30] | Expert rates (undisclosed) | $300 per 1K human labels vs. about $3-15 for LLM judges [28] |

---

## 10. Designing preference data collection: a checklist

Drawing on the systems above, the main design decisions are:

1. **Unit of judgment.** Pairwise comparison is easier and more reliable than absolute Likert scoring, and it feeds BT directly. Add "tie" and "both bad" options; "both bad" separates failure from indifference [21]. Use implicit signals (acceptance, persistence) when the workflow allows, as Copilot Arena does [20].
2. **Blinding and randomization.** Hide identities until after the vote and randomize position [2][20]. Equalize latency when speed would reveal the model [20]. Standardize identifying formats, or randomize them for analysis [22].
3. **Prompt provenance.** Live user prompts resist contamination [10] but skew toward easy prompts and greetings. Deduplicate [8], then stratify by category with a validated classifier [9].
4. **Rater selection.** Use crowds for breadth and preference. Use screened experts, with recorded agreement against gold labels, for correctness on hard or specialist prompts [27][30]. Record rater attributes if pluralistic analysis is planned [33].
5. **Redundancy.** Budget overlap labels (for example, 4 per item on a subset, as AlpacaFarm does) to estimate the agreement ceiling before trusting any automated proxy [28].
6. **Aggregation model.** Use batch BT with sampling reweighting, bootstrap CIs and an anchor model [6]. Add confounder covariates (length, markdown, sentiment) and publish both raw and controlled views [4][11].
7. **Abuse resistance.** Use CAPTCHA, per-identity vote caps, sequential outlier tests, prompt deduplication and delayed publication [5][7].
8. **Governance.** Publish the sampling policy, the pre-release testing rules, the retirement list and the full vote data [5][14]. Separate the votes used to select a variant from the votes used to report its score.
9. **Validation loop.** Check proxies such as LLM judges and reward models against held-out human labels using metrics that predict downstream outcomes. Pairwise accuracy beat rank correlation in PPE [19].

---

## Key takeaways for system designers

- **Use batch Bradley-Terry, not online Elo, for static models.** Fit it as weighted logistic regression, bootstrap the CIs, anchor the scale, and reweight for non-uniform sampling. FastChat's `rating_systems.py` is a compact reference implementation [6].
- **Treat style as a covariate.** Length is the dominant style effect. Formatting and positive sentiment also move votes. Publish controlled and uncontrolled leaderboards, and where possible randomize style at serving time rather than regressing it out afterward [4][11][22].
- **Selection bias is the arena's structural risk.** Best-of-N private testing, uneven sampling and silent deprecation can each move scores. Use fresh-vote holdouts for reported scores, public retirement lists and full vote releases [5][14][15].
- **Non-experts cannot reliably judge hard prompts.** Combine arena breadth with hard-prompt categories, expert-labeled subsets or private expert benchmarks (the SEAL pattern), and treat each as a different instrument [8][27][37].
- **Measure agreement before you automate.** Single-rater agreement on open-ended tasks is roughly 65-80%. An LLM judge at that level has hit the human ceiling. Report leave-one-out agreement, bias and variance, and model-level separability [17][25][28].
- **Disagreement is data.** Per-prompt and per-user leaderboards (P2L, PRISM) recover structure that a single global ranking averages away [18][33].
- **Auditability and contamination resistance pull against each other.** Public vote logs build trust but leak the distribution. Private sets resist gaming but require trust in the operator. Mature programs run both: public splits, private splits and locked re-grading [5][37].

---

## References

1. Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., Stoica, I. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." arXiv, 2024. https://arxiv.org/abs/2403.04132
2. Zheng, L., Sheng, Y., Chiang, W.-L., Zhang, H., Gonzalez, J. E., Stoica, I. (LMSYS). "Chatbot Arena" (launch post with initial Elo leaderboard). LMSYS Blog, 2023. https://lmsys.org/blog/2023-05-03-arena/
3. Chiang, W.-L., Li, T., Gonzalez, J. E., Stoica, I. "Chatbot Arena - New models & Elo system update." LMArena Blog, 2023. https://blog.lmarena.ai/blog/2023/leaderboard-elo-update/
4. Li, T., Angelopoulos, A., Chiang, W.-L. "Does Style Matter? Disentangling style and substance in Chatbot Arena." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/style-control/
5. LMArena. LMArena policy page: "LMArena Leaderboard Policy" (published March 2024, updated 2025). LMArena Blog. https://blog.lmarena.ai/blog/2024/policy/
6. LMSYS. FastChat `rating_systems.py` (Bradley-Terry, bootstrap, style control implementation). GitHub, 2024. https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/rating_systems.py
7. LMSYS. FastChat `elo_analysis.py` (anonymous-vote filtering, outlier user detection, vote limits). GitHub, 2024. https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/elo_analysis.py
8. LMArena Team. "Introducing Hard Prompts Category in Chatbot Arena." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/hard-prompts/
9. LMArena Team. "Chatbot Arena Categories." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/arena-category/
10. Dunlap, L., Lu, E., Gonzalez, J. E., Angelopoulos, A. N., Chiang, W.-L., Stoica, I., et al. "How Many User Prompts are New?" LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/freshness/
11. LMArena Team. "Does Sentiment Matter Too?" LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/sentiment-control/
12. Angelopoulos, A. N., et al. "Statistical Extensions of the Bradley-Terry and Elo Models." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/extended-arena/
13. LMArena Team. "Celebrating Community Impact: 3M+ votes, 400+ models, and 300+ pre-release tests." LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/two-year-celebration/
14. LMArena Team. "Our Response to 'The Leaderboard Illusion' Writeup." LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/our-response/
15. Singh, S., Nan, Y., Wang, A., D'Souza, D., Kapoor, S., Üstün, A., Koyejo, S., Deng, Y., Longpre, S., Smith, N. A., Ermis, B., Fadaee, M., Hooker, S. (Cohere Labs et al.). "The Leaderboard Illusion." arXiv, 2025. https://arxiv.org/abs/2504.20879
16. LMArena Team. "LMArena is Growing to Support our Community Platform." LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/new-beta/
17. Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., Stoica, I. "From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline." arXiv 2024 (https://arxiv.org/abs/2406.11939); "From Live Data to High-Quality Benchmarks - The Arena-Hard Pipeline" blog (https://blog.lmarena.ai/blog/2024/arena-hard/); Arena-Hard-Auto repository, 2025. https://github.com/lmarena/arena-hard-auto
18. Frick, E., et al. (LMArena). "Prompt-to-Leaderboard." arXiv, 2025 (https://arxiv.org/abs/2502.14855); code. https://github.com/lmarena/p2l
19. Frick, E., Li, T., Chen, C., Chiang, W.-L., Angelopoulos, A. N., Jiao, J., Zhu, B., Gonzalez, J. E., Stoica, I. "How to Evaluate Reward Models for RLHF" (Preference Proxy Evaluations). arXiv 2024 (https://arxiv.org/abs/2410.14872); blog. https://blog.lmarena.ai/blog/2024/preference-proxy-evaluations/
20. Chi, W., Chen, V., Angelopoulos, A. N., Chiang, W.-L., Mittal, A., Jain, N., Zhang, T., Stoica, I., Donahue, C., Talwalkar, A. "Copilot Arena." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/copilot-arena/
21. LMArena Team. "WebDev Arena." LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/webdev-arena/
22. Miroyan, M., Wu, T.-H., et al. "Search Arena: Analyzing Search-Augmented LLMs" (ICLR 2026). Blog 2025 (https://blog.lmarena.ai/blog/2025/search-arena/); code and data. https://github.com/lmarena/search-arena
23. LMArena Team. "RedTeam Arena." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/redteam-arena/
24. LMSYS and Kaggle. "LMSYS Chatbot Arena Kaggle Competition." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/kaggle-competition/
25. Zheng, L., Chiang, W.-L., Sheng, Y., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." arXiv, 2023 (https://arxiv.org/abs/2306.05685); FastChat llm_judge README. https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md
26. LMSYS. "Chatbot Arena Conversation Dataset Release" (incl. 3.3K MT-Bench expert annotations). LMArena Blog, 2023. https://blog.lmarena.ai/blog/2023/dataset/
27. LMSYS. "Chatbot Arena Leaderboard Updates (Week 4)." LMArena Blog, 2023. https://blog.lmarena.ai/blog/2023/leaderboard-week4/
28. Dubois, Y., Li, X., Taori, R., et al. (Stanford Tatsu Lab). AlpacaEval repository and evaluator documentation (human agreement, bias, variance, price). GitHub, 2023-2025. https://github.com/tatsu-lab/alpaca_eval
29. Dubois, Y., et al. "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback." arXiv, 2023. https://arxiv.org/abs/2305.14387
30. OpenAI. "InstructGPT Model Card." GitHub, 2022 (https://github.com/openai/following-instructions-human-feedback/blob/main/model-card.md); Ouyang, L., et al. "Training language models to follow instructions with human feedback." arXiv, 2022. https://arxiv.org/abs/2203.02155
31. Stiennon, N., et al. (OpenAI). "Learning to Summarize from Human Feedback." arXiv 2020 (https://arxiv.org/abs/2009.01325); data and code. https://github.com/openai/summarize-from-feedback
32. Bai, Y., et al. (Anthropic). "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv 2022 (https://arxiv.org/abs/2204.05862); HH-RLHF data. https://github.com/anthropics/hh-rlhf
33. Kirk, H. R., Whitefield, A., Röttger, P., et al. "The PRISM Alignment Project." arXiv 2024 (https://arxiv.org/abs/2404.16019); dataset repository. https://github.com/HannahKirk/prism-alignment
34. Cui, G., et al. (OpenBMB). "UltraFeedback." GitHub, 2023. https://github.com/OpenBMB/UltraFeedback
35. Touvron, H., et al. (Meta). "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv, 2023. https://arxiv.org/abs/2307.09288
36. Scale AI. "SEAL Leaderboards." Scale AI, 2024-2026. https://scale.com/leaderboard
37. Scale AI. "SWE-Bench Pro" repository (public and commercial/private leaderboards, V2 locked protocol). GitHub, 2025-2026. https://github.com/scaleapi/SWE-bench_Pro-os
38. Center for AI Safety and Scale AI. "Humanity's Last Exam" repository. GitHub, 2025. https://github.com/centerforaisafety/hle
39. Center for AI Safety and Scale AI. "The MASK Benchmark: Disentangling Honesty from Accuracy in AI Systems" repository. GitHub, 2025. https://github.com/centerforaisafety/mask
40. Boubdir, M., Kim, E., Ermis, B., Hooker, S., Fadaee, M. (Cohere). "Elo Uncovered: Robustness and Best Practices in Language Model Evaluation." arXiv, 2023. https://arxiv.org/abs/2311.17295
41. LMArena Team. "Arena Explorer." LMArena Blog, 2025. https://blog.lmarena.ai/blog/2025/arena-explorer/
