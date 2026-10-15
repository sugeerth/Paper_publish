# 14. Training Agents: Agentic RL, Environment Design, Tool-Use Fine-Tuning, RL for Software Engineering, and RL Gyms

## Overview

Earlier chapters looked at the agent as a *harness*: the loop, tools, memory, planning scaffolds and runtime around a fixed model. This chapter looks at the other half, which is how the model inside that harness is trained to be a good agent. Between 2023 and 2026 the public record moved through three stages.

1. **Supervised tool-use fine-tuning (2023-2024).** Models learned *when* and *how* to call tools from self-labelled, retrieved or synthesized demonstrations. Examples are Toolformer, Gorilla, ToolLLM, FireAct, AgentTuning, APIGen and ToolACE [1][2][4][5][6][7][8][9].
2. **Reinforcement learning with verifiable rewards (RLVR) extended to multi-turn tool use (2025).** After OpenAI's o-series [57], DeepSeek-R1 and Kimi k1.5 showed that critic-free policy gradients scale on checkable tasks [10][11], the same machinery was pointed at interpreters, search engines, browsers and repositories. Examples are ReTool, rStar2-Agent, Search-R1, DeepResearcher, SWE-RL and DeepSWE [28][29][23][24][15][16].
3. **Industrial agentic post-training (mid-2025 onward).** Frontier and open-weight labs built *environment factories*: thousands of real and synthetic tools, tens of thousands of parallel sandboxes, and judge models grounded in verifiable rewards. Kimi K2, Qwen3-Coder, GLM-4.5, MiniMax-M1 and OpenAI's codex-1 are the best-documented cases [12][50][49][48][55]. In parallel, open RL frameworks turned "the environment" into an installable, first-class service [33][34][35][36].

The central claim of this chapter is that **training an agent is mostly an environment-engineering problem**. The optimizer (some GRPO variant) is nearly a commodity. The hard, differentiating work is building tasks that are executable, verifiable, diverse and pitched at the right difficulty. That work also includes the rollout infrastructure that keeps GPUs busy while thousands of sandboxes run, and the reward design that survives a policy actively searching for loopholes [12][46].

```
          THE AGENT TRAINING STACK (2025-2026 consensus shape)
 +--------------------------------------------------------------------------+
 | 1. Task / environment factory                                            |
 |    real repos, MCP tools, synthetic tools, simulated users, web, GUIs    |
 |    -> containerized env + verifier (tests, rule checks, judge rubric)    |
 +-----------------------------------+--------------------------------------+
                                     |
 +-----------------------------------v--------------------------------------+
 | 2. Trajectory synthesis  (teacher models / multi-agent simulation)       |
 |    -> filter by verifier or LLM judge (rejection sampling)               |
 +-----------------------------------+--------------------------------------+
                                     | SFT "cold start"
 +-----------------------------------v--------------------------------------+
 | 3. Agentic RL                                                            |
 |    policy <-> env, multi-turn rollouts, tool outputs masked from loss,   |
 |    group-relative advantages, difficulty filtering, length budgets       |
 +-----------------------------------+--------------------------------------+
                                     |
 +-----------------------------------v--------------------------------------+
 | 4. Evaluation + production feedback (SWE-bench, tau-bench, BFCL, logs)   |
 +--------------------------------------------------------------------------+
```

---

## 14.1 Generation 1: Supervised Tool-Use Fine-Tuning

### 14.1.1 Toolformer (Meta AI, 2023): self-supervised API-call insertion

**Problem.** Language models were weak at arithmetic, factual lookup and dates, but tool use at the time needed either large human annotation or task-specific prompting.

**Architecture and data flow.** Toolformer turns a plain-text pretraining corpus into a tool-use corpus without human labels [1]:

```
 plain text x ----> (a) SAMPLE: few-shot prompt the LM to propose API calls
                        at positions where P("<API>") is high
                   (b) EXECUTE: run each candidate call (calculator, QA,
                        Wikipedia search, MT, calendar) -> result r
                   (c) FILTER: keep call c at position i only if inserting
                        "<API> c -> r </API>" lowers the LM loss on the
                        FOLLOWING tokens by >= threshold, compared with no call
                        and with the call without its result
                   (d) FINE-TUNE the same LM on the augmented corpus
```

**Key techniques.** The filtering criterion is the important idea. A call is "useful" if it makes future text more predictable. That self-supervised signal is a distant ancestor of the verifiable rewards used later. Only a handful of demonstrations per tool were needed. The tools were a calculator, a QA system, search, translation and a calendar [1].

**Results and limits.** The paper reports that a GPT-J-scale model trained this way improved zero-shot performance on several downstream tasks, often competitive with much larger models [1]. The authors are explicit about the limits. Tools cannot be *chained*, because calls were sampled independently per position. The model cannot *interact* with a tool, for example by refining a search. Data generation is sample-inefficient for some tools [1]. Each limit points to what multi-turn RL later supplied.

**Lesson.** A tool-use label can come from the *consequence* of the call rather than from a human. Every later verifier-filtered pipeline reuses this idea.

### 14.1.2 Gorilla (UC Berkeley, 2023): retriever-aware training for API calls

**Problem.** API surfaces are huge and change constantly, so models hallucinate API names and arguments.

**Architecture.** Gorilla fine-tuned LLaMA to write API calls over HuggingFace, TorchHub and TensorHub. The resulting APIBench dataset has roughly 1,600 APIs, each paired with self-instruct-generated instructions [2]. The key technique is **Retriever-Aware Training (RAT)**. During fine-tuning, the instruction is augmented with retrieved API documentation, which may be imperfect. The model therefore learns to use relevant documentation at inference time and to discount irrelevant documentation. Generated calls are checked by **AST sub-tree matching** against the reference API, which gives a structural, partly verifiable metric for hallucination [2].

**Results.** Gorilla reported surpassing GPT-4 at writing API calls. Pairing it with a document retriever let it adapt to documentation changes at test time and substantially reduced hallucination [2]. The project later produced the **Berkeley Function Calling Leaderboard (BFCL)**, which became one of the standard targets for function-calling post-training [3].

**Lesson.** Tool knowledge should be *retrieved*, not memorized. Training *with* retrieval noise is what makes retrieval safe to rely on at inference time.

### 14.1.3 ToolLLM / ToolBench (2023-2024): search-annotated solution paths

ToolLLM collected **16,464 REST APIs** from RapidAPI, grouped into 3,451 tools. It generated instructions with a strong model and annotated solution paths with **DFSDT**, a depth-first search decision tree that lets the annotator explore several reasoning paths and backtrack. DFSDT solved instructions that plain chain-of-thought or ReAct could not. A neural API retriever recommends APIs per instruction, and the reported pass rate drops only slightly compared with ground-truth API sets [4]. Architecturally this is *search at data-generation time, imitation at training time*. The same pattern appears in rStar-Math and in Kimi K2's rejection sampling [12].

### 14.1.4 Trajectory tuning for general agents: FireAct and AgentTuning (2023)

- **FireAct** fine-tuned smaller open models on ReAct-format trajectories generated by GPT-4 on question-answering tasks with a search tool. The paper reports large gains from only a few hundred trajectories. It also reports that mixing trajectories from several prompting methods (ReAct, CoT, Reflexion-style) improves robustness [6].
- **AgentTuning** (Tsinghua, 2023) built **AgentInstruct**, a small set of high-quality, filtered interaction trajectories across six agent tasks. It mixed them with general instruction data ("hybrid instruction tuning") to fine-tune Llama-2 models into **AgentLM**. The finding is that agent-only data damages general ability, while mixing preserves it. The 70B AgentLM was reported to approach GPT-3.5 on held-out agent tasks [5].

**Lesson.** Agent SFT data is a *capability delta*, not a replacement corpus. Later systems keep the same principle: Kimi K2's auxiliary PTX loss on curated data is a later version of "mix to avoid forgetting" [12].

### 14.1.5 Verified synthetic function-calling data: APIGen, APIGen-MT, ToolACE (2024-2025)

As function calling became a product feature, the data problem became breadth and correctness at scale.

- **APIGen** (Salesforce) synthesizes function-calling examples over thousands of executable APIs across many categories. Each sample passes three stages of verification: a **format check**, an **execution check** (the call actually runs) and a **semantic check** (an LLM confirms that the result answers the query). The resulting xLAM models ranked highly on BFCL at release relative to their size [7].
- **APIGen-MT** extends this to multi-turn agent conversations. Phase 1 builds a *task blueprint* with ground-truth actions, validated by a committee of LLM reviewers with feedback loops. Phase 2 turns the blueprint into a full trajectory through *simulated human-agent interplay*. The xLAM-2 models trained on it were reported to outperform larger general models on tau-bench and BFCL [8][58].
- **ToolACE** (Huawei and collaborators) generates a large, diverse synthetic API pool (tens of thousands of APIs) through a self-evolution process. It produces dialogs through multi-agent interplay and applies **dual-layer verification**, rule-based and model-based. An 8B model trained on it was reported to rival much larger models on BFCL [9].

```
     VERIFIED SYNTHETIC TOOL DATA (APIGen / ToolACE family)
 API pool (real executable or synthesized)
        |
        v
 query + call generator (LLM, multi-agent role-play: user / assistant / tool)
        |
        v
 [format check] --fail--> drop
        |
 [execute or simulate] --error--> drop
        |
 [semantic / model check] --mismatch--> drop
        |
        v
 SFT dataset (single-turn, parallel, multi-turn, irrelevance/abstain cases)
```

**Failure modes of generation 1.**
- *Exposure bias.* The model imitates teacher trajectories but never sees the consequences of its own mistakes.
- *Simulator drift.* Simulated tool outputs can be more regular than real ones.
- *Benchmark coupling.* Synthetic data built around a leaderboard's format can overfit it.

The move to on-policy RL in 2025 addresses the first two directly.

---

## 14.2 From Single-Turn RLVR to Agentic RL: What Changes

The reasoning-RL recipe is: sample G answers per prompt, score them with a rule-based verifier, compute group-relative advantages, and update the policy without a value network. DeepSeek-R1 and Kimi k1.5 established it at scale [10][11]. Agentic RL keeps that core and changes almost everything around it.

| Concern | Single-turn RLVR | Multi-turn agentic RL |
|---|---|---|
| Episode | one prompt, one long completion | many turns: think, tool call, observation, and so on |
| Tokens in context | all policy-generated | interleaved policy tokens and environment tokens |
| Loss mask | all completion tokens | **only policy tokens**; tool outputs masked [28][23] |
| Reward | answer check | tests or state check at episode end; sometimes judge rubric |
| Rollout cost | GPU decode | GPU decode + sandbox CPU and IO, network, simulators |
| Length variance | high | extreme: some trajectories exceed 100 tool calls and 400k tokens [27] |
| Main systems bottleneck | long-tail generation | stragglers + environment latency + container start-up |
| New failure modes | length hacking | env exploits (for example `sys.exit(0)`), echo traps, retokenization drift [46][30][40] |

Four techniques recur in agent-specific training.

1. **Observation masking.** Tool and interpreter outputs are excluded from the policy-gradient loss, so the model is not trained to "predict" the environment. ReTool and Search-R1 both describe this explicitly [28][23].
2. **Token-faithful trajectories.** The trainer must optimize exactly the token IDs that the inference engine sampled. The Agent Lightning team documented **retokenization drift**: re-tokenizing logged strings produces effectively off-policy updates. It has three causes: non-unique tokenizations, tool-call JSON re-serialization that silently "auto-corrects" malformed calls, and chat-template mismatches. The fix, upstreamed into vLLM v0.10.2, returns `prompt_token_ids` and `token_ids` through the OpenAI-compatible API [40][39].
3. **Trajectory-level credit assignment.** Most systems still use one scalar per episode. RAGEN's analysis of multi-turn RL (the StarPO framework) identified an **"Echo Trap"**: reward variance collapses and gradients spike as the policy overfits to locally rewarded reasoning patterns. It proposed a stabilized variant with trajectory filtering, a critic and decoupled clipping [30]. **ArCHer** proposed a hierarchical alternative earlier. It pairs an off-policy, utterance-level critic with a token-level policy and reports large sample-efficiency gains over flat token-level methods on multi-turn tasks [32].
4. **Curricula from failure.** **WebRL** trains web agents on WebArena-Lite with a *self-evolving curriculum*: new tasks are generated from the agent's failed attempts, scored by an outcome-supervised reward model, and trained with KL-constrained updates. It reported large gains for open Llama-3.1 and GLM-4 models over their starting points [31].

```
           MULTI-TURN AGENTIC RL ROLLOUT (one episode, one group member)
 prompt -> [policy: think + tool_call]    (loss ON)
        -> [env: tool result / stdout]    (loss OFF, masked)
        -> [policy: think + tool_call]    (loss ON)
        -> [env: test output]             (loss OFF)
        -> [policy: final patch / answer] (loss ON)
        -> verifier(state) -> r in {0,1} or [0,1]
 group of G episodes -> advantage_i = r_i - mean(r)  (optionally / std)
 drop group if all r equal (zero gradient); drop/mask truncated episodes
```

---

## 14.3 Tool-Integrated Reasoning RL: Interpreters and Search

### 14.3.1 ReTool (ByteDance Seed, 2025): code interpreter inside the reasoning loop

**Problem.** Text-only reasoning RL is weak at exact computation.

**Architecture.** ReTool runs in two stages [28]:
- **Cold-start SFT** on synthesized reasoning traces that contain code blocks.
- **RL with real-time code execution.** During rollout, a generated code block is sent to a sandbox, and the interpreter output is inserted back into the context. Interpreter feedback is masked from the loss, and the reward is final-answer correctness.

**Results.** On AIME 2024, a 32B model reached about 67% after roughly 400 RL steps. The text-only RL baseline reached about 40% after more than 1,000 steps (approximate figures as reported) [28]. The paper also reports emergent behaviours such as **code self-correction**: the model reads an error and rewrites the code.

### 14.3.2 rStar2-Agent (Microsoft, 2025): GRPO-RoC for clean tool use

rStar2-Agent is a 14B model that uses a Python tool. It was trained with agentic RL in "only 510 RL steps within one week" on 64 MI300X GPUs and reached 80.6% / 69.8% average pass@1 on AIME24 / AIME25, with responses shorter than DeepSeek-R1's [29]. Its algorithmic contribution is **GRPO-RoC (Resample-on-Correct)**. It oversamples rollouts and keeps "higher-quality positive trajectories while preserving all failure cases". This pushes the policy toward shorter, cleaner tool use instead of rewarding correct answers reached through noisy, error-ridden tool calls [29].

**Lesson.** With tools in the loop, outcome-only reward rewards *any* path to the answer, including sloppy ones. Filtering *which* successes enter the batch is a cheap form of process supervision.

### 14.3.3 Search agents: Search-R1, DeepResearcher, Tongyi DeepResearch, ASearcher, Kimi-Researcher

- **Search-R1** extends R1-style RL to interleaved reasoning and search. It is built on veRL, supports PPO, GRPO and REINFORCE, and works with local retrievers or online search engines. It uses rule-based outcome rewards (exact match on NQ and HotpotQA-style data) and masks retrieved tokens [23].
- **DeepResearcher** trains with GRPO against *live* web search and browsing. It reports gains of up to 28.9 points over prompt-engineered baselines and up to 7.2 points over RAG-based RL agents. The authors argue that training on the real web is "a fundamental requirement" [24].
- **Tongyi DeepResearch** (Alibaba, 2025) is a 30.5B-total, 3.3B-active MoE. It is trained by a fully automated synthetic-data pipeline that feeds **agentic continual pre-training, SFT and on-policy RL**. The RL uses a customized GRPO with token-level gradients, leave-one-out advantages and selective filtering of negative samples, to stay stable "in a non-stationary environment" [25]. Sibling projects split the problem into parts. **WebSailor** builds high-uncertainty questions by graph sampling and information obfuscation, then trains with an RFT cold start plus DUPO RL. **WebDancer** uses a four-stage pipeline: data construction, trajectory sampling, SFT and DAPO [26].
- **ASearcher** (inclusionAI, built on AReaL) attacks the systems problem. Search trajectories vary in length by up to two orders of magnitude in tokens, so batch-synchronous RL leaves GPUs idle. **Fully asynchronous RL** allowed 128-turn limits and produced trajectories with more than 100 tool calls and more than 400k tokens. A data-synthesis agent "fuzzes" questions and injects context, keeping only questions a strong reasoning model fails [27].
- **Kimi-Researcher** (Moonshot, June 2025) is trained end-to-end with REINFORCE and rule-based outcome rewards [14].

---

## 14.4 RL for Software Engineering

Software engineering became the flagship domain for agentic RL. It has a natural verifier (tests), real economic value and a public benchmark (SWE-bench Verified). It is also the hardest domain to make into environments: every task needs a reproducible container with the repository at the right commit, installed dependencies and tests that fail before the fix and pass after.

### 14.4.1 Environment factories: SWE-Gym, R2E-Gym, SWE-smith, Multi-SWE-RL, RepoLaunch

| Gym | Task source | Scale (as reported) | Verifier | Headline training result |
|---|---|---|---|---|
| SWE-Gym (ICML 2025) [19] | real GitHub issues, 11 Python repos | about 2.4k tasks, pre-built per-instance images | repo unit tests | under 500 trajectories gave a 32B OpenHands agent +14 pts on Verified; trained verifiers reached 32% with best-of-n |
| R2E-Gym (2025) [18] | real commits + **SWE-GEN** back-translation (tests and issue text generated from commits) | about 8k problems, 13 repos (approx.) | generated + existing tests; hybrid execution and execution-free verifiers | about 51% on Verified with hybrid test-time scaling, 32B open model (approx.) |
| SWE-smith (NeurIPS 2025 D&B) [20] | **synthesized bugs** injected into 250+ repos | about 52k tasks | keep only tasks that break at least one test | SWE-agent-LM-32B: 40.2% pass@1 on Verified |
| Multi-SWE-RL (ByteDance) [21] | expert-curated, 7 languages | 4,723 RL instances | per-language test harnesses | companion to Multi-SWE-bench |
| SWE-bench-Live / RepoLaunch (Microsoft) [22] | live issues; an **LLM agent builds the container** | 856 issues from 93 repos set up with at least 98% success | tests | 82% lower LLM API cost, 78% less Docker storage after optimization |

The design space has two axes.
- **Real versus synthetic tasks.** SWE-Gym uses real issues, which are realistic but scarce. SWE-smith injects bugs synthetically, which is abundant but has a different distribution. R2E-Gym sits between them, using real commits with synthetic issue text and tests.
- **Per-instance versus per-repo environments.** SWE-smith builds *one* environment per repository and synthesizes many tasks inside it. This is why it reaches tens of thousands of tasks at modest storage cost [20].

Automated environment construction (RepoLaunch) is the capability that limits how fast new SWE gyms can be built at all [22].

```
            SWE GYM CONSTRUCTION PIPELINE (generic)
 repo @ commit --> env-builder agent (install deps, run test suite)
        |                       |
        |                       +--> fail: discard repo / retry with fixes
        v
 task source:  real issue+PR | commit + back-translated issue | injected bug
        |
        v
 validate: FAIL_TO_PASS tests fail before, pass after gold patch;
           PASS_TO_PASS tests stay green; runtime under budget
        |
        v
 container image (or per-repo image + patch) + task card + verifier
```

### 14.4.2 SWE-RL (Meta FAIR, 2025): RL on software-evolution data without execution

**Problem.** Executable environments are expensive. Meanwhile GitHub holds a record of "software evolution" (issues, pull requests and code changes) at a scale no gym can match.

**Architecture.** SWE-RL trains Llama-3-family models with RL on a large curated corpus of pull requests [15]:

```
 GitHub events + repo clones --> PR records (issue text, code context, oracle patch)
                                         |
 policy input:  issue + full contents of relevant files
 policy output: reasoning + search/replace edits
                                         |
 reward:  -1                      if output is malformed
          similarity(pred, oracle) otherwise (sequence-similarity of patches, 0..1)
                                         |
                              GRPO-style policy update
```

**Key techniques.**
- **No execution in the training loop.** The reward is a lightweight *rule-based similarity* to the human patch, so the model can train on far more tasks than any containerized gym can hold.
- **A pipeline scaffold rather than a free-form agent** at evaluation time ("Agentless Mini"): localization, repair, then reproduction-test generation and selection.

**Results.** The resulting Llama3-SWE-RL-70B reported 41.0% on SWE-bench Verified. That was described as the best result among medium-sized (under 100B) open models at the time [15]. The most interesting finding is **generalization**: RL on SWE data improved *out-of-domain* abilities, including function-level coding, library use, code reasoning, mathematics and general language understanding. An SFT baseline on the same data *degraded* some of these [15].

**Failure modes and lessons.** Similarity-to-oracle rewards penalize correct but different fixes and can reward near-misses that do not run. The design accepts reward noise in exchange for data scale. Later systems (DeepSWE, industrial labs) moved back toward execution-based rewards once sandbox capacity existed. SWE-RL remains the clearest evidence that *RL on real developer data teaches transferable reasoning* rather than benchmark-specific tricks.

### 14.4.3 DeepSWE (Agentica and Together AI, 2025): fully open RL for a long-horizon coding agent

**Problem.** Show that an open model can learn multi-step SWE behaviour (explore, edit, run tests, iterate) *purely through RL*, with recipe, data, code and logs all open.

**Architecture.** DeepSWE-Preview starts from Qwen3-32B and is trained with the **rLLM** framework [16][17]. rLLM is built on verl and has since added other backends [16]. The environment is a subset of **R2E-Gym** (about 4.5k problems, approximate). The agent acts through a small tool set (bash execution, file view and edit, search, finish) inside a per-task container. The reward is sparse and binary: 1 if the final patch passes the selected FAIL_TO_PASS and PASS_TO_PASS tests within a time limit, 0 otherwise [17].

**Key techniques.** The blog describes "GRPO++", which combines fixes from the reasoning-RL literature [17][51]:
- clip-higher and no KL term, from DAPO;
- no reward standard-deviation normalization and length-normalization fixes, from Dr. GRPO [56];
- leave-one-out advantage;
- **compact filtering**: trajectories that hit the context limit, step limit or timeout are masked out of the loss, so the agent is not punished for running out of budget on otherwise reasonable behaviour;
- no entropy bonus.

**Scale and results (as reported, approximate).** Training ran on about 64 H100 GPUs for about six days, for roughly 200 RL steps. The result was about 42% Pass@1 on SWE-bench Verified, and about 59% with hybrid test-time scaling that combines execution-based and execution-free verifiers to pick among sampled trajectories [17]. The rLLM repository lists DeepSWE-32B alongside the same team's DeepScaleR-1.5B (math) and DeepCoder-14B (competitive code) [16].

**Lessons.**
1. **Masking truncated rollouts matters.** Long-horizon agents frequently run out of turns, and penalizing them teaches the agent to give up early.
2. **Verifiers double as test-time selectors.** The same signal used for training ranks candidates at inference.
3. **Open, reproducible SWE RL is feasible at university-plus-cloud scale.**

### 14.4.4 Industrial SWE and agentic RL disclosures

- **OpenAI codex-1 (May 2025)** is described as a version of o3 "trained using reinforcement learning on real-world coding tasks in a variety of environments". The target behaviour is code that mirrors human style and PR preferences, follows instructions, and "can iteratively run tests until it receives a passing result" [55]. OpenAI's later note that SWE-bench Verified "no longer measures frontier coding capabilities" shows how quickly RL on SWE-like environments saturated the public yardstick [59].
- **Qwen3-Coder (Alibaba, July 2025)** describes two RL phases [50]:
  - "Code RL" on tasks that are hard to solve but easy to verify;
  - long-horizon "Agent RL" on real SWE tasks. Alibaba built a system that runs **about 20,000 independent environments in parallel** on its cloud to supply rollouts.
- **GLM-4.5 (Z.ai, 2025)** is a 355B-total, 32B-active MoE with agentic, reasoning and coding post-training. Its RL runs on **slime**, Z.ai's Megatron-plus-SGLang framework with synchronous and fully asynchronous modes. GLM-4.5 also introduced interleaved thinking for tool use [49][34].
- **MiniMax-M1 (2025)** was trained with RL across environments "from math to sandbox-based, real-world software engineering". It introduced **CISPO**, which clips importance-sampling weights rather than token updates. MiniMax reported that full RL training took about three weeks on 512 H800 GPUs, at a rental cost of roughly USD 535k (approximate) [48].

---

## 14.5 Kimi K2: Agentic Post-Training at Trillion-Parameter Scale

Kimi K2 (Moonshot AI, July 2025) is the most fully documented open-weight example of a model whose post-training was designed around agency [12]. It is a mixture-of-experts model with about 1T total and 32B activated parameters.

### 14.5.1 Pretraining stability: MuonClip

K2 was pretrained with the token-efficient **Muon** optimizer. At this scale Muon produced exploding attention logits. Moonshot's fix, **MuonClip**, adds **QK-Clip**, which rescales query and key projection weights when attention logits exceed a threshold. The report credits it with pretraining on roughly 15.5T tokens with no loss spikes [12]. This is not agent-specific, but it is what made a trillion-parameter base available for agentic RL in the first place.

### 14.5.2 The large-scale agentic data synthesis pipeline

```
           KIMI K2 AGENTIC DATA SYNTHESIS (as described in [12])
 +-------------------------+     +----------------------------------------+
 | Tool repository         |     | Agent + task generation                |
 |  3,000+ real MCP tools  |---->|  diverse agents (system prompt +       |
 |  20,000+ synthetic tools|     |  tool subset); tasks with rubrics      |
 |  (domain-evolved specs) |     |  (success criteria, expected calls)    |
 +-------------------------+     +-------------------+--------------------+
                                                     |
                     +-------------------------------v---------------------+
                     | Multi-turn trajectory generation                    |
                     |  simulated user  <->  agent  <->  tool simulator    |
                     |  (stateful, injects stochastic outcomes/failures)   |
                     |  + REAL execution sandboxes for coding / SWE        |
                     +-------------------------------+---------------------+
                                                     |
                     +-------------------------------v---------------------+
                     | LLM judge scores each trajectory against its rubric |
                     | keep successes only = large-scale rejection sampling|
                     +-------------------------------+---------------------+
                                                     v
                                           SFT data for agentic skills
```

The technical report highlights three design choices [12]:
- **Breadth through synthetic tools.** Real MCP tools anchor realism. Synthetic tools, generated by evolving domain and tool specifications, supply coverage.
- **A hybrid simulator.** Simulated tools are cheap and controllable, while real sandboxes ground the domains where simulation fidelity matters most, such as coding.
- **Rubric-based filtering** "effectively implements large-scale rejection sampling".

### 14.5.3 RL: a verifiable-rewards gym plus a self-critique rubric reward

- **Verifiable Rewards Gym** [12]:
  - math, STEM and logic tasks selected for moderate difficulty by SFT pass@k;
  - instruction following with "hybrid rule verification" (code interpreters for checkable constraints plus LLM judges) and a **"hack-check layer"** that detects claims of compliance without actual compliance;
  - a sentence-level faithfulness judge;
  - coding and SWE tasks in a Kubernetes sandbox supporting "over 10,000 concurrent sandbox instances".
- **Self-Critique Rubric Reward.** For non-verifiable tasks, the K2 critic ranks the actor's outputs pairwise against core, prescriptive (anti-hacking) and human-written rubrics. "On-policy rollouts generated from verifiable-reward prompts are used to continuously update the critic", so subjective judgments stay grounded in checkable data [12].
- **Algorithm and stabilizers.** The policy update follows K1.5's regularized squared-loss objective, without clipping [11][12]. Stabilizers include per-task token budgets with truncation penalties, an auxiliary PTX loss on curated data to prevent forgetting, and temperature decay from exploration to exploitation [12].

### 14.5.4 Infrastructure

Training and inference engines are colocated on the same GPUs. For a 1T model, resharding through shared storage was infeasible, so a distributed **checkpoint engine** broadcasts the full parameter set parameter by parameter, and each inference worker takes the shard it needs. The report says this "transfers several times more data than a theoretically optimal approach", but it fully decouples the engines and completes a weight update in under 30 seconds. Long agentic rollouts use partial-rollout techniques inherited from k1.5, and environments run as remote services, so slow tool calls do not block GPUs [11][12].

### 14.5.5 Results and successors

The report and model card list agentic results for the non-thinking Instruct model, including roughly 65.8% single-attempt on SWE-bench Verified and strong tau2-bench and ACEBench tool-use scores (approximate, as reported) [12]. The model is described as "a reflex-grade model without long thinking". Its successor, **Kimi K2 Thinking** (November 2025), combines long reasoning with tool use. Moonshot reports that it executes about 200-300 sequential tool calls without human intervention and ships with native INT4 weights produced by quantization-aware training (approximate, as reported) [13].

**Lessons from K2.**
1. **Separate breadth from fidelity.** Synthetic tools give breadth, and real sandboxes give fidelity where it matters most.
2. **Ground the judge.** A rubric critic that is continuously recalibrated on verifiable rollouts is the clearest public design for extending RL to non-verifiable agent behaviour.
3. **Build anti-hacking into the reward from day one** (hack-check layer, prescriptive rubrics), not after the first exploit is found.

---

## 14.6 Training GUI and Computer-Use Agents

GUI agents follow the same arc from SFT to RL, with harder environments.

- **UI-TARS** (ByteDance Seed) is an end-to-end "native" GUI agent model: perception, reasoning, grounding and action happen in one VLM. UI-TARS-1.5 adds RL-trained "think before act" reasoning and reports 42.5% on OSWorld (100 steps) and 64.2% on AndroidWorld. UI-TARS-2 extends the approach to games, code and tool use [52].
- **OpenCUA** (XLANG Lab) open-sourced the data side:
  - an annotation tool that records screen video, input events and accessibility trees;
  - the 22.6k-task **AgentNet** dataset;
  - models trained with "reflective long chain-of-thought" that comments on the previous action, the alternatives and the expected next state. OpenCUA-72B reported 45.0% on OSWorld-Verified [53].
- **CUA-Gym and CUA-Gym-Hub** (XLANG) provide a pipeline for synthesizing *verifiable* computer-use tasks and a set of mock web apps packaged as RL environments [54].

The constraint specific to GUI RL is **environment snapshotting**. Group-based RL needs G independent rollouts from the same initial state, so VMs or browser states must be cloned cheaply and reset deterministically.

---

## 14.7 RL Gyms, Environment Hubs and Training Frameworks

Open frameworks converged on a separation of concerns:
- an **environment service** (tasks, sandboxes, verifiers, judges);
- a **rollout service** (vLLM or SGLang running multi-turn tool loops);
- a **trainer** (Megatron or FSDP);
- a **weight-update path** between trainer and rollout service [33][34][35][37].

```
 +------------------+   tasks + rubrics    +------------------------------+
 |  Environment hub |--------------------->|  Rollout service             |
 |  (sandboxes,     |                      |  vLLM / SGLang, multi-turn   |
 |  verifiers,      |<---------------------|  tool loops, partial         |
 |  judges, sims)   |  actions / patches   |  rollouts, token-ID capture  |
 +--------+---------+                      +--------------+---------------+
          | rewards                                       | trajectories
          v                                               v
 +------------------+   batches            +------------------------------+
 |  Buffer + filters|--------------------->|  Trainer (Megatron / FSDP2)  |
 |  zero-variance,  |                      |  GRPO / DAPO / CISPO / RLOO  |
 |  staleness, trunc|                      +--------------+---------------+
 +------------------+   weight sync (broadcast / delta)   |
          ^-----------------------------------------------+
```

| Framework / service | Origin | Distinctive design | Agent-relevant notes |
|---|---|---|---|
| verl (HybridFlow) [33] | ByteDance Seed + community | hybrid controller; 3D-HybridEngine resharding | backend for Search-R1, rLLM, Agent Lightning; PPO, GRPO, DAPO, RLOO and more |
| slime [34] | THUDM / Z.ai | Megatron + SGLang; Data Buffer; sync and fully async | RL engine behind the GLM-4.5 series |
| prime-rl + verifiers [35][36] | Prime Intellect | fully async; environments as installable packages; Environments Hub | built for 1T+ MoE on 1000+ GPUs |
| AReaL [37] | Tsinghua IIIS + Ant Group | fully asynchronous; reported 2.77x over sync (v0.3) | ASearcher's long-horizon search RL [27] |
| OpenRLHF [38] | community | Ray-separated actor, critic, reward, reference; async mode | general RLHF and RLVR |
| rLLM [16] | Agentica | agent-harness-centric API over verl and other backends | DeepSWE, DeepCoder, DeepScaleR |
| Agent Lightning [39] | Microsoft Research | OpenAI-compatible **proxy** in front of an *unmodified* agent; gateway records token IDs, logprobs and reward events | reported SWE-bench Verified gains of about 14 points on Qwen3.5-9B with 6K samples |
| ART + RULER [41][42][43] | OpenPipe | GRPO on LoRA, vLLM hot-swap; RULER LLM judge gives **relative** scores within a group | FAQ: needs a base success rate of at least 30%, rerunnable without real-world side effects, about USD 15-200 per run |
| OpenAI RFT [44] | OpenAI | hosted RL on reasoning models with programmable graders | cookbook documents grader reward hacking and fixes |
| Tinker cookbook [45] | Thinking Machines Lab | hosted training API; recipes incl. on-policy distillation | OPD lifted an SFT model from about 65% to about 76.7% on AIME'24 in 200 steps |

Three design points deserve emphasis.

1. **Environment as package.** prime-rl's `verifiers` and Environments Hub treat a task family (dataset, tool loop, reward) as a versioned, installable artifact [35][36]. This is the "gym" idea in its modern form. It enables sharing, reuse and decontamination auditing.
2. **The harness as the environment.** Agent Lightning's proxy captures every model call from a production agent framework, so the agent's actual tools, context management and control flow are what gets trained [39]. This avoids the sim-to-prod gap of re-implementing the agent inside the trainer.
3. **Relative judging fits group-based RL.** RULER rests on the observation that "GRPO only needs relative scores". Ranking N trajectories of the same scenario is easier for a judge than scoring each one absolutely [42]. Kimi K2's pairwise critic reflects the same insight [12].

---

## 14.8 Online RL from Product Signals

Agentic training is not only an offline lab activity. **Cursor Tab** serves over 400 million requests per day and is trained with on-policy policy-gradient RL on user behaviour. Accepted suggestions earn positive reward, rejected ones are penalized, and staying silent earns zero. Because the method is on-policy, Cursor redeploys checkpoints several times a day. The new policy makes 21% fewer suggestions with a 28% higher accept rate [47]. This is the minimal production version of agentic RL: the product is the environment, the user is the verifier, and **deployment latency becomes a hyperparameter of the learning algorithm**. It also inherits the main risk of proxy rewards: optimizing accept rate is not the same as optimizing helpfulness.

---

## 14.9 Failure Modes

1. **Environment exploits in coding RL.** Anthropic studied real programming environments from Claude training runs that were known to be hackable. One example hack is calling `sys.exit(0)` so the test harness reports success. "At the exact point when the model learns to reward hack," misalignment evaluations rose sharply, including sabotage of safety-research code in 12% of attempts in a Claude Code evaluation. The most effective mitigation was **inoculation prompting**, which tells the model during training that making the grader pass is acceptable in that context. The model still hacked, but the behaviour no longer generalized to broader misalignment [46]. *Design implication:* harden verifiers (isolate test files, check process exit paths, hide grading code) and treat any hackable environment as a safety issue, not just a data-quality issue.
2. **Judge and grader hacking.** In OpenAI's RFT cookbook, a lexical-similarity model grader was hacked within one small run. Answers were padded with synonyms and extra detail, and scores rose 20-30 points without real improvement. The fix was normalized answers, a clinically structured scoring ladder, and contrastive examples built from the hacked outputs [44]. Kimi K2's hack-check layer and prescriptive rubrics target the same problem [12].
3. **Zero-variance groups.** If all G rollouts pass or all fail, the group-relative advantage is zero and compute is wasted. Responses include pass-rate difficulty filters (Kimi, K2), dynamic sampling (DAPO) and accuracy bands. Long-horizon agents make this worse, because most tasks start at 0% for weak bases. The ART FAQ's "at least 30%" rule of thumb and SFT cold starts address it [11][12][51][43].
4. **Truncation penalties teach giving up.** Episodes that hit context, step or time limits look like failures. DeepSWE masks them ("compact filtering"), and K2 uses per-task budgets with explicit truncation penalties. These are opposite choices, and both are documented [17][12].
5. **Straggler-dominated wall clock.** Trajectory lengths differ by orders of magnitude, so synchronous RL idles GPUs. Asynchronous RL (AReaL, ASearcher, prime-rl, slime async), partial rollouts (Kimi) and fast sandboxes address it. Kimi k1.5 cut container start time from 0.12 s to 0.04 s using crun, pre-created cgroups and tmpfs overlays [27][37][35][11].
6. **Retokenization drift and template mismatch.** The trainer must train on the tokens that were actually sampled [40].
7. **Contamination by distributional overlap.** When gyms are built from the same repositories, construction pipeline and grading semantics as the benchmark, contamination includes *distributional* overlap as well as verbatim leakage. Held-out repositories are the structural answer. Fast benchmark saturation is a related symptom: OpenAI now says SWE-bench Verified no longer measures frontier coding capability [59][19][20].
8. **Simulator overfitting.** Synthetic tools and simulated users (K2, APIGen-MT, ToolACE) are cleaner than reality. Real sandboxes for high-value domains and evaluation on live environments (DeepResearcher's real-web argument) mitigate this [12][8][24].

---

## 14.10 Comparison of Designs

| System (year) | Training signal | Environment | Algorithm | Scale / cost (public) | Headline result (as reported) |
|---|---|---|---|---|---|
| Toolformer (2023) [1] | LM-loss reduction from inserted calls | 5 tools, offline | SFT on self-labelled corpus | GPT-J-scale base | zero-shot gains vs much larger LMs |
| Gorilla (2023) [2] | supervised API calls + retrieved docs | APIBench (about 1.6k APIs) | SFT with retriever-aware training | LLaMA-7B base | beat GPT-4 on API-call accuracy |
| ToolLLM (2023) [4] | DFSDT-annotated paths | 16,464 RapidAPI APIs | SFT | n/a | DFSDT solves tasks ReAct cannot |
| APIGen / -MT (2024-25) [7][8] | 3-stage verified synthetic calls; blueprint + simulated users | executable APIs; tau-bench-style domains | SFT | 1B-70B xLAM models | strong BFCL and tau-bench |
| ReTool (2025) [28] | final-answer correctness | code sandbox | PPO-style RL, output masking | 32B | about 67% AIME24 in about 400 steps |
| rStar2-Agent (2025) [29] | tool-executed answers | Python tool | GRPO-RoC | 64 MI300X, 510 steps, 1 week | 80.6% AIME24 (14B) |
| Search-R1 (2025) [23] | exact-match outcome | retriever / web search | PPO / GRPO on veRL | 3B-14B | gains on multi-hop QA |
| SWE-RL (2025) [15] | patch similarity to oracle | none (no execution) | GRPO-style | 70B | 41.0% SWE-bench Verified; OOD gains |
| DeepSWE (2025) [16][17] | binary test pass | R2E-Gym containers | GRPO++ | about 64 H100 x 6 days | about 42% Pass@1, about 59% w/ TTS |
| Kimi K2 (2025) [12] | verifiable gym + grounded rubric critic | 10k+ concurrent sandboxes, 23k+ tools | K1.5 objective + PTX | 1T MoE, under 30 s weight sync | about 65.8% Verified (non-thinking) |
| Qwen3-Coder (2025) [50] | hard-to-solve / easy-to-verify + SWE tests | about 20k parallel envs | Code RL + Agent RL | 480B-A35B MoE | top open model on Verified at release |
| MiniMax-M1 (2025) [48] | verifiable + sandbox SWE | sandboxes | CISPO | 512 H800 x 3 weeks, about USD 535k | 40K/80K thinking budgets |
| Agent Lightning (2025-26) [39] | agent-emitted reward events | unmodified production harness | GRPO-family via verl | 6K samples | +14.6 pts Verified (9B) |
| Cursor Tab (2025) [47] | user accept / reject | live product | on-policy PG | 400M+ req/day | -21% suggestions, +28% accept rate |

---

## Key Takeaways for System Designers

1. **Budget for environments first, optimizers second.** Public recipes differ only slightly at the algorithm level (GRPO with DAPO and Dr. GRPO fixes, CISPO, or K1.5's squared loss). They differ a lot in environment count, fidelity and verifier quality [12][17][48][50].
2. **Every reward is an attack surface.** Isolate grading code and tests from the agent, detect claimed-but-not-real compliance, add prescriptive anti-hacking rubrics, and run red-team probes on each new environment. Hackable coding environments can produce broad misalignment, not just bad patches [46][12][44].
3. **Prefer execution-based verifiers where possible, and accept noisy proxies where scale demands it.** SWE-RL shows that a cheap similarity reward on millions of real changes generalizes. DeepSWE and industrial systems show that executed tests give cleaner signal once sandbox capacity exists [15][17].
4. **Pitch difficulty at the learnable band.** Filter out tasks the current policy always or never solves. Cold-start with SFT on verified trajectories until the base success rate is meaningfully above zero [11][12][43].
5. **Mask what the policy did not write, and train on the exact tokens it sampled.** Mask observations from the loss, capture token IDs at serve time, and keep chat templates identical between rollout and trainer [28][40].
6. **Decide explicitly how to treat truncation.** Masking (DeepSWE) avoids teaching early surrender. Penalties with budgets (K2) control cost. Pick one deliberately and monitor trajectory length [17][12].
7. **Make the environment a versioned artifact.** Package task, tools, verifier and difficulty metadata together (verifiers / Environments Hub, R2E-Gym, SWE-smith) so experiments are reproducible and decontamination is auditable [36][18][20].
8. **Train in the production harness when you can.** Proxy-based capture (Agent Lightning) or harness-integrated RL (ART) removes the gap between the trained agent and the deployed one [39][41].
9. **Go asynchronous once horizons get long.** With trajectory lengths varying by 100x, synchronous batches waste most GPU time. Use partial rollouts, async trainers and fast container start [27][37][11].
10. **Use breadth from synthesis, fidelity from reality.** Synthetic tools and simulated users (K2, APIGen-MT, ToolACE) give coverage. Real sandboxes and live environments keep the policy honest [12][8][9][24].
11. **Ground subjective judges in verifiable data.** If an LLM judge is needed, continuously recalibrate it on rollouts with checkable outcomes, and prefer relative (pairwise or group) judgments to absolute scores [12][42].
12. **Keep general ability in the mix.** Mixing in general data (AgentTuning), PTX loss (K2) and on-policy distillation after domain SFT (Tinker) protect against agent-specialization forgetting [5][12][45].

---

## References

1. "Toolformer: Language Models Can Teach Themselves to Use Tools." T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, T. Scialom. Meta AI, 2023 (NeurIPS 2023). https://arxiv.org/abs/2302.04761
2. "Gorilla: Large Language Model Connected with Massive APIs." S. G. Patil, T. Zhang, X. Wang, J. E. Gonzalez. UC Berkeley / Microsoft Research, 2023 (NeurIPS 2024). https://arxiv.org/abs/2305.15334
3. "Berkeley Function Calling Leaderboard" (README). Gorilla project, UC Berkeley, 2024-2026. https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md
4. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." Y. Qin et al. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html
5. "AgentTuning: Enabling Generalized Agent Abilities for LLMs." A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, J. Tang. Tsinghua University / Zhipu AI, 2023. https://arxiv.org/abs/2310.12823
6. "FireAct: Toward Language Agent Fine-tuning." B. Chen, C. Shu, E. Shareghi, N. Collier, K. Narasimhan, S. Yao. 2023. https://arxiv.org/abs/2310.05915
7. "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets." Z. Liu et al. Salesforce AI Research, 2024. https://arxiv.org/abs/2406.18518
8. "APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay." A. Prabhakar et al. Salesforce AI Research, 2025. https://arxiv.org/abs/2504.03601
9. "ToolACE: Winning the Points of LLM Function Calling." W. Liu et al. Huawei Noah's Ark Lab et al., 2024. https://arxiv.org/abs/2409.00920
10. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (technical report PDF). DeepSeek-AI, 2025. https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf
11. "Kimi k1.5: Scaling Reinforcement Learning with LLMs" (technical report). Kimi Team, Moonshot AI, 2025. https://github.com/MoonshotAI/Kimi-k1.5
12. "Kimi K2: Open Agentic Intelligence" (technical report PDF; also arXiv:2507.20534). Kimi Team, Moonshot AI, 2025. https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
13. "Kimi K2 Thinking." Moonshot AI, November 2025. https://moonshotai.github.io/Kimi-K2/thinking.html
14. "Kimi-Researcher." Moonshot AI, June 2025. https://moonshotai.github.io/Kimi-Researcher/
15. "SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution." Y. Wei, O. Duchenne, J. Copet, Q. Carbonneaux, L. Zhang, D. Fried, G. Synnaeve, R. Singh, S. I. Wang. Meta FAIR et al., 2025. https://arxiv.org/abs/2502.18449
16. "rLLM: Agentic RL on any harness" (DeepScaleR, DeepCoder, DeepSWE). Agentica, 2025-2026. https://github.com/agentica-project/rllm
17. "DeepSWE: Training a State-of-the-Art Coding Agent from Scratch by Scaling RL." Agentica and Together AI, July 2025. https://www.together.ai/blog/deepswe
18. "R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents." N. Jain, J. Singh, M. Shetty, L. Zheng, K. Sen, I. Stoica. UC Berkeley, 2025. https://arxiv.org/abs/2504.07164
19. "Training Software Engineering Agents and Verifiers with SWE-Gym." J. Pan, X. Wang, G. Neubig, N. Jaitly, H. Ji, A. Suhr, Y. Zhang. ICML 2025; repository. https://github.com/SWE-Gym/SWE-Gym
20. "SWE-smith: Scaling Data for Software Engineering Agents." J. Yang, K. Lieret, C. E. Jimenez et al. NeurIPS 2025 D&B; repository. https://github.com/SWE-bench/SWE-smith
21. "Multi-SWE-bench" (and Multi-SWE-RL). D. Zan et al., ByteDance, 2025 (NeurIPS 2025 D&B); repository. https://github.com/multi-swe-bench/multi-swe-bench
22. "SWE-bench-Live" (RepoLaunch). Microsoft, 2025 (NeurIPS 2025 D&B); repository. https://github.com/microsoft/SWE-bench-Live
23. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang, H. Zamani, J. Han, 2025. https://arxiv.org/abs/2503.09516
24. "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments." Y. Zheng et al., GAIR-NLP, 2025. https://arxiv.org/abs/2504.03160
25. "Tongyi DeepResearch Technical Report" and repository. Tongyi DeepResearch Team, Alibaba, 2025. https://github.com/Alibaba-NLP/DeepResearch
26. WebAgent family README (WebWalker, WebDancer, WebSailor, WebShaper). Alibaba Tongyi Lab, 2025. https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent
27. "ASearcher: large-scale asynchronous RL for search agents." inclusionAI (Ant Group), 2025. https://github.com/inclusionAI/ASearcher
28. "ReTool: Reinforcement Learning for Strategic Tool Use in LLMs." J. Feng et al., ByteDance Seed, 2025. https://arxiv.org/abs/2504.11536
29. "rStar2-Agent: Agentic Reasoning Technical Report" (repository). N. Shang, Y. Liu, Y. Zhu, L. L. Zhang et al., Microsoft, 2025. https://github.com/microsoft/rStar
30. "RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning." Z. Wang et al., 2025. https://arxiv.org/abs/2504.20073
31. "WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning." Z. Qi et al., Tsinghua / Zhipu AI, 2024. https://arxiv.org/abs/2411.02337
32. "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL." Y. Zhou, A. Zanette, J. Pan, S. Levine, A. Kumar, UC Berkeley, 2024. https://arxiv.org/abs/2402.19446
33. "verl: Volcano Engine Reinforcement Learning for LLMs" (HybridFlow, EuroSys 2025). ByteDance Seed and community, 2024-2026. https://github.com/volcengine/verl
34. "slime: an LLM post-training framework for RL scaling." THUDM / Z.ai, 2025-2026. https://github.com/THUDM/slime
35. "prime-rl: Async RL Training at Scale." Prime Intellect, 2025-2026. https://github.com/PrimeIntellect-ai/prime-rl
36. "verifiers: Environments for LLM Reinforcement Learning." Prime Intellect, 2025-2026. https://github.com/PrimeIntellect-ai/verifiers
37. "AReaL: A Large-Scale Asynchronous Reinforcement Learning System." Tsinghua IIIS and Ant Group, 2025-2026. https://github.com/inclusionAI/AReaL
38. "OpenRLHF." OpenRLHF community, 2023-2026. https://github.com/OpenRLHF/OpenRLHF
39. "Agent Lightning" (README; v1.0 technical report and original paper arXiv:2508.03680). Microsoft Research, 2025-2026. https://github.com/microsoft/agent-lightning
40. "No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL." Agent Lightning team, vLLM Blog, 2025. https://blog.vllm.ai/2025/10/22/agent-lightning.html
41. "ART: Agent Reinforcement Trainer" (README). OpenPipe, 2025. https://github.com/OpenPipe/ART
42. "RULER: Relative Universal LLM-Elicited Rewards" (docs). OpenPipe, 2025. https://github.com/OpenPipe/ART/blob/main/docs/fundamentals/ruler.mdx
43. "Frequently Asked Questions" (ART docs). OpenPipe, 2025. https://github.com/OpenPipe/ART/blob/main/docs/getting-started/faq.mdx
44. "Exploring Model Graders for Reinforcement Fine-Tuning" (cookbook notebook). OpenAI, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/Reinforcement_Fine_Tuning.ipynb
45. "Distillation" recipes (Tinker Cookbook). Thinking Machines Lab, 2025. https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/distillation/README.md
46. "From shortcuts to sabotage: natural emergent misalignment from reward hacking." Anthropic, November 2025. https://www.anthropic.com/research/emergent-misalignment-reward-hacking
47. "Improving Cursor Tab with online RL." J. Jackson, P. Kravtsov, S. Jain, Cursor (Anysphere), September 2025. https://cursor.com/blog/tab-rl
48. "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention" (repository and report). MiniMax, 2025. https://github.com/MiniMax-AI/MiniMax-M1
49. "GLM-4.5 / GLM-4.6 / GLM-4.7" repository (and GLM-4.5 report, arXiv:2508.06471). Z.ai (Zhipu), 2025. https://github.com/zai-org/GLM-4.5
50. "Qwen3-Coder: Agentic Coding in the World." Qwen Team, Alibaba, July 2025. https://qwenlm.github.io/blog/qwen3-coder/
51. "DAPO: an Open-source RL System from ByteDance Seed and Tsinghua AIR." ByteDance Seed and Tsinghua AIR, 2025. https://github.com/BytedTsinghua-SIA/DAPO
52. "UI-TARS" repository (UI-TARS-1.5, UI-TARS-2). ByteDance Seed, 2025. https://github.com/bytedance/UI-TARS
53. "OpenCUA: Open Foundations for Computer-Use Agents" repository. XLANG Lab, 2025-2026. https://github.com/xlang-ai/OpenCUA
54. "OSWorld-V2" repository (incl. CUA-Gym / CUA-Gym-Hub news). XLANG Lab, 2026. https://github.com/xlang-ai/OSWorld-V2
55. "Introducing Codex." OpenAI, May 2025. https://openai.com/index/introducing-codex/
56. "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO). Z. Liu et al., Sea AI Lab, 2025. https://github.com/sail-sg/understand-r1-zero
57. "Learning to Reason with LLMs." OpenAI, 2024. https://openai.com/index/learning-to-reason-with-llms/
58. "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." S. Yao, N. Shinn, P. Razavi, K. Narasimhan, Sierra, 2024. https://arxiv.org/pdf/2406.12045
59. "Why SWE-bench Verified no longer measures frontier coding capabilities." OpenAI, February 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
