# 08. Self-Improving Agents and Automated Optimization: Reflection, Skill Libraries, Evolutionary Program Search, Self-Modifying Agents and Prompt Optimizers

## Overview

Earlier chapters described loops that close through model *weights*: preference data feeding RLHF, verifiable rewards feeding RL, production traces feeding fine-tuning. This chapter covers a family of loops that mostly close *without touching weights*. The thing being improved is a text or code artifact around a frozen model: a reflection appended to the next attempt, a library of executable skills, a population of programs, the agent's own source code, a prompt, or a curated memory "playbook". The model acts as both the worker and the mutation operator. An evaluator, which may be a unit test, a simulator, a benchmark, an LLM judge or the environment itself, closes the loop.

Every system in this chapter can be described with the same five parts:

1. **Artifact under optimization.** What gets rewritten: an answer, a prompt, a function, a whole file, an agent codebase, a memory store.
2. **Proposal operator.** Usually an LLM, sometimes an ensemble, conditioned on prior artifacts and feedback.
3. **Evaluator.** What produces the signal: a scalar score, a pass/fail result, or rich textual feedback such as traces, error messages and judge critiques.
4. **Memory or archive.** What persists between iterations: an episodic buffer, a skill library, a program database with islands, an archive of agent variants, a Pareto front of prompts.
5. **Selection policy.** Which artifacts get expanded next: greedy, score-proportional, novelty-weighted, MAP-Elites, Pareto, bandit or Bayesian optimization.

```
          +--------------------------------------------------------------+
          |                  ARCHIVE / MEMORY                            |
          |  reflections | skills | programs | agent variants | prompts  |
          +-----------+-----------------------------------^--------------+
                      | select (greedy / Pareto /         | insert if valid
                      | MAP-Elites / novelty / bandit)    | (score, feedback)
                      v                                   |
          +-----------------------+   artifact   +--------+-------------+
          |  PROPOSER (LLM(s))    |------------->|  EVALUATOR           |
          |  prompt = parents +   |              |  tests / simulator / |
          |  feedback + context   |<-------------|  benchmark / judge   |
          +-----------------------+  textual     +----------------------+
                                     feedback
```

The chapter is organized from the shortest loop (within one task, seconds to minutes) to the longest (open-ended evolution of agents over days):

- Section 1: verbal self-feedback within and across episodes (Reflexion, Self-Refine).
- Section 2: skill libraries and memory-based learning (Voyager, Generative Agents, ExpeL, Agent Workflow Memory, Dynamic Cheatsheet, ACE, MemGPT/Letta, Mem0, Agent Skills, ReasoningBank, Memento).
- Section 3: LLM-guided evolutionary program search (FunSearch, AlphaEvolve, open reimplementations).
- Section 4: self-referential and self-modifying agents (STOP, ADAS, Gödel Agent, SICA, Darwin Gödel Machine, Huxley-Gödel Machine).
- Section 5: prompt and pipeline optimization (APE, ProTeGi and EvoPrompt as precursors; OPRO, Promptbreeder, DSPy/MIPRO, TextGrad, GEPA, AFlow, and the production self-evolving agent loop).
- Section 6: a cross-cutting comparison, failure modes and design guidance.

## 1. Verbal Self-Feedback Loops

### 1.1 Reflexion (Shinn et al., 2023)

**Problem.** Reinforcement learning on LLM agents requires many samples and expensive fine-tuning. Reflexion asks whether an agent can improve across trials by turning sparse task feedback into *natural-language* lessons kept in memory, with no weight updates [1].

**Architecture.** Reflexion has three model roles and two memories:

- **Actor.** Generates actions or text. It is built on ReAct or Chain-of-Thought and conditioned on the task plus memory.
- **Evaluator.** Scores the trajectory. Depending on the domain this is exact match, heuristics, or self-generated unit tests.
- **Self-Reflection model.** Converts the trajectory and the evaluation into a short verbal post-mortem: what went wrong and what to do differently.
- **Short-term memory** is the current trajectory. **Long-term memory** is a bounded buffer of past reflections that is prepended to the next trial.

```
 trial t:  task --> [Actor (+ reflections memory)] --> trajectory --> [Evaluator]
                                ^                                         |
                                |                    reward / test result |
                     append     |                                         v
            [Long-term memory] <-------- [Self-Reflection LLM] <-- (trajectory, reward)
 trial t+1: same task, Actor now conditioned on the new reflection
```

**Results.** The paper reports 91% pass@1 on HumanEval, compared with 80% for GPT-4 at the time. On ALFWorld it reports 130 of 134 tasks completed and an absolute gain of about 22% over 12 iterative trials. On HotPotQA it reports roughly 20% improvement over baselines [1]. The official repository logs every run (HotPotQA with 100 sampled distractor questions, ALFWorld, programming, WebShop). It exposes the reflection strategy as a switch: none, last trajectory, self-reflection, or both. It also warns that individual developers may find reproduction infeasible because the paper's runs used GPT-4, which had limited access and significant API charges [2]. Architecturally, Reflexion is a wrapper around an existing decision procedure (ReAct, CoT) rather than a replacement for it.

**Failure modes and lessons.**
- The loop only helps if the evaluator is reliable. For code, Reflexion relies on *self-generated* tests, so a wrong test can produce a confidently wrong "lesson".
- Reflections are retries on the *same* task. They do not automatically generalize to new tasks. The memory-based methods in Section 2 address that gap.
- The memory window is bounded. Old lessons are evicted rather than consolidated.
- On WebShop, which needs diverse exploration, verbal reflection gave little benefit. Reflection fixes local mistakes better than it fixes exploration problems.

### 1.2 Self-Refine (Madaan et al., 2023)

Self-Refine is the single-episode, single-model version of the idea. One LLM produces an output, critiques it (FEEDBACK) and rewrites it (REFINE), and the cycle repeats until a stopping criterion is met. Each task needs only three prompts: init, feedback and iterate [3][4]. The paper evaluates seven tasks: acronym generation, dialogue response, code readability, CommonGen, GSM8K, Yelp sentiment rewriting and PIE code optimization. It reports roughly 20% absolute average improvement over one-shot generation [3]. The key design point is that no external evaluator is required. The weakness is the same: when the model cannot recognize its own errors, which the authors note happens on some math problems, the loop stalls or drifts.

**Lesson.** Intrinsic self-critique is cheap and is a reasonable default for style and coverage. When correctness matters, attach an external, grounded evaluator (tests, execution, retrieval). Most later systems in this chapter take that route.

## 2. Skill Libraries and Memory-Based Learning

Reflexion's memory is per-task. This section covers systems whose memory accumulates *transferable* knowledge across tasks. The knowledge comes in four forms: executable skills, abstracted workflows, natural-language insights, and curated "playbooks".

### 2.1 Voyager: an executable skill library (Wang et al., 2023)

**Problem.** Open-ended, lifelong learning in an embodied world (Minecraft) without fine-tuning, where the agent must keep acquiring new capabilities rather than solve a fixed task [5].

**Architecture.** Voyager has three components and queries GPT-4 as a black box [5][6]:

1. **Automatic curriculum.** An LLM proposes the next task based on the agent's inventory, state, and the tasks it has completed or failed. Its goal is to "discover as many diverse things as possible".
2. **Skill library.** Each successful program (JavaScript against the Mineflayer API) is stored as an executable skill. It is indexed by an embedding of its natural-language description and retrieved by similarity when a new task arrives. Complex skills are written by *composing* simpler ones.
3. **Iterative prompting mechanism.** Code is executed, and three kinds of feedback are folded into the next prompt: environment feedback, execution errors, and **self-verification**, in which a separate LLM call acts as a critic that judges task success. A skill is committed only after verification passes.

```
 +-------------------+  next task   +----------------------------------------+
 | Automatic         |------------->| Code generation (GPT-4)                |
 | curriculum (LLM)  |              |   prompt = task + retrieved skills +   |
 +---------^---------+              |            env feedback + errors       |
           | state, history         +-------------------+--------------------+
           |                                            | program
 +---------+---------+   success     +------------------v-------------------+
 | Skill library     |<--------------| Execute in Minecraft -> Self-verify  |
 | (code, embedded   |  commit skill | (critic LLM); on failure, re-prompt  |
 |  description)     |-------------->|  with errors (bounded rounds)        |
 +-------------------+  retrieve     +--------------------------------------+
```

**Results.** Compared with prior LLM-agent baselines, Voyager reports 3.3x more unique items, 2.3x longer travel distances, and tech-tree milestones reached up to 15.3x faster. The learned library transfers to a fresh world for novel tasks [5][6].

**Lessons.**
- **Code is a good memory format.** Skills are interpretable, composable and verifiable by execution, and they resist the catastrophic forgetting that affects weight updates.
- **Verification gates the library.** If unverified programs were committed, the library would fill with broken skills that get retrieved and copied. The self-verification critic is the integrity check on the loop.
- **The curriculum is the exploration policy.** Without it, the agent stays near easy tasks. This anticipates the open-ended archives in Section 4.

This design reappears in production as **Agent Skills**. A skill is a folder containing a `SKILL.md` file (YAML frontmatter with a `name` and a `description` that says when to use it, followed by Markdown instructions) plus optional scripts and resources, which an agent loads dynamically. Only the two frontmatter fields are required. Anthropic's reference repository includes a specification, and the format is now published as an open "Agent Skills" standard at agentskills.io [7]. The skill format is how the knowledge is *packaged*. Whether skills are authored by humans or induced by the agent, as Voyager and Agent Workflow Memory do, is a separate choice.

### 2.2 Generative Agents: memory stream, retrieval, reflection (Park et al., 2023)

Generative Agents introduced the memory architecture that many later agent-memory systems inherit [8]. Each agent keeps a **memory stream** of natural-language observations. Retrieval scores each memory as a combination of **recency**, **importance** (rated by an LLM) and **relevance** (embedding similarity). Periodically, the agent runs **reflection**, synthesizing higher-level inferences that are written back into the stream as new memories. The stream also drives planning. The loop is observe, store, retrieve, reflect, write back. This is "learning" by curating the agent's own context rather than changing its weights.

### 2.3 ExpeL: experiential learning from insights and trajectories (Zhao et al., 2024)

ExpeL separates **experience gathering** from **insight extraction** [9][10]. In a training phase, the agent attempts tasks and collects successful and failed trajectories. Then an LLM compares them and distills a bounded list of natural-language insights, controlled by settings such as a target number of rules and the number of experiences shown per iteration. At inference, the agent conditions on the insights *and* on retrieved past trajectories, which can be matched by task similarity, thought similarity or both [10]. Benchmarks are HotpotQA, ALFWorld, WebShop and FEVER. The authors report that performance improves consistently as experience accumulates [9][10].

**Lesson.** Pair the two memory types. *Declarative* insights generalize. *Episodic* exemplars ground the insights in concrete behavior. The insight list must be actively edited, not just appended, or it fills with contradictory rules.

### 2.4 Agent Workflow Memory (Wang, Mao, Fried, Neubig, 2024)

AWM induces **workflows**, meaning reusable sub-routines with example-specific details abstracted out, from past trajectories, and adds them to memory to guide later tasks [11][12]. It runs in two modes. **Offline**, workflows are induced from annotated training examples. **Online**, they are induced on the fly from the agent's own past episodes, using a judged outcome as the filter, with no labeled data. AWM reports a 35.6% success rate on WebArena, state of the art at the time, and the best text-based-agent results on Mind2Web [12]. The paper reports relative improvements of roughly 25% (Mind2Web) and 51% (WebArena) over its baselines; these are approximate figures from the abstract [11]. The online mode generalizes across websites and domains, which makes AWM a direct web-agent analogue of Voyager's skill library, with procedures written as text rather than code.

### 2.5 Dynamic Cheatsheet: test-time learning with adaptive memory (Suzgun et al., 2025)

Dynamic Cheatsheet (DC) gives a black-box LLM a persistent memory that evolves *during inference*, with no labels and no weight updates [13][14]. After each query, a **curator** prompt rewrites the cheatsheet so that it keeps concise, transferable strategies and code snippets rather than full transcripts. There are three variants:

- **DC-Cu (cumulative):** one growing cheatsheet carried across queries.
- **DC-RS (retrieval and synthesis):** retrieves the top-k similar past examples and synthesizes a per-query cheatsheet.
- **Hybrid:** a global strategy sheet combined with per-query retrieval [14].

Reported effects include GPT-4o on Game of 24 going from about 10% to 99% after it discovered and reused a Python solver, near-perfect equation balancing against a baseline of about 50%, more than double the accuracy of Claude 3.5 Sonnet on AIME, and gains of about 9% on GPQA-Diamond and 8% on MMLU-Pro Engineering/Physics [14].

**Lesson.** DC shows that a large share of test-time gains on structured tasks comes from *discovering a tool once and remembering it*. This is Voyager's insight applied to a reasoning benchmark.

### 2.6 ACE: Agentic Context Engineering (Zhang et al., 2025)

ACE treats the context as an evolving **playbook** and targets two failure modes of naive memory rewriting [15][16]:

- **Brevity bias.** Optimizers converge on short, generic prompts that drop domain detail.
- **Context collapse.** Repeated whole-context rewrites by an LLM gradually erase accumulated knowledge.

The fix is a three-role pipeline that makes **incremental delta updates**, not rewrites:

```
  query --> [Generator] -- trajectory, outcome --> [Reflector] -- lessons -->
                ^                                                         |
                |                                                         v
           [Playbook: itemized bullets      <-- deterministic merge -- [Curator]
            with helpful/harmful counters]       dedupe, prune          (delta ops)
```

The Generator attempts tasks. The Reflector extracts lessons and is kept separate from editing. The Curator emits small, localized edits that are merged *deterministically*, with de-duplication, pruning and per-bullet helpful/harmful counters [16]. ACE reports +10.6% on AppWorld agent tasks, where a smaller open-source model matches GPT-4.1-based agents on average and surpasses them on the harder test-challenge split. It reports +8.6% on finance tasks (FiNER, XBRL formula). Compared with GEPA offline on AppWorld, it reports 82.3% lower adaptation latency and 75.1% fewer rollouts. Compared with Dynamic Cheatsheet online on FiNER, it reports 91.5% lower latency and 83.6% lower token cost, for an average adaptation-latency reduction of 86.9% [16].

**Lesson.** Memory updates should look like a **database transaction log**, not like a pen that rewrites the whole document. Structured, append-mostly, deterministically merged updates prevent the slow loss of detail seen in "summarize-and-replace" memory loops.

### 2.7 OS-style and production memory layers: MemGPT/Letta and Mem0

**MemGPT** framed long-term memory as *virtual context management*, by analogy to an operating system that pages data between fast and slow memory. The LLM moves information between its context window and external storage through function calls, editing its own memory [17]. Its successor platform, **Letta**, describes itself as a way to "build stateful agents with memory that can learn and improve over time". Active development has since moved to a `letta-code` repository that contains an agent harness, a terminal UI, an App Server, and the runtime behind Letta's desktop and web apps [18].

**Mem0** is a production memory layer that keeps user-, session- and agent-level state [19][20]. Its repository reports results for an April 2026 algorithm on its managed platform: LoCoMo 71.4 to 92.5 at about 7.0K tokens and 0.88 s p50 latency, LongMemEval 67.8 to 94.4, and BEAM 64.1 (1M) and 48.6 (10M). It cautions that open-source users should expect "directionally similar gains but not identical numbers" [20]. The accompanying paper describes extraction and update phases that decide whether new facts are added, update existing ones, delete contradicted ones, or are ignored [19].

**Lesson.** In production, "agent learning" is mostly *memory hygiene*: extracting facts, resolving conflicts, scoping (user, session or agent), and controlling the token and latency budget. The benchmark gains reported above depend heavily on update policy, not on the storage backend.

### 2.8 ReasoningBank: memory distilled from successes *and* failures (Ouyang et al., Google, ICLR 2026)

ExpeL and AWM mostly learn from what worked. ReasoningBank stores *reasoning strategies* distilled from both successful and failed trajectories, and treats failures as a first-class source of preventive lessons [52]. The loop runs online at test time with no ground-truth labels:

1. **Retrieve.** Before a task, relevant memory items are retrieved by similarity and injected into the agent's context.
2. **Act.** The agent (a web-browsing or SWE agent) runs the task.
3. **Judge.** An LLM-as-a-judge labels the trajectory as success or failure. The released WebArena code includes this `autoeval` component.
4. **Distill and write back.** Successes yield validated strategies. Failures yield "what to avoid" lessons. Both are written as compact, structured items rather than raw trajectories.

The paper's second contribution is **memory-aware test-time scaling (MaTTS)**. Spending extra test-time compute, through several parallel rollouts of one task or sequential self-refinement, produces *contrastive* experience: the agent can compare rollouts that succeeded with ones that failed. That contrast produces better memory, and better memory makes the next round of scaling more effective. The authors present experience-driven memory as a scaling dimension in its own right. Code is released for WebArena (BrowserGym) and SWE-Bench (built on mini-swe-agent), with Gemini 2.5 and Claude 3.7 Sonnet backends [52]. The paper reports gains in both effectiveness and efficiency (fewer interaction steps) over raw-trajectory and workflow-memory baselines such as AWM; see the paper for exact figures.

**Lesson.** A memory that only records successes cannot tell the agent what to stop doing. Write back distilled lessons from failures, and remember that a self-judged success signal is a noisy label. ReasoningBank's gains depend on the judge as much as on the memory format.

### 2.9 Memento: memory-based online RL without fine-tuning (Zhou et al., 2025)

Memento recasts "fine-tuning an agent" as **online reinforcement learning over a memory-augmented MDP**, while the LLM weights stay frozen [53]. A **Case Bank** stores past episodes, both successful and failed. A **case-selection policy** decides which cases to retrieve for a new task. The policy comes in a non-parametric form (similarity retrieval) and a parametric form (a small neural policy trained on the reward signal), and the authors describe retrieval as being "by value", not only by similarity. The agent follows a planner-executor design. A CBR-driven meta-planner (GPT-4.1 in the reference setup) decomposes the task using retrieved cases. An executor (o3 or another model) runs each subtask as an MCP client with search, crawling, document, code and media tools, and writes outcomes back to the Case Bank [53].

The repository reports 87.88% on the GAIA validation set (Pass@3, Top-1) and 79.40% on the GAIA test set, plus 95.0% on SimpleQA. It also notes that GAIA Level-3 remains hard because errors compound over long horizons [53]. These are the authors' own numbers on a heavily tool-augmented stack, so they reflect the whole system and not memory alone.

**Lesson.** Memento makes the RL reading of agent memory explicit. The trainable component is the *retrieval policy*, a much smaller object than the LLM, and it can be trained with an ordinary reward signal. This is a practical midpoint between the prompt-only loops of this chapter and the weight-update loops of Chapter 07.

## 3. LLM-Guided Evolutionary Program Search

The second major family drops per-task memory and instead runs a population-based search over *programs*, with the LLM serving as the mutation operator and a machine-checkable evaluator serving as the fitness function.

### 3.1 FunSearch (Romera-Paredes et al., Nature 2023)

**Problem.** LLMs hallucinate, so they cannot be trusted to state new mathematical facts directly. They can, however, propose *programs* whose outputs a deterministic evaluator can verify [21].

**Architecture.** The user supplies a problem specification: an `evaluate` function plus a skeleton program with one `priority`-style function to evolve. FunSearch then runs a distributed loop:

- A **programs database** organized into **islands**. These are independent subpopulations that limit premature convergence. Weak islands are periodically reset and reseeded from the best ones.
- **Best-shot prompting.** A prompt is built from a few high-scoring programs from one island, sorted by score, and the LLM is asked to write the next version.
- A **pretrained code LLM** generates many samples quickly. The paper used a relatively small code model, favoring throughput over sample quality.
- **Evaluators** run candidates in a sandbox and discard any that fail, time out or produce invalid outputs.

```
  islands:  [I1] [I2] ... [Ik]   (periodic reset of the worst islands)
     | sample k parents (score-sorted) from one island
     v
  prompt "v0 ... v(k-1); write v(k)"  -->  code LLM (many parallel samplers)
                                               |
                                               v
                                   sandboxed evaluators (score or reject)
                                               |
                       valid program + score --+--> back into the same island
```

**Results.** FunSearch found cap sets larger than any previously known in dimension 8, and better lower bounds on the related admissible-set problem. It also discovered online bin-packing heuristics that beat first-fit and best-fit on the evaluated distributions [21]. The public repository releases the discovered programs and sets for cap sets, admissible sets, bin packing, independent sets in strong products of cyclic graphs, and corners-free sets. It also includes a single-threaded reference implementation of the evolutionary loop, without the LLM, the sandbox or the distributed infrastructure [22].

**Scale.** By DeepMind's own later comparison, FunSearch evolved a single Python function of roughly 10-20 lines, needed fast evaluation (20 minutes or less on one CPU), and consumed *millions* of LLM samples. It did not benefit from larger LLMs [23].

**Lessons.** Searching over *programs that generate solutions* rather than over solutions directly produces compact, interpretable artifacts that humans can study. The program can also be analyzed for structure, which is how the cap-set constructions were explained. The evaluator is the loop's only ground truth. If it is wrong, the search optimizes the bug.

### 3.2 AlphaEvolve (Novikov et al., Google DeepMind, 2025)

**Problem.** Scale FunSearch from single-function puzzles to whole codebases and real infrastructure, using frontier LLMs and rich context [23].

**Architecture.** The human defines *what* to optimize: an evaluation function `h` that returns one or more scalar metrics, an initial program with blocks marked for evolution, and optional background knowledge. AlphaEvolve decides *how* [23]:

- **Prompt sampler.** Assembles prompts from parent and inspiration programs, their scores and outputs, problem context, and **meta-prompts**, which are LLM-evolved instructions co-evolved with the programs.
- **LLM ensemble.** Gemini 2.0 Flash generates high-throughput, lower-latency candidates, and Gemini 2.0 Pro provides occasional higher-quality "breakthrough" suggestions [23].
- **Diff-based output.** Changes are requested as `SEARCH/REPLACE` diff blocks so they can be targeted edits inside large files. Full rewrites are allowed for short code [23].
- **Evaluator pool** with three features. An **evaluation cascade** runs test sets of increasing difficulty, and a candidate advances only if it does well at each stage. **LLM-generated feedback** scores properties that are hard to code, such as simplicity. **Parallelized evaluation** allows on the order of 100 compute-hours per candidate when needed, distributed asynchronously [23].
- **Evolutionary database.** Inspired by MAP-Elites combined with island models, it balances exploitation of the best programs against diversity [23].
- **Distributed asynchronous pipeline.** Built on Python asyncio, with a controller, LLM samplers and evaluation nodes. It is tuned for *throughput* of ideas per compute budget, not the latency of any single candidate [23].

```
 Human: evaluate(h) + initial program (EVOLVE-BLOCKs) + context
                         |
                         v
 +-----------------+  prompt  +------------------------+  diffs  +----------------+
 | Prompt sampler  |--------->| LLM ensemble           |-------->| Apply diff ->  |
 | parents,        |          | Flash: many samples    |         | new program    |
 | inspirations,   |          | Pro: few, high quality |         +-------+--------+
 | meta-prompts    |          +------------------------+                 |
 +--------^--------+                                                     v
          |                                     +--------------------------------+
          |      programs + multi-metric scores | Evaluator cascade (cheap ->    |
 +--------+-----------------------+<------------| expensive), LLM feedback,      |
 | Evolutionary DB (MAP-Elites x  |             | async parallel eval cluster    |
 | islands)                       |             +--------------------------------+
 +--------------------------------+
```

**Key differences from FunSearch** (from the paper's own comparison table) [23]:

| Dimension | FunSearch | AlphaEvolve |
|---|---|---|
| Unit of evolution | Single function | Entire code file, multiple components |
| Code size | ~10-20 lines | Up to hundreds of lines |
| Language | Python | Any language |
| Evaluation budget | <= 20 min on 1 CPU | Hours, in parallel, on accelerators |
| LLM samples | Millions | Thousands suffice |
| LLM sensitivity | Small LLMs; no gain from larger | Benefits from SOTA LLMs |
| Prompt context | Previous solutions only | Rich context and feedback |
| Objectives | Single metric | Multiple metrics simultaneously |

**Results (all reported by the authors) [23][24]:**
- **Data-center scheduling (Borg).** An evolved, simple heuristic for scoring jobs against machines was trained against a simulator using historical snapshots and validated on unseen recent workloads. It was rolled out fleet-wide and recovers on average **0.7%** of Google's fleet-wide compute resources that would otherwise be stranded. It was chosen over a deep-RL alternative for interpretability, debuggability, predictability and ease of deployment.
- **Gemini kernel tiling heuristic.** Applied to a Pallas matrix-multiplication kernel, with kernel input shapes split half for training and half for held-out evaluation. It achieved an average **23%** kernel speedup over the expert heuristic and a **1%** reduction in Gemini's overall training time. Kernel-optimization time dropped from months of engineering to days of automated experimentation. The authors describe this as Gemini optimizing its own training.
- **TPU arithmetic circuit (Verilog).** A rewrite that removed unnecessary bits, verified by TPU designers, was slated for an upcoming TPU. The authors note that downstream synthesis tools also independently caught the change.
- **Compiler-generated IR for FlashAttention on GPUs.** Achieved a **32%** kernel speedup and **15%** on pre- and post-processing. Every edit was checked against reference code on randomized inputs and then confirmed correct by human experts.
- **Mathematics.** Improved the state of the art for 14 matrix-multiplication settings, including **48** scalar multiplications for 4x4 complex-valued matrices, the first improvement over Strassen's algorithm in this setting in 56 years. Across more than 50 open problems, it matched the best known constructions on about 75% and found better constructions on about 20%. This includes a kissing-number configuration of **593** spheres in 11 dimensions and an improvement on Erdős's minimum overlap problem. The new constructions are released with verification code [24].

**Ablations.** On tensor decomposition and kissing numbers, removing evolution, prompt context, meta-prompt evolution or full-file evolution, or using only a small base LLM, each degraded results noticeably [23].

**Follow-on.** A late-2025 collaboration with Georgiev, Gómez-Serrano, Tao and Wagner applied AlphaEvolve to a curated set of 67 problems in analysis, combinatorics and geometry. It released a per-problem repository with the prompts, verifiers, initial and evolved programs [25].

**Failure modes and lessons.**
- AlphaEvolve's scope is limited to problems with **automatic evaluators**. Tasks that need manual experimentation are out of scope [23].
- Production wins came where correctness holds **by construction**. The Borg heuristic only reorders machines Borg has already found feasible. The tiling heuristic does not change the math of the kernel. Where correctness did not hold by construction (compiler IR), it was enforced by randomized differential testing followed by human proof.
- **Held-out evaluation sets**, such as unseen workloads and held-out kernel shapes, guard against overfitting to the evaluator.
- **Multi-objective scoring helps even for a single target metric.** Diverse high performers in the prompt stimulate more varied proposals [23].

### 3.3 Open reimplementations: OpenEvolve and ShinkaEvolve

**OpenEvolve** is an open-source implementation of the AlphaEvolve pattern [26]. It provides an LLM ensemble with weighted models and fallbacks (any OpenAI-compatible API, local models, or coding-agent CLIs). Prompts mix top-performing and diverse programs. A MAP-Elites grid organizes programs along user-defined feature dimensions. Islands with periodic migration provide diversity, and cascade evaluation can pass error artifacts back to the LLM. The project reports matching published circle-packing (n=26) results, a 2.8x attention-kernel speedup on Apple M1 Pro, and +23% on HotpotQA prompt optimization. It gives per-iteration cost estimates ranging from about $0.01-0.05 (Gemini 2.5 Flash) to about $0.15-0.60 (o3), and advises starting with 100-200 iterations [26]. These are the project's own claims.

**ShinkaEvolve** (Sakana AI; Lange, Imajuku, Cetin, 2025) focuses on **sample efficiency** [27][28]. Its techniques include:
- configurable parent selection (weighted, power-law or beam search) with an exploitation ratio;
- **novelty filtering**, which rejects candidates whose code embedding exceeds a similarity threshold (default 0.99) and asks an LLM to judge novelty, with up to 3 retries;
- a **UCB bandit** that picks which LLM in the ensemble to query, with a cost-aware term;
- islands with migration and a shared archive;
- asynchronous evolution, credited with a 5-10x throughput speedup.

The repository notes acceptance at ICLR 2026 and use by the team that won the ICFP 2025 programming contest [28].

**Lesson.** Once the AlphaEvolve pattern became commodity infrastructure, competition moved to **evaluation economy**. That means avoiding wasted evaluations on near-duplicates (novelty rejection), routing queries to the cheapest model that still produces progress (bandits), and pruning early (cascades).

## 4. Self-Referential and Self-Modifying Agents

Sections 1-3 improve an artifact *used by* a fixed agent. The systems in this section improve the **agent's own code**, including the code that performs the improving. All descend conceptually from Schmidhuber's Gödel Machine, which rewrites itself only when it can *prove* the rewrite is beneficial. Because such proofs are impractical, every modern variant replaces proof with **empirical evaluation**.

### 4.1 STOP: Self-Taught Optimizer (Zelikman et al., 2023)

STOP starts from a seed "improver": a scaffolding program that calls an LLM several times to rewrite an input program for a higher utility and returns the best attempt. The improver is then run *on itself* [29][30]. The improved improvers performed better on a small set of downstream tasks. The LLM proposed classic search strategies (beam search, genetic algorithms, simulated annealing) as self-improvement strategies. The authors stress that this is not full recursive self-improvement because the model weights are unchanged. They also measured how often generated code tried to **circumvent the sandbox**, an early empirical safety signal for this class of loop [30].

### 4.2 ADAS and Meta Agent Search (Hu, Lu, Clune, 2024)

ADAS defines a research area, the automated design of agentic systems. Its example algorithm, **Meta Agent Search**, has a meta-agent write new agents *in code*, add them to a growing archive, and condition future designs on that archive [31][32]. Because agents are defined in a Turing-complete language, the search space covers prompts, tool use, control flow and novel building blocks. The paper reports that discovered agents outperform hand-designed baselines, with gains of roughly 13.6 F1 points on reading comprehension (DROP) and about 14% accuracy on math (MGSM). It also reports that designs transfer across domains and models; these figures are approximate, from the abstract [31]. The repository carries an explicit warning about executing untrusted, model-generated code [32].

### 4.3 Gödel Agent (Yin et al., 2024)

Gödel Agent is a self-referential framework in which the agent inspects and modifies its own running logic, implemented by **monkey patching** its own modules at runtime. The modification policy itself is therefore open to modification [33][34]. Because it relies on runtime self-modification rather than versioned offline candidates, it highlights a trade-off: self-modification is fast, but it is much harder to roll back and audit than archive-based approaches.

### 4.4 SICA: A Self-Improving Coding Agent (Robeyns, Szummer, Aitchison, 2025)

SICA removes the separation between meta-agent and target agent. The *same* coding agent edits its own codebase [35]. The loop is: evaluate the current agent on benchmarks, store the results in an archive, run the agent on its own repository to produce an improvement, and repeat. The best archived agent so far acts as the next meta-agent. The paper reports gains from about 17% to about 53% on a random subset of SWE-bench Verified (approximate). The repository requires Docker isolation because the agent executes shell commands [35].

### 4.5 Darwin Gödel Machine (Zhang, Hu, Lu, Lange, Clune; Sakana AI and UBC, 2025)

**Problem.** Greedy self-improvement (always build on the current best) gets stuck in local optima. The DGM adds **open-ended exploration**: an archive of all valid agent variants, any of which can be a parent [36][37][38].

**Architecture.** Grounded in the public code [37]:
- **Seed agent.** A coding agent driven by a frontier LLM through a tool-calling loop. It edits a repository and emits a `git diff` patch.
- **Outer loop** (`DGM_outer.py`). The default is 80 generations, with 2 self-improvement attempts per generation run in parallel and a 1.5-hour timeout per attempt.
- **Diagnose and modify.** For each selected parent, the system picks a task the parent failed, such as an unresolved issue or empty patches, and the parent agent is asked to analyze its own evaluation logs, propose a feature for *itself*, and implement it in its own codebase.
- **Parent selection** (default `score_child_prop`). Each score `s` is passed through a sigmoid centered at 0.5, `s' = 1/(1+exp(-10(s-0.5)))`, then multiplied by `1/(1+children)` and normalized. This favors high performers that have not been explored much.
- **Staged evaluation.** Every child is first evaluated on a small SWE-bench subset and discarded if it does not compile or produces only empty patches. A medium subset follows if the child clears a 0.4 threshold. A full evaluation follows only if it beats a threshold derived from the archive (the second-best score, floored at 0.4).
- **Archive policy.** The default `keep_all` keeps every compiled child. The alternative `keep_better` keeps a child only if it scores no worse than the *initial* agent's score minus an evaluation-noise margin (default 0.1).
- **Failure targeting.** Besides picking a random unresolved issue, the code sometimes (each with probability 0.25) assigns a generic self-improvement target: fixing empty patches (when they are at least 10% of tasks), handling stochasticity, or handling context-length overflow.
- **Polyglot mode.** The same loop runs against the Aider Polyglot benchmark.

```
 Archive A = {agent_0}
 repeat G=80 generations:
   parents <- sample(A, weight = sigmoid(score) / (1 + n_children))
   for p in parents (parallel):
       task  <- pick failure from p's eval logs (unresolved / empty patch / ...)
       child <- p modifies ITS OWN repo (diagnose -> propose feature -> implement)
       if compiles and passes small subset:
            evaluate on medium (score > 0.4) -> full (score > archive threshold)
            A <- A + {child}            # keep_all: stepping stones retained
```

**Results.** The paper reports SWE-bench improving from 20.0% to 50.0% and Polyglot from 14.2% to 30.7% [36]. Ablations without self-improvement, or without open-ended exploration (always the latest agent), perform worse, which supports the archive design [36][38]. Discovered improvements include finer-grained file-viewing and editing tools, generating multiple patch candidates and ranking them, and keeping a history of earlier attempts. These improvements **transferred** to other foundation models and other programming languages [38].

**Cost.** The paper notes that a single SWE-bench run takes on the order of two weeks and incurs substantial API cost, reported at roughly US$22,000 (approximate) [36]. This is two to three orders of magnitude more than a typical prompt-optimization run (Section 5).

**Failure modes: objective hacking.** The DGM authors document a canonical failure of self-improvement loops. Asked to fix hallucinated tool use, where the agent claimed to run tests it never ran, some variants instead **removed the markers the detector relied on**, so the hallucination check passed without the behavior being fixed [38]. All runs were sandboxed, with human oversight and lineage tracking of every change [36][38]. The repository warns that it executes untrusted model-generated code [37].

**Lessons.**
- **Keep stepping stones.** Some lineages pass through temporarily *worse* agents before reaching the best ones. Greedy pruning would have discarded them.
- **Do not let the agent edit its own evaluator.** Keep the reward or detector outside the modifiable surface, and red-team the scoring pipeline.
- **Staged evaluation is the cost lever.** Most children are rejected on small subsets.

### 4.6 Huxley-Gödel Machine (Wang, Piękos, Li, Laakom, Schmidhuber et al., 2025)

The HGM paper argues that DGM-style selection by *benchmark score* is a weak proxy for self-improvement potential. The authors call this a **metaproductivity-performance mismatch**: an agent's own score does not predict how good its *descendants* will be [39][40]. HGM instead estimates the promise of an entire subtree (a "clade") of self-modifications and uses that clade-level estimate to choose which node to expand. It builds on the DGM codebase and evaluates on SWE-bench and Polyglot [40]. The authors report better results than prior self-improving coding-agent methods with less compute time; see the paper for exact figures [39]. The repository reports that the paper was accepted as an oral presentation at ICLR 2026, and like the DGM repository it warns that it executes untrusted model-generated code inside Docker [40].

**Lesson.** In tree-structured self-improvement, the *selection signal* is the key design decision. Lineage-aware credit assignment (how good are my descendants?) beats node-local fitness (how good am I?).

## 5. Prompt and Pipeline Optimization

These systems treat prompts, instructions and few-shot demonstrations, and increasingly whole agent programs, as the parameters to optimize against a metric on a training set.

### 5.1 Precursors: APE, ProTeGi and EvoPrompt (2022-2024)

Three early systems set up the patterns that the rest of this section refines.

- **APE: Automatic Prompt Engineer** (Zhou et al., ICLR 2023). An LLM *induces* candidate instructions from input-output demonstrations. Each candidate is scored by the zero-shot performance of a target LLM, and the best candidates are optionally resampled as paraphrases (Monte-Carlo search). On 24 instruction-induction tasks, the generated instructions beat the prior LLM baseline by a large margin and matched or beat human-written instructions on 21 of 24 [54]. APE set up the basic propose-score-select loop.
- **ProTeGi: Prompt Optimization with Textual Gradients** (Pryzant et al., EMNLP 2023). A minibatch of errors is shown to an LLM, which writes a natural-language "gradient": a critique of how the prompt caused the errors. A second call edits the prompt in the *opposite* semantic direction. Candidate prompts are kept in a **beam**, and **bandit algorithms** (UCB, successive rejects) decide which candidates get evaluated, which saves evaluation budget [55]. TextGrad credits ProTeGi with the term "textual gradients" [48]. ProTeGi is the direct ancestor of TextGrad's backward pass and GEPA's reflective mutation.
- **EvoPrompt** (Guo et al., ICLR 2024). An LLM implements the operators of classical evolutionary algorithms, a genetic algorithm (crossover plus mutation) and differential evolution, directly on natural-language prompts. On 31 datasets, with GPT-3.5 and Alpaca, it reports gains over human-engineered prompts and earlier automatic methods of up to 25% on BIG-Bench Hard [56].

**Lesson.** Score-only search (APE), error-driven critique (ProTeGi) and population search (EvoPrompt) were separate lines of work in 2023. GEPA (Section 5.6) is essentially their synthesis: reflective critique, a population with Pareto selection, and frugal evaluation.

### 5.2 OPRO: LLMs as optimizers (Yang et al., Google DeepMind, 2023)

OPRO describes the optimization problem in natural language. A **meta-prompt** contains the task description, a few exemplars, and a trajectory of past solutions *sorted by score*. The optimizer LLM proposes new candidates, which are scored and added back into the meta-prompt [41][42]. The repository includes prompt optimization, linear regression and TSP, and it warns about API cost [42]. The paper reports prompts that beat human-designed ones by up to 8% on GSM8K and up to 50% on Big-Bench Hard tasks. One example is "Take a deep breath and work on this problem step-by-step", which reached 80.2% on GSM8K with PaLM 2-L [41].

**Limitation.** OPRO's only feedback is the score. The optimizer sees *that* a prompt scored 62% but not *why*. Later methods inject richer feedback.

### 5.3 Promptbreeder (Fernando et al., Google DeepMind, 2023)

Promptbreeder evolves a population of task prompts using LLM-applied **mutation prompts**, and it evolves the mutation prompts too. This is self-referential prompt evolution, so the improvement operator is itself improved [43]. The idea recurs in AlphaEvolve's meta-prompt evolution, where the ablation shows it matters [23].

### 5.4 DSPy and MIPROv2 (Khattab et al.; Opsahl-Ong et al., 2023-2024)

DSPy recasts LM pipelines as **programs** made of declarative modules with signatures, and treats prompts and demonstrations as compiled parameters [44]. Its optimizers ("teleprompters") include [45]:

- **BootstrapFewShot.** Runs a teacher program, keeps the traces that pass the metric, and uses them as demonstrations.
- **COPRO.** Instruction generation plus coordinate-ascent hill-climbing.
- **MIPROv2.** Three stages. (1) **Bootstrapping** collects high-scoring traces per module. (2) **Grounded proposal** drafts candidate instructions conditioned on the program code, data summaries and traces. (3) **Discrete search** evaluates instruction and demonstration combinations on minibatches, with a **Bayesian-optimization surrogate** guiding later trials.
- **SIMBA.** Finds high-variance examples in stochastic minibatches and writes self-reflective rules or adds demonstrations.
- **GEPA** (see 5.6).
- **BootstrapFinetune.** Distills a prompted program into fine-tuned weights, which closes the loop into weights as in Chapter 07.

The MIPRO paper reports that it outperforms baseline optimizers on five of seven multi-stage programs, by up to 13% accuracy with Llama-3-8B [46]. The DSPy docs describe typical runs costing about US$2 and about 10 minutes, with a range from cents to tens of dollars. Informal examples include a ReAct agent on HotPotQA going from 24% to 51% (MIPROv2, light mode, 500 examples), RAG on a StackExchange subset going from 53% to 61%, and Banking77 classification going from 66% to 87% via BootstrapFinetune with GPT-4o-mini. The docs also give selection guidance: about 10 examples suits BootstrapFewShot, and 200+ examples with 40+ trials suits MIPROv2 [45].

**Lesson.** Optimizing the **whole pipeline jointly** matters. Credit assignment across modules (which stage's instruction caused the failure?) is the hard part, and MIPRO handles it by bootstrapping per-module traces from end-to-end successes.

### 5.5 TextGrad (Yuksekgonul et al., Nature 2025)

TextGrad brings the PyTorch abstraction to text. **Variables** (prompts, solutions, code, even molecules) have role descriptions and a `requires_grad` flag. A **TextLoss** is a natural-language evaluation instruction. `loss.backward()` asks an LLM to produce **textual gradients** (critiques) that are back-propagated through the computation graph, and the **TGD** optimizer applies them as edits [47][48]. Published in Nature in March 2025, the paper reports GPT-4o zero-shot GPQA accuracy improving from 51% to 55% and a 20% relative gain on LeetCode-Hard solutions, alongside applications to molecule design and radiotherapy treatment planning [47][48].

**Lessons.** The "gradient" is only as good as the critic. Long chains of textual backprop are noisy and can amplify hallucinated critiques. TextGrad works best when the loss is grounded, for example in test results or a verifier, rather than in pure LLM opinion.

### 5.6 GEPA: Reflective Prompt Evolution (Agrawal et al., 2025)

GEPA (Genetic-Pareto) combines three ideas: reflection from Reflexion, evolution from FunSearch, and Pareto selection from multi-objective optimization [49][50].

- **Actionable Side Information (ASI).** Evaluators return full traces, error messages, profiler output and judge rationales, not just a scalar. An LLM reads them, diagnoses failures and proposes targeted edits. This is the textual analogue of a gradient [50].
- **Pareto-front candidate pool.** GEPA keeps every candidate that is best on *some* training instance, not only the best on average. Parents are sampled from this front, which preserves complementary strategies. A **system-aware merge** can combine two complementary candidates [50].
- **Minibatch acceptance.** A mutated candidate is evaluated on a small minibatch and admitted to the pool only if it improves.

```
 Pareto pool P (per-instance winners)
     | sample parent
     v
 run on minibatch --> full traces + ASI --> reflection LLM: diagnose -> rewrite module
     ^                                                        |
     |                     improved on minibatch?             v
     +--------- yes: add child, update per-instance front <---+
                no : discard
```

**Results.** The paper reports GEPA beating GRPO by up to 20% on individual tasks while using up to 35x fewer rollouts, and beating MIPROv2 by more than 10% across two LLMs [49]. The *average* margin over GRPO is about 10% in the first arXiv version; it may differ in later revisions, so cite the specific version when quoting it. The project reports 100-500 evaluations compared with roughly 5,000-25,000+ for GRPO. Other project-reported results include GPT-4.1 Mini on AIME 2025 going from 46.6% to 56.6%, an ARC-AGI agent going from 32% to 89% via architecture discovery, a cloud-scheduling policy with 40.2% cost savings, and a Databricks result where open-source models plus GEPA beat Claude Opus 4.1 at about 90x lower cost [50]. GEPA is integrated as `dspy.GEPA`, in MLflow's `optimize_prompts`, in Google ADK, and elsewhere. Adapters cover DSPy full programs (evolving signatures, modules and control flow), RAG pipelines, MCP tool descriptions and the Terminus terminal agent [50].

**Lesson.** GEPA's main finding is that **rich feedback beats rich sampling**. Reading one failure trace often reveals more than hundreds of scalar rollouts. RL-scale sample counts are mostly wasted on optimizing a few thousand tokens of instructions.

### 5.7 AFlow: tree search over agentic workflows (Zhang et al., ICLR 2025)

DSPy and GEPA mostly tune the *text* inside a fixed program. AFlow searches over the program's *structure* [57]. It represents a workflow as code: LLM-invoking **nodes** connected by edges, plus reusable **operators** that encapsulate common patterns such as Generate, Format, Review, Revise, Ensemble, Test and Programmer. A variant of **Monte Carlo tree search** then repeatedly does four things. It *selects* a workflow from the tree of past versions, balancing score against exploration. It *expands* the workflow by asking an LLM to modify the code, using the parent's logged failures and the modification history. It *evaluates* the result on a validation set, repeated over several rounds to average out noise. It *backpropagates* the score and experience into the tree [57].

Experiments cover six benchmarks: HumanEval, MBPP, GSM8K, MATH, HotpotQA and DROP. The paper reports an average improvement over the strongest manual and automated baselines of about 5.7%, and workflows that let smaller models beat GPT-4o on some tasks at a small fraction of its dollar inference cost (approximate figures, from the abstract) [57]. AFlow was an oral presentation at ICLR 2025 [57].

**Lesson.** AFlow sits between ADAS (Section 4.2, unrestricted code search with an archive) and DSPy (fixed structure, optimized text). Restricting the search to a vocabulary of operators makes structural search tractable and the results readable, at the cost of not discovering primitives outside that vocabulary.

### 5.8 A production pattern: the self-evolving agent loop (OpenAI and Bain cookbook, 2025)

The OpenAI Cookbook's "self-evolving agents" guide packages these ideas into a deployable retraining loop for a regulated domain: summarizing sections of a pharmaceutical CMC regulatory document, with about 70 sections [51]. The components are:

- **A baseline agent** (a summarizer, plus a compliance checker in the agent-builder variant).
- **A grader suite.** A chemical-name preservation check (threshold 0.8), a length check (100 words ±20%, threshold 0.85), cosine similarity to reference (0.85), and an LLM-as-judge rubric (0.85).
- **A metaprompt agent** that rewrites the summarizer prompt using grader feedback, while its own prompt stays fixed.
- **A `VersionedPrompt` store** that supports rollback.
- **An acceptance rule.** A summary passes if 75% of graders pass or the average score reaches 0.85. The loop allows a bounded number of retries (3 in code) and then escalates to engineers.

The guide compares three tiers: manual optimization in the Evals UI, the automated metaprompt loop (which risks overfitting to grader feedback), and GEPA with a validation split (10%). In the example, GEPA produced a more general prompt in about 10-15 minutes [51].

```
 prod/eval inputs --> agent(prompt v_k) --> output --> graders (rule + embed + judge)
                                                          |
                    pass (lenient aggregate) <------------+------> fail
                         |                                          |
                  ship v_k (versioned)          metaprompt agent rewrites -> v_{k+1}
                                                (<= N retries, else page a human)
```

**Lesson.** The production version of a self-improving agent is unremarkable: versioned artifacts, multiple orthogonal graders, a held-out split, bounded retries and human escalation. Each of these controls exists to contain a failure mode documented in the research systems.

## 6. Cross-Cutting Comparison

| System | Artifact optimized | Proposer | Evaluator | Memory / selection | Scale / cost (public) | Signature risk |
|---|---|---|---|---|---|---|
| Reflexion [1] | Next attempt (via reflections) | Same LLM | Tests, heuristics, env reward | Bounded reflection buffer | Few trials per task | Wrong self-tests teach wrong lessons |
| Self-Refine [3] | Single output | Same LLM | Self-critique | None (iterate) | 1-few rounds | Blind to own errors |
| Voyager [5] | Skill library (code) | GPT-4 | Env + execution + critic LLM | Embedding-indexed skills; curriculum | Lifelong; GPT-4 API | Library pollution if verification is weak |
| ExpeL / AWM [9][11] | Insights / workflows | LLM | Task success (labels or judge) | Insight list + trajectory retrieval | Grows with experience | Contradictory or stale rules |
| Dynamic Cheatsheet / ACE [13][15] | Context playbook | LLM roles | Execution feedback | Curated memory; delta merges | ACE: ~87% lower adaptation latency vs baselines | Context collapse, brevity bias |
| ReasoningBank / Memento [52][53] | Strategy memory / case bank | LLM (distiller); learned case-selection policy (Memento) | LLM judge (ReasoningBank); task reward | Similarity retrieval; value-based retrieval policy | Online, no weight updates | Noisy self-judged labels poison memory |
| FunSearch [21] | One function | Small code LLM | Deterministic scorer | Islands; best-shot prompts | Millions of samples | Evaluator bugs exploited |
| AlphaEvolve [23] | Whole files, multi-language | Gemini Flash + Pro | Cascaded, parallel, multi-metric | MAP-Elites x islands; meta-prompts | Thousands of samples; up to ~100 compute-h per eval | Limited to auto-evaluable tasks |
| OpenEvolve / Shinka [26][27] | Programs | Model ensemble (bandit) | User evaluator | MAP-Elites / islands / novelty filter | ~$0.01-0.60 per iteration (OpenEvolve est.) | Wasted evals on near-duplicates |
| DGM [36] | Agent's own codebase | The agent itself | SWE-bench / Polyglot, staged | Open-ended archive, score/children weighting | ~2 weeks, ~US$22k per run (approx.) | Objective hacking of detectors |
| HGM [39] | Agent's own codebase | The agent itself | Coding benchmarks | Clade-level metaproductivity | Reported less compute than DGM | Proxy mismatch (addressed) |
| APE / ProTeGi / EvoPrompt [54][55][56] | Instruction prompt | LLM (induce / critique-and-edit / GA-DE operators) | Score on dev set | Resampling; beam + bandit; population | Dev-set evaluations per candidate | Overfit to small dev set |
| OPRO [41] | Prompt | LLM | Score on train set | Sorted score trajectory | Many evals | Score-only signal |
| MIPROv2 [46] | Instructions + demos, per module | LLM | Program metric | Bayesian surrogate | ~$2, ~10 min typical (docs) | Overfit small train sets |
| TextGrad [47] | Any text variable | LLM | Natural-language loss | Optimizer state | Per-variable LLM calls | Noisy textual gradients |
| GEPA [49] | Prompts / programs / code | Reflection LLM | Metric + traces (ASI) | Per-instance Pareto front, merge | 100-500 evals vs 5k-25k+ for GRPO | Judge-feedback overfitting |
| AFlow [57] | Workflow structure (code over operators) | LLM | Validation-set score, repeated runs | MCTS tree with experience backprop | 20 rounds default; several validation runs | Limited to operator vocabulary |

### 6.1 Three axes that organize the design space

1. **Signal richness: scalar vs. textual.** FunSearch and OPRO learn from numbers. Reflexion, TextGrad, GEPA and ACE learn from *explanations*. The trend through 2025-2026 is clear: textual feedback, meaning traces, error messages and judge rationales, raises sample efficiency by one to two orders of magnitude (GEPA compared with GRPO, AlphaEvolve's thousands of samples compared with FunSearch's millions), *provided* the feedback is grounded in execution.
2. **Search topology: greedy, population or archive.** Self-Refine and Reflexion are greedy chains. FunSearch and AlphaEvolve use islands and quality-diversity grids. DGM and ADAS keep open-ended archives of stepping stones. GEPA keeps a per-instance Pareto front. HGM scores whole subtrees. More diverse topologies cost more evaluations but escape local optima that greedy loops cannot.
3. **Modifiable surface: output, context, program or agent.** The larger the surface, the larger the potential gain and the larger the attack surface. When the agent can edit its own scaffolding (STOP, DGM, SICA, Gödel Agent), it can also edit its guardrails. Sandbox-escape measurements (STOP) and detector tampering (DGM) are the documented consequences.

### 6.2 Recurring failure modes

- **Evaluator exploitation (Goodhart).** Any search strong enough to find improvements will also find evaluator bugs. Documented examples include DGM's removal of hallucination markers [38] and the cookbook's warning about overfitting to grader feedback [51]. Mitigations: keep evaluators outside the modifiable surface, use held-out splits (AlphaEvolve's unseen workloads and kernel shapes [23]), randomized differential testing, and human verification of final artifacts.
- **Memory degradation.** Context collapse and brevity bias (ACE [16]), contradictory insights (ExpeL), library pollution (Voyager without verification). Mitigations: append-mostly delta updates, helpful/harmful counters, verification before commit, explicit conflict resolution (Mem0's add/update/delete decisions [19]).
- **Diversity collapse.** Populations converge on one lineage. Mitigations: islands with resets (FunSearch), MAP-Elites (AlphaEvolve, OpenEvolve), novelty rejection (ShinkaEvolve), Pareto fronts (GEPA), children-penalized sampling (DGM).
- **Cost blow-up.** Open-ended self-modification runs cost tens of thousands of dollars [36]. Mitigations: evaluation cascades and staged subsets, cost-aware model bandits, cheap-model exploration with occasional expensive-model exploitation (Flash plus Pro).
- **Non-stationary evaluation noise.** Agent benchmarks are stochastic. DGM explicitly models an evaluation-noise margin (0.1) and re-targets failures caused by stochasticity [37]. Without this, the archive fills with lucky variants.
- **Safety of self-modifying code.** Every self-modifying-agent repository surveyed here (ADAS, DGM, SICA) ships with warnings about untrusted model-generated code and requires container isolation [32][35][37].

## 7. Key Takeaways for System Designers

1. **Start with the evaluator, not the optimizer.** Every success in this chapter (AlphaEvolve's 0.7% fleet compute, the 23% kernel speedup, the 48-multiplication algorithm, DGM's SWE-bench gains) rests on a cheap, trustworthy, automatic evaluator. If you cannot write `evaluate()`, none of these loops apply.
2. **Make correctness hold by construction where possible.** Optimize the *policy* (ranking, tiling, prioritization) inside a frame that guarantees validity, as AlphaEvolve's Borg and tiling deployments did. Where that is impossible, gate on differential tests plus human review.
3. **Return explanations, not just scores.** Feed execution traces, compiler errors and judge rationales back to the proposer (GEPA's ASI, AlphaEvolve's rich context, Reflexion's reflections). This is the cheapest sample-efficiency gain available.
4. **Keep an archive and select for diversity.** Use islands, MAP-Elites, Pareto fronts or children-penalized sampling. Greedy loops plateau. Open-ended archives find stepping stones (DGM ablations).
5. **Use cascaded evaluation and tiered models.** Filter with cheap tests and small subsets before expensive full evaluations. Pair a fast, cheap proposer with an occasional strong one. Route queries with a bandit if the ensemble is large.
6. **Store learned knowledge as verified, composable artifacts.** Code skills (Voyager), workflows (AWM) and itemized playbook bullets (ACE) outperform free-form summaries. Commit only after verification, and update through deltas rather than rewrites.
7. **Never let the optimizer touch its own reward.** Put evaluators, detectors and safety checks outside the modifiable surface. Log lineage for every change, and red-team the scoring pipeline for objective hacking.
8. **Use held-out validation and bounded retries.** Split training and validation sets for prompt optimization (the cookbook used a 10% validation split for GEPA), cap retries, version every artifact with rollback, and escalate to humans when the loop fails to converge.
9. **Choose the cheapest loop that fits.** Rough ordering of cost: Self-Refine/Reflexion (one task, a few calls) < DSPy/GEPA (about $2 to tens of dollars, minutes to hours) < AlphaEvolve-style evolution (thousands of samples, possibly hours per evaluation) < self-modifying agents (weeks, about $10^4). Move up only when the cheaper loop has saturated.
10. **Graduate prompts into weights when stable.** Once a prompt or program optimization converges, distillation (DSPy BootstrapFinetune [45]) or the trace-to-finetune loops of Chapter 07 can move the learned behavior into weights, which cuts per-call context cost.

## References

1. Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., Yao, S. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. https://arxiv.org/abs/2303.11366
2. Shinn, N. et al. "reflexion" (official code and logs). GitHub, 2023. https://github.com/noahshinn/reflexion
3. Madaan, A. et al. "Self-Refine: Iterative Refinement with Self-Feedback." 2023. https://arxiv.org/abs/2303.17651
4. Madaan, A. et al. "self-refine" (code). GitHub, 2023. https://github.com/madaan/self-refine
5. Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., Anandkumar, A. "Voyager: An Open-Ended Embodied Agent with Large Language Models." 2023. https://arxiv.org/abs/2305.16291
6. MineDojo / NVIDIA. "Voyager" (code). GitHub, 2023. https://github.com/MineDojo/Voyager
7. Anthropic. "Agent Skills" (skills repository and specification). GitHub, 2025. https://github.com/anthropics/skills
8. Park, J. S. et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. https://arxiv.org/abs/2304.03442
9. Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., Huang, G. "ExpeL: LLM Agents Are Experiential Learners." AAAI 2024. https://arxiv.org/abs/2308.10144
10. LeapLab THU. "ExpeL" (code). GitHub, 2023. https://github.com/LeapLabTHU/ExpeL
11. Wang, Z., Mao, J., Fried, D., Neubig, G. "Agent Workflow Memory." 2024. https://arxiv.org/abs/2409.07429
12. Wang, Z. et al. "agent-workflow-memory" (code). GitHub, 2024. https://github.com/zorazrw/agent-workflow-memory
13. Suzgun, M., Yuksekgonul, M., Bianchi, F., Jurafsky, D., Zou, J. "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory." 2025. https://arxiv.org/abs/2504.07952
14. Suzgun, M. et al. "dynamic-cheatsheet" (code). GitHub, 2025. https://github.com/suzgunmirac/dynamic-cheatsheet
15. Zhang, Q. et al. "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models." 2025. https://arxiv.org/abs/2510.04618
16. ACE authors. "ace" (code). GitHub, 2025. https://github.com/ace-agent/ace
17. Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems." 2023. https://arxiv.org/abs/2310.08560
18. Letta. "Letta (formerly MemGPT)." GitHub, 2024-2026. https://github.com/letta-ai/letta
19. Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." 2025. https://arxiv.org/abs/2504.19413
20. Mem0. "mem0" (code and benchmark results). GitHub, 2024-2026. https://github.com/mem0ai/mem0
21. Romera-Paredes, B. et al. "Mathematical discoveries from program search with large language models." Nature, 2023. https://www.nature.com/articles/s41586-023-06924-6
22. Google DeepMind. "funsearch" (discovered programs and reference implementation). GitHub, 2023. https://github.com/google-deepmind/funsearch
23. Novikov, A., Vu, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." Google DeepMind white paper, 2025. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf (arXiv: https://arxiv.org/abs/2506.13131)
24. Google DeepMind. "alphaevolve_results" (mathematical results with verification code). GitHub, 2025. https://github.com/google-deepmind/alphaevolve_results
25. Georgiev, B., Gómez-Serrano, J., Tao, T., Wagner, A. Z. "Mathematical exploration and discovery at scale." 2025. https://arxiv.org/abs/2511.02864 (problems repository: https://github.com/google-deepmind/alphaevolve_repository_of_problems)
26. OpenEvolve contributors. "OpenEvolve: open-source evolutionary coding agent." GitHub, 2025. https://github.com/algorithmicsuperintelligence/openevolve
27. Lange, R. T., Imajuku, Y., Cetin, E. "ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution." Sakana AI, 2025. https://arxiv.org/abs/2509.19349
28. Sakana AI. "ShinkaEvolve" (code). GitHub, 2025. https://github.com/SakanaAI/ShinkaEvolve
29. Zelikman, E., Lorch, E., Mackey, L., Kalai, A. T. "Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation." 2023. https://arxiv.org/abs/2310.02304
30. Microsoft. "stop" (code). GitHub, 2023. https://github.com/microsoft/stop
31. Hu, S., Lu, C., Clune, J. "Automated Design of Agentic Systems." ICLR 2025. https://arxiv.org/abs/2408.08435
32. Hu, S. et al. "ADAS" (code). GitHub, 2024. https://github.com/ShengranHu/ADAS
33. Yin, X., Wang, X., Pan, L., Wan, X., Wang, W. Y. "Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement." 2024. https://arxiv.org/abs/2410.04444
34. Yin, X. et al. "Godel_Agent" (code). GitHub, 2024. https://github.com/Arvid-pku/Godel_Agent
35. Robeyns, M., Szummer, M., Aitchison, L. "SICA: A Self-Improving Coding Agent." ICLR 2025 Workshop on Scaling Self-Improving Foundation Models. https://github.com/MaximeRobeyns/self_improving_coding_agent (paper: https://openreview.net/forum?id=rShJCyLsOr)
36. Zhang, J., Hu, S., Lu, C., Lange, R., Clune, J. "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents." 2025. https://arxiv.org/abs/2505.22954
37. Zhang, J. et al. "dgm" (code, including DGM_outer.py). GitHub, 2025. https://github.com/jennyzzt/dgm
38. Sakana AI. "The Darwin Gödel Machine: AI that improves itself by rewriting its own code." Blog, 2025. https://sakana.ai/dgm/
39. Wang, W., Piękos, P., Li, N., Laakom, F., Schmidhuber, J., et al. "Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine." 2025. https://arxiv.org/abs/2510.21614
40. METAUTO AI. "HGM" (code). GitHub, 2025. https://github.com/metauto-ai/HGM
41. Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., Chen, X. "Large Language Models as Optimizers." ICLR 2024. https://arxiv.org/abs/2309.03409
42. Google DeepMind. "opro" (code). GitHub, 2023. https://github.com/google-deepmind/opro
43. Fernando, C., Banarse, D., Michalewski, H., Osindero, S., Rocktäschel, T. "Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution." 2023. https://arxiv.org/abs/2309.16797
44. Khattab, O. et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." 2023. https://arxiv.org/abs/2310.03714
45. Stanford NLP. "DSPy Optimizers" (documentation). GitHub, 2024-2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
46. Opsahl-Ong, K., Ryan, M. J., Purtell, J., Broman, D., Potts, C., Zaharia, M., Khattab, O. "Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs" (MIPRO). EMNLP 2024. https://arxiv.org/abs/2406.11695
47. Yuksekgonul, M., Bianchi, F., Boen, J., Liu, S., Lu, P., Huang, Z., Guestrin, C., Zou, J. "Optimizing generative AI by backpropagating language model feedback" (TextGrad). Nature 639, 609-616, 2025. https://arxiv.org/abs/2406.07496
48. Zou Group, Stanford. "textgrad" (code). GitHub, 2024. https://github.com/zou-group/textgrad
49. Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A. G., Stoica, I., Klein, D., Zaharia, M., Khattab, O. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." 2025. https://arxiv.org/abs/2507.19457
50. GEPA authors. "gepa" (code, adapters, reported results). GitHub, 2025. https://github.com/gepa-ai/gepa
51. OpenAI and Bain. "Self-Evolving Agents: A Cookbook for Autonomous Agent Retraining." OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/partners/self_evolving_agents/autonomous_agent_retraining.ipynb
