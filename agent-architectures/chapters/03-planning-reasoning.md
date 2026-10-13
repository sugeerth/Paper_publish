# 03. Planning and Reasoning Architectures: Plan-and-Execute, ReWOO, LLMCompiler, Tree and Graph Search, LATS, Reflection, and Reasoning Models with Interleaved Thinking

## Overview

Chapter 02 described the basic agent loop: call the model, run whatever tool calls it emits,
append the results, and repeat. That loop decides one step at a time. The model reasons
about the next action, acts, observes the result, and reasons again. ReAct made this pattern
canonical [1]. It is flexible and robust, but it has three structural weaknesses that every
design in this chapter tries to fix:

1. **No explicit long-horizon plan.** Each step is chosen greedily from the transcript so far.
   LangChain's tutorial on plan-and-execute names "explicit long term planning (which even
   really strong LLMs can struggle with)" as the main motivation for planning agents [3].
2. **Redundant, serial LLM calls.** Every tool observation triggers another full model call
   over a growing prefix. ReWOO's authors call this "huge computation complexity from
   redundant prompts and repeated execution" [4].
3. **No backtracking or self-correction.** A single trajectory cannot recover from an early bad
   decision unless the model happens to notice. Search methods (Tree of Thoughts, LATS) and
   reflection methods (Reflexion, Self-Refine) add that capability [11][15][17][19].

Since 2023, published work on these problems has moved through three generations:

- **Generation 1 (2022-2023): prompted scaffolds.** Planning, search, and reflection were
  built outside the model as prompt templates plus orchestration code. This group includes
  Plan-and-Solve, ReWOO, HuggingGPT, Tree of Thoughts, Graph of Thoughts, RAP, Reflexion, and
  Self-Refine [2][4][8][11][13][14][17][19].
- **Generation 2 (2023-2024): compiled and searched agents.** These systems treat a plan as a
  program or a search tree over tool-using actions. They include LLMCompiler (a DAG with
  parallel dispatch), LATS (Monte Carlo tree search with reflection), and ADaPT (recursive
  decomposition on failure) [6][15][9].
- **Generation 3 (2024-2026): reasoning models.** These models are trained with reinforcement
  learning to produce long chains of thought and, increasingly, to think between tool calls
  within one turn. Examples are OpenAI's o-series and GPT-5, Anthropic's extended and adaptive
  thinking, DeepSeek-R1, and open-weight "interleaved thinking" models such as MiniMax-M2 and
  GLM-4.5 through GLM-4.7 [33][35][36][24][38][41][42]. In this generation much of the
  explicit scaffolding moves into the model's weights. The engineering effort shifts to plan
  documents, effort and budget controls, and preserving reasoning state across tool calls
  [37][27][28][33].

This chapter takes each design family in turn. It covers the problem the design targets, its
components and data flow, its key techniques, any published cost or quality figures, its
failure modes, and the lessons practitioners have drawn. A comparison table and design
takeaways follow at the end.

A note on scope: memory and context management (Chapter 05 territory) and multi-agent
orchestration (Chapter 04 territory) overlap with planning. This chapter covers them only
where they serve as the planning substrate, for example plan files that persist across
context resets.

## 1. A Taxonomy of Planning Designs

Every design in this chapter answers four questions:

| Axis | Options seen in the literature |
|---|---|
| When is the plan made? | Up front (ReWOO, Plan-and-Execute), incrementally (ReAct), on failure only (ADaPT) |
| What shape is the plan? | Linear list, DAG with variables, tree of partial states, arbitrary graph |
| Who checks progress? | Nobody (open loop), a replanner, a value function or reflection, an external verifier |
| Where does reasoning live? | In prompts and orchestration code, or inside a trained reasoning model's hidden thinking |

The most consequential split is between **open-loop** designs, which commit to a plan and
execute it without looking at observations until the end, and **closed-loop** designs, which
re-plan, search, or reflect after observations. Open-loop designs save tokens and latency.
Closed-loop designs buy robustness at a cost multiple. The rest of the chapter can be read as
a series of positions on that trade-off.

## 2. Plan-and-Execute (Plan-and-Solve Lineage)

### Problem

ReAct agents reason "one step at a time" [3]. On multi-step tasks they lose track of the
overall goal, skip steps, or loop. Separately, zero-shot chain-of-thought prompting ("Let's
think step by step") produces calculation errors, missing steps, and misunderstandings on
reasoning benchmarks. Plan-and-Solve (PS) prompting, presented at ACL 2023, targets these
errors by asking the model to "first understand the problem and devise a plan," then "carry
out that plan step by step." Its PS+ variants add instructions such as "extract relevant
variables and their corresponding numerals" and "calculate intermediate variables" [2].

### Architecture

The agentic version separates a **planner** from an **executor**. LangChain's Plan-and-Execute
agent was explicitly "heavily inspired by the Plan-and-Solve paper as well as the Baby-AGI
project" [3][45]. The Plan-and-Solve repository records that the method was added to
LangChain's core library under that name [2].

```
            +-----------+   plan: [s1, s2, ..., sn]
 user ----->|  Planner  |------------------------------+
            | (strong   |                              |
            |   LLM)    |<--------------+              v
            +-----------+  revise plan  |     +------------------+
                  ^                     |     | Executor agent   |
                  |                     |     | (ReAct loop,     |
                  |  past_steps:        |     |  smaller model,  |
                  |  [(step, result)]   |     |  tools)          |
                  |                     |     +--------+---------+
            +-----+------+              |              |
            | Re-planner |--------------+   result of s_i
            | respond or |<-----------------------------+
            | new plan   |
            +-----+------+
                  |
                  v  final response
```

The LangGraph reference implementation keeps four pieces of state: the original input, the
current plan (a list of strings), the executed steps (a list of `(step, result)` tuples), and
the final response. The planner uses function calling to emit a structured plan. After each
step, a re-plan node either returns a final answer or produces a revised plan from the
remaining work [3].

### Key techniques

- **Model tiering.** The tutorial lists the "ability to use smaller/weaker models for the
  execution step, only using larger/better models for the planning step" as one of the two
  main advantages [3].
- **Structured plans.** The plan is a typed object (Pydantic in the reference
  implementation), not free text, so the harness can iterate over it deterministically [3].
- **Re-planning after every step.** This makes the design closed-loop at step granularity
  while keeping a global view of the task.

### Failure modes and lessons

The LangGraph authors name the main limitation directly: "each task is still executed in
sequence, meaning embarrassingly parallel operations all add to the total execution time."
They suggest representing tasks "as a DAG (similar to LLMCompiler), rather than a regular
list" [3]. Plans also go stale when early observations invalidate assumptions. The re-planner
mitigates this, but each re-plan costs a strong-model call.

## 3. ReWOO: Reasoning Without Observation

### Problem

In ReAct-style systems, "an LLM reasons to call an external tool, gets halted to fetch the
tool's response, and then decides the next action based on all preceding response tokens."
The system prompt and all prior steps are re-sent on every call [4][5].

### Architecture

ReWOO (Xu et al., 2023) "detaches the reasoning process from external observations" [4]. It
has three modules [5]:

1. **Planner (LLM).** Writes the whole tool-use plan in one pass. Placeholders refer to
   future evidence:
   ```
   Plan: <reasoning>
   #E1 = Tool[argument]
   Plan: <reasoning>
   #E2 = Tool[argument that references #E1]
   ```
2. **Worker (no LLM, or a cheap LLM tool).** Runs the tools in order and substitutes earlier
   evidence into later arguments.
3. **Solver (LLM).** Reads the plan and all the evidence, then writes the answer.

```
 task --> [Planner LLM] --plan with #E vars--> [Worker: tool1 -> tool2 -> ...]
                                                     |  evidence #E1..#En
                                                     v
                                  task + plan + evidence --> [Solver LLM] --> answer
```

The planner never sees an observation, so a task needs exactly two strong-model calls no
matter how many tools it uses. In the LangGraph implementation, an "LLM" tool that receives
only the substituted evidence "receives less of the prompt context and so can be more
token-efficient than the ReACT paradigm" [5].

### Published numbers

The authors report "5x token efficiency and 4% accuracy improvement on HotpotQA" and
"consistent performance enhancements" across six public NLP benchmarks and a curated dataset
[4]. Because planning data does not depend on tool outputs, ReWOO could be instruction-tuned
without running tools. The authors "offload reasoning ability from 175B GPT3.5 into 7B
LLaMA" and released a 7B planner based on Alpaca [4]. The maintainers also report that
integrating ReWOO into the Gentopia framework halved an agent's execution time compared with
ReAct [4].

### Failure modes and lessons

The LangGraph tutorial summarizes two limitations from the paper. First, "if little context
of the environment is available, the planner will be ineffective in its tool use," which
few-shot prompting or fine-tuning can mitigate. Second, tasks still run in sequence, so
latency is the sum of all tool calls [5]. More fundamentally, ReWOO is open-loop. A failed or
surprising observation cannot change the plan. The paper claims robustness under tool
failure [4], but the planner has no way to branch on intermediate results. ReWOO is best
suited to tasks whose information needs can be foreseen, which the authors call "foreseeable
reasoning" [4].

## 4. LLMCompiler: Plans as DAGs with Parallel Dispatch

### Problem

Both plan-and-execute and ReWOO serialize tool calls, even when the calls are independent.
LLMCompiler (Kim et al., ICML 2024) borrows from classical compilers. It identifies "which
tasks can run in parallel and which depend on each other" and runs independent calls
concurrently [6].

### Architecture

LLMCompiler has three components [6][7]:

1. **Planner.** Streams a DAG of tasks. Each task names a tool, its arguments, and its
   dependencies, written as `$k` references to earlier task outputs.
2. **Task Fetching Unit.** "Schedules and executes the tasks as soon as they are executable."
   It resolves variables as dependencies complete. With streaming enabled, it starts tasks
   while the planner is still emitting later ones [6][7].
3. **Executor / Joiner.** The repository names the third component the Executor. The
   LangGraph port adds a Joiner that "responds to the user or triggers a second plan" [6][7].

```
 Planner (streaming) ==> t1: search(A)        t2: search(B)       t3: math($1,$2)
                            |                    |                   ^
                            v                    v                   |
 Task Fetching Unit:   [run t1] ----------- [run t2] (parallel) ---- waits on $1,$2
                            \___________________/________________ [run t3]
                                                                     |
 Joiner: answer the user  OR  replan with observations  <------------+
```

### Published numbers and evaluation

The repository reports "latency speedup, cost saving, and accuracy improvement" on HotpotQA,
Movie Recommendation, and ParallelQA, with both open-source and closed-source models [6]. The LangGraph port notes that
the design also "saves costs on redundant token usage by reducing the number of calls to the
LLM" [7]. Exact speedup factors are in the paper and are not repeated here.

### Failure modes and lessons

- A DAG is only as good as the planner's dependency analysis. If a dependency is missing,
  tasks run on unresolved inputs. If the planner over-serializes, the speedup disappears.
- The joiner's re-plan path is what makes LLMCompiler closed-loop. Without it, the design has
  ReWOO's open-loop weakness.
- The broader lesson: once the plan is a data structure rather than prose, standard systems
  techniques apply, including topological scheduling, streaming, and variable binding. The
  production descendant is native parallel tool calling in model APIs. OpenAI's cookbook
  notes, however, that reasoning models "may produce a sequence of function calls that must
  be made in series" and recommends a general loop rather than assuming parallelism [34].

## 5. Hierarchical and As-Needed Decomposition

### HuggingGPT: an LLM controller over expert models

HuggingGPT (Shen et al., 2023) is an early, influential example of explicit task
decomposition. Its workflow has four stages: **task planning** (the LLM "analyze[s] the
requests of users to understand their intention, and disassemble[s] them into possible
solvable tasks"), **model selection** (choosing Hugging Face models "based on their
descriptions"), **task execution**, and **response generation** [8]. The design shows the
controller-plus-specialists pattern that later reappears as orchestrator-workers (Section 10)
and multi-agent research systems. The same group later released TaskBench for evaluating
task automation and EasyTool for tool instructions [8].

### Least-to-most and decomposition prompting

Least-to-most prompting (Zhou et al., 2022) splits a problem into ordered subproblems and
solves them in sequence, each answer feeding the next. LLM Reasoners groups it with
chain-of-thought, Tree of Thoughts, and RAP as a standard reasoning algorithm [10][14].

### ADaPT: decompose only when execution fails

ADaPT (Prasad et al., 2023; AI2, UNC, and Saarland University) answers the question of when to plan. It first
tries a subtask with an executor. It decomposes recursively only if the executor fails, and
the maximum depth `d` is a tunable parameter (`max_depth`) [9]. It was evaluated on ALFWorld,
WebShop, and TextCraft, a Minecraft-recipe crafting environment the authors added [9].

```
 ADaPT(task, depth):
     ok = Executor(task)                  # try directly first
     if ok or depth == d: return ok
     plan = Planner(task)                 # e.g. "AND"/"OR" of subtasks
     return combine(ADaPT(sub, depth+1) for sub in plan)
```

**Lesson.** Up-front planning wastes tokens when the executor could just do the task, and it
fails when the planner cannot foresee the difficulty. As-needed decomposition adapts the
amount of planning to the observed difficulty of the task. Adaptive thinking and effort
controls in 2025-2026 reasoning models express the same principle at the model level
(Section 9) [27][24].

## 6. Tree of Thoughts and Graph of Thoughts

### Problem

Chain-of-thought sampling is left-to-right and commits to every token. Deliberate problem
solving needs lookahead, evaluation of partial solutions, and backtracking.

### Tree of Thoughts (ToT)

ToT (Yao et al., NeurIPS 2023) breaks problem solving into **thoughts**, coherent
intermediate steps. It searches over them with four configurable parts [11][12]:

- **Thought generator.** `propose` generates sequential thoughts from one prompt and was used
  for Game of 24. `sample` draws independent thoughts and was used for Creative Writing.
- **State evaluator.** `value` scores each state on its own. `vote` compares states jointly.
- **Search algorithm.** The official implementation uses BFS for most tasks, with a greedy
  top-b beam whose width is `n_select_sample`. Crosswords uses DFS.
- **Budget knobs.** `n_generate_sample` and `n_evaluate_sample` set how many generation and
  evaluation calls are made per node [11].

```
                         [problem]
                /            |             \
          thought a      thought b      thought c       <- generate (k per node)
          value=0.9      value=0.2      value=0.6       <- evaluate (LLM or rule)
           /    \                         |             <- keep top-b (BFS) or DFS
        a1        a2                     c1
      (24!)     (dead)                 (dead) -> backtrack
```

**Numbers.** The repository reports that the paper's Game of 24 experiments reached 74%. A
later re-run scored 69%, which the authors attribute to "GPT decoding randomness" [11]. The
LangGraph re-implementation splits ToT into expand, score, and prune steps. It uses a
deterministic scorer for Game of 24, which checks that every number is used once and that
the expression equals 24 [12]. LLM Reasoners reports reproducing ToT's published performance
with its own library [14].

**Failure modes.** ToT multiplies model calls by roughly branching factor times evaluations
times depth. It depends on a usable evaluator. When the state value comes from an LLM rather
than a rule, search can confidently follow wrong branches.

### Graph of Thoughts (GoT)

GoT (Besta et al., AAAI 2024; ETH Zurich) generalizes the tree to an arbitrary graph. Thoughts
can be **aggregated** (several chains merged into one), **refined** through feedback loops,
and scored [13]. The framework has these parts:

- a **Graph of Operations (GoO)** that declares the transformation plan (Generate, Score,
  Aggregate, KeepBest, GroundTruth);
- a **Controller** that executes the GoO against an LLM;
- a **Prompter** and **Parser**, which are problem-specific;
- **Scoring and validation** by an LLM, a human, or a function [13].

The canonical example sorts 64 numbers. GoT splits the list, sorts the pieces, and merges
them with merge-sort-style prompts. Each thought is scored by its number of sorting errors
[13]. The poster states that the GoO "enables seamless specification of not only GoT, but
also existing schemes such as CoT, CoT-SC, ToT" [13]. In other words, the thought topology
becomes a programmable artifact, much like the DAG in LLMCompiler.

### Reasoning via Planning (RAP) and search libraries

RAP (Hao et al., 2023) treats the LLM as both a world model and a reasoning agent and plans
with MCTS. LLM Reasoners (Maitrix) packages RAP, ToT (BFS and DFS), self-evaluation-guided
beam search, GRACE decoding, and inference-time scaling with process reward models (Snell et
al., 2024) under one API. It has an SGLang backend for efficient structured generation and a
visualizer for search trees [14]. The library's existence reflects how standard these search
procedures had become by 2024-2025.

## 7. Language Agent Tree Search (LATS)

### Problem

ToT searches over thoughts inside a closed reasoning problem. Agent tasks such as web
navigation, multi-hop QA with search, and code with tests involve actions and environment
feedback. LATS (Zhou et al., ICML 2024) "unifies reasoning, acting and planning." It runs
Monte Carlo tree search over ReAct-style trajectories, with an LLM-generated value plus
reflection serving as the heuristic [15].

### Architecture

The LATS teaser figure shows an LLM agent coupled to an environment (observations and
rewards), a value and self-reflection module, memory, and a tree search that picks the best
node [15]. The LangGraph port describes the loop in four steps [16]:

1. **Select.** Pick the node with the highest upper confidence bound (UCT), which balances
   exploitation (average reward) against exploration (visit count).
2. **Expand and simulate.** Generate N candidate actions (5 in the tutorial) from that node
   and run them in parallel against tools.
3. **Reflect and evaluate.** Score each outcome with a reflection prompt, plus external
   feedback such as test results when available.
4. **Backpropagate.** Update ancestor values. Stop when a valid solution is found or when
   the rollout and depth limit is reached.

```
          root (user task)
          /     |      \             UCT select -> node n
        n1     n2      n3            expand: 5 candidate actions from n
       /  \             \            execute tools in parallel
     n11  n12           n31          reflect: LLM critique + value score (+ unit tests)
      |                              backprop: update mean value along path
   SOLVED  <-- terminate when a trajectory passes the external check
```

The official code has three main budget parameters: `n_generate_sample` (expansion width),
`n_evaluate_sample` (evaluation prompts per state), and `iterations` (maximum sampled
trajectories) [15]. Domains are HotPotQA, programming (built on the Reflexion codebase), and
WebShop [15].

### Failure modes and lessons

The LangGraph authors offer two operational cautions [16]:

- "The tree rollout can take additional compute time." For production they recommend
  streaming intermediate steps, or using LATS offline "for fine-tuning data to improve the
  single-shot accuracy and avoid long rollouts."
- "The candidate selection process is only as good as the reward you generate." When an
  external signal such as code test execution exists, it should replace or supplement
  self-reflection.

LATS also assumes the environment can be reset or that actions are reversible, so that
sibling branches can be explored. That holds for QA and sandboxed code but not for
irreversible real-world actions such as sending email or making payments. This is the main
reason MCTS-style agents are rare in production, while best-of-N sampling with a verifier is
common.

## 8. Reflection and Self-Correction

Reflection designs keep a single trajectory but add a critique-and-revise cycle. They differ
mainly in where the feedback comes from.

### Self-Refine: intrinsic feedback loop

Self-Refine (Madaan et al., 2023; CMU, AI2, and others) uses one LLM in three roles:
**generate**, **feedback**, and **refine**. It repeats FEEDBACK -> REFINE -> FEEDBACK until a
task-specific `is_refinement_sufficient` check passes [19]. Two design points matter. The
feedback is required to be **actionable**, meaning it should localize the problem and give an
instruction to improve it. The loop also **retains history** by appending earlier outputs and
feedback to the prompt, so the model avoids repeating mistakes [19]. Across seven tasks
(including review rewriting, acronym generation, code readability, constrained generation,
and toxicity removal), the project reports outputs "preferred by humans and by automated
metrics" over direct generation with GPT-3.5 and GPT-4. Gains range from "at least 5% to more
than 40%," roughly 20% absolute on average [19].

### Reflexion: verbal reinforcement across episodes

Reflexion (Shinn et al., NeurIPS 2023) frames reflection as reinforcement learning without
weight updates. Its architecture has three parts [17][18]:

- **Actor (LM).** Acts in the environment and keeps a short-term trajectory memory.
- **Evaluator.** Produces internal or external feedback. External feedback is task-specific,
  such as unit tests or environment reward.
- **Self-reflection (LM).** Turns the feedback into reflective text stored in **long-term
  experience memory**, which conditions the next attempt.

```
            +-------------------- long-term memory (reflections) ----------+
            v                                                              |
  task -> Actor LM --action--> Environment --obs/reward--> Evaluator ------+
            ^   (trajectory = short-term memory)              |            |
            +------------------------------------------- Self-reflection LM
```

The repository compares four context strategies: no prior-attempt information, the last
trajectory only, the reflection only, and both. It covers HotPotQA, AlfWorld (where
reflections persist in external memory), programming, and WebShop [17]. The LangGraph port
implements an initial responder and a revisor that cite evidence from search tools [18].

### The general reflection pattern and its limits

LangGraph's minimal reflection example, a five-paragraph-essay generator critiqued by a
"teacher" prompt, ends with the key caveat: "self-reflection is inherently cyclic: it is much
more effective if the reflection step has additional context or feedback (from tool
observations, checks, etc.)." Without grounding, reflection "can still benefit the output
quality (since the LLM then has multiple 'shots' at getting a good output), but it's less
guaranteed" [20]. The Reflexion port adds that the design "trades off execution time for
quality" and "usually (not always) increases the response quality" [18].

Practitioner guidance has converged on the same point. Anthropic's evaluator-optimizer
workflow is recommended "when clear evaluation criteria exist and feedback measurably
improves output" [21]. Claude Code's 2026 best-practice guide turns this into an operational
rule: "Give Claude a check it can run: tests, a build, a screenshot to compare." It
recommends a verification subagent with fresh context so that "the agent doing the work
isn't the one grading it." It also warns that a reviewer "prompted to find gaps will usually
report some, even when the work is sound," which leads to over-engineering if every finding
is chased [32].

### The "think" tool: reflection at tool boundaries

Anthropic's 2025 "think" tool is a no-op tool that only logs a thought. It gives Claude
"a dedicated space to pause mid-task" after tool results arrive, which differs from extended
thinking, which happens before the response starts [22]. On tau-bench with Claude 3.7
Sonnet, the airline domain went from a 0.332 baseline (pass^1, table value) to 0.584 with the
think tool plus an optimized prompt, and retail went from 0.783 to 0.812. Extended thinking
alone scored 0.412 and 0.770 respectively. The tool also gave about a 1.6% average isolated
improvement on SWE-bench [22]. Anthropic recommends it for "complex tool chains,
policy-heavy settings, and sequential decisions where mistakes are costly," and not for
single or parallel tool calls [22]. This bridged prompted reflection and the native
interleaved thinking described next.

## 9. Reasoning Models: Planning Moves Inside the Model

### From scaffold to weights

Reasoning models are trained, mostly with large-scale reinforcement learning, to produce a
long private chain of thought before answering. OpenAI describes o3 and o4-mini as "LLMs
trained with reinforcement learning to perform reasoning" that "excel in ... multi-step
planning for agentic workflows" [34]. DeepSeek-R1-Zero applied "reinforcement learning
directly to the base model" without supervised fine-tuning. The repository calls this the
first open research to validate that reasoning "can be incentivized purely through RL," and
reports emergent "self-verification, reflection, and long chain-of-thought" [38]. DeepSeek-R1
adds cold-start data and a pipeline of two RL stages and two SFT stages. It reports 79.8 on
AIME 2024 (pass@1) and 97.3 on MATH-500, compared with 79.2 and 96.4 for OpenAI o1-1217.
Distilled 32B and 70B dense models score 72.6 and 70.0 on AIME 2024 [38].

These results change the design calculus for the scaffolds above. Decomposition,
self-verification, backtracking, and reflection now happen in hidden reasoning tokens. In
effect, the model runs its own plan-and-solve and reflection loop. OpenAI's function-calling
guide for o3 and o4-mini is explicit: "a developer should not try to induce additional
reasoning before each function call by asking the model to plan more extensively. Asking a
reasoning model to reason more may actually hurt the performance" [35].

Anthropic reported two scaling axes when it introduced extended thinking on Claude 3.7 Sonnet
in February 2025 [23]:

- **Serial test-time compute.** Accuracy on 2024 AIME problems "improves predictably (roughly
  logarithmically)" with the number of allowed thinking tokens.
- **Parallel test-time compute.** Sampling many independent thought processes and selecting
  one with a learned scorer reached 84.8% on GPQA (96.5% physics subscore). This used the
  equivalent of 256 samples and a 64k thinking budget, with gains "beyond the limits of
  majority vote." The method was a research result, not a deployed feature [23].

Parallel sampling with a learned selector is the same idea as best-of-N or tree search with a
value function from Sections 6 and 7, moved inside the inference stack.

### Interleaved thinking: reasoning between tool calls

In early reasoning-model APIs, the model thought once at the start of a turn and then chained
tool calls without further reasoning. **Interleaved thinking** lets the model think after
each tool result inside one assistant turn. Anthropic's documentation shows the difference
with a two-tool example. Without interleaving, Response 2 (the database query) and Response 3
(the final answer) have no thinking blocks. With interleaving, the model reasons "Got $7,500.
Now I should query the database to compare..." before the second call, and "$7,500 vs $5,200
average - that's a 44% increase..." before answering [26].

```
 Without interleaving                     With interleaving
 [think] plan                             [think] plan
 [tool_use calc] -> 7500                  [tool_use calc] -> 7500
 [tool_use db]   -> 5200  (no thinking)   [think] interpret 7500, decide next
 [text answer]            (no thinking)   [tool_use db]   -> 5200
                                          [think] compare, check
                                          [text answer]
```

Vendor implementations as of 2025-2026:

- **Anthropic.** On Claude 4-era models in manual mode, interleaving requires the
  `interleaved-thinking-2025-05-14` beta header. In that mode `budget_tokens` can exceed
  `max_tokens`, because "the budget spans all thinking blocks within one assistant turn."
  With adaptive thinking (`thinking: {type: "adaptive"}`), "interleaved thinking is automatic"
  and needs no header. Claude Haiku 4.5 does not support it [25][24]. Fixed `budget_tokens` is
  deprecated on the 4.6 models and rejected from 4.7 onward. Depth is instead controlled by
  `output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`). Effort "affects all
  tokens in the response," including tool calls, and lower effort means "fewer and terser
  tool calls" [25][27].
- **OpenAI.** o3 and o4-mini "are trained to use tools natively within their chain of
  thought" and were "trained with its internal reasoning persisted between tool calls within
  a single turn" [35]. GPT-5 adds `reasoning_effort` (with a new `minimal` level) and a
  `verbosity` parameter that controls answer length separately from thinking length [36].
- **Open-weight models.** MiniMax-M2 (230B total parameters, 10B active) is described as "an
  interleaved thinking model." Its documentation warns that users must keep the
  `<think>...</think>` content in history and "do not remove" it, "otherwise, the model's
  performance will be negatively affected" [41]. Z.ai's GLM-4.5 introduced interleaved
  thinking. GLM-4.7 adds **Preserved Thinking**, which "automatically retains all thinking
  blocks across multi-turn conversations, reusing the existing reasoning instead of
  re-deriving from scratch," and **Turn-level Thinking**, which switches reasoning on or off
  per turn [42]. Qwen3 offered hybrid thinking and non-thinking modes, then split them into
  separate Instruct-2507 and Thinking-2507 models [43]. Kimi K2's Instruct release is
  described as "a reflex-grade model without long thinking" [44].

### Reasoning state is now part of the protocol

The most important engineering consequence of interleaved thinking is that **reasoning state
must round-trip through the client** during a tool loop:

- Anthropic requires that "when you return tool results, you must pass the thinking blocks
  from the assistant message back to the API, complete and unmodified." Each block carries a
  `signature`, which is "an encrypted copy of the full reasoning." Rebuilding the message or
  filtering out `redacted_thinking` blocks returns a 400 error [24][26]. Thinking text is a
  summary produced by a different model. Billing covers the full thinking tokens, not the
  summary. Newer models default to `display: "omitted"` to reduce time to first token [24].
- OpenAI's Responses API exposes reasoning items, persisted server-side via
  `previous_response_id` or client-side as `encrypted_content` for zero-data-retention
  customers. OpenAI reports about a **3% SWE-bench improvement** from including reasoning
  items "for the same prompt and setup" [33]. For GPT-5, Tau-Bench Retail rose "from 73.9% to
  78.2% just by switching to the Responses API and including `previous_response_id`," which
  avoids the need "to reconstruct a plan from scratch after each tool call" [36].
- Caching interacts with this state. OpenAI reports cache utilization rising "from 40% to
  80%" after moving to the Responses API [33]. Anthropic warns that changing `budget_tokens`
  or the thinking mode invalidates message-level cache breakpoints, so a fixed setting should
  be held for the life of a cached conversation [25]. Anthropic's per-message effort beta
  (`mid-conversation-output-config-2026-07-01`) changes effort without breaking the cache
  [27].

Earlier designs such as Plan-and-Execute and ReWOO kept the plan as explicit, inspectable
state that the harness owned. Reasoning models keep the plan as opaque, signed state owned by
the vendor and carried by the harness. The harness can no longer read or edit the plan. It
can only preserve it, which is why explicit plan artifacts returned in 2025 (Section 10).

### Training models to interleave search and reasoning

Open research has reproduced interleaved tool reasoning:

- **Search-o1** (EMNLP 2025) adds an agentic RAG step to a large reasoning model. The model
  emits search queries in the middle of reasoning when it "encounter[s] uncertain
  information." A Reason-in-Documents module condenses the retrieved pages before they go
  back into the chain. Inference uses "batch generation ... with interleaved search" [39].
- **Search-R1** trains "reasoning-and-searching interleaved LLMs" with RL (PPO, GRPO,
  REINFORCE) on veRL. It extends DeepSeek-R1(-Zero) with search-engine access and is offered
  as "an alternative and open solution to OpenAI DeepResearch" [40].

### Failure modes of reasoning models in agent loops

- **Overthinking and cost.** Thinking tokens are billed as output and count toward
  `max_tokens` [24]. Anthropic notes that `max` effort "adds significant cost for relatively
  small quality gains" on most workloads and "can lead to overthinking" on structured-output
  tasks [27]. Budgets above 32k thinking tokens should use batch processing to avoid
  timeouts [25].
- **Skipped thinking.** DeepSeek-R1 models "tend to bypass thinking pattern" on some queries,
  which hurts performance. The recommended fix is to force output to start with a thinking
  tag [38].
- **Hallucinated tool behavior.** OpenAI warns that o3 "may be more prone to hallucinations,"
  including "promising to call tools in the background without actually doing so" [35].
- **Tool-ordering errors.** o3 and o4-mini "can make mistakes in the order of the tool
  calls," so OpenAI recommends spelling out required orderings in the developer prompt [35].
- **Unfaithful reasoning.** Anthropic cautions that visible reasoning "may not fully reflect
  what drives the model's behavior," which limits the use of thought monitoring as a safety
  check [23].
- **Protocol fragility.** Dropping or editing reasoning blocks quietly degrades quality on
  MiniMax-M2 and GLM-style models, and returns hard errors on the Claude API [41][42][26].

## 10. Planning in Production Agents (2024-2026)

Production systems rarely run ToT or LATS. They combine a reasoning model with explicit,
externalized plan artifacts and verification gates.

### Orchestrator-workers and decomposition at scale

Anthropic's "Building effective agents" (2024) describes the **orchestrator-workers**
workflow: "a central LLM dynamically breaks down a task and delegates pieces to worker LLMs,"
used "when the subtasks can't be predicted in advance" [21]. The 2025 multi-agent Research
system builds on this pattern:

- The lead agent "uses extended thinking as a scratchpad" to choose tools, judge query
  complexity, and decide how many subagents to use and what each does. It "saves its plan to
  Memory" because contexts beyond 200,000 tokens get truncated [29].
- Subagents "plan, then use interleaved thinking after each tool result to assess quality,
  find gaps, and refine their next query" [29].
- Effort scaling rules are written into the prompt. Simple fact-finding uses one agent with 3
  to 10 tool calls. Comparisons use 2 to 4 subagents with 10 to 15 calls each. Complex
  research can use more than 10 subagents [29].
- A system with Claude Opus 4 as lead and Claude Sonnet 4 subagents outperformed single-agent
  Opus 4 by 90.2% on an internal research eval. On BrowseComp, token usage alone explained
  80% of performance variance. Multi-agent systems use about 15x the tokens of chat [29].
- Failure modes included spawning 50 subagents for simple queries, endless searches for
  nonexistent sources, duplicated subagent work caused by vague delegation, and continuing to
  search after enough results were found [29].

### Plan documents as durable state

As agent runs grew from minutes to hours, plans moved out of the context window into files:

- **OpenAI ExecPlans (PLANS.md), October 2025.** An `AGENTS.md` rule tells Codex to write an
  "ExecPlan" for complex features. The plan is a "living document" that must be
  self-contained enough "to restart from only the ExecPlan and no other work," with mandatory
  `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective`
  sections and independently verifiable milestones. OpenAI reports that a similar PLANS.md
  "enabled Codex to work for more than seven hours from a single prompt" [37].
- **Anthropic long-running harness, November 2025.** An **initializer agent** expands the
  prompt into a feature list (over 200 features in the chat-app example, all marked failing),
  an `init.sh`, a progress log, and an initial git commit. A **coding agent** then works one
  feature per session and leaves a clean, committed state. The feature list is JSON because
  "models are less likely to inappropriately overwrite it than Markdown." Observed failure
  modes were declaring victory early, leaving undocumented partial work, and marking features
  done without end-to-end testing [30].
- **Structured note-taking.** Anthropic's context-engineering guidance describes to-do lists
  and NOTES.md files as a "simple pattern" for tracking progress across many tool calls. It
  cites Claude playing Pokemon, which kept tallies "over thousands of game steps" and resumed
  multi-hour sequences from its own notes after context resets [31].

```
  +-------------------+     writes      +-----------------------------+
  | Planner session   |---------------->| PLAN / feature_list.json    |
  | (explore + plan,  |                 | progress log, decision log  |
  |  read-only mode)  |                 +-------------+---------------+
  +-------------------+                               | read at start
                                                      v
  +-------------------+  one milestone  +-----------------------------+
  | Executor session  |---------------->| code + tests + git commit   |
  | (fresh context)   |<----------------| verifier: tests / subagent  |
  +-------------------+  pass / fail    +-----------------------------+
          | update progress, surprises, decisions; repeat until all pass
```

### Plan mode and effort as user-facing controls

Claude Code's documented workflow is "Explore, Plan, Implement, Commit." Plan mode is
read-only, and the user can edit the plan before approving it. The guide also says when not
to plan: "If you could describe the diff in one sentence, skip the plan." Planning "is most
useful when you're uncertain about the approach, when the change modifies multiple files, or
when you're unfamiliar with the code" [32]. GPT-5's guide makes a related point about
**agentic eagerness**. Lower `reasoning_effort` "reduces exploration depth but improves
efficiency." At minimal reasoning, "prompted planning is likewise more important, as the
model has fewer reasoning tokens to do internal planning" [36]. GPT-5 also emits **tool
preambles** ("clear upfront plans and consistent progress updates"). Newer Claude models emit
**progress updates** between tool calls, which are user-facing notes separate from the
reasoning blocks [36][24].

### Budget-aware planning

Anthropic's task budgets (beta `task-budgets-2026-03-13`) give the model an advisory token
budget for an entire agentic loop. That covers thinking, tool calls, tool results, and
output. The model sees a server-side countdown and uses it "to prioritize work and finish
gracefully" [28]. Effort controls depth per step, while a task budget controls breadth across
the loop. Budgets are soft: `max_tokens` remains the hard cap, and the minimum budget is
20,000 tokens. A budget "clearly insufficient for the work" can cause "refusal-like
behavior," such as aggressive down-scoping or early stops [28]. The model, rather than an
outer controller, now decides how deeply to plan given its remaining resources.

## 11. Comparison of Planning and Reasoning Designs

| Design | Plan shape | Loop type | LLM calls per task (approx.) | Main benefit | Main failure mode | Best fit |
|---|---|---|---|---|---|---|
| ReAct [1] | Implicit, step by step | Closed, every step | 1 per tool step | Simple, adaptive | Myopia, loops, serial latency | Short, unpredictable tasks |
| Plan-and-Execute [2][3] | Linear list | Closed via re-planner | Plan + N executor steps + re-plans | Global plan, model tiering | Serial, stale plans | Multi-step tasks with weaker executor |
| ReWOO [4][5] | Linear with #E vars | Open | 2 (planner + solver) | About 5x fewer tokens on HotpotQA | Cannot react to observations | Foreseeable info-gathering |
| LLMCompiler [6][7] | DAG with $k deps | Closed via joiner | Plan + join (+ re-plan) | Parallel tool dispatch | Wrong dependency analysis | Parallelizable queries |
| ADaPT [9] | Recursive AND/OR tree | Closed, on failure | Grows only with difficulty | Plans only when needed | Depth bound, executor self-assessment | Mixed-difficulty household and web tasks |
| Tree of Thoughts [11] | Tree of thoughts | Closed via evaluator | b x k x depth | Lookahead and backtracking | Cost, noisy LLM values | Puzzles, closed reasoning |
| Graph of Thoughts [13] | Arbitrary graph (GoO) | Closed via scoring | Defined by GoO | Aggregation and merging | Hand-designed graphs | Decomposable tasks (sort, merge) |
| LATS [15][16] | MCTS over actions | Closed, value + reflection | iterations x (N + evals) | Search with env feedback | Needs resettable env, slow | Code with tests, QA, sandboxed web |
| Self-Refine [19] | Single draft | Closed, self-feedback | 1 + 2 per iteration | No training needed | Ungrounded critique | Writing, code style |
| Reflexion [17][18] | Episodic retries | Closed across trials | Trials x (act + reflect) | Learns from failures in memory | Needs evaluator signal | Coding with tests, ALFWorld |
| Think tool [22] | None (scratchpad) | Closed at tool boundary | Same as ReAct, more tokens | Policy adherence | Little gain on simple calls | Policy-heavy tool chains |
| Reasoning model + interleaved thinking [24][33][41] | Hidden, in weights | Closed, every tool result | 1 per tool step, long outputs | Native planning and checking | Opaque state, overthinking, protocol fragility | Default for 2025-2026 agents |
| Plan file + verifier harness [30][37] | Durable document | Closed per milestone | Many sessions | Survives context resets, hours-long runs | Premature "done," doc drift | Long-horizon coding |

## Key Takeaways for System Designers

1. **Start from a reasoning model with interleaved thinking, then add scaffolding only where
   evals show a gap.** Vendors now advise against prompting reasoning models to plan before
   each call, because it "may actually hurt the performance" [35]. Anthropic's general advice
   is to "add complexity only when it measurably improves outcomes" [21].
2. **Treat reasoning state as part of the wire protocol.** Pass thinking blocks and reasoning
   items back unchanged, keep `<think>` content for open-weight interleaved models, and
   prefer stateful APIs. The measured gains are real: about 3% on SWE-bench and 73.9% to
   78.2% on Tau-Bench Retail [33][36][24][41].
3. **Use explicit plans for durability and inspection, not for intelligence.** Plan files
   (ExecPlans, feature lists, progress logs) let work survive context resets, give humans an
   approval point, and support restarts [37][30][32]. Prefer structured formats such as JSON
   for state the agent must not silently rewrite [30].
4. **Choose open-loop plans only when information needs are foreseeable.** ReWOO-style plans
   cut tokens sharply [4] but cannot react to observations. If tool calls are independent,
   use a DAG with parallel dispatch, as in LLMCompiler [6].
5. **Search only where you have a reliable value signal and a resettable environment.** ToT,
   LATS, and parallel test-time compute all depend on the quality of the evaluator. A unit
   test beats an LLM-judged value [16][23]. LATS-style rollouts work well as offline
   data-generation engines to "improve the single-shot accuracy" [16].
6. **Ground reflection in external checks.** Ungrounded self-critique gives the model extra
   attempts but no guarantee [20]. Tests, builds, screenshots, and a fresh-context verifier
   subagent work better, but tell reviewers to flag only correctness-relevant gaps to avoid
   over-engineering [32].
7. **Match planning depth to task difficulty.** ADaPT decomposes only on failure [9].
   Adaptive thinking and effort let the model skip thinking on easy inputs [24][27]. Claude
   Code's rule of thumb is to skip planning when the diff fits in one sentence [32].
8. **Budget explicitly, at two levels.** Use effort or reasoning effort for depth per step,
   and task budgets or harness turn caps for breadth across the loop. Size budgets from
   measured p99 usage, because budgets that are too small cause early stops [27][28].
9. **Expect cost multipliers and design for them.** Multi-agent research uses about 15x chat
   tokens, and token usage explains 80% of BrowseComp variance [29]. Plan for decomposition
   only on high-value, parallelizable tasks [29].
10. **Instrument for the failure modes the literature names.** Watch for premature
    completion, runaway subagent spawning, repeated searches for nonexistent sources,
    hallucinated tool calls, wrong tool order, and skipped thinking. Each has a documented
    mitigation: feature lists and end-to-end tests, effort-scaling rules, explicit ordering
    in prompts, and forced thinking prefixes [30][29][35][38].

## References

1. Yao, S. et al. "ReAct: Synergizing Reasoning and Acting in Language Models." Princeton /
   Google, ICLR 2023 (arXiv 2022). https://arxiv.org/abs/2210.03629 (code:
   https://github.com/ysymyth/ReAct)
2. Wang, L. et al. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
   by Large Language Models." ACL 2023. https://arxiv.org/abs/2305.04091 (code:
   https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting)
3. LangChain. "Plan-and-Execute" (LangGraph tutorial, archived). 2024.
   https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
4. Xu, B. et al. "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented
   Language Models." 2023. https://arxiv.org/abs/2305.18323 (code and abstract:
   https://github.com/billxbf/ReWOO)
5. LangChain. "Reasoning without Observation" (LangGraph ReWOO tutorial, archived). 2024.
   https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/rewoo/rewoo.ipynb
6. Kim, S., Moon, S., Tabrizi, R., Lee, N., Mahoney, M., Keutzer, K., Gholami, A. "An LLM
   Compiler for Parallel Function Calling." ICML 2024. https://arxiv.org/abs/2312.04511
   (code: https://github.com/SqueezeAILab/LLMCompiler)
7. LangChain. "LLMCompiler" (LangGraph tutorial, archived). 2024.
   https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/llm-compiler/LLMCompiler.ipynb
8. Shen, Y., Song, K., Tan, X., Li, D., Lu, W., Zhuang, Y. "HuggingGPT: Solving AI Tasks with
   ChatGPT and its Friends in Hugging Face." Microsoft / Zhejiang University, 2023.
   https://arxiv.org/abs/2303.17580 (code: https://github.com/microsoft/JARVIS)
9. Prasad, A., Koller, A., Hartmann, M., Clark, P., Sabharwal, A., Bansal, M., Khot, T.
   "ADaPT: As-Needed Decomposition and Planning with Language Models." 2023.
   https://arxiv.org/abs/2311.05772 (code: https://github.com/archiki/ADaPT)
10. Zhou, D. et al. "Least-to-Most Prompting Enables Complex Reasoning in Large Language
    Models." Google, 2022. https://arxiv.org/abs/2205.10625
11. Yao, S. et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models."
    NeurIPS 2023. https://arxiv.org/abs/2305.10601 (code:
    https://github.com/princeton-nlp/tree-of-thought-llm)
12. LangChain. "Tree of Thoughts" (LangGraph tutorial, archived). 2024.
    https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/tot/tot.ipynb
13. Besta, M. et al. "Graph of Thoughts: Solving Elaborate Problems with Large Language
    Models." ETH Zurich, AAAI 2024. https://arxiv.org/abs/2308.09687 (code and poster:
    https://github.com/spcl/graph-of-thoughts)
14. Maitrix.org. "LLM Reasoners" library (RAP, ToT, beam search, PRM inference scaling).
    2023-2025. https://github.com/maitrix-org/llm-reasoners ; Hao, S. et al. "Reasoning with
    Language Model is Planning with World Model" (RAP), 2023.
    https://arxiv.org/abs/2305.14992 ; Snell, C. et al. "Scaling LLM Test-Time Compute
    Optimally..." 2024. https://arxiv.org/abs/2408.03314
15. Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., Wang, Y.-X. "Language Agent Tree
    Search Unifies Reasoning, Acting, and Planning in Language Models." ICML 2024.
    https://arxiv.org/abs/2310.04406 (code: https://github.com/lapisrocks/LanguageAgentTreeSearch)
16. LangChain. "Language Agent Tree Search" (LangGraph tutorial, archived). 2024.
    https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/lats/lats.ipynb
17. Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., Yao, S. "Reflexion:
    Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
    https://arxiv.org/abs/2303.11366 (code: https://github.com/noahshinn/reflexion)
18. LangChain. "Reflexion" (LangGraph tutorial, archived). 2024.
    https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/reflexion/reflexion.ipynb
19. Madaan, A. et al. "Self-Refine: Iterative Refinement with Self-Feedback." CMU / AI2 et
    al., 2023. https://arxiv.org/abs/2303.17651 (code and project page:
    https://github.com/madaan/self-refine)
20. LangChain. "Reflection" (LangGraph tutorial, archived). 2024.
    https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/reflection/reflection.ipynb
21. Anthropic. "Building effective agents." 2024.
    https://www.anthropic.com/engineering/building-effective-agents
22. Anthropic. "The 'think' tool: Enabling Claude to stop and think in complex tool use
    situations." 2025. https://www.anthropic.com/engineering/claude-think-tool
23. Anthropic. "Claude's extended thinking." February 24, 2025.
    https://www.anthropic.com/news/visible-extended-thinking
24. Anthropic. "Thinking" (Claude API documentation: adaptive thinking, interleaved thinking,
    preservation, display, encryption). Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/thinking
25. Anthropic. "Extended thinking" (manual budget mode, interleaved beta header, caching).
    Accessed 2026. https://platform.claude.com/docs/en/build-with-claude/extended-thinking
26. Anthropic. "Thinking in tool and multi-turn workflows." Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows
27. Anthropic. "Effort." Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/effort
28. Anthropic. "Task budgets" (beta). Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/task-budgets
29. Anthropic. "How we built our multi-agent research system." 2025.
    https://www.anthropic.com/engineering/multi-agent-research-system
30. Young, J. (Anthropic). "Effective harnesses for long-running agents." November 26, 2025.
    https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
31. Anthropic. "Effective context engineering for AI agents." September 29, 2025.
    https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
32. Anthropic. "Best practices for Claude Code." Accessed 2026.
    https://code.claude.com/docs/en/best-practices
33. OpenAI. "Better performance from reasoning models using the Responses API" (OpenAI
    Cookbook). 2025.
    https://github.com/openai/openai-cookbook/blob/main/examples/responses_api/reasoning_items.ipynb
34. OpenAI. "Managing Function Calls With Reasoning Models" (OpenAI Cookbook). 2025.
    https://github.com/openai/openai-cookbook/blob/main/examples/reasoning_function_calls.ipynb
35. OpenAI. "o3/o4-mini Function Calling Guide" (OpenAI Cookbook). 2025.
    https://github.com/openai/openai-cookbook/blob/main/examples/o-series/o3o4-mini_prompting_guide.ipynb
36. OpenAI. "GPT-5 prompting guide" (OpenAI Cookbook). 2025.
    https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/gpt-5_prompting_guide.ipynb
37. OpenAI. "Using PLANS.md for multi-hour problem solving" (OpenAI Cookbook). October 2025,
    updated January 2026. https://github.com/openai/openai-cookbook/blob/main/articles/codex_exec_plans.md
38. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
    Learning." 2025. https://arxiv.org/abs/2501.12948 (repository and model card:
    https://github.com/deepseek-ai/DeepSeek-R1)
39. Li, X. et al. "Search-o1: Agentic Search-Enhanced Large Reasoning Models." EMNLP 2025.
    https://arxiv.org/abs/2501.05366 (code: https://github.com/sunnynexus/Search-o1)
40. Jin, B. et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with
    Reinforcement Learning." 2025. https://arxiv.org/abs/2503.09516 (code:
    https://github.com/PeterGriffinJin/Search-R1)
41. MiniMax. "MiniMax-M2" model repository and usage notes. 2025.
    https://github.com/MiniMax-AI/MiniMax-M2
42. Z.ai (Zhipu). "GLM-4.5 / GLM-4.6 / GLM-4.7" repository (interleaved, preserved, and
    turn-level thinking). 2025. https://github.com/zai-org/GLM-4.5
43. Qwen Team (Alibaba). "Qwen3" repository (thinking and non-thinking modes, 2507
    releases). 2025. https://github.com/QwenLM/Qwen3
44. Moonshot AI. "Kimi K2: Open Agentic Intelligence" repository. 2025.
    https://github.com/MoonshotAI/Kimi-K2
45. Nakajima, Y. "BabyAGI." 2023. https://github.com/yoheinakajima/babyagi
