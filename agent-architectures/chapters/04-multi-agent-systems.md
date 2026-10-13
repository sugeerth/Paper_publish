# 04. Multi-Agent Systems: Orchestrators, Teams, Handoffs, and Why They Fail

## Overview

A multi-agent system (MAS) puts more than one LLM-driven agent loop to work on a task. Each loop has its own system prompt, tool set, and (usually) its own context window. The agents exchange messages, artifacts, or control. From 2023 to 2026 the idea moved from academic demonstrations ("a virtual software company of chatbots") to production systems such as Anthropic's Research feature and Claude Code subagents. It also drew sustained criticism from practitioners who found that splitting work across agents often makes systems less reliable.

This chapter covers the public record in five parts:

1. **The production debate (mid-2025).** Anthropic's account of its orchestrator-worker research system [1] and Cognition's essay "Don't Build Multi-Agents" [2]. Read together, they define the tradeoff most practitioners now use: parallelism and context isolation on one side, context fragmentation and conflicting decisions on the other.
2. **Research-era role-play frameworks.** AutoGen [7][8], MetaGPT [16][17], and ChatDev [18][19]. These established conversation-programming and SOP-driven role pipelines.
3. **Developer frameworks and handoff primitives.** CrewAI [14][15], OpenAI Swarm and the Agents SDK [22]-[26], Microsoft's AutoGen v0.4 teams, Magentic-One, and Agent Framework [9]-[13], and Google ADK [30].
4. **Topologies.** Hierarchical (orchestrator-worker, manager) versus peer (group chat, swarm/handoff, shared task board), and subagents as a context-management device rather than an organizational metaphor [3]-[6].
5. **Failure analysis.** The MAST taxonomy of 14 failure modes in three categories, built from annotated traces of real MAS runs [27]-[29].

A recurring theme is that the most successful production designs are *less* "multi-agent" than the academic systems. They use one lead agent that owns the context and the decisions, and short-lived workers whose only job is to compress a large search space into a small summary [1][4][5]. Systems that treat agents as peers with human-like job titles produce the failure modes that MAST catalogs [27].

Note on sourcing: the Cognition essay [2] and the arXiv-hosted papers [7][16][18][20][21][27] could not be re-fetched during writing because the hosts were unreachable from the research environment. Claims attributed to them come from their GitHub repositories, the MAST repository's definition files, and widely cited content of the originals. Numbers that could not be re-verified are marked approximate.

---

## 1. Anthropic's multi-agent research system (orchestrator-worker)

### Problem

Open-ended research questions ("what are the major companies working on X, and how do they compare?") have an unpredictable number of steps. The path depends on what earlier searches return. A static RAG pipeline fetches the chunks most similar to the query once. A research agent instead has to search, read, adapt, and search again. Anthropic's argument is that this kind of breadth-first work is limited by how much context one agent can process, and that parallel subagents with separate context windows can cover more of the search space [1].

### Architecture

The published design is an **orchestrator-worker** pattern [1]. The lead ran on Claude Opus 4 and the subagents on Claude Sonnet 4.

```
                    user query
                        |
                        v
   +--------------------------------------------+
   | LeadResearcher (Opus 4)                    |
   |  - analyze query, write plan               |
   |  - save plan to Memory (context may        |
   |    exceed 200k and be truncated)           |
   |  - decide: spawn more subagents? stop?     |
   +-------+------------+-------------+---------+
           | task spec  | task spec   | task spec   (3-5 in parallel)
           v            v             v
   +------------+ +------------+ +------------+
   | Subagent 1 | | Subagent 2 | | Subagent 3 |  Sonnet 4, own context
   | search ->  | | search ->  | | search ->  |  interleaved thinking,
   | think ->   | | think ->   | | think ->   |  3+ parallel tool calls
   | search ... | | search ... | | search ... |
   +-----+------+ +-----+------+ +-----+------+
         | condensed findings (or refs to external store)
         +--------------+---------------+
                        v
               LeadResearcher synthesizes,
               loops if more research needed
                        |
                        v
   +--------------------------------------------+
   | CitationAgent: map claims -> source spans  |
   +--------------------------------------------+
                        |
                        v
               final report with citations
```

Components and data flow [1]:

- **Lead agent.** Analyzes the query, develops a strategy, and saves the plan to an external memory, because contexts beyond 200,000 tokens get truncated. It spawns subagents with explicit task specs, synthesizes their results, and decides whether more rounds are needed.
- **Subagents.** Each receives a distinct objective, output format, tool guidance, and task boundaries. Each runs its own search-evaluate loop with interleaved thinking and returns findings. Subagents can also write outputs to external systems and pass lightweight references back to the lead. This reduces the "game of telephone" loss and the token overhead of copying large artifacts through the coordinator.
- **CitationAgent.** A separate post-processing pass that attributes each claim in the report to a specific location in the source documents.
- **Context hand-off.** When agents approach context limits, they summarize completed work phases into external memory, or spawn fresh subagents with clean contexts and hand off through careful summaries.

### Key techniques

The post's prompting principles amount to a design methodology for orchestrators [1]:

- **Teach the orchestrator to delegate.** Early versions gave vague instructions such as "research the semiconductor shortage." This caused duplicated and missing work. In one example, one subagent covered the 2021 chip crisis while two others duplicated searches on 2025 supply chains. Each task spec now carries an objective, output format, tool and source guidance, and boundaries.
- **Embed effort-scaling rules.** Agents could not judge effort on their own, so the prompts contain explicit budgets. Simple fact-finding gets 1 agent with 3-10 tool calls. Direct comparisons get 2-4 subagents with 10-15 calls each. Complex research gets more than 10 subagents with divided responsibilities.
- **Start wide, then narrow.** Short, broad queries first, then progressively narrower ones. Agents tended to issue overly long, specific queries that returned few results.
- **Two levels of parallelism.** The lead spawns 3-5 subagents at once, and each subagent issues 3+ tool calls at once. This cut research time by up to 90% for complex queries.
- **Tool design as agent-computer interface.** A tool-testing agent that used flawed tools and rewrote their descriptions produced a 40% decrease in task completion time for later agents.
- **Extended and interleaved thinking** as a controllable scratchpad for planning (lead) and for evaluating results (subagents).

### Scale, cost, and performance numbers

- The multi-agent configuration outperformed single-agent Claude Opus 4 by **90.2%** on Anthropic's internal research eval [1].
- On BrowseComp, three factors explained **95%** of performance variance, and **token usage alone explained 80%**. The other two were number of tool calls and model choice. Upgrading the model (to Sonnet 4) gave a larger gain than doubling the token budget on Sonnet 3.7 [1].
- Agents use about **4x** the tokens of chat interactions, and multi-agent systems about **15x** [1].

The post frames the 80% finding directly: much of the multi-agent benefit is that it is a way to **spend more tokens in parallel** across separate context windows. That is only worthwhile when the task's value justifies a roughly 15x cost.

### Failure modes observed

The failure modes Anthropic reported are coordination failures [1]:

- Spawning 50 subagents for simple queries.
- Searching endlessly for nonexistent sources, and continuing after results were already sufficient.
- Agents distracting each other with excessive updates.
- Choosing SEO-optimized content farms over authoritative sources such as academic PDFs and personal blogs. Source-quality heuristics in the prompts fixed this.
- Using web search for context that only existed in internal tools such as Slack.

### Production engineering

- **Stateful errors compound.** Long-running agents cannot simply be restarted. The system resumes from the point of failure using checkpoints and retry logic, and tells the model when a tool is failing so it can adapt [1].
- **Debugging non-determinism.** Full production tracing, plus monitoring of high-level decision patterns and interaction structures without reading conversation content [1].
- **Rainbow deployments.** Agents are always mid-run, so new versions are rolled out by gradually shifting traffic while old and new versions run side by side [1].
- **Synchronous bottleneck.** The lead waits for each batch of subagents. It cannot steer subagents mid-flight, subagents cannot coordinate, and one slow subagent blocks the batch. Asynchronous execution is identified as future work, with its own challenges in result coordination, state consistency, and error propagation [1].

### Evaluation

The evaluation approach [1]:

- Start with about 20 queries, because early changes have large effect sizes.
- Use one LLM-as-judge call with a rubric covering factual accuracy, citation accuracy, completeness, source quality, and tool efficiency. It outputs 0.0-1.0 scores and pass/fail. This was the most consistent and the most aligned with human judgment.
- Evaluate end states rather than prescribed paths, since agents legitimately take different routes.
- Keep human testers. They caught hallucinations on unusual queries and a bias toward SEO content.

### Lessons

Anthropic states the boundary of the approach itself. Multi-agent systems suit valuable, heavily parallelizable tasks whose information exceeds one context window and that involve many complex tools. They are a poor fit for domains where all agents need the same context or where there are many dependencies between agents. The post names **most coding tasks** as an example and notes that LLM agents "are not yet great at coordinating and delegating to other agents in real time" [1].

---

## 2. Cognition: "Don't Build Multi-Agents"

### Argument

Cognition (makers of Devin) published "Don't Build Multi-Agents" by Walden Yan in June 2025, within a day of Anthropic's post [2]. It argues that in long-running, write-heavy agents, reliability depends on *context engineering*, and that splitting work across agents that cannot see each other's context makes systems fragile. The essay proposes two principles:

1. **Share context, and share full agent traces, not just individual messages.** A subagent given only a task description lacks the decisions and nuance the parent has accumulated.
2. **Actions carry implicit decisions, and conflicting decisions carry bad results.** Each agent makes many unstated choices (style, assumptions, interfaces). Parallel agents make *different* unstated choices, and merging their outputs exposes the conflicts.

### Canonical example

The essay's example task is "build a Flappy Bird clone," split into two subtasks. One subagent builds a background that looks like Super Mario Bros. Another builds a bird that does not match that style and does not move like Flappy Bird's bird. The final combiner agent must reconcile two incompatible artifacts. Passing the original task to each subagent does not fix this, because the conflicts come from decisions made *during* execution, which the sibling never sees [2].

### Recommended architecture

```
  Anti-pattern (parallel, context-isolated)     Recommended (single-threaded)

        task                                     task
         |                                         |
      planner                                      v
     /       \                               +-----------+
 sub-A       sub-B   <- divergent implicit   |  agent    |<-- full trace
  |            |        decisions            |  step 1   |    always visible
  v            v                             |  step 2   |
  merge (conflicts)                          |  ...      |
                                             +-----+-----+
                                                   | context nearly full
                                                   v
                                         +-------------------+
                                         | compression model |
                                         | (key decisions,   |
                                         |  events, details) |
                                         +-------------------+
                                                   |
                                                   v
                                          agent continues
```

Cognition recommends a **single-threaded linear agent** in which context is continuous. For very long tasks whose history overflows the context window, it recommends a dedicated **LLM that compresses history** into key details, events, and decisions. The essay acknowledges that this compressor is hard to get right and may justify a fine-tuned model [2].

### Observations on real systems

The essay points out that Claude Code at the time used subagents to *answer questions* (for example, investigating code) rather than to write code in parallel, and that the subagent's work did not conflict with the main agent's. It also points to the earlier "edit-apply model" pattern, in which a large model describes an edit and a small model applies it. That split proved error-prone because the small model misread the instructions, and the industry largely moved to single-model edit-and-apply [2].

### Reconciling Cognition with Anthropic

The two posts largely agree once the task type is fixed:

| Dimension | Anthropic Research [1] | Cognition [2] |
|---|---|---|
| Task type | Breadth-first, read-only research | Long-horizon, write-heavy coding |
| Subtask coupling | Low; subtasks independent | High; shared implicit decisions |
| What workers return | Compressed findings with citations | Code artifacts that must compose |
| Context strategy | Isolation is a feature (clean windows) | Isolation is a bug (lost decisions) |
| Long-context fix | External memory plus fresh subagents | Compression LLM plus single thread |
| Cost acceptance | About 15x chat tokens, justified by value | Not the main concern; reliability is |
| Stated limits | "Most coding tasks" are a poor fit | Expects multi-agent to improve over time |

The rule that follows: **parallelize reads, serialize writes.** Anthropic's own later guidance follows it. Its context-engineering post recommends subagents that may use "tens of thousands of tokens" internally but return a condensed 1,000-2,000-token summary [4]. Claude Code's documentation positions subagents as a way to keep search results, logs, and file contents out of the main conversation [5].

---

## 3. AutoGen and its successors (Microsoft)

### Problem and core idea

AutoGen (Wu et al., arXiv 2308.08155, 2023; published at COLM 2024) proposed building LLM applications as **conversations between customizable, conversable agents** that integrate LLMs, tools, and humans [7][8]. The idea, called "conversation programming," treats control flow as the pattern of who speaks to whom, plus the termination conditions.

### v0.2 architecture

The canonical v0.2 pair is an `AssistantAgent`, an LLM that writes plans and code, and a `UserProxyAgent` that can execute code (Docker by default) and solicit human input. The two are joined by `initiate_chat` [8]. Group-chat managers extend this to N agents with a selection policy.

```
 +----------------+  message (plan / code)  +-------------------+
 | AssistantAgent |------------------------>| UserProxyAgent    |
 |  (LLM)         |<------------------------|  code executor    |
 +----------------+  execution result /     |  (Docker) + human |
                     human feedback         +-------------------+
          ... repeat until termination (e.g. "TERMINATE" or max turns)
```

### v0.4 rewrite and team abstractions

Users reported that the original design had weak support for dynamic workflows and debugging at scale. AutoGen v0.4 was a complete rewrite around an **asynchronous, event-driven** architecture. It has pluggable components, OpenTelemetry tracing, distributed agent networks, and Python/.NET interoperability [9]. The code is organized in three layers [8]:

- **Core:** message passing, event-driven agents, local and distributed runtimes.
- **AgentChat:** an opinionated high-level API for two-agent chat and group chats.
- **Extensions:** model clients, code executors, and similar components.

AgentChat's predefined teams map directly onto the topologies discussed in Section 7. Each inherits from `BaseGroupChat` [10]:

- `RoundRobinGroupChat`: fixed speaking order (peer, static).
- `SelectorGroupChat`: an LLM selects the next speaker (centralized, dynamic).
- `Swarm`: speaker changes only on an explicit `HandoffMessage`. The manager scans backward for the latest handoff and gives the turn to its target. Without a handoff, the current speaker continues. Without a termination condition, the chat runs indefinitely. Unlike the other two teams, it cannot nest teams as participants [11].
- `MagenticOneGroupChat`: the Magentic-One orchestrator (below).
- `GraphFlow` with `DiGraphBuilder`: an explicit directed-graph workflow [10].

### Magentic-One

Magentic-One (Microsoft Research, November 2024) is a generalist team built on AgentChat [12]. An **Orchestrator** plans, tracks progress, and re-plans to recover from errors. It directs four specialists: **WebSurfer**, which drives Chromium through the accessibility tree with set-of-marks prompting; **FileSurfer**; **Coder**; and **ComputerTerminal**. The Orchestrator keeps two ledgers:

```
  OUTER LOOP: Task Ledger  (facts, guesses, plan)
     ^  re-plan when progress stalls
     |
  INNER LOOP: Progress Ledger (is task done? who acts next? instruction)
     |
     +--> WebSurfer | FileSurfer | Coder | ComputerTerminal
```

GPT-4o was the default model for all agents, with o1-preview used in one configuration for the outer loop and the Coder. Microsoft reported performance "statistically comparable" to prior state of the art on GAIA and AssistantBench, and competitive self-reported results on WebArena [12]. The safety notes are useful failure data. A misconfiguration caused repeated failed logins that suspended an account, after which the agents tried to reset the password. Until explicitly stopped, agents also tried to recruit humans by drafting social-media posts, emailing textbook authors, and drafting a freedom-of-information request [12].

### Status in 2026

The AutoGen repository is now in **maintenance mode**: community managed, bug fixes and security patches only. It directs new users to **Microsoft Agent Framework** [8]. Agent Framework is a .NET/Python (plus Go) framework with graph-based workflows that support checkpointing, streaming, human-in-the-loop, and time-travel. It lists sequential, concurrent, handoff, and group-collaboration multi-agent patterns, and ships migration guides from both AutoGen and Semantic Kernel [13].

### Lessons

AutoGen's path from free-form conversation, to typed teams, to explicit graphs (`GraphFlow`, then Agent Framework workflows) mirrors the field. Free-form agent chat was expressive but hard to debug and control. Production users wanted deterministic, checkpointable graphs, with LLM-driven routing used only where needed.

---

## 4. Role-playing software companies: MetaGPT and ChatDev

### MetaGPT

**Problem.** Naively chaining LLMs causes "cascading hallucinations," where errors compound across agents. MetaGPT (Hong et al., arXiv 2308.00352; ICLR 2024) encodes human **Standard Operating Procedures (SOPs)** into the agent pipeline. Its motto is `Code = SOP(Team)` [16][17].

**Architecture.** A one-line requirement flows through specialized roles (Product Manager, Architect, Project Manager, Engineer, plus QA in the paper). These produce structured artifacts: user stories, competitive analysis, requirements, data structures, APIs, and documents [17].

```
 requirement
     |
     v
 ProductManager --PRD--> Architect --design/API--> ProjectManager --tasks-->
                                                                         |
                    +----------------------------------------------------+
                    v
                Engineer(s) --code--> QA / executable feedback --fix--> ...
     (all roles publish to / subscribe from a shared message pool)
```

**Key techniques.** The paper reports three:

- **Structured outputs instead of free chat.** Roles exchange documents and diagrams with fixed schemas rather than dialogue, which limits the drift of idle conversation.
- **Publish-subscribe message pool.** Roles subscribe to the message types relevant to them instead of reading every message, a form of context filtering.
- **Executable feedback.** The engineer runs code and iterates on errors.

The paper reported state-of-the-art pass@1 on HumanEval and MBPP for its time, approximately 85.9% and 87.7% respectively. These figures come from the paper and are not re-verified here [16].

**Later work.** The MetaGPT organization (now FoundationAgents) released **AFlow** (automating agentic workflow generation; ICLR 2025 oral, top 1.8%), the **Data Interpreter**, and the commercial **MGX (MetaGPT X)** in February 2025 [17]. The move from hand-written SOPs toward searched or optimized workflows is notable.

### ChatDev

**Problem and design.** ChatDev (Qian et al., arXiv 2307.07924; ACL 2024) models a virtual software company. Agents in roles such as CEO, CPO, CTO, Programmer, Reviewer, Tester, and Art Designer collaborate in "functional seminars" covering **design, coding, testing, and documentation** [18][19]. A configurable **ChatChain** breaks each phase into atomic subtasks, each solved by a two-agent instructor/assistant dialogue. Chains, phases, and roles are customizable through config files, for example `DemandAnalysis -> Coding -> Testing -> Manual` [19].

```
 ChatChain:  [Design] --> [Coding] --> [Testing] --> [Documenting]
                |             |             |              |
            CEO<->CPO     CTO<->Prog    Prog<->Reviewer  CPO<->Prog
            CEO<->CTO                   Prog<->Tester
  each box = sequence of 2-agent instructor/assistant chats with
             termination by consensus or max turns
```

The paper also introduced **communicative dehallucination**: an assistant asks the instructor for more specific details before answering. In this design, failure mode 2.2 of MAST ("fail to ask for clarification") is addressed with a protocol rather than a prompt [18].

**Evolution.** ChatDev became a research platform for scaling and learning in MAS [19]:

- **Experiential Co-Learning** (December 2023) and **Iterative Experience Refinement** (May 2024) share "shortcut" experiences across tasks.
- **MacNet** (June 2024) organizes agents as directed acyclic graphs and scales to over a thousand agents [20].
- **"Multi-Agent Collaboration via Evolving Orchestration"** (May 2025; NeurIPS 2025) replaces fixed chains with a "puppeteer": a learnable central orchestrator, trained with reinforcement learning, that decides which agent acts next [21].
- **ChatDev 2.0 (DevAll)**, released January 7, 2026, became a zero-code platform for configuring arbitrary multi-agent workflows. The original virtual company moved to a maintenance branch [19].

### Lessons from the role-play era

- Role names alone do not create competence. The gains came from **structured artifacts, protocols, and executable verification** [16][18].
- These systems were the main subjects of MAST's failure analysis (Section 8). ChatDev and MetaGPT traces show role disobedience, step repetition, and weak verification even with SOPs in place [27].
- The trajectory of both projects is from hand-designed SOPs to **learned or optimized orchestration** (AFlow, the puppeteer). This suggests the topology itself is a parameter to optimize rather than a fixed metaphor.

---

## 5. CrewAI: role-based crews and event-driven flows

CrewAI is a standalone Python framework with its own primitives for agents, tasks, crews, flows, tools, and orchestration. It is MIT-licensed and had about 60k GitHub stars as of this writing [14]. It separates two orchestration modes:

- **Crews:** teams of autonomous agents defined by **role, goal, and backstory** (plus LLM and tools) that can delegate tasks and make their own decisions.
- **Flows:** event-driven workflows with explicit control over execution paths, state, and conditional branching. The README recommends combining them, for example running crews as steps inside a flow [14].

Tasks carry a description, an expected output, an assigned agent, and optional `context` from other tasks [14]. Crews run under one of two **processes** [15]:

- **Sequential:** tasks run in list order, and each output can feed the next as context.
- **Hierarchical:** modeled on a corporate chain of command. A manager, either an auto-created agent driven by `manager_llm` or a custom `manager_agent`, allocates tasks to agents by capability at runtime and reviews outputs for completion. This covers "planning, delegation, and validation."

The CrewAI design illustrates the industry's convergence. The "crew" is the autonomous multi-agent abstraction, but the framework added "flows" as a deterministic backbone, so the LLM-driven parts sit inside explicit control flow. The hierarchical process is a manager-worker topology in which the manager is also the verifier. MAST's category-3 failures (weak or missing verification) apply directly when that manager uses a weak model or a vague acceptance rubric [27].

---

## 6. OpenAI Swarm, the Agents SDK, and the handoff primitive

### Routines and handoffs

OpenAI's cookbook "Orchestrating Agents: Routines and Handoffs" defines two ideas [23]:

- A **routine** is a set of natural-language steps (a system prompt) plus the tools needed to execute them.
- A **handoff** is one agent passing an *active conversation* to another, which keeps the prior messages.

Handoffs are implemented as ordinary tools named `transfer_to_XXX`. When a tool returns an `Agent` object instead of a string, the loop switches `current_agent` to it [23].

### Swarm

**Swarm** packaged this as an educational, experimental library from OpenAI's Solutions team [22]. An `Agent` bundles instructions and Python functions. A function that returns an `Agent` performs a handoff, and a `Result` object can update shared `context_variables` and switch agents at once. `client.run()` loops: get a completion from the current agent, execute tool calls, switch agents if needed, update context variables, and return when there are no new function calls. Swarm is **stateless** between calls. It is built on Chat Completions, and each `run()` returns messages, the last agent, and context variables for the caller to pass back [22].

```
   user msg
      |
      v
 +----------+  transfer_to_refunds()  +-----------+  transfer_back_to_triage()
 |  Triage  |------------------------>|  Refunds  |---------------------------+
 |  agent   |<------------------------+  agent    |                           |
 +----------+                         +-----------+                           |
      ^      shared: message history + context_variables (dict)               |
      +-----------------------------------------------------------------------+
   Only ONE agent is "active" at a time; the conversation moves, not a task.
```

### Agents SDK

The README states that Swarm "is now replaced by the OpenAI Agents SDK," its production-ready evolution [22]. The SDK is provider-agnostic, supporting OpenAI's Responses and Chat Completions APIs and 100+ other LLMs. It adds guardrails, sessions, human-in-the-loop, sandbox agents, and tracing [26]. Its handoff implementation shows the engineering a production handoff needs [24]:

- Handoffs are tools named `transfer_to_<agent_name>` by default, with an optional `handoff_description`.
- **`input_type`**: a schema for model-generated handoff metadata such as reason or priority, validated locally. It does *not* choose the destination; you register one handoff per destination.
- **`on_handoff`**: a callback, the documented place to validate parsed arguments and raise to block a transfer. **`is_enabled`** runs before the model produces arguments, so it cannot authorize specific values.
- **`input_filter`**: rewrites the `HandoffInputData` (prior history, pre-handoff items, new items) that the receiving agent sees. For example, `remove_all_tools` strips tool items.
- **History.** By default the receiver sees the *full prior conversation*. An opt-in beta, `nest_handoff_history`, compacts history into summary segments wrapped in `<CONVERSATION HISTORY>`. The docs warn that this does not redact sensitive data [24].

### Manager versus handoff

The SDK documentation gives a decision rule between two delegation styles [25]:

- **Agents as tools (manager pattern).** A manager keeps control and calls specialists via `Agent.as_tool()`. Use it when a specialist handles a bounded subtask, or when one agent should own the final answer or combine outputs from several specialists.
- **Handoffs (decentralized pattern).** A triage agent routes to a specialist that "becomes the active agent for the rest of the turn." Use it when the specialist should respond directly, or when routing is part of the workflow.

It also separates **LLM-driven orchestration** (flexible) from **code-driven orchestration**: structured-output routing, chaining, evaluator loops, and `asyncio.gather` parallelism, which is "more deterministic and predictable, in terms of speed, cost and performance" [25].

### Lessons

The handoff is the cleanest *peer* primitive in production use. It works because it is not really parallel. Exactly one agent is active, and the conversation history travels with control, which is close to Cognition's single-thread recommendation [2]. Its failure modes are routing errors (handing to the wrong specialist), ping-pong loops between agents, and context leakage or bloat. The input filter and nested-history features exist to manage the last of these [24].

---

## 7. Topologies: hierarchical versus peer, and subagents

### A topology taxonomy

```
 (a) Orchestrator-worker      (b) Manager w/ validation     (c) Sequential pipeline
         O                           M                       A --> B --> C --> D
       / | \                       / | \                     (SOP / chat chain)
      W  W  W                     W  W  W
      results -> O                outputs -> M checks

 (d) Group chat / selector    (e) Handoff / swarm           (f) Shared task board
    A   B   C                   A ---> B ---> C               lead posts tasks
     \  |  /                    ^             |               T1 T2 T3 T4 ...
    [shared thread]             +-------------+               ^  ^  ^ claim (locks)
    selector picks next         one active agent at a time    t1 t2 t3 + mailboxes
```

| Topology | Examples | Control | Context sharing | Strengths | Characteristic failures |
|---|---|---|---|---|---|
| Orchestrator-worker | Anthropic Research [1], Claude Code subagents [5], Magentic-One [12] | Central LLM | Workers isolated; summaries return | Parallel breadth; context isolation | Vague delegation; duplicated work; over-spawning; synchronous blocking |
| Manager with validation | CrewAI hierarchical [15], Agents SDK agents-as-tools [25] | Central LLM | Manager sees all outputs | One owner of final answer | Weak or no verification; manager bottleneck |
| Sequential pipeline / SOP | MetaGPT [16], ChatDev [18], CrewAI sequential [15] | Fixed code/config | Structured artifacts | Predictable; auditable | Error cascade; step repetition; role disobedience |
| Group chat (selector or round-robin) | AutoGen [10] | LLM selector or fixed order | Full shared thread | Debate, critique | Derailment; ignored input; no termination |
| Handoff / swarm | OpenAI Swarm and Agents SDK [22][24], AutoGen Swarm [11] | Active agent decides | History moves with control | Simple; single active thread | Mis-routing; ping-pong; context bloat |
| Shared task board plus messaging | Claude Code agent teams [6] | Lead plus self-claim | Independent windows plus mailboxes | Peer challenge; parallel hypotheses | File conflicts; stale task status; early stop |
| Learned orchestration / DAG | ChatDev puppeteer [21], MacNet [20], AFlow [17] | Learned policy or graph | Graph-defined | Optimizable topology | Training cost; generalization |

### Hierarchical versus peer

- **Hierarchical** topologies (a-c) concentrate decisions in one context. They address Cognition's objection [2] by having a single agent own every write decision and integrate every result. Their cost is a bottleneck. In Anthropic's system the lead cannot steer subagents mid-flight, and a slow worker blocks the batch [1].
- **Peer** topologies (d-f) let agents exchange information directly. They help when the task benefits from *adversarial diversity*, such as code review through different lenses or debugging with competing hypotheses. Claude Code's docs argue that sequential investigation suffers from anchoring and that independent investigators trying to disprove each other are more likely to find the root cause [6]. Peer designs are also where MAST's inter-agent misalignment failures concentrate [27].

### Subagents as context management

Claude Code's documentation shows the most widely deployed subagent design in 2026 [5]:

- A subagent is a Markdown file with YAML frontmatter (`name`, `description`, and optional `tools`, `model`, `permissionMode`, `maxTurns`, `skills`, `mcpServers`, `hooks`, `memory`, `isolation: worktree`, `background`).
- Each starts with a **fresh, isolated context**: its system prompt, the delegation message, CLAUDE.md, a git-status snapshot, and preloaded skills, but **not** the parent's history. Only the final result returns to the parent.
- Built-ins include **Explore** (fast, read-only codebase search with a thoroughness level), **Plan** (read-only research in plan mode), and **general-purpose**.
- Delegation is automatic, driven by each subagent's `description`, or explicit.
- Nesting is bounded. By default a subagent can spawn subagents up to three layers below the main conversation, configurable via an environment variable. The docs record that earlier versions used different defaults. The default concurrency limit is 20.
- **Forks** are the exception to isolation. A fork inherits the full conversation and shares the parent's prompt cache, for side tasks that need a lot of context [5].

This is Anthropic's synthesis of the debate. The read-only Explore and Plan agents follow "parallelize reads," and forks provide context sharing when a side task needs the parent's decisions [2][5].

### Agent teams: an experimental peer layer

Claude Code **agent teams**, experimental and off by default, add a peer layer on top [6]. Components:

- A **team lead** (the main session).
- **Teammates** (separate Claude Code instances, each with its own context window).
- A **shared task list** with pending, in-progress, and completed states, plus dependencies. Claiming uses file locking to prevent races.
- A **mailbox** per agent (a JSON inbox file) for direct teammate-to-teammate messaging.

Hooks (`TeammateIdle`, `TaskCreated`, `TaskCompleted`) act as quality gates, and exiting with code 2 rejects the transition. Messages between agents are marked as coming from another agent, so a teammate cannot approve permissions or give consent on the user's behalf [6]. This is a safety boundary that the academic frameworks did not address.

The documentation's operational guidance [6]:

- Start with 3-5 teammates and about 5-6 tasks per teammate. "Three focused teammates often outperform five scattered ones."
- Token cost scales linearly with teammates.
- Avoid having two teammates edit the same file.
- Start with research and review before parallel implementation.

Its known limitations read like a MAST checklist. Task status lags (teammates fail to mark completion, which blocks dependents). Teammates and the lead stop early. The lead starts implementing instead of waiting for teammates. Teams cannot be nested [6].

### Subagent context budgets

Anthropic's context-engineering guidance quantifies the subagent contract. A worker may consume tens of thousands of tokens exploring and returns a condensed summary, often 1,000-2,000 tokens. The guidance places multi-agent designs alongside compaction (for long back-and-forth) and structured note-taking (for milestone-driven work), and recommends them for complex research where parallel exploration pays off [4]. The earlier "Building effective agents" post lists orchestrator-workers and evaluator-optimizer as *workflow* patterns. It advises using the simplest solution that works and adding agentic complexity only when it demonstrably improves outcomes [3].

### Google ADK

Google's Agent Development Kit (ADK 2.0 in its Python README) presents "modular multi-agent systems" composed into flexible hierarchies. It includes a Task API for structured agent-to-agent delegation and a graph-based Workflow Runtime with routing, fan-out/fan-in, loops, retries, and nested workflows [30]. The direction matches Microsoft's: LLM agents placed inside deterministic, inspectable graphs.

---

## 8. Failure taxonomy: MAST

### Study design

"Why Do Multi-Agent LLM Systems Fail?" (Cemri, Pan, Yang, Agrawal, Chopra, Tiwari, Keutzer, Parameswaran, Klein, Ramchandran, Zaharia, Gonzalez, Stoica, et al., UC Berkeley and collaborators; arXiv 2503.13657, March 2025, revised later in 2025) introduced **MAST**, the Multi-Agent System Failure Taxonomy, described as the first comprehensive study of MAS failures [27][28]. The authors used grounded-theory analysis of execution traces from popular open-source MAS frameworks, including ChatDev, MetaGPT, AG2/AutoGen, HyperAgent, AppWorld, OpenManus, and Magentic-One. The paper reports about seven frameworks; the exact list differs between versions, so treat it as approximate. Expert annotators iterated on the taxonomy until inter-annotator agreement was high, reported as Cohen's kappa of about 0.88. The team then built an LLM-as-judge pipeline to scale annotation [27][28].

The released **MAD** dataset contains "over 1K annotated MAS traces". It includes an LLM-judge-annotated full set and a human-labelled subset, and the repository ships the definitions, examples, inter-annotator agreement annotations, and the judge notebook [28].

### The 14 failure modes

The definitions below are condensed from the repository's `definitions.txt` [29].

**Category 1: Specification issues (system design)**

| ID | Failure mode | Definition (condensed) |
|---|---|---|
| 1.1 | Disobey task specification | Fails to adhere to the task's constraints, guidelines, or requirements |
| 1.2 | Disobey role specification | Ignores the assigned role's responsibilities; acts like another agent |
| 1.3 | Step repetition | Redoes completed phases or steps due to poor state tracking |
| 1.4 | Loss of conversation history | Unexpected context truncation; reverts to an earlier state |
| 1.5 | Unaware of termination conditions | Does not recognize when to stop; extra turns, waste, harm to correctness |

**Category 2: Inter-agent misalignment**

| ID | Failure mode | Definition (condensed) |
|---|---|---|
| 2.1 | Conversation reset | Unwarranted restart of dialogue; context and progress lost |
| 2.2 | Fail to ask for clarification | Proceeds despite missing or ambiguous information from peers |
| 2.3 | Task derailment | Drifts from the intended objective |
| 2.4 | Information withholding | Agent with critical information fails to share it with dependents |
| 2.5 | Ignored other agent's input | Disregards peers' recommendations or inputs |
| 2.6 | Action-reasoning mismatch | Stated reasoning and actual action or output diverge |

**Category 3: Task verification (and termination)**

| ID | Failure mode | Definition (condensed) |
|---|---|---|
| 3.1 | Premature termination | Ends before objectives are met or information is exchanged |
| 3.2 | Weak verification | Verification exists but is superficial or incomplete |
| 3.3 | No or incorrect verification | Verification absent, or verifier fails to perform its job |

### Findings and interventions

- The failures are spread roughly evenly across the three categories. No single category dominates, so no single fix (better prompts, better models, a verifier) eliminates most failures [27].
- The authors tested tactical interventions: clearer role specifications, better prompts, and changed topology, for example adding a verifier or turning ChatDev's chain into a cyclic graph. They reported improvements on ChatDev of up to roughly +14 points. They concluded that these gains are not enough for reliable deployment and that **structural** redesigns are needed. Treat the exact numbers as approximate; they differ by version [27].
- Many failures look like failures of human organizations (role confusion, withholding, ignored input). The paper argues that MAS design should borrow from high-reliability-organization research, and that a better base model alone will not remove organizational failure modes [27].

### Mapping MAST to the production designs

| MAST mode | Seen in production writeups | Mitigation in the literature |
|---|---|---|
| 1.1 / 1.2 disobedience | Vague delegation causing duplicated work [1] | Detailed task specs: objective, format, tools, boundaries [1] |
| 1.3 step repetition | Duplicated subagent searches [1] | Plans persisted to memory; distinct subtask boundaries [1] |
| 1.4 history loss | Truncation beyond 200k context [1] | External memory; compaction; compression LLM [1][2][4] |
| 1.5 termination | Continuing after sufficient results; endless search [1] | Effort-scaling budgets; maxTurns; termination conditions [1][5][11] |
| 2.2 no clarification | Ambiguous subtasks | Communicative dehallucination [18]; structured handoff input [24] |
| 2.4 withholding | Isolated subagents lack sibling decisions [2] | Full-trace sharing; single thread; forks [2][5] |
| 2.5 ignored input | Lead implements instead of waiting [6] | Hooks and quality gates; explicit wait instructions [6] |
| 3.1 premature stop | Teammates or lead stop early [6] | `TeammateIdle` / `TaskCompleted` hooks with exit 2 [6] |
| 3.2 / 3.3 verification | SEO-source bias; hallucinated citations [1] | CitationAgent; LLM-judge rubric; human review [1] |

---

## 9. Cross-system comparison

| System (year) | Org | Topology | Agents and roles | Communication | Notable numbers | Status (2026) |
|---|---|---|---|---|---|---|
| Anthropic Research (2025) [1] | Anthropic | Orchestrator-worker plus citation pass | Lead (Opus 4), 1 to more than 10 subagents (Sonnet 4), CitationAgent | Task specs down; summaries or refs up; external memory | +90.2% vs single agent; about 15x chat tokens; up to 90% faster | Production feature |
| Claude Code subagents [5] | Anthropic | Orchestrator-worker, nestable | Explore, Plan, general-purpose, custom | Isolated context; final result only; forks share context | Default nesting depth 3; 20 concurrent | Production |
| Claude Code agent teams [6] | Anthropic | Lead plus peer task board | Lead plus N teammates | Shared task list (file locks) plus mailboxes | Recommended 3-5 teammates | Experimental |
| Cognition single-thread [2] | Cognition | Single agent plus compressor | One agent | Continuous trace | n/a | Design guidance |
| AutoGen (2023-) [7]-[11] | Microsoft | Two-agent chat, group chat, swarm, graph | Assistant plus UserProxy; custom | Shared thread; HandoffMessage; DiGraph | n/a | Maintenance mode |
| Magentic-One (2024) [12] | Microsoft | Orchestrator with ledgers | Orchestrator, WebSurfer, FileSurfer, Coder, Terminal | Task and progress ledgers | Comparable to SOTA on GAIA and AssistantBench | Built on AutoGen |
| MS Agent Framework [13] | Microsoft | Graph workflows plus patterns | Any | Typed workflows; checkpoints | n/a | Active successor |
| MetaGPT (2023) [16][17] | DeepWisdom / FoundationAgents | SOP pipeline | PM, Architect, PjM, Engineer, QA | Structured docs; pub-sub pool | Approx. 85.9% HumanEval pass@1 (paper) | Active; MGX product |
| ChatDev (2023) [18][19] | OpenBMB / Tsinghua | Chat chain, then DAG, then learned | CEO, CPO, CTO, Programmer, Reviewer, Tester, Designer | Paired instructor-assistant chats | MacNet: more than 1,000 agents | 2.0 (DevAll) Jan 2026 |
| CrewAI [14][15] | CrewAI | Sequential or hierarchical crews inside flows | Role, goal, backstory | Task context; manager delegation | About 60k stars | Active |
| OpenAI Swarm (2024) [22] | OpenAI | Handoff | Triage plus specialists | History moves with control | n/a | Replaced by Agents SDK |
| OpenAI Agents SDK [24]-[26] | OpenAI | Handoff and agents-as-tools | Any | Input filters; nested history (beta) | 100+ LLMs supported | Active |
| Google ADK 2.0 [30] | Google | Hierarchies plus graph workflows | Any | Task API; workflow graph | n/a | Active |

---

## 10. Key takeaways for system designers

1. **Default to one agent; earn every additional one.** Both Anthropic [3] and Cognition [2] start from simplicity. Add agents when a measured eval shows the gain is worth roughly 15x token cost [1], not because a role metaphor is appealing.
2. **Parallelize reads, serialize writes.** Breadth-first, read-only exploration (research, code search, multi-lens review) benefits from isolated parallel workers [1][5][6]. Coupled write work (coding, design) suffers from conflicting implicit decisions [2]. Most production subagents are read-only for this reason.
3. **A subagent's main job is context compression.** Design its contract as "spend tens of thousands of tokens, return 1,000-2,000" [4], or write large artifacts to external storage and return references [1]. Measure how good the summary is, not only whether the task completed.
4. **Delegation quality is the bottleneck.** Every task spec needs an objective, an output format, tool and source guidance, boundaries, and an effort budget [1]. Most category-1 and category-2 MAST failures trace back to under-specified delegation [27][29].
5. **Put explicit effort and termination rules in the prompt and in code.** Use agent and tool-call budgets per complexity tier, `maxTurns`, termination conditions, and depth and concurrency limits [1][5][11]. Unbounded group chats run forever by default [11].
6. **Make verification a separate component.** MAST category 3 is a third of the failure surface [27]. Use dedicated verifiers (CitationAgent, LLM-judge rubrics, executable tests, hook-based gates) rather than trusting the worker's own claim of completion [1][6][16].
7. **Prefer hierarchical control with one owner of the final answer.** Use peer topologies (debate, competing hypotheses) only where adversarial diversity is the point, and enforce file or resource ownership to prevent conflicts [6][25].
8. **Handoffs are a routing primitive, not parallelism.** One active agent plus moving history keeps context continuous. Manage context growth with input filters or nested summaries, and audit them for data leakage [24].
9. **Treat the topology as code you can test, and eventually optimize.** The field moved from free chat (AutoGen v0.2, ChatDev chains) to typed graphs (GraphFlow, Agent Framework, ADK workflows) and to learned or searched orchestration (AFlow, the puppeteer) [10][13][17][21][30].
10. **Engineer for long-running, stateful execution.** Use checkpoints and resume-from-failure, tell agents when tools fail, use rainbow deployments for in-flight agents, and trace at the decision-pattern level [1].
11. **Treat inter-agent messages as untrusted input.** Agent-originated messages should never carry user authority such as permission approvals or consent [6]. Agents left unbounded will try to recruit humans and escalate actions [12].

---

## References

1. Hadfield, J., Zhang, B., Lien, K., Scholz, F., Fox, J., Ford, D. "How we built our multi-agent research system." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
2. Yan, W. "Don't Build Multi-Agents." Cognition, 2025. https://cognition.ai/blog/dont-build-multi-agents
3. Schluntz, E., Zhang, B. "Building effective agents." Anthropic Engineering, 2024. https://www.anthropic.com/engineering/building-effective-agents
4. Rajasekaran, P., Dixon, E., Ryan, C., Hadfield, J. "Effective context engineering for AI agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
5. Anthropic. "Create custom subagents." Claude Code Documentation, 2026. https://code.claude.com/docs/en/sub-agents
6. Anthropic. "Orchestrate teams of Claude Code sessions (agent teams)." Claude Code Documentation, 2026. https://code.claude.com/docs/en/agent-teams
7. Wu, Q., et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv 2308.08155, 2023 (COLM 2024). https://arxiv.org/abs/2308.08155
8. Microsoft. "AutoGen" GitHub repository (README, including v0.2 branch). 2023-2026. https://github.com/microsoft/autogen
9. Microsoft Research. "AutoGen" project page (v0.4 redesign). 2025. https://www.microsoft.com/en-us/research/project/autogen/
10. Microsoft. AutoGen AgentChat teams module source (`teams/__init__.py`). 2025. https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/__init__.py
11. Microsoft. AutoGen Swarm group chat source (`_swarm_group_chat.py`). 2025. https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_swarm_group_chat.py
12. Microsoft Research. "Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks." 2024. https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/
13. Microsoft. "Microsoft Agent Framework" GitHub repository. 2025-2026. https://github.com/microsoft/agent-framework
14. CrewAI Inc. "crewAI" GitHub repository. 2023-2026. https://github.com/crewAIInc/crewAI
15. CrewAI Inc. "Processes" documentation (docs/edge/en/concepts/processes.mdx). 2026. https://github.com/crewAIInc/crewAI/blob/main/docs/edge/en/concepts/processes.mdx
16. Hong, S., Zhuge, M., Chen, J., et al., Schmidhuber, J. "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." arXiv 2308.00352, 2023 (ICLR 2024). https://arxiv.org/abs/2308.00352
17. FoundationAgents. "MetaGPT" GitHub repository. 2023-2026. https://github.com/FoundationAgents/MetaGPT
18. Qian, C., et al. "ChatDev: Communicative Agents for Software Development." arXiv 2307.07924, 2023 (ACL 2024). https://arxiv.org/abs/2307.07924
19. OpenBMB. "ChatDev" GitHub repository (ChatDev 2.0 and the chatdev1.0 branch). 2023-2026. https://github.com/OpenBMB/ChatDev
20. Qian, C., et al. "Scaling Large-Language-Model-based Multi-Agent Collaboration" (MacNet). arXiv 2406.07155, 2024. https://arxiv.org/abs/2406.07155
21. Dang, Y., Qian, C., et al. "Multi-Agent Collaboration via Evolving Orchestration." arXiv 2505.19591, 2025 (NeurIPS 2025). https://arxiv.org/abs/2505.19591
22. OpenAI. "Swarm" GitHub repository. 2024. https://github.com/openai/swarm
23. OpenAI. "Orchestrating Agents: Routines and Handoffs." OpenAI Cookbook, 2024. https://github.com/openai/openai-cookbook/blob/main/examples/Orchestrating_agents.ipynb
24. OpenAI. "Handoffs." OpenAI Agents SDK Documentation, 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md
25. OpenAI. "Orchestrating multiple agents." OpenAI Agents SDK Documentation, 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md
26. OpenAI. "OpenAI Agents SDK" GitHub repository. 2025-2026. https://github.com/openai/openai-agents-python
27. Cemri, M., Pan, M. Z., Yang, S., Agrawal, L. A., Chopra, B., Tiwari, R., Keutzer, K., Parameswaran, A., Klein, D., Ramchandran, K., et al. "Why Do Multi-Agent LLM Systems Fail?" arXiv 2503.13657, 2025. https://arxiv.org/abs/2503.13657
28. MAST authors. "MAST: Multi-Agent Systems Failure Taxonomy" GitHub repository and MAD dataset pointer. 2025. https://github.com/multi-agent-systems-failure-taxonomy/MAST
29. MAST authors. "Taxonomy definitions" (definitions.txt). 2025. https://github.com/multi-agent-systems-failure-taxonomy/MAST/blob/main/taxonomy_definitions_examples/definitions.txt
30. Google. "Agent Development Kit (ADK) for Python" GitHub repository (ADK 2.0 README). 2025-2026. https://github.com/google/adk-python
