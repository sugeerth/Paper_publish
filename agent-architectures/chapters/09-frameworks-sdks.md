# 09. Agent Frameworks and SDKs: LangGraph, OpenAI Agents SDK, Claude Agent SDK, Google ADK, Pydantic AI, smolagents, Mastra, Strands, Microsoft Agent Framework, LlamaIndex, DSPy

## Overview

Earlier chapters described agent architectures in the abstract: the tool loop (Chapter 02), planning (03), multi-agent topologies (04), and context engineering (05). This chapter covers the software that turns those ideas into code. By late 2026 the open-source framework landscape has settled around a few distinct design philosophies. Most major model vendors (OpenAI, Anthropic, Google, AWS, Microsoft) ship their own SDK, and several independent projects (LangGraph, Pydantic AI, smolagents, Mastra, LlamaIndex, DSPy) occupy specific niches.

The frameworks agree on a common core and disagree on what to add to it. Every one of them implements the same inner loop: call a model with tools, execute the requested tool calls, append the results, repeat until a stop condition. Anthropic's December 2024 guidance described agents as "LLMs using tools based on environmental feedback in a loop." It also warned that frameworks "often create extra layers of abstraction that can obscure the underlying prompts," and that "incorrect assumptions about what's under the hood are a common source of customer error" [1]. The design space is mostly about what a framework puts *around* that loop:

1. **Who owns control flow.** Some frameworks let the model drive the loop (Claude Agent SDK, Strands, OpenAI Agents SDK handoffs). Others put developer-defined graphs in charge (LangGraph, ADK 2.0 Workflow, Microsoft Agent Framework workflows, Mastra workflows, pydantic-graph). Most now offer both.
2. **Where state lives and how it survives failure.** Options include per-superstep checkpoints (LangGraph, Microsoft Agent Framework), event-sourced sessions (ADK), external durable-execution engines (Pydantic AI on Temporal or DBOS), suspend snapshots (Mastra), and on-disk transcripts with resume and fork (Claude Agent SDK).
3. **What the action space is.** It may be JSON tool calls, code as action (smolagents, Microsoft's CodeAct provider, DSPy's RLM and ProgramOfThought), or a whole computer with bash and a file system (Claude Agent SDK, OpenAI sandbox agents).
4. **How behavior is improved.** Most frameworks rely on hand-written prompts plus tracing and evals. DSPy instead *compiles* prompts and weights against a metric.
5. **Where policy is enforced.** Mechanisms include guardrails and tripwires (OpenAI), hooks and permission modes (Claude), middleware (LangChain v1, Microsoft), callbacks (ADK), and information-flow control (Microsoft's FIDES proposal).

The chapter takes each system in turn: its problem, architecture, key techniques, public numbers, failure modes and lessons. It ends with a comparison table and design takeaways.

A note on sourcing: this chapter relies mainly on the frameworks' own repositories, design records (ADRs), and documentation as of October 2026. Star counts and version numbers were read from GitHub at that time and are approximate.

---

## 1. A shared reference model

Before going framework by framework, it helps to fix one vocabulary. Nearly every system below breaks down into the same seven components:

```
  +------------------------------------------------------------------------+
  |  APPLICATION / HOST PROCESS                                            |
  |                                                                        |
  |   input --> [ Runner / Executor ] ---------------------------> output  |
  |                 |   ^      |                                           |
  |                 |   |      +--> [ Policy layer: guardrails, hooks,     |
  |                 v   |           middleware, permissions, approvals ]   |
  |           +-----------+     +-----------+     +---------------------+  |
  |           |  Agent    |---->|  Model    |     | Tools: functions,   |  |
  |           | (prompt,  |<----|  client   |     | MCP, hosted, code   |  |
  |           |  tools,   |     +-----------+     | exec, sub-agents    |  |
  |           |  output   |-------------------->  +---------------------+  |
  |           |  schema)  |                                                |
  |           +-----------+                                                |
  |                 |                                                      |
  |                 v                                                      |
  |   [ State: session / thread / checkpoint store ]  [ Tracing: OTel ]    |
  |   [ Orchestration: graph | handoff | agent-as-tool | code ]            |
  +------------------------------------------------------------------------+
```

The frameworks differ in which box they treat as primary. For LangGraph, Microsoft's workflows, and ADK 2.0 the orchestration graph is the main object. OpenAI and Strands center the agent and runner, Claude centers the tool surface and the computer, Mastra and Pydantic AI center typed data, and DSPy centers an optimizable *program*.

---

## 2. LangGraph (LangChain)

### 2.1 Problem

LangGraph describes itself as a "low-level orchestration framework for building, managing, and deploying long-running, stateful agents." It emphasizes durable execution, human-in-the-loop, and both short- and long-term memory. The README names Klarna, Replit and Elastic as users [2]. Its premise is that production agents need explicit, inspectable, resumable control flow more than a high-level "agent" abstraction.

### 2.2 Architecture: Pregel over channels

The README says LangGraph is "inspired by Pregel and Apache Beam," and its public interface draws on NetworkX [2]. The runtime is a literal implementation of Google's bulk-synchronous-parallel (BSP) model. The `Pregel` class combines **actors** (`PregelNode`s, which are LangChain Runnables) with **channels** that carry data between them. Each step has three phases [3]:

- **Plan.** Select the actors to run. On the first step these are the actors subscribed to the input channels. On later steps they are the actors subscribed to channels updated in the previous step.
- **Execution.** Run all selected actors in parallel until every one completes, one fails, or a timeout is reached. Channel writes stay invisible to other actors until the next step.
- **Update.** Apply the writes to the channels.

The loop ends when no actor is selected or a maximum step count is reached. Cycles are allowed: an actor may write to a channel it subscribes to [3].

```
   superstep t                    superstep t+1
  +-----------------+  writes   +-----------------+
  | node A  node B  | --------> | channels updated| --> plan: who subscribes?
  | (parallel, no   |  (buffer) | reducers apply  |     node C, node A ...
  |  shared view)   |           +-----------------+
  +-----------------+                    |
                                         v
                              checkpoint(thread_id, checkpoint_id)
```

Channel types encode the semantics of state merges. `LastValue` (the default) keeps the last write. `Topic` is a pub/sub accumulator that can deduplicate. `BinaryOperatorAggregate` folds each update with a reducer. `Context` manages a resource lifecycle [3]. The familiar `StateGraph` API, with typed state and per-key reducers, compiles down to this channel model.

Dynamic control flow comes from two primitives in `langgraph.types` [4]:

- **`Send(node, arg)`** pushes a task to a specific node with state that can differ from the graph's main state. This is the basis for map-reduce fan-out.
- **`Command(update, goto, resume, graph)`** lets a node update state and route in a single return value. Setting `graph=Command.PARENT` targets the enclosing graph, which is how sub-graph agents hand control back up.

### 2.3 Persistence, durability and human-in-the-loop

A checkpoint is saved at every superstep, keyed by `thread_id`, with an optional `checkpoint_id` to start from an earlier point [5]. The `BaseCheckpointSaver` interface is small: `put`, `put_writes`, `get_tuple`, `list`, `delete_thread`, and `get_next_version` for channel versions, with async variants. Separate SQLite and Postgres checkpoint packages are versioned on their own release trains [5][6].

The most important fault-tolerance detail is **pending writes**. If one node fails partway through a superstep, the writes from sibling nodes that succeeded are stored. On resume those nodes are skipped rather than re-run [5]. A coarser design would redo all of them, along with any side effects.

Durability is a tunable trade-off. `Durability = "sync" | "async" | "exit"` [4]:

- `sync` persists before the next step starts.
- `async` persists in the background while the next step runs.
- `exit` persists only when the graph finishes.

This is a direct latency versus crash-safety dial. Nodes can also carry a `RetryPolicy` (defaults: initial interval 0.5 s, backoff factor 2.0, max interval 128 s, 3 attempts, jitter on) and a `CachePolicy` with a key function and TTL [4].

Human-in-the-loop is built on the same machinery. `interrupt()` raises an `Interrupt(value, id)`, the checkpoint persists, and a later `Command(resume=...)` supplies the value, either per interrupt ID or for the next pending interrupt [4]. Seven stream modes expose the runtime at different granularities: `values`, `updates`, `custom`, `messages` (token by token), `checkpoints`, `tasks`, and `debug` [4].

### 2.4 The agent layer: LangChain v1 middleware

In the 1.x era, LangChain's `create_agent` is a prebuilt LangGraph graph whose behavior is customized through **middleware** rather than by subclassing. `AgentMiddleware` exposes `before_agent`, `before_model`, `wrap_model_call`, `after_model`, `wrap_tool_call`, and `after_agent`. A `wrap_*` handler can call the inner handler several times (to retry) or skip it (to short-circuit). Hooks can return `{"jump_to": "tools" | "model" | "end"}`, which compiles to conditional edges [7]. The shipped middleware catalog shows which concerns production users needed factored out [8]:

- summarization and context editing
- PII redaction
- human-in-the-loop
- model fallback and retry
- tool retry and tool-error handling
- model- and tool-call limits
- LLM tool selection and provider tool search
- a todo planner, file search, and a shell tool
- a tool emulator for testing

### 2.5 Failure modes and lessons

- **Default serialization is permissive.** By default the checkpoint serializer accepts arbitrary Python types. The README recommends `LANGGRAPH_STRICT_MSGPACK=true` or an explicit module allowlist for new applications [5], since deserializing untrusted checkpoints is a code-execution risk.
- **Reducer design is part of the API.** Two parallel nodes writing a `LastValue` key in the same superstep conflict. Accumulating keys need an explicit reducer.
- **Point releases are mostly runtime bugs.** Releases are frequent; the 1.2.x line was at 1.2.15 on 12 October 2026. Their notes are dominated by fixes to delta channels, checkpoint history and interrupts [6], which is where the hard bugs in a BSP runtime tend to sit.
- **Lesson:** separating "graph runtime" from "agent recipe" let LangChain keep a stable low-level core while replacing the high-level agent API (from `create_react_agent` to middleware-based `create_agent`).

---

## 3. OpenAI Agents SDK

### 3.1 Problem and lineage

OpenAI's Swarm was an "educational" and stateless library built on two primitives, agents and handoffs. The caller carried message history and context variables between calls [18]. The Agents SDK is described as its "production-ready evolution" [18]. It keeps the small set of primitives and adds guardrails, sessions, tracing, human-in-the-loop, voice/realtime agents, and (in 2026) **sandbox agents**. It is provider-agnostic: it supports the Responses and Chat Completions APIs "as well as 100+ other LLMs" [9].

### 3.2 The runner loop

`Runner.run()`, `run_sync()` and `run_streamed()` take a starting agent and an input. The input can be a string, a list of Responses items, or a `RunState` to resume a run. Each turn calls the current agent's model, then [10]:

1. If the output is final (text of the desired `output_type` with no tool calls), the loop returns.
2. If the model requested a handoff, the loop switches the current agent and repeats.
3. Otherwise it executes the tool calls, appends the results, and repeats.

```
            +---------------------------------------------+
  input --> | Runner: current_agent = starting_agent      |
            |   loop (max_turns):                          |
            |     resp = model(current_agent, history)     |
            |     final?  ---------------------------------+--> output guardrails
            |     handoff? -> current_agent = target       |        -> result
            |     tools?   -> execute (guarded), append    |
            +---------------------------------------------+
               ^ input guardrails (parallel or blocking)
```

`max_turns` raises `MaxTurnsExceeded`. Error handlers for `max_turns`, `model_refusal` and `invalid_final_output` can return a controlled final output instead of raising [10]. `RunConfig` exposes a lot of tuning, including:

- a cap on concurrent local function tools (`max_function_tool_concurrency`, uncapped by default)
- tool-not-found behavior (raise, or "return_error_to_model")
- a tool-name collision policy
- `call_model_input_filter`, which edits the model input immediately before each call

Per agent, `tool_use_behavior` decides whether tool results go back to the model (`run_llm_again`, the default) or end the run (`stop_on_first_tool`, `StopAtTools`, or a custom function). `reset_tool_choice` (on by default) sets `tool_choice` back to `auto` after a tool call so a forced tool choice cannot loop forever [11].

### 3.3 Handoffs versus agents-as-tools

A handoff is presented to the model as a tool named `transfer_to_<agent_name>`. `handoff()` options include `on_handoff` (a callback, for example to prefetch data), `input_type` (a schema for model-generated metadata such as a reason or priority, *not* application state), `input_filter` (rewrites the history the receiving agent sees), and `is_enabled` [12]. By default the receiving agent sees the full history. A beta `nest_handoff_history` mode condenses prior history into summaries, and the docs note that this does not redact sensitive data [12].

The SDK's multi-agent guide sets this out as a two-way choice [16]:

- **Agents as tools** (`Agent.as_tool()`): the manager keeps control, combines specialist outputs, and enforces shared guardrails.
- **Handoffs**: the specialist takes over the rest of the turn and replies directly.

The guide also separates "orchestrating via LLM" from "orchestrating via code" (structured-output routing, chaining, evaluator loops, `asyncio.gather` parallelism). It recommends code orchestration when speed, cost and performance need to be predictable [16].

### 3.4 Guardrails and their latency trade-off

There are three kinds of guardrail [13]:

- **Input guardrails** run only on the *first* agent.
- **Output guardrails** run only on the agent that produces the final output.
- **Tool guardrails** wrap each function-tool call. They do not apply to handoffs.

The key design decision is `run_in_parallel`. By default, input guardrails run *concurrently* with the agent to cut latency. If a tripwire fires, the agent may already have spent tokens and run tools. Blocking mode avoids those side effects at the cost of latency [13]. When an output tripwire fires, completed tool calls are still saved to the session but the rejected output is not, and terminal tool output can be replaced with a placeholder [13].

### 3.5 Sessions, tracing and sandbox agents

State can live in four places [10]:

- app memory (`to_input_list()`)
- an SDK `Session`
- OpenAI's Conversations API (`conversation_id`)
- server-side chaining (`previous_response_id`)

These are mutually exclusive in a single run. Built-in sessions cover SQLite (sync and async), Redis, SQLAlchemy, MongoDB and Dapr. There are also OpenAI-hosted conversations, an `OpenAIResponsesCompactionSession` wrapper that compacts long histories through the Responses API, an `AdvancedSQLiteSession` with branching and usage analytics, and an `EncryptedSession` wrapper with TTL [15].

Tracing is on by default. Each run is wrapped in an "Agent workflow" trace, with nested spans for agents, generations, tools, guardrails and handoffs. A `BatchTraceProcessor` exports to OpenAI's backend, and more than a dozen third-party processors are listed [14]. Several design details stand out:

- `trace_include_sensitive_data` defaults to **True**.
- Tracing is unavailable for organizations under Zero Data Retention.
- Processors are independent. A redactor registered *alongside* an exporter therefore does not guarantee the exporter only sees redacted data. The docs recommend putting redaction and delivery in one exporter [14].

**Sandbox agents** (`SandboxAgent`) give the model a persistent workspace. A `Manifest` declares files, repos and mounts. *Capabilities* add filesystem, shell, skills, memory and compaction tools. A `SandboxRunConfig` picks the compute client (Unix-local, Docker, hosted providers) and how later runs reconnect: session, session state, or snapshot [17]. The docs are explicit that Unix-local execution is not a sandbox. On Linux it adds no OS-level confinement, and on macOS `sandbox-exec` restricts the filesystem but not the network [17].

### 3.6 Lessons

The SDK's evolution shows that a small primitive set (agent, tool, handoff) was enough for control flow. Most later additions were operational: state backends, tracing hygiene, guardrail timing, sandboxed compute. The parallel-by-default guardrail is a clear example of a framework choosing latency over strict safety by default and documenting the escape hatch.

---

## 4. Claude Agent SDK (Anthropic)

### 4.1 Problem

The Claude Agent SDK takes the opposite approach from graph frameworks. It ships a full, opinionated harness (the one that powers Claude Code) and lets the model drive. In September 2025 Anthropic renamed the Claude Code SDK to the Claude Agent SDK because the same harness was powering "many non-coding agents." The stated principle is to "give your agents a computer": bash, file editing and file search, so the agent can work the way a person at a terminal does [19].

### 4.2 Architecture: a library wrapping an agent binary

The SDK is a Python/TypeScript library that runs the Claude Code binary, which both SDKs bundle. Anthropic positions it against three neighbors [20]:

- the **CLI**, for interactive use
- the **Client SDK**, a raw API where you write the tool loop yourself or use a beta tool runner
- **Managed Agents**, where Anthropic hosts the harness and sessions run in a managed or self-hosted sandbox

The Python package exposes `query()` (one-shot streaming) and `ClaudeSDKClient` (bidirectional, with custom tools and hooks). Custom tools are **in-process MCP servers** (`create_sdk_mcp_server`), which avoids subprocess and IPC overhead. They can be mixed with external stdio MCP servers [22].

```
  your app (Python/TS)
     |  query()/ClaudeSDKClient  -- stream of SystemMessage, AssistantMessage,
     v                               UserMessage, StreamEvent, ResultMessage
  +---------------------------+
  | bundled Claude Code binary|--> Claude API (prompt-cached prefix)
  |  agent loop               |
  |  built-in tools: Read,    |--> hooks (PreToolUse, PostToolUse, Stop,
  |  Edit, Write, Glob, Grep, |     SubagentStart/Stop, PreCompact, ...)
  |  Bash, WebSearch/Fetch,   |      run in YOUR process, cost no context
  |  ToolSearch, Agent, Skill |--> permissions: allow/deny rules + mode
  |  sessions on disk (+store)|--> MCP servers (in-process or external)
  +---------------------------+
```

### 4.3 The loop and its controls

The loop is "gather context, take action, verify work, repeat" [19]. Each turn is one model call with tool calls, executed automatically, until a response contains no tool calls. The SDK then emits a `ResultMessage` carrying cost, usage and a session ID [21]. Read-only tools (`Read`, `Glob`, `Grep`, and MCP tools annotated `readOnlyHint`) run **concurrently**. State-modifying tools (`Edit`, `Write`, `Bash`) run sequentially. Custom tools are sequential by default [21].

The loop has two independent caps. `max_turns` counts tool-use round trips, and `max_budget_usd` stops at an estimated spend. The budget covers subagents, and the docs warn it can be overshot by one response plus whatever in-flight subagents spend [21]. Result subtypes make termination machine-readable: `success`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution`, `error_max_structured_output_retries` [21]. An `effort` setting (`low` to `max`) trades reasoning depth for cost per turn [21].

### 4.4 Permissions and hooks as the policy layer

Policy has three parts that are evaluated in a fixed order [21][22]:

- **`allowed_tools`** is an auto-approve list, not a restriction. Unlisted tools fall through to the permission mode and the `canUseTool` callback.
- **`disallowed_tools`** hard-blocks tools.
- **Permission modes:**
  - `default`: prompt via callback, and deny if there is no callback
  - `acceptEdits`
  - `plan`: no edits
  - `dontAsk`: deny anything that would prompt
  - `auto`: a model classifier reviews actions
  - `bypassPermissions`: for isolated CI and containers only, and refused when running as root

Hooks such as `PreToolUse` are deterministic code that "the Claude Code application" invokes rather than the model. They can deny a call with a reason, which Claude then sees as the tool result [22]. Because hooks run in the host process, they consume no context [21].

### 4.5 Context management

Three mechanisms keep context under control [21]:

- **Automatic compaction.** Older history is summarized near the context limit, marked by a `compact_boundary` event. Instructions that must survive belong in CLAUDE.md, which is re-injected on every request.
- **Subagents.** Each starts with fresh history, and "only its final response returns to the parent."
- **Tool search.** It defers MCP tool schemas by default so that many MCP servers do not inflate every request.

Stable prefixes (system prompt, tool definitions, CLAUDE.md) are prompt-cached [21]. Sessions can be resumed and **forked**. A `session_store` adapter mirrors transcripts to your own backend so stateless hosts can resume them [21].

### 4.6 Numbers and lessons from the same harness

Anthropic's multi-agent research system reports useful cost and scale data [23]:

- An Opus 4 lead with Sonnet 4 subagents beat single-agent Opus 4 by **90.2%** on an internal research eval.
- Agents use roughly **4x** the tokens of chat, and multi-agent systems roughly **15x**.
- Token usage alone explained **80%** of performance variance on BrowseComp.
- Parallel subagents cut research time by "up to 90%" for complex queries.

The production lessons were checkpoint-and-resume instead of restart, full tracing, and "rainbow deployments" so that long-running agents are not broken mid-run [23]. Anthropic also notes that LLM-as-judge verification "is not very robust" and adds latency, and it recommends rule-based checks (linting, typed languages) first [19].

---

## 5. Google Agent Development Kit (ADK)

### 5.1 Problem

ADK is "code-first," model-agnostic but "optimized for Gemini," and ships in Python, TypeScript, Go, Java and Kotlin. It comes with a CLI that covers a dev UI (`adk web`), evals (`adk eval`), and deployment to Docker, Cloud Run, or Vertex AI Agent Engine [24]. Releases come roughly every two weeks [24].

### 5.2 Event-sourced runtime (1.x core, retained in 2.0)

ADK's distinguishing idea is an **event loop with commit-on-yield semantics**. The `Runner` orchestrates one invocation. Agent, tool and callback code ("execution logic") yields an `Event` carrying content plus `actions` (`state_delta`, `artifact_delta`) and then **pauses**. The Runner commits the deltas through `SessionService` and `ArtifactService`, forwards the event upstream, and only then lets the logic resume [25].

```
  Execution logic                       Runner                     Services
  ---------------                       ------                     --------
  build Event(content, state_delta) --> receive
  yield; PAUSE                          commit deltas  ----------> SessionService
                                        append to history          ArtifactService
                                        forward upstream (UI)
  RESUME (state now committed)  <------ signal done
```

The docs spell out the consequences [25]:

- **Dirty reads.** Code later in the same step can see local, uncommitted changes. If the invocation fails before the event is processed, those changes are lost, so critical transitions should be tied to a processed event.
- **Partial events.** Streaming events marked `partial=True` are forwarded to the UI but their actions are skipped. State is committed once, from the final event.
- **Scope prefixes.** State keys prefixed `temp:` are discarded at the end of the invocation.
- **Blocking tools.** Sync tools run inline on the asyncio loop, so blocking I/O stalls it.

`LlmAgent` is configured declaratively [26]:

- `description` is what *other* agents read when deciding whether to route to this one.
- `instruction` supports `{var}`, `{artifact.var}`, and an optional `{var?}` template filled from session state.
- `output_key` writes the final response into state.
- `include_contents='none'` hides prior history from the agent.
- Planners are either `BuiltInPlanner`, which uses native thinking budgets, or `PlanReActPlanner` for models without one.

### 5.3 From agent tree to graph: ADK 2.0

ADK 1.x composed agents as a tree. `LlmAgent`s delegated to `sub_agents`, and the deterministic **workflow agents** (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) orchestrated without an LLM [27]. ADK 2.0 adds a **Workflow Runtime** that "moves ADK from a hierarchical agent executor to a graph-based execution engine": agents, tools and functions all become nodes. Python went GA on 19 May 2026, Go on 30 June 2026 and TypeScript on 21 August 2026 [28]. The changes are breaking in several places [28]:

- `BaseAgent` now subclasses `BaseNode`.
- Overriding `_run_async_impl` is *silently ignored*.
- `Event` gains `node_info`, `output` and route fields, which breaks custom session stores with rigid schemas.
- The template workflow agents are deprecated in TypeScript.
- Catching `BaseException` in tool code now traps `NodeInterruptedError` and breaks human-in-the-loop pauses. Broad `except` blocks also hide failures from the new `RetryConfig`.

In the Python graph API, edges are tuples such as `("START", a, f, b)`. A router node returns `Event(route=...)` and a dict maps route values to handlers. Returning a list of routes fans out. A node's return value becomes the next node's `node_input`, so session state does not need to be used as a message bus [29]. Go adds explicit `NewJoinNode` fan-in and typed `StringRoute`, `IntRoute` and `BoolRoute` matchers [29].

**Collaborative workflows** formalize LLM delegation [30]. ADK auto-generates a delegation tool for each sub-agent. Each sub-agent has a *mode*:

- `chat` returns control only by explicit `transfer_to_agent`.
- `task` can ask the user clarifying questions and returns on `finish_task`.
- `single_turn` involves no user interaction and supports parallel instances.

Task and single-turn agents run in their own **session branch** and cannot see peers' events. That gives context isolation by construction [30]. An experimental TypeScript `RoutedAgent` adds code-defined routing with failover semantics. A failed agent key can be retried with a different agent only if it failed *before* yielding any events [31].

### 5.4 Lessons

ADK's path from a tree with template agents to a graph runtime mirrors LangGraph's and Microsoft's. Fixed Sequential/Parallel/Loop combinators proved too rigid. The commit-on-yield event model is a strong design for UI streaming and auditability. The 2.0 migration notes show that changing the execution substrate under existing agent subclasses is costly: silently ignored overrides are a hard class of bug.

---

## 6. Pydantic AI

### 6.1 Problem

Pydantic AI aims to bring type safety and dependency injection to agents. It describes its approach as "How Python does AI" [32]. Agents are generic over a dependency type and an output type. Tools receive a typed `RunContext`. Results are validated Pydantic models, and validation failures can be fed back to the model for retry. Observability is OpenTelemetry with one-line Logfire setup [32].

### 6.2 Durable execution as a pluggable layer

Pydantic AI does not build its own checkpointer. It delegates durability to existing engines and wraps "every model request and tool call as its own durable unit" [33]. Supported engines include:

- co-maintained with the vendors: Temporal, DBOS, Prefect, Restate and AWS Lambda
- other integrations: Kitaru, Airflow and Absurd
- a builder for adding further engines

Only one engine can be attached per agent [33]. The docs distinguish *durability* (keeping one run alive across failures) from *persistence* (storing chat threads), which many frameworks blur [33]. The cost is that you take on the operational model of the chosen engine. Its determinism rules also apply. For example, under Temporal a delegate tool receives a *copy* of the run context, so `ctx.usage` does not propagate back to the parent [34].

### 6.3 A ladder of multi-agent complexity

The documentation lays out five levels, with the explicit advice "start simple" [34]:

1. a single agent
2. **agent delegation**: an agent called inside a tool, sharing `usage=ctx.usage` so limits apply across the tree
3. **programmatic hand-off** in application code
4. **graph-based control flow** with `pydantic-graph`
5. **deep agents**, which combine planning, files, delegation, sandboxed code, approval and durability

`UsageLimits` (requests, tokens, tool calls, cost) guard against runaway loops [34]. The pydantic-graph library derives **edges from the return-type annotations** of node `run` methods. A union such as `NodeB | End[int]` *is* the edge set, so the type checker validates the graph and Mermaid diagrams are generated from it. A newer `GraphBuilder` API adds explicit step and edge wiring [35]. The docs call graphs "a nail gun" next to the agent "hammer" and say: "if you're not confident a graph-based approach is a good idea, it might be unnecessary" [35].

---

## 7. smolagents (Hugging Face)

### 7.1 Problem and thesis

smolagents is deliberately minimal: the core logic in `agents.py` is under 1,000 lines [36]. Its thesis is **code as action**. A `CodeAgent` writes Python in which tool calls are function calls. A `ToolCallingAgent` uses native JSON tool calling [36]. The rationale draws on the CodeAct paper (Wang et al., ICML 2024). That paper found code actions gave "up to 20% higher success rate" than text or JSON across 17 LLMs on API-Bank and the new M3ToolEval benchmark [40]. The README claims roughly **30% fewer steps** (and so fewer LLM calls) for code agents [36]. The docs give four reasons for preferring code [37]:

- composability, since actions nest and can be reused as functions
- object management, since outputs live in variables
- generality
- the large amount of code in training data

The docs also define an "agency spectrum" from simple processor through router, tool call, multi-step agent and multi-agent, up to code agent. They advise deterministic code whenever a fixed workflow is enough [37].

### 7.2 Loop and memory

`MultiStepAgent.run` loops up to `max_steps` (default 20) [39]. Memory is a typed log of steps (`TaskStep`, `ActionStep`, `PlanningStep`, `FinalAnswerStep`) that is re-rendered into messages on each turn. If `planning_interval` is set, a planning step runs on step 1 and every N steps after that, using a summary view of memory [39]. The `final_answer` tool ends the run. Optional `final_answer_checks` can reject an answer and keep the loop going. If `max_steps` is hit, the agent is asked for a best-effort final answer [39]. Managed agents are exposed to a parent as named callables that take a `task` string [39].

### 7.3 Executing untrusted code

Code-as-action moves the security problem from tool permissions to code execution. The default `LocalPythonExecutor` is **not** the standard interpreter. It walks the AST, blocks imports outside an allowlist, gates submodule access, caps the number of elementary operations to stop infinite loops, and errors on any undefined operation [38]. Even so, the docs state that "no local python sandbox can ever be completely secure." An allowed package such as Pillow could still be abused to fill a disk [38]. Remote executors (E2B, Modal, Docker, Blaxel) support two topologies [38]:

```
  Approach 1: snippets only                Approach 2: whole agent in sandbox
  +-----------+   code    +---------+      +---------------------------------+
  | agent+LLM |---------->| sandbox |      | sandbox: agent + model client + |
  | (host,    |<----------| (E2B,   |      |          tools + executor       |
  | API keys) |  result   | Docker) |      +---------------------------------+
  +-----------+           +---------+        stronger isolation, keys inside
  simple, keys stay local, no managed
  multi-agent, state shuttled each step
```

---

## 8. Mastra (TypeScript)

### 8.1 Problem

Mastra targets the TypeScript and web-application ecosystem: React, Next.js, Node, or a standalone server. It provides model routing across 40+ providers, agents, a graph workflow engine, memory, RAG, evals, observability and MCP exposure, and integrates with Vercel's AI SDK UI and CopilotKit [41].

### 8.2 Workflows, snapshots and recovery

Steps are declared with `createStep({id, inputSchema, outputSchema, execute})`. Schemas can be Zod, Valibot, ArkType or Standard JSON Schema. Steps are composed with `.then()` (plus `.branch()` and `.parallel()`) and finalized with `.commit()`. A workflow can itself be a step [41][42]. A run's status is a discriminated union: `success | failed | suspended | tripwire | paused` [42]. Shared `stateSchema` state persists across suspend and resume. Execution uses a built-in engine by default, or can be handed to runners such as Inngest. `restart()` resumes a run from its last active step, and local servers restart active runs automatically on startup [42].

Durability is **snapshot-on-suspend**. When a step calls `suspend()`, the whole execution state is serialized into a `workflow_snapshots` table keyed by `runId` and workflow name. That state includes per-step context, `activePaths`, `suspendedPaths`, `waitingPaths` and a serialized step graph. libSQL is the default store, with Postgres, Upstash and others available. `run.resume({step, resumeData})` validates the payload against the step's `resumeSchema` [43]. Mastra snapshots at suspension points rather than at every step, unlike LangGraph's per-superstep checkpoints.

### 8.3 Observational memory

Mastra's memory design addresses context growth without retrieval. Two background agents do the work [44]:

- An **Observer** turns raw message history into dense observations once messages exceed about 30k tokens (default), buffering roughly every 6k tokens.
- A **Reflector** condenses the observation log once it exceeds about 40k tokens.

Above about 1.2x those thresholds, processing becomes synchronous. The docs claim typical compression of **5x to 40x**; their example turns a 50k+ token Playwright page snapshot into a few hundred tokens. Because observations *append* rather than being retrieved per turn, the prompt prefix stays cacheable. Activation can be deferred until the provider's cache TTL has expired [44]. The docs claim higher accuracy and lower cost than semantic recall but do not publish figures [44].

---

## 9. Strands Agents (AWS)

### 9.1 Problem and philosophy

Strands launched as "a model-driven approach to building AI agents in just a few lines of code." In practice, a model, a prompt and tools are combined, and the model plans inside a simple loop [46]. By 2026 the project had become a Python and TypeScript monorepo (`harness-sdk`) with two layers [45]:

- a **harness** (`create_harness()`) with defaults for model, tools, memory, sessions and context management
- an **SDK** for teams that "would otherwise write their own agent loop"

It runs in-process with no hosted control plane. Amazon Bedrock, Anthropic, OpenAI and Gemini are first-class providers [45].

### 9.2 Design record

Strands publishes its tenets and decision log, which makes its design reasoning unusually visible. The tenets include "simple at any scale," "the obvious path is the happy path," "accessible to humans and agents," and "embrace common standards" [47]. The 2026 decisions include [48]:

- **Hooks are low-level primitives.** Retry strategies, session managers and conversation managers are domain-specific interfaces that extend `HookProvider`. The rationale is that a bare hook gives implementors "no guidance."
- **Pay for play.** Small breaking changes are allowed only for opt-in features.
- **Low- and high-level APIs together.** For example, `BidiAgent` has `send`/`receive` plus `run`.
- **Native vocabulary.** "Agent loop" is preferred over "event loop," which already means something in Python and JavaScript.
- **LLM-native units.** Sizes are expressed in tokens rather than characters.

Feature-wise, the loop exposes turn limits, token budgets, cancellation and stop reasons, and it is traced by default. *Steering handlers* let agents self-correct rather than fail silently [45][46].

---

## 10. Microsoft Agent Framework (successor to Semantic Kernel and AutoGen)

### 10.1 Consolidation

Microsoft ran two agent stacks in parallel:

- **Semantic Kernel** was an enterprise SDK built around a kernel, plugins (native functions, prompt templates, OpenAPI, MCP) and filters [56].
- **AutoGen** was a research-born multi-agent framework with a layered design: a Core API for event-driven message passing across local and distributed runtimes, AgentChat for group-chat patterns, and Extensions. It also produced Magentic-One [57].

Both now point to **Microsoft Agent Framework (MAF)**. AutoGen is in maintenance mode and "will not receive new features," and the Semantic Kernel README calls MAF "the enterprise-ready successor," available at 1.0 [56][57]. MAF supports Python and .NET, with Go in a separate repository. It provides agents plus graph-based **workflows** with checkpointing, streaming, human-in-the-loop and "time-travel," along with middleware, OpenTelemetry, a DevUI, and experimental "AF Labs" packages for benchmarking and RL [49].

### 10.2 Workflow runtime: Pregel again

MAF's runner is described in its source as "a class to run a workflow in Pregel supersteps" [50]. It works as follows [50]:

- **Delivery.** Pending messages are grouped by source executor and delivered over edge runners. Delivery is concurrent across targets, and order is preserved per edge.
- **State.** Shared-state changes commit at the superstep boundary and are discarded if the superstep fails.
- **Checkpoints.** A checkpoint is taken after each superstep. A checkpoint failure only logs a warning, and the next checkpoint is parented to the last good one.
- **Restore.** Restoring requires that the graph signature hash still matches.
- **Convergence.** The default limit is 100 supersteps. Hitting it with messages still pending raises a convergence error.
- **Human-in-the-loop.** Pending `request_info` events are restored along with the checkpoint.

LangGraph, MAF and ADK 2.0 have thus independently arrived at the same BSP-with-checkpoints core.

### 10.3 Design via ADRs

MAF's `docs/decisions` folder holds more than 40 ADRs and is a good primary source for framework design trade-offs:

- **Middleware (ADR-0007).** The team chose a decorator pattern (`DelegatingAIAgent` with a `.Use()` builder) over Semantic Kernel-style per-type filter interfaces or a central processor. Middleware types cover agent runs, function invocation and approval requests [51].
- **Long-running operations (ADR-0009, accepted October 2025).** Callers opt in per call with `AllowLongRunningResponses`, and responses carry provider-specific **continuation tokens** that callers poll with. This maps onto background modes such as the OpenAI Responses API. Long-running runs require a thread so that tool calls are captured [52].
- **Context compaction (ADR-0019, accepted February 2026).** The original architecture loaded history only once per run, and chat middleware only saw a copy of the messages. Neither allowed compaction *inside* a long tool loop. The fix is a standalone `CompactionStrategy` that runs before **every** model call, before history is written, and on explicit request against stored history. Strategies (truncation, summarization, selective removal of tool-call pairs, a sliding window anchored on system messages) chain under a `TokenBudgetComposedStrategy`. They must keep tool calls and their results "together" to avoid API errors [53].
- **Prompt-injection defense, FIDES (ADR-0024, proposed January 2026).** This applies information-flow control as middleware. Content carries integrity labels (TRUSTED/UNTRUSTED) and confidentiality labels, and the most restrictive label wins when content is combined. Untrusted content is moved into a variable store and referenced by placeholder so it never enters the LLM context. A `quarantined_llm` tool processes it in isolation. MCP tool annotations map onto policy [54]. The ADR acknowledges extra per-call latency and a manually maintained allowlist [54].
- **CodeAct (ADR-0038, proposed April 2026).** CodeAct is added as an optional context provider, with Hyperlight as the first sandbox. A single *bundled approval* before `execute_code` covers all tool calls inside the snippet, and interpreter state is clean on every call [55].

```
  MAF request path (Python), with ADR features annotated
  user msg -> Agent.run
               |-- agent middleware (ADR-0007)        [FIDES labels, ADR-0024]
               |-- context providers (CodeAct ADR-0038, memory, RAG)
               |-- chat client.get_response
               |      |-- compaction before every call (ADR-0019)
               |      |-- tool loop -> function middleware -> tools / approvals
               |      '-- background mode + continuation token (ADR-0009)
               '-- history provider (pre-write compaction) -> thread store
```

---

## 11. LlamaIndex agents and Workflows

LlamaIndex started as a RAG framework. Its repository now describes it as "the document processing platform for AI," centered on LlamaParse, while the open-source framework (300+ integration packages) remains the toolkit for agentic apps [58]. Its orchestration layer is **Workflows**, now packaged as LlamaAgents. It is event-driven and async-first: steps are async functions that consume and emit typed events (from `StartEvent` to `StopEvent`), with pluggable durability and optional coordination backends [59]. `llama-agents-server` exposes a workflow as a REST API with streaming, persistence and human-in-the-loop. `llamactl` deploys to LlamaParse, AWS Bedrock AgentCore, or custom infrastructure [59].

The documentation describes three escalating multi-agent patterns [60]:

1. **AgentWorkflow.** This is a "swarm" in which agents hand off to one another using `can_handoff_to`, starting from a `root_agent`, with shared `initial_state`.
2. **Orchestrator.** Specialists are wrapped as tools of a top-level agent, which always regains control.
3. **Custom planner.** The LLM emits an XML plan and application code dispatches each step.

The advice is to start with AgentWorkflow and move up only when needed [60]. This matches the handoff and agent-as-tool split in the OpenAI SDK almost exactly.

---

## 12. DSPy: programs instead of prompts

### 12.1 Problem

DSPy ("Declarative Self-improving Python") is "the framework for programming—rather than prompting—language models" [61]. It treats hand-written prompts as brittle hyperparameters. Developers write modular programs, and an optimizer compiles prompts (and optionally weights) against a metric. The founding paper is "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (October 2023), and the line of work continued through 2024 (MIPRO, BetterTogether) and 2025 (GEPA, "Reflective Prompt Evolution Can Outperform Reinforcement Learning") [61][66].

### 12.2 Architecture

- **Signatures** are declarative input/output specs, such as `"context: list[str], question: str -> answer: str"` or a `dspy.Signature` class. Field names carry meaning. Adapters turn signatures into prompts and parse the outputs, raising `AdapterParseError` when a required output field is missing [62].
- **Modules** are modeled on PyTorch's `nn.Module`. Built-in modules include `Predict`, `ChainOfThought` (which adds a `reasoning` field), `ProgramOfThought`, `ReAct`, `MultiChainComparison`, and `RLM`, a Recursive Language Model that explores large contexts in a sandboxed REPL. They compose in `forward()` [63].
- **Optimizers** take a program, a metric, and as few as 5 to 10 (possibly unlabeled) training inputs. The families are [64]:
  - few-shot: `BootstrapFewShot`, `KNNFewShot`
  - instruction: `COPRO`, `MIPROv2`, `SIMBA`, `GEPA`
  - finetuning: `BootstrapFinetune`
  - program transformations: `Ensemble`
  - meta-optimizers: `BetterTogether`

  A typical run is said to cost about **$2 and about 10 minutes**, ranging from cents to tens of dollars [64].

`dspy.ReAct` shows how DSPy expresses an agent as a signature [65]. The module extends the user's signature with a `trajectory` input and three outputs: `next_thought`, `next_tool_name` (typed as a `Literal` over the tool names) and `next_tool_args`. A `finish` tool is always added. The loop runs up to `max_iters=20`. Tool errors become observations. On `ContextWindowExceededError`, the oldest thought/tool/args/observation group is dropped, with up to three attempts. A final `ChainOfThought` extract step produces the declared outputs even if the iteration limit was hit [65]. Because each part is a module with a signature, the whole agent's instructions and demonstrations can be optimized.

### 12.3 Lessons

DSPy sits on a different axis from the others. It barely deals with deployment, durability or multi-agent routing, but it is the only framework here that treats prompt quality as something to optimize *automatically*. It can therefore be layered *inside* nodes of a graph framework.

---

## 13. Design comparison

### 13.1 Feature matrix

| Framework | Primary abstraction | Control flow | State and durability | Action space | Policy layer | Langs |
|---|---|---|---|---|---|---|
| LangGraph [2-8] | Graph of nodes over channels (Pregel) | Developer graph; `Send`/`Command` dynamic | Checkpoint per superstep; pending writes; sync/async/exit | JSON tools | Middleware (v1 agents); interrupts | Py, JS |
| OpenAI Agents SDK [9-17] | Agent + Runner | Model-driven handoffs; agents-as-tools; code orchestration | Sessions (SQLite, Redis, SQLAlchemy, Mongo, Dapr, OpenAI); RunState resume | JSON, hosted tools, MCP, sandbox | Input/output/tool guardrails; tripwires; HITL | Py, TS |
| Claude Agent SDK [19-22] | Harness (Claude Code binary) | Model-driven loop; subagents | On-disk sessions; resume/fork; session_store | Computer: bash, files, web, MCP, skills | Hooks; allow/deny rules; 6 permission modes | Py, TS |
| Google ADK [24-31] | Agent tree then Workflow graph (2.0) | Graph, dynamic, collaborative modes | Event-sourced sessions; commit-on-yield | JSON tools, AgentTool, code executor | Callbacks; tool confirmation; RetryConfig | Py, TS, Go, Java, Kotlin |
| Pydantic AI [32-35] | Typed Agent[Deps, Output] | Code; delegation; pydantic-graph | External durable engines (Temporal, DBOS, ...) | JSON tools, MCP | Typed validation; tool approval; UsageLimits | Py |
| smolagents [36-39] | MultiStepAgent | Model-driven; managed agents | In-memory step log | Python code (CodeAgent) or JSON | AST interpreter limits; remote sandboxes | Py |
| Mastra [41-44] | Agents + typed step workflows | `.then/.branch/.parallel` graph | Snapshot on suspend; restart active runs | JSON tools, MCP | Tripwire status; suspend/approval | TS |
| Strands [45-48] | Model-driven agent loop + harness | Model-driven; multi-agent patterns | Session managers; conversation managers | JSON tools, MCP | Hooks; steering; guardrails | Py, TS |
| MS Agent Framework [49-55] | Agents + Pregel workflows | Graph (executors, edges); handoff/group | Checkpoint per superstep; continuation tokens | JSON, MCP, CodeAct (Hyperlight) | Decorator middleware; FIDES IFC (prop.) | Py, .NET |
| LlamaIndex [58-60] | Event-driven Workflow steps | Events; AgentWorkflow handoffs | Pluggable workflow persistence | JSON tools; document tools | HITL via server | Py, TS |
| DSPy [61-65] | Module with Signature | Python `forward()` | None built-in | ReAct tools; PoT/RLM code | Metric-driven optimization | Py |

### 13.2 Orchestration semantics side by side

| Concern | LangGraph | MAF workflows | ADK 2.0 | Mastra | OpenAI SDK |
|---|---|---|---|---|---|
| Execution model | BSP supersteps | BSP supersteps | Event loop, graph nodes | Step graph engine | Turn loop |
| Checkpoint granularity | Every superstep (tunable) | Every superstep | Every committed event | On suspend | Session items per run |
| Partial-failure recovery | Pending writes skip done siblings | Superstep state discarded | Uncommitted deltas lost | Restart from last active step | Resume from RunState |
| Fan-out primitive | `Send` | Multi-target edges | Route lists, JoinNode (Go) | `.parallel()` | `asyncio.gather` in code |
| HITL primitive | `interrupt()` + `Command(resume)` | `request_info` events | `RequestedInput`, tool confirmation | `suspend()` + `resumeSchema` | Approval interruptions |
| Loop guard | Recursion/step limit | 100 supersteps default | RetryConfig, limits | Status union | `max_turns` |

Notes: the ADK 2.0 HITL entry draws on the 2.0 event fields (`RequestedInput`) and the README's tool confirmation feature [24][28].

### 13.3 Approximate adoption signals (GitHub stars, October 2026)

These indicate visibility, not quality: LlamaIndex about 52.5k [58], LangGraph about 43.2k [2], DSPy about 38.6k [61], smolagents about 29.8k [36], Mastra about 28.8k [41], Google ADK (Python) about 21.8k [24], Pydantic AI about 20.6k [32], Microsoft Agent Framework about 14.1k [49], Strands sdk-python about 8.8k [46]. Repositories differ in age and in how they are split across languages, so the figures are not directly comparable.

### 13.4 Convergent evolution

Several patterns appeared independently in multiple frameworks between 2025 and 2026:

1. **BSP graph runtimes with checkpoints.** LangGraph, MAF and ADK 2.0 all landed here, and ADK and MAF both moved *from* simpler combinators or chat patterns *to* graphs [3][28][50].
2. **Handoff versus agent-as-tool as the basic multi-agent choice.** OpenAI, LlamaIndex, ADK (chat versus task/single-turn modes) and Pydantic AI (delegation versus output-function handoff) all draw this line [16][30][34][60].
3. **Compaction as a first-class runtime concern.** It appears as Claude's auto-compaction, OpenAI's compaction session, MAF's ADR-0019, Mastra's observational memory, LangChain's summarization middleware and DSPy's trajectory truncation [8][15][21][44][53][65].
4. **Code as action.** The idea moved from research (CodeAct) to smolagents, then to MAF's CodeAct provider and DSPy's RLM, with sandboxing as the main cost [40][36][55][63].
5. **Hooks and middleware replacing subclassing.** LangChain v1 middleware, MAF decorators, Claude hooks, Strands `HookProvider` and ADK 2.0 callbacks all follow this pattern. ADK 2.0 explicitly ignores overridden execution methods [7][22][28][48][51].
6. **Harness offerings above the SDK.** Strands `create_harness`, Pydantic AI Harness and "deep agents," LangChain Deep Agents, Claude Agent SDK and Managed Agents, and OpenAI sandbox agents each pre-assemble planning, files, subagents and compaction [2][17][20][34][45].

---

## 14. Failure modes across frameworks

- **Hidden prompt state.** Framework-injected system prompts (handoff prefixes, sandbox base instructions, ReAct scaffolds) are invisible unless traced. Anthropic's warning about wrong assumptions "under the hood" still applies [1][12][17][65].
- **Side effects before policy decisions.** Parallel input guardrails let the agent spend tokens and run tools before a tripwire fires [13]. Budget caps overshoot by at least one response [21].
- **Unsafe defaults in tracing and persistence.** OpenAI traces sensitive data by default [14]. LangGraph's serializer accepts arbitrary types unless strict mode is on [5]. Local "sandboxes" are not security boundaries [17][38].
- **Substrate migrations.** ADK 2.0 silently ignores old `_run_async_impl` overrides, and a broad `except BaseException` breaks human-in-the-loop [28]. MAF restore fails if the graph signature changes [50].
- **Unbounded context in tool loops.** Before ADR-0019, MAF could not compact during a run [53]. Removing a tool call without its result breaks provider APIs [53].
- **Lost delegate accounting.** Under durable engines, sub-agent usage may not propagate to the parent [34]. Claude's `usage` field covers only the main loop, so `model_usage` is needed for the whole tree [21].

---

## 15. Key takeaways for system designers

1. **Decide who owns control flow before choosing a framework.** If the process is known, use a graph or code (LangGraph, MAF workflows, ADK Workflow, Mastra, pydantic-graph). If it is open-ended, use a model-driven harness (Claude Agent SDK, Strands, OpenAI SDK). Most production systems combine the two: a deterministic outer graph with agentic nodes.
2. **Treat durability as a spectrum with explicit costs.** Choose between per-step checkpoints (with a sync/async/exit dial), snapshot-on-suspend, event-commit semantics, and external durable-execution engines based on how expensive a re-run is and how idempotent your tools are. Pending-writes handling (skipping completed siblings) matters whenever tools have side effects.
3. **Draw the handoff versus agent-as-tool line deliberately.** Handoffs keep prompts focused but give up central control and shared guardrails. Agent-as-tool keeps one owner of the answer and isolates sub-contexts. Filter what history crosses the boundary [12][16][30].
4. **Put policy in deterministic code that sits outside the context window.** Hooks, middleware, permission modes and guardrails should enforce what prompts only request. Know whether each check runs before or in parallel with execution [13][21][22].
5. **Make compaction a runtime responsibility.** It must run *within* a tool loop, preserve tool-call/result pairs, keep stable prefixes cacheable, and let important instructions survive (for example, re-injected project files) [21][44][53].
6. **If you use code as action, budget for a real sandbox.** AST interpreters and local executors reduce accidents, not attacks. Decide whether the sandbox holds only snippets or the whole agent, and how approvals cover tool calls made inside code [38][55].
7. **Separate the runtime from the recipe.** Frameworks that kept a stable low-level runtime (LangGraph's Pregel, MAF's runner) were able to replace their high-level agent APIs. Frameworks that changed the substrate under user subclasses (ADK 2.0) paid for it in migration cost.
8. **Instrument everything with OpenTelemetry and keep redaction in the exporter.** Multi-agent token costs (about 15x chat in Anthropic's data) make per-agent attribution necessary [14][23][34].
9. **Consider optimizing prompts rather than only hand-tuning them.** DSPy-style compilation (about $2 and 10 minutes for a typical run) can sit inside any framework's nodes [64].
10. **Keep the escape hatch.** Every framework here can be dropped below to the raw loop. Anthropic's 2024 advice to start with direct API calls, and to adopt a framework only once its abstractions are understood, remains the safest default [1].

---

## References

1. "Building effective agents." Anthropic (E. Schluntz, B. Zhang), 2024. https://www.anthropic.com/research/building-effective-agents
2. LangGraph repository README. LangChain Inc., 2026. https://github.com/langchain-ai/langgraph
3. LangGraph `pregel/main.py` (Pregel class docstring). LangChain Inc., 2026. https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py
4. LangGraph `types.py` (Durability, RetryPolicy, CachePolicy, Interrupt, Command, Send, StreamMode). LangChain Inc., 2026. https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py
5. langgraph-checkpoint README. LangChain Inc., 2026. https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint/README.md
6. LangGraph releases. LangChain Inc., 2026. https://github.com/langchain-ai/langgraph/releases
7. LangChain v1 agent middleware `types.py`. LangChain Inc., 2026. https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/middleware/types.py
8. LangChain v1 built-in middleware directory. LangChain Inc., 2026. https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/agents/middleware
9. OpenAI Agents SDK (Python) README. OpenAI, 2026. https://github.com/openai/openai-agents-python
10. "Running agents." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/running_agents.md
11. "Agents." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/agents.md
12. "Handoffs." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md
13. "Guardrails." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md
14. "Tracing." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md
15. "Sessions." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/sessions/index.md
16. "Orchestrating multiple agents." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md
17. "Sandbox agents." OpenAI Agents SDK docs, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/sandbox_agents.md
18. Swarm README. OpenAI, 2024-2025. https://github.com/openai/swarm
19. "Building agents with the Claude Agent SDK." Anthropic (T. Shihipar et al.), 2025. https://claude.com/blog/building-agents-with-the-claude-agent-sdk
20. "Agent SDK overview." Anthropic Claude Code docs, 2026. https://code.claude.com/docs/en/agent-sdk/overview
21. "How the agent loop works." Anthropic Claude Code docs, 2026. https://code.claude.com/docs/en/agent-sdk/agent-loop
22. Claude Agent SDK for Python README. Anthropic, 2026. https://github.com/anthropics/claude-agent-sdk-python
23. "How we built our multi-agent research system." Anthropic (J. Hadfield et al.), 2025. https://www.anthropic.com/engineering/multi-agent-research-system
24. Agent Development Kit (adk-python) README. Google, 2026. https://github.com/google/adk-python
25. "Event loop." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/runtime/event-loop.md
26. "LLM agents." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/agents/llm-agents.md
27. "Workflow agents." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/agents/workflow-agents/index.md
28. "ADK 2.0." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/2.0/index.md
29. "Graph-based workflows." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/graphs/index.md
30. "Collaborative workflows." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/workflows/collaboration.md
31. "Agent routing." ADK docs, Google, 2026. https://github.com/google/adk-docs/blob/main/docs/agents/routing.md
32. Pydantic AI README. Pydantic, 2026. https://github.com/pydantic/pydantic-ai
33. "Durable execution overview." Pydantic AI docs, 2026. https://github.com/pydantic/pydantic-ai/blob/main/docs/durable_execution/overview.md
34. "Multi-agent applications." Pydantic AI docs, 2026. https://github.com/pydantic/pydantic-ai/blob/main/docs/multi-agent-applications.md
35. "Graphs." Pydantic AI docs, 2026. https://github.com/pydantic/pydantic-ai/blob/main/docs/graph.md
36. smolagents README. Hugging Face, 2026. https://github.com/huggingface/smolagents
37. "Introduction to agents." smolagents docs, Hugging Face, 2026. https://github.com/huggingface/smolagents/blob/main/docs/source/en/conceptual_guides/intro_agents.md
38. "Secure code execution." smolagents docs, Hugging Face, 2026. https://github.com/huggingface/smolagents/blob/main/docs/source/en/tutorials/secure_code_execution.md
39. smolagents `agents.py`. Hugging Face, 2026. https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py
40. "Executable Code Actions Elicit Better LLM Agents" (CodeAct). X. Wang et al., ICML 2024. https://github.com/xingyaoww/code-act
41. Mastra README. Mastra, 2026. https://github.com/mastra-ai/mastra
42. "Workflows overview." Mastra docs, 2026. https://github.com/mastra-ai/mastra/blob/main/docs/src/content/en/docs/workflows/overview.mdx
43. "Snapshots." Mastra docs, 2026. https://github.com/mastra-ai/mastra/blob/main/docs/src/content/en/docs/workflows/snapshots.mdx
44. "Observational memory." Mastra docs, 2026. https://github.com/mastra-ai/mastra/blob/main/docs/src/content/en/docs/memory/observational-memory.mdx
45. Strands Agents harness-sdk README. Strands Agents / AWS, 2026. https://github.com/strands-agents/harness-sdk
46. Strands Agents sdk-python README. Strands Agents / AWS, 2025-2026. https://github.com/strands-agents/sdk-python
47. Strands "TENETS.md." Strands Agents, 2026. https://github.com/strands-agents/harness-sdk/blob/main/team/TENETS.md
48. Strands "DECISIONS.md." Strands Agents, 2026. https://github.com/strands-agents/harness-sdk/blob/main/team/DECISIONS.md
49. Microsoft Agent Framework README. Microsoft, 2026. https://github.com/microsoft/agent-framework
50. Agent Framework workflow runner `_runner.py`. Microsoft, 2026. https://github.com/microsoft/agent-framework/blob/main/python/packages/core/agent_framework/_workflows/_runner.py
51. ADR-0007 "Agent filtering middleware." Microsoft, 2025. https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0007-agent-filtering-middleware.md
52. ADR-0009 "Support long-running operations." Microsoft, 2025. https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0009-support-long-running-operations.md
53. ADR-0019 "Python context compaction strategy." Microsoft, 2026. https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0019-python-context-compaction-strategy.md
54. ADR-0024 "Prompt injection defense (FIDES)." Microsoft, 2026. https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0024-prompt-injection-defense.md
55. ADR-0038 "CodeAct integration." Microsoft, 2026. https://github.com/microsoft/agent-framework/blob/main/docs/decisions/0038-codeact-integration.md
56. Semantic Kernel README. Microsoft, 2026. https://github.com/microsoft/semantic-kernel
57. AutoGen README. Microsoft, 2026. https://github.com/microsoft/autogen
58. LlamaIndex README. LlamaIndex, 2026. https://github.com/run-llama/llama_index
59. LlamaAgents / workflows-py README. LlamaIndex, 2026. https://github.com/run-llama/workflows-py
60. "Multi-agent patterns in LlamaIndex." LlamaIndex docs, 2026. https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/understanding/agent/multi_agent.md
61. DSPy README. Stanford NLP, 2026. https://github.com/stanfordnlp/dspy
62. "Signatures." DSPy docs, 2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/programming/signatures.md
63. "Modules." DSPy docs, 2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/programming/modules.md
64. "Optimizers." DSPy docs, 2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
65. DSPy `predict/react.py`. Stanford NLP, 2026. https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/react.py
66. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." O. Khattab et al., 2023. https://arxiv.org/abs/2310.03714
