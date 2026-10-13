# 01. Agent Foundations and Taxonomy: Workflows, Agents, ReAct, and Levels of Autonomy

## Overview

The word "agent" is used for almost anything with an LLM in it, from a single classification call to a fleet of cooperating models running for hours. This chapter sets out the shared vocabulary for the rest of the Atlas. It draws on the documents that practitioners treat as reference points:

- **Anthropic's "Building Effective Agents"** (December 2024) [1]. It separates *workflows* (LLMs and tools orchestrated through predefined code paths) from *agents* (LLMs that dynamically direct their own processes and tool usage). It also names the *augmented LLM* building block and five workflow patterns: prompt chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer.
- **ReAct** (Yao et al., 2022; ICLR 2023) [2][3]. This is the research ancestor of nearly every tool-using agent loop: interleave free-text reasoning ("thoughts") with actions against an environment, and feed the observations back in.
- **OpenAI's "A Practical Guide to Building Agents"** (April 2025) [6][7]. It frames agents as *model + tools + instructions*, gives criteria for when an agent is worth building, and distinguishes single-agent runs from multi-agent *manager* and *decentralized (handoff)* patterns, all wrapped in layered guardrails.
- **Levels-of-autonomy frameworks.** These include DeepMind's "Levels of AGI" autonomy axis [25], the University of Washington user-role levels L1-L5 [24], Hugging Face's "agency spectrum" [20], and Anthropic's empirical measurements of how much autonomy users actually grant agents in production [17].

The chapter also follows how these foundations changed through 2025-2026:

- *context engineering* [9]
- tool and agent-computer-interface (ACI) design [11][12]
- long-running harnesses [13][14]
- the brain/hands decoupling in hosted agent runtimes [15]
- classifier-gated autonomy [16]
- agent evaluation methodology [18]
- cross-vendor orchestration catalogs from Microsoft [26] and the OpenAI Agents SDK [8]

The thread running through all of them is simple. Use the least autonomy and the least structure that reliably solves the task, and keep re-checking that choice as models improve.

---

## 1. Workflows vs. agents: the core distinction

### 1.1 Definitions

Anthropic groups every LLM-plus-tools system under the term *agentic systems*. It then splits them by who owns control flow [1]:

- **Workflows:** "systems where LLMs and tools are orchestrated through predefined code paths."
- **Agents:** "systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."

OpenAI's guide draws the line in a similar place. An agent is a system that independently accomplishes tasks on a user's behalf. A chatbot, a single-turn LLM call, or a sentiment classifier is *not* an agent, because the LLM does not control workflow execution [6]. OpenAI names two defining traits [6]:

1. **LLM-driven control:** The model runs the workflow, recognizes when it is complete, corrects its own mistakes, and halts and returns control to the user on failure.
2. **Tool use within guardrails:** The agent picks tools dynamically based on workflow state and operates within clearly defined limits.

Hugging Face's smolagents documentation [20] goes further and argues that "agent" is not binary. Agency is a *spectrum*, defined by how much the LLM's output decides the program's control flow (Section 5.3).

Microsoft's Azure Architecture Center (updated February 2026) offers a third, operational framing as three complexity tiers [26]:

1. A **direct model call** for single-step tasks such as classification, summarization, or translation.
2. A **single agent with tools**, often the enterprise default, with iteration limits to prevent infinite tool loops.
3. **Multi-agent orchestration**, for cross-domain problems, distinct security boundaries, or parallel specialization. This tier adds overhead, latency, and new failure modes.

Its guiding rule: "Use the lowest level of complexity that reliably meets your requirements" [26].

### 1.2 When to use which

The sources agree to a striking degree on when to use which design.

| Question | Anthropic [1] | OpenAI [6] | Microsoft [26] | Hugging Face [20] |
|---|---|---|---|---|
| Default starting point | Single optimized LLM call with retrieval and in-context examples | Deterministic solution unless the use case clearly qualifies | Direct model call, then single agent | Hard-coded workflow if it covers all queries |
| Signal to add agency | Open-ended problems; number of steps unpredictable | Complex judgment; brittle, hard-to-maintain rulesets; heavy unstructured data | Varied queries within a domain that need dynamic tool selection | Workflow cannot be predetermined; fixed paths fail too often |
| Cost of agency | Latency, cost, compounding errors | Overhead of evals and maintenance; multi-agent overhead | Coordination overhead, latency, new failure modes | Reliability risk from unpredictable LLM decisions |
| Multi-agent trigger | Subtasks unpredictable (orchestrator-workers) | Complex conditional logic; tool overload | Cross-domain, security boundaries, parallel specialization | Agent triggers another agent (top of spectrum) |

OpenAI's three qualifying criteria each come with a canonical example [6]:

- **Complex decision-making**, such as refund approval in customer service.
- **Difficult-to-maintain rules**, such as vendor security reviews.
- **Heavy reliance on unstructured data**, such as processing a home-insurance claim.

The guide contrasts a rules engine, which works "like a checklist," with an LLM agent that behaves "like a seasoned investigator." In payment-fraud analysis, for example, the agent can flag suspicious patterns even when no explicit rule is violated [6].

Anthropic adds the key economic caveat: agentic systems "trade latency and cost for better task performance" [1]. Its later multi-agent work puts numbers on that trade (Section 6.1). Agents use about 4x the tokens of chat interactions, and multi-agent systems about 15x [10].

### 1.3 Frameworks vs. direct API use

Anthropic advises developers to "start by using LLM APIs directly." Frameworks speed up setup but can obscure the underlying prompts and responses, which makes debugging harder and tempts teams to over-engineer [1]. The current version of the page lists the Claude Agent SDK, Strands Agents SDK, Rivet, and Vellum as examples, and adds a note that much of the tooling has changed since December 2024 [1].

OpenAI's guide takes a code-first position. It contrasts *declarative graph* frameworks, which require every branch and loop to be predefined, with its Agents SDK, which expresses workflow logic in ordinary programming constructs [6].

The SDK itself is deliberately small [8]. It has three primitives:

- **Agents:** LLMs equipped with instructions and tools.
- **Agents-as-tools and handoffs:** two ways to delegate.
- **Guardrails:** input and output validation.

It also has two stated design principles: offer enough features to be worth using with few enough primitives to learn quickly, and work well out of the box while staying customizable.

---

## 2. The augmented LLM: the building block

```
                 +-------------------------------------------+
                 |              Augmented LLM                |
   input  ---->  |  +-------------+                          |  ----> output
                 |  |     LLM     |<--> Retrieval (queries)  |
                 |  |             |<--> Tools (calls/results)|
                 |  |             |<--> Memory (read/write)  |
                 |  +-------------+                          |
                 +-------------------------------------------+
```

**Problem.** A bare LLM is limited to its parametric knowledge and cannot act on the world.

**Architecture.** An LLM augmented with **retrieval, tools, and memory**. The model actively generates its own search queries, selects tools, and decides what to retain [1]. Anthropic stresses two implementation priorities [1]:

- Tailor these capabilities to the specific use case.
- Give them an easy, well-documented interface.

The Model Context Protocol (MCP) is offered as one standardized way to attach third-party tools [1].

OpenAI's decomposition is the same in substance: **model, tools, and instructions** [6]. It sorts tools into three types [6]:

| Tool type | Purpose | Examples [6] |
|---|---|---|
| Data | Retrieve context and information | Query databases or CRMs, read PDFs, search the web |
| Action | Change systems or send information | Send email or SMS, update a CRM record, hand a ticket to a human |
| Orchestration | Agents exposed as tools for other agents | Refund agent, research agent, writing agent |

For legacy systems without APIs, the guide notes that computer-use models can drive application UIs directly [6].

**Model selection.** OpenAI recommends a three-step process [6]:

1. Build the prototype with the most capable model to set a performance baseline, measured with evals.
2. Focus on meeting the accuracy target.
3. Optimize cost and latency by swapping in smaller models wherever they still meet that target.

Simple retrieval or intent classification can run on small models. Harder decisions, such as approving a refund, may need a more capable one [6].

**Instructions.** OpenAI's practices for writing instructions [6]:

- Derive routines from existing operating procedures, support scripts, and policy documents.
- Break dense resources into smaller, clearer steps.
- Tie every step to a concrete action or output.
- Anticipate edge cases with conditional branches.

The guide also recommends **prompt templates**: one flexible base prompt with policy variables, instead of many hand-maintained prompts [6].

**Evolution: context engineering.** By September 2025, Anthropic had reframed the augmented-LLM problem as *context engineering*: curating "the smallest possible set of high-signal tokens" across system prompt, tools, external data, and message history at every inference step [9].

The motivation is **context rot**. As the window fills, recall accuracy degrades in all models, at different rates. Every token spends part of a finite "attention budget," and the n² pairwise relationships of transformer attention contribute to this [9].

The prescribed techniques are [9]:

- **Compaction:** summarize near the limit. Claude Code keeps the summary plus the five most recently accessed files.
- **Structured note-taking:** persistent memory outside the window, such as a NOTES.md or a to-do list.
- **Sub-agent architectures:** sub-agents explore with tens of thousands of tokens and return condensed 1,000-2,000-token summaries.
- **Just-in-time retrieval:** keep lightweight identifiers such as paths, queries, and links, and load the data through tools at runtime. Hybrid designs pre-load some context, as Claude Code does with CLAUDE.md, and search the rest with glob and grep.

---

## 3. The five workflow patterns

Anthropic's five patterns [1] have become the field's shared vocabulary. Microsoft's catalog [26] and the OpenAI Agents SDK's code-orchestration patterns [8] map almost one-to-one onto them (see the table in Section 3.6).

### 3.1 Prompt chaining

```
 In --> [LLM 1] --> out1 --> (Gate: pass?) --yes--> [LLM 2] --> [LLM 3] --> Out
                                   |
                                   +--no--> Exit / fix
```

- **Problem:** A task that is too large or too error-prone for one call, but splits cleanly into fixed subtasks.
- **Architecture:** A sequence of LLM calls, each consuming the previous output. Programmatic **gates** check intermediate results to keep the process on track [1].
- **Trade-off:** Higher latency in exchange for accuracy, since each call becomes an easier task [1].
- **Examples:** Write marketing copy, then translate it. Write an outline, check it against criteria, then write the document [1].
- **Equivalents:** Microsoft's *sequential* pattern ("pipeline," "linear delegation") [26]. The OpenAI SDK's *chaining*, for example research, outline, draft, critique, revise [8].
- **Failure modes:** Errors in early stages propagate downstream, and there is no parallelism [26]. Microsoft advises against the pattern when stages are embarrassingly parallel, when backtracking is needed, or when routing depends on intermediate results [26].

### 3.2 Routing

```
                    +--> [Specialized prompt / model A]
 In --> [Router] ---+--> [Specialized prompt / model B]  --> Out
                    +--> [Specialized prompt / model C]
```

- **Problem:** Inputs fall into distinct categories, and optimizing one prompt for all of them hurts the others.
- **Architecture:** Classify the input, using an LLM or a traditional classifier, and dispatch it to a specialized downstream task [1].
- **Examples:** Separate general, refund, and technical-support flows. Send easy questions to small, cheap models and hard ones to more capable models [1].
- **Equivalents:** OpenAI SDK "structured outputs for routing" [8]. Microsoft's *handoff* pattern is the dynamic, agentic version [26].
- **Failure modes:** Misrouting. Microsoft notes that when the right agent is identifiable from the input, a simple classifier or deterministic rule beats agent-driven handoff. Agent-driven routing risks infinite handoff loops and agents bouncing work back and forth [26].

### 3.3 Parallelization (sectioning and voting)

```
            Sectioning                         Voting
        +--> [LLM: aspect 1] --+          +--> [LLM run 1] --+
 In ----+--> [LLM: aspect 2] --+--> Agg   In --+--> [LLM run 2] --+--> Vote/threshold
        +--> [LLM: aspect 3] --+          +--> [LLM run 3] --+
```

- **Problem:** Independent subtasks could run concurrently, or a single answer needs more confidence.
- **Architecture:** Two variants, with outputs aggregated programmatically [1]:
  - **Sectioning:** independent subtasks run in parallel.
  - **Voting:** the same task runs several times to get diverse outputs.
- **Key insight:** For complex tasks with several considerations, "LLMs generally perform better when each consideration is handled by a separate LLM call" [1].
- **Sectioning examples:** One model instance answers the user while another screens the input for inappropriate content. Separate eval calls grade different aspects of performance [1].
- **Voting examples:** Several prompts review code for vulnerabilities. Several judges assess content, with vote thresholds tuned to balance false positives and negatives [1].
- **Equivalents:** Microsoft's *concurrent* pattern ("fan-out/fan-in," "scatter-gather," "map-reduce") [26]. The OpenAI SDK runs agents in parallel with `asyncio.gather` [8].
- **Failure modes:** Conflicting results need a resolution strategy such as voting, weighting, or LLM synthesis. Shared mutable state produces inconsistency. Model quotas can cap concurrency [26].

### 3.4 Orchestrator-workers

```
                 +--------------------+
   In ---------> |  Orchestrator LLM  | -- plans subtasks dynamically
                 +--------------------+
                  |        |        |
                  v        v        v
              [Worker] [Worker] [Worker]   (count and content decided at runtime)
                  |        |        |
                  +--------+--------+
                           v
                   [Synthesizer (often orchestrator)] --> Out
```

- **Problem:** Complex tasks whose subtasks cannot be predicted in advance.
- **Architecture:** A central LLM breaks the task down, delegates to worker LLMs, and synthesizes their results [1].
- **The key difference from parallelization:** the subtasks are not predefined. The orchestrator decides them from the specific input [1].
- **Examples:** Coding changes across an unknown number of files. Search tasks that gather information from many sources [1].
- **Equivalents:** OpenAI's *manager pattern* (agents-as-tools) [6][8]. Microsoft's *magentic* pattern adds an explicit task ledger [26]. Magentic-One's Orchestrator keeps a Task Ledger of facts and educated guesses and a Progress Ledger, and replans when progress stalls [27].
- **Production example:** Anthropic's Research feature is the canonical large-scale instance (Section 6.1) [10].

### 3.5 Evaluator-optimizer

```
           +---------------------------------------+
           v                                       |
 In --> [Generator LLM] --> candidate --> [Evaluator LLM] --accept--> Out
                                               |
                                               +--feedback (loop)
```

- **Problem:** Output quality improves measurably with articulated critique.
- **Architecture:** One LLM generates and another evaluates and gives feedback, in a loop [1].
- **Fit test:** Use it when (a) human feedback demonstrably improves responses and (b) an LLM can produce that kind of feedback [1].
- **Examples:** Literary translation with nuance. Complex multi-round search where the evaluator decides whether more searching is needed [1].
- **Research lineage:**
  - **Self-Refine** (Madaan et al., 2023) uses one LLM to generate, critique, and refine in a loop until a stopping criterion. Its tasks range from dialogue response to code readability and GSM-8K [23]. The paper reports roughly 20% average absolute improvement over one-step generation (approximate; see [23]).
  - **Reflexion** (Shinn et al., NeurIPS 2023) stores verbal self-reflections in an episodic memory across trials instead of updating weights. It reports 91% pass@1 on HumanEval, against 80% for GPT-4 [22].
- **Equivalents:** Microsoft's *maker-checker loop* [26] and the OpenAI SDK's *evaluator loop* [8].
- **Failure modes:**
  - **Infinite refinement.** Mitigate with an iteration cap and a fallback such as human escalation or a quality warning [26].
  - **Evaluator leniency.** Anthropic's 2026 harness work found the out-of-the-box QA agent "talked itself into" approving real defects and tested superficially [14].
  - **Self-evaluation leniency.** Agents praise their own mediocre work, especially on subjective tasks [14].
  - **No crisp criteria.** Microsoft recommends against the pattern when clear, consistent pass/fail criteria cannot be defined [26].

### 3.6 Cross-vendor pattern mapping

| Anthropic pattern [1] | Microsoft name [26] | OpenAI SDK idiom [8] | Control owner | Typical failure mode |
|---|---|---|---|---|
| Prompt chaining | Sequential | Chaining (output to input) | Code | Error propagation down the chain |
| Routing | Handoff (agentic) / classifier | Structured-output routing; handoffs | Code or LLM | Misrouting; handoff loops |
| Parallelization | Concurrent | `asyncio.gather` parallel agents | Code | Conflicting outputs; quota limits |
| Orchestrator-workers | Magentic | Manager / agents-as-tools | LLM | Duplicated work; slow convergence |
| Evaluator-optimizer | Maker-checker | Evaluator `while` loop | Code loop + LLM judge | Endless refinement; lenient judge |
| Autonomous agent | Single agent with tools | `Runner.run` loop with `max_turns` | LLM | Compounding errors; runaway cost |
| (none) | Group chat | (none) | Chat manager | Conversation loops; keep to about 3 agents |

The patterns compose. Anthropic stresses that they are "common patterns" to shape and combine, not prescriptions, and that complexity should be added "only when it demonstrably improves outcomes" [1]. Microsoft lists "forcing one workflow into a single pattern when different stages suit different patterns" as an antipattern [26].

---

## 4. Autonomous agents and the ReAct loop

### 4.1 Anthropic's autonomous agent

Anthropic describes the lifecycle this way [1]:

1. The agent starts from a human command or interactive discussion.
2. Once the task is clear, it plans and operates independently.
3. It may return to the human for information or judgment at checkpoints or when blocked.

During execution it needs "ground truth" from the environment at every step, such as tool results or code execution, to assess progress. Agents are "typically just LLMs using tools based on environmental feedback in a loop," with stopping conditions such as a maximum iteration count to keep control [1]. They fit open-ended problems where the number of steps cannot be predicted and the path cannot be hard-coded. Using them requires some trust in the model's decisions, and Anthropic warns of "higher costs, and the potential for compounding errors," and recommends extensive sandbox testing and guardrails [1].

```
            +-----------------------------------------------------+
            |                                                     |
 Human ---> |  LLM --(action / tool call)--> Environment / tools   |
  goal      |   ^                                    |            |
            |   +-------(observation / feedback)-----+            |
            |                                                     |
            |  stop when: task done | max iterations | needs human|
            +-----------------------------------------------------+
```

### 4.2 ReAct: reasoning and acting, interleaved

**Problem.** In 2022, reasoning (chain-of-thought) and acting (action-plan generation, as in WebGPT or SayCan) were studied separately [3].

- Reason-only models hallucinate and propagate errors, because they rely on internal knowledge.
- Act-only models can take the same actions and see the same observations, yet fail to synthesize a final answer [3].

**Method.** ReAct (Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao; Princeton and Google) prompts an LLM to produce *thoughts* and *actions* in an interleaved sequence [2][3]. Actions produce observations from an external environment. Thoughts do not touch the environment; they update the model's internal context. In the paper's words, reasoning traces help the model "induce, track, and update action plans as well as handle exceptions," while actions let it "interface with external sources" [3].

A ReAct prompt is a set of few-shot human-written trajectories made of reasoning, actions, and observations [3]. For knowledge tasks, thoughts and actions alternate densely. For decision tasks such as ALFWorld, trajectories use *sparse* reasoning and let the model decide when to think and when to act [3].

```
 Question: <task>
 Thought 1: I need to find X, then compare with Y.
 Action 1:  Search[X]
 Observation 1: <Wikipedia API result>
 Thought 2: X is ...; now I need Y.
 Action 2:  Lookup[Y]
 Observation 2: <result>
 Thought 3: So the answer is Z.
 Action 3:  Finish[Z]
```

**Results.**

- On HotpotQA and FEVER, using only a simple Wikipedia API, ReAct overcomes the hallucination and error-propagation problems of chain-of-thought and produces more interpretable trajectories [3]. It beats vanilla action-only generation and is competitive with chain-of-thought [5].
- On ALFWorld and WebShop, it beats imitation-learning and reinforcement-learning baselines by **34% and 10% absolute success rate** respectively, prompted with only one or two in-context examples [3].
- The official repository reports these scores [4]:

| Task | Metric | PaLM-540B (paper) | GPT-3 text-davinci-002 (repo rerun) |
|---|---|---|---|
| HotpotQA | Exact match | 29.4 | 30.4 |
| FEVER | Exact match | 62.2 | 54.0 |
| ALFWorld | Success rate | 70.9 | 78.4 |
| WebShop | Success rate | 40.0 | 35.8 |

The HotpotQA and FEVER reruns use 500 random validation examples [4].

**Lessons that carried forward.**

1. **Interpretability as a control surface.** The authors show that a human can correct an ALFWorld failure by editing a couple of thoughts. They frame this as a new route to human alignment, and it anticipates today's practice of exposing planning steps for transparency [1][3].
2. **Fine-tuning on trajectories.** Initial results suggest ReAct is the best fine-tuning format across model sizes, and that *ReAct-fine-tuned smaller models outperform prompted larger models* [3]. This foreshadows today's agentic post-training.
3. **Native tool calling absorbed the format.** Modern APIs make the action a structured tool call rather than a parsed "Action:" string. The loop is otherwise the same as the one in the OpenAI Agents SDK runner and the Claude Agent SDK (Sections 4.3-4.4).

**Failure modes.** ReAct inherits every weakness of open-loop prompting: limited context and limited learning support [3]. Bad search results can derail reasoning, and grounding is only as good as the tools. Later work added memory and reflection on top: Reflexion runs ReAct-style agents and adds self-reflection across trials [22].

### 4.3 The OpenAI run loop and exit conditions

OpenAI's guide defines a *run* as a loop that continues until an **exit condition** is reached [6]:

- a final-output tool is invoked
- the model returns a response with no tool calls
- a structured output is produced
- an error occurs
- a maximum number of turns is reached

The Agents SDK runner implements it [8]:

```
 loop:
   resp = LLM(current_agent, input)
   if resp is final_output (correct type, no tool calls): return result
   elif resp is handoff:   current_agent = target; input = updated; continue
   elif resp has tool calls: run tools; append results to input; continue
   if turns > max_turns: raise MaxTurnsExceeded (or handler returns fallback)
```

Conversation state can be carried four ways [8]:

- client-side `to_input_list()`
- SDK-managed `session` storage
- server-side `conversation_id`
- `previous_response_id` chaining

The SDK warns against mixing client-side and server-side history, which can duplicate context [8].

### 4.4 Anthropic's agent loop in the Claude Agent SDK

Anthropic describes the loop behind Claude Code and the Claude Agent SDK as **gather context, take action, verify work, repeat** [30]:

- **Gather context:** agentic search over the file system (grep, tail), which the post recommends before semantic or vector search because it is more accurate and more transparent; sub-agents with isolated context windows; and compaction.
- **Take action:** tools, bash, code generation, and MCP integrations.
- **Verify work:** rules-based feedback such as linting, visual feedback from screenshots, and LLM-as-judge. The post notes LLM-as-judge is less robust and adds latency [30].

The SWE-bench work shows the minimal-scaffold philosophy in its purest form [12]. The scaffold is a short prompt suggesting a workflow (explore, reproduce, edit, rerun, check edge cases) plus two tools: bash, and a `str_replace_editor` that requires absolute paths and exact, unique string matches. The philosophy is to "give as much control as possible to the language model itself."

- **Score:** Claude 3.5 Sonnet (new) reached 49% on SWE-bench Verified, against 45% for the previous state of the art [12]. The same scaffold scored 33% with the older Sonnet and 22% with Claude 3 Opus.
- **Run length:** Some runs succeeded in about 12 steps. Many took hundreds of turns and more than 100k tokens [12].

---

## 5. Tools and the agent-computer interface (ACI)

### 5.1 The ACI principle

Anthropic lists three core principles for agents [1]:

1. **Simplicity.**
2. **Transparency:** explicitly show the agent's planning steps.
3. **A carefully crafted agent-computer interface (ACI)**, built through tool documentation and testing.

On the third, its claim is that ACI deserves as much investment as human-computer interface (HCI) design [1].

Concrete guidance from the appendix [1]:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep formats close to text that occurs naturally on the internet.
- Avoid formatting overhead. Diffs require knowing chunk-header line counts in advance, and code inside JSON requires escaping newlines and quotes.
- Write tool descriptions like docstrings for a junior developer, with examples, edge cases, and boundaries.
- Test many inputs.
- Apply **poka-yoke**: change arguments so that mistakes are hard to make.

The emblematic anecdote comes from SWE-bench. The team spent more time optimizing tools than the overall prompt. After the model made relative-path mistakes once it left the root directory, the team required absolute paths, and "the model used this method flawlessly" [1].

### 5.2 Tool-design findings, 2025-2026

Anthropic's September 2025 tool-writing guide [11] turns ACI into an eval-driven process:

1. Prototype the tools.
2. Generate dozens of realistic, multi-call tasks with verifiable outcomes.
3. Run simple agent loops while tracking accuracy, runtime, call counts, tokens, and errors.
4. Let Claude refactor the tools from transcripts, validating on held-out sets.

The design rules [11]:

- **Consolidate.** A `schedule_event` tool replaces separate list and create calls; don't just wrap API endpoints.
- **Namespace.** For example, `asana_projects_search`.
- **Return high-signal fields.** Prefer names over UUIDs, and expose a `response_format` enum. In one example a concise response used 72 tokens against 206 for the detailed one.
- **Paginate and truncate.** Claude Code caps tool responses at 25,000 tokens by default.
- **Write actionable errors**, not opaque codes.

OpenAI's guide makes a complementary point about *how many* tools one agent can handle [6]. The issue is less the count than how similar and overlapping the tools are:

- Some deployments manage more than 15 well-defined, distinct tools.
- Others struggle with fewer than 10 overlapping ones.
- Splitting into multiple agents is warranted only after better naming and descriptions fail.

The context-engineering post states the same test: "If a human engineer can't definitively say which tool should be used ... an AI agent can't be expected to do better" [9].

### 5.3 Code as the action space

Two lines of work argue that the *form* of actions matters:

- **CodeAct** (Wang et al., ICML 2024) folds all actions into executable Python run in an interpreter. Across 17 LLMs on API-Bank and the authors' M3ToolEval benchmark, it reports up to 20% higher success rate than JSON or text actions. The authors also release CodeActInstruct, about 7k multi-turn interactions [21].
- **smolagents** adopts code agents as its top level of agency. Its argument is that code gives composability, object management, generality, and better coverage in training data [20].

Anthropic's November 2025 "code execution with MCP" applies the same idea to tool overload [19]:

- **Mechanism:** MCP servers are presented as a filesystem of code APIs. The agent reads only the tool files it needs, and intermediate data is filtered inside the sandbox.
- **Result:** In a Google Drive-to-Salesforce example, token usage dropped from about 150,000 to about 2,000 (98.7%).
- **Cost:** A secure sandbox with resource limits and monitoring [19].

---

## 6. Multi-agent extensions of the foundational patterns

### 6.1 Orchestrator-workers at production scale: Anthropic Research

```
 User query
     |
     v
 +-------------------+  saves plan  +---------+
 |  LeadResearcher   |------------->| Memory  |  (survives >200k-token truncation)
 | (extended think)  |<-------------+---------+
 +-------------------+
   | spawn 3-5 subagents in parallel (effort scaled to query)
   v            v            v
 [Subagent] [Subagent] [Subagent]  each: own context, 3+ parallel tool calls,
   |            |            |      interleaved thinking, returns condensed findings
   +------------+------------+
                v
       LeadResearcher synthesizes; loops if more research is needed
                |
                v
         [CitationAgent] --> cited report
```

**Problem.** Open-ended research where the path depends on intermediate findings. A static RAG retrieval cannot follow those leads [10].

**Architecture.** An orchestrator-worker system [10]:

- The lead agent (Claude Opus 4) plans and saves its plan to memory, because context beyond 200,000 tokens gets truncated.
- It spawns parallel subagents (Claude Sonnet 4), each with its own context, and a final CitationAgent attributes the claims.

**Numbers** [10]:

- **Quality:** The multi-agent system beat single-agent Opus 4 by **90.2%** on an internal research eval.
- **Token budget:** Token usage alone explains **80%** of performance variance on BrowseComp. Token usage, tool-call count, and model choice together explain 95%.
- **Cost:** Agents use about 4x the tokens of chat, and multi-agent systems about 15x.
- **Time:** Parallelization cut research time by up to 90% for complex queries.

**Techniques** [10]:

- **Effort-scaling rules in the prompt:** one agent with 3-10 calls for simple fact-finding; 2-4 subagents with 10-15 calls each for comparisons; more than 10 subagents for complex research.
- **Delegation briefs:** each subagent gets an objective, an output format, tool guidance, and boundaries.
- **Search strategy:** start with broad queries, then narrow.
- **Self-improving tools:** a tool-testing agent rewrote tool descriptions, cutting task completion time by 40%.
- **Lightweight handoffs:** subagents write outputs to a filesystem and pass references back.

**Evaluation** [10]:

- Start with about 20 queries.
- Use a single LLM judge scoring factual accuracy, citation accuracy, completeness, source quality, and tool efficiency.
- Keep human testers. They caught a bias toward SEO content farms.

**Failure modes** [10]:

- **Duplicated work** when instructions are vague.
- **Over-spawning** on simple queries.
- **Compounding stateful errors**, handled with checkpoints and resumption rather than restarts.
- **Non-determinism** that makes debugging hard, handled with full tracing.
- **Deploys mid-run**, handled with rainbow deployments.
- **A synchronous bottleneck:** the lead waits for each batch of subagents to finish.

**Where it fits.** Anthropic notes that multi-agent systems excel at breadth-first, parallelizable work. They fit poorly where all agents need the same context or have many dependencies, as in most coding [10].

### 6.2 OpenAI: manager vs. decentralized

```
  Manager pattern (agents as tools)        Decentralized pattern (handoffs)

        User <--> [Manager]                  User <--> [Triage agent]
                 /   |    \                               | handoff (one-way)
           tool /  tool    \ tool                         v
        [Agent A] [Agent B] [Agent C]                [Specialist] <--> User
        (results return to manager)               (takes over the conversation)
```

OpenAI recommends getting the most out of a single agent first [6]. It splits into multiple agents only for two reasons: **complex logic**, meaning prompts full of conditionals that templates can't scale, or **tool overload** (Section 5.2).

- **Manager pattern:** A central LLM calls specialists as tools, keeps control and the user relationship, and synthesizes their outputs. The guide's example translates "hello" into Spanish, French, and Italian, with one agent per language [6].
- **Decentralized pattern:** Peers hand off. A handoff is a one-way transfer, implemented as a tool, that passes the latest conversation state. The receiving agent takes over the user interaction [6].

The SDK's decision rule [8]:

- Use agents-as-tools when one agent should own the final answer, combine specialist outputs, or enforce shared guardrails.
- Use handoffs when the specialist should answer directly with a focused prompt.
- The two combine: a triage agent hands off to a specialist that can still call other agents as tools.

### 6.3 Counterpoint: single-threaded agents

Cognition, the maker of Devin, argued against parallel multi-agent designs in mid-2025. Its reasoning, as reported by secondary write-ups [29]:

- Subagents that see only a summarized task miss upstream decisions. In the Flappy Bird example, two subagents build parts that don't fit together.
- "Actions carry implicit decisions." Parallel writers make conflicting choices about style, edge cases, and code patterns.

Cognition's 2026 follow-up keeps the core point for parallel *writer* swarms. It reports that patterns where several agents contribute intelligence while **writes stay single-threaded** do work [28].

This lines up with Anthropic's caution about coding [10], and with its context-engineering advice that sub-agents should return condensed results to a coordinator [9].

---

## 7. Guardrails and human-in-the-loop as part of the foundation

OpenAI treats guardrails as a layered defense, not a single filter [6]:

| Guardrail [6] | Function |
|---|---|
| Relevance classifier | Flags off-topic queries |
| Safety classifier | Detects jailbreaks and prompt injections |
| PII filter | Screens outputs for personally identifiable information |
| Moderation | Flags harmful content (hate, harassment, violence) |
| Tool safeguards | Rate each tool low/medium/high risk by read vs. write access, reversibility, permissions, and financial impact; gate high-risk calls |
| Rules-based protections | Blocklists, input-length limits, regex filters (e.g., for SQL injection) |
| Output validation | Keeps responses aligned with brand values |

**How to build them.** Start with data privacy and content safety, then add guardrails as real-world edge cases appear [6].

**Human intervention.** The guide names two triggers [6]:

1. **Exceeding failure thresholds**, for example repeatedly failing to understand customer intent.
2. **High-risk actions**, such as canceling orders, authorizing large refunds, or making payments. These stay under human oversight until reliability is proven.

**Implementation in the SDK** [8]:

- **Input guardrails** run only on the first agent, and **output guardrails** only on the agent that produces the final answer.
- **Tool guardrails** wrap function tools and run on every guarded call.
- **Execution mode trade-off:** Input guardrails run *in parallel* with the agent by default, for lower latency. If one trips, tokens and tool side effects may already be spent. *Blocking* mode guarantees the expensive model never starts.
- **Tripwires:** a violation sets `tripwire_triggered`, and the runner raises an exception that carries `run_data` for auditing.

Anthropic's *auto mode* for Claude Code (March 2026) is the most detailed public design for classifier-gated autonomy [16]. The motivation: under manual approval, users approve 93% of permission prompts, which produces approval fatigue.

**Architecture** [16]:

- A server-side **prompt-injection probe** scans tool outputs.
- A two-stage **transcript classifier** (Sonnet 4.6) reviews actions before they execute: a fast cautious filter, then chain-of-thought review only for flagged actions.
- The classifier is **reasoning-blind**. It sees user messages and tool calls but not the assistant's text or tool results, so the agent cannot argue its way past it.
- **Deny-and-continue:** a denial returns to the agent with a reason. Three consecutive or 20 total denials escalate to the human.

**Measured performance** [16]:

| Dataset | Rate |
|---|---|
| Real traffic (n=10,000) | 0.4% false positives (benign actions blocked) |
| Real overeager actions (n=52) | 17% false negatives (dangerous actions allowed) |
| Synthetic exfiltration (n=1,000) | 5.7% false negatives |

The authors say plainly that auto mode is not a drop-in replacement for careful human review on high-stakes infrastructure [16].

---

## 8. Levels of autonomy

### 8.1 Four taxonomies

| Level | DeepMind "Levels of AGI" autonomy [25] | UW user-role levels [24] | Hugging Face agency [20] | Anthropic / Microsoft complexity [1][26] |
|---|---|---|---|---|
| 0 | No AI | (none) | Simple processor: output does not affect flow | Direct model call |
| 1 | AI as a Tool | L1 Operator: user directs, agent acts | Router: output picks a branch | Workflow: chaining, routing |
| 2 | AI as a Consultant | L2 Collaborator: shared planning and execution | Tool call: output picks function and args | Workflow: parallelization, orchestrator-workers |
| 3 | AI as a Collaborator | L3 Consultant: agent leads, consults user | Multi-step agent: output controls looping | Single autonomous agent with tools |
| 4 | AI as an Expert | L4 Approver: user engaged only for risky or pre-set cases | Multi-agent: one agent triggers another | Multi-agent orchestration |
| 5 | AI as an Agent (fully autonomous) | L5 Observer: user only monitors | Code agent: acts in code, can define tools | (beyond published tiers) |

The rows only roughly correspond. Each taxonomy measures a different thing:

- **DeepMind's axis (Morris et al., ICML 2024)** is a *human-AI interaction paradigm*. It is deliberately decoupled from capability: higher capability levels "unlock" higher autonomy but do not decide it, and the right paradigm depends on context, including safety [25].
- **Feng, McDonald, and Zhang (University of Washington, June 2025)** define autonomy as a *design decision separate from capability*, classified by the role the user plays [24]. They propose **autonomy certificates**, third-party documents that cap an agent's autonomy level in single- and multi-agent systems, along with ways to evaluate autonomy [24].
- **Hugging Face's spectrum** is a *control-flow* measure: how much of the program's flow the LLM output decides [20]. It reads like a software engineer's version of Anthropic's workflow/agent split.
- **The vendor tiers** are *architectural*: how many LLM-controlled loops and agents exist [1][26].

When a system is described as "Level N," name the axis.

### 8.2 Autonomy measured in practice

Anthropic's February 2026 study [17] is the first large-scale empirical look at autonomy as it is actually *exercised*. It used privacy-preserving analysis of millions of interactions across Claude Code and the public API.

**Claude Code session length:**

- The median turn lasts about 45 seconds.
- The 99.9th-percentile turn duration nearly doubled, from under 25 minutes to over 45 minutes, between October 2025 and January 2026.
- Internally, success on the hardest tasks doubled from August to December 2025, while human interventions per session fell from 5.4 to 3.3.

**User trust grows with experience:**

- Auto-approve rises from about 20% of sessions for new users to over 40% by about 750 sessions.
- Interrupt rates *also* rise, from about 5% to about 9% of turns. The authors read this as a shift from approving each action to monitoring actively, not as giving up oversight.

**Agent-initiated stops:**

- On the most complex tasks, Claude asks for clarification more than twice as often as on minimal-complexity tasks.
- The top reasons Claude stops itself: choosing between approaches (35%), gathering diagnostics (21%), and clarifying vague requests (13%).
- The top reasons humans interrupt: missing technical context or corrections (32%), and slow or excessive behavior (17%).

**Public API (998,481 tool calls):**

- 80% of calls come from agents with at least one safeguard.
- 73% have some human in the loop. Both figures are upper bounds.
- 0.8% of actions appear irreversible.
- Software engineering accounts for nearly 50% of calls.

**Deployment overhang.** Models can handle more autonomy than users grant them. The study contrasts METR's estimate that Claude Opus 4.5 completes about 5-hour human tasks at 50% success, while noting the two metrics are not directly comparable [17].

**Conclusion.** The study concludes that autonomy is "co-constructed by the model, the user, and the product." It recommends that policymakers avoid mandating specific interaction patterns, such as approving every action, and focus on whether humans can effectively monitor and intervene [17].

---

## 9. From patterns to harnesses (2025-2026)

The foundational patterns now live inside *harnesses*, the code that loops the model, routes its tools, and manages its context. Anthropic's 2025-2026 engineering posts show how harness design has evolved.

### 9.1 Long-running harnesses

**Problem.** Agents working across many context windows start each session with no memory of earlier work. Compaction alone was not enough [13].

**The November 2025 harness** splits the work between two agents [13]:

- An **initializer agent** runs once. It writes:
  - `init.sh`
  - a `claude-progress.txt` log
  - an initial git commit
  - a JSON feature list with more than 200 features, all initially failing. JSON was chosen because the model is less likely to overwrite it inappropriately than Markdown.
- A **coding agent** runs every later session. It reads the progress file and git log, runs a basic end-to-end check, implements one feature at a time, tests it with browser automation, commits, and updates the notes.

**Failure modes it fixes** [13]:

- declaring victory early
- one-shotting the whole app and running out of context mid-implementation
- marking features done prematurely
- wasting time working out how to run the app

### 9.2 Generator/evaluator harnesses

The March 2026 harness adds a GAN-inspired **planner, generator, and evaluator** [14]:

- A *planner* expands a 1-4 sentence prompt into a spec.
- Before each sprint, the generator and evaluator negotiate **sprint contracts** that define "done."
- A Playwright-driven *evaluator* grades against hard thresholds.

**Cost and time** [14]:

| Run | Duration | Cost |
|---|---|---|
| Retro game maker, solo (Opus 4.5) | 20 min | $9 |
| Retro game maker, full harness (Opus 4.5) | 6 hours | $200 |
| Digital audio workstation, V2 harness (Opus 4.6) | 3 h 50 min | $124.70 |

The solo run's core feature (play mode) did not work [14].

**The lesson on scaffolding** [14]:

- *Every harness component encodes an assumption about what the model can't do alone*, and those assumptions go stale.
- With Opus 4.6 the sprint decomposition was removed, and the evaluator became a single end-of-run pass whose value depends on whether the task is beyond the model's solo reliability.
- Remove components one at a time. Radical simplification made it impossible to tell which pieces were load-bearing.

### 9.3 Brain/hands decoupling

The April 2026 Managed Agents post takes the same lesson to the infrastructure layer [15].

**Architecture.** Three components:

- the **brain:** Claude plus its harness loop
- the **hands:** sandboxes and tools behind a uniform `execute(name, input) -> string` interface
- the **session:** an append-only event log stored outside both

Each can fail or be replaced on its own. A crashed harness restarts with `wake(sessionId)`. Credentials stay out of the sandbox where generated code runs [15].

**Results.** Provisioning containers only when needed cut p50 time-to-first-token by about 60% and p95 by over 90% [15]. The post also notes that context resets that helped Sonnet 4.5 with "context anxiety" became unnecessary on Opus 4.5. That is one more instance of harness assumptions going stale [15].

### 9.4 Evaluating agents

Anthropic's January 2026 evals guide supplies the vocabulary that rigorous agent work needs [18]:

- **Terms:** task, trial, grader, transcript, outcome, and harness.
- **Grade the outcome, not the claim.** Check whether the reservation is actually in the database, not whether the agent says it made one.
- **pass@k vs. pass^k.** At a 75% per-trial success rate, the chance of passing three of three trials is about 42%. pass^k is the right metric for customer-facing consistency.
- **Capability vs. regression evals.** Capability suites start at low pass rates. Regression suites sit near 100% and catch backsliding.
- **Start early.** 20-50 tasks drawn from real failures is enough to begin.
- **Fix broken evals, not just agents.** CORE-Bench went from 42% to 95% for Opus 4.5 after grading fixes and a less constrained scaffold [18].

---

## 10. Key takeaways for system designers

1. **Choose by control-flow ownership, not by buzzword.**
   - If code can own the control flow reliably, build a workflow: chaining, routing, parallelization, or an evaluator loop [1][6][26].
   - Hand control to the LLM only when the path truly cannot be predicted.
   - Note which autonomy axis you mean (Section 8.1).
2. **Climb the complexity ladder one rung at a time.**
   - The order is: single call, then a workflow, then a single agent with tools, then multiple agents [1][26].
   - Move up only when evals show the current rung fails.
   - OpenAI's triggers for splitting agents are complex conditional logic and *overlapping* tools, not tool count alone [6].
3. **Budget explicitly for agency.** Agents use about 4x chat tokens, and multi-agent systems about 15x [10]. Write effort-scaling rules into prompts [10], set `max_turns` and iteration caps [8][26], and measure tokens per agent [26].
4. **Treat the ACI as a product.**
   - Consolidate and namespace tools, and return high-signal, paginated outputs [11].
   - Poka-yoke the arguments, for example with absolute paths [1][12].
   - Improve tools through evals and transcript review [11].
   - Consider code as the action space when tool composition or data volume dominates [19][21].
5. **Engineer context, not just prompts.** Use compaction, external notes and progress files, sub-agents that return condensed summaries, and just-in-time retrieval [9][13]. Keep durable state outside the context window [15].
6. **Use orchestrator-workers for breadth; keep writes single-threaded for coupled work.** Parallel subagents shine at research [10]. Parallel *writers* conflict through implicit decisions [28][29].
7. **Make evaluators earn their place.**
   - Evaluator-optimizer needs crisp criteria, iteration caps, and calibration against humans [1][18][26].
   - Out-of-the-box LLM judges are lenient [14].
   - Grade outcomes, and use pass^k where users need consistency [18].
8. **Layer guardrails and gate by risk.**
   - Combine relevance, safety, PII, moderation, rules-based, and output checks with tool risk ratings [6].
   - Choose blocking or parallel guardrail execution on purpose [8].
   - Escalate to a human on failure thresholds and high-risk actions [6].
   - Classifier gates can stand in for approval prompts, but they have measurable miss rates [16].
9. **Design for the oversight people actually practice.** Experienced users auto-approve more and interrupt more [17]. Give them trustworthy visibility and cheap intervention, and train agents to stop and ask when uncertain [17].
10. **Expect your scaffolding to expire.** Each harness component encodes a model limitation. Re-ablate after every model upgrade, one component at a time [14][15].

---

## References

1. "Building Effective Agents." Erik Schluntz and Barry Zhang, Anthropic, 2024 (page updated later). https://www.anthropic.com/engineering/building-effective-agents
2. "ReAct: Synergizing Reasoning and Acting in Language Models." Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao, arXiv 2210.03629 / ICLR 2023, 2022. https://arxiv.org/abs/2210.03629
3. "ReAct: Synergizing Reasoning and Acting in Language Models" (project page with abstract, examples, and fine-tuning results). Yao et al., 2022. https://react-lm.github.io/
4. "ReAct" official code repository (README with PaLM and GPT-3 results). Shunyu Yao, 2022-2023. https://github.com/ysymyth/ReAct
5. "ReAct: Synergizing Reasoning and Acting in Language Models" (blog post). Google Research, 2022. https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/
6. "A Practical Guide to Building Agents." OpenAI, 2025. https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf (Markdown transcription consulted: https://gist.github.com/testy-cool/86cafd426ba22e3e8c1d6d2c853506c4)
7. "A Practical Guide to Building AI Agents" (landing page). OpenAI, 2025. https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
8. OpenAI Agents SDK documentation: overview, multi-agent orchestration, guardrails, running agents. OpenAI, 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/index.md ; https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md ; https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md ; https://github.com/openai/openai-agents-python/blob/main/docs/running_agents.md
9. "Effective Context Engineering for AI Agents." Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield, Anthropic, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
10. "How We Built Our Multi-Agent Research System." Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, Daniel Ford, Anthropic, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
11. "Writing Effective Tools for Agents, with Agents." Ken Aizawa, Anthropic, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
12. "Raising the Bar on SWE-bench Verified with Claude 3.5 Sonnet." Anthropic, 2025. https://www.anthropic.com/engineering/swe-bench-sonnet
13. "Effective Harnesses for Long-Running Agents." Justin Young, Anthropic, 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
14. "Harness Design for Long-Running Application Development." Prithvi Rajasekaran, Anthropic, 2026. https://www.anthropic.com/engineering/harness-design-long-running-apps
15. "Scaling Managed Agents: Decoupling the Brain from the Hands." Lance Martin, Gabe Cemaj, Michael Cohen, Anthropic, 2026. https://www.anthropic.com/engineering/managed-agents
16. "How We Built Claude Code Auto Mode: A Safer Way to Skip Permissions." John Hughes, Anthropic, 2026. https://www.anthropic.com/engineering/claude-code-auto-mode
17. "Measuring AI Agent Autonomy in Practice." Miles McCain, Thomas Millar, Saffron Huang, et al., Anthropic, 2026. https://www.anthropic.com/research/measuring-agent-autonomy
18. "Demystifying Evals for AI Agents." Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, Jiri De Jonghe, Anthropic, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
19. "Code Execution with MCP: Building More Efficient Agents." Adam Jones and Conor Kelly, Anthropic, 2025. https://www.anthropic.com/engineering/code-execution-with-mcp
20. "Introduction to Agents" (smolagents conceptual guide). Hugging Face, 2025. https://github.com/huggingface/smolagents/blob/main/docs/source/en/conceptual_guides/intro_agents.md
21. "Executable Code Actions Elicit Better LLM Agents" (CodeAct). Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, Heng Ji, ICML 2024. https://arxiv.org/abs/2402.01030 (code: https://github.com/xingyaoww/code-act)
22. "Reflexion: Language Agents with Verbal Reinforcement Learning." Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao, NeurIPS 2023. https://arxiv.org/abs/2303.11366 (code: https://github.com/noahshinn/reflexion)
23. "Self-Refine: Iterative Refinement with Self-Feedback." Aman Madaan, Niket Tandon, et al., 2023. https://arxiv.org/abs/2303.17651 (code: https://github.com/madaan/self-refine)
24. "Levels of Autonomy for AI Agents." K. J. Kevin Feng, David W. McDonald, Amy X. Zhang, University of Washington / Knight First Amendment Institute, 2025. https://arxiv.org/abs/2506.12469 (also https://knightcolumbia.org/content/levels-of-autonomy-for-ai-agents-1)
25. "Levels of AGI for Operationalizing Progress on the Path to AGI." Meredith Ringel Morris et al., Google DeepMind, ICML 2024. https://arxiv.org/abs/2311.02462
26. "AI Agent Orchestration Patterns." Microsoft Azure Architecture Center, 2026. https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns
27. Magentic-One team implementation (docstring describing Orchestrator, ledgers, agents, and safety guidance). Microsoft AutoGen, 2024-2025. https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/teams/magentic_one.py
28. "Multi-Agents: What's Actually Working." Cognition, 2026. https://cognition.ai/blog/multi-agents-working
29. "Why Cognition Does Not Use Multi-Agent Systems" (secondary write-up of Cognition's "Don't Build Multi-Agents"). Jason Liu, 2025. https://jxnl.co/writing/2025/09/11/why-cognition-does-not-use-multi-agent-systems/
30. "Building Agents with the Claude Agent SDK." Thariq Shihipar, Anthropic, 2025. https://claude.com/blog/building-agents-with-the-claude-agent-sdk
