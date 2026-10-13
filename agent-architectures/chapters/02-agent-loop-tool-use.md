# 02. The Agent Loop and Tool Use: Function Calling, Tool Design, Selection at Scale, and Code-as-Action

## Overview

Almost every production agent described publicly between 2023 and 2026 runs the same core
mechanism. A language model is called with a prompt and a set of tool definitions. It either
answers or emits one or more structured tool calls. A harness runs those calls and appends the
results to the transcript. The loop repeats until a stopping condition fires. Anthropic's
widely cited definition reduces agents to "LLMs using tools based on environmental feedback in a
loop" [1]. The OpenAI Agents SDK runner, the Claude Agent SDK, and Hugging Face's smolagents
all implement versions of this loop [6][25][16].

The loop itself is simple. The engineering effort has gone into five areas around it, and this
chapter covers each:

1. **The function-calling protocol.** This covers how tool definitions, calls, and results are
   encoded in provider APIs, how tool choice is forced or constrained, and how schema adherence
   is guaranteed (Anthropic Messages API, OpenAI function calling, MCP).
2. **Tool design as an interface discipline.** Anthropic calls this the "agent-computer
   interface" (ACI) [1]. SWE-agent showed it empirically [33][34], and Anthropic's 2025 guide
   turned it into an evaluation-driven process [2].
3. **Tool selection at scale.** When an agent can reach hundreds or thousands of tools, loading
   every definition harms both cost and accuracy. Responses range from retrieval research
   (Gorilla, ToolLLM, ToolRet, RAG-MCP) to production deferred loading (Anthropic Tool Search,
   OpenAI tool search) [3][11][24][28][29][30][31].
4. **Code as the action space.** CodeAct, smolagents, Anthropic's Programmatic Tool Calling,
   "code execution with MCP," and Cloudflare's Code Mode replace one-call-per-turn JSON with
   generated programs that call tools [14][16][12][4][21].
5. **Error recovery and stopping conditions.** These include `is_error` results, instructive
   error messages, linters and guardrails, iteration and turn caps, stop-reason handling,
   compaction, and reliability metrics such as pass^k [9][10][25][36].

Throughout the chapter, a number is attributed to the organization that published it. Vendor
figures are internal evaluations unless stated otherwise.

---

## 1. Lineage: From Prompted ReAct to Native Function Calling

### 1.1 ReAct (2022)

**Problem.** Chain-of-thought reasoning helps models plan, but it is ungrounded. Acting without
reasoning cannot adapt plans to what the environment reports.

**Design.** ReAct interleaves free-text *Thought* steps with *Action* steps (calls into an
environment or API) and *Observation* steps (the returned results). It does this in a single
prompted trace [26]. Thoughts decompose goals, track subgoal completion, and decide what to
retrieve next.

**Results.** On ALFWorld and WebShop, ReAct beat imitation-learning and reinforcement-learning
baselines by 34 and 10 absolute points of success rate, using only one or two in-context
examples [26].

**Legacy.** The Thought/Action/Observation cycle is the direct ancestor of the modern loop.
smolagents still describes its `CodeAgent` as a ReAct-style agent [16].

### 1.2 Toolformer and Gorilla (2023): learning when and how to call

Toolformer (Meta AI, NeurIPS 2023) trained a model in a self-supervised way to decide which API
to call, when to call it, and with what arguments. It needed only a handful of demonstrations
per tool. The tools included a calculator, a QA system, search, translation, and a calendar
[27].

Gorilla fine-tuned LLaMA for writing API calls over HuggingFace, TorchHub, and TensorHub (the
APIBench dataset). It reported surpassing GPT-4 at writing API calls. Pairing the model with a
document retriever let it adapt to API documentation that changed at test time and
substantially reduced hallucination [28]. This was an early sign that tool knowledge should be *retrieved*, not
memorized. Section 4 returns to that idea.

### 1.3 Native function calling (2023 onward)

Providers then moved tool use into the API itself. A developer declares tools as JSON Schema
objects. The model emits typed tool-call blocks instead of free-text `Action:` lines. The
harness returns typed results that are matched to the calls by ID [8][9][23]. The protocol
details matter because they decide which failure modes are even possible.

---

## 2. The Canonical Agent Loop

### 2.1 Reference architecture

```
                 +-----------------------------------------------+
   user task --> |  context = system + tool defs + history       |
                 +-----------------------+-----------------------+
                                         |
                                         v
                              +--------------------+
                              |   LLM inference    |
                              +----------+---------+
                                         |
               +-------------------------+-------------------------+
               | stop_reason / output classification               |
               +----+-----------------+----------------+-----------+
                    |                 |                |
            end_turn / final     tool_use(s)      max_tokens, pause_turn,
            (no tool calls)           |           refusal, context-exceeded
                    |                 v                |
                    |       +-------------------+      v
                    |       | harness: validate,|   special handling
                    |       | authorize, execute|   (continue, retry,
                    |       | (parallel ok)     |    fallback, compact)
                    |       +---------+---------+
                    |                 |
                    |      tool_result blocks (is_error?) appended
                    |                 |
                    v                 +--> loop (turn counter += 1; check caps)
                 answer
```

Three implementations show how the vendors fill in the boxes.

- **Anthropic Messages API.** A response with `stop_reason: "tool_use"` contains one or more
  `tool_use` blocks, each with an `id`, a `name`, and an `input` [9]. The harness sends the next
  `user` message containing a `tool_result` for **every** `tool_use` block, matched by
  `tool_use_id`. The result blocks must come first in that message. No message may sit between
  the call and its result, and adding text after results can produce empty or failed turns [9][10].
  Unlike APIs that use a dedicated `tool` or `function` role, Claude carries tools inside the
  ordinary `user` and `assistant` messages [9].
- **OpenAI Agents SDK.** The runner calls the LLM and classifies its output. A *final output*
  ends the loop. A *handoff* switches the active agent and loops again. *Tool calls* are
  executed, their results are appended, and the loop runs again. An output counts as final when
  it contains text of the desired type and no tool calls. Going past `max_turns` raises
  `MaxTurnsExceeded` [25].
- **smolagents.** Memory starts with the user task. On each step, the LLM produces an action, the
  action is executed, and the result is appended to memory. The loop ends when the task is solved
  (via a `final_answer` call) [20][16].

Anthropic's Claude Agent SDK post describes the same loop at a higher level: **gather context →
take action → verify work → repeat** [6]. Its distinctive point is that verification is a
first-class part of the loop and not an afterthought (see Section 6.4).

### 2.2 The hidden cost of tool definitions

Tool definitions are not free metadata. Claude's API builds a dedicated system prompt from the
tool definitions, tool configuration, and the user's system prompt [8]. Definitions sit at the
front of the context, which affects both token cost and caching:

- Anthropic reports a five-server MCP setup (GitHub, Slack, Sentry, Grafana, Splunk) with 58
  tools that uses about 55K tokens before any work starts. It has seen definitions reach 134K
  tokens before optimization [3].
- The Manus team calls KV-cache hit rate "the most important metric" for a production agent.
  Their tasks average about 50 tool calls, with an input-to-output token ratio of roughly 100:1
  [22]. They cite cached Claude Sonnet input at $0.30 per million tokens against $3.00 uncached
  [22]. Because tool definitions sit near the front of the context, editing them mid-task
  invalidates the cache for everything after them [22].
- Changing `tool_choice` on Claude invalidates cached message blocks, although tool definitions
  and the system prompt stay cached [8].

### 2.3 Controlling the choice: `tool_choice`, strict mode, and masking

**Anthropic.** `tool_choice` takes four values [8]:

- `auto`: the default when tools are present.
- `any`: the model must call some tool.
- `tool`: the model must call one named tool.
- `none`: the default when no tools are present.

Forcing a tool works by prefilling the assistant turn, so no natural-language preamble or
thinking block comes before the call [8]. Recent documentation (2026) states that some newer
models reject `any` and `tool` with a 400 error. For those models it recommends `auto` combined
with **strict tool use** (`strict: true`, grammar-constrained decoding that guarantees
schema-valid inputs) or structured outputs [8]. The tool search documentation notes that the
strict-mode grammar is built from the *full* toolset, so deferred loading does not force a
grammar recompile [11].

**OpenAI.** OpenAI's function calling uses `tools`; the older `functions` parameter is
deprecated. `tool_choice` supports `auto`, `required`, and `none`, plus forcing a specific
function. Each result is paired to its call by `call_id` in the Responses API or `tool_call_id`
in Chat Completions [23]. With `strict: true`, the schema is pre-processed and generation is
constrained to it. Secondary guides report that strict schemas require every property to be
listed in `required` and `additionalProperties: false` [23]. One practitioner caution applies:
strict mode guarantees *shape*, not business validity. A schema-valid call can still name a city
that does not exist, so semantic validation stays in the runtime [23].

**Manus: mask, don't remove.** Manus gives a production argument against changing the toolset
dynamically. Removing a tool mid-task invalidates the KV cache. It also confuses the model when
earlier steps still refer to the removed tool, which leads to hallucinated or malformed actions
[22]. Manus keeps definitions fixed. A context-aware state machine then **masks token logits**
during decoding to prevent or enforce the selection of certain actions [22]. Commentary on the
post notes that most providers and inference frameworks support some form of response prefill,
which constrains the action space without touching the tool definitions [22].

```
  Manus-style action constraint (conceptual)
  +-------------------+     state      +------------------------------+
  | agent state       | ------------>  | allowed action set           |
  | machine           |                | (subset of fixed toolset)    |
  +-------------------+                +--------------+---------------+
                                                      |
  fixed tool definitions (cache-stable prefix)        v
  -------------------------------------------> decoder with logit mask
                                               over disallowed actions
```

---

## 3. Tool Design: The Agent-Computer Interface

### 3.1 Anthropic, "Building Effective Agents" (Dec 2024)

**Thesis.** Teams should spend as much effort on the agent-computer interface as on
human-computer interfaces [1]. Anthropic offers three heuristics:

- Put yourself in the model's shoes. If a tool is not obvious from its description and
  parameters, it is not obvious to the model either.
- Choose argument formats that are close to text commonly seen on the internet. Avoid formats
  that need overhead such as counting lines or escaping strings.
- **Poka-yoke** the tools, so that mistakes are structurally hard to make [1].

**Case study.** During Anthropic's SWE-bench work, the model made errors with relative file paths
after it had moved out of the root directory. The tool was changed to require absolute paths, and
the model then used it "flawlessly" [1].

### 3.2 SWE-agent: the ACI as an empirical variable (NeurIPS 2024)

**Problem.** Dropping an LM into a bare Linux shell gives it an interface designed for humans.

**Architecture.** SWE-agent places custom commands between the model and the repository [34]:

- a file viewer that shows **100 lines per turn**, with scrolling and in-file search;
- an edit command with an **integrated linter** that rejects edits leaving the file
  syntactically invalid;
- a directory-wide search that lists only the *file names* containing matches, because extra
  match context confused the model;
- an explicit message when a command produces no output: "Your command ran successfully and did
  not produce any output."

**Results.** The paper reported 12.5% pass@1 on SWE-bench and 87.7% on HumanEvalFix, which were
state-of-the-art for that setup at the time [33]. A secondary summary of the ablations reports
roughly 10.7 percentage points more solved issues on 300 SWE-bench Lite instances than an
otherwise identical agent with a plain shell [33]. Treat that figure as approximate.

**Lesson.** The interface is a first-order variable. Small affordances, such as a linter
guardrail, a bounded viewer, or a non-empty acknowledgment, are where error recovery starts.

### 3.3 Anthropic, "Writing Effective Tools for Agents, with Agents" (Sep 2025)

This is the most complete public method for tool design [2]. It frames tools as "a contract
between deterministic systems and non-deterministic agents." It has two parts: a process and a
set of principles.

**Process (evaluation-driven).**

```
  prototype tools --> generate realistic eval tasks --> run agent loop programmatically
        ^             (multi-call, verifiable)          (while-loop; reasoning +
        |                                                feedback blocks)
        |                                                       |
        |                                                       v
  agent refactors tools <-- concatenate transcripts <-- collect metrics: accuracy,
  (Claude Code) and          and failure analysis        runtime, #calls, tokens,
  descriptions                                           tool errors
        |
        +--> validate on held-out test set
```

- Tasks should be grounded in realistic workflows that may need dozens of calls. For example,
  schedule a meeting with a named colleague, attach prior notes, and book a room, rather than
  "schedule a meeting with x@y.com" [2].
- Verifiers should not reject correct answers over formatting. Expected tool sequences are
  optional and should not over-specify the strategy [2].
- The metrics to collect are accuracy, per-call and per-task runtime, total calls, total
  tokens, and tool errors. Redundant calls suggest that pagination or consolidation needs fixing.
  Many invalid-parameter errors suggest that descriptions or examples need fixing [2].
- One real case: Claude kept appending "2025" to web-search queries, which biased the results.
  The fix was a better description [2].
- Anthropic reports that Claude-optimized Slack and Asana MCP servers beat human-written versions
  on held-out sets. The post shows this as charts without exact figures [2].

**Principles.**

| Principle | Concrete guidance [2][8] |
|---|---|
| Choose tools deliberately | Don't wrap every endpoint. Prefer `search_contacts` to `list_contacts`. |
| Consolidate workflows | `schedule_event` instead of `list_users` + `list_events` + `create_event`. `get_customer_context` instead of three getters. Claude docs suggest one tool with an `action` parameter over `create_pr`/`review_pr`/`merge_pr`. |
| Namespace | `asana_search`, `jira_search`; `asana_projects_search`. Prefix versus suffix choice affects results and varies by model, so evaluate it. |
| Return meaningful context | Return `name`, `image_url`, `file_type`, not `uuid` or `mime_type`. Resolving UUIDs to readable or 0-indexed IDs improved precision. |
| Offer verbosity control | A `response_format` enum (`concise`/`detailed`). In one example, 72 vs 206 tokens. In Slack, concise was about 1/3 the tokens, but detailed returned IDs (`thread_ts`, `channel_id`) needed by follow-up calls. |
| Be token-efficient | Pagination, filtering, and truncation with sensible defaults. Claude Code caps tool responses at 25,000 tokens by default. A truncation message should steer toward narrower queries. |
| Make errors actionable | Explain what to fix and give an example of correct input, not opaque codes or tracebacks. |
| Prompt-engineer descriptions | Write for a new hire: name specialized formats and resource relationships. Use `user_id`, not `user`. Description refinements helped Claude 3.5 Sonnet reach SOTA on SWE-bench Verified. |
| Annotate | MCP tool annotations can flag open-world or destructive tools. |

Claude's API documentation adds that detailed descriptions are "by far the most important
factor in tool performance." It recommends at least 3-4 sentences per tool, covering what the
tool does, when to use it and when not to, what each parameter means, and caveats [8]. The
documentation also warns against parameters that ask the model to spell out its step-by-step
reasoning. A request for a short explanation or evidence is preferred [8].

The companion context-engineering post names **bloated, overlapping toolsets** as "one of the
most common failure modes," because overlap makes it unclear which tool to use. It argues for
a minimal viable toolset that is easier to maintain and prune over long interactions [5].

### 3.4 Tool Use Examples (Nov 2025)

**Problem.** JSON Schema states which inputs are valid. It does not show usage conventions such
as date formats, ID patterns, which optional parameters appear together, or how to tell
`create_ticket` from `create_incident`.

**Design.** An `input_examples` array on the tool definition. Each example must validate against
the schema, and an invalid example returns a 400 error [8]. Examples cost about 20-50 tokens
when simple and 100-200 tokens for complex nested objects [8]. Anthropic recommends 1-5
realistic examples (not placeholders like "string"), added only where the schema leaves usage
ambiguous [3].

**Result.** In Anthropic's internal tests on complex parameter handling, accuracy rose from 72% to
90% [3].

### 3.5 The "think" tool (Mar 2025; superseded Dec 2025)

Anthropic's think tool is a no-op tool whose only purpose is to give the model a scratchpad in
the *middle* of a response, mainly to process new tool results [7]. On tau-bench airline
(Claude 3.7 Sonnet), think plus an optimized prompt scored pass^1 0.584, against a 0.332
baseline. The post's text cites a 54% relative improvement over the next-best configuration
[7]. On SWE-bench, the isolated gain was about 1.6% on average [7]. Anthropic recommended it for
policy-heavy, sequential settings and not for single or parallel non-sequential calls. A
December 2025 update says improved extended thinking now makes a dedicated think tool
unnecessary in most cases [7]. This is a useful example of a tool-design pattern being absorbed
into the model.

---

## 4. Tool Selection at Scale

### 4.1 The two failure modes of "load everything"

1. **Context and cost bloat.** In Anthropic's examples, about 55K tokens go to definitions for 58
   tools, Jira alone takes about 17K, and one deployment reached 134K before optimization [3].
2. **Selection accuracy.** Claude's documentation states that the model's ability to pick the
   right tool "degrades once you exceed 30-50 available tools" [11]. OpenAI's function-calling
   guidance, as reproduced in third-party copies, recommends aiming for fewer than 20 functions
   available at any one time and evaluating performance at different counts [23]. On edge
   devices, "Less is More" (2024) cut the tools shown per request through recommendation plus
   similarity search. It reported up to 70% lower execution time and up to 40% lower power use
   [32].

### 4.2 Retrieval-based selection: the research line

- **ToolLLM / ToolBench (ICLR 2024).** The authors collected **16,464 REST APIs** from RapidAPI,
  grouped into 3,451 tools. They generated instructions and annotated solution paths with
  **DFSDT**, a depth-first search decision tree that lets the model evaluate several reasoning
  paths and backtrack. DFSDT solved instructions that plain CoT or ReAct could not. A neural API
  retriever recommends APIs per instruction, and the reported pass rate drops only slightly
  compared with ground-truth API sets [29].
- **ToolRet (ACL Findings 2025).** This benchmark has 7.6K retrieval tasks over a corpus of
  about 43K tools. Its central finding: models that score well on conventional IR benchmarks
  retrieve tools poorly, and this lowers downstream pass rates. Training on the 200K-instance
  ToolRet-train set raised tool-use pass rates by 10-20% [30].
- **RAG-MCP (2025).** A needle-in-a-haystack stress test presents N MCP schemas (one correct, N-1
  distractors). Retrieving and injecting only the best-matching MCP description raised selection
  accuracy from 13.62% to 43.13% and cut prompt tokens by more than half. Retrieval precision
  degrades as the registry grows [31].

**Lesson.** Tool retrieval is its own IR problem. Tool descriptions are short, functional, and
full of jargon, so generic embedding retrievers underperform on them. Description quality from
Section 3 becomes searchability.

### 4.3 Anthropic Tool Search Tool (Nov 2025; GA in later docs)

**Architecture.**

```
  request.tools = [ tool_search (non-deferred), top 3-5 hot tools (non-deferred),
                    N deferred tools (defer_loading: true, full defs still sent) ]
        |
        v
  context prefix = search tool + hot tools only  (cache-stable)
        |
  model --server_tool_use(pattern|query, limit)--> API-side regex or BM25 search over
        |                                         names, descriptions, arg names/descs
        |<-- tool_search_tool_result { tool_reference[] } (default up to 5)
        |
  API expands references inline into full definitions (appended, prefix untouched)
        |
  model --tool_use(discovered tool)--> client executes --> tool_result
```

Key properties, from the documentation [11]:

- **Two variants.** In the regex variant, the model writes Python `re.search` patterns (at most
  200 characters, case-insensitive). In the BM25 variant, it writes natural-language queries
  (at most 500 characters).
- **Limits.** Up to **10,000 deferred tools** per request. At least one tool must stay
  non-deferred, or the request fails with a 400 error.
- **Caching.** Deferred tools are kept out of the system-prompt prefix, and discovered
  definitions are appended inline. This preserves prompt caching, and discovered tools stay
  usable in later turns without another search.
- **Custom search.** Any client tool can return `tool_reference` blocks, which enables
  embedding-based search.
- **MCP.** Whole MCP servers can be deferred via `mcp_toolset.default_config`.

**Numbers.** The 58-tool example drops from about 77K tokens of overhead to about 8.7K (an 85%
reduction). Accuracy on internal MCP evaluations rose from 49% to 74% for Opus 4 and from 79.5%
to 88.1% for Opus 4.5 [3].

**When not to use it.** Fewer than 10 tools, every tool used in every request, or very small
definitions [11].

**Failure modes.** If no description contains the words the model searches for, the right tool
cannot be found [11]. Each search adds a round trip. Discovery accuracy now depends on naming
and description keywords, and the documentation recommends consistent service prefixes so that
one query matches a whole group [11].

### 4.4 OpenAI tool search and namespaces (2026)

OpenAI's Responses API added a hosted `tool_search` for GPT-5.4 and later models. Functions,
functions inside a **namespace**, or whole MCP servers can be marked `defer_loading: true`. When
the model needs a deferred tool, the output gains a `tool_search_call` item and a
`tool_search_output` item that lists the newly callable subset. Newly discovered tools are
injected at the *end* of the context to keep the cache intact [24]. Two vendors arriving at the
same design (deferred definitions, appended expansion, cache-preserving prefix) suggests it is
becoming the standard answer to toolset scale.

### 4.5 Progressive disclosure through a filesystem

Anthropic's "Code execution with MCP" (Nov 2025) takes a different approach to discovery. MCP
servers are generated into a file tree, `./servers/<server>/<tool>.ts`, and each file wraps one
MCP call. The agent lists directories and reads only the files it needs. An alternative is a
`search_tools` tool with adjustable detail levels, from name only up to the full schema [4].
Here discovery is ordinary agentic file exploration. This links tool selection to the code-as-
action designs in Section 5.

---

## 5. Code as the Action Space

### 5.1 CodeAct (ICML 2024)

**Problem.** JSON and text actions allow one predefined tool per turn. They cannot compose calls,
hold intermediate objects, loop, or branch without a model round trip each time.

**Design.** Executable Python is the single, unified action space. The agent emits code. A Python
interpreter runs it, and stdout, errors, and tracebacks become the next observation. This lets
the model use existing libraries and debug itself across multiple turns [14][15].

**Results.**

- Across 17 LLMs on API-Bank and the new **M3ToolEval** benchmark, CodeAct beat Text and JSON
  actions by up to **20% absolute success rate** [14][15].
- The project also released **CodeActInstruct** (about 7K multi-turn interactions) and
  **CodeActAgent** models fine-tuned from Llama-2-7B (4K context) and Mistral-7B (32K context)
  [15].
- smolagents and Hugging Face materials cite the CodeAct line of work as using about 30% fewer
  steps, and therefore fewer LLM calls, than JSON tool calling [16]. Treat this as approximate.

### 5.2 smolagents (Hugging Face, Dec 2024 onward)

**Architecture.** The agent logic fits in about 1,000 lines (`agents.py`) [16]. Two agent types
are offered:

- `CodeAgent` (recommended) writes Python snippets in which tools are plain function calls.
- `ToolCallingAgent` uses the provider's native JSON tool calling.

Tools can come from MCP servers, LangChain, or Hugging Face Spaces. Agents and tools can be
shared on the Hub [16].

**Why code?** The library gives four reasons [17][20]:

- *Composability*: actions nest and can be reused.
- *Object management*: an image output can live in a variable.
- *Generality*: code can express any computable task.
- *Training-data fit*: models have seen far more code than tool-call JSON.

**Evidence.** In Hugging Face's Open Deep Research replication, a CodeAgent reached **55.15%
on the GAIA validation set**, against about 46% for the previous best open framework
(Magentic-One). Switching the same system to JSON actions dropped it to **about 33%** [18]. This
is one of the cleanest public ablations of code versus JSON actions at system level, although it
is measured on the validation split.

**Execution security.** The built-in `LocalPythonExecutor` interprets the AST operation by
operation. It blocks imports (and submodules) outside an allowlist, raises an error on undefined
operations, and caps the number of elementary operations to stop runaway loops [19]. The docs
state that "no local python sandbox can ever be completely secure," and give an example: an
authorized package like Pillow could still be used to fill a disk. Remote sandboxes are
recommended instead: E2B, Modal, Blaxel, or Docker [19]. The hardened Docker example runs as
`nobody` with 512 MB of memory, a 100-process limit, dropped capabilities, and
`no-new-privileges` [19]. A sandbox that covers only code snippets does not support managed
multi-agent setups. Sandboxing the whole system may mean putting API keys inside the sandbox
[19].

### 5.3 Anthropic Programmatic Tool Calling (Nov 2025; GA in 2026 docs)

**Problem.** Every tool result in classic tool calling passes through the model's context. Every
dependent call costs another inference pass, typically hundreds of milliseconds to seconds each
[3].

**Architecture.**

```
  model ---writes Python---> [ code_execution container (sandbox) ]
                               |  tools with allowed_callers=[code_execution]
                               |  appear as async functions
                               |
                               |  await query_db({...}) --pause--> API returns tool_use
                               |                                 with caller=code_exec,
                               |                                 container id + expires_at
  client executes tool  <------------------------------------+
  client sends tool_result (tool_result only, same container) +
                               |  code resumes; filter/aggregate/loop/asyncio.gather
                               v
                     only final stdout/result enters model context
```

**Mechanics** [12]:

- Tools opt in through `allowed_callers` and are exposed as async Python functions that take a
  dict and return a string.
- Each tool call pauses execution and surfaces a `tool_use` block with a `caller` field, which is
  how the client tells programmatic calls apart from direct ones.
- Idle containers are reclaimed after about 5 minutes. A pending tool call times out after about
  4 minutes and raises `TimeoutError` inside the running code, where Claude typically sees it in
  stderr and retries.
- **Incompatibilities:** `strict: true` tools, forcing programmatic calls through `tool_choice`,
  `disable_parallel_tool_use`, MCP-connector tools, and schemas with recursive `$ref`.
- `allowed_callers` "is not a hard API-level block." Clients must still be ready for direct
  invocation, and the field must not be treated as a security boundary.

**Numbers** [3]:

- Average tokens on complex research tasks fell from 43,588 to 27,297 (-37%).
- Internal knowledge retrieval accuracy rose from 25.6% to 28.5%, and GIA benchmarks from 46.5%
  to 51.2%.
- In a budget-compliance example (20 employees, 2,000+ expense lines), the post describes about
  200 KB of raw data reduced to about 1 KB of results. The same post elsewhere describes the raw
  data as "50 KB+," so treat the magnitude as approximate.
- Orchestrating 20+ calls in one code block removes 19+ inference passes.

Anthropic recommends it for aggregating large datasets, three or more dependent calls, and fan-
out across many items. It recommends against it for single calls or when the model must reason
over every intermediate result [3].

**Downstream use.** In February 2026, Anthropic shipped web search and web fetch tools that
*dynamically filter* results by writing code before results enter context. Across BrowseComp
and DeepSearchQA this gave an average 11% improvement with 24% fewer input tokens. For example,
Opus 4.6 on BrowseComp rose from 45.3% to 61.6%. Price-weighted cost fell for Sonnet 4.6 but
*rose* for Opus 4.6, because it wrote more code [13][12].

### 5.4 Code execution with MCP and Cloudflare Code Mode

Anthropic's "Code execution with MCP" (Jones and Kelly, Nov 2025) generalizes the idea to
whole MCP ecosystems [4]:

- **Filesystem tool APIs.** Tools appear as files (Section 4.5).
- **Context-efficient results.** For example, log 5 rows of a 10,000-row sheet.
- **Native control flow.**
- **Privacy.** Intermediate data stays in the sandbox, and the MCP client can tokenize PII
  before it reaches the model.
- **State persistence** in `./workspace/`.
- **Skills.** Working code is saved in `./skills/` with a `SKILL.md` file.
- **Headline example.** A Google Drive → Salesforce workflow drops from about 150,000 tokens to
  about 2,000 (-98.7%).
- **Cost.** Secure sandboxing, resource limits, and monitoring add operational overhead that
  direct tool calls avoid [4].

Cloudflare reached the same conclusion independently. Its September 2025 "Code Mode" post argued
that agents should write code against typed APIs instead of calling MCP tools directly. In
February 2026 Cloudflare followed up with an MCP server that exposes its entire API through only
two tools [21]:

- `search()` runs JavaScript against the OpenAPI specification.
- `execute()` runs JavaScript against the API inside a Dynamic Worker (V8 isolate) sandbox.

Cloudflare reports that an equivalent non-Code-Mode server would need about **1.17 million
tokens** of tool definitions, so the two-tool design cuts input tokens by about 99.9% [21].
Secondary coverage describes 2,500+ endpoints served in about 1,000 tokens (approximate) [21].

### 5.5 Comparison of action-space designs

| Design | Action unit | Round trips per N dependent calls | Intermediate data in context | Selection mechanism | Main risks | Reported gains |
|---|---|---|---|---|---|---|
| Prompted ReAct [26] | Text "Action:" line | N | All | All tools in prompt | Parse errors, hallucinated tools | +34 / +10 pts (ALFWorld / WebShop) |
| Native JSON function calling [8][23] | Typed tool-call block | N (parallel calls help when independent) | All | All tools in prompt; `tool_choice` | Bloat; selection degrades past ~30-50 tools [11] | Baseline |
| JSON + tool search [11][24] | Typed call after discovery | N + searches | All | Regex/BM25/hosted search over deferred defs | Discovery misses if descriptions lack keywords | Opus 4 49%→74%; ~85% fewer tokens [3] |
| CodeAct / smolagents [14][16] | Python program | ~1 per program | Only printed output | Tools as functions in prompt | Sandbox escape; code bugs | Up to +20% SR [14]; GAIA 55% vs 33% JSON [18] |
| Programmatic Tool Calling [12] | Python in managed container calling client tools | 1 model pass; N client executions | Final output only | `allowed_callers` opt-in | Container timeouts; no strict mode | -37% tokens; GIA 46.5→51.2% [3] |
| Code exec with MCP / Code Mode [4][21] | TS/JS program against generated SDK | ~1 per program | Final output only | Filesystem or `search()` over spec | Ops overhead; isolation | 150K→2K tokens [4]; ~99.9% fewer input tokens [21] |
| Masked fixed toolset (Manus) [22] | Typed call, logit-constrained | N | All (file system used as memory) | State machine masks logits | Requires decoder control or prefill | KV-cache stability (qualitative) |

---

## 6. Error Recovery

### 6.1 Error taxonomy at the protocol layer

Two layers must be kept separate.

- **Protocol errors vs. execution errors (MCP).** A malformed or unsupported request returns a
  JSON-RPC error, which means the tool never ran. A valid request whose execution failed returns
  a *successful* JSON-RPC response with `isError: true` in the result. Only the second kind
  carries content the model can read and react to [38].
- **Claude client tools.** The harness returns a `tool_result` with `is_error: true` and an
  explanatory `content` field [9]. For invalid calls, such as a missing required parameter, the
  documentation states that Claude "will retry 2-3 times with corrections before apologizing to
  the user." It recommends `strict: true` to eliminate schema violations entirely [9].
- **Server tools** (web search, tool search, code execution) report errors inside 200-status
  result blocks with codes such as `too_many_requests`, `invalid_tool_input`, `unavailable`, and
  `max_uses_exceeded`. The model handles these transparently [9][11].

### 6.2 Error messages as prompts

Several sources agree that the content of an error message is itself a steering prompt:

- Write "Rate limit exceeded. Retry after 60 seconds.", not "failed" [9].
- Explain what to fix and show an example of correct input, not opaque codes or raw tracebacks
  [2].
- A truncation notice should tell the agent how to narrow its query [2].
- SWE-agent's linter rejects broken edits *before* they land. Its explicit "ran successfully and
  did not produce any output" message stops the model from misreading silence as failure [34].

### 6.3 Keep the failure in the transcript

The Manus team argues for leaving failed actions and their error observations *in* the context,
instead of cleaning them up or retrying silently, so that the model can learn from them [22].
This principle is reported in secondary summaries of the post as a counterintuitive but
central error-recovery practice [22]. CodeAct's design
reaches the same point from another direction. Tracebacks are ordinary observations, which is
what makes multi-turn self-debugging possible [14]. In Programmatic Tool Calling, a tool
`TimeoutError` appears in the code's stderr, and Claude typically retries [12].

### 6.4 Verification as a loop stage

The Claude Agent SDK guidance lists three verification mechanisms [6]:

- **Rules-based feedback**, such as linters and validators that say which rule failed. The
  guidance suggests generating TypeScript over JavaScript to gain extra feedback layers.
- **Visual feedback**, such as screenshots or renders, for example through Playwright MCP.
- **LLM-as-judge** for fuzzy criteria, described as less robust and adding latency and cost.

When the agent keeps failing, the post's prescriptions are about tools: check whether the agent
has the right tools, make search APIs easier, add formal rules to tool calls, and offer
alternative tools if the agent cannot fix its own errors [6].

### 6.5 Recovery-relevant protocol edge cases (Claude)

These come from the stop-reasons documentation [10]:

- **Incomplete tool call at `max_tokens`.** If the last block is a truncated `tool_use`, retry
  with a higher `max_tokens`. The partial call is not usable.
- **Empty `end_turn`.** A response of 2-3 tokens with no content, often after tool results, is
  usually caused by adding text right after `tool_result` blocks. Do not resend the empty
  response unchanged. As a last resort, add a new user message such as "Please continue."
- **`refusal`.** Returned with HTTP 200. Read `stop_details`, and on supported models retry on a
  fallback model.

---

## 7. Stopping Conditions

### 7.1 Why stopping is a design problem

An agent loop has no natural upper bound. Anthropic notes that tasks usually end on completion,
but that "it's also common to include stopping conditions (such as a maximum number of
iterations) to maintain control" [1]. Public implementations combine several layers of stops:

```
  +--------------------------- outer budget ---------------------------+
  |  wall-clock / cost / token budget (harness)                         |
  |  +----------------------- turn budget ---------------------------+  |
  |  |  max_turns (OpenAI SDK), max iterations (loop), server-tool    |  |
  |  |  loop cap -> pause_turn (default 10)                           |  |
  |  |  +------------------- semantic stop -----------------------+  |  |
  |  |  |  model ends turn with no tool calls / typed final output |  |  |
  |  |  |  explicit final_answer tool (smolagents)                 |  |  |
  |  |  |  verification passes (tests, linters, judge)             |  |  |
  |  |  +----------------------------------------------------------+  |  |
  |  +----------------------------------------------------------------+  |
  |  context limit -> compaction / tool-result clearing / truncation    |
  +---------------------------------------------------------------------+
```

### 7.2 Provider stop signals

| Signal | Provider | Meaning | Harness action |
|---|---|---|---|
| `end_turn` | Anthropic [10] | Model finished | Use the output; guard against empty responses |
| `tool_use` | Anthropic [10] | Client tool requested | Execute and return a `tool_result` for every call |
| `pause_turn` | Anthropic [10] | Server-tool sampling loop hit its limit (default 10) | Send the assistant content back unchanged; cap continuations |
| `max_tokens` | Anthropic [10] | Output cap hit | Raise the cap or continue; discard a partial `tool_use` |
| `model_context_window_exceeded` | Anthropic [10] | Context filled | Treat as truncated; compact |
| `refusal` | Anthropic [10] | Declined (HTTP 200) | Read `stop_details`; fall back to another model |
| Final output (no tool calls, correct type) | OpenAI Agents SDK [25] | Semantic completion | Return the result |
| `MaxTurnsExceeded` | OpenAI Agents SDK [25] | Turn cap hit (`max_turns=None` disables it) | Raise, or use the `max_turns` error handler to return a controlled `final_output` |
| `final_answer` tool call | smolagents [16][20] | Agent declares completion | Return its argument |

The OpenAI SDK's `max_turns` error handler is a useful pattern. Instead of failing hard, the
run returns a fallback `final_output`, and `include_in_history` controls whether that fallback
is saved into the session [25]. Developer reports on the OpenAI forum describe
`MaxTurnsExceeded` during tool chaining where partial results could not be recovered [25]. That
is an argument for persisting intermediate state outside the transcript.

### 7.3 Context exhaustion as a stopping (or continuation) condition

Long-horizon agents hit the context limit before they hit a turn cap. Anthropic describes three
responses [5]:

- **Compaction**: summarize the history near the limit, keeping architectural decisions and open
  bugs and dropping redundant tool output.
- **Tool-result clearing**: drop old raw results, described as "one of the safest, lightest touch
  forms of compaction."
- **Just-in-time retrieval**: keep identifiers such as paths and queries instead of payloads.

Manus uses the file system as externalized memory. Secondary accounts also describe it
rewriting a `todo.md` at each step so that current goals stay at the end of the context [22]. All of these turn a hard stop into a
continuation.

### 7.4 Reliability: stopping correctly, repeatedly

τ-bench (Sierra) evaluates tool-agent-user conversations by comparing the final database state
with an annotated goal state. It introduced **pass^k**: the probability that *all* k independent
trials succeed. At release, gpt-4o-class function-calling agents solved fewer than 50% of tasks,
and pass^8 in retail was below 25% [36][37]. Sierra reports that τ²-bench added a telecom domain
[37]. For loop design, pass^k shows that a loop which usually stops at the right state is not
production-ready. Consistency needs tool guardrails, policy reasoning (for example the think
tool's τ-bench gains [7]), and deterministic checks.

---

## 8. Measuring Function Calling

- **BFCL (Berkeley Function Calling Leaderboard).** BFCL describes itself as the first
  comprehensive and *executable* function-call evaluation [35]. Its versions:
  - v1: AST-checked simple, parallel, and multiple calls.
  - v2: live, user-contributed functions.
  - v3: multi-turn and multi-step calls. Aggregator descriptions say v3 checks the resulting API
    *state* and not only the call format [35].
  - v4: agentic categories (web search, memory management, and format sensitivity) [35].

  Aggregated per-model v4 snapshots show multi-turn accuracy far below single-turn AST accuracy
  for some small models, for example single digits against 70-87% [35]. These are single-model
  data points and should not be generalized.
- **τ-bench / τ²-bench.** Stateful, policy-constrained, multi-turn conversations with a simulated
  user, reported as pass^k [36][37].
- **Internal tool evaluations.** Anthropic's process in Section 3.3 is the reference method for
  evaluating *your own* tools: realistic tasks, verifiable outcomes, a held-out split, and
  transcripts read by humans and agents [2].

---

## 9. Cross-Cutting Comparison: Production Tool-Use Stacks

| Capability | Anthropic Claude API | OpenAI (Responses / Agents SDK) | Hugging Face smolagents | MCP (protocol) |
|---|---|---|---|---|
| Tool definition | `name`, `description`, `input_schema`, `input_examples`, `strict`, `defer_loading`, `allowed_callers` [8][11][12] | JSON Schema functions, `strict`, namespaces, `defer_loading` [23][24] | Python functions / `Tool` classes; MCP, LangChain, Spaces [16] | `tools/list` schemas plus annotations [2][38] |
| Choice control | `auto`/`any`/`tool`/`none`; forced choice prefills, skips thinking [8] | `auto`/`required`/`none`/specific [23] | Code or JSON agent type [16] | n/a (client decides) |
| Schema guarantee | Strict tool use (grammar) [8] | Strict mode / structured outputs [23] | Code parsing / interpreter errors | n/a |
| Scale mechanism | Tool Search (regex/BM25/custom), up to 10K deferred [11] | Hosted `tool_search` (GPT-5.4+) [24] | Code agent can import or define tools | Server-level grouping |
| Code orchestration | Programmatic Tool Calling; code execution tool [12] | (not covered in sources reviewed here) | Native (CodeAgent) [16] | Code-mode wrappers [4][21] |
| Error signal | `tool_result.is_error`; server error codes [9] | Function output content | Interpreter exceptions as observations [19] | `isError` vs JSON-RPC error [38] |
| Loop termination | `stop_reason` set incl. `pause_turn`, `refusal` [10] | Final output, `max_turns`, error handlers [25] | `final_answer`, max steps [16][20] | n/a |

---

## Key Takeaways for System Designers

1. **Treat tool definitions as prompts and the toolset as a product.** Description quality is
   "by far the most important factor" [8]. Ship tool changes only after an evaluation loop with
   realistic multi-call tasks, verifiable outcomes, and a held-out split [2].
2. **Design tools for workflows, not endpoints.** Consolidate (`schedule_event`,
   `get_customer_context`), namespace by service, return human-readable identifiers, and offer a
   `response_format` verbosity switch [2][8].
3. **Poka-yoke inputs and use constrained decoding.** Require absolute paths, use enums and
   unambiguous names like `user_id`, add `input_examples` for convention-heavy tools (72%→90%
   [3]), and turn on strict mode to remove schema errors entirely [1][8][9].
4. **Budget the context like a cache-sensitive system.** Keep the tool prefix stable. Prefer
   masking or deferred loading appended at the end over editing definitions mid-task. Measure KV
   cache hit rate [22][11][24].
5. **Above roughly 10-30 tools, switch to discovery.** Keep the 3-5 hot tools loaded and defer
   the rest behind search (regex, BM25, or embeddings). Be aware that generic retrievers are not
   "tool-savvy" [11][30][31].
6. **Use code as the action space when calls are dependent, data is large, or work fans out.**
   The evidence is consistent across groups: CodeAct (+20% SR), GAIA (55% vs 33%), PTC (-37%
   tokens), MCP code execution (-98.7% tokens), and Code Mode (~99.9% fewer definition tokens)
   [14][18][3][4][21]. Pay for it with real sandboxing. Do not treat an in-process interpreter
   or `allowed_callers` as a security boundary [19][12].
7. **Make errors informative, and keep them visible.** Return `is_error` / `isError` results with
   remediation hints. Keep failures in the transcript so the model can adapt. Separate protocol
   failures, where the tool never ran, from execution failures [9][22][38].
8. **Add verification to the loop itself.** Use linters, tests, renders, and judges, and give the
   agent alternative tools when it cannot fix its own errors [6][34].
9. **Layer stopping conditions.** Combine the semantic stop (no tool calls, typed final output,
   `final_answer`), turn caps with graceful handlers, handling for every provider stop reason
   (including `pause_turn` and truncated `tool_use`), and compaction for context limits
   [10][25][5].
10. **Optimize for pass^k, not pass@1.** Production reliability means repeated success. Many of
    the techniques above (guardrails, policy reasoning, deterministic tools) matter mainly through
    consistency [36][7].
11. **Expect techniques to be absorbed into models and platforms.** The think tool gave way to
    better extended thinking [7], and dynamic filtering moved code-based result pruning into
    first-party search tools [13]. Keep harness abstractions thin enough to drop scaffolding.

---

## References

1. "Building effective agents." Anthropic, 2024.
   https://www.anthropic.com/research/building-effective-agents
2. "Writing effective tools for agents — with agents." Anthropic Engineering, 2025.
   https://www.anthropic.com/engineering/writing-tools-for-agents
3. "Introducing advanced tool use on the Claude Developer Platform." Anthropic Engineering, 2025.
   https://www.anthropic.com/engineering/advanced-tool-use
4. "Code execution with MCP: Building more efficient agents." A. Jones, C. Kelly, Anthropic,
   2025. https://www.anthropic.com/engineering/code-execution-with-mcp
5. "Effective context engineering for AI agents." Anthropic Engineering, 2025.
   https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
6. "Building agents with the Claude Agent SDK." T. Shihipar, Anthropic, 2025.
   https://claude.com/blog/building-agents-with-the-claude-agent-sdk
7. "The 'think' tool: Enabling Claude to stop and think in complex tool use situations."
   Anthropic Engineering, 2025 (updated Dec 2025).
   https://www.anthropic.com/engineering/claude-think-tool
8. "Define tools." Claude API Documentation, Anthropic, 2026.
   https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
9. "Handle tool calls." Claude API Documentation, Anthropic, 2026.
   https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
10. "Stop reasons and fallback." Claude API Documentation, Anthropic, 2026.
    https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
11. "Tool search tool." Claude API Documentation, Anthropic, 2026.
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
12. "Programmatic tool calling." Claude API Documentation, Anthropic, 2026.
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling
13. "Improved web search with dynamic filtering." Anthropic (Claude blog), 2026.
    https://claude.com/blog/improved-web-search-with-dynamic-filtering
14. "Executable Code Actions Elicit Better LLM Agents" (CodeAct). X. Wang, Y. Chen, L. Yuan,
    Y. Zhang, Y. Li, H. Peng, H. Ji (UIUC, Apple), ICML 2024. https://arxiv.org/abs/2402.01030
15. "code-act" repository (CodeActInstruct, CodeActAgent, M3ToolEval). X. Wang et al., 2024.
    https://github.com/xingyaoww/code-act
16. "smolagents" repository README. Hugging Face, 2024-2026.
    https://github.com/huggingface/smolagents
17. "Introducing smolagents, a simple library to build agents." Hugging Face, 2024.
    https://huggingface.co/blog/smolagents
18. "Open-source DeepResearch – Freeing our search agents." Hugging Face, 2025.
    https://huggingface.co/blog/open-deep-research
19. "Secure code execution" (smolagents documentation). Hugging Face, 2025.
    https://huggingface.co/docs/smolagents/v1.10.0/tutorials/secure_code_execution
20. "Introduction to Agents" (smolagents conceptual guide). Hugging Face, 2025.
    https://raw.githubusercontent.com/huggingface/smolagents/main/docs/source/en/conceptual_guides/intro_agents.md
21. "Code Mode: give agents an entire API in 1,000 tokens." M. Carey, Cloudflare, 2026.
    https://blog.cloudflare.com/code-mode-mcp/
22. "Context Engineering for AI Agents: Lessons from Building Manus." Y. "Peak" Ji, Manus, 2025.
    https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
23. "Function calling" guide. OpenAI, 2025-2026.
    https://developers.openai.com/docs/guides/function-calling
24. "Tool search" guide. OpenAI (localized docs), 2026; and "Tool search" (Azure OpenAI in
    Foundry), Microsoft, 2026.
    https://developers.openai.com/ja-JP/api/docs/guides/tools-tool-search ;
    https://learn.microsoft.com/en-my/Azure/foundry/openai/how-to/tool-search
25. "Running agents" (OpenAI Agents SDK docs). OpenAI, 2025-2026.
    https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/running_agents.md ;
    see also forum report:
    https://community.openai.com/t/maxturnsexceeded-error-during-agent-tool-execution-unable-to-retrieve-final-output/1370708
26. "ReAct: Synergizing Reasoning and Acting in Language Models." S. Yao et al., ICLR 2023
    (arXiv 2022). https://arxiv.org/pdf/2210.03629
27. "Toolformer: Language Models Can Teach Themselves to Use Tools." T. Schick et al., Meta AI,
    NeurIPS 2023. https://arxiv.org/abs/2302.04761v1
28. "Gorilla: Large Language Model Connected with Massive APIs." S. Patil et al., 2023
    (NeurIPS 2024). https://arxiv.org/abs/2305.15334v1
29. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." Y. Qin et
    al., ICLR 2024.
    https://proceedings.iclr.cc/paper_files/paper/2024/hash/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html
30. "Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models"
    (ToolRet). Z. Shi, Y. Wang et al., Findings of ACL 2025. https://arxiv.org/abs/2503.01763v1
31. "RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented
    Generation." T. Gan, Q. Sun, 2025. https://arxiv.org/html/2505.03275v1
32. "Less is More: Optimizing Function Calling for LLM Execution on Edge Devices."
    V. Paramanayakam et al., 2024. https://arxiv.org/html/2411.15399v1
33. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." J. Yang,
    C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, O. Press, NeurIPS 2024.
    https://papers.neurips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html
34. "Agent-Computer Interface" (SWE-agent documentation). SWE-agent team, 2024-2025.
    https://raw.githubusercontent.com/SWE-agent/SWE-agent/main/docs/background/aci.md
35. "Berkeley Function Calling Leaderboard" (README). Gorilla project, UC Berkeley, 2024-2026.
    https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/README.md
36. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." S. Yao,
    N. Shinn, P. Razavi, K. Narasimhan (Sierra), 2024. https://arxiv.org/pdf/2406.12045
37. "Sierra's τ-bench is shaping the development and evaluation of agents." Sierra, 2025.
    https://sierra.ai/blog/tau-bench-shaping-development-evaluation-agents
38. "Tools" (Model Context Protocol specification, 2025-06-18). MCP project, 2025.
    https://modelcontextprotocol.io/specification/2025-06-18/server/tools
