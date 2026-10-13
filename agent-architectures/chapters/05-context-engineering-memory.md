# 05. Context Engineering and Memory: Caches, Compaction, Scratchpads, and Long-Horizon State

## Overview

Context engineering is the discipline of choosing which tokens a model sees at each step of an agent loop. Anthropic defines context as "the set of tokens included when sampling from a large-language model" and context engineering as curating and maintaining the best set of tokens during inference: system prompts, tool definitions, MCP resources, external data, and message history [1]. The term spread through practitioner writing in mid-2025. Manus published production lessons in July [2], LangChain grouped techniques into *write, select, compress, isolate* [26][27], and Anthropic published its own guide in September [1]. By 2026 most frontier labs and agent frameworks shipped context management as product features: server-side compaction, tool-result clearing, memory tools, prefix caches, and durable session logs [4]-[7][14][21]-[25].

Two pressures drive the discipline:

1. **Quality.** Long inputs degrade model performance gradually. Anthropic calls this *context rot* and frames context as an "attention budget" that every token draws down [1]. Chroma's 2025 report found that performance on even simple tasks changes, often unevenly, as input length grows across 18 models [28][29][30]. The earlier "Lost in the Middle" study tested how models use information placed at different positions in long inputs [31].
2. **Cost and latency.** Agents re-send their whole history on every step. Manus reports an average input-to-output token ratio of about 100:1 [2], so prefill cost, and therefore prefix-cache hit rate, dominates the bill. Anthropic reports that agents use about 4x the tokens of chat interactions, and multi-agent systems about 15x [16].

The chapter moves from a reference model of the context pipeline (Section 1), through Manus's production lessons (2) and prefix-cache mechanics (3), to the compaction designs of seven production systems (4). It then covers scratchpads and file-system memory (5), long-horizon task state (6), sub-agent isolation (7, treated in depth in Chapter 04), and long-term memory systems (8), and ends with failure modes and design takeaways.

Note on sourcing: several primary hosts (manus.im, arxiv.org, blog.langchain.com, cognition.ai, research.trychroma.com, and Lance Martin's personal blog) could not be reached from the research environment. Claims attributed to them come from the documents as originally published and from their GitHub repositories where those exist. Implementation details for Codex, Deep Agents, ADK, and Letta were read directly from source code and docs in their public repositories. Numbers that could not be re-verified are marked approximate.

---

## 1. A reference model of the context pipeline

### What goes into a request

Each step of an agent loop rebuilds one request. The order of its parts matters for caching, because providers cache by prefix. Anthropic's cache hierarchy is `tools`, then `system`, then `messages`, and a change at one level invalidates that level and every level after it [8].

```
 +-------------------------------- one model request --------------------------------+
 | [tools] -> [system prompt] -> [stable memory] -> [history ............] -> [new]  |
 |  rarely      rarely            per session       append-only, grows      per step |
 |  changes     changes           (CLAUDE.md,       (assistant msgs,                 |
 |                                MEMORY.md,        tool calls, results,             |
 |                                skill index)      thinking, summaries)             |
 +-----------------------------------------------------------------------------------+
        ^ cache breakpoint(s) go on the last byte-identical block before change
```

Claude Code publishes an interactive breakdown of what loads before the user types anything. In its simulated session the system prompt is about 4,200 tokens, auto memory (MEMORY.md) about 680, environment info about 280, deferred MCP tool names about 120, skill descriptions about 450, the user CLAUDE.md about 320, and the project CLAUDE.md about 1,800. Each file read then adds 1,000 to 2,500 tokens [10]. The figures are illustrative, but they show where the budget goes: tool and file observations, not instructions, fill most of a long session.

### Taxonomies

Three overlapping taxonomies recur in the literature.

- **Write / select / compress / isolate (LangChain).** *Write* saves information outside the window (scratchpads, checkpoints, long-term memory). *Select* pulls it back in when needed (retrieval, tool selection). *Compress* keeps only needed tokens (summarization, trimming). *Isolate* splits context across sub-agents, sandboxes, or state fields [26][27]. The companion notebooks report conversation summarization shrinking a run from 115k to 60k tokens [27].
- **Failure modes (Drew Breunig, as implemented by LangChain).** *Context poisoning* means a hallucination or error enters context and keeps being referenced. *Context distraction* means the model leans on accumulated history over its training. *Context confusion* means superfluous content degrades responses. *Context clash* means contradictory information accumulates in context. The six fixes are RAG, tool loadout, context quarantine, pruning, summarization, and offloading. In the demo, pruning cut a retrieval-heavy query from about 25k to about 11k tokens, and binding only the top 5 relevant tools reduced tool confusion [28].
- **Anthropic's long-horizon trio.** These are compaction, structured note-taking, and sub-agent architectures. Compaction suits extended back-and-forth, note-taking suits iterative work with clear milestones, and multi-agent setups suit parallel research [1].

The guiding principle is shared: "find the smallest set of high-signal tokens that maximize the likelihood of some desired outcome" [1].

---

## 2. Manus: lessons from building a production agent

### Problem

Manus is a general-purpose agent that runs tasks in a sandboxed virtual machine with browser, shell, and file tools. Its team chose in-context learning on frontier models over training an end-to-end model, so it could ship improvements in hours instead of weeks. Context design therefore became its main engineering lever. The team says it rebuilt its agent framework four times, a manual search process it calls "Stochastic Graduate Descent" [2].

### Architecture and data flow

```
 user task
    |
    v
 +--------------------+    append-only     +-------------------------------+
 | stable prefix      |------------------->| context = prefix + actions    |
 | (system + tools,   |                    |   + observations (+ todo.md   |
 |  no timestamps)    |                    |   recitation at the tail)     |
 +--------------------+                    +---------------+---------------+
                                                           | decode, with
                                                           | logit masks from
                                                           v a state machine
 +-------------------+   large observations  +-------------------------------+
 | sandbox file      |<----------------------| action: browser_* / shell_* / |
 | system (memory)   |---- path / URL ------>| file ops ...                  |
 +-------------------+   restorable stubs    +-------------------------------+
```

### Key techniques

1. **Design around the KV cache.** Manus calls KV-cache hit rate "the single most important metric for a production-stage AI agent." In its workloads the input-to-output ratio averages about 100:1. With Claude Sonnet, cached input costs USD 0.30 per million tokens versus USD 3.00 uncached, a 10x difference [2]. The practices that follow are:
   - Keep the prompt prefix stable. A timestamp precise to the second at the start of the system prompt invalidates the cache on every request.
   - Keep context append-only. Do not edit earlier actions or observations, and serialize deterministically, because many JSON libraries do not guarantee stable key order.
   - Mark cache breakpoints explicitly when the provider requires it, at least at the end of the system prompt. For self-hosted vLLM, enable prefix caching and use session IDs to route requests consistently across workers [2].
2. **Mask, don't remove.** Loading tools dynamically, RAG-style, breaks the cache because tool definitions sit near the front of the context. It also confuses the model when earlier steps reference tools that are no longer defined. Manus keeps every tool defined and uses a context-aware state machine to constrain decoding by masking token logits. Under the Hermes function-calling format, response prefill gives three modes: *auto* (the model may call a function or not), *required* (it must call one), and *specified* (it must call one from a subset). Tool names share prefixes such as `browser_` and `shell_`, so prefilling a prefix restricts the choice to one tool group without stateful logit processors [2].
3. **Use the file system as context.** Even 128K-token windows are too small or too costly for real tasks. Observations such as web pages and PDFs are huge, performance degrades past some length, and long inputs stay expensive even when cached. Manus treats the sandbox file system as unlimited, persistent, directly operable memory. Its compression is designed to be *restorable*: a web page's content can be dropped as long as the URL stays, and a document's content as long as its path stays [2]. The post also speculates that state-space models with file-based memory could become successors to Neural Turing Machines [2].
4. **Manipulate attention through recitation.** A typical Manus task takes about 50 tool calls. The agent creates a `todo.md` and rewrites it as it works, checking items off. This pushes the global plan into the end of the context, the model's most recent attention span, and counters goal drift and lost-in-the-middle effects [2].
5. **Keep the wrong stuff in.** Failed actions, error messages, and stack traces stay in context, so the model updates its beliefs and is less likely to repeat the mistake. The post calls error recovery "one of the clearest indicators of true agentic behavior" [2].
6. **Don't get few-shotted.** A context full of similar action-observation pairs makes the model imitate the pattern. The example is reviewing 20 resumes, where the agent falls into a rhythm and over-generalizes. Manus adds structured variation in serialization templates, phrasing, and ordering to break the pattern [2].

### Later refinements (2025 webinar)

In an October 2025 discussion with LangChain, summarized by Lance Martin [3], Manus refined its approach in three ways.

- **Compaction versus summarization.** Compaction is reversible: content that can be rebuilt from the environment, such as file contents, is swapped for a reference such as a path. Summarization is irreversible and is applied only when compaction stops freeing enough space. Older tool results are compacted first, and recent ones stay in full.
- **Context isolation.** Manus uses two modes. For simple, well-defined subtasks the sub-agent gets only instructions ("share memory by communicating"). For complex subtasks it gets the full context.
- **A layered action space.** A small set of atomic function calls sits on top. Broader capability comes from shell utilities in the sandbox and from code calling packages and APIs. This keeps the tool-definition prefix small and stable.

### Lessons

- Treat cost as a property of context *shape*. An append-only, deterministic, prefix-stable context is a cache design.
- Prefer reversible reductions (pointers into a file system) to lossy ones (summaries).
- Use the end of the context as a steering channel: recitation, to-do lists, and current objectives belong at the tail.

---

## 3. KV-cache-friendly design

### Mechanics

Prefix caching stores the attention key/value tensors computed for a prompt prefix, so a later request with the same prefix skips that prefill. In vLLM's design, each KV block is hashed from the parent block's hash, the block's tokens, and extra keys such as LoRA IDs, multimodal input hashes, and an optional per-tenant `cache_salt`. Only full blocks are cached. Eviction is LRU through a free-block queue, and blocks hit by a running request are removed from the queue so they cannot be evicted mid-request [33]. Because each hash chains to its parent, **one changed token invalidates everything after it**. This single fact explains most "cache-friendly" rules.

Hosted APIs expose the same mechanics with pricing. In Anthropic's implementation [8]:

| Aspect | Behavior |
|---|---|
| Prefix order | `tools` -> `system` -> `messages` |
| Breakpoints | Up to 4 explicit `cache_control` markers, or one automatic marker that advances with the conversation |
| TTL | 5 minutes by default (refreshed on read), or 1 hour at extra cost |
| Write price | 1.25x base input (5 min) or 2x (1 h) |
| Read price | 0.1x base input on most models (0.05x and 0.025x on some 2026 models) |
| Lookback | A read walks back up to 20 blocks from a breakpoint to find a prior write |
| Minimum size | 512 to 4,096 tokens depending on model; shorter prompts silently skip caching |
| Invalidators | Tool definition changes (whole cache); `tool_choice`, images (system and messages); thinking config (model-specific) |

The docs repeat the Manus advice in API terms. Put the breakpoint on the last block that stays identical across requests, not on content that changes every request such as timestamps or the new user message. Keep JSON key order stable in `tool_use` blocks. Wait for the first response before sending parallel requests. Pre-warm with `max_tokens: 0` [8]. Google ADK exposes similar knobs for Gemini through `ContextCacheConfig(min_tokens, ttl_seconds, cache_intervals)`, for example a 2,048-token minimum, a 10-minute TTL, and a refresh after 5 uses [32].

### Interactions with context editing

Context reduction and caching are in tension: every edit to the middle of the context throws away the cached suffix. Anthropic's tool-result clearing therefore has a `clear_at_least` parameter, so a clear happens only if it frees enough tokens to justify the cache write. Clearing old thinking blocks invalidates the cache at the point of clearing, while keeping them preserves cache hits [6]. When server-side compaction rewrites history, the docs recommend a breakpoint at the end of the system prompt, so that part stays cached even as the summary changes, and a breakpoint on the compaction block itself [7]. Anthropic's Managed Agents architecture states the principle directly: the harness decides how events are transformed before they enter the window, partly "to improve prompt cache hit rates" [14].

Across sources, the recurring rules are: freeze the prefix and push volatile data to the tail; keep tool sets static per session (masking, prefill, or a search tool instead of swapping definitions [2][18]); serialize deterministically; batch context edits so each cache rebuild pays for itself [6]; and route a session's requests to the same cache [2][8].

---

## 4. Compaction, pruning, and trimming

Compaction replaces old history with something smaller so the loop can continue. Designs differ on five axes: **trigger** (token threshold, turn count, or explicit call), **scope** (whole history, or older turns with a raw tail kept), **method** (LLM summary, deterministic clearing, or reversible offload), **where it runs** (client or server), and **what is re-injected afterward**.

```
 before:  [prefix][t1][t2][t3] ...................... [t40][t41][t42]  ~ trigger
                   \______________ summarize _______/  \__ keep raw __/
 after:   [prefix][SUMMARY block][t40][t41][t42] + re-injected state
                                                (instructions, recent files,
                                                 memory index, invoked skills)
```

### 4.1 Claude Code

Claude Code compacts automatically as the conversation approaches the model's limit. On models with a native 1M window it compacts at about 967K tokens by default; on 200K-window configurations it compacts at the 200K boundary. Users can set the window with `/autocompact`, a setting, or `CLAUDE_CODE_AUTO_COMPACT_WINDOW`, within a 100K to 1M range [11]. Its most instructive feature is **what survives** [10]:

- Startup content (system prompt, CLAUDE.md, memory, MCP tools) lives outside message history and reloads automatically.
- The conversation becomes a structured summary.
- Claude Code re-reads up to five of the most recently modified files the session touched. A file over 5,000 tokens comes back only as a path reference.
- Invoked skill bodies are re-injected but truncated to a per-skill cap, with the oldest dropped once a total budget is exceeded. The skill *listing* is not re-injected.
- Path-scoped rules and nested CLAUDE.md files were loaded into message history, so they are summarized away and reload only when a matching file is touched again.

Anthropic's guide adds that the summary should keep architectural decisions, unresolved bugs, and implementation details, and that compaction prompts should be tuned for recall first, then precision [1].

### 4.2 Claude API: context editing and server-side compaction

Anthropic exposes three server-side mechanisms.

- **Tool-result clearing** (`clear_tool_uses_20250919`, beta header `context-management-2025-06-27`). When input exceeds a trigger (default 100,000 tokens), the oldest tool results are replaced with placeholder text, keeping the most recent N tool uses (default 3). Options include `clear_at_least`, `exclude_tools`, and `clear_tool_inputs`. The response reports `cleared_tool_uses` and `cleared_input_tokens`. The client keeps the full history; only what the model sees is edited [6].
- **Thinking-block clearing** (`clear_thinking_20251015`) keeps the last N turns of thinking or all of it [6].
- **Compaction** (`compact_20260112`). When input reaches a trigger (default 150,000 tokens, minimum 50,000), the server writes a summary and returns a `compaction` block. Later requests drop everything before the last compaction block. `pause_after_compaction` returns control so the client can re-insert recent turns verbatim. `instructions` replaces the default summarization prompt, which asks for a continuity summary of state, next steps, and learnings in `<summary>` tags. Compaction is an extra sampling step reported in `usage.iterations` [7]. A later on-demand variant (beta `compact-2026-09-04`) lets the application decide when to compact. It supports keeping recent turns word for word and compacting in the background while work continues [7].

The launch post reported results on an internal agentic-search evaluation. Context editing alone improved performance by 29% over baseline, and context editing plus the memory tool by 39%. In a 100-turn web-search evaluation, context editing let agents finish workflows that otherwise failed from context exhaustion while cutting token consumption by 84% [4]. Sonnet 4.5 also introduced *context awareness*: the API injects `<budget:token_budget>` into the system prompt and a `Token usage: X/Y; Z remaining` warning after each tool call, so the model can pace itself [4][9].

### 4.3 OpenAI Codex

The open-source Codex CLI shows compaction implemented in a production harness [21]. Local compaction sends the history with a short prompt that frames the job as a handoff:

> "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task." The summary must cover current progress and key decisions, important context, constraints, or user preferences, what remains to be done, and any critical data, examples, or references needed to continue [21].

The new history is (a) re-rendered initial context, (b) the most recent *user* messages up to a 20,000-token budget (`COMPACT_USER_MESSAGE_MAX_TOKENS`), and (c) the summary, prefixed by a note that "another language model started to solve this problem and produced a summary of its thinking process" [21]. Keeping the user's own words verbatim protects the task specification from summarization drift. Codex also has a *remote* path in which the Responses API returns an opaque `Compaction` item with `encrypted_content`, plus pre- and post-compaction hooks [21].

### 4.4 OpenAI Agents SDK

OpenAI's cookbook presents two session-level techniques [22]:

| Dimension | Trimming (last N turns) | Summarizing (older turns -> summary) |
|---|---|---|
| Latency and cost | Lowest; no extra calls | Higher at refresh points |
| Long-range recall | Weak (hard cut-off) | Strong (compact carry-forward) |
| Risk | Context loss | Distortion and context poisoning |
| Observability | Simple logs | Summary prompts and outputs must be logged |
| Best for | Tool-heavy operations, short workflows | Analyst or concierge work, long threads |

Trimming cuts on turn boundaries (a user message plus everything up to the next one), so tool call/result pairs stay together [22]. A 2026 cookbook adds automatic server-side `Compaction()`, a `StaticCompactionPolicy` threshold, and forced checkpoints via `OpenAIResponsesCompactionSession.run_compaction({"force": True})`. It advises compacting at workflow phase boundaries and keeping cited facts in generated artifacts [23].

### 4.5 LangChain Deep Agents

Deep Agents is LangChain's open-source harness "inspired by Claude Code" [24]. Its middleware makes the policies concrete:

- **Summarization** triggers at 85% of the model's context window and keeps the most recent 10%, when the model's profile includes `max_input_tokens`. Otherwise it uses a conservative fixed trigger of 170,000 tokens and keeps 6 messages. Evicted messages are appended as Markdown to `/conversation_history/{session_id}.md`, so the summary is lossy but the raw history stays recoverable from the file system. A `compact_conversation` tool lets the agent or a human trigger compaction on demand [24].
- **Large tool-result eviction.** Tool results over 20,000 tokens, and human messages over 50,000, are written to the backend file system (for example `/large_tool_results/{tool_call_id}`) and replaced with a stub. On a context-overflow error, `read_file` results are head-sliced with a pointer back to the original path, since the file already exists [24].

This is the Manus "restorable compression" idea built into a framework.

### 4.6 Google ADK

ADK's `EventsCompactionConfig` supports two strategies [32]:

- **Token-based** (`token_threshold`, `event_retention_size`): when the session exceeds the threshold, older events are summarized and the last N events stay raw "for pronoun resolution."
- **Sliding window** (`compaction_interval`, `overlap_size`): every N invocations, older events are summarized, with an overlap so consecutive summaries share context.

If both are configured, the token-based strategy takes priority. Compaction runs in the request pipeline through a `CompactionRequestProcessor` [32].

### 4.7 Letta / MemGPT

The MemGPT lineage treats overflow as a *memory pressure* event the agent itself responds to. In Letta's legacy server, summarization triggers when usage exceeds `context_window * 0.9`. Before trimming, the agent receives a system alert: the history "will soon reach its maximum length and be trimmed," so it should save important information with `core_memory_append`, `core_memory_replace`, or `archival_memory_insert`. Tool returns are truncated (for example at 50,000 characters for function returns) [37]. This "flush to memory before compaction" pattern now appears in other harnesses that let the agent write a memory file before history is cleared [5][6].

### 4.8 What the evidence says about summarization

Summarization is not obviously better than simpler masking. JetBrains Research compared LLM summarization with *observation masking*, which replaces old tool observations with placeholders while keeping the agent's reasoning and actions, on SWE-bench Verified with SWE-agent and OpenHands. Both efficiency methods cut cost by roughly 50% without significantly hurting solve rates. LLM summarization did not consistently or significantly beat masking, and a hybrid saved a further 7% over masking alone and 11% over summarization alone [34]. Anthropic likewise describes tool-result clearing as one of the safest and lightest forms of compaction [1]. Summaries add a model call, can inject errors that later turns trust (poisoning), and may drop rare but critical details [22].

### 4.9 Comparison

| System | Trigger (default) | Kept raw | Method | Recoverability | Re-injected after |
|---|---|---|---|---|---|
| Claude Code [10][11] | Near model limit (~967K on 1M models) | None (summary) | LLM structured summary | Transcript on disk | Startup context, 5 recent files, invoked skills |
| Claude API clearing [6] | 100K input tokens | Last 3 tool uses | Placeholder replacement | Client holds full history | n/a |
| Claude API compaction [7] | 150K (min 50K) | Optional, via pause or on-demand | Server LLM summary | Client holds history | Developer's choice |
| OpenAI Codex [21] | Configurable token limit | Recent user msgs up to 20K tokens | Handoff summary, or encrypted remote item | Rollout files | Initial context |
| OpenAI Agents SDK [22][23] | Turn count or threshold | Last N turns | Trim, or server compaction | Session store | Notes re-injected (cookbook) |
| Deep Agents [24] | 85% of window | Last 10% | LLM summary + offload | `/conversation_history/*.md` | Files on demand |
| Google ADK [32] | Token threshold or N invocations | N events / overlap | LLM summary of events | Session events | n/a |
| Letta legacy [37] | 90% of window | Recent messages | Summary after memory-pressure alert | Recall memory (searchable) | Core memory always in context |
| Manus [2][3] | Not public | Recent tool results in full | Reversible compaction first, then summary | Sandbox file system | todo.md recitation |

---

## 5. Scratchpads, notes, and file-system memory

### 5.1 Structured note-taking

Anthropic describes *structured note-taking* as the agent regularly writing notes outside the context window and pulling them back later. A to-do list or a `NOTES.md` file is a minimal version [1]. In the guide's Pokemon example the agent kept exact tallies across thousands of game steps (for example, 1,234 steps of training in which Pikachu gained 8 levels toward a target of 10) and resumed from its own notes after context resets [1]. Manus's `todo.md` serves the same purpose and also steers attention, because it is rewritten at the tail of the context [2].

### 5.2 The Claude memory tool

The memory tool (`{"type": "memory_20250818", "name": "memory"}`) gives Claude a file-system interface to a `/memories` directory. Its commands are `view` (directory listings two levels deep, or numbered file lines with an optional `view_range`), `create`, `str_replace`, `insert`, `delete`, and `rename` [5]. Key design choices:

- **Client-side execution.** Claude only *requests* operations. The application maps `/memories` to real storage (files, a database, encrypted blobs) and returns results, so data residency stays with the developer [5].
- **An injected protocol.** When the tool is present, the API adds instructions to the system prompt: always view the memory directory first, record progress as you work, and "ASSUME INTERRUPTION: Your context window might be reset at any moment" [5].
- **Security is the implementer's job.** Handlers must reject path traversal (`../`, URL-encoded variants), cap file sizes, expire stale files, and strip sensitive data [5].
- **Pairing.** The tool is meant to be used with context editing (save before results are cleared) or compaction ("memory preserves the information that must survive summarization") [5][6].

The docs also define a *multisession software development pattern*. An initializer session writes a progress log, a feature checklist, and a startup-script reference. Every later session reads them first and updates the log before ending. A feature is marked complete only after end-to-end verification [5].

### 5.3 Claude Code: CLAUDE.md and auto memory

Claude Code has two cross-session memory layers [9]:

- **CLAUDE.md files** are human-written instructions at managed-policy, user (`~/.claude/CLAUDE.md`), project, and local (`CLAUDE.local.md`) scope. Files from the root down to the working directory are concatenated, not overridden, at launch. Files in subdirectories load on demand when Claude touches files there. `@path` imports nest up to four hops. The docs recommend under 200 lines per file because "longer files consume more context and reduce adherence." Path-scoped rules in `.claude/rules/` load only when matching files are touched [9].
- **Auto memory** is notes Claude writes for itself, stored per repository at `~/.claude/projects/<project>/memory/`. A `MEMORY.md` index (one line per memory) points to topic files. Only the first 200 lines or 25KB of the index load at session start. When the index nears that limit, Claude Code tells the model to shorten it, merge entries, and move detail into topic files [9].

This is a two-tier, index-plus-pages design: a small always-loaded index, with detail fetched just in time. Anthropic describes the general hybrid: load some data up front (CLAUDE.md) and let the agent explore the rest with tools [1].

### 5.4 Progressive disclosure with Agent Skills

Agent Skills apply the same layering to procedural knowledge. At startup only each skill's `name` and `description` are loaded. The full `SKILL.md` is read when relevant, and bundled files (references, forms, scripts) are read or executed only when needed. Because of this, the amount of context a skill can bundle is "effectively unbounded" [19]. Scripts run without their source or their inputs entering the context [19].

### 5.5 Keeping tools and tool outputs out of context

Tool definitions and intermediate results are a large, often hidden share of context.

- **Tool search.** A five-server MCP setup with 58 tools consumed about 55K tokens before the conversation began, and Anthropic saw tool definitions reach 134K tokens before optimization. Deferring definitions behind a Tool Search Tool cut total context from about 77K to about 8.7K tokens (85% less). MCP-eval accuracy rose from 49% to 74% on Opus 4 and from 79.5% to 88.1% on Opus 4.5 [18]. Claude Code defers MCP schemas by default and lists only tool names [10].
- **Programmatic tool calling.** When Claude orchestrates tools in code, intermediate results stay in the sandbox. Average tokens on complex research tasks fell from 43,588 to 27,297 (37%), and about 200KB of raw expense data was reduced to about 1KB of results in an example [18].
- **Code execution with MCP.** Presenting MCP servers as a file tree of code APIs lets the agent read only the definitions it needs and filter data before returning it. Anthropic's example drops from about 150,000 to about 2,000 tokens (98.7%) [17].
- **Tool-response hygiene.** Claude Code caps tool responses at 25,000 tokens by default. Anthropic recommends pagination, filtering, truncation with steering messages, and a `response_format` enum. In one example a concise response used 72 tokens versus 206 for a detailed one. It also recommends semantic identifiers over UUIDs [20].

### 5.6 Design pattern: the file system as the memory hierarchy

```
 tier            location                    loaded when             written by
 --------------  --------------------------  ----------------------  -------------
 L0 registers    tail of context (todo.md,   every step              agent
                 current objective)
 L1 always-on    system prompt: CLAUDE.md,   session start, after    human / agent
                 MEMORY.md index, skill list compaction (re-inject)
 L2 on demand    topic files, SKILL.md,      tool call (view, read,  agent / human
                 nested rules, tool schemas  tool search)
 L3 cold         offloaded tool results,     explicit read / grep    harness
                 conversation_history/*.md,
                 session event log, git
```

Every production system in this chapter is a variation on this hierarchy. The design decisions are which tier each fact lives in, who promotes or demotes it, and what is re-injected after compaction.

---

## 6. Long-horizon task state

### 6.1 Effective harnesses for multi-session coding (Anthropic, 2025)

**Problem.** Given a high-level prompt such as "build a clone of claude.ai," Claude Opus 4.5 in the Agent SDK fell short even with compaction. It either tried to one-shot the app and ran out of context mid-feature, leaving half-built, undocumented code, or a later session saw partial progress and declared the project finished [12].

**Architecture.** Two prompts share one harness [12]:

```
 session 1: INITIALIZER                   sessions 2..N: CODING AGENT
 +---------------------------+            +------------------------------------+
 | write feature_list.json   |            | pwd; read git log + progress file  |
 |  (200+ features, all      |  files +   | read feature list, pick top unmet  |
 |   "passes": false)        |  git repo  | run init.sh + basic e2e smoke test |
 | write init.sh             |----------->| implement ONE feature              |
 | write claude-progress.txt |            | verify as a user (browser MCP)     |
 | initial git commit        |            | set passes=true; commit; log entry |
 +---------------------------+            +------------------------------------+
```

**Key techniques.** The feature list is JSON rather than Markdown because models were less likely to edit or overwrite JSON inappropriately. Agents may change only the `passes` field. Each session works on one feature, ends with a commit and a progress-log entry, and verifies end-to-end with Puppeteer MCP before marking anything done [12].

**Failure modes and fixes.** Premature victory was fixed with the feature list and one-feature sessions. Undocumented progress was fixed with git and progress notes. Premature completion marks were fixed with mandatory self-verification. Time spent figuring out how to run the app was fixed with `init.sh`. Claude could not see browser-native alert modals, so features using them were buggier [12].

### 6.2 Context resets versus compaction (Anthropic, 2026)

A follow-up on long-running app development separates two continuity mechanisms [13]. A *context reset* clears the window and starts a fresh agent with a structured handoff artifact. *Compaction* summarizes in place, which keeps continuity but gives no clean slate. Some models show **context anxiety**: they wrap up early as they approach what they believe is their limit. With Sonnet 4.5 this was strong enough that resets were essential. Opus 4.5 largely removed the behavior, so the harness dropped resets and ran one continuous session with the Agent SDK's automatic compaction. With Opus 4.6 the sprint construct was also removed [13]. Planner, generator, and evaluator agents communicate through files; one full-harness run building a digital audio workstation took about 3 h 50 min and USD 124.70 [13].

The lesson is that context-management scaffolding is **model-relative**. A mechanism that compensates for one model's weaknesses becomes overhead for the next, consistent with the guide's advice to "do the simplest thing that works" [1].

### 6.3 Parallel agents and a 2,000-session build

Nicholas Carlini's C-compiler project ran 16 parallel Claude agents over nearly 2,000 Claude Code sessions in about two weeks, for about USD 20,000, producing a roughly 100,000-line Rust compiler that builds Linux 6.9 on x86, ARM, and RISC-V [15]. Its context lessons were [15]:

- Fresh containers need written memory: extensive READMEs and progress files, updated often.
- **Design test output for the context window.** Print a few lines, send detail to log files, put `ERROR` and the reason on one line so grep finds it, and precompute aggregate statistics.
- **Compensate for time blindness.** The model cannot judge elapsed time, so the harness offers a `--fast` mode that runs a deterministic 1% or 10% sample per agent.
- Keep a running document of failed approaches, so dead ends are not retried.
- Coordinate through lock files in `current_tasks/`, synchronized by git, with no orchestrator.

### 6.4 The session log as durable external context

Anthropic's Managed Agents architecture (April 2026) separates the "brain" (model plus harness) from the "hands" (sandboxes, tools) and from the **session**, an append-only durable event log that lives outside the context window [14]. The harness is stateless: after a crash a new instance calls `wake(sessionId)` and resumes from the last event. The brain reads positional slices with `getEvents()` to resume, rewind, or reread context before an action, so context removed by irreversible compaction stays recoverable. The interfaces guarantee only durability and queryability, deliberately leaving context strategy to the harness. Provisioning containers lazily cut p50 time-to-first-token by about 60% and p95 by over 90% [14].

```
 +-----------+  emitEvent   +----------------------+  getEvents(slice)  +-----------+
 | sandbox / |------------->| SESSION LOG          |------------------->| HARNESS   |
 | tools     |              | (durable, append-    |                    | transform |
 | (hands)   |<-------------| only, outside window)|                    | -> window |
 +-----------+  tool calls  +----------------------+                    +-----+-----+
                                                                              |
                                                                       model (brain)
```

This is the server-side version of the Deep Agents and Manus pattern: **lossy context over lossless storage**.

---

## 7. Isolation: sub-agents as a context-management device

Chapter 04 covers multi-agent topologies. From a context perspective the relevant facts are:

- In Anthropic's research system each subagent may use tens of thousands of tokens but returns a condensed summary, often 1,000 to 2,000 tokens [1]. The lead agent saves its plan to memory because context beyond 200,000 tokens is truncated. Subagents write outputs to the file system and pass back lightweight references. On BrowseComp, token usage alone explained 80% of performance variance, and the multi-agent system beat single-agent Opus 4 by 90.2% on an internal research eval [16].
- Cognition's counter-argument is that fragmenting context across agents causes conflicting implicit decisions. Its principles are "share context, and share full agent traces, not just individual messages" and "actions carry implicit decisions." For long tasks it suggests a single-threaded agent with a model dedicated to compressing history [43].
- Manus reconciles the two: pass only instructions for well-scoped subtasks, and the full context when the subtask depends on accumulated decisions [3].

Isolation is thus a form of compression: a sub-agent is a summarizer with tools.

---

## 8. Long-term memory systems

Long-term memory persists across sessions and users rather than within one task. Designs fall into four families: OS-style tiered memory, retrieval over memory streams, extracted fact stores and knowledge graphs, and evolving playbooks.

### 8.1 MemGPT and Letta: OS-style virtual context

MemGPT ("Towards LLMs as Operating Systems") treats the context window like RAM and external stores like disk, with the LLM paging data through function calls [35]. Letta's open-source implementation describes three tiers in the agent's own system prompt [37]:

```
 +----------------------------- context window --------------------------------+
 | system instructions | CORE MEMORY blocks (persona, human, ...)  | FIFO msgs |
 |                     | always visible, size-limited, self-edited |           |
 +---------------------+-------------------------------------------+-----------+
            core_memory_append/replace |                        | evicted
            memory_rethink             v                        v
               +-----------------------------+   +-------------------------+
               | ARCHIVAL MEMORY             |   | RECALL MEMORY           |
               | unbounded, vector-searched  |   | full message history,   |
               | archival_memory_insert/     |   | conversation_search     |
               | archival_memory_search      |   |                         |
               +-----------------------------+   +-------------------------+
```

- **Core memory** sits in the system prompt, is always visible, and is edited by the agent itself. In the legacy server the persona and human blocks are capped at 20,000 characters each by default [37].
- **Recall memory** is the searchable conversation history. **Archival memory** is described to the agent as "infinite size" storage it must search explicitly [37].
- **Memory pressure.** At 90% of the window, the agent is warned to save important information before the history is trimmed (Section 4.7) [37].
- **Sleep-time agents.** A background agent ("Letta-Sleeptime-Memory") reorganizes the primary agent's memory blocks with narrow edits or whole-block `rethink` rewrites. Its instructions require absolute dates instead of "today" because memories persist indefinitely, and tell it to be selective while aiming for high recall [37]. The research behind this, "Sleep-time Compute," proposes doing inference on context offline, before queries arrive, to cut test-time compute [39].
- **Letta Code (2026)** tracks all context, including memory blocks, in git ("MemFS"), supports agent-learned skills, and runs sleep-time compute, now called "dreaming," on a schedule or manually with `/dream` and `/reflect` [38].

### 8.2 Generative Agents: the memory stream

Park et al.'s simulated town (2023) introduced an architecture that is still the reference for retrieval-based agent memory. A *memory stream* records observations in natural language. Retrieval scores memories by recency, importance, and relevance. Periodic *reflection* synthesizes higher-level inferences that are stored back into the stream, and plans are generated from retrieved memories [41]. Most later systems keep this split between a raw log, a retrieval scorer, and consolidation into higher-level memories.

### 8.3 Mem0: extracted, ranked facts

Mem0 is a memory layer for user, session, and agent state ("Building Production-Ready AI Agents with Scalable Long-Term Memory") [40]. Its April 2026 pipeline uses single-pass, ADD-only extraction with one LLM call, so memories accumulate rather than being overwritten. It adds entity extraction and linking, parallel semantic, BM25, and entity retrieval fused into one ranking, and time-aware retrieval. The README reports, for the managed platform, LoCoMo 92.5 (up from 71.4) at about 7.0K tokens per query and 0.88 s p50 latency, LongMemEval 94.4 (up from 67.8) at about 6.8K tokens, and BEAM scores of 64.1 at 1M and 48.6 at 10M tokens. It notes that the open-source SDK lacks some platform optimizations [40]. These are vendor-reported numbers.

### 8.4 Zep and Graphiti: temporal knowledge graphs

Zep ("A Temporal Knowledge Graph Architecture for Agent Memory") builds memory as a graph of entities, facts, and episodes [42]. Graphiti, its open-source core, uses **bi-temporal** tracking. Facts carry validity windows, and superseded facts are marked invalid rather than deleted, so historical queries still work. Each derived fact traces back to the episodes (raw ingested data) that produced it. Updates are incremental, and retrieval combines embeddings, BM25, and graph traversal without LLM summarization at query time [42]. This design targets the *knowledge update* and *temporal reasoning* failures that flat vector stores handle poorly.

### 8.5 A-MEM: self-organizing notes

A-MEM (2025) applies Zettelkasten principles. Each new memory becomes a note with structured attributes, a contextual description, and tags. The system links it to related notes, and existing notes' tags and context *evolve* as new memories arrive. The authors report outperforming baselines across six foundation models [44].

### 8.6 ACE: contexts as evolving playbooks

Agentic Context Engineering (ACE, 2025) treats the system prompt and memory as a *playbook* refined by three roles. A Generator produces trajectories, a Reflector extracts lessons, and a Curator merges them as **incremental delta updates**: itemized entries with helpful/harmful counters, deterministic de-duplication, and pruning [36]. The design responds to two failures of iterative prompt rewriting: *brevity bias* and **context collapse**, where repeated whole-context rewrites erode detail over time. Reported gains are +10.6% on AppWorld agent tasks and +8.6% on finance benchmarks. Against baselines it reports 82.3% lower latency and 75.1% fewer rollouts (versus GEPA, offline) and 91.5% lower latency and 83.6% lower token cost (versus Dynamic Cheatsheet, online) [36].

### 8.7 State-based versus retrieval-based personalization

OpenAI's personalization cookbook argues for **state-based** memory where continuity matters. A structured profile plus notes is injected into the system prompt, a `save_memory_note` tool captures candidates during a run, session notes are re-injected after trimming, and an asynchronous job consolidates them into global memory with the precedence *latest user input, then session overrides, then global defaults* [25]. A companion notebook separates the two mechanisms: compaction carries one run's working state, memory carries reusable workflow lessons, and case conclusions belong in reviewed artifacts [23].

### 8.8 Measuring memory: LongMemEval

LongMemEval (ICLR 2025) tests five long-term memory abilities: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. It has 500 questions over histories of about 115k tokens (about 40 sessions) in the S variant and about 500 sessions in the M variant. Its design studies include index expansion with summaries, keyphrases, and user facts, and time-aware query expansion [45]. Memory vendors now report against it [40].

### 8.9 Comparison of memory designs

| Design | Unit of memory | Write path | Read path | Update semantics | Main strength | Main risk |
|---|---|---|---|---|---|---|
| Claude memory tool [5] | Files in `/memories` | Agent tool calls | `view` on demand | Agent edits/overwrites | Transparent, developer-hosted | Clutter; traversal security |
| Claude Code auto memory [9] | Index + topic files | Agent, from corrections | Index at start; topics on demand | Agent rewrites index | Bounded always-on cost | Index drift |
| MemGPT / Letta [35][37] | Core blocks + archival passages | Self-editing tools; sleep-time agent | Core always; search for rest | Replace / rethink | Explicit memory tiers | Agent forgets to save |
| Generative Agents [41] | Observations + reflections | Every observation | Recency x importance x relevance | Append + reflect | Consolidation | Retrieval noise |
| Mem0 [40] | Extracted facts + entities | One LLM extraction pass | Hybrid semantic/BM25/entity | ADD-only accumulation | Low tokens per query | Stale or conflicting facts |
| Zep / Graphiti [42] | Temporal graph edges | Incremental episode ingest | Embedding + BM25 + traversal | Bi-temporal invalidation | Knowledge updates, time | Extraction errors |
| A-MEM [44] | Linked notes | Note + link generation | Similarity + links | Notes evolve | Self-organizing | Uncontrolled rewriting |
| ACE [36] | Playbook bullets with counters | Reflector -> curator deltas | Whole playbook in context | Delta merge + prune | No collapse | Playbook growth |
| OpenAI state-based [25] | Profile + notes | Tool during run; async consolidation | Injected at start | Precedence rules | Coherent current state | Consolidation errors |

---

## 9. Failure modes

| Failure | Mechanism | Mitigations seen in sources |
|---|---|---|
| Context rot | Accuracy degrades as tokens grow, unevenly by model and task [1][30] | Compaction, clearing, sub-agents, minimal tool sets |
| Cache collapse | Volatile prefix (timestamps), tool swaps, non-deterministic JSON [2][8] | Frozen prefix, masking, deterministic serialization |
| Summary drift / poisoning | Summarizer drops or invents facts that later turns trust [22][28] | Keep user messages verbatim [21]; reversible offload [2][24]; recall-first prompt tuning [1] |
| Premature victory | New session sees partial work, declares done [12] | JSON feature list, verified `passes` flags |
| Context anxiety | Model wraps up early near its believed limit [13] | Context resets with handoff (older models); compaction plus budget signals [4][9][13] |
| Goal drift in long loops | Plan buried mid-context [2] | `todo.md` recitation at the tail [2] |
| Few-shot ruts | Repetitive action-observation pairs [2] | Structured variation in serialization [2] |
| Lost instructions after compaction | Rules loaded mid-history get summarized away [10] | Put persistent rules in root CLAUDE.md; re-inject on compaction [9][10] |
| Memory clutter / staleness | Unbounded notes, relative dates, overwritten facts [5][37][42] | Size caps, expiry, absolute dates, bi-temporal invalidation |
| Context collapse in self-improvement | Whole-context rewrites erode detail [36] | Incremental delta updates [36] |
| Tool-definition bloat | Tens of thousands of tokens of schemas [18] | Tool search, code-mode APIs, skills [17][18][19] |
| Time blindness | Agent cannot sense elapsed time; long test runs [15] | Sampled `--fast` tests, terse output, logs to files [15] |

---

## Key takeaways for system designers

1. **Budget context like memory, not like a prompt.** Every token competes for an attention budget [1]. Instrument per-step token composition (tools, instructions, history, observations), as Claude Code's context view does [10].
2. **Make the prefix a cache artifact.** Order content from stable to volatile, keep it append-only, serialize deterministically, and keep tool sets static per session. At input-to-output ratios near 100:1, hit rate is the cost model [2][8].
3. **Prefer reversible reductions.** Offload large observations to files and keep pointers (paths, URLs, event offsets). Summarize only when that runs out [2][3][24]. Keep a lossless log underneath the lossy window [14].
4. **Start with the cheapest compaction.** Observation masking and tool-result clearing match LLM summarization on coding agents at similar savings [34]. When you do summarize, frame it as a handoff, keep the user's own words verbatim, and keep a raw tail [7][21][32].
5. **Decide what is re-injected after compaction.** Instructions, recently touched files, invoked skills, and the plan should survive by construction, not by luck [10].
6. **Put the plan at the tail.** Recite objectives (`todo.md`, progress logs) so they stay in recent attention [2][12].
7. **Externalize long-horizon state into structured, verifiable artifacts.** Use JSON feature lists with pass flags, progress files, git history, and init scripts so any fresh session can rebuild state in a few tool calls [12][15].
8. **Use a two-tier memory: small always-on index, large on-demand pages.** Examples are MEMORY.md with topic files, skill metadata with SKILL.md, and core blocks with archival storage [9][19][37].
9. **Model knowledge updates and time explicitly** in long-term memory: absolute dates, validity intervals, and conflict precedence [25][37][42].
10. **Update memory by deltas, not rewrites,** to avoid context collapse, and consolidate offline (sleep-time) when latency matters [36][37][39].
11. **Treat scaffolding as model-relative.** Context resets, sprints, and budget warnings that a 2025 model needed became overhead for its successors. Re-evaluate harness complexity with each model generation [1][13].
12. **Secure the memory surface.** Memory tools are file-system tools: validate paths, cap sizes, expire stale entries, and keep secrets out of context [5][17].

---

## References

1. Rajasekaran, P., Dixon, E., Ryan, C., Hadfield, J., et al. "Effective context engineering for AI agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
2. Ji, Y. ("Peak"). "Context Engineering for AI Agents: Lessons from Building Manus." Manus blog, 2025. https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
3. Martin, L. "Context Engineering in Manus" (summary of the LangChain x Manus webinar). 2025. https://rlancemartin.github.io/2025/10/15/manus/
4. Anthropic. "Managing context on the Claude Developer Platform" (context editing and memory tool launch). Claude blog, 2025. https://claude.com/blog/context-management
5. Anthropic. "Memory tool." Claude Platform documentation, 2025-2026. https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
6. Anthropic. "Context editing." Claude Platform documentation, 2025-2026. https://platform.claude.com/docs/en/build-with-claude/context-editing
7. Anthropic. "Compaction overview" and "Compaction at a token threshold." Claude Platform documentation, 2026. https://platform.claude.com/docs/en/build-with-claude/compaction and https://platform.claude.com/docs/en/build-with-claude/compaction-threshold
8. Anthropic. "Prompt caching." Claude Platform documentation, 2024-2026. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
9. Anthropic. "How Claude remembers your project" (CLAUDE.md and auto memory). Claude Code documentation, 2026. https://code.claude.com/docs/en/memory
10. Anthropic. "Explore the context window." Claude Code documentation, 2026. https://code.claude.com/docs/en/context-window
11. Anthropic. "Model configuration" (default auto-compact thresholds). Claude Code documentation, 2026. https://code.claude.com/docs/en/model-config
12. Young, J., et al. "Effective harnesses for long-running agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
13. Rajasekaran, P. "Harness design for long-running application development." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/harness-design-long-running-apps
14. Martin, L., Cemaj, G., Cohen, M. "Scaling Managed Agents: Decoupling the brain from the hands." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/managed-agents
15. Carlini, N. "Building a C compiler with a team of parallel Claudes." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/building-c-compiler
16. Hadfield, J., Zhang, B., Lien, K., Scholz, F., Fox, J., Ford, D. "How we built our multi-agent research system." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
17. Jones, A., Kelly, C. "Code execution with MCP: Building more efficient agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/code-execution-with-mcp
18. Wu, B., et al. "Introducing advanced tool use on the Claude Developer Platform." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/advanced-tool-use
19. Zhang, B., Lazuka, K., Murag, M. "Equipping agents for the real world with Agent Skills." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
20. Aizawa, K., et al. "Writing effective tools for agents - with agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
21. OpenAI. "Codex" GitHub repository: compaction prompt (codex-rs/prompts/templates/compact/prompt.md), summary prefix, and codex-rs/core/src/compact.rs. 2025-2026. https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md
22. OpenAI. "Context Engineering - Short-Term Memory Management with Sessions." OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/session_memory.ipynb
23. OpenAI. "Building Reliable Agents with Memory and Compaction." OpenAI Cookbook, 2026. https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/building_reliable_agents_memory_compaction.ipynb
24. LangChain. "Deep Agents" GitHub repository (README; libs/deepagents/deepagents/middleware/summarization.py and filesystem.py). 2025-2026. https://github.com/langchain-ai/deepagents
25. OpenAI. "Context Engineering for Personalization - State Management with Long-Term Memory Notes." OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/context_personalization.ipynb
26. Martin, L. "Context Engineering for Agents." LangChain Blog, 2025. https://blog.langchain.com/context-engineering-for-agents/
27. LangChain. "context_engineering" GitHub repository (write, select, compress, isolate notebooks). 2025. https://github.com/langchain-ai/context_engineering
28. LangChain. "how_to_fix_your_context" GitHub repository (implements Drew Breunig's failure modes and fixes). 2025. https://github.com/langchain-ai/how_to_fix_your_context
29. Chroma. "context-rot" GitHub repository (code for the Context Rot report). 2025. https://github.com/chroma-core/context-rot
30. Hong, K., Troynikov, A., Huber, J. "Context Rot: How Increasing Input Tokens Impacts LLM Performance." Chroma Research, 2025. https://research.trychroma.com/context-rot
31. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. "Lost in the Middle: How Language Models Use Long Contexts." arXiv 2307.03172, 2023. https://arxiv.org/abs/2307.03172
32. Google. "Compress agent context for performance" and "Context caching with Gemini." Agent Development Kit documentation, 2025-2026. https://google.github.io/adk-docs/context/compaction/ and https://google.github.io/adk-docs/context/caching/
33. vLLM Project. "Automatic Prefix Caching" design document. 2024-2026. https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md
34. Lindenbauer, T., Slinko, I., Felder, L., Bogomolov, E., Zharov, Y. "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management." arXiv 2508.21433, 2025 (NeurIPS 2025 DL4Code workshop). https://arxiv.org/abs/2508.21433 (code: https://github.com/JetBrains-Research/the-complexity-trap)
35. Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., Gonzalez, J. E. "MemGPT: Towards LLMs as Operating Systems." arXiv 2310.08560, 2023. https://arxiv.org/abs/2310.08560
36. Zhang, Q., Hu, C., Upasani, S., et al., Zou, J., Olukotun, K. "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models." arXiv 2510.04618, 2025. https://arxiv.org/abs/2510.04618 (code: https://github.com/ace-agent/ace)
37. Letta. "Letta (formerly MemGPT)" legacy server source, archive branch (letta/constants.py, letta/functions/function_sets/base.py, letta/prompts/system_prompts/memgpt_chat.py and sleeptime_v2.py). 2023-2026. https://github.com/letta-ai/letta/tree/archive
38. Letta. "Letta Code" GitHub repository. 2026. https://github.com/letta-ai/letta-code
39. Lin, K., Snell, C., Wang, Y., Packer, C., Wooders, S., Stoica, I., Gonzalez, J. E. "Sleep-time Compute: Beyond Inference Scaling at Test-time." arXiv 2504.13171, 2025. https://arxiv.org/abs/2504.13171 (code: https://github.com/letta-ai/sleep-time-compute)
40. Chhikara, P., Khant, D., Aryan, S., Singh, T., Yadav, D. "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." arXiv 2504.19413, 2025; and Mem0 GitHub README (April 2026 algorithm update). https://arxiv.org/abs/2504.19413 and https://github.com/mem0ai/mem0
41. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., Bernstein, M. S. "Generative Agents: Interactive Simulacra of Human Behavior." arXiv 2304.03442, 2023 (UIST 2023). https://arxiv.org/abs/2304.03442
42. Rasmussen, P., et al. "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv 2501.13956, 2025; and Graphiti GitHub repository. https://arxiv.org/abs/2501.13956 and https://github.com/getzep/graphiti
43. Yan, W. "Don't Build Multi-Agents." Cognition, 2025. https://cognition.ai/blog/dont-build-multi-agents
44. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., Zhang, Y. "A-MEM: Agentic Memory for LLM Agents." arXiv 2502.12110, 2025. https://arxiv.org/abs/2502.12110 (code: https://github.com/agiresearch/A-mem)
45. Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., Yu, D. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." arXiv 2410.10813, 2024 (ICLR 2025). https://arxiv.org/abs/2410.10813
