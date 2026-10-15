# 26. Tool Retrieval: RAG over APIs, Function Catalogs and MCP Servers

Most of this atlas is about retrieving *passages* so a model can answer a question. This chapter covers a different corpus: *capabilities*. The model needs a function name, a JSON Schema and a description it can act on, not a fact to quote. When an agent can reach a handful of tools, all of their definitions fit in the prompt. When it can reach the 16,000+ REST endpoints in ToolBench [3], the thousands of servers and tools in public MCP registries [9][23], or the up to 10,000 deferred tools a single Claude API request accepts [6], the tool catalog becomes a retrieval problem in its own right.

Three pressures make tool retrieval its own discipline rather than plain RAG with a different corpus:

1. **Context cost.** Tool definitions are verbose, structured and loaded *before* the user's first token. Anthropic reports that a typical five-server MCP setup (GitHub, Slack, Sentry, Grafana, Splunk; 58 tools) uses about 55K tokens before the conversation starts, and that it has seen definitions reach 134K tokens before optimization [5][6].
2. **Selection accuracy.** Anthropic's documentation says Claude's ability to pick the right tool "degrades once you exceed 30-50 available tools" [6]. Its internal MCP evaluations with large tool libraries rose from 49% to 74% (Claude Opus 4) and from 79.5% to 88.1% (Claude Opus 4.5) once tool search replaced loading everything upfront [5].
3. **Retrieval mismatch.** A user query ("book me something cheap near the venue for Friday") rarely shares vocabulary with the tool that fulfils it (`hotels_search_v2(lat, lng, checkin, max_price)`). The ToolRet benchmark's title states the result bluntly: "Retrieval Models Aren't Tool-Savvy" [7].

The chapter follows the field's history. Section 1 covers Gorilla and retriever-aware training. Section 2 covers ToolLLM's DFSDT, its neural API retriever and AnyTool's hierarchical follow-up. Section 3 covers RAG-MCP and the measured cost of prompt bloat. Section 4 covers MCP-era discovery systems (MCP-Zero, LangGraph BigTool, LiveMCPBench's copilot) and the official MCP Registry as the catalog layer. Section 5 covers Anthropic's server-side tool search, deferred loading and code-based progressive disclosure. Section 6 covers benchmarks (ToolRet, COLT/ToolLens, StableToolBench, MCP-Bench, MCP-Universe). Section 7 covers generative tool retrieval (ToolGen). Section 8 covers how to embed tool descriptions, including synthetic-query (Re-Invoke) and usage-driven (Tool2Vec) representations. A comparison table and design takeaways close the chapter.

A note on sourcing: the research and fact-check passes for this chapter could not reach arXiv, Hugging Face, the ACL Anthology or other paper hosts directly. Numbers below were checked against the authors' GitHub READMEs and against vendor documentation and engineering posts (re-read in October 2026). Where a figure comes only from a paper abstract that could not be re-read, it is labelled *unverified here*.

---

## 1. Gorilla and APIBench: teaching a model to read retrieved API docs

### Problem

In 2023, frontier LLMs often hallucinated API calls: wrong argument names, deprecated model identifiers, functions that did not exist. Patil, Zhang, Wang and Gonzalez (UC Berkeley) asked whether a smaller fine-tuned model, connected to a retriever over API documentation, could write correct calls against a large, changing set of ML-model APIs [1][2].

### Architecture

```
            +------------------------------+
 user  ---> | instruction ("classify       |
 query      |  bird species in an image")  |
            +---------------+--------------+
                            |
                  (optional) retriever
                  BM25 / embedding / oracle
                            |
                            v
            +------------------------------+
            | top-1 API doc from APIBench  |   TorchHub / TensorFlow Hub /
            | (name, domain, args, example)|   HuggingFace model cards
            +---------------+--------------+
                            |
                            v
            +------------------------------+
            | Gorilla (fine-tuned LLaMA)   |
            | prompt = query +             |
            | "Use this API documentation  |
            |  for reference: <doc>"       |
            +---------------+--------------+
                            |
                            v
                  API call + explanation
                            |
                            v
             AST sub-tree match vs. APIBench
             (correct / wrong / hallucinated)
```

**APIBench** is the dataset. It holds instruction-API pairs built from model hubs, with separate train and evaluation JSONL files per hub in the Gorilla repository [1]. The repository describes the paper's evaluation as covering "1,600+ APIs" [1].

**Retriever-aware training** is the key idea. During fine-tuning, the instruction is paired with a retrieved (or oracle) API document, so the model learns to *condition on* documentation instead of only memorizing it. The repository calls this "retrieval-augmented training for test-time adaptation" [1]. If an API changes, you update the document store, not the weights.

**AST-based evaluation.** Generated calls are parsed and matched as sub-trees against the reference API, so "hallucination" can be measured as a call to an API that does not exist in the corpus.

### What followed

The Gorilla project grew into an ecosystem [1]:

- **OpenFunctions v2**: an OpenAI-compatible function-calling model with parallel and multiple calls and *relevance detection*, meaning it can decline to call when no supplied function fits.
- **Berkeley Function Calling Leaderboard (BFCL)**: V1 is single-turn, V2 uses enterprise-contributed functions, V3 is multi-turn and multi-step, and V4 Agentic (July 2025) adds web search, memory and format sensitivity, with cost and latency reported.
- **GoEx**: an execution runtime with post-facto validation, undo and damage confinement.
- **RAFT**: a domain-RAG fine-tuning recipe that trains with oracle and distractor documents.

The repository reports roughly 500K requests served and about 1,500 APIs supported by the Gorilla CLI [1].

### Failure modes and lessons

- **The retriever is the ceiling.** In retriever-aware setups, a wrong top-1 document can do more harm than no document, because the model has been trained to trust it. Later systems retrieve a top-k list and let the LLM choose.
- **Exact-match evaluation favors short calls.** AST matching suits single-call ML-hub APIs well, but it does not measure multi-step workflows. BFCL V3/V4 and ToolBench were built to fill that gap.
- **Lesson:** treat tool documentation as an index you can hot-swap. This design principle survives in every later system in this chapter.

---

## 2. ToolLLM / ToolBench: a neural API retriever plus DFSDT

### Problem

Gorilla's APIs were ML-model loaders. Real agents need thousands of heterogeneous REST APIs, often several per task. ToolLLM (Qin et al., Tsinghua/OpenBMB; ICLR 2024 spotlight) built a large instruction-tuning corpus over real APIs and asked an open model (LLaMA) to plan across them [3][4].

### Scale

Per the ToolBench README [3]:

| Item | Count |
|---|---|
| Tools (RapidAPI services) | 3,451 |
| APIs (endpoints) | 16,464 |
| Categories | 49 |
| Instances | 126,486 |
| Real API calls made during construction | 469,585 |

### Architecture

```
  instruction (I1 single-tool / I2 intra-category / I3 intra-collection)
        |
        v
  +---------------------+     16k API docs (name, desc, params) embedded
  |  API retriever      |<----  with a fine-tuned BERT bi-encoder
  |  (Sentence-BERT     |
  |   bert-base-uncased)|
  +----------+----------+
             | top-k candidate APIs
             v
  +-------------------------------------------+
  |  ToolLLaMA agent + DFSDT search           |
  |                                           |
  |   root --> thought/action --> obs         |
  |     |            \                        |
  |     |             +--> fail/give-up ----+ |
  |     |                                   | |
  |     +--> backtrack, try sibling node <--+ |
  |                                           |
  |   stop on "Finish" with answer            |
  +-------------------+-----------------------+
                      v
           ToolEval (pass rate, win rate)
```

**API retriever.** The README gives the training setup: `bert-base-uncased`, 5 epochs, batch size 32, learning rate 2e-5, 500 warm-up steps, max sequence length 256. The trained retriever was released so ToolLLaMA could run in an *open-domain* setting, where it is not told which APIs are relevant [3]. Training pairs come for free from data construction: each synthesized instruction was generated from a known set of APIs, which become its positives.

**DFSDT (depth-first search-based decision tree)** is the planner, described as a way to "bolster the planning and reasoning ability of LLMs" [3]. Unlike ReAct's single chain, the agent can abandon a failing branch (an API error, an empty result, a dead end) and try a sibling action. DFSDT was used both to *annotate* training data (it solved instructions that CoT and ReAct could not) and at inference.

**ToolEval** is an AlpacaEval-style automatic judge. It reports pass rate and win rate, and its ChatGPT evaluator agrees with humans 87.1% (pass) and 80.3% (win) of the time [3].

### Results (DFSDT, averaged; README [3])

| Model | Pass rate | Win rate vs ChatGPT-ReAct |
|---|---|---|
| ChatGPT | 64.8 | 64.3 |
| ToolLLaMA (oracle APIs) | 66.7 | 60.0 |
| ToolLLaMA + retriever | 67.3 | 63.1 |
| GPT-4 | 71.1 | 70.4 |

The retriever variant matches or slightly beats the oracle-API variant. This is a notable result: a retrieved top-k set can be *better* than the ground-truth set, because it surfaces substitute APIs when the annotated one fails.

### Follow-up: AnyTool and hierarchical retrieval over the same APIs

AnyTool (Du, Wei, Zhang, 2024, "Self-Reflective, Hierarchical Agents for Large-Scale API Calls") keeps ToolBench's 16K-API pool but replaces the flat bi-encoder with a GPT-4-driven hierarchy that follows RapidAPI's own structure: an agent narrows to categories, then to tools, then to APIs, and a self-reflection loop re-opens the search when the solver fails [22]. Its README reports an average pass rate of 58.2% on a filtered ToolBench subset (queries the authors judged solvable), against 31.8% for GPT-4 with DFSDT and ToolLLM's retriever, and 73.8% versus 36.6% on the authors' own AnyToolBench [22]. Two caveats matter: the filtering and the in-house benchmark make the comparison less than like-for-like, and the hierarchy multiplies LLM calls. The design idea, though, recurs in the MCP era as server-then-tool routing (Section 4).

### Failure modes and lessons

- **Benchmark instability.** Live RapidAPI endpoints go down, change, or rate-limit, so scores drift. StableToolBench (Guo et al.) responded with a virtual API server: a cache of real responses plus LLM simulators for unavailable APIs, a "MirrorAPI" model trained to mimic 7K+ APIs, and solvable-only metrics (SoPR/SoWR) whose solvability is fixed in advance by majority vote of three LLMs (gpt-4-turbo, gemini-pro and claude-2) [12]. Any tool-retrieval evaluation that executes live calls needs this kind of stabilization.
- **Search cost.** DFSDT multiplies API calls and LLM turns. Backtracking is valuable when calls are cheap and idempotent, and dangerous when they are side-effecting (see GoEx's undo semantics [1]).
- **Lesson:** tool retrieval and tool planning interact. A retriever with good recall and a planner that can backtrack compensates for imperfect ranking.

---

## 3. RAG-MCP: measuring prompt bloat

### Problem

When MCP made it trivial to attach many servers, the naive pattern became "list every tool from every server in the system prompt." RAG-MCP (Gan and Sun, 2025, "Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation") quantified what this costs and proposed the obvious fix [8].

### Architecture

```
 offline:  MCP registry --> for each server/tool: description text
                             --> embed --> vector index

 online:
   user task ----> query encoder ----> top-k MCP descriptions
                                        |
                                        v
                         (optional) validate: sample a test call
                                        |
                                        v
              LLM prompt = task + ONLY the k selected tool schemas
                                        |
                                        v
                                 tool call / answer
```

The design is ordinary RAG with tool descriptions as documents. The paper's contribution is the **MCP stress test**, a needle-in-a-haystack setup. One target MCP is placed among N distractor MCPs, N is swept upward, and the test records whether the model still selects the target. Selection accuracy drops as N grows, and the drop is sharper when every candidate is in the prompt than when a retriever pre-filters.

Headline numbers, from the paper's abstract (*unverified here*): about 43% tool-selection accuracy with RAG-MCP versus about 14% for the all-tools baseline on the benchmark task, and prompt tokens cut by more than half [8]. Treat these as indicative. They depend on the model, the distractor pool and the task.

### Failure modes

- **Retrieval-induced blind spots.** If the right server is not in the top-k, the model cannot recover. It does not know the tool exists. Every later production design (Section 5) addresses this with a *search tool the model can call again* rather than a one-shot pre-filter.
- **Description quality dominates.** MCP servers ship descriptions written for humans or not written at all. Embedding them as-is works poorly for terse or overlapping tools.
- **Lesson:** separate "what the model can see" from "what the model can reach." The full catalog stays reachable; only a slice is visible.

---

## 4. MCP-era discovery: active requests, hierarchical routing and tool registries

### MCP-Zero: the model asks for tools

MCP-Zero (Fei, Zheng, Feng, 2025) reverses the direction of retrieval. Instead of the framework guessing which tools to inject from the user query, the *agent* writes a structured request describing the tool it needs mid-task, and a matcher resolves it against an index [9]. The released **MCP-tools** dataset has **308 servers and 2,797 tools** filtered from the official MCP server repository. Each server carries a summary, a description and `text-embedding-3-large` embeddings (3,072 dimensions); summaries were extracted with Qwen2.5-72B-Instruct [9]. The paper is an arXiv preprint (2506.01056) and the README reports no accuracy figures. The repository ships a similarity matcher and a target-tool sampler for building evaluation needles [9].

The motivation: the model's own statement of a sub-goal ("I need a tool that converts a PDF page to PNG") is a much better retrieval query than the user's original high-level request. Because the dataset carries server-level summaries and embeddings alongside tool entries, it supports matching servers first and then tools within the chosen server, which keeps the candidate set small at each step. (The paper describes this hierarchical routing; the README alone does not detail it.)

### LangGraph BigTool: tools as entries in a memory store

LangChain's `langgraph-bigtool` treats a large tool registry as a LangGraph long-term memory namespace [10]:

- The tool registry is a dict mapping tool IDs to tool objects.
- Tool descriptions are embedded and indexed in a LangGraph store (in-memory or Postgres) under the namespace `("tools",)`. The quickstart uses OpenAI `text-embedding-3-small` at 1,536 dimensions.
- The agent gets a `retrieve_tools` tool that returns tool IDs. The graph then binds those tools for the next model call.
- `retrieve_tools_function` can be overridden. The README shows a non-semantic variant that returns fixed tool groups keyed by a `Literal` category argument [10].

The quickstart indexes about 50 functions from Python's `math` module. A query for "arc cosine calculation" retrieves `cos` and `acos`, and the agent then calls `acos(0.5)` [10]. The scale is toy, but the pattern is the important part: **retrieval is itself a tool**, so the agent can search again when the first result set is wrong.

### LiveMCPBench's MCP Copilot Agent

LiveMCPBench (Mo et al., 2025, "Can Agents Navigate an Ocean of MCP Tools?") pairs a benchmark with a baseline agent that routes through an embedding index of servers and then tools. Its configuration exposes `TOP_SERVERS=5`, `TOP_TOOLS=3` and a 1,024-dimension embedding model, and servers must be indexed before the agent runs [11]. The two-level server-then-tool fan-out shows up repeatedly in MCP-era work, because servers form natural clusters with shared descriptions and authentication.

```
  query --> [server index] --top 5--> [tool index within those servers] --top 3-->
            LLM sees <= 15 schemas; may call "route" again with a new query
```

### The official MCP Registry: the catalog layer

Retrieval needs a catalog. The Model Context Protocol project's official **MCP Registry** launched in preview on 8 September 2025, describing itself as "like an app store for MCP servers" [23]. Servers are published with an `mcp-publisher` CLI rather than by pull request, with namespace ownership verified at publish time (a GitHub login for `io.github.<user>/...` names, DNS or HTTP verification for domain names). The Registry API entered a v0.1 freeze on 24 October 2025 ahead of a planned GA [23]. For tool retrieval, the registry matters less as a search engine than as a source of normalized, ownership-verified `server.json` metadata: downstream clients and sub-registries can build their own server-level indices (the first hop of the server-then-tool routing above) from it, instead of scraping READMEs as MCP-Zero and LiveMCPBench had to.

---

## 5. Anthropic: server-side tool search, deferred loading and code-based disclosure

Anthropic's approach turns tool retrieval into a first-class API primitive, and it is the most fully documented production design in this space.

### 5.1 Tool search tool and `defer_loading`

Introduced on 24 November 2025 as part of "advanced tool use" [5]. The documentation lists both variants as supported on Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 and later models; Opus 4.1 and earlier do not support it [6]:

```
 request.tools = [ tool_search_tool_{regex|bm25}_20251119,      <- always loaded
                   hot_tool_1 .. hot_tool_k,                     <- defer_loading: false
                   tool_1 .. tool_N (N <= 10,000)                <- defer_loading: true ]

 +-------------------- Anthropic API (server side) ---------------------+
 |  system-prompt prefix = search tool + hot tools   (cacheable)        |
 |                                                                      |
 |  Claude --server_tool_use(pattern|query, limit)--> search over       |
 |          names, descriptions, arg names, arg descriptions            |
 |        <--tool_search_tool_result: [tool_reference x <=5 default]--  |
 |                                                                      |
 |  API expands each tool_reference inline into the full definition     |
 |  (prefix untouched -> prompt cache preserved)                        |
 |                                                                      |
 |  Claude --tool_use(discovered_tool)--> client executes, returns      |
 +----------------------------------------------------------------------+
```

Key details from the documentation [6]:

- **Two built-in retrievers.** `tool_search_tool_regex_20251119`: Claude writes case-insensitive Python `re.search` patterns, up to 200 characters. `tool_search_tool_bm25_20251119`: Claude writes natural-language queries, up to 500 characters. Both search tool names, descriptions, argument names and argument descriptions.
- **Results.** Up to 5 `tool_reference` blocks by default. Claude may set `limit` from 1 to 10,000.
- **Catalog size.** Up to 10,000 deferred tools per request. You still send every definition on every request, because the server needs them to search and expand. `defer_loading` controls what enters the *context window*, not the payload.
- **Constraints.** At least one tool must be non-deferred. A deferred tool cannot carry `cache_control`. Strict-mode grammars are built from the full toolset, so deferral does not force grammar recompilation.
- **Cache-friendly by construction.** Deferred tools are excluded from the system-prompt prefix. Discovered tools are appended inline as the conversation proceeds, so the cached prefix stays valid. This is the main systems trick: naive dynamic tool injection rewrites the tool block on every turn and destroys prompt caching.
- **Persistence.** References are expanded throughout the conversation history, so a tool found in turn 2 can be reused in turn 7 without another search.
- **Bring your own retriever.** A custom client tool can return `tool_reference` blocks inside a normal `tool_result`, which lets you plug in embedding search, hybrid search or a reranker. Anthropic's cookbook shows this with `all-MiniLM-L6-v2` (384-d) embeddings over a text rendering of each tool: name, description, then each parameter's name, type and description. Results are ranked by cosine similarity with a default top-5 [13].
- **MCP connector.** For remote MCP servers, `defer_loading` is set once on an `mcp_toolset`'s `default_config` or per tool in `configs`.
- **Billing.** Tool search is not metered separately. Discovered definitions count as normal input tokens.

**Numbers.** In Anthropic's example, a traditional setup consumed about 77K tokens of context before any work, versus about 8.7K with tool search, roughly an 85% reduction [5]. Accuracy on internal MCP evaluations rose from 49% to 74% for Opus 4 and from 79.5% to 88.1% for Opus 4.5 [5]. The documentation recommends tool search when there are 10+ tools, more than 10K tokens of definitions, aggregated MCP servers (200+ tools), or a growing library. It also recommends keeping the 3-5 most-used tools non-deferred [6].

**Operational guidance** [6]: namespace tool names by service (`github_`, `slack_`) so one search matches a whole group. Put the words users would actually type into descriptions. Add a system-prompt line listing the tool *categories* that are searchable. Monitor which tools get discovered.

### 5.2 Claude Code: tool search as the default MCP loading path

Claude Code applies the same mechanism to locally configured MCP servers [14]:

- With tool search on (the default), only tool names and server instructions load at session start. Claude Code imposes no fixed per-server cap; "the practical limit is your context window budget."
- `ENABLE_TOOL_SEARCH` accepts `true`, `false`, `auto` (load upfront until deferrable definitions reach 10% of the context window, then defer all of them) or `auto:N` (a custom percentage).
- Servers can opt out per configuration (`alwaysLoad: true`), or per tool via `_meta["anthropic/alwaysLoad"]`.
- **Description budgets differ by loading path.** Upfront-loaded tool descriptions and server instructions are truncated at 4,096 characters. Descriptions loaded through tool search are allowed 16,384 characters. Because search matches only within the shorter limit, the docs advise putting searchable keywords near the start [14].
- **Server instructions become routing metadata.** MCP server authors are told to describe what category of tasks the server handles and when Claude should search for its tools. This works like a skill description [14], and it amounts to a server-level summary for hierarchical retrieval, the same idea as MCP-Zero's and LiveMCPBench's server-first routing.
- Tool search is disabled automatically when `ANTHROPIC_BASE_URL` points at a non-first-party proxy, "since most proxies don't forward `tool_reference` blocks" [14]. This is a practical reminder that retrieval semantics live in the protocol, and middleboxes can break them.

### 5.3 Progressive disclosure through code and the filesystem

"Code execution with MCP" (Jones and Kelly, November 2025) proposes a third option alongside upfront loading and search: present each MCP server as a directory of code files, one file per tool, and let the agent `ls` and `cat` its way to the definitions it needs. Agents then call tools from code, so intermediate results never pass through the context window [15]. The post also suggests a `search_tools` function with a *detail-level* parameter (name only, name plus description, or full schema). The agent can skim many candidates cheaply and then expand a few. One example workflow went from about 150,000 tokens to about 2,000, a 98.7% reduction [15].

The related **Programmatic Tool Calling** feature lets Claude write Python that calls opted-in tools (`allowed_callers`) inside a sandbox. In one example, about 200KB of expense data reduced to about 1KB of results seen by the model, and average token use on complex research tasks fell 37% (from 43,588 to 27,297 tokens) [5]. **Tool use examples** (`input_examples`) address what retrieval does not: once the right tool is found, calling it correctly. Accuracy on complex parameter handling rose from 72% to 90% in internal testing [5]. Deferred tools carry their examples with them when expanded [6].

### 5.4 The context-engineering framing

Anthropic's "Effective context engineering" post names bloated tool sets as one of the most common agent failure modes: "If a human engineer can't definitively say which tool should be used in a given situation, an AI agent can't be expected to do better." It frames tool search as a case of *just-in-time retrieval*, where the agent keeps lightweight references and loads content on demand [16]. "Writing effective tools for agents" adds that consolidating thin endpoint wrappers into a few workflow-level tools, namespacing, and precise descriptions all change selection accuracy measurably [17]. In other words, catalog design comes before retrieval: deduplicating and merging tools shrinks N before any index is built.

---

## 6. Benchmarks for tool retrieval and large-catalog tool use

### ToolRet (ACL 2025)

"Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models" (Shi, Wang, Yan, Ren, Wang, Yin, Ren) presents itself as the first comprehensive benchmark for evaluating IR models on tool retrieval [7]. The repository evaluates 21 retrievers, 16 embedding models (E5, BGE, GTE, GritLM-7B, NV-Embed-v1 and others) and 5 rerankers (monoT5, bge-reranker-v2-m3, mxbai-rerank-large-v1 and others), each with and without task instructions. It also releases **ToolRet-train**, a large training set; the Hugging Face dataset name suggests about 200K examples [7]. The paper abstract describes the benchmark as aggregating thousands of retrieval tasks over a corpus of tens of thousands of tools (about 7.6K tasks and 43K tools; *unverified here*). Its central finding is in the title: retrievers that are strong on general IR benchmarks are weak on tool retrieval, and training on ToolRet-train improves them substantially.

Why general retrievers struggle:

- **Asymmetry.** Queries describe goals. Documents describe mechanisms, often as JSON Schema.
- **Multi-target relevance.** A task may need several tools at once, and recall@k over the full set matters more than the top-1 hit.
- **Near-duplicates.** Catalogs contain many tools with overlapping descriptions (five weather APIs, three "search" tools), so ranking quality depends on small details.
- **Instructions help.** Instruction-following embedders do better when told "retrieve tools that can accomplish this task," which is why ToolRet reports both settings.

### COLT and ToolLens (CIKM 2024)

"Towards Completeness-Oriented Tool Retrieval for Large Language Models" (Qu et al.) argues that tool retrieval should optimize for retrieving the *complete set* of tools a multi-faceted query needs, not just the most similar one [18]. COLT trains in two stages. The first is semantic learning, fine-tuning dense retrievers such as ANCE, TAS-B, co-Condenser and Contriever via BEIR. The second is collaborative learning, which models relationships between tools and between queries and tool sets. The authors release the ToolLens dataset and a processed ToolBench split [18].

### StableToolBench

See Section 2. Its relevance here is methodological: an end-to-end score over live APIs mixes retrieval quality with API availability, and StableToolBench's cache-plus-simulator server separates the two [12].

### MCP-native benchmarks (2025)

- **MCP-Bench** (Accenture; Wang et al.) runs 28 real MCP servers (BioMCP, Google Maps, Wikipedia, NASA data and others) on single-, two- and three-server tasks with "fuzzy" instructions that do not name the tools. It scores schema understanding, task completion, tool usage and planning. The overall score averages the four dimensions, with o4-mini as the LLM judge for task completion. Leaderboard at time of reading: gpt-5 0.749, o3 0.715, gpt-oss-120b 0.692, gemini-2.5-pro 0.690, claude-sonnet-4 0.681 [19].
- **MCP-Universe** (Salesforce AI Research; Luo et al.) evaluates on real MCP servers with live, time-sensitive ground truth across web search, location navigation, browser automation, financial analysis, repository management and 3D design [20].
- **LiveMCPBench** (Mo et al.) targets the "ocean of tools" setting directly, with a large MCP tool collection (LiveMCPTool), annotated tasks and the retrieval-routed copilot baseline from Section 4 [11].
- **BFCL V4** (Berkeley) adds agentic categories and reports cost and latency alongside accuracy [1].

### Benchmark comparison

| Benchmark | Year | Catalog | What is measured | Retrieval-specific? |
|---|---|---|---|---|
| APIBench (Gorilla) [1] | 2023 | 1,600+ ML hub APIs | AST-correct call, hallucination | Partly (retriever-aware) |
| ToolBench / ToolEval [3] | 2023 | 16,464 REST APIs, 49 categories | Pass/win rate; open-domain retriever | Yes (retriever track) |
| StableToolBench [12] | 2024 | ToolBench + virtual APIs | Solvable pass/win rate | No (stabilization) |
| COLT / ToolLens [18] | 2024 | ToolBench-derived + ToolLens | Completeness of retrieved set | Yes |
| ToolRet [7] | 2025 | ~43K tools (unverified here) | IR metrics for 21 retrievers | Yes (pure IR) |
| MCP-Zero MCP-tools [9] | 2025 | 308 servers / 2,797 tools | Needle tool selection | Yes |
| RAG-MCP stress test [8] | 2025 | N distractor MCPs, swept | Selection accuracy vs N | Yes |
| LiveMCPBench [11] | 2025 | Large live MCP pool | Task success with routing | Partly |
| MCP-Bench [19] | 2025 | 28 live servers | Multi-dim agent score | Partly (fuzzy tasks) |
| MCP-Universe [20] | 2025 | Real servers, 6 domains | Live-ground-truth success | No |
| BFCL V1-V4 [1] | 2024-25 | Function sets per task | Call correctness, relevance | Relevance detection |

---

## 7. Generative tool retrieval: ToolGen

ToolGen (Wang et al., ICLR 2025, "Unified Tool Retrieval and Calling via Generation") removes the external index entirely. Each of about **47,000 tools** becomes a new vocabulary token, initialized from the mean of the subword embeddings of its tool and API names. Training has three stages [21]:

1. **Tool memorization**: description to tool token.
2. **Retrieval training**: query to tool token, producing a "ToolGen-Retriever".
3. **End-to-end agent tuning** on trajectories, so the model emits the tool token and then its arguments in one decoding stream.

```
 query --> LLM (vocab + 47k tool tokens) --> <tool_8812> {"city": "Paris"} --> exec
                         ^ constrained decoding over tool-token space
```

This applies generative retrieval (Chapter 25) to tools. Its advantages are no prompt cost for definitions and no separate retriever latency. Its drawbacks mirror generative retrieval generally: adding or changing a tool needs retraining or careful incremental tuning, and closed-API models cannot be adapted this way. For catalogs that change daily, such as MCP marketplaces, index-based retrieval remains the practical choice.

---

## 8. Embedding tool descriptions: practical recipe

The sources converge on a set of practices.

**What to index (the "tool document").** Name, description, and each parameter's name, type and description, flattened to text. This is the cookbook's `tool_to_text` [13] and also what Anthropic's built-in search covers [6]. Many systems add:

- server or category summary (server instructions in MCP [14]; server summaries in MCP-Zero [9]);
- synthetic example queries ("doc2query" for tools), which close the goal-versus-mechanism gap that ToolRet identifies;
- usage examples (`input_examples` [5]), useful for the model and often useful as retrieval text.

**Beyond the description: synthetic queries and usage-driven embeddings.** Two lines of work attack ToolRet's goal-versus-mechanism gap from the document side rather than by fine-tuning the query encoder:

- **Re-Invoke** (Chen et al., Google, Findings of EMNLP 2024) is unsupervised. At indexing time an LLM generates diverse synthetic queries for each tool and stores them in the tool's document; at query time an LLM extracts the underlying intents from the user request, each intent is retrieved separately, and a multi-view similarity ranking combines the results [24]. It needs no labelled query-tool pairs, which suits new or private catalogs. (Its reported gains over BM25 and dense baselines are *unverified here*.)
- **Tool2Vec** (Moon et al., UC Berkeley SqueezeAI Lab, 2024) builds a tool's embedding from the embeddings of real or generated *usage queries* rather than from its description, adds ToolRefiner (a staged step that refines the retrieved set) and frames retrieval as multi-label classification. The authors report improvements of up to 27.28 Recall@K points on ToolBench and release the ToolBank datasets [25].

Both confirm the practical rule: the text that represents a tool in the index should look like the queries that will hit it.

**Granularity.** Index both servers and tools. Route in two levels (top servers, then top tools within them) once catalogs reach thousands, as in LiveMCPBench's 5x3 setting [11] and MCP-Zero's hierarchy [9].

**Retriever choice.**

- *Lexical* (BM25, regex) works well when names are well namespaced and the model writes the query itself. Anthropic ships both [6].
- *Dense* embeddings (MiniLM [13], `text-embedding-3-small/large` [9][10], instruction-tuned embedders [7]) handle paraphrase. Fine-tuning on tool-retrieval data (ToolBench retriever [3], ToolRet-train [7], COLT [18]) is the single largest lever reported in the literature.
- *Hybrid plus rerank*: ToolRet evaluates cross-encoder rerankers. A reranker over the top 50 is cheap compared with the cost of a wrong tool call.

**Who writes the query.** Prefer *model-issued* queries (an agent calling `tool_search` / `retrieve_tools` mid-task [6][9][10]) over one-shot pre-filtering on the raw user message (the RAG-MCP baseline design [8]). Model-issued queries describe the current sub-goal and can be retried.

**Keep a hot set.** Keep 3-5 frequently used tools always loaded [6] so that common turns pay no search round-trip.

**Write descriptions for search.** Put keywords first (Claude Code matches within the first 4,096 characters [14]), use consistent prefixes, and avoid overlapping tools [17].

```
              +------------------- catalog build (offline) -------------------+
 MCP servers->| normalize schema -> tool doc text -> +synthetic queries        |
 OpenAPI    ->| server summary  -> embed(server), embed(tool), BM25 index     |
 functions  ->| dedupe / merge near-duplicate tools                           |
              +------------------------------+--------------------------------+
                                             |
              +-------------- runtime (per turn) -----------------------------+
              | context: system + hot tools + search tool  (cached prefix)    |
              | model -> search(query) -> hybrid top-50 -> rerank -> top-5    |
              |       <- tool_reference x5 (expanded inline, not in prefix)   |
              | model -> tool_use -> result -> may search again               |
              | log: query, returned ids, chosen id, success -> eval set      |
              +---------------------------------------------------------------+
```

---

## 9. How selection accuracy falls as catalogs grow

The evidence is consistent across sources, even though the experiments differ:

- **Anthropic** puts the degradation threshold at about 30-50 tools in context [6]. Large-library MCP evaluations improved by 25 points for Opus 4 (49% to 74%) and 8.6 points for Opus 4.5 (79.5% to 88.1%) when search replaced upfront loading [5]. Stronger models are more robust to bloat, but still benefit.
- **RAG-MCP** sweeps the number of distractor MCPs and finds monotonic decline for all-in-context prompting. Retrieval roughly triples selection accuracy in its setup (*unverified here*) [8].
- **ToolBench** shows the converse: with a good retriever, an agent over 16K APIs matches oracle-tool performance (67.3 vs 66.7 pass rate) [3].
- **Context rot** is the general mechanism. Recall over context degrades as token count grows [16], and tool definitions are among the most token-dense, least-attended parts of a prompt.

The degradation has three separable components, and each needs its own mitigation:

| Component | Cause | Mitigation |
|---|---|---|
| Attention dilution | Too many tokens of schema in context | Deferred loading, search tool [6], code-based disclosure [15] |
| Confusability | Overlapping or near-duplicate tools | Consolidation and namespacing [17], reranking [7] |
| Recall failure | Retriever misses the right tool | Tool-tuned retriever [3][7][18], re-searchable tool, hot set [6] |

Retrieval mainly fixes the first component and introduces the third. Good catalog hygiene fixes the second. A system that only adds a vector index without addressing confusability will plateau.

---

## 10. Design comparison

| System | Retrieval unit | Retriever | Who queries | Catalog scale | Prompt cost control | Notable number |
|---|---|---|---|---|---|---|
| Gorilla [1] | API doc | BM25 / embed / oracle, top-1 | Framework | 1,600+ APIs | 1 doc in prompt | Retriever-aware fine-tune |
| ToolLLM [3] | API endpoint | Fine-tuned BERT bi-encoder | Framework | 16,464 APIs | top-k + DFSDT | 67.3 pass rate w/ retriever |
| AnyTool [22] | Category, then tool, then API | GPT-4 hierarchical agents + self-reflection | Agents | 16K APIs | Hierarchical narrowing | 58.2% vs 31.8% (filtered ToolBench) |
| RAG-MCP [8] | MCP server desc | Dense semantic | Framework | N swept | top-k only | ~3x accuracy (unverified here) |
| MCP-Zero [9] | Server, then tool | text-embedding-3-large | Agent (active) | 308 / 2,797 | Hierarchical | 2-level routing |
| BigTool [10] | Tool | Store semantic search | Agent tool | Any (store) | Bind retrieved only | Pluggable retriever |
| LiveMCPBench copilot [11] | Server, then tool | Dense, 1,024-d | Agent | Large MCP pool | 5 servers x 3 tools | <= 15 schemas |
| Claude tool search [5][6] | Tool | Regex / BM25 / custom | Model (server tool) | <= 10,000 deferred | Deferred, cache-safe | ~85% token cut; 49->74% |
| Claude Code [14] | MCP tool | Built-in search | Model | Context-bound | Names only upfront | `auto` at 10% of context |
| Code-exec MCP [15] | Tool file | Filesystem / search_tools | Model (code) | Any | Detail-level disclosure | ~150K -> 2K tokens |
| Tool2Vec [25] | Tool (usage-query centroid) | Usage-driven dense + ToolRefiner | Framework | ToolBench, ToolBank | top-k | Up to +27.28 Recall@K |
| ToolGen [21] | Tool token | Generative (vocab) | Model | ~47,000 tools | Zero definition tokens | Retrain to add tools |

---

## Key takeaways for system designers

1. **Count tokens before you count tools.** Measure the tokens your tool block occupies. Past roughly 10 tools or 10K tokens of definitions, defer and search [6]. A five-server MCP setup can cost about 55K tokens before the user speaks [5].
2. **Make retrieval a tool, not a pre-filter.** One-shot top-k on the user's message (RAG-MCP style [8]) fails silently when recall misses. A search tool the model can call repeatedly with sub-goal queries (Claude tool search [6], BigTool [10], MCP-Zero [9]) recovers.
3. **Protect the prompt cache.** Inject discovered tools inline after the cached prefix, never by rewriting the tool list [6]. Dynamic tool sets that break caching can cost more than they save.
4. **Keep a hot set.** Leave the 3-5 most common tools always loaded so typical turns skip the search round-trip [6].
5. **Fix the catalog before tuning the retriever.** Consolidate thin wrappers, remove overlaps, namespace by service, and lead descriptions with the words users actually type [14][17]. Confusability is not a retrieval problem.
6. **Index richer tool documents.** Include parameters, server summaries and synthetic queries. Route servers first, then tools, at thousands-of-tools scale [9][11][13].
7. **Off-the-shelf embedders are not tool-savvy.** Expect a gap. Fine-tune on tool-retrieval pairs (ToolBench, ToolRet-train, COLT) or add a reranker, and evaluate recall over the full required set, not just top-1 [3][7][18].
8. **Disclose progressively.** Return names first, descriptions on request and full schemas last. Code-based tool access can cut context by orders of magnitude on data-heavy workflows [15].
9. **Retrieval finds the tool; examples make the call right.** Pair retrieval with `input_examples` or few-shot calls for complex schemas (72% to 90% in Anthropic's testing [5]).
10. **Evaluate on stable, realistic suites.** Use cached or simulated APIs (StableToolBench [12]) for regression testing, and live MCP suites (MCP-Bench, MCP-Universe, LiveMCPBench) for realism. Log every search query, returned ID and chosen ID, and turn those logs into a retrieval evaluation set.
11. **Watch the middleboxes.** Proxies and gateways that do not understand `tool_reference` blocks silently disable tool search [14]. Retrieval semantics are part of your protocol surface.

---

## References

1. Gorilla: Large Language Model Connected with Massive APIs (repository: APIBench, OpenFunctions, BFCL, GoEx, RAFT). S. G. Patil, T. Zhang, X. Wang, J. E. Gonzalez / UC Berkeley, 2023-2025. https://github.com/ShishirPatil/gorilla
2. Gorilla: Large Language Model Connected with Massive APIs (paper). S. G. Patil, T. Zhang, X. Wang, J. E. Gonzalez, 2023. https://arxiv.org/abs/2305.15334
3. ToolBench / ToolLLM repository (data stats, API retriever, DFSDT, ToolEval). OpenBMB / Tsinghua University, 2023-2024. https://github.com/OpenBMB/ToolBench
4. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. Y. Qin et al., ICLR 2024. https://arxiv.org/abs/2307.16789
5. Introducing advanced tool use on the Claude Developer Platform. B. Wu et al. / Anthropic, 2025. https://www.anthropic.com/engineering/advanced-tool-use
6. Tool search tool (documentation). Anthropic, 2025-2026. https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
7. Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models (ToolRet). Z. Shi, Y. Wang, L. Yan, P. Ren, S. Wang, D. Yin, Z. Ren, ACL 2025. https://github.com/mangopy/tool-retrieval-benchmark (paper: https://arxiv.org/abs/2503.01763)
8. RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation. T. Gan, Q. Sun, 2025. https://arxiv.org/abs/2505.03275
9. MCP-Zero: Active Tool Discovery for Autonomous LLM Agents. X. Fei, X. Zheng, H. Feng, 2025. https://github.com/xfey/MCP-Zero
10. langgraph-bigtool. LangChain, 2025. https://github.com/langchain-ai/langgraph-bigtool
11. LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? G. Mo et al. / ICIP-CAS, 2025. https://github.com/icip-cas/LiveMCPBench
12. StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models. Z. Guo et al., 2024. https://github.com/THUNLP-MT/StableToolBench
13. Tool search with embeddings (cookbook). Anthropic, 2025. https://platform.claude.com/cookbook/tool-use-tool-search-with-embeddings
14. Connect Claude Code to tools via MCP (Scale with MCP tool search; Configure tool search). Anthropic, 2025-2026. https://code.claude.com/docs/en/mcp
15. Code execution with MCP: Building more efficient agents. A. Jones, C. Kelly / Anthropic, 2025. https://www.anthropic.com/engineering/code-execution-with-mcp
16. Effective context engineering for AI agents. P. Rajasekaran, E. Dixon, C. Ryan, J. Hadfield / Anthropic, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
17. Writing effective tools for agents - with agents. K. Aizawa / Anthropic, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
18. Towards Completeness-Oriented Tool Retrieval for Large Language Models (COLT). C. Qu et al., CIKM 2024. https://github.com/quchangle1/COLT
19. MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers. Z. Wang et al. / Accenture, 2025. https://github.com/Accenture/mcp-bench
20. MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers. Z. Luo et al. / Salesforce AI Research, 2025. https://github.com/SalesforceAIResearch/MCP-Universe
21. ToolGen: Unified Tool Retrieval and Calling via Generation. R. Wang, X. Han, L. Ji, S. Wang, T. Baldwin, H. Li, ICLR 2025. https://github.com/Reason-Wang/ToolGen
22. AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls. Y. Du, F. Wei, H. Zhang, 2024. https://github.com/dyabel/AnyTool (paper: https://arxiv.org/abs/2402.04253)
23. MCP Registry (official registry of MCP servers; preview launched 2025-09-08). Model Context Protocol project, 2025-2026. https://github.com/modelcontextprotocol/registry
24. Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval. Y. Chen, J. Yoon, D. S. Sachan, Q. Wang, V. Cohen-Addad, M. Bateni, C.-Y. Lee, T. Pfister / Google, Findings of EMNLP 2024. https://arxiv.org/abs/2408.01875
25. Efficient and Scalable Estimation of Tool Representations in Vector Space (Tool2Vec, ToolRefiner, ToolBank). S. Moon, S. Jha, L. E. Erdogan, S. Kim, W. Lim, K. Keutzer, A. Gholami, 2024. https://github.com/SqueezeAILab/Tool2Vec (paper: https://arxiv.org/abs/2409.02141)
