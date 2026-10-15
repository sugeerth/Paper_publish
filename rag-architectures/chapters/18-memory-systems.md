# 18. Long-Term Memory Systems for LLMs and Agents: MemGPT/Letta, Mem0, Zep/Graphiti, ChatGPT and Claude Memory, LangMem, and the Write/Read Policies Behind Them

## Overview

Classic RAG retrieves from a corpus that someone else wrote and that changes slowly.
Long-term memory is a different case. The agent writes the corpus itself, while it runs,
from its own conversations and actions. The corpus changes every turn, and facts in it
contradict one another as users change jobs, preferences and plans. The read path still
looks like retrieval (embed, search, rerank, inject into the prompt). The hard parts move
to the **write path**: what gets stored, in what form, when, by whom, and how newer facts
supersede older ones.

This chapter surveys the publicly documented designs, from the research prototypes of
2023 to the production assistant memories of 2025-2026:

- **OS-style tiered memory**: MemGPT and its successor Letta, where the model pages
  information in and out of its own context window with tool calls [1][2][3].
- **Extract-and-consolidate memory layers**: Mem0, which turns conversations into atomic
  facts and reconciles them with an existing store [8][9][10].
- **Temporal knowledge graphs**: Zep and its open-source engine Graphiti, which store
  facts as graph edges that carry validity intervals [12][13][14].
- **Toolkits and taxonomies**: LangMem's semantic/episodic/procedural split and its
  "hot path versus background" write policies [16][17], CoALA [30], Generative Agents
  [29], A-Mem [32], MemOS [33] and MIRIX [34].
- **Consumer assistant memory**: ChatGPT's saved memories and chat-history reference
  [18][19][21], and Claude's tool-based chat search, memory summaries and topics
  [22][24], plus Gemini's personal context [44].
- **Further designs** (Section 10): the LoCoMo benchmark [38], forgetting curves [39],
  procedural memory distilled from experience [40][41] and RL-trained memory policies
  [42][43].
- **Developer-facing file memory**: Anthropic's client-side memory tool, context editing
  and Claude Code's CLAUDE.md and auto memory [25][26][27][28].

A pattern runs through the whole period. The early designs leaned on complex machinery:
vector archival stores, LLM-driven UPDATE/DELETE reconciliation and graph construction.
By 2025-2026 several of the strongest results came from simpler substrates: plain files,
append-only fact logs and on-demand search over raw transcripts [7][9][26]. Evaluation is
contested territory. Vendors publish benchmark numbers that conflict with each other
[11][15], so this chapter reports them with their provenance.

```
                     THE MEMORY LOOP (generic)
  +-----------+   turn    +-------------------+   response
  |   user /  |---------->|   LLM agent       |------------> user / env
  |   env     |           | (context window)  |
  +-----------+           +---------+---------+
                             ^      |  WRITE POLICY
                 READ POLICY |      |  (what / when / who / form)
                 (what to    |      v
                 inject,     |  +---------------------------+
                 when, how   |  | memory store(s)           |
                 much)       +--| profile | facts | graph | |
                                | episodes | files | logs   |
                                +---------------------------+
                                     ^  CONSOLIDATION / FORGETTING
                                     |  (dedupe, invalidate, summarize,
                                     +--  reflect, expire) - often offline
```

## 1. Conceptual Foundations: Memory Types and Policies

### 1.1 Cognitive taxonomies

**CoALA** (Cognitive Architectures for Language Agents; Sumers, Yao, Narasimhan,
Griffiths, 2023, TMLR 2024) borrowed the vocabulary of cognitive science. It describes a
language agent as having modular memory components, a structured action space for acting
on internal memory and external environments, and a generalized decision procedure [30].
Most later systems use its four-way split: *working* memory (the active context),
*episodic* memory (past experiences), *semantic* memory (facts) and *procedural* memory
(skills and behavior, stored either in the weights or in code and prompts) [30].

**LangMem** turns that split into storage guidance [17]:

| Memory type | Purpose | Agent example | Typical storage |
|---|---|---|---|
| Semantic | Facts and knowledge | User preferences; knowledge triplets | Profile or collection |
| Episodic | Past experiences | Few-shot examples; summaries of past conversations | Collection |
| Procedural | System behavior | Core personality, response patterns | Prompt rules or collection |

LangMem also separates two forms of semantic memory [17]:

- A **collection** holds an unbounded set of records that are searched at run time. The
  system has to reconcile each new fact with what it already holds. "If the system
  over-extracts, this could lead to reduced precision... If it under-extracts, this could
  lead to low recall."
- A **profile** is a single schema-bound document, such as a Pydantic model of the user's
  name, preferred name and response style. Each update replaces the current state. A
  profile is cheap to inject and easy for users to edit by hand, but it keeps no history.

LangMem's guide also says that relevance "is more than just semantic similarity". It
argues that recall should combine similarity with a memory's *importance* and its
*strength* (how recently and how often it was used) [17]. That idea goes back to
Generative Agents.

### 1.2 Generative Agents: the memory stream

Park et al. (2023) built agents for a simulated town around a **memory stream**: "a
long-term memory module that records, in natural language, a comprehensive list of the
agent's experiences" [29]. The **read policy** ranks memories by a weighted combination
of three signals:

- **Recency**: exponential decay since the memory was last accessed.
- **Importance**: an integer score the LLM assigns when the memory is written.
- **Relevance**: embedding similarity to the current query.

The **consolidation policy** is *reflection*. Periodically the agent synthesizes related
observations into higher-level inferences and writes them back into the stream, where
they can be retrieved like any other memory [29]. Almost every later system uses these
two ideas in some form. Importance-weighted recall and offline reflection reappear as
LangMem's "strength", Letta's sleep-time agents and Claude's daily memory synthesis.

### 1.3 The four policy questions

Whatever the substrate, every memory design answers the same questions:

1. **What to write.** Raw transcripts, atomic facts, typed profiles, graph triples,
   reflections, or free-form notes in files.
2. **When and who writes.** The agent can write *in the hot path*, via tool calls during
   the turn. A background process can write after the turn or during idle time.
   LangMem's guide calls these "conscious" and "subconscious" formation: hot-path writes
   add "perceptible latency to user interactions" and give the agent one more decision
   to make, while background writes are listed with a latency impact of "None" but
   update with a delay [17].
3. **How to reconcile.** Overwrite, UPDATE/DELETE via the LLM, append-only with links, or
   bi-temporal invalidation.
4. **What and when to read.** Always inject a small profile, retrieve per query, or let
   the agent search on demand through tools.

The sections below show that the major systems differ mostly in how they answer these
four questions, not in their embedding models or vector stores.

## 2. MemGPT and Letta: The LLM as Operating System

### 2.1 Problem

MemGPT (Packer et al., UC Berkeley, October 2023) started from the fixed context window.
Long conversations and large documents do not fit, and naive truncation loses
information. The paper borrows virtual memory from operating systems. The LLM gets a
small fast tier (the prompt) and large slow tiers (external stores), and it moves data
between them itself, using function calls [1].

### 2.2 Architecture

```
  +--------------------------- MAIN CONTEXT (prompt tokens) --------------------------+
  | [system instructions] [working context: editable persona/user facts] [FIFO queue] |
  +-----------------------------------------------+-----------------------------------+
           ^  page in (search results)            |  page out (insert / evict)
           |                                      v
  +--------+---------------------+   +------------+-------------------------------+
  | RECALL STORAGE               |   | ARCHIVAL STORAGE                           |
  | full message log; search by  |   | read/write vector-backed DB of arbitrary   |
  | text / date                  |   | text; semantic search                      |
  +------------------------------+   +--------------------------------------------+
        QUEUE MANAGER: warns at ~70% of window ("memory pressure"); at 100% flushes
        ~half the queue and rewrites a recursive summary  |  FUNCTION EXECUTOR
```

The main context has three parts [1]:

- **System instructions**: read-only. They describe the control flow and the memory
  functions.
- **Working context**: a read/write scratchpad of key facts about the user and the
  agent's persona.
- **FIFO queue**: the rolling message history, headed by a recursive summary of evicted
  messages.

Outside the prompt, **recall storage** holds the complete message log and can be searched
by content or date. **Archival storage** is a read/write, vector-backed database for
arbitrary text [1]. A **queue manager** enforces the memory budget. When prompt tokens
pass a warning threshold (about 70% of the window, per the paper as summarized in
secondary sources), it inserts a "memory pressure" system message so that the model can
save important facts. At the limit it flushes the queue, evicting about half the messages
and updating the recursive summary [1]. The model can also chain calls: a function call
can request an immediate follow-up inference, so a multi-step search runs before the
agent answers the user.

### 2.3 Results

On the paper's Deep Memory Retrieval (DMR) task, built on the Multi-Session Chat
dataset, GPT-4 with a fixed context scored 32.1%. GPT-4 under MemGPT scored 92.5% [1].
Secondary summaries report 93.4% for GPT-4 Turbo with MemGPT and 66.9% for GPT-3.5 Turbo
with MemGPT, against 35.3% and 38.7% for the corresponding fixed-context baselines.
These pairs come from summaries rather than the paper's own table, so treat them as
approximate. The paper also reports conversation-opener, document-QA and nested
key-value retrieval experiments [1].

### 2.4 Letta: memory blocks, sleep-time agents, MemFS

The MemGPT authors went on to found Letta, which describes itself as "formerly MemGPT"
[2]. Its concepts moved through several stages:

- **Memory blocks.** The working context became labeled, size-limited, editable sections
  of the system prompt (for example `persona` and `human`). The agent edits them through
  memory tools, and a block can be shared across several agents [4][5].
- **Sleep-time agents.** A primary agent is paired with a background agent that runs
  between turns and rewrites the primary agent's memory blocks asynchronously. Letta's
  documentation describes it as triggering every N steps (5 by default) and labels the
  feature experimental [5]. The research paper behind it, *Sleep-time Compute: Beyond
  Inference Scaling at Test-time* (Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez,
  2025), reports these results [6]:
  - About 5x less test-time compute needed for the same accuracy on Stateful
    GSM-Symbolic and Stateful AIME.
  - Accuracy gains of up to 13% (GSM-Symbolic) and 18% (AIME) when sleep-time compute is
    scaled.
  - A 2.5x lower average cost per query when several related queries share the same
    pre-computation (Multi-Query GSM-Symbolic).
  - The gains are largest when queries are predictable from the context.
- **Filesystem memory.** In August 2025 Letta published *Benchmarking AI Agent Memory: Is
  a Filesystem All You Need?* [7]. The conversation history was uploaded as a single file
  and the agent got file tools (`grep`, a semantic `search_files`, `open`, `close`). With
  gpt-4o-mini, the agent scored 74.0% on LoCoMo, higher than the 68.5% that Mem0 reported
  for its best graph variant. Commentators pointed out that `search_files` is semantic,
  so "grep" undersells the setup, and that LoCoMo barely tests knowledge updates. Letta
  itself called framework-to-framework comparison apples-to-oranges [7].
- **Letta Code (2026).** The active codebase is now `letta-code`, and the V1 API server
  has been retired to an archive branch [2]. The README describes agents
  "rewriting their own memory, skills, prompts, and even the harness itself". All context, including memory blocks, is
  tracked in a git-versioned **MemFS** that can sync to a custom GitHub repository
  (`/memory-repository`). Periodic "dreaming" is
  configured with `/sleeptime`, a recall subagent searches past messages, and
  agent-scoped skills live in MemFS [3].

### 2.5 Failure modes and lessons

- **Self-management takes model capability.** The paper's own numbers show the gap: a
  weaker model (GPT-3.5 Turbo) reached 66.9% on DMR versus GPT-4's 92.5% [1]. Paging
  decisions are only as good as the model's judgment about tool use.
- **Hot-path memory editing costs latency.** Every memory edit is an extra inference
  step. Moving consolidation into sleep-time agents is Letta's answer [5][6].
- **Lesson.** The OS metaphor survived but the substrate got simpler. MemFS's git-tracked
  files and the filesystem benchmark both point to plain files plus good tools, rather
  than bespoke stores, as a strong default [3][7].

## 3. Mem0: Extract, Consolidate, Retrieve

### 3.1 Problem and positioning

Mem0 ("mem-zero", YC S24) offers a drop-in memory layer for applications. It scopes
memories at the user, session and agent level and ships as a library, a self-hosted
server and a managed cloud [9]. Its April 2025 paper, *Mem0: Building Production-Ready
AI Agents with Scalable Long-Term Memory*, targets the cost and latency of stuffing
entire histories into the prompt [8].

### 3.2 Original architecture (2025 paper, v1/v2 library)

```
 new message pair (m_t-1, m_t)
        |
        v
 +------------------- EXTRACTION -------------------+
 | context = async-refreshed conversation summary S |
 |           + last m messages (default m=10)       |
 | LLM -> list of candidate facts                   |
 +-------------------------+------------------------+
                           v
 +-------------------- UPDATE ----------------------+
 | for each fact: vector-search top s similar       |
 | memories (default s=10)                          |
 | LLM tool call -> ADD | UPDATE | DELETE | NONE     |
 +-------------------------+------------------------+
                           v
             vector store (+ graph store for Mem0g)
```

Extraction runs on each new message pair. Its context is a conversation summary that is
refreshed asynchronously, plus a window of recent messages (default 10 per secondary
analyses of the paper) [8]. For each candidate fact, the update phase fetches the top-s
most similar existing memories and asks the LLM to choose an operation. The library's
`DEFAULT_UPDATE_MEMORY_PROMPT` names them directly: "ADD: Add it to the memory as a new
element; UPDATE: Update an existing memory element; DELETE: Delete an existing memory
element; NONE: Make no change (if the fact is already present or irrelevant)" [10]. The
graph variant, **Mem0g**, also stores entities and relations in a graph database; the
paper's implementation used Neo4j, and the library's documentation has listed several
other graph backends over time [8].

### 3.3 Reported numbers (paper)

On LoCoMo the paper reports these figures [8]:

- **Search latency.** Mem0 had the lowest of all compared methods: p50 0.148 s, p95
  0.200 s.
- **Total latency.** Mem0's median was 0.708 s and its p95 was 1.440 s. The full-context
  baseline, which passes about 26,000 tokens, had a p50 of 9.870 s and a p95 of
  17.117 s.
- **Headline claims.** The paper reports 91% lower p95 latency and more than 90% lower
  token cost than full context.
- **Accuracy.** Mem0's companion blog reports an LLM-as-judge score of 66.9% for Mem0,
  against 52.9% for OpenAI Memory and 58.1% for LangMem, roughly a 26% relative gain over
  OpenAI [11].
- **Mem0g.** The graph variant scored somewhat higher on temporal questions (58.1 vs
  55.5) and lower on multi-hop questions. It used about twice the tokens per query
  (about 4K vs 2K, per the vendor blog) and had higher latency [11].

### 3.4 The 2026 redesign: ADD-only extraction

In April 2026 Mem0 published a new algorithm that drops the UPDATE/DELETE model [9].
The README lists these changes:

- **Single-pass ADD-only extraction.** "One LLM call, no UPDATE/DELETE. Memories
  accumulate; nothing is overwritten."
- **Agent-generated facts are first-class.** Information an agent confirms, such as an
  action it took, is stored with the same weight as user facts.
- **Entity linking.** Entities are extracted, embedded and linked across memories to
  boost retrieval.
- **Multi-signal retrieval.** Semantic, BM25 keyword and entity matches are scored in
  parallel and fused.
- **Temporal reasoning.** Retrieval ranks the right dated instance for questions about
  current state, past events and upcoming plans.

The new `ADDITIVE_EXTRACTION_PROMPT` in the codebase says: "Your sole operation is ADD."
When a new memory relates to an existing one ("updated/shifted preference, follow-up
event, or continuation of a narrative"), the extractor records the old memory's UUID in
`linked_memory_ids` instead of rewriting it [10].

| Benchmark (Mem0 managed platform) | Old | New | Tokens/query | p50 latency |
|---|---|---|---|---|
| LoCoMo | 71.4 | 92.5 | 7.0K | 0.88 s |
| LongMemEval | 67.8 | 94.4 | 6.8K | 1.09 s |
| BEAM (1M) | n/a | 64.1 | 6.7K | 1.00 s |
| BEAM (10M) | n/a | 48.6 | 6.9K | 1.05 s |

Source: Mem0 README [9]. These are self-reported figures for single-pass retrieval. The
README says the scores "reflect Mem0's managed platform" and that "open-source users
should expect directionally similar gains but not identical numbers".

### 3.5 Failure modes and lessons

- **Destructive updates lose history.** An LLM-chosen DELETE or UPDATE can erase a fact
  that was still true, or was true at a different time. Without timestamps, the store
  cannot answer "what did I prefer last year?". The 2026 switch to append-plus-link is,
  in effect, an admission that lossy reconciliation was the weak point [9][10].
- **Benchmark disputes.** Zep argued that Mem0's paper ran Zep incorrectly (wrong user
  role assignment, wrong timestamp handling). Zep then published a corrected LoCoMo
  score of 75.14% ± 0.17 for itself, after admitting an error in its own first
  calculation [15]. Third-party papers report yet other orderings [11]. Treat cross-vendor
  memory leaderboards as directional at best.
- **Lesson.** Separate extraction (high recall, append-only) from read-time resolution
  (temporal ranking, entity linking). This is the same conclusion Zep reached by a
  different route.

## 4. Zep and Graphiti: Bi-Temporal Knowledge Graph Memory

### 4.1 Problem

Zep (Rasmussen et al., January 2025) argues that enterprise agents need memory that
combines conversation with business data and keeps track of *when* facts held. Static
RAG and batch GraphRAG do neither well [12][13]. Graphiti is the open-source engine
underneath. Zep's managed service runs on a proprietary database, the "Context Graph
Engine", and advertises "sub-200ms performance at scale". Self-hosted Graphiti brings
its own graph database (Neo4j, FalkorDB or Amazon Neptune; Kuzu support is marked
deprecated), and its performance "depends on your setup" [13].

### 4.2 Architecture

```
  EPISODES (raw messages / JSON / text, each with a reference timestamp)
     |  ingest: entity extraction (window of recent messages) -> entity resolution
     |          (embedding + full-text candidate match, LLM dedupe) -> fact (edge)
     |          extraction + dedupe -> temporal extraction -> contradiction check
     v
 +------------------------------------------------------------------------+
 |  Episode subgraph   : raw episodes; every derived fact links back here |
 |  Semantic subgraph  : entity nodes (evolving summaries) + fact edges   |
 |                       edge = (fact text, embedding, valid_at,          |
 |                       invalid_at, created_at, expired_at, episodes[])  |
 |  Community subgraph : clusters of entities with summaries              |
 +------------------------------------------------------------------------+
     |  retrieve: search (cosine | BM25 | breadth-first traversal)
     |            -> rerank (RRF | MMR | node distance | episode mentions |
     |               cross-encoder) -> context constructor (facts + dates)
     v
  prompt context string
```

The graph has three tiers [12][13]:

- **Episodes**: raw data, kept as ground-truth provenance.
- **Semantic entities and facts**: the extracted knowledge.
- **Communities**: clusters of entities with high-level summaries.

Ingestion is incremental. New episodes are integrated "without requiring complete graph
recomputation", in contrast to GraphRAG's batch pipeline [13]. Developers can declare
custom entity and edge types as Pydantic models (a "prescribed" ontology) or let the
structure emerge from the data (a "learned" one) [13].

### 4.3 The bi-temporal model

Each fact edge carries two timelines [12][14]:

- **Valid time.** `valid_at` is when the fact became true in the world; `invalid_at` is
  when it stopped. The code says "soft deadlines are not invalid_at", and future-tense
  plans use the episode's reference time.
- **Transaction time.** `created_at` and `expired_at` record when the system learned the
  fact and when it retired the edge.

Each edge also stores the `reference_time` of the episode that produced it, which lets
Graphiti resolve relative expressions such as "two weeks ago" [14]. When a new edge
contradicts an existing one, the older edge is **invalidated, not deleted**. The system
sorts conflicting edges by `valid_at` and expires the one that occurred earlier in the
real world [12][13]. As a result, both "what is true now" and "what was true at time T"
can be answered.

### 4.4 Retrieval

Graphiti's search configuration defines these options [14]:

- **Edge and node search:** `cosine_similarity`, `bm25` and `breadth_first_search`.
- **Episode search:** BM25.
- **Community search:** cosine and BM25.
- **Rerankers:** reciprocal rank fusion, MMR, `node_distance` (graph proximity to a focal
  node), `episode_mentions` (how often a fact recurs) and a cross-encoder.

Sixteen predefined recipes in `search_config_recipes.py`, such as
`COMBINED_HYBRID_SEARCH_RRF` and `EDGE_HYBRID_SEARCH_NODE_DISTANCE`, package these
choices [14]. (Episode reranking is limited to RRF and the cross-encoder, and community
reranking to RRF, MMR and the cross-encoder.) The README stresses that retrieval does not depend on LLM
summarization at query time. That is how it reaches "typically sub-second" latency,
compared with "seconds to tens of seconds" for GraphRAG-style global search [13].

### 4.5 Reported numbers

All of these are self-reported:

- **DMR.** Zep 94.8% vs MemGPT 93.4% [12].
- **LongMemEval with gpt-4o.** Zep scored 71.2% against 60.2% for full context, using
  about 1.6K context tokens instead of about 115K. Median latency was 2.58 s for Zep
  against 28.9 s for full context, which Zep summarizes as roughly a 90% latency
  reduction. With gpt-4o-mini, Zep scored 63.8% against 55.4% [12][37].
- **Headline.** Zep cites "accuracy improvements of up to 18.5%". The paper reports the
  largest gains on cross-session synthesis and temporal questions [12].

### 4.6 Failure modes and lessons

- **Ingestion cost and quality.** Every episode triggers several LLM calls (extraction,
  resolution, dating, invalidation). Graphiti's README warns that providers without
  structured output, and especially smaller models, can produce "incorrect output
  schemas and ingestion failures" [13].
- **Entity resolution errors compound.** Two nodes for one person, or one node for two
  people, will corrupt every fact attached to them. This is why resolution uses both
  embedding and full-text candidate matching [12].
- **Lesson.** Time is a first-class retrieval dimension in agent memory. Storing validity
  intervals is cheap and removes a whole class of "stale fact" errors that similarity
  search cannot fix.

## 5. LangMem and LangGraph Memory: A Toolkit View

LangMem (LangChain, 2025) is not a single architecture. It is a set of primitives that
apply the policies from Section 1.3 [16][17]:

- **Core functional API** (storage-agnostic):
  - `create_memory_manager` extracts, updates and consolidates memories from
    conversations. It can produce free-form collections (`enable_inserts=True`) or
    schema-bound profiles.
  - `create_prompt_optimizer` rewrites prompt rules from conversations and feedback.
    This is procedural memory implemented as prompt evolution.
- **Stateful layer** on LangGraph's `BaseStore`:
  - Store managers persist extracted memories automatically.
  - `create_manage_memory_tool` and `create_search_memory_tool` let an agent write and
    search memory in the hot path.
- **Namespaces** segment memories hierarchically, for example by organization, user or
  app. Templates such as `("acme_corp", "{user_id}", "code_assistant")` are filled at run
  time from the `configurable` fields of the `RunnableConfig`.
  Retrieval supports direct key lookup, semantic search and metadata filtering [17].

```
          HOT PATH (conscious)                 BACKGROUND (subconscious)
  user -> agent --tool: manage_memory--> store <-- memory manager (after turn /
          agent --tool: search_memory--> store      after inactivity; reflects,
          (+latency, +tool-choice burden)           consolidates; no added latency)
```

The guide's write-policy table sums up the trade-off. Active formation has higher latency
and immediate updates, and suits critical context. Background formation adds no latency
but updates with a delay, and suits pattern analysis and summaries [17]. Episodic memory
in LangMem means storing *successful interactions*: the situation, the reasoning and why
it worked, for later use as few-shot examples. That is a narrower and more actionable
meaning than "a log of past chats" [17].

## 6. Consumer Assistant Memory: ChatGPT

### 6.1 Timeline and controls

OpenAI launched ChatGPT memory in 2024 as user-visible **saved memories**. On April 10,
2025 it expanded memory to "reference all past conversations" for Plus and Pro users
[18]. The feature now has two settings. **Reference saved memories** covers details the
user explicitly asked ChatGPT to remember. **Reference chat history** draws on past chats,
and OpenAI notes that what it remembers from them "can change over time" [19]. Saved
memories persist until the user deletes them, and turning the setting off does not erase
what is already stored [19]. On June 3, 2025 free users got a lighter version that
provides "short-term continuity across conversations" [20].

### 6.2 Reverse-engineered architecture

OpenAI has not published the internals. In December 2025, Manthan Gupta probed ChatGPT
in conversation and reported that, at least at that time, it did not appear to run
retrieval over a vector index of past conversations. He found a structured context
assembled for each message [21]:

```
  [system + developer instructions]
  [session metadata]        device/env info; injected per session, not persisted
  [user memory]             explicit long-term facts (saved memories)
  [recent conversations]    lightweight summaries of recent chats
  [current session window]  sliding window of this conversation's messages
```

This is a **pre-injection** read policy: a compact profile plus recent-chat digests sit in
every prompt, so nothing needs to be retrieved at answer time. Shlok Khemani's
independent analysis likewise described ChatGPT as combining user-controlled memories
with "a dense generated summary" of history that the user cannot see. In a 2026 talk he
put that profile at roughly 4,000 tokens [22][23]. These are inferences from probing,
not documentation, and they may be out of date.

### 6.3 Failure modes

- **Opacity.** Chat-history memory has no user-visible ledger, so users cannot audit why
  a detail came up. Saved memories, by contrast, are listed and editable [19][21].
- **Memory as an injection persistence channel.** In September 2024 Johann Rehberger
  showed "SpAIware" against the ChatGPT macOS app. A prompt injection planted in
  untrusted content (a web page or document) wrote attacker instructions into memory.
  Those instructions then rode along in every future conversation and exfiltrated chat
  content. OpenAI patched the exfiltration vector [35]. The general lesson: any
  automatic write policy that runs over untrusted content turns a one-shot injection into
  a persistent compromise.

## 7. Consumer Assistant Memory: Claude

### 7.1 Phase 1 (August 2025): retrieval-only, tool-mediated

Claude's first cross-chat memory used the opposite design to ChatGPT. Each conversation
started from a blank slate with no preloaded profile. Recall worked by searching raw past
conversations through two visible tools [22]:

- `conversation_search`: searches past conversations for relevant context.
- `recent_chats`: retrieves recent chats, with sort order, `before`/`after` datetime
  pagination and project filtering.

Because memory access shows up as tool calls, users can see when and how past context is
used. Khemani summed up the contrast this way: ChatGPT trades depth for automatic
continuity, while Claude trades automatic continuity for on-demand depth [22][23].

### 7.2 Phase 2 (September 2025 onward): synthesized summaries, then topics

Anthropic added automatic memory for Team and Enterprise on September 11, 2025, and
extended it to Pro and Max in October 2025 [36][24]. Claude's help center describes two
generations [24]:

- **Legacy memory.** Claude summarizes chats outside projects into a *memory summary*.
  The synthesis refreshes **every 24 hours**, while edits the user makes in chat apply
  immediately. Deleted conversations drop out of the synthesis.
- **New experience (topics).** Claude "saves memory as individual topics while you chat,
  not by summarizing conversations after they end". It saves either on its own initiative
  or when told to "remember this".

Several controls and scoping rules apply to both [24]:

- **Projects.** Each project has its own memory. Inside a project, Claude uses that
  project's memory plus the user's non-project memory, but never another project's;
  turning off "Use account memory" isolates a project completely. Past-chat search
  follows the same boundaries.
- **Sensitive topics.** Health, politics, religion and similar topics are not saved
  unless the user opts in, and Claude shows a notice when it saves one. Some data, such
  as government ID numbers, financial account numbers, criminal history and
  immigration status, is never saved.
- **User controls.** Users can pause or reset memory, turn it off for a single chat, use
  incognito chats, and view and edit topics. Import and export are experimental.
- **Deletion semantics.** In the new experience, deleting or expiring a conversation
  does *not* delete the memory entries made from it; they must be deleted individually.
  Reset deletes all memories, including project memories, irreversibly.
- **Availability.** As of 2026 the help center lists memory as on by default for Free,
  Pro and Max, and opt-in (owner-enabled) on Team and Enterprise; it is unavailable to
  organizations with HIPAA, public-sector or custom data-retention agreements.

Khemani reported that the later Claude profile is user-visible and about 1,000 tokens,
against his estimate of about 4,000 for ChatGPT [23]. Chat search remains a separate,
tool-based capability on paid plans [24].

The two phases show a broader convergence. The pure on-demand retrieval design gained a
small always-injected profile. The pure pre-injection design (ChatGPT) depends on
recency-weighted summaries. Production assistants end up as **hybrids**: a small,
editable profile in every prompt plus retrieval over history when needed.

## 8. Developer Memory at Anthropic: Files, Context Editing, Auto Memory

### 8.1 The memory tool (API)

Anthropic announced the memory tool and context editing on September 29, 2025, together
with Claude Sonnet 4.5 [25]. The design choices are deliberately minimal [26]:

- **File-based, client-side.** The model issues `view`, `create`, `str_replace`,
  `insert`, `delete` and `rename` commands against a virtual `/memories` directory. The
  developer's handler executes them on any backend (disk, database, per-user prefix).
  Anthropic never stores the memory.
- **Built-in write/read protocol.** When the tool is enabled, the API adds an instruction
  to the system prompt: "ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE...
  ASSUME INTERRUPTION: Your context window might be reset at any moment." Memory works as
  a checkpoint, not only as a profile.
- **Paging.** `view` truncates text over 16,000 characters, and the model pages through
  long files with `view_range`.
- **Security is the integrator's job.** The docs recommend validating paths against
  traversal (`/memories/../../secrets.env`, URL-encoded variants), capping file sizes,
  expiring files that have not been accessed, and stripping sensitive data before writes.
- **Multi-session pattern.** An initializer session writes a progress log and a feature
  checklist. Each later session reads them first and updates them before it ends.

Anthropic's internal agentic-search evaluation reported a 39% improvement from the
memory tool plus context editing over baseline, and 29% from context editing alone. In a
100-turn web-search evaluation, context editing (automatically clearing stale tool
results) let agents finish workflows that otherwise failed from context exhaustion, while
cutting token use by 84% [25]. The memory tool also pairs with server-side
**compaction**: compaction summarizes the conversation, and memory holds what must
survive that summarization [26].

Anthropic's context-engineering guidance places memory among three long-horizon
techniques [28]:

- **Compaction**, which suits long back-and-forth sessions.
- **Structured note-taking**, which suits iterative work with milestones.
- **Sub-agents**, which suit parallel exploration. Each returns a condensed summary,
  often 1,000-2,000 tokens.

The guide's Pokémon example shows unprompted note-taking at work. Claude kept step
tallies over thousands of game steps, maintained maps and strategy notes, and resumed
after context resets by reading its own notes [28].

### 8.2 Claude Code: CLAUDE.md plus auto memory

Claude Code keeps two complementary stores, and both are loaded at session start [27]:

| | CLAUDE.md files | Auto memory |
|---|---|---|
| Writer | Human | Claude |
| Content | Instructions, rules | Learnings, patterns |
| Scope | Org / project / user | Per repository, shared across worktrees |
| Load policy | Every session; imports via `@path` (max 4 hops) | `MEMORY.md` index: first 200 lines or 25KB |

Auto memory lives in `~/.claude/projects/<project>/memory/`. It holds a `MEMORY.md`
index with one line per memory, plus topic files whose frontmatter records a `type`:
`user`, `feedback`, `project` or `reference` [27]. Only the index is pre-injected. Topic
files are read on demand with ordinary file tools. When the index nears its limit, the
harness reminds Claude to "keep one line per entry, move detail into topic files, and
merge or drop stale entries" [27].

The documented write policy is explicitly selective [27]:

- Claude skips anything it can derive from the code or git history, and anything
  CLAUDE.md already says.
- It does not save something every session; it saves only what would help in a future
  conversation.
- Subagents do not inherit the main conversation's auto memory and keep separate
  directories.

This design shows the **index-plus-lazy-load** pattern clearly: a bounded, always-present
table of contents with unbounded detail behind it.

## 9. Other Research Designs Worth Knowing

- **A-Mem** (Xu et al., NeurIPS 2025) applies the Zettelkasten method [32]. Each memory
  becomes a note with LLM-generated keywords, tags, a contextual description, an
  embedding and links to related notes. **Memory evolution** means a new note can update
  the descriptions and attributes of older linked notes. The authors report gains across
  six foundation models.
- **MemOS** (2025) treats memory as a schedulable system resource [33]. The paper
  packages memory in **MemCubes** whose payload is plaintext, activation (KV cache) or
  parametric (for example, LoRA) memory, with metadata for governance. This extends
  "memory" beyond retrieved text to cached activations and weights. By mid-2026 the
  open-source project presented MemCubes mainly as composable, permissioned knowledge
  bases and self-reported 88.83 on LoCoMo and 89.20 on LongMemEval [33][45].
- **MIRIX** (2025) splits memory across six typed stores (Core, Episodic, Semantic,
  Procedural, Resource and Knowledge Vault), each run by its own manager agent under a
  meta memory manager [34]. The paper reports 85.4% on LoCoMo, plus 35% higher accuracy
  than a RAG baseline on ScreenshotVQA with 99.9% less storage (self-reported).
- **LongMemEval** (Wu et al., ICLR 2025) is the benchmark most later systems cite [31].
  It has 500 questions over timestamped multi-session histories and tests five
  abilities: information extraction, multi-session reasoning, temporal reasoning,
  knowledge updates and abstention. The authors found that commercial assistants and
  long-context LLMs show about a 30% accuracy drop when memorizing information across
  sustained interactions. They propose session decomposition, fact-augmented key
  expansion and time-aware query expansion as design fixes. They also note that LoCoMo does not test recall of assistant-provided
  information or reasoning over updated user information [31].

## 10. Further Designs and 2024-2026 Developments

### 10.1 LoCoMo: the other standard benchmark

Most vendor numbers in this chapter are reported on **LoCoMo** ("Evaluating Very
Long-Term Conversational Memory of LLM Agents"; Maharana, Lee, Tulyakov, Bansal,
Barbieri, Fang; ACL 2024) [38]. It is small: ten very long, multi-session conversations,
annotated for question answering and event summarization (its dialogs can also be used
for multimodal dialog generation). Its size explains why scores saturate quickly and why
harness details (speaker roles, timestamps, judge prompts) swing results by many points,
as the Mem0/Zep dispute showed [15]. LongMemEval [31] and newer suites such as BEAM [9]
were partly motivated by these limits. When comparing systems, prefer the benchmark whose
question mix (knowledge updates, temporal reasoning, abstention) matches your workload.

### 10.2 Forgetting as a policy: MemoryBank

**MemoryBank** (Zhong et al., 2023; AAAI 2024) [39] made *forgetting* an explicit part of
the write/consolidation policy. Inspired by the Ebbinghaus forgetting curve, it lets the
system "selectively forget or reinforce memories based on their significance and the
passage of time": memories that are recalled get reinforced, and unused ones decay. Most
production systems in this chapter never forget on their own (Mem0's 2026 design is
explicitly append-only; Zep invalidates but keeps edges). Decay-based pruning remains the
main published alternative to unbounded growth, alongside the "expire files that have not
been accessed" advice in Anthropic's memory-tool documentation [26].

### 10.3 Procedural memory from experience: AWM and ReasoningBank

The systems above mostly store *facts about the user*. A parallel line of work stores
*how to do tasks*, which is CoALA's procedural memory [30]:

- **Agent Workflow Memory (AWM)** (Wang, Mao, Fried, Neubig, 2024) [40] induces reusable
  workflows (sub-routines with abstracted variables) from an agent's past trajectories,
  offline or online, and injects them into later tasks. The project reports a 35.6%
  success rate on WebArena, state-of-the-art at publication.
- **ReasoningBank** (Ouyang et al., Google, 2025) [41] distills generalizable reasoning
  strategies from both *successful and failed* trajectories, judged by the agent itself,
  retrieves them for new tasks, and writes new lessons back after each task. This is
  memory as a self-improvement loop, close in spirit to LangMem's prompt optimizer [17]
  and Letta Code's self-rewriting skills [3].

The design lesson: episodic logs become far more useful when a consolidation step
abstracts them into strategies, and failures are worth recording as well as successes.

### 10.4 Learned memory policies: MEM1 and Memory-R1

Every write policy so far is hand-designed: a prompt tells the LLM what to extract and
when to UPDATE or DELETE. In 2025 several papers trained the policy with reinforcement
learning instead:

- **MEM1** (Zhou et al., 2025) [42] trains an agent end to end to keep a *constant-size*
  internal state. At each turn it merges prior memory with new observations and discards
  irrelevant or redundant information, instead of appending to the context. On a
  16-objective multi-hop QA task, MEM1-7B reports 3.5x better performance and 3.7x less
  memory than Qwen2.5-14B-Instruct, and it generalizes beyond its training horizon.
- **Memory-R1** (Yan et al., 2025) [43] fine-tunes a *memory manager* that chooses
  ADD/UPDATE/DELETE/NOOP operations (the Mem0 operation set) and an *answer agent* that
  filters retrieved memories, both with outcome-based RL from a small amount of training
  data.

Learned policies are still research systems, but they show that the operation set
matters less than whether the policy is optimized for downstream answers rather than for
an extraction prompt's notion of "important".

### 10.5 Gemini: the third consumer design

Google's Gemini app added a "Personal context" setting in August 2025 that lets Gemini
learn from and reference past chats automatically, alongside **Temporary Chats** that are
not used for personalization [44]. It sits between the two designs in Sections 6 and 7:
automatic like ChatGPT's chat-history reference, with user controls to switch it off per
chat. Google has not published its internals, so treat any architectural description as
speculative. The broader point stands: by 2026 all three major consumer assistants ship
automatic cross-chat memory with an incognito escape hatch, which makes the security
lessons of Section 6.3 universal rather than vendor-specific.

## 11. Comparison

| System | Primary substrate | Write policy (who/when) | Reconciliation | Read policy | Public latency/cost signal |
|---|---|---|---|---|---|
| MemGPT (2023) [1] | Prompt tiers + recall log + vector archive | Agent, hot path, pressure-triggered | Agent edits working context; recursive summary on eviction | Agent pages in via search tools | DMR 92.5% (GPT-4) vs 32.1% baseline |
| Letta / Letta Code [2][3][5] | Memory blocks in git-tracked MemFS + message search | Agent hot path + sleep-time agent | Agent/sleeper rewrites blocks | Blocks always in prompt; recall subagent | Sleep-time: ~5x less test-time compute [6] |
| Mem0 v1/v2 (2025) [8] | Vector facts (+ graph for Mem0g) | Pipeline after each message pair | LLM ADD/UPDATE/DELETE/NONE | Top-k retrieval per query | p95 total 1.44 s vs 17.1 s full context |
| Mem0 v3 (2026) [9] | Append-only facts + entity links | Single-pass extraction | ADD-only, `linked_memory_ids` | Fused semantic + BM25 + entity, time-aware | ~7K tokens, ~1 s p50 (self-reported) |
| Zep / Graphiti [12][13] | Bi-temporal KG: episodes, entities, communities | Pipeline per episode | Edge invalidation with validity intervals | Hybrid cosine/BM25/BFS + rerankers | ~1.6K vs 115K tokens; 2.58 s vs 28.9 s |
| LangMem [17] | LangGraph BaseStore namespaces | Hot-path tools or background manager | LLM consolidation; profiles overwrite | Semantic search + filters | Not published |
| ChatGPT [19][21] | Saved facts + generated history summary | Explicit ("remember") + automatic | Opaque | Pre-injected every turn | Not published |
| Claude apps [22][24] | Topics / summary + raw chat search | In-chat topic saves (new); 24 h synthesis (legacy) | User-editable topics | Profile injected; tool search on demand | Not published |
| Claude memory tool [26] | Developer-hosted `/memories` files | Agent via file commands | Agent `str_replace`/`delete` | Agent views directory first | +39% with context editing (internal eval) [25] |
| Claude Code auto memory [27] | `MEMORY.md` index + topic files | Agent, selective | Agent merges/drops stale entries | Index pre-loaded (200 lines/25KB); topics lazy | Not published |

## Key Takeaways for System Designers

1. **Design the write path first.** Read-side retrieval is mostly a solved RAG problem
   (hybrid search plus reranking, as in Chapter 5). The differences between memory
   systems come from what they store, when, and how they reconcile conflicts [8][12][17].
2. **Do not destroy history.** LLM-chosen UPDATE/DELETE silently loses facts. Zep's
   validity intervals and Mem0's 2026 ADD-only redesign converge on append, link, and
   resolve at read time with temporal ranking [9][12][14].
3. **Make time a first-class field.** Store both when a fact was true and when the system
   learned it. Resolve relative dates against the episode's timestamp. LongMemEval's
   temporal-reasoning and knowledge-update categories are where naive memories fail
   [12][31].
4. **Use a small always-on profile plus on-demand retrieval.** ChatGPT and Claude
   converged on this split from opposite starting points. Claude Code's 200-line index
   with lazy topic files is a concrete, bounded version [21][24][27].
5. **Move consolidation off the hot path.** Reflection, deduplication and summarization
   belong in background or sleep-time jobs. Reserve hot-path writes for facts the user
   explicitly asked to keep [5][6][17].
6. **Try files and good tools before building a bespoke store.** A filesystem agent hit
   74% on LoCoMo [7]. Anthropic's memory tool and Letta's MemFS are both file-shaped
   [3][26]. Files are inspectable, diffable and easy for users to edit.
7. **Scope memory aggressively.** Namespace by user, project and agent (LangMem
   namespaces, Claude project memories, Mem0 user/session/agent scopes). Cross-context
   leakage is both a quality bug and a privacy bug [9][17][24].
8. **Treat memory writes as a security boundary.** Automatic writes from untrusted
   content make prompt injections persistent (SpAIware). Validate paths, cap sizes, filter
   sensitive categories, show writes to the user, and allow pause and reset
   [24][26][35].
9. **Distrust cross-vendor leaderboards.** LoCoMo and LongMemEval scores for the same
   system differ by tens of points across vendor reports, and harness errors have been
   publicly conceded [11][15]. Run your own evaluation on your own conversation
   distribution, including knowledge-update and abstention cases.
10. **Budget the ingestion LLM.** Graph and fact extraction cost several LLM calls per
    episode and need structured-output-capable models [13]. Measure write-path cost and
    failure rate, not only read latency.

## References

1. C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, J. E. Gonzalez. "MemGPT: Towards LLMs as Operating Systems." arXiv, 2023. https://arxiv.org/abs/2310.08560
2. Letta. "Letta (f.k.a. MemGPT)" README. GitHub, 2026. https://github.com/letta-ai/letta
3. Letta. "Letta Code" README (MemFS, sleeptime, skills). GitHub, 2026. https://github.com/letta-ai/letta-code
4. Letta. "Memory Blocks." Letta blog, 2025. https://www.letta.com/blog/memory-blocks
5. Letta. "Sleep-time Compute." Letta blog, 2025. https://www.letta.com/blog/sleep-time-compute
6. K. Lin, C. Snell, Y. Wang, C. Packer, S. Wooders, I. Stoica, J. E. Gonzalez. "Sleep-time Compute: Beyond Inference Scaling at Test-time." arXiv, 2025. https://arxiv.org/abs/2504.13171
7. Letta. "Benchmarking AI Agent Memory: Is a Filesystem All You Need?" Letta blog, August 2025. https://www.letta.com/blog/benchmarking-ai-agent-memory
8. P. Chhikara, D. Khant, S. Aryan, T. Singh, D. Yadav (Mem0). "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory." arXiv, 2025. https://arxiv.org/abs/2504.19413
9. Mem0. "Mem0 - The Memory Layer for Personalized AI" README (April 2026 algorithm). GitHub, 2026. https://github.com/mem0ai/mem0
10. Mem0. `mem0/configs/prompts.py` (DEFAULT_UPDATE_MEMORY_PROMPT, ADDITIVE_EXTRACTION_PROMPT). GitHub, 2026. https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py
11. Mem0. "Benchmarked OpenAI Memory vs LangMem vs MemGPT vs Mem0 for Long-Term Memory." Mem0 blog, 2025. https://mem0.ai/blog/benchmarked-openai-memory-vs-langmem-vs-memgpt-vs-mem0-for-long-term-memory-here-s-how-they-stacked-up
12. P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, D. Chalef (Zep). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv, 2025. https://arxiv.org/abs/2501.13956
13. Zep. "Graphiti: A Framework for Building Temporal Knowledge Graphs" README. GitHub, 2026. https://github.com/getzep/graphiti
14. Zep. Graphiti source: `graphiti_core/search/search_config.py`, `search_config_recipes.py` and `graphiti_core/edges.py`. GitHub, 2026. https://github.com/getzep/graphiti/blob/main/graphiti_core/search/search_config.py
15. Zep. "Lies, Damn Lies, & Statistics: Is Mem0 Really SOTA in Agent Memory?" Zep blog, 2025. https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
16. LangChain. "LangMem" README. GitHub, 2025. https://github.com/langchain-ai/langmem
17. LangChain. "LangMem Core Concepts: Long-term Memory in LLM Applications." 2025. https://github.com/langchain-ai/langmem/blob/main/docs/docs/concepts/conceptual_guide.md
18. OpenAI. "Memory and new controls for ChatGPT" (with April 2025 update). OpenAI, 2024-2025. https://openai.com/index/memory-and-new-controls-for-chatgpt/
19. OpenAI Help Center. "How does Reference Saved Memories work?" OpenAI, 2025. https://help.openai.com/en/articles/11146739-how-does-reference-saved-memories-work
20. BleepingComputer. "ChatGPT rolls out memory upgrade for free users." 2025. https://bleepingcomputer.com/news/artificial-intelligence/chatgpt-rolls-out-memory-upgrade-for-free-users
21. M. Gupta. "I Reverse Engineered ChatGPT's Memory System, and Here's What I Found!" December 2025. https://manthanguptaa.in/posts/chatgpt_memory/ (summary: https://llmrefs.com/blog/reverse-engineering-chatgpt-memory)
22. S. Willison. "Claude Memory: A Different Philosophy" (commentary on S. Khemani). Simon Willison's Weblog, September 2025. https://simonwillison.net/2025/Sep/12/claude-memory/
23. S. Khemani. "Lessons from Studying Every Memory System." AI Engineer talk, 2026. https://ai.engineer/talks/lessons-from-studying-every-memory-system
24. Anthropic. "Claude memory" (Help Center article). Claude Help Center, 2025-2026. https://support.claude.com/en/articles/11817273
25. Anthropic. "Managing context on the Claude Developer Platform" (context editing and memory tool). September 2025. https://claude.com/blog/context-management
26. Anthropic. "Memory tool." Claude Platform documentation, 2026. https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
27. Anthropic. "How Claude remembers your project" (CLAUDE.md and auto memory). Claude Code documentation, 2026. https://code.claude.com/docs/en/memory
28. P. Rajasekaran, E. Dixon, C. Ryan, J. Hadfield (Anthropic). "Effective context engineering for AI agents." Anthropic Engineering, September 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
29. J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, M. S. Bernstein. "Generative Agents: Interactive Simulacra of Human Behavior." arXiv, 2023. https://arxiv.org/abs/2304.03442
30. T. R. Sumers, S. Yao, K. Narasimhan, T. L. Griffiths. "Cognitive Architectures for Language Agents." arXiv 2023 / TMLR 2024. https://arxiv.org/abs/2309.02427
31. D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, D. Yu. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." ICLR 2025. https://arxiv.org/abs/2410.10813
32. W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, Y. Zhang. "A-Mem: Agentic Memory for LLM Agents." NeurIPS 2025. https://arxiv.org/abs/2502.12110
33. MemTensor et al. "MemOS: A Memory OS for AI System." arXiv, 2025. https://arxiv.org/abs/2507.03724
34. Y. Wang, X. Chen. "MIRIX: Multi-Agent Memory System for LLM-Based Agents." arXiv, 2025. https://arxiv.org/abs/2507.07957
35. The Hacker News. "ChatGPT macOS Flaw Could've Enabled Long-Term Spyware via Memory Function" (J. Rehberger's SpAIware). September 2024. https://thehackernews.com/2024/09/chatgpt-macos-flaw-couldve-enabled-long.html
36. A. Ghoshal. "Anthropic adds memory to Claude for Team and Enterprise plan users." Computerworld, September 2025. https://www.computerworld.com/article/4056366/anthropic-adds-memory-to-claude-for-team-and-enterprise-plan-users.html
37. Zep. "State of the Art in Agent Memory" (LongMemEval results). Zep blog, 2025. https://blog.getzep.com/state-of-the-art-agent-memory/
38. A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, Y. Fang. "Evaluating Very Long-Term Conversational Memory of LLM Agents" (LoCoMo). ACL 2024. https://arxiv.org/abs/2402.17753 (data: https://github.com/snap-research/locomo)
39. W. Zhong, L. Guo, Q. Gao, H. Ye, Y. Wang. "MemoryBank: Enhancing Large Language Models with Long-Term Memory." arXiv 2023 / AAAI 2024. https://arxiv.org/abs/2305.10250
40. Z. Wang, J. Mao, D. Fried, G. Neubig. "Agent Workflow Memory." arXiv, 2024. https://arxiv.org/abs/2409.07429
41. S. Ouyang et al. (Google). "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory." arXiv, 2025. https://arxiv.org/abs/2509.25140
42. Z. Zhou, A. Qu, Z. Wu, S. Kim, A. Prakash, D. Rus, J. Zhao, B. K. H. Low, P. P. Liang. "MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents." arXiv, 2025. https://arxiv.org/abs/2506.15841
43. S. Yan et al. "Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning." arXiv, 2025. https://arxiv.org/abs/2508.19828
44. Google. "Temporary Chats and new privacy controls in the Gemini app" (personal context). The Keyword, August 2025. https://blog.google/products/gemini/temporary-chats-privacy-controls/
45. MemTensor. "MemOS" README (2026 news and benchmark table). GitHub, 2026. https://github.com/MemTensor/MemOS
