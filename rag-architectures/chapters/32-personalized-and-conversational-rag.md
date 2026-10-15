# 32. Personalized and Conversational RAG: User Profiles, Per-User Scopes, Conversational Search, Voice Latency, Clarification and Shopping Assistants

## Overview

Chapter 18 covered long-term memory: how an assistant decides what to remember about a user and how it stores and recalls those memories. This chapter covers the retrieval problems around that memory store. They arise when the same question should get different answers for different users, or when the query is not a self-contained string but the latest turn of a conversation. Six design problems recur:

1. **Profile-as-corpus personalization.** The user's own history (past reviews, emails, papers, tweets, purchases) is treated as a small private corpus. Retrieval picks the few items that should condition generation. LaMP and LongLaMP are the reference benchmarks [1][5].
2. **Personal-knowledge retrieval inside a conversation.** TREC iKAT gives each simulated user a Personal Text Knowledge Base (PTKB) of natural-language statements. Systems must rank those statements and web passages at every turn and produce a grounded, cited answer [7][9].
3. **Context-dependent query understanding.** TREC CAsT (2019-2022) made conversational query rewriting and conversational dense retrieval standard components [11][13][14].
4. **Per-user indexes and access scopes.** Personal corpora have to be isolated from each other at scale. Vector databases now offer tenant shards, payload-partitioned tenants and partition keys for this [25][26][27].
5. **Retrieval under a voice latency budget.** Realtime speech-to-speech assistants (OpenAI Realtime, Gemini Live) must hide retrieval latency behind speech. They use filler preambles, supervisor delegation and non-blocking tool calls [19][20][21][22].
6. **Clarify or answer.** Mixed-initiative systems must decide when to ask a clarifying question instead of retrieving. Qulac, ClariQ, CAsT 2022's mixed-initiative subtask and the clarification steps in deep-research agents cover this decision [15][16][17][18].

The chapter ends with retrieval for recommendation and shopping assistants: tool-based conversational recommenders [28], product-search relevance data [30], and the public outlines of Amazon Rufus and Instacart's LLM search features [31][32]. Section 32.9 then surveys 2024-2026 benchmarks for personalized QA, evolving preferences and multi-turn enterprise RAG [34]-[38].

**Coverage note.** arXiv, vendor documentation sites (platform.openai.com, ai.google.dev), amazon.science, aboutamazon.com, tech.instacart.com and trecikat.com could not be reached from the research environment for this chapter, and web search was unavailable. Facts here therefore come mostly from primary GitHub sources: benchmark repositories, official topic files, baseline code, SDK source and official cookbooks. Paper-level claims are taken from the abstracts the authors reproduce in their repositories. The Rufus and Instacart sections are short and flagged, because their primary posts could not be re-read. Statistics on iKAT topics were computed for this chapter from the official topic JSON files [9] and were re-computed during fact-checking with the same results. Section 32.9 adds 2024-2026 benchmarks (LaMP-QA, PersonaMem, PrefEval, CORAL, MTRAG and MTRAG-UN), again described from their official repositories [34]-[38].

```
               +-------------------- per-user scope (tenant / partition) -------+
               |  profile items   PTKB statements   memories   purchase history |
               +--------^-----------------^--------------^-------------^--------+
                        | profile retrieval (LaMP)      | (Ch. 18)    | recsys tools
 user turn t --> [context resolver] --> rewritten query + selected profile facts
   ^              (CAsT rewrite /          |
   |               ConvDR encoder)         v
   |                              [shared corpus retrieval] --> rerank --> generate
   |                                       |                                 |
   |   [clarify-or-answer policy] <--------+ (ambiguity / low confidence)    |
   +------------- clarifying question  <---+     cited answer / spoken reply <+
```

## 32.1 Profile-as-corpus: LaMP

### Problem

A generic LLM writes the same headline, email subject or product review for everyone. LaMP ("When Large Language Models Meet Personalization", Salemi, Mysore, Bendersky and Zamani, 2023) treats personalization as a retrieval problem. Each input comes with the user's profile, a set of that user's past documents. The model must produce the output that user would produce [1][2].

### Benchmark structure

The repository describes LaMP as "seven personalized tasks, spanning across three classification and four text generation tasks", with "multiple entries for each user profile" [2]. The paper's tasks are citation identification, movie tagging (in the current version), product rating, news headline generation, scholarly title generation, email subject generation and tweet paraphrasing [1]. Email subject generation (LaMP-6) uses the Avocado corpus. Avocado is not public, so the repository ships only sample IDs and a script to rebuild that task for licensed users [2]. Test-set scoring goes through a leaderboard. Dev-set scoring uses `eval/eval_all.py` or `eval/eval_task.py` [2].

### Architecture

The baseline is a retrieve-then-prompt pipeline over the profile [2]:

```
  input x (e.g., article text)     user profile P_u = {d_1 ... d_n} (user's past docs)
          |                                   |
          +------> rank P_u by relevance to x: bm25 | contriever | recency
                                   |  (optional: prepend timestamps to items)
                                   v
                       top num_retrieved items --> prompt template --> LLM --> y_u
```

The rank step is offline. Profiles are pre-sorted per input (`--is_ranked`), and the LLM is trained or evaluated with the top `num_retrieved` items [2]. The paper reports that models "utilizing profile augmentation outperform their counterparts that do not factor in profile information" [1].

### Optimizing the profile retriever

Follow-up work from the same group trains the retriever with feedback from the generator [3]:

- **ROPG-RL.** Reinforcement learning "whose reward function is defined using any arbitrary metric for personalized generation" (for example accuracy, ROUGE-1 or ROUGE-L) [2][3].
- **ROPG-KD.** Knowledge distillation from the downstream LLM into the retriever [3].
- **RSPG (Pre/Post).** A retriever selector that picks, for each input, one option from a pool. The pool holds Contriever, ROPG-RL, ROPG-KD, recency, BM25 and "no retrieval". The selector runs either before or after generation [2][3].

The authors report "statistically significant improvements in six out of seven datasets" [3]. The design lesson is that no single profile retriever wins everywhere. Sometimes recency beats relevance, and sometimes adding no profile at all is best. A learned router over retrievers is cheap insurance.

### RAG vs per-user fine-tuning

A third paper compares retrieval augmentation with per-user parameter-efficient fine-tuning (one LoRA adapter per user) on the seven LaMP datasets [4]. On average, RAG-based personalization improved over the non-personalized LLM by **14.92%**, PEFT by **1.07%**, and the combination by **15.98%**. PEFT's benefit grows with the amount of user data, so the authors conclude that RAG "is a better choice for cold-start users" [4]. The repository's PEFT recipe stores one checkpoint per user and loads it at inference [2]. That design has a real serving cost: adapter storage and per-request adapter swaps, against a profile index lookup.

### Failure modes

- **Wrong signal type.** Topical relevance to the input is not always what personalization needs. Style, which is what headline and tweet tasks need, may be better captured by recent items than by similar ones. This is why recency is a first-class retriever in the code and why RSPG helps [2][3].
- **Cold start.** Users with few profile items gain little from either approach. PEFT degrades fastest [4].
- **Private data in training.** LaMP-6's licensing constraint is a reminder that the most valuable personalization data (email) is often the hardest to share or benchmark [2].

## 32.2 Long-form personalization: LongLaMP

LaMP's outputs are short: a tag, a rating, a title. LongLaMP ("LongLaMP: Personalized Long-text Generation", 2024) says personalized long-text generation "remains largely unexplored" and builds a benchmark of long-form tasks [5][6]. Per the paper, the tasks are personalized email completion, abstract generation, review writing and topic writing. Data is distributed through a Hugging Face dataset, and the Avocado-based email task again needs a rebuild script [5][6].

The reference pipeline reuses LaMP's machinery: `--retriever` from {bm25, contriever, recency}, `--use_profile`, `--is_ranked` and `--num_retrieved`. It also has scripts for fine-tuning with RAG and for zero-shot RAG evaluation [6]. Long outputs change what retrieval must provide. A headline needs a few stylistic exemplars. A 300-word review needs exemplars of structure, vocabulary and opinion patterns, which makes the right `k` and the ordering of profile items matter more. Long-form generation also makes evaluation harder, because ROUGE-style overlap against one reference text is a weak proxy for "sounds like this user".

## 32.3 Personal knowledge inside conversations: TREC iKAT

### Problem

The TREC Interactive Knowledge Assistance Track (iKAT, 2023-2025) builds on four years of CAsT. Its stated focus is "collaborative information seeking conversational agents which can tailor and personalize their response based on what they learn about and from the user". It supports "multi-path, multi-turn, multi-perspective conversations", where "the direction and the conversation that evolves depends not only on the prior responses but also on the user" [7]. The organizers were from the University of Amsterdam, the University of Glasgow and the University of Strathclyde [7].

### Task definition

The user's personal information is given as a **PTKB**: "a set of natural language sentences" provided at the start of the conversation [7]. At each turn the system must [7]:

1. Read the context: the fixed previous responses with provenance, plus the PTKB. Using future turns is forbidden.
2. Rank PTKB statements by relevance to the current information need. This is scored as a ranking task.
3. Extract or generate a response from one or more ranked passages, each with provenance.

The collection is a subset of ClueWeb22 licensed through CMU [7]. The 2025 validator requires passage IDs of the form `clueweb22-enNNNN-NN-NNNNN:k`, which identifies a passage within a ClueWeb22 document [10].

### Topic statistics (computed from the official files)

| Year | Test topics | Turns | PTKB size (min / median / max) | Turns with gold PTKB provenance | Responses ending in "?" |
|---|---|---|---|---|---|
| 2023 | 25 | 332 | 6 / 11 / 17 | 112 (about 34%) | 64 (about 19%) |
| 2024 | 17 | 218 | 13 / 16 / 22 | 95 (about 44%) | 72 (about 33%) |
| 2025 | 17 | 188 | 19 / 20 / 21 | 64 (about 34%) | 61 (about 32%) |

Source: `2023_test_topics.json`, `2024_test_topics.json` and `2025_test_topics.json` [9]. The last column counts both clarifying questions and follow-up offers, so it is only an approximate proxy for mixed initiative. Two trends stand out. PTKBs grew from about 11 to about 20 statements, so the statement-ranking problem got harder. And only a third or so of turns need any PTKB fact, so a system that always injects profile facts adds noise to most turns.

A 2025 example shows the problem. The topic "Acid reflux" comes with 21 PTKB statements, mostly irrelevant ("I enjoy 3d printing for fun", "I'm an Android fan"). Turn 1 ("Can you tell me some food good for acid reflux?") gets a clarifying response ("do you mean acid reflux of the stomach?"). Turn 3 ("How about the fruits I mentioned before?") can only be answered by finding the PTKB facts about mangoes, pomegranates, apples, kiwi and watermelon [9]. The 2023 topic "Finding a diet" has a gold resolved utterance that folds five PTKB facts into the query: vegetarian, soy allergy, lactose intolerance, limited exercise and frequent water intake [9].

### Reference baseline architecture (2023)

The organizers' baseline uses Pyserini with Hugging Face components [8]:

```
 context + utterance --> t5-base-canard rewriter (automatic runs)
                              |      or resolved_utterance (manual runs)
 PTKB --> cross-encoder ms-marco-MiniLM-L-6-v2 scores each (query, statement) pair
                              |  top num_ptkb statements appended to rewritten query
                              v
                 BM25 (or QLD), optional RM3, k=100..1000 over ClueWeb22 subset
                              v
                 cross-encoder rerank --> top num_psg passages (default 3)
                              v
            t5-base-finetuned-summarize-news ("summarize: ..."), max 250 tokens
                              v
           response + passage provenance + PTKB provenance (official JSON)
```

The official baseline runs are named `ret_bm25_rm3--type_{automatic|manual}--num_ptkb_3--k_1000--num_psg_3` [8]. A notable design choice: PTKB personalization is done by **query expansion**. The selected statements are concatenated onto the rewritten query before BM25. The same cross-encoder serves as both the PTKB ranker and the passage reranker [8].

### 2025: run types, output limits and LLM pipelines

The 2025 validator accepts three run types: `manual`, `automatic` and `generation-only`. Generation-only runs use an organizer-provided retrieval run, which separates generation quality from retrieval quality [10][9]. Each response must carry `citations` (a dict from passage ID to score) and `ptkb_provenance`. The validator trims responses to a 250 whitespace-token limit and caps references at 1,000 entries. It also deduplicates citations and drops citations to passages that were trimmed from the reference list [10]. The example baseline file is named `gpt41mini-bm25-minilm-llama70b-gpt41mini.jsonl` [9]. The name suggests an LLM rewriter, BM25 retrieval, a MiniLM reranker and LLM generation stages, but the file name is the only public description seen for this chapter. The baseline file also includes an organizer run `orga-bm25-personal` whose first response ends in dozens of repeated "Bye." tokens [9]. Small generators fail this way, and the length limit and citation checks exist to catch it.

### Lessons from iKAT

- **Treat profile selection as ranking with abstention.** Most turns need zero PTKB facts. The evaluation scores statement ranking separately, which rewards systems that can say "none" [7][9].
- **Personalize the query, not just the prompt.** The baseline shows that adding profile facts to the retrieval query changes which web passages come back. A vegetarian's diet query should retrieve different documents, not just be summarized differently [8].
- **Provenance at two levels.** Responses cite both web passages and PTKB statements. That is the right audit shape for any assistant that personalizes answers using stored user facts [10].

## 32.4 Context-dependent retrieval: TREC CAsT and its techniques

### The track

TREC CAsT ran from 2019 to 2022 to create "a reusable benchmark for open-domain information centric conversational dialogues" [11].

- **Year 1 (2019).** MS MARCO passages plus the TREC CAR paragraph corpus. 30 training topics, with incomplete judgments for 5 topics (about 50 turns), and 50 evaluation topics. Manually resolved rewrites of the evaluation turns were released to support "manual" runs [11][12].
- **Year 2 (2020).** 25 evaluation topics in automatic and manual variants, and a BM25 + BERT baseline run over raw, automatically rewritten and manually rewritten utterances [11].
- **Year 3 (2021).** 25 topics over Washington Post 2020, KILT Wikipedia and MS MARCO documents. Baselines and a web UI used BM25 with T5 reranking and a T5-CANARD rewriter that conditions on previous queries and result passages [11].
- **Year 4 (2022).** MS MARCO v2 documents, KILT and WaPo. It added an optional **mixed-initiative subtask** with a pool of "over 4000 candidate questions" for clarification and suggestion [11].

iKAT then took over CAsT's "multi-path" turn toward mixed initiative and added personas [7].

### Technique 1: explicit query rewriting

The standard cascade rewrites the turn into a self-contained query. For example, "What about its side effects?" becomes "What are the side effects of metformin?". The rewritten query then goes to an ad hoc retriever. CAsT released manual rewrites that act as an upper bound, and later baselines used `castorini/t5-base-canard` for automatic rewriting [11][8]. The rewriter's input matters. CAsT 2021's raw rewrite used both prior queries and result passages, because a user's follow-up often refers to something the system said, not something the user said [11].

### Technique 2: conversational dense retrieval (ConvDR)

ConvDR ("Few-Shot Conversational Dense Retrieval", SIGIR 2021) skips the text rewrite. A student query encoder reads the whole conversational context and is trained, with an MSE knowledge-distillation loss, to produce the embedding that an ad hoc ANCE teacher produces for the manual rewrite [13]. The passage index stays the ordinary ANCE index, so only the query side changes. Training is few-shot: five-fold cross-validation on CAsT topics [13].

```
 teacher: manual rewrite --> ANCE query encoder --> e*  (frozen)
 student: [turn_1 .. turn_t] --> ConvDR encoder --> e_t     loss = MSE(e_t, e*)
 serve:   e_t --> ANN over the same ANCE passage index (no rewrite step at runtime)
```

### Technique 3: LLM rewriting with aggregation (LLM4CS)

LLM4CS (EMNLP Findings 2023) prompts an LLM in three modes [14]:

- **REW**: rewrite only.
- **RTR**: rewrite, then generate a hypothetical response.
- **RAR**: rewrite and response together.

Each mode can add a chain of thought tailored to conversational intent. The method samples several rewrites or responses and combines their embeddings into one "search intent vector" with **MaxProb**, **Mean** or **self-consistency (SC)** aggregation, then runs dense retrieval with ANCE [14]. This is HyDE-style expansion (Chapter 6) adapted to conversation. Sampling multiple rewrites hedges against one bad coreference resolution.

### Failure modes common to conversational retrieval

- **Topic shift.** Rewriters that copy entities from earlier turns pollute queries after the user changes subject. CAsT topics deliberately include shifts [11].
- **Reference to system output.** Follow-ups that point at the assistant's last answer need the response text in the rewriter's input [11].
- **Rewrite latency.** An LLM rewrite step adds a full generation call before retrieval can begin. ConvDR-style encoders avoid this, which matters in voice (Section 32.6).

## 32.5 Per-user indexes and access scopes

Chapter 16 covers permission-aware retrieval (ACL pre-filter vs post-filter, token-enforced document security). Personalization adds a related but different problem: millions of small, private corpora (profiles, PTKB-like fact stores, uploaded files, chat histories). Each must be invisible to everyone else and cheap when idle.

### Isolation patterns in vector databases

**Weaviate: tenant equals shard.** "Each tenant is stored on a separate shard. Data stored in one tenant is not visible to another tenant." Every CRUD and query call must name the tenant. Tenants can be `ACTIVE`, `INACTIVE` (on disk, not loaded) or `OFFLOADED` (moved to cloud storage), plus transitional `OFFLOADING`/`ONLOADING` states. Auto-tenant creation works on batch inserts [25]. Weaviate's concepts page says tenant shards are lightweight: "50,000, or more, active shards per node", so about "1M concurrently active tenants with just 20 or so nodes" [25]. Two caveats are documented. Backups do not include offloaded tenants (inactive tenants are backed up only from v1.37.0, with backports to v1.35.17 and v1.36.10; earlier releases back up active tenants only). And TTL-based expiry deletions are skipped for inactive or offloaded tenants until the tenant becomes active again [25]. Deleting a tenant deletes its shard and all its objects, which gives a clean primitive for "forget me" requests [25].

**Qdrant: one collection, three tiers.** Qdrant advises against a collection per tenant ("each collection carries its own resource overhead") except for a few tenants that need strict isolation [26]. It offers instead:

- **Partition by payload.** A keyword payload index on a tenant field such as `group_id`, with `is_tenant=true` so that "vectors of the same tenant" are co-located and read "in a single sequential pass". Queries must filter on the tenant field. To keep many tenants from making indexing slow, the global HNSW can be disabled (`m: 0`) with per-group graphs built via `payload_m` (for example 16) [26].
- **User-defined sharding**, with a dedicated shard per large tenant [26].
- **Tiered multitenancy.** Small tenants share a fallback shard. Large tenants are promoted to dedicated shards through internal shard transfer that is "completely transparent for the application", and fallback routing means callers need not know which tier a tenant is in [26].

**Milvus: partition key.** Setting `partition_key_field` hashes entities into partitions by key value (64 by default, and the docs recommend keeping it). Milvus suggests this as one way to implement multi-tenancy [27].

| Pattern | Isolation strength | Idle-tenant cost | Best fit | Main risk |
|---|---|---|---|---|
| Collection/index per user | Strong (separate object) | High per-collection overhead [26] | Few, large, regulated tenants | Operational sprawl |
| Shard per tenant (Weaviate) | Strong; tenant name required on every call [25] | Low; INACTIVE/OFFLOADED states [25] | Millions of personal stores | Offloaded tenants missing from backups [25] |
| Payload-partitioned, `is_tenant` (Qdrant) | Logical; relies on filter [26] | Very low | Many small, similar tenants | One missing filter leaks data |
| Tiered fallback + promotion (Qdrant) | Mixed | Low for small tenants | Skewed tenant sizes | Promotion complexity |
| Partition key (Milvus) | Logical; hashed partitions [27] | Low | Many tenants, shared schema | Many tenants share each hashed partition, so isolation still relies on the key filter |

### Design guidance

- **Make the scope part of the query type, not a free-form filter.** Weaviate's requirement that every call name a tenant is safer than an optional `where user_id = ...` clause that a code path can forget [25]. Chapter 16 argues the same for ACLs.
- **Separate scopes by data class.** A personal assistant typically queries three or more scopes per turn: the user's private store, shared org or catalog content, and the public web. Each has different freshness, ACL and retention rules, and the merge step must keep provenance per scope, as iKAT's separate PTKB and passage provenance does [10].
- **Plan for deletion and cold users.** Tenant offload and tenant deletion match the lifecycle of personal data: most users are idle, and some ask to be forgotten [25].

## 32.6 Retrieval inside voice and realtime assistants

Chapter 17 of the agent atlas covers realtime voice architectures in depth. This section covers only what changes for retrieval.

### The constraint

A realtime session is a long-lived, stateful connection. The model streams audio, calls tools and handles interruptions "without restarting a fresh request on every turn" [19]. Retrieval is a tool call in the middle of a spoken turn. Any time it takes is dead air unless the design hides it.

### OpenAI Realtime: preambles, supervisors and async tools

- **Tool-call preambles.** OpenAI's Realtime prompting guide recommends the model "output at the same time" a short phrase announcing the tool call. This "helps mask latency". Sample preamble phrases can go into each tool's description ("Let me look up your account by {email} now.") [21]. The guide's prompt skeleton has a `# Context` section explicitly for "retrieved context, relevant info" [21].
- **Chat-supervisor delegation.** In `openai-realtime-agents`, a realtime chat agent (`gpt-4o-realtime-mini` in the diagram) handles greetings and information gathering. It must call `getNextResponseFromSupervisor` for "ANY factual, account-specific, or process-related questions", and it must always speak a filler phrase first [20]. The text supervisor (`gpt-4.1`) runs the retrieval tool (`lookupPolicyDocument` in the demo) and returns a cited message such as "[Family Plan Policy](ID-010)" [20]. The README's example shows about 2 s between the end of "give me a moment to check on that" and the start of the answer. It argues this still beats a stitched pipeline, "where response latency is often 1.5s or longer after a user has finished speaking", because the realtime model "responds to the user right away" [20].
- **SDK hooks.** The Agents SDK realtime runner exposes `async_tool_calls`, tool approval events, and history events that keep "the server-side conversation ... aligned with what the user actually heard" after interruptions [19]. Retrieved context that was never spoken is still in history. Spoken text cut off by barge-in is truncated.

```
 user speech --> realtime model --(filler: "One moment...")--> audio out
                      | tool: getNextResponseFromSupervisor(relevantContext)
                      v
               text supervisor (gpt-4.1) --> lookupPolicyDocument / RAG / APIs
                      | message with citations   (~2 s in the README example)
                      v
               realtime model reads the answer aloud --> audio out
```

### Gemini Live: non-blocking functions and scheduling

Google's Live API takes a different approach. Tools run asynchronously while the conversation continues:

- A function declared with `behavior: NON_BLOCKING` lets the system "handle function responses as they become available while maintaining the conversation". `BLOCKING` waits for the result [23]. The cookbook says NON_BLOCKING is the default on the "Gemini 3.8 Live" generation and is required for its Extended Thinking variant. In blocking mode, calls act as a FIFO queue that "blocks subsequent requests" [22].
- Each function response carries a `scheduling` value [22][23]:
  - **`INTERRUPT`**: stop the current speech and use the result now.
  - **`WHEN_IDLE`** (the default): finish speaking, then use it.
  - **`SILENT`**: add the result to context without triggering generation.
- `will_continue` turns a function call into a generator that streams several partial responses, which suits progressive retrieval: fast cache hits first, slower deep results later. The SDK notes that `will_continue` and `scheduling` apply only to NON_BLOCKING calls, and that `will_continue` is not supported in Vertex AI [23].
- Google's reference web console streams microphone, webcam and screen input over a WebSocket session [24]. Built-in tools include Google Search grounding, with `grounding_metadata` attached to responses, and code execution [22]. The SDK's `VertexRagStore` grounding config has a `store_context` flag that is "currently only supported for Gemini Multimodal Live API". With it, "Gemini will leverage it to automatically memorize the interactions ... and retrieve context when needed" for "ongoing and future interactions" [23]. That is personalization memory built into the voice transport. The SDK's file-search tool takes `top_k` and a `metadata_filter` [23], which is where a per-user scope filter would go.

### Patterns and numbers

| Mechanism | OpenAI Realtime | Gemini Live |
|---|---|---|
| Hide retrieval latency | Spoken preamble at tool-call time [21] | Non-blocking call; model keeps talking [22] |
| Heavy reasoning or RAG | Text supervisor model via a tool, ~2 s example [20] | Extended Thinking variant (NON_BLOCKING required) [22] |
| Result delivery control | Client sends tool output, then requests a response [19] | `scheduling`: INTERRUPT / WHEN_IDLE / SILENT [23] |
| Incremental results | Application-managed | `will_continue` generator responses [23] |
| Built-in web grounding | Not in the sources reviewed here | Google Search tool [22] |
| Built-in memory | Application-managed | `store_context` on Vertex RAG store [23] |

Retrieval-specific failure modes in voice:

- **Long retrieved text read aloud.** Retrieved passages must be compressed into 2-3 spoken sentences (the guide's "Length" rule) [21].
- **Stale answers interrupting.** An `INTERRUPT`-scheduled result that arrives after the user has moved on breaks the conversation. `WHEN_IDLE` or `SILENT` is safer for background enrichment [23].
- **Rewriting cost.** An LLM query rewrite before every retrieval eats the latency budget. Supervisor patterns move rewriting off the speaking model, and ConvDR-style encoders remove the extra step [13][20].

## 32.7 Clarifying-question policies

### Benchmarks

**Qulac** (Aliannejadi, Zamani, Crestani and Croft, SIGIR 2019) was the first offline framework for clarifying questions in open-domain conversational search [15]. It is built on TREC Web Track 2009-2012 topics:

- 198 topics: 141 faceted and 57 ambiguous.
- 762 facets: 577 informational and 185 navigational.
- 2,639 questions and 10,277 question-answer pairs.

Its motivating example: two users issue the same query, "dinosaur", get the same first clarifying question, then diverge based on their answers [15].

**ClariQ** (ConvAI3, SCAI 2020; EMNLP 2021) splits the problem into two research questions: **when** to ask (score clarification need from 1 to 4) and **what** to ask (select from a question bank or generate) [16]. Its training split has 187 topics and its validation split 50. Stage 2 added human-in-the-loop evaluation of the top three systems, judging both search usefulness and naturalness [16]. CAsT 2022 brought the "when/what" decision into ad hoc conversational retrieval with its 4,000-question pool [11]. iKAT gold conversations contain clarification turns, as in the acid-reflux example [9].

### Production policies in LLM agents

- **Open Deep Research (LangChain).** A `clarify_with_user` node runs before research when `allow_clarification` is on. A structured-output call returns `need_clarification`, `question` and `verification` [18]. Its prompt encodes a **one-question budget**: "If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one". It also says to ask about acronyms or unknown terms and never to re-ask for information already given [18].
- **OpenAI Deep Research.** In ChatGPT, an "intermediate model (like gpt-4.1)" asks clarifying questions before research. The API skips this step and expects "fully-formed prompts". The cookbook offers two developer-side substitutes: a prompt rewriter, or a lightweight clarifier that asks "3-6 questions that would most reduce ambiguity", must not "invent preferences", and must "avoid overasking" [17]. The cookbook names the cost of not clarifying: broad rewrites cause "verbosity, higher latency, and increased token usage" [17].

### A clarify-or-retrieve policy

```
 turn --> resolve context --> ambiguity / missing-slot detector (ClariQ-style 1..4)
              |                               |
              | low need                      | high need and clarification budget left
              v                               v
       personalize from profile        ask ONE targeted question
       (fill slots from PTKB/memory)    (facets from retrieval results or a bank)
              v                               v
          retrieve + answer            user answer --> merge into query --> retrieve
```

Design notes:

- **Use the profile before asking.** Personal facts can answer what a clarifying question would ask. A user known to be vegetarian should not be asked about diet type; iKAT's "Finding a diet" resolution does exactly this [9].
- **Use retrieval to generate facets.** Qulac's facets and ClariQ's question bank are proxies for what a system can learn from the spread of first-pass results. Ambiguity that shows up as divergent top passages is a good trigger [15][16].
- **Ration questions.** Limit clarification to about one round and a few questions [17][18]. In voice, a question costs a full turn of latency.

## 32.8 Retrieval for recommendation and e-commerce assistants

### Tool-based conversational recommendation: InteRecAgent

Microsoft's InteRecAgent (part of RecAI) wraps classical recommender components as LLM tools. It splits them into **query** (SQL over an item profile table), **retrieval** (SQL filters for hard constraints, item-similarity retrieval for "similar to Call of Duty") and **ranking** (popularity, similarity or preference schemas, with user `prefer` and `unwanted` lists) [28]. The README makes a key separation explicit: the retrieval module handles "what user wants", not "user's personal profile". Personalization enters at the ranking stage [28]. Options include **plan-first** execution (the LLM writes a tool plan and an executor runs it), **reflection** after answering, and **dynamic demonstration selection** for in-context examples [28]. RecAI also includes RecLM-emb, a retrieval-oriented embedding model for items, and an evaluator for LLM recommenders [29].

```
 user: "popular sports games like FIFA, not too pricey"
   LLM planner --> SQL tool (genre=sports, price<X) --> candidate set
              --> item-similarity tool (FIFA)      --> candidate set (merged)
              --> ranker(schema=preference, prefer=[...], unwanted=[...]) --> top-k
   LLM --> grounded recommendation with item facts from the profile table
```

### Product search relevance data: ESCI

Amazon's Shopping Queries Dataset labels query-product pairs as **Exact, Substitute, Complement, Irrelevant**, with up to 40 results per query. The data covers English, Japanese and Spanish. The reduced version has 48,300 queries and 1,118,011 judgments. The larger version has 130,652 queries and about 2.62 million judgments. The README's prose says 2,621,738, but its per-language table sums to 2,621,288 [30]. Its three tasks (ranking, four-way classification, substitute identification) [30] map directly onto shopping-assistant retrieval. A good assistant must tell an exact match from an acceptable substitute (out of stock) and from a complement (cross-sell). A single relevance score cannot carry those distinctions.

### Amazon Rufus (primary post not re-fetched; treat as approximate)

Amazon Science's post on the technology behind Rufus [31] describes:

- A custom shopping LLM trained mostly on shopping data: catalog, customer reviews, community Q&A and web information.
- Retrieval augmentation from sources such as reviews, catalog data, community Q&A and Amazon store APIs.
- Reinforcement learning from customer feedback to improve answers over time.
- Serving on AWS AI chips with a streaming response architecture.

Specific numbers (chip counts, latency targets, Prime Day traffic) could not be verified for this chapter and are omitted.

### Instacart (primary post not re-fetched; treat as approximate)

Instacart's 2024 post "Supercharging Discovery in Search with LLMs" describes LLM-generated related and complementary results in grocery search [32]. A secondary summary describes Instacart query understanding that combines RAG-style context engineering with a fine-tuned Llama-3-8B (secondary source) [33]. The common pattern is that the LLM does not replace the catalog retriever. It expands intent ("complements for taco night") into retrievable product queries, and inventory-aware ranking then picks real items.

### Lessons for commerce retrieval

- **Separate intent retrieval from personal ranking**, as InteRecAgent does. This keeps hard constraints (price, dietary, size) exact and lets personal preference reorder results [28].
- **Model substitutes and complements explicitly.** The ESCI labels are a ready-made training and evaluation target [30].
- **Ground every product claim in catalog or review retrieval.** Generated product facts are a liability. The Rufus outline relies on reviews, catalog data and Q&A as retrieval sources [31].

## 32.9 Newer benchmarks for personalized and multi-turn RAG (2024-2026)

LaMP, iKAT and CAsT remain the reference points, but several newer benchmarks probe gaps the earlier ones leave open. They cover personalized question answering, preferences that evolve across sessions, preferences buried in long histories, and multi-turn RAG over enterprise corpora.

### LaMP-QA: personalized long-form question answering

LaMP-QA (Salemi and Zamani, 2025) moves the LaMP line from writing tasks to personalized long-form answers. Questions come from three categories (Arts & Entertainment, Lifestyle & Personal Development, Society & Culture), with "over 45 subcategories in total". The authors report that "incorporating the personalized context provided leads to performance improvements of up to 39%" [34]. The repository's baselines show the retrieval shape. RAG-Personalization ranks each user's profile against the question with Contriever, once per dataset, and conditions generation on the top items. **PlanPers** first trains a planner on dataset labels, using 10 personal contexts, to decide what the answer should cover before generation [34]. Responses are scored by an LLM evaluator (Qwen2.5-32B-Instruct in the default script) rather than ROUGE [34]. That is a direct response to the long-form evaluation problem noted in Section 32.2.

### PersonaMem: evolving profiles across sessions

PersonaMem ("Know Me, Respond to Me", Jiang et al., 2025) tests whether a model can "infer evolving user profiles and generate personalized responses". Each sample is a persona with static attributes (for example demographics) and dynamic ones (evolving preferences), followed by multi-session chats on topics such as food recommendation, travel planning and therapy consultation. The data ships in three context-length versions: 32k, 128k and 1M tokens [35]. A PersonaMem-v2 release focuses on 1,000 more implicit user preferences [35]. The retrieval lesson complements iKAT. iKAT's PTKB is a fixed list of facts. PersonaMem's facts change over time, so a profile retriever must prefer the latest statement of a preference over older, contradicting ones. Recency and conflict resolution become part of profile retrieval.

### PrefEval: preferences stated many turns earlier

PrefEval (Zhao et al., 2025; Amazon Science repository) asks whether models follow a user preference stated earlier in a conversation when a later question tempts a generic answer. It includes explicit preferences and two implicit forms: choice-based conversations and persona-driven conversations [36]. The README's leaderboard for the Travel-Restaurants topic shows the gap clearly. With 10 intervening turns and no help, most listed models score below 0.1. Adding a reminder raises several to 0.9 or above. At 300 turns, even with a reminder, most models fall to 0.36 or below; o1-preview is the exception at 0.98 [36]. For RAG designers this argues for retrieving the stated preference and placing it next to the current question, instead of relying on the model to find it in a long transcript.

### CORAL: conversational RAG with citations and topic shifts

CORAL (Cheng, Mao et al., 2024; dataset updated in 2025) is a large conversational RAG benchmark built from Wikipedia. Its listed features are open-domain coverage, knowledge-intensiveness, free-form responses, topic shifts and citation labeling [37]. It scores three tasks separately: conversational passage retrieval, response generation and citation labeling [37]. Conversations are sampled with four flow strategies (abbreviated LDS, SIDS, STRW and DTRW), each with 1,800 training and 200 test conversations [37]. CORAL extends the CAsT setup to full generation with attribution, and its topic-shift flows test the rewriter failure mode described in Section 32.4.

### MTRAG and MTRAG-UN: human-written multi-turn enterprise RAG

IBM's MTRAG (Katsis et al., TACL 2025) is described as "the first end-to-end human-generated multi-turn RAG benchmark". It has 110 conversations turned into 842 evaluation tasks over four corpora: ClapNQ (Wikipedia), Cloud (technical documentation), FiQA (finance) and Govt (government), plus 200 synthetic conversations [38]. MTRAG-UN (Rosenthal et al., ACL Findings 2026) adds 666 tasks across six domains, including new Banking and Telco corpora. It targets UNanswerable, UNderspecified and NONstandalone questions and UNclear responses [38]. The pair served as training and evaluation data for SemEval-2026 Task 8 (MTRAGEval) [38]. MTRAG-UN's categories map onto three earlier sections of this chapter. Non-standalone questions need context resolution (32.4). Underspecified questions call for a clarify-or-answer policy (32.7). Unanswerable questions need the system to abstain when retrieval finds nothing.

| Benchmark | Year | Personal signal | Conversation | What it adds |
|---|---|---|---|---|
| LaMP-QA [34] | 2025 | Per-user profile | Single question | Personalized long-form QA, LLM-judged |
| PersonaMem [35] | 2025 | Static + evolving persona | Multi-session, up to 1M tokens | Tracking preference change |
| PrefEval [36] | 2025 | Stated or implied preferences | Up to 300 intervening turns | Preference following over long context |
| CORAL [37] | 2024-2025 | None | Multi-turn, topic shifts | Retrieval + generation + citation labeling |
| MTRAG / MTRAG-UN [38] | 2025-2026 | None | Human multi-turn, enterprise corpora | Unanswerable, underspecified, non-standalone turns |

## 32.10 Comparison of designs

| Design | Personal signal | Retrieval unit | Query side | Latency profile | Primary eval |
|---|---|---|---|---|---|
| LaMP RAG [1][2] | User's past documents | Profile item | Task input | Offline/batch | Task metrics (acc, ROUGE) |
| ROPG / RSPG [3] | Same | Profile item | Learned retriever / selector | Batch | Same, 6 of 7 significant |
| Per-user PEFT [4] | Same, in weights | None | n/a | Adapter swap per user | Weaker than RAG on avg [4] |
| LongLaMP [5][6] | Past long texts | Profile item | Task input | Batch | Long-form overlap |
| iKAT baseline [8] | PTKB statements | PTKB + ClueWeb22 passages | CANARD rewrite + PTKB expansion | Multi-stage | PTKB ranking, passages, response |
| ConvDR [13] | Conversation only | Passage | Context encoder (no rewrite) | One encoder pass | CAsT ranking |
| LLM4CS [14] | Conversation only | Passage | Multiple LLM rewrites, aggregated | Several LLM calls | CAsT ranking |
| Realtime supervisor [20] | Account data via tools | Policy docs / APIs | Supervisor LLM | ~2 s masked by filler | Product UX |
| Gemini Live async [22][23] | Optional stored context | Search / RAG store | Model tool call | Non-blocking | Product UX |
| InteRecAgent [28] | prefer / unwanted lists | Items (SQL, similarity) | LLM tool plan | Several tool calls | Rec. metrics |
| LaMP-QA PlanPers [34] | User profile | Profile item | Contriever ranking + learned planner | Batch | LLM-judged answer quality |

## Key takeaways for system designers

1. **Treat the user's history as a private corpus and learn how to retrieve from it.** BM25, dense retrieval and recency each win on different personalization tasks. A retriever selector, including a "no profile" option, gave significant gains on six of seven LaMP datasets [3].
2. **Default to RAG for personalization, and add per-user fine-tuning only for data-rich users.** Average gains were 14.92% for RAG vs 1.07% for PEFT on LaMP, and RAG is better for cold-start users [4].
3. **Selecting profile facts is ranking with abstention.** In iKAT, only about a third of turns have gold PTKB provenance, and PTKBs hold about 20 mostly irrelevant facts by 2025 [9]. Inject profile facts only when they are relevant.
4. **Personalize the retrieval query, not just the generation prompt.** Profile facts change which documents should be retrieved [8].
5. **Resolve conversational context before retrieval, and pick the mechanism by latency budget.** Use an LLM rewrite with sampling for accuracy [14], a distilled context encoder for speed [13], and always include the system's own previous response in the context [11].
6. **Make per-user scope a structural property of the store.** Use tenant shards or required tenant parameters, with idle-tenant offload and whole-tenant deletion [25][26][27].
7. **In voice, hide retrieval behind speech or run it in parallel.** Use preambles and supervisor delegation [20][21], or non-blocking tools with explicit scheduling [22][23]. Keep spoken answers short.
8. **Ask at most one round of targeted clarification, and check the profile first.** Clarification costs a turn, and LLM clarifiers over-ask unless the prompt budgets them [17][18].
9. **In commerce, keep hard constraints in structured retrieval and personal preference in ranking.** Model exact, substitute and complement relevance explicitly [28][30].
10. **Keep provenance per scope.** Cite web passages and personal facts separately so personalized answers can be audited and corrected [10].
11. **Retrieve preferences explicitly, and keep them current.** Models often miss preferences stated many turns earlier unless the preference is placed next to the question [36], and profiles change over time [35]. Retrieve the latest version of a preference instead of trusting the transcript.
12. **Evaluate on multi-turn RAG benchmarks that include unanswerable and underspecified turns** [37][38], not only on answerable, self-contained questions.

## References

1. Alireza Salemi, Sheshera Mysore, Michael Bendersky, Hamed Zamani. "LaMP: When Large Language Models Meet Personalization." 2023. https://arxiv.org/abs/2304.11406
2. LaMP Benchmark. "Codes for papers on Large Language Models Personalization (LaMP)" (RAG, ROPG, RSPG, PEFT code; evaluation). GitHub, 2023-2025. https://github.com/LaMP-Benchmark/LaMP
3. Alireza Salemi, Surya Kallumadi, Hamed Zamani. "Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation." SIGIR 2024. https://arxiv.org/abs/2404.05970
4. Alireza Salemi, Hamed Zamani. "Comparing Retrieval-Augmentation and Parameter-Efficient Fine-Tuning for Privacy-Preserving Personalization of Large Language Models." 2024. https://arxiv.org/abs/2409.09510
5. Ishita Kumar et al. "LongLaMP: Personalized Long-text Generation" (as cited by the authors' repository). 2024. https://www.arxiv.org/abs/2407.11016
6. LongLaMP Benchmark. "Long text Large Language Model Personalization (LongLaMP)" repository. GitHub, 2024. https://github.com/LongLaMP-benchmark/LongLaMP-Benchmark
7. M. Aliannejadi, Z. Abbasiantaeb, S. Chatterjee, J. Dalton, L. Azzopardi. "TREC Interactive Knowledge Assistance Track (iKAT)" repository and task description. University of Amsterdam IRLab, 2023-2025. https://github.com/irlabamsterdam/iKAT
8. TREC iKAT organizers. "Baseline Runs" (baseline_runs.py and README; Pyserini, t5-base-canard, MiniLM cross-encoder, T5 summarizer) and official baseline runs. 2023. https://github.com/irlabamsterdam/iKAT/tree/main/2023/scripts/baselines
9. TREC iKAT organizers. Test topics 2023, 2024 and 2025 (2023_test_topics.json, 2024_test_topics.json, 2025_test_topics.json) and 2025 baseline runs. 2023-2025. https://github.com/irlabamsterdam/iKAT/tree/main/2025/data
10. TREC iKAT organizers. "validate_trec_ikat25.py" (run types, response and citation limits, provenance fields). 2025. https://github.com/irlabamsterdam/iKAT/blob/main/2025/scripts/validate_trec_ikat25.py
11. J. Dalton, C. Xiong, J. Callan, M. Aliannejadi, L. Azzopardi, P. Owoicho, J. Trippas, S. Vakulenko. "TREC Conversational Assistance Track (CAsT)" website repository, Years 1-4. 2019-2022. https://github.com/daltonj/treccastweb
12. Jeffrey Dalton, Chenyan Xiong, Jamie Callan. "CAsT 2019: The Conversational Assistance Track Overview." 2020. https://arxiv.org/abs/2003.13624
13. Shi Yu, Zhenghao Liu, Chenyan Xiong, Tao Feng, Zhiyuan Liu. "Few-Shot Conversational Dense Retrieval" (ConvDR), SIGIR 2021; code at https://github.com/thunlp/ConvDR. Paper: https://arxiv.org/abs/2105.04166
14. Kelong Mao et al. "Large Language Models Know Your Contextual Search Intent: A Prompting Framework for Conversational Search" (LLM4CS). EMNLP Findings 2023. https://github.com/kyriemao/LLM4CS
15. Mohammad Aliannejadi, Hamed Zamani, Fabio Crestani, W. Bruce Croft. "Asking Clarifying Questions in Open-Domain Information-Seeking Conversations" (Qulac dataset). SIGIR 2019. https://github.com/aliannejadi/qulac
16. M. Aliannejadi, J. Kiseleva, A. Chuklin, J. Dalton, M. Burtsev. "ConvAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems (ClariQ)" (arXiv 2009.11352) and "Building and Evaluating Open-Domain Dialogue Corpora with Clarifying Questions" (EMNLP 2021). https://github.com/aliannejadi/ClariQ
17. OpenAI. "Introduction to Deep Research API" (clarifying questions vs prompt rewriting). OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb
18. LangChain. "Open Deep Research" (clarify_with_user node and clarify_with_user_instructions prompt). 2025-2026. https://github.com/langchain-ai/open_deep_research
19. OpenAI. "Realtime agents guide." OpenAI Agents SDK (Python) documentation, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/realtime/guide.md
20. OpenAI. "OpenAI Realtime Agents" (chat-supervisor and sequential handoff patterns; chatSupervisor agent config). 2025. https://github.com/openai/openai-realtime-agents
21. OpenAI. "Realtime Prompting Guide" (prompt skeleton, tool-call preambles). OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/Realtime_prompting_guide.ipynb
22. Google. "Gemini - Multimodal live API: Tool use" (Get_started_LiveAPI_tools). Gemini Cookbook, 2026. https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI_tools.ipynb
23. Google. "Google Gen AI Python SDK types" (Behavior, FunctionResponseScheduling, VertexRagStore.store_context, FileSearch). 2026. https://github.com/googleapis/python-genai/blob/main/google/genai/types.py
24. Google. "Live API - Web Console." Gemini, 2024-2026. https://github.com/google-gemini/live-api-web-console
25. Weaviate. "Multi-tenancy operations" and "Data structure" (tenant states, shards per node). Weaviate documentation, 2026. https://github.com/weaviate/docs/blob/main/docs/weaviate/manage-collections/multi-tenancy.mdx and https://github.com/weaviate/docs/blob/main/docs/weaviate/concepts/data.md
26. Qdrant. "Multitenancy" (partition by payload, is_tenant, user-defined sharding, tiered multitenancy). Qdrant documentation, 2026. https://github.com/qdrant/landing_page/blob/master/qdrant-landing/content/documentation/manage-data/multitenancy.md
27. Milvus. "Use Partition Key." Milvus documentation v2.4.x, 2024. https://github.com/milvus-io/milvus-docs/blob/v2.4.x/site/en/userGuide/use-partition-key.md
28. Microsoft. "InteRecAgent" README; paper "Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations" (arXiv 2308.16505). 2023-2025. https://github.com/microsoft/RecAI/blob/main/InteRecAgent/README.md
29. Microsoft. "RecAI: Leveraging LLMs for Next-Generation Recommender Systems." 2023-2025. https://github.com/microsoft/RecAI
30. Amazon Science. "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search." 2022. https://github.com/amazon-science/esci-data
31. Amazon Science. "The technology behind Amazon's GenAI-powered shopping assistant, Rufus." 2024 (not re-fetched for this chapter). https://www.amazon.science/blog/the-technology-behind-amazons-genai-powered-shopping-assistant-rufus
32. Instacart Tech. "Supercharging Discovery in Search with LLMs." 2024 (not re-fetched for this chapter). https://tech.instacart.com/supercharging-discovery-in-search-with-llms-556c585d4720
33. ZenML LLMOps Database. "Revamping Query Understanding with LLMs in E-commerce Search" (Instacart case; secondary). n.d. https://www.zenml.io/llmops-database/revamping-query-understanding-with-llms-in-e-commerce-search
34. Alireza Salemi, Hamed Zamani. "LaMP-QA: A Benchmark for Personalized Long-form Question Answering." 2025 (arXiv 2506.00137). Code and data: https://github.com/LaMP-Benchmark/LaMP-QA
35. Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, Dan Roth. "Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale" (PersonaMem; arXiv 2504.14225), and PersonaMem-v2 (arXiv 2512.06688). 2025. https://github.com/bowen-upenn/PersonaMem
36. Siyan Zhao et al. "Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs" (PrefEval; arXiv 2502.09597). 2025. https://github.com/amazon-science/PrefEval
37. Yiruo Cheng, Kelong Mao, Ziliang Zhao, Guanting Dong, Hongjin Qian, Yongkang Wu, Tetsuya Sakai, Ji-Rong Wen, Zhicheng Dou. "CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmentation Generation" (arXiv 2410.23090). 2024. https://github.com/Ariya12138/CORAL
38. Yannis Katsis, Sara Rosenthal et al. "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems." TACL 13, 2025 (doi 10.1162/TACL.a.19); Sara Rosenthal et al. "MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations." Findings of ACL 2026; SemEval-2026 Task 8 (MTRAGEval). https://github.com/IBM/mt-rag-benchmark
