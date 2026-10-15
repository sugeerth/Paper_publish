# 21. RAG System Design Interview Playbook: Enterprise Q&A, Support Bots, Code Assistants, and Web Answer Engines

## Overview

"Design a retrieval-augmented assistant" has become a standard prompt in ML, AI-engineer, and forward-deployed-engineer interviews. Typical versions include "design a system that answers employee questions from our internal wiki", "design a customer-support bot", "design context retrieval for a coding assistant", and "design a Perplexity-style answer engine" [1][2][3]. Prep platforms and candidate reports agree on what separates strong answers. Strong candidates treat retrieval as the hard part and generation as the easy part. They scope the corpus (size, change rate, formats such as scanned PDFs and tables) before naming any technology. They spend the closing minutes on evaluation and failure diagnosis: "a wrong RAG answer is either a retrieval failure or a generation failure, and the interview is mostly about proving you can tell which" [2][4][5].

This chapter is a playbook built on publicly documented production systems. It covers:

1. A reusable interview framework: the 45-60 minute arc, the requirement questions, and a canonical component diagram.
2. Back-of-envelope sizing formulas with worked numbers. These are illustrative arithmetic based on published prices and index-memory formulas.
3. Four reference designs, each anchored to real systems:
   - **Enterprise Q&A:** Glean, Microsoft 365 Copilot, Uber Genie, Dropbox Dash.
   - **Customer support:** LinkedIn's knowledge-graph RAG, DoorDash, Klarna, Intercom Fin.
   - **Code assistants:** GitHub Copilot, Sourcegraph Cody, Cursor, Augment.
   - **Web answer engines:** Perplexity on Vespa, plus the FreshLLMs line of research.
4. A comparison table, a bank of common follow-up questions with model answers, and takeaways.

The other chapters of this atlas go deep on components: chunking (Ch. 2), embeddings (Ch. 3), vector databases (Ch. 4), hybrid search (Ch. 5), evaluation (Ch. 11), and security (Ch. 16). This chapter covers how to assemble them under interview time pressure and defend the tradeoffs.

## 1. The interview framework

### 1.1 The 45-60 minute arc

One published mock-interview transcript for a corporate RAG platform uses checkpoints at roughly minutes 10, 30, 45, and 60 [3]. Its main advice is to connect each design decision back to a stated requirement or scale number rather than listing technologies. A workable arc:

| Minutes | Phase | What to produce |
|---|---|---|
| 0-8 | Requirements and scoping | Users, corpus, freshness, ACLs, latency SLO, quality bar |
| 8-15 | Back-of-envelope sizing | Chunks, index RAM, QPS, tokens per query, cost per query |
| 15-30 | High-level design | Offline ingestion path and online answer path, drawn separately |
| 30-45 | Deep dives (interviewer picks) | Retrieval quality, permissions, freshness, hallucination control |
| 45-55 | Evaluation and operations | Offline eval set, online metrics, LLM-as-judge, feedback loop |
| 55-60 | Follow-ups and evolution | Scale x10, multilingual, agentic multi-hop, long context vs RAG |

### 1.2 Requirement questions that change the design

Ask questions whose answers change the architecture. Generic questions do not earn credit.

- **Corpus:** How many documents, and how many tokens on average? What formats (HTML, PDF with tables, scans, slides, code, tickets)? What is the change rate? A static manual set and a constantly edited wiki need different ingestion pipelines [2].
- **Permissions:** Is the data public to all users or ACL-scoped per user or group? In enterprise settings this is the defining constraint. Educative's framing of enterprise RAG puts access control, citation traceability, and answer accuracy at the center [1].
- **Freshness:** Is minutes-stale acceptable (wiki), or must answers reflect seconds-old changes (code edits, stock prices, news)?
- **Latency SLO:** Time to first token versus full answer. Interactive chat typically tolerates 1-3 s to first token. IDE autocomplete tolerates far less.
- **Quality bar and risk:** Is a wrong answer an annoyance (internal wiki) or a compliance incident (payments support)? This determines whether you need a pre-delivery guardrail and human fallback.
- **Answer form:** Free text with citations, structured action (refund, ticket), or code edit?
- **Traffic:** Daily active users, queries per user per day, and peak-to-mean ratio.

### 1.3 The canonical component diagram

Almost every RAG design question reduces to two paths. Draw them separately; interviewers check that you know ingestion is a separate, asynchronous system [3].

```
OFFLINE / ASYNC INGESTION PATH
 sources --> connectors/crawlers --> parse & clean --> chunk --> enrich
 (wiki, tickets,  (incremental,       (OCR, tables,    (struct-   (title, path,
  code, web)       webhooks, ACLs)     dedupe)          aware)     context blurb)
                                                                  |
          +-------------------+------------------+----------------+
          v                   v                  v
     lexical index       vector index       metadata / ACL store
     (BM25 / inverted)   (HNSW / IVF / BQ)  (doc->groups, freshness)

ONLINE ANSWER PATH
 user --> auth --> query understanding --> hybrid retrieve --> fuse (RRF)
          |        (rewrite, condense,     (BM25 + dense,     |
          |         route, decompose)       ACL pre-filter)   v
          |                                              rerank (cross-enc.)
          |                                                   |
          +--> conversation memory          context assembly (top-k, order,
                                            dedupe, token budget)
                                                   |
                                                   v
                         LLM generation with citations --> guardrails --> user
                                                   |
                        logs, feedback, LLM-judge <-+--> eval sets, dashboards
```

Each box maps to a documented technique:

- **Contextual enrichment of chunks:** Anthropic's Contextual Retrieval prepends an LLM-generated context blurb to each chunk before embedding and BM25 indexing [6].
- **Rank fusion:** reciprocal rank fusion (RRF) combines lexical and dense result lists using ranks only, with k typically 60 [7].
- **Reranking:** a cross-encoder rescoring stage [6].
- **Retrieval-quality gating:** Corrective RAG (CRAG) uses a lightweight evaluator that decides whether to use, refine, discard, or supplement retrieved evidence with web search [8].

### 1.4 Numbers worth memorizing

Interviewers reward candidates who anchor choices in published data:

- **Contextual Retrieval [6]:** contextual embeddings cut the top-20 retrieval failure rate about 35% (5.7% to 3.7%). Adding contextual BM25 brings the reduction to about 49% (2.9%). Adding a reranker brings it to about 67% (1.9%). Gains vary by domain. A critique notes the reduction falls to about 41% on a vague-codebase category [9].
- **Long context vs RAG [10]:** the Google DeepMind / Michigan EMNLP 2024 study found long-context models outperform RAG on average when resourced sufficiently, but RAG is far cheaper because it shrinks input tokens. Their Self-Route method routes each query to RAG or long context based on the model's self-reflection, keeping comparable accuracy at much lower cost.
- **Lost in the middle [11]:** accuracy follows a U-shape over the position of relevant evidence in the prompt. It is best at the start or end and degrades as context grows. This argues for small, well-ordered contexts rather than stuffing.
- **Binary quantization [12]:** 1-bit embeddings cut memory 32x. Rescoring a larger binary candidate set with float32 query vectors preserves about 96% of retrieval quality.
- **RRF [7]:** score(d) = sum over lists of 1/(k + rank). k = 60 is the default in many engines.
- **The seven failure points [13]:** missing content, missed top-ranked documents, not in context (consolidation), not extracted, wrong format, incorrect specificity, incomplete answer. Barnett et al. also conclude that RAG robustness "evolves rather than is designed in", and that validation is only feasible during operation.

## 2. Back-of-envelope sizing toolkit

All numbers in this section are illustrative arithmetic. They combine published unit prices and index-memory formulas with assumed workloads. State your assumptions aloud in the interview.

### 2.1 Corpus to chunks

```
chunks = documents x avg_tokens_per_doc / (chunk_size - overlap)
```

OpenAI's hosted file search defaults to 800-token chunks with 400-token overlap. Overlap may be at most half the chunk size, and chunk size ranges from 100 to 4,096 tokens [14]. With a 50% overlap the effective stride is 400 tokens, so chunk count doubles compared with non-overlapping chunks. Say this when you justify a smaller overlap at scale.

### 2.2 Embedding cost

Third-party price trackers list text-embedding-3-small at about $0.02 per million input tokens, about $0.01 via batch, with 1,536 default dimensions [15]. Embedding cost is therefore usually negligible compared with LLM generation. The real ingestion costs are parsing (OCR, table extraction) and optional LLM enrichment. Anthropic estimated contextual chunk generation at roughly $1.02 per million document tokens with prompt caching [6][9].

### 2.3 Vector index memory

OpenSearch-style HNSW sizing for float32 vectors is approximately 1.1 x (4·d + 8·M) bytes per vector, where d is the dimension and M the graph degree [16]. Its binary-quantized variant uses 1.1 x (d/8 + 8·M) style terms. The exact constants differ by engine. ScyllaDB, for example, uses N x (D x B + 16m) x 1.2 [16].

Worked example (d = 1,536, M = 16): 1.1 x (6,144 + 128) ≈ 6.9 KB per vector, so about 6.8-6.9 GB per million vectors in RAM.

| Chunks | float32 HNSW (approx.) | Binary-quantized vectors only (d/8 = 192 B) |
|---|---|---|
| 1 M | ~6.9 GB | ~0.2 GB (+ graph links) |
| 100 M | ~690 GB | ~19 GB (+ graph links) |
| 1 B | ~6.9 TB | ~190 GB (+ graph links) |

The lesson: past roughly 100M chunks, a float32 in-memory HNSW is a multi-node, high-RAM deployment. Propose quantization (binary or product quantization) with float rescoring from disk [12], dimension reduction, or sharding by tenant.

### 2.4 Traffic and tokens

```
avg_QPS   = DAU x queries_per_user_per_day / 86,400
peak_QPS  ~ 3-10 x avg_QPS (state assumption)
input_tok = system_prompt + history + k_chunks x chunk_tokens + question
LLM_cost  = (input_tok x p_in + output_tok x p_out) x queries
```

Example: 50,000 employees x 5 questions/day = 250,000 queries/day, which is about 2.9 QPS on average and perhaps 30 QPS at peak. Each query carries 8 chunks x 500 tokens plus about 1,000 tokens of instructions and history, so about 5,000 input tokens. That totals about 1.25 billion input tokens per day. The point to make in the interview: generation tokens, not vector search, dominate cost and latency. The levers are fewer, better chunks (reranking), prompt caching of static prefixes, smaller models for routing and condensation, and caching frequent answers.

### 2.5 Latency budget (interactive chat, illustrative)

```
auth + ACL lookup           10-30 ms
query rewrite (small LLM)   100-300 ms   (skip when query is self-contained)
hybrid retrieval            20-100 ms    (BM25 + ANN in parallel)
rerank top-50..100          50-200 ms    (cross-encoder, GPU)
LLM time-to-first-token     300-1000 ms  (depends on model and prompt size)
guardrail check             50-500 ms    (cheap similarity first, LLM judge if needed)
```

Dropbox's Dash team describes a retrieval target of completing more than 95% of requests within roughly 1-2 seconds. They treat retrieval as the layer that shapes both speed and answer quality [17].

## 3. Design A: Enterprise Q&A assistant

### 3.1 Problem

"Employees should ask questions in natural language over wiki, drive, tickets, chat, and email, and get cited answers. They must never see content they lack permission to read."

### 3.2 Requirements that dominate

- **Permissions** are the hard constraint. Microsoft 365 Copilot's semantic index inherits Microsoft Graph security trimming, so users only receive content they can already open [18]. Microsoft's own ISE guidance for custom RAG over SharePoint copies document ACLs (Entra ID object IDs) into index fields at ingestion. It then filters on them at query time, and the ingestion app uses least-privilege Graph permissions (Sites.Selected) [19]. A practitioner warning makes the risk concrete: get ACLs wrong and "you've just built a single search index that ignores years of permission work in the source system" [20].
- **Heterogeneous connectors** with incremental sync. Glean describes the foundation as a robust, scalable crawler that connects to all enterprise data, together with its permission rules [21].
- **Mixed query types:** navigational ("the Q3 OKR doc"), factual ("what is our parental leave policy"), and people/expertise queries. Glean argues that vector search alone is insufficient. It combines a self-learning language model, lexical search, and a knowledge graph, with authority signals inspired by PageRank [21][22].

### 3.3 Architecture

```
 +-----------+   webhooks / polling   +-------------------+
 | SaaS apps |----------------------->| connector workers |--+
 | (wiki,    |   (per-source rate     | fetch content +   |  |
 |  drive,   |    limits, backoff)    | ACLs + activity   |  |
 |  tickets) |                        +-------------------+  |
 +-----------+                                               v
                       +----------------------------------------------+
                       | normalize -> chunk -> embed -> index         |
                       | per-chunk fields: tenant, acl_principals[],  |
                       | source, updated_at, author, doc_type         |
                       +---------------------+------------------------+
                                             |
         +---------------+-------------------+--------------+
         v               v                                  v
   inverted index   vector index (tenant-sharded)   knowledge graph
   (BM25)           with ACL filter fields          (people, docs, teams,
                                                     activity edges)
 ONLINE:
 user -> SSO -> expand principals (user + groups, cached) -> query rewrite
      -> parallel BM25 + ANN with filter acl_principals IN user_principals
      -> RRF -> rerank (+authority/recency/graph features) -> top 5-10
      -> LLM answer with inline citations -> post-check: cited doc still visible?
```

### 3.4 Key techniques and tradeoffs

- **Pre-filter versus post-filter for ACLs.** Pre-filtering (ACL principals as index filter fields) is safe and gives full recall within the user's visible set. It requires ANN engines that handle selective filters well, and an ACL change triggers re-indexing of metadata. Post-filtering (retrieve top-N, drop unauthorized) is simpler but leaks through timing and counts. It can also return empty results when the user sees only a small slice of the corpus. Third-party analyses of Glean describe it as filtering by permission from the start rather than retrieve-then-filter [23]. Treat that as commentary, not official documentation. Always add a final "can this user still read the cited doc?" check at render time, because ACLs change between indexing and query.
- **RAG over fine-tuning.** Glean argues fine-tuning is poorly suited to factual recall and permissioning, and recommends a centralized, permission-aware index with RAG [21]. Uber chose RAG for its on-call copilot Genie mainly for time to market, since fine-tuning needs curated examples and ongoing compute [24].
- **Hybrid retrieval plus reranking** is table stakes. Use the 35% / 49% / 67% failure-reduction ladder from Contextual Retrieval to justify each added stage [6].
- **Agentic escalation for hard questions.** Dropbox Dash pairs RAG for fast single-shot answers with multi-step agents (planning plus a custom interpreter) for complex tasks [17]. Uber later moved its security and privacy domain to an "Enhanced Agentic RAG" design. After traditional RAG fell short on a curated 100+ query test set, it reported about 27% relative more acceptable answers and about 60% relative fewer incorrect-advice answers (as summarized secondhand) [25].

### 3.5 Scale and outcomes (public)

Uber Genie launched in September 2023. Secondary summaries of Uber's reports say it served 154 Slack channels and answered more than 70,000 questions, with a 48.9% helpfulness rate and an estimated ~13,000 engineering hours saved [24][25]. Take two lessons from these numbers. A useful internal assistant need not be "right" on every question. And helpfulness must be measured explicitly, because Genie built a user-feedback pipeline and LLM-as-judge evaluation [24].

### 3.6 Failure modes to name

- Permission leakage through stale ACLs, flattened group membership, or a non-delegated service connection. Microsoft's guidance warns that a non-delegated Azure AI Search connection has no security trimming [18].
- Duplicate and outdated documents (five versions of a policy). Mitigate with recency/authority features and dedupe at ingestion.
- Missing content, the first of Barnett's seven failure points [13]. The system should say "I couldn't find this" rather than synthesize an answer.
- Tables and scanned PDFs parsed badly. This is a corpus-scoping question to raise early [2].

## 4. Design B: Customer support bot

### 4.1 Problem

"Deflect a large share of inbound support conversations with accurate, policy-compliant answers. Hand off to humans gracefully."

### 4.2 What is different from enterprise Q&A

- The corpus is smaller and curated: help-center articles, macros, resolved tickets. Permissions are simpler, usually per-tenant or per-customer-account.
- Risk is higher. Wrong refund or policy statements create financial and compliance exposure, so pre-delivery guardrails and escalation are core components.
- Conversations are multi-turn. The retrieval query must be distilled from the whole dialogue, not taken from the last message.
- Actions matter: order lookup, refunds, account changes. The bot needs tools as well as retrieval.

### 4.3 Reference architectures

**DoorDash Dasher support [26].** There are three parts:

1. A RAG system that condenses the multi-message conversation into the core issue, then retrieves knowledge-base articles and similar previously resolved cases.
2. An online guardrail. Every response must pass all checks before the Dasher sees it. Summaries describe this as two tiers: a cheap semantic-similarity check, then an LLM evaluator [26].
3. An offline LLM judge that scores retrieval correctness, response accuracy, grammar, coherence with context, and relevance to the request.

DoorDash reports that the guardrail reduced hallucinations by about 90% and severe compliance issues by about 99%. When guardrail latency is a problem, the system defaults to human agents rather than trade off quality [26].

**LinkedIn customer-service KG-RAG [27].** Instead of chunking historical tickets as flat text, LinkedIn builds a knowledge graph from them. The graph preserves each ticket's internal structure (fields such as summary, description, steps to reproduce) as a tree, and links tickets to each other via explicit references and embedding similarity. At query time, the system parses the customer query into entities and intents and retrieves matching ticket sub-graphs. It then generates the answer from those sub-graphs. The paper (SIGIR 2024) reports a 77.6% MRR improvement over a flat-text baseline. After roughly six months in production it reports a 28.6% reduction in median per-issue resolution time.

**Klarna [28].** In its first month, Klarna's assistant handled 2.3 million conversations, about two-thirds of its customer-service chats. Klarna equated that to the workload of 700 full-time agents. It operates in 23 markets and more than 35 languages. Klarna reported customer satisfaction on par with humans and a 25% drop in repeat inquiries, with resolution time down from about 11 to under 2 minutes. These are company-reported launch numbers.

**Intercom Fin [29].** Third-party guides describe Fin as a bespoke multi-phase RAG pipeline: query refinement, retrieval, reranking, generation, and validation. They say it uses custom retrieval and reranker models trained on customer-service interactions; the phase counts differ between sources. Anthropic's case study reports a 51% average resolution rate across Intercom's customers, and up to 86% for some [29]. Note the metric definition. Guides point out that "assumed" resolutions (the customer leaves after the last answer) count alongside confirmed ones, so headline rates may overstate satisfaction [29]. This makes a strong follow-up talking point about metric hygiene.

### 4.4 Architecture

```
 customer msg --> session store (history, account ctx, locale)
       |
       v
 intent + condense (small LLM): "core issue" + entities (order id, product)
       |
       +--> policy router: sensitive intent? (fraud, legal, safety) --> human
       |
       v
 retrieval: help-center (hybrid) + resolved-ticket KG / similar cases
       |                                  + tool calls (order status API)
       v
 rerank --> grounded generation (answer + citations + proposed action)
       |
       v
 guardrail tier 1: similarity(answer, retrieved evidence) >= tau ?
 guardrail tier 2: LLM judge for policy / grounding (only if tier 1 uncertain)
       |pass                                 |fail or timeout
       v                                     v
   send to user                       escalate to agent with summary
       |
       v
 offline: LLM-judge sampling, CSAT, resolution, repeat-contact, reopen rate
```

### 4.5 Sizing sketch (illustrative)

At Klarna-like volume (2.3M conversations per month), that is about 77k conversations per day, or about 0.9 conversations per second on average. If each conversation has about 4 bot turns, that is roughly 3.5 LLM turns per second on average, and an order of magnitude more at peak. The knowledge base is typically thousands to tens of thousands of articles plus millions of resolved tickets. The vector index fits on a single node even in float32 (Section 2.3). The cost drivers are LLM turns and guardrail calls, so make the expensive LLM judge conditional on the cheap check.

### 4.6 Failure modes

- Retrieval keyed on the last message instead of the condensed issue. DoorDash condenses first for this reason [26].
- A policy article that is outdated but faithfully cited. Faithfulness metrics such as RAGAS measure consistency with context, not truth [30].
- Over-deflection: optimizing resolution rate leads to hiding the human-handoff path. Track repeat contacts and CSAT alongside deflection [28][29].
- Guardrail latency spikes. Use a timeout that falls back to a human, as DoorDash does [26].

## 5. Design C: Code assistant (autocomplete and chat)

### 5.1 Problem

"Give completions and chat answers that use the developer's codebase, not just the open file, without leaking code and under tight latency."

### 5.2 What is different

- **Two products, two latency classes.** Inline completion must feel instant and runs on every keystroke pause. Chat can afford seconds.
- **Extreme freshness.** The "document" changes as the user types, and per-branch state differs per developer.
- **Code structure** (symbols, imports, call graphs) is a stronger signal than semantic similarity of prose.
- **Security.** Source code is often the most sensitive asset a company has.

### 5.3 Reference techniques

- **GitHub Copilot [31].** The prompt is assembled from the current file and "neighboring tabs": other open files are searched for snippets similar to the code around the cursor. A/B tests found a very low inclusion threshold worked best and gave about a 5% relative increase in accepted suggestions. Fill-in-the-middle (sending code both before and after the cursor) gave about a 10% relative boost. Caching let both run without added latency.
- **Sourcegraph Cody [32][33].** Cody frames context retrieval as recommender-style candidate generation followed by ranking. The system must balance recall and precision because only a few items fit. A larger context window does not fully solve this, since it raises inference cost and big codebases still overflow it [33]. Retrievers include keyword search with query rewriting, Sourcegraph code search (with a BM25 variant among its ranking signals), and code-graph context, plus snippets from open IDE files [32]. Sourcegraph also tested a 1M-token Gemini 1.5 Flash context and reported a substantial improvement on its "Essential Recall" benchmark [34]. That result shows long context and retrieval being combined, not one replacing the other.
- **Cursor [35].** Cursor builds a Merkle tree over the workspace: each file is hashed, and each folder's hash derives from its children. Client and server compare trees and sync only the entries that diverge. For a 50,000-file workspace, sending all filenames and SHA-256 hashes would be about 3.2 MB per update, which is what the Merkle comparison avoids. Changed files are split into syntactic chunks and embedded asynchronously, with embeddings cached by chunk content. To reuse a teammate's index safely, the client uploads its own Merkle tree with a similarity hash, and the server filters results against that tree, so users only see code they actually have [35].
- **Augment [36].** Augment claims index updates within seconds of edits (versus "10-minute" delays elsewhere) and custom retrieval models rather than generic embeddings. It keeps per-developer indices to handle branch switching, and shares overlapping index segments between users to control RAM. These are vendor claims.

### 5.4 Architecture

```
 IDE plugin                                   server
 +-----------------------------+              +-------------------------------+
 | cursor ctx (prefix/suffix)  |  completion  | low-latency code LLM (FIM)    |
 | open-tab snippet matcher    |------------->| prompt cache                  |
 | local symbol index (LSP)    |  <~ hundreds |                               |
 | Merkle tree of workspace    |  of ms total +-------------------------------+
 +-------------+---------------+
               | diff sync (changed hashes only)
               v
 +---------------------------------------------------------------+
 | chunk by syntax (functions/classes) -> embed (async, cached   |
 | by content hash) -> per-repo/branch vector index + code search |
 | + symbol/code graph                                           |
 +-----------------------------+---------------------------------+
                               |  chat / agent queries
                               v
 retrievers: keyword(+rewrite) | embeddings | graph (defs/refs) | open files
         -> rank (recsys-style) -> pack under token budget -> chat LLM
         -> answer/edit; agent may iterate (grep, read file, run tests)
```

### 5.5 Sizing sketch (illustrative)

Take a 100,000-file monorepo averaging 300 lines per file: 30M lines. With function-level chunks of about 40 lines, that is about 750k chunks, roughly 5 GB of float32 HNSW at 1,536 dimensions using the formula in Section 2.3. Per-branch indices multiply this, which is why reusing shared content-addressed chunks and storing only branch deltas matters [35][36]. Re-embedding cost is driven by churn, not repo size, so content-hash caching is the key lever.

### 5.6 Failure modes

- Retrieving semantically similar but wrong-version code (other branch, deprecated module).
- Long-context overconfidence: stuffing the repo and then "losing in the middle" [11].
- Leakage across repos or users through shared caches. Cursor's server-side filtering against the client's own tree shows the mitigation [35].
- Evaluating with generic benchmarks instead of product-specific ones. Cody built its own context-recall benchmarks [33][34].

## 6. Design D: Web answer engine

### 6.1 Problem

"Answer open-domain questions from the live web with citations, at consumer scale and latency."

### 6.2 What is different

- **The corpus is the web:** billions of documents, adversarial SEO, and continuous change.
- **Freshness** is a first-class requirement. FreshLLMs showed all tested LLMs struggle with fast-changing and false-premise questions [37].
- **Scale.** Perplexity's research post describes a production search system handling about 200 million daily queries, built on hybrid retrieval, multi-stage ranking, distributed indexing, and a self-improving content-understanding module [38]. That works out to an average of roughly 2,300 QPS (my arithmetic).
- **Granularity.** Perplexity argues that a search API for AI should treat sections and spans of documents as retrievable units, not only whole pages [38].

### 6.3 Reference architecture: Perplexity on Vespa

Perplexity moved its search in-house onto Vespa (announced April 2025). Its stated requirements were: combine keyword and semantic search, filter, rank individual passages, and index new content continuously at consumer latency. It wanted all of this in one platform rather than stitched-together databases, vector stores, and ranking services [39]. Vespa's materials also highlight distributed machine-learned ranking inference that combines many relevance signals [39]. Perplexity published an open evaluation framework (search_evals) and reported a median latency of about 358 ms for its Search API. These are company-run benchmarks [38].

The method lineage is public. FreshPrompt injects search-engine results (answer box, organic results, related questions) into the prompt, ordered by date. With GPT-4 it gave about +49 points of absolute accuracy on FreshQA under strict evaluation. Both the number and the order of evidence snippets mattered, and asking for concise answers reduced hallucination [37]. Trade press has described Perplexity's "online LLMs" as inspired by this work [40].

```
  crawler fleet --> fetch queue (priority by freshness/authority)
       |                       ^
       v                       | recrawl signals (news, change rate)
  parse + boilerplate removal + spam/quality classifiers
       |
  split into sections/spans --> per-span: text, embedding, signals
       |
       v
  distributed index (sharded by doc id), lexical + vector + attributes
 ----------------------------------------------------------------------
 ONLINE (~hundreds of ms budget for retrieval):
 query --> classify (navigational / fresh / complex) --> rewrite / decompose
       --> fan-out to all shards: hybrid retrieval of spans
       --> stage-1 cheap rank (per shard) --> merge top-N
       --> stage-2 ML ranker (many signals) --> stage-3 cross-encoder (top-K)
       --> LLM synthesize answer with numbered citations (streamed)
       --> complex queries: iterate (search -> read -> search) as agent
```

### 6.4 Tradeoffs to discuss

- **Build versus buy search.** A third-party search API ships quickly but limits control of ranking, freshness, and cost per query. Perplexity's move in-house shows the trajectory at scale [39].
- **Page versus span indexing.** Spans improve precision and context efficiency but multiply index size (10-100 spans per page) [38].
- **Freshness tiers.** Use a small, rapidly refreshed news/real-time index next to a large, slowly refreshed index, with the query classifier routing between them.
- **Single shot versus multi-hop.** Agentic research modes improve complex answers at several times the latency and cost. Route by query complexity, analogous to Self-Route [10].

### 6.5 Failure modes

- Citing low-quality or SEO spam sources. Quality classifiers and authority signals belong before the LLM.
- False-premise questions answered confidently [37].
- Citation-answer mismatch, where the claim is not supported by the cited span. Add a post-hoc attribution check.
- Fan-out tail latency across thousands of shards. Hedge requests and set per-shard timeouts.

## 7. Comparison of the four designs

| Dimension | Enterprise Q&A | Support bot | Code assistant | Web answer engine |
|---|---|---|---|---|
| Corpus size | 10^6-10^8 docs | 10^3-10^6 articles/tickets | 10^4-10^6 files per repo | 10^9+ pages |
| Hardest constraint | Per-user ACLs | Policy correctness, safety | Latency + freshness per keystroke | Scale + freshness + spam |
| Freshness need | Minutes-hours | Hours-days (policy changes) | Seconds | Minutes for news |
| Retrieval core | Hybrid + graph/authority [21] | Condense + hybrid / ticket KG [26][27] | Keyword + embeddings + code graph [32] | Hybrid span retrieval, multi-stage ML rank [38][39] |
| Guardrails | Render-time ACL recheck | Pre-delivery 2-tier guardrail, human fallback [26] | Repo/tenant isolation [35] | Source quality, attribution |
| Public outcome numbers | Genie: 70k+ Qs, 48.9% helpful [24] | DoorDash -90% hallucinations [26]; LinkedIn -28.6% resolution time [27] | Copilot +5% (tabs), +10% (FIM) acceptance [31] | ~200M queries/day, ~358 ms median [38] |
| Primary metric | Helpfulness, citation precision | Resolution, CSAT, repeat contacts | Acceptance rate, context recall | Answer accuracy, latency, freshness |
| Typical escalation | Agentic multi-step [17][25] | Human agent | Agent loop (grep/read/test) | Multi-hop research mode |

## 8. Common follow-up questions and strong answers

**"How do you know if a bad answer is a retrieval or a generation failure?"**
Log retrieved chunk IDs for every answer. Build a labeled set of question, gold passages, and gold answer. Measure retrieval recall@k against the gold passages separately from answer correctness given gold context. If the gold passage was not retrieved, the failure is retrieval (failure points 1-2 in [13]). If it was retrieved but the answer was wrong, the failure is consolidation, extraction, or generation (points 3-7). Interviewers report this diagnosis skill is the core of the evaluation discussion [4].

**"What metrics would you track?"**
- Offline: recall@k and MRR for retrieval (LinkedIn reported MRR [27]); faithfulness, answer relevance, and context relevance via RAGAS-style reference-free LLM judging [30]; context recall when references exist.
- Online: product metrics such as acceptance rate [31], resolution and CSAT [28][29], and helpfulness votes [24].
- Monitoring: LLM-judge sampling over production traffic [26].
- Caveat: faithfulness does not catch faithfully cited stale content [30].

**"Why not just fine-tune the model on our docs?"**
Fine-tuning is weak at factual recall. It cannot enforce per-user permissions, needs retraining whenever content changes, and gives no citations [21][24]. Fine-tune retrievers, rerankers, or style instead. Intercom reportedly trains its own retrieval and reranker models [29].

**"Context windows are 1M tokens now. Do we still need RAG?"**
Long context can win on accuracy when cost is no object. RAG is far cheaper, and hybrid routing (Self-Route) keeps most of the accuracy [10]. Long prompts also suffer position effects [11], and corpora still exceed windows [33]. Best answer: retrieval narrows the corpus, and long context lets you include whole documents or files rather than tiny chunks.

**"How do you handle a document whose permissions change?"**
Store ACL principals per chunk. Propagate ACL change events through the same incremental pipeline as content changes, updating only metadata with no re-embedding. Cache user-to-group expansion with a short TTL, and re-check access at render time [19][20].

**"The corpus grows 10x. What breaks first?"**
RAM for float32 HNSW (Section 2.3). Mitigate with binary or product quantization plus rescoring [12], tenant sharding, and tiered storage. Re-embedding on a model upgrade becomes a multi-day batch, so version the index and dual-write during migration.

**"How do you reduce hallucinations?"**
- Retrieval gating: CRAG-style evaluator [8], or abstain when the top rerank score is low.
- Explicit instructions to answer only from context and to cite.
- Concise answers [37].
- Pre-delivery guardrails [26].
- Human fallback for high-risk intents.
- Measure faithfulness continuously.

**"How do you make it faster and cheaper?"**
- Skip query rewriting for self-contained queries.
- Run BM25 and ANN in parallel.
- Rerank only the top 50-100 results.
- Send fewer, better chunks.
- Use prompt caching for static system prompts and document prefixes. Anthropic's contextual chunking cost estimate assumes caching [6].
- Use a small model for routing and condensation.
- Cache semantically equivalent frequent questions, keyed by ACL set.
- Stream tokens.

**"How do you handle multi-hop questions?"**
Decompose the query or iterate as an agent. Dropbox [17] and Uber [25] both moved complex queries to agentic flows. Bound the iterations, and route only complex queries there to protect latency.

**"How do you prevent prompt injection from retrieved documents?"**
Treat retrieved text as data:
- Delimit it clearly in the prompt.
- Strip active content at ingestion.
- Restrict tool permissions to the user's own scope.
- Require confirmation for side-effecting actions in support bots.
- Run output guardrails.

See Chapter 16 for detail.

## 9. Interview anti-patterns

- Drawing "user -> vector DB -> LLM" and stopping. There is no ingestion path, no ACLs, no reranker, and no evaluation [2][3].
- Choosing a vector database by brand before stating corpus size, QPS, or filter selectivity.
- Assuming 50% chunk overlap and float32 vectors at a billion scale without doing the RAM math (Section 2.3).
- Offering "use a bigger context window" as the answer to retrieval quality [10][11][33].
- Reporting a single headline metric (deflection, resolution) without its definition [29].
- Ignoring the "missing content" case. The system must be able to say it does not know [13].

## Key takeaways for system designers

1. **Draw two systems.** Asynchronous ingestion (connectors, parsing, chunking, enrichment, ACL capture) and the online answer path have different SLOs, failure modes, and scaling axes.
2. **The defining constraint picks the architecture:**
   - permissions for enterprise Q&A [18][19];
   - policy safety for support [26];
   - keystroke latency and freshness for code [31][35];
   - scale, freshness, and spam for the web [38][39].
   Name it in the first five minutes.
3. **Hybrid retrieval plus reranking is the default.** Published ablations show each stage cuts retrieval failures: about 35%, then 49%, then 67% cumulative [6]. RRF is the simple fusion baseline [7].
4. **Do the memory math.** Float32 HNSW costs about 6.9 GB per million 1,536-d vectors. Quantization with rescoring keeps about 96% quality at 1/32 the vector memory [12][16].
5. **Generation tokens dominate cost and latency.** Fewer, better-ordered chunks beat more chunks [11]. Route easy queries cheaply and hard ones to long context or agents [10].
6. **Guardrails and fallbacks are components, not afterthoughts.** DoorDash's two-tier guardrail with human fallback is the canonical pattern [26].
7. **Structure beats flat text when the data has it.** LinkedIn's ticket knowledge graph improved MRR by about 78% [27], code graphs help code assistants [32], and authority graphs help enterprise search [22].
8. **Evaluate retrieval and generation separately, offline and online.** Expect robustness to come from operation, not upfront design [13]. Uber, DoorDash, and LinkedIn all built judge-based evaluation pipelines [24][26][41].
9. **Read headline metrics critically.** Resolution and deflection rates depend on definitions [29]. Company benchmarks are self-reported [38][39].

## References

1. "Enterprise RAG: Problem Framing and Requirements" (Grokking the ML System Design Interview), Educative, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/enterprise-rag-problem-framing-requirements
2. "What interviewers actually grade in a RAG design round", techinterview.org, 2026. https://www.techinterview.org/post/3233476088/rag-system-design-interview/?format=md
3. "RAG mock interview transcript", System Design Space, 2025. https://system-design.space/en/chapter/rag-mock-interview-transcript
4. "What RAG evaluation interviews test beyond recall@k" (AI interviews tag), techinterview.org, 2026. https://www.techinterview.org/post/tag/ai-interviews/
5. "Design a Retrieval-Augmented Generation (RAG) system", Hello Interview community, 2025. https://www.hellointerview.com/community/questions/rag-system-design/cmoaemvhf1emv09adwqq9wk3f
6. "Introducing Contextual Retrieval", Anthropic, 2024. https://www.anthropic.com/news/contextual-retrieval
7. G. V. Cormack, C. L. A. Clarke, S. Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
8. S.-Q. Yan et al., "Corrective Retrieval Augmented Generation", arXiv:2401.15884, 2024. https://www.datacamp.com/tutorial/corrective-rag-crag (tutorial summary; paper: arXiv 2401.15884)
9. "Responding to Anthropic's Contextual Retrieval for RAG Apps... Why Context is NOT All you Need", Almond AI (Medium), 2024. https://medium.com/almond-ai/responding-to-anthropics-contextual-retrieval-for-rag-apps-why-context-is-not-all-you-need-af503985aa55
10. Z. Li et al. (Google DeepMind, Univ. of Michigan), "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach", EMNLP 2024 Industry. https://arxiv.org/abs/2407.16833v1
11. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", TACL 2024 (arXiv 2023). https://arxiv.org/abs/2307.03172v3
12. "Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval", Hugging Face and mixedbread.ai, 2024. https://huggingface.co/blog/embedding-quantization ; see also Sentence Transformers docs: https://sbert.net/examples/applications/embedding-quantization
13. S. Barnett et al., "Seven Failure Points When Engineering a Retrieval Augmented Generation System", CAIN 2024. https://arxiv.org/abs/2401.05856v1
14. "Dear all, I have the questions about the openAI assistant chunk size and chunk overlap of file search tool?", OpenAI Developer Community, 2024. https://community.openai.com/t/dear-all-i-have-the-questions-about-the-openai-assistant-chunk-size-and-chunk-overlap-of-file-search-tool/831342
15. "OpenAI Embeddings Pricing 2026", CostBench, 2026. https://www.costbench.com/software/embedding-apis/openai-embeddings/
16. "Faiss product quantization / binary quantization memory estimation", OpenSearch Documentation 3.x, 2025. https://docs.opensearch.org/3.1/vector-search/optimizing-storage/binary-quantization/ ; ScyllaDB vector search sizing: https://cloud.docs.scylladb.com/master/vector-search/vector-search-sizing.html
17. "Building Dash: How RAG and AI agents help us meet the needs of businesses", Dropbox Tech Blog, 2024. https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users
18. "Enhance AI responses by using Retrieval Augmented Generation" (Copilot Studio guidance), Microsoft Learn, 2025. https://learn.microsoft.com/en-za/microsoft-Copilot-studio/guidance/retrieval-augmented-generation
19. "SharePoint document-level access control for RAG", Microsoft ISE Developer Blog, 2026. https://devblogs.microsoft.com/ise/sharepoint-doc-level-access/
20. "Why Copilot without security trimming is just a very polite insider threat", DEV Community, 2025. https://dev.to/saschadev/why-copilot-without-security-trimming-is-just-a-very-polite-insider-threat-4jk2
21. "How to build an AI assistant for the enterprise", Glean, 2024. https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise ; "What sets great retrieval augmented generation apart", Glean: https://www.glean.com/blog/hybrid-vs-rag-vector
22. "Agentic reasoning: the future of work AI", Glean, 2025. https://www.glean.com/jp/blog/agentic-reasoning-future-ai
23. "Enterprise AI system design: Atlassian and Glean architecture", K. RT (Medium), 2025. https://medium.com/@kevinrt6911/enterprise-ai-system-design-atlassian-glean-architecture-a36d80f5bc7f
24. "Genie: Uber's Gen AI On-Call Copilot", Uber Engineering, 2024. https://www.uber.com/blog/genie-ubers-gen-ai-on-call-copilot
25. "Enhanced Agentic RAG for On-Call Engineering Support" (summary of Uber), ZenML LLMOps Database, 2025. https://zenml.io/llmops-database/enhanced-agentic-rag-for-on-call-engineering-support
26. "Path to high-quality LLM-based Dasher support automation", DoorDash Engineering, 2024. https://careersatdoordash.com/?p=12568 ; summary: https://www.zenml.io/llmops-database/llm-based-dasher-support-automation-with-rag-and-quality-controls
27. Z. Xu, M. J. Cruz et al. (LinkedIn), "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering", SIGIR 2024. https://arxiv.org/abs/2404.17723v1
28. "Klarna AI assistant handles two-thirds of customer service chats in its first month", coverage of Klarna announcement, 2024. https://the-decoder.com/klarna-says-openais-ai-agent-does-the-work-of-700-support-agents/ ; https://siliconcanals.com/klarna-unveils-ai-assistant/
29. "Intercom" customer case study, Anthropic, 2025. https://www.anthropic.com/customers/intercom ; "Intercom Fin AI explained", Macha: https://www.getmacha.com/blog/intercom-fin-ai-explained
30. S. Es, J. James, L. Espinosa-Anke, S. Schockaert, "Ragas: Automated Evaluation of Retrieval Augmented Generation", EACL 2024 demo (arXiv 2023). https://arxiv.org/pdf/2309.15217
31. "How GitHub Copilot is getting better at understanding your code", GitHub Blog, 2023. https://github.blog/ai-and-ml/github-copilot/how-github-copilot-is-getting-better-at-understanding-your-code/
32. "Cody Context" documentation, Sourcegraph, 2024. https://5.5.sourcegraph.com/cody/core-concepts/context
33. J. Hartman et al., "AI-assisted Coding with Cody: Lessons from Context Retrieval and Evaluation for Code Recommendations", RecSys 2024. https://arxiv.org/html/2408.05344v1 ; blog: https://sourcegraph.com/blog/lessons-from-building-ai-coding-assistants-context-retrieval-and-evaluation
34. "Supercharging AI Coding Assistants with Gemini Models' Long Context", Google Developers Blog, 2024. https://developers.googleblog.com/en/supercharging-ai-coding-assistants-with-massive-context/
35. "Securely indexing large codebases" (secure codebase indexing), Cursor, 2025. https://cursor.com/blog/secure-codebase-indexing
36. "A real-time index for your codebase: Secure, personal, scalable", Augment Code, 2025. https://www.augmentcode.com/blog/a-real-time-index-for-your-codebase-secure-personal-scalable
37. T. Vu et al., "FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation", Findings of ACL 2024. https://arxiv.org/abs/2310.03214v2
38. "Architecting and Evaluating an AI-First Search API", Perplexity Research, 2025. https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api ; coverage: https://infoq.com/news/2025/09/perplexity-search-api
39. "Perplexity at a Glance" case study and "Perplexity Partners with Vespa.ai to Bring its Search Function In-House", Vespa.ai, 2025. https://vespa.ai/case-studies/perplexity/ ; https://vespa.ai/perplexity-partners-with-vespa-ai/
40. "How Perplexity's Online LLM Was Inspired by FreshLLMs Paper", The New Stack, 2023. https://thenewstack.io/how-perplexitys-online-llm-was-inspired-by-freshllms-paper/
41. "5 Lessons From LinkedIn's First Foray Into GenAI Development", The New Stack, 2024. https://thenewstack.io/5-lessons-from-linkedins-first-foray-into-genai-development/
