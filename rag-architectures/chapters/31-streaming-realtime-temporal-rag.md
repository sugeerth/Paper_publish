# 31. Real-Time and Time-Aware RAG: Streaming Ingestion, Temporal Retrieval and Staleness Evaluation

## Overview

Most RAG systems quietly assume the corpus is fixed: documents are parsed, chunked, embedded
and indexed once, and every query reads the same snapshot. That assumption fails in three
common situations.

1. **The source of truth changes continuously.** Product catalogs, tickets, CRM rows, wiki
   pages, prices, on-call rotas. If the index lags the database, the system answers from
   the past without saying so.
2. **The question is about "now".** "Who is the current CEO?", "What did the Fed decide
   this week?", "Which alerts fired in the last hour?" Here the right document depends on
   when the question is asked, not only on what it says.
3. **The question is about a time in the past.** "Who coached the team in 2014?",
   "What did the runbook say before the March change?" Here the system has to pick the
   version of a fact that was valid at a stated time, and ignore newer versions.

This chapter treats these as one design space with three layers:

- **Freshness of the index**: change data capture (CDC), Kafka/Flink/Pathway-style
  streaming pipelines, and in-database "vectorizers" that keep embeddings in step with the
  source [14][15][16][17][19].
- **Time-awareness of retrieval and ranking**: recency decay functions, time-weighted
  retrievers, valid-time (bi-temporal) fact stores and versioned datasets
  [20][21][22][13][26].
- **Evaluation of staleness and temporal reasoning**: FreshQA, RealTime QA, TimeQA,
  SituatedQA, StreamingQA, TempLAMA, TempRAGEval, TemporalWiki and LongMemEval
  [1][4][6][8][9][10][11][30][31].

It closes with two application families where time is the main axis: news- and
market-feed RAG [25], and event-stream/log RAG for observability and SIEM copilots
[23][24].

A note on sourcing: claims below are tied to repository READMEs, documentation sources or
source code read directly. Where a paper's numbers could not be checked against its PDF,
the text says so and does not quote them.

```
          THE THREE LAYERS OF TIME IN A RAG SYSTEM

  +-------------------+     +---------------------+     +-------------------+
  |  SOURCE SYSTEMS   |     |  INDEX / KNOWLEDGE  |     |  QUERY + ANSWER   |
  |  OLTP DB, docs,   | --> |  vectors, BM25,     | --> |  retrieve, rank,  |
  |  feeds, logs      |     |  temporal KG, vers. |     |  generate, cite   |
  +-------------------+     +---------------------+     +-------------------+
        |                           |                          |
   (1) INDEX FRESHNESS        (2) TEMPORAL MODEL         (3) TEMPORAL INTENT
   lag from commit to         valid-time vs              "now" vs "as of T"
   searchable                 transaction-time           recency vs relevance
        \___________________________|__________________________/
                                    |
                      (4) STALENESS EVALUATION
              index lag SLOs, fast-changing QA sets, canaries
```

---

## 1. Why time breaks the default RAG pipeline

### 1.1 Failure taxonomy

Time-related RAG failures fall into a small number of types. Benchmarks in Section 5 test
most of them.

| Failure | Example | Root cause | Layer |
|---|---|---|---|
| Index lag | Price changed 10 min ago, answer quotes old price | Batch re-index, slow embed queue | Ingestion |
| Zombie documents | Deleted policy still retrieved | Deletes not propagated to vector store | Ingestion |
| Recency blindness | Retrieves a 2019 article for "latest guidance" | Pure semantic similarity ranks on topic, not date | Ranking |
| Over-recency | Retrieves yesterday's tweet over the canonical spec | Decay too steep, or applied to time-invariant queries | Ranking |
| Wrong version | Answers "as of 2014" with the 2022 holder | No valid-time model; latest version overwrote history | Data model |
| Parametric override | LLM ignores fresh context and answers from training data | Knowledge conflict (see Chapter 29) | Generation |
| False premise | "Why did X resign?" when X never resigned | Model accepts the premise | Generation |

FreshQA's design targets the last row on purpose: it includes false-premise questions
next to never-changing, slow-changing and fast-changing ones [1][2].

### 1.2 Two clocks: valid time and transaction time

Database theory separates two timestamps, and temporal RAG needs both:

- **Valid time**: when a fact was true in the world ("Alice was VP of Sales from
  2021-03 to 2024-07").
- **Transaction (system) time**: when the system learned or recorded the fact (the row
  was written on 2024-07-15; the vector was indexed at 10:02:13).

Recency ranking usually runs on one of these, sometimes on the wrong one. Zep's Graphiti
framework keeps both explicitly. Its README describes "explicit bi-temporal tracking with
automatic fact invalidation", where "old facts are invalidated, not deleted" so a caller
can "query what's true now, or what was true at any point in time" [13]. A news article
adds a third clock, publication time, which StreamingQA records as `evidence_ts` next to
the question's own `question_ts` [9].

---

## 2. Streaming ingestion into vector indexes

### 2.1 The CDC pattern: Debezium + Kafka as the change log

**Problem.** An application writes to Postgres/MySQL/Mongo. The RAG index must reflect
inserts, updates and deletes without the application doing "dual writes" to the database
and the vector store.

**Architecture.** Debezium describes itself as "a low latency data streaming platform for
change data capture". It watches a database's replication log and emits one event per
committed row-level change. "Only committed changes are visible, so your application
doesn't have to worry about transactions or changes that are rolled back" [14]. Each
connector runs inside Kafka Connect, watches one upstream server, and writes changes to
Kafka topics, "typically one topic per database table". Kafka gives total ordering per
table, so consumers that stop and restart "resume exactly where they left off" [14].

The Debezium README names "update search indexes" as a reason to avoid dual writes. If an
application crashes after the database commit but before the secondary write, the systems
fall out of sync. CDC moves that work into a separate consumer that cannot miss events
[14]. A vector index is one more such derived read model, in the CQRS sense the README
also describes [14].

```
   CDC -> STREAM -> EMBED -> VECTOR SINK (canonical streaming RAG ingestion)

  +-----------+  WAL/binlog  +-----------+   topic per table   +----------------+
  | Postgres  | -----------> | Debezium  | ------------------> | Kafka          |
  | (OLTP)    |              | (Connect) |   c/u/d events      | (ordered log)  |
  +-----------+              +-----------+                     +-------+--------+
                                                                       |
                         +---------------------------------------------+
                         v
              +------------------------+     +-------------------+
              | Stream processor       |     | Embedding service |
              | Flink / Pathway /      |<--->| (batched, cached, |
              | Bytewax: parse, chunk, |     |  rate-limited)    |
              | dedupe, key by doc_id  |     +-------------------+
              +-----------+------------+
                          | upsert(doc_id#chunk_i, vec, payload{ts,...})
                          | delete(doc_id#*) on tombstone
                          v
              +------------------------+       +----------------------+
              | Vector / hybrid index  | <---- | Query service (RAG)  |
              | Qdrant/Milvus/Mongo/ES |       | filters on ts fields |
              +------------------------+       +----------------------+
```

**Key techniques.**

- **Key everything by a stable document ID.** A changed row must replace all of its old
  chunks. Re-chunking a longer text can produce a different number of chunks, so the
  sink must delete `doc_id#*` before upserting, or keep a doc-to-chunk manifest.
- **Propagate tombstones.** Deletes are the change most often lost. Sink support
  varies. Qdrant's sink "supports streaming dense/sparse/multi vectors into Qdrant
  collections", requires the collection "to be created in advance with the appropriate
  vector dimensions", and does handle Kafka tombstones: on a record with a `null` value
  it deletes the point whose `id` appears in the record key [18]. Note that this deletes
  one point, so a document split into many chunks needs one tombstone per chunk ID (or a
  manifest-driven delete upstream). The Milvus sink requires the Kafka message schema to
  match the collection schema, "containing one and only one vector field" [18]. Check how
  each sink handles deletes before relying on it.
- **Carry timestamps as payload.** Each point should store at least `source_updated_at`
  (transaction time from the CDC event), `indexed_at`, and, where known, valid-time or
  publication-time fields. These feed the recency ranking and staleness metrics later.

**Failure modes.** Schema drift between topic and collection [18]; embedding-API outages
that back up the stream; cross-partition reordering of one document's chunks; lost
deletes; and re-embedding storms when a field outside the embedded text changes.

### 2.2 Flink-native RAG: Confluent Cloud streaming agents

Confluent's public quickstart shows a version where the whole RAG flow is a chain of Flink
SQL tables over Kafka topics [17]. Lab 2, "Vector Search & RAG", deploys a chain of Flink
tables (the walkthrough says six, but lists seven), named so they sort in pipeline order: `documents`,
`documents_embed`, `documents_vectordb_lab2`, `queries`, `queries_embed`,
`search_results` and `search_results_response`. It also deploys two registered models
(`llm_embedding_model`, `llm_textgen_model`) and a MongoDB sink connector that "stream[s]
embeddings from `documents_embed` to Atlas" [17].

The design choice that matters: **queries are events too.** A question "land[s] in the
`queries` topic, which immediately feeds into `queries_embed`", and search results and
answers appear as rows downstream [17]. Retrieval and generation become durable,
replayable, joinable streams, which helps auditing and offline evaluation at the cost of
per-hop latency and a less natural synchronous API. Other labs add MCP tool calling and
`CREATE AGENT` definitions [17]. Apache Flink now also has a Flink Agents sub-project
[19], and open-source Flink itself has gained native AI operators (Section 2.5) [29].

```
     QUERIES-AS-STREAMS (Confluent quickstart Lab 2 shape)

  documents --> documents_embed --> [Mongo Atlas sink] --> vector index
                                                               ^
  queries ----> queries_embed ----> search_results ------------+ (vector search)
                                         |
                                         v
                              search_results_response  (LLM answer + snippets)
```

### 2.3 Incremental dataflow engines: Pathway

**Problem.** Keep a vector plus full-text index live over files, SharePoint, Google Drive,
Kafka and Postgres, without running a separate vector DB, cache and API layer.

**Architecture.** Pathway is a Python framework whose code runs on "a scalable Rust engine
based on Differential Dataflow" that "performs incremental computation" [16]. Its LLM app
templates "connect and sync (all new data additions, deletions, updates)" with sources
including "Kafka, PostgreSQL, real-time data APIs". They build vector, hybrid and full-text
indexes "in-memory, with cache": usearch for vectors and Tantivy for full text [16]. The
live document-indexing template runs connectors that "poll the changes with low latency".
Each change is reflected in internal tables, then parsed with `unstructured`, chunked,
embedded and indexed. The service exposes `/v1/retrieve`, `/v1/statistics` (indexer
health) and `/v1/inputs` (files currently indexed) [16].

**Distinctive technique: answer-change alerting.** The Google Drive alert template goes
further than a fresh index. After answering a query, Pathway "monitors changes to
documents and selectively re-triggers potentially affected queries. If the new answer is
significantly different from the previously presented one, a new notification is
created" [16]. This is a standing query: the reverse of normal RAG, where a data change
triggers the answer. It suits compliance and deal-desk uses ("tell me if the campaign
start date changes").

**Consistency and scale claims.** Pathway "manages late and out-of-order points by
updating its results whenever new (or late...) data points come into the system". The
free version gives at-least-once consistency; the enterprise version gives exactly-once
[16]. The templates are said to scale to "millions of pages of documents" [16]. The
README also states Pathway is "made to outperform" Flink, Spark and Kafka Streams. That
is a vendor claim this chapter has not verified independently.

### 2.4 In-database vectorizers: pgai Vectorizer

**Problem.** Many RAG corpora already live in Postgres. Running Kafka and Flink just to
keep embeddings in sync is heavy.

**Architecture.** Timescale's pgai Vectorizer treats embeddings as "a declarative,
DDL-like feature, akin to an index". A single `ai.create_vectorizer(...)` call names the
source table, the column to embed, the destination table and the embedding model.
Embeddings then stay "continuously synchronized with source data (asynchronously)" [15].
The system has three parts: the application, the database (which holds a work queue)
and "stateless vectorizer workers" that "process the data queue into embeddings and
chunked text" and write results back [15].

**Key properties.**

- **Failure isolation.** Application writes only enqueue work; the background worker
  makes embedding generation "resilient to intermittent LLM endpoint failures" [15], so
  an embedding outage grows the backlog instead of failing the source transaction.
- **Tunable freshness.** Each worker drains the queue until it is empty. It then sleeps
  for a configurable `--poll-interval` (default five minutes) and runs with a
  configurable `-c/--concurrency` number of async tasks [15]. So the index-lag ceiling is roughly
  poll interval plus queue drain time.
- **Built-in staleness telemetry.** The `ai.vectorizer_status` view exposes
  `pending_items`. It returns max-bigint once the backlog passes 10,000; an exact count
  needs `ai.vectorizer_queue_pending(id, exact_count=>true)` [15]. This is the simplest
  public example of index lag exposed as a first-class metric.

**Lesson / caveat.** The pgai README states that "as of February 2026, this project is no
longer being maintained or supported" [15]. The pattern (queue table, async workers,
pending-items metric) is still a sound reference design. Teams adopting it should treat
the code as a starting point to fork, not a supported dependency.

Python teams that want custom logic in the "stream processor" box often use Bytewax, "a
Python framework and Rust-based distributed processing engine for stateful event and
stream processing" with Kafka inputs, windowing and state recovery [19]. Checkpointed
state means a restart does not re-embed the whole topic.

### 2.5 Native AI operators in open-source Flink (2.1 and 2.2, 2025)

The Confluent design in Section 2.2 relied on vendor-specific SQL extensions. Apache
Flink has since moved much of this into the open-source engine. The Flink 2.1 release
notes describe a Model DDL that lets users "define and manage AI models programmatically
through the Table API in both Java and Python", and an expanded `ML_PREDICT`
table-valued function "to perform realtime model inference in SQL queries" [29]. Flink
2.2 adds `VECTOR_SEARCH`, "provided in Flink 2.2 to enable users to perform streaming
vector similarity searches", closing a gap the notes state directly: Flink previously
"lacks real-time online querying and similarity analysis capabilities for vector spaces"
[29]. Together they make an all-Flink pipeline possible: CDC topic in, `ML_PREDICT` for
embeddings, `VECTOR_SEARCH` against an external store for each streaming query event,
`ML_PREDICT` again for generation. The queries-as-streams trade-offs of Section 2.2
(durable, replayable, auditable, but higher per-hop latency) apply unchanged.

### 2.6 Comparison of ingestion designs

| Design | Change source | Freshness knob | Deletes | Ops weight | Best fit |
|---|---|---|---|---|---|
| Debezium + Kafka + sink [14][18] | DB replication log | Consumer lag; sink batch size | Tombstones; check sink support | High (Kafka, Connect) | Many sources, many consumers, audit |
| Flink SQL pipeline (Confluent; OSS Flink 2.1+/2.2) [17][29] | Kafka topics | Per-hop streaming latency | Modeled in SQL | Medium (managed) | Queries and answers as replayable streams |
| Pathway live index [16] | Connectors (FS, Drive, SharePoint, Kafka, PG) | Connector polling; incremental dataflow | Yes ("deletions, updates") | Low-medium (single service) | Doc-centric live RAG, answer-change alerts |
| pgai Vectorizer [15] | Postgres table changes via queue | `--poll-interval`, `--concurrency` | Synced with source rows | Low (Postgres + workers) | Postgres-resident corpora; now unmaintained |
| Bytewax / custom [19] | Kafka, files, sockets | Window size, checkpoint cadence | Developer-implemented | Medium | Python teams with custom logic |
| Nightly batch re-index | Full scan | Cron period (hours to days) | Implicit on full rebuild | Low | Slow-changing corpora only |

---

## 3. Recency-weighted ranking and time decay

Once timestamps are in the index, the ranker has to decide how much age should count.
Public implementations fall into three groups: **search-engine decay functions**,
**framework time-weighted retrievers**, and **hard recency filters or sorts**.

### 3.1 Elasticsearch/OpenSearch decay functions

Elasticsearch's `function_score` query supports `gauss`, `exp` and `linear` decay over
numeric, date or geo fields. Each is set by `origin`, `scale`, optional `offset` and
optional `decay`. For date fields, `origin` defaults to now and accepts date math such as
`now-1h` [20]. The documented semantics, written in plain text:

- `decay` is the score a document gets at distance `scale` from `origin +/- offset`
  (default 0.5). Documents within `offset` of origin are not penalized [20].
- **gauss:** `S = exp(-max(0, |t - origin| - offset)^2 / (2 * sigma^2))`, with
  `sigma^2 = -scale^2 / (2 * ln(decay))` [20].
- **exp:** `S = exp(lambda * max(0, |t - origin| - offset))`, with
  `lambda = ln(decay) / scale` [20].
- **linear:** falls to exactly 0 once the distance passes `scale / (1 - decay)`. With
  the default decay of 0.5 that is twice the scale. "In contrast to the normal and
  exponential decay, this function actually sets the score to 0" past that point [20].

The docs' own date example uses `origin: 2013-09-17, scale: 10d, offset: 5d, decay: 0.5`.
Documents from 09-12 to 09-22 get weight 1.0, and documents 15 days out get 0.5 [20].
Function scores combine with the text or vector score through `score_mode` and
`boost_mode`, and per-function `weight` [20].

**Design guidance.** `exp` with `scale = half-life` and `decay = 0.5` gives a true
half-life. `gauss` keeps recent items nearly flat and then drops sharply, which suits
"this week's news". `linear` is the only one that removes old items entirely, which is
dangerous for evergreen content. The `offset` parameter is the main tool against
over-recency: it defines a grace window in which age does not count.

### 3.2 LangChain TimeWeightedVectorStoreRetriever

LangChain's retriever "combines embedding similarity with recency" [21]. From the source:

- Score = `(1 - decay_rate) ** hours_passed + vector_relevance + sum(other_score_keys)`.
  The terms are **added**, not multiplied, and `other_score_keys` can bring in fields
  such as `importance` [21].
- `decay_rate` defaults to 0.01, so a document loses half its recency term after about
  69 hours (0.99^69 is about 0.5).
- `hours_passed` is measured from **`last_accessed_at`**, not creation time. Each
  retrieval resets `last_accessed_at` for the returned documents "to ensure frequently
  accessed memories aren't forgotten" [21].
- The retriever always considers the last `k` items in its in-memory `memory_stream`,
  plus the vector store's top results (default `search_kwargs={"k": 100}`), and returns
  `k` documents (default 4) [21].
- The `last_accessed_at` refresh is applied to the in-memory copies only; a TODO in the
  source notes the vector store entries are not updated [21]. After a restart, recency
  falls back to whatever timestamp was persisted.

This scoring comes from agent memory, in the style of Generative Agents' combination of
recency, importance and relevance [27]. Its semantics are **access recency**, not
**content freshness**. Pointed at a news or policy corpus without changes, it keeps
popular stale documents high because reading them refreshes their timestamp. For content
freshness, swap the date field for a publication or update time and turn off the
access refresh.

### 3.3 LlamaIndex recency postprocessors

LlamaIndex provides three node postprocessors [22]:

- **FixedRecencyPostprocessor**: sorts candidates by a metadata `date` field and keeps
  the newest `top_k` (default 1). This is a hard "latest wins" rule.
- **EmbeddingRecencyPostprocessor**: sorts by date, then walks from newest to oldest.
  It drops older nodes whose embedding similarity to a newer node passes a cutoff
  (default 0.7). The query template tells the embedder that the other documents "are
  older than the current document, meaning that certain details may be changed" [22].
  This is **version de-duplication**: keep the newest version of each near-duplicate,
  drop its ancestors.
- **TimeWeightedPostprocessor**: score = similarity + `(1 - time_decay) ** hours_passed`,
  using a `__last_accessed__` key and refreshing it on access by default [22].

**A parameter trap.** In `TimeWeightedPostprocessor` the default is `time_decay=0.99`,
and the formula is `(1 - time_decay) ** hours`. So the recency term is 0.01 after one
hour and close to 0 after two [22]. LangChain uses the same exponent form with
`decay_rate=0.01`, which gives a 69-hour half-life [21]. Two libraries with the same
formula and similar parameter names can therefore produce half-lives that differ by about
two orders of magnitude. Always convert decay settings into an explicit half-life and
test them. The source also contains a commented-out, unused LLM prompt that classifies
whether a question "requires finding the most recent context" [22]. That is the
query-intent gate discussed next.

### 3.4 Temporal-intent gating

Every public implementation above applies decay without regard to the query. That fixes
recency blindness but causes over-recency on time-invariant questions. FreshQA's split of
questions into never-changing, slow-changing and fast-changing [1][2] shows the problem:
decay helps the third group and can hurt the first. SituatedQA's first task,
"Context-Dependent Question Identification", is a binary classifier for "whether the
answer to a given question does or does not depend on the extra-linguistic contexts" [8].
That is the gate a production system needs. A practical design:

```
              TEMPORAL-INTENT-GATED RETRIEVAL

  query --> [temporal intent classifier] --+-- "timeless"  --> pure relevance
              (LLM prompt or small model)  |
                                           +-- "current"   --> relevance x decay(now)
                                           |                   + filter valid_to=NULL
                                           |
                                           +-- "as of T"   --> valid_from<=T<valid_to
                                           |                   decay centered on T
                                           +-- "range"     --> filter ts in [T1,T2]
                                                               (logs, events)
```

The "as of T" branch reuses Elasticsearch's `origin` parameter. Centering the decay on a
past date instead of now [20] turns recency ranking into **temporal-proximity ranking**,
which is what TimeQA- and SituatedQA-style questions need [6][8].

### 3.5 Research: time-aware retrievers

TempRALM (Gade and Jetcheva, 2024) adds a temporal term to a retriever's score next to
semantic relevance. It is a published example of learned or parameterized time-aware
scoring inside an Atlas-style RAG model [12]. MRAG (Zhang et al., 2024) proposes a
modular framework for time-sensitive QA. It separates the question's main content from
its temporal constraint, retrieves and summarizes evidence, and then ranks with a hybrid
semantic-plus-temporal score. The same paper introduces TempRAGEval [11]. The PDFs of
[11] and [12] could not be fetched in this research session. This chapter therefore
describes only their overall approach and does not quote their numbers.

---

## 4. Valid-time and versioned document retrieval

### 4.1 Temporal knowledge graphs: Zep / Graphiti

**Problem.** Agent memory and enterprise knowledge contain facts that change ("Kendra
loves Adidas shoes (as of March 2026)") [13]. Overwriting the old fact loses history.
Keeping both without validity windows lets the retriever surface contradictions.

**Architecture.** Graphiti builds a "temporal context graph" of entities (nodes), facts
(edges as entity-relationship-entity triplets "with temporal validity windows"), episodes
(the raw ingested data, which carries provenance) and communities [13]. Its key
mechanisms:

- **Incremental construction.** "New data integrates immediately without batch
  recomputation" [13].
- **Automatic invalidation.** When new information contradicts an edge, the old edge is
  closed (its validity ends). It is not deleted, so point-in-time queries still work
  [13].
- **Hybrid retrieval.** Semantic embeddings, BM25 and graph traversal, with "typically
  sub-second latency". The README contrasts this with "seconds to tens of seconds" for
  batch GraphRAG-style summarization [13].
- **LLM-dependent extraction.** Entity/edge extraction and dedupe depend on structured
  JSON output, and the README routes edge resolution and dedupe to specific models [13].

The design is described in the Zep paper, "Zep: A Temporal Knowledge Graph Architecture
for Agent Memory" [13].

```
        BI-TEMPORAL FACT EDGE (Graphiti-style)

   (Alice) --[ROLE: VP Sales]--> (Acme)
        valid_from = 2021-03     valid_to = 2024-07      <- world time
        created_at = 2021-03-04  invalidated_at = 2024-07-15  <- system time
        episode_ref = slack#88213 (provenance)

   (Alice) --[ROLE: CRO]-------> (Acme)
        valid_from = 2024-07     valid_to = NULL (current)

   "Who runs sales now?"      -> edges with valid_to = NULL
   "Who ran sales in 2022?"   -> edges with valid_from <= 2022 < valid_to
   "What did we believe on 2024-07-10?" -> filter on created/invalidated (system time)
```

**Failure modes.** Extraction errors become permanent "facts". Contradiction detection
depends on the LLM. Valid-time has to be inferred from text, and text often states dates
loosely ("last spring"). The third query type above, "what did the system believe at
time T", only works if system time is stored separately from valid time.

### 4.2 Versioned vector datasets

When the unit is a document, not a fact, versioning the index itself is often simpler.
Lance, the format under LanceDB, advertises "zero-copy versioning: automatic versioning
with ACID transactions, time travel, tags, and branches" [26]. For RAG this enables
reproducible evals pinned to a dataset version, as-of retrieval for audit ("what would
the assistant have said on date D"), and safe re-embedding on a branch before promotion.
The costs are storage growth and compaction. Version time travel follows transaction
time only, not valid time inside a document.

### 4.3 Document-level valid-time schema

For a plain vector store, the minimal schema that supports both "current" and "as of"
queries is metadata on each chunk:

| Field | Clock | Used for |
|---|---|---|
| `doc_id`, `version` | - | Replace/delete all chunks of a version; dedupe |
| `valid_from`, `valid_to` (NULL = current) | Valid time | "Current" filter; "as of T" filter |
| `published_at` / `source_updated_at` | Valid/publication | Recency decay origin |
| `indexed_at` | Transaction time | Index-lag metrics; audit |
| `superseded_by` | - | Point the generator to the newer version |

An update then writes the new version and **closes** the old one (sets `valid_to`)
instead of deleting it. Queries with "current" intent filter `valid_to IS NULL`. This is
Graphiti's invalidate-not-delete rule applied to documents [13]. Benchmarks such as
SituatedQA store gold answers with explicit start and end dates. Their sampled contexts
are drawn "between an answer's start and end dates" [8], and that is exactly the data
needed to test the as-of filter.

### 4.4 Contrast: append-only memory with time-aware ranking (Mem0)

Not every memory layer invalidates. Mem0's 2026 README describes the opposite choice:
"Single-pass ADD-only extraction -- one LLM call, no UPDATE/DELETE. Memories accumulate;
nothing is overwritten." Temporal correctness then moves from the write path to the read
path, as "time-aware retrieval that ranks the right dated instance for queries about
current state, past events, and upcoming plans" [32]. The README reports large gains on
LongMemEval (Section 5.9) and LoCoMo for this version, but labels them as managed-platform
results that open-source users should expect to be "directionally similar ... but not
identical" [32]. Treat them as vendor claims.

The two designs trade cost differently. Invalidate-on-write (Graphiti) pays an LLM
contradiction check per ingest and keeps reads simple (`valid_to IS NULL`). Append-only
(Mem0) keeps ingest cheap but puts every "which version is current?" decision on the
ranker at query time, so a ranking error shows up as a stale answer with no record
marked superseded.

---

## 5. Temporal QA benchmarks

### 5.1 FreshQA / FreshLLMs (Google, UMass; 2023, continuously updated)

**What it tests.** Whether an LLM, with or without search augmentation, answers questions
whose answers change, including false-premise questions. The paper is "FreshLLMs:
Refreshing Large Language Models with Search Engine Augmentation" [1].

**Distinctive design: a living benchmark.** The dataset is published as dated snapshots.
The repository lists versions from February 2024 through April 21, 2026, roughly weekly
at first and less often later. It states the maintainers "update our dataset weekly or
upon request" [2]. An eval result is therefore only meaningful with its snapshot date.

**Method: FreshPrompt.** The paper's few-shot prompting method puts search-engine results
into the prompt. A public notebook implements it [2]. The README states that FreshQA and
FreshPrompt have "inspired or been used for the development of" Google's Gemini,
Perplexity's online LLMs, You.com's API and Contextual AI's RAG 2.0 [2].

**Evaluation.** Two modes. **Relaxed** judges whether the main answer is correct.
**Strict** also penalizes any hallucinated or outdated claim in the response. The
maintainers say human raters are the most reliable judges. They also release FreshEval,
"a simple automatic metric that uses few-shot in-context learning to teach an LLM to
judge model responses" with "high agreement with human raters", and recommend a specific
GPT-4 snapshot (`gpt-4-1106-preview`) as the judge [2]. You.com reported FreshQA *recall* as a cheaper overlap
metric [2]. The README points to **SealQA** as a follow-up that builds on FreshQA
"with a stronger focus on reasoning", released on Hugging Face (`vtllms/sealqa`) [3].

**Lesson.** The strict-versus-relaxed gap measures staleness inside the answer. A RAG
system can retrieve the current fact and still repeat an outdated parametric claim
elsewhere in its response. Only strict mode catches that.

### 5.2 RealTime QA (UW, AI2, Tohoku; NeurIPS 2023 Datasets and Benchmarks)

**What it tests.** "A dynamic question answering (QA) platform that inquires about the
present." The organizers announce about 30 questions on a regular schedule (weekly in
the released version) and evaluate baseline systems such as GPT-3 and T5 [5]. The
submission window "closes when the next set of questions is announced (3 am GMT on
every Saturday)" [5].

**Baselines.** Six real-time baselines run every week. They pair retrieval from Google
Custom Search or DPR with readers, in multiple-choice and open-generation settings. The
repository publishes the retrieval results of each run [5]. Real-time baselines started
on June 17, 2022, and "backnumber" questions cover the earlier period [5]. The paper is
Kasai et al. [4].

**Lesson.** RealTime QA treats freshness as a scheduled, continuously graded property:
every week brings questions no training set could contain, a template for internal
staleness evaluation (Section 7).

### 5.3 TimeQA (UCSB; NeurIPS 2021 Datasets and Benchmarks)

**What it tests.** "Existing reading comprehension models' capability to perform
temporal reasoning, and ... whether they are sensitive to the temporal description in
the given question" [7]. Crowd workers annotated passages with "time-evolving facts".
Template-synthesized train/dev/test splits were then built in **easy** and **hard**
versions, plus a human-written test set [7]. Baselines are BigBird (extractive) and
Fusion-in-Decoder (generative), initialized from NQ or TriviaQA checkpoints. An
open-domain setting uses a TF-IDF/BM25 retriever [7]. The paper is Chen, Wang and Wang
[6].

**Lesson for RAG.** TimeQA separates **retrieving the right passage** from **reading the
right time slice inside it**. One Wikipedia passage can hold several valid-time
intervals. Chunking that splits a passage across interval boundaries, or a reader that
ignores "in 2009" or "before 2009", fails even with perfect retrieval.

### 5.4 SituatedQA (UT Austin/UW; EMNLP 2021)

SituatedQA covers questions whose answers depend on temporal or geographical context [8].
Its temporal task pairs each question with a date. Dates are either sampled "between an
answer's start and end dates" or set to the answer's start or end. Gold answers come
from NaturalQuestions, and an `edited_question` folds the context into the text [8]. Its
context-dependence classification task is a ready-made training set for the temporal
intent gate in Section 3.4 [8].

### 5.5 StreamingQA (DeepMind; 2022)

StreamingQA evaluates "adaptation to new knowledge over time in question answering
models". It is built on a deduplicated subset of the English WMT News Crawl, with
document files spanning 2007 to 2020 [9]. Each QA item records `question_ts` (when the
question is asked) and `evidence_ts` (when the evidence article was published). It also
records a `recent_or_past` split and whether the item was written by humans or generated
[9]. The authors estimate that about 0.5% of items remain toxic after filtering. They
warn that answers "reflect information from the news articles, and ... may not always be
factually correct" [9]. Both points matter for news-RAG evaluation sets built the same
way.

### 5.6 TempLAMA (Google; TACL 2022)

TempLAMA probes language models as temporal knowledge bases, using cloze queries built
from a November 1, 2020 Wikidata dump [10]. That dump is no longer downloadable, so
rebuilding today gives "slightly different" data [10], a small staleness lesson in its
own right.

### 5.7 TempRAGEval (2024)

TempRAGEval was introduced with MRAG [11]. It evaluates time-sensitive QA end to end in
a RAG setting, scoring both retrieval and answer accuracy on questions with temporal
constraints. The paper's dataset statistics and results could not be fetched in this
session, so the chapter only names it as the RAG-specific complement to the
reader-focused TimeQA and SituatedQA.

### 5.8 TemporalWiki (KAIST; EMNLP 2022)

TemporalWiki is "a lifelong benchmark for training and evaluating ever-evolving language
models" [30]. It is built from consecutive English Wikipedia snapshots (the released data
covers five monthly snapshots, 2021.08 to 2021.12). `TWiki_Diffsets` hold only what
changed between snapshots, and `TWiki_Probes` test whether updated models know the new and
unchanged facts [30]. It frames freshness as a **diff problem**: how cheaply can a system
absorb only the changed knowledge? That is the parametric counterpart of CDC-driven
re-indexing in Section 2, and the diffsets are a convenient test corpus for incremental
indexing pipelines.

### 5.9 LongMemEval (ICLR 2025)

LongMemEval benchmarks "chat assistants on long-term interactive memory" with 500
questions over long multi-session chat histories [31]. Two of its five core abilities are
temporal: **knowledge updates** (the user's facts changed across sessions; answer with the
latest) and **temporal reasoning** (questions that depend on when things were said), next
to information extraction, multi-session reasoning and abstention [31]. It is the de facto
yardstick for agent-memory systems such as Zep/Graphiti and Mem0 (Section 4), and the
knowledge-update category is a direct test of invalidate-versus-append designs.

### 5.10 Benchmark comparison

| Benchmark | Year | Corpus / source | Temporal axis tested | Live? | Eval |
|---|---|---|---|---|---|
| FreshQA [1][2] | 2023 to 2026 | Curated Qs; web search | Never/slow/fast-changing, false premise | Yes, dated snapshots | Human; FreshEval relaxed/strict |
| SealQA [3] | 2025 | Builds on FreshQA | Reasoning over noisy/conflicting search | Not stated | See paper |
| RealTime QA [4][5] | 2022 to 2023 | News; GCS/DPR retrieval | "Now" (this week's events) | Yes, weekly | MC accuracy; open-gen |
| TimeQA [6][7] | 2021 | Wikipedia time-evolving facts | Time-scoped reading (easy/hard) | No | EM/F1 style |
| SituatedQA [8] | 2021 | NaturalQuestions | Answer as of a date; context-dependence ID | No | EM style |
| StreamingQA [9] | 2022 | WMT News 2007 to 2020 | Question time vs evidence time; recent vs past | No (time-sliced) | Multi-reference |
| TempLAMA [10] | 2022 | Wikidata (Nov 2020) | Facts with time qualifiers | No | Cloze accuracy |
| TempRAGEval [11] | 2024 | Derived from temporal QA sets | RAG retrieval + answer under time constraints | No | Retrieval + QA |
| TemporalWiki [30] | 2022 | Monthly Wikipedia snapshots + diffs | Absorbing changed vs unchanged knowledge | Time-sliced | Probe accuracy |
| LongMemEval [31] | 2024 to 2025 | Multi-session chat histories (500 Qs) | Knowledge updates; temporal reasoning in memory | No | QA accuracy (LLM-judged) |

---

## 6. Application families where time is the main axis

### 6.1 News- and market-feed RAG

**Problem.** Finance is "highly dynamic". FinGPT's README makes the economic case for
retrieval over retraining. It cites BloombergGPT's training at about 53 days and roughly
$3M; its own table estimates about $2.67M on 512 A100s. It contrasts that with "less than
$300 per fine-tuning" for lightweight adaptation [25][28]. Even weekly fine-tuning leaves
gaps of days, so the newest market information has to arrive through retrieval.

**Architecture patterns (FinGPT).** FinGPT describes a layered stack. A data-source layer
"addressing the temporal sensitivity of financial data through real-time information
capture". A data-engineering layer for "real-time NLP data processing" that handles "high
temporal sensitivity and low signal-to-noise ratio". Above those sit model and
application layers [25]. Two components are relevant here:

- **FinGPT-RAG**: "a retrieval-augmented large language model framework specifically
  designed for financial sentiment analysis", published at ICAIF-23 [25].
- **FinGPT-Forecaster**: takes a ticker, a prediction date, "the number of past weeks
  where market news are retrieved", and an optional flag to add the latest basic
  financials. It then produces an analysis and a next-week price-movement prediction
  [25]. The retrieval window here is an **explicit, user-set time range anchored to a
  chosen date**, the "range" branch of the intent gate. It also prevents look-ahead
  leakage in backtests.

**Lessons for feed RAG.**

- **Point-in-time correctness is a hard constraint in finance.** A backtest that
  retrieves news published after the prediction date is invalid. Filter on publication
  time no later than the as-of date. StreamingQA's `question_ts` and `evidence_ts`
  fields are the evaluation version of this rule [9].
- **Dedupe syndicated stories.** Wire stories get republished many times. Newest-wins
  near-duplicate collapse (LlamaIndex's EmbeddingRecencyPostprocessor pattern [22])
  stops one event from filling the context window.
- **Decay on publication time, not on ingestion or access time.** See the pitfalls in
  Sections 3.2 and 3.3.

### 6.2 Event-stream and log RAG: observability copilots

Logs, metrics and traces are append-only, high-volume and mostly time-range queried.
Embedding every log line is rarely worth it. Public observability copilots instead treat
the time-series backends as **tools** and the LLM as the query planner.

**Grafana MCP server.** Grafana's MCP server exposes tools that let an agent run PromQL
range and instant queries, run LogQL log and metric queries against Loki, read Loki label
and stream metadata, and pull **log patterns** "detected by Loki to identify common log
structures and anomalies" [24]. It also covers Elasticsearch/OpenSearch, Quickwit,
ClickHouse, CloudWatch and Google Cloud Logging. Cloud Logging results come back
"newest-first with severity, body, labels, and trace ID". Other tools cover alert rules,
incidents and a hand-off to Grafana Assistant that "may use tools, metrics, logs, and
other stack context" [24].

Retrieval here means generating a time-bounded query, so freshness is close to perfect
by construction. The hard parts move elsewhere:

- **Choosing the time window.** "What broke after the deploy?" needs the deploy time
  first.
- **Compressing results.** Log patterns are a pre-aggregation that turns millions of
  lines into a few dozen templates the LLM can read [24].
- **Permissions.** Each tool maps to fine-grained Grafana RBAC scopes, such as
  `datasources:query` on a specific datasource UID [24].

```
        OBSERVABILITY COPILOT: TOOL-MEDIATED TEMPORAL RETRIEVAL

  user: "why did p99 latency spike after 14:00?"
     |
     v
  +--------------+  1. alert rules / incidents   +---------------------+
  |  LLM agent   | ----------------------------> | Grafana / Alerting  |
  |  (planner)   |  2. PromQL range [13:30,15:00]| Prometheus / Mimir  |
  |              | ----------------------------> |                     |
  |              |  3. Loki patterns, same range | Loki                |
  |              | ----------------------------> |                     |
  |              |  4. runbook / postmortem RAG  | vector KB (static)  |
  +------+-------+ ----------------------------> +---------------------+
         |
         v   answer = live telemetry (exact time range) + static knowledge (semantic)
```

### 6.3 SIEM copilots: Elastic AI Assistant for Security

Elastic's AI Assistant for Security is a clear public example of **mixing a time-windowed
live feed with a semantic knowledge base**:

- **Alerts as recency-bounded context.** The "Knowledge base for alerts" feature sends
  `open` or `acknowledged` alerts **from the last 24 hours**. They are "ordered by risk
  score, then by the most recently generated", and building-block alerts are excluded
  [23]. A slider sets how many alerts to include. The docs warn that too many alerts "may
  cause your request to exceed the maximum token length", so the count is a user-tuned
  context budget [23]. This is a hard recency window followed by a sort on importance
  then recency, which is the same idea as LangChain's importance-plus-recency score [21],
  done with filters instead of a decay function.
- **Indices as live knowledge sources.** Admins can add whole Elasticsearch indices
  (asset inventories, on-call matrices, threat intel reports, vulnerability scans). "When
  you update the index with new information, AI Assistant will gain access to the new
  information". Each source has a natural-language "Query Instruction" and a list of
  output fields [23]. Freshness is inherited from the index, with no separate embedding
  pipeline. Indices must have a `semantic_text` field, or from 9.1 a plain text field,
  which "perform[s] better for retrieval based on specific document values or
  attributes, such as email or username" [23].
- **A dated static corpus.** The knowledge base comes pre-loaded with Elastic Security
  Labs articles "current through September 30, 2024" [23]. Even a vendor's curated
  corpus has a stated freshness horizon. Stating it, as Elastic does, is good practice.
- **User-asserted memory with time.** Entries such as "Remember that I changed my
  password today, October 24, 2024" carry their date in the text [23], a simple form of
  valid-time stamping.
- From 9.1 the agent uses "index names and field metadata" to improve ES|QL generation
  [23]. Event retrieval is again partly query generation, not only vector search.

---

## 7. Evaluating staleness

The benchmarks in Section 5 measure whether a model handles time-dependent questions. A
production team also needs to measure whether **its own pipeline** is stale. The public
sources above suggest a layered method. The metric names below are this chapter's
synthesis; the components come from the cited systems.

### 7.1 Pipeline-level metrics (does the index lag the source?)

| Metric | Definition | Public precedent |
|---|---|---|
| Index lag (p50/p99) | `indexed_at - source_updated_at` per change event | Debezium event timestamps [14]; pgai queue [15] |
| Backlog | Changes captured but not yet embedded | pgai `pending_items` [15]; Kafka consumer lag |
| Delete propagation lag | Time from source delete to absence in top-k | Tombstones in CDC [14] |
| Coverage | Source docs present in index / source docs | Pathway `/v1/inputs`, `/v1/statistics` [16] |
| Corpus horizon | Newest `published_at` in each static KB | Elastic Security Labs "through Sept 30, 2024" [23] |

Set **freshness SLOs** per source, for example "p99 index lag under 60 s for pricing;
under 24 h for wiki". Alert on them like any other data-pipeline SLO. With poll-based
workers, the poll interval plus the drain time bounds lag from below. A five-minute
default poll [15] means sub-minute freshness is impossible without retuning.

### 7.2 Canary facts (end-to-end freshness probes)

Borrow RealTime QA's continuous schedule [5] for internal use:

1. On a schedule, write a synthetic, uniquely identifiable fact into each source system
   ("Canary 7f3a: the test SKU price is 41.17"). Also update or delete a previous canary.
2. Query the RAG endpoint with the matching question on a schedule.
3. Record **time-to-correct-answer**, **time-to-forget** for deleted canaries, and
   whether the answer cites the new version.

This catches failures that pipeline metrics miss: caching layers, a reranker that buries
fresh chunks, decay misconfiguration (Section 3.3), and the generator preferring
parametric knowledge.

### 7.3 Answer-level temporal evaluation

- **Build a FreshQA-style internal set** split into never-, slow- and fast-changing
  questions [1][2]. Re-label answers for fast-changing items on a schedule and version
  the set by date, as FreshQA does [2].
- **Score strict and relaxed** [2]. The strict-minus-relaxed gap shows outdated claims
  mixed into otherwise-correct answers.
- **Include false-premise and "as of T" items.** For as-of items, store gold answers with
  validity intervals, following SituatedQA's start and end dates [8]. Test both the
  "current" and the "as of a past date" query paths.
- **Check for look-ahead leakage** in any backtest or historical eval. Assert that every
  retrieved `evidence_ts` is no later than `question_ts` [9].
- **Pin the corpus version** for regression tests, using dataset time travel [26].
  Otherwise score changes mix retriever changes with corpus drift.
- **LLM judges.** FreshEval shows few-shot LLM judges can approximate human ratings on
  freshness [2]. Its authors still prefer human raters and pin a specific judge model,
  because judge drift is itself a temporal confound.

### 7.4 Ranking-level diagnostics

- Compare the **age distribution of retrieved chunks** with the corpus's, per intent
  class: timeless queries should show little skew, "current" queries should skew young.
- Track the **superseded-hit rate**, the share of retrieved chunks with a
  `superseded_by` pointer. A high rate means valid-time filtering is broken.

---

## 8. Reference architecture

```
            REFERENCE: TIME-AWARE STREAMING RAG

  SOURCES             INGEST                    STORES                QUERY
  -------             ------                    ------                -----
  OLTP DB --CDC--+                        +-> vector/hybrid idx --+
  Docs/Drive ----+--> stream processor ---+   (valid_from/to,      |   +----------+
  News feeds ----+    chunk, dedupe,      |    published_at,       +-->| intent   |
  Kafka topics --+    version, embed      |    indexed_at)         |   | gate     |
                      (batch+retry)       +-> temporal KG ---------+   +----+-----+
                           |              |   (bi-temporal edges)  |        |
                           |              +-> versioned snapshot --+   timeless/now/
                           v                  (time travel, eval)  |   as-of/range
                    lag + backlog metrics                          |        |
                           |                LIVE TELEMETRY ---------+   +---v------+
                           v                (Prom/Loki/SIEM via     |   | retrieve |
                    freshness SLO alerts     tools, time-bounded)   +-->| + decay  |
                           ^                                            | + filter |
                           |                                            +----+-----+
                    canary writer/prober <------------------------------+    |
                                                                       +-----v-----+
                                                                       | generate  |
                                                                       | cite dates|
                                                                       +-----------+
```

---

## Key takeaways for system designers

1. **Use CDC instead of dual writes.** A vector index is a derived read model. Feed it
   from the database's committed change log (Debezium + Kafka) or an in-database queue
   (the pgai pattern), never from application-side dual writes [14][15].
2. **Deletes and re-chunking are the hard part.** Key chunks by stable `doc_id` and
   version, delete all chunks of a superseded version, and test tombstone handling in
   every sink connector [14][18].
3. **Expose index lag as a first-class metric.** pgai's `pending_items` view [15] and
   Pathway's `/v1/statistics` endpoint [16] are the public examples. Set per-source
   freshness SLOs, and remember that poll intervals set a floor on lag.
4. **Store at least three timestamps per chunk:** valid time, publication/update time and
   indexed time. Most temporal bugs come from decaying or filtering on the wrong clock.
5. **Convert every decay parameter into a half-life.** LangChain's default is about 69
   hours. LlamaIndex's TimeWeightedPostprocessor default effectively zeroes the recency
   term after about an hour. Both use the same formula shape [21][22]. Elasticsearch's
   `exp` with `decay=0.5` makes `scale` the half-life, and `offset` gives a grace window
   [20].
6. **Access recency is not content freshness.** Retrievers that refresh
   `last_accessed_at` on read are memory heuristics [21][27]. For document corpora, decay
   on publication or update time.
7. **Gate decay on temporal intent.** Classify queries as timeless, current, as-of-T or
   range (SituatedQA provides training data for this gate [8]). Apply decay only to
   "current". Re-center on T for as-of. Use hard filters for ranges.
8. **Invalidate, don't delete, when history matters.** Bi-temporal edges (Graphiti [13])
   or `valid_to` closure on document versions support both "now" and "as of" queries,
   plus audit of what the system believed at a given time.
9. **For logs and telemetry, retrieval is query generation.** Observability and SIEM
   copilots get freshness by running time-bounded PromQL/LogQL/ES|QL through tools, then
   compressing results (log patterns, risk-sorted alerts) to fit the context budget
   [23][24].
10. **Evaluate freshness continuously, not once.** Use dated, re-labeled question sets
    (FreshQA [2]), scheduled question batches (RealTime QA [5]), strict scoring for
    outdated claims [2], canary facts for end-to-end lag, and look-ahead-leakage checks
    on any historical eval [9].
11. **State the corpus horizon to users.** If a knowledge base is current only through a
    date, say so, as Elastic does for its Security Labs corpus [23]. That makes a silent
    staleness failure a visible, known limit.

---

## References

1. Vu, T. et al. "FreshLLMs: Refreshing Large Language Models with Search Engine
   Augmentation." Google / UMass Amherst, 2023. https://arxiv.org/abs/2310.03214
2. FreshLLMs authors. "FreshQA / FreshPrompt / FreshEval repository (dated dataset
   snapshots through April 2026)." GitHub, 2023-2026. https://github.com/freshllms/freshqa
3. Pham, T. et al. (Vu, T., senior author). "SealQA: Raising the Bar for Reasoning in
   Search-Augmented Language Models" (benchmark building on FreshQA; announced in the
   FreshQA README), 2025. https://arxiv.org/abs/2506.01062 ; dataset:
   https://huggingface.co/datasets/vtllms/sealqa (arXiv page not retrievable in this
   session; title and authors from prior knowledge)
4. Kasai, J., Sakaguchi, K., Takahashi, Y., Le Bras, R., Asai, A., Yu, X. V., Radev, D.,
   Smith, N. A., Choi, Y., Inui, K. "RealTime QA: What's the Answer Right Now?" NeurIPS
   Datasets and Benchmarks, 2023. https://arxiv.org/abs/2207.13332
5. RealTime QA team (UW, AI2, Tohoku). "realtimeqa_public repository." GitHub, 2022-2023.
   https://github.com/realtimeqa/realtimeqa_public
6. Chen, W., Wang, X., Wang, W. Y. "A Dataset for Answering Time-Sensitive Questions"
   (TimeQA). NeurIPS Datasets and Benchmarks, 2021. https://arxiv.org/abs/2108.06314
7. UCSB NLP group. "Time-Sensitive-QA repository." GitHub, 2021.
   https://github.com/wenhuchen/Time-Sensitive-QA
8. Zhang, M. J. Q., Choi, E. "SituatedQA: Incorporating Extra-Linguistic Contexts into
   QA." EMNLP, 2021. https://arxiv.org/abs/2109.06157 (repository:
   https://github.com/mikejqzhang/SituatedQA)
9. Liska, A., Kocisky, T., Gribovskaya, E., Terzi, T. et al. "StreamingQA: A Benchmark for
   Adaptation to New Knowledge over Time in Question Answering Models." DeepMind, 2022.
   https://arxiv.org/abs/2205.11388 (repository:
   https://github.com/google-deepmind/streamingqa)
10. Dhingra, B., Cole, J. R., Eisenschlos, J. M., Gillick, D., Eisenstein, J., Cohen, W. W.
    "Time-Aware Language Models as Temporal Knowledge Bases" (TempLAMA). TACL 10, 2022.
    https://doi.org/10.1162/tacl_a_00459 (construction code:
    https://github.com/google-research/language/tree/master/language/templama)
11. Zhang, S. et al. "MRAG: A Modular Retrieval Framework for Time-Sensitive Question
    Answering" (introduces TempRAGEval), 2024. https://arxiv.org/abs/2412.15540
    (identifier from prior knowledge; PDF not retrievable in this research session)
12. Gade, A., Jetcheva, J. "It's About Time: Incorporating Temporality in Retrieval
    Augmented Language Models" (TempRALM), 2024. https://arxiv.org/abs/2401.13222
    (identifier from prior knowledge; PDF not retrievable in this research session)
13. Zep AI. "Graphiti: A Framework for Building Temporal Knowledge Graphs" (repository)
    and Rasmussen, P. et al. "Zep: A Temporal Knowledge Graph Architecture for Agent
    Memory," 2025. https://github.com/getzep/graphiti ; https://arxiv.org/abs/2501.13956
14. Debezium project (Red Hat and community). "Debezium: change data capture for a variety
    of databases" (README). GitHub, accessed 2026. https://github.com/debezium/debezium
15. Timescale. "pgai: Power your RAG and Agentic applications with PostgreSQL" (README,
    Vectorizer overview and worker docs), 2024-2026. https://github.com/timescale/pgai ;
    https://github.com/timescale/pgai/blob/main/docs/vectorizer/overview.md ;
    https://github.com/timescale/pgai/blob/main/docs/vectorizer/worker.md
16. Pathway. "Pathway Live Data Framework" and "LLM App templates" (document indexing,
    Google Drive alert, adaptive RAG), 2023-2026. https://github.com/pathwaycom/pathway ;
    https://github.com/pathwaycom/llm-app ;
    https://github.com/pathwaycom/llm-app/tree/main/templates/document_indexing ;
    https://github.com/pathwaycom/llm-app/tree/main/templates/drive_alert
17. Confluent. "Streaming Agents on Confluent Cloud Quickstart" and "Lab2: Vector Search /
    RAG Walkthrough," 2025-2026. https://github.com/confluentinc/quickstart-streaming-agents ;
    https://github.com/confluentinc/quickstart-streaming-agents/blob/master/LAB2-Walkthrough.md
18. Qdrant. "Qdrant Sink Connector for Kafka Connect"; Zilliz. "Kafka Connect Milvus Sink."
    GitHub, accessed 2026. https://github.com/qdrant/qdrant-kafka ;
    https://github.com/zilliztech/kafka-connect-milvus
19. Bytewax. "Bytewax: Python Stateful Stream Processing Framework"; Apache Software
    Foundation. "Apache Flink Agents." GitHub, accessed 2026.
    https://github.com/bytewax/bytewax ; https://github.com/apache/flink-agents
20. Elastic. "Function score query" (decay functions), Elasticsearch reference
    documentation source, accessed 2026.
    https://github.com/elastic/elasticsearch/blob/main/docs/reference/query-languages/query-dsl/query-dsl-function-score-query.md
21. LangChain. "TimeWeightedVectorStoreRetriever" (source code), accessed 2026.
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/time_weighted_retriever.py
22. LlamaIndex. "Node recency post-processors" (FixedRecency, EmbeddingRecency,
    TimeWeighted; source code), accessed 2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/node_recency.py
23. Elastic. "AI Assistant for Security" and "AI Assistant Knowledge Base" (documentation
    source), 2024-2026.
    https://github.com/elastic/docs-content/blob/main/solutions/security/ai/ai-assistant.md ;
    https://github.com/elastic/docs-content/blob/main/solutions/security/ai/ai-assistant-knowledge-base.md
24. Grafana Labs. "mcp-grafana: MCP server for Grafana" (README), 2025-2026.
    https://github.com/grafana/mcp-grafana
25. AI4Finance Foundation. "FinGPT: Open-Source Financial Large Language Models" (README,
    FinGPT-RAG, FinGPT-Forecaster), 2023-2026. https://github.com/AI4Finance-Foundation/FinGPT ;
    AI4Finance authors. "Enhancing Financial Sentiment Analysis via Retrieval Augmented Large
    Language Models," ICAIF 2023. https://arxiv.org/abs/2310.04027
26. LanceDB. "Lance: modern columnar data format for ML" (README; zero-copy versioning and
    time travel), accessed 2026. https://github.com/lancedb/lance
27. Park, J. S. et al. "Generative Agents: Interactive Simulacra of Human Behavior," 2023.
    https://arxiv.org/abs/2304.03442 (identifier from prior knowledge)
28. Wu, S. et al. "BloombergGPT: A Large Language Model for Finance," Bloomberg, 2023.
    https://arxiv.org/abs/2303.17564
29. Apache Software Foundation. "Release notes: Flink 2.1" and "Release notes: Flink 2.2"
    (Model DDL, `ML_PREDICT`, `VECTOR_SEARCH`), 2025.
    https://github.com/apache/flink/blob/master/docs/content/release-notes/flink-2.1.md ;
    https://github.com/apache/flink/blob/master/docs/content/release-notes/flink-2.2.md
30. Jang, J. et al. "TemporalWiki: A Lifelong Benchmark for Training and Evaluating
    Ever-Evolving Language Models." EMNLP, 2022. https://arxiv.org/abs/2204.14211
    (repository: https://github.com/joeljang/temporalwiki)
31. Wu, D. et al. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
    Memory." ICLR, 2025. https://arxiv.org/abs/2410.10813 (repository:
    https://github.com/xiaowu0162/LongMemEval)
32. Mem0. "Mem0: the memory layer for AI agents" (README, 2026 algorithm notes and
    benchmark table) and Chhikara, P. et al. "Mem0: Building Production-Ready AI Agents
    with Scalable Long-Term Memory," 2025. https://github.com/mem0ai/mem0 ;
    https://arxiv.org/abs/2504.19413
