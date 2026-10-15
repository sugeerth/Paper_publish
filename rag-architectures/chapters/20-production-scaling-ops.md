# 20. Production RAG Operations: Ingestion, Freshness, Migrations, Latency, Cost, Observability, Caching and Debugging

## Overview

Most RAG literature is about the query path: how to chunk, embed, retrieve, rerank and prompt. Teams that have run RAG for a year or more report something else. The hard problems live in the data plane and the operations around it:

- keeping an index synchronized with a changing source of truth, including deletes and permission changes;
- changing the embedding model without downtime or a recall cliff;
- keeping a multi-stage pipeline within a latency SLO;
- predicting and controlling cost at the storage, embedding and generation layers;
- seeing what happened inside a single failing request, and turning that into a fix.

Barnett et al. summarize the core lesson of their three-case-study experience report: a RAG system's robustness evolves over time and cannot be fully designed in up front, and validation is really only feasible during operation [1]. A 2026 paper by Budigi and Sirigiri makes a similar point from the infrastructure side. It argues that production RAG failures mostly originate in the data layer, not the language model, and names staleness, access-control fragility and "query composition explosion" across split systems as the three dominant failure modes [39].

This chapter treats RAG as an operational system. It covers ingestion and incremental indexing (Section 1), freshness, deletes and permissions (Section 2), index maintenance inside vector engines (Section 3), re-embedding migrations (Section 4), latency budgets (Section 5), cost modeling (Section 6), caching at four layers plus prefix-aware serving (Section 7), observability and tracing (Section 8), and failure modes with debugging playbooks (Section 9). Case studies from Notion, Cursor, Uber, Dropbox, Pinecone, turbopuffer and Microsoft GraphRAG are woven through.

A note on sourcing: numbers here come from vendor docs, company engineering blogs and papers. Vendor-reported figures are labelled as such, and figures that only appear in secondary coverage are marked approximate.

## 1. Ingestion pipelines and incremental indexing

### 1.1 The canonical ingestion pipeline

Every production system examined here has the same skeleton. It differs mainly in how changes are detected and how much work is skipped.

```
  sources (wiki, drive, tickets, code, DB)
        |  connectors: full crawl + webhooks / CDC / polling
        v
  +--------------+   +------------+   +-----------+   +------------------+
  | fetch + ACLs |-->| parse /    |-->| chunk +   |-->| change detection |
  | + metadata   |   | normalize  |   | enrich    |   | (hash per doc /  |
  +--------------+   +------------+   +-----------+   |  per chunk)      |
                                                      +--------+---------+
                                     unchanged: skip <---------+
                                                               | changed / new
                                                               v
                              +-------------------+    +----------------+
                              | embed (batch GPU  |--->| upsert vectors |
                              | or API, cached by |    | + lexical idx  |
                              | content hash)     |    | + metadata     |
                              +-------------------+    +-------+--------+
                                                               |
                       record manager / docstore  <------------+
                       (source_id -> {chunk ids, hashes, ts, model ver})
                                                               |
                                    cleanup: delete stale ids  v
```

The component that separates a demo from a production system is the record manager, sometimes called the docstore, page-state table or manifest. It maps each source document to the chunk IDs and content hashes currently in the index. Without it you cannot answer "which vectors belong to the document that was just edited or deleted?", and the index drifts.

### 1.2 Framework implementations: LangChain and LlamaIndex

**LangChain indexing API.** LangChain's `index()` function pairs a vector store with a `RecordManager` that records a hash of each document, its write time and its source ID [2]. Its cleanup modes encode the main design choices:

- `None`: deduplicate only. Nothing is ever deleted.
- `incremental`: as each batch is written, delete older versions of documents from the same source. Stale data is visible only briefly. If one source's chunks span several batches, some work is repeated. The docs say the end state is correct but the process is "not 100% efficient" [2].
- `full`: cleanup runs after all batches are written. Any document not seen in this run is deleted, which is the only mode that catches deletions at the source. The loader must therefore return the entire dataset. A partial load would wipe valid content [2].
- `scoped_full` (a later addition in the 0.3 line): cleanup also runs at the end of indexing, but only over the source IDs seen in the run, so it is parallelizable and suits loaders that cannot return everything at once. The docs are explicit that, unlike `full`, it does **not** remove documents whose source was deleted [2].

The documented caveats are operationally important. Do not use the API on a store that was populated by other means, because the record manager cannot know about those vectors. The vector store must support add-by-ID and delete-by-ID. Cleanup depends on a time-based mechanism (write timestamps): the docs note that if two back-to-back indexing tasks finish before the clock value changes, the second may fail to clean up, and clock skew between writers matters for the same reason [2].

**LlamaIndex IngestionPipeline.** LlamaIndex attaches a persistent docstore to an ingestion pipeline and offers three strategies [3][4]:

- `UPSERTS` (the default): re-run transformations when a document ID is new or its hash has changed.
- `DUPLICATES_ONLY`: skip only exact hash duplicates.
- `UPSERTS_AND_DELETE`: also delete documents missing from the current input.

Older versions silently fall back to `DUPLICATES_ONLY` when no vector store is attached [4], a quiet behavioral change worth checking in tests. The pipeline also caches transformation outputs, so re-running unchanged nodes through the same transformations, including embedding, costs nothing.

**Lesson.** Both frameworks make deletion an explicit, opt-in mode that needs a complete view of the source. That is the root cause of many "ghost document" incidents, discussed in Section 2.

### 1.3 Case study: Notion's dual-path pipeline and Page State

Notion's February 2026 retrospective, "Two years of vector search at Notion: 10x scale, 1/10th cost", is one of the most detailed public accounts of operating RAG ingestion at scale [5]. The figures below are from that post and from secondary summaries of it [6].

*Launch architecture.* Indexing had two paths:

- an offline path, in which Spark batch jobs chunked existing pages, called an embedding API and bulk-loaded vectors into the vector database;
- an online path, in which Kafka consumers processed individual page edits in near real time.

To add capacity, Notion chose not to reshard under live traffic. It created new index clusters tagged with a generation ID and routed new workspaces to them [5][6].

```
  page edits --> Kafka --> online consumers --+
                                              |      +-------------------+
  historical pages --> Spark batch -----------+----> | embed + upsert to |
                                                     | vector DB (gen N) |
  routing table: workspace_id -> generation  ------> +-------------------+
```

*Redesign.* Over two years Notion reports roughly 10x scale at about one-tenth the cost [5]. The secondary summaries attribute this to several changes [6]:

1. A migration to turbopuffer, an object-storage-first engine, including a full re-index and a move to a newer embedding model. Each namespace is an independent index, so the sharding and generation-routing logic disappeared. Reported savings were 60% on search-engine spend and 35% on AWS EMR compute.
2. A Page State system that hashes page text and metadata separately. Only changed spans are re-embedded, a reported 70% reduction in data volume sent for embedding.
3. A move of the near-real-time embedding pipeline from Spark plus an external embedding API to Ray on Anyscale with self-hosted models, starting July 2025. CPU preprocessing and GPU inference run in one pipeline, removing an S3 hand-off.
4. Query latency improving from about 70-100 ms to about 50-70 ms (approximate, per secondary summaries).

*Lessons.*

- Change detection at sub-document granularity is the biggest cost lever in ingestion.
- Choosing a storage engine whose unit of isolation (the namespace) matches the tenant boundary removes a whole class of operational work.
- A full re-index is easiest to justify when it is bundled with an engine or model migration you were going to do anyway.

### 1.4 Case study: Cursor's Merkle-tree sync and index reuse

Code indexing is an extreme incremental-indexing workload: many clients, frequent small edits, and near-duplicate corpora. Cursor's "Securely indexing large codebases" post describes the approach [7].

- **Merkle-tree change detection.** The client computes a cryptographic hash of every file. Each folder hash is derived from its children's hashes. Client and server compare trees from the root down and descend only where hashes differ, so an edit costs O(depth) comparisons rather than a full scan [7].
- **Index reuse across teammates.** Clones of the same repository average 92% similarity across users within an organization. Cursor therefore reuses an existing teammate's index instead of building from scratch, cutting time-to-first-query on the largest repos "from hours to seconds" [7].
- **Matching and seeding (secondary source).** A third-party analysis reports how the matching works [8]. The client computes a simhash summarizing its file-hash distribution, and the server finds the nearest existing index. If similarity exceeds a threshold, the new namespace is seeded with turbopuffer's `copy_from_namespace`. The same analysis says content hashes double as embedding-cache keys, so unchanged chunks are never re-embedded.

```
 client                                   server
 +-----------------+   root hash      +----------------------+
 | Merkle tree of  |----------------->| stored tree per      |
 | file hashes     |<-- diff paths ---| namespace            |
 +--------+--------+                  +----------+-----------+
          | changed files only                    |
          v                                       v
   chunk (AST) -> hash -> [embedding cache hit?] -> embed -> upsert
```

**Lesson.** Content-addressed chunks give you three properties at once: delta sync, embedding caching and cross-tenant deduplication. Where data is legitimately shared, as with identical code, deduplication can turn a cold-start problem into a copy.

### 1.5 Case study: Uber Genie's Spark ETL

Uber's on-call copilot Genie ingests internal wiki (Engwiki) and Stack Overflow content through a custom Spark application [31]. Data prep produces a dataframe of (source link, content). Embeddings are computed on Spark executors and pushed into Uber's in-house vector database. Secondary coverage names it Sia and describes a "pusher" with bootstrap and merge jobs [32]. A separate ETL joins Genie's responses, from Hive, with Slack metadata and user feedback, and feeds an LLM-as-judge evaluator [31].

The pattern to note is that the evaluation pipeline is itself a batch data pipeline that sits alongside ingestion. A later Uber iteration, Enhanced Agentic RAG, reported a 27% relative increase in acceptable answers and a 60% relative drop in incorrect advice in the security/privacy domain [32].

### 1.6 Case study: GraphRAG incremental updates

Graph-based indexes are harder to maintain incrementally because new content can change global structure. Microsoft's GraphRAG 1.0 added an `update` command. It computes deltas between the existing index and new content and merges them "to minimize re-indexing". Microsoft also notes that new content can alter the community structure enough that much of the index must be recomputed anyway [35]. LLM caching is on by default, so re-runs over unchanged content are much cheaper. Update runs write to a separately configured `update_output_storage` location "to preserve your original outputs", so the live index is not overwritten [35][51]. The archived Azure GraphRAG Solution Accelerator does not support incremental indexing; adding documents there means a re-index [36]. As of 2026 the GraphRAG repository itself describes the project as "largely in maintenance mode", not accepting new PRs or features [51], which matters for anyone planning to depend on its incremental-update path long term.

**Lesson.** Derived artifacts such as summaries, community reports and contextualized chunks have their own invalidation rules, which can be far coarser than per-chunk hashing.

### 1.7 Parsing is the least-tested ingestion stage

Change detection and embedding get most of the attention, but many quality incidents trace back to the parser: a PDF table flattened into a run of numbers, a two-column layout read across columns, headers and footers repeated into every chunk, or scanned pages that silently yield no text. Layout-aware converters such as IBM Research's open-source Docling, now hosted by the LF AI & Data Foundation, convert PDF, Office, HTML and other formats into a structured document model with reading order and tables preserved before chunking [44]. Operationally:

- Version the parser alongside the embedding model. A parser upgrade changes chunk text, and therefore every content hash, so it behaves like a re-embedding migration (Section 4) even when the model is unchanged.
- Track parse-level metrics per source: pages with empty text, OCR fallback rate, table count and characters per page. A sudden shift is an ingestion regression, not a retrieval one.
- Keep a small golden set of hard documents (scanned, multi-column, table-heavy) and diff parser output on every upgrade.

## 2. Freshness, deletes and permissions

### 2.1 Measuring freshness

Treat freshness as an SLO with three measured lags:

```
  t_src_change --(detect lag)--> t_event --(pipeline lag)--> t_indexed
           --(engine visibility lag)--> t_queryable
```

Each lag has a different owner:

- **Detection lag** depends on the connector. Webhooks or change data capture (CDC) take seconds. Polling takes as long as the polling interval. Nightly full crawls take up to 24 hours.
- **Pipeline lag** depends on queueing, batching and embedding throughput.
- **Engine visibility lag** depends on the store. Pinecone's serverless architecture acknowledges a write once it reaches a write-ahead log on object storage. An index builder pulls up to 10,000 records at a time into an in-memory memtable, and those records are queryable before they are flushed. Pinecone says "seconds is the norm" for freshness [9][10]. Pinecone also offers an eventually consistent read mode that bypasses the memtable for lower latency and higher availability, at the cost of a small amount of staleness [10]. turbopuffer writes each batch as a WAL file in object storage, and a namespace commits at most one WAL entry per second, with concurrent writes batched [11].

Dropbox Dash relies on periodic syncs plus webhooks for real-time changes. Secondary coverage notes that index-based retrieval "presents freshness challenges if rate limits aren't managed well" [34]: source API rate limits often set the floor for detection lag.

### 2.2 Deletes: the most common correctness bug

Practitioner write-ups repeatedly identify deletes as the main freshness failure [40]:

- Ingestion code is usually written for inserts, with updates and deletes added later.
- A row that has been hard-deleted from the source cannot be queried, so a timestamp-polling connector never learns it is gone. The orphaned vectors stay retrievable indefinitely.

The fixes, in increasing order of robustness:

1. **Full-scan reconciliation.** Periodically list all source IDs and delete index entries not in the set. This is LangChain `full` or LlamaIndex `UPSERTS_AND_DELETE` [2][3]; LangChain's `scoped_full` does not catch source-side deletions. It is correct but expensive, and dangerous if the listing is partial.
2. **Log-based CDC.** Tools such as Debezium emit explicit delete events from the database log [40].
3. **Soft deletes with tombstones in the source**, propagated as a `deleted=true` metadata update and filtered at query time, then physically purged later.
4. **Idempotent keys.** Key every write by (source_id, version), so retries and out-of-order events converge instead of duplicating [40].

A subtle variant is the *edit that shrinks a document*. If a 10-chunk page becomes 7 chunks and the pipeline upserts chunks 0-6 by position, chunks 7-9 survive as orphans. The record manager must delete the full previous chunk-ID set for the source, not only overwrite it.

### 2.3 Permission freshness

Access-control lists (ACLs) are freshness-sensitive data with security consequences:

- A nightly sync can let a departed employee query sensitive documents for hours [43].
- Amazon Bedrock Knowledge Bases' managed ACL filtering says explicitly that it is not a security boundary. Permission changes are not reflected until the next ingestion job unless the connector supports real-time ACL verification [37].
- Azure AI Search documents that changing SharePoint permissions without triggering an update leaves stale ACLs on already-ingested files [43].

Recommended practice [42][43]:

- Store ACL principals as chunk metadata and filter *inside* the vector query. Post-filtering wastes context and can silently collapse recall when most of the top-k results are unauthorized.
- Subscribe to permission-change events and run periodic full reconciliation.
- Set an explicit staleness target, for example five minutes.
- Include the permission scope in every cache key (Section 7).

Budigi and Sirigiri go further. They argue for one transactional store, PostgreSQL with pgvector and HNSW, so that similarity, freshness, filtering and tenant isolation are evaluated in a single SQL query, removing the cross-system consistency gaps [39].

### 2.4 Deletes as a privacy obligation

Deletes are also a compliance requirement. Embeddings are sometimes treated as anonymized derivatives of text, but Morris et al.'s Vec2Text showed that an iterative correction model can reconstruct much of the original input from a text embedding alone; the paper reports exact recovery of a large majority of 32-token inputs for the GTR-base encoder, and recovery of sensitive fields such as names from clinical notes [45]. The practical consequences:

- Treat vectors, and any payload text stored beside them, as having the same sensitivity as the source document.
- An erasure request (for example under GDPR's right to erasure) must reach every derived copy: vectors in all live index generations, the rollback collection of an in-flight migration (Section 4), embedding caches, semantic-cache answers that cite the document (Section 7), traces that captured retrieved text (Section 8), and backups or object-storage snapshots.
- Log-structured engines acknowledge a delete long before compaction physically removes the bytes (Section 3). If you must certify physical removal, check the engine's compaction and retention guarantees, not just query-time invisibility.

## 3. Index maintenance inside the vector engine

### 3.1 Graph indexes under churn

HNSW and similar graph indexes are optimized for reads, not mutation. Most engines implement delete as a tombstone: the node stays in the graph for routing but is filtered from results, and a background job repairs the graph later. Consequences seen in practice:

- **Recall drift.** As the deleted fraction rises, searches route through dead nodes, and neighbor lists fragment under insert/delete cycles. Practitioners recommend incremental writes plus periodic full rebuilds, for example weekly or monthly [40].
- **Cleanup cost.** Weaviate runs tombstone cleanup periodically, every `cleanupIntervalSeconds`. Newer versions expose `TOMBSTONE_DELETION_MIN_PER_CYCLE` (default 0; cycles with fewer tombstones are skipped), `TOMBSTONE_DELETION_MAX_PER_CYCLE` (default unlimited) and `TOMBSTONE_DELETION_CONCURRENCY` (default half the available cores). For a cluster with 300M-object shards, Weaviate's example values are a 1M minimum and a 10M maximum per cycle [13]. Users report CPU spikes at `tombstone_cleanup_begin` after bulk deletes [13].
- **Disk not reclaimed.** One Weaviate forum report describes deleting about 600k vectors with no change in LSM store size even after cleanup finished. A maintainer explained that space is not freed until segments compact [14].
- **Non-determinism after compaction.** Rebuilding a graph after compaction can yield different neighbors, because insertion order and deletion state shape the edges. Golden-set retrieval tests can therefore change after a compaction with no code change.

### 3.2 Log-structured engines

Object-storage-first engines turn mutation into an LSM-style problem. In Pinecone, the memtable flushes immutable L0 "slabs" to object storage, and compaction merges L0 slabs into larger L1 slabs [9][10]. turbopuffer keeps object storage as the source of truth, with NVMe and RAM caches for hot namespaces [11][12]. Deletes and updates become log entries, and compaction absorbs the graph-repair cost. Operationally you trade in-place graph repair for compaction scheduling, and cold-cache latency becomes the main tail-latency risk (Section 5).

### 3.3 Update-friendly and filter-aware index designs

Two research lines target the maintenance problems above directly.

- **Streaming updates without rebuilds.** FreshDiskANN (Singh et al., 2021) showed that a graph index can absorb a continuous stream of inserts and deletes while keeping recall stable, by consolidating deletes into the graph incrementally instead of rebuilding [46]. IP-DiskANN (2025) extends this with in-place deletion, and Microsoft's DiskANN library now cites the pair as the basis for real-time updates that need "no merges, rebuilds, or patches" [46][47]. If your engine depends on periodic full rebuilds to recover recall, these designs are the benchmark to ask vendors about.
- **Filtered search.** Section 2.3 recommends filtering ACLs inside the vector query. Naive pre-filtering can force brute-force scans, and post-filtering collapses recall when filters are selective. ACORN (Patel et al., SIGMOD 2024) makes HNSW predicate-agnostic by building a denser graph and traversing only nodes that pass the filter, so restrictive, arbitrary predicates such as per-user ACLs keep high recall without a separate index per predicate [48]. Filtered-DiskANN offers a label-based alternative [47]. Test recall at your real filter selectivities: a tenant or ACL that admits 0.1% of the corpus behaves very differently from one that admits 50%.

## 4. Re-embedding migrations

Changing the embedding model invalidates every stored vector: the old and new vector spaces are not comparable. Changing chunking settings has the same effect. Elastic's `semantic_text` documentation states that updated chunking settings are not applied to existing documents until they are reindexed [17]. A practitioner essay adds a quieter risk: even a "better" model silently re-ranks the whole corpus, so queries that used to work can regress [41].

### 4.1 Strategy catalogue

| Strategy | Downtime | Extra serving cost | Recall during migration | Rollback | Notes / sources |
|---|---|---|---|---|---|
| In-place re-index | Yes, or a mixed state | None | Broken while mixed | Hard | Rarely acceptable |
| Blue-green collection + alias swap | None | ~2x storage during copy | Old model until cutover | Swap alias back | Dual-write during backfill; needs the source text [15] |
| Named-vector side-by-side | None | Extra vector per point | Old model until cutover | Easy until old vector deleted | Collection must use named vectors; updates must write both; pause updates/deletes during backfill [15] |
| Shadow index + gradual traffic shift | None | ~2x during ramp | Measured per cohort | Feature flag | Ramp by percentage while watching retrieval metrics [41] |
| Query-side adapter (Drift-Adapter) | Near zero | Adapter cost only | ~95-99% of re-embed recall (paper) | Remove adapter | Defers full re-embedding [16] |

### 4.2 Blue-green and named vectors (Qdrant pattern)

Qdrant's zero-downtime guide gives two concrete recipes [15]:

1. **New collection.** Create a second collection for the new model, dual-write all changes to both, backfill by re-embedding the original text (from the primary database, or from payloads stored on the points), then switch search over.
   The backfill should use insert-only upserts so that it never overwrites a newer dual-written point, and the switch is best done by atomically repointing a collection alias. The cost is duplicated payloads, which is significant for text-heavy data [15].
2. **Named vectors.** Add a second named vector to each point in the existing collection (the guide targets Qdrant 1.18+ for this path), backfill only the new vector with `update_vectors`, and switch the query's `using` parameter. No second collection is needed, IDs and payloads stay intact, and rollback is easy until the old vector is deleted, after which restoring it means re-embedding. Conditions: the collection must have been created with named vectors, and the dual-write logic must update both vectors together, or they diverge [15].

Neither path makes deletes and in-place modifications free. In blue-green, a delete issued against the new collection can fail because the point has not been migrated yet, and the backfill can later re-add it. In the named-vector path, updates and deletes to existing points should be paused during backfill, since a concurrent update can let the backfill write an embedding of stale text. The guide lists the operations to pause or special-case: `delete`, `update_vectors`, `delete_vectors`, the payload mutations and `batch_update_points` [15].

```
            writes ---------------+-------------------+
                                  v                   v
                         +--------------+     +--------------+
   queries --> alias --> | coll_v1      |     | coll_v2      | <-- backfill
               "docs"    | (model A)    |     | (model B)    |     worker
                         +--------------+     +--------------+
   cutover: point alias "docs" -> coll_v2 ; keep v1 for rollback window
```

Operational checklist distilled from these sources:

- Persist the raw chunk text and chunk boundaries. Without them you cannot re-embed.
- Store `embedding_model_version` on every vector, and refuse queries that mix spaces.
- Freeze or dual-write during backfill, and reconcile the deletes that landed during the copy.
- Run the golden-set retrieval evaluation on both indexes before cutover, then ramp traffic gradually and never flip 100% at once [41].
- Budget beyond the embedding API bill: re-chunking, write amplification, index build CPU, and double storage for the rollback window [41].

### 4.3 Drift-Adapter: bridging spaces instead of re-embedding

Vejendla's Drift-Adapter (EMNLP 2025) takes a different route. It learns a lightweight map from new-model query embeddings into the legacy space, so the existing ANN index keeps serving while full re-embedding is deferred or skipped [16]. Three parameterizations are compared: orthogonal Procrustes, low-rank affine, and a small residual MLP. Each is trained on a small sample of paired old/new embeddings. On MTEB text corpora and a 1M-item CLIP image upgrade, the paper reports recovery of 95-99% of full-re-embedding Recall@10/MRR, under 10 microseconds of added query latency, and over 100x lower recompute cost than full re-indexing or dual-index serving [16].

Treat these as the authors' own results. The approach also caps quality at what the old document embeddings can express, so it is a bridge, not a destination.

## 5. Latency budgets

### 5.1 Where the time goes

A typical synchronous RAG request is a chain of network hops:

```
 client
   |-- auth/ACL lookup ............ 5-50 ms   (often underestimated)
   |-- query rewrite (LLM, opt.) .. 100-500 ms
   |-- query embedding ............ 5-50 ms
   |-- ANN + BM25 (parallel) ...... 5-100 ms  (cold object-store: 100s of ms)
   |-- rerank (cross-encoder) ..... 30-400 ms (depends on candidates x length)
   |-- prompt assembly + fetch .... 10-80 ms
   |-- LLM prefill  -> TTFT ....... dominant for long contexts
   '-- LLM decode ................. dominates total wall time
```

The ranges in the diagram are illustrative. The sourced numbers are these:

- NVIDIA's enterprise RAG sizing guide attributes about 80% of RAG time-to-first-token (TTFT) to LLM prefill over the augmented prompt. Reranking is about 10%, embedding about 5%, and vector search and the RAG server each under 2% [27]. It suggests scaling a component once it exceeds those shares, for example the reranker above 10% of TTFT [27].
- A ZeroEntropy playbook works through a 700 ms p95 search-request budget: first-pass retrieval 30-50 ms, a large cross-encoder reranker 300-400 ms, assembly and network 40-80 ms, and an LLM summary 150-300 ms. It notes that about half the reranker time is per-batch model-load and tokenization overhead [28].
- Dropbox Dash reportedly targeted sub-2-second responses for over 95% of queries [34]. Dropbox's earlier search-engine work found that metadata fetches for ACL checks and result decoration, not retrieval itself, drove much of the latency. The team found this only by instrumenting each component [33].
- turbopuffer's docs report a p50 of 874 ms for a first query against a 1M-document namespace read cold from object storage, versus 14 ms once cached on NVMe [11]. Cache warmth is therefore a first-class tail-latency variable for object-storage engines, and turbopuffer introduced namespace pinning (billed in GB-hours) for hot namespaces in 2026 (approximate, per secondary sources) [11].
- RAGCache, discussed in Section 7, overlaps retrieval with inference to hide retrieval latency [23].

### 5.2 Budgeting practice

1. Set a p95 budget per stage, not just end-to-end. Track percentiles per stage in traces, since averages hide the cold-cache and reranker-batch tails.
2. Run lexical and dense retrieval in parallel, and bound the reranker's candidate count. Reranking around 20 candidates with a small cross-encoder is a commonly cited compromise [28].
3. Stream tokens. For chat, TTFT is the user-perceived metric, and it is dominated by prefill, which grows with context length. Retrieving fewer, better chunks therefore reduces latency directly.
4. Pre-warm caches for tenants likely to be active, or pin hot namespaces on object-storage engines.
5. Make ACL and metadata lookups batchable and cacheable. They are a known hidden cost [33].

## 6. Cost modeling

### 6.1 Cost components

A useful model splits cost into one-time, steady-state and per-query terms:

```
 C_total = C_ingest_once + C_reembed_per_migration
         + months * (C_storage + C_index_compute + C_churn_embedding)
         + queries * (C_query_embed + C_search + C_rerank + C_llm_in + C_llm_out)
```

Reference points from public sources:

- **Embedding.** OpenAI lists `text-embedding-3-small` at $0.02 per 1M tokens and `text-embedding-3-large` at $0.13 [20]. Third parties report a 50% Batch API discount, to $0.01 per 1M, which is not confirmed on the page as extracted. Treat it as approximate. For most corpora, the initial embedding bill is small next to LLM generation. The operational cost is churn re-embedding and migrations, which is why Notion's hash-based Page State (reported 70% volume reduction) mattered [6].
- **Contextualized chunks.** Anthropic's Contextual Retrieval prepends LLM-generated context to each chunk. Anthropic estimates a one-time cost of $1.02 per million document tokens with prompt caching, assuming 8k-token documents, 800-token chunks and about 100 tokens of generated context per chunk. The top-20 retrieval failure rate fell from 5.7% to 3.7% (35%) with contextual embeddings, to 2.9% (49%) with contextual BM25 added, and to 1.9% (67%) with reranking added [38]. Enrichment like this multiplies ingestion cost and must be re-run on every content change, so it has to be included in the churn term.
- **Storage tiers.** turbopuffer's documentation gives approximate per-GB-month prices of about $0.02 for object storage, about $0.08 for SSD and about $2 for DRAM [11]. This roughly 100x spread is the economic argument for object-storage-first engines when most namespaces are cold. Notion's reported 60% search-engine savings after moving to turbopuffer is a concrete instance [6].
- **Vector compression.** Hugging Face and Mixedbread report that int8 scalar quantization gives 4x smaller vectors and binary quantization gives 32x. For `mxbai-embed-large-v1` on MTEB retrieval (NDCG@10), int8 retains about 97% of float32 performance (52.79 vs 54.39), and binary retains about 96.5% with rescoring but only about 92.5% without it [18]. The recommended pattern is to search a binary index, then rescore the top candidates with int8 or float32 vectors. The same post reports that Matryoshka truncation of `nomic-embed-text-v1.5` keeps 95.8% of performance at 3x compression and about 90% at 6x [18]. Vespa documents combining Matryoshka with binary quantization [19]. The int8 calibration set strongly affects quality, so recalibrate after domain shift [18].
- **Generation.** LLM input tokens usually dominate per-query cost, because retrieved context is long. Anthropic's prompt caching prices cache reads at 0.1x the base input price and 5-minute cache writes at 1.25x (1-hour writes at 2x) [21][22]. Anthropic advertised cost reductions of up to 90% and latency reductions of up to 85% for long prompts. In its launch examples, time to first token for a cached 100k-token prompt ("chat with a book") dropped from 11.5 s to 2.4 s [21].

### 6.2 Cost levers ranked by typical impact

1. Fewer, better context tokens per query, through reranking and tighter top-k. This also cuts latency (Section 5).
2. Prompt caching for stable prefixes, such as system prompts, tool definitions and frequently reused documents.
3. Change detection that avoids re-embedding and re-enriching unchanged content.
4. A storage tier matched to the access pattern: object storage for cold tenants, RAM or SSD only for hot ones.
5. Quantization and dimension truncation, with rescoring.
6. Self-hosting embedding models once volume justifies GPUs. Notion's Ray migration is an example [6].

## 7. Caching

RAG systems can cache at four layers. Each has its own invalidation problem.

| Layer | Key | Value | Hit-rate driver | Invalidation risk | Sources |
|---|---|---|---|---|---|
| Embedding cache | hash(chunk text, model ver) | vector | content churn | Model change: must include model version | [7][8] |
| Retrieval / semantic cache | query embedding (+ tenant, ACL scope) | retrieved ids or final answer | repeated intents | Stale answers after doc change; false positives; cross-tenant leaks | [25][26][42] |
| Provider prompt cache | exact prompt prefix | provider-side KV state | stable prefix reuse within TTL | Any byte change in the prefix misses | [21][22] |
| KV cache of retrieved docs | chunk id (+ position) | attention KV tensors | popular chunks | Non-prefix reuse needs recomputation | [23][24] |

### 7.1 Semantic caches

A semantic cache returns a previous answer when a new query embeds close to a cached one. GPTCache's documentation notes that such caches produce false positives on hits and false negatives on misses, and it reports hit ratio, latency and recall as the metrics to watch [25]. Its maintainers now state that they no longer add support for new APIs or models, so treat it as a reference design rather than a growing dependency [25]. Vendor guidance on similarity thresholds varies widely, from about 0.75 to 0.97 cosine. One vendor table pairs 0.95 with a 15-25% hit rate and 1-3% false positives, and 0.90 with a 35-55% hit rate and 7-15% false positives [26]. These are vendor figures without a published method.

The defensible practice is to label several hundred real query pairs from your own traffic, choose the threshold that meets a precision target, and consider per-category thresholds [26]. Two operational rules:

- Scope cache keys by tenant and permission set. A shared namespace with a loose threshold can serve one tenant's answer to another [42].
- Invalidate or TTL entries whose supporting documents changed. This requires storing the source document IDs alongside each cached answer.

### 7.2 Provider prompt caching

For RAG, put stable content first (system prompt, instructions, tool schemas, sometimes a fixed "core corpus") and retrieved chunks after it. Anthropic's cache has a short default lifetime that is refreshed on each hit, and the 1.25x write premium means sparse traffic can make caching cost more than it saves [21][22].

### 7.3 KV caching of retrieved knowledge

Two systems papers address the fact that retrieved chunks appear in different combinations and orders, which defeats ordinary prefix caching:

- **RAGCache** (Peking University and ByteDance) organizes the KV states of retrieved documents in a "knowledge tree" across GPU and host memory. It uses a replacement policy aware of RAG retrieval patterns and overlaps retrieval with inference. It reports up to 4x lower TTFT and up to 2.1x higher throughput than vLLM with Faiss [23].
- **CacheBlend** (EuroSys 2025 best paper) reuses precomputed per-chunk KV caches even when they are not a prefix, and selectively recomputes a small subset of tokens to restore cross-attention. It reports 2.2-3.3x lower TTFT and 2.8-5x higher throughput, with quality comparable to full prefill. Sources differ on whether the baseline is prefix caching or full recompute [24]. Because recomputation is pipelined with KV loading, caches can live on SSD; the LMCache team frames this as approaching a 100% KV-cache hit rate for RAG [24].

```
 retrieved: [c7][c2][c9]  ---> KV store (GPU/CPU/SSD) lookup per chunk
                                   |hit       |hit       |miss
                                   v          v          v
                              [KV c7]    [KV c2]   prefill c9
                                    \        |        /
                                selective recompute of ~few % tokens
                                             v
                                       decode answer
```

### 7.4 Prefix-aware serving engines

Since 2024, much of the KV-cache reuse that RAG depends on has moved into the serving layer. SGLang's RadixAttention keeps KV caches for all served requests in a radix tree and automatically reuses the longest shared prefix across requests, with cache-aware scheduling to raise hit rates [49]. vLLM's automatic prefix caching provides block-level prefix reuse, and LMCache extends both directions: it offloads KV blocks to CPU memory, disk or remote storage, shares them across engine instances, and, through CacheBlend, reuses blocks at non-prefix positions with selective recomputation [24][50]. Two operational consequences follow:

- **Routing is now a cache decision.** With several replicas, a load balancer that ignores cache contents scatters requests sharing a system prompt or popular documents across instances and destroys the hit rate. KV-aware or prefix-hash routing, offered for example in NVIDIA Dynamo and the vLLM production stack, should be part of the latency budget design [50].
- **Prompt layout is shared across systems.** The same rule as provider prompt caching applies: stable instructions first, retrieved chunks after, and nothing volatile such as timestamps or request IDs in the prefix.

## 8. Observability and tracing

### 8.1 What to trace

A RAG trace should let an engineer replay any answer. At minimum, each request records:

- the raw query, the rewritten queries and the router decision;
- for each retriever: index name, embedding model version, filters (including ACL scope), top-k IDs with scores, and latency;
- the reranker's input and output IDs, scores and the cut-off;
- the final prompt (or a hash plus template version), model, token counts in, out and cached, and TTFT;
- citations emitted and links to the source document versions;
- user feedback and any online evaluator scores.

### 8.2 Standards: OpenTelemetry GenAI and OpenInference

The OpenTelemetry GenAI semantic conventions now live in a dedicated `open-telemetry/semantic-conventions-genai` repository [29]. Its span specification lists well-known `gen_ai.operation.name` values including `chat`, `embeddings`, `retrieval`, `execute_tool` and `invoke_agent`, and defines a retrieval CLIENT span named `{gen_ai.operation.name} {gen_ai.data_source.id}` [29]. Cached input tokens are reported via `gen_ai.usage.cache_read.input_tokens` and counted within the input-token total, and companion metrics cover token usage and operation duration [29]. The conventions, including the retrieval span, are still marked Development, and the new repository had not yet published a stable schema URL. Pin a version, and expect attribute renames.

OpenInference, Arize's OpenTelemetry-based convention used by Phoenix, explicitly covers retrieval from vector stores and tool use. Its helpers extract retrieved documents and question/answer/reference triples from spans into dataframes, so evaluators such as Ragas faithfulness and context precision can run over production traces [30].

```
 app (LangChain / LlamaIndex / custom)
   | OTel SDK + GenAI / OpenInference instrumentation
   v
 OTel collector --> trace store (Phoenix / Langfuse / APM vendor)
                      |
                      +--> sampled traces --> offline evaluators (faithfulness,
                      |                       context precision, citation check)
                      +--> dashboards: p95 per span, cache hit %, $/query,
                                       empty-retrieval %, index lag
```

### 8.3 Online metrics worth alerting on

- **Freshness:** index lag (p50 and p95 of t_queryable - t_src_change), connector error rate, delete-propagation lag, ACL sync age.
- **Retrieval health:** empty or low-score retrieval rate, top-1 score distribution drift (a sudden shift often means a model or index change), percentage of queries with no reranked chunk above threshold.
- **Generation:** refusal or "I don't know" rate (high is FP1 Missing Content, low with bad answers is hallucination), citation coverage, answer length drift.
- **Cost and latency:** tokens in, out and cached per request, cache hit rates per layer, per-stage p95, cold-namespace query rate.

Uber's Genie shows the loop closing: production responses, Slack metadata and user feedback flow through an ETL into an LLM-as-judge evaluator [31]. Dash-style systems reportedly mine failed retrievals from production traces into test sets [34].

## 9. Failure modes and debugging playbooks

### 9.1 A taxonomy

Barnett et al.'s seven failure points come from three deployments in research, education and biomedical settings [1]. Commonly cited labels are:

1. FP1 Missing Content: the answer is not in the corpus, and the system answers anyway instead of declining.
2. FP2 Missed Top-Ranked Documents: the answer is in the corpus but ranks below the cut-off.
3. FP3 Not in Context (consolidation limits): the document was retrieved but dropped during context assembly.
4. FP4 Not Extracted: the answer is in the context, but the model fails to extract it.
5. FP5 Wrong Format.
6. FP6 Incorrect Specificity: the answer is too general or too specific.
7. FP7 Incomplete.

Operations adds failure classes that the paper does not separate:

- stale content (index lag);
- ghost content (undeleted orphans);
- permission leakage or over-filtering;
- mixed embedding spaces after a partial migration;
- cache poisoning or false-positive semantic hits;
- silent recall drift from index churn or compaction;
- cold-cache tail latency.

### 9.2 Triage flow

```
 bad answer reported
   |
   v
 1. Replay trace. Is the correct source doc in the corpus?  --no--> FP1 / ingestion
   |yes                                                         (connector, parser,
   v                                                             filter, ACL?)
 2. Is the current version indexed? compare src hash vs index --stale--> freshness
   |yes                                                                 playbook
   v
 3. Was it in retriever top-N (pre-rerank)? --no--> FP2: query rewrite, hybrid,
   |yes                                              chunking, model version mix?
   v
 4. Survived rerank / context assembly?  --no--> FP3: reranker, top-k, token budget
   |yes
   v
 5. In prompt but answer wrong?  --yes--> FP4-FP7: prompt, model, format, citations
   |
   v
 6. Was a cache involved? semantic hit / stale prompt cache --> cache playbook
```

### 9.3 Playbooks

**Playbook A: "It cites a document we deleted."**

1. Look up the chunk IDs in the record manager. If they are missing, they are orphans written outside the indexing API, which LangChain explicitly warns against [2].
2. Check whether the connector emits delete events. If it polls timestamps, it cannot see hard deletes [40].
3. Run a scoped full reconciliation for that source.
4. Purge semantic-cache entries that reference the IDs.
5. Add a canary: a synthetic document that is created and deleted hourly, with an alert if it remains retrievable after the SLO.

**Playbook B: "User saw something they should not have."**

1. Treat it as a security incident first.
2. Compare the chunk's ACL metadata with the live source ACL. A mismatch means sync lag [37][43].
3. Check whether a cache served the answer without a permission-scoped key [42].
4. Fix the filter-in-query path, shorten the ACL sync interval, and consider real-time ACL verification on the final candidates. AWS describes its managed ACL filtering as not a security boundary [37].

**Playbook C: "Retrieval quality dropped overnight; no deploy."**

1. Check the top-1 score distribution and the empty-retrieval rate for a step change.
2. Check whether a background compaction or rebuild ran, since HNSW rebuilds can change neighbors, and whether tombstone counts are rising [13][14].
3. Check whether any vectors carry a different `embedding_model_version`, which indicates a partially applied migration [15].
4. Check whether a parser or connector update changed the chunk text, and therefore the hashes, causing a mass re-embed with a new chunking.
5. Re-run the golden set against a snapshot if your engine supports one.

**Playbook D: "p95 latency doubled."**

1. Break the latency down per span [27][33].
2. Typical culprits are:
   - cold namespaces on object storage [11];
   - reranker batch size or candidate inflation [28];
   - longer contexts increasing prefill [27];
   - a prompt-cache miss caused by a changed prefix, for example a timestamp in the system prompt [22];
   - ACL lookup fan-out [33].

**Playbook E: "Costs jumped."**

1. Split by ingestion and query.
2. On the ingestion side, look for churn spikes, such as a source that rewrites metadata on every sync and invalidates hashes. Notion's separate hashing of text and metadata addresses this [6]. Also check for enrichment re-runs [38] and GraphRAG community recomputation [35].
3. On the query side, look at the context-token growth and cache hit rate per layer.

### 9.4 Comparison: operational posture of reference designs

| Design | Change detection | Delete handling | Freshness (public) | Migration approach | Cost lever highlighted |
|---|---|---|---|---|---|
| LangChain indexing API [2] | Doc hash + source id | `incremental` / `full` / `scoped_full` cleanup | Batch-dependent | Re-index into a new store | Skip unchanged docs |
| LlamaIndex IngestionPipeline [3] | Doc id + hash in docstore | `UPSERTS_AND_DELETE` | Batch-dependent | Re-run pipeline | Transformation cache |
| Notion [5][6] | Hashes of page text and metadata | Online Kafka path + rebuilds | Near real time | Full re-index during engine move | 70% less embed volume; 60% search savings |
| Cursor [7][8] | Merkle tree of file hashes | Tree diff | Periodic client sync (interval undisclosed) | Seeding from similar index | Index reuse; embed cache |
| Pinecone serverless [9][10] | Client-side | WAL + compaction | Seconds (vendor) | Client-side | Object storage + slabs |
| turbopuffer [11][12] | Client-side | WAL entries | 1 WAL commit/s per namespace | `copy_from_namespace` (reported) | ~$0.02/GB-mo object storage |
| Qdrant migration guide [15] | n/a | Pause or special-case during backfill (both paths) | n/a | Collections or named vectors | Rollback safety |
| GraphRAG 1.0 [35] | New/changed input files | Delta merge | Batch | `update` to separate output | LLM cache on by default |

## 10. Key takeaways for system designers

1. **Build the record manager first.** A durable map from source document to chunk IDs, hashes, embedding-model version and ACLs is what makes updates, deletes, migrations and debugging possible. Neither LangChain nor LlamaIndex will delete anything unless you opt in with a complete view of the source [2][3].
2. **Deletes and permissions are the freshness SLO that matters.** Prefer CDC or webhooks with explicit delete events, run periodic full reconciliation, filter ACLs inside the query, and measure delete- and ACL-propagation lag separately from insert lag [37][40][43].
3. **Hash at the finest granularity you can afford.** Separate content hashes from metadata hashes. Notion's selective re-embedding and Cursor's Merkle diffing show this is the largest ingestion cost lever [6][7].
4. **Plan every embedding change as a migration.** Version every vector, keep the raw text, and use alias or named-vector cutovers with dual writes and a rollback window [15]. Evaluate before and during the ramp. Adapters such as Drift-Adapter can buy time but cap quality at the old index [16].
5. **Expect index entropy.** Tombstones, compaction and rebuilds change recall and disk usage without code changes [13][14]. Schedule rebuilds, and re-run golden sets after them, or choose engines built for streaming updates and filtered search [46][48]. Remember that a delete is also a privacy obligation reaching every derived copy [45].
6. **Budget latency per stage, with percentiles.** Prefill usually dominates TTFT, so fewer, better context tokens save both latency and money [27]. Watch the hidden costs: ACL and metadata fetches [33], reranker batches [28] and cold object-storage reads [11].
7. **Model cost in three terms: one-time, churn and per-query.** Enrichment such as contextual chunks and graph summaries moves cost into the churn term [35][38]. Storage-tier choice and quantization change the steady-state term by an order of magnitude [11][18].
8. **Cache at the right layer, with the right key.** Embedding caches are keyed by content and model version. Semantic caches need tenant and ACL scope plus a doc-change invalidation path [26][42]. Prompt caches need byte-stable prefixes [22]. KV-cache systems such as RAGCache and CacheBlend target prefill directly [23][24].
9. **Make every answer replayable.** Trace retriever inputs, outputs, scores, model versions and prompt versions with OpenTelemetry GenAI or OpenInference conventions. Pin the convention version, because the spec is still in Development status [29][30].
10. **Debug with a fixed decision tree.** Walk each case through: in corpus, then current, then retrieved, then survived rerank, then in prompt, then extracted. Each failure point maps to a different owner and fix [1]. Feed every resolved incident back into the evaluation set [31][34].

## References

1. Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. "Seven Failure Points When Engineering a Retrieval Augmented Generation System." Deakin University / CAIN 2024, 2024. https://arxiv.org/abs/2401.05856
2. LangChain. "How to use the LangChain indexing API" (Record Manager, cleanup modes). LangChain docs, 2024-2025. https://python.langchain.com/docs/how_to/indexing
3. LlamaIndex. "Ingestion API reference" (docstore strategies). LlamaIndex, 2025. https://developers.llamaindex.ai/python/framework-api-reference/ingestion/
4. LlamaIndex. "Ingestion Pipeline + Document Management." LlamaIndex docs, 2024. https://docs.llamaindex.ai/en/latest/examples/ingestion/document_management_pipeline
5. Notion. "Two years of vector search at Notion: 10x scale, 1/10th cost." Notion Engineering Blog, 2026. https://www.notion.com/blog/two-years-of-vector-search-at-notion
6. ZenML LLMOps Database. "Notion: Scaling Vector Search Infrastructure for AI-Powered Workspace Search." ZenML, 2026. https://www.zenml.io/llmops-database/scaling-vector-search-infrastructure-for-ai-powered-workspace-search
7. Cursor. "Securely indexing large codebases." Cursor Blog, 2025-2026. https://cursor.com/blog/secure-codebase-indexing
8. Gupta, M. "How Does Cursor Index Your Codebase?" Personal blog, 2026. https://manthanguptaa.in/posts/how_cursor_index_your_codebase/
9. Pinecone. "How Pinecone Works." Pinecone, 2025-2026. https://www.pinecone.io/how-pinecone-works/
10. Pinecone. "Slab architecture" (serverless write path, memtable, compaction). Pinecone, 2025. https://www.pinecone.io/learn/slab-architecture/
11. turbopuffer. "Architecture." turbopuffer docs, 2025-2026. https://turbopuffer.com/docs/architecture
12. turbopuffer. "turbopuffer: fast search on object storage." turbopuffer blog, 2024. https://turbopuffer.com/blog/turbopuffer
13. Weaviate. "Vector index configuration" (HNSW tombstone cleanup settings) and "Environment variables" (`TOMBSTONE_DELETION_*`). Weaviate docs, 2025-2026. https://docs.weaviate.io/weaviate/config-refs/schema/vector-index and https://docs.weaviate.io/deploy/configuration/env-vars
14. Weaviate Community Forum. "Disk usage for weaviate and tombstone cleanup." 2025. https://forum.weaviate.io/t/disk-usage-for-weaviate-and-tombstone-cleanup/9870
15. Qdrant. "Migrate to a New Embedding Model with Zero Downtime in Qdrant." Qdrant docs, 2025. https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/
16. Vejendla, M. "Drift-Adapter: A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector Databases." EMNLP 2025. https://arxiv.org/abs/2509.23471
17. Elastic. "Semantic text field type reference." Elasticsearch docs, 2025. https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text-reference
18. Shakir, A., Aarsen, T., Lee, S. (Mixedbread and Hugging Face). "Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval." Hugging Face Blog, 2024. https://huggingface.co/blog/embedding-quantization
19. Bergum, J. K. "Combining Matryoshka with Binary-Quantization using the Hugging Face embedder." Vespa Blog, 2024. https://blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder/
20. OpenAI. "text-embedding-3-small" model page. OpenAI API docs, 2026. https://developers.openai.com/api/docs/models/text-embedding-3-small
21. Anthropic. "Prompt caching with Claude." Anthropic, 2024 (now served at claude.com). https://claude.com/blog/prompt-caching
22. Anthropic. "Prompt caching." Claude API documentation, 2025-2026. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
23. Jin, C. et al. "RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation." Peking University / ByteDance, 2024. https://arxiv.org/abs/2404.12457
24. Yao, J. et al. "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion." EuroSys 2025; with LMCache blog "CacheBlend (Best Paper @ ACM EuroSys'25)", 2025. https://arxiv.org/abs/2405.16444 and https://blog.lmcache.ai/2025-03-31-eurosys
25. Zilliz. "GPTCache: A Library for Creating Semantic Cache for LLM Queries." GitHub, 2023-2025. https://github.com/zilliztech/GPTCache
26. Portkey. "Semantic caching thresholds and why they matter." Portkey Blog, 2025. https://portkey.ai/blog/semantic-caching-thresholds/
27. NVIDIA. "RAG Scaling Guidelines." NVIDIA Enterprise Reference Architectures, 2025. https://docs.nvidia.com/enterprise-reference-architectures/enterprise-rag-retrieval-scaling-and-sizing-guide/latest/rag-scaling-guidelines.html
28. ZeroEntropy. "Reranker on the request path" (latency budget playbook). ZeroEntropy, 2026. https://zeroentropy.dev/playbooks/reranker-on-the-request-path/
29. OpenTelemetry. "Semantic Conventions for Generative AI (GenAI)" (spans, including retrieval; metrics). open-telemetry/semantic-conventions-genai, 2026; earlier versions at opentelemetry.io/docs/specs/semconv/gen-ai/. https://github.com/open-telemetry/semantic-conventions-genai
30. Ragas. "Observability" (Arize Phoenix / OpenInference integration). Ragas docs, 2025. https://docs.ragas.io/en/stable/howtos/observability/
31. Uber Engineering. "Genie: Uber's Gen AI On-Call Copilot." Uber Blog, 2024. https://www.uber.com/blog/genie-ubers-gen-ai-on-call-copilot
32. ZenML LLMOps Database. "Enhanced Agentic RAG for Internal On-Call Support Copilot" (Uber). ZenML, 2025. https://www.zenml.io/llmops-database/enhanced-agentic-rag-for-internal-on-call-support-copilot
33. Dropbox Tech. "Validating performance and reliability of the new Dropbox search engine." Dropbox, 2022. https://dropbox.tech/machine-learning/validating-performance-and-reliability-of-the-new-dropbox-search-engine
34. ZenML LLMOps Database. "Building a Universal Search Product with RAG and AI Agents" (Dropbox Dash). ZenML, 2025. https://www.zenml.io/llmops-database/building-a-universal-search-product-with-rag-and-ai-agents
35. Microsoft Research. "Moving to GraphRAG 1.0 - Streamlining ergonomics for developers and users." Microsoft Research Blog, 2024. https://www.microsoft.com/en-us/research/blog/moving-to-graphrag-1-0-streamlining-ergonomics-for-developers-and-users/
36. Microsoft Q&A. "How do I add new documents to my Microsoft Graph RAG solution accelerator deployed on Azure?" Microsoft Learn, 2025. https://learn.microsoft.com/en-us/answers/questions/5578950/how-do-i-add-new-documents-to-my-microsoft-graph-r
37. Amazon Web Services. "Access Control Lists awareness enablement" (Bedrock Knowledge Bases managed ACL). AWS docs, 2026. https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-acl.md
38. Anthropic. "Introducing Contextual Retrieval." Anthropic, September 2024. https://www.anthropic.com/news/contextual-retrieval
39. Budigi, V. K. P., Sirigiri, S. C. "Beyond Similarity Search: A Unified Data Layer for Production RAG Systems." arXiv, 2026. https://arxiv.org/abs/2605.03275
40. Pan, T. "RAG Knowledge Base Freshness: The Staleness Problem Teams Solve Last." tianpan.co, 2026. https://tianpan.co/blog/2026/04/20/rag-knowledge-base-freshness-index-rot
41. Pan, T. "The Embedding Upgrade That Silently Re-Ranks Your Entire Corpus." tianpan.co, 2026. https://tianpan.co/blog/2026-05-17-embedding-upgrade-silently-re-ranks-corpus
42. Let's Data Science. "RAG system design: permissions." 2025-2026. https://letsdatascience.com/blog/rag-system-design-permissions
43. Particula. "RAG document ACL sync and stale permissions." Particula Tech blog, 2025-2026. https://particula.tech/blog/rag-document-acl-sync-stale-permissions
44. Auer, C. et al. "Docling Technical Report." IBM Research, arXiv:2408.09869, 2024; project at https://github.com/docling-project/docling. https://arxiv.org/abs/2408.09869
45. Morris, J. X., Kuleshov, V., Shmatikov, V., Rush, A. M. "Text Embeddings Reveal (Almost) As Much As Text." EMNLP 2023. https://arxiv.org/abs/2310.06816 (code: https://github.com/jxmorris12/vec2text)
46. Singh, A., Subramanya, S. J., Krishnaswamy, R., Simhadri, H. V. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." arXiv:2105.09613, 2021. https://arxiv.org/abs/2105.09613
47. Microsoft. "DiskANN" repository (real-time updates via FreshDiskANN and IP-DiskANN, arXiv:2502.13826; Filtered-DiskANN). GitHub, 2025-2026. https://github.com/microsoft/DiskANN
48. Patel, L., Kraft, P., Guestrin, C., Zaharia, M. "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." SIGMOD 2024. https://dl.acm.org/doi/10.1145/3654923 (code: https://github.com/stanford-futuredata/ACORN)
49. Zheng, L. et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024. https://arxiv.org/abs/2312.07104
50. LMCache. "LMCache" repository (KV offloading, cross-instance sharing, non-prefix reuse via CacheBlend; vLLM and NVIDIA Dynamo integration). GitHub, 2024-2026. https://github.com/LMCache/LMCache
51. Microsoft. "GraphRAG" repository and configuration docs (`update_output_storage`; maintenance-mode notice). GitHub, 2024-2026. https://github.com/microsoft/graphrag
