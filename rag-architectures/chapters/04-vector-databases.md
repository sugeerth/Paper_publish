# 04. Vector Databases and ANN Indexing: From In-Memory Graphs to Object-Storage-Native Engines

## Overview

The retrieval stage of almost every RAG system ends in an approximate nearest neighbor (ANN) query. A query embedding goes in, and the top-k most similar stored vectors come back, usually restricted by metadata (tenant, ACL, date, document type). The component that answers that query is the vector database. It may be a dedicated engine (Pinecone, turbopuffer, Milvus, Weaviate, Qdrant, Chroma), a library embedded in a table format (Lance), or a vector index added to a general-purpose engine (pgvector, Elasticsearch/OpenSearch, Vespa, MongoDB, Redis).

This chapter treats the vector database as a systems problem with four coupled axes:

1. **Index structure.** Graphs (HNSW, Vamana/DiskANN), partition or cluster indexes (IVF, SPANN/SPFresh, DiskBBQ), and the quantizers layered on them (PQ, SQ, binary, RaBitQ, BBQ). These set the recall/latency/memory trade-off.
2. **Filtering.** Metadata predicates can break an ANN index's connectivity or its partition pruning. Getting filtered recall right is the hardest correctness problem in production vector search.
3. **Multi-tenancy.** RAG products usually serve thousands to hundreds of millions of small, isolated corpora. Tenancy layout drives cost far more than raw index speed does.
4. **Storage architecture.** The field has moved from "the whole index lives in RAM on a fixed shard set" toward "object storage is the source of truth, and compute is a cache." Pinecone serverless, turbopuffer, Milvus 2.6 (Woodpecker, tiered storage), distributed Chroma, and Lance are all versions of this move.

```
                 the generic vector-database data path (2024-2026 designs)
 writes                                                               queries
   |                                                                     |
   v                                                                     v
+--------+   +-----------------+   +------------------+   +-----------------------+
| router |-->| WAL / log       |-->| index builder /  |-->| immutable index files |
| (auth, |   | (object storage,|   | compactor        |   | (object storage)      |
| quota) |   |  Kafka, quorum) |   | (cluster, quant.,|   +-----------+-----------+
+--------+   +--------+--------+   |  graph build)    |               | lazy load
                      |            +------------------+               v
                      |  tail unindexed suffix          +-----------------------------+
                      +-------------------------------->| query node: RAM/SSD cache,  |
                         ("freshness layer")            | ANN + filter + merge + top-k|
                                                        +-----------------------------+
```

Sections 4.1 to 4.3 cover index families, filtering and tenancy as cross-cutting techniques. Section 4.4 walks through each system's architecture. Section 4.5 pulls out the object-storage-native patterns, Section 4.6 compares the systems in one table, and Section 4.7 covers further designs that sit beside the main systems: object-store-native vector buckets, GPU graph indexes, data-oblivious quantizers, multi-vector encodings and Postgres extensions beyond pgvector.

---

## 4.1 ANN index families

### 4.1.1 HNSW: the default in-memory graph

**Design.** Hierarchical Navigable Small World [1] builds a multi-layer proximity graph. Every vector is in layer 0, and each higher layer holds an exponentially smaller sample. A query greedily descends from a top-layer entry point, then does a best-first beam search (beam width `ef`/`ef_search`) in layer 0. Build parameters are `M` (max links per node per layer) and `ef_construction`. HNSW needs no training step, so pgvector notes that an HNSW index can be created on an empty table, unlike IVFFlat [24].

**Cost model.** HNSW is memory-bound. Faiss, the reference ANN library [4], gives the per-vector footprint as `d*4 + M*2*4` bytes for float32 vectors [3]. Weaviate's sizing table puts nodes at 4 bytes x dimensions (2 to 12 kB) and edges at about 10 B x 20 connections (about 200 B), so 100M vectors need 200 to 1,200 GB for vectors alone [40]. Quantizing the vectors does not shrink the graph. MongoDB documents that automatic quantization cuts `mongot` RAM by 3.75x (scalar) or 24x (binary) even though vector values shrink by 4x and 32x, "but the HNSW graph itself does not shrink" [59].

**Typical defaults.** pgvector uses `m=16`, `ef_construction=64` and `hnsw.ef_search=40` [24]. Vespa examples use `max-links-per-node` 16 to 24 and `neighbors-to-explore-at-insert` 100 to 200 [51]. Qdrant's default `m=16` gave about 21 links per node on layer 0 in a 1M-point benchmark [20].

**Failure modes.**
- *Updates and deletes.* Pinecone argued in 2023 that bolted-on HNSW indexes see memory consumption "grow... a lot" under frequent updates and deletes, and need parameter retuning or periodic rebuilds [26]. Weaviate marks deleted objects as tombstones that stay connected to the graph until an async cleanup reassigns edges [40]. pgvector advises `REINDEX INDEX CONCURRENTLY` before `VACUUM`, because vacuuming HNSW is slow [24].
- *Segment fan-out.* Lucene-based engines (Elasticsearch, OpenSearch) build one graph per immutable segment, so a query searches many graphs and merges results. Vespa explicitly advertises "one graph per tensor field per content node," which is mutable in place and has no segmented graphs to fan out over [51].
- *Filtering breaks connectivity* (Section 4.2).

### 4.1.2 IVF, PQ and IVF-PQ

**Design.** An inverted-file (IVF) index runs k-means to get `nlist` centroids, assigns each vector to its nearest centroid's posting list, and at query time scans the `nprobe` closest lists. Product quantization (PQ) [2] splits each vector into `M` sub-vectors and replaces each with a codebook index. Distances are then computed asymmetrically with per-query lookup tables ("asymmetric hashing" in ScaNN's terms [16]).

**Sizing rules from primary sources.**
- pgvector IVFFlat: `lists = rows/1000` up to 1M rows and `sqrt(rows)` beyond, with `probes = sqrt(lists)` as a starting point [24].
- Faiss: `K` between `4*sqrt(N)` and `16*sqrt(N)` below 1M vectors. Above that, `IVF65536_HNSW32` (1M to 10M), `IVF262144_HNSW32` (10M to 100M) and `IVF1048576_HNSW32` (100M to 1B), where an HNSW index over the centroids does cluster assignment. Training needs 30*K to 256*K samples [3].
- Faiss storage: `PQ<M>` is M bytes per vector, and the 4-bit FastScan variant `PQ<M>x4fs` is M/2 bytes [3].

**Trade-offs.** IVF builds faster and uses less memory than HNSW, but its speed/recall curve is worse [24]. It also needs representative training data, and recall drifts when the data distribution moves away from the trained centroids. Pinecone describes this as the difficulty of "maintaining well-separated clusters as the data distribution changes over time" [25]. That drift motivates the incremental-clustering designs in Section 4.1.5.

### 4.1.3 Quantization as a first-class layer

By 2025-2026 almost every engine treats quantization as orthogonal to the index structure. Lance makes this explicit, naming each index `{clustering}_{sub_index}_{quantization}` (for example `IVF_PQ`, `IVF_HNSW_SQ`, `IVF_RQ`) [48].

| Technique | Bits/dim | Where documented | Notes |
|---|---|---|---|
| Scalar int8 (SQ8) | 8 | ES `int8_hnsw`, Faiss, Lance SQ, MongoDB | 4x smaller; ES keeps raw floats on disk (+~25%) [54] |
| int4 | 4 | ES `int4_hnsw` | +~12.5% disk for quantized copy [54] |
| Product quantization | ~0.5-2 | Faiss, Milvus IVF_PQ, DiskANN in-RAM codes | needs training; codebooks [2][3][38] |
| Binary (sign bit) | 1 | Qdrant BQ, pgvector `binary_quantize`, Vespa bit | Qdrant: "up to 40x" speed-up; poorer below ~1024 dims [23] |
| RaBitQ | 1 (+multi-bit) | Faiss, Milvus, Lance, turbopuffer, ES, Qdrant, Weaviate | random rotation + theoretical error bound [17] |
| BBQ (Elastic) | 1 index / 4 query | Elasticsearch | 32x plus 14 bytes of corrective factors per vector [53] |
| Rotational quantization (RQ) | 8/4/1 | Weaviate | training-free; used by HFresh [40] |

RaBitQ [17] (SIGMOD 2024) has become the de facto 1-bit scheme. Its library README [63] lists adoption in Milvus, Faiss, NVIDIA cuVS, VectorChord, Elasticsearch, turbopuffer, LanceDB, Qdrant and Weaviate. Faiss sizes it at `d/8 + 8` bytes per vector, recommends `IVF<K>,RaBitQfs<N>` for speed, and notes that a random rotation is required for accuracy [3].

The universal companion technique is **oversample then rescore**. The quantized index returns `k * oversample` candidates, which are rescored against full-precision vectors kept on disk. Qdrant exposes `oversampling` and `rescore` per query [23]. Elasticsearch BBQ ties default oversampling to bit width (for `bbq_disk`, `bits=1` gives oversampling 3.0 and `bits=2` gives 1.5) [53]. OpenSearch's `on_disk` mode defaults to 32x compression with rescoring and `oversample_factor` 2.0 [55]. Pinecone's VQ-bench (September 2026, with a VecDB@VLDB 2026 paper) describes seven quantization primitives, re-expresses 25 common quantizers as pipelines of them, and benchmarks them in one MIT-licensed harness. Its authors note it favors composability over kernel speed [65].

### 4.1.4 DiskANN / Vamana: graphs that live on SSD

**Problem.** HNSW needs the graph and vectors in RAM. DiskANN [5] (NeurIPS 2019) put the graph and full-precision vectors on SSD and kept only PQ codes in memory. Microsoft's research overview describes it as "the first practical, high-performance SSD-based index that could index 10x more vectors per machine than previous in-memory systems" [8].

**Search loop.** As Milvus's DISKANN documentation describes it [38]:

```
 query --> entry point (near global centroid)
   loop:
     expand frontier using PQ codes held in RAM  (cheap approximate distances)
     pick best beam_width candidates
     read their SSD blocks (full vector + adjacency list in one I/O)
     compute exact distances, update candidate list
   until candidate list converges --> top-k
 knobs: max_degree, search_list_size, beam_width_ratio,
        pq_code_budget_gb_ratio, search_cache_budget_gb_ratio
```

**Lineage.** Follow-up work covers streaming updates with stable recall through merges (FreshDiskANN [6]) and in-place edits (IP-DiskANN [64]), filtered search (Filtered-DiskANN [7]), and a 50-billion-point index distributed across 1,000 machines [8]. The current Rust rewrite, "DiskANN3" (the legacy C++ code now lives on a `cpp_main` branch), is a *stateless orchestrator* that reads and writes index terms through a `DataProvider` trait the host database implements. This lets it inherit the host's durability [9]. Providers include in-memory, SSD, Garnet (a key-value store) and Bf-tree, and the Azure Cosmos DB integration is described in a VLDB 2025 paper [11]. DiskANN ideas are also used in pgvectorscale, DataStax JVector, Intel SVS (exposed in Redis as `SVS-VAMANA`), and NVIDIA cuVS [8][61]. The NeurIPS 2021 and 2023 Big-ANN competitions, co-organized by the same group, supplied many of the public billion-scale, filtered and streaming benchmarks used to compare these designs [8][62].

**Vendor-reported numbers.** Microsoft reports that DiskANN3 with the Garnet provider, measured with VectorDBBench on Wikipedia-10M Cohere embeddings (768-d) while ingesting 1,000 vectors/s in the background, held p99 under 5 ms with roughly 2x the QPS of the next-best system [66]. These are self-reported results.

### 4.1.5 SPANN and SPFresh: cluster indexes that update in place

SPANN [12] (NeurIPS 2021) keeps only cluster centroids in memory, with a graph or tree over them, and stores posting lists on SSD. Boundary vectors are assigned to several nearby posting lists so that a small `nprobe` still finds them. SPFresh [13] (SOSP 2023) adds a lightweight rebalancing protocol: posting lists that grow too large are split and their neighbors reassigned locally, so the index stays balanced without global rebuilds.

This family matters because it fits object storage. turbopuffer's docs argue that "a centroid-based index works well for object storage as it minimizes roundtrips and write-amplification, compared to graph-based indexes like HNSW or DiskANN": a cold query downloads the centroid index, then fetches every selected cluster "in one, massive roundtrip" [28]. Weaviate's HFresh index (based on SPFresh) keeps an HNSW over centroids in RAM with 8-bit RQ, stores posting lists on disk with 1-bit RQ, and rescores the top candidates against uncompressed vectors. Its tuning knobs are `searchProbe` and `replicas`, the number of posting lists each vector joins [40]. Elasticsearch's DiskBBQ uses hierarchical k-means with a default `cluster_size` of 384 vectors per cluster (configurable from 64 to 65,536) [54]. Its docs say it "performs well for recall levels up to around 95%", while 99%+ recall favors HNSW [53].

### 4.1.6 ScaNN

Google's ScaNN [14][15][16] runs in three phases: optional partitioning (a k-means tree), scoring with anisotropic vector quantization (AVQ), and optional exact rescoring. AVQ weights the quantization error component parallel to the datapoint more heavily, because that component dominates inner-product error. SOAR (NeurIPS 2023) adds "spilled" assignments of a vector to extra partitions, chosen so their residuals are orthogonal. The documented rules of thumb are brute force below about 20k points, AH plus rescore below 100k, partition plus AH plus rescore above that, `num_leaves` around `sqrt(n)`, and `dimensions_per_block=2` [16].

### 4.1.7 Index-family comparison

| Family | Memory residency | Update behavior | Object-storage fit | Filter handling | Representative systems |
|---|---|---|---|---|---|
| HNSW (float) | graph + vectors in RAM | incremental insert; deletes leave tombstones | poor (random access) | graph traversal with allow-list; ACORN | pgvector, Qdrant, Weaviate, Vespa, Redis, Lucene |
| HNSW + 1-bit/SQ | graph + codes in RAM, floats on disk | same | poor | same | ES `bbq_hnsw`, Qdrant BQ, OpenSearch `on_disk` |
| IVF / IVF-PQ | centroids (+codes) in RAM | needs retraining as data drifts | good (sequential lists) | list-level pruning plus post-filter | Faiss, Milvus, Lance, pgvector IVFFlat |
| Vamana / DiskANN | PQ codes in RAM, graph on SSD | Fresh/IP-DiskANN streaming | moderate (SSD) | label-aware graph (Filtered-DiskANN) | Milvus DISKANN, Cosmos DB, pgvectorscale, JVector |
| SPANN / SPFresh | centroid index in RAM | local split/reassign | very good | cluster-level filter indexes | turbopuffer, Weaviate HFresh, Chroma Cloud |
| Hierarchical k-means + BBQ | centroids in RAM | segment merges | good | segment fallback to exact | Elasticsearch DiskBBQ |
| ScaNN (partition + AVQ) | codes in RAM | batch-oriented | good | partition pruning | ScaNN library |

---

## 4.2 Filtered vector search

### 4.2.1 Why filters break ANN

Every RAG query carries filters: tenant ID, ACL group, recency, source. There are three naive strategies, and each fails in its own way.

- **Post-filtering.** Run ANN, then drop non-matching results. pgvector gives the arithmetic: with HNSW's default `ef_search=40` and a filter matching 10% of rows, "only 4 rows will match on average" [24]. Weaviate notes that with restrictive filters the result set can be empty [41].
- **Pre-filtering by brute force.** Compute the allowed set, then scan it exactly. This is exact and works well when the allowed set is small, but it is O(matches) in the general case.
- **In-graph filtering.** Traverse the graph but only admit allowed nodes into results. Qdrant's 2019 analysis framed the failure using percolation theory: if a fraction of nodes uncorrelated with the vectors is removed, the graph fragments below a threshold near `1/<k>` [19]. Qdrant's 2026 benchmark makes this concrete. With about 21 links per node, filtering out 96% of points leaves fewer than one surviving link per node on average, so traversal is stranded [20].

### 4.2.2 Repairs at index time: filterable HNSW and Filtered-DiskANN

**Qdrant filterable HNSW** [19][20] adds extra HNSW edges among points that share a payload value. These edges are built per indexed field, using the same construction algorithm. For categorical fields this at most doubles total edges regardless of category count [19]. Numeric ranges reduce to equal-population buckets with neighbor links, and geo filters to geohash cells [19]. The 2026 article quantifies the cost and the gaps [20]:

- Build time on 1M points was 116 s without extra edges and 507 to 652 s with them (4.4x to 5.6x).
- A value shared by more points than `segment_points / avg_links * 4` gets no extra edges (about 19% of the collection in the benchmark), because the main graph is assumed to keep that many points connected.
- Edges are per field, never per combination, so `AND` intersections are uncovered.

**Filtered-DiskANN** [7] builds label-aware Vamana graphs. The DiskANN team's 2026 evaluation compares five filtered-search algorithms on one graph: inline, beta (distance discount for matching points), multi-hop, two-queue, and adaptive-L, which re-scales the search list after 1,000 distance computations based on the observed match rate. The evaluation uses an 8M-document caselaw dataset embedded with OpenAI text-embedding-small, stratified by filter specificity [10].

### 4.2.3 Repairs at query time: ACORN and its descendants

ACORN [18] (SIGMOD 2024) has two variants. ACORN-gamma expands neighbor lists at build time, at 8.8x to 33.1x HNSW build time per the paper's table as quoted by Qdrant [20]. ACORN-1 keeps a standard HNSW and, when a neighbor fails the predicate, evaluates neighbors-of-neighbors at search time. Implementations shipping in 2026 include:

- **Weaviate.** ACORN is the default filter strategy from v1.34. It ignores non-matching objects in distance computations, explores multi-hop neighborhoods, and seeds extra entry points that match the filter [41]. The alternatives are `sweeping` (traverse normally, admit only allowed IDs) and `pathseer`, from the SIGMOD 2026 PathSeer paper [45]. PathSeer applies ACORN-style two-hop checks while the result set is filling and skips non-matching neighbors of non-matching candidates once it is full. Weaviate reports PathSeer fastest or within a few percent on random, topical and combined filters, with ACORN still better for sparse filters negatively correlated with the query [41].
- **Qdrant.** ACORN-1 is an opt-in per-query parameter, applied only below `max_selectivity` (default 0.4) [20].
- **Vespa.** It supports pre- and post-filtering, beam exploration, and "filtering before distance calculation ('Acorn 1')" [51].

**The Qdrant v1.18.2 benchmark** (1M deep-image-96 vectors, recall@10 at `hnsw_ef=64`) is the most detailed public head-to-head [20]:

| Filter (selectivity) | Plain HNSW | HNSW + ACORN-1 | Filterable HNSW | Planner + ACORN |
|---|---|---|---|---|
| One keyword (20%) | 62.9% @ 1.6 ms | 98.9% @ 4.4 ms | 94.8% @ 1.2 ms | 99.9-100% |
| One keyword (1%) | 0.1% @ 1.6 ms | 67.7% @ 4.7 ms | 99.8% @ 1.0 ms | 100% @ 1.5 ms |
| Two keywords AND (4%) | ~0.1% | 95.2% @ 7.7 ms | 63.7% @ 1.2 ms | 99.9% @ 13.9 ms |
| Two keywords AND (0.012%) | n/a | 0.6% @ 2.6 ms | 1.8% @ 2.6 ms | 100% @ 1.3 ms |

The lessons are that index-time edges win for single, mid-selectivity filters; ACORN wins where edges do not exist (values over the cap, AND intersections); and for very selective filters the right move is to skip the graph and read the payload index directly. On the 1% intersection, the planner sent 471 of 500 queries straight to the payload index [20].

### 4.2.4 Cost-based planners: the convergent design

Every mature engine has converged on a cardinality-estimating planner that picks among exact scan of the allowed set, filtered graph traversal, and post-filtering.

```
            estimate |allowed| from inverted / payload / bitmap index
                                   |
          +------------------------+-------------------------+
          | tiny                   | medium                  | large (>~ 40-95%)
          v                        v                         v
   exact scan of allowed IDs  graph walk w/ allow-list   plain ANN + post-filter
   (payload index, bitmaps)   (+ACORN / extra edges)     (filter rarely bites)
          |                        |                         |
          +---------- fallback: if ANN returns < k, rerun exactly -------+
```

- **Qdrant**: `full_scan_threshold` (in KB of vector data) routes to the payload index [20].
- **Weaviate**: the flat-search cutoff, about 15% of the dataset in the documented example, switches to brute force, and allow-lists come from roaring-bitmap inverted indexes [41].
- **Vespa**: pre-filtering by default, with `approximate-threshold` and `post-filter-threshold` per rank profile. Pre-filter evaluation can be multi-threaded (since 8.78), but HNSW search is always single-threaded per query [51].
- **Elasticsearch/Lucene**: per segment, if the filtered count is no greater than the candidates the search would examine, the graph is skipped. If graph exploration visits more nodes than there are matching docs, it stops and scores the matches directly. Only filters *inside* the kNN clause pre-filter; a sibling `bool.filter` post-filters [52].
- **OpenSearch Faiss engine**: a `filtered_exact_search_threshold`, plus an exact-search fallback when ANN returns fewer than k results even though at least k documents match (this can be disabled for latency) [56].
- **pgvector 0.8.0+**: iterative index scans (`strict_order` or `relaxed_order`) keep scanning until enough rows pass, bounded by `hnsw.max_scan_tuples` (default 20,000) or `ivfflat.max_probes`. The alternatives it suggests are partial indexes for a few distinct values and partitioning for many [24].
- **turbopuffer**: "native filtering" makes its clustering index filter-aware, so the engine can use exact metadata indexes to skip clusters with no matching documents instead of post-filtering [28][34].

---

## 4.3 Multi-tenancy

RAG platforms (per-user notes, per-customer knowledge bases, per-repo code search) have a long-tailed tenant distribution. The design question is where the tenant boundary sits physically.

| Pattern | Physical isolation | Typical scale | Systems |
|---|---|---|---|
| Separate collection/index per tenant | full | low (Qdrant Cloud default caps clusters at 1,000 collections) | Qdrant (discouraged) [21], Chroma (encouraged) [46] |
| Namespace as hard partition inside an index | segments never mix tenants | 100s of thousands of namespaces per index | Pinecone serverless [25] |
| Namespace = object-storage prefix | separate WAL and index per namespace | 250M+ namespaces observed | turbopuffer [28][30] |
| Shard per tenant | own vector index, inverted index, files | ~18-19k active tenants per node in a test cluster | Weaviate [43] |
| Shared shard, payload partition + per-tenant graph | tenant-colocated storage, `payload_m` graphs | many small tenants | Qdrant `is_tenant`, `m=0` [21] |
| Tiered: shared fallback shard + promoted dedicated shards | mixed | skewed tenant sizes | Qdrant 1.16+ [21] |
| Hash-to-partition key | partitions shared by many tenants | beyond the 1,024-partition manual cap | Milvus partition key [39] |
| SQL list partitioning | Postgres partitions | moderate | pgvector [24] |

Design notes from the sources:

- **A shared ANN index leaks performance across tenants.** pgvector warns that "vectors from one tenant can affect recall (and speed) for other tenants" in a shared approximate index [24]. Qdrant's fix is to disable the global graph (`m=0`) and build per-tenant subgraphs (`payload_m=16`), at the cost of slower global queries [21].
- **Tenant temperature is a first-class state.** Weaviate tenants can be `Active` (RAM/disk), `Inactive` (disk only, not queryable, fast to reactivate) or `Offloaded` (moved to S3 through the `offload-s3` module) [42]. Its dynamic index starts each tenant as a disk-backed flat index and switches it one-way to HNSW past 10,000 objects (by default, with async indexing) [40].
- **Tenant promotion should be transparent.** Qdrant tiered multitenancy routes writes with a `(target, fallback)` shard-key pair, so a growing tenant can be moved to its own shard through internal shard transfer without application changes [21].
- **Statistics leak too.** In Qdrant, BM25/IDF statistics are shard-wide by default, so v1.19 added a per-query `idf` filter to scope IDF to one tenant [21]. Lexical scoring in multi-tenant hybrid search (Chapter 5) has to account for this.

---

## 4.4 System architectures

### 4.4.1 Pinecone serverless

**Problem.** Pinecone's pod architecture used scatter-gather over shards that held their whole index locally, which made cost proportional to total data rather than data touched. Pinecone illustrated this with two workloads: Gong's infrequent labeling queries over billions of vectors, and Notion's thousands of tenants with sub-100 ms interactive queries [25].

**Architecture (January 2024).** Blob storage is the source of truth [25].

```
 writers --commit--> blob storage + log (sequences all mutations)
                          |                         |
                 tail log |                         | tail log
                          v                         v
          +-------------------------+     +------------------------+
          | index builder           |     | freshness layer        |
          | geometric partitioning  |     | compact index over data|
          | (coarse->fine hierarchy)|     | newer than manifest's  |
          | immutable segments ->   |     | max indexed version    |
          | blob; update manifest   |     +-----------+------------+
          +-----------+-------------+                 |
                      | segments (lazy load + cache)  |
                      v                               v
          +--------------------------------------------------------+
          | query executors: load only relevant partitions of the  |
          | target namespace, merge with freshness results         |
          +--------------------------------------------------------+
```

Key techniques:
- **Geometric partitioning.** Centroids represent regions, and new points go to the nearest centroid. A hierarchy of partitions is refined as data arrives, so the index can be maintained incrementally instead of retrained.
- **A lambda architecture for freshness.** Indexed segments plus a freshness layer make writes searchable "within seconds."
- **Namespaces as hard partitions.** Builders only partition within a namespace, so a query touches only that namespace's segments [25].

**Numbers (Pinecone-reported).** Gong reports a 10x cost reduction after moving to serverless, which Pinecone's Gong case study also states. The launch post's latency comparisons against pods (an average reduction of roughly half, with larger gains on a Cohere-768 dataset) could not be re-verified for this edition, so treat them as indicative. Cold starts are "a couple of seconds" for most datasets and up to about 20 s at billion scale [25]. Pinecone has since described a "slab-based architecture" update, linked from the same post [25].

**Evolution.** In December 2025 Pinecone added Dedicated Read Nodes (DRN): provisioned replicas for throughput and shards for capacity, with data kept warm in memory and local SSD, hourly per-node pricing, and the same write path as on-demand [27]. This mirrors turbopuffer's namespace pinning (Section 4.4.2). Both object-storage-first vendors added a reserved-capacity tier for sustained high-QPS workloads.

### 4.4.2 turbopuffer

**Problem.** turbopuffer is built for very many namespaces (per user, per repo), most of them cold at any moment, where RAM-resident indexes are uneconomical.

**Architecture** [28][29]:

```
 client --API--> stateless ./tpuf query nodes (multi-tenant, Rust)
                 |  route by namespace for cache locality; any node can serve any ns
                 v
          +----------------------+   miss   +------------------------------------+
          | memory + NVMe cache  |--------->| object storage, prefix per         |
          +----------------------+          | namespace:                         |
                                            |  /wal/  one file per group commit  |
                                            |  /index/ centroids.bin,            |
                                            |          clusters-N.bin,           |
                                            |          exact filter + BM25 idx   |
                                            +------------------------------------+
 writes: buffer -> group commit (<= 1 WAL entry/s/ns) -> CAS -> async indexer
 reads (strong): indexed data + scan of unindexed WAL tail; ~10 ms consistency check
```

Documented numbers [28][29][30]:
- Cold query p50 is 874 ms on 1M documents, and warm p50 is 14 ms. ANN and BM25 indexes are object-storage-optimized for about 500 ms cold latency on 1M docs.
- Writes return only after a durable object-storage commit: p50 of 285 ms for 500 kB, "up to 200ms" in the tradeoffs page, and 10,000+ vectors/s throughput. The documented per-namespace limit is 10k writes/s at 32 MB/s, and at most 2 GB of ingested-but-unindexed data.
- Reads are strongly consistent by default, with a ~10 ms floor set by object-storage conditional-GET metadata latency (S3 p50 10 ms). Eventual consistency removes the floor but can be stale by up to about one hour in the worst case.
- Production observations include 4T+ documents (15 PB+), 250M+ namespaces, 10M+ writes/s and 25k+ queries/s globally, 500M documents / 2 TB per namespace, and recall@10 of 90 to 100%.

**Indexing.** The vector index is based on SPFresh, chosen to minimize round trips and write amplification on object storage [28]. ANN v3 (January 2026) reports search over 100 billion vectors (about 200 TiB of dense data) in a single index at 200 ms p99, using a hierarchical clustering tree and RaBitQ-compressed vectors [32]. Treat these as vendor claims.

**Lesson: the ANN index should not be the primary key.** In "RIP, vector database" (30 September 2026), turbopuffer described v1/v2 as keying all data by ANN address (`ClusterId`, `LocalId`). It then listed the problems this caused [33]:
- *Storage amplification.* Documents with several vectors had their content copied for every vector, which is why namespaces had a vector-column limit.
- *Write amplification.* An SPFresh rebalance moves the document and every attribute and full-text posting that points at it.
- *Limited vectorization.* Query plans were tied to cluster sizes of roughly 100 to 200 documents, versus about 2,048-row (DuckDB) or 256-document (Lucene) blocks. Moving FTS v2 to fixed blocks of about 256 made the index 10x smaller and queries up to 20x faster.

v3 decouples primary document storage from ANN, which becomes one secondary index among several. The general lesson is that a vector database that grows into a document or search database tends to converge on a general storage layout.

**Capacity tiers.** Namespace pinning reserves compute and NVMe for one namespace, billed in GB-hours instead of per-TB-queried, with break-even "typically around 10 queries per second" [31].

### 4.4.3 Milvus (Zilliz)

**Architecture (Milvus 2.6 / 3.0 docs)** [35]. Milvus is a shared-storage design with four independently scalable layers:

```
 SDK --> [ Access layer: stateless proxies (MPP reduce of partial results) ]
                 |
         [ Coordinator: exactly one active; DDL/DCL, TSO/time-tick,
           WAL<->streaming-node binding, query views, compaction/index tasks ]
                 |
   +-------------+-----------------+--------------------------+
   | Streaming node                | Query node               | Data node
   | shard "mini-brain": WAL       | loads sealed segments    | compaction,
   | consumption, growing-segment  | from object storage,     | index build
   | search, growing -> sealed     | serves historical data   |
   +-------------+-----------------+--------------------------+
                 |
   [ Storage: etcd (meta) | WAL (Woodpecker/Kafka/Pulsar) | object storage (S3/MinIO) ]
```

Key techniques:
- **Woodpecker** (2.6) replaces Kafka/Pulsar with a "zero-disk" WAL written directly to object storage, with metadata in etcd. In *MemoryBuffer* mode the client batches writes in memory and flushes them. In *QuorumBuffer* mode a write is acknowledged after 2 of 3 quorum nodes have it, "typically completing within single-digit milliseconds," and is flushed to object storage asynchronously. On Milvus's single-stream benchmark, Woodpecker on S3 reached 750 MB/s at 166 ms latency, against Kafka at 129.96 MB/s and 58 ms. Local file system mode reached 450 MB/s at 1.8 ms [36].
- **Tiered storage** (2.6.4+, beta) makes query nodes load only segment metadata. Field data is fetched on demand at chunk granularity and whole per-segment indexes on first use, with configurable warm-up at cluster, collection or field/index level. This lets a query node serve datasets "that far exceed their physical memory," at the price of first-query latency [37].
- **Index menu.** HNSW, IVF_FLAT/SQ8/PQ, IVF_RABITQ, SCANN, DISKANN and GPU variants [35][38].
- **Partition key** hashes a scalar field into a fixed number of partitions, so filters on that field prune partitions and tenant counts can exceed the 1,024 manual-partition cap [39].

**Failure modes.** A single active coordinator is a control-plane bottleneck. Before tiered storage, the "full-load" mode required every query node to hold all fields and indexes of its segments [37].

### 4.4.4 Weaviate

**Storage.** Each collection is an index made of shards. Each shard contains an LSM-tree object store and inverted index (roaring bitmaps, including roaring-bitmap slices for range filters) alongside an independent vector index that is not segmented [44][41]. Placing the vector index *next to* the LSM store rather than inside it avoids segment fan-out on the vector side [44].

**Indexes.** The options are `hnsw` (a custom implementation with full CRUD), `flat` (disk-backed, for small tenants), `dynamic` (flat to HNSW at 10k objects) and `hfresh` (SPFresh-style, Section 4.1.5) [40]. Async indexing (v1.28+) uses a persistent on-disk queue for all vector index operations, which decouples ingest latency from graph insertion at the cost of a short visibility delay for vector search [40].

**Tenancy** is one shard per tenant, with tenant states and S3 offload (Section 4.3) [42][43].

### 4.4.5 Qdrant

**Design.** Collections are split into segments, each with its own HNSW and payload indexes, which is why the filterable-HNSW edge cap is computed per segment [20]. In 1.13 Qdrant replaced RocksDB with **Gridstore** for payload and sparse-vector storage. RocksDB's LSM compaction caused "random latency spikes," and Qdrant only needed sequential integer keys, so Gridstore uses a pointer-array tracker, fixed-size block data grid, used-block bitmask and gaps layer [22].

**Distinctive techniques.**
- Filterable HNSW plus ACORN plus planner (Section 4.2) [19][20].
- Binary quantization with search-time oversampling and rescoring, keeping quantized vectors and graph in RAM and originals on disk [23].
- `is_tenant` payload indexes that co-locate a tenant's vectors for sequential reads [21].
- User-defined sharding and tiered multitenancy with tenant promotion [21].

### 4.4.6 Chroma

**Deployment modes** are local (embedded), single-node (typically under 10M records) and distributed (Chroma Cloud) [46]. The distributed design has five services:

```
 client -> Gateway (auth, quota, rate limit, logical plan)
              | writes                         | reads: rendezvous hash(collection_id)
              v                                v
          Log (WAL) ------------------->  Query executor (mem + SSD-cached indexes,
              |      consistent read        consults log tail for freshness)
              v
          Compactor: log -> vector/FTS/metadata indexes -> object storage;
                     registers new index versions in System DB (SQL catalog)
```

**wal3** is Chroma's log library. It is "a linearizable log... built entirely on top of object storage," relying on conditional writes (`If-Match`) instead of an external lock service [47]. Records live in immutable *fragments*, with a mutable *manifest* at the root and immutable *snapshots* as interior nodes, like a B+ tree. A setsum (an associative, commutative hash) verifies log integrity incrementally. Leader election only needs to be best-effort, because at most one of two concurrent writers can succeed [47]. Chroma's schema exposes HNSW for single-node and SPANN for Chroma Cloud [46]. Its performance guidance is to shard by collection, one collection per customer, because collections have isolated cold starts and rate limits [46].

### 4.4.7 LanceDB and the Lance format

**Problem.** Lance targets multimodal AI data that has to support both scans (training, analytics) and random access (retrieval), on object storage, with versioning. It claims "100x faster than Parquet or Iceberg for random access" [48].

**Design.** Vector indexes are ordinary Lance files: `index.idx` holds the structure and an auxiliary file holds quantized vectors. IVF is the clustering layer, FLAT or HNSW the sub-index, and PQ/SQ/RabitQ/FLAT the quantizer, with all partitions in one file written in order [48]. Commits use MVCC with optimistic concurrency and rely only on object-store `put-if-not-exists` or `rename-if-not-exists`. A V2 manifest naming scheme (`u64::MAX - version`) lets the latest version be found by lexicographic listing [49]. An experimental **MemWAL** adds LSM-style shards (MemTable, per-shard WAL, flushed small datasets, compaction into the base table) for streaming writes while keeping vector and full-text reads indexed [50].

**Lesson.** When the vector index is just another file in a versioned table format, the same data serves training, analytics and RAG retrieval, with time travel included.

### 4.4.8 pgvector

pgvector (v0.8.7 as of this writing) adds `vector` (up to 2,000 dims indexed), `halfvec` (4,000), `bit` (64,000) and `sparsevec` (1,000 non-zeros) types, along with HNSW and IVFFlat access methods [24]. Its strengths are transactional consistency with relational data, joins, and Postgres operations tooling. HNSW builds are much faster when the graph fits in `maintenance_work_mem`, and builds parallelize (`max_parallel_maintenance_workers`). Scaling options are `halfvec`, binary quantization with re-ranking, read replicas, and sharding through Citus or PgDog [24]. The project recommends monitoring recall by comparing against exact search with `enable_indexscan=off` [24].

**Failure modes.** Post-filtering before 0.8.0 (Section 4.2), slow HNSW vacuums, and shared-index tenant interference. DiskANN-style indexes for Postgres come from extensions such as pgvectorscale [8].

### 4.4.9 Vespa

Vespa treats ANN as one operator (`nearestNeighbor`) inside a full query and ranking engine. That allows multi-vector fields (a "map of vectors" per document, for chunk-level retrieval), several ANN fields per schema, real-time CRUD on a single mutable HNSW per field per content node, multi-threaded indexing, and value types from double down to single-bit [51]. The design centers on `targetHits`/`totalTargetHits`: ANN supplies candidates to phased ranking, and the strict per-node HNSW single-threading is offset by multi-threaded pre-filter evaluation [51].

### 4.4.10 Elasticsearch and OpenSearch

**Elasticsearch.** Lucene builds a graph or IVF structure per segment, which merges consolidate. Float vectors are quantized by default. In Stack 9.0 they are always indexed as `int8_hnsw`. From 9.1 the default is `bbq_hnsw` at 384 dimensions and above and `int8_hnsw` below 384. From Stack 9.4, `bbq_disk` (DiskBBQ) is the default where licensed [54]. Raw floats stay on disk for rescoring and requantization [54]. Search can preload `vex`/`veq`/`veb`/`cenivf` files into the page cache [54]. Tuning is through `num_candidates` for HNSW and `visit_percentage` for DiskBBQ, and an auto-calibration mode picks quantization and oversampling per merged segment [52].

**OpenSearch k-NN.** It offers Faiss, Lucene and JVector engines (JVector via the `disk_ann` method plugin). `mode: on_disk` (2.17+) defaults to Faiss HNSW with 32x compression and rescoring [55]. A remote **GPU index-build service** (3.0+) builds Faiss HNSW graphs off-cluster through a vector repository [57]. Efficient filtering differs by engine (Section 4.2.4) [56].

### 4.4.11 MongoDB Atlas Vector Search and Redis

**MongoDB** runs vector search in a separate `mongot` process that holds the HNSW index. The recommended production topology puts it on dedicated **Search Nodes** (S30+) apart from `mongod`, so ANN memory and CPU scale independently of the operational database [58]. ANN queries use `numCandidates`, which the docs recommend setting to at least 20x `limit`, and `exact: true` gives ENN [60].

**Redis** offers `FLAT` (suggested under 1M vectors), `HNSW` (over 1M) and `SVS-VAMANA`, Intel's Vamana implementation with LVQ/LeanVec compression. `GRAPH_MAX_DEGREE` defaults to 32, the docs equate it to HNSW's `M*2`, and proprietary LVQ/LeanVec optimizations are not in Redis Open Source [61]. Redis is the choice when vector search has to share sub-millisecond infrastructure with caching and session state.

---

## 4.5 Object-storage-native design patterns

Across Pinecone, turbopuffer, Milvus 2.6, Chroma and Lance, the same building blocks recur.

**1. WAL on object storage with compare-and-swap.** S3/GCS conditional writes (`If-Match`, put-if-absent) are enough to build a linearizable log without ZooKeeper, Kafka or BookKeeper:
- turbopuffer: a WAL file per group commit, CAS on commit [28].
- Chroma: wal3 manifest plus fragments [47].
- Lance: manifest put-if-not-exists [49].
- Milvus: Woodpecker [36].

The cost is write latency (hundreds of ms) and per-namespace commit rate (turbopuffer's one WAL entry per second). Group commit and quorum buffers (Woodpecker QuorumBuffer) are the mitigations.

**2. Indexed base plus unindexed tail.** Readers merge immutable indexed segments with a brute-force or compact index over the log suffix. Pinecone calls this the freshness layer [25], Milvus uses growing segments on streaming nodes [35], Chroma's query executor consults the log [46], and turbopuffer scans the unindexed WAL [28]. Bounding the unindexed tail matters: turbopuffer documents a 2 GB cap [30].

**3. Stateless compute as a cache hierarchy.** RAM, then NVMe, then object storage, with routing for cache affinity: Chroma uses rendezvous hashing on collection ID [46], and turbopuffer routes namespaces to the same node [28]. Cold-start latency (Pinecone up to about 20 s at billion scale [25], turbopuffer 874 ms p50 at 1M [28], Milvus first-query fetch [37]) is the key failure mode. The mitigations are cache-warm hints (turbopuffer's `hint_cache_warm` API), configured warm-up (Milvus [37]) and pinning or reserved nodes [27][31].

**4. Cluster indexes over graphs on object storage.** Graph traversal means dependent random reads, one round trip per hop. Centroid indexes turn a query into one centroid lookup plus a single parallel batch of ranged reads [28]. This is why object-storage-first engines (turbopuffer, Chroma Cloud's SPANN, Weaviate HFresh, Elasticsearch DiskBBQ) chose SPANN/SPFresh or hierarchical k-means, while RAM-first engines kept HNSW.

**5. A reserved-capacity escape hatch.** Pure usage pricing on object storage becomes expensive at sustained high QPS. Pinecone DRN [27] and turbopuffer pinning [31] both added provisioned, always-warm capacity with per-node or GB-hour pricing, and turbopuffer cites break-even near 10 QPS.

**6. Decoupling primary storage from the ANN index.** turbopuffer's v3 rewrite [33] and Weaviate's choice to place the vector index next to the LSM store [44] reach the same conclusion: documents, attributes and inverted indexes should not be keyed by vector-cluster address once hybrid and lexical queries matter.

---

## 4.6 System comparison

| System | Primary ANN structures | Durability / storage | Filtering approach | Tenancy unit | Notable public numbers |
|---|---|---|---|---|---|
| Pinecone serverless | geometric partitions, hierarchical (proprietary) | blob storage + log; freshness layer | metadata filters within namespace | namespace (hard partition) | cold start ~2 s to ~20 s at 1B; -46.9% avg latency vs pods [25] |
| turbopuffer | SPFresh hierarchy + RaBitQ | object-storage WAL (CAS), NVMe/RAM cache | exact attribute indexes, filter-aware clusters | namespace = S3 prefix | 874 ms cold / 14 ms warm p50 at 1M; 100B @ 200 ms p99 [28][32] |
| Milvus 2.6/3.0 | HNSW, IVF*, DISKANN, SCANN, RaBitQ, GPU | Woodpecker WAL on S3; segments in S3 | scalar indexes, partition key pruning | collection / partition key | Woodpecker S3 750 MB/s [36] |
| Weaviate | HNSW, flat, dynamic, HFresh | per-shard LSM + vector index; S3 offload | roaring-bitmap allow-list + ACORN/PathSeer | shard per tenant | ~18-19k active tenants/node (test) [43] |
| Qdrant | HNSW (+payload edges), BQ/SQ/PQ | segments; Gridstore KV | filterable HNSW + ACORN-1 + planner | payload partition / shard key / tiered | 99.9-100% filtered recall with planner [20] |
| Chroma (distributed) | SPANN (cloud), HNSW (single node) | wal3 on object storage; SSD cache | metadata inverted indexes | collection | single node <10M records [46] |
| LanceDB / Lance | IVF_PQ, IVF_HNSW_SQ, IVF_RQ | Lance files + MVCC manifests on object store | scalar indexes (btree, bitmap, zonemap...) | table / dataset | "100x" random access vs Parquet [48] |
| pgvector | HNSW, IVFFlat | Postgres WAL/heap | iterative scans; partial indexes | list partition / table | dims 2,000/4,000/64,000 by type [24] |
| Vespa | single mutable HNSW per field/node | content nodes | pre/post-filter thresholds, ACORN-1 | application-level | HNSW search single-threaded per query [51] |
| Elasticsearch | HNSW, BBQ (hnsw/flat/disk), int8/int4 | Lucene segments | in-clause pre-filter; exact fallback | index / filter | BBQ 32x + 14 B/vector [53] |
| OpenSearch | Faiss HNSW/IVF, Lucene, JVector | Lucene segments; remote GPU build | engine-specific efficient filters | index | on_disk 32x default [55] |
| MongoDB Atlas | HNSW in `mongot` | separate Search Nodes | pre-filter fields in index | collection / filter | quantization cuts RAM 3.75x / 24x [59] |
| Redis | FLAT, HNSW, SVS-VAMANA | in-memory | hybrid queries over secondary index | key prefix / index | SVS max degree 32 default [61] |
| Amazon S3 Vectors | managed (undisclosed) | S3 vector buckets | metadata filters (up to 50 keys/vector) | vector index (10,000 per bucket) | 2B vectors/index; ~100 ms frequent, <1 s infrequent [67] |

---

## 4.7 Further designs and 2025-2026 developments

### 4.7.1 Amazon S3 Vectors: the object store becomes the vector store

Section 4.5 describes engines that put their source of truth on object storage. AWS took the next step and made the object store itself answer similarity queries. S3 Vectors adds *vector buckets* holding *vector indexes*, queried through an S3 API with no cluster to provision. At general availability (2 December 2025) AWS raised the limit to 2 billion vectors per index, 40x the 50M preview limit, and up to 10,000 indexes per bucket. AWS says infrequent queries return in under one second and frequent queries in about 100 ms or less, quotes 1,000 vectors/s for streaming single-vector writes, a top-k ceiling of 100, and up to 50 metadata keys per vector for filtering, and claims up to 90% lower cost for uploading, storing and querying vectors [67].

The design point is cold, large, cheap retrieval with modest top-k and QPS. It sits below the serverless engines on both price and capability. In practice it works as a cold tier beneath hotter engines that serve high-QPS or hybrid workloads. Its index structure is not documented, so its recall behavior under filters has to be measured, not assumed.

### 4.7.2 GPU graph construction and search: CAGRA and cuVS

Graph build time is often the bottleneck when a corpus is re-embedded or re-indexed. CAGRA [68] (ICDE 2024) builds a fixed-degree k-NN graph on the GPU and then prunes and reorders edges to improve reachability, using a highly parallel algorithm. The authors report graph construction 2.2x to 27x faster than HNSW, large-batch query throughput 33x to 77x higher than HNSW at 90 to 95% recall (and 3.8x to 8.8x higher than earlier GPU methods), and single-query search 3.4x to 53x faster than HNSW at 95% recall. CAGRA ships in NVIDIA's cuVS library [69]. A common production pattern is to build the graph on a GPU and serve it on CPUs: a CAGRA graph can be converted to an HNSW-compatible layout. OpenSearch's remote GPU index-build service (Section 4.4.10) [57] and Milvus's GPU index variants [35] follow this split between building and serving. The design lesson is that GPUs pay off first in index builds and batch queries. For single-query RAG serving at low QPS, CPU graphs or cluster indexes on object storage usually remain cheaper.

### 4.7.3 Data-oblivious quantizers: TurboQuant

RaBitQ (Section 4.1.3) showed that a random rotation plus a very simple per-coordinate code can carry a provable error bound. TurboQuant [70] (Google Research; ICLR 2026) extends the same family. It randomly rotates each vector so that its coordinates follow a concentrated Beta distribution, applies a per-coordinate MSE-optimal scalar quantizer, and then applies a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform to the residual, which yields an unbiased inner-product estimator. The scheme is *online* and data-oblivious: there is no codebook training, so indexing time is "virtually zero." The authors report better recall than product-quantization baselines for nearest-neighbor search. The same quantizer is also aimed at LLM KV-cache compression. For vector databases the appeal is operational. A training-free quantizer does not drift when the data distribution shifts (Section 4.1.2), and it can be applied per segment as data arrives. Qdrant has published material on TurboQuant-style quantization alongside 1-bit RaBitQ bit-plane scoring [63]. Expect PQ to give way to rotation-based codes as the default compressed representation in new engines. Vendor and paper figures for these codes are not yet independently reproduced.

### 4.7.4 Multi-vector (late-interaction) retrieval: MUVERA

ColBERT-style text models and ColPali-style page-image models emit one vector per token or patch. That multiplies index size and breaks the single-vector ANN contract. Vespa handles this natively with tensor fields and phased ranking (Section 4.4.9). MUVERA [71] (NeurIPS 2024) takes the other route. It compresses a multi-vector set into a single *fixed dimensional encoding* (FDE) whose inner product approximates Chamfer (MaxSim) similarity, so an ordinary MIPS index can retrieve candidates before exact multi-vector re-scoring. The paper reports PLAID-level recall while retrieving 2x to 5x fewer candidates, and on BEIR about 10% higher recall with 90% lower latency than PLAID. Weaviate added MUVERA encoding in v1.31, after multi-vector embeddings became generally available in v1.30. On LoTTE it reports about 70% lower memory and import times falling from over 20 minutes to 3 to 6 minutes, at some cost in recall that higher `ef` can win back [72]. For multimodal RAG over PDFs and screenshots (Chapter 10), the choice is between native multi-vector storage (exact but large) and FDE compression (cheap but approximate).

### 4.7.5 Postgres beyond pgvector: pgvectorscale and VectorChord

Two extensions bring Section 4.1's disk-oriented designs into Postgres while keeping pgvector's types:

- **pgvectorscale** (Timescale) adds a `diskann` access method called StreamingDiskANN, plus Statistical Binary Quantization and label-based filtered search derived from Filtered-DiskANN [7]. Arbitrary `WHERE` clauses still post-filter, but in a streaming way that stays correct. Its README reports 28x lower p95 latency and 16x higher throughput than Pinecone's storage-optimized s1 pods at 99% recall on 50M Cohere-768 vectors, at 75% lower self-hosted cost. This is a vendor benchmark against a pod tier that serverless has since superseded [73].
- **VectorChord** (TensorChord) adds a `vchordrq` index: IVF partitioning with RaBitQ codes and automatic reranking, plus native 4-bit and 8-bit RaBitQ types. It claims 100M vectors on a single i4i.xlarge and a path to 1B+ [74].

Both extensions keep Postgres transactions and joins while moving the index off the "graph plus floats in shared buffers" model. That model is what limits pgvector HNSW at large scale (Section 4.4.8).

---

## Key takeaways for system designers

1. **Choose by working-set economics, not benchmark QPS.** HNSW's RAM cost (`d*4 + M*2*4` bytes per float32 vector [3]) makes sense only for hot, high-QPS corpora. For long-tailed RAG tenants, object-storage-first designs (Pinecone serverless, turbopuffer, Milvus tiered storage) trade 10 to 20 ms warm and 0.5 to 20 s cold latency for order-of-magnitude cost reductions [25][28].
2. **Quantize by default, always keep a rescoring path.** 1-bit schemes (RaBitQ, BBQ, BQ) are standard practice in 2026 [17][53]. Budget oversampling (2x to 3x) and full-precision rescoring from disk, and remember the graph does not shrink when the vectors do [59].
3. **Treat filtered recall as a correctness requirement.** Measure recall per filter shape, especially broad values (around 20% of the corpus) and AND intersections. Qdrant's default configuration returned 39.7% recall on one such filter while every other filter was above 90% [20].
4. **Rely on a planner rather than a single filter algorithm.** The best results come from cardinality estimation that routes among exact scan of the allowed set, graph-with-repairs (extra edges, ACORN, PathSeer) and post-filtering, plus a fallback when fewer than k results return [20][41][52][56].
5. **Put tenancy into the physical layout.** Namespaces or prefixes as hard partitions [25][28], shard-per-tenant with temperature states [42], or tiered promotion [21]. Shared global indexes leak recall and latency across tenants [24], and shared BM25 statistics leak relevance [21].
6. **Use centroid or cluster indexes on object storage and graphs in RAM or on SSD.** Round trips dominate object-storage latency, and SPANN/SPFresh-style indexes need one centroid lookup plus one batched fetch [28].
7. **Design freshness explicitly.** A WAL plus an unindexed-tail scan gives read-your-writes consistency [25][28][46], but the tail needs a cap [30] and the consistency floor (about 10 ms per object-storage metadata check [29]) should be exposed as a per-query knob.
8. **Do not make the ANN address the primary key.** turbopuffer's 2026 v3 rewrite shows the write, storage and vectorization costs of clustering documents by vector [33]. Keep documents and secondary indexes independent of ANN partition churn.
9. **Plan a provisioned tier.** Usage-priced serverless loses to reserved capacity above roughly 10 sustained QPS per namespace in turbopuffer's accounting [31], and both major serverless vendors now sell one [27][31].
10. **Monitor recall in production.** Both pgvector and turbopuffer document comparing ANN results against exact search on live data [24][28]. Index degradation under updates, deletes and drift is the most common silent failure.

---

## References

1. Malkov, Y. A., Yashunin, D. A. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv, 2016 (IEEE TPAMI 2018). https://arxiv.org/abs/1603.09320
2. Jégou, H., Douze, M., Schmid, C. "Product Quantization for Nearest Neighbor Search." IEEE TPAMI, 2011. https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
3. Meta (Faiss team). "Guidelines to choose an index." Faiss wiki, 2026 revision. https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
4. Douze, M., et al. "The Faiss library." arXiv, 2024. https://arxiv.org/pdf/2401.08281
5. Subramanya, S. J., et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS, 2019. https://harsha-simhadri.org/pubs/DiskANN19.pdf
6. Singh, A., et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." arXiv, 2021. https://arxiv.org/abs/2105.09613
7. Gollapudi, S., et al. "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters." WWW, 2023. https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf
8. Microsoft Research. "DiskANN Project and Research Overview (2018-present)." DiskANN wiki, 2025-2026. https://github.com/microsoft/DiskANN/wiki/DiskANN-Project-and-Research-Overview-(2018%E2%80%90present)
9. Microsoft. "DiskANN3: A Composable Vector Indexing Library" (README). GitHub, 2026. https://github.com/microsoft/DiskANN
10. Microsoft. "Evaluation of Filtered Search Algorithms." DiskANN wiki, 2026. https://github.com/microsoft/DiskANN/wiki/Evaluation-of-Filtered-Search-Algorithms
11. Microsoft (Azure Cosmos DB and MSR). "Cost-Effective, Low Latency Vector Search with Azure Cosmos DB." arXiv / VLDB, 2025. https://arxiv.org/pdf/2505.05885
12. Chen, Q., et al. "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search." NeurIPS, 2021 (code and citation in SPTAG repository). https://github.com/microsoft/SPTAG
13. Xu, Y., et al. "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search." SOSP, 2023. https://dl.acm.org/doi/10.1145/3600006.3613166 (arXiv: https://arxiv.org/abs/2410.14452)
14. Guo, R., et al. "Accelerating Large-Scale Inference with Anisotropic Vector Quantization." ICML, 2020. https://arxiv.org/abs/1908.10396
15. Sun, P., et al. "SOAR: Improved Indexing for Approximate Nearest Neighbor Search." NeurIPS, 2023. https://arxiv.org/abs/2404.00774
16. Google Research. "ScaNN Algorithms and Configuration." GitHub, 2024-2026. https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
17. Gao, J., Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." SIGMOD, 2024. https://doi.org/10.1145/3654970
18. Patel, L., et al. "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data." SIGMOD, 2024. https://arxiv.org/abs/2403.04871
19. Vasnetsov, A. "Filterable HNSW Without Recall Loss." Qdrant, 2019. https://qdrant.tech/articles/filterable-hnsw/
20. Couzon, D., Ghafouri, M. "Filtered Vector Search: What ACORN Fixes, and What Fixes ACORN." Qdrant, July-August 2026. https://qdrant.tech/articles/filtered-vector-search-acorn/
21. Qdrant. "Configure Multitenancy." Qdrant documentation, 2026. https://qdrant.tech/documentation/manage-data/multitenancy/
22. Cossio, L., Gourlay, A., Myriel, D. "Introducing Gridstore: Qdrant's Custom Key-Value Store." Qdrant, 2025. https://qdrant.tech/articles/gridstore-key-value-storage/
23. Kasliwal, N. "Binary Quantization: 40x Faster Vector Search." Qdrant, 2023. https://qdrant.tech/articles/binary-quantization/
24. Kane, A., et al. "pgvector" README (v0.8.7). GitHub, 2026. https://github.com/pgvector/pgvector
25. Sriharsha, R. "Reimagining the vector database to enable knowledgeable AI" (Pinecone serverless architecture). Pinecone, January 2024. https://www.pinecone.io/blog/serverless-architecture/ (Gong figures also in Pinecone's Gong case study: https://www.pinecone.io/customers/gong/)
26. Pinecone. "Great Algorithms Are Not Enough." Pinecone, October 2023. https://www.pinecone.io/blog/hnsw-not-enough/
27. Johnson, G., Zhu, J., Umbaugh, B. "Pinecone Dedicated Read Nodes are now in Public Preview." Pinecone, December 2025. https://www.pinecone.io/blog/dedicated-read-nodes/
28. turbopuffer. "Architecture." turbopuffer documentation, 2026. https://turbopuffer.com/docs/architecture
29. turbopuffer. "Tradeoffs." turbopuffer documentation, 2026. https://turbopuffer.com/docs/tradeoffs
30. turbopuffer. "Limits." turbopuffer documentation, 2026. https://turbopuffer.com/docs/limits
31. turbopuffer. "Namespace Pinning." turbopuffer documentation, 2026. https://turbopuffer.com/docs/pinning
32. VanBenschoten, N. "ANN v3: 200ms p99 query latency over 100B vectors." turbopuffer blog, 21 January 2026. https://turbopuffer.com/blog/ann-v3
33. Harrison, D. "RIP, vector database." turbopuffer blog, September 2026. https://turbopuffer.com/blog/rip-vector-database
34. Serafimov, B. "Native filtering" (filtered vector search on a clustering index). turbopuffer blog, January 2025. https://turbopuffer.com/blog/native-filtering
35. Milvus / Zilliz. "Milvus Architecture Overview." Milvus documentation (v3.0.x), 2026. https://milvus.io/docs/architecture_overview.md
36. Milvus / Zilliz. "Woodpecker." Milvus documentation, 2025-2026. https://milvus.io/docs/woodpecker_architecture.md
37. Milvus / Zilliz. "Tiered Storage Overview." Milvus documentation, 2025-2026. https://milvus.io/docs/tiered-storage-overview.md
38. Milvus / Zilliz. "DISKANN." Milvus documentation, 2026. https://milvus.io/docs/diskann.md
39. Milvus / Zilliz. "Use Partition Key." Milvus documentation, 2026. https://milvus.io/docs/use-partition-key.md
40. Weaviate. "Vector Indexing" (HNSW, flat, dynamic, HFresh). Weaviate documentation, 2026. https://docs.weaviate.io/weaviate/concepts/vector-index
41. Weaviate. "Filtering" (ACORN, sweeping, PathSeer, flat-search cutoff). Weaviate documentation, 2026. https://docs.weaviate.io/weaviate/concepts/filtering
42. Weaviate. "Manage tenant states & temperature." Weaviate documentation, 2026. https://docs.weaviate.io/weaviate/manage-collections/tenant-states
43. Weaviate. "Data structure" (multi-tenancy limits). Weaviate documentation, 2026. https://docs.weaviate.io/weaviate/concepts/data
44. Weaviate. "Storage." Weaviate documentation, 2026. https://docs.weaviate.io/weaviate/concepts/storage
45. Li, Z., Zhang, G., et al. "PathSeer: Adaptive Neighbor Handling for Efficient Filtered ANN Search." SIGMOD, 2026. https://doi.org/10.1145/3802098
46. Chroma. "Architecture Overview" and "Distributed Architecture." Chroma documentation source, 2026. https://github.com/chroma-core/chroma/blob/main/docs/mintlify/reference/architecture/distributed.mdx
47. Chroma. "wal3" README (write-ahead log on object storage). GitHub, 2025-2026. https://github.com/chroma-core/chroma/blob/main/rust/wal3/README.md
48. Lance / LanceDB. "Vector Indices" format specification and project README. GitHub, 2026. https://github.com/lancedb/lance/blob/main/docs/src/format/index/vector/index.md
49. Lance / LanceDB. "Transaction Specification." GitHub, 2026. https://github.com/lancedb/lance/blob/main/docs/src/format/table/transaction.md
50. Lance / LanceDB. "MemTable & WAL Specification (Experimental)." GitHub, 2026. https://github.com/lancedb/lance/blob/main/docs/src/format/table/mem_wal.md
51. Vespa.ai. "Approximate nearest neighbor search using HNSW index." Vespa documentation source, 2026. https://github.com/vespa-engine/documentation/blob/master/en/querying/approximate-nn-hnsw.md
52. Elastic. "Approximate kNN search" and "Filtered kNN search." Elastic docs source, 2026. https://github.com/elastic/docs-content/blob/main/solutions/search/vector/knn/approximate-knn.md
53. Elastic. "Better Binary Quantization (BBQ)" reference. Elasticsearch docs source, 2026. https://github.com/elastic/elasticsearch/blob/main/docs/reference/elasticsearch/mapping-reference/bbq.md
54. Elastic. "Dense vector field type" reference. Elasticsearch docs source, 2026. https://github.com/elastic/elasticsearch/blob/main/docs/reference/elasticsearch/mapping-reference/dense-vector.md
55. OpenSearch Project. "Disk-based vector search." OpenSearch documentation source, 2024-2026. https://github.com/opensearch-project/documentation-website/blob/main/_vector-search/optimizing-storage/disk-based-vector-search.md
56. OpenSearch Project. "Efficient k-NN filtering." OpenSearch documentation source, 2026. https://github.com/opensearch-project/documentation-website/blob/main/_vector-search/filter-search-knn/efficient-knn-filtering.md
57. OpenSearch Project. "Building vector indexes remotely using GPUs." OpenSearch documentation source, 2025-2026. https://github.com/opensearch-project/documentation-website/blob/main/_vector-search/remote-index-build.md
58. MongoDB. "Review Deployment Options" (Vector Search). MongoDB docs source, 2026. https://github.com/mongodb/docs/blob/main/content/vector-search/source/deployment/deployment-options.txt
59. MongoDB. "Vector Quantization." MongoDB docs source, 2026. https://github.com/mongodb/docs/blob/main/content/vector-search/source/about/vector-quantization.txt
60. MongoDB. "Run Vector Search ANN and ENN Queries" ($vectorSearch stage). MongoDB docs source, 2026. https://github.com/mongodb/docs/blob/main/content/vector-search/source/query/aggregation-stages/vector-search-stage.txt
61. Redis. "Vectors" (FLAT, HNSW, SVS-VAMANA). Redis docs source, 2026. https://github.com/redis/docs/blob/main/content/develop/ai/search-and-query/vectors/_index.md
62. Simhadri, H. V., et al. "Results of the Big ANN: NeurIPS'23 competition." arXiv, 2024. https://arxiv.org/abs/2409.17424
63. Gao, J., et al. "RaBitQ-Library" (adoption list and implementations). VectorDB-NTU, GitHub, 2025-2026. https://github.com/VectorDB-NTU/RaBitQ-Library
64. Microsoft Research and collaborators. "In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search" (IP-DiskANN). arXiv, 2025. https://arxiv.org/abs/2502.13826
65. Padaki, A., Ingber, A., Liberty, E. "VQ-bench: A Composable Vector Quantization Framework." Pinecone blog, 17 September 2026, and VecDB@VLDB 2026 paper. https://arxiv.org/abs/2608.11240
66. Microsoft. "Perf: Garnet Providers vs other Vector DBs (Zilliz, Pinecone, etc.)." DiskANN wiki, 2026. https://github.com/microsoft/DiskANN/wiki/Perf:-Garnet-Providers-vs-other-Vector-DBs-(Zilliz,-Pinecone,-etc.)
67. Amazon Web Services. "Amazon S3 Vectors is now generally available with 40 times the scale of preview." AWS What's New, 2 December 2025. https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/
68. Ootomo, H., Naruse, A., Nolet, C., Wang, R., Feher, T., Wang, Y. "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs." ICDE, 2024. https://arxiv.org/abs/2308.15136
69. NVIDIA RAPIDS. "cuVS: Vector Search and Clustering on the GPU." GitHub, 2024-2026. https://github.com/rapidsai/cuvs
70. Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR, 2026; see also Google Research blog, "TurboQuant: Redefining AI efficiency with extreme compression," March 2026. https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
71. Dhulipala, L., Hadian, M., Jayaram, R., Lee, J., Mirrokni, V. "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings." NeurIPS, 2024. https://arxiv.org/abs/2405.19504
72. Weaviate. "MUVERA" (multi-vector encoding in Weaviate v1.31). Weaviate blog, 2025. https://weaviate.io/blog/muvera
73. Timescale. "pgvectorscale" README. GitHub, 2024-2026. https://github.com/timescale/pgvectorscale
74. TensorChord. "VectorChord" README. GitHub, 2025-2026. https://github.com/tensorchord/VectorChord

Source note: vendor documentation was read from each project's public documentation source repository. The turbopuffer documentation pages [28]-[31] were read from a verbatim mirror of turbopuffer.com/docs. The turbopuffer blog posts [32]-[34] were read through third-party excerpts and summaries, so their figures should be checked against the originals before reuse. In the 2026-10 fact-check, documentation sources [3], [9], [24], [47], [51], [53]-[55], [59], [61], [63], [73] and [74] were re-read from their GitHub sources and the cited figures matched. The Weaviate filtering page [41] was re-read from the weaviate/docs repository, and it confirms ACORN as the default from v1.34 and the PathSeer strategy. Claims from [20], [27], [32], [33], [36], [65] and [67] were cross-checked through search results and secondary coverage only. Every performance number in this chapter is as reported by the vendor or authors and was not independently reproduced.
