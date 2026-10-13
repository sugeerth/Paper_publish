# 07. GraphRAG and Knowledge Graphs: Community Summaries, Graph Memory, and When Graphs Beat Vectors

## Overview

Vector RAG treats a corpus as a bag of independent chunks. Each chunk is embedded, and at
query time the top-k nearest neighbours go into the prompt. That model fails in two ways,
and the GraphRAG family of designs was built to fix them. Microsoft Research named both
failures in its first public GraphRAG post [3]:

1. **Connecting the dots.** An answer may need facts spread across documents that share
   an entity, but no single chunk is close to the query in embedding space. In Microsoft's
   example, asked "What has Novorossiya done?", a LangChain vector-RAG baseline said the
   text did not say, because none of the retrieved chunks mentioned the term. GraphRAG found
   the entity in its graph and answered in detail [3].
2. **Holistic understanding.** Some questions are about the whole corpus, such as "What are
   the top 5 themes in the data?". No top-k set of chunks can answer them. On this query
   the baseline matched chunks that contained the word "theme". GraphRAG returned themes
   that described the dataset [3].

The GraphRAG paper reframes the second failure. A global question is a *query-focused
summarization* (QFS) task, not a retrieval task, and earlier QFS methods do not scale to
RAG-sized corpora [1][2]. Everything in this chapter is some answer to the same question:
**which structure should be built over the corpus at index time, so that multi-hop and
corpus-level questions become cheap at query time?**

The designs differ mainly in where they spend LLM tokens:

```
  index-time LLM cost  ------------------------------------------------>  high
  |                                                                         |
  vector RAG     LazyGraphRAG   FastGraphRAG    HippoRAG/LightRAG    GraphRAG (standard)
  (embed only)   (NLP concept   (NLP graph +    (LLM triples or      (LLM entities +
                  graph, no     LLM community    entity/relation      relationships +
                  summaries)    reports)         profiles)            summaries + claims
                                                                      + community reports)
  query-time cost, global questions: high <-------------------------> low (pre-summarized)
```

The chapter covers, in order:

- Microsoft GraphRAG: indexing, global/local/DRIFT search, dynamic community selection,
  auto-tuning, FastGraphRAG, and LazyGraphRAG.
- LightRAG: dual-level keyword retrieval over an entity/relation graph.
- HippoRAG and HippoRAG 2: OpenIE knowledge graphs with Personalized PageRank (PPR) as
  associative memory. Also fast-graphrag and GFM-RAG, which build on the same idea.
- The Neo4j GraphRAG stack: property-graph retrievers, Text2Cypher, the KG builder, and
  LLM Graph Builder.
- Other industrial designs: AWS GraphRAG Toolkit, Zep/Graphiti temporal knowledge graphs,
  Ant Group's KAG, Think-on-Graph, and LinkedIn's customer-service KG.
- KG construction with LLMs: schemas, extraction, entity resolution, and pruning.
- The evidence on when graphs beat vectors, and when they do not.

---

## 1. Microsoft GraphRAG

### 1.1 Problem and origins

GraphRAG came out of Microsoft Research. The first blog post (February 2024) used the
VIINA dataset: thousands of Russian- and Ukrainian-language news articles from June 2023,
translated into English. The data was too large for a context window and recent enough to
be absent from the base model's training data [3]. The paper "From Local to Global: A
Graph RAG Approach to Query-Focused Summarization" followed in April 2024 [1][2]. Code
was released on GitHub in July 2024, with an Azure "GraphRAG Accelerator" for code-free
deployment [4][48]. The paper evaluates on "global sensemaking questions over datasets in
the 1 million token range" [2].

### 1.2 Indexing pipeline

The default dataflow in the documentation has six phases [13]:

```
 documents
    |
    v
 [1] Compose TextUnits ---- chunk (default 1,200 tokens); unit of extraction + provenance
    |
    v
 [2] Document processing -- link documents <-> TextUnits; export Documents table
    |
    v
 [3] Graph extraction ----- LLM per TextUnit -> entities (title,type,description)
    |                       + relationships (source,target,description)
    |                       merge by (title,type) / (source,target)
    |                       LLM summarizes merged description lists
    |                       optional: claim extraction -> Covariates (off by default)
    v
 [4] Graph augmentation --- hierarchical Leiden community detection, recursive until
    |                       communities reach a size threshold -> multi-level hierarchy
    v
 [5] Community summarization -- LLM report per community (overview, key entities,
    |                            relationships, claims), then a shorter summary
    v
 [6] Text embeddings ------ TextUnit text, entity descriptions, report content
                            -> vector store (LanceDB default; Azure AI Search)
 outputs: Documents, TextUnits, Entities, Relationships, Communities,
          Community Reports, (Covariates)
```

Key techniques:

- **Chunk size is a fidelity knob.** The docs warn that "larger chunks result in
  lower-fidelity output and less meaningful reference texts" [13].
- **Merge, then summarize.** Each chunk yields a subgraph. Entities with the same title and
  type are merged across chunks, and their descriptions are collected into a list. A second
  LLM pass reduces each list to one description [13]. This is a simple form of entity
  resolution. Because it matches on exact title and type, spelling variants stay separate
  unless the prompt normalizes them.
- **Claims as covariates.** Optional claim extraction produces time-bounded factual
  statements with a status. It is off by default because it "usually requires prompt
  tuning" [13].
- **Hierarchical Leiden.** Clustering produces levels (C0 root, C1, C2, ...). Each level
  partitions the graph at a different granularity. A report is written for every community
  at every level [13].

**Cost profile.** The repository's main warning is that "GraphRAG indexing can be an
expensive operation": read the cost docs and start small [12]. The indexing-methods page
estimates that graph extraction is about 75% of indexing cost [14]. The project is now in
**maintenance mode**: bug fixes and CVE-driven dependency updates continue, but no new
features are accepted [12].

### 1.3 Query modes

GraphRAG ships several query engines over the same index.

**Global search (map-reduce over community reports).** This engine is for whole-corpus
questions [15]. Reports from one chosen hierarchy level are split into chunks of fixed
size. *Map*: each chunk produces an intermediate answer made of points, and each point
carries a numeric importance rating. *Reduce*: the highest-rated points are filtered,
aggregated, and used to write the final answer. Lower levels give more detailed reports
and usually more thorough answers, but they cost more time and LLM calls because there are
more reports [15]. Configuration includes `max_data_tokens`, separate map/reduce LLM
parameters, `concurrent_coroutines` for map-stage parallelism, and
`allow_general_knowledge`. That last flag lets the reduce step use outside knowledge, and
the docs warn it may increase hallucination [15].

```
          query
            |
   +--------+---------+---------+          level-k community reports
   v        v         v         v          (batched into context-size chunks)
 [map]    [map]     [map]     [map]   --> rated points {text, score}
   \        |         |        /
    +-------+----+----+-------+
                 v
        filter/sort by score, pack to token budget
                 v
             [reduce] --> final answer (+ report/source references)
```

**Local search (entity-anchored).** This engine answers questions about specific entities
[16]:

1. Embed the query, plus optional conversation history, and match it against entity
   description embeddings to find entry points.
2. Fan out from those entities to linked TextUnits, community reports, neighbouring
   entities, relationships, and covariates.
3. Rank and filter each candidate set.
4. Pack everything into one context window of fixed size and generate the answer [16].

**DRIFT search (Dynamic Reasoning and Inference with Flexible Traversal).** DRIFT mixes
global and local search [6][17]:

- **Primer.** The query is expanded with HyDE and compared with the top-K most relevant
  community reports. This yields an initial broad answer and follow-up questions.
- **Follow-up.** Each follow-up runs as a local-search variant and produces more
  intermediate answers and follow-ups. The loop stops at a termination criterion, set to
  two iterations at the time of the post.
- **Output.** A hierarchy of question/answer pairs ranked by relevance. Unlike naive
  map-reduce, intermediate answers are not weighted equally.

On 50 "local" questions over more than 5,000 AP News articles, an LLM judge preferred
DRIFT to local search 78% of the time on comprehensiveness and 81% on diversity [6]. The
post gives no cost numbers. It lists a "lite-global" variant and a query router as future
work [6].

There is also a **basic search** mode, which is plain vector RAG over the same TextUnits.
It is useful as a baseline.

### 1.4 Published results

From the July 2024 release post, which summarizes the paper [4]:

- Datasets: podcast transcripts and news articles (linked to the MultiHop-RAG dataset).
  GPT-4 generated "activity-centered sense-making questions" from short dataset
  descriptions. An LLM judge compared answers on comprehensiveness, diversity, and
  empowerment.
- GraphRAG at **any** community level beat naive RAG on comprehensiveness and diversity,
  with roughly 70-80% win rates.
- Intermediate- and low-level community summaries beat hierarchical source-text
  summarization while using about **20-70%** of its tokens per query.
- Root-level summaries were competitive with source-text summarization at about **2-3%**
  of its tokens per query. That is roughly 97-98% fewer tokens.

The February 2024 post also reported that SelfCheckGPT faithfulness was similar to
baseline RAG [3]. That result matters: the gains come from coverage, not from fewer
hallucinations.

### 1.5 Prompt auto-tuning

Default prompts are tuned for news: organization, geography, and person entities, with 15
hand-written entity examples and 12 relationship examples [5]. Auto-tuning works in four
steps:

1. Sample the corpus. The example given is 1% of 10,000 chemistry papers.
2. Have the LLM identify the domain.
3. Generate a persona.
4. Generate the extraction, summarization, and community-report prompts in parallel, with
   few-shot examples taken from the corpus itself [5].

On the *Behind the Tech* podcast with GPT-4 Turbo, auto-tuned prompts increased extracted
entities from 1,796 to 4,896, relationships from 2,851 to 8,210, and communities from 352
to 1,027 [5]. The CLI can select sample chunks randomly (the default), take the first *n*,
use all of them, or use `auto`. The `auto` method embeds up to `n_subset_max` = 300 chunks
and picks the k = 15 nearest to the centroid. The tuning chunk size defaults to 200 tokens
[18]. The README "strongly recommends" prompt tuning, because out-of-the-box results may
be poor [12].

### 1.6 Making global search cheaper: dynamic community selection

Static global search runs map-reduce over every report at the chosen level. Dynamic
community selection (November 2024) instead walks the hierarchy from the root [7]:

1. An LLM rates each community report for relevance to the query.
2. Irrelevant reports are pruned, along with their whole subtree.
3. The children of relevant reports are rated in turn.
4. Only the surviving reports go into map-reduce.

A GPT-4o-mini rater was used for pruning and GPT-4o for map-reduce. The authors say the
small rater reached "a very similar retrieval rate" to a GPT-4o rater [7]. Results on AP
News with 50 global questions [7]:

| Configuration                    | Reports into map-reduce | Token cost vs static L1 | Quality vs static L1                  |
|----------------------------------|-------------------------|-------------------------|---------------------------------------|
| Static global search, level 1    | ~1,500                  | baseline                | baseline                              |
| Dynamic selection, up to level 1 | ~470 (average)          | -77% (average)          | no significant difference             |
| Dynamic selection, up to level 3 | more than L1 dynamic    | about +34%              | comprehensiveness 58.8%, empowerment 60.0% win rate |

The lesson is that a cheap relevance classifier, placed in front of an expensive
summarizer, recovers most of the efficiency of retrieval without giving up corpus-level
coverage.

### 1.7 FastGraphRAG: an NLP-built graph

The `--method fast` indexer replaces LLM extraction with NLP [14]:

- **Entities** are noun phrases, found with NLTK plus regexes by default, or with spaCy.
  They have no descriptions; the source TextUnit serves as context.
- **Relationships** are co-occurrences within a TextUnit, again without descriptions.
- Entity/relationship summarization and claim extraction are skipped.
- The LLM still writes community reports.

The docs recommend much smaller chunks (50-100 tokens) so the co-occurrence graph is less
noisy. They describe the result as much cheaper but noisier and "less useful outside
GraphRAG". Use Standard when you need high-fidelity entities and graph exploration. Use
Fast when the main need is global summarization at low cost [14].

### 1.8 GraphRAG 1.0 engineering changes

The 1.0 release (December 2024) changed the engineering more than the algorithm [9]:

- Embeddings moved out of parquet into a default vector store (LanceDB, or Azure AI
  Search). Parquet disk usage fell by 80%, and total disk usage by 43%.
- 88 DataShaper workflows were collapsed into 11 functional workflows.
- CLI start-up time fell from 148 s to 2 s on average.
- A new `update` command computes deltas against an existing index and merges them.
  Reruns use LLM caching, so incremental updates are often cheaper than the first run.

Because 1.0 broke backward compatibility, it needs a migration [9]. Incremental update
matters because the Leiden hierarchy and its reports depend on the whole graph. Appending
documents can change community membership, and changed communities need new reports.

### 1.9 LazyGraphRAG: deferring all LLM work to query time

LazyGraphRAG (November 2024) removes the main barrier to adoption, which is indexing cost
[8]:

```
INDEX (no LLM)                         QUERY (budgeted LLM work)
--------------                         --------------------------------------------
docs -> chunks                         query -> LLM: 3-5 subqueries, expanded query,
   -> noun-phrase concepts                      refine with matching graph concepts
   -> co-occurrence concept graph      per subquery:
   -> graph-statistics optimization       best-first: rank chunks by embedding sim,
   -> hierarchical communities                       rank communities by top-k chunks
   -> chunk embeddings                    breadth-first: LLM sentence-level relevance
                                                  test on top-k untested chunks of
                                                  communities in rank order
                                          iterative deepening: recurse into
                                                  sub-communities after z empty
                                                  communities in a row
                                          stop: no relevant communities left or
                                                relevance-test budget / q reached
                                       map: concept subgraph -> group chunks by
                                            community -> LLM extracts claims
                                            -> rank, filter to context window
                                       reduce: LLM answers expanded query from claims
```

The cost-quality trade-off is controlled by one parameter, the **relevance-test budget**
[8]. Reported numbers, from 5,590 AP news articles and 100 synthetic queries (50 local, 50
global) judged on comprehensiveness, diversity, and empowerment [8]:

- Indexing cost is the same as vector RAG and **0.1% of full GraphRAG**.
- At query cost comparable to vector RAG, it outperformed all competing methods on local
  queries.
- On global queries it matched GraphRAG global search quality at **more than 700x lower**
  query cost.
- At 4% of GraphRAG global search (C2) query cost, it significantly beat all competitors on
  both local and global queries.

Three configurations were reported [8]:

- **Z100_Lite**: budget 100, low-cost LLM at every step. Its cost matched an 8K-context
  vector baseline.
- **Z500**: budget 500. Low-cost LLM for relevance tests, a stronger LLM for refinement and
  map/reduce.
- **Z1500**: budget 1,500, with the same split as Z500.

At budget 100, LazyGraphRAG beat everything except GraphRAG global search on global
queries. At 500 it beat everything, and win rates kept rising at 1,500. The competitors
included GraphRAG C1/C2/C3-dynamic, local search, DRIFT, vector RAG with 8K and 64K
contexts, and RAPTOR [8].

BenchmarkQED (June 2025) added a stronger baseline [10]. It compared LazyGraphRAG with
vector RAG using a **1M-token** context window, on 200 queries: 50 in each of the four
AutoQ classes (data-local, activity-local, data-global, activity-global). Each query had
six trials, and GPT-4.1 was the judge. LazyGraphRAG won every comparison. The one
non-significant result was relevance on data-local queries, which the authors attribute
to vector RAG's strength at ranking directly relevant chunks [10]. A June 2025 editor's
note says LazyGraphRAG was integrated into Microsoft Discovery and Azure Local (public
preview) [8][11].

**Lesson.** LazyGraphRAG shows that the costly part of GraphRAG (LLM summaries
precomputed for every community) is not needed for the gains. The structural part, a
community hierarchy over a concept graph that guides breadth-first exploration, can be
built with cheap NLP. The LLM is then spent only on the parts of the corpus a query
touches.

### 1.10 Failure modes observed in GraphRAG-style systems

- **Index cost and staleness.** Full LLM indexing is expensive [12]. Updates can change
  community structure, so reports must be regenerated [9].
- **Domain mismatch.** News-tuned prompts under-extract in other domains: about 2.7x fewer
  entities on the podcast corpus [5].
- **Entity fragmentation.** Exact-match merging on (title, type) [13] leaves aliases as
  separate nodes, which splits communities.
- **Detail loss in global search.** Community summaries are lossy. Questions that need
  exact figures fall back to local or basic search. The BenchmarkQED data-local relevance
  result is consistent with this [10].
- **Judge-based evaluation.** Most published GraphRAG results are pairwise LLM-judge win
  rates on comprehensiveness, diversity, and empowerment [4][7][8][10]. These metrics
  reward long, multi-perspective answers. They do not measure exact-answer accuracy.

---

## 2. LightRAG

### 2.1 Problem

LightRAG (HKU Data Intelligence Lab, October 2024; listed as EMNLP 2025) presents itself
as a lighter alternative to GraphRAG [20][21]. It keeps an entity/relation graph but drops
the community hierarchy and its reports, and it supports incremental insertion.

### 2.2 Architecture

```
INDEX                                          QUERY
-----                                          -----
chunks --LLM--> entities + relations           query --LLM (keyword role)-->
          (name, type, description;               low-level keywords  (entities)
           relation keywords/description)         high-level keywords (themes)
   |                                                 |                 |
   v                                                 v                 v
dedupe/merge into graph  <------------------  match entity       match relation
   |                                          vectors (local)    vectors (global)
   +--> KV store: entity/relation profiles           \               /
   +--> vector store: entity, relation, chunk         +--> 1-hop neighbourhood +
   +--> graph store: nodes/edges                      |    source chunks
   +--> doc-status store                              v
                                              (mix mode: + naive chunk vectors)
                                                      v
                                              [optional reranker] -> LLM answer
```

Key techniques:

- **Dual-level retrieval.** An LLM pulls *low-level* keywords (specific entities) and
  *high-level* keywords (broad themes) from the query. Low-level keywords are matched
  against entity vectors and high-level keywords against relation vectors. The results are
  expanded to their graph neighbourhood and source text [20][21].
- **Query modes** [21]: `local` (entity-focused), `global` (relationship-focused),
  `hybrid`, `naive` (plain chunk vectors), and `mix` (all of these combined; now the
  default). A reranker is the default in mixed queries since August 2025. The README says
  it adds about 1-2 s of latency and recommends `BAAI/bge-reranker-v2-m3` locally [21].
- **Incremental update.** New documents are extracted and merged into the existing graph
  without rebuilding any community structure. Deleting a document rebuilds the affected
  entities and relations from the LLM cache [21].
- **Model roles.** The README separates four model roles [21]. Extraction should use a
  fast non-thinking model; Qwen3-30B-A3B-Instruct is called "a reasonable minimum"
  locally. The query model should be stronger and may be a thinking model. The keyword
  model should be fast. A VLM handles multimodal input. The embedding model must not
  change after indexing starts; bge-m3 is suggested locally.

**Production storage.** The defaults (JSON KV, NanoVectorDB, NetworkX) are for testing
only. The README recommends PostgreSQL for all four storage roles, and lists MongoDB and
OpenSearch as single-backend options. Milvus or Qdrant can serve vectors, and Neo4j or
Memgraph the graph [21]. 2025-2026 changes include citations, a WebUI, RAGAS evaluation,
Langfuse tracing, selectable chunking strategies, and in May 2026 the RAG-Anything
multimodal pipeline (MinerU/Docling parsing) merged into the core [21].

### 2.3 Results and cost claims

The evaluation uses the UltraDomain datasets (Agriculture, CS, Legal, Mix). Questions are
generated GraphRAG-style: an LLM imagines 5 users with 5 tasks each and writes 5
whole-dataset questions per user-task pair. Answers are compared pairwise by an LLM judge
[22]. (Microsoft's BenchmarkQED toolkit [10][19] packages this kind of synthetic query
generation and pairwise judging for reuse.) Overall win rates for LightRAG [21]:

| Baseline  | Agriculture | CS    | Legal | Mix   |
|-----------|-------------|-------|-------|-------|
| NaiveRAG  | 67.6%       | 61.2% | 84.8% | 60.0% |
| RQ-RAG    | 67.6%       | 62.0% | 85.6% | 60.0% |
| HyDE      | 75.2%       | 58.4% | 73.6% | 57.6% |
| GraphRAG  | 54.8%       | 52.0% | 52.8% | 49.6% |

Against GraphRAG the margins are small. Most of the gap is on *diversity* (for example
77.2% in Agriculture), while comprehensiveness is close to parity (51.6-54.4%). On Mix,
GraphRAG is slightly ahead overall [21][22].

On cost, the paper argues that GraphRAG global retrieval sends hundreds of community
reports through the LLM per query, while LightRAG needs a single short keyword-extraction
call plus retrieval [20]. The paper's worked example puts GraphRAG at roughly 610,000
tokens for its level-2 communities, against under 100 tokens for LightRAG keyword
generation. These figures are approximate, taken from the paper's cost-analysis section.

### 2.4 Failure modes and lessons

- **Extraction quality depends on the model.** The README says LightRAG needs stronger
  LLMs than plain RAG because of entity-relation extraction, and a query LLM that can
  handle long, noisy context [21].
- **Fixed embedding model.** Changing the embedding model forces a full re-index [21].
- **No corpus-level summaries.** "Global" in LightRAG means relation-centric retrieval, not
  map-reduce over the whole corpus. Truly corpus-wide questions are answered from
  whatever relations the high-level keywords match.
- **Lesson.** Much of what GraphRAG gains on sensemaking benchmarks can be had without
  community reports, if retrieval units are graph elements (entities and relations with
  LLM-written descriptions) and not raw chunks.

---

## 3. HippoRAG, HippoRAG 2, and PPR-based graph memory

### 3.1 HippoRAG (NeurIPS 2024)

HippoRAG (Ohio State NLP group) takes its design from the hippocampal indexing theory of
human long-term memory [23][25]. The LLM plays the neocortex, which processes text. A
schemaless knowledge graph plays the hippocampal index, which stores associations.
Retrieval is pattern completion by **Personalized PageRank** [23][26].

```
OFFLINE INDEXING                              ONLINE RETRIEVAL
passages --LLM OpenIE--> (s, p, o) triples    query --LLM NER--> query entities
                     |                                         |
                     v                          retrieval encoder: nearest KG nodes
          schemaless KG (noun-phrase nodes)                    |
                     +                                         v
   retrieval encoder (ColBERTv2/Contriever):    seed PPR at those nodes (node
   synonymy edges between similar nodes         specificity weighting)
   (example scripts: cosine >= 0.8)                            |
                     |                                         v
          node -> passage incidence matrix      PPR mass over nodes -> aggregate to
                                                passages -> rank -> LLM reader
```

Key techniques [23][26]:

- **OpenIE with an LLM.** The original release used GPT-3.5 Turbo for NER and triple
  extraction, and also supported Llama-3 and REBEL.
- **Synonymy edges.** An embedding encoder links near-duplicate nodes, using a 0.8
  similarity threshold in the example scripts. This makes up for not having strict entity
  resolution.
- **PPR from query entities.** One graph walk spreads relevance across multi-hop paths, so
  multi-hop retrieval happens in a single step. The example commands use a damping value
  of 0.5.
- **Node specificity.** Seed nodes are weighted by how rare they are across passages, an
  IDF-like signal.

Results (from the paper abstract [23]): on multi-hop QA (MuSiQue, 2WikiMultiHopQA,
HotpotQA), HippoRAG beat state-of-the-art retrievers by up to about 20%. Single-step
HippoRAG matched or beat iterative retrieval with IRCoT while being about 10-30x cheaper
and 6-13x faster. It can also be combined with IRCoT for further gains [23][26].

PathRAG (BUPT) is a related idea. It retrieves relational *paths* rather than
neighbourhoods, and prunes them with a propagation decay rate (`alpha`) and a pruning
threshold before prompting [42].

### 3.2 HippoRAG 2 (ICML 2025)

"From RAG to Memory: Non-Parametric Continual Learning for Large Language Models" recasts
the system as long-term memory for LLMs [24][25]. It is evaluated on three axes:

- **Factual memory**: NaturalQuestions, PopQA.
- **Sense-making**: NarrativeQA.
- **Associativity**: MuSiQue, 2Wiki, HotpotQA, LV-Eval.

The motivation is that structure-augmented RAG systems, GraphRAG and LightRAG among them,
often fall *below* a strong dense retriever on simple factual QA even when they help on
multi-hop questions. HippoRAG 2 aims to improve associativity and sense-making without that
regression [25]. As described in the paper [24]:

- Passages are added as nodes next to phrase nodes, which ties dense and sparse
  (entity-level) signals together in one graph.
- Query-to-triple matching is followed by an LLM "recognition memory" filter that drops
  irrelevant triples before seeding PPR.

The repository claims HippoRAG 2 beats other methods in all three categories. It describes
online retrieval as cost- and latency-efficient and offline indexing as using
"significantly fewer resources" than GraphRAG, RAPTOR, or LightRAG [25]. The paper reports
roughly a 7-point average gain on associative tasks over a strong embedding baseline (an
approximate figure). Supported back ends include OpenAI/Azure, Bedrock, vLLM (for example
Llama-3.3-70B-Instruct), NV-Embed-v2, and GritLM embeddings, with Parquet, Qdrant, Chroma,
or Milvus as stores [25].

### 3.3 Descendants: fast-graphrag and GFM-RAG

- **fast-graphrag** (Circlemind) uses Personalized PageRank to explore the graph and
  credits HippoRAG for the method. It reports $0.08 against $0.48 for GraphRAG on *The
  Wizard of Oz*, about 6x cheaper, and says the savings grow with data size and the number
  of insertions. No methodology is published for this comparison [27].
- **GFM-RAG** (NeurIPS 2025) replaces hand-tuned PPR with a pre-trained **graph
  foundation model**, a query-conditioned GNN based on ULTRA. It runs over a "universal
  graph index" with entity, document, and summary nodes [40]. The 8M-parameter checkpoint
  was pre-trained on 286 KGs (June 2025 release). It claims single-step multi-hop
  retrieval and transfer to unseen datasets without fine-tuning. The follow-up G-reasoner
  (34M parameters) was accepted at ICLR 2026 [40].

**Lesson.** PPR, or a GNN that learns to imitate it, is a cheap and deterministic way to
do multi-hop retrieval. It moves the "hop" from repeated LLM calls (IRCoT) into a graph
walk that costs milliseconds.

---

## 4. The Neo4j GraphRAG stack

Neo4j approaches GraphRAG from the database side. The graph is a typed property graph
that the application owns, and retrieval is a **Cypher query composed with vector or
full-text search**.

### 4.1 Retrievers (neo4j-graphrag-python)

The official package [28] provides a `GraphRAG` orchestrator: retriever, then prompt
template, then LLM. It has a `return_context` option and a `response_fallback` that skips
the LLM call when nothing is retrieved [29]. The retrievers are [29]:

| Retriever                          | First stage                         | Graph use                              | Notes |
|------------------------------------|-------------------------------------|----------------------------------------|-------|
| `VectorRetriever`                  | Neo4j vector index (ANN)            | none                                   | `return_properties` |
| `VectorCypherRetriever`            | vector index                        | `retrieval_query` Cypher from `node`, `score` | the core GraphRAG pattern |
| `HybridRetriever`                  | vector + full-text index            | none                                   | no filters |
| `HybridCypherRetriever`            | vector + full-text                  | `retrieval_query` traversal            | |
| `Text2CypherRetriever`             | LLM writes Cypher from question (+ schema, examples) | the whole query is graph | no embedder; queries may be invalid (`Text2CypherRetrievalError`) |
| `ToolsRetriever`                   | LLM selects 0..n retriever tools    | any                                    | attributes results to tools |
| `Weaviate/Pinecone/QdrantNeo4jRetriever` | external vector DB            | map ids to Neo4j nodes                 | provider filter syntax |

The central pattern is **vector entry plus Cypher expansion**:

```
question --embed--> vector index on :Chunk(embedding) --top-k (node, score)-->
   retrieval_query:
     MATCH (node)<-[:PART_OF_CHUNK]-(e:__Entity__)-[r]-(nbr:__Entity__)
     RETURN node.text, collect(e.name + ' ' + type(r) + ' ' + nbr.name)
   --> context strings --> RagTemplate(context, question, examples) --> LLM
```

This is close to GraphRAG local search (section 1.3), but the developer writes the
expansion as an explicit, testable Cypher query instead of a fixed library heuristic.
(The Cypher above is an illustration, not code from the docs.)

### 4.2 KG construction pipeline

The `SimpleKGPipeline` wrapper takes an LLM, a Neo4j driver, an embedder, and a schema of
node types, relationship types, and allowed patterns. It accepts text or PDF [28]. Its
components run in this order [30]:

1. **Loader**: PDF or Markdown.
2. **Splitter**: `FixedSizeSplitter`, or LangChain/LlamaIndex adapters.
3. **TextChunkEmbedder** (optional).
4. **Schema**: written by hand with `SchemaBuilder`, or derived by an LLM with
   `SchemaFromTextExtractor`. It can be saved as JSON or YAML.
5. **LexicalGraphBuilder**: `Document` and `Chunk` nodes, with `NEXT_CHUNK` and
   `PART_OF_DOCUMENT` relationships.
6. **LLMEntityRelationExtractor**: schema-guided extraction with structured output and an
   `on_error` policy. Entities are linked to their chunk with `PART_OF_CHUNK`.
7. **GraphPruning**: removes elements that break schema patterns or `EXISTENCE`
   constraints, drops empty nodes, and fixes relationship direction.
8. **Neo4jWriter**: batched writes, 1,000 by default.
9. **Entity resolution**: exact match on label and name; fuzzy match with RapidFuzz
   Levenshtein distance (faster, less precise); or semantic match with spaCy embeddings
   (better, but limited by static vectors).

The KG-builder components live in an experimental namespace and may change [28].

```
            LEXICAL LAYER                         DOMAIN LAYER
(:Document)<-[:PART_OF_DOCUMENT]-(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
                                    ^
                                    | [:PART_OF_CHUNK]
                              (:__Entity__:Person)-[:WORKS_AT]->(:__Entity__:Org)
```

Keeping a **lexical graph** (documents and chunks) next to the **domain graph** (entities)
recurs across vendors. It preserves provenance and lets every retriever fall back to
chunk text.

### 4.3 LLM Graph Builder

Neo4j Labs' LLM Graph Builder is an application built on these ideas [31]:

- **Ingestion** from local files, GCS, S3, YouTube, Wikipedia, and web pages, using
  LangChain-based extraction with OpenAI, Gemini, Diffbot, and (in dev deployments)
  Bedrock, Anthropic, Ollama, and others.
- **Optional schema** constraints.
- **Post-processing** that cleans the graph and merges duplicates by similarity score.
- **Chat modes**: `vector`, `graph_vector`, `graph`, `fulltext`, `graph_vector_fulltext`,
  `entity_vector`, and `global_vector`.

It is a practical reference for multi-strategy retrieval over one graph.

---

## 5. Other industrial and research designs

### 5.1 AWS GraphRAG Toolkit: hierarchical lexical graph

The AWS `graphrag-toolkit` has two parts [34][35]:

- **lexical-graph** builds a "hierarchical lexical graph" from unstructured text and
  answers questions with a traversal-based search engine. In the toolkit's docs, the
  hierarchy runs from sources and chunks, through topics, statements, and facts, to
  entities.
- **BYOKG-RAG** answers questions over a knowledge graph the user already has.

Supported stores are Neptune Analytics, Neptune Database, and Neo4j for the graph, and
Neptune Analytics, OpenSearch Serverless, S3 Vectors, and Postgres with pgvector for
vectors. Models run on Bedrock. Batch extraction uses Bedrock batch inference, and
"versioned updates" support documents that change over time [35]. The design choice to
note: the graph is built around *statements* (sentence-level propositions), so retrieval
returns compact facts with their provenance, not whole chunks.

### 5.2 Zep and Graphiti: temporal knowledge graphs for agent memory

Graphiti is the open-source core of Zep. It targets *dynamic* data such as conversations
and business events, not static corpora [32][33]:

- **Episodes** are ingested incrementally, without recomputing the graph in batch.
- **Facts** are entity-relationship-entity triplets with validity windows. When a fact
  changes, the old one is *invalidated*, not deleted, so past states can still be queried.
  The README calls this bi-temporal tracking.
- **Retrieval** combines embeddings, BM25, and graph traversal, without LLM summarization
  at query time.
- **Custom entity types** are defined as Pydantic models.
- **Back ends**: Neo4j, FalkorDB, Neptune (with OpenSearch Serverless), and Kuzu, which
  is now deprecated.

The README contrasts Graphiti's "typically sub-second" query latency with GraphRAG's
"seconds to tens of seconds". It also says Zep's managed platform gets sub-200 ms at
scale, but publishes no benchmark for that [32]. The Zep paper reports better results than
MemGPT on the Deep Memory Retrieval benchmark, and accuracy and latency gains on
LongMemEval. Treat these as vendor-reported [33].

### 5.3 KAG (Ant Group / OpenSPG)

KAG ("Knowledge Augmented Generation") targets professional domains where vector
similarity is ambiguous and OpenIE adds noise [36][37]:

- It has a kg-builder and a kg-solver. The kag-model component is not yet open-sourced.
- Its knowledge representation follows the DIKW hierarchy and supports both schema-free
  and schema-constrained extraction.
- **Mutual indexing** links graph elements and text chunks in both directions.
- **Logical-form-guided hybrid reasoning** breaks a question into planning, retrieval, and
  reasoning operators. Each step can use exact match, text retrieval, numerical
  computation, or semantic reasoning [36].

Release 0.7 (April 2025) added a "Lightweight Build" mode that the README says cuts
construction token cost by 89%. Release 0.8 (June 2025) added MCP support and KAG-Thinker
[36].

### 5.4 Think-on-Graph (ICLR 2024): the LLM as graph explorer

Think-on-Graph has the LLM explore an existing KG (Freebase or Wikidata) iteratively. It
expands and prunes candidate relation and entity paths, then reasons over the paths it
kept [41]. This is the "agentic traversal" end of the design space: no precomputed
summaries, with an LLM call at every hop. It is suited to curated KGs rather than graphs
extracted from a private corpus.

### 5.5 LinkedIn: a KG over support tickets

LinkedIn's customer-service QA system (SIGIR 2024) builds a graph from historical support
tickets [45]:

- Each ticket is parsed into a tree of sections, which keeps the ticket's internal
  structure.
- Tickets are linked to each other by explicit references and by similarity.
- At query time, the system extracts entities and intents, and retrieves sub-graphs either
  by translating the question into a graph query or by embedding search.

The paper reports large retrieval gains over chunk-based RAG, roughly a 77.6% gain in MRR,
and about a 28.6% drop in median per-issue resolution time after deployment. These figures
are approximate, taken from the paper's abstract. The design point is that the graph
encodes **document structure** (intra-ticket trees) as well as entities.

### 5.6 Survey context

The Graph RAG survey by Peng et al. [46] splits the field into three stages: graph-based
indexing, graph-guided retrieval, and graph-enhanced generation. The systems in this
chapter map onto those stages. G-Retriever [47] is an example of the "retrieve a subgraph,
then encode it for the LLM" family; it is mainly aimed at textual-graph question
answering.

---

## 6. KG construction with LLMs: the shared engineering problem

Every design above, except LazyGraphRAG and FastGraphRAG, depends on LLM extraction. The
same issues come up in each.

**Schema: open or closed.**

- Open extraction (GraphRAG, HippoRAG, LightRAG) maximizes recall but gives heterogeneous
  types and predicates.
- Closed schemas (the Neo4j `SchemaBuilder` with allowed patterns [30], KAG's
  schema-constrained mode [36], Graphiti's Pydantic entity types [32]) produce queryable
  graphs, but they miss anything outside the schema.
- A common hybrid has the LLM *propose* a schema from samples, then constrains extraction
  to it. Examples are GraphRAG auto-tuning [5] and `SchemaFromTextExtractor` [30].

**Extraction granularity.**

| Unit extracted          | Example systems                  | Strength                         | Weakness |
|-------------------------|----------------------------------|----------------------------------|----------|
| Entities + relations with descriptions | GraphRAG, LightRAG [13][21] | rich, summarizable nodes | costly; descriptions drift |
| OpenIE triples          | HippoRAG [23]                    | cheap, precise links for PPR     | noisy predicates; loses context |
| Statements/facts        | AWS lexical graph [35], Graphiti [32] | compact, citable, time-aware | needs a proposition model |
| Noun-phrase co-occurrence | LazyGraphRAG, FastGraphRAG [8][14] | no LLM cost                  | no relation semantics |
| Schema-typed property graph | Neo4j KG builder [30], KAG [36] | Cypher-queryable, constrained | brittle to schema gaps |

**Entity resolution** is the step most often underestimated. The options, from cheapest
to most expensive:

1. Exact match on name and type: GraphRAG [13], Neo4j's exact resolver [30].
2. Embedding-similarity synonymy edges, which leave nodes separate: HippoRAG [26].
3. Fuzzy or semantic merging: Neo4j's RapidFuzz and spaCy resolvers [30], LLM Graph
   Builder's similarity merge [31].
4. LLM-assisted clustering of entities and edges: KGGen [38][39].

KGGen (Stanford, NeurIPS 2025) generates triples with LiteLLM and DSPy, then clusters
similar entities and edges, so the graph is less sparse and less redundant. It is
evaluated on the MINE benchmark [38][39].

**Prompt tuning has a large effect.** GraphRAG's own numbers show that domain-adapted
prompts can nearly triple the number of extracted entities and communities [5]. KAG's
lightweight build and LazyGraphRAG show how much token cost can be saved by moving
extraction to cheaper models or to NLP [8][36].

**Pruning and validation.** Schema-based pruning drops invalid patterns and fixes
relationship direction [30]. `on_error` policies decide whether a malformed LLM output
breaks the pipeline [30]. Structured-output support is "important for reliable ingestion"
according to Graphiti [32].

A reference construction pipeline, assembled from the components above:

```
docs -> parse -> chunk (+lexical graph: Document/Chunk/NEXT_CHUNK)
     -> [sample] -> LLM schema proposal / prompt auto-tune
     -> per-chunk LLM extraction (structured output, schema-constrained)
     -> validate + prune (schema patterns, direction, empties)
     -> entity resolution (exact -> fuzzy/semantic -> LLM adjudication)
     -> write graph (batched) + embed nodes/relations/chunks
     -> optional: community detection (Leiden) + LLM reports
     -> optional: temporal validity on edges (valid_from / invalidated_at)
```

---

## 7. When graphs beat vectors, and when they do not

### 7.1 The evidence

- **Corpus-level sensemaking.** On global questions judged for comprehensiveness and
  diversity, every GraphRAG variant beat naive RAG with roughly 70-80% win rates [4].
  LazyGraphRAG beat vector RAG even when vector RAG had a 1M-token context [10].
- **Multi-hop association.** PPR over an OpenIE graph improved multi-hop retrieval by up
  to about 20%, and single-step graph retrieval rivalled iterative LLM retrieval at a
  fraction of the cost [23].
- **Entity-anchored questions that need many mentions combined.** The Novorossiya example
  [3] and DRIFT's results over local search [6].
- **Structured, relational domains.** These include ticket graphs [45], professional
  knowledge bases with logical constraints [36], and agent memory where facts change over
  time [32][33].

On the other side:

- **Single-hop factual lookup.** HippoRAG 2 was motivated by structure-augmented RAG
  falling below strong dense retrievers on simple factual QA [25]. BenchmarkQED's one
  non-significant result was relevance on data-local queries, where vector RAG ranks
  directly relevant chunks well [10]. GraphRAG-Bench starts from the observation that
  "GraphRAG frequently underperforms vanilla RAG on many real-world tasks" [43]. It sorts
  tasks into four levels: fact retrieval, complex reasoning, contextual summarization, and
  creative generation. The paper's analysis finds graph structure helps mainly on the
  harder reasoning and summarization levels [43]. A separate systematic comparison, "RAG
  vs. GraphRAG", reports the same split: vector RAG is better on detailed single-hop
  questions, and graph methods help on multi-hop and summarization queries [44].
- **The judge can favour graph methods.** Comprehensiveness and diversity favour long
  answers built from many reports. LightRAG's lead over GraphRAG comes mostly from
  diversity, with comprehensiveness close to parity [21]. Systems built for exact answers
  should also be evaluated on correctness, which BenchmarkQED's AutoE supports when ground
  truth exists [10].
- **Cost and freshness.** If the corpus changes often and global questions are rare, full
  GraphRAG indexing is hard to justify. LazyGraphRAG [8], LightRAG's incremental merge
  [21], or Graphiti's episodic updates [32] fit better.

### 7.2 Comparison table

| System | Graph built from | Index-time LLM | Retrieval mechanism | Global questions | Multi-hop | Incremental | Public cost signal |
|---|---|---|---|---|---|---|---|
| GraphRAG (standard) [1][13] | LLM entities/relations + Leiden + reports | very high (extraction about 75% of it) | global map-reduce; local; DRIFT | strong | moderate (local/DRIFT) | `update` with deltas | root summaries at 2-3% of TS tokens [4] |
| GraphRAG dynamic selection [7] | same | same | LLM-rated hierarchy pruning, then map-reduce | strong | n/a | same | -77% tokens vs static L1 |
| FastGraphRAG [14] | noun phrases + co-occurrence, LLM reports | reports only | as GraphRAG | good | weak | same | "much cheaper" (no number) |
| LazyGraphRAG [8] | NLP concept graph + communities | none (same as vector RAG) | budgeted best-first + breadth-first, map-reduce over claims | matches GraphRAG global | good | cheap rebuild | 0.1% of GraphRAG index; >700x cheaper global query |
| LightRAG [20][21] | LLM entities/relations | high | dual-level keyword match plus 1-hop expansion (+ chunks, reranker) | relation-level | moderate | merge and delete | one keyword call per query (paper) |
| HippoRAG / 2 [23][24] | LLM OpenIE triples + synonymy edges (+ passage nodes in v2) | medium | PPR from query entities or triples | limited (v2: sense-making) | strong | append | 10-30x cheaper than IRCoT (v1) |
| fast-graphrag [27] | LLM graph | medium | PPR | limited | strong | append | $0.08 vs $0.48 on one book |
| Neo4j GraphRAG [28][29] | schema-typed property graph + lexical graph | medium-high | vector/hybrid entry + Cypher; Text2Cypher; tools | app-defined | strong (explicit Cypher) | DB writes | n/a |
| AWS lexical graph [35] | sources, chunks, topics, statements, facts, entities | high | traversal-based search | moderate | good | versioned updates | n/a |
| Graphiti / Zep [32] | episodic, time-stamped triplets | per episode | embeddings + BM25 + traversal | weak | good | native, real-time | sub-second (claimed) |
| KAG [36] | schema-constrained KG with mutual chunk index | high (lightweight build -89% tokens) | logical-form operators | moderate | strong | rebuild | n/a |

### 7.3 Decision heuristics

```
Is the question about the WHOLE corpus (themes, trends, "what are the main...")?
   yes -> community-hierarchy method: LazyGraphRAG (cheap index) or GraphRAG global /
          dynamic selection (precomputed reports, repeated queries on a stable corpus)
   no  -> Does the answer need >= 2 hops through shared entities?
            yes -> PPR graph (HippoRAG 2 / fast-graphrag) or vector entry + Cypher
                   expansion (Neo4j); DRIFT if a GraphRAG index already exists
            no  -> Is the domain inherently relational/typed with exact constraints
                   (org charts, tickets, products, regulations)?
                     yes -> property graph + Text2Cypher / curated KG (KAG, Neo4j)
                     no  -> hybrid vector + BM25 + reranker (chapter 05); add a graph
                            only when evaluations show multi-hop failures
Do facts change over time and must history be queryable? -> temporal KG (Graphiti)
```

---

## 8. Cross-cutting failure modes

1. **Extraction noise propagates.** A wrong edge in the KG becomes a confident
   multi-hop path. KAG cites OpenIE noise as a motivation [36]. HippoRAG 2 filters triples
   with an LLM recognition step before PPR [24].
2. **Entity fragmentation or over-merging.** Too little resolution splits communities and
   breaks PPR paths. Too much merges distinct entities, for example two people with the
   same name. Resolution needs its own evaluation set.
3. **Summary drift.** Merged entity descriptions and community reports are LLM summaries
   of summaries. Their claims should be traced back to TextUnits. GraphRAG reports cite
   source records for this reason [3]. Microsoft's later VeriTrail work on tracing
   provenance in multi-step pipelines targets the same risk [11].
4. **Text2Cypher brittleness.** Generated queries may be syntactically or semantically
   wrong [29]. Production systems pair Text2Cypher with templated retrievers or route
   through a `ToolsRetriever` [29].
5. **Hidden cost cliffs.** Lower community levels, larger relevance-test budgets, and
   DRIFT iterations all multiply LLM calls [8][15][6]. Budgets must be explicit
   configuration, not defaults nobody looked at.
6. **Benchmark mismatch.** Most published wins are LLM-judge win rates on synthetic
   sensemaking queries [4][8][21]. Check them against domain queries that have ground
   truth before committing to a costly index.

---

## Key takeaways for system designers

1. **Name the query class first.** Graphs pay off for corpus-level sensemaking and for
   multi-hop association. They often do not for single-hop fact lookup, where a tuned
   hybrid vector stack is as good or better [10][25][43][44].
2. **Precompute only what repeated queries will reuse.** Full GraphRAG community reports
   make sense for a stable corpus that gets many global questions. Otherwise LazyGraphRAG
   gets similar global quality with an index at 0.1% of the cost [8].
3. **Put a cheap model in front of an expensive one.** A GPT-4o-mini relevance rater cut
   global-search tokens by 77% [7]. LazyGraphRAG uses a low-cost LLM for its relevance
   tests [8]. LightRAG uses a fast model for extraction and keywords [21].
4. **Use graph walks for multi-hop, not repeated LLM calls.** PPR (HippoRAG,
   fast-graphrag) or a learned GNN (GFM-RAG) replaces iterative retrieval at a fraction
   of the cost [23][27][40].
5. **Keep a lexical graph under the knowledge graph.** Document, chunk, and statement
   nodes linked to entities keep provenance and give a fallback to raw text [30][35].
6. **Treat entity resolution and prompt tuning as first-class work.** Domain-tuned
   prompts nearly tripled extracted structure in Microsoft's test [5]. Resolution quality
   decides whether communities and paths mean anything.
7. **Make incrementality a requirement.** Choose designs whose update cost scales with
   the delta: GraphRAG `update`, LightRAG merge and delete, Graphiti episodes, AWS
   versioned updates [9][21][32][35].
8. **Build a router, not a single engine.** GraphRAG's own design ships global, local,
   DRIFT, and basic modes. LightRAG defaults to `mix`. Neo4j offers `ToolsRetriever`
   [6][21][29]. Route by query class.
9. **Evaluate on two axes.** Use LLM-judge win rates for sensemaking, and correctness
   against ground truth for factual queries (BenchmarkQED AutoQ/AutoE, GraphRAG-Bench)
   [10][43].

---

## References

1. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., Larson, J. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv, 2024. https://arxiv.org/abs/2404.16130
2. Microsoft Research. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (publication page). 2024. https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/
3. Larson, J., Truitt, S. (Microsoft Research). "GraphRAG: Unlocking LLM discovery on narrative private data." 2024. https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
4. Edge, D., Trinh, H., Truitt, S., Larson, J. (Microsoft Research). "GraphRAG: New tool for complex data discovery now on GitHub." 2024. https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/
5. Guevara Fernández, A., et al. (Microsoft Research). "GraphRAG auto-tuning provides rapid adaptation to new domains." 2024. https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/
6. Whiting, J., Hills, Z., Guevara Fernández, A., Trinh, H., Bradley, A., Larson, J. (Microsoft Research / Uncharted Software). "Introducing DRIFT Search: Combining global and local search methods to improve quality and efficiency." 2024. https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/
7. Li, B., Trinh, H., Edge, D., Larson, J. (Microsoft Research). "GraphRAG: Improving global search via dynamic community selection." 2024. https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/
8. Edge, D., Trinh, H., Larson, J. (Microsoft Research). "LazyGraphRAG: Setting a new standard for quality and cost." 2024 (editor's note 2025). https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
9. Evans, N., Guevara Fernández, A., Bradley, J. (Microsoft Research). "Moving to GraphRAG 1.0: Streamlining ergonomics for developers and users." 2024. https://www.microsoft.com/en-us/research/blog/moving-to-graphrag-1-0-streamlining-ergonomics-for-developers-and-users/
10. Edge, D., Trinh, H., Morales Esquivel, A., Larson, J. (Microsoft Research). "BenchmarkQED: Automated benchmarking of RAG systems." 2025. https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/
11. Microsoft Research. "Project GraphRAG." 2024-2025. https://www.microsoft.com/en-us/research/project/graphrag/
12. Microsoft. "graphrag" GitHub repository (README). 2024-2026. https://github.com/microsoft/graphrag
13. Microsoft. "GraphRAG Indexing Dataflow" (docs/index/default_dataflow.md). 2025. https://github.com/microsoft/graphrag/blob/main/docs/index/default_dataflow.md
14. Microsoft. "GraphRAG Indexing Methods: Standard vs FastGraphRAG" (docs/index/methods.md). 2025. https://github.com/microsoft/graphrag/blob/main/docs/index/methods.md
15. Microsoft. "Global Search" (docs/query/global_search.md). 2025. https://github.com/microsoft/graphrag/blob/main/docs/query/global_search.md
16. Microsoft. "Local Search" (docs/query/local_search.md). 2025. https://github.com/microsoft/graphrag/blob/main/docs/query/local_search.md
17. Microsoft. "DRIFT Search" (docs/query/drift_search.md). 2025. https://github.com/microsoft/graphrag/blob/main/docs/query/drift_search.md
18. Microsoft. "Auto Prompt Tuning" (docs/prompt_tuning/auto_prompt_tuning.md). 2025. https://github.com/microsoft/graphrag/blob/main/docs/prompt_tuning/auto_prompt_tuning.md
19. Microsoft. "benchmark-qed" GitHub repository. 2025. https://github.com/microsoft/benchmark-qed
20. Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C. "LightRAG: Simple and Fast Retrieval-Augmented Generation." arXiv, 2024 (EMNLP 2025). https://arxiv.org/abs/2410.05779
21. HKUDS. "LightRAG" GitHub repository (README, evaluation and news). 2024-2026. https://github.com/HKUDS/LightRAG
22. HKUDS. "LightRAG Reproduce" (docs/Reproduce.md). 2024. https://github.com/HKUDS/LightRAG/blob/main/docs/Reproduce.md
23. Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., Su, Y. "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models." NeurIPS 2024. https://arxiv.org/abs/2405.14831
24. Gutiérrez, B. J., Shu, Y., Qi, W., Zhou, S., Su, Y. "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models" (HippoRAG 2). ICML 2025. https://arxiv.org/abs/2502.14802
25. OSU NLP Group. "HippoRAG" GitHub repository (HippoRAG 2 README). 2025. https://github.com/OSU-NLP-Group/HippoRAG
26. OSU NLP Group. "HippoRAG" legacy branch (v1 README). 2024. https://github.com/OSU-NLP-Group/HippoRAG/tree/legacy
27. Circlemind. "fast-graphrag" GitHub repository. 2024-2025. https://github.com/circlemind-ai/fast-graphrag
28. Neo4j. "neo4j-graphrag-python" GitHub repository. 2024-2026. https://github.com/neo4j/neo4j-graphrag-python
29. Neo4j. "User Guide: RAG" (neo4j-graphrag-python docs, user_guide_rag.rst). 2025. https://github.com/neo4j/neo4j-graphrag-python/blob/main/docs/source/user_guide_rag.rst
30. Neo4j. "User Guide: Knowledge Graph Builder" (neo4j-graphrag-python docs, user_guide_kg_builder.rst). 2025. https://github.com/neo4j/neo4j-graphrag-python/blob/main/docs/source/user_guide_kg_builder.rst
31. Neo4j Labs. "LLM Graph Builder" GitHub repository. 2024-2026. https://github.com/neo4j-labs/llm-graph-builder
32. Zep. "Graphiti" GitHub repository. 2024-2026. https://github.com/getzep/graphiti
33. Rasmussen, P., et al. (Zep). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv, 2025. https://arxiv.org/abs/2501.13956
34. AWS Labs. "graphrag-toolkit" GitHub repository. 2024-2026. https://github.com/awslabs/graphrag-toolkit
35. AWS Labs. "lexical-graph" (graphrag-toolkit README). 2025-2026. https://github.com/awslabs/graphrag-toolkit/tree/main/lexical-graph
36. OpenSPG / Ant Group. "KAG: Knowledge Augmented Generation" GitHub repository. 2024-2025. https://github.com/OpenSPG/KAG
37. Liang, L., et al. (Ant Group). "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation." arXiv, 2024. https://arxiv.org/abs/2409.13731
38. Stanford STAIR Lab. "kg-gen" GitHub repository. 2025. https://github.com/stair-lab/kg-gen
39. Mo, B., Yu, K., Kazdan, J., Cabezas, J., Mpala, P., Yu, L., Cundy, C., Kanatsoulis, C., Koyejo, S. "KGGen: Extracting Knowledge Graphs from Plain Text with Language Models." arXiv, 2025 (NeurIPS 2025). https://arxiv.org/abs/2502.09956
40. Luo, L., et al. "GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation" (GitHub repository; arXiv 2502.01113; NeurIPS 2025). 2025. https://github.com/RManLuo/gfm-rag
41. Sun, J., et al. "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" (ICLR 2024; GitHub repository). 2024. https://github.com/IDEA-FinAI/ToG
42. Chen, B., et al. (BUPT GAMMA). "PathRAG" GitHub repository. 2025. https://github.com/BUPT-GAMMA/PathRAG
43. GraphRAG-Bench. "When to use Graphs in RAG: A Comprehensive Benchmark and Analysis for Graph Retrieval-Augmented Generation" (GitHub repository; arXiv 2506.05690). 2025. https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
44. Han, H., et al. "RAG vs. GraphRAG: A Systematic Evaluation and Key Insights." arXiv, 2025. https://arxiv.org/abs/2502.11371
45. Xu, Z., et al. (LinkedIn). "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering." SIGIR 2024. https://arxiv.org/abs/2404.17723
46. Peng, B., et al. "Graph Retrieval-Augmented Generation: A Survey." arXiv, 2024. https://arxiv.org/abs/2408.08921
47. He, X., et al. "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering." arXiv, 2024. https://arxiv.org/abs/2402.07630
48. Azure Samples / Microsoft. "GraphRAG Accelerator" GitHub repository. 2024. https://github.com/Azure-Samples/graphrag-accelerator
