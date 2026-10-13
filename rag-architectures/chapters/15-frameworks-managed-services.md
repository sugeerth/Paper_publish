# 15. RAG Frameworks and Managed Services: From Orchestration Libraries to Retrieval-as-a-Service

## Overview

Most production RAG systems are not written from scratch. They are assembled from one of two kinds of building blocks:

1. **Open-source orchestration frameworks.** LangChain/LangGraph, LlamaIndex, Haystack, DSPy, Semantic Kernel (now Microsoft Agent Framework), the Vercel AI SDK and RAGFlow. These give you abstractions for loaders, splitters, embedders, vector stores, retrievers, rerankers, prompts and agents. You still own the infrastructure.
2. **Managed retrieval services.** Amazon Bedrock Knowledge Bases, Google's Vertex AI RAG Engine and Vertex AI Search, Azure AI Search, OpenAI file search and vector stores, and Anthropic's Citations API. These take over some or all of ingestion, chunking, embedding, indexing, ranking and grounded generation behind an API.

Between 2023 and 2026 the two camps converged. The frameworks moved from fixed "retrieve then generate" chains toward agent loops in which retrieval is one tool among many. LangChain v1 moved its classic retrievers into a legacy package and now centres on `create_agent` [3]. Haystack documents a "pipelines vs agents" decision [14]. LlamaIndex says its "primary focus has shifted towards LlamaParse" [5]. The managed services moved the same way, from single-shot vector search toward *agentic retrieval*: LLM query planning, parallel subqueries and iterative refinement run server-side. Azure AI Search's knowledge bases with retrieval reasoning effort levels [49][50], Bedrock's query decomposition [35] and Vertex AI RAG Engine's LLM ranker [44] are all examples.

This chapter describes each system's architecture, data flow, key techniques, published limits and numbers, and failure modes. It closes with comparison tables and design takeaways.

A note on sources. Where possible this chapter relies on each project's own documentation repositories, sample notebooks and engineering blogs as they stood in October 2026. Star counts and version numbers are snapshots and are marked approximate.

## 15.1 A reference model: who owns which layer?

Every framework and service in this chapter can be placed on a five-layer stack. The useful design question is which layers you delegate.

```
 +--------------------------------------------------------------------------------+
 | L5 ORCHESTRATION  agent loop / graph / pipeline; tools; memory; HITL; retries  |
 |    LangGraph, Haystack Pipeline+Agent, LlamaIndex Workflows, AI SDK agents,    |
 |    Agent Framework, DSPy modules (+ optimizer that compiles prompts/weights)   |
 +--------------------------------------------------------------------------------+
 | L4 SYNTHESIS      prompt assembly, grounded generation, citations, guardrails  |
 |    Anthropic Citations, Bedrock RetrieveAndGenerate, Azure answer synthesis,   |
 |    OpenAI file_search annotations, Vertex grounding / check-grounding          |
 +--------------------------------------------------------------------------------+
 | L3 RETRIEVAL      query rewrite/decompose, hybrid search, fusion, rerank,      |
 |                   metadata filters, thresholds                                 |
 +--------------------------------------------------------------------------------+
 | L2 INDEX          vector store / inverted index / graph / SQL; ACLs            |
 +--------------------------------------------------------------------------------+
 | L1 INGEST         connectors, parsing/OCR/layout, chunking, enrichment, embed  |
 +--------------------------------------------------------------------------------+

  Frameworks: thin over L1-L3 (adapters), thick at L5.
  Managed services: thick at L1-L4, increasingly reaching into L5 (agentic retrieval).
  Anthropic Citations: L4 only (bring your own retrieval).
```

## 15.2 Open-source frameworks

### 15.2.1 LangChain, LangGraph and Deep Agents

**Problem addressed.** LangChain began as a way to chain LLM calls with retrievers and tools behind interchangeable interfaces. Its repository now calls itself "The agent engineering platform". It has roughly 147.7k GitHub stars and a standard interface for models, embeddings, vector stores and retrievers across hundreds of integrations [1].

**Architecture (v1).** The v1 release reorganised the project around three ideas [3]:

- **`create_agent`** is the standard agent constructor. It replaces `langgraph.prebuilt.create_react_agent` and runs the basic loop: call the model, execute any tools it picks, stop when it calls none.
- **Middleware** is the extension point for "context engineering": dynamic prompts, summarisation, selective tool access, state management and guardrails. Prebuilt examples include `PIIMiddleware`, `SummarizationMiddleware` and `HumanInTheLoopMiddleware`.
- **Standard `content_blocks`** give provider-neutral access to reasoning, citations and other modern output types.

The legacy material moved into **`langchain-classic`**: old chains, the retrievers from the former `langchain.retrievers` module (for example `MultiQueryRetriever`), the indexing API, the hub module and `langchain-community` exports [3]. This changes how RAG is built in LangChain. The 2023-era `RetrievalQA` and `MultiQueryRetriever` patterns are now legacy, and retrieval is expected to be exposed as a *tool* inside an agent. LangGraph 1.0 sits underneath. It is the lower-level graph runtime whose release notes stress stable core APIs and "durable execution with checkpointing, persistence, streaming, and human-in-the-loop" [3].

**Three RAG architectures.** LangChain's retrieval documentation sets out a taxonomy worth adopting [2]:

| Architecture | Description | Control | Flexibility | Latency | Example |
|---|---|---|---|---|---|
| 2-Step RAG | Retrieval always precedes generation | High | Low | Fast, predictable | FAQ and docs bots |
| Agentic RAG | An LLM agent decides when and how to retrieve | Low | High | Variable | Research assistants |
| Hybrid | Query enhancement, retrieval validation, answer validation, with loops | Medium | Medium | Variable | Domain Q&A with QA checks |

The documentation explains why 2-step latency is predictable: "the maximum number of LLM calls is known and capped". It also notes that retrieval-side latency (API calls, database queries) can still vary [2].

**Deep Agents patterns.** For heavier workloads, LangChain's Deep Agents layer adds planning, sub-agents, a filesystem backend, skills and grading rubrics. Its RAG guide names four patterns [4]:

- **Skills-guided retrieval.** A skill file tells the agent which index to use, how to phrase queries and how to format citations.
- **Rubric-checked grounding.** A grader sub-agent built with `RubricMiddleware` checks that the draft answer is grounded and forces revisions up to an iteration cap. This is in beta and requires `deepagents>=0.6.5`.
- **Todo-driven investigation.** The agent writes a to-do list of pages or queries to work through.
- **Retrieve, offload and delegate.** Retrieved chunks are written to the agent filesystem instead of kept in the orchestrator's context. Sub-agents then read and summarise the files in parallel, so the main context stays clean.

```
 user question
      |
      v
 +-----------+   vector search tool   +--------------+
 |orchestrator|--------------------->| vector store |
 |  (agent)   |<--- chunk ids -------+--------------+
 +-----+------+
       | writes chunks as files (offload)
       v
 +-------------+   parallel    +-----------+  +-----------+
 | agent FS    |-------------->| subagent 1|  | subagent n|  read + summarise
 +-------------+               +-----+-----+  +-----+-----+
                                     \             /
                                      v           v
                               orchestrator synthesises answer + citations
```

**Failure modes and lessons.**

- Abstractions churn quickly. Code written against 2023 chain classes now depends on `langchain-classic` [3].
- Agentic RAG gives up latency predictability [2]. The rubric loop and fan-out patterns multiply LLM calls, so iteration caps are mandatory [4].
- The offload-to-filesystem pattern targets one specific failure: orchestrator context bloat when many long chunks are retrieved.

### 15.2.2 LlamaIndex and LlamaParse

**Problem addressed.** LlamaIndex (MIT, roughly 52.5k stars) began as a data framework for connecting documents to LLMs. It has more than 300 integration packages [5]. The README now states that the company's "primary focus has shifted towards LlamaParse", that "agents are the new consumers of documents", and that the OSS framework remains available [5].

**Core abstractions.**

- **Documents and Nodes.** Documents are parsed into Nodes by node parsers. The API reference lists sentence, token, semantic, hierarchical, sentence-window, markdown, HTML, JSON, code and third-party parsers such as Docling and Chonkie [10].
- **`IngestionPipeline`.** This applies a sequence of *transformations* (parse, extract metadata, embed) and can write directly to a vector store. Every node-plus-transformation pair is hashed and cached, so a re-run on unchanged data is close to free once the cache is persisted [6]. If the pipeline writes to a vector store, embedding must be one of its stages, or building the index later fails [6].
- **Query stages.** The documentation splits querying into **retrieval** (top-k semantic retrieval by default), **postprocessing** (rerank, transform, filter by metadata or similarity cut-off) and **response synthesis**. A `RetrieverQueryEngine` composes the three [7].
- **Advanced indexes and retrievers.** These include `PropertyGraphIndex`, which pairs implicit, schema-guided and simple LLM path extractors with retrievers based on text-to-Cypher, Cypher templates, LLM synonym expansion and vector context [10]. There are also BM25, auto-merging, recursive, router and query-fusion retrievers, and sub-question and multi-step query engines [10].

**Workflows.** Orchestration moved to `llama-index-workflows`. Workflows are event-driven: "steps are async Python functions that emit and consume events". Steps are marked with `@step`, a run is bounded by `StartEvent` and `StopEvent`, and persistence is pluggable ("save and resume runs from a file, or connect to a database"). `llama-agents-server` exposes a workflow as a REST API with streaming and human-in-the-loop [9].

**LlamaParse (formerly LlamaCloud).** This is the hosted platform. It offers Parse (agentic OCR across 130+ formats), Extract (schema-driven structured extraction), Classify, Split and Index (managed ingestion and retrieval). It plugs into the framework as readers, indexes and retrievers. The older `llama-parse` and `llama-cloud-services` packages are deprecated in favour of `llama-cloud` 2.x [5][8].

**Lessons.**

- LlamaIndex's history shows where the hard problem in RAG ended up: parsing and document understanding, not orchestration. The company reorganised around that layer [5][8].
- Ingestion caching is an underrated cost control for corpora that are re-indexed often [6].
- The README warns that it "is not updated as frequently as the documentation". Pin versions, and treat the docs site as the source of truth [5].

### 15.2.3 Haystack (deepset)

**Problem addressed.** Haystack is a Python framework (roughly 26.7k stars) for "modular pipelines and agent workflows with explicit control over retrieval, routing, memory, and generation" [11]. Its README lists users including Apple, Meta, Databricks, NVIDIA, the European Commission, Airbus and Netflix. deepset sells a managed or self-hosted Haystack Enterprise Platform [11]. At the time of writing, the main branch's version file read `3.4.0-rc0`.

**Architecture.**

- **Components** declare typed inputs and outputs with `@component.output_types(...)` and a `run()` method. Heavy components (embedders, LLMs) load lazily through `warm_up()` [12].
- **Pipelines** are "directed multigraphs" of components. They support branching (for example, routing files to per-type converters), loops and standalone components [12].
- **Loops** let a component re-run whenever its required inputs are ready again. Termination is guaranteed by `max_runs_per_component`, which defaults to 100 and raises `PipelineMaxComponentRuns` when exceeded [13]. This is the built-in safeguard against runaway self-correction loops.
- **Async execution.** `run_async`, `run_async_generator` and `stream` run independent branches (for example several retrievers) in parallel [12].
- **SuperComponents.** The `@super_component` decorator wraps a whole pipeline as a single component. The canonical example is a `HybridRetriever` that runs BM25 and embedding retrieval in parallel and merges the results [15].
- **Agents** are themselves components with lifecycle hooks (`before_llm`, `before_tool`, `on_exit`) and step and token accounting [11]. A pipeline can be exposed to an agent as one deterministic tool through `PipelineTool` [14].

**Design guidance.** Haystack's documentation states the predictability-versus-freedom trade-off plainly [14]:

- A fixed pipeline gives bounded LLM calls, predictable latency and cost, and reproducible evaluation.
- An agent handles requests you did not enumerate in advance, but its step count, latency and cost are not fixed.
- "Pipeline loops are a middle ground": for example, a bounded retry-until-valid loop.

**Lessons.** Explicit typed graphs make RAG systems easier to audit and serialise. Bounded loops give you most of the benefit of "agentic" self-correction without handing control flow to the LLM [13][14].

### 15.2.4 DSPy

**Problem addressed.** Hand-written prompts in multi-stage RAG pipelines are brittle and tied to one model. DSPy (Stanford NLP, roughly 38.6k stars, `version="3.4.0"`) treats an LM pipeline as a *program* whose prompts and even weights are *compiled* against a metric. The tagline is "programming—not prompting—language models" [16][19].

**Architecture.**

- **Signatures** declare I/O behaviour, for example `'context, question -> response'`.
- **Modules** such as `dspy.ChainOfThought` implement a signature. They compose into Python classes with a `forward()` method.
- **Optimizers** (formerly teleprompters) tune the program [19]:
  - Few-shot synthesis: `BootstrapFewShot`, `BootstrapFewShotWithRandomSearch`, `KNNFewShot`.
  - Instruction optimisation: `MIPROv2` uses Bayesian optimisation over instructions and demonstrations [17]. `SIMBA` uses mini-batch introspection. `GEPA` is a reflective, genetic-Pareto prompt evolver that can use *textual* feedback as well as scalar scores [18].
  - Weight optimisation: `BootstrapFinetune` distils a prompted program into a fine-tuned model.
  - Optimizers compose. For example, MIPROv2 followed by BootstrapFinetune is "partly the essence of `dspy.BetterTogether`" [19].

A minimal DSPy RAG program [20]:

```
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')
    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
```

**Published numbers (official tutorials).**

- *RAG tutorial* [20]. The setup is a RAG-QA Arena tech corpus, documents truncated to 6,000 characters, OpenAI `text-embedding-3-small` at 512 dimensions, and k=5. Semantic F1 went from about 42% (chain-of-thought, no retrieval) to 55% (unoptimised RAG) to about 61% after `MIPROv2(auto="medium")`. The optimisation run cost about $1.5 and took 20-30 minutes on 200 examples. MIPROv2 splits these 20% train and 80% validation by default.
- *Multi-hop search tutorial* [21]. A 4-hop `Hop` module (query generation plus note-taking) over BM25 went from about 30% to "a little below 60%" top-5 recall. This took about 35 minutes and about $5 of GPT-4o calls used to optimise prompts for Llama-3.1-8B.

**Failure modes.** The tutorial itself warns that prompt optimisation "is not a magic button" and can overfit the training set. Held-out evaluation is essential [20]. DSPy also moves the difficulty into *metric design*: the bootstrap filters on traces that score well, so a weak metric (for example, an LLM-judged semantic F1) bounds what you can reach.

**Lesson.** DSPy is the only framework here that treats retrieval-pipeline prompts as learnable parameters. That makes it complementary to the others, which handle orchestration, deployment and integrations.

### 15.2.5 Semantic Kernel and Microsoft Agent Framework

**Status.** Semantic Kernel (roughly 28.7k stars; Python, .NET, Java) now carries the banner "Semantic Kernel is now Microsoft Agent Framework!". The README calls Agent Framework the "enterprise-ready successor", now at 1.0 with stable APIs, and links migration guides from both Semantic Kernel and AutoGen [22][26].

**Semantic Kernel's RAG abstractions.**

- **Vector Store abstraction.** Python implementations ship with SK. In .NET the abstraction is provided through the .NET Foundation, with connectors shipped by database vendors and the community toolkit (MEVD) [23]. It covers collection management and record CRUD, so you can prototype against a local store and switch to a hosted service later.
- **Text Search.** RAG works by wrapping any `IVectorSearchable<TRecord>` as a Text Search implementation, which can then be exposed to the model as a plugin or function [23].

**Agent Framework RAG.** RAG becomes a property of the agent through **AI Context Providers**. These run around each invocation, adding context before execution and processing results after it [24]. The built-in `TextSearchProvider` (.NET) has two modes, set by `SearchTime` [24]:

- `BeforeAIInvoke`: search on every agent run. The search input can include recent chat turns (`RecentMessageMemoryLimit`, default 0).
- On-demand function calling: the model decides when to search.

Results are wrapped with a default `ContextPrompt` and `CitationsPrompt` ("Include citations to the source document with document name and link..."). Optional `SourceName` and `SourceLink` fields let the agent cite [24].

Packaged context providers exist for Azure AI Search, Microsoft Foundry file search, Neo4j GraphRAG, Redis, Cosmos DB and Mem0 [24]. The Go preview of Agent Framework does not yet support RAG [25].

**Lesson.** The pre-invoke versus on-demand switch is the 2-step versus agentic choice from 15.2.1, exposed as one configuration flag. Making that a deploy-time decision rather than a rewrite is good framework design.

### 15.2.6 Vercel AI SDK

**Problem addressed.** The AI SDK is a TypeScript toolkit for building AI features in web apps. The `ai` package was at version 7.0.x at the time of writing. RAG is not a separate subsystem. It is built from `embed`/`embedMany`, `cosineSimilarity`, a `rerank` function and tool-calling agents [27][28].

**Reference RAG design (official guide)** [27]:

```
 Next.js useChat UI --POST--> /api/chat route --streamText(tools)--> LLM
                                     |                                |
                                     |   tool: getInformation(q)  <---+
                                     v
                     embed(q) -> SQL: 1 - cosineDistance(...) > 0.5
                                     |   ORDER BY sim DESC LIMIT 4
                                     v
          Postgres + pgvector: embeddings(1536-d, HNSW, vector_cosine_ops)
                                     ^
     tool: addResource(content) -> chunk on '.' -> embedMany -> INSERT
```

Details from the guide [27]:

- The model gets two tools. `getInformation` queries the knowledge base. `addResource` writes new knowledge, so the agent can grow its own corpus.
- The system prompt restricts answers to tool results and requires "Sorry, I don't know." when nothing relevant comes back.
- `stopWhen: isStepCount(5)` lets the model answer after a tool call instead of ending on it.
- The guide chunks naively on periods and notes that larger embedding inputs tend to produce lower-quality embeddings.

**Reranking and loop control.** `rerank()` works with provider rerankers, for example `cohere.reranking('rerank-v4.0-pro')` or `bedrock.reranking('cohere.rerank-v3-5:0')`, and accepts structured JSON documents as well as strings [28]. The `ToolLoopAgent` stops after 20 steps by default (`isStepCount(20)`). `WorkflowAgent` has no default limit, so the docs advise setting one explicitly to bound cost [29].

**Lessons.** The AI SDK's minimal design shows that in 2025-2026 a capable tool-calling model plus a SQL vector index is enough for many app-embedded RAG use cases. The hard parts left are chunking quality, thresholds and the refusal policy [27].

### 15.2.7 RAGFlow (InfiniFlow)

**Problem addressed.** RAGFlow (roughly 92.1k stars) is a full RAG *application and engine* rather than a library. It covers deep document understanding, chunking you can see and edit, hybrid retrieval, an agent canvas and APIs, and it presents itself as "a superior context layer for LLMs" [30].

**Architecture.**

- Services include Admin, Ingestor, Syncer and API. Dependencies are MySQL, MinIO, NATS and Kvrocks, plus a document engine: Elasticsearch, or InfiniFlow's own Infinity database [30].
- **DeepDoc** does layout analysis, OCR and table-structure recognition. In the 1.0 open-source release these run on CPU [30].
- **Template-based chunking** is "intelligent and explainable": parsed chunks can be visualised and edited by humans before indexing [30].
- **Retrieval** fuses keyword and vector similarity. *Vector similarity weight* sets the mix (0.3 means 0.7 keyword), and a *similarity threshold* (default 0.2) drops weak candidates. An optional rerank model joins the combined score at a latency cost [32].

**2026 evolution.**

- **v1.0.0-rc1** (September 2026) is "a comprehensive rewrite of RAGFlow in Go", aimed at lower resource use and a lighter deployment. The data upgrade from v0.27.2 is irreversible [31].
- **Knowledge Compilation** (August 2026) generates document-level and dataset-level Wiki, Graph, Tree, PageIndex, Mind Map, Timeline and Skills artefacts. It replaces the earlier GraphRAG and RAPTOR features, which are deprecated in the UI, although content already generated stays searchable [30][31].
- **Agentic Retrieval** decomposes questions, retrieves and verifies evidence, with user-selectable Low/Medium/High/Ultra thinking modes [30].
- Earlier releases added checkpoint/resume for GraphRAG community extraction and entity resolution, which the release notes call "the most expensive and time-consuming parts" of graph indexing. They also added a RAPTOR agglomerative (AHC) tree-building mode and metadata-filter push-down into Infinity [31].

**Lessons.** RAGFlow's history shows a common arc: adopt GraphRAG and RAPTOR, find that graph construction is the cost bottleneck (hence checkpointing), then fold the structures into a general "compile the corpus into derived artefacts" step. Human-editable chunks are a practical answer to parsing errors that no automatic pipeline removes [30][31].

## 15.3 Managed RAG services

### 15.3.1 Amazon Bedrock Knowledge Bases

**Problem addressed.** Bedrock Knowledge Bases (KB) is a fully managed ingest-index-retrieve-generate service tied into Bedrock models, Agents and Guardrails.

**Architecture and data flow.**

```
 S3 / SharePoint / Salesforce / Confluence / web          Redshift (structured)
          |                                                       |
          v  StartIngestionJob (one job at a time per KB)         v
 parse: default | BEDROCK_FOUNDATION_MODEL | BEDROCK_DATA_AUTOMATION
          |                                           metadata/schema ingestion
 chunk: FIXED_SIZE | HIERARCHICAL | SEMANTIC | NONE | Lambda custom
          |                                                       |
 embed (e.g. Titan Text v2, Nova Multimodal Embeddings)           |
          v                                                       v
 vector store: OpenSearch Serverless | Aurora | Pinecone |   NL -> SQL generation
   Redis | MongoDB Atlas | Neptune Analytics (GraphRAG) |
   S3 Vectors | Kendra GenAI index (managed retriever)
          |
 Retrieve / RetrieveAndGenerate
   overrideSearchType HYBRID|SEMANTIC, implicitFilterConfiguration,
   QUERY_DECOMPOSITION, rerankingConfiguration (amazon.rerank-v1:0,
   cohere.rerank-v3-5:0), Guardrails contextual grounding
```

**Key techniques, from AWS's official sample repository** [33]:

- **Chunking strategies** [34]. The choices are `FIXED_SIZE`, `HIERARCHICAL`, `SEMANTIC` and `NONE`, or a custom Lambda. Example settings: fixed at 300 tokens with 20% overlap; hierarchical with 1,500-token parents, 300-token children and 60-token overlap; semantic with a 300-token maximum, buffer size 1 and a 95th-percentile breakpoint threshold. The chunking strategy cannot be changed after the data source is created. The repository also shows Anthropic-style contextual retrieval implemented as a custom Lambda chunker [33].
- **Two query APIs.** `Retrieve` returns chunks with S3 location, score and metadata. `RetrieveAndGenerate` also generates a cited answer [34][42].
- **Query decomposition** [35]. `orchestrationConfiguration.queryTransformationConfiguration.type = QUERY_DECOMPOSITION` splits a multi-part question into sub-queries that are retrieved separately, then pooled and ranked. The sample shows a two-intent question about a building's location and a whistleblower scandal. Without decomposition, the query embedding is dominated by the scandal, and decomposition recovers the chunks about the building.
- **Reranking** with Bedrock-hosted reranker models overrides the default ranking. Retrieving fewer but better chunks also cuts generation cost and latency [36].
- **Autogenerated metadata filters** (`implicitFilterConfiguration`). An FM reads the query against a declared metadata schema and emits filter expressions [37].
- **GraphRAG** [38]. Choosing Neptune Analytics as the store makes KB build and maintain a graph of entities and structural elements (such as section titles) alongside the embeddings. It is queried through the same `Retrieve`/`RetrieveAndGenerate` APIs.
- **Structured RAG** [39]. Natural-language questions over Amazon Redshift are turned into SQL using the schema, query history and supplied context, "without the need to move or preprocess the data".
- **Storage and modality options.** S3 Vectors is available as a low-cost vector store [40]. Multimodal KBs with Amazon Nova Multimodal Embeddings put text, documents, images, video and audio in one vector space and support image-to-content search [41]. A Kendra GenAI Enterprise index can act as a managed hybrid-plus-rerank retriever; the sample warns that index creation can take up to 40 minutes [42].
- **Guardrails contextual grounding** blocks responses whose *grounding* or *relevance* score falls below configurable thresholds [43].

**Failure modes and operational notes.**

- Chunking is fixed at data-source creation, so experimenting with chunking means re-creating the source [34].
- Ingestion jobs serialise per KB [38].
- Embedding a multi-intent query dilutes its semantics. That is the motivation for decomposition [35].
- Grounding thresholds trade false blocks against hallucination risk and must be tuned per domain [43].

**Lesson.** Bedrock KB has become a *menu of retrieval backends* (vector, graph, SQL, Kendra, S3 Vectors) behind one retrieval API. That is useful for portability across backends, but evaluating each combination is left to you. AWS ships a KB evaluation-job notebook for this purpose [33].

### 15.3.2 Google: Vertex AI RAG Engine and Vertex AI Search

**Positioning.** Google offers two layers:

- **Vertex AI Search** (Discovery Engine): a Google-quality managed search engine with connectors.
- **RAG Engine**: a managed RAG orchestration layer that can use Vertex AI Search *or* a vector database as its backend.

RAG Engine became generally available on January 10, 2025 [44]. By 2026 the sample notebooks call it "Agent Platform RAG Engine" under the Gemini Enterprise Agent Platform branding [45].

**RAG Engine architecture.** The documentation lists six stages: ingestion, transformation (chunking), embedding, indexing into a **corpus**, retrieval and generation [45].

- **Sources:** Cloud Storage, Google Drive, Jira, Slack, local files and websites, among others [44].
- **Chunking:** configurable per import. The samples use `chunk_size=1024, chunk_overlap=100` and `512/50`, with `max_embedding_requests_per_min` throttling [45].
- **Backends:** RAG Engine's managed vector storage, Vertex AI Vector Search, Pinecone, Weaviate, Feature Store, or a Vertex AI Search engine. The last is presented as the option for "low-latency retrieval at scale" [44][46].
- **Retrieval:** `similarity_top_k`, `vector_distance_threshold` and an **LLM ranker** (`rag.Ranking(llm_ranker=...)`) [44][45]. A sample comment notes that only one corpus could be queried per call at that time [46].
- **Generation:** retrieval is attached to Gemini as a `Tool.from_retrieval(...)`. Alternatively, `retrieve_contexts` returns raw contexts for use with any model [45].

```
 GCS / Drive / Slack / Jira / web --import_files(chunking cfg)--> RagCorpus
                                                                   |
     +---------------------------+---------------------------------+
     | backend: managed store | Vector Search | Pinecone | Weaviate |
     |          Feature Store | Vertex AI Search engine            |
     +---------------------------+---------------------------------+
                                 |
  Gemini generate_content(tools=[Retrieval(VertexRagStore(top_k, threshold,
                                 llm_ranker))])  or  retrieve_contexts -> any LLM
```

**Grounding stack.** Google's 2024 grounding announcement made a set of RAG building-block APIs generally available: document parsing, embedding generation, semantic ranking, grounded answer generation and a fact-checking **check-grounding** service. It also introduced:

- **dynamic retrieval**, which decides per query whether to ground in Google Search;
- an experimental **high-fidelity mode**, a fine-tuned Gemini 1.5 Flash that answers only from supplied context, with per-sentence sources and confidence scores;
- hybrid (vector plus keyword) Vector Search in preview.

Vertex AI Vector Search is described as scaling to billions of vectors with nearest-neighbour lookups in "a few milliseconds" [48].

**Evaluation guidance.** Google's RAG Engine evaluation notebook tunes chunk size, overlap and top-k on BEIR-FiQA using recall@k, precision@k and nDCG@k. Its heuristics: shrink chunks or raise top-k to improve recall, and start chunk overlap at about 20% of chunk size [47].

**Lesson.** Google separates "search engine as backend" from "RAG orchestration" more cleanly than its peers. A team can begin on a managed vector store and later move the same corpus API onto Vertex AI Search for enterprise-scale relevance without changing generation code [44][46].

### 15.3.3 Azure AI Search: from hybrid + semantic ranker to agentic retrieval

**Classic pipeline.** Azure AI Search is the reference design for the "hybrid plus L2 rerank" stack.

- **Hybrid fusion.** BM25 and one or more vector queries run in parallel and are merged with **Reciprocal Rank Fusion**. Each document scores `1/(rank + k)` per list, with k defaulting to a small constant "such as 60". Vector-query weights scale each list's contribution [53].
- **Semantic ranker.** A multilingual cross-encoder "adapted from Microsoft Bing" reranks only the **top 50** fused results [52]:
  - Inputs come from the title (128 tokens), keywords (128) and content fields.
  - Since November 2024, summary strings of up to 2,048 tokens per document are scored (previously 256).
  - It produces a `@search.rerankerScore` from 0 to 4 and returns verbatim captions and answers. No generative model composes text in this step.
  - A preview query-rewrite feature creates up to 10 query variants [52].
- **Integrated vectorization.** Indexers crack documents, chunk them (Text Split skill, which is free, or Content Understanding / layout-based semantic chunking) and embed them during indexing. *Index projections* write chunk-level secondary indexes linked to parent documents [54].

**Agentic retrieval (knowledge bases).** First previewed in 2025 and now the engine behind **Foundry IQ** [49]:

1. The app calls a **knowledge base** `retrieve` action with the query and conversation history.
2. At `low` or `medium` *retrieval reasoning effort*, an Azure OpenAI model **plans** focused subqueries and selects **knowledge sources**. At `minimal`, planning is skipped and all sources are queried directly.
3. Subqueries run **in parallel** (keyword, vector or hybrid). Each result list is **semantically reranked**, and references are kept for citation.
4. Results are **merged** into a unified grounding payload, with an optional activity log and optional **answer synthesis** (preview) that produces citation-backed answers [49].

```
 query + chat history
        |
        v
 +---------------- knowledge base (retrievalReasoningEffort) ----------------+
 | minimal: no LLM | low: 1 planning pass | medium: + semantic classifier    |
 |                 |                      |   -> 1 follow-up iteration       |
 | auto: light pass first, escalate up to medium only if grounding is weak   |
 +-------------------+-------------------------------+-----------------------+
                     | parallel subqueries           |
        +------------+------------+------------------+-------------+
        v            v            v                  v             v
   search index   Blob/OneLake  SharePoint(remote)  Web    MCP server, Work IQ,
   (hybrid+L2)    (indexed)     (user-delegated)          Fabric, Azure SQL
        \____________|____________|__________________|_____________/
                                  v
        merged grounding docs + references (+ activity log, + answer synthesis)
```

The **medium** level adds a high-precision semantic classifier. It judges whether the first results are good enough and, if not, runs one more iteration with a revised plan. **Auto** (2026-08-01-preview) starts with a lightweight pass and escalates only when needed [50].

By 2026 knowledge-source types included search indexes, Blob, OneLake, indexed and remote SharePoint, web, Azure SQL, Fabric Data Agent and Fabric Ontology, Work IQ (Microsoft 365) and arbitrary **MCP servers** [55]. Recent preview additions [55]:

- Purview sensitivity labels in responses
- freshness-aware ranking
- query hints that let the planner generate filters and boosts
- per-source rerank bypass
- server-sent-event streaming of plan and source activity
- `citationUrl` lookups

Knowledge bases and some source types became GA in REST API `2026-04-01`, for extractive retrieval at `minimal` effort. Planning, answer synthesis and higher efforts remain in preview [49][55].

**Limits and cost (published)** [49][51]:

- At most **10 knowledge sources per knowledge base** on S1 and above.
- Retrieve runtime defaults to **90 s** (minimum 10 s, maximum 600 s).
- Billing is **token-based** for agentic retrieval, compared with per-query for the classic semantic ranker.

Microsoft's worked example assumes:

- 2,000 retrievals, each with 3 subqueries;
- 50 chunks reranked per subquery, at 500 tokens per chunk, giving 150M reranking tokens;
- planning with gpt-4o-mini at 2,000 input and 350 output tokens per retrieval.

That works out to about $3.30 for reranking plus $1.02 for planning, roughly **$4.32 in total**. Microsoft labels the per-token rate in the example as hypothetical [49].

**Failure modes and lessons.**

- Agentic retrieval "adds latency compared to a single-query pipeline" [49].
- Cost grows with fan-out (sources multiplied by subqueries multiplied by reranked chunks). Microsoft's own tips are to consolidate indexes, lower reasoning effort and curate summary content [49].
- Reranker score distributions can drift with model updates, so do not set thresholds "too granular" [52].
- Design lesson: Azure exposes the agentic-retrieval *dial* (minimal, low, medium, auto) as configuration. That makes the cost and latency versus completeness trade-off explicit and tunable per request [50].

### 15.3.4 OpenAI file search and vector stores

**Architecture.** OpenAI's hosted RAG has two parts:

- **Vector stores**, which receive uploaded files that OpenAI parses, chunks, embeds and indexes.
- The **`file_search` tool** in the Responses API. It searches one or more vector stores and returns an answer with file-citation annotations. It was originally an Assistants API feature, and the Responses API version adds metadata filtering [56].

There is also a standalone `vector_stores.search` endpoint for retrieval without generation [56][57].

**Parameters, from the OpenAPI specification** [57]:

- **Chunking.** `static` chunking with `max_chunk_size_tokens` between 100 and 4,096 (default **800**) and `chunk_overlap_tokens` (default **400**), which must not exceed half the chunk size.
- **Ingestion.** `file_batches` accepts up to **2,000 files** per batch, with optional per-file `attributes` and chunking.
- **Metadata.** Each file can carry up to 16 key-value `attributes`, which back comparison and compound `filters` at query time.
- **Results.** `max_num_results` ranges from 1 to 50. The tool's default for gpt-4-class models is 20.
- **Ranking.** `ranking_options.ranker` is `auto` or a pinned `default_2024_08_21`. `score_threshold` runs from 0 to 1, where higher values return "only the most relevant results, but may return fewer". `hybrid_search.embedding_weight` and `text_weight` weight the RRF fusion of dense and sparse keyword matches.
- **Search endpoint options.** The search endpoint adds `rewrite_query` (default false) and accepts multiple queries.
- **Lifecycle.** `expires_after` policies control vector-store lifetime.

**Measured behaviour.** In OpenAI's cookbook, file search over 21 OpenAI blog PDFs with LLM-generated questions (gpt-4o-mini, k=5, `tool_choice="required"`) scored **Recall@5 0.905, MRR 0.905, MAP 0.895** [56]. The misses were telling. A generic question ("What is OpenAI's mission as stated in the document?") matched a different document, and another question resolved to a more on-topic file than the one it was generated from. The cookbook itself says synthetic questions are "imperfect" and recommends human-verified evaluation sets [56].

**Lessons.**

- File search is the lowest-effort path to hosted hybrid RAG with citations. The levers are coarse, though: chunk size and overlap, the threshold, hybrid weights and filters.
- Parsing is opaque, so PDF quality problems surface as retrieval misses.
- Pin `ranker` when you need reproducible evaluations across platform updates [57].

### 15.3.5 Anthropic Citations API and search result blocks

**Problem addressed.** Prompt-based citation ("quote your sources in brackets") is unreliable. Models fabricate or misattribute quotes, and output tokens spent on quotes cost money. Anthropic's Citations API moves citation into the API contract. It is generally available on the Anthropic API and Google Cloud's Vertex AI. The announcement page on claude.com is dated June 23, 2025, with a June 30, 2025 update adding Amazon Bedrock [58]. The current documentation lists GA on the Claude API, Bedrock, Google Cloud and Microsoft Foundry [59]. It handles **L4 only**: retrieval stays with you.

**Mechanics** [59]:

- **Documents** are passed as `document` content blocks with `citations: {enabled: true}`:
  - *Plain text* is chunked into sentences and cited by character index (`char_location`).
  - *PDF* text is extracted and sentence-chunked, and cited by page (`page_location`).
  - *Custom content* uses your own blocks with no further chunking, and is cited by block index (`content_block_location`). Use it for bullet lists, transcripts or pre-chunked RAG passages.
- **Responses** interleave text blocks with citation objects that point to exact spans. The API guarantees that citations contain valid pointers to the supplied documents. The `cited_text` field does *not* count toward output tokens, and is not counted as input when passed back in later turns.
- **Search result blocks** (`type: "search_result"` with `source`, `title` and `content`) let custom RAG tools return results that Claude cites the same way it cites web search. They can be returned from tool calls (agentic RAG) or placed top-level in the user turn (pre-fetched RAG) [60].
- **Prompt caching** works on the source documents, but citation blocks themselves are not cached [59].

**Reported results.**

- Anthropic's internal evaluations found built-in citations "outperform most custom implementations", with recall accuracy up to 15% higher.
- Endex reported that source hallucinations and formatting issues fell "from 10% to 0%", with 20% more references per response.
- Thomson Reuters uses the feature in CoCounsel [58].

**Limitations** [59]:

- Citations are **incompatible with structured outputs**. Enabling both returns a 400 error, because interleaved citation blocks cannot satisfy a strict JSON schema.
- Image citations are not supported, so scanned PDFs without extractable text cannot be cited.
- Enabling citations adds a small number of input tokens for system-prompt additions and chunking.

**Design lesson.** Citation granularity is set by how you package context. Put each retrieved chunk in its own plain-text document if you want sentence-level citations inside chunks. Use custom-content documents if each chunk should be cited as a whole [59].

## 15.4 Comparison tables

### 15.4.1 Open-source frameworks

| Framework | Primary abstraction | RAG composition style | Agentic support | Distinctive technique | Hosted companion |
|---|---|---|---|---|---|
| LangChain v1 / LangGraph | `create_agent` + middleware; graph runtime | Retrieval as tool; legacy chains in `langchain-classic` | Native; Deep Agents adds planning, sub-agents, FS | Offload-and-delegate; rubric grading [4] | LangSmith (eval, deployment) |
| LlamaIndex | Documents/Nodes, indexes, query engines, Workflows | Retrieval, postprocess, synthesis; cached ingestion | Event-driven Workflows | PropertyGraphIndex; ingestion caching [6][10] | LlamaParse (parse, extract, index) |
| Haystack | Typed components in directed multigraphs | Explicit pipelines, SuperComponents | Agent is a component; PipelineTool | Bounded loops (`max_runs_per_component`) [13] | deepset Enterprise Platform |
| DSPy | Signatures, modules, optimizers | Python program compiled to a metric | ReAct-style modules | MIPROv2, GEPA, BootstrapFinetune [19] | none |
| SK / Agent Framework | Kernel, plugins; agents + context providers | `TextSearchProvider` pre-invoke or on-demand | Agents, workflows, harness agent | One flag switches 2-step vs agentic [24] | Microsoft Foundry |
| Vercel AI SDK | `streamText`/agents, `embed`, `rerank` | Tools over any DB (e.g. pgvector) | `ToolLoopAgent` (20-step default) | Frontend-native streaming of tool calls [27] | Vercel AI Gateway |
| RAGFlow | End-to-end engine + UI + API | Template chunking, hybrid + rerank | Agentic retrieval, thinking modes | DeepDoc; Knowledge Compilation [30] | self-hosted (Docker) |

### 15.4.2 Managed services

| Service | Ingest & chunking control | Index backends | Retrieval features | Agentic retrieval | Citations / grounding |
|---|---|---|---|---|---|
| Bedrock Knowledge Bases | FM or BDA parsing; fixed, hierarchical, semantic, none, Lambda | OpenSearch, Aurora, Pinecone, Redis, MongoDB, Neptune (graph), S3 Vectors, Kendra, Redshift (SQL) | Hybrid, rerank models, auto metadata filters | Query decomposition | RetrieveAndGenerate citations; Guardrails grounding |
| Vertex AI RAG Engine / Search | Chunk size and overlap per import | Managed store, Vector Search, Pinecone, Weaviate, Feature Store, Vertex AI Search | top-k, distance threshold, LLM ranker | Via Gemini tool use | Grounding APIs, check-grounding [48] |
| Azure AI Search | Integrated vectorization; Text Split, Content Understanding | Search index; remote SharePoint, web, MCP, Fabric and others | BM25 + vector RRF (k=60), semantic ranker on top 50 | Knowledge bases: minimal, low, medium, auto | Answer synthesis (preview), references, `citationUrl` |
| OpenAI file search | Auto or static (800/400 default, up to 4,096) | Hosted vector stores | Hybrid RRF weights, score threshold, filters, 1-50 results | Model-driven tool calls | File-citation annotations |
| Anthropic Citations | You chunk (sentence auto-chunking or custom blocks) | None (bring your own) | None | Search-result blocks from tools | Span-level citations; `cited_text` not billed as output |

## 15.5 Cross-cutting patterns and failure modes

1. **"Retrieval as a tool" became the default.** LangChain v1 [3], Agent Framework's on-demand mode [24], the AI SDK guide [27], Anthropic's search-result blocks [60] and Gemini's retrieval tool [45] all model retrieval as a function the LLM calls. Fixed 2-step pipelines remain the predictable, cheaper option [2][14].
2. **Agentic retrieval is moving server-side.** Azure's reasoning-effort levels [50], Bedrock's query decomposition [35], RAGFlow's thinking modes [30] and Vertex's LLM ranker [44] put query planning and iteration inside the retrieval service. Developers lose some visibility. Azure partly compensates with activity logs and SSE streaming of plans [49][55].
3. **Hybrid retrieval plus a cross-encoder rerank is the consensus first stage.** Azure documents RRF with k=60 and the top-50 semantic rerank [52][53]. OpenAI exposes RRF weights [57]. RAGFlow exposes a vector/keyword weight [32]. Bedrock exposes `HYBRID` search and rerankers [33][36]. The AI SDK and Haystack provide rerank and hybrid primitives [15][28].
4. **Thresholds are fragile.** Similarity and reranker thresholds appear everywhere: OpenAI `score_threshold`, Vertex `vector_distance_threshold`, RAGFlow's 0.2 default, the AI SDK's 0.5. Azure warns that score distributions shift with model updates [52]. Tune thresholds against an evaluation set and re-check them after platform updates.
5. **Parsing and chunking decisions are sticky.** Bedrock's chunking cannot change after data-source creation [34]. RAGFlow's v1 migration is irreversible [31]. Plan re-indexing as a routine operation.
6. **Graph structures are being productised and then generalised.** Bedrock's Neptune GraphRAG [38], LlamaIndex's PropertyGraphIndex [10], Agent Framework's Neo4j provider [24] and RAGFlow's move from GraphRAG/RAPTOR to Knowledge Compilation [31] all show this. Graph-build cost is the recurring pain point.
7. **Cost scales with fan-out.** In agentic retrieval, reranking tokens dominate. In Microsoft's example, $3.30 of the $4.32 total is reranking [49]. In DSPy, the optimisation itself costs dollars per run [20][21]. Budget for the multiplicative fan-out factors (sources x subqueries x reranked chunks) [49].
8. **Citations should be structural, not prompted.** Anthropic's API-level citations [59], Azure references and `citationUrl` [55], OpenAI annotations [56] and Bedrock RetrieveAndGenerate citations [34] replace prompt-format conventions. Beware incompatibilities such as citations plus structured outputs [59].

## Key takeaways for system designers

- **Decide which layers to own.** Use a managed service when connectors, ACL-aware indexing and ranking quality matter more than control. Use a framework when you need custom retrieval logic, portability or offline evaluation. Many mature stacks combine the two: a managed retriever exposed as a tool inside a framework agent (for example, Agent Framework's Azure AI Search provider [24], or Bedrock KB sample notebooks using LangChain [33]).
- **Start 2-step, earn agentic.** Begin with a fixed retrieve-rerank-generate pipeline with bounded cost. Add query decomposition, planning or iterative retrieval only for query classes where evaluation shows gains. Prefer configuration dials (Azure `minimal`/`low`/`medium`/`auto` [50], Agent Framework `SearchTime` [24]) and bounded loops (Haystack's `max_runs_per_component` [13], the AI SDK's `stopWhen` [29]) over open-ended agents.
- **Make hybrid + rerank your baseline.** RRF fusion (k of about 60) of BM25 and dense retrieval, followed by a cross-encoder over the top 50 or so candidates, is what Azure, OpenAI, Bedrock and RAGFlow converged on [32][36][52][53][57].
- **Treat chunking and parsing as versioned, re-runnable artefacts.** Cache ingestion (LlamaIndex [6]), keep raw sources, and expect to rebuild indexes when you change chunking or embeddings [31][34].
- **Optimise prompts against metrics, not intuition.** DSPy shows sizeable gains for single-digit dollar costs (about 42% to 61% semantic F1; about 30% to almost 60% multi-hop recall), but only with held-out validation to catch overfitting [20][21].
- **Get citations from the API contract.** Use structured citation features where they exist, and decide citation granularity when you package context [59][60].
- **Instrument fan-out.** Log subqueries, sources hit, chunks reranked and tokens per stage. Azure's activity log [49] and LangSmith traces [4] are examples. In agentic retrieval, cost and latency are emergent properties of the plan.
- **Expect churn.** Within roughly 18 months, LangChain moved retrievers into a legacy package [3], Semantic Kernel became Agent Framework [22], LlamaCloud was renamed LlamaParse [8], RAGFlow was rewritten in Go [31], and Vertex AI was rebranded under the Gemini Enterprise Agent Platform [45]. Isolate vendor APIs behind your own retrieval interface.

## References

1. LangChain repository README. LangChain, Inc. 2026. https://github.com/langchain-ai/langchain
2. "Retrieval" (LangChain OSS docs source, incl. 2-Step/Agentic/Hybrid RAG table). LangChain, Inc. 2026. https://github.com/langchain-ai/docs/blob/main/src/oss/deepagents/retrieval.mdx
3. "What's new in LangChain v1" (release notes; `create_agent`, middleware, `langchain-classic`). LangChain, Inc. 2025-2026. https://github.com/langchain-ai/docs/blob/main/src/oss/python/releases/langchain-v1.mdx
4. "Retrieval Augmented Generation (RAG) with Deep Agents". LangChain, Inc. 2026. https://github.com/langchain-ai/docs/blob/main/src/oss/deepagents/rag.mdx
5. LlamaIndex repository README. LlamaIndex (run-llama). 2026. https://github.com/run-llama/llama_index
6. "Ingestion Pipeline" (module guide). LlamaIndex. 2026. https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/loading/ingestion_pipeline/index.md
7. "Querying" (Understanding RAG). LlamaIndex. 2026. https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/understanding/rag/querying/index.mdx
8. "LlamaParse (formerly LlamaCloud)". LlamaIndex. 2026. https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/llama_cloud/index.md
9. LlamaIndex Workflows (`workflows-py`) repository README. LlamaIndex. 2026. https://github.com/run-llama/workflows-py
10. LlamaIndex API reference: property graph index, retrievers and node parsers. LlamaIndex. 2026. https://github.com/run-llama/llama_index/tree/main/docs/api_reference/api_reference
11. Haystack repository README. deepset. 2026. https://github.com/deepset-ai/haystack
12. "Pipelines" and "Components" (Haystack concepts docs). deepset. 2026. https://github.com/deepset-ai/haystack/blob/main/docs-website/docs/concepts/pipelines.mdx
13. "Pipeline Loops". deepset. 2026. https://github.com/deepset-ai/haystack/blob/main/docs-website/docs/concepts/pipelines/pipeline-loops.mdx
14. "Choosing Between Pipelines and Agents". deepset. 2026. https://github.com/deepset-ai/haystack/blob/main/docs-website/docs/concepts/agents/choosing-between-pipelines-and-agents.mdx
15. "SuperComponents". deepset. 2026. https://github.com/deepset-ai/haystack/blob/main/docs-website/docs/concepts/components/supercomponents.mdx
16. Khattab, O. et al. "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines". Stanford NLP. 2023. https://arxiv.org/abs/2310.03714
17. Opsahl-Ong, K. et al. "Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs" (MIPRO). Stanford NLP. 2024. https://arxiv.org/abs/2406.11695
18. Agrawal, L. A. et al. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning". 2025. https://arxiv.org/abs/2507.19457
19. "DSPy Optimizers" (learn/optimization). Stanford NLP. 2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
20. "Tutorial: Retrieval-Augmented Generation (RAG)". DSPy. 2025-2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/rag/index.ipynb
21. "Tutorial: Multi-Hop Search". DSPy. 2025-2026. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/multihop_search/index.ipynb
22. Semantic Kernel repository README. Microsoft. 2026. https://github.com/microsoft/semantic-kernel
23. "What are Semantic Kernel Vector Stores?". Microsoft. 2024-2026. https://github.com/MicrosoftDocs/semantic-kernel-docs/blob/main/semantic-kernel/concepts/vector-store-connectors/index.md
24. "RAG" (Microsoft Agent Framework; TextSearchProvider, context providers). Microsoft. 2026. https://github.com/MicrosoftDocs/semantic-kernel-docs/blob/main/agent-framework/agents/rag.md
25. "Microsoft Agent Framework Overview". Microsoft. 2026. https://github.com/MicrosoftDocs/semantic-kernel-docs/blob/main/agent-framework/overview/index.md
26. Microsoft Agent Framework repository README. Microsoft. 2026. https://github.com/microsoft/agent-framework
27. "RAG Agent Guide" (AI SDK cookbook). Vercel. 2025-2026. https://github.com/vercel/ai/blob/main/content/cookbook/00-guides/01-rag-chatbot.mdx
28. "Reranking" (AI SDK Core docs). Vercel. 2026. https://github.com/vercel/ai/blob/main/content/docs/03-ai-sdk-core/31-reranking.mdx
29. "Loop Control" (AI SDK agents docs). Vercel. 2026. https://github.com/vercel/ai/blob/main/content/docs/03-agents/04-loop-control.mdx
30. RAGFlow repository README. InfiniFlow. 2026. https://github.com/infiniflow/ragflow
31. RAGFlow release notes (v0.21 through v1.0.0-rc1). InfiniFlow. 2025-2026. https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md
32. "Retrieval testing" (RAGFlow dataset guide). InfiniFlow. 2026. https://github.com/infiniflow/ragflow/blob/main/docs/guides/dataset/retrieval_testing.md
33. Amazon Bedrock Knowledge Bases feature examples (sample repository). AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/tree/main/rag/knowledge-bases/features-examples
34. "Advanced chunking options" notebook. AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/03-optimizing-accuracy-retrieved-results/advanced_chunking_options.ipynb
35. "Query Reformulation Supported by Amazon Bedrock Knowledge Bases" notebook. AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/03-optimizing-accuracy-retrieved-results/query_reformulation.ipynb
36. "Re-ranking" notebook. AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/03-optimizing-accuracy-retrieved-results/re_ranking_using_kb.ipynb
37. "Autogenerated filters" notebook. AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/03-optimizing-accuracy-retrieved-results/autogenerated_metadata_filters.ipynb
38. "Graph RAG using Amazon Bedrock Knowledge Bases" notebook. AWS. 2025-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/08-graph-rag/01-graph-rag-using-kb.ipynb
39. "Structured RAG using Amazon Bedrock Knowledge Bases (Redshift)" notebook. AWS. 2025-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/07-structured-rag/structured_rag_using_kb.ipynb
40. "Amazon Bedrock Knowledge Bases with S3 Vectors" notebook. AWS. 2025-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/10-s3-vectors/knowledge-base-s3-vector.ipynb
41. "Amazon Bedrock Multimodal Knowledge Bases with S3 Vectors" (Nova Multimodal Embeddings) notebook. AWS. 2025-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/06-multi-modal-rag/nova-mme-multimodal-rag/multimodal-knowledge-bases.ipynb
42. "Amazon Bedrock Knowledge base with Kendra GenAI index" notebook. AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/09-managed-index/kb_with_kendra_genai_Index.ipynb
43. "Combining Guardrails Contextual Grounding and Knowledge Bases" notebook. AWS. 2024-2026. https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/05-responsible-ai/contextual_grounding.ipynb
44. Velez, C. and Liu, L. "Introducing Vertex AI RAG Engine". Google Cloud Blog. 2025. https://cloud.google.com/blog/products/ai-machine-learning/introducing-vertex-ai-rag-engine
45. Skinner, H. "Intro to Building a Scalable and Modular RAG System with RAG Engine". Google Cloud (generative-ai repo). 2025-2026. https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb
46. Dorozhkin, A. "Vertex AI RAG Engine with Vertex AI Search". Google Cloud. 2025. https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_vertex_ai_search.ipynb
47. Tsoi, E. "Vertex RAG Engine Retrieval Quality Evaluation and Hyperparameters Tuning". Google Cloud. 2025. https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/rag_engine_evaluation.ipynb
48. Gokturk, B. "Google Cloud expands grounding capabilities on Vertex AI". Google Cloud Blog. 2024. https://cloud.google.com/blog/products/ai-machine-learning/rag-and-grounding-on-vertex-ai
49. "Agentic retrieval in Azure AI Search" (overview, architecture, billing example). Microsoft. 2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/agentic-retrieval-overview.md
50. "Set the retrieval reasoning effort (preview)". Microsoft. 2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/agentic-retrieval-how-to-set-retrieval-reasoning-effort.md
51. "Service limits in Azure AI Search" (agentic retrieval limits). Microsoft. 2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/search-limits-quotas-capacity.md
52. "Semantic ranking in Azure AI Search". Microsoft. 2024-2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/semantic-search-overview.md
53. "Relevance scoring in hybrid search using Reciprocal Rank Fusion (RRF)". Microsoft. 2024-2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/hybrid-search-ranking.md
54. "Integrated data chunking and embedding in Azure AI Search". Microsoft. 2024-2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/vector-search-integrated-vectorization.md
55. "What's new in Azure AI Search". Microsoft. 2026. https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/search/whats-new.md
56. "Using file search tool in the Responses API" (cookbook). OpenAI. 2025. https://github.com/openai/openai-cookbook/blob/main/examples/File_Search_Responses.ipynb
57. OpenAI OpenAPI specification (vector stores, file_search, ranking options). OpenAI. 2026. https://github.com/openai/openai-openapi
58. "Introducing Citations on the Anthropic API". Anthropic. 2025. https://claude.com/blog/introducing-citations-api
59. "Citations" (Claude API documentation). Anthropic. 2026. https://platform.claude.com/docs/en/build-with-claude/citations
60. "Search results" (Claude API documentation). Anthropic. 2026. https://platform.claude.com/docs/en/build-with-claude/search-results
