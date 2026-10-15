# 22. Master Index of RAG Engineering Blogs, Talks and Guides

This chapter is the atlas's bibliography in working form. It lists the engineering blogs, conference talks, vendor guides, practitioner essays and papers that people building retrieval-augmented generation (RAG) systems actually cite, grouped by company and by topic. Each entry has a one-line summary and a numbered reference that carries the full URL. The second half is a flat annotated index of 240+ entries.

The first half reads the most informative primary sources as one body of evidence. It covers the problem each team set out to solve, the architecture they shipped, the numbers they published, and the failures they admitted. The pattern that emerges is consistent across companies. Production RAG wins came mostly from retrieval and data preparation, not from the generator. Systems moved from single-shot "embed, top-k, stuff" pipelines toward agentic retrieval and structured indexes (graphs, Merkle-synced code indexes, page images). Evaluation and guardrails became first-class components rather than afterthoughts.

Method and caveats. The entries were collected through web searches and direct reads of primary sources in October 2026. Many company blogs could not be fetched directly from the research environment. For those posts the facts below come from search-result extracts of the primary page or from clearly labeled secondary summaries; the ZenML LLMOps database is the most common one. Numbers that come from secondary sources are marked "(secondary)". Numbers that could not be checked against the original are marked approximate. Coverage is broad but not complete. A second research pass (October 2026) added text-to-SQL and data-agent posts (Uber, Swiggy), more DoorDash, LinkedIn, Dropbox, Instacart and Airbnb posts, Cursor's semantic-search study, Google's "sufficient context" work, and more vendor and framework guides (entries 172-235). A third editorial pass (October 2026) checked headline numbers against the Anthropic and Microsoft Research primary posts, corrected several of them, and added entries 236-242 on tool protocols, agent tool design, GraphRAG 1.0, HippoRAG 2 and reasoning-model search agents. Several well-known company posts (for example from Spotify, Netflix, Notion, Slack and Shopify) still did not show up in searches. They are left out rather than cited from memory.

---

## 22.1 How to use this index

The index is organized along two axes:

```
                        WHO PUBLISHED IT
          +-------------+-------------+-------------+-------------+
          | Product co. | Cloud/LLM   | Infra vendor| Practitioner|
          | eng. blogs  | providers   | (vector DB, | essays,     |
          | (Uber,      | (Anthropic, | frameworks) | courses,    |
          | DoorDash..) | MSR, AWS..) |             | talks       |
  +-------+-------------+-------------+-------------+-------------+
W | Ingest| Cursor,     | Contextual  | Jina late   | Applied LLMs|
H | /chunk| GitHub      | Retrieval   | chunking,   |             |
A |       |             |             | Vespa ColPali             |
T +-------+-------------+-------------+-------------+-------------+
  | Retr- | Pinterest,  | GraphRAG,   | Weaviate    | Eugene Yan, |
I | ieve  | Ramp,Harvey | LazyGraphRAG| hybrid,     | Hamel       |
T |       |             |             | Pinecone rr |             |
  +-------+-------------+-------------+-------------+-------------+
C | Gen / | DoorDash    | Citations,  | RAGAS, HF   | J. Liu      |
O | guard | guardrails  | BenchmarkQED| cookbook    | course      |
V | / eval|             |             |             |             |
E +-------+-------------+-------------+-------------+-------------+
R | Agent | Uber EAg-RAG| Multi-agent | LangChain   | Chip Huyen  |
S | -ic   |             | research,   | decon. RAG  |             |
  |       |             | ctx. eng.   |             |             |
  +-------+-------------+-------------+-------------+-------------+
```

To answer a design question, read across a row. For example, to decide on a chunking strategy, compare Anthropic's Contextual Retrieval [91], Jina's late chunking [117], LlamaIndex's decoupled retrieval and synthesis chunks [123], and Cursor's syntactic chunks [61]. To see how a single company's thinking changed over time, read down a column.

---

## 22.2 Deep reads: company engineering case studies

### 22.2.1 Uber: Genie and Enhanced Agentic RAG

**Problem.** On-call engineers at Uber asked roughly 45,000 questions a month in Slack help channels. The answers were spread across Engwiki, an internal Stack Overflow and other stores [44]. Uber chose RAG over fine-tuning mainly for time to market: fine-tuning needs curated, diverse examples and recurring compute to stay current [44].

**Architecture (v1, 2023-2024).** Internal sources go through ETL into chunks, then into embeddings and a vector store. A Slack bot retrieves the top chunks and generates an answer with citations, and users leave helpfulness feedback. According to a secondary summary, Genie reached 154 Slack channels, answered more than 70,000 questions with a 48.9% helpfulness rate, and saved about 13,000 engineering hours (approximate) [164].

**Architecture (v2, May 2025: EAg-RAG).** Pointed at more than 40 security and privacy policy documents stored as PDFs, the traditional pipeline "revealed significant gaps" [45]. The redesign added richer document processing: custom Google Docs loaders, and LLM-driven content formatting so that tables and structure survive parsing. It also added pre-processing agents (query optimization, source identification) and post-processing agents (context refinement) around a hybrid vector and BM25 retriever [45][165]. The reported result was a 27% relative increase in acceptable answers and a 60% relative reduction in incorrect advice (secondary) [165].

```
 Slack question
      |
      v
 +-----------------+    +--------------------+    +---------------------+
 | Query optimizer |--->| Source identifier  |--->| Hybrid retrieval    |
 | (agent)         |    | (agent: which docs)|    | vector + BM25 top-k |
 +-----------------+    +--------------------+    +----------+----------+
                                                             |
                     +---------------------------+           v
 answer + cites <----| Generator (LLM)           |<---| Post-processor     |
                     +---------------------------+    | (context refine)   |
                                                      +--------------------+
 Offline: PDF/GDoc loaders -> LLM formatting (tables, headings) -> chunks -> index
```

**Lesson.** The jump in quality came from document enrichment plus query-side agents, not from changing the generator. This agrees with the controlled comparison in "Is Agentic RAG worth it?" [37]: agentic retrieval helps on hard, policy-heavy corpora but costs latency and tokens.

### 22.2.2 DoorDash: RAG with a two-tier guardrail and LLM judge

**Problem.** DoorDash wanted to automate support for Dashers (delivery drivers) without hallucinated or policy-violating answers [46].

**Architecture.** The multi-turn chat is first condensed into a single issue statement. That summary drives retrieval over knowledge-base articles and past resolved cases, and an LLM then drafts the reply [46]. DoorDash first tried a more sophisticated guardrail model but found it too slow and too token-expensive. The shipped version has two tiers. The first is a cheap in-house semantic-similarity check between the response and the knowledge-base segments. Only responses that fail it go to an LLM evaluator. A separate LLM Judge monitors quality offline [46][151].

**Numbers.** Secondary sources cite a 90% reduction in hallucinations and a 99% reduction in compliance issues across thousands of daily requests [153][166]. These were not verified against the original post and should be treated as approximate. A 2025 follow-up extends the knowledge base with user-generated content, because a manually curated knowledge base does not scale [46].

**Lesson.** Run the expensive grounding check only on the tail. A cheap similarity filter in front of an LLM evaluator keeps guardrail latency close to retrieval latency.

### 22.2.3 LinkedIn: knowledge-graph RAG for support, and the GenAI product postmortem

LinkedIn's SIGIR 2024 paper argues that treating historical tickets as flat text loses both the structure inside each ticket and the links between tickets [40]. The paper builds a dual-layer knowledge graph. The intra-issue layer is a tree of ticket sections. The inter-issue layer links tickets through explicit metadata and through implicit links inferred from embeddings. Queries are parsed into entities and intents, sub-graphs are retrieved, and the answer is generated from them. Reported results: MRR improved by 77.6%, and median per-issue resolution time fell by 28.6% after deployment on LinkedIn's customer service team [40][171].

The companion "Musings on building a generative AI product" post [47] is one of the most widely quoted postmortems. According to secondhand accounts, the team reached about 80% of the target experience in roughly one month, then spent about four more months pushing past 95%, with each additional 1% "discouragingly" hard because of hallucinations. Earlier LinkedIn posts and a 2025 talk describe the platform that grew out of this work [48][49][150][167].

### 22.2.4 Pinterest: RAG for table selection in text-to-SQL

Pinterest's text-to-SQL feature lives in Querybook, its open-source SQL tool. The first version assumed users knew which tables to query. The second added RAG for table discovery: an LLM writes a summary of each table from its schema and sample queries, those summaries and historical queries are embedded into a vector index, and only "top-tier" tables are indexed so that low-quality datasets never reach the prompt [50][51][53]. At query time, the question is embedded and matched against the index. An LLM re-selects the best tables from the candidates, the user confirms them, and SQL is generated. Reported impact: 35% faster task completion and a first-shot acceptance rate that rose from about 20% to over 40% [50][52]. A later Pinterest write-up on unified context-intent embeddings for analytics tables (covered in Korean by [54]) and academic follow-ups such as CSR-RAG [38] continue the line.

### 22.2.5 Ramp and Grab: RAG as a constrained classifier and an analytics engine

Ramp's January 2025 post (by Ryne Carbone) rebuilds industry classification as RAG over the NAICS taxonomy [57]. Retrieval restricts the LLM to valid six-digit codes, which turns open generation into a multiple-choice question. A secondary summary describes two stages: optimized embeddings produce candidate codes, then a dual-prompt LLM makes the final selection. The same summary reports about 60% better recommendation retrieval and 5-15% better final accuracy (secondary) [168]. Ramp's engineering index also lists a separate post on transaction embeddings trained with triplet loss [58].

Grab's Integrity Analytics team paired an internal LLM tool (Spellvault) with a data middleware layer (Data-Arks) that exposes vetted SQL and API outputs as retrievable context. A secondary summary reports savings of 3-4 hours per report and fraud investigations shortened to minutes [55][56].

### 22.2.6 Harvey: vector-store selection for legal RAG

Harvey's February 2025 post describes three data scales [66][68]:

- Ad-hoc uploads of about 1-50 documents
- Project "vaults" of about 100-100,000 documents
- Third-party corpora of millions of legal documents, which a related talk puts at tens of millions [67]

Harvey runs two vector stores in parallel. Postgres with pgvector uses brute-force KNN, which gives high accuracy at small-to-medium scale. LanceDB uses an IVF-PQ index with tunable recall versus latency, and its decentralized storage can sit in per-customer cloud buckets, which matters for confidentiality and customer isolation [66]. The comparison table quotes LanceDB latency at 15M rows with metadata filtering [66].

**Lesson.** In regulated domains, tenancy and retention shape the index topology as much as recall does.

### 22.2.7 Code assistants: GitHub Copilot, Cursor, Sourcegraph

**GitHub (2025).** A new Copilot embedding model raised the average retrieval score from 0.362 to 0.498 (37.6%) and lifted code-acceptance ratios for C# users in VS Code by 110.7% [59]. InfoQ adds that index throughput doubled and memory dropped eightfold [60]. Training on "hard negatives" (code that looks right but is wrong) is reported by weaker sources [59][60].

**Cursor.** Cursor hashes every file and builds folder hashes from their children's hashes, forming a Merkle tree. Client and server compare trees and sync only the branches that differ. Changed files are split into syntactic chunks and embedded in the background, with embeddings cached by chunk content [61]. Third-party write-ups add that a new teammate's index can be bootstrapped by copying a similar existing index from the team namespace, and that the vector store is reported to be Turbopuffer [62][63]. These details are unverified.

**Cursor, semantic search for agents (November 2025) [192].** In an era of grep-capable agents, Cursor asked whether an embedding index still earns its keep. It trained a custom embedding model using agent sessions as training data, then compared agents that had grep only with agents that had grep plus semantic search. Search-result extracts of the post report:

- Question-answering accuracy rose by an average of 12.5% (6.5% to 23.5%, depending on the model).
- Code retention also rose, by a smaller 2.6% on large codebases (1,000+ files).

Lesson: lexical tools and dense retrieval work well together for agents, and the gain grows with repository size [192][193].

**Sourcegraph Cody.** Sourcegraph chose RAG over fine-tuning because code changes too quickly for a fine-tuned snapshot to stay current [64]. Context comes from three sources: keyword search with automatic query rewriting, the Sourcegraph search backend, and a code graph that follows structural relationships. Users can also pull context in explicitly with @-mentions [65].

```
      Cursor-style incremental code index
 local repo --hash--> Merkle tree --diff vs server--> changed files only
                                                          |
                                         syntactic (AST) chunking
                                                          |
                         content-hash cache hit? --yes--> reuse vector
                                    | no
                                    v
                           embed -> vector namespace (per repo / team)
```

### 22.2.8 Enterprise search: Glean and Dropbox Dash

Glean's posts describe a knowledge graph built from three pillars: content, people and activity. It is assembled per customer from 100+ connectors through continuous crawling rather than federated search [69][70]. The RAG layer runs plan, retrieve, generate. The planner learns to query Glean's own hybrid engine (semantic search, lexical search and the graph) and to rewrite queries with knowledge of the enterprise. Retrieval enforces permissions, so the LLM never has blanket access to company data [69][71]. The "Scholastic" semantic model is fine-tuned on each customer's corpus and click data [71]. Newer posts move to agentic reasoning built on the same retrieval layer [72].

Dropbox's engineering blog lists "How Dash uses context engineering for smarter AI" (November 2025), a December 2025 post on the feature store behind Dash, and a January 2026 post on knowledge graphs, MCP and DSPy in Dash [75]. A podcast with Dropbox's VP of Engineering covers the "context engine" in more depth [74]. The Dropbox posts themselves could not be fetched, so no numbers are reported here. The earlier "Building Dash" post [182] describes Dash as universal search plus knowledge management with content access control. A third-party summary describes its retrieval as a fast lexical candidate search followed by an embedding rerank, with documents sliced at query time rather than pre-chunked (secondary) [182].

### 22.2.9 Text-to-SQL and data agents: Uber QueryGPT, Swiggy Hermes

Text-to-SQL is RAG in which the "documents" are table schemas and example queries. The hard part is again the retrieval step, here choosing tables. Pinterest (22.2.4) and the two systems below converge on the same fix: scope retrieval to a business domain before searching.

- **Uber QueryGPT [172].** Uber's data platform serves about 1.2 million queries a month, and Operations issues about 36% of them (secondary). QueryGPT groups SQL samples and table schemas into "workspaces" by business domain (Mobility, Ads, Core Services and others). It then retrieves similar samples and schemas from a vector index within the selected workspace. The later Finch post [173] turns the same idea into a conversational financial data agent.
- **Swiggy Hermes [174][175][176].** V1 was a thin GPT-3.5 prompt. V2 moved to a RAG architecture on GPT-4o, partitioned into business-unit "charters", each with its own metadata and knowledge base. It answers several thousand queries for hundreds of users, with turnaround under 2 minutes. A V3 summary reports a conversational multi-turn analyst and query accuracy rising from 54% to 93% (secondary; approximate).

```
 question --> domain router --> [workspace / charter]
                                    |  table summaries + example SQL (vector index)
                                    v
                         candidate tables (top-N) --> LLM re-select (top-K)
                                    |                       |
                                    +------> user confirms --+--> SQL gen --> run/validate
```

**Lesson.** A domain partition does three jobs at once. It shrinks the retrieval search space, raises precision, and gives each team ownership of its metadata quality.

---

## 22.3 Deep reads: model-provider and research-lab blogs

### 22.3.1 Anthropic

**Contextual Retrieval (September 2024) [91].**
- Before indexing, Claude 3 Haiku reads the whole document and writes a 50-100 token context for each chunk. That context is prepended to the chunk before both embedding and BM25 indexing.
- Failure rate is measured as 1 minus recall@20. With the best embedding model tested (Gemini Text 004), the baseline failure rate was 5.7%. Contextual embeddings alone cut failures by 35% (to 3.7%). Adding contextual BM25 brought the cut to 49% (2.9%). Adding a reranker brought it to 67% (1.9%).
- With reranking, the top 150 chunks are retrieved and reranked down to 20. Twenty chunks beat 5 or 10.
- Prompt caching brings the one-time indexing cost to about $1.02 per million document tokens. The estimate assumes 8k-token documents, 800-token chunks, 50 tokens of instructions and 100 tokens of context per chunk.
- Cohere's reranker was used in the tests (Voyage's was mentioned but not tested).
- Secondary coverage [155][156], and a critique arguing that context is "not all you need" [157], are indexed below.

**Citations (2025) [92].** The API splits PDF and text sources into sentences (developers can also pass their own chunks) and returns grounded citations. Anthropic's internal evaluations show up to 15% better recall accuracy than most custom citation implementations; this is a vendor claim, not an independent benchmark. Endex reports that source hallucinations and formatting issues fell from 10% to 0%, with 20% more references per response.

**Building effective agents (December 2024) [95].** The post frames the "augmented LLM" (retrieval, tools, memory) as the basic building block of agentic systems. It notes that a single call with retrieval and in-context examples is often enough, so agentic complexity is not always needed.

**Multi-agent research system (June 2025) [94].** A lead agent plans and spawns parallel subagents that search and compress what they find. A citation agent then attributes each claim to a source. With Claude Opus 4 as lead and Claude Sonnet 4 subagents, this setup beat single-agent Claude Opus 4 by 90.2% on an internal research eval. The cost is tokens: single agents use about 4x the tokens of a chat interaction and the multi-agent system about 15x. Token usage alone explained about 80% of the variance on BrowseComp; with tool-call count and model choice it explained about 95%. The lead agent runs 3-5 subagents in parallel, and each subagent runs 3+ tools in parallel, which cut research time by up to 90% on complex queries. Admitted failure modes include:

- spawning 50 subagents for simple queries
- duplicated work from vague delegation
- a bias toward SEO-optimized content farms over authoritative sources such as academic PDFs, fixed by adding source-quality heuristics to the prompts

The post explicitly contrasts its adaptive multi-step search with static top-k RAG.

**Effective context engineering (September 2025) [93].** The post argues for "just-in-time" retrieval. The agent holds lightweight references (paths, queries, links) and loads content through tools such as glob and grep when it needs it. A hybrid approach preloads a little context (for example CLAUDE.md files) and explores the rest at runtime, which avoids stale indexes. The trade-off is that runtime exploration is slower than querying a pre-built index.

### 22.3.2 Microsoft Research: the GraphRAG family

| Post | Date | Core idea | Published numbers |
|---|---|---|---|
| GraphRAG intro [96] | Feb 2024 | LLM-built entity graph, hierarchical community summaries | Pairwise LLM-judge wins on VIINA news dataset |
| GraphRAG on GitHub [97] | Jul 2024 | Map-reduce over community summaries for global questions | ~70-80% win rate vs naive RAG; root summaries ~2-3% of tokens |
| Auto-tuning [98] | Sep 2024 | Auto-generate domain extraction prompts | Larger graph on podcast data |
| DRIFT search [99] | Oct 2024 | Primer -> follow-up local searches -> ranked hierarchy | Beats local search 78% (comprehensiveness), 81% (diversity) |
| Dynamic community selection [100] | Nov 2024 | Cheap LLM prunes irrelevant communities top-down | 77% average cost reduction vs static global search at level 1 (~470 vs ~1,500 reports); ~34% higher cost when searching to level 3 |
| LazyGraphRAG [101] | Nov 2024 | NLP noun-phrase graph, no LLM summaries at index time | Indexing ~0.1% of GraphRAG cost (same as vector RAG); global-search-level quality at >700x lower query cost |
| GraphRAG 1.0 [238] | Dec 2024 | Production release: incremental `update` indexing, default vector store | Parquet output -80%, total disk -43%, CLI startup 148 s to 2 s |
| BenchmarkQED [102] | Jun 2025 | AutoQ / AutoE / AutoD benchmarking suite | LazyGraphRAG won all 96 same-model (GPT-4o) comparisons, all but one significant; higher win rates than 1M-token vector RAG (GPT-4.1) |

```
 GraphRAG (eager)                         LazyGraphRAG (deferred)
 docs -> LLM entity/rel extraction        docs -> noun-phrase extraction (NLP)
      -> graph -> Leiden communities           -> concept graph -> communities
      -> LLM community summaries          query -> LLM expands to 3-5 subqueries
 query -> map over summaries                    -> best-first chunk ranking
       -> reduce -> answer                      -> LLM relevance tests (budget)
                                                -> map claims -> reduce answer
 cost paid at index time                  cost paid at query time, tunable
```

The arc of this series is instructive. Microsoft first showed that global "sensemaking" questions need structure [17][96]. It then spent a year driving down the cost of that structure [98][100][101], and finally built a benchmark that measures it without hand-written test sets [102]. The BenchmarkQED result is relevant to the long-context debate: even vector RAG with a 1M-token window (on GPT-4.1) lost to LazyGraphRAG across query classes, with only one comparison (answer relevance on local queries) short of statistical significance [102].

### 22.3.3 Databricks, AWS, Google, Microsoft Azure, OpenAI

- **Databricks [77][24].** More than 2,000 experiments on 13 LLMs (20 in the paper). Llama-3.1-405B performance starts to fall after 32k tokens of retrieved context and GPT-4-0125-preview after 64k. Failure modes vary by model: some answer incorrectly, some fail to follow instructions, and some refuse.
- **AWS Generative AI Innovation Center [78][79].** Part 1 covers text RAG: hybrid search, preprocessing that enriches context, query rewriting, and both quantitative and qualitative evaluation. Part 2 extends to tables, databases and images. AWS prescriptive guidance [80] and the Bedrock Knowledge Bases documentation [81] cover the managed path.
- **Azure [82][83]** and **Google Cloud [84][85][86]** publish reference architectures and design guides. **OpenAI's** accuracy guide frames RAG, prompting and fine-tuning as separate levers to tune [87], and its file-search documentation describes the managed retrieval tool [88]. The current Responses API documentation describes file search as a hosted tool that combines semantic and keyword search over uploaded files [235].
- **Google Research, "sufficient context" (ICLR 2025) [196][197].** This work separates RAG errors into two kinds: the retrieved context was insufficient, or the model failed despite sufficient context. Large models answer well when the context is sufficient, but they tend to guess rather than abstain when it is not. A selective-generation method that uses sufficiency signals improves the share of correct answers among the responses given by about 2-10% across Gemini, GPT and Gemma.
- **AWS Bedrock GraphRAG GA (March 2025) [198].** Managed GraphRAG on Neptune Analytics. It extracts entities and relationships alongside embeddings, and offers semantic, hierarchical, fixed-size or Lambda-based custom chunking. AWS's worked example costs about $0.48 per hour for Neptune Analytics.
- **Azure AI Search [200][201]** runs full-text and vector search in parallel and merges them with Reciprocal Rank Fusion. Microsoft's query-engine whitepaper adds generative query rewriting with a fine-tuned small language model and a new semantic ranker. **Vertex AI RAG Engine [199]** is Google's managed pipeline covering ingestion, chunking, embedding, indexing, retrieval and generation.

### 22.3.4 Tools, protocols and memory: 2024-2026 developments

The first two passes of this index concentrated on retrieval pipelines. Several 2024-2026 developments change where retrieval lives in a system, and they are cited often enough to belong in the core reading list.

- **Model Context Protocol (Anthropic, November 2024) [236].** MCP is an open standard for connecting assistants to "the systems where data lives". It launched with prebuilt servers for Google Drive, Slack, GitHub, Git, Postgres and Puppeteer. For RAG it moves connectors out of each application into reusable servers, so a retriever becomes one tool among many that an agent can call. The Dropbox Dash posts on MCP [75] are an example.
- **Writing effective tools for agents (Anthropic, September 2025) [237].** This is the tool-side companion to context engineering [93]. Its retrieval advice: prefer targeted `search_*` tools to `list_*` tools that dump everything into context; expose a `response_format` switch (in the post's Slack example, concise responses used about a third of the tokens of detailed ones); return readable fields instead of opaque IDs, which reduced hallucinations and improved retrieval precision; and paginate or truncate with sensible defaults. Claude Code caps tool responses at 25,000 tokens by default.
- **GraphRAG 1.0 (Microsoft Research, December 2024) [238].** The production release of the library [103]. It adds an incremental `update` command so a changed corpus does not need full re-indexing, builds a default vector store (LanceDB, with Azure AI Search supported) during indexing, cuts Parquet output by 80% and total disk use by 43%, and drops CLI startup from about 148 s to 2 s. It is not backward compatible with pre-1.0 indexes. It answers the "stale graph" failure mode listed in the technique table below.
- **HippoRAG 2 (OSU, ICML 2025) [239][240].** "From RAG to Memory" recasts graph retrieval as non-parametric continual learning. It extends the Personalized PageRank memory of HippoRAG [18] so that new knowledge can be added without retraining. It is the main research counterpoint to treating graph RAG only as a summarization device.
- **Reasoning models that search (2025) [241][242].** Search-o1 [241] puts an agentic search step and a "reason-in-documents" condensation step inside a large reasoning model's chain of thought. OpenAI's deep research [242] productized the same pattern (multi-step browsing with citations) in February 2025. Together with Search-R1 [33] and Anthropic's research system [94], these mark the 2025 shift from "retrieve then generate" to retrieval as a tool that a reasoning loop calls repeatedly.

**Lesson.** By 2026 the retriever is increasingly a tool behind a protocol, not a fixed pipeline stage. Tool ergonomics (response size, search granularity, readable identifiers) have become retrieval-quality levers in their own right.

---

## 22.4 Deep reads: infrastructure vendors and techniques

- **Late chunking (Jina, August 2024) [117][21].** Run a long-context embedding model (8,192 tokens) over the whole document, then split into chunks after the transformer and just before mean pooling. Each chunk vector is then conditioned on the surrounding document.
- **ColPali on Vespa [112]-[116][20].** Embed page screenshots as multi-vectors, which skips OCR and chunking. Vespa reports better ViDoRe results than BM25 and BGE-M3. For billions of pages, it adds a Hamming-distance MaxSim over binary-quantized vectors inside a phased ranking pipeline [112].
- **Hybrid fusion (Weaviate) [109][110].** Run BM25 and vector search in parallel, normalize, and fuse. Weaviate offers rankedFusion and relativeScoreFusion; the latter has been the default since v1.24. AutoCut (v1.20) trims the tail of the result list.
- **Two-stage retrieval (Pinecone) [105]-[108].** A bi-encoder retrieves candidates and a cross-encoder reranks them. Pinecone hosts rerankers such as bge-reranker-v2-m3 as a separate inference step.
- **Decoupled chunks (LlamaIndex) [123].** Use different chunks for retrieval and for synthesis: embed small units for precision, then expand to a window or the parent section before generation ("small-to-big").
- **Query transformations (LangChain) [125][126].** Multi-query, RAG-Fusion, decomposition, step-back and HyDE [9], plus a multi-vector retriever that indexes summaries but returns the full documents.

### Comparison of headline techniques by evidence

| Technique | Primary source | Where it acts | Reported gain | Cost profile | Main failure mode |
|---|---|---|---|---|---|
| Contextual Retrieval | Anthropic [91] | Index time | -49% failures (-67% with rerank) | ~$1.02/M doc tokens once | Context generation quality; reindex cost on churn |
| Late chunking | Jina [117] | Embedding | Gains on BEIR (paper) [21] | One long-context pass | Needs a long-context embedder |
| Hybrid + rerank | Weaviate/Pinecone [109][105] | Query time | Qualitative; widely adopted | Reranker latency per query | Reranker domain mismatch |
| GraphRAG global | MSR [97] | Index + query | ~70-80% win rate vs naive RAG | High LLM indexing cost | Extraction errors; stale graph |
| LazyGraphRAG | MSR [101] | Query time | Matches global quality at >700x lower query cost | Vector-RAG-level indexing | Relevance-test budget tuning |
| ColPali | Vespa [112] | Index time | Beats BM25/BGE-M3 on ViDoRe | Multi-vector storage | Storage blow-up without quantization |
| Agentic pre/post agents | Uber [45] | Query time | +27% acceptable, -60% incorrect (sec.) | Extra LLM calls | Latency; agent drift |
| Two-tier guardrail | DoorDash [46] | Post-generation | ~90% fewer hallucinations (sec.) | Cheap first tier | Similarity false-positives |
| Domain embedding model | GitHub [59] | Retrieval | +37.6% retrieval score | Training cost | Benchmark-to-product gap |
| Semantic search beside grep | Cursor [192] | Agent tools | +12.5% avg QA accuracy | Index + custom embedder | Small repos gain little |
| Dynamic community selection | MSR [100] | Query time | 77% lower global-search cost (level 1) | Cheap rater LLM | Rater misses relevant communities |
| Domain-scoped schema RAG | Uber, Swiggy [172][174] | Query time | 54% to 93% accuracy (Hermes, sec.) | Metadata curation | Wrong domain routing |
| Sufficiency-aware abstention | Google [196] | Post-retrieval | +2-10% correct among answered | Extra classifier call | Lower answer coverage |

---

## 22.5 Practitioner canon: essays, courses, talks

- **Eugene Yan [139][140].** Seven patterns for LLM systems, with RAG as the pattern for adding fresh external knowledge. His 2023 AI Engineer keynote covers evals, RAG, guardrails and feedback.
- **Applied LLMs [141][142].** Written by Yan, Bischof, Frye, Husain, Liu and Shankar (May-June 2024). Do not neglect keyword search, prefer RAG over fine-tuning for new knowledge, and long-context models will not make RAG obsolete.
- **Jason Liu [144][145][146].** Treat RAG as a recommender system. Measure retrieval precision, recall and MRR on synthetic evaluation sets, and separate leading from lagging metrics.
- **Hamel Husain [143].** A multi-part "Stop saying RAG is dead" series featuring guest talks.
- **Chip Huyen [147].** Places retrieval inside a full GenAI platform reference architecture.
- **Simon Willison [148].** A running tag archive of RAG experiments and commentary.
- **Jerry Liu, AI Engineer [149].** A production-ready RAG talk: start with table-stakes tuning (chunk size, metadata filters, hybrid search), then move to small-to-big retrieval and agents.

---

## 22.6 The annotated master index

Format: **Title** (Org/author, year) [n]: one-line summary. "n.d." marks an undated page.

### 22.6.1 Company engineering blogs and case studies

1. **Genie: Uber's Gen AI On-Call Copilot** (Uber, 2024) [44]: RAG over Engwiki and internal Stack Overflow for about 45k questions a month in Slack; chose RAG over fine-tuning.
2. **Enhanced Agentic-RAG** (Uber, 2025) [45]: Query and source agents, document enrichment, and hybrid retrieval for policy PDFs.
3. **Gen AI On-Call Copilot (summary)** (ZenML, n.d.) [164]: 154 channels, 70k+ questions, 48.9% helpful, about 13k hours saved.
4. **Enhanced Agentic RAG (summary)** (ZenML, n.d.) [165]: +27% acceptable answers and -60% incorrect advice, relative.
5. **Path to high-quality LLM-based Dasher support automation** (DoorDash, 2024) [46]: Condensed issue, then knowledge-base and case retrieval, then a two-tier guardrail and an LLM Judge.
6. **DoorDash RAG guardrails (summary)** (ZenML, n.d.) [166]: Secondary account of the quality-control loop.
7. **How DoorDash mitigates hallucinations** (Athina AI, n.d.) [151]: Hands-on replication of the two-tier guardrail.
8. **How DoorDash's AI cut errors by 90%** (Simform newsletter, n.d.) [153]: Newsletter digest of the DoorDash numbers.
9. **RAG with Knowledge Graphs for Customer Service QA** (LinkedIn, 2024) [40]: Dual-layer ticket graph; +77.6% MRR and -28.6% median resolution time.
10. **LinkedIn KG-RAG lesson** (Activeloop, n.d.) [171]: Course module that walks through the LinkedIn paper.
11. **Musings on building a Generative AI product** (LinkedIn, 2024) [47]: Postmortem on routing, retrieval and evaluation; the "last 20%" problem.
12. **Our learnings from the early days of generative AI** (LinkedIn, 2023) [48]: Lessons from the first GenAI product sprints.
13. **Lessons from building LinkedIn's GenAI platform** (AI Engineer, 2025) [49]: Talk on how the shared platform evolved.
14. **Building and scaling a production GenAI assistant** (ZenML, n.d.) [167]: Summary of the LinkedIn assistant architecture.
15. **The evolution of LinkedIn's generative AI tech stack** (ByteByteGo, n.d.) [150]: Newsletter synthesis of LinkedIn's posts.
16. **Text-to-SQL with RAG-enhanced table selection** (ZenML, n.d.) [50]: Pinterest Querybook table-summary index; acceptance 20% to 40%+.
17. **Pinterest's text-to-SQL system** (MLOps newsletter, n.d.) [51]: Digest of Pinterest's design.
18. **Build a text-to-SQL system: replicating Pinterest** (Analytics Vidhya, 2025) [52]: Step-by-step reimplementation.
19. **Deep dive: how Pinterest built text-to-SQL** (Wren AI, n.d.) [53]: Analysis of the table-retrieval stage.
20. **Unified context-intent embeddings for text-to-SQL** (devday.kr, n.d.) [54]: Korean summary of later Pinterest work across 100k+ tables.
21. **Leveraging RAG-powered LLMs for analytical tasks** (Grab, n.d.) [55]: Data-Arks middleware plus Spellvault for analytics RAG.
22. **RAG-powered analytics and fraud investigation** (ZenML, n.d.) [56]: 3-4 hours saved per report (secondary).
23. **From RAG to Richness: industry classification** (Ramp, 2025) [57]: RAG constrains the LLM to valid NAICS codes.
24. **Ramp engineering blog index** (Ramp, n.d.) [58]: Includes transaction-embedding retrieval with triplet loss.
25. **RAG-based industry classification (summary)** (ZenML, n.d.) [168]: About +60% retrieval and +5-15% accuracy (secondary).
26. **Copilot new embedding model** (GitHub, 2025) [59]: +37.6% retrieval score; 110.7% acceptance lift for C#.
27. **GitHub embedding model coverage** (InfoQ, 2025) [60]: 2x throughput and 8x less index memory.
28. **Securely indexing large codebases** (Cursor, n.d.) [61]: Merkle-tree sync, syntactic chunks, embedding cache.
29. **How Cursor indexes codebases fast** (Engineer's Codex, n.d.) [62]: Third-party teardown of Cursor indexing.
30. **How Cursor's codebase indexing works (2026 guide)** (Bito, 2026) [63]: Updated third-party guide.
31. **How Cody understands your codebase** (Sourcegraph, n.d.) [64]: RAG over code; keyword search, code graph, search backend.
32. **Cody context docs** (Sourcegraph, n.d.) [65]: Retrieval sources and @-mention context.
33. **Indexing large codebases for Copilot with local RAG** (Medium/xorets, n.d.) [76]: Community MCP-based local code RAG.
34. **Enterprise-grade RAG systems** (Harvey, 2025) [66]: pgvector KNN vs LanceDB IVF-PQ for legal tenancy.
35. **Scaling enterprise-grade RAG: legal frontier** (AI Engineer, 2025) [67]: Harvey and LanceDB talk on 10M+ document corpora.
36. **Building enterprise-grade RAG: lessons** (StartupHub, 2025) [68]: Talk summary with data-scale tiers.
37. **Enterprise-grade RAG for legal AI (summary)** (ZenML, n.d.) [169]: Secondary account of Harvey's design.
38. **What sets great RAG apart; vector search isn't enough** (Glean, n.d.) [69]: Hybrid search plus graph, plan/retrieve/generate.
39. **Why enterprise AI requires a knowledge graph** (Glean, n.d.) [70]: Graph of content, people and activity.
40. **Enterprise search with LLM technology** (Glean, n.d.) [71]: Scholastic semantic model fine-tuned on corpus and clicks.
41. **Agentic reasoning: the future of Work AI** (Glean, n.d.) [72]: Agentic layer on top of Glean's RAG.
42. **How we built AI-powered workplace search** (Glean webinar, n.d.) [73]: Engineering webinar.
43. **Dropbox Tech blog listing (Dash posts)** (Dropbox, 2025-2026) [75]: Context engineering, feature store, KG/MCP/DSPy in Dash.
44. **SDS 951: context engineering with Dropbox's Josh Clemm** (SuperDataScience, n.d.) [74]: Podcast on the Dash context engine.

### 22.6.2 Model providers and research labs

45. **Introducing Contextual Retrieval** (Anthropic, 2024) [91]: Chunk-context prepending; -49% failures, -67% with rerank.
46. **Introducing Citations** (Anthropic, 2025) [92]: Sentence-level grounded citations; up to +15% recall.
47. **Effective context engineering for AI agents** (Anthropic, 2025) [93]: Just-in-time vs pre-computed retrieval.
48. **How we built our multi-agent research system** (Anthropic, June 2025) [94]: Parallel subagent search; +90.2%, about 15x chat tokens.
49. **Building effective agents** (Anthropic, 2024) [95]: The augmented LLM, with retrieval as a core augmentation.
50. **GraphRAG: unlocking LLM discovery on narrative private data** (Microsoft Research, 2024) [96]: Introduced GraphRAG on the VIINA dataset.
51. **GraphRAG now on GitHub** (Microsoft Research, 2024) [97]: Global map-reduce; ~70-80% win rate.
52. **GraphRAG auto-tuning** (Microsoft Research, 2024) [98]: Domain-adapted extraction prompts.
53. **Introducing DRIFT search** (Microsoft Research, 2024) [99]: Hybrid of global and local search; 78%/81% wins.
54. **Dynamic community selection** (Microsoft Research, 2024) [100]: 77% cheaper global search at community level 1.
55. **LazyGraphRAG** (Microsoft Research, 2024) [101]: Deferred, budgeted graph RAG.
56. **BenchmarkQED** (Microsoft Research, 2025) [102]: Automated RAG benchmarking (AutoQ/AutoE/AutoD).
57. **microsoft/graphrag repository** (Microsoft, n.d.) [103]: Reference implementation.
58. **GraphRAG documentation** (Microsoft, n.d.) [104]: Indexing and query-mode documentation.
59. **Long context RAG performance of LLMs** (Databricks, 2024) [77]: Performance degrades past 32k-64k tokens for many models.
60. **From RAG to fabric, Part 1** (AWS GenAIIC, 2024) [78]: Text RAG optimization and evaluation.
61. **From RAG to fabric, Part 2** (AWS GenAIIC, 2024) [79]: Structured and multimodal RAG.
62. **RAG options and architectures on AWS** (AWS, n.d.) [80]: Prescriptive guidance on build vs managed.
63. **Amazon Bedrock Knowledge Bases** (AWS, n.d.) [81]: Managed ingestion and retrieval.
64. **RAG in Azure AI Search** (Microsoft, n.d.) [82]: Hybrid and semantic ranking for RAG.
65. **RAG solution design and evaluation guide** (Microsoft Azure, n.d.) [83]: Phased design: chunking, enrichment, retrieval, evaluation.
66. **RAG guide** (Google Cloud, n.d.) [84]: Overview of RAG on Google Cloud.
67. **What is RAG** (Google Cloud, n.d.) [85]: Conceptual primer.
68. **RAG with foundation models** (Google Cloud, n.d.) [86]: Developer walkthrough.
69. **Optimizing LLM accuracy** (OpenAI, n.d.) [87]: RAG vs prompting vs fine-tuning as separate levers.
70. **File search tool** (OpenAI, n.d.) [88]: Managed vector-store retrieval in the API.
71. **What is retrieval-augmented generation** (NVIDIA, 2023) [89]: Primer with history.
72. **What is RAG** (IBM Research, 2023) [90]: Primer on grounding and freshness.

### 22.6.3 Vector databases, search engines and retrieval vendors

73. **Rerankers and two-stage retrieval** (Pinecone, n.d.) [105]: Bi-encoder recall plus cross-encoder precision.
74. **Pinecone Rerank v0** (Pinecone, n.d.) [106]: Hosted reranker announcement.
75. **Rerank guide** (Pinecone docs, n.d.) [107]: Rerank as a query step or a standalone step.
76. **Chunking strategies** (Pinecone, n.d.) [108]: Fixed, recursive, semantic and structural chunking.
77. **Hybrid search explained** (Weaviate, 2025) [109]: Dense plus sparse fusion.
78. **Hybrid search concepts** (Weaviate docs, n.d.) [110]: rankedFusion vs relativeScoreFusion.
79. **Hybrid search for web developers** (Weaviate, n.d.) [111]: Applied tutorial.
80. **Scaling ColPali to billions of PDFs** (Vespa, n.d.) [112]: Binary-quantized Hamming MaxSim with phased ranking.
81. **Visual RAG in practice** (Vespa, n.d.) [113]: End-to-end visual RAG application.
82. **Vespa announces ColPali support** (Vespa, 2024) [114]: Press release.
83. **ColPali visual retrieval demo** (Vespa / Hugging Face Space, n.d.) [115]: Open-source demo.
84. **Transforming information retrieval with ColPali** (Vespa, n.d.) [116]: Explainer.
85. **Late chunking in long-context embedding models** (Jina AI, 2024) [117]: Chunk after encoding.
86. **Elasticsearch Labs** (Elastic, n.d.) [118]: Hub for hybrid search, ELSER and RAG tutorials.
87. **Qdrant articles** (Qdrant, n.d.) [119]: Hub for hybrid, sparse and quantization articles.
88. **Improving retrieval in RAG with reranking** (Unstructured, n.d.) [120]: Reranking applied to parsed documents.
89. **Advanced RAG apps with LlamaIndex** (Zilliz, n.d.) [121]: Small-to-big and sentence-window patterns.
90. **RAG on Notion docs via LangChain** (Zilliz, n.d.) [122]: Tutorial; chunk size and overlap tuning.
91. **The architect's guide to production RAG** (Ragie, n.d.) [158]: Vendor guide to scaling challenges.

### 22.6.4 Frameworks, evaluation and tooling

92. **Building performant RAG applications for production** (LlamaIndex, n.d.) [123]: Decoupled chunks, structured retrieval.
93. **OpenAI cookbook: evaluating RAG systems** (LlamaIndex, 2023) [124]: Retrieval metrics (hit rate, MRR) and response evaluation.
94. **Deconstructing RAG** (LangChain, n.d.) [125]: Taxonomy of query transforms, routing and indexing.
95. **rag-from-scratch** (LangChain, n.d.) [126]: Notebook series covering multi-query, Fusion, HyDE and RAPTOR.
96. **Pinecone rerank integration** (LangChain docs, n.d.) [127]: Retriever integration.
97. **Ragas documentation** (Ragas, n.d.) [128]: Faithfulness and context precision/recall metrics.
98. **DSPy** (Stanford NLP, n.d.) [129]: Programmatic optimization of RAG prompts.
99. **Haystack** (deepset, n.d.) [130]: Pipeline framework for retrieval and generation.
100. **LightRAG repository** (HKUDS, n.d.) [131]: Lightweight graph-plus-vector RAG.
101. **ColPali repository** (Illuin Tech, n.d.) [132]: Visual late-interaction retriever.
102. **RAGatouille** (Answer.AI, n.d.) [133]: ColBERT made easy to use.
103. **RAG evaluation cookbook** (Hugging Face, n.d.) [134]: Synthetic evaluation set plus LLM-as-judge.
104. **Optimizing Ragas to evaluate a RAG pipeline** (Braintrust, n.d.) [135]: ContextRecall and AnswerCorrectness.
105. **Build a simple RAG app** (LangWatch, n.d.) [136]: Retrieval-first evaluation that is fast, cheap and objective.
106. **Decreasing hallucinations in RAG pipelines** (Future AGI, n.d.) [137]: Grid search over chunking and retrieval strategies.
107. **RAG using LlamaIndex on ROCm** (AMD, n.d.) [138]: RAG on AMD GPUs.

### 22.6.5 Practitioner essays, courses, talks and newsletters

108. **Patterns for building LLM-based systems and products** (Eugene Yan, 2023) [139]: RAG as one of seven patterns.
109. **Building blocks for LLM systems** (Eugene Yan, AI Engineer, 2023) [140]: Keynote on evals, RAG, guardrails and feedback.
110. **What we've learned from a year of building with LLMs** (Applied LLMs, 2024) [141]: Practical RAG lessons.
111. **What we learned... (O'Reilly edition)** (Yan et al., 2024) [142]: Book edition of the same essays.
112. **Stop saying RAG is dead** (Hamel Husain, n.d.) [143]: Multi-part talk series.
113. **Systematically improving RAG applications** (Jason Liu, n.d.) [144]: Course on eval-driven RAG.
114. **Why your RAG system is broken** (TWIML / Jason Liu, n.d.) [145]: Podcast on test sets and metrics.
115. **The real battle-tested RAG playbook** (Decoding ML, n.d.) [146]: Newsletter distillation.
116. **Building a generative AI platform** (Chip Huyen, 2024) [147]: Where retrieval sits in the platform.
117. **RAG tag archive** (Simon Willison, n.d.) [148]: Running notes and experiments.
118. **Production-ready RAG** (Jerry Liu, AI Engineer, n.d.) [149]: Table-stakes tuning, then advanced retrieval.
119. **Agentic RAG: context-augmented agents** (Cobus Greyling, n.d.) [154]: Agentic RAG patterns.
120. **Anthropic contextual retrieval coverage** (InfoQ, 2024) [155]: News analysis.
121. **Contextual retrieval tutorial** (DataCamp, n.d.) [156]: Hands-on implementation.
122. **Why context is NOT all you need** (Almond AI, 2024) [157]: Critique of Contextual Retrieval.
123. **RAG in production: practitioner view** (Enterprise DNA, n.d.) [159]: Most of the gains came from retrieval.
124. **RAG in production (book)** (Leanpub, n.d.) [160]: Messy-PDF realities.
125. **RAG: the architecture of reliable AI (book)** (INNOQ, n.d.) [161]: Book-length treatment.
126. **RAG query transformation in depth** (dev.to/jamesli, n.d.) [162]: Multi-query, decomposition and step-back.
127. **Scaling RAG: data engineering hurdles** (dev.to, n.d.) [163]: Ingestion and pipeline scaling.
128. **RAG examples** (Evidently AI, n.d.) [152]: Catalogue of company RAG deployments.
129. **Optimizing RAG systems: lessons from production** (ZenML, n.d.) [170]: AWS GenAIIC distillation.

### 22.6.6 Foundational and frontier papers

130. **RAG for knowledge-intensive NLP** (Lewis et al., 2020) [1]: The original RAG model.
131. **REALM** (Guu et al., 2020) [2]: Retrieval-augmented pretraining.
132. **RETRO** (Borgeaud et al., 2021) [3]: Retrieval from trillions of tokens.
133. **Atlas** (Izacard et al., 2022) [4]: Few-shot learning with retrieval.
134. **DPR** (Karpukhin et al., 2020) [5]: Dense passage retrieval.
135. **ColBERT** (Khattab and Zaharia, 2020) [6]: Late interaction.
136. **ColBERTv2** (Santhanam et al., 2021) [7]: Residual compression.
137. **SPLADE** (Formal et al., 2021) [8]: Learned sparse retrieval.
138. **HyDE** (Gao et al., 2022) [9]: Hypothetical document embeddings.
139. **IRCoT** (Trivedi et al., 2022) [10]: Interleaving retrieval with chain-of-thought.
140. **FLARE** (Jiang et al., 2023) [11]: Active retrieval.
141. **Self-RAG** (Asai et al., 2023) [12]: Reflection tokens.
142. **Corrective RAG** (Yan et al., 2024) [13]: A retrieval evaluator triggers corrective actions.
143. **Adaptive-RAG** (Jeong et al., 2024) [14]: Routes queries by complexity.
144. **RAFT** (Zhang et al., 2024) [15]: Fine-tuning for RAG with distractor documents.
145. **RAPTOR** (Sarthi et al., 2024) [16]: Recursive summary tree.
146. **From local to global: GraphRAG** (Edge et al., 2024) [17]: Query-focused summarization.
147. **HippoRAG** (Gutiérrez et al., 2024) [18]: Personalized PageRank memory.
148. **LightRAG** (Guo et al., 2024) [19]: Dual-level graph retrieval.
149. **ColPali** (Faysse et al., 2024) [20]: Vision-language document retrieval.
150. **Late chunking** (Günther et al., 2024) [21]: Contextual chunk embeddings.
151. **Lost in the middle** (Liu et al., 2023) [22]: Position bias in long contexts.
152. **Retrieval meets long context LLMs** (Xu et al., 2023) [23]: Retrieval helps even long-context models.
153. **Long context RAG performance** (Leng et al., 2024) [24]: Databricks paper.
154. **RAG for LLMs: a survey** (Gao et al., 2023) [25]: Naive, advanced and modular RAG taxonomy.
155. **Seven failure points of RAG** (Barnett et al., 2024) [26]: Engineering failure taxonomy.
156. **Searching for best practices in RAG** (Wang et al., 2024) [27]: Ablation of pipeline choices.
157. **RAGAS** (Es et al., 2023) [28]: Reference-free RAG evaluation.
158. **ARES** (Saad-Falcon et al., 2023) [29]: Automated RAG evaluation with trained judges.
159. **BEIR** (Thakur et al., 2021) [30]: Zero-shot IR benchmark.
160. **MTEB** (Muennighoff et al., 2022) [31]: Embedding benchmark.
161. **Matryoshka representation learning** (Kusupati et al., 2022) [32]: Truncatable embeddings.
162. **Search-R1** (Jin et al., 2025) [33]: RL-trained search-calling LLMs.
163. **Agentic RAG survey** (Singh et al., 2025) [34]: Taxonomy of agentic RAG.
164. **RAGVA** (2025) [35]: Engineering RAG virtual assistants in practice.
165. **RAG survey: architectures and robustness** (2025) [36]: Robustness-focused survey.
166. **Is agentic RAG worth it?** (2026) [37]: Experimental comparison of RAG approaches.
167. **CSR-RAG** (2026) [38]: Enterprise-scale retrieval for text-to-SQL.
168. **Hierarchical Re-ranker Retriever** (2025) [39]: Chunk-granularity-aware reranking.
169. **RAG for NASA aerospace documentation** (PMLR, 2026) [41]: Domain retrieval and user evaluation.
170. **RankGPT** (Sun et al., 2023) [42]: LLMs as rerankers.
171. **FreshLLMs** (Vu et al., 2023) [43]: Search-augmented freshness.

### 22.6.7 Additions from the October 2026 pass

Company engineering blogs:

172. **QueryGPT: natural language to SQL** (Uber, 2024) [172]: Domain "workspaces" of schemas and SQL samples narrow table retrieval.
173. **Finch: conversational AI data agent** (Uber, 2025) [173]: Financial-data agent built on the same domain-scoped retrieval.
174. **Hermes: a text-to-SQL solution** (Swiggy, 2024) [174]: Slack-based text-to-SQL with charter-scoped knowledge bases.
175. **Swiggy Hermes V3 conversational analyst** (InfoQ, 2026) [175]: V3 moves from one-shot SQL to a multi-turn data analyst.
176. **Evolution of Hermes V3 (summary)** (ZenML, n.d.) [176]: Reports 54% to 93% accuracy (secondary).
177. **Enhancing chatbot knowledge with user-generated content** (DoorDash, 2025) [177]: Scales the support knowledge base beyond manual curation.
178. **How DoorDash leverages LLMs for better search retrieval** (DoorDash, n.d.) [178]: LLM query understanding for search retrieval.
179. **Building DoorDash's product knowledge graph with LLMs** (DoorDash, 2024) [179]: LLM attribute extraction feeding a product graph.
180. **Building the next generation of job search** (LinkedIn, n.d.) [180]: LLM-era semantic job retrieval.
181. **Automated GenAI-driven search quality evaluation** (LinkedIn, n.d.) [181]: LLM judges for search relevance at scale.
182. **Building Dash: RAG and multi-step AI agents** (Dropbox, 2024) [182]: Universal search with access control, lexical then embedding rerank.
183. **AI-powered answers and summaries in file previews** (Dropbox, n.d.) [183]: Per-file Q&A in the web previewer.
184. **Improving Pinterest search relevance using LLMs** (Pinterest, n.d.) [184]: LLM relevance models distilled for search.
185. **Supercharging discovery in search with LLMs** (Instacart, 2024) [185]: LLM-generated related and complementary results.
186. **Revamping query understanding with LLMs (summary)** (ZenML, n.d.) [186]: Instacart RAG context engineering plus a fine-tuned Llama-3-8B.
187. **Automation Platform v2: conversational AI** (Airbnb, 2024) [187]: Platform for LLM-powered support flows with retrieval.
188. **LLM-friendly knowledge representation for customer support** (Airbnb, COLING 2025) [188]: Intent-Context-Action format for policy documents.
189. **Video Q&A with RAG and speaker detection (summary)** (ZenML, n.d.) [189]: Vimeo's multi-granularity transcript index with timestamps.
190. **Navigating fair housing guardrails in LLMs** (Zillow, 2024) [190]: Compliance guardrail for real-estate assistants.
191. **Leveraging RAG-powered LLMs for analytical tasks** (Grab, 2024) [191]: Primary post behind entry 21.
192. **Improving agent with semantic search** (Cursor, 2025) [192]: +12.5% average accuracy from embedding search beside grep.
193. **Custom semantic search for coding agents (summary)** (ZenML, n.d.) [193]: Secondary account of the Cursor study.
194. **Notion-based RAG Slackbot hackathon** (DataDome, n.d.) [194]: One-day internal RAG bot that cites Notion pages.
195. **CoCounsel Knowledge Search launch** (Thomson Reuters, 2025) [195]: Retrieval across firm repositories for legal work.

Labs, clouds and research:

196. **Sufficient context** (Joren et al., Google, ICLR 2025) [196]: Taxonomy of RAG errors plus selective abstention.
197. **Sufficient context publication page** (Google Research, 2025) [197]: Publication page with authors and venue.
198. **Bedrock Knowledge Bases GraphRAG GA** (AWS, 2025) [198]: Managed graph and vector retrieval on Neptune Analytics.
199. **Vertex AI RAG Engine overview** (Google Cloud, n.d.) [199]: Managed ingestion-to-generation pipeline.
200. **Hybrid search overview** (Azure AI Search, n.d.) [200]: Parallel BM25 and vector search with RRF merge.
201. **AI Search query engine performance whitepaper** (Microsoft, n.d.) [201]: Query rewriting with an SLM and a new semantic ranker.
202. **Long-context RAG: OpenAI o1 and Gemini** (Databricks, 2024) [202]: o1 leads up to 128k tokens; Gemini stays stable to 2M.
203. **Mosaic AI Agent Evaluation** (Databricks docs, n.d.) [203]: LLM judges for retrieval and response quality, latency and cost.
204. **RAFT: teaching LLMs to be better at RAG** (Microsoft, 2024) [204]: Domain fine-tuning with distractor documents.

Vendors and frameworks:

205. **What is semantic reranking** (Elastic Search Labs, n.d.) [205]: Zero-shot reranking; it cannot fix low BM25 recall.
206. **Qdrant BM42 hybrid search** (Blocks and Files, 2024) [206]: Attention-derived term weights that keep BM25's IDF.
207. **Snippet selection and URL ranking in DeepSearch** (Jina AI, 2025) [207]: Retrieval inside deep-research agents.
208. **Enhancing RAG pipelines with re-ranking** (NVIDIA, n.d.) [208]: Reranker stage in NVIDIA's RAG reference design.
209. **Easy introduction to multimodal RAG** (NVIDIA, n.d.) [209]: Text, image and audio retrieval patterns.
210. **The GraphRAG manifesto** (Neo4j, 2024) [210]: Argument for knowledge graphs inside RAG.
211. **RAG is dead, long live agentic retrieval** (LlamaIndex, 2025) [211]: Auto-routed retrieval modes across indexes.
212. **Query transformations** (LangChain, 2023) [212]: LLM rewriting of queries before embedding.
213. **Self-reflective RAG with LangGraph** (LangChain, 2024) [213]: Self-RAG and CRAG as state machines with loops.
214. **open_deep_research** (LangChain, n.d.) [214]: Open deep-research agent (archived August 2026).
215. **Agentic RAG tutorial** (LangGraph docs, n.d.) [215]: Agent decides when to retrieve.
216. **Self-RAG project page** (Asai et al., 2023) [216]: Code and models for reflection-token RAG.
217. **Search-R1 code** (Jin et al., 2025) [217]: RL training of search-calling LLMs.
218. **RAG_Techniques** (N. Diamant, n.d.) [218]: 30+ runnable notebooks of RAG techniques.
219. **Awesome-RAG** (Danielskry, n.d.) [219]: Curated list of RAG papers, tools and posts.
220. **LLMs in production** (primaprashant, n.d.) [220]: Curated index of company LLM engineering posts.

Practitioners, talks and secondary analyses:

221. **What we learned from a year of building with LLMs, Part I** (O'Reilly Radar, 2024) [221]: Tactical RAG lessons.
222. **RAG evaluation is broken (and how to fix it)** (Belfer and Granot, AIEWF 2025) [222]: Talk on RAG benchmark flaws.
223. **The state of AI-powered search and retrieval** (Frank Liu, AIEWF 2025) [223]: Talk surveying retrieval in 2025.
224. **RAG is more than just embedding search** (Jason Liu, 2023) [224]: Critiques "dumb" single-vector RAG as demoware.
225. **Hugging Face open deep research** (TechCrunch, 2025) [225]: 54% on GAIA vs 67.36% for OpenAI deep research.
226. **RAG (deep dive)** (Pragmatic Engineer, n.d.) [226]: Newsletter explainer on how RAG is built.

Papers:

227. **RAG-Fusion** (2024) [227]: Multi-query generation with reciprocal rank fusion.
228. **REFRAG** (2025) [228]: Compresses retrieved context into embeddings to cut decoding latency.
229. **VideoRAG over video corpora** (2025) [229]: Retrieval and generation over video with LVLMs.
230. **Don't do RAG: cache-augmented generation** (2024) [230]: Preloads small corpora into the KV cache.
231. **A-RAG** (2026) [231]: Agentic RAG with keyword, semantic and chunk-level tools.
232. **Embedding-based retrieval in Facebook search** (Facebook, 2020) [232]: Industrial two-tower retrieval baseline.
233. **Embedding-based retrieval at Scribd** (Scribd, 2021) [233]: Production dense retrieval before LLMs.

Managed-tool and framework documentation:

234. **Query expansion cookbook** (Haystack, n.d.) [234]: Recall boost from LLM query expansion.
235. **File search tool (Responses API)** (OpenAI, n.d.) [235]: Hosted semantic plus keyword retrieval.

### 22.6.8 Additions from the editorial pass

236. **Introducing the Model Context Protocol** (Anthropic, 2024) [236]: Open standard for connecting assistants to data systems; launch servers for Drive, Slack, GitHub, Postgres.
237. **Writing effective tools for agents** (Anthropic, 2025) [237]: Search over list tools, concise/detailed response formats, pagination and truncation.
238. **Moving to GraphRAG 1.0** (Microsoft Research, 2024) [238]: Incremental indexing, default vector store, 43% less disk.
239. **HippoRAG repository (incl. HippoRAG 2)** (OSU NLP Group, n.d.) [239]: Code for PPR-based graph memory.
240. **From RAG to Memory: non-parametric continual learning** (Gutiérrez et al., ICML 2025) [240]: HippoRAG 2 paper.
241. **Search-o1** (Li et al., 2025) [241]: Agentic search inside a reasoning model's chain of thought.
242. **Introducing deep research** (OpenAI, 2025) [242]: Productized multi-step browsing research agent with citations.

---

## 22.7 Cross-cutting lessons from the corpus

The same failure modes come up again and again across the sources:

1. **Parsing loses structure.** Uber's policy PDFs [45], the messy-PDF warnings in [160], and the ColPali line of work [20][112] all point at ingestion as the first place quality drops.
2. **Chunks lose their document.** Contextual Retrieval [91], late chunking [117] and small-to-big retrieval [123] are three answers to the same problem at different points in the pipeline: before embedding, inside the encoder, and after retrieval.
3. **Lexical signals still matter.** Hybrid BM25 plus vector search shows up at Uber [45], Glean [69], Anthropic [91] and AWS [78], and in Applied LLMs' advice [141].
4. **Global questions break top-k.** GraphRAG [96][97] and BenchmarkQED [102] show that bigger context windows do not substitute for structure on whole-corpus questions. Databricks [77] shows that quality falls past 32-64k tokens for many models.
5. **The last 20% is most of the work.** LinkedIn's 80%-to-95% timeline [47] and Genie's 48.9% helpfulness rate [164] suggest planning for months of evaluation-driven iteration.
6. **Agentic retrieval is a cost decision.** Anthropic's 15x token multiplier [94] and the Uber and EAg-RAG trade-offs [45][37] put agentic retrieval in the "high-value queries only" tier.

---

## 22.8 Key takeaways for system designers

- **Start from retrieval metrics, not answer vibes.** Build a labeled or synthetic retrieval set and track recall@k, MRR and failure rate before tuning prompts [91][124][144]. Anthropic's 5.7% to 1.9% failure progression is a template for this kind of ablation.
- **Default to hybrid retrieval plus a reranker,** and over-fetch before reranking (for example 150 candidates down to 20) [91][105][109].
- **Fix context loss at index time when the corpus is stable** (Contextual Retrieval at about $1.02 per million tokens) and at query time when it churns (just-in-time agent retrieval) [91][93].
- **Choose index topology by question type.** Use vectors for local lookups. Use graph or community structures for global sensemaking, and prefer budgeted, deferred variants (LazyGraphRAG) when indexing cost matters [101][102].
- **Constrain the output space when you can.** Ramp's NAICS classification and Pinterest's table selection turn generation into choosing among retrieved candidates, which removes whole classes of hallucination [57][50].
- **Put cheap checks first.** Use a similarity guardrail before an LLM evaluator, and a cheap LLM to prune communities before an expensive one [46][100].
- **Tenancy is architecture.** Permission-aware retrieval (Glean) and per-customer storage (Harvey with LanceDB) must be designed in from the start [69][66].
- **Code RAG is an indexing problem.** Incremental Merkle sync, syntactic chunking, content-hash caches and domain-trained embedders give the largest gains [59][61].
- **Treat agentic RAG as a tier, not a default.** Route only complex or high-stakes queries through multi-step agents [94][95][37].
- **Design retrieval tools for agents, not humans.** Targeted search tools, bounded and paginated responses, and readable identifiers behind a standard protocol such as MCP [236][237].
- **Partition retrieval by domain before you search.** Uber's workspaces, Swiggy's charters and Pinterest's top-tier table index all shrink the candidate set and assign ownership of metadata quality [172][174][50].
- **Let the system say "not enough context".** Sufficiency signals and abstention trade a little coverage for accuracy. DoorDash's guardrail does the same at the post-generation layer [196][46].
- **Benchmark continuously and automatically.** Synthetic query generation and LLM-judge pairwise evaluation (BenchmarkQED, Ragas, ARES) make regression testing affordable [102][28][29].

---

## References

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Lewis et al. (Facebook AI). 2020. https://arxiv.org/abs/2005.11401
2. REALM: Retrieval-Augmented Language Model Pre-Training. Guu et al. (Google). 2020. https://arxiv.org/abs/2002.08909
3. Improving Language Models by Retrieving from Trillions of Tokens (RETRO). Borgeaud et al. (DeepMind). 2021. https://arxiv.org/abs/2112.04426
4. Atlas: Few-shot Learning with Retrieval Augmented Language Models. Izacard et al. (Meta). 2022. https://arxiv.org/abs/2208.03299
5. Dense Passage Retrieval for Open-Domain Question Answering. Karpukhin et al. (Facebook AI). 2020. https://arxiv.org/abs/2004.04906
6. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. Khattab, Zaharia (Stanford). 2020. https://arxiv.org/abs/2004.12832
7. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. Santhanam et al. (Stanford). 2021. https://arxiv.org/abs/2112.01488
8. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. Formal et al. (Naver Labs). 2021. https://arxiv.org/abs/2107.05720
9. Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). Gao et al. (CMU). 2022. https://arxiv.org/abs/2212.10496
10. Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT). Trivedi et al. 2022. https://arxiv.org/abs/2212.10509
11. Active Retrieval Augmented Generation (FLARE). Jiang et al. 2023. https://arxiv.org/abs/2305.06983
12. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. Asai et al. 2023. https://arxiv.org/abs/2310.11511
13. Corrective Retrieval Augmented Generation. Yan et al. 2024. https://arxiv.org/abs/2401.15884
14. Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity. Jeong et al. 2024. https://arxiv.org/abs/2403.14403
15. RAFT: Adapting Language Model to Domain Specific RAG. Zhang et al. (UC Berkeley). 2024. https://arxiv.org/abs/2403.10131
16. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. Sarthi et al. (Stanford). 2024. https://arxiv.org/abs/2401.18059
17. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. Edge et al. (Microsoft). 2024. https://arxiv.org/abs/2404.16130
18. HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs. Gutiérrez et al. (OSU). 2024. https://arxiv.org/abs/2405.14831
19. LightRAG: Simple and Fast Retrieval-Augmented Generation. Guo et al. (HKU). 2024. https://arxiv.org/abs/2410.05779
20. ColPali: Efficient Document Retrieval with Vision Language Models. Faysse et al. (Illuin Tech). 2024. https://arxiv.org/abs/2407.01449
21. Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models. Günther et al. (Jina AI). 2024. https://arxiv.org/abs/2409.04701
22. Lost in the Middle: How Language Models Use Long Contexts. Liu et al. (Stanford). 2023. https://arxiv.org/abs/2307.03172
23. Retrieval Meets Long Context Large Language Models. Xu et al. (NVIDIA). 2023. https://arxiv.org/abs/2310.03025
24. Long Context RAG Performance of Large Language Models. Leng et al. (Databricks Mosaic Research). 2024. https://arxiv.org/abs/2411.03538
25. Retrieval-Augmented Generation for Large Language Models: A Survey. Gao et al. (Tongji/Fudan). 2023. https://arxiv.org/abs/2312.10997
26. Seven Failure Points When Engineering a Retrieval Augmented Generation System. Barnett et al. (Deakin). 2024. https://arxiv.org/abs/2401.05856
27. Searching for Best Practices in Retrieval-Augmented Generation. Wang et al. (Fudan). 2024. https://arxiv.org/abs/2407.01219
28. RAGAS: Automated Evaluation of Retrieval Augmented Generation. Es et al. 2023. https://arxiv.org/abs/2309.15217
29. ARES: An Automated Evaluation Framework for RAG Systems. Saad-Falcon et al. (Stanford). 2023. https://arxiv.org/abs/2311.09476
30. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. Thakur et al. 2021. https://arxiv.org/abs/2104.08663
31. MTEB: Massive Text Embedding Benchmark. Muennighoff et al. (Hugging Face). 2022. https://arxiv.org/abs/2210.07316
32. Matryoshka Representation Learning. Kusupati et al. 2022. https://arxiv.org/abs/2205.13147
33. Search-R1: Training LLMs to Reason and Leverage Search Engines with RL. Jin et al. 2025. https://arxiv.org/abs/2503.09516
34. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG. Singh et al. 2025. https://arxiv.org/abs/2501.09136
35. RAGVA: Engineering Retrieval Augmented Generation-based Virtual Assistants in Practice. 2025. https://arxiv.org/abs/2502.14930
36. Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers. 2025. https://arxiv.org/abs/2506.00054
37. Is Agentic RAG worth it? An experimental comparison of RAG approaches. 2026. https://arxiv.org/abs/2601.07711v1
38. CSR-RAG: An Efficient Retrieval System for Text-to-SQL on the Enterprise Scale. 2026. https://arxiv.org/html/2601.06564v1
39. Hierarchical Re-ranker Retriever (HRR). 2025. https://arxiv.org/pdf/2503.02401
40. Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering. Xu et al. (LinkedIn), SIGIR. 2024. https://arxiv.org/abs/2404.17723
41. RAG in the Aerospace Domain: Retrieval, Generation, and User Evaluation for NASA Documentation. Petniunas et al. (PMLR v307). 2026. https://proceedings.mlr.press/v307/petniunas26a.html
42. Is ChatGPT Good at Search? LLMs as Re-Ranking Agents (RankGPT). Sun et al. 2023. https://arxiv.org/abs/2304.09542
43. FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation. Vu et al. (Google). 2023. https://arxiv.org/abs/2310.03214
44. Genie: Uber's Gen AI On-Call Copilot. Uber Engineering. 2024. https://www.uber.com/blog/genie-ubers-gen-ai-on-call-copilot/
45. Enhanced Agentic-RAG: What If Chatbots Could Deliver Near-Human Precision? Chakraborty, Settles, Raghavendra (Uber). 2025. https://www.uber.com/nl/en/blog/enhanced-agentic-rag
46. Path to High-Quality LLM-based Dasher Support Automation. DoorDash Engineering. 2024. https://careersatdoordash.com/?p=12568
47. Musings on Building a Generative AI Product. LinkedIn Engineering. 2024. https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product
48. Our Learnings from the Early Days of Generative AI. LinkedIn Engineering. 2023. https://engineering.linkedin.com/blog/2023/our-learnings-from-the-early-days-of-generative-ai
49. Lessons from Building LinkedIn's GenAI Platform (talk). AI Engineer. 2025. https://ai.engineer/talks/n9rjuBuShko-lessons-from-building-linkedins-genai-platform
50. Text-to-SQL System with RAG-Enhanced Table Selection (Pinterest case). ZenML LLMOps Database. n.d. https://www.zenml.io/llmops-database/text-to-sql-system-with-rag-enhanced-table-selection
51. Pinterest's Text-to-SQL System. MLOps Newsletter. n.d. https://mlops.substack.com/p/pinterests-text-to-sql-system-through
52. Build a Text-to-SQL System: A Guide to Replicating Pinterest's Approach. Analytics Vidhya. 2025. https://www.analyticsvidhya.com/blog/2025/10/build-a-text-to-sql-system/
53. Deep Dive into How Pinterest Built Its Text-to-SQL Solution. Wren AI. n.d. https://getwren.ai/post/deep-dive-into-how-pinterest-built-its-text-to-sql-solution
54. Unified Context-Intent Embeddings for Scalable Text-to-SQL (Korean summary). devday.kr. n.d. https://devday.kr/article/unified-context-intent-embeddings-for-scalable-text-to-sql
55. Grab Engineering: Analytics category (incl. "Leveraging RAG-powered LLMs for analytical tasks"). Grab. n.d. https://engineering.grab.com/categories/analytics/
56. RAG-Powered LLM System for Automated Analytics and Fraud Investigation (Grab case). ZenML. n.d. https://www.zenml.io/llmops-database/rag-powered-llm-system-for-automated-analytics-and-fraud-investigation
57. From RAG to Richness: How Ramp Revamped Industry Classification. Ryne Carbone (Ramp). 2025. https://engineering.ramp.com/industry_classification
58. Ramp Engineering Blog (index). Ramp. n.d. https://engineering.ramp.com
59. GitHub Copilot Gets Smarter at Finding Your Code: Inside Our New Embedding Model. GitHub. 2025. https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/
60. GitHub Embedding Model (news). InfoQ. 2025. https://www.infoq.com/news/2025/10/github-embedding-model
61. Securely Indexing Large Codebases. Cursor. n.d. https://cursor.com/blog/secure-codebase-indexing
62. How Cursor Indexes Codebases Fast. Engineer's Codex. n.d. https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast
63. How Cursor's Codebase Indexing Works (2026 Guide). Bito. 2026. https://bito.ai/blog/how-cursors-codebase-indexing-works-2026-guide/
64. How Cody Understands Your Codebase. Sourcegraph. n.d. https://sourcegraph.com/blog/how-cody-understands-your-codebase
65. Cody Context (docs). Sourcegraph. n.d. https://sourcegraph.com/docs/cody/core-concepts/context
66. Enterprise-Grade RAG Systems. Harvey. 2025. https://www.harvey.ai/blog/enterprise-grade-rag-systems
67. Scaling Enterprise-Grade RAG Systems: Lessons from the Legal Frontier (talk). AI Engineer. 2025. https://ai.engineer/talks/scaling-enterprise-grade-rag-systems-lessons-from-the-legal-frontier
68. Building Enterprise-Grade RAG: Lessons from the Legal Frontier. StartupHub.ai. 2025. https://www.startuphub.ai/ai-news/ai-video/2025/building-enterprise-grade-rag-lessons-from-the-legal-frontier
69. What Sets Great Retrieval Augmented Generation Apart, and Why Vector Search Isn't Enough. Glean. n.d. https://www.glean.com/blog/hybrid-vs-rag-vector
70. Why Great Enterprise Search and Generative AI Requires a Comprehensive Knowledge Graph. Glean. n.d. https://www.glean.com/blog/enterprise-ai-knowledge-graph
71. Unlocking the Power of Enterprise Search with LLM Technology. Glean. n.d. https://glean.com/blog/enterprise-search-llm-tech
72. Agentic Reasoning: The Future of Work AI. Glean. n.d. https://glean.com/blog/agentic-reasoning-future-ai
73. How We Built AI-Powered Workplace Search at Glean (webinar). Glean. n.d. https://www.glean.com/webinars/how-we-built-ai-powered-workplace-search-at-glean
74. SDS 951: Context Engineering, Multiplayer AI and Effective Search, with Dropbox's Josh Clemm. SuperDataScience. n.d. https://superdatascience.com/951
75. Dropbox Tech blog listing (Dash context engineering, feature store, KG/MCP/DSPy posts). Dropbox. 2025-2026. https://dropbox.tech/tag-results
76. Indexing Large Codebases for GitHub Copilot with Local RAG. xorets (Medium). n.d. https://medium.com/@xorets/indexing-large-codebases-for-github-copilot-with-local-rag-bdbf8472e21c
77. Long Context RAG Performance of LLMs. Databricks. 2024. https://databricks.com/blog/long-context-rag-performance-llms
78. From RAG to Fabric: Lessons Learned from Building Real-World RAGs at GenAIIC, Part 1. AWS. 2024. https://aws.amazon.com/blogs/machine-learning/from-rag-to-fabric-lessons-learned-from-building-real-world-rags-at-genaiic-part-1/
79. From RAG to Fabric: Lessons Learned from Building Real-World RAGs at GenAIIC, Part 2. AWS. 2024. https://aws.amazon.com/blogs/machine-learning/from-rag-to-fabric-lessons-learned-from-building-real-world-rags-at-genaiic-part-2
80. Retrieval Augmented Generation Options and Architectures on AWS. AWS Prescriptive Guidance. n.d. https://docs.aws.amazon.com/en_us/prescriptive-guidance/latest/retrieval-augmented-generation-options/introduction.html
81. Amazon Bedrock Knowledge Bases (User Guide). AWS. n.d. https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
82. Retrieval Augmented Generation in Azure AI Search. Microsoft. n.d. https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview
83. RAG Solution Design and Evaluation Guide. Microsoft Azure Architecture Center. n.d. https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide
84. Retrieval Augmented Generation (RAG) Guide. Google Cloud Blog. n.d. https://cloud.google.com/blog/products/ai-machine-learning/retrieval-augmented-generation-rag-guide
85. What Is Retrieval-Augmented Generation. Google Cloud Blog. n.d. https://cloud.google.com/blog/products/ai-machine-learning/what-is-retrieval-augmented-generation
86. Retrieval Augmented Generation with Foundation Models. Google Cloud Blog. n.d. https://cloud.google.com/blog/topics/developers-practitioners/retrieval-augmented-generation-foundation-models
87. Optimizing LLMs for Accuracy. OpenAI. n.d. https://platform.openai.com/docs/guides/optimizing-llm-accuracy
88. File Search (tools guide). OpenAI. n.d. https://platform.openai.com/docs/guides/tools-file-search
89. What Is Retrieval-Augmented Generation? NVIDIA Blog. 2023. https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
90. What Is Retrieval-Augmented Generation? IBM Research. 2023. https://research.ibm.com/blog/retrieval-augmented-generation-RAG
91. Introducing Contextual Retrieval. Anthropic. 2024. https://www.anthropic.com/news/contextual-retrieval
92. Introducing Citations on the Anthropic API. Anthropic. 2025. https://claude.com/blog/introducing-citations-api
93. Effective Context Engineering for AI Agents. Anthropic. 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
94. How We Built Our Multi-Agent Research System. Anthropic. 2025. https://www.anthropic.com/engineering/multi-agent-research-system
95. Building Effective Agents. Anthropic. 2024. https://www.anthropic.com/engineering/building-effective-agents
96. GraphRAG: Unlocking LLM Discovery on Narrative Private Data. Larson, Truitt (Microsoft Research). 2024. https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
97. GraphRAG: New Tool for Complex Data Discovery Now on GitHub. Microsoft Research. 2024. https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/
98. GraphRAG Auto-Tuning Provides Rapid Adaptation to New Domains. Microsoft Research. 2024. https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/
99. Introducing DRIFT Search: Combining Global and Local Search Methods. Microsoft Research. 2024. https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/
100. GraphRAG: Improving Global Search via Dynamic Community Selection. Microsoft Research. 2024. https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/
101. LazyGraphRAG: Setting a New Standard for Quality and Cost. Microsoft Research. 2024. https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
102. BenchmarkQED: Automated Benchmarking of RAG Systems. Edge, Trinh, Morales Esquivel, Larson (Microsoft Research). 2025. https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/
103. microsoft/graphrag (repository). Microsoft. n.d. https://github.com/microsoft/graphrag
104. GraphRAG Documentation. Microsoft. n.d. https://microsoft.github.io/graphrag/
105. Rerankers and Two-Stage Retrieval. Pinecone. n.d. https://pinecone.io/learn/series/rag/rerankers/
106. Pinecone Rerank v0 Announcement. Pinecone. n.d. https://www.pinecone.io/blog/pinecone-rerank-v0-announcement/
107. Rerank (docs). Pinecone. n.d. https://docs.pinecone.io/guides/inference/rerank
108. Chunking Strategies for LLM Applications. Pinecone. n.d. https://www.pinecone.io/learn/chunking-strategies/
109. Hybrid Search Explained. Weaviate. 2025. https://weaviate.io/blog/hybrid-search-explained
110. Hybrid Search (concepts). Weaviate Docs. n.d. https://docs.weaviate.io/weaviate/concepts/search/hybrid-search
111. Hybrid Search for Web Developers. Weaviate. n.d. https://weaviate.io/blog/hybrid-search-for-web-developers
112. Scaling ColPali to Billions of PDFs with Vespa. Jo Kristian Bergum (Vespa). n.d. https://blog.vespa.ai/scaling-colpali-to-billions/
113. Visual RAG in Practice. Thomas H. Thoresen (Vespa). n.d. https://blog.vespa.ai/visual-rag-in-practice/
114. Vespa.ai Announces Support for ColPali in RAG. Vespa. 2024. https://vespa.ai/vespa-ai-announces-support-for-colpali-in-retrieval-augmented-generation-rag/
115. ColPali Vespa Visual Retrieval Demo (About). Vespa / Hugging Face Spaces. n.d. https://vespa-engine-colpali-vespa-visual-retrieval.hf.space/about-this-demo
116. Transforming the Future of Information Retrieval with ColPali. Bonnie Chase (Vespa). n.d. https://blog.vespa.ai/Transforming-the-Future-of-Information-Retrieval-with-ColPali/
117. Late Chunking in Long-Context Embedding Models. Michael Günther, Han Xiao (Jina AI). 2024. https://jina.ai/news/late-chunking-in-long-context-embedding-models/
118. Elasticsearch Labs. Elastic. n.d. https://www.elastic.co/search-labs
119. Qdrant Articles. Qdrant. n.d. https://qdrant.tech/articles/
120. Improving Retrieval in RAG with Reranking. Unstructured. n.d. https://www.unstructured.io/blog/improving-retrieval-in-rag-with-reranking
121. Building Advanced RAG Apps with LlamaIndex. Zilliz. n.d. https://zilliz.com/blog/advanced-rag-apps-with-llamaindex
122. Retrieval Augmented Generation on Notion Docs via LangChain. Zilliz. n.d. https://zilliz.com/blog/retrieval-augmented-generation-on-notion-docs-via-langchain
123. Building Performant RAG Applications for Production. LlamaIndex. n.d. https://developers.llamaindex.ai/python/framework/optimizing/production_rag/
124. OpenAI Cookbook: Evaluating RAG Systems. LlamaIndex. 2023. https://blog.llamaindex.ai/openai-cookbook-evaluating-rag-systems-fe393c61fb93
125. Deconstructing RAG. LangChain. n.d. https://www.langchain.com/blog/deconstructing-rag
126. rag-from-scratch (repository). LangChain. n.d. https://github.com/langchain-ai/rag-from-scratch
127. Pinecone Rerank (retriever integration). LangChain Docs. n.d. https://docs.langchain.com/oss/python/integrations/retrievers/pinecone_rerank
128. Ragas Documentation. Ragas. n.d. https://docs.ragas.io/
129. DSPy. Stanford NLP. n.d. https://dspy.ai/
130. Haystack. deepset. n.d. https://haystack.deepset.ai/
131. LightRAG (repository). HKUDS. n.d. https://github.com/HKUDS/LightRAG
132. ColPali (repository). Illuin Technology. n.d. https://github.com/illuin-tech/colpali
133. RAGatouille (repository). Answer.AI. n.d. https://github.com/AnswerDotAI/RAGatouille
134. RAG Evaluation (Open-Source AI Cookbook). Hugging Face. n.d. https://huggingface.co/learn/cookbook/en/rag_evaluation
135. Optimizing Ragas to Evaluate a RAG Pipeline. Braintrust. n.d. https://braintrust.dev/docs/cookbook/recipes/SimpleRagas
136. Build a Simple RAG App. LangWatch. n.d. https://docs.langwatch.ai/cookbooks/build-a-simple-rag-app
137. Decreasing Hallucinations in RAG Pipelines. Future AGI. n.d. https://docs.futureagi.com/docs/cookbook/decrease-hallucination
138. Retrieval Augmented Generation (RAG) using LlamaIndex. AMD ROCm Blogs. n.d. https://rocm.blogs.amd.com/artificial-intelligence/rag-llamaindex/README.html
139. Patterns for Building LLM-based Systems & Products. Eugene Yan. 2023. https://eugeneyan.com/writing/llm-patterns/
140. AI Engineer 2023 Keynote: Building Blocks for LLM Systems. Eugene Yan. 2023. https://eugeneyan.com/speaking/ai-eng-summit
141. What We've Learned From A Year of Building with LLMs. Yan, Bischof, Frye, Husain, Liu, Shankar. 2024. https://applied-llms.org/
142. What We Learned from a Year of Building with LLMs (O'Reilly). Yan et al. 2024. https://oreilly.com/library/view/what-we-learned/9781098176716
143. Stop Saying RAG Is Dead. Hamel Husain. n.d. https://hamel.dev/notes/llm/rag/not_dead.html
144. Systematically Improving RAG Applications (course). Jason Liu (Maven). n.d. https://maven.com/applied-llms/rag-playbook
145. Why Your RAG System Is Broken, and How to Fix It with Jason Liu. TWIML AI Podcast. n.d. https://twimlai.com/podcast/twimlai/why-your-rag-pipeline-is-broken-and-how-to-fix-it
146. The Real Battle-Tested RAG Playbook. Decoding ML. n.d. https://decodingml.substack.com/p/the-real-battle-tested-rag-playbook
147. Building A Generative AI Platform. Chip Huyen. 2024. https://huyenchip.com/2024/07/25/genai-platform.html
148. RAG (tag archive). Simon Willison. n.d. https://simonwillison.net/tags/rag/
149. Production-Ready RAG (talk). Jerry Liu, AI Engineer. n.d. https://ai.engineer/talks/production-ready-rag
150. The Evolution of LinkedIn's Generative AI Tech Stack. ByteByteGo. n.d. https://blog.bytebytego.com/p/the-evolution-of-linkedins-generative
151. How DoorDash Mitigates Hallucinations in Its RAG System with a Two-Tier LLM Guardrail. Athina AI. n.d. https://hub.athina.ai/athina-originals/how-doordash-mitigates-hallucinations
152. RAG Examples. Evidently AI. n.d. https://www.evidentlyai.com/blog/rag-examples
153. How DoorDash's AI Cut Errors by 90%. Simform Newsletter. n.d. https://simformnewsletter.substack.com/p/how-doordash-ai-cut-errors-by-90
154. Agentic RAG: Context-Augmented OpenAI Agents. Cobus Greyling. n.d. https://cobusgreyling.substack.com/p/agentic-rag-context-augmented-openai
155. Anthropic Contextual Retrieval (news). InfoQ. 2024. https://www.infoq.com/news/2024/09/anthropic-contextual-retrieval/
156. Contextual Retrieval by Anthropic (tutorial). DataCamp. n.d. https://www.datacamp.com/tutorial/contextual-retrieval-anthropic
157. Responding to Anthropic's Contextual Retrieval for RAG Apps: Why Context Is NOT All You Need. Almond AI (Medium). 2024. https://medium.com/almond-ai/responding-to-anthropics-contextual-retrieval-for-rag-apps-why-context-is-not-all-you-need-af503985aa55
158. The Architect's Guide to Production RAG. Ragie. n.d. https://www.ragie.ai/blog/the-architects-guide-to-production-rag-navigating-challenges-and-building-scalable-ai
159. Practitioner RAG in Production. Enterprise DNA. n.d. https://enterprisedna.co/resources/blog/practitioner-rag-in-production
160. RAG in Production (book). Leanpub. n.d. https://leanpub.com/rag-production
161. RAG: Retrieval Augmented Generation, The Architecture of Reliable AI (book). INNOQ. n.d. https://www.innoq.com/en/books/retrieval-augmented-generation/
162. In-depth Understanding of RAG Query Transformation Optimization. James Li (dev.to). n.d. https://dev.to/jamesli/in-depth-understanding-of-rag-query-transformation-optimization-multi-query-problem-decomposition-and-step-back-27jg
163. Scaling RAGs: Data Engineering Hurdles in High-Performance AI. dev.to. n.d. https://dev.to/mabualzait/scaling-rags-data-engineering-hurdles-in-high-performance-ai-438g
164. Gen AI On-Call Copilot for Engineering Support (Uber case). ZenML. n.d. https://www.zenml.io/llmops-database/gen-ai-on-call-copilot-for-engineering-support
165. Enhanced Agentic RAG for Internal On-Call Support Copilot (Uber case). ZenML. n.d. https://www.zenml.io/llmops-database/enhanced-agentic-rag-for-internal-on-call-support-copilot
166. LLM-based Dasher Support Automation with RAG and Quality Controls (DoorDash case). ZenML. n.d. https://www.zenml.io/llmops-database/llm-based-dasher-support-automation-with-rag-and-quality-controls
167. Building and Scaling a Production Generative AI Assistant for Professional Networking (LinkedIn case). ZenML. n.d. https://www.zenml.io/llmops-database/building-and-scaling-a-production-generative-ai-assistant-for-professional-networking
168. RAG-Based Industry Classification System for Financial Services (Ramp case). ZenML. n.d. https://www.zenml.io/llmops-database/rag-based-industry-classification-system-for-financial-services
169. Enterprise-Grade RAG Systems for Legal AI Platform (Harvey case). ZenML. n.d. https://zenml.io/llmops-database/enterprise-grade-rag-systems-for-legal-ai-platform
170. Optimizing RAG Systems: Lessons from Production (AWS GenAIIC case). ZenML. n.d. https://www.zenml.io/llmops-database/optimizing-rag-systems-lessons-from-production
171. RAG with Knowledge Graphs for Customer Service QA by LinkedIn (lesson). Activeloop. n.d. https://courses.activeloop.ai/advanced-rag-and-graphrag/retrieval-augmented-generation-with-knowledge-graphs-for-customer-service-question-answering-by-linkedin-corporation
172. QueryGPT: Natural Language to SQL Using Generative AI. Uber Engineering. 2024. https://www.uber.com/en-JP/blog/query-gpt/
173. Unlocking Financial Insights with Finch: Uber's Conversational AI Data Agent. Uber Engineering. 2025. https://www.uber.com/en-JP/blog/unlocking-financial-insights-with-finch/
174. Hermes: A Text-to-SQL Solution at Swiggy. Swiggy Bytes. 2024. https://bytes.swiggy.com/hermes-a-text-to-sql-solution-at-swiggy-81573fb4fb6e
175. Swiggy Hermes Conversational AI (news). InfoQ. 2026. https://www.infoq.com/news/2026/01/swiggy-hermes-conversational-ai
176. Evolution of Hermes V3: Building a Conversational AI Data Analyst (Swiggy case). ZenML. n.d. https://www.zenml.io/llmops-database/evolution-of-hermes-v3-building-a-conversational-ai-data-analyst
177. A Scalable LLM Approach to Enhancing Chatbot Knowledge with User-Generated Content. DoorDash Engineering. 2025. https://careersatdoordash.com/blog/doordash-llm-chatbot-knowledge-with-ugc/
178. How DoorDash Leverages LLMs for Better Search Retrieval. DoorDash Engineering. n.d. https://careersatdoordash.com/blog/how-doordash-leverages-llms-for-better-search-retrieval/
179. Building DoorDash's Product Knowledge Graph with Large Language Models. DoorDash Engineering. 2024. https://doordash.engineering/2024/04/23/building-doordashs-product-knowledge-graph-with-large-language-models/
180. Building the Next Generation of Job Search at LinkedIn. LinkedIn Engineering. n.d. https://www.linkedin.com/blog/engineering/ai/building-the-next-generation-of-job-search-at-linkedin
181. Automated GenAI-Driven Search Quality Evaluation. LinkedIn Engineering. n.d. https://www.linkedin.com/blog/engineering/ai/automated-genai-driven-search-quality-evaluation
182. Building Dash: How RAG and AI Agents Help Us Meet the Needs of Businesses. Dropbox. 2024. https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users
183. Bringing AI-Powered Answers and Summaries to File Previews on the Web. Dropbox. n.d. https://dropbox.tech/machine-learning/bringing-ai-powered-answers-and-summaries-to-file-previews-on-the-web
184. Improving Pinterest Search Relevance Using Large Language Models. Pinterest Engineering. n.d. https://medium.com/pinterest-engineering/improving-pinterest-search-relevance-using-large-language-models-4cd938d4e892
185. Supercharging Discovery in Search with LLMs. Instacart Tech. 2024. https://tech.instacart.com/supercharging-discovery-in-search-with-llms-556c585d4720
186. Revamping Query Understanding with LLMs in E-commerce Search (Instacart case). ZenML. n.d. https://www.zenml.io/llmops-database/revamping-query-understanding-with-llms-in-e-commerce-search
187. Automation Platform v2: Improving Conversational AI at Airbnb. Airbnb Engineering. 2024. https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb
188. LLM-Friendly Knowledge Representation for Customer Support. Airbnb authors (COLING 2025 Industry). 2025. https://arxiv.org/pdf/2510.10331
189. Building a Video Q&A System with RAG and Speaker Detection (Vimeo case). ZenML. n.d. https://www.zenml.io/llmops-database/building-a-video-q-a-system-with-rag-and-speaker-detection
190. Navigating Fair Housing Guardrails in LLMs. Zillow Tech. 2024. https://www.zillow.com/tech/navigating-fair-housing-guardrails-in-llms/
191. Leveraging RAG-Powered LLMs for Analytical Tasks. Grab Engineering. 2024. https://engineering.grab.com/transforming-the-analytics-landscape-with-RAG-powered-LLM
192. Improving Agent with Semantic Search. Cursor. 2025. https://cursor.com/blog/semsearch
193. Enhancing AI Coding Agent Performance with Custom Semantic Search (Cursor case). ZenML. n.d. https://www.zenml.io/llmops-database/enhancing-ai-coding-agent-performance-with-custom-semantic-search
194. Notion-Based RAG Slackbot Hackathon. DataDome Engineering. n.d. https://datadome.co/engineering/notion-based-rag-slackbot-hackathon/
195. Thomson Reuters Launches CoCounsel Knowledge Search. Thomson Reuters. 2025. https://thomsonreuters.com/en/press-releases/2025/july/thomson-reuters-launches-cocounsel-knowledge-search-an-ai-powered-experience-to-manage-content-and-institutional-knowledge
196. Sufficient Context: A New Lens on Retrieval Augmented Generation Systems. Joren, Zhang, Ferng, Juan, Taly, Rashtchian (Google). 2024/2025. https://arxiv.org/abs/2411.06037
197. Sufficient Context (publication page). Google Research. 2025. https://research.google/pubs/sufficient-context-a-new-lens-on-retrieval-augmented-generation-systems/
198. Announcing General Availability of Amazon Bedrock Knowledge Bases GraphRAG with Amazon Neptune Analytics. AWS. 2025. https://aws.amazon.com/blogs/machine-learning/announcing-general-availability-of-amazon-bedrock-knowledge-bases-graphrag-with-amazon-neptune-analytics/
199. RAG Engine Overview (Vertex AI). Google Cloud. n.d. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-overview
200. Hybrid Search Overview (Azure AI Search). Microsoft. n.d. https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview
201. AI Search Query Engine Performance (whitepaper). Microsoft. n.d. https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/ai-search-query-engine-performance-whitepaper.pdf
202. Long Context RAG Capabilities of OpenAI o1 and Google Gemini. Databricks. 2024. https://www.databricks.com/blog/long-context-rag-capabilities-openai-o1-and-google-gemini
203. What Is Mosaic AI Agent Evaluation? Databricks Docs. n.d. https://docs.databricks.com/generative-ai/agent-evaluation/index.html
204. RAFT: A New Way to Teach LLMs to Be Better at RAG. Microsoft Tech Community. 2024. https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/raft-a-new-way-to-teach-llms-to-be-better-at-rag/ba-p/4084674
205. What Is Semantic Reranking and How to Use It? Elastic Search Labs. n.d. https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-1
206. Qdrant Launches Combined Vector and Keyword Search for RAG and AI Apps. Blocks and Files. 2024. https://blocksandfiles.com/2024/07/02/qdrant-launches-combined-vector-and-keyword-search-for-rag-and-ai-apps/
207. Snippet Selection and URL Ranking in DeepSearch/DeepResearch. Jina AI. 2025. https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/
208. Enhancing RAG Pipelines with Re-Ranking. NVIDIA Technical Blog. n.d. https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/
209. An Easy Introduction to Multimodal Retrieval-Augmented Generation. NVIDIA Technical Blog. n.d. https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation/
210. The GraphRAG Manifesto: Unlock Better GenAI Results with Knowledge Graphs. Philip Rathle (Neo4j). 2024. https://neo4j.com/?p=332054
211. RAG Is Dead, Long Live Agentic Retrieval. LlamaIndex. 2025. https://www.llamaindex.ai/blog/rag-is-dead-long-live-agentic-retrieval
212. Query Transformations. LangChain. 2023. https://www.langchain.com/blog/query-transformations
213. Self-Reflective RAG with LangGraph. Ankush Gola (LangChain). 2024. https://blog.langchain.com/agentic-rag-with-langgraph
214. open_deep_research (repository). LangChain. n.d. https://github.com/langchain-ai/open_deep_research
215. Agentic RAG (LangGraph tutorial). LangChain. n.d. https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
216. Self-RAG (project page). Asai et al. 2023. https://selfrag.github.io/
217. Search-R1 (repository). Jin et al. 2025. https://github.com/PeterGriffinJin/Search-R1
218. RAG_Techniques (repository). Nir Diamant. n.d. https://github.com/NirDiamant/RAG_Techniques
219. Awesome-RAG (repository). Danielskry. n.d. https://github.com/Danielskry/Awesome-RAG
220. LLMs in Production (repository). primaprashant. n.d. https://github.com/primaprashant/llms-in-production
221. What We Learned from a Year of Building with LLMs (Part I). O'Reilly Radar. 2024. https://oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/
222. RAG Evaluation Is Broken! (And How to Fix It). Yuval Belfer, Niv Granot (AI Engineer World's Fair). 2025. https://ai.engineer/speakers/niv-granot
223. The State of AI-Powered Search and Retrieval. Frank Liu (AI Engineer World's Fair). 2025. https://ai.engineer/speakers/frank-liu
224. RAG Is More Than Just Embedding Search. Jason Liu (Instructor docs, mirror). 2023. https://git.kennethreitz.org/kennethreitz/instructor/src/commit/4054720004fa533b1e5531bbbb26ba0a23d9ea49/docs/blog/posts/rag-and-beyond.md
225. Hugging Face Researchers Aim to Build an Open Version of OpenAI's Deep Research Tool. TechCrunch. 2025. https://techcrunch.com/2025/02/04/hugging-face-researchers-aim-to-build-an-open-version-of-openais-deep-research-tool
226. RAG (deep dive). The Pragmatic Engineer. n.d. https://newsletter.pragmaticengineer.com/p/rag
227. RAG-Fusion: A New Take on Retrieval-Augmented Generation. 2024. https://arxiv.org/abs/2402.03367
228. REFRAG: Rethinking RAG-based Decoding. 2025. https://arxiv.org/pdf/2509.01092
229. VideoRAG: Retrieval-Augmented Generation over Video Corpus. 2025. https://arxiv.org/abs/2501.05874
230. Don't Do RAG: When Cache-Augmented Generation Is All You Need for Knowledge Tasks. 2024. https://arxiv.org/abs/2412.15605
231. A-RAG (agentic RAG exposing keyword, semantic and chunk-level retrieval interfaces; title abbreviated). 2026. https://arxiv.org/abs/2602.03442
232. Embedding-based Retrieval in Facebook Search. Huang et al. (Facebook). 2020. https://arxiv.org/abs/2006.11632
233. Embedding-based Retrieval at Scribd. Scribd Tech. 2021. https://tech.scribd.com/blog/2021/embedding-based-retrieval-scribd.html
234. Query Expansion (cookbook). Haystack (deepset). n.d. https://haystack.deepset.ai/cookbook/query-expansion
235. File Search (tools guide, Responses API). OpenAI. n.d. https://developers.openai.com/api/docs/guides/tools-file-search
236. Introducing the Model Context Protocol. Anthropic. 2024. https://www.anthropic.com/news/model-context-protocol
237. Writing Effective Tools for AI Agents, Using AI Agents. Anthropic. 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
238. Moving to GraphRAG 1.0: Streamlining Ergonomics for Developers and Users. Microsoft Research. 2024. https://www.microsoft.com/en-us/research/blog/moving-to-graphrag-1-0-streamlining-ergonomics-for-developers-and-users/
239. HippoRAG (repository, includes HippoRAG 2). OSU NLP Group. n.d. https://github.com/OSU-NLP-Group/HippoRAG
240. From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (HippoRAG 2). Gutiérrez et al. (OSU), ICML. 2025. https://arxiv.org/abs/2502.14802
241. Search-o1: Agentic Search-Enhanced Large Reasoning Models. Li et al. 2025. https://arxiv.org/abs/2501.05366
242. Introducing Deep Research. OpenAI. 2025. https://openai.com/index/introducing-deep-research/
