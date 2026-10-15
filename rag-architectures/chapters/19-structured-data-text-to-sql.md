# 19. Structured-Data RAG: Text-to-SQL Architectures, Semantic Layers, Table QA and Schema Retrieval

## Overview

Most RAG systems in this atlas retrieve passages of unstructured text. A large share of enterprise questions, though, have answers that sit in relational tables: "How many trips did we complete in Seattle last week?", "What was weekly active usage for feature X by country?", "Which accounts churned after the price change?" For these, retrieving prose is the wrong move. The system has to retrieve the right *schema* and *business context*, write an executable query, run it against the warehouse, and report the result. This family of designs is usually called text-to-SQL, NL2SQL or conversational analytics. Seen through the RAG lens, it is structured-data RAG, and the retrieval target is metadata, not documents.

Three facts shape every production design in this chapter.

1. **The retrieval problem is schema selection.** A data warehouse can hold hundreds of thousands of tables. Pinterest's later context-layer work cites roughly 400,000 ungoverned tables [5], and LinkedIn describes cutting "millions" of tables down to a few thousand before retrieval [6][9]. Nothing generates SQL correctly until it has found the right 3-10 tables and the right columns. Uber, Pinterest and LinkedIn all report that table selection, not SQL syntax, was the dominant failure mode [1][3][7].
2. **The context that matters is mostly not in the DDL.** Join keys, metric definitions, which of five near-duplicate tables is "certified", what `status = 3` means: all of this lives in query logs, wikis, dbt models and analysts' heads. Production systems invest heavily in harvesting it, as table summaries (Pinterest), knowledge graphs (LinkedIn), workspaces (Uber), semantic models (Snowflake) or knowledge stores (Databricks).
3. **Benchmarks overstate real-world accuracy.** Systems reach about 80% execution accuracy on BIRD [35][37] and above 90% on Spider 1.0 [29]. The enterprise-style Spider 2.0 started with a best agent solving only 21.3% of tasks [29]. Vendor-internal BI benchmarks show the same gap: GPT-4o drops to 51% on Snowflake's internal set [11], and a leading coding agent scores 32% on Databricks' internal data-analysis tasks [18].

The chapter first covers the generic architecture, then the major industrial systems (Uber QueryGPT, Pinterest, LinkedIn SQL Bot, Swiggy Hermes, Snowflake Cortex Analyst, Databricks Genie, Google Conversational Analytics), then the semantic-layer approach, schema-linking and candidate-selection research (DIN-SQL, CHESS, CHASE-SQL, RL-trained generators), table QA without SQL (TableRAG, Chain-of-Table), and finally the benchmarks.

## 19.1 The canonical structured-data RAG pipeline

Across the systems surveyed, a recurring decomposition appears:

```
 User question (NL)
        |
        v
+-------------------+   domain / workspace / semantic model routing
| Intent & routing  |---------------------------------------------+
+-------------------+                                             |
        |                                                         v
        v                                          +---------------------------+
+-------------------+   vector + keyword search    | Metadata / context store  |
| Schema retrieval  |<---------------------------->|  - table & column docs    |
| (tables, columns) |                              |  - LLM table summaries    |
+-------------------+                              |  - historical SQL / logs  |
        |                                          |  - verified Q->SQL pairs  |
        v                                          |  - metrics, joins, synonyms|
+-------------------+                              |  - sampled column values  |
| LLM re-rank /     |                              +---------------------------+
| column pruning    |
+-------------------+
        |
        v
+-------------------+  few-shot examples, business rules, dialect hints
| SQL generation    |  (single shot, multi-candidate, or agentic loop)
+-------------------+
        |
        v
+-------------------+  EXPLAIN / dry-run / syntax check / unit tests
| Validate & repair |----> error? ---> regenerate (bounded retries)
+-------------------+
        |
        v
+-------------------+
| Execute + explain |---> result table, chart, NL summary, the SQL itself
+-------------------+
```

The industrial systems differ mainly in (a) what goes into the context store and who curates it, (b) how many LLM calls are spent on narrowing context before generation, and (c) whether a human stays in the loop. Most expose the generated SQL for editing, and several ask users to confirm the selected tables.

## 19.2 Uber QueryGPT

### Problem

Uber's data platform serves about 1.2 million interactive queries per month, and operations teams account for roughly 36% of them [1][2]. Writing a query means finding the right tables among a very large catalog and understanding their columns. Uber estimated each query took about 10 minutes to author and that QueryGPT could produce "sufficiently reliable" queries in about 3 minutes [1]. Secondary write-ups extrapolate this to roughly 140,000 hours saved per month. That figure is a third-party calculation, not an Uber-published number, so treat it as approximate [2].

### Evolution

QueryGPT began as a proposal at Uber's Generative AI Hackdays in May 2023 [1][2]. The first version was a plain RAG system. It vectorized the user prompt, ran a similarity search over SQL samples and table schemas, and put the top results into a few-shot prompt. Sources report more than 20 iterations of the algorithm before the current design [2]. Uber described the design publicly in September 2024 [1].

The naive version failed for predictable reasons. Similarity search between a natural-language question and raw SQL/DDL was a poor relevance signal. Some Tier-1 tables have more than 200 columns and can consume 40-60K tokens of the request, so large schemas exhausted context budgets [1].

### Architecture

```
 question
    |
    v
+-----------------+     maps question -> business domain(s)
|  Intent Agent   |---> "system" workspaces (Mobility, Ads, Core Services, ...)
+-----------------+     or user-defined "custom" workspaces
    |
    v  (RAG search radius narrowed to the selected workspaces)
+-----------------+     proposes tables; user ACKs or edits the list
|  Table Agent    |<--> human in the loop
+-----------------+
    |
    v
+-----------------+     LLM call removes irrelevant columns from each schema
| Column Prune    |     -> "skinnier" schemas, fewer tokens, lower latency/cost
|     Agent       |
+-----------------+
    |
    v
+-----------------+     few-shot SQL samples from workspace + pruned schemas
| SQL generation  |---> SQL + explanation shown to user
+-----------------+
```

Key techniques [1][2]:

- **Workspaces.** These are curated collections of SQL samples and table schemas per business domain, such as Mobility, Ads and Core Services. Users can also create custom workspaces. Picking a domain first lets Uber "drastically narrow the search radius for RAG" [1].
- **Intent agent.** An LLM call classifies the question into one or more workspaces before any retrieval runs.
- **Table agent with human acknowledgement.** Users reported that QueryGPT often chose the wrong tables. The table agent now shows the proposed tables, and the user can ACK them or edit the list before generation [1].
- **Column prune agent.** It was added after a wider rollout, when some requests hit token limits even with GPT-4 Turbo's 128K context window. Uber says pruning "massively improved" token size, cost per call and latency. No exact percentages were published [1].
- **Prompt enhancer.** Uber notes the need for a "prompt enhancer" or "prompt expander" to "massage" short user questions into context-rich ones before they reach the LLM [1].

### Lessons and failure modes

Uber's conclusions emphasize that hallucinations (non-existent tables or columns) and very large schemas remain open problems [1]. Third-party write-ups say Uber evaluates components separately (intent accuracy, table selection, execution success, output quality) against a set of manually verified queries. That detail is secondhand [2]. The main design lesson is to narrow first: route to a domain, then select tables with a human check, then prune columns, and only then generate.

## 19.3 Pinterest: RAG table selection in Querybook, then a governed context layer

### Problem

Pinterest analysts write SQL in Querybook, Pinterest's open-source query IDE. The first text-to-SQL feature asked users to name the tables. The model then received the question and the selected table schemas, and returned SQL [3][4]. In practice, users often did not know which tables to use, and that became the bottleneck.

### Architecture (2024 design)

```
OFFLINE INDEXING                                ONLINE
------------------------------------------      ---------------------------------
metadata store --> table schema ---+            question
recent queries on the table -------+--> LLM        |
                                   |   summary     v
                                   v            embed --> similarity search
                    table-summary embeddings ----+   (table + query indices)
                    query-summary embeddings ----+          |
                         (OpenSearch vector store)          v
                                                     top-N candidate tables
                                                            |
                                                            v
                                                 LLM picks top-K using summaries
                                                            |
                                                            v
                                                 user confirms / edits tables
                                                            |
                                                            v
                                                 SQL generation (streamed) in IDE
```

Key techniques [3][4]:

- **Index only top-tier tables.** Pinterest restricted the index to high-quality, top-tier tables rather than the whole warehouse.
- **LLM-written table summaries.** For each table, an offline job pulls the schema from the metadata store, gathers recent sample queries that use the table, packs as many as fit into a summarization prompt, and asks an LLM for a summary. The summary is embedded into OpenSearch.
- **Query summaries.** Historical queries are summarized and embedded the same way. A question can then match a table through "what people asked of it", not only through its column names.
- **Weighted scoring.** Table-summary hits are weighted more heavily than query-summary hits [4].
- **Two-stage retrieval.** Vector search returns the top N candidates. An LLM then selects the top K by reading the question alongside the table summaries. Pinterest reported that this LLM re-selection step improved results [3].

### Numbers

- **Search hit rate.** Without table documentation in the embeddings, the hit rate was 40%. Performance "increased linearly with the weight placed on table documentation" up to 90% [3]. This is one of the clearest public data points that metadata quality dominates structured RAG.
- **First-shot acceptance.** Acceptance of generated SQL on the first try rose from about 20% to above 40% [3]. Pinterest notes that most generated queries still take several rounds of human or AI iteration before they are final.
- **Task completion speed.** A secondary summary reports a 35% improvement in task completion speed for SQL writing [4].

### 2025-2026: the analytics agent and a governed context layer

Later Pinterest work, described in a DataHub customer story, builds its analytics agent on a centralized context layer. The layer combines enforced data governance, AI-generated documentation and "intent-indexed" query history, and it starts from roughly 400,000 ungoverned tables [5]. The vendor reports that the agent sees about 10x the usage of any other internal tool and that AI-generated documentation removed roughly 70% of manual documentation work [5]. These are vendor-reported figures. Pinterest's arc shows a pattern this chapter keeps finding: a retrieval trick (summaries in a vector index) first, then a data-governance program, because retrieval quality is capped by the quality of the metadata.

## 19.4 LinkedIn SQL Bot

### Problem

LinkedIn's SQL Bot sits inside DARWIN, LinkedIn's data science platform, and lets employees self-serve data insights [6][7]. The constraints: a huge and constantly changing table catalog, many near-duplicate or deprecated datasets, and a population of users who cannot judge whether SQL is correct.

### Architecture

The December 2024 engineering post and the 2025 paper "Text-to-SQL for Enterprise Data Analytics" [7] describe the system:

```
                 +------------------- knowledge graph --------------------+
                 | nodes: tables, columns, owners, domains, metrics,      |
                 |        historical queries, wiki docs, code             |
                 | sources: DataHub-style metadata, query logs, wikis, git|
                 +--------------------------------------------------------+
                                  ^                     ^
question -> intent classification |                     |
              |                   | EBR over name,      | example queries
              v                   | descriptions, tags  |
      +----------------+          |                     |
      | candidate gen. |----------+---------------------+
      | ~20 tables     |  (a) embedding retrieval filtered by access popularity
      +----------------+  (b) tables used in retrieved example queries
              |           (c) tables named explicitly in the question
              v
      +----------------+
      | LLM re-rank    |---> ~7 tables, then relevant columns (two tiers)
      +----------------+
              |
              v
      +----------------+     writes SQL + stated assumptions + explanation
      | Query writer   |
      +----------------+
              |
              v
      +----------------+     validates; fixes hallucinated columns, syntax errors
      | Query fixer    |---> loop
      +----------------+
              |
              v
       chat UI in DARWIN ("Fix with AI", table/column explanations, follow-ups)
```

Key techniques [6][7][8][9]:

- **Popularity pre-filter.** Access popularity cuts millions of tables to a few thousand before any embedding retrieval runs [9]. This is cheap and very effective, because unused tables are rarely the right answer.
- **Dataset certification.** Domain experts nominated the key tables in their areas and supplied mandatory table descriptions and optional field descriptions. The embedding index covers table names, descriptions and tags [9].
- **Three-source candidate generation.** Candidates come from embedding retrieval, tables referenced in retrieved example queries, and tables the user names. Around 20 candidates go to an LLM re-ranker that narrows them to about 7 tables, then selects columns [9].
- **Knowledge graph as context store.** It indexes database metadata, historical query logs, wikis and code [7].
- **Agentic writer and fixer.** The writer is built on LangChain and LangGraph and documents its assumptions [6]. A fixer agent self-corrects hallucinations and syntax errors [7].
- **Intent routing.** Different question types (find a table, explain a table, write a query, fix a query) go to specialized handlers [6][9].
- **Five deployment strategies** from LinkedIn's Trino Summit 2024 talk: quality metadata, query validation, interactive chat UX, self-serve customization, and ongoing benchmarking [8].

### Evaluation and numbers

The paper reports more than 300 weekly users. On an internal benchmark of 133 questions across 10 product areas (167 ground-truth tables; 60% of questions have more than one valid answer), experts judged 53% of responses correct or close to correct [7]. Of the 124 questions experts rated, 77% scored 3 or higher, meaning the answer at least helped identify the right tables and columns [7]. The blog-era survey reported that about 95% of users rated query accuracy as "passes" or better [6]. That is a satisfaction metric and is not comparable to the 53% expert-judged correctness. The pairing is worth remembering: user satisfaction can be high even when strict correctness is near 50%, because the tool speeds up table discovery even when its SQL is imperfect.

LinkedIn measures table recall and column recall (the share of ground-truth tables and columns that appear in the bot's query), plus human and LLM-as-judge scores [7]. Recall-style metrics are useful because enterprise questions often have several valid SQL answers, which makes exact-match and even execution-match too strict.

## 19.5 Swiggy Hermes

Swiggy's Hermes is a useful mid-scale reference [40]. In V1, users pasted their own metadata into the prompt. V2 moved from GPT-3.5 to a RAG design on GPT-4o and split the company into "charters" (business units). Each charter has its own metadata and knowledge base, so retrieval only searches the relevant charter's tables and columns. This is the same idea as Uber's workspaces. InfoQ reports that Hermes V3 raised SQL generation accuracy from 54% to 93%. It does this by retrieving a curated set of previously executed queries and their prompts with vector similarity, keeping conversational context, and using a ReAct-style orchestrator for intent parsing, metadata lookups, SQL generation and optional clarifying questions [40]. Reported usage is hundreds of users running several thousand queries, with average turnaround under 2 minutes. These figures come from secondary summaries.

## 19.6 Snowflake Cortex Analyst (and Cortex Agents)

### Problem and stance

Snowflake's position is that "given only the raw schema, it's challenging for any analyst to write SQL accurately" [10]. Cortex Analyst therefore depends on an author-provided **semantic model**, which narrows both the retrieval problem and the generation problem.

### Semantic model

The semantic model is a YAML file. Newer deployments can instead use native semantic views, which Snowflake now recommends while still supporting YAML [13][16]. It contains one or more **logical tables**, each mapped to a base table, with these sections [13]:

- `dimensions`: categorical columns, with descriptions and synonyms
- `time_dimensions`: date and timestamp columns
- `measures` / `facts`: numeric columns, optionally with a `default_aggregation`
- `filters`: named, commonly used predicates
- relationships (join paths) between logical tables
- `verified_queries`: the **Verified Query Repository (VQR)**, a list of (name, question, SQL, verification metadata) entries. The SQL must use the logical names, not physical ones [14].

At inference time, Cortex Analyst uses the semantic-model metadata (names, types, descriptions) only for SQL generation [15]. When a question resembles a verified question, the system draws on the matching verified SQL [14]. The VQR is therefore a few-shot retrieval index that domain experts curate.

### Agentic architecture

Snowflake's engineering post (August 2024) describes an agentic system built on a collection of LLMs, including Meta Llama and Mistral models. Multiple agents interact, with guardrails at every step to prevent hallucinations [10]. Public material describes the flow in broad terms: classify the question (answerable, ambiguous, out of scope), enrich context from the semantic model and VQR, generate SQL, check and correct errors, and return SQL plus an explanation. The per-agent internals have not been published in detail.

```
question --> [classify: answerable? ambiguous? off-topic?] --> clarify / refuse
                 |
                 v
        [context enrichment: semantic model (logical tables, measures,
         dimensions, filters, joins) + similar VQR entries]
                 |
                 v
        [SQL generation (several LLMs)] --> [error detection & correction]
                 |
                 v
        SQL over logical names --> compiled to physical SQL --> Snowflake
```

### Numbers (all Snowflake-internal)

- On an internal set of 150 BI-style questions (sales, marketing, finance) in three difficulty tiers (filtering, aggregation, trends), Cortex Analyst reaches 90%+ SQL accuracy. Single-prompt GPT-4o scores 51% on the same set, so Cortex Analyst is "close to 2x" more accurate, and it is about 14% more accurate than another text-to-SQL product [10][11].
- A later post on agentic semantic-model improvement reports an average of 57% for vanilla Claude 3.5 Sonnet versus 78% for Cortex Analyst across four datasets [12].
- Snowflake argues that public benchmarks overstate real-world performance. Their questions are simpler and their schemas cleaner than typical BI schemas, and they underrepresent time-series tables [11].

### Evaluation tooling and evolution

Snowsight can run evaluations over a semantic view. Verified queries serve as ground truth, and a verified query under evaluation is temporarily removed from the semantic view so it cannot leak into the answer [11]. Snowflake has since positioned **Cortex Agents** as the successor. Cortex Agents adds agentic orchestration, self-correction and a unified interface across structured data (Analyst) and unstructured data (Cortex Search), and existing semantic views work unchanged [16]. On the model side, Snowflake's **Arctic-Text2SQL-R1** is trained with GRPO using execution correctness as a simple reward. Its 32B model reported 71.83% on the BIRD leaderboard, and the 14B model was described as the first open model above 70% [34].

## 19.7 Databricks AI/BI Genie (Genie Agents)

### Architecture

Genie became generally available on 12 June 2025 [17]. An author configures a **Genie space**, renamed "Genie Agent" in newer docs [20], with:

- Unity Catalog tables or views, which supply schema, comments, lineage and usage signals [21]
- **Knowledge store**: Databricks calls it "a living semantic model that Genie consults each time it answers a question" [17]. It holds column-level synonyms and sampled values, and table-level context such as primary/foreign-key joins and certified metrics.
- **General instructions**: natural-language business rules
- **Example SQL**: these guide generation and are not necessarily executed
- **Trusted assets**: parameterized SQL queries and Unity Catalog functions registered as trusted. Answers that come from them are labeled "trusted" in the UI [17].
- **Benchmarks**: curated questions with expected SQL, used only for evaluation and never fed back as context [20]

The 2024 Data + AI Summit material and later docs describe Genie as a compound AI system. Multiple LLM calls, retrievers and validators work together, and each answer comes with an explanation of how it was derived [17][21].

```
                   +--------------- Genie space / agent -----------------+
question --------> | planner -> knowledge search (semantic indices over   |
                   |            UC assets, knowledge store, examples)      |
                   |        -> SQL generation (multiple LLMs / sub-agents) |
                   |        -> parallel reasoning paths + judge/aggregate  |
                   |        -> trusted-asset match? -> "trusted" label     |
                   +-------------------------------------------------------+
                                 |
                                 v
               Databricks SQL warehouse -> table/chart + explanation
               (Agent mode: decompose -> many SQL queries -> report)
```

### 2025-2026 techniques and numbers

Databricks' research post "Pushing the Frontier for Data Agents with Genie" names three techniques [18]:

1. **Specialized knowledge search.** Semantic search indices are built over existing data assets to improve asset discovery. A summary reports table-search gains of up to 40% (approximate).
2. **Parallel thinking.** Data questions have no unit tests to verify against, so Genie samples several reasoning paths and aggregates them into one answer.
3. **Multi-LLM sub-agents.** Different models handle planning, search, code generation and judging.

On an internal benchmark of real-world analysis tasks, these raised accuracy relative to a leading coding agent from 32% to over 90%, at lower cost and latency [18]. The October 2026 Genie One / Genie Agents / Genie Ontology launch reports 84.5% first-attempt accuracy on a 28-question internal enterprise benchmark. The strongest general-purpose coding agent scored 52.4% and the weakest 25%, and Genie was about 2x faster [19]. **Agent mode** breaks complex questions into sub-tasks, runs multiple SQL queries and returns a structured report. Agent-mode benchmarks are graded by an LLM judge, while chat-mode benchmarks compare results against reference SQL [20]. All of these numbers are internal and use small question sets.

## 19.8 Google Conversational Analytics (BigQuery and Looker)

Google's Conversational Analytics API exposes a **context retrieval tool**. For BigQuery it pulls schema plus table and column descriptions from Dataplex. For Looker it reads the LookML model, which acts as the semantic layer. Both feed one shared NL2Query engine [22]. The design point: when a semantic layer exists (LookML), the agent generates queries against it rather than raw SQL. When one does not exist, catalog descriptions stand in for it.

## 19.9 Semantic layers: generating against metrics, not tables

A semantic layer (dbt Semantic Layer/MetricFlow, LookML, Cube, Snowflake semantic views, Databricks metric views and knowledge store) defines entities, joins, dimensions and metrics once. The LLM's job then shrinks to choosing a metric, dimensions, filters and a time grain, and a deterministic compiler writes the SQL. This removes whole classes of errors: wrong join paths, double counting from fan-out joins, and private re-definitions of "active user".

Evidence:

- **data.world benchmark (2023).** Sequeda, Allemang and Jacob built an insurance-domain benchmark: 13 tables from a real industry model and 43 questions. They showed that querying through a knowledge-graph representation of the schema substantially improved GPT-4 accuracy [25]. The follow-up paper summarizes the earlier result as an improvement "from 16% to 54%" [26].
- **dbt Labs (2023 and 2026).** Using the same questions, dbt Labs reported that the semantic layer answered about 83% of questions [24]. Delphi later reported 100% using Cube as the semantic layer [24]. dbt's 2026 rerun with newer models found that schema modeling alone raised plain text-to-SQL accuracy substantially (64.5% to 90.0% in one summary), and that semantic-layer grounding added several more points. Summaries cite Claude Sonnet 4.6 going from 90.0% to 98.2% and GPT-5.3-Codex from 84.1% to 100% [23]. These figures come from secondary summaries of a small question set (reported as 11 questions), so treat them as approximate.
- **Snowflake.** The 57% to 78% average lift from adding a semantic model is in effect the same experiment run by a vendor [12].

Takeaway: semantic layers trade coverage for correctness. Questions that fit the modeled metrics become close to deterministic. Questions outside them need a fallback to raw text-to-SQL, which is why Cortex Analyst, Genie and Looker all combine curated objects with free-form generation.

## 19.10 Research systems: schema linking, decomposition and candidate selection

### DIN-SQL (2023)

DIN-SQL decomposes the task into **schema linking**, **query classification** (easy / non-nested complex / nested complex), **SQL generation** with class-specific prompts, and **self-correction** [30]. It reached 85.3% execution accuracy on the Spider holdout test set (previous SOTA 79.9) and 55.9% on BIRD, improving few-shot prompting by about 10 points across three LLMs [30]. It established the decompose-then-correct template that most industrial pipelines echo.

### CHESS (2024)

CHESS (Stanford and collaborators) uses four agents [31]:

```
Information Retriever --> Schema Selector --> Candidate Generator --> Unit Tester
 (keywords, LSH over      (prunes hundreds     (writes + revises      (NL unit tests
  DB values, vector DB     of columns to         SQL using execution    pick best
  over catalog docs)       usually < 10)         feedback)              candidate)
```

Two ideas carried over to production. The first is **value retrieval**: locality-sensitive hashing over millions of database values, so "NYC" in a question can be matched to `'New York'` in a column. The second is **aggressive schema pruning** down to fewer than ten columns. A third-party review reports 71.10% on the BIRD test set and a large reduction in LLM calls compared with prior agentic approaches. Both figures are approximate and should be checked against the paper.

### "The Death of Schema Linking?" (2024)

Maamari et al. argue that strong reasoning models can find the relevant schema elements themselves when the whole schema fits in context. Their pipeline skips explicit schema linking in that case and reached 71.83% on BIRD with a fine-tuned GPT-4o [33]. This does not contradict the industrial systems: a single BIRD database fits in context, and an enterprise catalog of 10^5 tables does not. Table *retrieval* stays mandatory at enterprise scale. Fine-grained *column* linking becomes optional once the candidate set is small.

### CHASE-SQL (2024)

Google Cloud and Stanford's CHASE-SQL generates diverse candidates through three generators: divide-and-conquer decomposition, chain-of-thought based on query execution plans, and instance-aware synthetic few-shot examples. A fine-tuned binary selection model then compares candidates pairwise [32]. It reported 73.0% on the BIRD test set, first on the leaderboard at the time (ICLR 2025).

### Test-time scaling and RL (2025-2026)

- **Agentar-Scale-SQL** orchestrates test-time scaling (many candidates, iterative refinement, selection) and reported 81.67% execution accuracy on the BIRD test set, ranked first as of September 2025 [35].
- **Arctic-Text2SQL-R1** (Snowflake) shows that GRPO with a plain execution-correctness reward, without elaborate reward shaping, produces strong open models [34].
- **Gemini-SQL2** was reported in June 2026 at 80.04% on BIRD's single-model track, with no public model card or technical report at the time [37]. **ReViSQL** reports exceeding BIRD's 92.96% proxy human score on the Mini-Dev subset [36]. That is a different split from the main test set and should not be read as "BIRD solved".

The research trajectory is: schema linking, then decomposition, then multi-candidate generation with learned selection, then RL on execution reward. Production systems borrow selectively, mainly value retrieval, pruning, self-correction and, increasingly, parallel sampling with judging (Databricks [18]).

## 19.11 Table QA without SQL: TableRAG and Chain-of-Table

Not every structured question goes through a database engine. Some arrive with a single large table (a spreadsheet, a CSV, an HTML table) or need reasoning that SQL expresses badly.

- **TableRAG** (Google, NeurIPS 2024) targets tables of up to millions of tokens [38]. Feeding the whole table to the LM causes positional bias and context overflow. TableRAG instead (1) expands the question into separate schema queries and cell queries, (2) runs **schema retrieval** over column names and metadata with a pre-trained encoder, and (3) runs **cell retrieval** over a database of distinct column-value pairs. A *cell-encoding budget* keeps only the most frequent values when a column has too many distinct values. The authors built the million-token benchmarks ArcadeQA and BirdQA (derived from Arcade and BIRD-SQL) plus an augmented TabFact, and report state-of-the-art results with much shorter prompts [38].
- **Chain-of-Table** (Google, ICLR 2024) has the LLM iteratively plan table operations (add column, select rows, group, sort), apply them, and feed the transformed table into the next step. The evolving table *is* the reasoning chain, and the final table is used to answer [39]. It reported state-of-the-art results on WikiTQ, FeTaQA and TabFact with PaLM 2-S and GPT-3.5 backbones.

```
TableRAG                                      Chain-of-Table
---------                                     --------------
question -> expand -> schema queries          table T0, question
                    -> cell queries             | plan op1 (e.g. f_select_row)
schema retrieval  (column embeddings)           v
cell retrieval    (distinct col=value pairs,  T1 -> plan op2 (f_group_by) -> T2
                   capped by encoding budget)   | ... until [E] (end)
        |                                       v
        v                                     final table -> LLM answer
compact prompt -> program-aided LM (e.g. Python/pandas) -> answer
```

TableRAG's split between schema retrieval and cell retrieval is the same split CHESS uses (schema selector vs. value retriever) and the same one enterprise systems implement with "sampled values" (Genie knowledge store) and synonym lists (Cortex Analyst dimensions). Retrieving literal values matters as much as retrieving column names.

## 19.12 Open-source building blocks

**Vanna** is an open-source Python framework that "trains" a RAG store on three item types: DDL statements, documentation snippets and question-SQL pairs. It retrieves related items per question (about 10 in one write-up) before generation, and it supports vector stores such as ChromaDB, Qdrant or Milvus. NVIDIA's NeMo Agent Toolkit ships a Vanna-based text-to-SQL function that keeps separate Milvus collections for SQL, DDL and documentation [41]. The three-collection layout is a compact version of the industrial context store. Its main gaps are the lack of governance, popularity signals or table-level summaries.

## 19.13 Benchmarks

| Benchmark | Year | Scale | What it stresses | Reference results |
|---|---|---|---|---|
| Spider 1.0 [27] | 2018 | 10,181 questions, 5,693 SQL, 200 DBs, 138 domains | Cross-domain generalization; small clean DBs | Best model at release ~12% exact match; ~91% EX by 2024 [29] |
| BIRD [28] | 2023 | 12,751 pairs, 95 DBs, 33.4 GB, 37 domains | Dirty values, external knowledge ("evidence"), efficiency (VES) | Human 92.96%; GPT-4 54.89% at release; ~80-82% top EX in 2025-26 [35][37] |
| Spider 2.0 [29] | 2024 | 632 enterprise workflow tasks; BigQuery, Snowflake, SQLite, DuckDB, PostgreSQL | Huge schemas, dialects, multi-step workflows, docs | o1-preview agent 21.3% at release; Spider2-lite best ~5.7% |
| Spider2-Snow [29] | 2024 | 547 tasks hosted on Snowflake | Snowflake dialect, large schemas | Used as base for later variants |
| data.world insurance [25] | 2023 | 13 tables, 43 questions | Enterprise schema + KG/semantic layer | 16% -> 54% with KG (as summarized in [26]) |
| Snowflake internal [11] | 2024 | 150 BI questions, 3 tiers | Real BI schemas, time series | GPT-4o 51%; Cortex Analyst 90%+ |
| LinkedIn internal [7] | 2025 | 133 questions, 10 areas, 167 tables | Enterprise catalog, multiple valid answers | 53% correct/close (expert) |
| Databricks internal [19] | 2026 | 28 enterprise questions | Agentic multi-query analysis | Genie 84.5%; best coding agent 52.4% |

Metrics. Spider originally emphasized exact set match. BIRD uses **execution accuracy (EX)**, meaning the result sets match, plus a **Valid Efficiency Score (VES)** that penalizes correct but slow SQL [28]. Enterprise teams add **table/column recall** [7], **first-shot acceptance** [3], **search hit rate** [3] and LLM-as-judge grading for agentic reports [20]. BIRD's human baseline of 92.96% assumes oracle external knowledge and falls to about 72% without it, as reported in secondary sources. This is a reminder that much of the "difficulty" in text-to-SQL is missing business context, not SQL skill.

## 19.14 Comparison of production designs

| System | Context store | Schema retrieval | Generation & repair | Human in loop | Public metrics |
|---|---|---|---|---|---|
| Uber QueryGPT [1] | Workspaces of SQL samples + schemas | Intent agent -> workspace -> table agent; column-prune agent | LLM few-shot over pruned schemas | User ACKs/edits tables; edits SQL | ~10 -> ~3 min per query; 1.2M queries/month platform |
| Pinterest Querybook [3] | LLM table summaries + query summaries in OpenSearch | Top-N vector search -> LLM top-K | LLM generation in IDE | User confirms tables | Hit rate 40% -> 90%; acceptance 20% -> 40%+ |
| LinkedIn SQL Bot [7] | Knowledge graph: metadata, logs, wikis, code; certified tables | Popularity filter -> EBR + example tables (~20) -> LLM re-rank (~7) -> columns | LangGraph writer + fixer agents | Chat, "Fix with AI" | 300+ weekly users; 53% expert-correct; ~95% user "passes" |
| Swiggy Hermes [40] | Per-charter metadata + past queries | Charter routing + vector retrieval | ReAct orchestrator, clarification | Clarifying questions | 54% -> 93% accuracy (V3) |
| Snowflake Cortex Analyst [10][11] | Semantic model / view + VQR | Semantic model scopes the tables; VQR similarity | Multi-LLM agents with guardrails, error correction | Author curates model | 90%+ internal vs GPT-4o 51% |
| Databricks Genie [17][18][19] | UC metadata + knowledge store + instructions + trusted assets | Semantic indices over data assets | Multi-LLM sub-agents, parallel paths + judge; Agent mode | Author curates; benchmarks | 32% -> 90%+ (internal); 84.5% vs 52.4% |
| Google Conv. Analytics [22] | Dataplex descriptions / LookML | Context retrieval tool | NL2Query engine | n/a | n/a |

## 19.15 Cross-cutting failure modes

1. **Wrong table, correct-looking SQL.** This is the most common and most dangerous failure, because the query runs and returns plausible numbers. Mitigations: show the selected tables (Uber, Pinterest), popularity and certification priors (LinkedIn), trusted-asset labels (Genie).
2. **Hallucinated columns and joins.** Mitigations: column pruning (Uber), fixer agents with schema validation (LinkedIn), declared join paths in semantic models (Snowflake, Genie knowledge store).
3. **Literal-value mismatch** ("NYC" vs `'New York'`, enum codes). Mitigations: value retrieval via LSH (CHESS), cell retrieval (TableRAG), sampled values and synonyms (Genie, Cortex Analyst).
4. **Metric ambiguity** (several definitions of "active"). Mitigations: semantic layers, certified metrics, clarifying questions (Cortex Analyst classification, Hermes).
5. **Context overflow from wide tables.** 200+ column tables consumed 40-60K tokens at Uber [1]. Mitigations: pruning and summaries.
6. **Evaluation mismatch.** Multiple valid SQLs (60% of LinkedIn's questions [7]) make exact-match and single-reference execution-match unreliable. Use recall-style and judge-based metrics, and keep held-out verified queries out of the context (Snowflake [11], Databricks [20]).
7. **Satisfaction vs. correctness divergence.** LinkedIn's ~95% "passes" versus 53% expert-correct [6][7] shows that user surveys alone cannot certify accuracy.

## Key takeaways for system designers

- **Treat table retrieval as the core RAG problem.** At warehouse scale (10^4-10^6 tables), apply cheap priors first (domain routing, access popularity, certification, top-tier-only indexing), then embedding retrieval, then an LLM re-rank to roughly 5-10 tables. Pinterest's 40% -> 90% hit-rate jump from documentation weighting [3] is the strongest single data point in this space.
- **Index meaning, not just DDL.** Embed LLM-generated table summaries built from schema plus real historical queries, embed summarized past queries, and store verified question-SQL pairs. Query logs are the most underused metadata source.
- **Retrieve values as well as names.** Build a value or cell index (LSH, distinct-value catalogs, synonyms) so filters match the literal values actually stored.
- **Prune before generating.** Column pruning cuts tokens, cost and latency and reduces hallucination (Uber [1], CHESS [31]). Long-context models can skip fine-grained linking only once the candidate schema is small [33].
- **Use a semantic layer where the question space is known.** Generating against metrics and dimensions is close to deterministic for covered questions. Keep a free-form fallback, and route between the two.
- **Close the loop with execution.** Validate with dry-run or EXPLAIN, repair with bounded retries, and for hard questions sample several candidates and select with a judge or tests (CHASE-SQL [32], Genie parallel thinking [18]).
- **Keep humans in the loop at the right step.** Table confirmation is cheap for the user and catches the most expensive errors. Always show the SQL and the assumptions behind it.
- **Build your own benchmark.** Public leaderboards (BIRD ~80%) do not predict enterprise accuracy (Spider 2.0 ~21% at release; internal sets 32-57% for unaided models). Curate 100-200 verified questions per domain, measure table/column recall plus execution accuracy, keep the evaluation questions out of the context, and rerun after every metadata change.
- **Expect the work to become data governance.** Every mature program here (Pinterest, LinkedIn, Snowflake, Databricks) ended up investing in certification, documentation and semantic models. The LLM part plateaued faster than the metadata part.

## References

1. "QueryGPT - Natural Language to SQL using Generative AI." Uber Engineering, 2024. https://www.uber.com/blog/query-gpt/
2. "Uber: Natural Language to SQL Query Generation at Scale." ZenML LLMOps Database, 2024. https://www.zenml.io/llmops-database/natural-language-to-sql-query-generation-at-scale
3. "How we built Text-to-SQL at Pinterest." Pinterest Engineering, 2024. https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff
4. "Pinterest: Text-to-SQL System with RAG-Enhanced Table Selection." ZenML LLMOps Database, 2024. https://www.zenml.io/llmops-database/text-to-sql-system-with-rag-enhanced-table-selection
5. "How Pinterest Powers Its #1 AI Agent With DataHub." DataHub (customer story), 2026. https://datahub.com/customer-stories/pinterest/
6. "LinkedIn: Building a Production Text-to-SQL Assistant with Multi-Agent Architecture." ZenML LLMOps Database (summary of LinkedIn Engineering post, Dec 2024). https://www.zenml.io/llmops-database/building-a-production-text-to-sql-assistant-with-multi-agent-architecture
7. Chen, A. et al. "Text-to-SQL for Enterprise Data Analytics." LinkedIn, arXiv:2507.14372, 2025. https://arxiv.org/abs/2507.14372
8. "LinkedIn AI at Trino Summit 2024" (slides). LinkedIn / Trino, 2024. https://trino.io/assets/blog/trino-summit-2024/trino-summit-2024-linkedin-ai.pdf
9. "What LinkedIn Learned Building an AI Data Agent for 300+ Weekly Users." Dot, 2025. https://www.getdot.ai/blog/linkedin-sql-bot-data-agent
10. Sehrawat, N. "Snowflake Cortex Analyst: Behind the Scenes." Snowflake Engineering, 2024. https://www.snowflake.com/en/engineering-blog/snowflake-cortex-analyst-behind-the-scenes/
11. "Snowflake Cortex Analyst: Evaluating Text-to-SQL Accuracy for Real-World BI Scenarios." Snowflake Engineering, 2024. https://www.snowflake.com/en/blog/engineering/cortex-analyst-text-to-sql-accuracy-bi/
12. "Agentic Semantic Model Improvement: Elevating Text-to-SQL Performance." Snowflake Engineering, 2025. https://www.snowflake.com/en/blog/engineering/agentic-semantic-model-text-to-sql/
13. "Cortex Analyst semantic model specification." Snowflake Documentation, 2025. https://docs.snowflake.com/user-guide/snowflake-cortex/cortex-analyst/semantic-model-spec
14. "Cortex Analyst Verified Query Repository." Snowflake Documentation, 2025. https://docs.snowflake.com/en/user-guide/views-semantic/verified-query-repository
15. "Cortex Analyst." Snowflake Documentation, 2025. https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst
16. "Upgrade from Cortex Analyst to Cortex Agents." Snowflake Developer Guides, 2026. https://www.snowflake.com/en/developers/guides/migrate-cortex-analyst-to-cortex-agents/
17. "AI/BI Genie is now Generally Available." Databricks, 2025. https://www.databricks.com/blog/aibi-genie-now-generally-available
18. "Pushing the Frontier for Data Agents with Genie." Databricks, 2026. https://www.databricks.com/blog/pushing-frontier-data-agents-genie
19. "Introducing Genie One, Genie Agents, and Genie Ontology." Databricks, 2026. https://www.databricks.com/blog/introducing-genie-one-genie-ontology-and-genie-agents
20. "Genie Agents concepts" and "Test and monitor a Genie Agent." Databricks / Microsoft Learn, 2026. https://docs.databricks.com/aws/en/genie-agents/concepts ; https://learn.microsoft.com/en-us/azure/databricks/genie-agents/monitor
21. "AI/BI concepts." Microsoft Learn (Azure Databricks), 2025. https://learn.microsoft.com/en-us/azure/databricks/ai-bi/concepts
22. "Use the Conversational Analytics API for natural language AI." Google Cloud Blog, 2025. https://cloud.google.com/blog/products/business-intelligence/use-conversational-analytics-api-for-natural-language-ai
23. "Semantic Layer vs. Text-to-SQL (2026)." dbt Labs, 2026. https://docs.getdbt.com/blog/semantic-layer-vs-text-to-sql-2026
24. "Delphi at 100% - dbt semantic layer benchmark." Delphi, 2024. https://delphihq.substack.com/p/delphi-at-100-dbt-semantic-layer
25. Sequeda, J., Allemang, D., Jacob, B. "A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model's Accuracy for Question Answering on Enterprise SQL Databases." data.world, arXiv:2311.07509, 2023. https://arxiv.org/abs/2311.07509
26. Allemang, D., Sequeda, J. "Increasing the LLM Accuracy for Question Answering: Ontologies to the Rescue!" arXiv:2405.11706, 2024. https://arxiv.org/abs/2405.11706
27. Yu, T. et al. "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task." EMNLP 2018. https://arxiv.org/abs/1809.08887
28. Li, J. et al. "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs (BIRD)." NeurIPS 2023. https://arxiv.org/abs/2305.03111
29. Lei, F. et al. "Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows." ICLR 2025. https://arxiv.org/abs/2411.07763
30. Pourreza, M., Rafiei, D. "DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction." NeurIPS 2023. https://arxiv.org/abs/2304.11015
31. Talaei, S. et al. "CHESS: Contextual Harnessing for Efficient SQL Synthesis." Stanford, arXiv:2405.16755, 2024. https://arxiv.org/abs/2405.16755
32. Pourreza, M. et al. "CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL." Google Cloud / Stanford, ICLR 2025. https://arxiv.org/abs/2410.01943
33. Maamari, K. et al. "The Death of Schema Linking? Text-to-SQL in the Age of Well-Reasoned Language Models." Distyl AI, NeurIPS 2024 workshop. https://arxiv.org/abs/2408.07702
34. "Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL." Snowflake, arXiv:2505.20315, 2025; and "Smaller Models, Smarter SQL: Arctic-Text2SQL-R1 Tops BIRD and Wins Broadly." Snowflake Engineering, 2025. https://arxiv.org/abs/2505.20315 ; https://www.snowflake.com/en/blog/engineering/arctic-text2sql-r1-sql-generation-benchmark/
35. "Agentar-Scale-SQL: Advancing Text-to-SQL through Orchestrated Test-Time Scaling." arXiv:2509.24403, 2025. https://arxiv.org/abs/2509.24403
36. "ReViSQL: Achieving Human-Level Text-to-SQL." arXiv:2603.20004, 2026. https://arxiv.org/abs/2603.20004
37. "Google's Gemini-SQL2 Tops BIRD Text-to-SQL at 80.04%." AI Weekly, 2026. https://aiweekly.co/alerts/googles-gemini-sql2-tops-bird-text-to-sql-at-8004
38. Chen, S.-A. et al. "TableRAG: Million-Token Table Understanding with Language Models." Google, NeurIPS 2024. https://arxiv.org/abs/2410.04739
39. Wang, Z. et al. "Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding." Google, ICLR 2024; Google Research Blog. https://arxiv.org/abs/2401.04398 ; https://research.google/blog/chain-of-table-evolving-tables-in-the-reasoning-chain-for-table-understanding/
40. "Swiggy Hermes conversational AI" (news coverage of Hermes V3). InfoQ, 2026. https://www.infoq.com/news/2026/01/swiggy-hermes-conversational-ai
41. "Text-to-SQL with Vanna." NVIDIA NeMo Agent Toolkit documentation, 2025. https://docs.nvidia.com/nemo/agent-toolkit/1.6/components/functions/text-to-sql.html
