# 12. Enterprise RAG in Production: Case Studies from Engineering Blogs

## Overview

The earlier chapters describe RAG one component at a time: parsing, embeddings, vector stores, hybrid retrieval, query transformation, graphs, agents, long context and evaluation. This chapter looks at whole systems. It covers what companies built when they had to ship retrieval-augmented features to millions of users, thousands of support agents, or a few hundred regulated professionals, and what they wrote about it afterwards.

The case studies fall into five groups. Each group faces a different main constraint.

1. **Workplace search and assistants** (Glean, Microsoft 365 Copilot, Notion AI, Dropbox Dash, Slack AI). The hard problems are permission-aware retrieval over many SaaS sources, tenant isolation, and freshness.
2. **Customer and internal support** (DoorDash, LinkedIn, Klarna, Airbnb, Uber Genie). The hard problems are hallucination control, policy compliance, and deciding when to hand off to a human.
3. **Product copilots and structured-data RAG** (LinkedIn Premium assistant, Uber QueryGPT, Shopify Sidekick, Ramp, Stripe). Here "retrieval" often means choosing tools, schemas or label sets rather than passages.
4. **Regulated professional domains** (Morgan Stanley, Harvey, Thomson Reuters CoCounsel and Westlaw). The hard problems are citation fidelity, evaluation by experts, and liability.
5. **Platform vendors' own measurements** (Databricks). Their published studies on long-context RAG and LLM judges shaped how many of the other teams evaluate.

A note on evidence. Almost every number below is **self-reported** by the company, usually in a blog post written partly to recruit or to sell. Few teams publish their baselines, sample sizes or confidence intervals. The one independent audit in this chapter, the Stanford study of legal research tools [24], found hallucination rates far above what vendor marketing implied. Treat the numbers as directional. Figures marked "about" or "approximately" are rounded or were reported loosely in the source.

### A reference architecture distilled from the case studies

Almost every system in this chapter fits the same skeleton. The differences lie in which boxes a team invested in.

```
                        +---------------------- offline / async ----------------------+
 sources (wikis, docs,  |  connectors -> parse/normalize -> enrich (LLM summaries,    |
 tickets, chats, CRM,   |  keywords, FAQs) -> chunk -> embed -> index (lexical+vector |
 code, policies, DBs)   |  + ACL metadata + graph edges) -> eval sets from prod logs  |
                        +--------------------------------+----------------------------+
                                                         |
 user --> [input guard] --> [router / intent] --> [query rewrite / decompose]
                                   |                         |
                                   |          [retrieve: hybrid + ACL filter + rerank]
                                   |                         |
                                   v                         v
                            [tools / APIs / SQL]   [context assembly + citations]
                                   \                         /
                                    +-----> [LLM generate] <+
                                                 |
                         [output guard: grounding, policy, PII, format]
                                                 |
                         pass -> answer + sources   fail -> retry / human handoff
                                                 |
                         logs -> LLM judge + human review -> fixes to KB/prompts/index
```

Three observations hold across nearly all the write-ups:

- **The knowledge base matters more than the model.** DoorDash, Uber and Thomson Reuters all report that fixing source documents (missing articles, tables lost during PDF parsing, stale policies) improved results more than swapping models [1][6][25].
- **The guard after generation is a first-class component.** DoorDash's two-tier guardrail, Microsoft's post-processing stage and Slack's output checks all sit between the LLM and the user [1][12][11].
- **Evaluation is the bottleneck for iteration speed.** LinkedIn, Shopify and Databricks each describe building LLM judges calibrated against humans because manual review could not keep up [3][4][21][30].

---

## 1. Workplace search and assistants

### 1.1 Glean: search-first enterprise RAG

**Problem.** Knowledge workers spread information across dozens of SaaS tools, each with its own permission model. An assistant that answers from that information is only as good, and only as safe, as the search engine underneath it.

**Architecture.** Glean's public material describes its assistant as a layer on top of an enterprise search product that came first [17]. The main components it describes are:

- **Connectors** to over 100 SaaS applications. They crawl content and also each system's permission model, so access control can be enforced at query time to match the source system.
- A **hybrid index** that combines lexical and semantic (embedding) retrieval.
- An **enterprise knowledge graph** linking content, people and activity: who wrote what, who works with whom, and which documents a team actually uses. Its signals feed ranking.
- **Company-specific language adaptation.** Glean describes models that learn each customer's vocabulary (project codenames, acronyms) from that customer's corpus.
- An **LLM layer**, model-agnostic in the public description. It plans retrieval, issues one or more searches and synthesizes a cited answer. From 2025 Glean marketed an "agentic reasoning" engine that breaks requests into multi-step plans that run against search and actions [17].

**Key techniques.** The lesson Glean stresses most is that **retrieval quality dominates answer quality** and that enterprise relevance is a ranking problem. Lexical match matters for codenames and IDs. Semantic match matters for paraphrase. Personalization signals from the graph (your team, your recent documents) break ties. The design starts from permissions: the index carries ACLs, and the LLM only ever sees documents the querying user could open.

**Failure modes.** Glean's public writing stays at a high level, and it gives few public latency or quality numbers. Connector freshness and permission sync lag are inherent risks. A document shared or unshared minutes ago may not yet be reflected in the index. Any system that mirrors ACLs rather than checking them live against the source carries this window.

### 1.2 Microsoft 365 Copilot: grounding through Microsoft Graph and the Semantic Index

**Problem.** Ground an LLM in a tenant's email, files, chats, meetings and calendar without moving data outside the Microsoft 365 compliance boundary, and without letting any user see content they cannot already access.

**Architecture.** Microsoft's documentation describes this flow [12][14]:

```
 user prompt (Word/Teams/Outlook/...)
        |
        v
 [Copilot orchestration] --pre-processing ("grounding")--> Microsoft Graph
        |                     (search: Semantic Index + keyword;   (mail, files,
        |                      only content the user can view)      chats, calendar)
        v
 modified prompt + grounding data --> [LLM (Azure OpenAI, in service boundary)]
        |
        v
 [post-processing: responsible-AI checks, security/compliance/privacy review,
  app command generation]
        |
        v
 response + citations to the app
```

- **Grounding.** Copilot does not send the raw prompt straight to the model. It first queries Microsoft Graph and adds the retrieved context to the prompt [12].
- **Semantic Index for Copilot.** Microsoft describes vectorized indexes built at two levels: a **tenant-level** index over SharePoint and OneDrive content and a **user-level** index over the user's own mailbox and related content. Lexical and semantic matching are combined, so a query for "the budget deck from last week" matches on meaning as well as keywords [13].
- **Permissions.** Copilot shows only data that the individual user has at least view permission to, using existing Microsoft 365 access controls, sensitivity labels and retention policies [14].
- **Data boundary.** Prompts, retrieved data and responses stay within the Microsoft 365 service boundary, are not used to train foundation models, and follow existing commitments such as the EU Data Boundary [14].

**Failure modes.**

1. **Oversharing.** Because Copilot faithfully respects existing permissions, it also faithfully exposes years of overly broad sharing ("Everyone except external users" links, open SharePoint sites). Content that was technically accessible but practically undiscoverable becomes one prompt away. Microsoft's deployment guidance accordingly pushes permission audits and restricted search modes before rollout [14].
2. **Prompt injection through retrieved content.** In June 2025 Aim Security disclosed **EchoLeak** (CVE-2025-32711), a zero-click vulnerability in Microsoft 365 Copilot. A crafted external email, later retrieved as grounding context, could instruct the model to exfiltrate data from the user's context through a link or image URL that bypassed existing defenses. Aim called the class an "LLM scope violation". Microsoft patched it server-side [15][16]. It is the canonical demonstration that **every retrieved document is untrusted input**, and that RAG over a mailbox means RAG over attacker-controlled text.

**Lessons.** Enforcing permissions correctly is necessary but not sufficient. The real risk surface is the permission *state* of the tenant and the provenance of every retrieved chunk.

### 1.3 Notion AI: multi-tenant vector search over a block graph

**Problem.** Notion's data model is a tree of billions of "blocks", and every workspace is a separate tenant. Notion AI's Q&A feature (launched in late 2023) has to answer questions over a single workspace interactively, while serving a very large number of workspaces of very different sizes.

**Architecture and evidence.**

- **Vector search at tenant scale.** Pinecone's serverless architecture post used Notion as its motivating example of a workload with thousands of tenants that need sub-100 ms interactive queries. Pod-based scatter-gather made cost proportional to total data instead of the data a query actually touches [18]. The serverless design separates storage (blob storage as the source of truth) from compute, so small tenants cost almost nothing while idle (see Chapter 4).
- **Data platform.** Notion's data-lake post describes block data doubling roughly every six to twelve months, to the order of hundreds of billions of blocks, and a move to a change-data-capture pipeline (Debezium/Kafka into an S3 data lake processed with Hudi and Spark). Notion credits the lake with supporting its AI features, including the embedding and indexing workloads behind search and Q&A [19].

**Key techniques.** These are per-workspace namespaces, incremental re-embedding driven by change data capture rather than batch re-indexing, and permission filtering at the page level within a workspace.

**Lessons.** For SaaS products, the vector store's *tenancy model* is often the deciding factor in cost. A design where idle tenants cost nothing is worth more than a few points of recall. Notion's story is mostly about the data plumbing (CDC, the lake, incremental indexing). The LLM prompt is a small part of it.

### 1.4 Dropbox Dash: from file-preview Q&A to universal search and agents

Dropbox has published one of the most detailed public RAG engineering records, in three stages.

**Stage 1: AI answers and summaries in file previews (2023).** Dropbox added summarization and Q&A to its web file previews by building on **Riviera**, its existing preview and transcoding framework, which already converted many file types to text [9]. Key techniques:

- Files are converted to text and split into chunks, and each chunk is embedded.
- For **summaries**, chunk embeddings are clustered (k-means) and representative chunks from each cluster are summarized. This covers the whole document without passing every token to the LLM.
- For **Q&A**, the chunks most similar to the question are retrieved and passed to the LLM. Multi-file questions work the same way across several files.
- Embeddings and intermediate results are cached, so repeated questions about the same file avoid re-processing.

Dropbox reported large improvements from this re-architecture. Per-summary cost fell by about 93% and per-query cost by about 64%. Approximate p75 latency fell from about 115 s to about 4 s for summaries and from about 25 s to about 5 s for Q&A. These are approximate figures as reported [9].

**Stage 2: Dash universal search and answers.** Dash connects to third-party applications (Google Workspace, Microsoft 365, Slack, Salesforce and others) as well as Dropbox. In "Building Dash" Dropbox describes its retrieval choice explicitly as a trade-off among latency, quality and freshness [8]:

- A pure vector index over all connected content was judged too costly to keep fresh and too slow for their latency budget.
- They chose **traditional information retrieval (lexical) as the first stage**, followed by **on-the-fly chunking and semantic reranking** of the candidate documents. This gives reasonable relevance with fresh data and interactive latency (the post targets responses of a couple of seconds for most queries).
- The system is **model-agnostic**. It evaluates and switches among several LLMs depending on cost and quality.

**Stage 3: agents with a planning DSL.** For multi-step requests ("find the latest project plan, then summarize open risks"), Dash has the LLM write a plan in a constrained, Python-like domain-specific language. An interpreter then executes it. Dropbox describes **static analysis and type checking** of the generated code before execution, plus retries when a step fails. This catches malformed plans deterministically instead of trusting the LLM to self-correct [8].

**Later lessons (2025).** In a talk write-up, Dropbox's VP of Engineering Josh Clemm described where Dash went next [10]:

- A **knowledge graph** over people, documents and activity, used to build richer "bundles" of context for ranking.
- **Limiting tool sprawl.** Exposing many MCP tools to the model degraded tool selection and filled the context window, so Dash consolidated retrieval behind a small number of well-designed tools.
- **DSPy** to optimize prompts for components such as relevance judges against labeled data.

**Lessons.** Dash shows that "RAG" in enterprise search can begin with BM25. A lexical first stage plus a semantic reranker on fresh candidates beats a stale vector index. It also shows that agent plans should be compiled and checked, not just executed.

### 1.5 Slack AI: RAG inside the trust boundary

**Problem.** Summarize channels and threads and answer search questions across a workspace's messages. Slack's enterprise customers require that their data never leaves Slack's control, is never used for training, and respects every channel's membership rules.

**Architecture.** Slack Engineering's security post sets out these principles and the architecture that follows from them [11]:

- **Hosting.** LLMs are hosted inside Slack's own AWS infrastructure using an **escrow VPC** arrangement on Amazon SageMaker. The model provider never has access to customer data, and Slack never has access to the provider's weights.
- **No training on customer data.** Slack chose **RAG instead of fine-tuning**. Customer data is supplied only as context at inference time, so no customer data is baked into a model that could leak it to another tenant.
- **Permission-scoped retrieval.** Slack AI retrieves only content the requesting user can already see, through the same access-control checks used by Slack search. That includes private channels the user belongs to, and excludes channels they have left.
- **Ephemeral output.** AI outputs such as summaries are not stored beyond what the feature needs, and existing compliance tools such as data retention settings and encryption key management (EKM) keep working [11].

```
 Slack user --> Slack AI service --(user's ACL)--> Slack search / message store
                     |                                    |
                     |<----------- relevant messages -----+
                     v
        [escrow VPC in Slack's AWS: hosted LLM]  (provider: no data access)
                     |
                     v
     summary / answer (ephemeral) --> user
```

**Lessons.** Slack framed the choice of RAG over fine-tuning as a *security* decision, not a quality one. For multi-tenant SaaS, inference-time context is easy to scope, audit and delete. Weights are not.

---

## 2. Customer and internal support

### 2.1 DoorDash: Dasher support with a two-tier guardrail and an LLM judge

**Problem.** Dashers (delivery drivers) contact support mid-delivery with time-sensitive issues. DoorDash's earlier flow-based automation could not handle the variety of issues, and the knowledge base articles were written for humans [1].

**Architecture.** DoorDash describes this pipeline [1]:

```
 Dasher chat --> [conversation summarizer: extract core issue]
                          |
                          v
              [retrieve: similar resolved cases + KB articles]
                          |
                          v
              [LLM: draft response from retrieved context]
                          |
                          v
     +--------------[LLM Guardrail]---------------+
     | tier 1: cheap semantic-similarity check     |
     |   (response vs retrieved KB content)        |
     | tier 2: LLM evaluator, only if tier 1 flags |
     |   (grounding, coherence, policy compliance) |
     +-------------------+-------------------------+
            pass         |          fail
             v           |            v
         send reply      |   retry / escalate to human agent
                         v
     [LLM Judge offline: retrieval correctness, response accuracy,
      grammar, coherence to context, relevance] + human review
                         |
                         v
     [quality pipeline: fix KB articles, retrieval, prompts; regression tests]
```

**Key techniques.**

- **Summarize before retrieving.** A multi-turn chat is condensed into the core issue before it is used as a search query, which keeps chit-chat out of the query.
- **Two-tier guardrail.** A cheap similarity check runs on every response. The more expensive LLM evaluator runs only on responses the first tier flags. This controls both cost and latency, which matter in a live chat.
- **The LLM Judge** scores five dimensions offline: retrieval correctness, response accuracy, grammar and language accuracy, coherence to context, and relevance to the Dasher's request. Human reviewers calibrate it.
- **Prompt engineering lessons.** Break complex prompts into smaller parts, avoid negative instructions ("don't..."), use chain-of-thought for the evaluator, and run independent steps in parallel to save latency.
- **Regression prevention.** Prompt changes are tested against a suite of known cases before release, in the spirit of unit tests.

**Numbers (as reported).** The guardrail reduced hallucinations by about 90% and severe compliance issues by about 99%. The system handles thousands of support requests per day [1].

**Lessons.** DoorDash found that many failures traced back to **knowledge-base gaps** (missing or ambiguous articles), not to the model. They then made KB improvement a standing part of the quality loop. The guardrail's tiered design is a reusable pattern: check everything cheaply, and check deeply only what is suspicious.

### 2.2 LinkedIn: a knowledge graph over support tickets

**Problem.** Chunk-based RAG over historical support tickets loses two kinds of structure. One is a ticket's **internal structure** (summary, description, steps to reproduce, resolution). The other is the **relationships between tickets** (clones, duplicates, references) [2].

**Architecture.** The SIGIR 2024 paper by Xu et al. builds a two-level graph [2]:

```
 historical tickets
   |  parse each ticket into a tree (rule-based + LLM)
   v
 intra-ticket trees: ticket -> sections (summary, description, priority, resolution)
   |  connect tickets: explicit links (clone/related) + implicit (embedding similarity)
   v
 inter-ticket graph  --(section embeddings)--> vector index
 ------------------------------------------------------------------------------
 query: [LLM extracts entities + intents]
          -> [embedding search finds candidate tickets]
          -> [LLM translates question to graph query (Cypher-like) over sub-graphs]
          -> [retrieved sub-graphs -> LLM answer]
```

**Numbers (as reported).**

- Retrieval MRR improved by about 77.6% over a chunk-based baseline.
- BLEU improved by 0.32.
- The system was deployed in LinkedIn's customer-service team for about six months, and median per-issue resolution time fell by about 28.6% compared with the control group [2].

**Lessons.** Keeping the *document schema* (sections as nodes) is a cheap and robust form of GraphRAG. It needs no open-ended entity extraction, and it lets retrieval target "the resolution section of tickets similar to this one". Chapter 7 discusses this design alongside other graph approaches.

### 2.3 Klarna: high-volume assistant and the limits of automation

**Problem.** Klarna serves customers in many markets and languages with questions about refunds, returns, payments and disputes.

**Reported results (February 2024).** Klarna's press release on its OpenAI-powered assistant claimed the following for the first month [27]:

- 2.3 million conversations, about two-thirds of all customer-service chats.
- Work equivalent to about 700 full-time agents.
- Resolution in under 2 minutes, versus about 11 minutes before.
- A 25% drop in repeat inquiries.
- Customer satisfaction on par with human agents.
- Availability in 23 markets and more than 35 languages.
- An estimated USD 40 million profit improvement in 2024.

**Architecture.** Klarna's own release gives little architectural detail. LangChain's customer story says Klarna built the assistant with LangGraph for controllable, stateful agent flows and LangSmith for tracing and evaluation. It describes the assistant as personalized with customer and transaction context, and it reports a large cut in average resolution time (about 80% in LangChain's telling) [28]. In other words, the assistant combines retrieval over policy and help content with tool calls into account data. Most questions ("where is my refund?") are answered from the customer's own records, not from documents.

**Failure modes and lessons.** In 2025 Klarna's leadership said publicly that they would again make human agents readily available after quality concerns. Press reports at the time quoted the CEO as saying that cost had been weighted too heavily. That is press reporting, not an engineering source, so treat it accordingly. The lesson for designers is that **containment rate is not the objective**. A clear, easy path to a human is part of the architecture, as DoorDash's escalation path and Airbnb's handoff show.

### 2.4 Airbnb: LLMs inside a workflow platform

**Problem.** Airbnb's customer-support automation had evolved as rigid, predefined workflows. They were reliable and auditable, but brittle when faced with open-ended conversation [20].

**Architecture.** Airbnb's "Automation Platform v2" post describes a hybrid design [20]:

- **LLM-driven workflows** ("chain of thought" workflows) in which the LLM reasons over which tools or actions to invoke. They run beside the traditional deterministic workflows, which remain in place for high-stakes or well-understood flows.
- A **context management** layer that gathers what the LLM needs, such as user role, trip status, relevant policies and past interactions, rather than relying on retrieval alone.
- A **guardrails framework** that checks LLM inputs and outputs (for example, against unsafe content or policy violations) before actions run.
- **Tool and data integration** through managed APIs, plus developer tooling (a prompt playground and evaluation) for teams that build on the platform.

**Lessons.** Airbnb treats the LLM as one execution mode within a workflow engine, not as a replacement for it. Retrieval is part of context management, together with structured state about the user and the reservation. For transactional support, the reservation record is usually more relevant than any help article.

### 2.5 Uber Genie and Enhanced Agentic RAG

**Problem.** Uber's internal Slack support channels (platform, data, infrastructure) received thousands of repetitive questions whose answers already existed in wikis and internal Q&A. On-call engineers spent hours answering them [5].

**Genie architecture (2023-2024).** Uber describes this design [5]:

```
 sources: internal wikis, internal Stack Overflow, engineering docs
     |  Spark ETL: scrape + chunk
     v
 embeddings (LLM embedding API via Uber's Michelangelo gateway)
     |
     v
 Sia (Uber's in-house vector database)   <-- refreshed on schedule
     |
 Slack question --> Knowledge Service --> retrieve top chunks --> LLM (via gateway)
     |                                                          |
     |<----------------- answer + source links -----------------+
     v
 user feedback buttons (resolved / helpful / not relevant) --> metrics dashboard
```

Design choices include only ingesting **sources pre-approved as non-sensitive**, citing sources in every answer so users can verify, and collecting structured feedback in-thread.

**Numbers (as reported).** Between its September 2023 launch and the blog post, Genie had answered about 70,000 questions across about 154 Slack channels, with an estimated saving of about 13,000 engineering hours [5].

**Enhanced Agentic RAG (EAg-RAG, 2025).** When Uber extended Genie to security and privacy policy questions, the baseline failed in ways that needed a redesign [6]:

1. **Document processing.** Standard PDF loaders flattened tables and lost structure. Uber switched to sources that keep structure (Google Docs exported as HTML and converted to a markdown-like format) so tables survived as tables.
2. **Enriched metadata.** An LLM generates summaries, FAQs and keywords for each document or chunk at ingestion. These improve lexical and semantic match for questions phrased differently from the policy text.
3. **Agents at query time**, built with LangGraph:
   - A **Query Optimizer** rewrites ambiguous queries and splits compound ones.
   - A **Source Identifier** narrows the search to the relevant policy documents.
   - **Hybrid retrieval** combines BM25 with vector search.
   - A **Post-Processor** removes duplicates and orders the retrieved context by document structure before generation.
4. **Evaluation.** An LLM-as-judge scores answers against a golden set curated by subject-matter experts. This replaced slow manual review.

**Numbers (as reported).** Uber reported about a 27% relative increase in acceptable answers and about a 60% relative reduction in incorrect advice compared with the earlier pipeline [6].

**Lessons.** Uber's progression mirrors the field. It moved from a naive pipeline to adding structure at ingestion, then to agentic query planning and an automated judge. Its biggest gains came from fixing parsing and enriching metadata, which are steps that involve no generation at all.

---

## 3. Product copilots and structured-data RAG

### 3.1 LinkedIn Premium assistant: routing, retrieval, generation

LinkedIn's "Musings on building a Generative AI product" describes the assistant behind Premium features such as job-fit assessment and post takeaways [3].

**Architecture.**

```
 user query + page context (job post, profile, feed post)
     |
     v
 [Routing: small/fast LLM picks a specialized "agent"]  (job assessment, knowledge, ...)
     |
     v
 [Retrieval (recall-oriented): agent decides which internal APIs ("skills") to call:
   people search, job data, company data, external search; plus embedding-based
   retrieval of few-shot examples]
     |
     v
 [Generation (precision-oriented): larger LLM composes answer from noisy context]
     |
     v
 streamed response (all stages stream, to cut time to first token)
```

**Key techniques.**

- **Recall in retrieval, precision in generation.** Retrieval casts a wide net, and the generator filters. The intent is that over-retrieval should hurt less than missing evidence.
- **Internal APIs as skills.** LinkedIn wrapped internal APIs as LLM-callable skills with descriptions. Function calls are emitted in **YAML rather than JSON** to save tokens. A defensive YAML parser plus prompt fixes reduced malformed outputs from about 10% to about 0.01% [3].
- **Smaller models for routing and retrieval** and larger ones for generation, to manage latency and GPU capacity.

**Lessons.** LinkedIn reported reaching about 80% of its quality target in the first month and then needing about four more months to get past about 95%. The last stretch was dominated by hallucination edge cases and evaluation work [3]. Their evaluation was limited by throughput: human annotators following written guidelines could review only a limited daily sample. An earlier LinkedIn post describes each prompt iteration requiring hours of inference and spreadsheet-based review [4]. Their conclusion is to **budget for the last 15%**, which costs far more than the first 80%.

### 3.2 Uber QueryGPT: RAG for text-to-SQL

**Problem.** Uber engineers and analysts write a very large number of SQL queries against a large warehouse, and writing them requires knowing which of many tables and columns to use [7].

**Architecture.** QueryGPT's production version, as Uber describes it, is a chain of narrow components [7]:

- **Workspaces** curate domain-specific sets of tables and SQL samples, for example Mobility or Ads.
- An **Intent Agent** maps the question to a business domain or workspace.
- A **Table Agent** proposes relevant tables, which the user can confirm or edit (human-in-the-loop).
- A **Column Prune Agent** removes irrelevant columns from very wide schemas so the prompt fits the context window and contains less noise.
- **Retrieval of similar SQL samples** and the schemas as few-shot context, followed by generation of the query and an explanation.

**Numbers (as reported).** Uber estimated a large query-authoring workload (on the order of a million queries per month) and reported that QueryGPT cut authoring time from about 10 minutes to about 3 minutes for a typical query [7].

**Lessons.** In structured-data RAG, *what is retrieved is the schema*, and the main failure is choosing the wrong table. Uber chose to put the table decision in front of the user instead of hiding it. Pruning columns is the schema equivalent of reranking chunks.

### 3.3 Shopify Sidekick: tool retrieval, JIT instructions and calibrated judges

**Problem.** Sidekick is Shopify's merchant assistant. It answers questions and takes actions across analytics, customers, products, orders and settings. As tools were added, the system prompt became a pile of special-case instructions [21].

**Key techniques (as described by Shopify, 2025).**

- **Tool-count scaling.** Shopify describes a progression. With roughly 0 to 20 tools, boundaries are clear. From about 20 to 50 they blur. Past about 50, there are many plausible ways to do the same task and behavior becomes unpredictable [21].
- **Just-in-time (JIT) instructions.** Instead of a monolithic system prompt, relevant instructions are returned *alongside tool results* when they apply. This is effectively retrieval of guidance at the point of use. It keeps the base prompt small, makes caching more effective and makes behavior easier to reason about.
- **Ground-truth sets sampled from production.** Shopify chose real merchant conversations labeled by humans over hand-written "golden" examples.
- **Calibrated LLM judges.** Judges were iterated until their agreement with humans approached human-to-human agreement. Shopify reports Cohen's kappa rising from about 0.02 to about 0.61, against a human baseline of about 0.69 [21].
- **User simulators** replay conversations so candidate changes can be tested before release.
- **RL fine-tuning with gated rewards.** Shopify describes reward hacking it observed (for example, the model opting out of hard tasks, or putting values into tags instead of the right fields), and procedural checks before LLM-judged rewards to catch it [21].

**Lessons.** For agentic assistants, the "retrieval" problem moves from documents to **instructions and tools**. Delivering context just in time beats loading everything up front, which matches Dropbox's experience with MCP tool sprawl [10].

### 3.4 Ramp: RAG as constrained classification

**Problem.** Ramp needed to assign customers standard industry codes (NAICS) rather than labels from a homegrown taxonomy. The label space is large and hierarchical, and a classifier trained on a few examples per code would perform poorly [22].

**Architecture.** In the post "From RAG to Richness: How Ramp Revamped Industry Classification", Ramp describes RAG used for classification [22]:

```
 business profile (name, description, website text, ...)
       |  embed
       v
 [retrieve top-k candidate NAICS codes by embedding similarity]
       |
       v
 [LLM picks from candidates only, constrained output; optional second pass
  at a finer level of the hierarchy]
       |
       v
 validated code (must be in candidate set) + explanation
```

**Key techniques.**

- **Separate metrics for each stage.** Ramp measures retrieval as "is the correct code in the top-k?" and the LLM stage as accuracy given the candidates. The two stages can then be tuned independently. For example, a larger k raises recall but makes the choice harder for the LLM.
- **Constrained selection.** The LLM chooses from retrieved candidates instead of generating codes freely, which removes invented codes.

**Lessons.** RAG works well as a **candidate generator for extreme classification**. Because the label space itself is retrieved, the system stays maintainable when the taxonomy changes.

### 3.5 Stripe: documentation Q&A and internal use

Stripe's public material on LLMs, much of it in OpenAI's customer story, describes internal experimentation. Teams across the company proposed and prototyped GPT-4 use cases, and some shipped. These included **Stripe Docs** question answering, where the model reads Stripe's extensive developer documentation to answer integration questions, and support and fraud-screening tasks [31]. Stripe has given few public architectural details about its retrieval stack, so this atlas lists it only as an example of the common "docs assistant" pattern. In that pattern a well-structured, versioned corpus of developer documentation is the retrieval source, and answers link back to specific documentation pages.

---

## 4. Regulated professional domains

### 4.1 Morgan Stanley: wealth-management knowledge assistant with evals as a gate

**Problem.** Financial advisors need fast answers from a large internal library of research, procedures and product information. Wrong answers carry regulatory risk [26].

**Architecture and process.** OpenAI's customer story describes **AI @ Morgan Stanley Assistant**, built on GPT-4 over a corpus of about 100,000 documents, and a later meeting-summary tool, **AI @ Morgan Stanley Debrief** [26]. The theme of the story is **evaluation as a gate for deployment**:

- Evals were built for each use case (for example, summarization quality and accuracy against expert-written answers). Advisors and subject-matter experts graded outputs.
- Retrieval was tuned until the evals passed.
- Pilot rollouts to small advisor groups came before firm-wide release.

**Numbers (as reported).** More than 98% of advisor teams adopted the assistant. Document-retrieval efficiency (finding the right document for a question) rose from about 20% to about 80% through iterative work on retrieval [26].

**Lessons.** In regulated settings the evaluation suite *is* the deployment approval. A 20% to 80% improvement in retrieval efficiency shows where the effort went: retrieval, not generation.

### 4.2 Harvey: domain models plus retrieval with sentence-level citations

**Problem.** Lawyers need answers grounded in authoritative sources such as case law, statutes and the client's own documents, with every claim traceable to a citation.

**Architecture and evidence.** OpenAI's Harvey story describes a **custom-trained case-law model**. Harvey and OpenAI added a large amount of legal text (on the order of 10 billion tokens, starting from Delaware case law and expanding to US case law) to a GPT-4-class model. The model was designed so that statements are supported by the cases it cites. In the reported evaluation, attorneys from large law firms preferred its output over GPT-4's about 97% of the time [23]. Harvey's products combine this with retrieval over firm and client documents. Its Vault product handles large document sets for bulk review and Q&A, and Harvey's blog discusses enterprise retrieval and security requirements [23].

**Key techniques.**

- **Domain adaptation and RAG together.** The fine-tuned model knows the shape of legal reasoning and citation, and retrieval supplies the specific authorities.
- **Sentence-level citations** that a lawyer can check.
- **Evaluation by practicing lawyers** rather than generic metrics. Harvey has also published legal task benchmarks.

**Lessons.** For expert domains, *style and structure* (how legal answers are written and cited) can be learned through training, while *facts* (which case says what) must come from retrieval.

### 4.3 Thomson Reuters: CoCounsel, Westlaw AI and the independent audit

**Problem and products.** Thomson Reuters acquired Casetext (maker of CoCounsel) in 2023 and launched AI-assisted research in Westlaw Precision. Both use RAG over Thomson Reuters' editorial legal content, which is curated and annotated. TR Labs' engineering blog also describes RAG for **customer support**: retrieving from support knowledge bases to help agents answer product questions faster, with embeddings, a vector store and an LLM generating cited responses [25].

**The independent audit.** In 2024 Stanford RegLab and HAI researchers (Magesh et al.) published the first preregistered evaluation of commercial legal RAG tools [24]. They ran a set of more than 200 legal queries against Lexis+ AI, Westlaw AI-Assisted Research and Ask Practical Law AI, with GPT-4 as a baseline. Their findings:

- Vendor claims of "hallucination-free" legal research were not supported. The tools hallucinated on roughly 17% to 33% of queries. Westlaw AI-Assisted Research had the highest rate among the products tested (about 33%), and Lexis+ AI was about 17%. These are approximate figures from the paper.
- The study gave a precise **taxonomy of RAG-specific failure**. One category is responses that are *misgrounded*: a correct-sounding statement cites a source that does not support it. Another is responses that are *incomplete*, refusing or giving unhelpful answers. In the authors' framing, RAG reduces hallucination compared with a general chatbot but does not eliminate it.
- They identified causes particular to legal retrieval. Relevance is not the same as authority: a case may be relevant but overruled. Retrieval can match on surface text from the wrong jurisdiction. And the generator may reason incorrectly even from correct documents.

Thomson Reuters responded at the time that the tool is intended to help researchers, not replace verification. Later versions of CoCounsel added more agentic, multi-step research workflows.

**Lessons.** This is the clearest public case that **"grounded" is not "correct"**. A citation link proves provenance, not entailment. Systems in high-stakes domains need claim-level support checks (see Chapter 11) and a check for *authority*, not just relevance.

---

## 5. Platform vendors' published measurements: Databricks

Databricks is both a RAG platform vendor (Vector Search, Mosaic AI Agent Framework, Agent Evaluation) and a publisher of influential measurement studies that many teams in this chapter echo.

**Long-context RAG study (2024).** Databricks Mosaic Research ran RAG with increasing amounts of retrieved context across many commercial and open models and several corpora [29]. More retrieved context helped up to a point. Only a few frontier models kept accuracy past about 64K tokens, and many open models degraded well before their advertised limits. The long-context failures were model-specific (refusals, copyright-related declines, repetition, instruction-following breakdowns). Chapter 9 discusses it. For enterprise designers, the practical result is that **the context budget is a per-model hyperparameter you must measure**, not a constant.

**LLM-judge best practices (2023).** Databricks' study of automated RAG evaluation found that [30]:

- LLM judges (GPT-4 at the time) agreed with human graders at high rates on their documentation-QA task (over about 80% agreement in their report).
- **Low-precision grading scales** (for example 0 to 3, or 1 to 5) were more consistent than fine-grained ones.
- **Few-shot examples** for each score made judges, including cheaper models, much more consistent.
- A judge's grading rubric, once tuned, transferred poorly to new use cases without re-checking.

These conclusions show up in DoorDash's judge, Shopify's kappa-driven calibration and Uber's EAg-RAG evaluation [1][21][6].

---

## 6. Comparison of designs

| System | Primary use | Retrieval core | Distinctive technique | Guard / eval | Reported impact (self-reported, approx.) |
|---|---|---|---|---|---|
| Glean [17] | Workplace search + assistant | Hybrid lexical+semantic, 100+ connectors | Enterprise knowledge graph; per-company language adaptation | ACLs mirrored from sources | Few public numbers |
| M365 Copilot [12][13] | Assistant in Office apps | Microsoft Graph + Semantic Index (tenant + user) | Grounding pre-processing; data stays in service boundary | Post-processing RAI/compliance; user-view ACLs | EchoLeak showed injection risk [15] |
| Notion AI [18][19] | Workspace Q&A | Per-tenant vector search | CDC-driven incremental indexing | Page permissions | Thousands of tenants, sub-100 ms target |
| Dropbox Dash [8][9][10] | Universal search, Q&A, agents | Lexical first stage + on-the-fly chunk + rerank | Python-like plan DSL with static checks; KG; DSPy | LLM judges | Preview Q&A cost -64%, p75 25 s -> 5 s |
| Slack AI [11] | Summaries, search answers | Slack search with user ACLs | Escrow-VPC hosted LLMs; RAG not fine-tuning | Ephemeral outputs, EKM, retention | Not reported |
| DoorDash [1] | Dasher support chat | Resolved cases + KB articles | Summarize-then-retrieve; 2-tier guardrail | LLM Judge (5 dims) + humans | Hallucinations -90%, severe compliance -99% |
| LinkedIn CS [2] | Support ticket QA | KG of ticket trees + embeddings | Intra/inter-ticket graph; LLM to graph query | Offline MRR/BLEU | MRR +77.6%; resolution time -28.6% |
| Klarna [27][28] | Consumer support | Help/policy content + account tools | LangGraph-controlled flows | LangSmith tracing/eval | 2/3 of chats; 11 -> <2 min |
| Airbnb [20] | Support automation | Context management + tools | LLM workflows inside workflow engine | Guardrails framework | Not reported |
| Uber Genie / EAg-RAG [5][6] | Internal on-call Q&A, policy Q&A | Sia vectors; later BM25+vector | Enriched metadata; query optimizer, source identifier agents | LLM judge on SME golden set | ~70k questions, ~13k hours; +27% acceptable, -60% incorrect |
| Uber QueryGPT [7] | Text-to-SQL | Workspaces, SQL samples, schemas | Intent / table / column-prune agents | Human confirms tables | ~10 -> ~3 min per query |
| LinkedIn Premium [3] | Product assistant | Internal APIs as skills + few-shot EBR | Router; YAML calls; full streaming | Annotator-based eval | Malformed output ~10% -> ~0.01% |
| Shopify Sidekick [21] | Merchant agent | Tools + JIT instructions | JIT instructions; user simulator; GRPO | Judges calibrated to kappa ~0.61 | Not reported as a single KPI |
| Ramp [22] | Industry classification | Embedding retrieval of NAICS codes | Retrieve-then-select, constrained output | Per-stage accuracy | Not reported here |
| Morgan Stanley [26] | Advisor knowledge assistant | ~100k internal documents | Evals as deployment gate | Expert-graded evals | 98% adoption; retrieval 20% -> 80% |
| Harvey [23] | Legal research and review | Case law + client docs (Vault) | Custom case-law model plus RAG | Lawyer evaluation | 97% preference vs GPT-4 |
| TR CoCounsel / Westlaw [24][25] | Legal research; support | Editorial legal corpus | RAG over curated authority | Independent audit | ~33% hallucination (Westlaw AI-AR, 2024 audit) |
| Databricks [29][30] | Platform + research | Vector Search platform | Long-context sweeps; judge rubrics | Low-precision rubrics, few-shot | >~80% judge-human agreement |

---

## 7. Cross-cutting patterns and failure modes

### 7.1 Permission-aware retrieval is the defining enterprise constraint

Every workplace system (Glean, Microsoft, Notion, Dash, Slack) enforces access control at retrieval time, so the generator never sees a document the user cannot open [11][13][17]. There are two main designs:

- **Mirrored ACLs in the index.** Glean, Microsoft and Notion store permissions as index metadata. This is fast, but there is a sync-lag window.
- **Delegated checks against the source of truth.** Slack runs retrieval through its own search and permission system. This is always current, but it is only easy when the system owns the data.

Both inherit the tenant's permission hygiene, so oversharing becomes discoverable [14].

### 7.2 Retrieved content is an attack surface

EchoLeak [15][16] turned a theoretical concern into a CVE. Any RAG system that retrieves content written by outsiders (email, tickets, shared documents, web pages) must assume that content contains instructions. Mitigations seen in the case studies include:

- Isolating the model's context from egress channels (Slack's escrow VPC limits where data can go).
- Post-generation filters on links and images.
- Treating tool outputs and retrieved text as data, never as policy.

### 7.3 Ingestion quality beats model choice

The largest reported jumps come from upstream work:

- Uber fixed table parsing and added LLM-generated metadata [6].
- Morgan Stanley improved retrieval efficiency from about 20% to about 80% [26].
- DoorDash rewrote KB articles [1].
- Dropbox rebuilt its preview pipeline [9].

Several posts state outright that a better LLM would not have fixed these problems.

### 7.4 "Retrieve" increasingly means retrieving tools, schemas and instructions

These systems retrieve things other than passages:

- QueryGPT retrieves tables and SQL examples [7].
- Ramp retrieves label candidates [22].
- LinkedIn retrieves skills and few-shot examples [3].
- Shopify retrieves instructions just in time [21].
- Dash consolidated tools to keep tool selection reliable [10].

The common principle is to **put the smallest sufficient context in front of the model at the moment it is needed**.

### 7.5 Guardrails are tiered for cost and latency

DoorDash checks everything cheaply and escalates only suspicious responses to an LLM evaluator [1]. Microsoft runs pre-processing (grounding) and post-processing (compliance) stages around the model [12]. Airbnb keeps deterministic workflows for high-stakes paths [20]. The shared pattern is: **deterministic check, then cheap model, then expensive model, then human**, each tier triggered by the one before it.

### 7.6 Evaluation: calibrate the judge, sample from production

The mature teams converge on three practices:

- Golden sets built from real traffic, not invented examples (Shopify, Uber's SME sets).
- LLM judges with low-precision rubrics and few-shot anchors (Databricks), calibrated against human agreement (Shopify's kappa).
- Separate retrieval and generation metrics (Ramp's per-stage accuracy, LinkedIn's MRR alongside answer quality) [2][21][22][30].

Even so, the independent Stanford audit [24] is a reminder that internal evals may not measure what users actually experience.

### 7.7 The automation ceiling

Klarna's reported 2025 shift back toward human availability, DoorDash's escalation path and LinkedIn's "80% in a month, four months for the next 15%" all point the same way [1][3][27]. Designing the handoff is as important as designing the retrieval.

---

## Key takeaways for system designers

1. **Start from the permission model, not the vector store.** Decide whether ACLs are mirrored or checked live, measure the sync lag, and audit tenant oversharing before launch. Copilot shows that correct enforcement of bad permissions is still a breach in practice [13][14].
2. **Treat every retrieved chunk as untrusted input.** Separate instructions from data, strip or proxy outbound links and images in generated output, and red-team with content that has been planted in the corpus [15][16].
3. **Invest in ingestion first.** Parsing that keeps structure (tables, sections), LLM-enriched metadata (summaries, FAQs, keywords) and KB hygiene produced the largest reported gains at Uber, DoorDash and Morgan Stanley [1][6][26].
4. **Use hybrid retrieval with a cheap first stage.** BM25 or lexical search plus a semantic reranker on fresh candidates (Dropbox, Uber EAg-RAG) is often better than a single stale vector index [6][8].
5. **Keep document structure.** LinkedIn's ticket trees and Uber's structure-aware ordering show that schema-level structure is a cheap, robust form of graph RAG [2][6].
6. **Retrieve the minimal context just in time.** That applies to passages, tools, schemas and instructions alike. Fewer, well-designed tools beat a large tool catalog [3][7][10][21].
7. **Tier your guardrails.** Use a cheap check on every response, an LLM check on flagged responses, and a human for the rest. Make escalation a designed path, not an exception [1][20].
8. **Make evaluation the release gate.** Use production-sampled golden sets, low-precision rubrics with examples, judges calibrated against human agreement, and separate retrieval and generation metrics [21][26][30].
9. **Prefer RAG over fine-tuning for customer data** when tenancy, deletion and auditability matter (Slack). Use fine-tuning for domain style and reasoning when the domain demands it (Harvey) [11][23].
10. **Budget for the long tail.** Expect the first 80% of quality quickly and the last 15% slowly, and plan headcount for evaluation accordingly [3].
11. **Distrust single headline metrics.** Containment rates, hours saved and "hallucination-free" claims are self-reported. Use independent or adversarial evaluation where stakes are high [24][27].

---

## References

1. DoorDash Engineering. "Path to high-quality LLM-based Dasher support automation." DoorDash, 2024. https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/
2. Xu, Z., Cruz, M. J., Guevara, M., Wang, T., Deshpande, M., Wang, X., Li, Z. (LinkedIn). "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering." SIGIR 2024. https://arxiv.org/abs/2404.17723
3. Bottaro, J. P., Ramgopal, K. (LinkedIn). "Musings on building a Generative AI product." LinkedIn Engineering, 2024. https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product
4. LinkedIn Engineering. "Our learnings from the early days of generative AI." LinkedIn, 2023. https://engineering.linkedin.com/blog/2023/our-learnings-from-the-early-days-of-generative-ai
5. Uber Engineering. "Genie: Uber's Gen AI On-Call Copilot." Uber, 2024. https://www.uber.com/blog/genie-ubers-gen-ai-on-call-copilot/
6. Uber Engineering. "Enhanced Agentic-RAG: What If Chatbots Could Deliver Near-Human Precision?" Uber, 2025. https://www.uber.com/blog/enhanced-agentic-rag/
7. Uber Engineering. "QueryGPT - Natural Language to SQL Using Generative AI." Uber, 2024. https://www.uber.com/blog/query-gpt/
8. Dropbox. "Building Dash: How RAG and AI agents help us meet the needs of businesses." Dropbox Tech Blog, 2024. https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users
9. Dropbox. "Bringing AI-powered answers and summaries to file previews on the web." Dropbox Tech Blog, 2023. https://dropbox.tech/machine-learning/bringing-ai-powered-answers-and-summaries-to-file-previews-on-the-web
10. Clemm, J. (Dropbox). "Knowledge graphs, MCP, and DSPy in Dash" (talk write-up). Dropbox Tech Blog, 2025. https://dropbox.tech/machine-learning/vp-josh-clemm-knowledge-graphs-mcp-and-dspy-dash
11. Slack Engineering. "How We Built Slack AI To Be Secure and Private." Slack, 2024. https://slack.engineering/how-we-built-slack-ai-to-be-secure-and-private/
12. Microsoft. "Microsoft 365 Copilot overview" (architecture and grounding flow). Microsoft Learn, 2024-2025. https://learn.microsoft.com/en-us/copilot/microsoft-365/microsoft-365-copilot-overview
13. Microsoft. "Semantic indexing for Microsoft 365 Copilot." Microsoft Learn, 2024-2025. https://learn.microsoft.com/en-us/microsoftsearch/semantic-index-for-copilot
14. Microsoft. "Data, Privacy, and Security for Microsoft 365 Copilot." Microsoft Learn, 2024-2025. https://learn.microsoft.com/en-us/copilot/microsoft-365/microsoft-365-copilot-privacy
15. Microsoft Security Response Center. "CVE-2025-32711: Microsoft 365 Copilot Information Disclosure Vulnerability." Microsoft, 2025. https://msrc.microsoft.com/update-guide/vulnerability/CVE-2025-32711
16. Aim Labs (Aim Security). "EchoLeak: zero-click AI vulnerability in Microsoft 365 Copilot." Aim Security, 2025. https://www.aim.security/lp/aim-labs-echoleak-blogpost
17. Glean. Glean engineering and product blog (posts on RAG, hybrid search, the enterprise knowledge graph, permissions and agentic reasoning). Glean, 2023-2025. https://www.glean.com/blog
18. Sriharsha, R. "Reimagining the vector database to enable knowledgeable AI" (Pinecone serverless architecture; Notion workload example). Pinecone, 2024. https://www.pinecone.io/blog/serverless-architecture/
19. Notion. "Building and scaling Notion's data lake." Notion Blog, 2024. https://www.notion.com/blog/building-and-scaling-notions-data-lake
20. Airbnb Engineering. "Automation Platform v2: Improving Conversational AI at Airbnb." The Airbnb Tech Blog, 2024. https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb
21. Shopify Engineering. "Building production-ready agentic systems: Lessons from Shopify Sidekick." Shopify, 2025. https://shopify.engineering/building-production-ready-agentic-systems
22. Ramp Engineering. "From RAG to Richness: How Ramp Revamped Industry Classification." Ramp Engineering Blog, 2025. https://engineering.ramp.com/
23. OpenAI. "Harvey: customizing models for legal professionals" (customer story; custom case-law model). OpenAI, 2024. https://openai.com/index/harvey/ (see also Harvey blog: https://www.harvey.ai/blog)
24. Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., Ho, D. E. (Stanford RegLab / HAI). "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools." 2024. https://arxiv.org/abs/2405.20362
25. Thomson Reuters Labs. "Better Customer Support Using Retrieval-Augmented Generation (RAG) at Thomson Reuters." TR Labs ML Engineering Blog, 2023. https://medium.com/tr-labs-ml-engineering-blog
26. OpenAI. "Morgan Stanley uses AI evals to shape the future of financial services" (customer story). OpenAI, 2024. https://openai.com/index/morgan-stanley/
27. Klarna. "Klarna AI assistant handles two-thirds of customer service chats in its first month." Klarna press release, February 2024. https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/
28. LangChain. "How Klarna's AI assistant redefined customer support at scale" (customer story). LangChain Blog, 2024-2025. https://blog.langchain.dev/customers-klarna/
29. Leng, Q., Portes, J., Havens, S., Zaharia, M., Carbin, M. (Databricks Mosaic Research). "Long Context RAG Performance of Large Language Models." Databricks, 2024. https://www.databricks.com/blog/long-context-rag-performance-llms
30. Databricks. "Best Practices for LLM Evaluation of RAG Applications." Databricks Blog, 2023. https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
31. OpenAI. "Stripe" (customer story: GPT-4 for Stripe Docs, support and fraud use cases). OpenAI, 2023. https://openai.com/index/stripe/
