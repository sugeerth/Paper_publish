# 24. Design an LLM-Powered Enterprise Q&A / RAG Assistant (Chat over Company Documents)

> How the prompt is usually worded: "Design an assistant that lets employees ask natural-language
> questions over internal documents (wiki, Google Drive, Confluence, Slack, tickets, PDFs)
> and get grounded answers with citations. Users must only see what they are allowed to see."
> Variants: "Chat with your docs / ChatPDF", "LLM customer-support bot over a help center",
> "Enterprise search with LLM answers", "On-call copilot over runbooks", "Agentic RAG".

> Verification note (review pass): the shared web-search budget was used up, and only
> github.com and anthropic.com resolved. uber.com, linkedin.com, arxiv.org, huyenchip.com,
> github.blog, pinecone.io and igotanoffer.com did not. Tags used below:
> - (fetched): opened and checked in this run.
> - (repo-link): the exact URL was found in a GitHub list fetched in this run ([3], [11] or
>   [21]). The page itself could not be opened.
> - (ch-ref): verified in ch. 01 of this atlas.
> - (not verified): neither opened nor found in a fetched list.
>
> Figures marked "approx." come from memory of the cited post and should be checked against it.

## Company tags

| Company / context | Basis | Production system it mirrors |
|---|---|---|
| Any company with a "GenAI / LLM system design" round | alirezadir lists "Enterprise search or RAG assistant" as **Tier 1: Almost always** in 2026 GenAI rounds [2] (fetched). The ML-SD template lists "RAG document Q&A / chat with your docs" and "Enterprise / semantic search with LLM answers" [1] (fetched) | Generic retrieve -> rerank -> ground -> generate |
| OpenAI, Anthropic | Their question banks lean toward LLM systems ("Design ChatGPT" and similar) [22][23][29] (ch-ref) | Chat-assistant products with retrieval and file Q&A |
| Uber | Genie, an on-call copilot that answers in Slack over internal docs, upgraded to "Enhanced Agentic RAG". The alirezadir RAG overview cites it as the production agentic-RAG example [3] (fetched) [14] (repo-link) | **The real system in Diagram 4** |
| LinkedIn | Published a GenAI product post-mortem covering routing, retrieval, generation and evaluation [13] (repo-link via [11]) | Router + RAG over internal APIs |
| Instacart, Salesforce, Vimeo, Airbnb, GitHub | Internal assistant (Ava), Einstein Search Answers, a support bot, a conversational-AI automation platform, and Copilot [15]-[19] (repo-link via [21]) | Internal / support RAG assistants |
| Glean, Perplexity (reference products, not interview tags) | Named as enterprise RAG examples in [3] (fetched) | Permission-aware enterprise search + LLM answers |

**Portal coverage**

| Portal / source | What it covers | Link |
|---|---|---|
| alirezadir ML-Interviews + Agentic-AI-Systems | Enterprise RAG template, must-cover list, bar-raiser moves, RAG taxonomy (CRAG, Self-RAG, Adaptive, KG-RAG) | [1][2][3][4] |
| Educative | "RAG knowledge assistant" case in Grokking MLSD (ch-ref). Grokking the Generative AI System Design and Advanced RAG Techniques are listed in [5] (fetched) | [24][25][26] |
| ByteByteGo (Aminian et al.) | The "Generative AI System Design Interview" book is reported to have a RAG chapter (not verified in this run) | [30] |
| IGotAnOffer | GenAI system design guide and OpenAI system design guide | [22][23] |
| Chip Huyen, "AI Engineering" (O'Reilly) | Ch. 6 "RAG and Agents": RAG Architecture, Retrieval Algorithms, Retrieval Optimization, RAG Beyond Texts, Agents, Memory [11] (fetched). GenAI platform post [12] | [11][12] |
| Anthropic engineering | Contextual Retrieval (measured retrieval-failure reductions). "Building effective agents": routing, prompt chaining, evaluator-optimizer | [6][34] (fetched) |
| Hello Interview / Exponent / PracHub | MLSD delivery frameworks. LLM-system question banks | [27][28][29] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Ask questions in natural language. Get a grounded answer with citations. Multi-turn follow-ups. Say "I don't know" when unsupported | [2] |
| Corpus sources | Wiki/Confluence, Drive/Docs, PDFs, Slack, tickets, code. Connectors per source. MCP is an option for standard tool/data connectors, but treat each protocol boundary as a trust boundary | [4] |
| Access control | Per-document ACLs and tenant isolation, applied at **retrieval time** | [2] |
| Freshness | New or edited docs searchable in under 15 min. ACL revocations take effect in under 5 min | typical assumption |
| Corpus size | 10M docs -> about 200M chunks (about 20 chunks per doc, chunks of a few hundred tokens) [6] | assumption |
| Index size | 200M x 1024-d fp16 is about 400 GB raw. About 25-50 GB with PQ / int8 | derived |
| Users / QPS | 50k employees x 10 q/day = 500k q/day: about 6 QPS average, about 30 QPS peak. A multi-tenant SaaS with 1k tenants reaches about 3-5k QPS peak | assumption |
| Latency | Retrieval + rerank p95 under 400 ms. Time to first token (TTFT) under 1.5 s. Full answer under 8 s, streamed | typical assumption |
| Quality | Faithfulness of 0.9 or higher on the golden set. Citation precision 0.9 or higher. Hallucinated-answer rate under 2% | typical target |
| Cost | Track tokens per query. Route easy queries to a small model. Cache prompts and responses | [1][2] |
| Safety | Defend against prompt injection (including from retrieved docs). Redact PII. Keep audit logs. Honor deletion / GDPR | [1][4] |

## Diagram 1 - Online serving path (query -> retrieve -> filter -> rerank -> generate -> response)

**Figure 24.1 - End-to-end online architecture of a permission-aware RAG assistant**

```text
 +---------+  question  +--------------+  user_id, groups   +-------------------------+
 | Client  |----------->| API gateway  |------------------->| Conversation svc        |
 | web /   |            | authN, rate  |                    | session, history,       |
 | Slack   |            | limit        |                    | memory summary          |
 +----^----+            +--------------+                    +-------------------------+
      |                                                                  |
      |                                                                  v
      |                                                     +-------------------------+
      |                                                     | Input guardrail         |
      |                                                     | PII, jailbreak,         |
      |                                                     | policy classifier       |
      |                                                     +-------------------------+
      |                                                                  |
      |                                                                  v
      |               +------------------------+  lookup    +-------------------------+
      |               | Semantic answer        |<-----------| Query understanding     |
      +--hit----------| cache  key = query     |            | router: chit-chat /     |
      |               | emb + ACL-group hash   |            | RAG / tool; rewrite     |
      |               +------------------------+            | with history; HyDE;     |
      |                                                     | decompose multi-hop     |
      |                                                     +-------------------------+
      |                                             miss: 1..n queries   |
      |                            +-------------------------------------+
      |                            |                                     |
      |                            v                                     v
      |               +------------------------+            +-------------------------+
      |               | Dense ANN (HNSW/IVF)   |            | BM25 inverted index     |
      |               | top-100                |            | top-100                 |
      |               | ACL pre-filter:        |            | ACL pre-filter:         |
      |               | tenant, groups, type   |            | tenant, groups, type    |
      |               +------------------------+            +-------------------------+
      |                            |                                     |
      |                            +------------------+------------------+
      |                                               v
      |                                +----------------------------+
      |                                | Fusion (RRF) + dedupe      |
      |                                | ~150 candidates            |
      |                                +----------------------------+
      |                                               v
      |                                +----------------------------+
      |                                | Cross-encoder reranker     |
      |                                | -> top-k (k = 5..20)       |
      |                                | ACL re-check post-filter   |
      |                                +----------------------------+
      |                                               v
      |                                +----------------------------+
      |                                | Context builder: token     |
      |                                | budget, best-first order,  |
      |                                | chunk ids for citations    |
      |                                +----------------------------+
      |                                               v
      |   +------------------+         +----------------------------+
      |   | Model registry   |-------->| LLM gateway / router       |
      |   | prompt store     |         | small vs large model,      |
      |   +------------------+         | fallback, prompt + KV      |
      |                                | cache, streaming           |
      |                                +----------------------------+
      |                                               v
      |  answer + citations (streamed) +----------------------------+
      +--------------------------------| Output guardrail           |
                                       | groundedness, cite check,  |
                                       | abstain, redact            |
                                       +----------------------------+
                                                      | traces, spans, thumbs
                                                      v
                                       +----------------------------+
                                       | Logging / observability    |
                                       | query, chunks, prompt,     |
                                       | answer, tokens, cost,      |
                                       | latency, feedback          |
                                       +----------------------------+
```

- The "feature store" of classic MLSD maps to the **chunk store + metadata/ACL store**. The
  "model store" maps to the **model registry + prompt store** [12].
- Apply ACL filtering **before** generation, never after. A post-filter re-check guards
  against stale ACLs [2]. Context selection should also weigh recency, authority and
  permission filters [4].
- The semantic-cache key includes the caller's ACL-group hash. Without it, a cached answer
  can leak one user's documents to another user.
- Prompt caching of the stable system prompt and document prefix can cut cost by up to 90%
  and latency by more than 2x [6].

## Diagram 2 - Offline pipeline: ingestion, indexing, evaluation, feedback loop

**Figure 24.2 - Offline indexing, model adaptation, and evaluation pipeline**

```text
 +-------------+  +-------------+  +-------------+  +-------------+
 | Confluence  |  | Drive / PDF |  | Slack       |  | Tickets /   |
 | wiki        |  | docs        |  | threads     |  | code        |
 +-------------+  +-------------+  +-------------+  +-------------+
        |                |                |                |
        +----------------+-------+--------+----------------+
                                 | connectors: full crawl + CDC / webhooks
                                 v (edits, deletes, ACL changes)
                +---------------------------------+
                | Parse + clean: OCR, layout,     |
                | tables -> markdown, dedupe,     |
                | PII tagging, ACL capture        |
                +---------------------------------+
                                 v
                +---------------------------------+       +---------------------------+
                | Chunking: heading-aware,        |       | Contextualize chunk       |
                | 300-800 tok + overlap,          |------>| (LLM prepends 50-100 tok  |
                | parent-child links              |       | of doc context) [6]       |
                +---------------------------------+       | optional                  |
                                 |                        +---------------------------+
                                 |      chunks (+ context prefix)       |
                  +--------------+--------------------------+-----------+
                  |                                         |
                  v                                         v
      +------------------------+             +----------------------------+
      | Embedding service      |             | BM25 / inverted index      |
      | batch GPU, versioned   |             | build (same chunks)        |
      +------------------------+             +----------------------------+
                  |                                         |
                  v                                         v
      +--------------------------------------------------------------------+
      | Vector DB + lexical index + chunk store + ACL / metadata store     |
      | (versioned; tombstones for deletes; blue/green on embedder change) |
      +--------------------------------------------------------------------+

 ====================== evaluation & model adaptation loop ======================

 +---------------+     +-------------------+     +------------------------------------+
 | Prod logs:    |---->| Label / join:     |---->| Golden set: (question, gold        |
 | queries, ctx, |     | thumbs, citation  |     | chunks, reference answer);         |
 | answers,      |     | clicks, SME,      |     | SME + synthetic Q from docs        |
 | feedback      |     | LLM judge         |     +-----------------+------------------+
 +-------^-------+     +-------------------+                       v
         |             +--------------------------------------------------------------+
         |             | Train / tune: embedder (contrastive, hard negatives),        |
         |             | reranker (cross-encoder), optional LoRA on LLM,              |
         |             | prompt versions                                              |
         |             +--------------------------------------------------------------+
         |                                                         v
         |             +--------------------------------------------------------------+
         |             | Offline eval: recall@k, MRR, nDCG (retrieval);               |
         |             | faithfulness, answer relevance, context precision            |
         |             | (Ragas-style, LLM judge); regression + red-team              |
         |             +--------------------------------------------------------------+
         |                                                         | pass gates
         |                                                         v
         |             +---------------------+    +-----------------------------------+
         |             | Registry: embedder, |--->| Deploy: shadow -> canary -> A/B;  |
         |             | reranker, prompt,   |    | re-embed if embedder changed      |
         |             | LLM config          |    | (blue/green index)                |
         |             +---------------------+    +-----------------------------------+
         |                                                         |
         +-------------------- feedback loop ----------------------+
```

## Diagram 3 - Model architecture: bi-encoder retriever + cross-encoder reranker + LLM generator

**Figure 24.3 - The three learned components and their inputs/outputs**

```text
   query q (rewritten)                         chunk c (offline)
          |                                           |
          v                                           v
 +-------------------+                      +-------------------+
 | Query encoder     |  shared or separate  | Doc encoder       |
 | (BERT/E5-style)   |  weights (bi-encoder)| (same family)     |
 +---------+---------+                      +---------+---------+
           | e_q (d=768..1024)                        | e_c, stored in ANN index
           +---------------------+--------------------+
                                 v
                    +-------------------------+   loss: InfoNCE with in-batch
                    | sim = cos(e_q, e_c)     |   + mined hard negatives
                    | ANN top-100  (+ BM25)   |
                    +------------+------------+
                                 | candidates
                                 v
                    +-------------------------+
                    | Cross-encoder reranker  |   input: [CLS] q [SEP] c [SEP]
                    | full attention over q,c |   output: relevance score s(q,c)
                    | -> top-k                |   loss: BCE / listwise softmax
                    +------------+------------+
                                 | top-k chunks + ids
                                 v
 +----------------------------------------------------------------------+
 | Prompt = system policy + "answer only from sources, cite [id],       |
 |          say 'not found' otherwise" + chat history summary +         |
 |          [1] chunk ... [k] chunk + user question                     |
 +-----------------------------------+----------------------------------+
                                     v
                    +-------------------------+
                    | Decoder-only LLM        |   out: answer tokens with [id]
                    | (API or self-hosted,    |   citations, streamed
                    | optional LoRA)          |   loss (if tuned): next-token
                    +------------+------------+   cross-entropy on grounded answers
                                 v
                    +-------------------------+
                    | Groundedness verifier   |   NLI / LLM judge: is each claim
                    | (small model)           |   supported by the cited chunk?
                    +-------------------------+   -> keep, rewrite, or abstain
```

- The original RAG paper (Lewis et al. 2020) trained a DPR retriever and a BART generator
  jointly [7] (repo-link via [11]), building on the retriever-reader design of DrQA [9]. Production systems usually keep the components **decoupled**
  and use a frozen LLM [8][11].

## Diagram 4 - Real system: Uber Genie "Enhanced Agentic RAG" (EAg-RAG)

**Figure 24.4 - Agentic RAG variant modelled on Uber's on-call copilot (approx., per [14])**

```text
 +-------------+  question    +--------------------------------------+
 | Engineer in |------------->| Genie bot (Slack)                    |
 | Slack       |<------+      +--------------------------------------+
 +-------------+       |                          |
   cited answer        |                          v
   (bot reply)         |        +-----------------------------------+
                       |        | Query Optimizer agent: rewrite    |
                       |        | ambiguous query, split into       |
                       |        | sub-queries                       |
                       |        +-----------------------------------+
                       |                          v
                       |        +-----------------------------------+
                       |        | Source Identifier agent: pick     |
                       |        | the relevant doc set / policy     |
                       |        | area (narrows search space)       |
                       |        +-----------------------------------+
                       |                          v
                       |        +-----------------------------------+
                       |        | Hybrid retrieval: vector + BM25   |
                       |        | over enriched chunks (tables      |
                       |        | kept intact, LLM summaries,       |
                       |        | metadata)                         |
                       |        +-----------------------------------+
                       |                          v
                       |        +-----------------------------------+
                       |        | Post-processor agent: dedupe,     |
                       |        | order by doc structure,           |
                       |        | assemble context                  |
                       |        +-----------------------------------+
                       |                          v
                       |        +-----------------------------------+
                       +--------| LLM answer generation             |
                                | with citations                    |
                                +-----------------------------------+
                                                  | logged Q/A pairs
                                                  v
                    +----------------------------------------------------------+
                    | OFFLINE: LLM-as-judge over a golden set (SME-curated),   |
                    | scored acceptable / incorrect; drives doc-processing and |
                    | agent-prompt changes                                     |
                    +----------------------------------------------------------+
```

- Reported results (approx., not verified in this run; check [14]): a sizeable relative
  gain in "acceptable" answers and a large relative drop in incorrect advice versus the
  earlier plain-RAG Genie.
  Most of the gain came from **better document processing**, not a better LLM.
- The same pattern shows up elsewhere. LinkedIn routes intents to "skills", retrieves via
  internal APIs, then generates. Its post says most of the effort went into evaluation
  guidelines, and that pushing quality past "good enough" took far longer than the first
  build [13] (repo-link via [11]). Practitioner write-ups report the same pattern [32].

## Data & labels

| Data | Source | Label / use | Pitfall |
|---|---|---|---|
| Documents + ACLs | Connectors, CDC | Corpus, permission filter | Stale ACL -> leak |
| (query, positive chunk) pairs | Citation clicks, SME annotation, synthetic Q from chunks (LLM) | Train the embedder and reranker | Synthetic Q are too "easy" (lexical overlap) |
| Hard negatives | BM25 / ANN top-k that is not gold | Contrastive training | False negatives (duplicate docs) |
| Golden QA set (200-2k) | SMEs + logs | Offline regression eval | Goes stale as docs change |
| Answer feedback | Thumbs, edits, escalations, re-asks | Online quality signal, judge calibration | Sparse; skews negative |
| LLM-judge labels | Judge model + rubric | Faithfulness, relevance at scale | Judge bias; calibrate on human labels [13] |

## Features / signals used in ranking and context building

| Group | Examples |
|---|---|
| Query | Rewritten text, intent class, entities, language, conversation history summary |
| Chunk content | Dense embedding, BM25 terms, contextual prefix [6], title/heading path |
| Chunk metadata | Source, doc type, author, last-modified (freshness), authority / official flag, popularity |
| User | Team, org, permissions (groups), recent docs (light personalization) |
| Cross | Cross-encoder score, RRF rank, dense/lexical agreement, same-doc neighbours |

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Knowledge strategy | Long context (stuff the docs) | RAG. Plus LoRA for style/format. Agentic RAG [1][10] | RAG = fresh + citable. Fine-tune = behaviour, not facts [10] |
| Chunking | Fixed 512 tok + overlap | Heading/semantic chunking, parent-child, contextual chunks [6][31] | Bigger chunks: recall up, precision and tokens worse. Contextual chunks cost about $1.02 per 1M doc tokens once, with prompt caching [6] |
| Retrieval | BM25 | Dense bi-encoder + BM25 hybrid with RRF. Fine-tuned embedder | Dense misses IDs/codes. BM25 misses paraphrase |
| Rerank | None | Cross-encoder on top-150 -> top-20 [6] | +50-150 ms. Big precision gain |
| Query processing | Raw query | Rewrite with history, decompose multi-hop, HyDE [1] | Extra LLM call (latency) |
| Generator | Hosted frontier LLM | Router: small model for easy queries, large for hard. Self-host + vLLM for scale [1] | Cost vs quality. Data residency |
| Control flow | Fixed pipeline (workflow: routing, prompt chaining [34]) | Agentic: decide when and where to retrieve, iterate, abstain (CRAG / Self-RAG / Adaptive). Evaluator-optimizer loop for answer checking [3][4][14][33][34] | Better on multi-hop. Harder to bound latency and cost |

## Loss functions

| Component | Loss |
|---|---|
| Bi-encoder retriever | InfoNCE / multiple-negatives ranking loss (in-batch + hard negatives) |
| Cross-encoder reranker | Pointwise BCE, or listwise softmax CE over (1 pos, n neg) |
| LLM fine-tune (optional) | Next-token cross-entropy on grounded, cited answers (SFT). DPO on preferred vs rejected answers |
| Groundedness verifier | BCE on (claim, evidence) -> supported / not supported (NLI) |

## Offline metrics

| Layer | Metric | Notes |
|---|---|---|
| Retrieval | Recall@k (e.g. @20), MRR, nDCG@10 | Anthropic: top-20 retrieval failure 5.7% -> 3.7% with contextual embeddings (-35%), -> 2.9% adding contextual BM25 (-49%), -> 1.9% adding a reranker (-67%) [6] (fetched) |
| Context | Context precision, context recall | Ragas definitions [20] (fetched) |
| Generation | Faithfulness = supported claims / total claims | Ragas [20] |
| Generation | Answer relevancy, answer correctness, factual correctness vs reference | LLM judge + embeddings [20] |
| Context | Context entities recall (entity-heavy domains: names, IDs, products) | Ragas [20] |
| Citations | Citation precision / recall (does the cited chunk support the sentence?) | Human audit sample |
| Robustness | Noise sensitivity, abstain rate on unanswerable questions, prompt-injection pass rate | Red-team set [20] |
| Cost / latency | Tokens per answer, p95 latency, TTFT | Gate in CI |

## Online metrics / A-B

| Type | Metric |
|---|---|
| Primary | Answer acceptance: thumbs-up rate, or "resolved without follow-up" rate |
| Engagement | Weekly active askers, queries per user, citation click-through |
| Business | Ticket deflection (support), time-to-resolution, escalations avoided |
| Guardrails | Thumbs-down rate, reported hallucinations, leak incidents (must be 0), refusal rate, p95 latency, cost per query |
| Method | Shadow -> canary -> A/B on users or teams. Interleaving for retriever changes. LLM-judge on sampled live traffic |

## Serving & latency budget (target under 8 s full answer, TTFT about 1.5 s; typical assumption)

| Step | Budget (p95) | Tactics |
|---|---|---|
| Gateway + auth + guardrail | 50 ms | Small classifier, cached group lookups |
| Query rewrite / router | 150-300 ms | Small model, skip for single-turn simple queries |
| Hybrid retrieval + ACL filter | 50-100 ms | HNSW with metadata pre-filter, sharded by tenant |
| Rerank top-150 | 80-150 ms | Distilled cross-encoder on GPU, batch |
| Prompt build | 10 ms | Token budget, dedupe |
| LLM TTFT | 500-900 ms | Prompt caching, KV cache, continuous batching (vLLM), streaming [1] |
| LLM decode (300 tok) | 3-6 s (streamed) | Small model routing, speculative decoding |
| Output check | 100-300 ms (async or partial) | Verify after the stream; retract or flag if needed |

## Monitoring

| What | Signal | Action |
|---|---|---|
| Retrieval health | Empty-result rate, top-1 score distribution, recall on a canary set | Alert. Check embedder / index version |
| Quality drift | Sampled LLM-judge faithfulness, thumbs-down trend by source | Golden-set regression. Re-index |
| Freshness | Ingestion lag, CDC backlog, % of docs stale | Scale connectors |
| Security | ACL mismatch audits, injection detections, PII in outputs | Block. Page security |
| Cost / latency | Tokens per query, cache hit rate, provider errors, TTFT | Route to a smaller model. Fail over |
| Traces | Per-step spans (rewrite, retrieve, rerank, generate, tool calls), cost and evidence per step | Trace replay to debug bad answers. Online evals on live traces [2][4] |
| Safety events | Prompt injection, policy refusal, PII exposure, unsafe tool request | Count, alert, add to the red-team set [4] |

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| Hallucination / unsupported claim | Weak context, model prior | "Answer only from sources", groundedness check, abstain |
| Permission leak | ACL filter after generation, stale ACL sync, cache shared across users | Pre-filter at retrieval. Cache key includes ACL hash. Fast ACL CDC |
| Retrieval miss | Bad chunking (tables split), vocabulary mismatch, IDs/acronyms | Hybrid BM25, contextual chunks, structure-aware parsing [6][14] |
| Lost in the middle / context overflow | Too many chunks | Rerank, fewer chunks, order best-first |
| Indirect prompt injection | Malicious text inside a retrieved doc | Treat retrieved text as data, spotlight/delimit it, output filters, restrict tool scopes |
| Stale or conflicting answers | Old doc versions | Freshness and authority features, dedupe versions, show dates |
| Deletion / compliance | Deleted doc still in index or cache | Tombstones, cache invalidation, re-embed jobs |
| Agent acts with stale or broad permissions | Over-permissioned tools or connectors, cached credentials | Scope credentials per tool, task, user and tenant. Re-check ACLs at call time [4] |
| Cost blow-up | Long histories, agent loops | History summarization, step and budget caps |
| Embedder upgrade breaks index | Vector-space mismatch | Blue/green re-index, dual-read during migration |

## Interviewer follow-ups

- RAG vs fine-tuning vs long context: when would you use each, and when would you combine them? [1][10]
- How do you enforce document permissions? Pre-filter vs post-filter. How fast must ACL changes propagate? [2]
- How do you pick chunk size and chunking strategy for PDFs with tables and headings? [6][31]
- Why hybrid search? Give a query that dense retrieval fails on. How do you fuse the scores?
- How do you evaluate without labelled data? Golden set, synthetic questions, LLM-as-judge and its biases [13][20]
- How do you detect and reduce hallucinations at serving time? When should the system abstain?
- How do you handle multi-turn follow-ups ("what about for contractors?")? Query rewriting with history.
- Multi-hop questions that span several docs: decomposition or agentic RAG? How do you bound the latency? [3][14]
- How do you keep the index fresh, and how do you handle deletes and right-to-be-forgotten? [2]
- How would you cut cost by 5x? Caching, routing, smaller models, prompt compression.
- How do you defend against prompt injection hidden inside documents? [1][4]
- How do you scale to 1,000 tenants? Tenant isolation, sharding, noisy neighbours.
- You upgrade the embedding model: how do you migrate a 200M-vector index safely?
- Which online metric would you ship on, and what are your guardrails?
- Where does the LLM fit, and where should deterministic systems take over? [1]
- Workflow or agent? When is a fixed routing / chaining pipeline enough, and when does an agent loop earn its cost? [34]
- How do connectors reach Slack, Drive and tickets (custom crawlers vs MCP), and how do you scope their credentials? [4]
- Contextual retrieval: what does prepending doc context to each chunk buy, and what does it cost to index? [6]

## Portal comparison

| Portal | Framing | Emphasis | Distinctive point |
|---|---|---|---|
| alirezadir Agentic-AI-Systems [2][3] | "Enterprise RAG assistant" template (Tier 1) | Ingestion, hybrid search, ACL and tenant isolation, faithfulness evals | Bar-raiser list: multi-hop, deletion, abstention, semantic cache, agentic RAG |
| alirezadir ML-SD template [1] | GenAI lens on the 9-step MLSD formula | "Evaluation is the new system design". Cost, latency, guardrails | RAG triad (faithfulness, answer relevance, context relevance) + recall@k / MRR / nDCG |
| Chip Huyen, AI Engineering [11][12] | RAG as context construction inside a GenAI platform | Retrieval algorithms and optimization, guardrails, router, gateway, cache | Finetuning vs RAG analysis. Platform-level view |
| Educative (Grokking MLSD / GenAI SD) [24][25] | RAG knowledge-assistant case study | Step-by-step framework, training + inference | Course format with model and serving sections |
| ByteByteGo GenAI SDI [30] | RAG chapter in the 7-step ML framework (not verified in this run) | Indexing, retrieval, generation, evaluation | Diagram-heavy, interview-script style |
| Uber / LinkedIn blogs [13][14] | Production post-mortems | Document processing, agents, evaluation effort | Real numbers and lessons learned. Not interview-shaped |

## References

1. alirezadir, "ML System Design" (GenAI / LLM System Design 2026 section), GitHub, 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
2. alirezadir, "GenAI and Agentic System Design Interview Prep" (Enterprise RAG Assistant template, topic tiers), GitHub, 2026. https://github.com/alirezadir/Agentic-AI-Systems/blob/main/06_interview_prep/genai-agentic-system-design.md (fetched)
3. alirezadir, "Retrieval-Augmented Generation (RAG)" overview, Agentic-AI-Systems, GitHub, 2025. https://github.com/alirezadir/Agentic-AI-Systems/blob/main/03_system_design/RAGs/README.md (fetched)
4. alirezadir, "2026 Agentic AI System Design Update", Agentic-AI-Systems, GitHub, 2026. https://github.com/alirezadir/Agentic-AI-Systems/blob/main/03_system_design/2026-agentic-ai-system-design.md (fetched)
5. alirezadir, "GenAI resources", GitHub, 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/genai-resources.md (fetched)
6. Anthropic, "Introducing Contextual Retrieval", anthropic.com, 2024. https://www.anthropic.com/news/contextual-retrieval (fetched)
7. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", arXiv, 2020. https://arxiv.org/abs/2005.11401 (repo-link via [11])
8. Y. Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey", arXiv, 2023. https://arxiv.org/abs/2312.10997 (repo-link via [11])
9. D. Chen et al., "Reading Wikipedia to Answer Open-Domain Questions", arXiv, 2017. https://arxiv.org/abs/1704.00051 (repo-link via [11])
10. O. Ovadia et al., "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs", arXiv, 2024. https://arxiv.org/abs/2312.05934 (repo-link via [11])
11. C. Huyen, "AI Engineering" book resources (Ch. 6 RAG and Agents) and table of contents, GitHub, 2025. https://github.com/chiphuyen/aie-book/blob/main/resources.md (fetched); https://github.com/chiphuyen/aie-book/blob/main/ToC.md (fetched)
12. C. Huyen, "Building A Generative AI Platform", huyenchip.com, 2024. https://huyenchip.com/2024/07/25/genai-platform.html (not verified)
13. J. P. Bottaro & K. Ramgopal, "Musings on Building a Generative AI Product", LinkedIn Engineering, 2024. https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product (repo-link via [11])
14. Uber Engineering, "Enhanced Agentic-RAG: What If Chatbots Could Deliver Near-Human Precision?", uber.com, 2025. https://www.uber.com/blog/enhanced-agentic-rag/ (repo-link via [3])
15. Instacart, "Scaling Productivity with Ava - Instacart's Internal AI Assistant", tech.instacart.com, 2023. https://tech.instacart.com/scaling-productivity-with-ava-instacarts-internal-ai-assistant-ed7f02558d84 (repo-link via [21])
16. Salesforce AI Research, "Resolve Cases Quickly with Interactive Einstein Search Answers", blog.salesforceairesearch.com, 2023. https://blog.salesforceairesearch.com/einstein-search-answers/ (repo-link via [21])
17. Vimeo Engineering, "From idea to reality: Elevating our customer support through generative AI", Medium, 2023. https://medium.com/vimeo-engineering-blog/from-idea-to-reality-elevating-our-customer-support-through-generative-ai-101a2c5ea680 (repo-link via [21])
18. Airbnb Engineering, "Intelligent Automation Platform: Empowering Conversational AI and Beyond at Airbnb", Medium, 2022. https://medium.com/airbnb-engineering/intelligent-automation-platform-empowering-conversational-ai-and-beyond-at-airbnb-869c44833ff2 (repo-link via [21])
19. GitHub, "How to build an enterprise LLM application: Lessons from GitHub Copilot", github.blog, 2023. https://github.blog/2023-09-06-how-to-build-an-enterprise-llm-application-lessons-from-github-copilot/ (repo-link via [21])
20. Ragas, "Available metrics" docs (faithfulness, context precision/recall, answer relevancy, noise sensitivity), GitHub, 2025. https://github.com/explodinggradients/ragas/tree/main/docs/concepts/metrics/available_metrics (fetched)
21. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
22. IGotAnOffer, "Generative AI System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/generative-ai-system-design-interview (ch-ref)
23. IGotAnOffer, "OpenAI System Design Interviews", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/openai-system-design-interview (ch-ref)
24. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework (ch-ref)
25. Educative, "Grokking the Generative AI System Design", educative.io, 2025. https://www.educative.io/courses/generative-ai-system-design (repo-link via [5])
26. Educative, "Advanced RAG Techniques", educative.io, 2025. https://www.educative.io/courses/advanced-rag-techniques (repo-link via [5])
27. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
28. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
29. PracHub, "ML System Design Interview Questions", prachub.com, 2026. https://prachub.com/categories/ml-system-design?page=20 (ch-ref)
30. ByteByteGo, home page (courses incl. ML and Generative AI System Design Interview), bytebytego.com, 2025. https://bytebytego.com (ch-ref; chapter title not verified)
31. Pinecone, "Chunking Strategies for LLM Applications", pinecone.io, 2023. https://www.pinecone.io/learn/chunking-strategies/ (repo-link via [11])
32. E. Yan et al., "Applied LLMs: What We've Learned From A Year of Building with LLMs", applied-llms.org, 2024. https://applied-llms.org/ (repo-link via [11])
33. LangGraph, "Agentic RAG" example notebook, GitHub, 2024. https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb (fetched)
34. E. Schluntz & B. Zhang, "Building effective agents", Anthropic Engineering, 2024. https://www.anthropic.com/engineering/building-effective-agents (fetched)
