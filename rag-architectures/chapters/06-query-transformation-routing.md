# 06. Query Understanding and Transformation: HyDE, Multi-Query, RAG-Fusion, Step-Back, Decomposition, Conversational Rewriting, Routing and Self-Query

## Overview

Most of a RAG pipeline is designed around documents: how they are parsed, chunked, embedded
and indexed (Chapters 02-04). The query gets far less design attention, even though it is the
weakest input in the system. Users type short, ambiguous, misspelled or context-dependent
text ("what about the second one?"). They mix several questions into one sentence. They state
constraints ("from last year", "under three minutes") that belong in a filter rather than in
an embedding. And they use vocabulary that differs from the vocabulary of the corpus. A single
embedding of the raw string tends to blur all of this together. The AWS Bedrock team describes
the effect as the embedding step that "may mask or dilute important components of the query"
[27].

*Query understanding and transformation* is the stage between the user's raw input and the
retrievers. Its job is to turn what the user typed into what the retrieval system can act on.
This chapter covers the publicly documented designs for that stage:

1. **Generative expansion.** HyDE writes a hypothetical answer document and searches with its
   embedding [1][2]. Query2doc appends a generated pseudo-document to a sparse query [3].
   Later work shows when expansion hurts strong retrievers and how to steer it with the
   corpus itself [45][46][47].
2. **Multi-query and RAG-Fusion.** An LLM writes several paraphrases, each is retrieved
   separately, and the ranked lists are merged with Reciprocal Rank Fusion (RRF) [6][11][12].
   This section includes the 2026 debate over whether fusion gains survive reranking [7][8][9],
   and adaptive rewrite-strategy selection (DMQR-RAG) [48].
3. **Abstraction.** Step-back prompting asks a more general question first and retrieves for
   it alongside the original question [5][16].
4. **Decomposition.** Least-to-most, Self-Ask, IRCoT, sub-question engines, and the managed
   query planners in Amazon Bedrock and Azure AI Search [19][25][27][34][35][36].
5. **Learned rewriters.** Rewrite-Retrieve-Read, RQ-RAG, DeepRetrieval and Search-R1 train
   the query writer with retrieval or answer feedback instead of prompting it [4][37][41][42].
6. **Conversational rewriting.** Condense-question and history-aware retrievers, and the
   conversational-search research line (LLM4CS, ConvGQR, CHIQ, ConvSearch-R1)
   [13][22][38][39][40][49].
7. **Routing.** LLM classifiers, embedding-based semantic routers, complexity routers and
   effort routers [17][21][26][29][32][33][50].
8. **Structured query construction.** Metadata filter extraction and "self-query", in which
   an LLM turns natural language into a search string plus a typed filter expression
   [14][15][23][28].

These techniques are composable. A production query layer often stacks several: condense the
conversation, route, extract filters, then decompose or expand.

```
                         QUERY UNDERSTANDING LAYER
  user turn + chat history
          |
          v
  +-------------------+   standalone question   +----------------------------+
  | 1. Contextualize  |------------------------>| 2. Route                   |
  | (condense/rewrite)|                         | (index / tool / effort /   |
  +-------------------+                         |  no-retrieval / model)     |
                                                +-------------+--------------+
                                                              |
              +-----------------------------------------------+
              v
  +---------------------------+     +---------------------------------------+
  | 3. Structure              |     | 4. Expand / transform                 |
  | query text + filter       |---->| multi-query | HyDE | step-back |      |
  | (self-query, implicit     |     | decomposition into sub-questions      |
  |  filters, top_k)          |     +-------------------+-------------------+
  +---------------------------+                         |
                                                        v
                         N retrievals in parallel (dense / BM25 / hybrid, filtered)
                                                        |
                                                        v
                     fuse (RRF / union + dedupe) -> rerank -> truncate -> generator
```

A note on sources. This chapter relies mainly on primary artifacts: official code
repositories for papers, framework source code, and vendor documentation, plus paper
abstracts. Figures that are reported only by a paper's authors or by a vendor are marked
as such.

## 1. Taxonomy: what is being transformed, and why

The query-side techniques fall into seven families. They differ in what they assume is wrong
with the raw query:

| Family | Diagnosis of the raw query | Output | Representative designs |
|---|---|---|---|
| Expansion by generation | Query and documents look different in embedding space (short question vs long answer passage) | A document-shaped string | HyDE [1], Query2doc [3], CSQE [46] |
| Paraphrase / multi-query | One phrasing misses relevant documents because of vocabulary mismatch | N alternative queries | MultiQueryRetriever [12], RAG-Fusion [6], DMQR-RAG [48], Azure query rewrite [24] |
| Abstraction | The query is too specific to retrieve the principles or background it needs | A higher-level question | Step-back [5] |
| Decomposition | The query contains several information needs, or needs multiple hops | Sub-questions, possibly sequential | Least-to-most [36], Self-Ask [34], IRCoT [35], sub-question engines [19], Bedrock/Azure planners [25][27] |
| Contextualization | The query is not standalone because it depends on earlier turns | A standalone question | Condense-question [22], history-aware retriever [13], LLM4CS [38], ConvSearch-R1 [49] |
| Routing | The query should go somewhere specific (an index, a tool, a model, an effort level, or no retrieval at all) | A decision label, optionally with parameters | Semantic router [29], RouterQueryEngine [21], Adaptive-RAG [33], RouteLLM [32] |
| Structuring | The query mixes semantic content with hard constraints | Query text + typed filter + limit | SelfQueryRetriever [15], auto-retriever [23], Bedrock implicit filters [28] |

Each family costs something. All but the embedding-based router add at least one LLM call
before retrieval can start, and that call is on the critical path. Measured data on this cost
is collected in Section 3.4.

## 2. Generative expansion: HyDE and Query2doc

### 2.1 HyDE (Hypothetical Document Embeddings)

**Problem.** Zero-shot dense retrieval with no relevance labels. Questions and answer
passages differ in length, style and vocabulary. An unsupervised encoder such as Contriever
therefore places a question far from the passages that answer it.

**Architecture.** HyDE was introduced by Gao, Ma, Lin and Callan (CMU / Waterloo) in December
2022 and published at ACL 2023 [1]. It splits retrieval into two steps. First, an
instruction-following LLM (InstructGPT, `text-davinci-003` in the released notebook) is
prompted zero-shot to "write a passage to answer the question". Second, the generated
passage, not the question, is encoded with the unsupervised retriever, and the nearest real
documents are returned. The authors describe the encoder's dense bottleneck as a filter for
the hallucinated details in the fake document. The fake passage only needs to sit in the
right neighbourhood; its facts do not need to be correct [1][2].

```
  query q --> LLM ("write a passage that answers q") x N samples
                    |  d1 ... dN  (hypothetical, may contain false facts)
                    v
  encoder f:  v = mean( f(q), f(d1), ..., f(dN) )      # official repo averages q + docs
                    |
                    v
  ANN search over real corpus embeddings --> top-k real documents
```

**Key techniques visible in the reference code [2]:**

- *Task-specific prompt templates.* The released `promptor.py` has one template per corpus:
  "write a scientific paper passage to support/refute the claim" (SciFact), "write a counter
  argument for the passage" (ArguAna), "write a financial article passage" (FiQA), "write a
  news passage about the topic" (TREC-News), and a multilingual "write a passage in {lang}"
  template (Mr.TyDi). The form of the hypothetical document is a design parameter: it should
  look like the documents in the target corpus.
- *Multiple samples, averaged.* `HyDE.encode` embeds the original query and every generated
  document, then averages the vectors into a single search vector. The DL19 notebook
  generates 8 hypothetical documents per query (its output files are tagged `8rep`).
- *Frameworks keep the query.* LlamaIndex's `HyDEQueryTransform` defaults to
  `include_original=True`. It adds the hypothetical passage to the query's embedding strings
  and keeps the original string as well [18].

**Numbers.** The repository's TREC DL19 notebook, run with saved outputs, reports Contriever
alone at nDCG@10 0.445, MAP 0.240 and Recall@1000 0.746. Contriever with HyDE (GPT-3, 8
samples) reaches nDCG@10 0.597, MAP 0.404 and Recall@1000 0.873 [2]. That is roughly +15
nDCG points with no training. The paper's own Table 1 reports a slightly higher DL19
figure for HyDE (nDCG@10 61.3, mAP 41.8, against Contriever's 44.5 and 24.0), close to the
fine-tuned Contriever-ft (62.1) [1]; the notebook rerun is the reproducible number. The paper also claims gains across web search, QA and fact
verification, and across languages (e.g. Swahili, Korean, Japanese), with performance
"comparable to fine-tuned retrievers" [1].

**Failure modes and lessons.**

- *Knowledge-bound.* HyDE helps only when the LLM knows roughly what an answer looks like.
  On private or very new corpora, the hypothetical passage can point at the wrong
  neighbourhood with full confidence.
- *Latency.* A full generated passage (and in the reference setup, eight of them) must be
  written before retrieval starts. This is the most expensive step in the pipeline.
- *Strong-encoder regime.* HyDE was designed for the label-free setting. A large study
  across 24 retrievers found that generative expansion helps weak retrievers but tends to
  hurt strong ones (Section 2.3) [45], so gains against modern supervised embedders should
  be validated per corpus rather than assumed.
- *Mitigation.* Keep the original query in the embedding mix (the default in both the
  reference code and LlamaIndex) so that a bad hypothesis cannot fully take over the search
  [2][18].

### 2.2 Query2doc

Query2doc (Wang, Yang and Wei, Microsoft, EMNLP 2023) applies the same idea to sparse and
dense retrieval. It few-shot prompts an LLM to write a pseudo-document, then concatenates it
to the original query, repeating the query several times so that it is not outweighed by the
longer pseudo-document in BM25. The abstract reports BM25 gains of roughly 3% to 15% on
ad-hoc benchmarks such as MS MARCO and TREC DL without fine-tuning, and gains for
state-of-the-art dense retrievers as well [3]. The
lesson: generative expansion also works for keyword retrieval, and the original query must
be explicitly re-weighted against the much longer generated text.

### 2.3 When expansion hurts, and corpus-steered expansion

Three follow-up papers sharpen the guidance on generative expansion:

- **Prompt form matters.** Jagerman et al. (Google, 2023) compared zero-shot, few-shot and
  chain-of-thought (CoT) prompts for LLM query expansion on MS MARCO and BEIR, and found CoT
  prompts the most effective, because they make the model reason step by step and emit
  many related terms. Like Query2doc, they up-weight the original query by repeating it
  (five times) before the expansion text. A CoT/PRF variant also feeds the top-3 retrieved
  documents into the prompt [47].
- **Expansion helps weak retrievers and hurts strong ones.** Weller et al. (Findings of EACL
  2024) tested eleven expansion techniques across twelve datasets and twenty-four retrieval
  models. They report a strong negative correlation between a retriever's baseline quality
  and its gain from expansion: expansion adds recall but also noise, which makes it harder
  to rank the truly relevant documents above near misses. Their recommendation is to use
  expansion for weaker models or when the target corpus differs strongly in format from
  the retriever's training data, and otherwise to avoid it [45]. This is the strongest
  published evidence behind the "strong-encoder regime" caveat in Section 2.1.
- **Ground the expansion in the corpus.** Corpus-Steered Query Expansion (CSQE; Lei et al.,
  EACL 2024) targets HyDE's knowledge-bound failure mode. It retrieves once, asks the LLM to
  identify the pivotal sentences in the initially retrieved documents, and adds those
  corpus sentences to the query alongside the LLM's own generated expansion. The authors
  report strong training-free results, especially on queries for which the LLM lacks
  knowledge [46]. This is a pseudo-relevance-feedback loop with an LLM as the relevance
  judge, and it costs one extra retrieval round before the final search.

## 3. Multi-query retrieval and RAG-Fusion

### 3.1 The basic multi-query retriever

**Problem.** Distance-based similarity search is sensitive to wording. A relevant chunk
phrased differently from the query can fall outside top-k.

**Architecture.** LangChain's `MultiQueryRetriever` is the canonical implementation. Its
default prompt asks an LLM to "generate 3 different versions of the given user question to
retrieve relevant documents from a vector database" so as "to help the user overcome some of
the limitations of distance-based similarity search". A line parser splits the output, the
retriever runs once per generated query, and the results are de-duplicated by union. The
original question is *not* included unless `include_original=True` is set [12]. That default
is risky, because a poor rewrite then replaces the user's words instead of adding to them.
The LangChain "RAG from scratch" teaching notebooks use a five-perspective variant of the same
prompt [16].

### 3.2 RAG-Fusion: multi-query plus Reciprocal Rank Fusion

**Origin.** Adrian Raudaschl proposed RAG-Fusion in a 2023 article and accompanying
repository [6]. It adds two things to plain multi-query: rank-aware fusion and, in its later
form, a reranker. An Infineon team reported an industrial evaluation for product-information
chat in early 2024 [10].

**Reciprocal Rank Fusion.** RRF (Cormack, Clarke and Büttcher, SIGIR 2009) scores each
document as the sum, over all ranked lists in which it appears, of `1 / (k + rank)`, with
k = 60 [11]. Documents that appear high in many lists rise; one outlier list cannot dominate.
LangChain's reference implementation uses exactly `fused_scores[doc] += 1 / (rank + k)` with
`k=60` [16], and the RAG-Fusion repository uses the same constant [6].

**The current recommended architecture** (as of the repository's September-October 2026
write-ups) is more elaborate than the original vector-only demo [6][8]:

```
                    +--> BM25(q)  ------+
  q ----------------+--> vector(q) -----+
  |                                     |
  +--> LLM: 4 diverse rewrites r1..r4   |
         each ri --> BM25(ri), vector(ri)--> 10 ranked lists
                                        |
                                        v
                      RRF: score(d) = sum 1/(60 + rank_i(d))
                                        |
                              pool of 50 candidates
                                        v
                    reranker (cross-encoder or LLM judge) --> top 5-10 --> LLM
```

The repository calls this `hybrid_diverse+rerank`. "Diverse" refers to a rewrite prompt that
explicitly asks for different angles, synonyms, and narrower and broader framings [6].

### 3.3 Does fusion survive reranking? The 2026 evidence

The most useful public evidence on multi-query in production comes from a 2026 exchange.

- **The industry paper.** "Scaling Retrieval Augmented Generation with RAG Fusion: Lessons
  from an Industry Deployment" (Medrano, Verma and Chhabra, Dell Technologies; arXiv
  2603.02153, March 2026) reported that on 115 synthetic enterprise-support queries,
  two-query fusion (the original query plus one LLM rewrite), followed by FlashRank reranking
  and top-K truncation, matched or slightly underperformed a single-query baseline. Its
  proposed mechanism is that the reranker re-anchors on the original query, and the
  rewrite's marginal candidates are squeezed out at truncation. The paper concedes that
  fusion still helps in "recall-scarce" regimes. The abstract's own framing is that fusion
  raised raw recall but the gains "largely" disappeared after reranking and truncation,
  while adding redundancy and latency [9] (configuration details as summarized in [7]).
- **The replication.** Raudaschl replicated the setup on BEIR NFCorpus (3,633 biomedical
  documents, n=200 queries, paired-bootstrap 95% CIs) and disclosed his stake as the
  technique's originator [7]. The paper's exact configuration (vector-only, N=1 rewrite,
  pool of 10) showed no significant lift: +0.005 NDCG@10 [-0.016, +0.027] after a parser
  bug fix. The strong variant, `hybrid_diverse+rerank`, did show a significant lift. Against
  a vector baseline with the same reranker, NDCG@10 rose by +0.027 (bge-reranker-base), +0.015
  (FlashRank MiniLM) and +0.025 (bge-reranker-large), all with CIs excluding zero [7].
- **Stronger rerankers do not absorb the gain.** With LLM-based "decision model" rerankers,
  the lift grew rather than shrank: +0.035 to +0.039 with two vendor decision APIs and +0.050
  with the strongest one tested [6][8]. Vector-only fusion "collapses toward zero once a
  strong reranker is added" [6].
- **Answer quality is real but noisy.** End-to-end LLM-judge comparisons favoured fusion in
  all three judge runs but reached significance in only two. The author revised his earlier
  "measurably better answers" claim to "better retrieval, and probably somewhat better
  answers" [7].

Three methodological lessons from the replication apply beyond RAG-Fusion:

1. **Parse the rewrites defensively.** A rewrite-parser bug caused "for 96 of the 200
   queries, two of the four 'rewrites' [to be] the LLM's preamble line and an empty string".
   In the strict N=1 condition, the preamble was used as the only rewrite for 48% of queries
   [7]. LangChain's `MultiQueryRetriever` splits raw output on newlines [12], so the same
   failure is possible there. Structured output (a JSON array) is the safer contract.
2. **Small samples mislead.** The write-up records four reversals between n=30 and n=200
   runs [7].
3. **Retrieval-only tables overstate production value.** The repository's retrieval-only
   table shows +0.057 NDCG@10 and +0.089 MRR for hybrid-diverse fusion over a vector
   baseline. After reranking, the lift is roughly half that or less [6].

### 3.4 Cost and latency of query expansion

The RAG-Fusion write-up includes the only detailed public per-stage latency breakdown found
for this chapter (30 queries, medians, Apple M2 Max laptop, so absolute timings are
indicative only) [8]:

| Stage | Median latency | API cost per 1,000 queries |
|---|---|---|
| Vector search, top 50 | 80 ms | self-hosted |
| Hybrid-diverse retrieval (10 lists) | 426 ms | self-hosted |
| Rewrite call (4 rewrites, small LLM) | 1,557 ms | about $0.055 |
| Rerank 50 with bge-reranker-base (laptop GPU) | 692 ms | self-hosted |
| Rerank 50 with bge-reranker-large (laptop GPU) | 2,162 ms | self-hosted |
| Query-type router call (LLM classifier) | 243 ms | about $0.017 |

The author's conclusion: "Fusion's real price is latency, not money". The rewrite call costs
about five cents per thousand queries but adds about 1.5 seconds, and "that call sits in
front of everything else" [8]. The same source calls hybrid BM25 plus vector search (no
rewrites) "close to free and worth doing everywhere" [6].

**When multi-query pays.** According to [6], fusion earns its cost when (1) user vocabulary
diverges from corpus vocabulary, (2) recall matters more than precision, and (3) the
downstream consumer can handle topically broad context. Good fits are literature search,
patent prior art, e-discovery, long-tail e-commerce and cold-start specialist corpora. Poor
fits are FAQ bots, latency-critical voice or autocomplete, code and identifier search, and
structured or SQL-backed retrieval.

**Routing fusion selectively** (see Section 7) is the obvious optimization, but it has not
yet been shown to work. An LLM query-type router fused 78% of queries and saved about a fifth
of rewrite calls. It did not target the queries where fusion helps, and its own cost exceeded
the savings when the rewrite model was cheap. The author proposes a router keyed on a
retrieval-weakness signal (for example, a low top reranker score) as the next experiment
[6][8].

### 3.5 Managed multi-query: Azure AI Search generative query rewriting

Azure AI Search ships multi-query as a service feature (preview) [24]:

- The request sets `"queryRewrites": "generative|count-5"` (1 to 10 rewrites). Semantic
  ranking must be on (`queryType: semantic`), and `queryLanguage` must be set; the docs list
  dozens of supported locales.
- The service sends the query "(or a variation of it)" to a generative model. It retrieves
  with the original and the rewritten queries, then applies the semantic reranker. Rewrites
  also apply to vector queries whose text matches the search text.
- The stated purpose is to correct typos and add synonyms. The sample rewrites the hotel
  query "newer hotel near the water with a great restaurant" into five variants such as "new
  waterfront hotels with top-rated eateries".
- **Documented failure mode:** "The rewritten queries might not contain all of the exact
  terms the original query had. This might impact search results if the query was highly
  specific and required exact matches for unique identifiers or product codes" [24]. This is
  the same identifier-search weakness noted in [6].

### 3.6 Adaptive multi-query: choosing the rewrite strategy per query

RAG-Fusion uses one paraphrase prompt for every query. DMQR-RAG (Li et al., Renmin
University and Kuaishou, November 2024) instead defines four rewriting strategies that
operate at different levels of information (for example, keeping the query's information
fixed, expanding it, or reducing it to its core keywords), and adds an adaptive selection
step, driven by lightweight prompting with few-shot examples, that picks which strategies
to apply to each query so as to minimize the number of rewrites. The authors report that
multi-query rewriting generally beat single-query rewriting on academic and industrial
datasets, and that DMQR-RAG outperformed vanilla RAG-Fusion; their ablation attributes much
of the gain to query-specific strategy selection [48]. The design converges with the 2026
RAG-Fusion "diverse" prompt (Section 3.2): diversity of rewrite *type*, not merely the
number of paraphrases, is what recovers missed documents.

## 4. Step-back prompting

**Problem.** Some questions are hard to answer directly because they are full of specifics,
such as a particular temperature and pressure in a physics problem, or a person's employer
in a particular year. Retrieving for the literal question can return nothing useful, while
retrieving for the governing principle or the broader history works.

**Technique.** Step-back prompting (Zheng et al., Google DeepMind, 2023; ICLR 2024) is a
two-step method. The model first generates a *step-back question* about a higher-level
concept or principle, for example "what is the education history of this person?" rather
than "which school did Estella Leopold attend between August and November 1954?". It then reasons over the answer to
that question to solve the original one [5]. The abstract reports that with PaLM-2L, step-back
prompting improved MMLU Physics and Chemistry by about 7% and 11%, TimeQA by about 27%, and
MuSiQue by 7% (abstract figures). The authors also report gains with GPT-4 and
Llama2-70B [5].

**RAG architecture.** In retrieval pipelines, the step-back question is used as an extra
retrieval query, not as a replacement. LangChain's reference chain retrieves twice in
parallel, once with the original question (`normal_context`) and once with the generated
step-back question (`step_back_context`). Both context sets go into a response prompt that
tells the model to use them if relevant and ignore them otherwise [16].

```
  q ---------------------------------------> retrieve --> normal_context ---+
  |                                                                         |
  +--> LLM (few-shot: "paraphrase to a more generic step-back question")    |
         --> q_sb --> retrieve --> step_back_context ------------------------+
                                                                            v
                                     answer prompt(q, normal_ctx, step_back_ctx)
```

**Fit and failure modes.** Step-back suits textbook-style corpora, policies and technical
documentation, where principles are written separately from the cases they govern. It costs
one extra short LLM call and one extra retrieval. The main failure mode is
over-abstraction: the step-back question becomes generic enough to retrieve broad
boilerplate that crowds out the specific evidence. Keeping the original-query retrieval in
the pipeline, as the reference implementation does, guards against this [16].

## 5. Query decomposition

Decomposition targets questions that contain more than one information need. They are either
*parallel* ("Where is the waterfront building and how did the whistleblower scandal affect
the company?") or *sequential*, where the answer to one hop is needed to form the next
query.

### 5.1 Research foundations

- **Least-to-most prompting** (Zhou et al., Google, 2022) breaks a problem into simpler
  subproblems and solves them in order, feeding each answer into the next step [36].
- **Self-Ask** (Press et al., Findings of EMNLP 2023) names the *compositionality gap*:
  models can often answer every sub-question correctly yet fail on the composed question.
  The model is prompted to ask itself explicit follow-up questions, which can be answered by
  a search engine, before giving the final answer. The work also introduced the Bamboogle
  dataset [34].
- **IRCoT** (Trivedi et al., ACL 2023) interleaves retrieval with chain-of-thought. Each
  reasoning sentence becomes the query for the next retrieval step, and the retrieved
  paragraphs condition the next reasoning step. It was evaluated on HotpotQA,
  2WikiMultihopQA, MuSiQue and IIRC. The paper reports gains of up to about 21 retrieval
  points and up to 15 QA points with GPT-3, with gains also holding out of distribution
and with much smaller models such as Flan-T5 [35].

### 5.2 Framework patterns: answer individually vs. answer recursively

LangChain's teaching material separates two execution strategies [16]:

```
  PARALLEL ("answer individually")         SEQUENTIAL ("answer recursively")
  q -> LLM -> [s1, s2, s3]                 q -> LLM -> [s1, s2, s3]
        |      |     |                      s1 -> retrieve -> a1
       ret    ret   ret                     s2 + (s1,a1) -> retrieve -> a2
        |      |     |                      s3 + (s1,a1),(s2,a2) -> retrieve -> a3
       a1     a2    a3                      answer(q) = a3 (or synthesize)
        \      |     /
   synthesize(q, Q/A pairs)
```

The decomposition prompt asks for sub-questions "that can be answered in isolation". The
recursive variant passes the accumulated "background question + answer pairs" into each
later step. That is the least-to-most / IRCoT pattern, which the notebook cites directly
[16].

LlamaIndex's **SubQuestionQueryEngine** adds *tool assignment* to decomposition. The question
generator receives the user question plus a JSON map of tools (one query engine per data
source) and emits sub-questions, each tagged with a `tool_name`. Its built-in example turns
"Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021" into four
sub-questions, two routed to `uber_10k` and two to `lyft_10k` [19][20]. The sub-questions can
run concurrently with `asyncio.gather` (`use_async`), and the answers are synthesized
afterwards [19]. LlamaIndex also has a `StepDecomposeQueryTransform` for multi-step
sequential querying [18].

### 5.3 Managed decomposition: Amazon Bedrock Knowledge Bases

Amazon Bedrock Knowledge Bases offers decomposition as a one-line configuration
(`orchestrationConfiguration.queryTransformationConfiguration.type = "QUERY_DECOMPOSITION"`)
on the `RetrieveAndGenerate` API. Sub-queries each go "through their own retrieval steps",
and "the resulting chunks will then be pooled and ranked together before passing them to
the FM" [27]. AWS's sample notebook shows both the failure and the fix. Without
decomposition, the compound query about Octank's waterfront building and its whistleblower
scandal returned only scandal-related chunks, because the embedding was pulled toward one
half of the question. With decomposition, the invocation log shows a `<standalone_question>`
followed by three `<query>` sub-queries, and the retrieved chunks covered both topics [27].
The `<standalone_question>` element suggests that conversational contextualization and
decomposition run in the same model call.

### 5.4 Managed decomposition: Azure AI Search agentic retrieval

Azure AI Search's *agentic retrieval* (Foundry IQ) is the most fully specified public query
planner [25][26]:

```
  retrieve(query, conversation history, reasoningEffort)
       |
       v
  [query planning LLM]  (skipped at "minimal")
       |  focused subqueries + knowledge-source selection
       v
  subqueries run in parallel over knowledge sources
  (keyword | vector | hybrid; indexed indexes or remote sources)
       |  each subquery semantically reranked (L2)
       v
  [medium only] semantic classifier (L3) judges sufficiency
       |  insufficient -> ONE follow-up iteration with revised plan
       v
  merged grounding content (+ optional references, activity log, answer synthesis)
```

- **Effort levels act as a router over the planner itself** [26]. `minimal` disables
  LLM planning and searches every knowledge source directly. `low` (the default) runs one
  pass of planning and source selection. `medium` adds a high-precision semantic classifier
  and at most one retry ("Each iteration adds latency and cost, so the system constrains
  retry to one pass"). `auto` starts with a lightweight pass and escalates to planning, up to
  medium effort, only if grounding is insufficient. Microsoft introduced `auto` in an August
  2026 preview and, in its own evaluation, reports that it retained 97.6% of
  `medium`'s evidence recall while cutting average query-planning token usage by 46.9%
  (vendor-reported figures) [50].
- **Limits** (preview): at `low`, 5,000 answer tokens and up to 50 documents for semantic
  ranking (10 with L3 classification). At `medium`, 10,000 answer tokens and 50 documents (20
  with L3) [26].
- **Cost model.** Billing is per token rather than per query. Microsoft's worked example
  assumes gpt-4o-mini for planning, 2,000 input tokens of chat context, 350 output tokens per
  plan, three subqueries per plan and 50 reranked chunks of 500 tokens per subquery. On those
  assumptions, 2,000 agentic retrievals cost about $3.30 for reranking and $1.02 for query
  planning, $4.32 in total. Fan-out (number of sources and subqueries) dominates; planning is
  under a quarter of the cost [25].
- The docs state the trade-off directly: agentic retrieval "adds latency compared to a
  single-query pipeline, but it handles query complexity that a single query can't". The
  examples given are multi-ask questions, dependence on earlier turns, paraphrase and
  spelling mistakes [25].

### 5.5 Decomposition in graph and agentic systems

Decomposition also appears inside larger architectures:

- **LazyGraphRAG** (Microsoft Research, November 2024) defers all LLM work to query time.
  Its query-refinement step uses an LLM to "identify relevant subqueries and recombine them
  into a single expanded query", and to refine subqueries with matching concepts from a
  concept graph. Each of q = 3-5 subqueries then drives chunk ranking and iterative deepening
  over communities. Indexing cost is reported as identical to vector RAG and 0.1% of full
  GraphRAG [43]. The graph-RAG chapter of this atlas covers the full design.
- **Anthropic's multi-agent Research system** (June 2025) applies decomposition at the agent
  level. A lead agent plans, "determining query complexity and subagent count", and spawns
  subagents with separate search tasks. The prompts include explicit effort-scaling rules:
  "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might
  need 2-4 subagents with 10-15 calls each, and complex research might use more than 10
  subagents". The system (Claude Opus 4 as lead agent, Claude Sonnet 4 subagents)
  outperformed single-agent Claude Opus 4 by 90.2% on an internal research eval. Multi-agent systems used about 15x the tokens of chat interactions [44].
  The post also documents a query-formulation failure: agents "default to overly long,
  specific queries that return few results". The fix was to prompt them to "start with
  short, broad queries ... then progressively narrow focus" [44].

**Decomposition failure modes.** (1) Over-decomposition of simple queries, which multiplies
cost and latency for no gain. The Anthropic effort rules and Azure's `minimal`/`auto` levels
both exist to prevent this [26][44]. (2) Sub-questions that lose the shared constraint, such
as "for year 2021", so that each one retrieves off-period evidence. (3) Sequential chains in
which an early wrong answer contaminates every later query, the error compounding noted in
[44]. (4) Unbounded iteration, which Azure caps at one retry [26].

## 6. Learned query rewriters

Prompted rewriting is easy to build but cannot be tuned against retrieval outcomes. A
separate research line trains the rewriter using feedback from retrieval or from the reader.

- **Rewrite-Retrieve-Read** (Ma et al., EMNLP 2023) replaces retrieve-then-read with
  rewrite-retrieve-read. An LLM first generates the search query, and a web search engine
  (Bing) retrieves the context. To "better align the query to the frozen modules", a small
  trainable rewriter (T5-based in the paper) is trained by reinforcement learning, using the
  black-box LLM reader's feedback as reward. Results were reported on open-domain and
  multiple-choice QA [4].
- **RQ-RAG** (Chan et al., COLM 2024) trains a Llama-2-7B model end to end to rewrite,
  decompose and disambiguate queries. The training data is built by search-augmenting
  existing data (with DuckDuckGo in the released curation scripts), and both the dataset and
  the checkpoint are public [37].
- **DeepRetrieval** (Jiang et al., 2025) trains a 3B LLM with RL to write queries, using
  retrieval metrics directly as reward, with no reference rewrites. The model reasons in a
  `<think>` block and emits the query in an `<answer>` block. The reward is tiered by recall
  (for example, +5.0 for recall of 0.7 or more, down to -3.5 for recall under 0.05 on PubMed
  search). The authors report 65.07% recall on publication search (previous best 24.68%) and
  63.18% on clinical-trial search (previous best 32.11%), and say the 3B model outperforms
  GPT-4o and Claude-3.5-Sonnet as a query writer. The method also covers BM25/dense retrieval
  and SQL search [41].
- **Search-R1** (Jin et al., 2025) extends the idea from a single rewrite to *interleaved*
  reasoning and search. It uses RL (PPO, GRPO) on veRL to train models to decide when and
  what to search during multi-turn reasoning, with local sparse or dense retrievers or
  online search engines as the environment. By late 2025 it had been integrated into veRL,
  SkyRL and Thinking Machines' Tinker cookbook [42].

**Lesson.** The research frontier has moved the query writer from a fixed prompt to a
policy optimized for retrieval reward. For system designers, the practical takeaway is that
a small fine-tuned rewriter (3B-7B) can beat a large prompted model at writing queries. It
is also cheaper and faster on the critical path [37][41]. The cost is a training pipeline
and a reward signal. The DeepRetrieval reward table, tiered by recall, shows how a reward can
be built from retrieval metrics alone [41].

## 7. Conversational query rewriting

**Problem.** In multi-turn chat, the latest message is often not retrievable on its own
("and how does that compare to last year?"). Embedding it directly retrieves noise.

### 7.1 Condense-question and history-aware retrievers

Both major frameworks use the same pattern: an LLM rewrites (chat history + follow-up) into a
standalone question, which is then sent to the retriever.

- **LlamaIndex `CondenseQuestionChatEngine`** prompt: "Given a conversation (between Human
  and Assistant) and a follow up message from Human, rewrite the message to be a standalone
  question that captures all relevant context from the conversation" [22].
- **LangChain `create_history_aware_retriever`** wraps the rewrite in a `RunnableBranch`. *If
  there is no chat history, the input is passed straight to the retriever, so the LLM call
  is skipped.* Otherwise the prompt, LLM and string parser produce a search query [13]. This
  skip is a simple latency optimization: the first turn of every conversation costs no
  rewrite.
- **Managed services build it into the planner.** Azure's agentic retrieval sends
  "your query and conversation history" to the planning LLM [25]. The Bedrock decomposition
  logs show a `<standalone_question>` produced before the sub-queries [27].

```
  history H = [(u1,a1), ..., (u_{t-1}, a_{t-1})],  follow-up u_t
        |
        +-- H empty? --yes--> retrieve(u_t)
        |
        no
        v
  LLM: rewrite(H, u_t) -> standalone q_t  -->  retrieve(q_t) --> answer(H, u_t, docs)
                                              (answer step still sees the raw turn)
```

A key design decision is what the *answer* step sees. The rewritten query is for retrieval
only. Generating from the rewritten query alone can lose tone, constraints or meta-requests
that appear in the raw turn ("answer in a table"), so the answer prompt should still receive
the original message and the history.

### 7.2 Research line: conversational search

The TREC CAsT and iKAT conversational-search benchmarks led to a series of rewriting methods:

- **LLM4CS** (Mao et al., Findings of EMNLP 2023) prompts an LLM to generate several rewrites
  and hypothetical responses, then aggregates them into one search-intent representation. It
  defines three prompting modes, Rewrite (REW), Rewrite-Then-Response (RTR) and
  Rewrite-And-Response (RAR), plus chain-of-thought tailored to intent understanding. It was
  evaluated on CAsT-19/20/21 with exemplars from CAsT-22 [38]. Generating a hypothetical
  response alongside the rewrite is HyDE applied to conversation.
- **ConvGQR** (Mo et al., ACL 2023) combines generative query rewriting with generated
  potential answers for query expansion [39].
- **CHIQ** (Mo et al., EMNLP 2024) first improves the conversational *history* with an LLM
  (resolving ambiguity, for example) and only then rewrites the query. It uses open-source
  LLMs where earlier work relied mostly on closed models, and reports state-of-the-art
  results in most settings across five benchmarks, including TREC iKAT [40].
- **ConvSearch-R1** (Zhu et al., Fudan / ByteDance, EMNLP 2025) brings the learned-rewriter
  idea of Section 6 to conversation. It needs no human or LLM rewrite labels: a warm-up
  stage uses retrieval-guided self-distillation to solve the cold start, and an RL stage
  optimizes the rewriter with a rank-based reward from the downstream retriever, which
  densifies the otherwise sparse retrieval-metric signal. The authors report more than 10%
  improvement on TopiOCQA with a 3B model, and gains on QReCC [49]. It addresses the main
  weakness of prompted condense-question rewriters: they are not aligned with the retriever
  that actually consumes the rewrite.

**Failure modes.** (1) *Over-carrying context*: the rewrite drags in entities from earlier
turns after the user has changed topic. (2) *Under-carrying*: pronouns are left unresolved.
(3) Latency on every turn after the first. (4) The rewrite becomes the hidden single point
of failure. Logging the rewritten query next to the retrieved documents is the minimum
needed for debugging, which is why both the Azure activity log [25] and the Bedrock
CloudWatch invocation logs [27] expose it.

## 8. Routing

Routing decides *where* a query goes before or instead of transforming it. Public designs
route on five different axes:

| Routing axis | Decision | Mechanism | Example |
|---|---|---|---|
| Data source / index | Which index or tool | LLM structured output / function calling | LangChain `RouteQuery` [17], LlamaIndex `RouterQueryEngine` [21] |
| Intent / guardrail | Which pipeline or prompt, or deflect | Embedding similarity to example utterances | semantic-router [29] |
| Complexity / strategy | No retrieval vs single-step vs multi-step | Trained small classifier | Adaptive-RAG [33] |
| Effort | How much planning and iteration | Service setting plus sufficiency check | Azure `minimal`/`low`/`medium`/`auto` [26] |
| Model | Cheap vs strong LLM | Trained router on preference data | RouteLLM [32] |

### 8.1 LLM-based logical routing

LangChain's reference router defines a Pydantic schema with a `Literal["python_docs",
"js_docs", "golang_docs"]` field, binds it with `with_structured_output` (function calling),
and branches on `result.datasource` [17]. LlamaIndex's `RouterQueryEngine` uses a *selector*
to choose one or several query engines from their text descriptions. The default prompt
lists the choices as a numbered set of summaries and asks for the most relevant one "using
only the choices above and not prior knowledge". With `select_multi=True`, it may choose
several ("no more than {max_outputs}, but only select what is needed"), and their responses
are merged by a `TreeSummarize` summarizer [21]. These routers are only as good as the
**descriptions** of the destinations. A vague tool description is the most common cause of
misrouting.

### 8.2 Semantic (embedding) routing

**Problem.** An LLM call to classify every query adds hundreds of milliseconds. The
RAG-Fusion benchmark measured 243 ms for a hosted classifier call [8].

**Architecture.** Aurelio's open-source `semantic-router` replaces the LLM classifier with
nearest-neighbour matching in embedding space [29]. Each `Route` is defined by a list of
example `utterances`. At query time, the query is embedded once and compared to the
utterance vectors (stored locally or in Pinecone/Qdrant). The best route above its
`score_threshold` wins. If none passes, the router returns `None`, which is an explicit
out-of-domain signal [29].

```
  offline: route r -> utterances u_r1..u_rn -> encoder -> utterance index
  online:  q -> encoder -> sim(q, u_ri) aggregated per route
           best route score >= threshold_r ?  -> route name (static)
                                              -> route + LLM-extracted args (dynamic)
           else -> None (fallback pipeline / general LLM)
```

**Key techniques.**

- *Threshold fitting.* Default thresholds of 0.5 gave 34.85% accuracy on a larger test set
  in the library's optimization notebook. Fitting per-route thresholds from labelled
  (utterance, route) pairs moved them to about 0.22-0.26 and raised accuracy to 90.91%. The
  notebook warns that optimal values vary greatly by encoder; OpenAI's Ada-002 tends to
  produce scores in the 0.5-0.8 range [30].
- *Dynamic routes.* A route can carry function schemas. After the embedding match, an LLM
  call extracts the arguments, so the route returns both a decision and parameters. This
  combines routing and slot filling, and the LLM is called only when the route needs it [31].
- *Hybrid routes* (sparse plus dense encoders) and local encoders and LLMs are supported. The
  library ships a "fully local" healthcare administrative-routing example with threshold
  optimization across seven workflows [29].
- As of 2026, the 0.x line (ending at 0.2.0) is in maintenance and a breaking 1.x rewrite of
  the routing layer is in development, so integrators should pin a version [29].

**Failure modes.** Embedding routers are fast and cheap, but they match surface similarity.
Negation ("not about politics"), rare intents with few utterances, and drift in user
language all degrade them. They need a labelled evaluation set and periodic re-fitting, and
the `None` route must be wired to a sensible fallback.

### 8.3 Complexity routing: Adaptive-RAG

Adaptive-RAG (Jeong et al., NAACL 2024) routes each query to *no retrieval*, *single-step
retrieval* or *iterative multi-step retrieval* (IRCoT-style), based on predicted complexity
[33]. The router is "a smaller LM trained to predict the complexity level of incoming
queries", with labels collected automatically from which strategy actually answered
correctly and from dataset biases (single-hop vs multi-hop sources). It was evaluated on
single-hop (NQ, TriviaQA, SQuAD) and multi-hop (MuSiQue, HotpotQA, 2WikiMultiHopQA) QA with
FLAN-T5 and GPT readers [33]. The design has been reproduced as LangGraph and LlamaIndex
cookbooks [33]. The lesson for production: decomposition and iteration should be gated, not
always on. Azure's `auto` effort level [26][50] and Anthropic's effort-scaling rules [44] are
industrial versions of the same idea.

### 8.4 Model routing: RouteLLM

Model routing chooses which LLM answers, not where to retrieve. It sits in the same
query-understanding layer and uses the same signals. LMSYS's RouteLLM provides routers
trained on preference data behind an OpenAI-compatible client. The authors report cost
reductions of up to 85% while keeping 95% of GPT-4 performance on MT Bench, and say the
routers matched commercial routing offerings while being more than 40% cheaper [32].

## 9. Metadata filter extraction and self-query

**Problem.** "Songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes in the
dance pop genre" contains one semantic need (teenage romance) and three hard constraints
(artist, length, genre). Embedding the whole sentence treats the constraints as soft
similarity. That loses precision, and on filtered ANN indexes it wastes the filter support the
store already has (Chapter 04).

### 9.1 LangChain SelfQueryRetriever

The self-query design [14][15] splits the query in two:

```
  q --> LLM(query-constructor prompt: schema + attribute descriptions
           + allowed comparators/operators + few-shot examples)
     --> {"query": "teenager love",
          "filter": "and(or(eq(artist,'Taylor Swift'), eq(artist,'Katy Perry')),
                         lt(length,180), eq(genre,'pop'))",
          "limit": optional int}
     --> parse into StructuredQuery IR (Comparison / Operation / limit)
     --> store-specific Translator (Pinecone, Qdrant, Chroma, Weaviate, Milvus,
         Elasticsearch, OpenSearch, PGVector, MongoDB Atlas, Neo4j, Redis, ...)
     --> vectorstore.search(query, filter=native_filter, k=limit)
```

Details of the prompt contract that matter in practice [14]:

- "The query string should contain only text that is expected to match the contents of
  documents. Any conditions in the filter should not be mentioned in the query as well."
  This stops the constraint words from also skewing the embedding.
- "Make sure that filters only refer to attributes that exist in the data source", only
  allowed comparators and operators are used, dates use `YYYY-MM-DD`, and if no filter
  applies the model must return `NO_FILTER`.
- An optional `limit` field turns "three songs about love" into k = 3.
- The retriever has `use_original_query`, which keeps the user's wording for the vector
  search and uses only the extracted filter [15]. This is a useful hedge when the LLM's
  shortened query loses meaning.
- An intermediate representation plus per-store `Translator` visitors makes the LLM output
  portable. More than twenty vector stores have built-in translators [15].

### 9.2 LlamaIndex VectorIndexAutoRetriever

LlamaIndex's auto-retriever adapts the LangChain prompt (the source credits it) to its own
`VectorStoreInfo` / `MetadataInfo` schema and outputs a `VectorStoreQuerySpec` with query,
filters and optional `top_k` [23]. It adds operational safeguards:

- `max_top_k` (default 10) caps any LLM-requested top_k.
- If the inferred query string is empty, meaning the request is filter-only, as in "all
  reports from 2023", it uses `empty_query_top_k` and an optional
  `default_empty_query_vector`.
- `extra_filters` are applied unconditionally on top of whatever the LLM produced. They
  cannot be OR conditions. This is the right place for security and tenancy filters, which
  must never depend on the LLM [23].

### 9.3 Structured search schemas with function calling

LangChain's query-construction notebook shows the modern form: a Pydantic `TutorialSearch`
model with separate `content_search` and `title_search` strings, plus optional typed
range filters (`min_view_count`, `earliest_publish_date`, `max_length_sec` and so on), each
described as "Only use if explicitly specified". The model is bound through structured
output [17]. Generating *two* query strings for two fields (a full-text content query and a
short, keyword-only title query) is a small but effective pattern for multi-field indexes.

### 9.4 Managed: Amazon Bedrock implicit (autogenerated) filters

Bedrock Knowledge Bases exposes self-query as `implicitFilterConfiguration` inside
`vectorSearchConfiguration`. The caller lists `metadataAttributes` (key, type such as
`NUMBER`/`STRING`, and a description with possible values, for example "Possible values
include ['AMZN']"), and names the model that generates filters (Claude 3.5 Sonnet in the
sample) [28]. The documented example: "marketing reports from last year" yields a date
filter. AWS recommends CloudWatch log analysis to "verify their consistency across different
user query variations" [28], which is an implicit admission that generated filters can vary
from one phrasing to the next.

### 9.5 Failure modes of filter extraction

1. **Over-filtering.** The single most damaging error. A wrongly inferred filter (wrong year,
   misspelled entity value) returns zero or wrong results, and no reranker can recover
   documents that were filtered out. Mitigations: tell the model to filter only when the
   constraint is explicit (the "Only use if explicitly specified" pattern [17]), enumerate
   allowed values in attribute descriptions [28], and fall back to an unfiltered or relaxed
   search when a filtered search returns too few results.
2. **Value normalization.** "Q3", "last year" and "2024" need a reference date and a
   canonical format. The LangChain prompt forces `YYYY-MM-DD` [14], but relative dates still
   need the current date injected into the prompt.
3. **Constraint leakage.** Constraint words left in the semantic query double-count. The
   self-query prompt explicitly forbids this [14].
4. **Security.** Access-control filters must be applied outside the LLM, as with
   `extra_filters` [23]. An LLM-generated filter is user-influenced input.

## 10. Side-by-side comparison

### 10.1 Technique comparison

| Technique | Extra LLM calls before retrieval | Retrievals per query | Best for | Main failure mode | Primary evidence |
|---|---|---|---|---|---|
| HyDE | 1 (long output; N samples in reference setup) | 1 (averaged vector) | Zero-shot, label-free corpora; question/passage asymmetry | Confident wrong hypothesis on private or new data | DL19 nDCG@10 0.445 -> 0.597 [2] |
| Query2doc | 1 (long output) | 1 | BM25 / sparse stacks | Pseudo-doc outweighs query | +3-15% BM25 (abstract) [3] |
| Corpus-steered expansion (CSQE) | 1-2 (relevance judging + expansion) | 2 | Queries outside the LLM's knowledge | Weak first-pass retrieval feeds weak sentences | Training-free gains (abstract) [46] |
| Multi-query (union) | 1 (short output) | N | Vocabulary mismatch | Parser junk; drift without original | Framework default [12] |
| Adaptive multi-query (DMQR-RAG) | 1-2 (strategy selection + rewrites) | Varies per query | Mixed traffic with noisy queries | Selector errors | Beats vanilla RAG-Fusion (authors) [48] |
| RAG-Fusion (hybrid + RRF + rerank) | 1 | 2(N+1) | Recall-scarce, jargon-heavy domains | ~1.5 s added latency; little gain with vector-only variant | +0.015 to +0.050 NDCG@10 after rerank [6][7][8] |
| Step-back | 1 (short) | 2 | Principle or background questions | Over-abstraction | MMLU, TimeQA, MuSiQue gains (abstract) [5] |
| Decomposition (parallel) | 1 | k sub-queries | Multi-ask questions | Lost shared constraints; over-decomposition | Bedrock sample [27]; Azure planner [25] |
| Decomposition (sequential / IRCoT) | k (interleaved) | k | Multi-hop questions | Error compounding; latency | Up to ~21 retrieval points (abstract) [35] |
| Conversational condense | 1 (skipped on turn 1) | 1 | Multi-turn chat | Over- or under-carrying context | Framework defaults [13][22] |
| LLM router | 1 (tiny output) | 0 extra | Several indexes or tools | Vague destination descriptions | [17][21] |
| Semantic router | 0 (embedding only) | 0 extra | Intents, guardrails, high QPS | Surface similarity; threshold drift | 34.85% -> 90.91% after fitting [30] |
| Complexity router | 0 (small classifier) | 0, 1 or many | Mixed simple and multi-hop traffic | Mislabelled complexity | Adaptive-RAG [33] |
| Self-query / implicit filters | 1 (structured) | 1 (filtered) | Catalogues, reports, dated corpora | Over-filtering to zero results | [14][15][23][28] |
| Learned rewriter (RL) | 1 (small model) | 1+ | High-volume, domain-specific search | Training cost; reward hacking | DeepRetrieval recall 65.07% vs 24.68% [41]; ConvSearch-R1 [49] |

### 10.2 Platform and framework comparison

| Capability | LangChain [12]-[17] | LlamaIndex [18]-[23] | Azure AI Search [24]-[26] | Amazon Bedrock KB [27][28] |
|---|---|---|---|---|
| Multi-query / rewrite | `MultiQueryRetriever` (3 by default, original optional) | Query transforms | `queryRewrites: generative\|count-N` (1-10) | Within decomposition |
| HyDE | Cookbook | `HyDEQueryTransform` (keeps original) | No (not documented) | No (not documented) |
| Decomposition | Notebook patterns | `SubQuestionQueryEngine` (tool-tagged, async) | Agentic retrieval query planning | `QUERY_DECOMPOSITION` |
| Conversational rewrite | `create_history_aware_retriever` (skips on empty history) | `CondenseQuestionChatEngine`, condense-plus-context | Planner consumes chat history | Standalone question in decomposition |
| Routing | Structured-output router; embedding router | `RouterQueryEngine` + selectors | Knowledge-source selection; effort levels incl. `auto` | Not documented |
| Filter extraction | `SelfQueryRetriever` + 20+ translators | `VectorIndexAutoRetriever` + `extra_filters` | Not documented as automatic | `implicitFilterConfiguration` |
| Observability | Tracing (LangSmith) | Verbose / callbacks | `debug: queryRewrites`, activity log | CloudWatch invocation logs |

## 11. Cross-cutting failure modes and engineering lessons

1. **The query layer sits on the critical path.** Every LLM call before retrieval adds its
   full latency to time-to-first-token. The measured rewrite cost of about 1.5 s [8] is
   larger than the retrieval it improves (80-426 ms). Parallelize what can run in parallel
   (sub-queries [19][25], step-back plus original retrieval [16]), skip work when it is not
   needed (empty history [13], `minimal` effort [26]), and use embedding routers instead of
   LLM classifiers where accuracy allows [29][30].
2. **Always keep the original query in the mix.** HyDE averages it in [2]. LlamaIndex keeps
   it by default [18]. RAG-Fusion searches it alongside the rewrites [6]. Step-back retrieves
   for it in parallel [16]. Self-query can use it for the vector search [15]. The cases that
   fail are the ones that replace it: LangChain's `include_original=False` default [12], and
   Azure's warning about product codes [24].
3. **Exact identifiers do not survive paraphrase.** SKUs, error codes, ticket IDs and legal
   citations should bypass generative rewriting or be protected by a BM25 leg [6][24].
4. **Use structured output for every transformation.** Free-text parsing of rewrites caused
   a large, silent evaluation error in the most-cited public RAG-Fusion benchmark [7]. Every
   managed service and the newer framework patterns use JSON or tagged output
   [14][17][19][27].
5. **Evaluate after the reranker and on answers, with confidence intervals.** Gains measured
   on retrieval alone shrink after reranking, and answer-level gains are noisier still. The
   2026 paper and its replication disagreed largely because they tested different variants
   at different sample sizes [6][7][9].
6. **Gate complexity.** Decomposition, iteration and multi-query should be routed on, not
   always on: Adaptive-RAG [33], Azure `auto` [26], Anthropic's effort rules [44]. The open
   problem is choosing the gating signal. Query-type classification did not target fusion
   wins in [8]; retrieval-confidence signals, such as reranker score or Azure's L3
   sufficiency classifier [26], look more promising.
7. **Make the transformation visible.** Rewritten queries, generated filters and sub-query
   plans should be logged next to the results. Azure's `debug` and activity log, Bedrock's
   CloudWatch logs and LangSmith traces exist for this purpose [16][24][25][27].
8. **Treat generated filters as untrusted.** Tenancy and ACL constraints belong in
   non-LLM `extra_filters`-style parameters [23].

## Key takeaways for system designers

- **Start with the cheapest wins.** Hybrid BM25 plus vector retrieval with RRF costs almost
  nothing and helps nearly everywhere [6]. A conversational condense step is mandatory for
  multi-turn chat [13][22]. Self-query filter extraction is mandatory for corpora with
  meaningful metadata [14][28].
- **Pick query expansion by diagnosis, not habit.** HyDE addresses question/passage
  asymmetry in zero-shot settings [1][2], and generative expansion in general tends to help
  weak retrievers and hurt strong ones [45]; steer it with the corpus when the LLM may not
  know the domain [46]. Multi-query and RAG-Fusion address vocabulary
  mismatch in recall-scarce domains [6]. Step-back addresses over-specific questions [5].
  Decomposition addresses multi-ask and multi-hop questions [25][27][35].
- **Budget latency explicitly.** One rewrite call is about 1-2 s with a small hosted model
  [8], several times the cost of retrieval itself. Embedding routers run in a single
  embedding call [29]. Managed planners bill per token, and fan-out across subqueries
  dominates the cost [25].
- **If you deploy RAG-Fusion, deploy the hybrid variant behind your strongest reranker.**
  Vector-only fusion with one rewrite shows no significant gain after reranking. Hybrid
  fusion with diverse rewrites shows +0.015 to +0.050 NDCG@10 [6][7][8].
- **Never let a transformation remove information.** Keep the original query, protect
  exact identifiers, apply filters only when explicit, and fall back when filters return
  nothing [2][12][17][24].
- **Route by complexity.** Use no retrieval, single-shot or iterative retrieval as the query
  requires, with capped iterations (Azure allows one retry [26]) and explicit effort budgets
  [33][44][50].
- **Watch the learned-rewriter frontier.** RL-trained 3B-7B query writers have outperformed
  large prompted models on recall-oriented search [37][41], and RL now trains when and what
  to search during reasoning [42]. For high-volume vertical search, a trained rewriter is
  becoming a realistic component.
- **Instrument everything.** The rewritten query is part of your system's behaviour. Log it,
  evaluate it, and version its prompt like code [7][24][27].

## References

1. L. Gao, X. Ma, J. Lin, J. Callan. "Precise Zero-Shot Dense Retrieval without Relevance
   Labels" (HyDE). CMU / University of Waterloo, arXiv 2022; ACL 2023.
   https://arxiv.org/abs/2212.10496
2. L. Gao et al. "HyDE" official code repository (README, `promptor.py`, `hyde.py`,
   `hyde-dl19.ipynb` with saved outputs). GitHub, 2022-2023. https://github.com/texttron/hyde
3. L. Wang, N. Yang, F. Wei. "Query2doc: Query Expansion with Large Language Models."
   Microsoft, EMNLP 2023. https://arxiv.org/abs/2303.07678
4. X. Ma, Y. Gong, P. He, H. Zhao, N. Duan. "Query Rewriting in Retrieval-Augmented Large
   Language Models" (Rewrite-Retrieve-Read). EMNLP 2023.
   https://aclanthology.org/2023.emnlp-main.322 (code: https://github.com/xbmxb/RAG-query-rewriting)
5. H. S. Zheng, S. Mishra, X. Chen, H.-T. Cheng, E. H. Chi, Q. V. Le, D. Zhou. "Take a Step
   Back: Evoking Reasoning via Abstraction in Large Language Models." Google DeepMind, arXiv
   2023; ICLR 2024. https://arxiv.org/abs/2310.06117
6. A. Raudaschl. "RAG-Fusion" repository README (architecture, evaluation, cost guidance).
   GitHub, 2023-2026. https://github.com/Raudaschl/rag-fusion
7. A. Raudaschl. "A small replication of arXiv 2603.02153v1 on NFCorpus" (with 27 September
   2026 correction). GitHub, 2026.
   https://github.com/Raudaschl/rag-fusion/blob/master/experiments/arxiv-2603-02153-replication/README.md
8. A. Raudaschl. "Jev inside the RAG-Fusion pipeline" (rerankers, cost and latency per
   stage, routing experiment). GitHub, September-October 2026.
   https://github.com/Raudaschl/rag-fusion/blob/master/experiments/jev-in-the-pipeline/README.md
9. L. Medrano, A. Verma, M. Chhabra (Dell Technologies). "Scaling Retrieval Augmented
   Generation with RAG Fusion: Lessons from an Industry Deployment." arXiv 2603.02153,
   March 2026. https://arxiv.org/abs/2603.02153
10. Z. Rackauckas. "RAG-Fusion: a New Take on Retrieval-Augmented Generation." Infineon
    Technologies, arXiv 2024; International Journal on Natural Language Computing 13(1),
    2024. https://arxiv.org/abs/2402.03367
11. G. V. Cormack, C. L. A. Clarke, S. Büttcher. "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods." SIGIR 2009.
    https://dl.acm.org/doi/10.1145/1571941.1572114
12. LangChain. `MultiQueryRetriever` source (`multi_query.py`). GitHub, 2023-2026.
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/multi_query.py
13. LangChain. `create_history_aware_retriever` source. GitHub, 2024-2026.
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/chains/history_aware_retriever.py
14. LangChain. Query-constructor prompt (`query_constructor/prompt.py`). GitHub, 2023-2026.
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/chains/query_constructor/prompt.py
15. LangChain. `SelfQueryRetriever` and built-in translators (`self_query/base.py`). GitHub,
    2023-2026.
    https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/self_query/base.py
16. L. Martin / LangChain. "RAG From Scratch: Query Transformations" (Parts 5-9: multi-query,
    RAG-Fusion, decomposition, step-back, HyDE). GitHub notebook, 2024.
    https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
17. L. Martin / LangChain. "RAG From Scratch: Routing and Query Construction" (Parts 10-11).
    GitHub notebook, 2024.
    https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb
18. LlamaIndex. Query transforms (`HyDEQueryTransform`, `DecomposeQueryTransform`,
    `StepDecomposeQueryTransform`). GitHub, 2023-2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/query/query_transform/base.py
19. LlamaIndex. `SubQuestionQueryEngine` source. GitHub, 2023-2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/sub_question_query_engine.py
20. LlamaIndex. Sub-question generator prompts (`question_gen/prompts.py`). GitHub,
    2023-2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/question_gen/prompts.py
21. LlamaIndex. `RouterQueryEngine` and selector prompts. GitHub, 2023-2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/router_query_engine.py
22. LlamaIndex. `CondenseQuestionChatEngine` source. GitHub, 2023-2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/chat_engine/condense_question.py
23. LlamaIndex. `VectorIndexAutoRetriever` and auto-retriever prompts. GitHub, 2023-2026.
    https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/vector_store/retrievers/auto_retriever/auto_retriever.py
24. Microsoft. "Rewrite queries with semantic ranker in Azure AI Search (preview)." Azure AI
    docs, 2024-2026. https://learn.microsoft.com/azure/search/semantic-how-to-query-rewrite
25. Microsoft. "Agentic retrieval in Azure AI Search." Azure AI docs, 2025-2026.
    https://learn.microsoft.com/azure/search/agentic-retrieval-overview
26. Microsoft. "Set the retrieval reasoning effort (preview)." Azure AI docs, 2026.
    https://learn.microsoft.com/azure/search/agentic-retrieval-how-to-set-retrieval-reasoning-effort
27. AWS. "Query Reformulation Supported by Amazon Bedrock Knowledge Bases" (sample notebook).
    aws-samples, GitHub, 2024-2026.
    https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/03-optimizing-accuracy-retrieved-results/query_reformulation.ipynb
28. AWS. "Autogenerated filters" (implicit metadata filtering, sample notebook).
    aws-samples, GitHub, 2024-2026.
    https://github.com/aws-samples/amazon-bedrock-samples/blob/main/rag/knowledge-bases/features-examples/03-optimizing-accuracy-retrieved-results/autogenerated_metadata_filters.ipynb
29. Aurelio AI. "Semantic Router" README. GitHub, 2023-2026.
    https://github.com/aurelio-labs/semantic-router
30. Aurelio AI. "Route Threshold Optimization" notebook. GitHub, 2024.
    https://github.com/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization.ipynb
31. Aurelio AI. "Dynamic Routes" notebook. GitHub, 2024.
    https://github.com/aurelio-labs/semantic-router/blob/main/docs/02-dynamic-routes.ipynb
32. I. Ong et al. / LMSYS. "RouteLLM: Learning to Route LLMs with Preference Data"
    (repository and paper). 2024. https://github.com/lm-sys/RouteLLM ;
    https://arxiv.org/abs/2406.18665
33. S. Jeong et al. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
    Models through Question Complexity." NAACL 2024 (official repository).
    https://github.com/starsuzi/Adaptive-RAG ; https://arxiv.org/abs/2403.14403
34. O. Press, M. Zhang, S. Min, L. Schmidt, N. Smith, M. Lewis. "Measuring and Narrowing the
    Compositionality Gap in Language Models" (Self-Ask). Findings of EMNLP 2023.
    https://aclanthology.org/2023.findings-emnlp.378 (code: https://github.com/ofirpress/self-ask)
35. H. Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for
    Knowledge-Intensive Multi-Step Questions" (IRCoT). ACL 2023.
    https://arxiv.org/abs/2212.10509 (code: https://github.com/StonyBrookNLP/ircot)
36. D. Zhou et al. "Least-to-Most Prompting Enables Complex Reasoning in Large Language
    Models." Google, arXiv 2022; ICLR 2023. https://arxiv.org/abs/2205.10625
37. C.-M. Chan et al. "RQ-RAG: Learning to Refine Queries for Retrieval Augmented
    Generation." COLM 2024. https://arxiv.org/abs/2404.00610 (code:
    https://github.com/chanchimin/RQ-RAG)
38. K. Mao et al. "Large Language Models Know Your Contextual Search Intent: A Prompting
    Framework for Conversational Search" (LLM4CS). Findings of EMNLP 2023.
    https://github.com/kyriemao/LLM4CS
39. F. Mo et al. "ConvGQR: Generative Query Reformulation for Conversational Search." ACL
    2023. https://github.com/fengranMark/ConvGQR
40. F. Mo et al. "CHIQ: Contextual History Enhancement for Improving Query Rewriting in
    Conversational Search." EMNLP 2024. https://aclanthology.org/2024.emnlp-main.135
    (code: https://github.com/fengranMark/CHIQ)
41. P. Jiang et al. "DeepRetrieval: Hacking Real Search Engines and Retrievers with Large
    Language Models via Reinforcement Learning." arXiv 2025.
    https://github.com/pat-jj/DeepRetrieval ; https://arxiv.org/abs/2503.00223
42. B. Jin et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with
    Reinforcement Learning." arXiv 2025. https://github.com/PeterGriffinJin/Search-R1 ;
    https://arxiv.org/abs/2503.09516
43. D. Edge, H. Trinh et al. "LazyGraphRAG: Setting a new standard for quality and cost."
    Microsoft Research blog, November 2024.
    https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
44. Anthropic. "How we built our multi-agent research system." Anthropic Engineering, June
    2025. https://www.anthropic.com/engineering/multi-agent-research-system
45. O. Weller, K. Lo, D. Wadden, D. Lawrie, B. Van Durme, A. Cohan, L. Soldaini. "When do
    Generative Query and Document Expansions Fail? A Comprehensive Study Across Methods,
    Retrievers, and Datasets." Findings of EACL 2024.
    https://aclanthology.org/2024.findings-eacl.134 ; https://arxiv.org/abs/2309.08541
46. Y. Lei, Y. Cao, T. Zhou, T. Shen, A. Yates. "Corpus-Steered Query Expansion with Large
    Language Models." EACL 2024 (short papers). https://aclanthology.org/2024.eacl-short.34
47. R. Jagerman, H. Zhuang, Z. Qin, X. Wang, M. Bendersky. "Query Expansion by Prompting
    Large Language Models." Google, arXiv 2023. https://arxiv.org/abs/2305.03653
48. Z. Li et al. "DMQR-RAG: Diverse Multi-Query Rewriting for Retrieval-Augmented
    Generation." arXiv 2024. https://arxiv.org/abs/2411.13154
49. C. Zhu et al. "ConvSearch-R1: Enhancing Query Reformulation for Conversational Search
    with Reasoning via Reinforcement Learning." EMNLP 2025.
    https://aclanthology.org/2025.emnlp-main.1349 ; https://arxiv.org/abs/2505.15776
50. Microsoft Foundry Blog. "Let the query decide: introducing auto retrieval reasoning
    effort." Microsoft Tech Community, 2026.
    https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/let-the-query-decide-introducing-auto-retrieval-reasoning-effort/4558943

Source note: For this edition, the HyDE notebook outputs [2], the LangChain, LlamaIndex,
aws-samples and semantic-router sources, and the RAG-Fusion write-ups [6][7][8] were
re-fetched, and the abstract-level figures for [3], [5], [9], [10], [35] and [45]-[49] were
checked against their published abstracts or listings. The Azure documentation references
now point at Microsoft Learn rather than the docs source repository. The vendor-reported
`auto` effort figures [50] and the DMQR-RAG and ConvSearch-R1 results are the authors' own
and were not independently reproduced. The RAG-Fusion latency figures [8] were measured on
a laptop by the technique's originator, who discloses his interest, and are indicative only.
