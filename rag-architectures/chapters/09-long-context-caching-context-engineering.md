# 09. Long Context vs RAG: Cache-Augmented Generation, Prompt and KV Caching, Context Rot, and Context Engineering

## Overview

Each jump in context window size, from 4K to 32K, 128K, 200K, 1M, and 2M tokens, brings back
the same question: if the whole corpus fits in the prompt, is retrieval still needed? This
chapter answers it. The short answer from the evidence is that context length and retrieval
are complements, not substitutes. The reasons have changed over time, though. Early on the
problem was that the window was too small. Now there are three problems:

1. **Effective context is much shorter than advertised context.** Benchmarks that go beyond
   literal needle matching (RULER, NoLiMa, Chroma's "context rot" study, LongBench v2) show
   that quality falls well before the nominal limit. It falls fastest when the needle and
   the question share few words, when distractors are present, or when the task needs
   reasoning rather than lookup [4][5][6][7].
2. **Position matters.** "Lost in the middle" showed a U-shaped accuracy curve over where
   the relevant passage sits [1]. Any RAG system that concatenates retrieved chunks
   inherits that bias.
3. **Prefill is expensive, and caching changes the economics.** Provider prompt caching
   (Anthropic, OpenAI, Google) and engine-level KV reuse (vLLM, SGLang, LMCache, Mooncake)
   make a stable, reused prefix cheap. They do nothing for the variable part of a RAG
   prompt, which is the retrieved chunks. A research line (Prompt Cache, RAGCache,
   CacheBlend, TurboRAG, Block-Attention, EPIC, APE, KVLink) tries to make retrieved-chunk
   KV caches reusable at arbitrary positions [24]-[31].

In 2025 a fourth thread joined the other three: **context engineering**. Agent builders
(Anthropic, Manus, LangChain, and practitioners such as Drew Breunig) recast "what goes in
the window" as a resource-management problem. They treat it as an attention budget to
curate, compact, offload to files, and split across sub-agents [37][43][44][45].

The chapter is organized as follows:

```
   Section 1  Evidence: how models actually use long inputs
              (NIAH -> lost-in-the-middle -> RULER/NoLiMa/HELMET -> context rot
               -> Michelangelo/MRCR)
   Section 2  Long context vs RAG: head-to-head studies and hybrid routers
              (Self-Route, OP-RAG, LongRAG, Databricks, "RAG meets LC", LaRA,
               sufficient context)
   Section 3  Cache-augmented generation (CAG): preload instead of retrieve
   Section 4  Prefix / prompt caching: providers and serving engines
   Section 5  Non-prefix KV reuse for RAG: Prompt Cache, RAGCache, CacheBlend,
              TurboRAG, Block-Attention, EPIC, APE, KVLink; KVzip compression
   Section 6  KV-cache infrastructure: LMCache, Mooncake, llm-d, Dynamo
   Section 7  Context engineering for agents (incl. ACE playbooks)
   Section 8  Comparison table, decision guide, takeaways
```

A note on sources: wherever possible this chapter cites the paper, repository, or vendor
documentation directly. Numbers marked "approximately" or "reported" come from abstracts or
announcements and were not reproduced. Check vendor pricing and limits against current
documentation before relying on them, because they change often.

---

## 1. How models actually use long inputs

### 1.1 Needle in a Haystack (NIAH)

**Design.** Greg Kamradt's 2023 test hides one fact, the "needle", at a chosen depth inside
filler text, the "haystack", which by default is Paul Graham essays. It sweeps a grid of
context lengths against needle depths and scores whether the model retrieves the fact [2].
The original runs covered GPT-4 128K and Claude 2.1 [2]. The current version of the tool
adds multi-needle, UUID, and "uuid_chain" multi-hop variants. It also stores a recipe so
that the exact prompt can be rebuilt [2].

**Lesson from the Claude 2.1 episode.** Anthropic's write-up shows how fragile NIAH scores
are. The planted sentence was "The best thing to do in San Francisco is eat a sandwich and
sit in Dolores Park on a sunny day". Claude 2.1 often answered that the document gave no
definitive answer. Prefilling the response with "Here is the most relevant sentence in the
context:" raised accuracy on the 200K-token test from 27% to 98% [3]. A NIAH score
therefore measures prompting and refusal behaviour as well as retrieval capacity.

**Failure modes of NIAH as a benchmark.** The needle is lexically similar to the question
and semantically out of place in the haystack, so the model can find it by surface
matching. Every later benchmark in this section removes one of those shortcuts.

### 1.2 Lost in the middle

**Setup.** Liu et al. (Stanford, UC Berkeley, Samaya AI) studied two tasks [1]:

- Multi-document QA over NaturalQuestions-Open, with 10, 20, or 30 Contriever-retrieved
  passages, exactly one of which contains the answer. For the 20-document setting, the gold
  passage is placed at positions 0, 4, 9, 14, or 19.
- Synthetic key-value retrieval over JSON objects of UUID pairs.

**Finding.** Accuracy is highest when the relevant passage is at the start (primacy) or
the end (recency) of the context. It is lowest when the passage is in the middle, and this
U-shape held for both open and closed models, including models sold as long-context [1].
In the multi-document setting, mid-context accuracy for some models fell *below* their
closed-book accuracy, meaning the retrieved documents actively hurt [1]. Extended-context
variants of a model were not better at using the context they already had [1].

**Implications for RAG designers.**

- The order of chunks is a modelling decision, not a cosmetic one. Common responses are to
  put the highest-scoring chunks at the edges, to keep document order (OP-RAG, Section
  2.2), or to retrieve fewer, better chunks.
- Prompt compression with question-aware reordering, as in LongLLMLingua, explicitly
  targets the effect. It reports up to 21.4% better RAG performance using about a quarter
  of the tokens [46].
- Anthropic's long-context guidance points the same way: put long documents at the top
  and the query at the end, which "can improve response quality by up to 30%" in their
  tests, especially with multi-document inputs [42]. It also recommends asking the model to
  extract relevant quotes before answering [42].

### 1.3 RULER: effective vs claimed context length

RULER (NVIDIA) is a configurable synthetic suite. It has 13 tasks in four categories:
retrieval (NIAH variants with multiple keys and values), multi-hop tracing (variable
tracking), aggregation (common- and frequent-word extraction), and QA [4]. It defines
*effective length* as the longest length at which a model still beats Llama-2-7B's 4K
score of 85.6% [4]. The repository's leaderboard shows the gap between claimed and
effective length. Llama 3.1 70B claims 128K and is effective to about 64K. Mixtral-8x22B
claims 64K and is effective to 32K. Mistral 7B claims 32K and is effective to 16K. A few
models, such as Gemini 1.5 Pro and Jamba-1.5-large, stay above threshold past 128K [4].
The maintainers note that almost all models fall below threshold before their claimed
length, and only about half handle 32K effectively [4].

### 1.4 NoLiMa: removing the literal match

NoLiMa (Adobe Research and LMU Munich, ICML 2025) builds needle/question pairs with minimal
lexical overlap. The model has to make a latent association, for example knowing that a
character who lives next to a landmark lives in a particular city [5]. In the paper's
original evaluation of 12 models claiming at least 128K context, 10 fell below 50% of their
short-context baseline at 32K. GPT-4o fell from 99.3% to 69.7% [5]. The repository has
since expanded the evaluation to 22 models, and in the expanded table most of them fall
below 50% of base at 32K. Its effective-length table, using an 85% of base score
threshold, is striking (selected rows) [5]:

| Model | Claimed | Effective (NoLiMa) |
|---|---|---|
| GPT-4.1 | 1M | 16K |
| GPT-4o | 128K | 8K |
| Claude 3.5 Sonnet | 200K | 4K |
| Gemini 2.0 Flash | 1M | 4K |
| Gemini 1.5 Pro | 2M | 2K |
| Llama 3.3 70B | 128K | 2K |
| Llama 4 Maverick | 1M | 2K |
| Llama 4 Scout | 10M | 1K |

The authors attribute the decline to attention struggling to locate relevant content when
no literal match is available [5]. Embedding-based RAG has the same weakness: if the query
and the evidence share no surface form, both the retriever and the reader suffer.

### 1.5 HELMET and LongBench v2: realistic tasks

HELMET (Princeton and Intel, ICLR 2025) evaluates seven categories: synthetic recall, RAG,
re-ranking, citation generation, long-document QA, summarization, and many-shot ICL [8].
It was built because synthetic NIAH-style tasks were seen as poor predictors of downstream
long-context ability [8]. LongBench v2 (Tsinghua/Zhipu, ACL 2025) has 503 multiple-choice questions
over contexts of 8K to 2M words, most under 128K. Human experts with a 15-minute limit
score 53.7%. The best model answering directly scores 50.1%, and o1-preview with extended
reasoning scores 57.7% [7]. The takeaway is that long-context *reasoning*, as opposed to
lookup, remains hard, and inference-time reasoning helps.

On the vendor side, the Gemini 1.5 technical report claimed near-perfect needle recall
(above 99%) up to millions of tokens in text, video, and audio [9]. That claim and the
NoLiMa and RULER results are consistent: near-perfect *literal* recall does not imply
robust *semantic* use of the context.

### 1.6 Context rot (Chroma, 2025)

Chroma's technical report "Context Rot: How Increasing Input Tokens Impacts LLM
Performance" (Hong, Troynikov, Huber, 14 July 2025) [6] tested 18 frontier and open models,
including GPT-4.1, Claude 4, Gemini 2.5, and Qwen3 variants.
The design holds task difficulty constant while varying only input length. Experiments
(code in [6]):

- **NIAH extension.** Needle-question pairs of varying semantic similarity, optional
  *distractors* (plausible but wrong statements such as advice attributed to a professor
  rather than a classmate), and haystacks of Paul Graham essays or arXiv papers, either
  coherent or *shuffled* at the sentence level. An LLM judge (GPT-4.1 by default) scores
  answers.
- **LongMemEval.** A *focused* prompt containing only the relevant conversation turns is
  compared with the *full* prompt of about 113K tokens, which contains irrelevant history
  [6][47].
- **Repeated words.** The model must reproduce a long sequence of a repeated word with one
  unique word inserted. The task is trivial at short lengths.

Reported findings [6]:

- Performance degrades non-uniformly as input grows, even on these simple tasks.
- Degradation is steeper when needle-question similarity is lower.
- A single distractor hurts, several hurt more, and the distractors do not all hurt
  equally.
- Counterintuitively, models did *better* on shuffled haystacks than on coherent ones.
  Logically flowing text appears to pull attention away from the needle.
- On LongMemEval, all models did much better on focused prompts than on full ones.
- Model families fail differently. The report observed Claude models abstaining more often
  under uncertainty, and GPT models more often giving confident wrong answers.

Anthropic later adopted "context rot" as a named concept. Its context-engineering guide
describes an *attention budget* that every token draws down, with recall degrading
gradually as the window fills [37].

**Lesson.** For RAG, context rot argues against sending "everything that might be
relevant". Each irrelevant but similar chunk acts as a distractor. Precision of the
retrieved set matters as much as recall.

### 1.7 Michelangelo and MRCR: beyond haystacks

Michelangelo (Vodrahalli et al., Google DeepMind, September 2024) proposes *Latent
Structure Queries* (LSQ): the model must filter out irrelevant context to reconstruct a
hidden structure, then answer questions about it [48]. It contributes three diagnostic
tasks over code and natural language. One of them, Multi-Round Coreference Resolution
(MRCR), asks the model to reproduce, say, "the second poem about tapirs" from a long
synthetic conversation full of near-duplicate requests. MRCR-style tests have since become
standard in vendor long-context reporting (for example OpenAI's MRCR release with
GPT-4.1 in 2025), because they combine the distractor and coreference pressures that
literal NIAH lacks [48]. For RAG builders, MRCR is a close proxy for "find the right one of
many near-identical chunks", which is exactly the hard-negative problem of Section 2.3.

---

## 2. Long context vs RAG: head-to-head studies

### 2.1 Self-Route (Google DeepMind and University of Michigan, EMNLP 2024 Industry)

**Problem.** Should a query go to long-context (LC) inference over the full document, or
to RAG?

**Findings.** Li et al. compared RAG and LC with Gemini-1.5-Pro, GPT-4o, and GPT-3.5-Turbo
on LongBench and ∞Bench [10]. Given enough resources, LC outperformed RAG on average.
RAG was far cheaper. For a large share of queries the two produced the *same* prediction
[10].

**Architecture.**

```
            query + document
                   |
         +---------v----------+
         |  RAG step (top-k)  |   cheap: k chunks + query
         |  "Is this          |
         |   answerable from  |
         |   these chunks?"   |
         +----+----------+----+
     answerable|          |"unanswerable"
              v           v
         RAG answer   +---v-----------------+
                      | LC step: full doc   |   expensive, used only
                      | (whole context)     |   for the remainder
                      +---------------------+
```

The model judges answerability itself through a self-reflection prompt. Most queries are
resolved in the cheap RAG step and only the remainder escalates [10]. The paper reports
performance comparable to LC at much lower cost. Reported savings are up to roughly 65% of
cost for Gemini-1.5-Pro and about 39% for GPT-4o. Secondary summaries of the paper differ on
whether these are token or cost reductions and on the exact setting, so treat them as
approximate and check the paper's tables [10].
The authors also analysed why RAG fails: multi-step reasoning, general or implicit
queries, long and complex queries, and chunking breaking up the evidence [10].

**Lesson.** Routing is a strong default. Treat LC as an escalation path rather than the
first resort.

### 2.2 OP-RAG: "In defense of RAG in the era of long-context" (NVIDIA, 2024)

Yu et al. argue that very long context dilutes focus. They propose **order-preserving
RAG**: retrieved chunks are presented in their original document order rather than by
score [11]. As the number of retrieved chunks grows, answer quality first rises and then
falls, an inverted U. At the best point, OP-RAG with Llama 3.1 70B beat the same model fed
the full ~128K context while using far fewer tokens (exact F1 values are in the paper)
[11]. This is the "lost in the middle" lesson again: past a point, more context is noise.

### 2.3 "Long-Context LLMs Meet RAG" (Google Cloud AI, ICLR 2025)

Jin et al. studied what happens when a long-context reader receives *more* retrieved
passages [12]. They also found that quality first improves and then declines. The cause
is **hard negatives**: passages that the stronger retriever ranks highly but that do not
contain the answer. A stronger retriever makes this worse, because its negatives are more
convincing [12]. Their remedies were training-free *retrieval reordering*, which puts top
passages at the start and end to exploit the U-curve, and RAG-specific fine-tuning, with
or without intermediate reasoning [12].

### 2.4 LongRAG (University of Waterloo, 2024)

LongRAG changes the balance from the retriever's side. It retrieves much longer units,
about 4K tokens of grouped documents, roughly 30x longer than typical passages. A
long-context reader such as GPT-4o or Gemini 1.5 Pro then extracts the answer [13]. Fewer,
larger units reduce the burden on the retriever and move the work to the reader. This
design assumes the reader is strong at long contexts, so it is exposed to the effects in
Section 1.

### 2.5 Databricks long-context RAG study (2024)

Databricks Mosaic Research ran a large sweep of RAG with increasing retrieved-context
length across a dozen-plus commercial and open models on several corpora [14]. Their
reported conclusions, summarized approximately: more retrieved context helps up to a
point. Only a few frontier models kept accuracy past about 64K tokens, and many open
models degraded well before their limits. The long-context failures were model-specific,
including refusals, copyright-related declines, instruction-following breakdowns, and
repetition [14]. Treat it as evidence that the best context length is a property of the
model and the task, and that it must be measured.

### 2.6 The vendor position: "just put it in the prompt" for small corpora

Anthropic's Contextual Retrieval post gives an explicit threshold. If a knowledge base is
smaller than about 200,000 tokens (about 500 pages), include the whole thing in the prompt
and use prompt caching instead of building RAG [15]. For larger corpora, the same post
introduces contextual chunk augmentation. It reduced top-20 retrieval failure by 35% with
contextual embeddings, 49% with contextual BM25 added, and 67% with reranking as well.
The one-time preprocessing cost was about $1.02 per million document tokens *because*
prompt caching keeps the full document cached while each chunk is contextualized [15].
That is a clear example of caching enabling a RAG indexing technique rather than replacing
RAG.

### 2.7 LaRA: "no silver bullet" (Alibaba and collaborators, ICML 2025)

Li et al. argue that earlier RAG-versus-LC comparisons were inconclusive because of
benchmark artifacts such as leaked answers and contexts that fit easily in the window
[49]. LaRA has 2,326 test cases at 32K and 128K context lengths, spanning four QA task
categories (including location, reasoning, and hallucination-detection questions) over
three kinds of naturally occurring long text: books, academic papers, and financial
documents. Across eleven LLMs, the better choice depended on model capability, context
length, context type, and task type [49]. One consistent pattern is that weaker models
gain more from retrieval, while stronger long-context models narrow or reverse the gap.
The practical upshot matches Self-Route: build a router and calibrate it per model rather
than adopting one paradigm.

### 2.8 Sufficient context (Google Research, ICLR 2025)

Joren et al. ask whether a RAG error comes from the model misusing the context or from the
context lacking the answer [50]. They define *sufficient context*, build an LLM autorater
that labels whether a query-context pair contains enough information, and find that
strong models (Gemini 1.5 Pro, GPT-4o, Claude 3.5) answer well when context is sufficient
but tend to answer wrongly instead of abstaining when it is not. Smaller open models
hallucinate or abstain often even with sufficient context [50]. A selective-generation
method that uses the sufficiency signal for guided abstention raised the share of correct
answers among those the model chose to give by 2-10% [50]. This is the formal version of
Self-Route's "is this answerable from these chunks?" step, and it gives a principled
trigger for escalating to long context or further retrieval.

### 2.9 Synthesis

| Factor | Favors long context | Favors RAG |
|---|---|---|
| Corpus size | Fits comfortably (well under effective length) | Exceeds window or effective length |
| Query type | Global, multi-hop within one document, summarization | Point lookup, freshness, many sources |
| Reuse | Same corpus queried many times (cache amortizes) | Corpus or user varies per query |
| Cost per query | Acceptable with cache hits | Must be minimal |
| Distractor density | Low (curated corpus) | High; retrieval filters noise |
| Updates | Rare | Frequent (re-indexing is incremental) |
| Access control | Single tenant / single permission | Per-document ACLs per user |

---

## 3. Cache-augmented generation (CAG)

### 3.1 Design

Chan et al. (National Chengchi University and Academia Sinica; ACM Web Conference 2025
short paper) propose dropping retrieval entirely when the knowledge base is bounded [16]:

```
  OFFLINE (once per knowledge base)                ONLINE (per query)
  +-----------------------------+                  +--------------------------+
  | all documents D             |                  | query q                  |
  |   -> one long prompt        |                  |   -> append q tokens     |
  |   -> forward pass (prefill) |   KV cache C_D   |   -> decode using C_D    |
  |   -> save KV cache C_D -----+----------------->|   (no retrieval step)    |
  +-----------------------------+   (disk/GPU)     +------------+-------------+
                                                               |
                                    reset: truncate C_D back   |
                                    to its original length  <--+
                                    (drop appended q/answer)
```

**Key techniques.** (1) Precompute the KV cache of the full document set once. (2) At
inference, load it and append only the query. (3) Between queries, *truncate* the cache
back to the document prefix instead of recomputing it [16]. Their implementation uses
Llama-3.1-8B-Instruct on SQuAD and HotpotQA reference sets of increasing size. The
repository gives knowledge sizes ranging from about 21K tokens (3 SQuAD documents) up to
about 10M tokens for all 7,405 HotpotQA articles. The configurations that CAG can serve
directly are bounded by the model's 128K window [16].

**Results.** The authors report BERTScore comparable to or better than sparse and dense
RAG baselines, and large reductions in generation time compared with feeding the full
context uncached [16]. In practice CAG is "prompt caching taken to its logical end": the
whole corpus is the cached prefix.

**Failure modes (stated by the authors).** The whole knowledge source must fit in the
window. Quality may degrade on very long contexts [16]. By Section 1's evidence, the
second limitation binds long before the first. A 128K-token CAG prompt sits in the range
where NoLiMa and context-rot degradation is measurable.

**When it fits.** Small, stable corpora queried heavily: product manuals, policy
handbooks, a single codebase, the "under 200K tokens" regime in [15]. Hosted APIs offer
the same pattern through provider caching (Section 4). The economics depend on the cache
read discount and TTL rather than on owning the KV tensors.

---

## 4. Prefix and prompt caching

### 4.1 The mechanism

All prefix caching rests on one fact. In a causal transformer, a token's KV entries depend
only on the tokens before it. If two requests share an identical token prefix, the KV
cache for that prefix is identical and can be reused, skipping prefill for those tokens.
Exact-match caching therefore requires a stable prefix and puts variable content last.

```
  request N   [ tools | system | docs (static) | history | new user turn ]
  request N+1 [ tools | system | docs (static) | history | turn | new turn ]
               \______________ cache hit ______________/ \_ new prefill _/
  any byte change at position p invalidates everything after p
```

### 4.2 Serving engines

**vLLM automatic prefix caching.** KV memory is split into fixed-size blocks. Each full
block is hashed from the *parent block's hash*, the block's tokens, and "extra keys" such
as LoRA ID, multimodal input hashes, and a per-tenant *cache salt*. The chain therefore
encodes the entire prefix [22]. Only full blocks are cached. Free blocks sit in an LRU
free queue and are evicted from the head. When a request finishes, its blocks return in
reverse order, so the tail blocks, which are least likely to be shared, go first [22].
Limits: matching is block-granular, so a partially matching block yields no reuse.
Non-cryptographic hash options (xxhash) carry collision and cross-tenant leakage risk.
The default sha256 is not guaranteed reproducible across versions [22].

**SGLang RadixAttention.** SGLang keeps KV caches in a radix tree keyed by token
sequences. Any request reuses the longest matching prefix automatically, with LRU eviction
of leaves. It also schedules cache-aware, ordering requests to maximize hits [23]. This
suits programs with branching prompts (few-shot, tree-of-thought, multi-call agents)
where many requests share prefixes of different lengths.

### 4.3 Hosted provider caching

**Anthropic (explicit breakpoints).** Developers mark `cache_control` on content blocks,
or use automatic caching, which moves the breakpoint forward as a conversation grows.
Up to 4 breakpoints are allowed per request, each with a 20-block lookback window [18].
The cache hierarchy is `tools -> system -> messages`. Changing tool definitions
invalidates everything. Toggling features such as web search or citations invalidates
system and messages. Changing `tool_choice` or images invalidates
messages; thinking-setting changes always invalidate messages and, on some models, earlier
levels too [18]. The default TTL is 5 minutes, refreshed on each hit, with an optional 1-hour
TTL. Writes cost 1.25x base input for 5 minutes or 2x for 1 hour. Reads cost 0.1x on
standard models, with lower read multipliers on some newer models. There is a per-model
minimum cacheable length of 512 to 4,096 tokens, and shorter prompts are silently not
cached [18]. The launch post reported up to 90% cost and up to 85% latency reduction. For
"chat with a book" with a 100K-token cached prompt, TTFT fell from 11.5 s to 2.4 s (-79%)
and cost fell by 90% [17].

**OpenAI (automatic).** Caching applies automatically to prompts over 1,024 tokens, with
no code change. Usage reports `cached_tokens`. Caches are scoped to the organization and
are eligible for zero data retention. OpenAI cites up to 80% latency reduction for prompts
over 10K tokens [19]. The launch announcement (October 2024) described a 50% discount on
cached input tokens, and caches typically cleared after minutes of inactivity [20]. Later
models and options changed discounts and retention, so check current documentation.
Official best practice matches the general rule: static content first, identical tool
definitions and ordering, append-only messages, and identical image inputs including the
`detail` parameter [19].

**Google Gemini (explicit and implicit).** Explicit context caches are created as named
objects through `caches.create` with model, contents, and system instruction. They are
model-specific, have a default TTL of one hour that can be updated, and incur a storage
charge in addition to discounted reads [21]. Gemini also offers implicit (automatic)
caching on newer models [21]. The explicit-object model maps most directly onto CAG: the
"knowledge base" becomes a named, TTL'd cache resource that many requests reference.

### 4.4 Design rules for cache-friendly RAG prompts

1. **Layout:** `[tools][system][static corpus or few-shot][retrieved chunks][history]
   [query]`. Retrieved chunks vary per query, so they sit after the cacheable region, and
   prefix caching alone gives them no benefit. That gap is what Section 5 addresses.
2. **Determinism:** stable serialization (sorted JSON keys, fixed tool order), no
   timestamps or request IDs in the system prompt. Manus calls out the timestamp
   anti-pattern explicitly [43].
3. **Breakpoint placement:** on the last block that is identical across requests, never on
   a per-request block, or every call pays a write and never gets a read [18].
4. **Measure:** track `cache_read_input_tokens` and `cache_creation_input_tokens` (or
   `cached_tokens`), and treat hit rate as a first-class SLO. Manus calls KV-cache hit
   rate "the single most important metric" for a production agent [43].
5. **Context editing invalidates caches.** Clearing old tool results rewrites the prefix.
   Anthropic's docs recommend `clear_at_least`, so each invalidation frees enough tokens to
   pay for the new cache write [40].

---

## 5. Non-prefix KV reuse for RAG

### 5.1 The problem

In RAG the expensive tokens are the retrieved chunks, typically thousands of tokens per
query. The same popular chunks recur across many queries, but in different combinations
and positions. Prefix caching only reuses chunk *i*'s KV cache if every token before it is
identical. The obvious fix is to precompute each chunk's KV independently and concatenate
them, but that is wrong in two ways:

- **Positional mismatch.** RoPE-encoded keys carry absolute positions. A chunk cached at
  position 0 but placed at position 3,000 has incorrect positional encoding.
- **Missing cross-attention.** A token in chunk 2 never attended to chunk 1 when its KV was
  computed independently. Multi-hop answers that need cross-chunk links degrade.

```
  Full prefill (correct, slow):     [sys][c1][c2][c3][q]  all tokens attend causally
  Naive concat (fast, wrong):       [sys][c1*][c2*][c3*][q]
                                         ^    ^    ^  each precomputed alone,
                                         |    |    |  position 0-based,
                                         +----+----+  no cross-chunk attention
  Fixes:  (a) re-rotate positions   (b) recompute a subset of tokens  (c) fine-tune the
          model to expect block-local attention  (d) add link/summary tokens
```

The systems below are different combinations of (a) through (d).

### 5.2 Prompt Cache (Yale, 2023)

Gim et al. introduced *modular attention reuse* [24]. Prompts are written in a Prompt
Markup Language (PML) schema of reusable *modules*, each pre-encoded at a fixed position
offset reserved in the schema. Alternative modules in a *union* share an offset, and
parameters have a maximum length plus a scaffold placeholder [24]. Modules can opt out of
caching (`cache="false"`) to be recomputed each time. In the repository's persona example,
TTFT fell from about 287 ms to 78 ms on an RTX 4090 and from about 17.9 s to 0.51 s on a
CPU [24]. Lesson: reserving positions up front sidesteps re-rotation, at the cost of
requiring a schema and accepting the loss of cross-module attention.

### 5.3 RAGCache (Peking University and ByteDance, 2024)

RAGCache caches the intermediate KV states of retrieved documents in a *knowledge tree*
organized by document order. The tree is spread across GPU and host memory, uses a
RAG-aware replacement policy (PGDSF), and *speculatively pipelines* retrieval with
inference so prefill can start before vector search finishes [25]. Because it stays
prefix-faithful (a document's cache is reused only after the same preceding documents),
it is exact but gets lower hit rates when document combinations vary. Reported gains are
up to about 4x lower TTFT and about 2x higher throughput compared with vLLM plus Faiss
(approximate, from the abstract) [25].

### 5.4 CacheBlend (University of Chicago, EuroSys 2025)

**Problem.** Reuse precomputed chunk KV caches at *non-prefix* positions without the
quality loss of naive concatenation, and without fine-tuning the model [26].

**Architecture.**

```
   retrieved chunks c1..cn                       query q
          |                                         |
   +------v---------------------+                   |
   | KV store (GPU/CPU/disk)    |  precomputed per  |
   | KV(c_i) computed alone     |  chunk, offline   |
   +------+---------------------+                   |
          | load (pipelined, layer by layer)        |
   +------v-------------------------------------------v------+
   | Layer l:                                                |
   |  1. fix positions of reused K (RoPE re-rotation)        |
   |  2. pick "high-KV-deviation" tokens: those whose KV     |
   |     would differ most under full prefill (selected in   |
   |     early layers, refined layer by layer)               |
   |  3. recompute KV only for those tokens (~10-20%) with   |
   |     full cross-chunk attention; keep the rest reused    |
   |  4. while computing layer l, load KV for layer l+1      |
   +-----------------------------+---------------------------+
                                 v
                           decode answer
```

**Key techniques.** *Selective KV recompute*: only a small fraction of tokens, those with
the highest deviation from their full-prefill KV, are updated, which restores most of the
cross-attention. The token set is chosen progressively across layers. *Pipelining*:
recomputation is overlapped with fetching KV from slower storage, so caches can live on
CPU RAM or SSD without adding latency [26].

**Numbers.** The abstract reports TTFT reductions of 2.2-3.3x and throughput gains of
2.8-5x compared with full KV recompute, across three open-source LLMs and four benchmark
datasets, without compromising generation quality. The repository's example scripts use
MuSiQue, 2WikiMQA, and SAMSum [26]. The paper received a EuroSys 2025 Best Paper award
[26].

**Productization.** CacheBlend's maintained implementation lives in LMCache. It supports
non-prefix reuse "even when the new prompt is not a prefix of the cached one, such as RAG
with swapped documents", running in-process with vLLM and offloading to CPU or disk
[33].

**Failure modes.** The recompute ratio trades quality for speed. Prompts need chunk
separators so boundaries can be found. Integration currently needs engine patches,
marked experimental [33]. Quality guarantees are empirical, not exact.

### 5.5 TurboRAG (Moore Threads, EMNLP 2025)

**Design.** TurboRAG precomputes the KV cache of each document chunk offline (512-token
chunks in the reference code) and stores it. At query time it retrieves chunk caches,
concatenates them, and prefills only the query [27]. To make independently computed chunks
consistent, the paper (a) rearranges position IDs so that concatenated chunks behave as if
they had been encoded in place, and (b) uses an *independent attention mask* during
fine-tuning, so the model learns to answer from chunks that never attended to each other
[27].

**Numbers.** The repository's demo shows TTFT for 10 retrieved documents of about 0.65 s
with cached KV compared with 4.13 s without, roughly 6x [27]. The paper reports TTFT
reductions of up to 9.4x (8.6x on average) compared with conventional RAG, with accuracy
comparable to standard RAG on the evaluated benchmarks [27]. The speedup depends heavily
on where the cache lives: in the paper's A100 batch experiments, including the CPU-to-GPU
transfer over PCIe cut the gain to roughly 4-5x [27].

**Contrast with CacheBlend.** TurboRAG pays a one-time fine-tuning cost and gets
near-zero query-time recomputation. CacheBlend needs no fine-tuning and pays a small,
tunable recompute at query time. TurboRAG ties the cache to a specific fine-tuned
checkpoint. CacheBlend works with stock models.

### 5.6 Block-Attention (ICLR 2025)

Block-Attention splits retrieved documents into blocks that compute KV independently. Only
the *final* block, holding the user query, attends to everything [28]. It re-encodes
positions and fine-tunes the model on this attention pattern. The model can switch between
block and full attention without loss. The authors report comparable accuracy to full
attention across 11 RAG, ICL, and general benchmarks after block fine-tuning. For a 32K
input, TTFT was 45 ms, which is 98.7% lower TTFT and 99.8% fewer FLOPs than full attention
[28].

### 5.7 EPIC, APE, KVLink: the 2025 refinements

- **EPIC** (position-independent context caching) formalizes the problem as
  *Position-Independent Caching*. Its LegoLink algorithm recomputes only a few tokens at
  each chunk's start, where attention-sink effects concentrate, instead of selecting tokens
  by deviation. It reports substantial TTFT and throughput gains over prefix-only caching
  with small accuracy loss (figures in the paper) [29].
- **APE** (Adaptive Parallel Encoding, CMU and NVIDIA, ICLR 2025) encodes contexts in
  parallel and corrects the attention-distribution mismatch with a shared prefix, an
  attention temperature, and a scaling factor. The defaults (0.9) keep over 90% of
  sequential-encoding performance on few-shot tasks without training [30].
- **KVLink** (UCSB, 2025) re-positions independently cached segments and adds trainable
  *link/summary tokens* per segment, which reconnect cross-document context under a
  segment attention mask [31]. It is the "add tokens" fix in the taxonomy above.

### 5.8 KVzip: query-agnostic compression for reusable caches (NeurIPS 2025)

Most KV-eviction methods (SnapKV-style) score entries by attention from the *current*
query, so a compressed cache is only good for that query. That defeats the purpose of CAG
and chunk-cache stores, where one cache must serve many future queries. KVzip (Kim et al.,
Seoul National University and NAVER AI Lab, NeurIPS 2025 oral) scores each KV pair by how
much the model needs it to *reconstruct the original context* from the cache, and evicts
the least important pairs [51]. The resulting cache is query-agnostic. The authors report
a 3-4x smaller KV cache and about 2x lower FlashAttention decoding latency with negligible
loss on QA, retrieval, reasoning, and code tasks, on LLaMA 3.1, Qwen2.5, and Gemma 3 at up
to 170K tokens. Query-aware baselines degraded even at a 90% cache budget in multi-query
settings [51]. For CAG and LMCache-style stores, query-agnostic compression directly
reduces the storage bottleneck discussed below.

### 5.9 Lessons across the KV-reuse line

1. **There is no free lunch on cross-chunk attention.** Each system either recomputes some
   tokens (CacheBlend, EPIC), adds tokens (KVLink), or trains the model to need less of it
   (TurboRAG, Block-Attention). Pure concatenation without either loses accuracy on
   multi-hop questions.
2. **Storage becomes the bottleneck.** KV for a chunk is far larger than its text. Systems
   compress caches (CacheGen encodes KV into a compact bitstream for streaming [32]) and
   tier them across GPU, CPU, SSD, and remote storage (Section 6).
3. **Model coupling.** KV caches are tied to an exact model checkpoint, and with LoRA, to
   the adapter. Any model upgrade invalidates the whole chunk-cache store, which must be
   planned like an embedding re-index.

---

## 6. KV-cache infrastructure

**LMCache** is a vendor-neutral KV cache layer that sits beside the engine. It offloads
KV from GPU to CPU RAM, local disk, and remote backends (Redis/Valkey, Mooncake,
S3-compatible stores, NIXL, GDS). It supports non-prefix reuse via CacheBlend and
prefill/decode disaggregation transfers over NVLink, RDMA, or TCP. It runs as a standalone
daemon so cached KV survives an engine crash [33]. It integrates with vLLM, NVIDIA Dynamo,
and the vLLM Production Stack [33].

**Mooncake** (Moonshot AI, FAST 2025 Best Paper) is the serving platform behind Kimi. It
is KV-cache-centric and disaggregated, "trading more storage for less computation". A
distributed KV store over DRAM and SSD feeds prefill and decode pools through a Transfer
Engine that reaches up to 87 GB/s on 4x200 Gbps and 190 GB/s on 8x400 Gbps RoCE, by the
project's own measurements [34]. The project says it let Kimi handle 75% more requests
under real workloads while meeting SLOs; no baseline or methodology is given in the
repository [34]. For long-context and RAG-heavy traffic,
this architecture turns prefix reuse into a cluster-wide resource rather than a per-GPU
accident.

**Cache-aware routing.** A cache hit requires the request to land on a worker holding the
prefix. llm-d (founded by Red Hat, Google Cloud, IBM Research, CoreWeave, and NVIDIA; a
CNCF sandbox project since March 2026) reports that prefix-cache-aware routing gave about
3x output throughput and 2x faster TTFT compared with round-robin for Llama 3.1 70B on four
AMD MI300X accelerators, and that hierarchical KV offloading gave about 13.9x throughput at
250 concurrent users compared with GPU-only on four H100s [35]. These are project-reported
benchmarks. NVIDIA Dynamo routes on "worker load and KV cache overlap" and reports about 2x
faster TTFT from KV-aware routing in a single third-party (Baseten) benchmark on
Qwen3-Coder 480B. Its KV Block Manager tiers caches across GPU, CPU, SSD, and remote
storage, including S3 and Azure Blob in version 1.0 [36].

```
                  +-------------------------------+
   requests ----->| cache-aware router            |  hashes prefix blocks, looks up
                  | (llm-d / Dynamo / gateway)    |  global KV index, picks worker
                  +---+-----------+-----------+---+
                      |           |           |
                 +----v---+  +----v---+  +----v---+     prefill / decode pools
                 | vLLM / |  | vLLM / |  | SGLang |     (disaggregated or not)
                 | worker |  | worker |  | worker |
                 +----+---+  +----+---+  +----+---+
                      |  KV connector (LMCache / NIXL / Mooncake TE)
                 +----v--------------------------------------------+
                 | tiered KV store: HBM -> CPU DRAM -> SSD -> S3   |
                 +-------------------------------------------------+
```

---

## 7. Context engineering for agents

### 7.1 From prompt engineering to context engineering

In mid-2025, "context engineering" became the name for curating *everything* in the window
at each step of an agent loop: system prompt, tools, retrieved data, message history, tool
results, and memory. Anthropic's definition centres on finding "the smallest possible set
of high-signal tokens" that produce the desired behaviour, because context rot makes extra
tokens expensive in quality as well as dollars [37]. LangChain's Lance Martin grouped
techniques into four verbs: **write** (persist outside the window: scratchpads, memories),
**select** (retrieve into the window: RAG over tools, memories, knowledge), **compress**
(summarize or trim), and **isolate** (split across sub-agents or sandboxes) [44]. Drew
Breunig catalogued how long contexts fail [45]:

- **Context poisoning:** a hallucination or error enters the context and is repeatedly
  referenced.
- **Context distraction:** the context grows so long that the model over-focuses on it and
  neglects what it learned in training.
- **Context confusion:** superfluous content, such as too many tools, influences the
  response.
- **Context clash:** parts of the context contradict each other.

### 7.2 Anthropic's techniques

The engineering guide [37] and API features [39][40][41] set out a layered toolkit:

- **Just-in-time retrieval.** Keep lightweight references (file paths, queries, URLs) and
  load data through tools only when needed. Hybrid designs preload some data upfront [37].
  This is agentic RAG with the agent controlling the retrieval timing.
- **Compaction.** Summarize a nearly full window and restart from the summary [37]. The API
  now offers server-side compaction, on demand or at a token threshold, optionally keeping
  recent turns verbatim, because "response quality degrades as a conversation grows" [41].
- **Context editing.** Automatically clear old tool results (`clear_tool_uses`) or old
  thinking blocks (`clear_thinking`) past a trigger, which defaults to 100K input tokens,
  keeping the last N tool uses and leaving placeholders [40].
- **Structured note-taking / memory tool.** The agent writes notes such as a to-do list or
  NOTES.md to a file-based memory outside the window and reads them back later [37][39].
- **Sub-agents.** Specialized agents work in clean windows and return condensed summaries.
  In Anthropic's multi-agent research system, a lead Opus 4 with Sonnet 4 sub-agents beat
  single-agent Opus 4 by 90.2% on an internal eval. Token usage alone explained about 80%
  of BrowseComp performance variance, and multi-agent runs used about 15x the tokens of
  chat [38]. Sub-agents can write outputs to a filesystem and pass references back, which
  reduces "game of telephone" loss [38].

Reported impact: on an internal agentic-search eval, the memory tool plus context editing
improved results 39% over baseline, and context editing alone 29%. In a 100-turn web-search
eval, context editing let workflows complete that otherwise failed from context
exhaustion, while cutting token use by 84% [39].

### 7.3 Manus: designing around the KV cache

Manus's July 2025 post by Yichao "Peak" Ji is the best-known practitioner account of
cache-aware agent design [43]:

- **KV-cache hit rate is the key production metric.** Agent loops have very skewed
  input-to-output ratios (about 100:1 in Manus), so prefill dominates cost. With Claude
  Sonnet, cached input was cited at $0.30/MTok against $3/MTok uncached, a 10x difference
  [43].
- **Stable prefix, append-only context, deterministic serialization.** Avoid timestamps at
  the start of the system prompt and unstable JSON key order. Mark cache breakpoints
  explicitly where needed [43].
- **Mask, don't remove.** Adding or removing tools mid-loop invalidates the cache and
  confuses the model when earlier actions refer to tools that no longer exist. Manus keeps
  tool definitions fixed and constrains action selection by masking token logits through a
  state machine [43].
- **File system as context.** The sandbox file system serves as unlimited, persistent,
  agent-operable memory. Compression is kept *restorable*: drop a web page's content but
  keep its URL [43].
- **Recitation.** Rewriting a `todo.md` at the end of the context pushes the global plan
  into recent attention, a direct counter to lost-in-the-middle over long trajectories
  [43].
- **Keep errors in.** Leaving failed actions and stack traces in context lets the model
  update away from repeating them [43].
- **Avoid few-shot ruts.** Uniform repeated action/observation patterns make the agent
  imitate itself, so introduce structured variation [43].

### 7.4 Context-engineering reference architecture

```
  +-------------------------------- context window (per step) -------------------+
  | [tools: fixed set, masked]  [system prompt: stable]   <- cached prefix       |
  | [pinned knowledge / CAG corpus (optional)]            <- cached prefix       |
  | [compacted summary of older turns]                    <- re-cached on compact|
  | [recent turns verbatim] [recent tool results]                                |
  | [recitation: current plan / todo]  [new observation]  <- uncached tail       |
  +---------------------------------+--------------------------------------------+
            ^ select (JIT retrieval) |  write (notes, files)    | isolate
            |                        v                          v
  +---------+--------+   +--------------------+   +-----------------------------+
  | RAG / search /   |   | memory dir, files, |   | sub-agents with clean       |
  | code grep tools  |   | todo.md, NOTES.md  |   | windows -> condensed results|
  +------------------+   +--------------------+   +-----------------------------+
```

### 7.5 ACE: contexts as evolving playbooks (ICLR 2026)

Agentic Context Engineering (Zhang et al., Stanford, SambaNova, and UC Berkeley; ICLR
2026) treats the context itself as the thing being optimized [52]. Instead of rewriting a
prompt or summary wholesale, ACE maintains a structured *playbook* of itemized strategies
and lessons, updated by a loop of Generator, Reflector, and Curator roles. It applies
small incremental "delta" edits rather than full rewrites. The authors identify two
failure modes of earlier prompt-optimization and memory methods: *brevity bias*, where
summaries drop domain detail, and *context collapse*, where repeated rewriting erodes
accumulated knowledge [52]. Building on Dynamic Cheatsheet, ACE reports gains of about
10.6% on agent benchmarks and 8.6% on finance tasks, with lower adaptation latency and
rollout cost, and can adapt from execution feedback without labels. On the AppWorld
leaderboard it matched the top production agent on average and beat it on the harder
test-challenge split while using a smaller open model [52]. It is a counterpoint to
aggressive compaction: for knowledge that should accumulate, grow and curate the context
instead of summarizing it away. Delta updates are also friendlier to prefix caching than
full rewrites.

---

## 8. Comparison and decision guide

### 8.1 Comparison of designs

| Design | Where reuse happens | Exactness | Model change | Main win | Main risk |
|---|---|---|---|---|---|
| Plain long context | none | exact | none | simplicity, global reasoning | cost, context rot, mid-loss |
| Self-Route [10] | none (routing) | exact | none | LC quality at lower cost | judge errors in routing |
| CAG [16] | whole corpus prefix | exact | none | no retrieval latency | corpus must fit; rot |
| Provider caching [17]-[21] | exact prefix | exact | none | up to ~90% input cost cut | TTL misses, prefix drift |
| vLLM APC / RadixAttention [22][23] | exact prefix blocks | exact | none | free prefill reuse | block granularity, routing |
| Prompt Cache [24] | schema modules | approximate | none | modular reuse | no cross-module attention |
| RAGCache [25] | doc-order prefix tree | exact | none | RAG prefix reuse + pipelining | low hit rate on new combos |
| CacheBlend [26] | any chunk position | approximate | none | ~2-3x TTFT, quality kept | recompute ratio tuning |
| TurboRAG [27] | any chunk position | approximate | fine-tune | ~6-9x TTFT | checkpoint lock-in |
| Block-Attention [28] | any block position | approximate | fine-tune | ~99% TTFT cut at 32K | training required |
| EPIC / APE / KVLink [29]-[31] | any position | approximate | none / none / train | lighter-weight fixes | newer, less proven |
| KVzip [51] | compressed reusable cache | lossy | none | 3-4x smaller KV, multi-query | one-time reconstruction scoring cost |
| Context editing / compaction [39]-[41] | n/a (shrinks window) | lossy | none | long agent runs | cache invalidation, info loss |

### 8.2 Decision guide

```
  Is the knowledge static and < ~effective context length of your model?
     |-- yes --> Is it queried repeatedly within the cache TTL?
     |             |-- yes --> CAG / provider prompt caching (pin it in the prefix)
     |             |-- no  --> long context per query, or small RAG
     |-- no  --> RAG.
                 Are the same chunks hot across many queries, and is TTFT critical?
                   |-- yes, can fine-tune --> TurboRAG / Block-Attention
                   |-- yes, stock model   --> CacheBlend via LMCache (or EPIC/APE)
                   |-- no                 --> standard RAG, cache only the static prefix
  Hard global / multi-hop questions?  --> route escalations to long context (Self-Route)
  Agent loop?  --> context engineering: stable prefix, JIT retrieval, compaction,
                   notes, sub-agents; track cache hit rate
```

---

## Key takeaways for system designers

1. **Plan around effective context, not advertised context.** Measure your model on
   semantically indirect, distractor-laden tasks, in the style of NoLiMa or context rot,
   at the lengths you intend to use. Literal NIAH results overstate capability [4][5][6].
2. **Precision beats volume.** Quality versus number of retrieved chunks is an inverted U.
   Hard negatives from strong retrievers hurt long-context readers most [11][12]. Rerank
   and cut aggressively, and consider document-order presentation or edge placement.
3. **Order is a parameter.** Put long material first and the query last. Put the
   strongest evidence at the edges, and have the model quote before answering [1][42].
4. **Route, don't choose.** Cheap RAG first, with escalation to full long context when
   the model judges the chunks insufficient, keeps most of LC's quality at a fraction of
   the cost [10].
5. **For small, stable corpora, prompt caching often beats a RAG stack.** Below roughly
   200K tokens, pinning the corpus in a cached prefix (CAG) removes retrieval failure
   modes entirely [15][16]. Re-evaluate as the corpus grows into the context-rot regime.
6. **Design prompts for the cache.** Static to dynamic order, deterministic serialization,
   no timestamps up front, fixed tool sets (mask rather than remove), breakpoints on the
   last stable block, and hit rate tracked as an SLO [18][19][43].
7. **Retrieved chunks need non-prefix reuse.** If the same chunks are hot and TTFT
   matters, use selective recompute (CacheBlend) for stock models or fine-tuned
   block-attention (TurboRAG, Block-Attention) where you control the model. Budget for KV
   storage tiers and full cache invalidation on every model upgrade [26][27][28][33].
8. **Make the cache cluster-wide.** Prefix hits only happen if routing sends requests to
   the worker that holds the prefix. Cache-aware routing and tiered KV stores (llm-d,
   Dynamo, Mooncake, LMCache) are now standard parts of RAG serving [33][34][35][36].
9. **Agents need context engineering as much as retrieval.** Just-in-time retrieval,
   compaction, context editing, file-based memory, recitation, and sub-agent isolation
   address context rot inside long trajectories. Each of them interacts with caching, so
   make every invalidation count [37][39][40][43].
10. **Evaluate failure behaviour, not just accuracy.** Models differ in how they fail on
    long inputs: abstaining, confidently answering wrongly, refusing, or repeating
    themselves [6][14]. Production guardrails should match the failure profile of the
    model you deploy.

---

## References

1. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P.
   "Lost in the Middle: How Language Models Use Long Contexts." TACL, 2024 (arXiv 2023).
   https://arxiv.org/abs/2307.03172 (code: https://github.com/nelson-liu/lost-in-the-middle)
2. Kamradt, G. "LLMTest_NeedleInAHaystack" (Needle in a Haystack test). GitHub, 2023.
   https://github.com/gkamradt/LLMTest_NeedleInAHaystack
3. Anthropic. "Long context prompting for Claude 2.1." 2023.
   https://claude.com/blog/claude-2-1-prompting
4. Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y.,
   Ginsburg, B. (NVIDIA). "RULER: What's the Real Context Size of Your Long-Context
   Language Models?" 2024. https://arxiv.org/abs/2404.06654 (repository and leaderboard:
   https://github.com/NVIDIA/RULER)
5. Modarressi, A., et al. (Adobe Research, LMU Munich). "NoLiMa: Long-Context Evaluation
   Beyond Literal Matching." ICML, 2025. https://arxiv.org/abs/2502.05167 (repository:
   https://github.com/adobe-research/NoLiMa)
6. Hong, K., Troynikov, A., Huber, J. (Chroma). "Context Rot: How Increasing Input Tokens
   Impacts LLM Performance." 2025. https://research.trychroma.com/context-rot (code:
   https://github.com/chroma-core/context-rot)
7. Bai, Y., et al. (Tsinghua University, Zhipu AI). "LongBench v2: Towards Deeper
   Understanding and Reasoning on Realistic Long-context Multitasks." ACL, 2025 (arXiv 2024).
   https://arxiv.org/abs/2412.15204 (repository: https://github.com/THUDM/LongBench)
8. Yen, H., Gao, T., Hou, M., Ding, K., Fleischer, D., Izsak, P., Wasserblat, M., Chen, D.
   "HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly."
   ICLR, 2025. https://arxiv.org/abs/2410.02694 (repository:
   https://github.com/princeton-nlp/HELMET)
9. Gemini Team, Google. "Gemini 1.5: Unlocking multimodal understanding across millions of
   tokens of context." 2024. https://arxiv.org/abs/2403.05530
10. Li, Z., Li, C., Zhang, M., Mei, Q., Bendersky, M. (Google DeepMind, University of
    Michigan). "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study
    and Hybrid Approach." EMNLP Industry, 2024. https://arxiv.org/abs/2407.16833
11. Yu, T., Xu, A., Akkiraju, R. (NVIDIA). "In Defense of RAG in the Era of Long-Context
    Language Models." 2024. https://arxiv.org/abs/2409.01666
12. Jin, B., Yoon, J., Han, J., Arik, S. O. (Google Cloud AI Research, UIUC).
    "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG." ICLR, 2025.
    https://arxiv.org/abs/2410.05983
13. Jiang, Z., Ma, X., Chen, W. (University of Waterloo). "LongRAG: Enhancing
    Retrieval-Augmented Generation with Long-context LLMs." 2024.
    https://arxiv.org/abs/2406.15319 (code: https://github.com/TIGER-AI-Lab/LongRAG)
14. Leng, Q., et al. (Databricks Mosaic Research). "Long Context RAG Performance of Large
    Language Models." 2024. https://www.databricks.com/blog/long-context-rag-performance-llms
15. Anthropic. "Introducing Contextual Retrieval." 2024.
    https://www.anthropic.com/news/contextual-retrieval
16. Chan, B. J., Chen, C.-T., Cheng, J.-H., Huang, H.-H. "Don't Do RAG: When
    Cache-Augmented Generation is All You Need for Knowledge Tasks." ACM Web Conference
    (short paper), 2025. https://arxiv.org/abs/2412.15605 (code:
    https://github.com/hhhuang/CAG)
17. Anthropic. "Prompt caching with Claude." 2024 (updated 2025).
    https://claude.com/blog/prompt-caching
18. Anthropic. "Prompt caching" (Claude API documentation). Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/prompt-caching
19. OpenAI. "Prompt Caching 101" (OpenAI Cookbook). 2024.
    https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_Caching101.ipynb
20. OpenAI. "Prompt Caching in the API." 2024. https://openai.com/index/api-prompt-caching/
21. Google. "Context caching" (Gemini API documentation) and "Caching" cookbook notebook.
    https://ai.google.dev/gemini-api/docs/caching ;
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/Caching.ipynb
22. vLLM Project. "Automatic Prefix Caching" (design document). Accessed 2026.
    https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md
23. Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs"
    (RadixAttention). 2023/NeurIPS 2024. https://arxiv.org/abs/2312.07104
24. Gim, I., Chen, G., Lee, S., Sarda, N., Khandelwal, A., Zhong, L. (Yale). "Prompt Cache:
    Modular Attention Reuse for Low-Latency Inference." 2023.
    https://arxiv.org/abs/2311.04934 (code: https://github.com/yale-sys/prompt-cache)
25. Jin, C., et al. (Peking University, ByteDance). "RAGCache: Efficient Knowledge Caching
    for Retrieval-Augmented Generation." 2024. https://arxiv.org/abs/2404.12457
26. Yao, J., Li, H., Liu, Y., Ray, S., Cheng, Y., Zhang, Q., Du, K., Lu, S., Jiang, J.
    (University of Chicago). "CacheBlend: Fast Large Language Model Serving for RAG with
    Cached Knowledge Fusion." EuroSys, 2025 (Best Paper). https://arxiv.org/abs/2405.16444
    (code: https://github.com/YaoJiayi/CacheBlend; award note:
    https://blog.lmcache.ai/2025-03-31-eurosys)
27. Lu, S., Wang, H., Rong, Y., Chen, Z., Tang, Y. (Moore Threads). "TurboRAG:
    Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked
    Text." EMNLP, 2025 (arXiv 2024). https://arxiv.org/abs/2410.07590 (code:
    https://github.com/MooreThreads/TurboRAG)
28. Ma, D., Wang, Y., Lan, T. "Block-Attention for Efficient Prefilling." ICLR, 2025.
    https://arxiv.org/abs/2409.15355 (code: https://github.com/TemporaryLoRA/Block-Attention)
29. Hu, J., et al. "EPIC: Efficient Position-Independent Context Caching for Serving Large
    Language Models." 2024. https://arxiv.org/abs/2410.15332
30. Yang, X., Chen, T., Chen, B. "APE: Faster and Longer Context-Augmented Generation via
    Adaptive Parallel Encoding." ICLR, 2025. https://arxiv.org/abs/2502.05431 (code:
    https://github.com/Infini-AI-Lab/APE)
31. Yang, J., Hou, B., Wei, W., Bao, Y., Chang, S. "KVLink: Accelerating Large Language
    Models via Efficient KV Cache Reuse." 2025. https://arxiv.org/abs/2502.16002 (code:
    https://github.com/UCSB-NLP-Chang/KVLink)
32. Liu, Y., et al. (University of Chicago). "CacheGen: KV Cache Compression and Streaming
    for Fast Large Language Model Serving." SIGCOMM, 2024.
    https://arxiv.org/abs/2310.07240 (code: https://github.com/UChi-JCL/CacheGen)
33. Cheng, Y., et al. "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM
    Inference." 2025. https://arxiv.org/abs/2510.09665 (repository:
    https://github.com/LMCache/LMCache)
34. Qin, R., et al. (Moonshot AI, Tsinghua University). "Mooncake: Trading More Storage for
    Less Computation" (FAST 2025) / "Mooncake: A KVCache-centric Disaggregated
    Architecture for LLM Serving." https://arxiv.org/abs/2407.00079 (repository:
    https://github.com/kvcache-ai/Mooncake)
35. llm-d project (Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA). "llm-d."
    GitHub, 2025. https://github.com/llm-d/llm-d
36. NVIDIA. "Dynamo: datacenter-scale distributed inference serving framework." GitHub,
    2025. https://github.com/ai-dynamo/dynamo
37. Anthropic Applied AI team. "Effective context engineering for AI agents." 2025.
    https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
38. Anthropic. "How we built our multi-agent research system." 2025.
    https://www.anthropic.com/engineering/multi-agent-research-system
39. Anthropic. "Managing context on the Claude Developer Platform" (context editing and
    memory tool). 2025. https://claude.com/blog/context-management
40. Anthropic. "Context editing" (Claude API documentation). Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/context-editing
41. Anthropic. "Compaction overview" (Claude API documentation). Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/compaction
42. Anthropic. "Prompting best practices: Long context prompting" (Claude API
    documentation). Accessed 2026.
    https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
43. Ji, Y. "Peak" (Manus). "Context Engineering for AI Agents: Lessons from Building
    Manus." 2025.
    https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
44. Martin, L. (LangChain). "Context Engineering for Agents." 2025.
    https://blog.langchain.com/context-engineering-for-agents/
45. Breunig, D. "How Long Contexts Fail." 2025.
    https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html
46. Jiang, H., et al. (Microsoft). "LongLLMLingua: Accelerating and Enhancing LLMs in Long
    Context Scenarios via Prompt Compression." ACL, 2024.
    https://aclanthology.org/2024.acl-long.91/ (code: https://github.com/microsoft/LLMLingua)
47. Wu, D., et al. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive
    Memory." ICLR, 2025. https://arxiv.org/abs/2410.10813
48. Vodrahalli, K., et al. (Google DeepMind). "Michelangelo: Long Context Evaluations
    Beyond Haystacks via Latent Structure Queries." 2024. https://arxiv.org/abs/2409.12640
    (see also OpenAI MRCR dataset: https://huggingface.co/datasets/openai/mrcr)
49. Li, K., Zhang, L., Jiang, Y., Xie, P., Huang, F., Wang, S., Cheng, M. "LaRA:
    Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet
    for LC or RAG Routing." ICML, 2025. https://proceedings.mlr.press/v267/li25dv.html
    (arXiv: https://arxiv.org/abs/2502.09977; code: https://github.com/Alibaba-NLP/LaRA)
50. Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., Rashtchian, C. (Google).
    "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems." ICLR, 2025.
    https://arxiv.org/abs/2411.06037
51. Kim, J.-H., Kim, J., Kwon, S., Lee, J. W., Yun, S., Song, H. O. "KVzip: Query-Agnostic
    KV Cache Compression with Context Reconstruction." NeurIPS, 2025 (oral).
    https://neurips.cc/virtual/2025/poster/118741
52. Zhang, Q., et al. (Stanford, SambaNova, UC Berkeley). "Agentic Context Engineering:
    Evolving Contexts for Self-Improving Language Models." ICLR, 2026.
    https://arxiv.org/abs/2510.04618
