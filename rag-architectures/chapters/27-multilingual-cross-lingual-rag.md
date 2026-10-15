# 27. Multilingual and Cross-Lingual RAG: Benchmarks, Embeddings, Language Bias, Tokenization and Sovereign Deployments

## Overview

Most RAG tutorials assume that the query, the corpus and the answer all share one language, usually English. Real deployments rarely work that way. A support bot for a Swiss bank gets questions in German, French, Italian and English and has to answer from legal texts published in all four. A government assistant in India may need to retrieve Hindi, Tamil and English circulars for a question typed in romanized Hindi. An intelligence analyst at a US agency issues English queries against Persian, Russian and Chinese news. Each case breaks assumptions that are built into monolingual pipelines: one tokenizer, one BM25 analyzer, one chunker, one embedding space, and one language for the prompt.

This chapter organizes the design space into six layers:

1. **Benchmarks** that define the task variants: monolingual non-English retrieval (MIRACL), multilingual open QA with language-independent answers (MKQA), cross-lingual open-retrieval QA (XOR-TyDi QA), cross-language and multilingual ad-hoc retrieval plus report generation over news (TREC NeuCLIR and its successor TREC RAGTIME), and multilingual RAG answer-quality arenas (MIRAGE-Bench) [1-9, 33, 34].
2. **Retrieval architecture.** The choice is between translating text into a pivot language (translate-then-retrieve, or translate the documents) and encoding everything into one shared multilingual space with models such as BGE-M3, multilingual-e5, jina-embeddings-v3 and Qwen3-Embedding, built on encoder backbones such as XLM-R or the newer mmBERT and trained on data such as SWIM-IR. A third option, used for neural CLIR, is to distill translation into the retriever itself [10-17, 35, 36].
3. **Language preference and bias.** Retrievers and generators favor some languages and scripts over others, and benchmark artifacts can make English look better than it is [18-23].
4. **Tokenization economics.** The same content can cost several times more tokens in a low-resource language. That multiplies price, latency and context pressure. Byte-level architectures such as BLT are one research response [24, 25, 37].
5. **Script-aware chunking** for CJK, Thai, Arabic and Indic scripts, where whitespace heuristics fail [18, 26-28].
6. **Sovereign and regional-language serving.** Regional models (SEA-LION, Apertus) and regional MT (IndicTrans2) change the build-versus-buy calculation and the data-residency posture [28-32].

A note on sourcing. Most figures in this chapter were checked against project repositories and documentation pages (mainly on GitHub) in October 2026. Paper links to arXiv are given for completeness. Specific numbers are quoted only where a fetched source stated them, and figures that could not be re-verified are marked approximate.

```
             THE MULTILINGUAL RAG DESIGN SPACE
 +-----------+   +-------------------+   +-----------------+   +-------------+
 | Query     |   | Retrieval         |   | Context         |   | Generation  |
 | (lang Lq) |-->| - translate q?    |-->| - which langs?  |-->| - prompt    |
 | script,   |   | - shared space?   |   | - translate     |   |   language  |
 | code-mix  |   | - per-lang index? |   |   passages?     |   | - answer in |
 +-----------+   | - hybrid lexical  |   | - token budget  |   |   Lq?       |
                 +-------------------+   +-----------------+   | - citations |
                          ^                                    +-------------+
                          |
          +-------------------------------+
          | Corpus (langs L1..Ln)         |
          | script-aware chunking,        |
          | per-language analyzers,       |
          | optional doc translation      |
          +-------------------------------+
```

## The benchmark landscape

Benchmarks matter here more than usual. Each one encodes a different assumption about where the knowledge lives and which language the answer should be in. If you pick the wrong one, you can end up optimizing the wrong system.

### MIRACL: monolingual retrieval in 18 languages

**Problem.** Before MIRACL, multilingual retrieval evaluation mostly reused translated English data or had shallow judgments. MIRACL ("Multilingual Information Retrieval Across a Continuum of Languages") offers native-speaker queries with human relevance judgments over each language's own Wikipedia [1, 2].

**Design.** MIRACL covers 18 languages. Corpora and query sets were released for 16 "known" languages, and 2 "surprise" languages were held back for a later reveal. The corpora are passage-segmented Wikipedia dumps in JSONL (`docid`, `title`, `text`). Their sizes vary by more than two orders of magnitude, from about 132K passages for Swahili to about 32.9M for English. The training split has 40,203 queries with 343,177 judgments. The dev split has 13,071 queries with 126,076 judgments. Judgments include explicit negatives, which makes the data useful for hard-negative training [1].

**Baselines.** The official baselines are BM25, mDPR and a BM25+mDPR hybrid, all reproducible with Pyserini. The project also releases mContriever and per-language mDPR checkpoints fine-tuned on MIRACL [1].

**What it measures and what it does not.** MIRACL is *monolingual* retrieval in many languages: Thai queries against Thai Wikipedia. It does not test cross-lingual transfer of knowledge. A system can score well on MIRACL and still fail when the answer exists only in English. Its spin-off NoMIRACL (below) reuses the same judgments to test whether generators abstain when the retrieved passages are all non-relevant [23].

### MKQA: aligned open-domain QA in 26 languages

**Problem.** MKQA asks whether the *same* question gets the *same* answer across languages. It does this by fully aligning 10,000 questions sampled from Google's Natural Questions across 26 typologically diverse languages, for 260,000 question-answer pairs in total [3, 4].

**Design.** Answers are language-independent where possible and are typed: entity (4,221), long answer (1,815), unanswerable (1,427), date (1,174), number (485), number-with-unit (394), short phrase (346) and binary (138). The official evaluation scores exact match and F1, split into answerable and unanswerable questions. Submissions include a `no_answer_prob`. The official macro-average needs prediction files for all 26 languages [3].

**Why it matters for RAG.** MKQA has no in-language corpus. Most answers come from English Wikipedia knowledge, so it naturally tests the cross-lingual setting: a Korean query whose evidence lives in English. That is why multilingual RAG toolkits such as NAVER's BERGEN use MKQA, alongside XOR-TyDi QA, as their multilingual-query testbeds. BERGEN's guide notes that MKQA questions are "grounded in English Wikipedia by design" [18, 19].

### XOR-TyDi QA: cross-lingual open retrieval

**Problem.** Information-seeking questions written by native speakers often have no answer in their own-language Wikipedia, because that Wikipedia is small. XOR QA formalizes "retrieve across languages, answer in the user's language" [5, 6].

**Design.** Questions come from native speakers of 7 typologically diverse languages: Arabic, Bengali, Finnish, Japanese, Korean, Russian and Telugu. There are three sub-tasks [5]:

- **XOR-Retrieve:** find an English passage that answers a target-language question. It is scored with R@2kt and R@5kt (recall within the first 2,000 or 5,000 tokens retrieved), which normalizes for passage length.
- **XOR-EnglishSpan:** output the short answer in English. It is scored with F1, EM and BLEU.
- **XOR-Full:** output the answer in the target language, retrieving from any language. It is scored with F1 and EM.

The release includes about 30k human-annotated question translations for training. The authors ask users to use the 2019-02-01 Wikipedia dump. Baselines combine DPR, Path Retriever or BM25 (Elasticsearch) with multilingual readers [5]. The token-normalized recall metric is worth copying into internal evaluations, because passage lengths differ across languages and segmenters.

### TREC NeuCLIR: neural CLIR, MLIR and multilingual report generation

**Problem.** Academic QA benchmarks use Wikipedia. Analysts work with news and technical literature in languages they cannot read. NeuCLIR is a TREC shared task that measures how neural methods change cross-language retrieval and, since 2024, generation [7].

**Design.** NeuCLIR Collection 1 consists of Common Crawl-derived news in Chinese, Persian and Russian. The Chinese portion alone is about 3.18M documents (the download tooling validates a line count of 3,179,209). Access requires TREC participant credentials [8]. By its third edition in 2024 the track had four tasks [7]:

- **CLIR:** English queries retrieve Chinese, Persian or Russian news.
- **MLIR:** English queries retrieve from the *combined* three-language collection and produce one ranking. Topics are designed to have relevant documents in more than one language.
- **Technical documents:** English queries against Chinese technical abstracts (the `csl` collection). This was a pilot in 2023 and a full task in 2024.
- **Report generation (pilot):** systems write English reports with citations to the foreign-language news. Reports are judged on the information they include and on whether citations are appropriate.

Collections are exposed through `ir_datasets` as `neuclir/1`, `neuclir/1/multi` and `csl` [7]. The track overview papers give per-run results [9].

**Why it matters for RAG.** The NeuCLIR report-generation task is effectively a benchmark for *cross-lingual, citation-grounded long-form RAG*. The answer is in English while all of the evidence is in other languages. The MLIR task also exposes a problem that monolingual systems never face: **score calibration across languages**. A single ranked list mixes Persian and Chinese documents whose retrieval scores may not be comparable.

### TREC RAGTIME: the successor to NeuCLIR (2025-2026)

**Problem.** NeuCLIR's 2024 report-generation pilot showed that cross-lingual, citation-grounded report writing deserved its own track. TREC RAGTIME is that track; its website describes NeuCLIR as its predecessor [33].

**Design.** RAGTIME benchmarks **multilingual report generation**: given a "report request" and a collection of news in several languages, systems write an English report whose claims carry citations to supporting documents. The emphasis is on *multi-faceted reports* rather than factoid QA, with citation-based evaluation [33].

- **2025 (first edition):** documents in English, Arabic, Chinese and Russian. Tasks were multilingual report generation, monolingual English report generation (English documents only) and CLIR/MLIR retrieval as supporting subtasks [33].
- **2026 (second edition):** the language set changed to English, Spanish, Chinese and Russian. The track runs multilingual report generation and multilingual retrieval, adds an **auto-nuggetization** task (decompose a report request into single-sentence questions a report should answer), and offers a hosted search service to registered participants [33].

**Why it matters for RAG.** The monolingual-English versus multilingual report-generation split in 2025 is a clean controlled experiment: does adding non-English evidence improve or hurt a report? Auto-nuggetization turns the "what should the answer cover?" step, usually hidden inside evaluation, into a first-class planning component, which is useful for long-form multilingual RAG agents.

### MIRAGE-Bench: an arena for multilingual RAG generation

Most of the benchmarks above score retrieval or short answers. MIRAGE-Bench (Thakur et al., NAACL 2025) targets the *generation* step of multilingual RAG across 18 languages [34]. It scores answers with heuristic features (for example support and fluency), runs pairwise LLM-as-a-judge comparisons (GPT-4o judgments across 19 models are released) and fits a Bradley-Terry model with bootstrapping to build an offline multilingual RAG leaderboard. It then trains a cheap **surrogate judge** (a regression over heuristic features) to approximate the expensive LLM judge [34]. The design point is reusable in production: an expensive judge calibrates a cheap per-language scorer that can run on every release.

### Benchmark comparison

| Benchmark | Query langs | Evidence langs | Corpus | Output | Primary metric | Tests |
|---|---|---|---|---|---|---|
| MIRACL [1] | 18 (16 released) | Same as query | Per-language Wikipedia passages, 132K-32.9M | Ranked passages | nDCG@10, recall | Monolingual non-English retrieval |
| NoMIRACL [23] | ~18 | Same as query | MIRACL subsets | Answer or abstain | Hallucination rate, error rate | Robustness to non-relevant context |
| MKQA [3] | 26 | Language-agnostic (mostly English knowledge) | None prescribed | Short typed answers | EM / F1 | Answer consistency across languages |
| XOR-TyDi QA [5] | 7 + English | English or any | Wikipedia 2019-02-01 | Passages, English span, or native answer | R@2kt/5kt, F1, EM | Cross-lingual open retrieval |
| NeuCLIR [7] | English | zh, fa, ru (+ technical zh) | Common Crawl news, CSL abstracts | Rankings and cited English reports | nDCG; nugget/citation judgments | CLIR, MLIR, cross-lingual report generation |
| TREC RAGTIME [33] | English report requests | en, ar, zh, ru (2025); en, es, zh, ru (2026) | Multilingual news | Cited English reports, rankings, nugget lists | Nugget- and citation-based judgments | Multilingual report generation, MLIR |
| MIRAGE-Bench [34] | 18 | Same as query | MIRACL-derived | Generated answers | Heuristic features, pairwise LLM judge, Bradley-Terry | Multilingual RAG answer quality |
| MMTEB [17] | Hundreds | Varies | Many tasks | Embeddings | Task means | Broad embedding quality |

## Architectural choices: translate versus shared multilingual space

There are three main architectures for cross-lingual retrieval and one hybrid that most production teams end up with.

```
 (A) TRANSLATE-THEN-RETRIEVE (query translation, "English pivot")
     q(Lq) --MT--> q(en) --> [English index] --> passages(en) --> LLM --> answer(Lq)

 (B) DOCUMENT TRANSLATION (translate corpus offline)
     docs(L1..Ln) --MT offline--> docs(en) --> [English index]
     q(Lq) --MT--> q(en) --------------^      (keep originals for citation)

 (C) SHARED MULTILINGUAL SPACE
     q(Lq) --enc--> v_q --+
                          +--> [one index over docs(L1..Ln) encoded by same model]
     docs(Li) --enc--> v_d+       --> passages(mixed langs) --> LLM --> answer(Lq)

 (D) DISTILLED NEURAL CLIR (ColBERT-X / Translate-Distill)
     teacher: MT + English cross-encoder scores (q_en, d_en)
     student: multilingual late-interaction encoder scores (q_en, d_Li) directly
```

### (A) Translate-then-retrieve

**Approach.** Machine-translate the query into the corpus language, or into a pivot language (usually English), then use a strong monolingual retriever. BERGEN's multilingual configuration lets you add NLLB-3.3B query translation in front of retrieval for exactly this reason [18].

**Strengths.** It reuses the best monolingual retrievers, rerankers and BM25 analyzers. It is easy to debug because everything is in one language. It works well when the knowledge really lives in English, as in MKQA and XOR-EnglishSpan.

**Failure modes.**
- **Entity and transliteration damage.** MT can mangle named entities, for example by rendering a Korean person's name in a romanization that does not match the English Wikipedia spelling. This is why BERGEN evaluates with character 3-gram recall instead of word-level match: it tolerates spelling variation in named entities [18].
- **Lost cultural specificity.** The English pivot under-retrieves locally relevant sources. A question about a regional tax rule should hit the regional-language corpus.
- **Latency and cost.** MT adds a model call per query, or per passage if you translate retrieved context back.
- **Benchmark-inflated benefit.** Park et al. (2026) argue that much of the apparent advantage of "English pivoting" comes from structural biases in benchmarks: exposure bias, gold-answer availability priors and cultural priors. After debiasing, retrievers prefer *monolingual alignment*, meaning query and document in the same language, over English as such [21].

### (B) Document translation

Translating the whole corpus offline into a pivot language is expensive at ingest but cheap at query time. NeuCLIR-style evaluations historically supplied machine translations of the collection so that teams could build English-side systems. The ColBERT-X training recipe "Translate-Distill" uses translated passages from neuMARCO (MS MARCO translated into Chinese, Persian and Russian) [10]. Keep the original text alongside the translation for citation and display. Users and auditors need to see the source, and LLM generators can often read the original directly.

### (C) Shared multilingual embedding space

**Approach.** One encoder maps queries and passages from all languages into one vector space. A single ANN index then serves every language pair. This is now the default for new builds, because modern multilingual embedders are strong enough that, for many language pairs, the translation step no longer pays for itself.

#### BGE-M3 (BAAI, 2024)

BGE-M3's name stands for its three "M"s: multi-linguality (more than 100 working languages), multi-granularity (inputs up to 8,192 tokens) and multi-functionality (dense, sparse and multi-vector retrieval from a single model) [11, 12].

- **Outputs.** A 1,024-dimensional dense vector; per-token lexical weights (a learned sparse representation that can replace BM25); and ColBERT-style multi-vector token embeddings.
- **Training.** *Self-knowledge distillation* uses the combined score of the three retrieval modes as a teacher signal to improve each single mode, especially sparse and multi-vector. Efficient batching makes long-text fine-tuning practical. MCLS improves long-document performance without extra fine-tuning. The release model is a unified fine-tune from `bge-m3-unsupervised` [11].
- **Recommended pipeline.** Hybrid retrieval (dense plus sparse, or embedding plus BM25) followed by a cross-encoder reranker such as bge-reranker. The authors state that rerankers are more accurate than bi-encoders and that **BM25 remains a competitive baseline, especially for long documents** [11]. Reproduction setups in practice use BGE-M3 for both the first stage and a second-stage cross-encoder reranker. BERGEN does exactly this and ships prebuilt BGE-M3 indexes for 12 Wikipedia languages [18].
- **Benchmarks.** The authors report results on MIRACL, MKQA, MLDR (their multilingual long-document retrieval set) and NarrativeQA. The GitHub page shows them only as images, so see the paper for numbers [12].

```
          BGE-M3 ONE-PASS, THREE-SIGNAL RETRIEVAL
  text --> XLM-R-style encoder (8192 ctx) --> hidden states H
              |                 |                       |
       [CLS] -> dense      token weights -> sparse   all tokens -> multi-vec
       1024-d ANN index    inverted index           late interaction (rerank)
              \_________________|_______________________/
                     weighted score fusion --> cross-encoder rerank
```

**Lesson.** The sparse head matters most in multilingual settings. Lexical matching is what rescues rare entities, codes and transliterated names, which dense vectors blur. Yet traditional BM25 needs a language-specific analyzer: stemmer, segmenter, stopwords. A learned sparse head inherits the model's multilingual tokenizer and needs no per-language analyzer.

#### multilingual-e5 (Microsoft, 2024)

The mE5 family comes in small (12 layers, 384-d), base (12 layers, 768-d), large (24 layers, 1,024-d) and large-instruct (24 layers, 1,024-d) sizes. Reported BEIR averages are 46.6, 48.9, 51.4 and 52.5 respectively [13, 14]. The models are trained with the E5 recipe of weakly supervised contrastive pre-training on multilingual text pairs followed by supervised fine-tuning, as described in the technical report [14]. Operationally, the non-instruct models expect asymmetric `query: ` and `passage: ` prefixes, and the instruct variant takes a task instruction on the query side. Forgetting the prefixes is a common silent quality bug (per the model cards; the GitHub page does not restate this [13]). The small model's 384-d vectors make it attractive for cost-sensitive, many-language indexes.

#### jina-embeddings-v3 (Jina AI, 2024)

jina-embeddings-v3 is a multilingual model of roughly 570M parameters built on an XLM-RoBERTa backbone with rotary position embeddings for an 8,192-token context. It adds **task-specific LoRA adapters** selected at inference time: retrieval-query, retrieval-passage, separation (clustering), classification and text-matching. Matryoshka representation learning lets you truncate the default 1,024-d output to much smaller sizes (down to 32-d) with graceful degradation [15]. Treat parameter counts and language counts as approximate; they come from the technical report, which could not be re-fetched for this chapter. Note also that the open weights were released under a non-commercial (CC BY-NC 4.0) license, with commercial use through Jina's API or a separate license; check the current model card before building on it. Asymmetric adapters for query and passage play the same role as mE5's prefixes. Matryoshka truncation matters for multilingual indexes, which tend to be large because they hold many languages' corpora.

#### Qwen3-Embedding (Alibaba, 2025)

Qwen3-Embedding is the LLM-backbone generation of multilingual embedders. It comes in 0.6B, 4B and 8B sizes (28, 36 and 36 layers), all with 32K sequence length. Embedding dimensions are 1,024, 2,560 and 4,096, with MRL support and instruction-awareness. The series claims "over 100 languages" inherited from Qwen3 [16]. On the MTEB multilingual leaderboard the mean task scores are 64.33, 69.45 and 70.58, and the 8B model was listed as number one as of June 5, 2025. Matching Qwen3-Reranker models score 66.36, 72.74 and 72.94 on MMTEB-R, the retrieval subset of MMTEB used for reranker evaluation [16]. The trade-off is cost: a 4,096-d vector from an 8B encoder is about an order of magnitude more expensive to compute and store than mE5-small or a truncated jina-v3 vector.

#### Backbones and training data for the long tail: mmBERT and SWIM-IR

Most multilingual embedders above sit on an XLM-R-era encoder (512-token native context, 2019 data) or on an LLM. **mmBERT** (JHU CLSP, 2025) is a modern ModernBERT-style multilingual encoder trained on 3T tokens across 1,833 languages. It uses *annealed language learning*: pretraining covers 60 languages (2.0T tokens), mid-training 110 languages (600B), and the decay phase all 1,833 languages (100B). The models come in small (140M) and base (307M) sizes with an 8,192-token context, and the authors report that they outperform XLM-R [36]. mmBERT is an encoder, not a ready-made retriever; it is a backbone for fine-tuning embedders and rerankers for languages that existing models barely cover.

The other bottleneck is training data. **SWIM-IR** (Thakur et al., Google, NAACL 2024) is a synthetic multilingual retrieval training set of 28 million query-passage pairs in 33 languages. The queries are generated by PaLM-2 from Wikipedia passages with *summarize-then-ask prompting*. Models trained on it perform well on XOR-Retrieve and MIRACL, and the authors report state-of-the-art results on XTREME-UP for under-represented languages [35]. The recipe (LLM-generated queries over native-language passages) is how most teams can bootstrap in-domain multilingual fine-tuning data when human judgments are unavailable.

#### Benchmarks for choosing an embedder

MMTEB (Enevoldsen et al., 2025) greatly expanded MTEB into a massive multilingual benchmark and is now the reference leaderboard for multilingual embedder selection [17]. Leaderboard means can still hide per-language gaps. Always evaluate on your own languages, ideally on MIRACL splits for those languages, plus an in-domain set.

### (D) Distilled neural CLIR: ColBERT-X, Translate-Distill and PLAID-X

The Johns Hopkins HLTCOE line of work, developed largely around NeuCLIR, takes a middle path [10]:

- **ColBERT-X** (Nair et al., ECIR 2022) is a cross-language version of ColBERT. It uses a multilingual encoder for late interaction and is trained by *translate-train*: English MS MARCO queries paired with machine-translated passages.
- **Translate-Distill** (Yang et al., ECIR 2024) distills from a translate-then-rerank teacher. The teacher scores (English query, translated passage) pairs with a strong English cross-encoder, and the student learns to reproduce those scores directly on (English query, native-language passage). The translation cost is paid once at training time, never at query time.
- **Multilingual Translate-Distill** (Yang et al., SIGIR 2024) extends this to MLIR by mixing multiple languages' training data. Mixing strategies are `entries`, `passages` or `round-robin`. This targets the cross-language score calibration problem that NeuCLIR's MLIR task exposes.
- **PLAID-X** generalizes the PLAID compressed late-interaction engine to CLIR. The HLTCOE repository that hosts ColBERT-X and PLAID-X was archived on May 28, 2026; its README says it is no longer maintained and recommends PyLate for new work [10].

**Lesson.** If you have query-time latency constraints and a fixed language set, distillation folds MT into the retriever. You get most of the translate-pipeline quality with single-encoder cost.

### Choosing among architectures

| Criterion | Translate query (A) | Translate corpus (B) | Shared multilingual space (C) | Distilled CLIR (D) |
|---|---|---|---|---|
| Query-time cost | +1 MT call per query | +1 MT call per query (if query not in pivot) | Single encode | Single encode |
| Ingest cost | None | Very high (MT entire corpus) | Encode once | Encode once |
| Entity fidelity | Weak (MT errors) | Medium | Good with sparse/lexical head | Good |
| Local-source recall | Weak (pivot bias) | Medium | Good if monolingual alignment preserved | Good |
| Debuggability | High | High | Medium | Low |
| New language | Needs MT pair | Needs MT + re-ingest | Free if model covers it | Needs retraining |
| Typical tools | NLLB, IndicTrans2 [18, 28] | NLLB, commercial MT | BGE-M3, mE5, jina-v3, Qwen3-Emb [11-16] | ColBERT-X, PLAID-X [10] |

## Language preference and bias in retrieval and generation

Multilingual RAG adds a failure surface that monolingual RAG lacks: the system can be *right about relevance but biased about language*. Four lines of 2024-2026 work describe it.

### Retriever and generator preference (Park and Lee, ACL Findings 2025)

"Investigating Language Preference of Multilingual RAG Systems" measures preference separately for retrievers and generators [20]:

- **Retrievers** prefer high-resource languages and the query's own language. That preference does not consistently translate into better generation quality.
- **Generators** prefer the query language or **Latin-script** text, which leads to inconsistent outputs when evidence is spread across scripts.
- **Mitigation (DKM-RAG, Dual Knowledge Multilingual RAG):** fuse *translated* multilingual passages with the generator's complementary parametric knowledge. The authors report reduced language preference in generation and better performance across language settings. The released code includes an `mlrs.py` script that measures retriever language preference from initial rankings [20].

### Debiasing the "English is best" result (Park et al., 2026)

The follow-up DELTA paper changes the earlier conclusion [21]:

- It introduces **DeLP** (Debiased Language Preference), a calibrated metric that removes exposure bias, gold-availability priors and cultural priors to estimate a language's *intrinsic* preference.
- After debiasing, **the apparent English preference largely disappears**. Retrievers instead favor *monolingual alignment*, where query and document are in the same language.
- **DELTA** (DEbiased Language preference-guided Text Augmentation) is a lightweight, query-level framework that uses monolingual alignment, augmenting or fusing query variants so that each document language is matched by a same-language query. Across seven languages (ar, de, es, ja, ko, th, zh) and three generators (Qwen3-235B, Gemini-2.5-Flash, DeepSeek-v3.1), it beats English pivoting. On MKQA with BGE-M3 retrieval and character 3-gram recall, it averages about 62.9 with Qwen3-235B, versus 58.8 for English translation and 51.3 for a MultiRAG baseline [21].

```
          DELTA-STYLE QUERY FUSION (monolingual alignment)
  q(ko) --+-- as is ------------------> search(ko docs)  --+
          +-- MT -> q(en) ------------> search(en docs)  --+--> fuse/rerank --> LLM
          +-- MT -> q(ja), q(zh) ... -> search(ja,zh docs)-+
          (each language partition is queried in its own language)
```

**Design implication.** Instead of "translate everything to English", translate the *query* into each corpus language and search each partition in its own language. That is cheap (queries are short), and it exploits the monolingual alignment that retrievers actually prefer.

### Citation-level "linguistic nepotism" (Ki et al., ICML 2026 Spotlight)

"Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG" asks whether generators *cite* documents differently depending on their language when relevance is held fixed [22]. The method uses model internals. It holds document content constant by machine-translating the human-picked relevant documents of 270 ELI5 queries (with Google Translate, and COMET-QE to check translation quality). It then generates verified, citation-supported reference reports and detects preference as a drop in next-token accuracy for the correct citation ID when only the cited document's language changes. The released code also includes logit-lens, ContextCite, query-language and relevance-language analyses. The authors are from UMD and JHU, the NeuCLIR community. The title states the finding: models trade citation quality for language preference. For RAG systems that show citations, this means the *citation* step needs its own fairness audit, separate from retrieval.

### Robustness: knowing when you don't know (NoMIRACL)

NoMIRACL builds two subsets from MIRACL across about 18 languages. In the *non-relevant* subset, every top-k passage is judged non-relevant. In the *relevant* subset, at least one passage is relevant. It reports a **hallucination rate** FP/(FP+TN), meaning the model answers when it should abstain, and an **error rate** FN/(FN+TP), meaning the model abstains when it could answer [23]. Baseline outputs for models such as GPT-4, Llama-2-13B-chat and Aya-101 are published at 250 queries per language. This is the standard multilingual test for whether your generator's abstention behavior holds up outside English.

### Language fidelity of the output

A practical, unglamorous failure is answering in the wrong language. English-centric generators often drift into English when the context is English. NAVER's BERGEN tracks a **correct language rate**, with responses classified against Flores language codes, as an auxiliary metric [18]. It found that a *translated, language-specific prompt* (`basic_translated_langspec/{lang}`) gives the highest correct-language rate. Even with good prompts, English-centric models vary by language: SOLAR handled French well but struggled with Korean, while Command-R-35B served as the main multilingual generator [18]. Chirkova et al. report the broader study behind the toolkit [19].

**Lessons on bias:**
- Measure retrieval language mix, citation language mix and output language as first-class metrics.
- Don't conclude "English pivot is better" from a single benchmark, because benchmark priors inflate it [21].
- Per-language query fusion with monolingual alignment is a cheap, well-supported default [21].
- Write the system prompt *in the target language* and explicitly ask for answer and entity names in that language [18].

## Tokenization costs for low-resource languages

### The tokenization premium

Petrov et al. (NeurIPS 2023) showed that translations of the same text can differ in tokenized length by **up to 15x** across languages. The disparity persists across 17 tokenizers, including ones designed for multilingual use, and byte- or character-level models still show gaps above 4x for some language pairs [24]. The disparity arises *at tokenization*, before the model runs, and it carries through to:

- **Price.** Per-token API pricing charges speakers of some languages several times more for the same content. Ahia et al. (2023) make the same economic argument in "Do All Languages Cost the Same?" [25].
- **Latency.** More tokens mean more prefill and decode time.
- **Effective context.** A 128K context holds several times less Telugu or Amharic content than English content.

### Consequences specific to RAG

```
  Same 5 passages, same information, different language
  English    [#####                ]  ~1.0x tokens
  Mid-res    [##########           ]  ~2x   (approximate, varies by tokenizer)
  Low-res    [####################+]  up to ~4-15x  [24]
  -> fewer passages fit in the budget, higher prefill cost, slower TTFT
```

1. **Chunk sizes defined in tokens are not comparable across languages.** A 512-token chunk in a high-premium language may carry a fraction of the information of a 512-token English chunk. This interacts with embedder context limits; mE5 has a 512-token limit, for example.
2. **Top-k must be budgeted by tokens, not by count.** Use a token-budgeted context packer per language, or rerank and truncate harder for high-premium languages.
3. **Translate-to-English context can be a cost optimization.** It is not always a quality one. Translating retrieved passages into a low-premium language before generation reduces generator tokens but adds MT cost and error. Given the DELTA findings, do this for cost reasons, not on the assumption that it improves quality [21].
4. **Embedding cost has the same premium.** Encoders with multilingual SentencePiece vocabularies (XLM-R lineage: BGE-M3, mE5, jina-v3) are fairer than English-centric BPE but not perfectly fair [24].
5. **Regional models attack the premium at the root** by training tokenizers with more regional-script vocabulary. That is one motivation for sovereign models (below).

### Tokenizer-free models: the Byte Latent Transformer

A structural fix for the premium is to drop the fixed subword vocabulary. Meta's **Byte Latent Transformer** (BLT, 2024) encodes raw bytes into dynamically sized *patches*, segmented by the entropy of the next byte, so compute goes where the data is complex. The authors report the first scaling study of byte-level models up to 8B parameters and 8T training bytes, matching tokenization-based LLM performance with better inference efficiency and robustness and with improvements on long-tail generalization [37]. Petrov et al. caution that byte- and character-level encodings still differ across languages by more than 4x for some pairs, since UTF-8 spends more bytes on many non-Latin scripts [24]. Dynamic patching may narrow this gap, but there is not yet a production RAG stack built on BLT-style generators. Treat it as a direction to watch, not a deployment option.

**Practical guidance.** Measure your own premium. Take a parallel sample (FLORES-200, which Petrov et al. use [24]) and tokenize it with your generator's and embedder's tokenizers. Record the ratio per language and use it to set per-language chunk sizes, top-k and cost forecasts.

## Script-specific chunking

Chunkers inherited from English pipelines assume that whitespace separates words and that `.`, `?` and `!` end sentences. Neither holds widely across the world's scripts.

### The default-splitter trap

LangChain's `RecursiveCharacterTextSplitter` falls back to four separators: paragraph break, newline, space, and finally the empty string (character-level split). Its built-in language-specific separator lists cover *programming* and markup languages (Python, Go, Markdown, HTML, LaTeX and so on), not natural-language scripts [26]. Chinese, Japanese and Thai text has no spaces between words. Once a paragraph is longer than the chunk size, the splitter falls through to **arbitrary character-level cuts mid-sentence or mid-word**. You need to add script-aware separators, such as CJK sentence punctuation, or better, run a real sentence segmenter first.

### CJK (Chinese, Japanese, Korean)

- **Segment by sentence punctuation:** the ideographic full stop and the fullwidth question and exclamation marks, with paragraph breaks as the higher-level boundary. Use the ideographic and fullwidth commas only as a last resort.
- **Size chunks in characters, not words.** BERGEN splits multilingual Wikipedia into 100-word chunks but uses **100 Unicode characters** for Chinese, Japanese and Thai, because "word" is undefined without segmentation [18].
- **BM25 needs a word segmenter or character n-grams.** For Chinese and Japanese, whitespace tokenization produces one giant "token" per sentence. Use a morphological analyzer or CJK bigrams, or rely on a learned sparse head such as BGE-M3's, which uses the model's subword vocabulary [11].
- **Korean** uses spaces between eojeol, but those are agglutinative units, so lexical search benefits from morphological analysis.

### Thai, Lao, Khmer, Burmese

These scripts have no inter-word spaces and often no sentence-final punctuation; spaces may mark phrase or sentence boundaries instead. Dictionary or model-based word segmentation, as in the PyThaiNLP toolkit for Thai, is a prerequisite for both lexical indexing and sensible chunk boundaries [27]. BERGEN treats Thai like CJK, with character-based chunk sizing [18].

### Arabic and Perso-Arabic scripts

- **Right-to-left text with bidirectional runs.** Numbers, Latin-script terms and URLs inside Arabic text cause display reordering. Chunk on logical (storage) order, never on visual order, and never "fix" bidi by reversing strings in preprocessing.
- **Orthographic normalization for lexical search:** alef variants, ta marbuta and ha, alef maqsura and ya, optional diacritics (tashkeel), and tatweel (kashida) elongation. Without normalization, BM25 misses matches that native readers see as identical. Apply the *same* normalization at index and query time.
- **Rich morphology.** Clitics attach to words (conjunctions, prepositions, articles), so light stemming or subword-aware sparse retrieval helps.
- **Persian** (a NeuCLIR language [7]) uses the zero-width non-joiner inside words. Splitters and normalizers must not treat it as a boundary or strip it blindly.

### Indic scripts

- **Grapheme clusters.** A visible character (akshara) in Devanagari, Tamil, Telugu, Bengali and others is often a cluster of a consonant, virama and dependent vowel signs. A character-level cut can split a cluster and produce invalid text. Split on extended grapheme cluster boundaries (Unicode text segmentation) at a minimum, and preferably at sentence boundaries. The danda is the sentence terminator in Hindi and several other languages.
- **Multiple scripts per language and script unification.** IndicTrans2 covers all 22 scheduled Indian languages plus English. Several languages are handled in more than one script (Kashmiri, Sindhi, Manipuri), and the model applies script unification where feasible across five scripts: Perso-Arabic, Devanagari, Ol Chiki, Meitei and Latin [28]. A retrieval system for the region must decide whether to transliterate to a canonical script before indexing.
- **Romanized and code-mixed queries.** Users often type Hindi in Latin script mixed with English words. Native-script documents will not lexically match. Transliteration at query time, or a multilingual dense retriever robust to romanization, is required. Test romanized inputs explicitly.

### Script-aware chunking pipeline

```
 raw doc --> Unicode NFC normalize --> language/script ID (per paragraph)
         --> per-script sentence segmenter
               CJK: punctuation rules     Thai: dictionary/model segmenter [27]
               Arabic/Persian: punctuation + ZWNJ-safe normalization
               Indic: danda + grapheme-cluster-safe boundaries
         --> pack sentences into chunks sized in TOKENS of the embedder,
             with per-language targets derived from the measured premium [24]
         --> attach title/heading prefix (BERGEN prefixes article title) [18]
         --> store: original text, lang tag, script tag, optional translation
```

Two cross-cutting lessons. First, **tag each chunk with language and script** at ingest. Every later stage needs those tags: per-language query fusion [21], per-language analyzers, output-language checks [18] and bias audits [20, 22]. Second, **do language identification per paragraph, not per document.** Mixed-language documents are common (bilingual legal texts, code-mixed forums, English tables inside Arabic reports).

## Serving sovereign and regional-language deployments

"Sovereign AI" deployments are driven by three pressures at once: language coverage that global models serve poorly, data-residency and legal requirements, and the tokenization and cultural gaps described above.

### Regional model families

**SEA-LION (AI Singapore).** An open model family for Southeast Asian languages, explicitly aimed at "under-represented population groups and low resource languages in the SEA region" [29]. The family spans generations: v1 (3B and 7B), v2 (8B), v3 (8B, 9B, 70B), v3.5 (8B and 70B reasoning), v4 (4B to 32B, including VLMs), v4.5 (E2B and 27B), and v4.8 (30B-A3B and 120B-A12B mixture-of-experts with 256K context, covering 13 regional languages including Burmese and Malay plus key sub-national languages). For RAG specifically, the project ships **SEA-LION-Embedding** (300M and 600M), which it reports as setting records on 10 regional languages, and **SEA-GUARD** safety models (4B, 8B, 12B). Its evaluation suite **SEA-HELM** scores English performance, SEA-language chat, instruction following and linguistic tasks [29]. Licensing generally follows MIT but inherits base-model terms (Llama 3 or Gemma) for some variants, so check each model card [29]. The family also shows sovereign stacks being composed across borders. SEA-LION v4 includes an 8B variant fine-tuned from Swiss **Apertus-8B-Instruct-2509**, and v4.8 is continued pretraining of NVIDIA Nemotron-3 bases. The SEA-LION-Embedding models were released in March 2026 and evaluated on SEA-BED, a Southeast Asian embedding benchmark built from human-curated native data rather than machine translation [29].

**Apertus (Swiss AI Initiative).** Fully open 8B and 70B models trained on 15T tokens covering more than 1,800 languages, with roughly 40% non-English pretraining data. They are trained "exclusively on openly available data", with retroactive honoring of content-owner opt-outs, filtering of non-permissive, toxic and PII content, and the Goldfish objective to reduce verbatim memorization [30]. Development artifacts (data scripts, checkpoints, evaluation suites, training code) are released under a permissive license [30]. Low-resource national languages get dedicated pipelines. For example, the Swiss AI Initiative published scripts for processing Romansh (Rumantsch) data for Apertus v1 [32].

**IndicTrans2 (AI4Bharat).** A regional MT backbone rather than an LLM, and very useful as the translation component in architectures (A), (B) and DELTA-style fusion. It covers all 22 scheduled Indian languages, including En-Indic, Indic-En and Indic-Indic directions. 1B base models and 200M or 320M distilled models are available, along with RoPE variants for 2,048-token inputs, CTranslate2 inference and LoRA fine-tuning scripts [28]. The distilled 200M models are small enough to run on the same GPU as the retriever, which removes the latency argument against query translation.

### Reference architecture: a sovereign multilingual RAG stack

```
                    REGION / JURISDICTION BOUNDARY (data residency)
 +-------------------------------------------------------------------------+
 |  Ingest                                                                 |
 |   docs --> NFC + lang/script ID --> script-aware chunker --> chunks     |
 |              |                                              (lang tags) |
 |              +--> optional MT to pivot (IndicTrans2 / NLLB) [18,28]      |
 |                                                                         |
 |  Index                                                                  |
 |   multilingual embedder (BGE-M3 / mE5 / regional embedder) [11,13,29]   |
 |   + per-language lexical or learned-sparse index                        |
 |   partitions by language for monolingual-alignment search [21]          |
 |                                                                         |
 |  Query                                                                  |
 |   q --> lang ID --> query variants per corpus language (MT, short) [21] |
 |     --> hybrid search per partition --> cross-lingual reranker          |
 |     --> token-budgeted packer (per-language premium) [24]               |
 |                                                                         |
 |  Generate                                                               |
 |   regional/open LLM (SEA-LION / Apertus / ...) [29,30]                  |
 |   prompt written in target language; answer + entities in Lq [18]      |
 |   abstain policy tested on NoMIRACL-style data [23]                     |
 |                                                                         |
 |  Evaluate                                                               |
 |   per-language: retrieval nDCG, char-3gram recall, correct-language     |
 |   rate, citation-language mix, hallucination/error rate [18,20,22,23]   |
 +-------------------------------------------------------------------------+
```

### A worked example: multilingual Swiss legal RAG

CertusAI, a Swiss AI Weeks 2025 hackathon project that won "Best Usage of Apertus", shows the minimum viable pattern [31]. It answers questions about Swiss federal law using Fedlex legal texts, with a RAG pipeline that combines semantic search with BM25 keyword retrieval. Orchestration uses DSPy, Apertus does generation, and answers are targeted in German, French, Italian and English. The README claims Swiss data-residency compliance but does not document how it is achieved, and it gives only local development instructions [31]. The architecture is typical of sovereign prototypes: a domestic open model, a domestic official corpus, a hybrid lexical-plus-dense retriever, and multiple official languages. It also shows where such prototypes are typically thin: residency is asserted rather than engineered, embedding choice and per-language evaluation are not documented, and indexing and refresh are unspecified.

### Serving considerations

- **Model placement.** Sovereign requirements often rule out cross-border API calls for embedding *and* generation. Budget GPUs for an embedder, a reranker, an MT model and a generator, and consider small distilled MT models [28] and Matryoshka-truncated embeddings [15, 16] to fit within domestic capacity.
- **Multiple languages, one index or many.** A single shared-space index is operationally simple. Language partitions (or a language filter field) enable DELTA-style per-language queries and per-language analyzers [21]. Most teams use one physical index with a language metadata filter.
- **Cross-language score calibration.** When merging results from per-language partitions, raw scores are not comparable. Use reciprocal-rank fusion or a single cross-lingual reranker over the union. NeuCLIR's MLIR task exists precisely because of this problem [7, 10].
- **Evaluation localization.** Generic leaderboards (MMTEB [17]) and regional ones (SEA-HELM [29]) should feed model selection, but the deciding evaluation should be an in-domain set in each official language with native-speaker judgments, following MIRACL's methodology [1].
- **Safety localization.** Safety classifiers trained mostly on English miss regional harms. SEA-LION ships dedicated SEA-GUARD models for this reason [29].
- **Licensing heterogeneity.** Regional families often inherit base-model licenses that vary across variants [29]. Track licensing per artifact, including embedders and rerankers.

## Key takeaways for system designers

1. **Pick the benchmark that matches your knowledge topology.** In-language corpora point to MIRACL. Answers living in English for non-English queries point to MKQA and XOR-TyDi. English users reading foreign-language evidence, or a single ranking over mixed languages, points to NeuCLIR CLIR, MLIR and report generation, and to its successor TREC RAGTIME for long-form multilingual reports. For generation quality across languages, MIRAGE-Bench's judge-plus-surrogate design is a practical template [1, 3, 5, 7, 33, 34].
2. **Default to a shared multilingual embedder with a lexical or sparse companion,** then rerank. BGE-M3's dense, sparse and multi-vector heads in one pass are a strong open baseline, and BM25 is still competitive for long documents [11].
3. **Don't assume an English pivot.** Debiased analysis shows that retrievers prefer monolingual alignment. Translate the short *query* into each corpus language and fuse, rather than translating the corpus to English [21].
4. **Treat language as a measured dimension.** Track retrieved-language mix, citation-language mix, correct-output-language rate and per-language hallucination or abstention rates [18, 20, 22, 23].
5. **Prompt in the target language** and explicitly request answers and entity names in that language. It measurably improves output-language fidelity [18].
6. **Budget by tokens per language.** Measure your tokenizer premium on parallel data. Gaps of up to 15x are documented and directly affect cost, latency and how many passages fit [24, 25].
7. **Chunk by script.** Never let whitespace-based splitters cut CJK, Thai or Indic text at arbitrary characters [26]. Size chunks in characters for unsegmented scripts [18], segment Thai properly [27], normalize Arabic consistently, and respect grapheme clusters in Indic scripts.
8. **Tag every chunk with language and script at ingest.** Fusion, filtering, analyzers and audits all depend on those tags.
9. **For latency-critical CLIR with fixed languages, consider distillation** (Translate-Distill or ColBERT-X style). It moves MT cost from query time to training time [10].
10. **Sovereign stacks are composable.** A regional generator (SEA-LION, Apertus), a regional MT model (IndicTrans2), a multilingual or regional embedder and a localized safety model can all run within a jurisdiction. Engineer and document residency explicitly; don't just assert it [28-31].

## References

1. MIRACL: Multilingual Information Retrieval Across a Continuum of Languages (GitHub project page). Zhang et al., University of Waterloo / Huawei. 2022-2023. https://github.com/project-miracl/miracl
2. MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages. Xinyu Zhang et al. TACL 2023. https://arxiv.org/abs/2210.09984
3. MKQA: Multilingual Knowledge Questions and Answers (dataset repository). Apple. 2020. https://github.com/apple/ml-mkqa
4. MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering. Shayne Longpre, Yi Lu, Joachim Daiber (Apple). 2020. https://arxiv.org/abs/2007.15207
5. XOR QA / XOR-TyDi QA repository. Akari Asai et al. 2021. https://github.com/AkariAsai/XORQA
6. XOR QA: Cross-lingual Open-Retrieval Question Answering. Akari Asai, Jungo Kasai, Jonathan H. Clark, Kenton Lee, Eunsol Choi, Hannaneh Hajishirzi. NAACL 2021. https://arxiv.org/abs/2010.11856
7. TREC NeuCLIR Track website (2024 edition: CLIR, MLIR, technical documents, report generation). NeuCLIR organizers / NIST TREC. 2024. https://github.com/NeuCLIR/neuclir.github.io
8. NeuCLIR Collection 1 download tooling. NeuCLIR organizers. 2022. https://github.com/NeuCLIR/download-collection
9. Overview of the TREC 2022 NeuCLIR Track. Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W. Oard, Luca Soldaini, Eugene Yang. 2023. https://arxiv.org/abs/2304.12367
10. ColBERT-X / Translate-Distill / Multilingual Translate-Distill / PLAID-X (repository and citations; archived May 2026). JHU HLTCOE (Nair, Yang, Lawrie, Mayfield et al.). 2022-2026. https://github.com/hltcoe/ColBERT-X
11. BGE-M3 (FlagEmbedding research page). BAAI. 2024. https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3
12. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu (BAAI). 2024. https://arxiv.org/abs/2402.03216
13. E5 / multilingual-e5 model family (repository). Microsoft (unilm). 2024. https://github.com/microsoft/unilm/tree/master/e5
14. Multilingual E5 Text Embeddings: A Technical Report. Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei (Microsoft). 2024. https://arxiv.org/abs/2402.05672
15. jina-embeddings-v3: Multilingual Embeddings With Task LoRA. Saba Sturua et al. (Jina AI). 2024. https://arxiv.org/abs/2409.10173
16. Qwen3 Embedding (repository: models, MTEB multilingual and MMTEB-R scores). Qwen Team, Alibaba. 2025. https://github.com/QwenLM/Qwen3-Embedding
17. MMTEB: Massive Multilingual Text Embedding Benchmark. Kenneth Enevoldsen et al. (MTEB community). 2025. https://arxiv.org/abs/2502.13595 (code: https://github.com/embeddings-benchmark/mteb)
18. BERGEN multilingual RAG guide (MKQA / XOR-TyDi, BGE-M3 indexes, NLLB translation, prompts, correct-language rate). NAVER LABS Europe. 2024. https://github.com/naver/bergen/blob/main/documentation/multilingual.md
19. Retrieval-augmented generation in multilingual settings. Nadezhda Chirkova, David Rau, Hervé Déjean, Thibault Formal, Stéphane Clinchant, Vassilina Nikoulina (NAVER LABS Europe). 2024. https://arxiv.org/abs/2407.01463
20. Investigating Language Preference of Multilingual RAG Systems. Jeonghyun Park, Hwanhee Lee. Findings of ACL 2025. https://arxiv.org/abs/2502.11175 (code: https://github.com/jeonghyunpark2002/LanguagePreference)
21. Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion (DeLP / DELTA). Jeonghyun Park, Byeongjeong Kim, Seojin Hwang, Hwanhee Lee (Chung-Ang University). arXiv preprint, 2026. https://arxiv.org/abs/2601.02956 (code: https://github.com/jeonghyunpark2002/DELTA)
22. Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG. Dayeon Ki, Marine Carpuat, Paul McNamee, Daniel Khashabi, Eugene Yang, Dawn Lawrie, Kevin Duh (UMD / JHU). ICML 2026 (Spotlight). https://arxiv.org/abs/2509.13930 (code: https://github.com/dayeonki/lang_preference)
23. "Knowing When You Don't Know": A Multilingual Relevance Assessment Dataset for Robust Retrieval-Augmented Generation (NoMIRACL). Nandan Thakur et al. Findings of EMNLP 2024. https://github.com/project-miracl/nomiracl (preprint: https://arxiv.org/abs/2312.11361)
24. Language Model Tokenizers Introduce Unfairness Between Languages. Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, Adel Bibi. NeurIPS 2023. https://github.com/AleksandarPetrov/tokenization-fairness (paper: https://arxiv.org/abs/2305.15425)
25. Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models. Orevaoghene Ahia et al. 2023. https://arxiv.org/abs/2305.13707
26. LangChain text splitters: RecursiveCharacterTextSplitter source (default and language-specific separators). LangChain. 2024-2026. https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
27. PyThaiNLP: Thai natural language processing in Python. PyThaiNLP contributors. 2016-2026. https://github.com/PyThaiNLP/pythainlp
28. IndicTrans2: translation models for all 22 scheduled Indian languages. AI4Bharat. 2023-2024. https://github.com/AI4Bharat/IndicTrans2
29. SEA-LION: Southeast Asian Languages in One Network (models, SEA-LION-Embedding, SEA-GUARD, SEA-HELM). AI Singapore. 2023-2026. https://github.com/aisingapore/sealion
30. Apertus technical report repository. Swiss AI Initiative (EPFL, ETH Zurich, CSCS). 2025. https://github.com/swiss-ai/apertus-tech-report
31. CertusAI: Swiss legal assistant using Apertus (Swiss AI Weeks 2025). Luca Saur et al. 2025. https://github.com/saurluca/CertusAI
32. Swiss AI Romansh data-processing scripts for Apertus v1. Swiss AI Initiative. 2025. https://github.com/swiss-ai/Swiss-AI-Romansh-Scripts
33. TREC RAGTIME: multilingual report generation track (2025 and 2026 editions; successor to NeuCLIR). RAGTIME organizers / NIST TREC. 2025-2026. https://trec-ragtime.github.io/ (source: https://github.com/trec-ragtime/trec-ragtime.github.io)
34. MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented Generation Systems. Nandan Thakur et al. NAACL 2025. https://arxiv.org/abs/2410.13716 (code: https://github.com/vectara/mirage-bench)
35. Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval (SWIM-IR). Nandan Thakur et al. (Google). NAACL 2024. https://arxiv.org/abs/2311.05800 (data: https://github.com/google-research-datasets/SWIM-IR)
36. mmBERT: A Modern Multilingual Encoder with Annealed Language Learning. JHU CLSP. 2025. https://arxiv.org/abs/2509.06888 (code: https://github.com/JHU-CLSP/mmBERT)
37. Byte Latent Transformer: Patches Scale Better Than Tokens. Artidoro Pagnoni et al. (Meta FAIR). 2024. https://github.com/facebookresearch/blt (paper: https://arxiv.org/abs/2412.09871)
