# 11. RAG Evaluation: Metrics, Judges, Benchmarks, Golden Datasets and Production Monitoring

## Overview

A RAG system is a pipeline of at least two learned components: a retriever (often with a reranker) and a generator. Each fails in its own ways. An end-to-end "is the answer right?" score cannot tell you which component to fix. Older QA metrics such as exact match (EM), token F1, BLEU and ROUGE miss most of what matters in RAG: whether the answer is *supported by* the retrieved evidence, whether the evidence was *sufficient*, whether the citations point to the passages that actually support each claim, and whether the system *declines* when the evidence is missing.

Between 2021 and 2026 the field built a layered evaluation stack:

1. **Retrieval-level IR metrics** (Recall@k, Precision@k, MRR, MAP, nDCG), utility-based retrieval labels (eRAG) and benchmarks (BEIR, MTEB/MMTEB, BRIGHT) [6][23][24][25][37].
2. **Component-level RAG metrics** scored by LLM judges or NLI models. Examples are the RAGAS metrics (faithfulness, context precision, context recall, answer relevancy) [1][2], the TruLens "RAG triad" [5], ARES's fine-tuned judges with statistical confidence intervals [3][4], and RAGChecker's claim-level diagnostics [8].
3. **Specialised groundedness and hallucination detectors**: small NLI-style models such as Vectara HHEM and MiniCheck, open LLM judges such as Lynx, plus annotated corpora such as RAGTruth, HaluBench and LLM-AggreFact [9][10][11][12][38].
4. **Citation evaluation** (ALCE's citation recall and precision) [7].
5. **Nugget-based evaluation** in the TREC style (UMBRELA, AutoNuggetizer, Ragnarok, open-rag-eval) [18][19][20][21].
6. **End-to-end RAG benchmarks and challenges** (CRAG, RGB, MultiHop-RAG, FlashRAG, BERGEN, BenchmarkQED, RAGBench, LiveRAG) [15][16][17][31][32][22][26][40], and context-sufficiency analysis [39].
7. **Methodology for LLM-as-a-judge** (agreement studies, bias, open evaluator models) [13][14][30].
8. **Production observability**: OpenTelemetry tracing, online evaluators and feedback loops (TruLens, Arize Phoenix, Langfuse, Azure AI Evaluation, DeepEval) [5][27][28][29][33].

The diagram below maps where each family of metrics attaches to the RAG data flow. The rest of the chapter uses it as a reference.

```
              reference answer / nuggets / gold doc IDs  (golden dataset)
                    |                 |                         |
                    v                 v                         v
 query -> [retriever] -> top-k -> [reranker] -> context -> [generator] -> answer+cites
   |           |                            |          |              |              |
   |   IR metrics: Recall@k, nDCG,  context precision  |   faithfulness/groundedness  |
   |   MRR, MAP (needs qrels / IDs) context recall     |   (answer vs context, NLI    |
   |                                (claims of ref vs  |    or LLM judge)             |
   |                                 context)          |                              |
   +---------------- answer relevancy (answer vs query) ---------------------------+
                     answer correctness (answer vs reference / nuggets)
                     citation recall / precision (each sentence vs cited passages)
                     refusal / negative rejection (when evidence is absent)
```

The axis that matters most for practitioners is **reference-based vs reference-free**. Reference-free metrics such as faithfulness, answer relevancy and context relevance compare the system's own artefacts with each other. They need no labels, so they can run on production traffic. Reference-based metrics such as context recall, factual correctness, nugget recall and IR metrics need a golden dataset. They are the only way to detect an answer that is fully grounded in the wrong documents. The TruLens documentation states this limit outright: the triad guarantees freedom from hallucination only "up to the limit of its knowledge base" [5].

## 1. A taxonomy of RAG failure modes and the metrics that catch them

Before looking at specific frameworks, it helps to list what can go wrong and which metric is sensitive to each failure.

| Failure | Where it originates | Detecting metric(s) | Needs labels? |
|---|---|---|---|
| Relevant doc not retrieved | retriever / chunking / query | Recall@k, context recall, claim recall | yes |
| Relevant doc retrieved but ranked low | retriever / reranker | nDCG, MRR, context precision@k | yes (or LLM relevance labels) |
| Irrelevant "noise" chunks in context | retriever, k too large | context precision / relevance, noise sensitivity | partly |
| Answer adds unsupported claims | generator | faithfulness / groundedness, hallucination rate | no |
| Answer ignores relevant evidence | generator | context utilization, claim recall | yes |
| Answer grounded in wrong/outdated doc | corpus / retriever | answer correctness vs reference | yes |
| Answer off-topic or evasive | generator / prompt | answer relevancy | no |
| Citations missing or wrong | generator | citation recall / precision | no (NLI) |
| Fails to abstain when evidence absent | generator / prompt | negative rejection, CRAG "missing" vs "incorrect" | yes |
| Swallows false facts from context | generator | counterfactual robustness (RGB) | yes |

The RGB benchmark (Chen et al.) frames the generator-side abilities as four tests: noise robustness, negative rejection, information integration, and counterfactual robustness. Its repository runs them through a `noise_rate` sweep, a rejection rate at `noise_rate=1`, `*_int` integration sets, and `*_fact` counterfactual sets with error-detection and error-correction rates, in English and Chinese [16]. CRAG's scoring encodes the abstention trade-off directly. In automatic evaluation a correct answer scores +1, a "missing" answer ("I don't know") scores 0, and an incorrect answer scores -1, so a system that hallucinates is ranked below one that abstains [15].

### 1.1 Sufficient context: is the failure retrieval's or the generator's?

**Problem.** Relevance labels say whether a chunk is *on topic*, not whether the context as a whole *contains enough information* to answer. Joren et al. (Google, UC San Diego, Duke; ICLR 2025) define **sufficient context** as context from which a plausible answer can be derived, and build an LLM autorater that classifies (query, context) pairs as sufficient or insufficient without needing a reference answer [39].

**Findings.** Stratifying errors by sufficiency separates the two root causes in the table above. Strong proprietary models (Gemini 1.5 Pro, GPT-4o, Claude 3.5) answer well when context is sufficient, but often answer *incorrectly instead of abstaining* when it is not. Smaller open models (Llama 3.1, Mistral 3, Gemma 2) hallucinate or abstain often even with sufficient context. A selective-generation method that combines the sufficiency signal with model confidence raised the fraction of correct answers among those given by 2% to 10% [39].

**Lesson.** Add a sufficiency label to every golden item and every sampled production trace. "Insufficient context, wrong answer" is a retrieval or abstention bug; "sufficient context, wrong answer" is a generation bug. Neither the RAG triad nor faithfulness alone makes that split.

## 2. Retrieval metrics and IR benchmarks

### 2.1 Classical rank metrics

Retrieval evaluation predates RAG by decades. The standard set is:

- **Recall@k**: the fraction of relevant items for a query that appear in the top k. For RAG this is often the single most important retrieval number, because the generator can only use what is in context.
- **Precision@k**: the fraction of the top k that is relevant. It matters more in RAG than in web search, because irrelevant chunks are passed to the generator and can contaminate the answer (see noise sensitivity, Section 3).
- **MRR** (mean reciprocal rank): the mean of 1/rank of the first relevant item. It suits single-answer lookups.
- **MAP**: the mean of average precision across queries.
- **nDCG@k**: discounted cumulative gain with graded relevance, normalised by the ideal ranking. It is the headline metric of BEIR and MTEB retrieval.
- **Hit rate / Hits@k**: whether any relevant item appears in the top k.

Toolkits such as **ranx** implement Hits, Hit Rate, Precision, Recall, F1, r-Precision, Bpref, rank-biased precision, MRR, MAP, DCG and nDCG. ranx tests its metrics against TREC Eval for correctness, uses Numba for speed, and provides paired t-tests, Fisher's randomization test and Tukey's HSD for significance [30]. Significance testing gets little use in RAG practice. With golden sets of a few hundred queries, nDCG differences of one or two points are often inside the noise.

**Design note: ID-based vs content-based matching.** IR metrics need *qrels*, meaning the IDs of relevant documents. RAG pipelines re-chunk documents, so gold IDs go stale whenever the chunker changes. Ragas therefore offers three variants of context recall [2c]:

- an **LLM-based** version that splits the *reference answer* into claims and checks whether each claim is attributable to the retrieved context. This removes the need for gold contexts.
- a **non-LLM** version that compares retrieved contexts with reference contexts using string-similarity measures.
- an **ID-based** version that compares `retrieved_context_ids` with `reference_context_ids` directly.

FlashRAG similarly computes `retrieval_doc_recall`, precision, F1 and MAP only when `golden_doc_ids` are present [31].

### 2.2 BEIR: zero-shot heterogeneity

**Problem.** In-domain MS MARCO results overstated how well dense retrievers generalise.

**Design.** BEIR (Thakur, Reimers, Rücklé, Srivastava, Gurevych; NeurIPS 2021 Datasets and Benchmarks) bundles heterogeneous retrieval tasks under one evaluation harness. The datasets are MS MARCO, TREC-COVID, NFCorpus, BioASQ, NQ, HotpotQA, FiQA-2018, Signal-1M, TREC-NEWS, Robust04, ArguAna, Touché-2020, CQADupstack, Quora, DBPedia, SCIDOCS, FEVER, Climate-FEVER and SciFact. Fifteen are publicly downloadable; four (BioASQ, Signal-1M, TREC-NEWS, Robust04) must be reproduced from instructions. The harness computes nDCG@k, MAP@k, Recall@k and Precision@k for k in {1, 3, 5, 10, 100, 1000}, plus MRR [6]. The paper's abstract describes 18 datasets across domain and task combinations. The repository's own counts vary ("15+", "17", 19 table rows), so cite the exact subset you use.

**Lessons.** BEIR's central finding was that BM25 is a strong zero-shot baseline, and that many dense retrievers trained on MS MARCO did not beat it out of domain. Reranking and late-interaction models were the most robust, at higher compute cost [6]. For RAG builders the practical lesson is to evaluate on your own domain. Public leaderboard rank transfers poorly. The official leaderboard and reproducible baselines now live with the "Resources for Brewing BEIR" paper (SIGIR 2024 resource track) [6].

### 2.3 MTEB / MMTEB

MTEB (Muennighoff et al., 2022) widened embedding evaluation beyond retrieval to reranking, clustering, STS, classification and bitext mining. It was greatly expanded in **MMTEB** (Enevoldsen et al., 2025) to many languages and modalities [24]. MTEB's retrieval subset is the usual way embedding vendors report quality. That makes it a target for overfitting, which is another reason a private, in-domain retrieval set remains essential (Section 9).

### 2.4 BRIGHT: reasoning-intensive retrieval

BRIGHT (Su, Yen, Xia et al., ICLR 2025) targets queries for which keyword or semantic overlap is not enough. Its 12 datasets span biology, economics, robotics, math, code and more, with long StackExchange-style posts and math/code questions as queries. The repository evaluates 13 retrievers, including BM25, E5, BGE and the OpenAI, Cohere and Voyage embeddings [25]. The paper reports that models scoring well on BEIR-style benchmarks drop sharply on BRIGHT (see the paper for exact nDCG@10 values). This motivates query rewriting and reasoning-augmented retrieval (Chapters 6 and 8), and it means retrieval evals should include "hard" queries that need inference to connect to the evidence.

### 2.5 eRAG: judging retrieval by downstream utility

**Problem.** Query-document relevance labels correlate only weakly with how a RAG system actually performs, and end-to-end evaluation for every retriever variant is expensive.

**Design.** eRAG (Salemi and Zamani, SIGIR 2024) feeds *each retrieved document on its own* to the generator, scores the resulting output against the task's ground truth, and uses that score as the document's relevance label. Standard set or rank metrics (Precision, Recall, MAP, nDCG) are then computed over these utility labels [37].

**Numbers.** Across the KILT-based tasks studied, eRAG's retrieval scores correlated better with end-to-end RAG performance than baseline methods, with Kendall's tau improvements of 0.168 to 0.494, and it used up to 50 times less GPU memory than end-to-end evaluation. It is distributed as the `erag` Python package [37].

**Lesson.** "Relevant to a human" and "useful to this generator" differ. When choosing a retriever or reranker for a fixed generator, utility labels are a cheap middle ground between qrels and full end-to-end runs.

## 3. RAGAS: reference-free component metrics

**Problem.** In 2023 most teams had no labelled data for their RAG application. RAGAS (Es, James, Espinosa-Anke, Schockaert; arXiv 2309.15217, EACL 2024 demo track) proposed scoring RAG pipelines *without human references*, using an LLM to decompose and verify [1]. The paper introduced three dimensions (faithfulness, answer relevance and context relevance) and a small validation set, WikiEval, built from Wikipedia pages. It reported that its LLM-decomposition metrics agreed with human pairwise judgments better than direct GPT-3.5 scoring, with the strongest agreement on faithfulness (see [1] for exact agreement values). The open-source library has since become one of the most widely used RAG evaluation toolkits (about 16k GitHub stars in 2026) [2].

### 3.1 Metric definitions (current library)

The definitions below are taken from the Ragas metric documentation [2a-2g].

**Faithfulness** (reference-free) [2a]. The response is split into atomic statements. For example, "Einstein was born in Germany on 20th March 1879" becomes two statements. Each statement is then judged as supported or not supported by the retrieved context:

```
faithfulness = |claims supported by retrieved context| / |claims in response|
```

A variant, `FaithfulnesswithHHEM`, replaces the LLM verifier with Vectara's open HHEM-2.1-Open T5 classifier. It runs on CPU by default (batch size 10), which cuts the cost of the verification step [2a].

**Context precision** (rank-aware) [2b]:

```
Context Precision@K = sum_{k=1..K} (Precision@k * v_k) / (# relevant items in top K)
Precision@k         = TP@k / (TP@k + FP@k),   v_k in {0,1}
```

This is average precision over the retrieved list. An irrelevant chunk at rank 1 roughly halves the score in the documentation's example. The same chunk at rank 2 leaves it unchanged. Three implementations exist:

- with a reference answer (`LLMContextPrecisionWithReference`)
- without one (`LLMContextPrecisionWithoutReference`, also called *context utilization*), which judges each chunk against the generated response
- a non-LLM string-similarity variant (`NonLLMContextPrecisionWithReference`) that needs `rapidfuzz` [2b]

**Context recall** [2c]: the fraction of reference-answer claims supported by the retrieved context, with the non-LLM and ID-based variants described in Section 2.1.

**Answer (response) relevancy** [2d]. An LLM reverse-engineers N questions from the answer (N = 3 by default, controlled by `strictness`). The score is the mean cosine similarity between the embeddings of these questions and the original query. Incomplete or padded answers produce generated questions that drift from the query, which lowers the score. The metric explicitly does *not* measure factual accuracy. Because cosine similarity can be negative, the score is not strictly bounded to [0, 1].

**Noise sensitivity** [2e]: the fraction of response claims that are *incorrect* (checked against the ground truth), computed separately for claims tied to relevant and to irrelevant retrieved contexts (`mode="irrelevant"`). Lower is better. It measures how easily the generator is misled by distractors.

**Factual correctness** [2f]: both the response and the reference are decomposed into claims, and NLI then counts true positives, false positives and false negatives to give precision, recall or F1 (F1 is the default). Two knobs, *atomicity* and *coverage*, control how finely sentences are decomposed. This makes the metric's granularity a tunable hyperparameter, so it must be held fixed when comparing runs.

### 3.2 Architecture

```
   sample = {user_input, retrieved_contexts, response, reference?, reference_contexts?}
                                   |
               +-------------------+--------------------+
               v                   v                    v
     [claim extractor LLM]   [question generator]  [chunk relevance judge]
        response -> claims     response -> N qs     (chunk, ref|resp) -> 0/1
               |                   |                    |
     [NLI verifier: LLM or   [embed + cosine vs     [rank-aware AP over v_k]
      HHEM T5 classifier]     user_input]               |
               v                   v                    v
          faithfulness       answer relevancy      context precision / recall
```

### 3.3 Failure modes and lessons

- **Judge-dependence.** Every metric involves at least one LLM call (claim extraction, verification), often several. Scores move when the judge model or prompt changes. Pin both, and re-baseline when either changes.
- **Decomposition variance.** Claim splitting is not deterministic. Two runs can produce different claim counts, which changes the denominator. Averaging over a few hundred samples dampens this, but per-sample scores are noisy.
- **Cost scales with claims.** Faithfulness needs one extraction call plus verification per response, which is why the HHEM substitution exists.
- **Reference-free does not mean correctness-free.** High faithfulness together with low factual correctness signals a retrieval or corpus problem, not a generation problem.

## 4. The TruLens RAG triad

**Problem.** TruLens wanted a minimal, label-free check that covers every edge of the query, context and answer triangle [5].

**Design.** The "RAG triad" consists of three feedback functions [5b]:

1. **Context relevance**: is each retrieved chunk relevant to the query? This verifies retrieval quality, because irrelevant context "can be woven into the LLM's answer."
2. **Groundedness**: each claim in the response is checked for support in the retrieved context.
3. **Answer relevance**: does the response address the original question?

The documented claim is that if all three pass, the application is free of hallucination "up to the limit of its knowledge base" [5b]. This is a useful framing, and it states its own blind spot: a correct-looking answer drawn faithfully from an outdated or wrong document passes all three checks.

**Architecture (2025-2026).** TruLens (MIT licence, repository under the `truera` org, with Snowflake-hosted community and a Snowflake Cortex provider) now instruments applications with OpenTelemetry spans. Each step records inputs, outputs, latency, tokens and cost, and traces can be exported to OTLP backends such as Jaeger, Grafana Tempo, Datadog or New Relic. A "Metric" pairs a judge implementation with *selectors* that pull the right span fields (for example, the query and retrieved context). Metrics can run inline as the app executes or in batch over a dataset. Judges can be OpenAI, Azure OpenAI, LiteLLM, Gemini, Bedrock, Snowflake Cortex or Hugging Face models [5]. The project has extended the triad to agents with seven evaluators: LogicalConsistency, ExecutionEfficiency, PlanAdherence, PlanQuality, ToolSelection, ToolCalling and ToolQuality [5]. These derive from the Goal-Plan-Action ("Agent GPA") framework of Jia et al. (Snowflake AI Research and Stanford, 2025) [34].

**Published numbers (project claims, treat as vendor-reported).** The TruLens README reports a groundedness F1 of 0.81 on an 11,000-example LLM-AggreFact holdout, which it says is ahead of Bespoke-MiniCheck-7B on F1, precision and recall. Citing a third-party (AIMultiple) comparison dated March 2026, it also reports a context-relevance nDCG@5 of 0.93 against WandB Weave, RAGAS, DeepEval and UpTrain, and a 4.2:1 win-loss rate (next best 3.3:1) on adversarial near-duplicate passages with one fact swapped. The README states that Agent GPA judges catch 267 of 281 (about 95%) human-annotated agent errors on TRAIL/GAIA, against 55% for a baseline trace judge; the paper's own abstract gives a range of 80% to over 95% across error types and 86% agreement with humans on error localisation [5][34]. None of these figures has been independently replicated.

## 5. ARES: trained judges plus prediction-powered inference

**Problem.** LLM-judge scores are point estimates of unknown bias. Teams comparing two RAG configurations need to know whether a difference is real.

**Design.** ARES (Saad-Falcon, Khattab, Potts, Zaharia; Stanford; arXiv 2311.09476, NAACL 2024) scores the same three dimensions as the triad: context relevance, answer faithfulness and answer relevance. It replaces prompted judges with *fine-tuned lightweight classifiers* and attaches *confidence intervals* using prediction-powered inference (PPI) [3][4].

```
  corpus docs ----> [LLM few-shot query/answer generator] ----> synthetic (q, passage, a)
                                                                 + generated negatives
                                                                         |
                                                                         v
                                               [fine-tune classifier judges, one per
                                                dimension: ctx-rel, faithful, ans-rel]
                                                                         |
   RAG system outputs (unlabeled, large) ----------------------------->  | predictions
   human-labelled validation set (small, ~50 to several hundred) --------+
                                                                         v
                                   [Prediction-Powered Inference: correct judge bias using
                                    labelled set; output score + 95% confidence interval]
```

**Key techniques.**

- **Synthetic training data.** Queries and answers are generated from the target corpus with few-shot prompts. The example configuration samples 6,189 documents.
- **Per-dimension classifier judges.** The quick-start trains for 10 epochs, patience 3, learning rate 5e-6, batch size 1 with 32-step gradient accumulation. The paper uses small encoder models as judges.
- **PPI.** A small human-labelled set calibrates the judge's error rate, so the final estimate is unbiased with a valid confidence interval [4].

The repository says at least 50 labelled examples are required and "several hundred" is ideal. Its PPI example uses 300 [4].

**Numbers.** The repository's worked example for context relevance reports an ARES estimate of 0.606 with a confidence interval of [0.547, 0.664], against ground truth of 0.600. The judge's raw accuracy was 0.789, over an evaluation set of 4,421 examples [4]. The example shows the main point of PPI: a judge that is only about 79% accurate still yields an interval that brackets the true value. The paper evaluates on KILT and SuperGLUE-derived tasks and reports that ARES ranks RAG configurations more accurately than RAGAS (see [3] for Kendall's tau values).

**Failure modes.** Fine-tuned judges are domain-specific: a judge trained on one corpus's synthetic data must be retrained for another. Synthetic queries may not match real user phrasing. Only the context-relevance checkpoint is distributed pre-trained [4].

**Lesson.** ARES's lasting contribution is statistical: **report a confidence interval, and spend a small human-labelling budget to de-bias the automatic judge.** That pattern applies regardless of judge type.

## 6. RAGChecker: claim-level diagnostics (Amazon Science)

**Problem.** Triad-style scores say *that* something is wrong but not *why*. RAGChecker (arXiv 2408.08067, NeurIPS 2024 Datasets and Benchmarks Track) uses claim-level entailment to separate retriever and generator faults [8].

**Design.** The response and the ground-truth answer are both decomposed into claims. Each claim is checked for entailment against the ground-truth answer and against each retrieved chunk. From this matrix RAGChecker derives:

- **Overall:** precision (the fraction of response claims that are correct), recall (the fraction of ground-truth claims covered) and F1.
- **Retriever:** *claim recall*, meaning the fraction of ground-truth claims present in some retrieved chunk, and *context precision*.
- **Generator:**
  - context utilization: the use of the relevant information that was retrieved
  - noise sensitivity in relevant and irrelevant chunks: incorrect claims caused by each
  - hallucination: incorrect claims supported by no chunk
  - self-knowledge: correct claims supported by no chunk, i.e. drawn from parametric memory
  - faithfulness

The only required annotation per query is a ground-truth answer. The README's examples use Llama 3.1 70B Instruct on AWS Bedrock (`bedrock/meta.llama3-1-70b-instruct-v1:0`) as both extractor and checker; any LiteLLM-compatible model can be substituted [8].

**Meta-evaluation.** RAGChecker was validated against 280 human-annotated instances, each labelled by two annotators, with baseline outputs from TruLens, ARES, RAGAS and CRUD-RAG. The paper reports better correlation with human judgments than these baselines (Tables 2 and 5 of [8]).

**Lesson.** The *self-knowledge* vs *hallucination* split is diagnostically valuable. A correct claim with no supporting chunk passes correctness checks but fails groundedness. Whether that is acceptable depends on the product, for example a regulated domain that requires every claim to be cited.

## 7. Groundedness and hallucination detectors

LLM judges are slow and expensive on production volume. A separate line of work trains small models to answer one question: *is claim c supported by document d?*

### 7.1 Vectara HHEM and the hallucination leaderboard

HHEM-2.1-Open is a T5-based classifier trained to detect hallucinations relative to a source. Ragas wraps it as a drop-in faithfulness verifier [2a]. Vectara's public **hallucination leaderboard** has each LLM summarise more than 7,700 private articles (about 50 to 24K words) under a "facts-only" prompt at temperature 0, with summaries capped at 20% of the source length. The commercial HHEM-2.3 then scores each summary. The leaderboard reports:

- factual consistency rate
- hallucination rate (100 minus consistency)
- answer rate (refusals are excluded and scores are computed on documents every model summarised)

As of the September 22, 2026 update, the three leading entries showed hallucination rates of 1.8% to 3.3%, and the fifth-ranked model 4.1% [11]. Note the scope: this measures *consistency with the source* in summarisation, not open-domain factuality or answer quality [11].

### 7.2 MiniCheck and LLM-AggreFact

MiniCheck (Tang, Laban, Durrett; EMNLP 2024) trains small fact-checkers on synthetic data: RoBERTa-large, DeBERTa-v3-large, Flan-T5-large and Bespoke-MiniCheck-7B. The ~770M-parameter Flan-T5 model is reported to reach GPT-4-level performance. Bespoke-MiniCheck-7B is presented as the strongest of the family. Inputs are one sentence against one document, so multi-sentence answers must be split first. With vLLM prefix caching, checking a 29K-example test set takes about 30 minutes on one A6000, against 55 without caching [10]. **LLM-AggreFact**, released alongside it, aggregates 11 public factual-consistency datasets covering closed-book and grounded generation. Most claims come from real model outputs rather than injected errors. It is now the common yardstick for groundedness judges, which is why TruLens reports its F1 there [10][5].

### 7.3 RAGTruth: span-level hallucination labels

RAGTruth (Niu, Wu et al., ACL 2024; arXiv 2401.00396) provides nearly 18,000 (17,790) naturally generated responses over 2,965 instances across QA, data-to-text (Yelp) and summarisation (CNN/DM and news). 7,664 of the responses contain hallucinations. Labels are *word-level spans* with type labels (for example, "Evident Baseless Info") and flags such as `implicit_true`, for correct information not in the context [12]. Span-level labels make it possible to train and evaluate detectors that *highlight* unsupported text, which is a better UX than a single score.

### 7.4 RAGBench / TRACe

Galileo's RAGBench packages component datasets (HotpotQA, MS MARCO, HAGRID, ExpertQA and others) with scripts that reproduce baseline judge scores for GPT-3.5, RAGAS and TruLens [26]. The accompanying TRACe framework defines utilization, relevance, adherence and completeness. Its paper reports that a fine-tuned small encoder can be competitive with prompted LLM judges on these labels; consult [26] for numbers.

### 7.5 Lynx and HaluBench

Lynx (Patronus AI with Contextual AI and Stanford, arXiv 2407.08488, 2024) is an open-weight hallucination judge fine-tuned from Llama 3 in 8B and 70B sizes. Unlike sentence-level NLI classifiers, it takes (question, context, answer) and emits a pass/fail verdict with chain-of-thought reasoning. It was released with **HaluBench**, about 15k samples drawn from sources including CovidQA, PubMedQA, DROP and FinanceBench, where hallucinated answers are produced by perturbing correct ones so that they are hard to spot [38]. The authors report that Lynx-70B beats GPT-4o and Claude 3 Sonnet as a judge on HaluBench. These are vendor results on a vendor benchmark, and the licence of the weights should be checked before commercial use [38]. Lynx sits between MiniCheck-class classifiers and frontier LLM judges in cost, and is a common choice when groundedness must be judged on-premises.

**Design pattern.** Use a small NLI-style checker such as HHEM or MiniCheck for **100% online coverage** (it is cheap and runs on CPU or a small GPU). Escalate low-confidence or flagged samples to an LLM judge or a human. This tiering appears repeatedly across vendor documentation and is the main cost lever in production evaluation.

## 8. Citation accuracy

**Problem.** Answers that cite sources are only trustworthy if (a) every claim is backed by a citation and (b) every citation actually supports the claim it is attached to.

**ALCE** (Gao, Yen, Yu, Chen; EMNLP 2023) set the standard formulation [7]. It covers three datasets (ASQA, QAMPARI and ELI5) and three axes:

- **fluency** (MAUVE)
- **correctness** (short-answer recall on ASQA, recall over answer lists on QAMPARI, NLI over sub-claims on ELI5)
- **citation quality**

Citation quality is measured with an NLI model:

- **Citation recall**: a statement is fully supported if the concatenation of its cited passages entails it.
- **Citation precision**: a citation is *irrelevant* if it does not entail the statement on its own and removing it does not affect entailment by the remaining citations.

The paper reports that even the strongest models of the time lacked complete citation support for a large fraction of ELI5 generations (about half), and that retrieval quality limited the results [7].

```
answer sentences s_1..s_n, each with citation set C_i
  for each s_i:
     recall_i = NLI( concat(passages in C_i) => s_i )            # 1 or 0
     for each c in C_i:
        irrelevant(c) = not NLI(c => s_i) and NLI(C_i \ {c} => s_i)
  citation_recall    = mean_i recall_i
  citation_precision = 1 - (#irrelevant citations / #citations)
```

Later systems apply the same idea with LLM judges. RAGElo's rubric evaluators include a citation-quality criterion that checks whether claims are backed by citations and whether the citations include relevant excerpts [35]. In production, the cheapest robust check is to run a sentence-level NLI checker (Section 7) against *only the cited* chunks, not the whole context. Comparing that score with whole-context faithfulness shows whether the model is grounded but mis-attributing.

## 9. Nugget-based evaluation and the TREC RAG Track

Long-form RAG answers have no single gold string. The IR community's answer, revived for the TREC 2024 RAG Track, is **nugget evaluation**: define the atomic facts a good answer should contain, then measure how many each answer covers.

```
 topics --> [retrieval runs over MS MARCO v2.1 segments] --> pooled passages
                                                                  |
                                       [UMBRELA: LLM relevance labels 0..3]
                                                                  |
                       passages with judgment >= 2 --> [AutoNuggetizer: create nuggets]
                                                                  |  label vital / okay
 RAG answers (with [n] citations) --> [assign: support / partial / not] <-+
                                                                  |
                       vital_score, all_score, strict_* (full=1.0, partial=0.5)
```

- **UMBRELA** (Upadhyay, Pradeep, Thakur, Craswell, Lin, 2024) is an open-source reproduction of Microsoft Bing's LLM relevance assessor prompt. It produces 4-level (0 to 3) relevance labels, with cloud or open-weight judges [18]. It turns unlabeled corpora into graded qrels and so makes nDCG computable without human assessors.
- **AutoNuggetizer** (castorini/nuggetizer) uses LLMs to create nuggets from relevant passages, label them *vital* or *okay*, and assign each nugget to an answer as fully or partially supported. Scores are `strict_vital_score` and `strict_all_score` (full support only) and `vital_score` and `all_score` (full = 1.0, partial = 0.5) [19]. The TREC 2024 RAG Track's initial results paper (arXiv 2411.09607) and "The Great Nugget Recall" (SIGIR 2025) report that fully automatic nugget scores track manual, assessor-based nugget scores at the run level [19].
- **Ragnarök** (Pradeep et al., ECIR 2025) is the reusable baseline framework for the track, using the MS MARCO v2.1 segmented collection [20].
- **open-rag-eval** (Vectara, 2025) packages UMBRELA and AutoNuggetizer with HHEM-based hallucination detection and citation-aware scoring, so a team can evaluate a RAG pipeline *without golden chunks or golden answers*. It has connectors for Vectara, LlamaIndex and LangChain, and an optional `GoldenAnswerEvaluator` when references exist [21].

**Lesson.** Nugget recall is the most principled answer-completeness metric for long-form RAG. It also degrades more gracefully than claim-level F1 against a single reference, because many differently worded answers can cover the same nuggets.

## 10. End-to-end RAG benchmarks

| Benchmark | Year / venue | Size & scope | What it stresses | Scoring |
|---|---|---|---|---|
| ALCE [7] | 2023, EMNLP | ASQA, QAMPARI, ELI5 | citations, long-form | MAUVE, correctness, citation P/R (NLI) |
| RGB [16] | 2023/24 | EN + ZH | noise, rejection, integration, counterfactual | accuracy, rejection rate, error detection/correction |
| MultiHop-RAG [17] | 2024, COLM | 2,556 queries, 2-4 docs each | cross-document reasoning, metadata | retrieval + QA scripts |
| CRAG [15] | 2024, Meta | 5 domains, 8 question types, mock web + KG APIs | dynamism, long-tail, false premise | +1 / 0 / -1 (correct / missing / incorrect) |
| RAGTruth [12] | 2024, ACL | 17,790 responses, 3 tasks | span hallucinations | word-level labels |
| RAGChecker [8] | 2024, NeurIPS D&B | 280 meta-eval instances x 2 annotators | diagnostic decomposition | claim-level P/R/F1 + 7 sub-metrics |
| FlashRAG [31] | WWW 2025 Resource Track | 36 datasets, 23 methods (incl. 7 reasoning-based) | method comparison | EM, F1, doc recall/precision |
| BERGEN [32] | 2024, Naver | 20+ retrievers, 4 rerankers, 20+ LLMs | reproducible ablations | Match, EM, LLMEval |
| BenchmarkQED [22] | 2025, Microsoft | AutoQ/AutoE/AutoD; 2 released corpora | local vs global queries (GraphRAG) | pairwise LLM judge, assertions |
| TREC RAG 2024 [19][20] | 2024-25 | MS MARCO v2.1 segments | long-form, citations | nuggets (vital/okay), UMBRELA qrels |
| BRIGHT [25] | ICLR 2025 | 12 datasets | reasoning-intensive retrieval | nDCG@10 |
| LiveRAG [40] | SIGIR 2025 | FineWeb-10BT corpus, fixed Falcon3-10B generator, 500 unseen live questions | retrieval design under fixed generator | LLM judge (correctness, faithfulness) + manual review |

Notes on individual benchmarks:

- **CRAG** labels each question with domain (finance, music, movie, sports, open), type (simple, simple with condition, comparison, aggregation, set, false premise, post-processing, multi-hop), temporal dynamism (static, slow-changing, fast-changing, real-time) and popularity (head, torso, tail). It provides up to 50 HTML pages per question (5 sampled from the top 10 for tasks 1 and 2) plus mock knowledge-graph APIs [15]. These slices are more useful than the aggregate. They show, for example, whether a system's errors concentrate on fast-changing or tail-entity questions. CRAG's licence is CC BY-NC 4.0 [15].
- **MultiHop-RAG** spreads evidence over 2 to 4 news documents per query and depends on metadata, which exposes chunk-level retrievers that cannot gather complementary evidence [17].
- **BERGEN** exists because RAG papers were hard to compare: different retrievers, prompts and metrics. Its baseline tables use a lenient *Match* metric (for example, Solar-10.7B scores 76.2 on ASQA and 70.2 on NQ) alongside LLM-based evaluation [32].
- **BenchmarkQED** reflects the GraphRAG lineage (Chapter 7). AutoQ synthesises queries along a *local to global* scope axis, including data-linked multi-hop local queries. AutoE runs side-by-side LLM judging on relevance, comprehensiveness, diversity and empowerment, plus correctness and assertion-based scoring when ground truth exists. It ships a podcast-transcript corpus (70 episodes) and 1,397 AP health news articles [22]. Global "sensemaking" questions have no gold answer, so pairwise preference is the only practical signal.
- **LiveRAG** (SIGIR 2025 LiveRAG Challenge, run by TII with Pinecone and AI71) fixed both the corpus (FineWeb-10BT) and the generator (Falcon3-10B-Instruct), so teams competed only on retrieval, reranking and prompting. 70 teams from 27 countries answered 500 unseen questions in a two-hour live window. Questions were generated with TII's **DataMorgana**, which produces diverse synthetic Q&A pairs from configurable user and question categorisations. Scoring used an LLM judge for correctness and faithfulness, followed by manual review of the top entries [40]. The design is a useful template for internal bake-offs: freeze the generator, hold back an unseen test set, and use a synthetic generator with explicit diversity controls for the development set.

## 11. LLM-as-a-judge for RAG: methodology

Almost every metric above relies on an LLM or NLI judge. What is known about judge reliability:

- **Agreement.** Zheng et al. (MT-Bench / Chatbot Arena, 2023) found that a GPT-4 judge and humans agree over 80% of the time, the same level at which humans agree with each other. This was based on 3.3K expert votes over 80 questions [13]. This result is the usual justification for judge-based evaluation. It was measured on general chat, not on RAG groundedness.
- **Position bias.** Wang et al. ("Large Language Models are not Fair Evaluators", 2023) showed severe position bias in pairwise judging. They proposed Multiple Evidence Calibration (sample k rationales) and Balanced Position Calibration (score both orders) [14]. Any pairwise RAG comparison (RAGElo, BenchmarkQED AutoE, Arena-style) should evaluate both orders.
- **Open evaluator models.** Prometheus 2 (7B and 8x7B, built by merging Mistral-based models trained for direct assessment and for pairwise ranking) supports absolute 1 to 5 rubric grading and pairwise ranking. The paper reports Pearson correlations above 0.5 with humans, GPT-4 and Claude 3 Opus across its direct-assessment benchmarks (for example 0.555 with humans on FLASK for the 8x7B model, against 0.679 for GPT-4), and high pairwise agreement with human preferences; see [36] for per-benchmark values. The 7B model fits on a single consumer or workstation GPU [36]. This makes local, privacy-preserving judging possible for regulated RAG data.
- **Tournament ranking.** RAGElo ranks RAG agents with pairwise LLM games and Elo ratings with confidence intervals. It can place a new agent after a few games, and it has separate retrieval evaluators that grade documents independently of answers [35].

**Practices that the sources converge on:**

1. **Decompose before judging.** Claim-level or nugget-level verification (RAGAS, RAGChecker, ALCE, AutoNuggetizer) is more stable and more interpretable than holistic "rate 1 to 10" prompts.
2. **Prefer binary or low-cardinality labels** (supported or not, 0 to 3 relevance) over fine-grained scales. UMBRELA's 0 to 3 scale and CRAG's four tiers are examples.
3. **Calibrate against humans and report uncertainty.** ARES's PPI [3][4] is the formal version. The minimal version is to measure judge accuracy on a few hundred human labels before trusting it.
4. **Pin the judge.** Record the judge model version and prompt hash with every score, because score drift after a judge upgrade is common.
5. **Do not let a model grade itself on its own facts.** Use a different model family, or use a small NLI checker, for groundedness.
6. **Use specialised checkers for volume.** HHEM and MiniCheck give near-LLM accuracy on grounding at a fraction of the cost [10][11].

## 12. Building golden datasets

Every reference-based metric depends on a golden set. The public frameworks suggest the following recipe.

```
  1. SEED              2. SYNTHESIZE               3. CURATE              4. FREEZE
+----------------+  +-------------------------+  +-------------------+  +---------------+
| prod query logs|  | KG over corpus (ragas): |  | SME review: keep, |  | versioned set |
| SME-written qs |->| nodes, NER, similarity  |->| fix, drop; label  |->| + slices      |
| support tickets|  | edges -> scenarios ->   |  | gold doc IDs,     |  | + judge cal.  |
| failure cases  |  | single/multi-hop,       |  | reference answer, |  | subset (PPI)  |
+----------------+  | specific/abstract qs    |  | nuggets (vital?)  |  +-------+-------+
                    +-------------------------+  +-------------------+          |
         ^                                                                       |
         +------------------ 5. GROW: add every production failure -------------+
```

**Synthetic generation.** Ragas builds a knowledge graph over the corpus [2g]:

- a splitter creates nodes
- extractors add properties (for example, `NERExtractor` adds entities)
- relationship builders link nodes (for example, `JaccardSimilarityBuilder` on entity overlap)
- query synthesizers sample node sets into *scenarios* (nodes, query length, style, persona) and emit a query, reference contexts and a reference answer

Ragas targets four query types: single-hop specific, single-hop abstract, multi-hop specific and multi-hop abstract [2g]. ARES generates synthetic queries and answers per document with few-shot prompts [4]. BenchmarkQED's AutoQ varies scope from local to global [22]. TII's DataMorgana, used for the LiveRAG Challenge, lets the builder define weighted categorisations (for example, question phrasing, factoid vs open-ended, expert vs novice user) and samples combinations to force diversity [40].

**Why synthetic alone is not enough.**

- Synthetic questions are written *from* a passage, so they share its vocabulary. That inflates lexical and dense retrieval scores relative to real user queries, the same out-of-distribution gap BEIR and BRIGHT documented for retrievers [6][25].
- Generators avoid unanswerable and false-premise questions unless told to include them. CRAG deliberately includes false-premise questions [15], and RGB includes negative-rejection sets [16].

**Recommended composition** (synthesised from the benchmarks above):

- **Slices**, as in CRAG: query type (simple, comparison, aggregation, multi-hop, false-premise), temporal dynamism, entity popularity, document type. Report per slice, because averages hide regressions.
- **Unanswerable fraction**: questions whose answer is not in the corpus, to measure abstention.
- **Gold evidence as IDs and as claims.** Store document or section IDs for IR metrics and reference answers or nuggets for claim-level metrics. Because chunk IDs change with re-chunking (Chapter 2), store *document-level* or *span-offset* provenance and map it to chunks at evaluation time. Ragas's ID-based and LLM-based recall variants exist for this reason [2c].
- **A human-labelled calibration subset** (at least 50, ideally several hundred, per ARES [4]) with judge-dimension labels. It lets you measure judge accuracy and apply PPI.
- **Size.** Public RAG evaluation sets run from a few hundred (RAGChecker meta-evaluation: 280) to a few thousand items (MultiHop-RAG: 2,556) [8][17]. A few hundred well-sliced, human-verified items usually discriminate between configurations better than thousands of unverified synthetic ones.

**Grow from production.** Every user-reported failure, thumbs-down or low groundedness alert should become a candidate golden item after review. Langfuse, Phoenix and TruLens all support turning traces into versioned datasets for experiments [27][28][5].

## 13. Production monitoring

Offline evaluation tells you whether a change is better. Production monitoring tells you whether the live system is still good: corpora change, user intent drifts, and upstream models are upgraded.

```
 user --> app (instrumented: OpenTelemetry / OpenInference spans)
            | retrieve span: query, chunk ids, scores, latency
            | rerank span
            | llm span: prompt, context, answer, citations, tokens, cost
            v
     [trace store] --(sample 100%)--> cheap checks: HHEM/MiniCheck groundedness,
            |                         citation-NLI, refusal detector, PII, latency/cost
            |--(sample 1-10%)-------> LLM judges: context relevance, answer relevance,
            |                         completeness (pinned judge + prompt version)
            |--> user feedback (thumbs, edits, escalations)
            v
     dashboards + alerts (per slice: tenant, intent, doc source, language)
            |
            +--> annotation / review queue --> golden dataset (Section 12) --> CI gate
```

**Tooling (2025-2026):**

- **TruLens**: OTel-native tracing; inline or batch feedback functions; the RAG triad plus agent evaluators; export to Jaeger, Tempo, Datadog, New Relic or any OTLP backend [5].
- **Arize Phoenix**: open-source, self-hosted. It traces with OpenTelemetry and OpenInference instrumentation for OpenAI, Anthropic, LangChain, LlamaIndex and others. It offers `arize-phoenix-evals` for RAG relevance and answer relevance, versioned datasets and experiments, and an MCP server so coding agents can query traces. Arize positions its commercial AX product for managed production monitoring [27].
- **Langfuse**: tracing of LLM, retrieval and embedding calls; evaluation via LLM-as-a-judge, code evaluators, user feedback and manual labelling; datasets for pre-deployment experiments; a jump from a bad trace into the prompt playground; and self-hosting via Docker or Kubernetes [28].
- **Azure AI Evaluation SDK**: built-in AI-assisted `GroundednessEvaluator`, `RelevanceEvaluator` and `RetrievalEvaluator`, run over `query`, `context` and `response` columns [29].
- **DeepEval**: a pytest-style framework with Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall and Contextual Relevancy metrics plus G-Eval custom criteria. `deepeval test run` makes RAG metrics into **CI regression gates** [33].

**Monitoring practices:**

1. **Log retrieval artefacts, not just answers.** Without chunk IDs and scores in the trace, you cannot compute context metrics after the fact or reproduce failures.
2. **Tier the evaluators by cost.** Small NLI checkers on all traffic; LLM judges on a sample; humans on flagged items. Vectara's open-rag-eval and the HHEM variant in Ragas show this division of labour [21][2a].
3. **Alert on distributions, per slice.** Track the share of answers below a groundedness threshold, the refusal rate and the empty-retrieval rate, for each tenant, source and intent. A global average can hide a broken connector for one source.
4. **Separate "the model refused" from "retrieval returned nothing".** These have different fixes. CRAG's missing vs incorrect split is the offline analogue [15].
5. **Re-run the golden set on every change** (embedding model, chunker, reranker, prompt, LLM, judge). Report deltas with confidence intervals (PPI or bootstrap) and per-slice breakdowns [3][30].
6. **Close the loop.** Failures from monitoring feed the golden set (Section 12), so the offline benchmark tracks real-world difficulty.

## 14. Framework comparison

| Framework | Origin | Judge type | Needs reference? | Metrics (core) | Statistical rigor | Best use |
|---|---|---|---|---|---|---|
| RAGAS [1][2] | Exploding Gradients / academic, 2023 | Prompted LLM (+ HHEM option) + embeddings | optional | faithfulness, ctx precision/recall, answer relevancy, noise sensitivity, factual correctness | none built in | fast offline iteration, synthetic test sets |
| TruLens triad [5] | TruEra / Snowflake | Prompted LLM, many providers | no | context relevance, groundedness, answer relevance (+ agent evals) | none built in | tracing + online feedback |
| ARES [3][4] | Stanford, 2023 | Fine-tuned classifiers | small labelled set | ctx relevance, faithfulness, answer relevance | PPI confidence intervals | comparing configs with rigor |
| RAGChecker [8] | Amazon, 2024 | LLM claim extractor + checker (Llama 3.1 70B in examples) | ground-truth answer | overall P/R/F1, claim recall, ctx precision, utilization, noise sens., hallucination, self-knowledge, faithfulness | meta-evaluated vs humans | root-cause diagnosis |
| ALCE [7] | Princeton, 2023 | NLI model | for correctness | citation recall/precision, correctness, MAUVE | n/a | citation quality |
| AutoNuggetizer + UMBRELA [18][19] | Waterloo / TREC, 2024 | LLM | no (auto nuggets) | nugget vital/all scores, graded qrels | validated vs TREC assessors | long-form answer completeness |
| open-rag-eval [21] | Vectara, 2025 | LLM + HHEM | no | UMBRELA, AutoNuggetizer, hallucination, citations | n/a | label-free pipeline eval |
| HHEM / MiniCheck [10][11] | Vectara; UT Austin | small classifiers | no | grounded / not (sentence-doc) | benchmarked on LLM-AggreFact | 100%-traffic groundedness |
| DeepEval [33] | Confident AI | LLM (G-Eval etc.) | optional | RAG metrics + custom | n/a | CI unit tests |
| Phoenix / Langfuse [27][28] | Arize; Langfuse | LLM + human + code | optional | relevance, QA, custom | n/a | production tracing & monitoring |

## Key takeaways for system designers

1. **Evaluate retrieval and generation separately, then end-to-end.** Use Recall@k or context recall to tell whether the evidence was available, faithfulness to tell whether it was used honestly, and correctness or nugget recall to tell whether the answer was right. A single blended score hides which component regressed.
2. **Reference-free metrics are necessary but not sufficient.** The RAG triad's own documentation limits its guarantee to the correctness of the knowledge base [5]. Keep a reference-based golden set for correctness and abstention.
3. **Make recall the headline retrieval number.** In RAG, a missed document cannot be recovered by the generator, while a noisy one only might hurt. Add rank-aware context precision once recall is acceptable, because rank-1 noise measurably degrades answers [2b][2e].
4. **Decompose into claims or nuggets.** Claim-level verification (RAGAS, RAGChecker, ALCE) and nugget scoring (AutoNuggetizer) are more stable and more diagnosable than holistic grades. Hold decomposition settings fixed across runs [2f].
5. **Treat judges as instruments that need calibration.** Measure judge accuracy on a few hundred human labels, report confidence intervals (ARES/PPI), evaluate pairwise comparisons in both orders [14], and pin judge model and prompt versions.
6. **Use small specialised checkers for scale.** HHEM and MiniCheck-class models make it affordable to check every answer's groundedness. Reserve LLM judges for sampled, harder dimensions [10][11].
7. **Measure citation precision as well as recall.** A cited but non-supporting passage is worse than no citation, because it creates false confidence. ALCE's leave-one-out precision definition is a practical template [7].
8. **Build golden sets that look like production.** Seed from real queries, use synthetic generation (knowledge-graph-based, multi-hop, abstract) for coverage, include unanswerable and false-premise questions, slice by type and freshness, and add every production failure [2g][15][16].
9. **Benchmarks are for choosing components; your golden set is for shipping.** BEIR, MTEB, BRIGHT, CRAG and FlashRAG tell you which retrievers and methods are worth trying. Public-leaderboard gains often do not transfer, as BEIR and BRIGHT show [6][25].
10. **Instrument with OpenTelemetry from day one.** Retrieval spans with chunk IDs and scores make offline replay, online evaluation and dataset growth possible. Every major 2025-2026 tool (TruLens, Phoenix, Langfuse) is built on this [5][27][28].

## References

1. Es, S., James, J., Espinosa-Anke, L., Schockaert, S. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." arXiv / EACL 2024 System Demonstrations, 2023. https://arxiv.org/abs/2309.15217
2. VibrantLabs / Exploding Gradients. "Ragas: Supercharge Your LLM Application Evaluations" (GitHub repository), 2024-2026. https://github.com/explodinggradients/ragas
   - 2a. Ragas docs, "Faithfulness." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md
   - 2b. Ragas docs, "Context Precision." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/context_precision.md
   - 2c. Ragas docs, "Context Recall." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/context_recall.md
   - 2d. Ragas docs, "Answer Relevancy." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/answer_relevance.md
   - 2e. Ragas docs, "Noise Sensitivity." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/noise_sensitivity.md
   - 2f. Ragas docs, "Factual Correctness." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/factual_correctness.md
   - 2g. Ragas docs, "Testset Generation for RAG." https://github.com/explodinggradients/ragas/blob/main/docs/concepts/test_data_generation/rag.md
3. Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." arXiv 2311.09476 (NAACL 2024), 2023. https://arxiv.org/abs/2311.09476
4. Stanford FutureData. "ARES" (GitHub repository). https://github.com/stanford-futuredata/ARES
5. TruEra / Snowflake. "TruLens" (GitHub repository), 2020-2026. https://github.com/truera/trulens
   - 5b. TruLens docs, "The RAG Triad." https://github.com/truera/trulens/blob/main/docs/getting_started/core_concepts/rag_triad.md
6. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2104.08663 ; code and leaderboard: https://github.com/beir-cellar/beir
7. Gao, T., Yen, H., Yu, J., Chen, D. "Enabling Large Language Models to Generate Text with Citations" (ALCE). EMNLP 2023. https://arxiv.org/abs/2305.14627 ; code: https://github.com/princeton-nlp/ALCE
8. Ru, D. et al. (Amazon Science). "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation." NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2408.08067 ; code and meta-evaluation: https://github.com/amazon-science/RAGChecker
9. Vectara. "HHEM-2.1-Open: Hallucination Evaluation Model" (model card), 2024. https://huggingface.co/vectara/hallucination_evaluation_model
10. Tang, L., Laban, P., Durrett, G. "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents." EMNLP 2024. https://github.com/Liyan06/MiniCheck
11. Vectara. "Hallucination Leaderboard" (GitHub repository, updated September 22, 2026). https://github.com/vectara/hallucination-leaderboard
12. Niu, C., Wu, Y., Zhu, J. et al. "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models." ACL 2024 (arXiv 2401.00396). https://arxiv.org/abs/2401.00396 ; data: https://github.com/ParticleMedia/RAGTruth
13. Zheng, L. et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." 2023. https://arxiv.org/abs/2306.05685 ; code: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
14. Wang, P. et al. "Large Language Models are not Fair Evaluators." 2023. https://arxiv.org/abs/2305.17926 ; code: https://github.com/i-Eval/FairEval
15. Yang, X. et al. (Meta). "CRAG: Comprehensive RAG Benchmark." 2024. https://arxiv.org/abs/2406.04744 ; data: https://github.com/facebookresearch/CRAG
16. Chen, J., Lin, H., Han, X., Sun, L. "Benchmarking Large Language Models in Retrieval-Augmented Generation" (RGB). 2023. https://arxiv.org/abs/2309.01431 ; code: https://github.com/chen700564/RGB
17. Tang, Y., Yang, Y. "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries." COLM 2024. https://arxiv.org/abs/2401.15391 ; data: https://github.com/yixuantt/MultiHop-RAG
18. Upadhyay, S., Pradeep, R., Thakur, N., Craswell, N., Lin, J. "UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor." 2024. https://arxiv.org/abs/2406.06519 ; code: https://github.com/castorini/umbrela
19. Pradeep, R. et al. "Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework." 2024. https://arxiv.org/abs/2411.09607 ; Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Lin, J. "The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models." SIGIR 2025. https://arxiv.org/abs/2504.15068 ; code: https://github.com/castorini/nuggetizer
20. Pradeep, R. et al. "Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track." ECIR 2025. https://arxiv.org/abs/2406.16828 ; code: https://github.com/castorini/ragnarok
21. Vectara. "open-rag-eval" (GitHub repository), 2025. https://github.com/vectara/open-rag-eval
22. Microsoft. "BenchmarkQED: Automated benchmarking of RAG systems" (GitHub repository), 2025. https://github.com/microsoft/benchmark-qed
23. Muennighoff, N. et al. "MTEB: Massive Text Embedding Benchmark." 2022. https://arxiv.org/abs/2210.07316
24. Enevoldsen, K. et al. "MMTEB: Massive Multilingual Text Embedding Benchmark." 2025. https://arxiv.org/abs/2502.13595 ; code: https://github.com/embeddings-benchmark/mteb
25. Su, H., Yen, H., Xia, M. et al. "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval." ICLR 2025. https://arxiv.org/abs/2407.12883 ; code: https://github.com/xlang-ai/BRIGHT
26. Galileo. "RAGBench" (GitHub repository and TRACe framework), 2024. https://github.com/rungalileo/ragbench
27. Arize AI. "Phoenix: AI Observability and Evaluation" (GitHub repository). https://github.com/Arize-ai/phoenix
28. Langfuse. "Langfuse: Open Source LLM Engineering Platform" (GitHub repository). https://github.com/langfuse/langfuse
29. Microsoft. "Azure AI Evaluation client library for Python" (README). https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/evaluation/azure-ai-evaluation/README.md
30. Bassani, E. "ranx: A Blazing-Fast Python Library for Ranking Evaluation, Comparison, and Fusion" (ECIR 2022; CIKM 2022). https://github.com/AmenRa/ranx
31. Jin, J., Zhu, Y., Dou, Z. et al. "FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research." WWW 2025 Resource Track. https://arxiv.org/abs/2405.13576 ; code: https://github.com/RUC-NLPIR/FlashRAG
32. Rau, D. et al. (Naver Labs Europe). "BERGEN: A Benchmarking Library for Retrieval-Augmented Generation." 2024. https://arxiv.org/abs/2407.01102 ; code: https://github.com/naver/bergen
33. Confident AI. "DeepEval: The LLM Evaluation Framework" (GitHub repository). https://github.com/confident-ai/deepeval
34. Jia, A. S. et al. (Snowflake AI Research / Stanford). "What Is Your Agent's GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignment." arXiv 2510.08847, 2025. https://arxiv.org/abs/2510.08847
35. Zeta Alpha. "RAGElo: Elo-based evaluation of RAG agents" (GitHub repository). https://github.com/zetaalphavector/RAGElo
36. Kim, S. et al. "Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models." 2024. https://arxiv.org/abs/2405.01535 ; code: https://github.com/prometheus-eval/prometheus-eval
37. Salemi, A., Zamani, H. "Evaluating Retrieval Quality in Retrieval-Augmented Generation" (eRAG). SIGIR 2024. https://arxiv.org/abs/2404.13781 ; code: https://github.com/alirezasalemi7/eRAG
38. Ravi, S. S. et al. (Patronus AI). "Lynx: An Open Source Hallucination Evaluation Model." 2024. https://arxiv.org/abs/2407.08488 ; blog: https://patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model
39. Joren, H. et al. "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems." ICLR 2025. https://arxiv.org/abs/2411.06037 ; blog: https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/
40. Carmel, D. et al. "SIGIR 2025 - LiveRAG Challenge Report." 2025. https://arxiv.org/abs/2507.04942
