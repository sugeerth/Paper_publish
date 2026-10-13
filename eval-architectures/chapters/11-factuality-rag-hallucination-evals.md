# 11. Factuality, Hallucination and RAG Evaluation: From Atomic Facts to Grounding Judges

## Overview

"Hallucination" names at least three different failures, and each needs its own evaluation design:

1. **Parametric (closed-book) factuality.** The model states something about the world that is false, with no source document in play. SimpleQA, TruthfulQA, FActScore, LongFact/SAFE and HalluLens's extrinsic tasks measure this.
2. **Faithfulness or groundedness to a supplied context.** The model summarizes, answers from or rewrites a document and adds or contradicts content. Summarization consistency metrics (FactCC, QAFactEval, SummaC, AlignScore), detectors (HHEM, MiniCheck, Lynx, LettuceDetect), the Vectara Hallucination Leaderboard, RAGTruth and FACTS Grounding measure this.
3. **Attribution or citation correctness.** The model cites a source, and the question is whether the cited passage actually supports the sentence it is attached to. AIS, ALCE, AttrScore and AttributionBench measure this, along with the citation metrics in RAG toolkits.

HalluLens [45] formalizes the first two as *extrinsic* hallucination (content inconsistent with training data that the input cannot support or refute) and *intrinsic* hallucination (content inconsistent with the input context). That split is the main organizing idea of this chapter. An extrinsic eval needs a world-knowledge oracle: a curated answer key, Wikipedia, or a live search engine. An intrinsic eval needs only the context and a strong entailment judge. Most production RAG stacks need both, plus attribution checks.

Almost every system below follows the same pipeline. It decomposes a response into checkable units (atomic facts, claims, triplets, sentences, nuggets or tokens), retrieves or assigns evidence for each unit, verifies each unit with an NLI model, a fine-tuned checker or an LLM judge, and aggregates the verdicts into a precision-, recall- or rate-style score. The designs differ in four choices: unit granularity, evidence source, verifier, and how they treat abstention and length. The comparison table and takeaways at the end of the chapter set these choices side by side.

A note on sourcing. During this research pass only GitHub-hosted primary sources (official code repositories, READMEs, grader source code) could be fetched; arXiv, vendor blogs and Hugging Face were unreachable. Facts taken from repositories are cited directly. A few figures come from the papers themselves (SimpleQA, FACTS Grounding, SimpleQA Verified, "Why Language Models Hallucinate", the HHEM model card) and could not be re-fetched in this pass. Those are marked "(paper-reported, approximate)" and should be checked against the cited URL before being quoted.

```
                 GENERIC FACTUALITY / FAITHFULNESS EVALUATION PIPELINE
 +----------+   +-------------+   +----------------+   +-------------+   +--------+
 | response |-->| decompose   |-->| evidence       |-->| verifier    |-->| aggre- |
 | (+prompt,|   | atomic facts|   | - answer key   |   | - LLM grader|   | gate:  |
 | context) |   | claims      |   | - Wikipedia DB |   | - NLI model |   | prec., |
 +----------+   | triplets    |   | - web search   |   | - small     |   | F1@K,  |
                | sentences   |   | - given context|   |   checker   |   | rate,  |
                | tokens      |   | - cited passage|   +-------------+   | AUROC  |
                +-------------+   +----------------+          |          +--------+
                      ^                                       |
                      +------ abstention / relevance filter --+
```

---

## 1. Short-form parametric factuality

### 1.1 TruthfulQA (Oxford / OpenAI, 2021; revised January 2025)

**Problem.** Larger models trained to imitate web text can reproduce common misconceptions, so scaling might make answers *less* truthful. TruthfulQA, "Measuring How Models Imitate Human Falsehoods" (Lin, Hilton, Evans), was built to probe this [16][17].

**Architecture.** The set has 817 adversarially written questions across categories such as misconceptions, law and health (817 is the count used in later papers' tables [50]). It has two task formats [17]:

- **Generation.** The model writes a 1-2 sentence answer. The answer is scored for truthfulness and informativeness by *GPT-judge* and *GPT-info*, which are fine-tuned GPT-3 classifiers, with BLEURT, ROUGE and BLEU reported as similarity metrics against the reference true and false answers. BLEURT is the recommended similarity metric.
- **Multiple choice.** MC1 picks the single true answer from 4-5 options. MC2 measures the normalized probability mass the model assigns to the set of true answers.

**January 2025 revision.** The authors added a **binary multiple-choice** format: the "Best Answer" against the "Best Incorrect Answer", shown as (A)/(B) in random order. They recommend it over MC1/MC2 while noting that most models score about the same on both. They also removed questions that had become invalid or that conflicted with how LLMs are now used (for example, questions about browsing), and dropped the "Indexical Error: Time" category. They state that the remaining questions "have not been fully validated" [17].

**Failure modes and lessons.**

- *Judge drift.* GPT-judge depended on OpenAI fine-tuning of a now-retired model family. The repo warns that the judges are not externally available and should not be expected to generalize [17]. Benchmarks whose metric depends on a vendor-hosted fine-tuned judge age badly.
- *Exploitable MC formats.* The authors moved to a binary format because multiple-choice variants let models exploit answer-option artifacts.
- *Contamination.* TruthfulQA questions are widely reproduced, and the benchmark is now mostly used as a regression check or ablation target. Many decoding papers still report MC1/MC2/%Truth×Info on it [50].

### 1.2 SimpleQA (OpenAI, October 2024)

**Problem.** Frontier models had saturated TriviaQA-style QA, and long-form factuality is hard to grade. SimpleQA narrows the problem to short, fact-seeking questions with a single, indisputable, time-invariant answer, so that grading is almost mechanical and the remaining signal is about *knowing what you know* [3][4].

**Data construction (paper-reported, approximate).** The set has 4,326 questions written by AI trainers. Each question was answered independently by a second trainer, and only questions where both agreed were kept. Questions were also filtered adversarially so that at least one GPT-4 sample answered incorrectly. The authors estimate the intrinsic label error rate at about 3% [3][4].

**Grading architecture.** The reference grader is open source in `simple-evals` [6]. A prompted LLM grader receives (question, gold target, predicted answer) and returns one of three grades:

- **A: CORRECT.** The gold target is fully contained and nothing contradicts it. Hedging is allowed: "Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions."
- **B: INCORRECT.** Any contradicting factual statement, *including hedged* ones ("I think it's either Malia and Sasha. Or it could be Malia and Jackie").
- **C: NOT_ATTEMPTED.** The gold information is missing and nothing contradicts it ("I don't know"; "I need more context").

Numeric answers must match the gold answer to its last significant figure, and vague approximations count as not attempted. If the grader's reply contains no A, B or C, the grade defaults to C. The grader model is a pluggable `SamplerBase` [6].

**Metrics.** The code computes `is_correct`, `is_incorrect` and `is_not_attempted` (rates over all questions), `accuracy_given_attempted = correct / (correct + incorrect)`, and an **F-score**, the harmonic mean of overall correct rate and accuracy-given-attempted [6]. The F-score rewards both answering and being right when answering. The paper also measures **calibration** in two ways: stated confidence against accuracy, and answer frequency over repeated sampling against accuracy [3][4].

```
     question ---> model under test ---> predicted answer
                                              |
     gold target -----------------------------+
                                              v
                                    +-------------------+
                                    | LLM grader prompt |  few-shot rubric with
                                    | (A / B / C)       |  Obama-children examples
                                    +-------------------+
                                              |
                        correct% / incorrect% / not_attempted%
                        acc|attempted = C/(C+I);  F = HM(correct%, acc|attempted)
```

**Reported scores.** The final `simple-evals` table, frozen in July 2025 when the repo stopped tracking new models, lists SimpleQA accuracy as follows [5]:

| Model | SimpleQA accuracy |
|---|---|
| GPT-4.5-preview | 62.5 |
| o3 (high / medium-low) | 48.6 / 49.4 |
| o1 | 42.6 |
| GPT-4.1 | 41.6 |
| gpt-4o-2024-08-06 | 40.1 |
| Claude 3.5 Sonnet | 28.9 |
| o4-mini | 19.3-20.2 |
| gpt-4o-mini | 9.5 |

The o3/o4-mini scores were measured *without tools* [5]. SimpleQA is a closed-book parametric-knowledge test, and with search enabled the benchmark is close to trivial. That is why OpenAI positioned BrowseComp as the browsing-agent successor [5].

**Lessons.**

1. The three-way grade is the key design choice. It makes abstention a measurable behavior rather than a failure.
2. Hedges are graded by *content*, not by tone: a hedge that contains a wrong candidate counts as wrong.
3. Short, single-answer targets turn the grader into a near-deterministic string-equivalence judge. That makes grader error small compared with model differences.
4. Accuracy correlates strongly with model size, so SimpleQA also serves as a proxy for how much knowledge a model stores.

### 1.3 SimpleQA Verified (Google DeepMind / Google Research, 2025)

**Problem.** An audit of SimpleQA found noisy or incorrect labels, topical skew and redundant near-duplicate questions [7].

**Design (paper-reported, approximate).** The authors filtered the set down to **1,000** prompts through de-duplication, topic re-balancing and source reconciliation, and they improved the autorater prompt. The paper reports Gemini 2.5 Pro reaching an F1 of about 55.6 on the verified set [7].

**Lesson.** Popular benchmarks need versioned "verified" subsets once frontier scores approach the label-noise ceiling. SWE-bench Verified followed the same path for coding.

### 1.4 Incentives: "Why Language Models Hallucinate" (OpenAI, September 2025)

Kalai, Nachum, Vempala and Zhang argue that hallucinations persist partly because almost every leaderboard uses **binary accuracy**, which rewards guessing over abstaining [8]. Their proposed fix is an evaluation-design change rather than a new dataset. Mainstream benchmarks would state an explicit confidence target in the instructions (answer only if more than *t* confident), penalize errors (by t/(1-t) points), and give abstentions zero (paper-reported, approximate). OpenAI's accompanying post used SimpleQA to show a newer reasoning model with a much higher abstention rate and far fewer errors than an older model with slightly higher raw accuracy. This is exactly the trade-off the SimpleQA three-way grade exposes [8].

**Design implication.** Any factuality eval that reports only accuracy will, over many optimization cycles, select for confident guessers. Report error rate and abstention rate as first-class metrics, or use a penalized score.

---

## 2. Long-form parametric factuality

### 2.1 FActScore (UW / AI2 / Meta, EMNLP 2023)

**Problem.** Long generations mix true and false statements, so a binary judgment of a whole response is too coarse and human annotation is expensive. FActScore measures **factual precision**: the fraction of atomic facts in a generation that a given knowledge source supports [1][2].

**Architecture.**

1. **Atomic fact generation.** An LLM (originally InstructGPT/ChatGPT) splits each sentence into short statements that each carry one piece of information.
2. **Knowledge source.** A Wikipedia dump (2023/04/01 by default). Users can register any `.jsonl` corpus with `title` and `text` fields. Indexing all of English Wikipedia takes about 8 hours [2].
3. **Verification.** Passages are retrieved for the topic entity, and a verifier LM decides whether each fact is supported. The two recommended estimators are `retrieval+ChatGPT` and `retrieval+llama+npm`. The second uses an Inst-LLaMA-7B plus a nonparametric masked-LM probability. The two have a **0.99 Pearson correlation** [2].
4. **Abstention and length handling.** Abstentions are detected and reported as a `respond_ratio`. A length penalty `gamma` (default 10) reduces the score when a response yields fewer facts than the threshold, which stops models from gaming precision by saying almost nothing. `init_score` is the score without the penalty [2].

```
 topic (person) --> LM writes bio --> sentence split --> LLM: atomic facts
                                                                 |
                       Wikipedia DB (BM25/GTR retrieval by topic)|
                                     |                           v
                                     +------> verifier (ChatGPT | LLaMA+NPM)
                                                       |
                         FActScore = mean_over_responses( supported / total facts )
                                    x length penalty (gamma), plus respond_ratio
```

**Numbers.** The released estimator results show the trade-off between coverage and precision clearly [2]:

| Model | % respond | # facts / response | FActScore (A) | FActScore (B) |
|---|---|---|---|---|
| GPT-4 | 88.2 | 60.8 | 73.1 | 59.9 |
| ChatGPT | 84.2 | 37.0 | 71.6 | 60.4 |
| Alpaca 65B | 100.0 | 17.1 | 55.6 | 46.3 |
| Vicuna 13B | 76.6 | 50.9 | 46.6 | 40.7 |
| Dolly 12B | 100.0 | 24.6 | 21.7 | 17.1 |

The paper's human-annotated biographies reported ChatGPT at about 58% precision, and the estimator approximated human scores to within about 2 points of error rate (paper-reported, approximate) [1]. Running costs about **$1 of API cost per 100 sentences** [2].

**Failure modes.**

- Precision only: a short, safe answer scores well. The length penalty and respond ratio are only partial fixes.
- Restricted to entities that have a Wikipedia page. Facts that are true but missing from Wikipedia are marked unsupported.
- Atomic-fact splitting is itself LLM-dependent and not unique, so different decomposers change the denominator.
- Estimator (A) and (B) give different absolute levels (73.1 against 59.9 for GPT-4), so only compare models scored with the same estimator.

### 2.2 LongFact and SAFE (Google DeepMind, 2024)

**Problem.** FActScore is tied to Wikipedia and to biographies. Google DeepMind wanted open-domain long-form factuality across many topics, with a verifier that can reach the whole web [9][10].

**LongFact.** 2,280 fact-seeking prompts that require long-form responses, generated with GPT-4 and split into object-centric and concept-centric subsets across 38 topics (topic count paper-reported) [10].

**SAFE (Search-Augmented Factuality Evaluator)** extends the FActScore decomposition into an agentic verifier [11]:

1. **Split** the response into atomic facts, reusing FActScore's `get_atomic_facts` code.
2. **Revise** each fact to be self-contained, replacing pronouns and vague references with the entities they refer to.
3. **Relevance check.** Decide whether each fact is relevant to answering the prompt, in the context of the whole response.
4. **Search-and-rate loop.** For each relevant fact, the LLM issues Google Search queries through the Serper API. The defaults are `num_searches=3` and `max_steps=5`, at temperature 0.5. The LLM reasons over the snippets and labels the fact *supported* or *not supported* [11].

Outputs per response are the counts of supported, not-supported and irrelevant facts.

```
 response --> [split atomic] --> [make self-contained] --> [relevant?] --no--> irrelevant
                                                               | yes
                                                               v
                                         +---- LLM agent: issue query --> Serper/Google
                                         |        ^                          |
                                         |        +------ read snippets <----+
                                         |        (up to 5 steps, 3 results each)
                                         v
                                 supported / not supported
                                         |
       F1@K:  Prec = S/(S+N);  Recall@K = min(S/K, 1);  F1@K = HM(Prec, Recall@K)
```

**F1@K.** This metric adds *recall* to factuality. K is a "human-preferred" number of supported facts; the paper reports K=64 (median response length) and K=178 (maximum), paper-reported. A response with fewer than K supported facts loses recall, which counteracts the "say less, be precise" exploit [9][10].

**Agreement and cost.** On about 16k individual facts SAFE agreed with crowd-sourced human annotators **72%** of the time. On a random sample of 100 disagreements SAFE was judged correct **76%** of the time, against 19% for humans [11]. With GPT-3.5-Turbo the cost is about **$0.20 per prompt-response pair** (about $0.10 in Serper credits plus $0.10 in LLM calls, up to $0.40 for long responses), against about **$4.00 per pair** for FActScore's human annotation. The paper summarizes this as more than 20x cheaper [11].

**Failure modes.** Search results are non-stationary, so scores are not perfectly reproducible over time. The "humans lost 76% of disagreements" result reflects the crowd-worker baseline (annotators with limited time and Wikipedia-only evidence) rather than expert raters. Relevance filtering is itself an LLM judgment that changes the denominator.

### 2.3 VeriScore, FacTool, HALoGEN and HalluLens: successors

- **VeriScore** (UMass, 2024) keeps the three stages of extract, search and verify, but extracts only *verifiable* claims, so opinions and advice are not penalized. It supports binary (supported/unsupported) or ternary (supported/contradicted/inconclusive) labels and ships fine-tuned open extractor and verifier models alongside GPT-4-class defaults: gpt-4-0125-preview for extraction and gpt-4o for verification, with 10 Serper results per claim by default [43].
- **FacTool** (GAIR, 2023) generalizes the pipeline to four domains with domain-appropriate tools: knowledge QA (search), code generation (execution), math (calculation) and scientific literature review (scholar search to catch hallucinated citations). Each claim gets a label with reasoning, error and correction [44].
- **HALoGEN** (ACL 2025) scales atomic-fact verification to 10,923 prompts across 9 tasks: code packages, scientific attribution, summarization, simplification, biographies, historical events, false presuppositions, and binary and numerical rationalization. Each task has its own decomposer and verifier. It adds a Type A/B/C error taxonomy tied to training-data provenance [46][50].
- **HalluLens** (Meta FAIR, ACL 2025) is built around contamination resistance. Its tasks are **dynamically generated** at evaluation time: PreciseWikiQA for short answers, LongWiki for long form, and NonExistentRefusal, which uses plausible but fictitious entities. Metrics are Hallu (incorrect among non-refused answers), Correct (refusals count as incorrect) and False Accept (the model does not refuse a nonexistent entity), averaged over three trials [45].

**Lesson.** Long-form factuality has settled on "claim decomposition + retrieval + LLM verifier." Current work competes on which claims count (verifiable, relevant), evidence breadth (Wikipedia, web, code execution), contamination resistance (dynamic generation) and refusal handling (nonexistent entities).

---

## 3. Grounded faithfulness: summarization consistency metrics

Before LLM judges, faithfulness research came mostly from abstractive summarization. The architectures from this period are still the cheapest production detectors.

### 3.1 FactCC (Salesforce, 2019): weak supervision from synthetic corruptions

FactCC trains a BERT-style classifier on **rule-based transformations** of source sentences: entity swaps, number swaps, negation, pronoun swaps, paraphrase and noise. It is trained jointly on three tasks: classify consistent or inconsistent, extract the supporting span in the source, and extract the inconsistent span in the summary. The authors found that the auxiliary span tasks helped human verifiers [26]. This established the recipe of training a checker on synthetic negatives.

### 3.2 QA-based metrics: QAGS, QuestEval, QAFactEval

QA-based metrics generate questions from the summary, answer them against both summary and source, and compare the answers. QAFactEval (Salesforce, NAACL 2022) ran an extensive ablation of question generation, answerability filtering and answer-overlap scoring (using the LERC learned answer-overlap metric). It added question-consistency filtering and reported a **14% average improvement over previous QA-based metrics on the SummaC benchmark** [27].

**Trade-off.** QA metrics are interpretable, since you can see which question failed. They are also slow (several seq2seq passes) and fragile when question generation fails.

### 3.3 SummaC (TACL 2022): NLI at the right granularity

Off-the-shelf NLI models did poorly on summary consistency because they were trained on sentence pairs, not document-summary pairs. SummaC splits both document and summary into sentences and builds an NLI **score matrix** over (document sentence, summary sentence) pairs. SummaC-ZS takes the max over document sentences and averages over summary sentences. SummaC-Conv learns a 1-D convolution over the histogram of each column's scores, and the repo recommends it [28]. The accompanying **SummaC Benchmark** standardizes six summary-consistency datasets from CNN/DM and XSum into binary classification [28].

```
          doc sents d1..dM
        +-----------------+
   s1   | e11 e12 ... e1M |--> ZS: max_j  | Conv: histogram -> conv -> score
   s2   | e21 ...         |--> ...
   ..   |                 |
   sN   | eN1 ...     eNM |-->
        +-----------------+
   summary score = mean over summary sentences
```

### 3.4 TRUE and TrueTeacher (Google, 2022-2023): meta-evaluation and synthetic labels

TRUE standardized **11 datasets** across summarization (FRANK, SummEval, MNBM, QAGS-C/X), dialogue (BEGIN, Q², DialFact), fact verification (FEVER, VitaminC) and paraphrase (PAWS) into one binary scheme. It argued for **ROC AUC** as the threshold-free meta-metric, since accuracy, precision and recall all need a threshold and the default of 0.5 suits few metrics [29]. Google released a **T5-11B NLI model** trained on the TRUE mixture (`google/t5_xxl_true_nli_mixture`) [29]. Later systems reused it as the default verifier; for example, ALCE uses it for citation checking. **TrueTeacher** then used LLMs to *label* large numbers of model-generated summaries and distilled those labels into smaller student checkers [51].

### 3.5 AlignScore (ACL 2023): one alignment function from 4.7M examples

AlignScore trains RoBERTa (125M base / 355M large) as a general **alignment** function on about **4.7M** examples from 7 task families: NLI, QA, paraphrase, fact verification, IR, semantic similarity and summarization. At inference the context is chunked into pieces of about 350 tokens and the claim is split into sentences. Each sentence takes its maximum alignment over chunks, and these maxima are averaged [30]. AlignScore-large scores **88.6 AUC on SummaC** and **83.8 on TRUE**. Its average Spearman correlation on SummEval and QAGS (59.3) beats G-Eval with GPT-4 (57.6) and ChatGPT (49.6) [30].

**Lesson.** A 355M-parameter model trained on diverse alignment data can match much larger prompted LLM judges on faithfulness, at a tiny fraction of the cost. Section 4 shows the same result repeated with newer models.

---

## 4. Hallucination detectors as production components

### 4.1 Vectara HHEM and the Hallucination Leaderboard

**Problem.** Vectara wanted a public, regularly refreshed ranking of how often LLMs add unsupported content when acting as the "generator" in RAG. It also wanted a cheap classifier customers could run inline [12][14].

**Detector.** HHEM (Hughes Hallucination Evaluation Model) is a cross-encoder classifier that scores a (premise, hypothesis) pair between 0 and 1, where higher means more consistent. The open variant **HHEM-2.1-Open** is based on Flan-T5 and is distributed on Hugging Face and Kaggle. The model card claims better results than GPT-3.5-Turbo and GPT-4 as zero-shot detectors on AggreFact and RAGTruth, CPU-friendly inference, and no fixed short context limit (paper/model-card-reported, approximate) [13]. The leaderboard itself uses the commercial **HHEM-2.3**, which is also multilingual [12]. Downstream toolkits embed HHEM directly. Ragas offers `FaithfulnesswithHHEM`, which swaps the LLM verification step for HHEM-2.1-Open on CPU with batch size 10 [39]. Vectara's Open RAG Eval uses it as its hallucination metric (truncating input at 8,192 characters, 2 CPU threads) [42].

**Leaderboard architecture** [12]:

```
  private corpus (>7,700 articles, 50 to 24K words; news, tech, science,
  medicine, legal, sports, business, education)
           |
           v
  each LLM (temperature 0) summarizes with fixed prompt:
   "Summarize using only the information in the given passage. Do not infer...
    never exceed 20 percent of the original text's length...
    else respond 'I am unable to summarize this text.'"
           |
           +--> refusals / very short outputs --> Answer Rate
           v
  keep only documents that ALL models summarized (common subset)
           |
           v
  HHEM-2.3 (source, summary) -> consistent?  --> Factual Consistency Rate
                                                 Hallucination Rate = 100 - FCR
                                                 + Avg summary length (words)
```

**Key techniques.**

- The dataset is **kept private** to avoid overfitting. It was refreshed from an earlier, older dataset, and earlier versions are kept as git branches (`hhem-1.0-final`, `hhem-2.3-old-dataset`) [12].
- Evaluation runs on the **common subset** of documents that every model answered, so refusal-happy models cannot improve their rate by skipping hard documents. Answer rate and average length are published next to the hallucination rate to expose degenerate strategies [12].
- Vectara justifies the choice of summarization as a proxy: open-ended hallucination cannot be measured without knowing training data, whereas consistency with a provided document can. Summarization is also what a RAG generator does with search results [12].

**Scale and recent numbers.** As of the 22 September 2026 update, the lowest hallucination rates were antgroup/finix_s1_32b at 1.8% (99.5% answer rate) and gpt-5.4-nano at 3.1%. Several frontier "thinking" configurations sat well above 10%, for example gpt-5.1-high at 12.1%. Small models such as Phi-4-mini-instruct were above 20% [12]. The spread is the point: reasoning-heavy models do not automatically hallucinate less when summarizing.

**Failure modes.** The repo says plainly that the metric captures *consistency in summarization*, not summary quality, and not every kind of hallucination [12]. A single detector serving as the judge creates detector-specific blind spots, and HHEM's own errors become the leaderboard's errors. To measure this, Vectara built **FaithBench** (NAACL 2025), a human-annotated set of *hard* summarization hallucinations. Samples were chosen for maximal disagreement among five detectors (GPT-4o, GPT-3.5-Turbo, TrueTeacher, TRUE NLI and HHEM-2.1-Open). Labels are Consistent, Questionable, Benign hallucination or Unwanted (intrinsic/extrinsic) hallucination, with span-level notes from several annotators [15].

### 4.2 MiniCheck and LLM-AggreFact (UT Austin, EMNLP 2024)

**Problem.** Checking every sentence of RAG output with GPT-4 is too expensive.

**Architecture.** The input is (document, single sentence) and the output is a support probability. Multi-sentence claims should be split first. The checkpoints are RoBERTa-large, DeBERTa-v3-large, Flan-T5-large (770M) and **Bespoke-MiniCheck-7B**. The small models are trained on 14K **synthetic** examples built by two procedures (C2D and D2C) that create claims needing multi-sentence reasoning over a document [31]. The repo states that MiniCheck-Flan-T5-Large "reaches GPT-4 performance" and that Bespoke-MiniCheck-7B is state of the art regardless of size. The paper also reports a cost about two orders of magnitude lower than GPT-4 (paper-reported) [31].

**Benchmark.** **LLM-AggreFact** unifies 11 human-labeled datasets of claims, mostly LLM-generated, covering closed-book and grounded generation. Its leaderboard tracks 32+ checkers, including Qwen-2.5, Llama-3.x, Mistral Large 2 and Claude 3.5 [31].

**Engineering detail.** Enabling vLLM **automatic prefix caching** (reusing the document's KV cache across the many claims checked against it) cut a 29K-example run on one A6000 from about 55 to 30 minutes with no change in accuracy [31]. This is a useful pattern for any claim-level verifier where many claims share one context.

### 4.3 Lynx and HaluBench (Patronus AI, 2024)

Lynx is a fine-tuned Llama-3-70B-Instruct (with a smaller 8B variant) trained on about 2,400 samples from RAGTruth, DROP, CovidQA and PubMedQA, using GPT-4o-generated reasoning traces. It outputs a binary, response-level faithfulness judgment with an explanation. **HaluBench** samples examples (on the order of 500 per source, approximate) from CovidQA, PubMedQA, DROP and FinanceBench and adds perturbation-based hallucinated answers generated with GPT-4o [33][50]. The design shows the "reasoning judge distilled into an open model" pattern, which suits enterprises that cannot send context to third-party APIs.

### 4.4 RAGTruth and token-level detectors (LettuceDetect)

**RAGTruth** (2024) is the main *training and evaluation* corpus for RAG hallucination detection. It contains about 18k responses (17,790) to 2,965 source instances across QA, summarization (CNN/DM and recent news) and data-to-text (Yelp). The responses come from six LLMs (GPT-3.5/4, Llama-2 7B/13B/70B, Mistral-7B). It has **word-level span annotations**: 7,664 hallucinated responses containing 14,289 spans, with type labels such as "Evident Baseless Info", plus flags for implicit-true content and incorrect refusals [32].

**LettuceDetect** (2025-2026) frames detection as **token classification**. A ModernBERT encoder (4K context; EuroBERT 8K for multilingual) labels each answer token as supported or hallucinated, given context and question. This follows the encoder-based Luna design. The v2 release (June 2026) adds a generative Qwen-2B span tagger, an mmBERT encoder and "TinyLettuce" models of 17-68M parameters. It reports span-F1 on the RAGTruth slice of 0.574 (Qwen-2B) and 0.528 (mmBERT-base) [34]. Token-level output lets products highlight the offending span instead of just rejecting the answer.

### 4.5 Reference-free detection: SelfCheckGPT and semantic entropy

When no context or answer key exists, consistency across samples is the only signal.

- **SelfCheckGPT** (Cambridge, EMNLP 2023) samples N extra responses (temperature 1.0) and scores each sentence of the main response by how well the samples support it. Variants use BERTScore, MQAG question answering, n-gram, **NLI** (DeBERTa-v3-large MNLI; the recommended practical variant) and **LLM-prompt**. On 238 annotated GPT-3 WikiBio passages, SelfCheck-Prompt with gpt-3.5-turbo reached 93.42 NonFact AUC-PR, against 92.50 for NLI and 83.21 for the grey-box Avg(-logP) baseline [35].
- **Semantic entropy** (Oxford, *Nature* 2024) clusters sampled answers by bidirectional entailment and computes entropy over *meanings* rather than token strings. It was evaluated with AUROC/AURAC on TriviaQA, SQuAD, BioASQ, NQ-Open and SVAMP, plus a new FactualBio set for biographies [36][50]. Reproduction costs about $5-30 of API per sentence-length run, and 70B models need 2xA100-80GB [36].

**Meta-evaluation warning.** "The Illusion of Progress" (EMNLP 2025) showed that ROUGE-based correctness labels in QA hallucination-detection papers inflate results. Under human-aligned LLM-judge labels, detectors such as Perplexity, EigenScore and eRank lost up to **45.9% AUROC**, and simple length heuristics matched Semantic Entropy [49]. Before trusting any detector number, check how the ground-truth labels were produced.

---

## 5. Grounding benchmarks for frontier models: FACTS Grounding

**Problem.** Summarization-consistency leaderboards use short outputs and single detectors. Google DeepMind wanted a leaderboard of a model's ability to produce **long-form responses fully grounded in long input documents** (up to about 32k tokens) across realistic enterprise tasks, scored by several frontier judges to reduce self-preference [18][19].

**Architecture (paper-reported, approximate).**

- **Data.** 1,719 examples, split into public (860) and private held-out (859) sets. Each pairs a system instruction ("answer only from the document"), a long document (finance, technology, retail, medicine, law) and a user request (summarize, Q&A, rewrite) [18][19].
- **Two-phase judging.** (1) An **eligibility** check: does the response actually address the user's request? Responses that dodge the request (for example, a vague but technically grounded answer) are disqualified. (2) A **grounding** check: is every claim fully supported by the document? [18]
- **Multi-judge ensemble.** Gemini 1.5 Pro, GPT-4o and Claude 3.5 Sonnet each judge, and the final score averages across judges. This explicitly reduces a judge's bias toward its own model family. Judge prompt templates were selected by agreement with human raters [18].
- **Leaderboard** hosted on Kaggle with the private split kept back to resist overfitting [19]. Google DeepMind later extended it into a broader FACTS benchmark suite covering parametric, search-based and multimodal factuality alongside grounding; check the Kaggle FACTS pages for the current composition.

```
 (system instr, long doc <=32k tok, user request)
              |
              v
       model under test --> response
              |
   +----------+-----------+----------------+
   v                      v                v
 Judge A (Gemini)     Judge B (GPT-4o)  Judge C (Claude)
   | eligible?            | eligible?      | eligible?     -- no --> score 0
   | fully grounded?      | grounded?      | grounded?
   +----------+-----------+----------------+
              v
     mean over judges, mean over examples -> FACTS Grounding score
```

**Lessons.** The eligibility gate is the key design choice. A pure groundedness metric can be maximized by saying nothing substantive, much like precision-only FActScore. The judge ensemble trades three times the judging cost for lower variance and less self-preference.

---

## 6. Attribution and citation evaluation

### 6.1 AIS (Google, 2021): the definitional anchor

"Attributable to Identified Sources" (Rashkin et al.) defines a two-stage human annotation. First, is the sentence interpretable on its own? Second, would a generic hearer agree that "according to source P, s" holds? Most later automatic attribution metrics ("AutoAIS") approximate the second judgment with an NLI model [22]. ExpertQA later reported AutoAIS precision/recall/F1 against expert labels alongside FActScore [50].

### 6.2 Verifiability audit of generative search engines (Stanford, 2023)

Liu, Zhang and Liang audited four commercial generative search engines (Bing Chat, NeevaAI, perplexity.ai, YouChat) with human annotators. They defined **citation recall** (the fraction of verification-worthy sentences fully supported by their citations) and **citation precision** (the fraction of citations that support their sentence). The engines' responses were fluent, but only about half of the sentences were fully supported, and about a quarter of citations did not support their sentence (paper-reported, approximate) [23]. These two definitions became the standard.

### 6.3 ALCE (Princeton, EMNLP 2023): automatic citation benchmark

ALCE evaluates LLMs that must answer *and cite* retrieved passages on **ASQA**, **QAMPARI** and **ELI5**. It reports three axes [20][21]:

- **Fluency:** MAUVE.
- **Correctness:** exact-match recall of short answers (ASQA), recall-5 (QAMPARI), and claim recall via NLI (ELI5).
- **Citation quality:** **citation recall** (is each statement entailed by the concatenation of its cited passages?) and **citation precision** (is each individual citation needed, i.e. not irrelevant?). Both are judged by the TRUE T5-11B NLI model [20][29].

```
 statement s_i  with citations {c1, c2}
    recall_i    = NLI( concat(c1,c2) => s_i )
    precision_j = c_j is "irrelevant" if  NLI(c_j => s_i) fails AND
                  NLI(other cites => s_i) still holds without c_j
```

### 6.4 AttrScore and AttributionBench (Ohio State, 2023-2024)

**AttrScore** treats attribution as three-way classification: *attributable*, *extrapolatory* (the reference lacks enough information) and *contradictory*. It trains judges on data repurposed from QA, fact-checking, NLI and summarization, and evaluates on a simulated set and a real generative-search set (AttrEval-GenSearch). Zero-shot GPT-4 reached 85.1 overall on GenSearch but only 55.6 on the harder simulated set, and contradiction detection was the weakest class (23.2-45.0) [24]. **AttributionBench** (ACL Findings 2024) unifies several attribution datasets into one binary task with in-domain and out-of-domain test splits. The paper's title question, "How Hard is Automatic Attribution Evaluation?", has the answer: hard. Fine-tuned NLI-style checkers and LLMs remain far from solved, especially on fine-grained information mismatches [25].

### 6.5 Citation metrics in production toolkits

Vectara's **Open RAG Eval** implements a citation metric judged by an LLM (GPT-4o by default). Each answer part with its cited passages is rated Full (1.0), Partial (0.5) or No support (0.0). Weighted precision is the sum of citation scores over citations, weighted recall is the sum of part scores over parts, and F1 combines the two [42].

---

## 7. End-to-end RAG evaluation frameworks

Section 4 covered the *verifier*. RAG eval frameworks wrap verifiers into a diagnostic that **attributes failures to the retriever or to the generator**.

### 7.1 The RAG triad (TruLens) and Ragas

TruLens's **RAG triad** places an eval on each edge of query, context and answer [40]:

```
                 query
                /     \
   context     /       \   answer relevance
   relevance  /         \
          context ------- answer
                groundedness
     (claim-split answer; find supporting evidence for each claim in context)
```

Its argument is that if all three pass, the app is "hallucination free up to the limit of its knowledge base" [40]. **Ragas** computes **faithfulness** = (claims supported by retrieved context) / (total claims). For example, "born in Germany on 20 March 1879" against a context giving 14 March 1879 decomposes into two claims, one of which is supported, so the score is 0.5. Ragas offers an HHEM-backed variant for cheaper verification [39].

### 7.2 RAGChecker (Amazon, NeurIPS 2024 D&B)

RAGChecker runs **claim-level entailment in both directions** between response, ground-truth answer and retrieved chunks [38]:

- **Overall:** claim precision, recall and F1 of the response against the ground-truth answer.
- **Retriever diagnostics:** *claim recall* (fraction of ground-truth claims covered by the retrieved context) and *context precision*.
- **Generator diagnostics:** *context utilization*, *noise sensitivity* (in relevant and irrelevant chunks), *hallucination* (incorrect claims supported by no chunk), *self-knowledge* (correct claims not found in context) and *faithfulness*.

The example output in the repo, for instance hallucination 4.2 and self-knowledge 25.0, shows that a quarter of correct claims can come from parametric memory rather than retrieval. That is invisible to a faithfulness-only metric [38]. RAGChecker grew out of Amazon's **RefChecker**, which extracts (subject, predicate, object) **triplets** and labels each one entailment, neutral or contradiction under three settings: zero context, noisy context (RAG) and accurate context (summarization). RefChecker's benchmark has 300 prompts and about 2.1k human-annotated responses from seven LLMs [37].

### 7.3 ARES (Stanford, 2023): statistically calibrated judges

ARES generates synthetic queries and answers from the target corpus, fine-tunes lightweight judges for **context relevance, answer faithfulness and answer relevance**, and then uses **prediction-powered inference (PPI)**. PPI combines judge predictions on many unlabeled examples with a small human-labeled set (at least 50 examples, ideally several hundred; the quick-start uses 300) to produce **confidence intervals** rather than point estimates [41]. This addresses a gap in most LLM-judge RAG evals, which report a judge's mean with no account of judge error.

### 7.4 TREC-style nugget evaluation: Open RAG Eval

Vectara's Open RAG Eval implements the TREC 2024 RAG Track metrics, which need **no golden answers or golden chunks** [42]:

- **UMBRELA** relevance judging (0-3 scale, GPT-4o by default), with derived Precision@K, AP@K and MRR (relevant means a score of 2 or more).
- **AutoNuggetizer.** Up to 30 atomic "nuggets" (1-12 words) are extracted from relevant passages over up to 5 refinement iterations, and the top 20 are kept. Each nugget is labeled *vital* or *okay* and assigned to the answer as support (1), partial (0.5) or none (0). The scores reported are All, Vital, Weighted (vital + 0.5·okay) and Strict.
- **Citation F1** (Section 6.5), **HHEM hallucination**, and **no-answer detection** for a "questions answered" rate.

Nuggets are the *recall* counterpart to claim-level faithfulness. They ask whether the answer contains the important facts that the retrieved evidence supports, which a faithfulness score never checks.

---

## 8. Meta-benchmarks for factuality evaluators

How good are the checkers themselves? Several "evaluating the evaluators" sets exist:

- **FELM** (HKUST, 2023): 847 ChatGPT responses across world knowledge, science/tech, writing/recommendation, reasoning and math, with **segment-level** error labels, reference links and error reasons (4,427 labels) [48].
- **HaluEval** (RUC, 2023): 35K samples, made up of 10K each of QA (HotpotQA), dialogue (OpenDialKG) and summarization (CNN/DM) with ChatGPT-generated, filtered hallucinated counterparts, plus 5K human-labeled general queries [47]. Its synthetic negatives are easier than natural errors. Detectors scoring near-perfect AUC on HaluEval-QA while doing poorly on HaluBench subsets is a known pattern [50].
- **LLM-AggreFact** (Section 4.2), **TRUE** and **SummaC** (Section 3), **FaithBench** (Section 4.1), **AttributionBench** (Section 6.4) and **RAGTruth** (Section 4.4) round out the set.

---

## 9. Comparison table

| System | Hallucination type | Unit | Evidence source | Verifier | Headline metric | Abstention / length handling | Cost profile |
|---|---|---|---|---|---|---|---|
| TruthfulQA [17] | Extrinsic (misconceptions) | Whole answer / MC choice | Reference true/false answers | Fine-tuned GPT-3 judges; MC likelihood | %True, %Info, MC1/MC2, binary MC | Informativeness score | Cheap; judges no longer available |
| SimpleQA [3][6] | Extrinsic, short-form | Whole short answer | Single gold target | Prompted LLM grader (A/B/C) | Correct, Not-attempted, F-score | Explicit NOT_ATTEMPTED class | 1 grader call per item |
| SimpleQA Verified [7] | Extrinsic | Whole answer | Cleaned gold targets (1,000) | Improved autorater | F1 | Same as SimpleQA | Cheap |
| FActScore [1][2] | Extrinsic, long-form bios | Atomic fact | Wikipedia dump | Retrieval + ChatGPT or LLaMA+NPM | Factual precision | gamma length penalty, respond ratio | ~$1 / 100 sentences |
| LongFact + SAFE [9][11] | Extrinsic, open-domain | Self-contained relevant atomic fact | Google Search (Serper) | Multi-step LLM search agent | F1@K | Recall@K penalizes short answers | ~$0.20-0.40 / response |
| VeriScore [43] | Extrinsic | Verifiable claim | Google Search | GPT-4-class or fine-tuned open | Precision/recall-style | Excludes unverifiable claims | Similar to SAFE |
| HalluLens [45] | Extrinsic | Answer / claims | Dynamically generated from Wiki | LLM judges | Hallu, Correct, False Accept | Refusal tracked; fake entities | Regenerated per run |
| SummaC / AlignScore [28][30] | Intrinsic | Sentence x chunk | Source document | NLI / RoBERTa alignment | AUC, consistency score | n/a | Sub-1B, CPU/GPU cheap |
| HHEM + leaderboard [12][13] | Intrinsic (summaries) | Whole summary | Source article | Flan-T5 cross-encoder (2.1-Open) / HHEM-2.3 | Hallucination rate | Answer rate, common subset, length | Very cheap per call |
| MiniCheck [31] | Intrinsic | Sentence | Grounding doc | 0.4B-7B checker | Balanced acc. (LLM-AggreFact) | Split claims first | ~100x+ cheaper than GPT-4 (approx.) |
| Lynx [33] | Intrinsic (RAG) | Whole response | Retrieved context | Fine-tuned Llama-3 70B/8B | Accuracy (HaluBench) | n/a | Self-hosted 70B |
| LettuceDetect [34] | Intrinsic (RAG) | Token / span | Context | ModernBERT / EuroBERT / Qwen-2B | Span F1 (RAGTruth) | n/a | Encoder-cheap |
| FACTS Grounding [18] | Intrinsic, long-doc | Whole response | Doc up to ~32k tokens | 3 frontier LLM judges | Mean grounded % | Eligibility gate | 3 judge calls per item |
| ALCE [20] | Attribution | Statement + citations | Cited passages | TRUE T5-11B NLI | Citation recall/precision | n/a | One NLI pass per citation |
| RAGChecker [38] | Both, diagnostic | Claim | GT answer + chunks | LLM extractor + checker | Claim P/R/F1 + 7 diagnostics | Self-knowledge separated | Several LLM calls per claim |
| ARES [41] | RAG triad | Query/context/answer | Corpus | Fine-tuned judges + PPI | Scores with CIs | n/a | Needs 50-300 human labels |
| Open RAG Eval [42] | Both + recall | Nugget, citation | Retrieved passages | GPT-4o + HHEM | Nugget scores, citation F1 | No-answer detection | No golden answers needed |

---

## 10. Cross-cutting failure modes

1. **Precision-only gaming.** Any metric of the form supported/total can be maximized by short, generic or refusing answers. Countermeasures in the literature: FActScore's gamma penalty, SAFE's F1@K recall, FACTS's eligibility gate, Vectara's answer rate on a common subset, SimpleQA's F-score, and TREC nugget recall.
2. **Guessing is rewarded by accuracy.** If the metric does not separate wrong answers from abstentions, optimization pressure creates confident hallucinators [8]. Report error rate as its own number.
3. **Decomposition is a hidden variable.** Atomic facts, claims, triplets and nuggets are LLM-generated and non-unique, so changing the decomposer changes scores. Pin decomposer model, prompt and version as part of the metric definition.
4. **Evidence-source drift.** Wikipedia snapshots age, and live search changes daily (SAFE, VeriScore). Archive retrieved evidence alongside verdicts so runs can be reproduced.
5. **Judge self-preference and single-judge blind spots.** FACTS uses three judge families. Vectara built FaithBench specifically to audit its own detector against others [15][18].
6. **Label-construction artifacts in meta-evals.** Synthetic negatives (HaluEval, FactCC-style corruptions) are easier than natural errors. ROUGE-derived labels inflate detector AUROC [49]. Contradiction is consistently the hardest attribution class [24].
7. **Contamination.** Public QA sets leak into training data. Responses include private splits (Vectara, FACTS), dynamic generation (HalluLens) and verified re-releases (SimpleQA Verified).
8. **Tool confounding.** Closed-book factuality numbers are meaningless once search is enabled. Always record the tool configuration, as `simple-evals` did for o3 [5].
9. **Faithful but wrong, or correct but unfaithful.** A RAG answer can be faithful to a wrong document, or correct from parametric memory while unsupported by context. RAGChecker's self-knowledge and the TruLens triad separate these cases. A single "hallucination score" cannot [38][40].

---

## Key takeaways for system designers

1. **Decide which hallucination you are measuring first.** Extrinsic (world knowledge) needs an answer key or search, intrinsic (context faithfulness) needs only an entailment judge, and attribution needs per-citation checks. Most RAG products need all three, reported separately.
2. **Use a three-outcome grade, not binary accuracy.** Copy SimpleQA's correct / incorrect / not-attempted scheme and publish error rate and abstention rate. If you need one number, use a penalized score or an F-score that cannot be maximized by guessing or by refusing.
3. **Always pair a precision metric with a coverage or recall check.** Faithfulness should go with nugget recall, answer relevance or an eligibility gate. FActScore precision should go with F1@K. Hallucination rate should go with answer rate on a common subset.
4. **Use tiered verifiers.** Put a cheap encoder (HHEM-2.1-Open, AlignScore, MiniCheck, LettuceDetect) on 100% of production traffic and in CI. Escalate to an LLM judge or judge ensemble for disagreements, high-stakes domains and offline benchmark runs. Small checkers trained on diverse or synthetic alignment data reach GPT-4-level accuracy on standard benchmarks at a small fraction of the cost [30][31].
5. **Verify at claim or span granularity, and keep the evidence.** Claim-level verdicts with stored evidence snippets make failures debuggable, enable span highlighting in the UI, and make scores reproducible despite search drift.
6. **Pin the decomposer and the judge as versioned parts of the metric.** Treat "FActScore with retrieval+ChatGPT" and "FActScore with LLaMA+NPM" as different metrics. Their absolute levels differ by more than 10 points [2].
7. **Use private or dynamic test sets for anything you will optimize against.** Vectara and FACTS keep private splits, and HalluLens regenerates its tests. A public static set becomes a training target.
8. **Meta-evaluate your evaluator on hard, natural errors.** Use FaithBench, LLM-AggreFact, RAGTruth or in-domain human labels, with threshold-free AUC as TRUE recommends. Do not trust detector scores validated only on synthetic negatives or ROUGE-labeled data.
9. **Attribute RAG failures to a component.** Retriever claim recall, generator noise sensitivity and parametric self-knowledge (RAGChecker), or the context-relevance / groundedness / answer-relevance triad, tell you *what to fix*, not just that something is wrong.
10. **Report uncertainty.** With small human-labeled sets, use PPI-style corrections (ARES) or bootstrap intervals. Judge error is itself a source of variance.

---

## References

1. Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., Hajishirzi, H. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP 2023. https://arxiv.org/abs/2305.14251
2. Min, S. et al. "FActScore" official repository (README, estimators, results table). GitHub, 2023. https://github.com/shmsw25/FActScore
3. OpenAI. "Introducing SimpleQA." 2024. https://openai.com/index/introducing-simpleqa/
4. Wei, J., Karina, N., Chung, H. W., Jiao, Y. J., Papay, S., Glaese, A., Schulman, J., Fedus, W. "Measuring short-form factuality in large language models" (SimpleQA paper). OpenAI, 2024. https://cdn.openai.com/papers/simpleqa.pdf
5. OpenAI. "simple-evals" repository (README, results table, July 2025 deprecation note). GitHub, 2024-2025. https://github.com/openai/simple-evals
6. OpenAI. "simpleqa_eval.py" (grader template and metric code). GitHub, 2024. https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
7. Haas, L. et al. "SimpleQA Verified: A Reliable Factuality Benchmark to Measure Parametric Knowledge." Google DeepMind / Google Research, 2025. https://arxiv.org/abs/2509.07968
8. Kalai, A. T., Nachum, O., Vempala, S. S., Zhang, E. "Why Language Models Hallucinate." OpenAI, 2025. https://arxiv.org/abs/2509.04664
9. Wei, J., Yang, C., Song, X., Lu, Y., Hu, N., Huang, J., Tran, D., Peng, D., Liu, R., Huang, D., Du, C., Le, Q. V. "Long-form factuality in large language models." Google DeepMind, 2024. https://arxiv.org/abs/2403.18802
10. Google DeepMind. "long-form-factuality" repository (LongFact, SAFE, F1@K). GitHub, 2024. https://github.com/google-deepmind/long-form-factuality
11. Google DeepMind. "SAFE" README (pipeline steps, agreement and cost figures). GitHub, 2024. https://github.com/google-deepmind/long-form-factuality/tree/main/eval/safe
12. Vectara. "Hallucination Leaderboard" repository (methodology, prompt, dataset, rankings updated 22 September 2026). GitHub, 2023-2026. https://github.com/vectara/hallucination-leaderboard
13. Vectara. "HHEM-2.1-Open: hallucination_evaluation_model" model card. Hugging Face, 2024. https://huggingface.co/vectara/hallucination_evaluation_model
14. Vectara. "Cut the Bull... Detecting Hallucinations in Large Language Models." Blog, 2023. https://vectara.com/blog/cut-the-bull-detecting-hallucinations-in-large-language-models/
15. Vectara et al. "FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs." NAACL 2025; repository. https://github.com/vectara/FaithBench (paper: https://aclanthology.org/2025.naacl-short.38.pdf)
16. Lin, S., Hilton, J., Evans, O. "TruthfulQA: Measuring How Models Imitate Human Falsehoods." ACL 2022. https://arxiv.org/abs/2109.07958
17. Lin, S., Hilton, J., Evans, O. "TruthfulQA" repository (metrics, January 2025 binary-MC update). GitHub, 2021-2025. https://github.com/sylinrl/TruthfulQA
18. Jacovi, A. et al. "The FACTS Grounding Leaderboard: Benchmarking LLMs' Ability to Ground Responses to Long-Form Input." Google DeepMind / Google Research, 2025. https://arxiv.org/abs/2501.03200
19. Google DeepMind. "FACTS Grounding: A new benchmark for evaluating the factuality of large language models." Blog, December 2024. https://deepmind.google/discover/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/
20. Gao, T., Yen, H., Yu, J., Chen, D. "Enabling Large Language Models to Generate Text with Citations" (ALCE). EMNLP 2023. https://arxiv.org/abs/2305.14627
21. Princeton NLP. "ALCE" repository. GitHub, 2023. https://github.com/princeton-nlp/ALCE
22. Rashkin, H. et al. "Measuring Attribution in Natural Language Generation Models" (AIS). Google, 2021/2023. https://arxiv.org/abs/2112.12870
23. Liu, N. F., Zhang, T., Liang, P. "Evaluating Verifiability in Generative Search Engines." Stanford, Findings of EMNLP 2023. https://arxiv.org/abs/2304.09848
24. Yue, X. et al. "Automatic Evaluation of Attribution by Large Language Models" (AttrScore); repository with results. Ohio State, 2023. https://github.com/OSU-NLP-Group/AttrScore (paper: https://arxiv.org/abs/2305.06311)
25. Li, Y. et al. "AttributionBench: How Hard is Automatic Attribution Evaluation?" Findings of ACL 2024; repository. https://github.com/OSU-NLP-Group/AttributionBench
26. Kryscinski, W., McCann, B., Xiong, C., Socher, R. "Evaluating the Factual Consistency of Abstractive Text Summarization" (FactCC). Salesforce, EMNLP 2020; repository. https://github.com/salesforce/factCC (paper: https://arxiv.org/abs/1910.12840)
27. Fabbri, A. R., Wu, C.-S., Liu, W., Xiong, C. "QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization." NAACL 2022; repository. https://github.com/salesforce/QAFactEval (paper: https://arxiv.org/abs/2112.08542)
28. Laban, P., Schnabel, T., Bennett, P. N., Hearst, M. A. "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization." TACL 2022; repository. https://github.com/tingofurro/summac (paper: https://aclanthology.org/2022.tacl-1.10.pdf)
29. Honovich, O. et al. "TRUE: Re-evaluating Factual Consistency Evaluation." Google, NAACL 2022; repository. https://github.com/google-research/true (paper: https://arxiv.org/abs/2204.04991)
30. Zha, Y., Yang, Y., Li, R., Hu, Z. "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function." ACL 2023; repository. https://github.com/yuh-zha/AlignScore (paper: https://arxiv.org/abs/2305.16739)
31. Tang, L., Laban, P., Durrett, G. "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents." EMNLP 2024; repository (LLM-AggreFact, Bespoke-MiniCheck-7B). https://github.com/Liyan06/MiniCheck (paper: https://arxiv.org/abs/2404.10774)
32. Niu, C. et al. (ParticleMedia). "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models." ACL 2024; repository. https://github.com/ParticleMedia/RAGTruth (paper: https://arxiv.org/abs/2401.00396)
33. Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., Qian, R. "Lynx: An Open Source Hallucination Evaluation Model." Patronus AI, 2024. https://arxiv.org/abs/2407.08488
34. KR Labs. "LettuceDetect" repository (v2, June-July 2026). GitHub, 2025-2026. https://github.com/KRLabsOrg/LettuceDetect (paper: https://arxiv.org/abs/2502.17125)
35. Manakul, P., Liusie, A., Gales, M. J. F. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." EMNLP 2023; repository. https://github.com/potsawee/selfcheckgpt (paper: https://arxiv.org/abs/2303.08896)
36. Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. "Detecting hallucinations in large language models using semantic entropy." Nature, 2024; code repository. https://www.nature.com/articles/s41586-024-07421-0 (code: https://github.com/jlko/semantic_uncertainty)
37. Amazon Science. "RefChecker" repository. GitHub, 2023-2024. https://github.com/amazon-science/RefChecker
38. Ru, D. et al. "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation." Amazon, NeurIPS 2024 Datasets and Benchmarks; repository. https://github.com/amazon-science/RAGChecker (paper: https://arxiv.org/abs/2408.08067)
39. Exploding Gradients. "Ragas: Faithfulness" metric documentation (including FaithfulnesswithHHEM). GitHub, 2024-2026. https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/available_metrics/faithfulness.md
40. TruEra / Snowflake. "The RAG Triad" (TruLens documentation). GitHub. https://github.com/truera/trulens/blob/main/docs/getting_started/core_concepts/rag_triad.md
41. Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M. "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems." Stanford, 2023; repository. https://github.com/stanford-futuredata/ARES (paper: https://arxiv.org/abs/2311.09476)
42. Vectara. "Open RAG Eval" repository and METRICS.md (UMBRELA, AutoNuggetizer, citation, HHEM). GitHub, 2025-2026. https://github.com/vectara/open-rag-eval/blob/main/METRICS.md
43. Song, Y., Kim, Y., Iyyer, M. "VeriScore: Evaluating the factuality of verifiable claims in long-form text generation." 2024; repository. https://github.com/Yixiao-Song/VeriScore (paper: https://arxiv.org/abs/2406.19276)
44. Chern, I.-C. et al. "FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios." GAIR-NLP, 2023; repository. https://github.com/GAIR-NLP/factool (paper: https://arxiv.org/abs/2307.13528)
45. Bang, Y. et al. "HalluLens: LLM Hallucination Benchmark." Meta FAIR, ACL 2025; repository. https://github.com/facebookresearch/HalluLens (paper: https://arxiv.org/abs/2504.17550)
46. Ravichander, A. et al. "HALoGEN: Fantastic LLM Hallucinations and Where to Find Them." ACL 2025. https://aclanthology.org/2025.acl-long.71/
47. Li, J. et al. "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models." EMNLP 2023; repository. https://github.com/RUCAIBox/HaluEval (paper: https://arxiv.org/abs/2305.11747)
48. Chen, S. et al. "FELM: Benchmarking Factuality Evaluation of Large Language Models." NeurIPS 2023 D&B; repository. https://github.com/hkust-nlp/felm
49. "The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs." EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1761/
50. EdinburghNLP. "awesome-hallucination-detection" (curated paper list with metric/dataset summaries). GitHub, 2023-2026. https://github.com/EdinburghNLP/awesome-hallucination-detection
51. Gekhman, Z. et al. "TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models." Google, EMNLP 2023. https://arxiv.org/abs/2305.11171
