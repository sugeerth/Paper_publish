# 24. Domain-Specific RAG for Regulated and Expert Fields: Medicine, Law, Finance and Science

## Overview

General-purpose RAG pipelines (chunk, embed, top-k, generate) work well enough for help-desk FAQs. They break down in fields where the answer has to be traceable to an authoritative source, where vocabulary is specialized and inconsistent, and where errors carry clinical, legal or financial liability. Four fields have produced most of the public, well-documented domain RAG work:

- **Medicine.** The questions are licensing-exam style or clinical, the corpora are huge (PubMed indexes more than 36M citations; the MedRAG snapshot alone uses about 23.9M abstracts), synonyms are everywhere (MI, myocardial infarction and heart attack name the same thing), and a confident wrong answer can cause harm.
- **Law.** Answers have to cite controlling authority. Documents are long contracts and opinions, retrieval precision has to reach the clause or character span, and "hallucinated case law" has already caused court sanctions, starting with *Mata v. Avianca* (S.D.N.Y., 2023) [34].
- **Finance.** Answers depend on numbers buried in 10-K/10-Q tables. Retrieval has to scope to the right company and period, and arithmetic over the retrieved figures is part of the task.
- **Science.** Questions need synthesis across many papers, recency matters, contradictions are common, and citation accuracy is the main currency.

This chapter covers the main public systems and benchmarks for each field: MedRAG/MIRAGE, i-MedRAG, Almanac, Self-BioRAG, MedGraphRAG and KG-RAG (medicine); LegalBench-RAG, CLERC and the commercial legal assistants Lexis+ AI and CoCounsel (law); FinanceBench, FinQA/TAT-QA, BloombergGPT and SEC-filing pipelines (finance); and PaperQA2, OpenScholar, Ai2 ScholarQA, Elicit and the 2025 deep-research platforms (science). Four cross-cutting themes come up across all of them: corpus curation, ontology-aware retrieval (UMLS, MeSH), citation requirements, and per-domain evaluation.

A note on sourcing: this chapter relies mainly on the systems' public code repositories and their associated papers. Figures taken from papers that could not be re-checked against the primary PDF while writing are marked "approximately" or "reported". Commercial products change quickly; descriptions of them reflect public documentation as of 2025-2026 and should be re-checked before being relied on.

```
            GENERIC RAG                         DOMAIN RAG (regulated / expert)
  +-----------------------------+      +---------------------------------------------+
  | crawl -> chunk -> embed     |      | CURATED corpus (licensed, versioned, dated)  |
  | top-k cosine                |      | + ontology/KG (UMLS, MeSH, XBRL, citators)   |
  | stuff prompt -> answer      |      | hybrid + domain retriever + reranker         |
  |                             |      | metadata scoping (company, year, jurisdiction)|
  |                             |      | evidence extraction / summarization          |
  |                             |      | answer WITH span-level citations             |
  |                             |      | abstain ("insufficient evidence") path       |
  |                             |      | domain benchmark + expert review             |
  +-----------------------------+      +---------------------------------------------+
```

---

## 1. Medicine

### 1.1 MedRAG and the MIRAGE benchmark

**Problem.** Before 2024, medical RAG papers each used their own corpus, retriever and question set, so their results couldn't be compared. Xiong, Jin, Lu and Zhang (Findings of ACL 2024) built a systematic benchmark, MIRAGE, and a reference toolkit, MedRAG, to test which corpus/retriever/LLM combinations actually help [1][2].

**Benchmark.** MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation) contains 7,663 questions drawn from five datasets [3]:

| Dataset | Questions | Type |
|---|---|---|
| MMLU-Med | 1,089 | Exam (six biomedical MMLU subjects) |
| MedQA-US | 1,273 | USMLE-style exam |
| MedMCQA | 4,183 | Indian medical entrance exams (dev set) |
| PubMedQA* | 500 | Research yes/no/maybe, given contexts removed |
| BioASQ-Y/N | 618 | Research yes/no, gold snippets removed |

The design choices are deliberate. Removing the provided contexts from PubMedQA and BioASQ forces the system to retrieve. The evaluation has four settings: zero-shot, multi-choice evaluation, retrieval-augmented generation, and question-only retrieval. In question-only retrieval the answer options are hidden from the retriever, which matches clinical use, where no option list exists [3].

**Corpora.** MedRAG ships a layered corpus called MedCorp [2]:

| Corpus | Raw docs | Snippets | Avg length (tokens) | Domain |
|---|---|---|---|---|
| PubMed | 23.9M | 23.9M | 296 | Biomedical abstracts |
| StatPearls | 9.3k | 301.2k | 119 | Point-of-care clinical |
| Textbooks | 18 | 125.8k | 182 | Medical textbooks (USMLE prep) |
| Wikipedia | 6.5M | 29.9M | 162 | General |
| MedCorp (all) | 30.4M | 54.2M | 221 | Mixed |

**Retrievers.** BM25 (lexical), Contriever (general dense, 110M parameters), SPECTER (scientific dense, 110M), and MedCPT (biomedical dense, 109M) [2]. MedCPT is a contrastively trained biomedical retriever and reranker that learned from PubMed search logs [4]. MedRAG also tests fusing the rankings from several retrievers.

```
                 question (options optionally hidden)
                              |
          +---------+---------+----------+----------+
          |         |         |          |          |
        BM25   Contriever  SPECTER    MedCPT     (any subset)
          |         |         |          |
          +----+----+---------+----------+
               | rank fusion across retrievers
               v
  MedCorp snippets: PubMed | StatPearls | Textbooks | Wikipedia
               |
               v  top-k snippets (k swept, e.g. up to 32)
        CoT prompt -> LLM (GPT-4, GPT-3.5, Mixtral, Llama 2, Meditron, PMC-LLaMA)
               |
               v
        option letter + rationale
```

**Key findings** [1][2]:

- Across 41 corpus/retriever/LLM combinations, MedRAG improved six LLMs by up to 18% over chain-of-thought prompting alone. It lifted GPT-3.5 and Mixtral to roughly GPT-4 level without retrieval.
- The best results came from **combining multiple medical corpora and multiple retrievers**. No single retriever won on every dataset.
- The authors observed a **log-linear scaling** relationship between the number of retrieved snippets and accuracy, and a **"lost-in-the-middle"** effect: evidence placed in the middle of long contexts was used less.
- Corpus choice depends on the task. Exam questions benefit from textbooks and StatPearls. Research questions (PubMedQA*, BioASQ) benefit from PubMed.

**Lessons.** A medical RAG system should treat its corpus as a portfolio, not a single source. It should ensemble retrievers, especially a domain dense retriever such as MedCPT combined with BM25 for drug names and gene symbols. It should also order context deliberately. MIRAGE's question-only setting is a useful guard against benchmarks that leak the answer through the retrieval query.

### 1.2 i-MedRAG: iterative follow-up queries

MedRAG retrieves once. Complex clinical vignettes need several hops, for example "which drug class" followed by "what is the contraindication in pregnancy". i-MedRAG (Pacific Symposium on Biocomputing 2025) has the LLM generate follow-up queries over several rounds, answers each one with retrieval, and appends the query-answer pairs to the context before giving the final answer [5]. The reference implementation exposes this through `follow_up=True`, with `n_rounds` (for example 4) and `n_queries` per round (for example 3) [2]. The same group later released RAG-Gym, a framework for training agentic medical search agents [2].

```
 Q --> LLM: generate n_queries follow-ups --> retrieve+answer each --> append Q/A log
  ^                                                                   |
  +----------------------------- repeat n_rounds ---------------------+
                                       |
                                       v
                          final answer conditioned on Q/A log
```

**Failure modes.** Cost grows with n_rounds x n_queries retrieval-plus-generation calls. The follow-up questions can also drift onto topics that are plausible but irrelevant. In practice, the round budget is the main lever on latency.

### 1.3 Almanac: curated-source retrieval for clinical medicine

**Problem.** Clinicians need answers grounded in trusted clinical references, not the open web, and with citations they can verify at the bedside. Almanac (Zakka et al., Stanford; first released 2023, later published in NEJM AI) was one of the first retrieval-augmented LLM systems built specifically for clinical medicine [6].

**Architecture (as described in the paper).** Almanac has three parts. A *database/retriever* fetches content from a curated set of medical sources rather than generic web results. An *evaluator* checks the relevance of retrieved passages. A *generator* writes answers that cite those passages. Clinicians evaluated it on a clinical question set, scoring factuality, completeness, user preference and adversarial safety. The authors reported gains over general-purpose chat LLMs, particularly in factuality, and better resistance to adversarial prompts. Exact margins vary by axis and should be read from the paper [6].

**Lessons.** Almanac shows the most important design choice in clinical RAG: **restrict the evidence base to sources the profession already trusts**, and judge the system with clinicians, not string-match metrics. Almanac's adversarial-safety axis tests whether the system refuses or corrects dangerous premises. That axis has since become standard in medical RAG evaluation.

### 1.4 Clinical-guideline RAG: a recurring pattern

Many hospital and health-system deployments follow a common pattern: answering questions over clinical practice guidelines (CPGs) from specialty societies, national bodies, or local formularies and protocols. Public descriptions vary, but the architecture tends to converge on the same elements:

```
  guideline PDFs (versioned, dated, issuing body)
        |
        v
  structure-aware parsing: recommendation units, strength-of-evidence grade,
  population/intervention tags, tables & flowcharts -> text
        |
        v
  index: hybrid (BM25 + dense) + metadata {body, version, date, population}
        |
  query --> concept normalization (UMLS CUIs / MeSH) --> filtered retrieval
        |
        v
  answer that quotes recommendation text + grade + guideline/version citation
        |
        v
  abstain / escalate if no recommendation covers the population
```

Design points:

- **Chunk on recommendations, not on fixed token counts.** A guideline recommendation carries its strength of evidence and its target population, and separating the two produces dangerous answers.
- **Versioning matters.** Guidelines get superseded, so the index needs effective dates and retrieval should prefer the current version.
- **Locality matters.** Local protocols, such as antibiotic stewardship rules, should override national guidance. A good design makes that precedence explicit in the ranking or in the prompt.

### 1.5 Ontology- and graph-aware medical retrieval: UMLS, MeSH, SPOKE

Biomedical vocabulary is the main reason embedding-only retrieval underperforms in this field. Two curated resources from the US National Library of Medicine are standard:

- **UMLS (Unified Medical Language System)** maps millions of terms from source vocabularies (SNOMED CT, ICD, RxNorm, MeSH and others) to Concept Unique Identifiers (CUIs) and semantic relations [7].
- **MeSH (Medical Subject Headings)** is the controlled vocabulary used to index PubMed. Every MEDLINE citation carries curated MeSH descriptors [8].

Common techniques:

1. **Query expansion through the ontology.** Normalize the query's entities to CUIs, then expand with synonyms, and with children in the hierarchy when the query is broad. Run BM25 over the expanded query.
2. **MeSH-filtered PubMed retrieval.** Use MeSH descriptors and publication-type filters (for example randomized controlled trial, meta-analysis, or practice guideline) as hard metadata filters before dense ranking. This puts the evidence hierarchy directly into retrieval.
3. **Knowledge-graph RAG.** **KG-RAG** (Soman et al., 2023) retrieves from SPOKE, a biomedical knowledge graph with more than 27M nodes of 21 types and 53M edges of 55 types. It extracts disease entities from the prompt, matches them against a disease vector index, and pulls a minimal, prompt-aware subgraph context instead of whole documents [9]. **Medical Graph RAG** (Wu, Zhu and Qi; arXiv 2024, listed as ACL 2025) builds a three-tier graph [10]:

```
  Tier 1 (top)    : private user data (e.g., MIMIC-IV notes) -> LLM-extracted graph
                          |  "trinity" links
  Tier 2 (middle) : medical books + papers (MedC-K, PubMed-linked S2ORC) -> graph
                          |
  Tier 3 (bottom) : medical dictionary (UMLS) -- already graph-structured
  Retrieval walks top -> down so every patient-level claim can be grounded
  in literature and finally in a controlled-vocabulary definition.
```

The MedGraphRAG repository notes a real trade-off: building graphs with an LLM gives higher quality, while non-learning extraction is faster and cheaper but noisier [10]. Graph construction over a large private corpus can be the dominant cost of the whole system. For more on GraphRAG cost profiles, see Chapter 7.

### 1.6 Self-reflective biomedical RAG and commercial evidence engines

**Self-BioRAG** (Jeong et al., Bioinformatics 2024) adapts Self-RAG to biomedicine [35]. A domain-tuned generator emits reflective tokens that decide *whether* to retrieve, judge whether each retrieved passage is relevant and supportive, and score the final answer. Its retriever is MedCPT over PubMed, PMC full text, clinical guidelines and medical textbooks. The design point is that adaptive retrieval matters in medicine: many exam questions are answerable from parametric knowledge, and forcing retrieval on them can add noise. Self-BioRAG reported clear gains over same-size open baselines on MedQA, MedMCQA and MMLU-Med.

On the commercial side, **OpenEvidence** became one of the most widely used clinician-facing evidence engines in 2024-2025. Its distinguishing design choice is corpus licensing rather than model architecture: it announced content agreements with publishers such as NEJM Group and the JAMA Network in 2025, so answers can cite full text from those journals rather than abstracts alone [36]. Public technical detail is limited, so treat it as an example of the "licensed, curated corpus" pattern from Section 5.1 rather than as a documented architecture.

### 1.7 Medical failure modes and lessons

- **Premise acceptance.** The LLM answers a question built on a false clinical premise. Mitigate with adversarial test sets (as Almanac did) and explicit premise checks.
- **Outdated evidence.** Older PubMed abstracts outnumber recent ones. Add recency and evidence-level priors, and filter by publication type.
- **Exam contamination.** MedQA-style questions may be in pretraining data. MIRAGE's research subsets with contexts removed partly address this [3].
- **Benchmarks vs. bedside.** Multiple-choice accuracy does not measure safety or completeness. Clinician-rated rubrics are still needed.

---

## 2. Law

### 2.1 Why legal RAG is different

Legal answers must cite **controlling authority** (the right jurisdiction, still good law), and a fabricated citation is a professional-conduct problem, not just a quality issue. Dahl, Magesh, Suzgun and Ho ("Large Legal Fictions", Journal of Legal Analysis, 2024) profiled legal hallucinations in general LLMs, including GPT-4, GPT-3.5, PaLM 2 and Llama 2, by sampling cases from US federal reporters. They found that hallucinations were pervasive in closed-book use, with reported rates of roughly 58% to 88% on verifiable questions about federal cases [11][12]. RAG became the industry's answer, and the field then had to measure how well RAG actually works.

### 2.2 LegalBench and LegalBench-RAG

**LegalBench** (Guha et al., 2023) is a collaboratively built benchmark of 162 legal reasoning tasks contributed by 40 contributors (lawyers, law professors, legal-impact labs). It measures reasoning, not retrieval [13][14].

**LegalBench-RAG** (Pipitone and Alami, ZeroEntropy, 2024) isolates the **retrieval** step [15][16]. The authors argue that legal RAG quality is limited by retrieval precision. Long, near-duplicate contract clauses mean that retrieving a whole document, or a large chunk, wastes context and invites hallucination.

**Construction.** The benchmark is built from four expert-annotated legal datasets: ContractNLI (NDAs), CUAD (commercial contracts), MAUD (merger agreements), and PrivacyQA (privacy policies). Each test case pairs a query with ground-truth snippets defined as **file path plus exact character range** in the corpus [16]. The paper reports 6,858 query-answer pairs over a corpus of more than 79M characters, plus a smaller "mini" subset for fast iteration [15].

**Metric.** Precision and recall are computed **deterministically at the character level** against the gold spans [16]. This penalizes retrieving an over-long chunk that merely contains the answer, which matches the legal need for pinpoint citations.

```
 query: "Does the NDA allow the receiving party to share with consultants?"
     |
     v
 retriever under test --> returned spans [(file, start, end), ...]
     |
     v
 overlap with gold spans at CHARACTER granularity
     -> precision@k = chars(returned ∩ gold) / chars(returned)
     -> recall@k    = chars(returned ∩ gold) / chars(gold)
```

**Findings (reported in the paper) [15].** Recursive-character text splitting beat naive fixed-size chunking. A general-purpose commercial reranker (Cohere's, in the paper's experiments) did *not* reliably help on this legal data and sometimes hurt. Precision was low across the board, especially on MAUD's long merger agreements. The practical message is that legal retrieval needs legal-specific reranking or fine-tuning, and that chunking strategy has a large effect.

**Caveat.** The benchmark was generated with an LLM-assisted pipeline, so regenerating it does not exactly reproduce the released data [16].

### 2.3 Commercial legal assistants: Lexis+ AI, Westlaw / CoCounsel

The major legal research vendors built RAG directly on their proprietary, editorially curated corpora. LexisNexis's Lexis+ AI and Thomson Reuters' AI-Assisted Research in Westlaw Precision and CoCounsel (which Thomson Reuters obtained by acquiring Casetext in 2023) use the same building blocks:

```
  user question (+ jurisdiction, practice area)
        |
        v
  query understanding -> legal-issue decomposition
        |
        v
  retrieval over curated corpus: cases, statutes, regulations,
  secondary sources (treatises, practice notes) + editorial metadata
  (headnotes, key numbers / topic taxonomies, citator status)
        |
        v
  citator filter: drop / flag overruled or negatively treated authority
        |
        v
  LLM drafts memo-style answer with linked citations
        |
        v
  lawyer verifies (UI links each citation to the source document)
```

Vendor marketing has sometimes described these products as "hallucination-free". **Magesh, Surani, Dahl, Suzgun, Manning and Ho ("Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools", Stanford RegLab/HAI, 2024; later published in the Journal of Empirical Legal Studies)** tested that claim with a preregistered set of more than 200 queries [17]. They reported that the RAG-based legal tools reduced hallucinations compared with general chatbots but did not eliminate them. Lexis+ AI and Ask Practical Law AI hallucinated on roughly one in six or more queries, and Westlaw's AI-Assisted Research on roughly one in three, with variation across query types. The paper also defines a taxonomy that matters for legal-specific evaluation:

- **Incorrect**: the answer misstates the law.
- **Misgrounded**: the statement may be right, but the cited source does not support it. This failure is unique to RAG and is especially dangerous, because a real citation creates trust.
- **Incomplete / refusal**: the tool fails to answer when it should.

**Lessons.**

1. Grounding is not the same as correctness. The citation must *support the proposition*, so evaluation needs entailment checks between each claim and its cited passage, not just checks that a link resolves.
2. Legal retrieval has to respect **jurisdiction and hierarchy**, such as binding vs. persuasive authority, and **temporal validity**, meaning citator status.
3. Fluent legal prose makes misgrounding harder to spot. Interfaces should show the quoted supporting text inline.

### 2.4 Case-law retrieval benchmarks: CLERC

LegalBench-RAG covers contracts and policies. **CLERC** (Hou et al., 2024) covers case law [37]. Built from the Caselaw Access Project, it defines two tasks: retrieving the cases that a given opinion cites, using the citing paragraph as the query, and generating an analytical paragraph with correct citations from the retrieved cases. The paper reports that BM25 is a strong baseline that off-the-shelf dense retrievers struggle to beat on long legal text without legal fine-tuning, and that generated analyses frequently include citations the retrieved context does not support. That is the same misgrounding failure the Stanford audit found in commercial tools [17].

---

## 3. Finance

### 3.1 FinanceBench

**Problem.** Enterprises wanted to know whether off-the-shelf LLM+RAG could answer analyst questions over public filings. FinanceBench (Islam, Kannappan, Kiela, Qian, Scherrer and Vidgen; Patronus AI and Contextual AI, 2023) supplies the test [18][19].

**Dataset.** It contains 10,231 questions about publicly traded companies, each with an answer and an evidence string. A 150-question open sample is released with source PDFs. Document types include 10-K, 10-Q, 8-K and earnings reports. Questions fall into three types: metrics-generated (for example, compute the FY2018 capex-to-revenue ratio), domain-relevant, and novel-generated. Each record is tagged with a GICS sector [19].

**Configurations.** The authors tested 16 model configurations: GPT-4-Turbo, Llama 2 and Claude 2, each closed-book, with a shared vector store over all filings, with a per-document vector store, with long-context prompts, and with oracle evidence. The configurations were run on the 150-case open sample, and the authors manually reviewed all 2,400 resulting answers (16 x 150) [18][19].

**Headline result.** GPT-4-Turbo with a shared-vector-store retrieval setup **answered incorrectly or refused 81% of questions** [19]. Long-context configurations did much better but were described as impractical, because of latency and because many filings exceed even long context windows [19].

```
   Configuration                          Retrieval scope        Outcome (qualitative)
   ------------------------------------   --------------------   -------------------------
   closed book                            none                   mostly wrong / refuse
   shared vector store (all filings)      corpus-wide top-k      81% wrong/refuse (GPT-4T)
   single-document vector store           the right filing       better
   long context (whole filing in prompt)  entire document        better; slow, limited
   oracle (gold evidence page)            perfect                upper bound
```

**Why shared-store RAG fails.** Filings from different companies and years are semantically almost identical ("Total revenues increased..."). Without **metadata scoping** by ticker, fiscal period and form type, top-k similarity pulls the right concept from the wrong company or the wrong year. Many questions also need **arithmetic over several table cells**, which retrieval of prose chunks does not support.

### 3.2 SEC-filing pipeline design

Production pipelines over EDGAR filings tend to include these elements:

```
  EDGAR (full-text search / submissions / XBRL APIs)
        |
        +--> XBRL facts (us-gaap tags, units, periods) --> SQL/columnar store
        |                                                     ^
        +--> filing HTML/PDF --> section parser (Item 1A Risk, Item 7 MD&A, Notes)
                    |                table extractor (keep row/col headers + units)
                    v
             chunks with metadata {CIK, ticker, form, fiscal_year, period, section}
                    |
  query --> entity+period resolver (ticker, "FY23", "last quarter") --> HARD filter
        --> router: numeric? -> XBRL/SQL tool  |  narrative? -> hybrid retrieval
        --> calculator / code tool for ratios, growth, sums
        --> answer with citation (form, period, section, page) + computed trace
```

Key techniques:

- **Prefer structured facts.** The SEC publishes XBRL financial data through its APIs [20]. When a number exists as a tagged fact, querying it (see Chapter 19 on text-to-SQL) is more reliable than having an LLM read it out of a PDF table.
- **Use hard metadata filters, not soft ranking.** Company and period should be filters applied before similarity search.
- **Chunk with tables in mind.** Keep table headers, units ("in millions") and period columns attached to every table chunk. Losing "in thousands" causes 1000x errors.
- **Separate computation from generation.** Route ratio and growth calculations to a code tool, and show the inputs alongside the answer.
- **Long-document benchmarks.** Follow-on datasets such as DocFinQA extend FinQA-style numerical reasoning to full-length filings, which tests whether retrieval plus reasoning survives documents far longer than a context window [21].

### 3.3 BloombergGPT vs. RAG

BloombergGPT (Wu et al., 2023) is a 50B-parameter (50.6B) decoder-only model trained on a mixed corpus of roughly 363B tokens of Bloomberg financial data plus roughly 345B tokens of general data. It outperformed similarly sized general models on financial NLP tasks while keeping general benchmark performance [22]. It is the main public example of the **"domain pretraining" alternative** to RAG.

The comparison that matters for system designers:

| Dimension | Domain-pretrained LLM (BloombergGPT) | General LLM + domain RAG |
|---|---|---|
| Upfront cost | Very high (full pretraining run) | Low to moderate (index + pipeline) |
| Freshness | Frozen at training cutoff | As fresh as the index |
| Attribution | None intrinsic | Citations to filings/pages |
| Jargon/format fluency | Strong | Depends on base model; improves with retrieval |
| Proprietary data use | Baked into weights (hard to revoke) | Access-controlled at query time |
| Keeping up with frontier | Overtaken by later general models | Swap base model freely |

Later evaluations reported that newer general-purpose models matched or beat BloombergGPT on several financial tasks [23]. The field's consensus moved toward **frontier model + curated financial retrieval + tools**, with domain fine-tuning reserved for narrow tasks such as extraction or classification. Chapter 17 covers the general fine-tuning vs. RAG trade-off.

### 3.4 Hybrid table-text benchmarks: FinQA, TAT-QA, ConvFinQA

Much of the finance RAG toolkit was shaped by earlier benchmarks that pair report text with tables and require multi-step arithmetic. **FinQA** (Chen et al., EMNLP 2021) annotates S&P 500 earnings-report questions with an executable reasoning program, so a system is judged on whether it retrieved the right facts *and* computed the right result [38]. **TAT-QA** (Zhu et al., ACL 2021) mixes tabular and textual evidence from financial reports and adds scale (thousands, millions, percent) as an explicit answer field [39], which is the units problem from Section 3.2 made measurable. **ConvFinQA** (Chen et al., EMNLP 2022) extends FinQA to multi-turn conversations in which later questions refer back to earlier numbers [40]. These datasets motivate the "retrieve facts, emit a program, execute it" pattern that production pipelines use, and DocFinQA [21] extends the setting to full-length filings.

---

## 4. Science

Scientific question answering stresses three things: synthesis across dozens of papers, attribution at the claim level, and handling contradictory evidence.

### 4.1 PaperQA and PaperQA2 (FutureHouse)

**Problem.** Researchers need literature answers that are correct, cited, and willing to say "I don't know". The original PaperQA (Lála et al., 2023) framed RAG over papers as an agent with tools [24]. PaperQA2 (Skarlinski et al., 2024, "Language agents achieve superhuman synthesis of scientific knowledge") hardened that design and evaluated it against human experts [25][26].

**Architecture** [26]:

```
                       +------------------ agent LLM (ToolSelector) ------------------+
 question ----------->  |  chooses tools in any order, within budget/timeouts           |
                       +--------------------------------------------------------------+
         |                         |                          |                   |
         v                         v                          v                   v
  [paper_search]           [gather_evidence]          [citation traversal]  [gen_answer]
  LLM writes keyword       embed query, top-k         follow refs/citers    best summaries
  queries -> metadata      chunks (evidence_k=10)     of good papers        (<=5 sources)
  providers (Semantic      -> per-chunk, query-       (internal versions; -> cited answer
  Scholar, Crossref,       specific SUMMARY + score    not in OSS repo)   or "insufficient
  Unpaywall, OpenAlex)     -> LLM re-score/select                            information"
  -> fetch PDFs, chunk,       = "RCS": rerank +
     embed, add to state      contextual summarization
```

Key techniques:

- **RCS (re-ranking and contextual summarization).** Each retrieved chunk is summarized *relative to the question*, scored by an LLM, and only the top summaries go into the answer prompt. This works as a cross-encoder reranker and context compressor in one step. It costs more LLM calls but produces much cleaner contexts [26].
- **Metadata and quality signals.** Citation counts, venue and **retraction checks** come from metadata providers, so retracted papers can be down-weighted or excluded [26].
- **Abstention.** LitQA2 questions include an "insufficient information" option. The system is scored on *precision* (accuracy on questions it chose to answer) as well as accuracy, which rewards calibrated abstention [25][27].
- **Configurable presets.** The open-source package ships `high_quality`, `fast`, `wikicrow` and `contracrow` settings. Defaults are an agent `search_count` of 8, `evidence_k` of 10, and `answer_max_sources` of 5, with GPT-4o-class LLMs and `text-embedding-3-small` embeddings [26].

**Results (reported in the paper) [25].** On LitQA2, a set of hard multiple-choice questions whose answers sit in specific recent papers (part of LAB-Bench [27]), PaperQA2 reported precision of 85.2% and accuracy of 66.0%. Its precision exceeded that of PhD-level biology experts under time limits, and its accuracy was comparable. **WikiCrow** used PaperQA2 to write Wikipedia-style articles on human genes, and evaluators judged these more accurate than the existing Wikipedia articles. **ContraCrow** detected contradictions in the literature. The paper reports an average of 2.34 contradictions per sampled biology paper, about 70% of which human experts validated.

**Failure modes and lessons.** RCS multiplies LLM calls, roughly one summary call per candidate chunk, so cost and latency scale with evidence_k times the number of searches. Full-text access is the binding constraint: open-access and licensing gaps silently limit coverage. The main lesson is that **agentic search, per-chunk evidence distillation, and abstention-aware evaluation** together produce expert-grade literature QA.

### 4.2 OpenScholar (Ai2 / University of Washington)

**Problem.** Proprietary LLMs fabricate scientific citations, and closed systems can't be inspected or self-hosted. OpenScholar (Asai, He, Shao et al., Hajishirzi; 2024) is a fully open alternative: open model, open datastore, open retriever [28][29].

**Architecture.**

```
  query
    |
    v
  bi-encoder retriever (Contriever continued-pretrained on peS2o)
    |  over OpenScholar Datastore: ~45M open-access papers, ~236M passage embeddings
    |  (+ optional Semantic Scholar API and web search)
    v
  cross-encoder reranker (trained from BGE-reranker-large)
    |
    v
  generator: OpenScholar-8B (Llama 3.1 8B, ~13k instruction examples)
    |
    v
  SELF-FEEDBACK LOOP: model critiques its draft ("missing evidence on X",
    "citation unsupported") -> issues new retrievals -> revises
    |
    v
  citation verification pass -> final answer with per-sentence citations
```

[29]

**Training data trick.** The 8B generator was trained on synthetic data produced by running the retrieve, generate, feedback and refine pipeline with a larger model and then filtering. This transfers the inference-time loop into a small open model [28].

**Evaluation: ScholarQABench.** The benchmark has subsets for computer science (ScholarQA-CS, with expert per-question rubrics), multi-domain (ScholarQA-Multi, covering CS, biomedicine and physics, with expert-written answers), biomedicine (ScholarQA-Bio) and neuroscience (ScholarQA-Neuro). It also reuses the single-paper tasks SciFact, PubMedQA and QASA. Metrics cover citation accuracy (precision/recall), string-based correctness, rubric-based correctness, and Prometheus-judged organization, relevance and coverage [30]. Experts wrote the rubrics between July and September 2024, so systems that retrieve papers published after that may legitimately know more than the reference answers [30].

**Results (reported in the paper, approximate) [28].** OpenScholar-8B outperformed GPT-4o and PaperQA2 on correctness on ScholarQABench, despite being far smaller. GPT-4o without retrieval **hallucinated a large majority of its citations**, reported at around 78-90%, while OpenScholar's citation accuracy was close to that of human experts. In expert pairwise evaluation, experts preferred OpenScholar answers over expert-written answers in a substantial share of comparisons, reported at about half for the 8B model and about 70% for an OpenScholar pipeline using GPT-4o.

**Lessons.** A small open model plus a strong domain datastore and reranker can match frontier closed models on attribution-heavy tasks. Self-feedback with additional retrieval is a cheap way to improve coverage. Citation accuracy should be measured explicitly, because fluency metrics miss fabricated references.

### 4.3 Ai2 ScholarQA

**Problem.** Researchers want an organized, multi-section literature review, not a single paragraph answer, with every claim traceable to an exact quote. Ai2 ScholarQA (Singh et al., 2025) is a deployed system with an open-source library [31].

**Pipeline** [31]:

```
 1. Query decomposition (LLM): rewrite query; extract filters (year, venue,
    field of study); produce keyword-search variant
 2. Retrieval: Semantic Scholar snippet search over ~11M+ full texts
    and ~100M+ abstracts; keyword search adds abstracts
 3. Rerank: cross-encoder (mxbai-rerank-large-v1); aggregate passages to PAPER level
 4. Quote extraction (LLM): pull exact supporting quotes per paper
 5. Outline + clustering (LLM): plan sections, choose format (paragraph vs list),
    assign quotes to sections
 6. Section generation (LLM): write each section from its quotes + prior sections,
    inline citations to quotes
 7. Literature-review tables for list-type sections (ArxivDIGESTables method)
```

Implementation details: retrieval is wrapped in a `PaperFinder` component and generation in a `MultiStepQAPipeline`. The default generation model is a Claude Sonnet model, with GPT-4o as fallback, and any LiteLLM-supported model can be swapped in. A moderation filter screens harmful queries [31]. The comparison-table step builds on ArxivDIGESTables, which generates literature-review tables whose columns are aspects shared across papers [32].

**Design notes.** The **quote-extraction step** splits "find the evidence" from "write the prose". Each sentence's citation points to a verbatim quote, which makes verification fast and cuts misgrounding. **Paper-level aggregation** after passage reranking prevents one long paper from dominating the context. **Outline-first generation** keeps long reports coherent and lets the UI stream section by section.

### 4.4 Elicit

Elicit is a commercial research assistant focused on systematic-review-style workflows [33]. It searches a large scholarly corpus and ranks papers, then extracts structured fields from each paper into a table (population, intervention, outcome, sample size and similar), with supporting quotes linked to each extracted cell. Architecturally this is **RAG as structured extraction**: the "answer" is a matrix of paper by attribute, each cell grounded in a quote. That makes per-cell accuracy measurable and lets reviewers audit the table instead of rereading every paper. The pattern resembles the ScholarQA table step and is the closest commercial analogue to how systematic reviewers already work.

### 4.5 Deep-research platforms (2025)

In 2025 the science systems above turned into multi-agent research platforms. **FutureHouse** released a platform (May 2025) that exposes PaperQA2-derived agents for different jobs: concise cited answers, broad literature reviews, and "has anyone done X?" precedent search [41]. **Ai2** rolled ScholarQA, paper search and data-analysis agents into **Asta** (August 2025) and released **AstaBench**, a suite for evaluating scientific research agents under controlled cost and tool settings [42]. General-purpose "deep research" products from frontier labs run similar loops (plan, search, read, synthesize) over the open web.

For domain designers the lesson is mostly about evaluation. These agents read hundreds of sources per question, so citation-level checks (ScholarQABench-style citation precision and recall) and cost-normalized comparisons matter more than single-answer accuracy. Restricting the agent to a curated scholarly index, instead of the open web, is still the main defense against citing low-quality or retracted sources.

### 4.6 Science comparison

| System | Corpus | Retrieval | Evidence step | Generator | Citation unit | Key eval |
|---|---|---|---|---|---|---|
| PaperQA2 | Fetched PDFs via S2/Crossref/OpenAlex | Agentic keyword search + dense top-k | RCS: per-chunk LLM summary + score | GPT-4o-class | Chunk / paper | LitQA2 (precision + accuracy) |
| OpenScholar | ~45M OA papers (peS2o), 200M+ passages | Trained bi-encoder + trained reranker | Self-feedback loop with re-retrieval | OpenScholar-8B (or GPT-4o) | Sentence -> passage | ScholarQABench |
| Ai2 ScholarQA | S2 snippets: ~11M full text, ~100M abstracts | S2 snippet + keyword search, mxbai reranker | Verbatim quote extraction | Claude Sonnet (GPT-4o fallback) | Sentence -> quote | Human / rubric eval |
| Elicit | Large scholarly corpus | Semantic search + ranking | Per-paper structured extraction | Proprietary pipeline | Table cell -> quote | Extraction accuracy |

---

## 5. Cross-Cutting Design Dimensions

### 5.1 Corpus curation

In every field covered here, the most important quality decision is made before retrieval: **what goes in the index**.

| Domain | Authoritative core | Common additions | Curation hazards |
|---|---|---|---|
| Medicine | Textbooks, StatPearls, guidelines, PubMed (MeSH-indexed) | Wikipedia, local protocols, EHR notes | Superseded guidelines, retracted papers, PHI leakage |
| Law | Primary law (cases, statutes, regs) + citator | Treatises, practice notes, firm work product | Overruled authority, jurisdiction mixing, privilege |
| Finance | EDGAR filings, XBRL facts | Transcripts, broker research, news | Restatements, period mismatch, units, MNPI controls |
| Science | Peer-reviewed full text, preprints | Abstract-only records, patents | Paywalls, retractions, preprint vs published versions |

MedRAG's results show that mixing sources helps when each is labeled, so that retrieval and prompts can account for source type [1]. PaperQA2's retraction checks and Lexis/Westlaw's citator filters both apply the same principle: **the validity of a document has to be a retrievable attribute**.

### 5.2 Ontology-aware retrieval

```
  raw query --> NER --> normalize to IDs ------------------------------+
                       medicine: UMLS CUI / MeSH descriptor / RxNorm   |
                       law:      citation parser, key-number/topic     |
                       finance:  ticker->CIK, XBRL us-gaap concept     |
                       science:  S2 paper IDs, field-of-study, DOIs    |
                                                                       v
       expansion (synonyms, children)  +  hard filters  +  graph neighbors
                                       |
                                       v
                    hybrid retrieval (BM25 on expanded terms + dense)
```

Ontologies do three jobs. They provide **recall** through synonym expansion, **precision** through hard filters (MeSH publication types, jurisdiction, ticker and period), and **reasoning scaffolds** through graph neighbors (KG-RAG's SPOKE subgraphs, MedGraphRAG's UMLS tier) [9][10]. Most of these vocabularies are maintained by public institutions, such as NLM's UMLS and MeSH [7][8] and the SEC's XBRL taxonomy [20], which makes them cheaper and more stable than LLM-extracted graphs.

### 5.3 Citation requirements by domain

| Domain | Required citation granularity | Validity check | Typical failure |
|---|---|---|---|
| Medicine | Source + section; guideline version + recommendation grade | Currency, evidence level | Citing an outdated or low-evidence source |
| Law | Pinpoint cite (case, reporter, page) | Citator status, jurisdiction | Misgrounding: a real cite that does not support the claim [17] |
| Finance | Filing, period, section/page, XBRL fact | Restatement / amendment | Right concept, wrong company or year [19] |
| Science | Paper + passage/quote, DOI | Retraction status | Fabricated references (closed-book LLMs) [28] |

Across all four, the strongest designs (ScholarQA quotes, LegalBench-RAG character spans, Elicit cell-level quotes) push citations down to **verbatim spans**. That lets an entailment check, or a human, verify each claim in seconds.

### 5.4 Per-domain evaluation

| Benchmark | Domain | Size | Tests | Key metric |
|---|---|---|---|---|
| MIRAGE [3] | Medicine | 7,663 Qs, 5 datasets | End-to-end RAG QA | Accuracy; question-only retrieval setting |
| HealthBench [43] | Medicine | 5,000 multi-turn conversations, physician rubrics | Open-ended clinical dialogue (not RAG-specific) | Rubric score from physician-written criteria |
| CLERC [37] | Law | Case-law citation retrieval + analysis generation | Retrieval and cited generation | Recall@k; citation support |
| FinQA / TAT-QA [38][39] | Finance | Report text + tables | Numerical reasoning over hybrid evidence | Execution / exact-match accuracy |
| LegalBench [14] | Law | 162 tasks | Legal reasoning (no retrieval) | Task accuracy |
| LegalBench-RAG [15][16] | Law | ~6.9k queries (approx.), 4 source datasets | Retrieval only | Character-level precision/recall |
| Stanford legal RAG audit [17] | Law | Preregistered query set | Commercial tools end-to-end | Hallucination / misgrounding rate |
| FinanceBench [18][19] | Finance | 10,231 Qs (150 open) | Open-book QA over filings | Human-graded correct / incorrect / refusal |
| LitQA2 / LAB-Bench [27] | Biology | ~2,400 Qs in 8 categories (LAB-Bench); LitQA2 ~200 Qs | Literature retrieval QA | Precision and accuracy with abstain option |
| ScholarQABench [30] | Multi-science | ~2,967 expert-written queries; CS, Bio, Neuro, Multi + single-paper tasks | Long-form synthesis | Citation P/R, rubric correctness, Prometheus |

Recurring lessons from these benchmarks:

1. **Evaluate retrieval separately from generation** (LegalBench-RAG). End-to-end scores hide whether the retriever or the generator is at fault.
2. **Reward abstention** (LitQA2 precision vs. accuracy). In regulated fields, "I don't know" beats a confident error.
3. **Grade grounding separately from correctness** (Stanford's misgrounded category).
4. **Use human experts for long-form answers.** ScholarQABench rubrics, Almanac's clinician panels and FinanceBench's 2,400 manual reviews all show that automatic metrics are not enough.
5. **Watch for temporal leakage and staleness** (ScholarQABench's dated rubrics, PubMed recency).

---

## 6. Comparative Summary of Domain Architectures

| System | Domain | Retrieval style | Domain knowledge source | Iteration | Citation | Open? |
|---|---|---|---|---|---|---|
| MedRAG [1][2] | Medicine | Multi-corpus, multi-retriever fusion | MedCorp; MedCPT | Single shot | Snippet | Yes |
| i-MedRAG [5] | Medicine | Iterative follow-up queries | MedCorp | n_rounds x n_queries | Snippet | Yes |
| Almanac [6] | Clinical | Curated clinical sources | Trusted references | Evaluator step | Source link | Paper |
| Self-BioRAG [35] | Biomedical | Adaptive retrieval with reflective tokens | MedCPT over PubMed/PMC/guidelines/textbooks | Retrieve-on-demand + self-critique | Passage | Yes |
| KG-RAG [9] | Biomedical | KG subgraph retrieval | SPOKE (27M nodes) | Single shot | KG triples | Yes |
| MedGraphRAG [10] | Medicine | 3-tier graph, top-down | MIMIC + MedC-K + UMLS | Graph walk | Graph to source | Yes |
| LegalBench-RAG [15] | Law | (benchmark) | CUAD/MAUD/ContractNLI/PrivacyQA | n/a | Char spans | Yes |
| Lexis+ AI / CoCounsel [17] | Law | Curated legal DB + citator | Editorial metadata | Vendor-specific | Linked cites | No |
| FinanceBench [19] | Finance | (benchmark) | SEC filings | n/a | Evidence page | Partial |
| PaperQA2 [25][26] | Science | Agentic search + RCS | S2/Crossref/OpenAlex metadata | Agent loop | Chunk/paper | Yes |
| OpenScholar [28][29] | Science | Trained retriever + reranker | 45M-paper datastore | Self-feedback | Sentence-level | Yes |
| Ai2 ScholarQA [31] | Science | S2 snippet search + rerank | S2 corpus | Pipeline stages | Verbatim quote | Yes (lib) |
| Elicit [33] | Science | Search + structured extraction | Scholarly corpus | Per-paper | Cell-level quote | No |
| Asta / FutureHouse platform [41][42] | Science | Multi-agent deep research | Scholarly indexes (S2, full text) | Agent loops | Passage / quote | Partial |

---

## Key Takeaways for System Designers

1. **Curate before you retrieve.** Build the index from sources the profession already trusts (textbooks and guidelines, primary law with a citator, EDGAR/XBRL, peer-reviewed full text). Make validity (retraction, overruling, supersession, restatement) a filterable attribute.
2. **Scope with hard metadata filters.** Company, fiscal period, jurisdiction, guideline version and publication type should be filters applied *before* similarity search. FinanceBench's 81% failure rate with a shared vector store is the cautionary example [19].
3. **Ensemble retrievers and use domain encoders.** BM25 catches drug names, tickers and citations exactly. Domain dense retrievers (MedCPT, the peS2o-trained Contriever) catch paraphrases. MedRAG's best configurations fused both across multiple corpora [1].
4. **Use the field's ontology.** UMLS/MeSH normalization, legal citation parsing and XBRL concepts give recall, precision and graph context at low cost. Reserve LLM-built knowledge graphs for private corpora where no curated ontology exists.
5. **Distill evidence before generating.** Per-chunk contextual summaries (PaperQA2's RCS) or verbatim quote extraction (ScholarQA) give cleaner contexts and verifiable citations, at the cost of extra LLM calls.
6. **Cite spans, then check entailment.** Make the citation granularity a verbatim span and test automatically that each claim is supported by its span. Misgrounding, where a real citation does not support the claim, is the characteristic RAG failure in law [17].
7. **Design for abstention.** Include an "insufficient evidence" path and measure precision as well as accuracy (LitQA2) [25][27].
8. **Iterate with a budget.** Multi-round retrieval (i-MedRAG, PaperQA2 agents, OpenScholar self-feedback) helps on multi-hop questions. Cap the rounds, since cost grows multiplicatively.
9. **Route numbers to tools.** In finance, query XBRL facts or tables through SQL or code. Do not have an LLM read numbers out of chunked PDFs.
10. **Evaluate per domain, with experts.** Combine a retrieval-only benchmark (LegalBench-RAG-style span metrics), an end-to-end benchmark (MIRAGE, FinanceBench, ScholarQABench) and expert review of long-form outputs. Re-audit vendor "hallucination-free" claims independently.
11. **Prefer RAG over domain pretraining for freshness and attribution.** BloombergGPT showed what domain pretraining can do, but it was later overtaken by general models. RAG plus tools on a strong base model is easier to update, audit and access-control [22][23].

---

## References

1. Xiong, G., Jin, Q., Lu, Z., Zhang, A. "Benchmarking Retrieval-Augmented Generation for Medicine." Findings of ACL, 2024. https://arxiv.org/abs/2402.13178
2. Xiong, G. et al. "MedRAG Toolkit" (GitHub repository and README). 2024-2025. https://github.com/Teddy-XiongGZ/MedRAG
3. Xiong, G. et al. "MIRAGE: Medical Information Retrieval-Augmented Generation Evaluation" (GitHub repository). 2024. https://github.com/Teddy-XiongGZ/MIRAGE
4. Jin, Q. et al. "MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval." Bioinformatics / arXiv, 2023. https://arxiv.org/abs/2307.00589
5. Xiong, G., Jin, Q., Wang, X., Zhang, M., Lu, Z., Zhang, A. "Improving Retrieval-Augmented Generation in Medicine with Iterative Follow-up Questions." Pacific Symposium on Biocomputing 2025 / arXiv, 2024. https://arxiv.org/abs/2408.00727
6. Zakka, C. et al. "Almanac: Retrieval-Augmented Language Models for Clinical Medicine." arXiv 2023; NEJM AI 2024. https://arxiv.org/abs/2303.01229
7. U.S. National Library of Medicine. "Unified Medical Language System (UMLS)." https://www.nlm.nih.gov/research/umls/index.html
8. U.S. National Library of Medicine. "Medical Subject Headings (MeSH)." https://www.nlm.nih.gov/mesh/meshhome.html
9. Soman, K. et al. "Biomedical Knowledge Graph-Optimized Prompt Generation for Large Language Models" (KG-RAG). arXiv, 2023; code at https://github.com/BaranziniLab/KG_RAG . Paper: https://arxiv.org/abs/2311.17330
10. Wu, J., Zhu, J., Qi, Y. "Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation." arXiv 2024 / ACL 2025; code: https://github.com/SuperMedIntel/Medical-Graph-RAG . Paper: https://arxiv.org/abs/2408.04187
11. Dahl, M., Magesh, V., Suzgun, M., Ho, D. E. "Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models." Journal of Legal Analysis, 2024. https://arxiv.org/abs/2401.01301
12. Stanford RegLab. "legal_hallucinations" (code for Dahl et al. 2024). https://github.com/reglab/legal_hallucinations
13. Guha, N. et al. "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models." 2023. https://arxiv.org/abs/2308.11462
14. HazyResearch. "LegalBench" (GitHub repository). 2023. https://github.com/HazyResearch/legalbench
15. Pipitone, N., Alami, G. H. "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain." ZeroEntropy, 2024. https://arxiv.org/abs/2408.10343
16. ZeroEntropy. "legalbenchrag" (GitHub repository). 2024. https://github.com/zeroentropy-ai/legalbenchrag
17. Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., Ho, D. E. "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools." Stanford RegLab/HAI, 2024; Journal of Empirical Legal Studies, 2025. https://arxiv.org/abs/2405.20362
18. Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N., Vidgen, B. "FinanceBench: A New Benchmark for Financial Question Answering." Patronus AI, 2023. https://arxiv.org/abs/2311.11944
19. Patronus AI. "financebench" (GitHub repository and README). 2023. https://github.com/patronus-ai/financebench
20. U.S. Securities and Exchange Commission. "EDGAR Application Programming Interfaces (submissions and XBRL data APIs)." https://www.sec.gov/edgar/sec-api-documentation
21. Reddy, V. et al. "DocFinQA: A Long-Context Financial Reasoning Dataset." 2024. https://arxiv.org/abs/2401.06915
22. Wu, S. et al. "BloombergGPT: A Large Language Model for Finance." Bloomberg, 2023. https://arxiv.org/abs/2303.17564
23. Li, X. et al. "Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks." 2023. https://arxiv.org/abs/2305.05862
24. Lála, J. et al. "PaperQA: Retrieval-Augmented Generative Agent for Scientific Research." 2023. https://arxiv.org/abs/2312.07559
25. Skarlinski, M. D., Cox, S., Laurent, J. M., Braza, J. D., Hinks, M., Hammerling, M. J., Ponnapati, M., Rodriques, S. G., White, A. D. "Language Agents Achieve Superhuman Synthesis of Scientific Knowledge." FutureHouse, 2024. https://arxiv.org/abs/2409.13740
26. FutureHouse. "paper-qa (PaperQA2)" (GitHub repository and README). 2024-2025. https://github.com/Future-House/paper-qa
27. Laurent, J. M. et al. "LAB-Bench: Measuring Capabilities of Language Models for Biology Research." FutureHouse, 2024. https://arxiv.org/abs/2407.10362 ; data: https://github.com/Future-House/LAB-Bench
28. Asai, A., He, J., Shao, R., et al., Hajishirzi, H. "OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs." Ai2 / University of Washington, 2024. https://arxiv.org/abs/2411.14199
29. Asai, A. et al. "OpenScholar" (GitHub repository). 2024. https://github.com/AkariAsai/OpenScholar
30. Asai, A. et al. "ScholarQABench" (GitHub repository). 2024. https://github.com/AkariAsai/ScholarQABench
31. Singh, A. et al. (Ai2). "Ai2 Scholar QA: Organized Literature Synthesis with Attribution" and ai2-scholarqa-lib (GitHub repository). 2025. https://github.com/allenai/ai2-scholarqa-lib
32. Newman, B. et al. "ArxivDIGESTables: Synthesizing Scientific Literature into Tables using Language Models." 2024. https://arxiv.org/abs/2410.22360
33. Elicit. "Elicit: The AI Research Assistant" (product site). https://elicit.com
34. *Mata v. Avianca, Inc.*, No. 22-cv-1461 (PKC), Opinion and Order on Sanctions (S.D.N.Y. June 22, 2023). https://www.courtlistener.com/?q=%22Mata+v.+Avianca%22
35. Jeong, M., Sohn, J., Sung, M., Kang, J. "Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models" (Self-BioRAG). Bioinformatics, 2024. https://arxiv.org/abs/2401.15269
36. OpenEvidence. Product site and announcements of content agreements with NEJM Group and JAMA Network (2025). https://www.openevidence.com
37. Hou, A. B. et al. "CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation." 2024. https://arxiv.org/abs/2406.17186
38. Chen, Z. et al. "FinQA: A Dataset of Numerical Reasoning over Financial Data." EMNLP, 2021. https://arxiv.org/abs/2109.00122
39. Zhu, F. et al. "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance." ACL, 2021. https://arxiv.org/abs/2105.07624
40. Chen, Z. et al. "ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering." EMNLP, 2022. https://arxiv.org/abs/2210.03849
41. FutureHouse. "FutureHouse Platform" (research agents built on PaperQA2), 2025. https://platform.futurehouse.org
42. Allen Institute for AI. "Asta" (scientific research agents) and AstaBench, 2025. https://allenai.org/asta
43. Arora, R. K. et al. "HealthBench: Evaluating Large Language Models Towards Improved Human Health." OpenAI, 2025. https://arxiv.org/abs/2505.08775
