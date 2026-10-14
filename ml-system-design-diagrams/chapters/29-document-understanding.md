# 29. Design a Document Understanding / Invoice Extraction System (OCR + Layout + LLM)

> How the prompt is usually worded: "Design a system that takes scanned or digital invoices
> (PDF, photo, email attachment) and returns structured fields (vendor, invoice number,
> dates, totals, tax, line items), then posts them to an ERP / accounts-payable system."
> Variants: "Receipt scanning for expense reports", "Intelligent document processing (IDP)
> like AWS Textract or UiPath Document Understanding", "KYC / ID-document extraction",
> "Parse PDFs into Markdown/JSON for an LLM pipeline", "Insurance-claim form digitization".

> Source note (review pass, 2026-10): github.com and raw.githubusercontent.com resolved;
> uber.com, docs.aws.amazon.com, docs.uipath.com and arxiv.org were blocked by the proxy.
> - (fetched): read in a fetch run.
> - (listed in [n]): the URL appears in curated list [n], which was fetched.
> - (ch-ref): verified in ch. 01 or ch. 24 of this atlas.
> - (not opened): canonical URL cited from memory. Check before relying on it.
>
> Numbers marked "approx." or "assumption" are typical values, not sourced figures.

## Company tags

| Company / context | Basis | Production system it mirrors |
|---|---|---|
| Amazon (AWS) | Owns Textract, whose AnalyzeExpense API returns normalized invoice/receipt fields and line items [10] (fetched). AWS IDP reference pipeline: classify -> extract -> enrich -> validate -> human review [9] (fetched) | Textract + Comprehend + A2I + Bedrock (Diagram 4) |
| Uber | "Uber's Real-Time Document Check" for driver documents [13] (listed in [7]). A GenAI invoice-processing post [11] is cited but not opened | Document verification; OCR -> LLM -> review |
| Google | Research on extracting structured data from templatic documents (form-like invoices) [14] (listed in [8]) | Candidate generation + field scoring |
| Microsoft | LayoutLMv2 / LayoutLMv3 / TrOCR / DiT research [3][5][6] (fetched) | Layout-aware transformer extractor |
| Dropbox | Built an in-house OCR pipeline (word detector + word recognizer) and indexed text from billions of images [15][16] (listed in [7][8]) | OCR stage of Diagram 1 |
| Scribd | Document-type classification and metadata extraction [17][18][30] (listed in [7][8]) | Classifier + extractor stages |
| Nanonets | Receipt information extraction with graph convolutional networks [8] (listed in [8]) | Graph-based KIE over OCR boxes |
| UiPath | Sells Document Understanding; product docs describe a digitize -> classify -> extract -> validate -> train loop [26] (not opened) | Human validation station feeds retraining |
| Any "GenAI system design" loop | alirezadir lists "OCR/Text recognition system" under CV prompts and LLM content-processing systems under GenAI [1] (fetched) | OCR + LLM structured output |

**Portal coverage**

| Portal / source | What it covers | Link |
|---|---|---|
| alirezadir ML-Interviews | "OCR/Text recognition system" listed as a CV MLSD prompt. 9-step MLSD template | [1] |
| Chip Huyen, MLSD exercises | Closest exercise: #23, QA that "extract[s] an answer from a large collection of documents" [2] (fetched) | [2] |
| Educative (Grokking MLSD / GenAI SD) | 6-step framework. GenAI course covers document / multimodal pipelines (case title not verified) | [20][21] |
| ByteByteGo / Aminian & Xu | No dedicated invoice chapter verified. Reuse the 7-step framework and the GenAI book's multimodal chapters | [22] |
| IGotAnOffer, Exponent, Hello Interview | Generic MLSD / GenAI delivery frameworks used to structure this answer | [23][24][25] |
| Curated case-study lists | Engineer1999 and eugeneyan/applied-ml index the OCR, layout and extraction posts above | [7][8] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Ingest PDF, TIFF, JPG, email. Classify doc type. Extract header fields + line-item table. Normalize (dates, currency, vendor ID). Validate. Send low-confidence docs to human review. Post to ERP | [9][10] |
| Output schema | ~20-40 header fields + N line items (description, qty, unit price, amount, tax code) | typical assumption; CORD has 5 superclasses / 42 subfields [12] |
| Volume | 1M docs/day at a large enterprise or BPO. ~12 docs/s average, ~50/s peak. ~2-3 pages per doc | assumption |
| Latency | Async batch: p95 under 60 s end-to-end. Sync API (receipt app): p95 under 3-5 s per page | assumption |
| Accuracy | Field-level F1 of 0.95 or higher on key fields. Totals exact-match 99%+ after validation. Straight-through-processing (STP) rate 70-90% | typical target |
| Vendors / templates | 100k+ vendors, long-tail layouts. New layouts appear daily | assumption |
| Languages | 50-100+ scripts. Handwriting is a minority but real | assumption; PP-OCR covers 50-109 languages [19] |
| Compliance | PII and bank details: encryption, retention limits, audit trail, region pinning | typical assumption |
| Cost | Target under $0.01-0.05 per page. LLM calls only where needed | assumption |

## Diagram 1 - Online serving path (ingest -> digitize -> classify -> extract -> validate -> route)

The ranking template maps as follows: candidate generation = OCR tokens and regions, ranking
= field tagging / LLM extraction, re-ranking/policy = validation rules and confidence routing.

**Figure 29.1 - End-to-end online architecture of an invoice extraction service**

```text
 +-----------+  upload / email  +--------------+ doc_id, tenant  +-------------------+
 | Sources   |----------------->| Ingest API   |---------------->| Queue (SQS/Kafka) |
 | portal,   |                  | dedupe(hash) |                 | per-tenant shards |
 | email,    |                  | virus scan   |                 +---------+---------+
 | scanner,  |                  +--------------+                           |
 | mobile    |                                                             v
 +-----------+                          +-----------------------------------------+
                                        | Pre-processing                          |
                                        | PDF split, render 300 dpi, deskew,      |
                                        | denoise, rotate, born-digital text grab |
                                        +--------------------+--------------------+
                                                             | page images (+ text)
                                                             v
 +--------------------+  model   +-------------------------------------------------+
 | Model registry     |--------->| Digitize: OCR + layout                          |
 | OCR / layout /     | versions | text detection -> recognition -> words + bboxes |
 | classifier / KIE / |          | layout: blocks, tables, key-value pairs         |
 | LLM prompts        |          +------------------------+------------------------+
 +---------+----------+                                   | tokens, boxes, page img
           |                                              v
           |                     +------------------------------------------------+
           +-------------------->| Classify & split                               |
           |                     | invoice / receipt / credit note / PO / other   |
           |                     | page-to-document boundaries in multi-doc PDFs  |
           |                     +------------------------+-----------------------+
           |                                              | doc type
           |                                              v
           |   +-------------------+     +------------------------------------------+
           |   | Vendor store      |---->| Extract (KIE)                            |
           |   | vendor master,    | hint| layout transformer (token tags) or       |
           |   | known templates,  |     | VLM / LLM -> JSON schema (+ bbox grounds)|
           |   | past corrections  |     | per-field confidence                     |
           |   +-------------------+     |                                          |
           +---------------------------->| models + prompts pinned by registry      |
                                         +---------------------+--------------------+
                                                               | fields + conf
                                                               v
                                         +------------------------------------------+
                                         | Normalize & validate                     |
                                         | dates, currency, VAT IDs, sum(lines) =   |
                                         | subtotal, subtotal + tax = total,        |
                                         | PO / receipt 3-way match, dup invoice    |
                                         +---------------------+--------------------+
                                                               |
                               +-------------------------------+--------------+
                               | all checks pass and conf >= threshold        | else
                               v                                              v
                   +-------------------------+                 +---------------------+
                   | Straight-through post   |                 | Human review queue  |
                   | ERP / AP system, webhook|<----------------| validation station, |
                   +------------+------------+  corrected doc  | bbox highlight, keys|
                                |                              +----------+----------+
                                v                                         | corrections
                   +---------------------------------------------------------------+
                   | Logging store: pages, OCR output, predictions, confidences,   |
                   | validation results, human corrections, latency, cost/page     |
                   +---------------------------------------------------------------+
```

- The "feature store" of classic MLSD maps to the **vendor store** (vendor master data,
  per-vendor layout priors, past corrected values). The "model store" maps to the
  **model registry + prompt store**.
- Confidence routing is the policy layer. Thresholds are set per field and per tenant to hit
  a target precision on auto-posted fields.
- The AWS reference pipeline has the same stages: classification, extraction, enrichment,
  validation, human review (A2I) [9].

## Diagram 2 - Offline training & data pipeline (corrections -> labels -> train -> eval)

**Figure 29.2 - Offline data flywheel for document extraction**

```text
 +----------------------+   join on doc_id  +-------------------------------------+
 | Logging store        |------------------>| Label builder                       |
 | predictions, conf,   |                   | human corrections = gold fields     |
 | OCR, page images     |                   | ERP posted values = weak labels     |
 +----------------------+                   | map value -> OCR tokens (BIO tags)  |
                                            | via fuzzy string + bbox matching    |
 +----------------------+                   +-----------------+-------------------+
 | Annotation tool      |  boxes + fields                     |
 | (Label Studio-like)  |-------------------------------------+
 | active-learning      |                                     v
 | sample: low conf,    |                   +-------------------------------------+
 | new vendors, errors  |                   | Dataset versioning                  |
 +----------^-----------+                   | stratify by vendor, doc type, lang, |
            |                               | scan quality. Split BY VENDOR       |
            |                               +-----------------+-------------------+
            |                                                 |
 +----------+-----------+                                     v
 | Synthetic data       |  rendered docs  +---------------------------------------+
 | template renderer,   |---------------->| Training                              |
 | SynthDoG-style text, |                 | 1) OCR det/rec fine-tune              |
 | augment: blur, skew, |                 | 2) classifier (DiT / LayoutLMv3)      |
 | stamps, handwriting  |                 | 3) KIE token tagger or VLM / LLM SFT  |
 +----------------------+                 |    or prompt + few-shot set           |
                                          +-------------------+-------------------+
                                                              v
                                          +---------------------------------------+
                                          | Offline evaluation                    |
                                          | field F1 / exact match per field,     |
                                          | line-item table F1, STP @ precision,  |
                                          | per-vendor slices, unseen vendors     |
                                          +-------------------+-------------------+
                                                 pass gates   v
 +----------------------+                 +---------------------------------------+
 | Calibration          |<----------------| Model registry                        |
 | per-field thresholds |  champion /     | versioned models + prompts + schema   |
 | for target precision |  challenger     +-------------------+-------------------+
 +----------+-----------+                                     |
            |                                                 v
            |                             +---------------------------------------+
            +---------------------------->| Deployment: shadow -> canary ->       |
                                          | full. Compare STP, review rate, error |
                                          +-------------------+-------------------+
                                                              | new predictions
                                                              v
                                       (back to Logging store: feedback loop)
```

- Human corrections are the main label source. This is UiPath's "Digitize -> Classify ->
  Extract -> Validate -> Train" loop [26] (not opened).
- Split by vendor, not by document. A random split leaks layouts and inflates F1.

## Diagram 3 - Model architecture: OCR + layout transformer, and the OCR-free / LLM variant

**Figure 29.3 - Three extractor families and their inputs/outputs**

```text
 (A) Pipeline: OCR -> layout-aware transformer (LayoutLMv3-style) [3]
 +-------------+  words w_i   +-------------------------------------+
 | OCR engine  |------------->| Text embedding + 1D position        |
 | DB detector |  boxes b_i   | + 2D layout (x0,y0,x1,y1,w,h)       |
 | + CRNN/TrOCR|------------->| per token / segment                 |---+
 +-------------+              +-------------------------------------+   |
 +-------------+  patches     +-------------------------------------+   | concat
 | Page image  |------------->| Linear patch embedding (no CNN)     |---+
 +-------------+              +-------------------------------------+   |
                                                                        v
                                 +--------------------------------------------------+
                                 | Multimodal transformer (12 or 24 layers)         |
                                 | pretrain: MLM + masked image model + word-patch  |
                                 | alignment                                        |
                                 +--------------------------+-----------------------+
                                                            |
                                  +-------------------------+-------+
                                  v                                 v
                    +---------------------------+   +---------------------------+
                    | Token head: BIO tags      |   | Relation head: key<->value|
                    | B-TOTAL, I-VENDOR, ...    |   | line-item row grouping    |
                    +-------------+-------------+   +-------------+-------------+
                                  +---------------+---------------+
                                                  v
                                 fields + bboxes + softmax confidence

 (B) OCR-free encoder-decoder (Donut-style) [4]
 +-------------+  pixels  +---------------------+  tokens  +---------------------------+
 | Page image  |--------->| Swin vision encoder |--------->| Text decoder -> JSON-like |
 +-------------+          +---------------------+          | <s_total>12.50</s_total>  |
                                                           +---------------------------+

 (C) VLM / LLM with schema (GenAI answer)
 +-------------+  +--------------+  +------------------+  +---------------------------+
 | Page image  |  | OCR text +   |  | JSON schema +    |  | VLM / LLM                 |
 | (optional)  |->| layout as MD |->| vendor few-shot  |->| constrained JSON decoding |
 +-------------+  +--------------+  +------------------+  | + cite bbox / source span |
                                                          +-------------+-------------+
                                                                        v
                                     fields + evidence; confidence from logprobs,
                                     self-consistency, or OCR-span match
```

- (A) is cheap, fast and well calibrated. It needs labeled token data per field.
- (B) avoids OCR errors and is a single model. Reported CORD score about 91 (91.3 TED
  accuracy for the v2 checkpoint) and 0.7-1.2 s per image [4] (fetched).
- (C) is zero/few-shot and handles new vendors and fields. It costs more per page, can
  hallucinate values, and its confidence is harder to calibrate.

## Diagram 4 - Real system: AWS intelligent document processing (IDP) reference pipeline

**Figure 29.4 - AWS IDP reference flow (Textract + Comprehend + A2I + Bedrock) [9][10]**

```text
 +----------+ S3 put  +------------------+  pages   +----------------------------+
 | Upload   |-------->| S3 raw bucket    |--------->| Classification             |
 | to S3    |  event  | -> Lambda / SQS  |          | Bedrock prompt flow or     |
 +----------+         +------------------+          | Comprehend custom classify |
                                                    +-------------+--------------+
                                                                  | doc type
                                                                  v
                                                    +----------------------------+
                                                    | Extraction: Amazon Textract|
                                                    | AnalyzeExpense: summary    |
                                                    | fields + line-item groups, |
                                                    | normalized types + conf    |
                                                    +-------------+--------------+
                                                                  v
                                                    +----------------------------+
                                                    | Enrichment: Comprehend     |
                                                    | entities / PII, LLM fixes  |
                                                    +-------------+--------------+
                                                                  v
                                                    +----------------------------+
                                                    | Validation rules / agents  |
                                                    | (Validator, Matcher)       |
                                                    +------+--------------+------+
                                                           | ok           | low conf
                                                           v              v
                                            +----------------+  +--------------------+
                                            | DynamoDB / ERP |  | Amazon A2I human   |
                                            | downstream     |<-| review loop        |
                                            +----------------+  +--------------------+
```

**AnalyzeExpense output contract (what to mirror in your own schema) [10]**

| Element | Content |
|---|---|
| Summary fields | Normalized `Type` per field (e.g. `TOTAL`, `VENDOR_NAME`); unmatched keys returned as `OTHER` with the raw label |
| Line-item groups | Each row is an `EXPENSE_ROW` with `ITEM`, `QUANTITY`, `PRICE`, `UNIT_PRICE`, `PRODUCT_CODE` |
| Grounding | Page number + bounding box + polygon for every label and value |
| Confidence | Per label and per value; drives the A2I routing threshold |
| Sibling APIs | AnalyzeID for ID documents (KYC variant); `analyze_expense` / `analyze_id` wrappers in Textractor [31] |

- Keep the raw key text next to the normalized type. Reviewers and the label builder need it.
- The AWS repo also ships an agentic variant on Bedrock AgentCore with Analyzer, Matcher,
  Extractor, Validator and Troubleshooter agents, plus a Bedrock Prompt Flows variant with
  per-document flows and A2I review [9] (fetched).
- Uber's GenAI invoice post [11] (not opened) describes the same shape: OCR -> LLM extraction
  -> human-in-the-loop UI. Its numbers are not verified here.

## Data & labels

| Source | Label type | Notes |
|---|---|---|
| Human review corrections | Gold field values (+ bbox) | Best signal. Biased toward low-confidence docs |
| ERP posted values | Weak labels (value only) | Map to OCR tokens with fuzzy match. Noisy for repeated numbers |
| Annotation campaigns | Boxes + field + relation labels | Costly. Spend on new vendors and active-learning samples |
| Public sets | FUNSD [27], CORD [12], SROIE [28], DocVQA [29], RVL-CDIP | Pretraining / benchmarks. Domain shift vs real AP invoices |
| Synthetic | Rendered templates, SynthDoG [4] | Cold start, rare fields, new languages |

## Features / signals

| Group | Examples |
|---|---|
| Text | OCR tokens, subword embeddings, OCR confidence per word |
| Layout | Normalized bbox, reading order, block/table/cell IDs, distance to keywords ("Total", "IVA") |
| Visual | Page patches, font size/bold, lines and table rulings, logos, stamps |
| Document | Doc type, page index, language, born-digital vs scanned, DPI / blur score |
| Vendor priors | Vendor ID (from VAT ID / IBAN / logo), known field positions, past values, currency |
| Context | PO data for matching, tenant schema, expected tax rates |

## Model choices (baseline -> advanced)

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| OCR | Tesseract / cloud OCR | DB detector + CRNN / TrOCR [6], PP-OCRv5 (109-language rec) / PP-OCRv6 (1.5M-34.5M params, 50 langs) [19] | Accuracy on noisy scans vs cost; handwriting needs TrOCR-class models |
| Layout / tables | Heuristic line detection | DiT / LayoutLMv3 layout, table-structure model, PP-StructureV3 [19] | Better tables vs GPU cost |
| Classifier | TF-IDF + logistic regression on OCR text | DiT / LayoutLMv3 fine-tuned [5][3] | Visual cues matter for look-alike docs |
| KIE | Regex + per-vendor templates | LayoutLMv2/v3 token tagging [3][5]; graph models [8] | Templates break on layout drift; learned models generalize |
| End-to-end | n/a | Donut (OCR-free) [4]; PaddleOCR-VL-1.6, 0.9B params [19] | One model, but harder grounding and long-page limits |
| GenAI | Prompted LLM on OCR text | VLM with JSON schema + few-shot from vendor store; fine-tuned small VLM | Zero-shot new fields vs hallucination, cost, latency |
| Router | Single model | Cheap model first, escalate low-confidence docs to LLM | Cuts cost; adds complexity |

## Loss functions

| Model | Loss |
|---|---|
| Text detector (DB) | BCE / dice on probability + threshold maps |
| Text recognizer | CTC (CRNN) or cross-entropy (TrOCR seq2seq) |
| Classifier | Cross-entropy, class-weighted for rare doc types |
| Token tagger | Token-level cross-entropy over BIO tags (+ CRF optional) |
| Relation head | Binary cross-entropy over key-value / row-membership pairs |
| Donut / VLM SFT | Next-token cross-entropy on the serialized JSON |
| Pretraining (LayoutLMv3) | MLM + masked image modeling + word-patch alignment [3] |

## Offline metrics

| Metric | Use |
|---|---|
| CER / WER | OCR quality, sliced by scan quality and language |
| Field-level precision / recall / F1, exact match | Core KIE metric. Report per field (total matters more than memo) |
| Normalized edit distance / ANLS | Partial credit for strings (DocVQA style) |
| Line-item table F1 (row + cell) | Hardest part of invoices |
| Tree edit distance accuracy | Donut-style JSON output [4] |
| STP rate at precision >= 99% | Business-aligned: share of docs that skip review |
| Calibration (ECE), coverage-risk curve | Needed to set routing thresholds |
| Unseen-vendor slice | Generalization to new layouts |

## Online metrics / A-B

| Metric | Notes |
|---|---|
| Straight-through-processing rate | Primary. Docs auto-posted without human touch |
| Human touch time per doc, fields corrected per doc | Reviewer effort |
| Post-posting error rate | Errors found later in ERP (payment errors, duplicate payments) |
| End-to-end latency, backlog age | SLA |
| Cost per page | OCR + GPU + LLM tokens |

- A-B by tenant or by document hash. Shadow mode first: compare to the champion on the same docs.
- Guardrail: auto-posted precision must not drop, even if STP rises.

## Serving & latency budget (async, p95 under 60 s per 3-page doc; sync receipt p95 under 5 s)

| Step | Budget (approx.) | Notes |
|---|---|---|
| Ingest, split, render | 0.5-2 s | CPU pool |
| OCR + layout | 0.3-1 s / page | GPU batch; skip OCR for born-digital text |
| Classify | 50-100 ms | Small model |
| KIE layout transformer | 100-300 ms / page | GPU, batch by page |
| LLM / VLM extraction (if routed) | 2-10 s | Only for low-conf or new vendors; constrained JSON |
| Validate + normalize | under 100 ms | Rules + vendor lookups |
| Queueing slack | rest | Autoscale on queue depth |

## Monitoring

| Signal | Alert on |
|---|---|
| OCR confidence distribution, blank-page rate | Scanner or upload change |
| Per-field confidence and review rate | Drift for a vendor or tenant |
| Correction rate per field / vendor | New layout or model regression |
| Validation failure rate (sum checks) | Line-item extraction breaking |
| Doc-type mix, language mix | New document types arriving |
| LLM: JSON parse failures, invented values (not in OCR text), token cost | GenAI-specific |
| Latency, queue depth, GPU use | SLA risk |

## Failure modes

| Failure | Mitigation |
|---|---|
| Poor scans, skew, stamps over text | Pre-processing, quality score, ask for re-upload |
| Multi-page tables, line items spanning pages | Page-level context, table stitching, sum validation |
| Look-alike numbers (invoice # vs PO # vs order #) | Key-value relation head, vendor priors, LLM with bbox citation |
| Locale formats (1.234,56 vs 1,234.56; DD/MM vs MM/DD) | Locale from vendor country, currency, validation |
| LLM hallucinated fields | Require the value to appear in OCR text; abstain; route to review |
| Prompt injection in document text | Treat doc text as data; schema-constrained output; no tool calls from doc content |
| New vendor layouts | VLM fallback, active learning, fast few-shot via vendor store |
| Duplicate / fraudulent invoices | Hash + fuzzy dedupe, bank-detail-change alerts |
| Label noise from weak ERP labels | Confidence-weighted loss, audit samples |

## Interviewer follow-ups

- How do you get training labels without a big annotation budget? (review corrections, ERP weak labels, synthetic data)
- OCR + layout model vs OCR-free (Donut) vs VLM/LLM: which do you pick and why?
- How do you extract line-item tables, including tables that span pages?
- How do you produce a calibrated per-field confidence from an LLM?
- How do you set thresholds for auto-posting vs human review? What is the business metric?
- How do you handle a vendor never seen before?
- How do you stop the LLM from inventing a total or a bank account number?
- How do you split train/test to avoid layout leakage?
- How do you handle multi-document PDFs (invoice + PO + delivery note in one file)?
- How would you cut cost per page by 10x?
- How do you support 50+ languages and handwriting?
- How do you detect drift when a big vendor changes its template?
- How do you protect PII and bank details, and meet data-residency rules?
- What do you log so reviewer corrections become training data?
- Buy (Textract / Document AI style API) or build? When does a managed API's fixed schema (`OTHER` fields) stop being enough?
- How do you keep raw key text, normalized type and bbox together so every field is auditable? [10]

## Portal comparison

| Portal / source | Framing | Emphasis | Distinctive point |
|---|---|---|---|
| alirezadir ML-Interviews [1] | "OCR/Text recognition system" as a CV prompt; 9-step template | Problem framing, data, metrics, serving | Generic template applied to OCR |
| Chip Huyen MLSD exercises [2] | Document QA exercise (#23), not extraction | Retrieval over documents | Bridge to RAG (ch. 24) |
| Educative / ByteByteGo [20][22] | Framework-driven; no dedicated invoice case verified | Step-by-step MLSD | Reuse CV + GenAI chapters |
| AWS IDP repo [9] | Production reference, not interview-shaped | Classify -> extract -> enrich -> validate -> A2I | Agentic validator / matcher variant |
| UiPath DU [26] | Product framework | Digitize -> Classify -> Extract -> Validate -> Train | Human validation station feeds retraining |
| Model repos (unilm, Donut, PaddleOCR) [3][4][19] | Model-level | Architecture, benchmarks | Concrete numbers: FUNSD F1 0.92 (v3 large), CORD ~91 (Donut) |

## References

1. alirezadir, "ML System Design" (CV and GenAI prompt lists), GitHub, 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
2. C. Huyen, "Machine Learning Systems Design: Exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
3. Y. Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking", ACM Multimedia, 2022. https://github.com/microsoft/unilm/tree/master/layoutlmv3 (fetched); https://arxiv.org/abs/2204.08387 (linked from repo)
4. G. Kim et al., "OCR-free Document Understanding Transformer" (Donut), ECCV, 2022. https://github.com/clovaai/donut (fetched); https://arxiv.org/abs/2111.15664 (linked from repo)
5. Microsoft, "unilm" repo (LayoutLMv2, DiT and other Document AI models), GitHub, 2024. https://github.com/microsoft/unilm (fetched); LayoutLMv2 https://arxiv.org/abs/2012.14740 (linked from repo); DiT https://arxiv.org/abs/2203.02378 (linked from repo)
6. M. Li et al., "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models", arXiv, 2021. https://arxiv.org/abs/2109.10282 (linked from repo)
7. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
8. E. Yan, "applied-ml" (includes Nanonets "Information Extraction from Receipts with Graph Convolutional Networks", 2021), GitHub, 2024. https://github.com/eugeneyan/applied-ml (fetched); https://nanonets.com/blog/information-extraction-graph-convolutional-networks/ (listed in [8])
9. AWS Samples, "aws-ai-intelligent-document-processing", GitHub, 2025. https://github.com/aws-samples/aws-ai-intelligent-document-processing (fetched)
10. AWS, "Analyzing Invoices and Receipts" (Textract AnalyzeExpense), Amazon Textract Developer Guide, 2024. https://docs.aws.amazon.com/textract/latest/dg/invoices-receipts.html; GitHub source of the page https://github.com/awsdocs/amazon-textract-developer-guide/blob/master/doc_source/invoices-receipts.md (fetched, raw)
11. Uber Engineering, "Advancing Invoice Processing Efficiency at Uber with GenAI", uber.com, 2024. https://www.uber.com/blog/advancing-invoice-processing-with-genai/ (not opened)
12. S. Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", NeurIPS Document Intelligence Workshop, 2019. https://github.com/clovaai/cord (fetched); https://openreview.net/pdf?id=SJl3z659UH (linked from repo)
13. Uber Engineering, "Uber's Real-Time Document Check", uber.com, 2022. https://www.uber.com/en-GB/blog/ubers-real-time-document-check/ (listed in [7])
14. Google AI, "Extracting Structured Data from Templatic Documents", Google AI Blog, 2020. https://ai.googleblog.com/2020/06/extracting-structured-data-from.html (listed in [8])
15. Dropbox, "Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning", dropbox.tech, 2017. https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning (listed in [8])
16. Dropbox, "Using Machine Learning to Index Text from Billions of Images", dropbox.tech, 2018. https://dropbox.tech/machine-learning/using-machine-learning-to-index-text-from-billions-of-images (listed in [8])
17. Scribd, "Identifying Document Types at Scribd", tech.scribd.com, 2021. https://tech.scribd.com/blog/2021/identifying-document-types.html (listed in [7][8])
18. Scribd, "Information Extraction at Scribd", tech.scribd.com, 2021. https://tech.scribd.com/blog/2021/information-extraction-at-scribd.html (listed in [7])
19. PaddlePaddle, "PaddleOCR" (PP-OCRv5/v6, PP-StructureV3, PaddleOCR-VL), GitHub, 2026. https://github.com/PaddlePaddle/PaddleOCR (fetched)
20. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework (ch-ref)
21. Educative, "Grokking the Generative AI System Design", educative.io, 2025. https://www.educative.io/courses/generative-ai-system-design (ch-ref)
22. ByteByteGo, home page (ML and Generative AI System Design Interview courses), bytebytego.com, 2025. https://bytebytego.com (ch-ref)
23. IGotAnOffer, "Generative AI System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/generative-ai-system-design-interview (ch-ref)
24. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
25. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
26. UiPath, "Document Understanding" documentation (framework: digitize, classify, extract, validate, train), docs.uipath.com, 2025. https://docs.uipath.com/document-understanding (not opened)
27. G. Jaume et al., "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents", arXiv, 2019. https://arxiv.org/abs/1905.13538 (not opened)
28. Z. Huang et al., "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction" (SROIE), arXiv, 2021. https://arxiv.org/abs/2103.10213 (not opened)
29. M. Mathew et al., "DocVQA: A Dataset for VQA on Document Images", arXiv, 2020. https://arxiv.org/abs/2007.00398 (not opened)
30. Scribd, "Categorizing user-uploaded documents", tech.scribd.com, 2021. https://tech.scribd.com/blog/2021/categorizing-user-uploaded-documents.html (listed in [7])
31. AWS Samples, "amazon-textract-textractor" (analyze_expense, analyze_id), GitHub, 2025. https://github.com/aws-samples/amazon-textract-textractor (fetched)
