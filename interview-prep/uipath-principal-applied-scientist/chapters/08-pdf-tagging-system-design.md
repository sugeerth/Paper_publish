# 08. End-to-End ML System Design: A PDF Tagging System (Structure, Accessibility and Document Understanding)

> **How to read this chapter.** The rest of the pack touches PDF tagging in scattered pieces: chapter 04 section C (Document Understanding), chapter 05 sections A and B (ML coding on layout and tables), and the leadership diagram in chapter 06. This chapter puts the whole system in one place. It opens with one end-to-end diagram. Each section then has a smaller diagram for its stage, numbered questions and follow-up probes. It has questions only, with no answers. The diagrams are plain text so they render the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one of two tags:
> - **[Reported] [n]**: a candidate report links this question to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role (Principal Applied Scientist), the level and UiPath's document products: Document Understanding (DU Modern), IXP, validation in Action Center, and published project versions and tags. No candidate has reported it. A **[Likely] [n]** tag means source [n] confirms the *premise* of the question, such as a product behaviour, dataset or metric definition. It does not mean a candidate was asked it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **"Tag" has two meanings here.** (1) A *PDF structure tag*: an element of the tagged-PDF structure tree, such as H1, P, L/LI, Table/TR/TH/TD, Figure with alt text, or Span. This is what PDF/UA accessibility needs. (2) A *semantic tag or label*: a document class, a key-value field or an entity, as in Document Understanding and IXP. UiPath also uses "tag" for a published project version [1]. A principal-level answer names which meaning it is using, and the chapter tests both.
>
> **Research limits (please read).** This pass (October 2026) could not run any web searches: the workflow's shared search budget was already used up. The fetch proxy could not resolve or reach glassdoor.com, teamblind.com, uipath.com, docs.uipath.com, w3.org or pdfa.org. Only GitHub-hosted sources were reachable. The pass read UiPath's own SDK source and public repositories for document layout, tables, reading order and PDF/UA validation [1]-[11]. Those sources confirm product behaviour and metric definitions, not interview content. So **no question below is tagged [Reported]**, and the chapter does not invent reports. Check product names and current PDF/UA details (ISO 14289-1 and -2, the Matterhorn Protocol, the PDF 2.0 structure namespace) against primary documents before the interview.

## Format notes for this round

| Item | Expected shape (inferred, not confirmed) |
|---|---|
| Round | ML / applied science system design. For a document-AI team it may be a deep dive on one document pipeline, or a design prompt followed by a probe on your own past document work |
| Duration | About 60 min: 5 min scoping, 30-35 min design, 15 min deep dives (evaluation, labelling, scale), 5 min your questions |
| Interviewer | Principal or staff scientist, or an architect from the document / IXP side; possibly a product manager for the human-review and customer angle |
| Likely prompt | "Design a system that turns any incoming PDF (native or scanned) into a correctly tagged, accessible PDF and structured data, for enterprise customers at scale" |
| Principal-level bar | You define the tag taxonomy and the success metrics before you pick models. You treat evaluation, labelling and human review as first-class systems, not afterthoughts. You separate the deterministic work (PDF parsing, schema validation, PDF/UA checks) from the probabilistic work (layout, reading order, semantics). You reason about cost per page, latency tiers, on-prem and regulated deployment, and versioning. You name the open research problems and a multi-quarter plan. |
| Common failure modes | Treating it as "just OCR plus an LLM"; no reading-order story; ignoring nested tables and multi-page objects; no plan for the confidence threshold and the review queue; metrics that do not match what the customer (screen-reader user, auditor or downstream robot) experiences |

### Reference diagram: the end-to-end PDF tagging system

```
 INGEST                     ROUTE                         PERCEIVE
┌──────────────┐   ┌─────────────────────────┐   ┌───────────────────────────────┐
│ API / email /│   │ Native vs scanned vs    │   │ Native: text runs, fonts,     │
│ queue / batch│──>│ hybrid page detector    │──>│ vector graphics, existing tags│
│ dedup, virus │   │ (text layer quality,    │   │ Scanned: deskew, denoise,     │
│ scan, PII    │   │ image coverage, fonts,  │   │ OCR (lines, words, boxes,     │
│ policy, split│   │ existing /StructTreeRoot│   │ word confidences, script/lang)│
└──────────────┘   └─────────────────────────┘   └───────────────┬───────────────┘
                                                                 │ page tokens + boxes + image
 UNDERSTAND                                                      ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Layout detection (regions: title, heading, paragraph, list, table, figure,     │
│ caption, header/footer, footnote, formula, form field)                         │
│ Reading order (within region, across columns, across pages)                    │
│ Table structure (rows, cols, spans, header cells, multi-page tables)           │
│ Logical structure: heading levels, list nesting, captions <-> figures,         │
│ key-value pairs, entities, spans (language, emphasis, links)                   │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ candidate document tree + confidences
 SCHEMA                                ▼
┌──────────────────────────────┐   ┌────────────────────────────────────────────┐
│ Tag taxonomy + schema        │──>│ Tree builder + validator: canonical tree   │
│ (versioned; maps internal    │   │ (JSON), schema checks, role map to PDF     │
│ labels -> PDF std structure  │   │ standard types, alt-text generation        │
│ types, and -> DU/IXP fields) │   │ for figures, language tags                 │
└──────────────────────────────┘   └───────────────┬────────────────────────────┘
                                                   │
                    ┌──────────────────────────────┼─────────────────────────────┐
                    ▼                              ▼                             ▼
          ┌──────────────────┐         ┌──────────────────────┐      ┌──────────────────────┐
          │ Confidence gate  │  low    │ HUMAN REVIEW         │      │ OUTPUT WRITERS        │
          │ (per element +   │────────>│ validation station / │      │ Tagged PDF (struct    │
          │ per document)    │         │ Action Center task,  │      │ tree, MCIDs, RoleMap, │
          └────────┬─────────┘         │ tag editor           │      │ /Alt, /Lang) ; JSON;  │
                   │ high              └──────────┬───────────┘      │ Markdown/HTML; fields │
                   └─────────────────────────────>│──────────────────>└──────────┬───────────┘
                                                  │ corrections                  │
                                                  ▼                              ▼
 LEARN                                ┌────────────────────────┐   ┌──────────────────────────┐
┌─────────────────────────────────┐   │ VERSIONED LABEL STORE  │   │ PDF/UA validator         │
│ Annotation tooling (expert      │──>│ (doc, page, tree, tax- │   │ (machine-checkable rules)│
│ guidelines, multi-annotation,   │   │ onomy version, source: │   │ + delivery + audit log   │
│ adjudication, active learning)  │   │ human/model/weak)      │   └──────────────────────────┘
└─────────────────────────────────┘   └───────────┬────────────┘
                                                  ▼
                         ┌───────────────────────────────────────────────────┐
                         │ TRAIN + EVAL: tag F1, tree-edit distance, reading- │
                         │ order accuracy, table TEDS, PDF/UA pass rate, human│
                         │ minutes per doc; model registry; staged rollout    │
                         └───────────────────────────────────────────────────┘
```

## A. Problem framing, users and success metrics

1. [Likely] Design a system that takes any enterprise PDF and produces (a) a tagged, accessible PDF and (b) structured data for downstream automation. Where do you start, and what do you ask before you draw anything?
   - Who are the users: screen-reader users, compliance officers, downstream robots or agents, or data teams? Do they need the same tree?
   - What does "done" mean for one document: PDF/UA machine-check pass, human-verified, or fields extracted?
   - What volume, page mix (native vs scanned), languages and latency tiers do you assume, and how would you check those assumptions?
2. [Likely] [1] UiPath already exposes classify, extract and validate as separate steps with published project versions and tags. How would a structure-tagging capability fit next to those steps rather than duplicate them?
   - Does structure tagging run before extraction, as shared infrastructure, or as a separate product?
   - What contract (schema) would extraction consume from the tagging stage?
3. [Likely] What is your north-star metric for this system, and what are the guardrail metrics?
   - How do you turn "accessible" into something you can measure offline?
   - How do you weigh human-correction minutes per document against model accuracy?
4. [Likely] Which parts of this problem are deterministic and which are probabilistic? How does that split shape the architecture?
   - What should never be left to a model (for example, writing valid PDF syntax or checking the schema)?
5. [Likely] An enterprise customer has 40 million legacy PDFs to remediate for accessibility in 12 months. How does that change your design compared with a real-time stream of new invoices?
   - What would you do in month one so that you learn the most?
6. [Likely] What would you build versus buy versus take from open source (for example, layout models, table models, OCR, PDF/UA validators)?
   - How do licences shape that choice, for example a model released under a non-commercial licence [7] or a dataset limited to research use [11]?

## B. Ingest, routing and native vs scanned detection

```
PDF ──> sanitise (encryption, corrupt xref, JS, embedded files) ──> page split
     ──> per-page features: text-layer char count, glyph->Unicode mapping quality,
         image area ratio, invisible OCR text layer present?, existing StructTreeRoot?
     ──> router: NATIVE | SCANNED | HYBRID | ALREADY-TAGGED (verify / repair)
```

7. [Likely] How do you decide, per page, whether a PDF is native, scanned or hybrid? What features and labels would you use?
   - How do you detect a text layer that exists but is garbage (broken ToUnicode maps, ligatures, Type 3 fonts)?
   - What do you do with a scanned page that already has an invisible OCR layer from a previous tool?
8. [Likely] Why decide per page instead of per document? What breaks downstream if you route a whole document one way?
9. [Likely] Some incoming PDFs are already tagged, but badly. Do you trust, repair or discard the existing tags? How do you decide?
   - How would you measure the quality of existing tags without a human label?
10. [Likely] What security and privacy controls belong at ingest (encrypted PDFs, embedded JavaScript, attachments, PII), and how do they interact with the ML stages?
    - What changes for an on-prem or air-gapped customer?
11. [Likely] How do you handle very long documents (1,000+ pages) and very large pages (engineering drawings) in the pipeline?
    - Where do you parallelise, and which stages need the whole document?

## C. OCR and the perception layer

12. [Likely] For scanned pages, what does your OCR stage output, and what extra outputs does the tagging stage need beyond plain text?
    - Why do word-level boxes and confidences matter for reading order and review?
13. [Likely] How would you choose between a classical OCR engine, a document-specific OCR model and an end-to-end vision-language model that outputs structure directly?
    - What are the cost, latency, hallucination and auditability trade-offs?
    - How do you guard against a generative model "reading" text that is not on the page?
14. [Likely] How do you handle mixed scripts, right-to-left text, vertical text, handwriting and low-quality faxes?
    - How do these affect the /Lang attribute and reading order in the output?
15. [Likely] For native PDFs, the text is "free", but the order of drawing operators often does not match the reading order. How do you turn content-stream order into usable tokens?
    - How do you handle text drawn as vector paths, or text hidden under images?
16. [Likely] Where does pre-processing (deskew, dewarp, binarisation, rotation detection) belong, and how do you stop it from hurting native pages?

## D. Layout analysis and reading order

```
page image + tokens ──> layout detector (regions + class + score)
                    ──> region-level reading order (columns, sidebars, floats)
                    ──> token-level order inside regions
                    ──> cross-page links (continued paragraphs, tables, lists)
```

17. [Likely] [3] Define the layout classes you would detect. Public datasets such as DocLayNet use 11 classes over about 80,000 pages from six document categories [3]. Is that the right taxonomy for an enterprise tagging product?
    - What would you add or merge (form fields, signatures, stamps, key-value regions)?
    - How do you keep the layout classes separate from, but mappable to, the PDF structure tags?
18. [Likely] [3] DocLayNet annotated a fraction of pages two or three times to estimate annotation uncertainty [3]. How would you use multiply-annotated pages when you set targets for your own layout model?
    - If humans agree on only 80% of boxes for a class, what accuracy target is meaningful for the model?
19. [Likely] [7] Compare a vision-only detector, a multimodal model that combines text, layout and image (for example, LayoutLMv3 [7]) and a vision-language model for layout. Which would you use for which stage, and why?
    - How does the choice change if the customer runs on CPU-only on-prem hardware?
20. [Likely] [6] How would you model reading order? Compare heuristic sorting (XY-cut, column detection), a learned seq2seq model such as LayoutReader [6] and graph or pairwise-ranking approaches.
    - LayoutReader reports large gains over a heuristic baseline on ReadingBank [6]. Why might those gains shrink on your customers' documents?
    - ReadingBank's labels come from weak supervision on Word documents [6]. What biases does that bring?
21. [Likely] How do you define the "correct" reading order when two reasonable people would disagree (sidebars, pull quotes, footnotes, multi-column with figures)?
    - How does that ambiguity affect your metric?
22. [Likely] How do you handle objects that continue across pages: paragraphs split by a page break, tables with repeated headers, numbered lists?
    - Where in the pipeline do you merge them, and how do you represent the merge in the tagged PDF?
23. [Likely] Headers, footers, page numbers and watermarks should usually become artifacts, not content. How do you detect them robustly across a whole document?
    - What happens to a screen-reader user if you get this wrong in either direction?

## E. Logical structure tagging: headings, lists, tables, figures, key-value pairs, spans

24. [Likely] How do you infer heading levels (H1-H6) from a page that only shows font sizes and positions? What signals do you use beyond font size?
    - How do you keep heading levels consistent across a 300-page document?
    - Is document-level inference (a global pass) worth the cost over page-level inference?
25. [Likely] How do you detect lists and their nesting, including lists whose bullets are images or whose numbers are drawn separately from the text?
    - How do you represent label (Lbl) and body (LBody) separately?
26. [Likely] [4] [8] Design the table pipeline: detection, structure recognition (rows, columns, spanning cells), header detection and cell content mapping. Which parts would you learn and which would you compute?
    - Public models such as TableFormer predict structure and cell boxes [8], trained on sets such as PubTabNet, FinTabNet and TableBank [4] [8] [11]. Why may these not transfer to scanned invoices or bank statements?
    - How do you decide which cells are headers, and how do you emit header-to-cell associations (scope or IDs) for accessibility?
    - How do you handle borderless tables, nested tables and tables split across pages?
27. [Likely] When is something a table for accessibility purposes but not for data extraction, or the other way round (for example, a layout table or a form laid out as a grid)?
28. [Likely] How do you link figures to their captions, and how do you decide whether a figure is meaningful or decorative?
29. [Likely] Design alt-text generation for figures, charts and diagrams. How do you evaluate it, and when does a human have to write or approve it?
    - What are the risks of a vision-language model writing confident but wrong alt text for a chart in a financial filing?
    - How long should alt text be, and how do you use surrounding context?
30. [Likely] How do you detect key-value pairs on forms and semi-structured documents, and how do they map both to structure tags and to DU or IXP fields?
    - Should a key-value pair be one structure element or two? Who decides?
31. [Likely] Which inline spans matter (language changes, emphasis, links, abbreviations with expansions, formulas), and how do you detect and tag them?
    - How would you produce accessible output for mathematical formulas?
32. [Likely] Would you use one large model that outputs the whole tree, or a cascade of specialised models? Defend the choice on accuracy, debuggability, cost and how fast you can iterate.
    - How do errors compound through a cascade, and how would you measure that?
    - How do you ensure a generated tree is always well formed (constrained decoding, post-validation, repair)?

## F. Tag taxonomy and schema

```
internal taxonomy v(N)            role map                  PDF standard structure types
 title, section_heading(level) ──────────────────────────> H1..H6 (or H in PDF 2.0 usage)
 paragraph, list(item, label) ───────────────────────────> P, L / LI / Lbl / LBody
 table(row, header_cell, cell) ──────────────────────────> Table / TR / TH / TD
 figure(alt), caption ───────────────────────────────────> Figure (/Alt), Caption
 kv_pair(key, value), entity ──> custom types ──RoleMap──> standard type (+ DU/IXP field)
 header/footer/page_no ──────────────────────────────────> Artifact (not in tree)
```

33. [Likely] Design the tag taxonomy and its schema. What is the canonical internal representation, and why not use PDF structure types directly?
    - How do you represent confidence, provenance (model, version, human) and bounding boxes on each node?
    - Docling, for example, uses one internal document model and exports it to Markdown, HTML, DocTags and lossless JSON [2]. What do you gain and lose with a single canonical tree?
34. [Likely] How do you version the taxonomy, and what happens to old labels, old models and customer outputs when you split or merge a class?
    - How do you migrate a label store from taxonomy v3 to v4 without relabelling everything?
35. [Likely] Customers want their own custom tags, such as "ClauseIndemnity" or "InvoiceLineItem". How do you support them without breaking accessibility output?
    - How does role mapping to standard types keep a custom tag valid for assistive technology?
36. [Likely] How strict should the schema be? Which rules are hard constraints (a TD must sit inside a TR) and which are soft (a document should have one H1)?
    - Where do you enforce each: in decoding, in post-processing, or in validation?
37. [Likely] How would you make the same tree serve accessibility, extraction and RAG chunking for agents? Where do these uses conflict?

## G. Tagged-PDF and PDF/UA output

```
canonical tree ──> structure tree writer: StructTreeRoot, StructElems, parent tree,
                   marked-content IDs (MCIDs) linking tree <-> page content,
                   RoleMap for custom types, /Alt, /ActualText, /Lang, Artifacts
               ──> PDF/UA validator (machine-checkable rules) ──> report
               ──> residual checks needing human judgement (alt-text quality, order)
```

38. [Likely] Walk through how you write the tagged PDF: how the structure tree links to the page content, and how you mark headers and footers as artifacts.
    - What goes wrong if marked content is split, or if one content stream draws text for several elements?
    - Do you rewrite content streams or only add structure? What are the risks of each?
39. [Likely] [9] [10] Open-source validators such as veraPDF check PDF/UA-1, PDF/UA-2 and WTPDF rules, with WCAG 2.2 profiles in an experimental stage [9] [10]. What share of accessibility can a machine validator check, and what remains for human judgement?
    - How do you report "passes machine checks" versus "human-verified accessible" honestly to customers?
40. [Likely] How do you set /Lang, /Alt, /ActualText and expansion text, and which of them can you generate automatically?
41. [Likely] How do you handle PDF 2.0 and PDF/UA-2 versus PDF 1.7 and PDF/UA-1 outputs? Does the internal tree change?
42. [Likely] How do you test the output with real assistive technology (screen readers), not only validators? How would you scale that testing?
43. [Likely] Writing a tagged PDF must not change how the page looks. How do you guarantee visual fidelity and detect regressions in the writer?
    - What is your regression test corpus for the writer, and how is it different from the ML eval set?

## H. Confidence, calibration and human review

```
per-element scores ──> calibrated confidence ──> document-level risk score
   ├─ all elements >= threshold and validator passes ──> auto-publish
   ├─ some elements low ──> targeted review (only the risky regions)
   └─ document very low / unsupported type ──> full manual remediation queue
corrections ──> label store (with reviewer id, time spent, taxonomy version)
```

44. [Likely] [1] UiPath's document APIs include validation actions that block until a person completes them in Action Center [1]. Design the human-review step for structure tags: what does the reviewer see, and what do they fix?
    - How do you show a tree, reading order and table structure so that review is fast?
    - What is the unit of review: a page, an element, or a document?
45. [Likely] How do you produce calibrated confidence for a structured output like a tree or a table, not just for a single class?
    - How do you combine element confidences into a document-level decision?
    - How do you check calibration after a model update?
46. [Likely] How do you set the auto-publish threshold? Who owns the trade-off between review cost and risk, and how does it vary by customer or document type?
    - What happens when the threshold sends 60% of a large customer's documents to review on day one?
47. [Likely] How do you measure and reduce reviewer effort (minutes per page, edits per page), not just model error?
    - How would you run an A/B test on the review UI itself?
48. [Likely] How do you prevent reviewers from rubber-stamping model output (automation bias), and how would you detect it in the data?
    - Would you inject known errors to audit reviewers? What are the ethical and practical concerns?
49. [Likely] Review corrections are a training signal but they are biased: reviewers only see low-confidence items. How do you use them without biasing the next model?

## I. Annotation and labelling tooling

50. [Likely] Design the annotation tool for structure tags. How is it different from a bounding-box tool?
    - How do annotators label reading order, nesting and table spans efficiently?
    - How do you pre-fill with model output without anchoring annotators?
51. [Likely] Write the outline of the annotation guideline for headings, lists and tables. Which edge cases would you settle up front?
    - How do you measure inter-annotator agreement for trees and orders, not only boxes?
    - What process resolves disagreements (adjudication), and how do guideline changes flow back into old labels?
52. [Likely] [3] [6] [11] Public datasets use different labelling strategies: expert manual labelling [3] and weak supervision from Word or LaTeX source files [6] [11]. When would you use each for your product, and how would you combine them?
    - Could you build synthetic tagged PDFs from Word or HTML sources? What gap would remain to real scanned documents?
53. [Likely] Design an active-learning loop. Which documents should humans label next, and how do you avoid only labelling the hardest outliers?
    - How do you balance labelling budget across document types, languages and tag classes?
54. [Likely] Customers' documents are confidential. How do you get labelled data from them (consent, on-tenant labelling, federated or per-tenant fine-tuning), and how do you keep tenants separate?
55. [Likely] Would you use LLMs or VLMs as annotators or pre-labellers? How do you check the quality of machine labels before they enter training?

## J. Versioned label store and data management

```
label record = (doc_id, page, element_id, geometry, tag, parent, order_index,
                attributes, taxonomy_version, source{human|model|weak|synthetic},
                annotator_id, adjudicated?, created_at, supersedes)
snapshots: train/val/test frozen by hash  ──>  model registry links model <-> snapshot
```

56. [Likely] Design the versioned label store. What is the record schema, and how do you represent a tree, an order and a table so they can be diffed and queried?
    - How do you store several competing labels for the same element (multi-annotation, model vs human)?
57. [Likely] How do you make training and evaluation reproducible: frozen snapshots, dataset hashes, and links from each model version to the exact data it saw?
58. [Likely] How do you prevent leakage between train and test when the same template or vendor appears many times?
    - Would you split by document, by template, by customer or by time? Why?
59. [Likely] A customer asks you to delete all their data, including labels and anything derived from it. What does your label store and model lineage need to support?
60. [Likely] How do you track label quality over time and find bad labels in the store (for example, with model disagreement or confident-learning methods)?

## K. Evaluation

```
offline:  layout mAP | tag F1 (per class, per level) | tree-edit distance (whole tree)
          reading-order accuracy (pairwise / BLEU / edit distance) | table TEDS (+structure-only)
          alt-text quality (human rubric) | PDF/UA machine-check pass rate
online:   auto-publish rate | review minutes per doc | post-publish defect rate
          customer-reported accessibility issues | cost per page | p95 latency
```

61. [Likely] Define tag F1 for a tree. How do you match predicted elements to ground-truth elements before you count true positives?
    - Do you require geometric overlap, text overlap or both? What IoU threshold, and why?
    - Do you report micro or macro F1? How do you weight rare but critical tags (table headers) against frequent ones (paragraphs)?
62. [Likely] Why is element-level F1 not enough for a tree? Define a tree-edit-distance metric for the whole structure tree.
    - What are the insert, delete and rename costs, and should a wrong heading level cost as much as a missing table?
    - How do you normalise it so that scores compare across documents of different sizes?
63. [Likely] [4] [5] TEDS scores tables as 1 minus the tree edit distance divided by the size of the larger tree. Renaming a cell costs the normalised Levenshtein distance of its content, and a structure-only mode ignores the content [4] [5]. What are the weaknesses of TEDS for your use case?
    - The reference code does not clamp the score, so in theory it can fall below 0 [5]. Does that matter in reporting?
    - When would you report structure-only TEDS, full TEDS and cell-level adjacency F1 side by side?
    - How would you score header-cell identification, which accessibility depends on?
64. [Likely] [6] How do you measure reading-order accuracy? Compare page-level BLEU, average relative distance [6], Kendall's tau and pairwise accuracy within regions.
    - Should you measure reading order on tokens, on lines or on regions? What does each hide?
    - How do you score order when the ground truth allows several valid orders?
65. [Likely] How do you build an evaluation set that reflects your customers and not public benchmarks? How large must it be per class to detect a 1-point regression?
    - How do you stratify (native vs scanned, language, document type, page complexity)?
66. [Likely] Offline metrics improved but customers report more accessibility complaints. How do you debug the gap?
67. [Likely] How do you evaluate the system end to end with screen-reader users or accessibility experts, and how do you turn their findings into a repeatable metric?
68. [Likely] How do you evaluate alt text and other generated text without relying only on an LLM judge? If you do use an LLM judge, how do you validate it?
69. [Likely] Define the release gate for a new model version. Which metrics block a release, and which are only monitored?
    - How do you handle a model that improves table TEDS but makes heading F1 worse?

## L. Serving at scale

```
         ┌─ realtime API (single doc, seconds; p95 SLA) ─┐
request ─┤                                               ├─> stage DAG on GPU/CPU pools
         └─ batch / backlog (millions of pages, hours) ──┘     (page-level fan-out,
                                                               doc-level fan-in)
caches: page hash -> OCR/layout result ; model registry ; per-tenant config + taxonomy
observability: per-stage latency, queue depth, confidence drift, validator failures
```

70. [Likely] Design serving for two workloads: an interactive API (one document, a few seconds) and a backlog of millions of pages. What is shared and what is separate?
    - Which stages run per page in parallel and which need the whole document?
    - How do you schedule GPU work across tenants fairly?
71. [Likely] Estimate the cost per 1,000 pages for your design. Which stage dominates, and what would you change to cut cost by 5x?
    - When is a smaller distilled model plus targeted review cheaper than a large VLM on every page?
72. [Likely] How do you cache and deduplicate work (identical pages, repeated templates, re-runs after a model update)?
73. [Likely] How do you roll out a new model version safely: shadow mode, canary by tenant, rollback? How do customers pin a version [1]?
    - What does a customer expect when the same document gives a different tree after an upgrade?
74. [Likely] What do you monitor in production to detect drift (new templates, new scanners, new languages) before customers notice?
    - Which signals work without labels: confidence distribution, validator failure rate, review edit rate?
75. [Likely] How does the design change for an on-prem, air-gapped deployment with no GPU?
76. [Likely] How do you make the pipeline fail safe? If a stage crashes or times out on a 500-page document, what does the customer get?
    - How do retries and idempotency work when outputs are written to the customer's storage?

## M. Integration with UiPath products and agents

77. [Likely] [1] How would robots and agents consume the tagging output, for example, to drive classification, extraction and validation steps [1]?
    - What do agents need from the tree that a screen reader does not (stable element IDs, provenance, confidence)?
78. [Likely] How would you use the structure tree to improve retrieval (chunking by section, keeping tables whole) for agent grounding over enterprise documents?
    - How would you measure whether structure-aware chunking actually improves answer quality?
79. [Likely] An agent asks a question about a table on page 47, and the answer is wrong because the table was split across pages. Whose bug is this, and how would your metrics have caught it?
80. [Likely] How would you govern this system for regulated customers: audit logs, explainability of each tag, data residency, and model risk management?

## N. Research depth and open problems

81. [Likely] What are the hardest open research problems in document structure understanding today, and which would you fund first?
    - Where do you expect vision-language models to replace the cascade within two years, and where not?
82. [Likely] How would you build a single model that predicts layout, reading order and hierarchy jointly? What would the training targets and the loss look like?
    - How do you represent a tree as a target sequence, and how do you keep the decoded output valid?
83. [Likely] How would you adapt the system to a new document domain with 50 labelled documents?
    - Which parts transfer, and where do you spend the labels?
84. [Likely] How would you use self-supervised or weakly supervised signals from native PDFs that already have good tags, or from Word and HTML sources, to train models for scanned documents?
85. [Likely] How would you estimate and report uncertainty over a whole tree rather than over single nodes?
86. [Likely] What would make you stop using public benchmarks such as DocLayNet, PubTabNet or ReadingBank [3] [4] [6] for decisions? What would you publish instead?

## O. Principal-level scoping, roadmap and trade-offs

87. [Likely] Present a 4-quarter roadmap for this system. What ships in quarter 1 to deliver customer value while you build the learning loop?
    - What do you deliberately not build?
88. [Likely] Which teams own which parts (OCR, layout, tables, review UI, PDF writer, label operations, platform)? Where do you expect friction, and how do you resolve it?
89. [Likely] Product wants "100% PDF/UA compliant" in marketing. How do you respond, and what claim would you sign off on?
90. [Likely] A large customer escalates: 2% of tagged PDFs read tables in the wrong order to screen-reader users. Walk through your first week.
    - How do you separate a model problem from a writer bug or a taxonomy gap?
    - What process change stops this class of issue from recurring?
91. [Likely] How would you decide between investing in a better model and investing in a better review experience, given a fixed budget?
92. [Likely] What did you learn from a document-AI or structured-prediction system you shipped, and how would that experience change this design?
    - What metric did you get wrong at first, and how did you find out?

## References

1. UiPath, Documents service source, `uipath-platform` package of `uipath-python`: classify, extract (DU Modern and IXP), published project version and tag parameters, and validation actions that block until a person completes them in Action Center; marked experimental (official; fetched in this pass). GitHub, 2025-2026. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/documents/_documents_service.py
2. Docling project, "Docling" README: PDF understanding including page layout, reading order, table structure, OCR for scanned PDFs, a unified DoclingDocument model, and export to Markdown, HTML, DocTags and lossless JSON (fetched). GitHub. https://raw.githubusercontent.com/docling-project/docling/main/README.md
3. DS4SD (IBM Research), "DocLayNet" README: 80,863 expert-annotated pages from six document categories, 11 layout classes, with a fraction of pages annotated two or three times (fetched). GitHub, 2022. https://raw.githubusercontent.com/DS4SD/DocLayNet/main/README.md
4. IBM, "PubTabNet" README: over 568k table images with HTML structure annotations and cell boxes; introduces the TEDS metric (fetched). GitHub, 2019-2020. https://raw.githubusercontent.com/ibm-aur-nlp/PubTabNet/master/README.md
5. IBM, PubTabNet TEDS reference implementation, `metric.py`: TEDS = 1 - edit distance / max node count, normalised Levenshtein rename cost for cells, structure-only option (fetched). GitHub. https://raw.githubusercontent.com/ibm-aur-nlp/PubTabNet/master/src/metric.py
6. Microsoft, "LayoutReader" README (unilm): reading-order prediction with text and layout, ReadingBank benchmark of about 500K pages with weakly supervised labels, average page-level BLEU and ARD metrics; EMNLP 2021 (fetched). GitHub. https://raw.githubusercontent.com/microsoft/unilm/master/layoutreader/README.md
7. Microsoft, "LayoutLMv3" README (unilm): unified text and image masking, word-patch alignment, results on FUNSD and PubLayNet; content licensed CC BY-NC-SA 4.0 (fetched). GitHub, 2022. https://raw.githubusercontent.com/microsoft/unilm/master/layoutlmv3/README.md
8. Docling project, "docling-ibm-models" README: TableFormer table structure and cell box prediction; lists PubTabNet, FinTabNet and TableBank (fetched). GitHub. https://raw.githubusercontent.com/docling-project/docling-ibm-models/main/README.md
9. veraPDF, "veraPDF-library" README: industry-supported PDF/A and PDF/UA validation (fetched). GitHub. https://raw.githubusercontent.com/veraPDF/veraPDF-library/integration/README.md
10. veraPDF, "veraPDF-validation-profiles" README: rules and profiles for PDF/UA-1, PDF/UA-2 and WTPDF 1.0; WCAG 2.2 profiles experimental (fetched). GitHub. https://raw.githubusercontent.com/veraPDF/veraPDF-validation-profiles/integration/README.md
11. Microsoft / doc-analysis, "TableBank" README: 417,234 tables labelled by weak supervision from Word and LaTeX sources; detection and structure recognition tasks; research use only (fetched). GitHub. https://raw.githubusercontent.com/doc-analysis/TableBank/master/README.md
12. UiPath Documentation portal (Document Understanding, IXP, Action Center). Not reachable in this pass; listed as the place to check product names and current behaviour. https://docs.uipath.com
