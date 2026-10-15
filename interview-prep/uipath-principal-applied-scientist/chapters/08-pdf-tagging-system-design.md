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
> **Research limits (please read).** Two passes (October 2026) checked this chapter. In both, the workflow's shared web-search budget was already used up, so no search could run. The fetch proxy could not resolve glassdoor.com, teamblind.com, uipath.com, docs.uipath.com, arxiv.org, w3.org or pdfa.org. Only GitHub-hosted sources were reachable. The second pass re-fetched all 11 original sources and confirmed each claim the questions rely on. It also fetched 15 more GitHub sources [12]-[26] on benchmarks, OCR, table metrics, label quality and the PDF/UA rule set. These sources confirm product behaviour, datasets and metric definitions, not interview content. So **no question below is tagged [Reported]**, and the chapter does not invent reports. The UiPath documentation portal could not be reached, so it was removed from the references. Check product names and current PDF/UA details (ISO 14289-1 and -2, the Matterhorn Protocol, the PDF 2.0 structure namespace) against primary documents before the interview.

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
    - How do licences shape that choice, for example a model released under a non-commercial licence [7], a dataset limited to research use [11], or Apache-licensed code whose model weights carry a revenue-capped licence [16]?
7. [Likely] The same PDF can need PDF/A (archiving) and PDF/UA (accessibility) at once, and tools such as OCRmyPDF already output PDF/A by default [17]. Where do the two goals conflict in your design, and which do you give up when they cannot both hold?
    - Who in the customer organisation owns that decision, and how does the system record it?
8. [Likely] A signed contract or a certified form arrives for tagging. Adding a structure tree changes the file. How do you handle digital signatures, incremental updates and legal evidentiary value?
    - When do you refuse to tag, and when do you return a tagged copy next to the untouched original?

## B. Ingest, routing and native vs scanned detection

```
PDF ──> sanitise (encryption, corrupt xref, JS, embedded files) ──> page split
     ──> per-page features: text-layer char count, glyph->Unicode mapping quality,
         image area ratio, invisible OCR text layer present?, existing StructTreeRoot?
     ──> router: NATIVE | SCANNED | HYBRID | ALREADY-TAGGED (verify / repair)
```

9. [Likely] How do you decide, per page, whether a PDF is native, scanned or hybrid? What features and labels would you use?
    - How do you detect a text layer that exists but is garbage (broken ToUnicode maps, ligatures, Type 3 fonts)?
    - What do you do with a scanned page that already has an invisible OCR layer from a previous tool?
10. [Likely] Why decide per page instead of per document? What breaks downstream if you route a whole document one way?
11. [Likely] Some incoming PDFs are already tagged, but badly. Do you trust, repair or discard the existing tags? How do you decide?
    - How would you measure the quality of existing tags without a human label?
12. [Likely] What security and privacy controls belong at ingest (encrypted PDFs, embedded JavaScript, attachments, PII), and how do they interact with the ML stages?
    - What changes for an on-prem or air-gapped customer?
13. [Likely] How do you handle very long documents (1,000+ pages) and very large pages (engineering drawings) in the pipeline?
    - Where do you parallelise, and which stages need the whole document?
14. [Likely] [1] UiPath's document APIs already run a separate digitization step (start, then poll for a result) before classification and extraction [1]. Should the structure tree become part of the digitization output? What compatibility duties does that create for every existing extraction project?
    - How do you ship a richer digitization result without changing extraction results that customers have already validated?

## C. OCR and the perception layer

15. [Likely] For scanned pages, what does your OCR stage output, and what extra outputs does the tagging stage need beyond plain text?
    - Why do word-level boxes and confidences matter for reading order and review?
16. [Likely] How would you choose between a classical OCR engine, a document-specific OCR model and an end-to-end vision-language model that outputs structure directly?
    - What are the cost, latency, hallucination and auditability trade-offs?
    - How do you guard against a generative model "reading" text that is not on the page?
17. [Likely] How do you handle mixed scripts, right-to-left text, vertical text, handwriting and low-quality faxes?
    - How do these affect the /Lang attribute and reading order in the output?
18. [Likely] For native PDFs, the text is "free", but the order of drawing operators often does not match the reading order. How do you turn content-stream order into usable tokens?
    - How do you handle text drawn as vector paths, or text hidden under images?
19. [Likely] Where does pre-processing (deskew, dewarp, binarisation, rotation detection) belong, and how do you stop it from hurting native pages?
20. [Likely] [18] OCR-free models such as Donut read the page image straight into structured output, with no OCR engine [18]. For a *tagging* product, every structure element must point to real marked content on the page. What do you lose when the model gives you text without word boxes tied to glyphs, and how would you recover the link?
21. [Likely] [14] A VLM-based OCR project lists, among its fixes, hallucinations on blank documents and auto-rotation detection [14]. Design an adversarial test set for a generative OCR stage: blank pages, rotated pages, watermark-only pages, near-duplicate pages and pages in unseen scripts. Which failures are silent, and how would you catch them in production without labels?
22. [Likely] [23] Some open models target scientific documents and output LaTeX for formulas [23]. How would you decide which regions to send to a specialised formula model, and how would the result appear in the tagged PDF (Formula element, alternate text, or MathML in PDF 2.0)?
    - How do you evaluate formula output when two different LaTeX strings render the same?

## D. Layout analysis and reading order

```
page image + tokens ──> layout detector (regions + class + score)
                    ──> region-level reading order (columns, sidebars, floats)
                    ──> token-level order inside regions
                    ──> cross-page links (continued paragraphs, tables, lists)
```

23. [Likely] [3] Define the layout classes you would detect. Public datasets such as DocLayNet use 11 classes over about 80,000 pages from six document categories [3]. Is that the right taxonomy for an enterprise tagging product?
    - What would you add or merge (form fields, signatures, stamps, key-value regions)?
    - How do you keep the layout classes separate from, but mappable to, the PDF structure tags?
24. [Likely] [3] DocLayNet annotated a fraction of pages two or three times to estimate annotation uncertainty [3]. How would you use multiply-annotated pages when you set targets for your own layout model?
    - If humans agree on only 80% of boxes for a class, what accuracy target is meaningful for the model?
25. [Likely] [7] Compare a vision-only detector, a multimodal model that combines text, layout and image (for example, LayoutLMv3 [7]) and a vision-language model for layout. Which would you use for which stage, and why?
    - How does the choice change if the customer runs on CPU-only on-prem hardware?
26. [Likely] [6] How would you model reading order? Compare heuristic sorting (XY-cut, column detection), a learned seq2seq model such as LayoutReader [6] and graph or pairwise-ranking approaches.
    - LayoutReader reports large gains over a heuristic baseline on ReadingBank [6]. Why might those gains shrink on your customers' documents?
    - ReadingBank's labels come from weak supervision on Word documents [6]. What biases does that bring?
27. [Likely] How do you define the "correct" reading order when two reasonable people would disagree (sidebars, pull quotes, footnotes, multi-column with figures)?
    - How does that ambiguity affect your metric?
28. [Likely] How do you handle objects that continue across pages: paragraphs split by a page break, tables with repeated headers, numbered lists?
    - Where in the pipeline do you merge them, and how do you represent the merge in the tagged PDF?
29. [Likely] Headers, footers, page numbers and watermarks should usually become artifacts, not content. How do you detect them robustly across a whole document?
    - What happens to a screen-reader user if you get this wrong in either direction?
30. [Likely] [19] [20] PubLayNet built its labels by matching PDFs to their XML sources [20]. DocBank built token-level labels for 12 kinds of semantic unit from LaTeX sources by weak supervision [19]. Does a tagger need token-level labels, region-level labels or both? What does each label type let you train, and what does it make hard?
31. [Likely] [13] OmniDocBench labels 28 block-level and 4 span-level element types over 1,651 pages from 10 document types [13]. How would you map an outside benchmark's taxonomy onto your own, and how do you report results for classes that do not map one to one?

## E. Logical structure tagging: headings, lists, tables, figures, key-value pairs, spans

32. [Likely] How do you infer heading levels (H1-H6) from a page that only shows font sizes and positions? What signals do you use beyond font size?
    - How do you keep heading levels consistent across a 300-page document?
    - Is document-level inference (a global pass) worth the cost over page-level inference?
33. [Likely] How do you detect lists and their nesting, including lists whose bullets are images or whose numbers are drawn separately from the text?
    - How do you represent label (Lbl) and body (LBody) separately?
34. [Likely] [4] [8] Design the table pipeline: detection, structure recognition (rows, columns, spanning cells), header detection and cell content mapping. Which parts would you learn and which would you compute?
    - Public models such as TableFormer predict structure and cell boxes [8], trained on sets such as PubTabNet, FinTabNet and TableBank [4] [8] [11]. Why may these not transfer to scanned invoices or bank statements?
    - How do you decide which cells are headers, and how do you emit header-to-cell associations (scope or IDs) for accessibility?
    - How do you handle borderless tables, nested tables and tables split across pages?
35. [Likely] [12] The Table Transformer project releases pre-trained structure weights only for PubTables-1M and FinTabNet, and asks users to train their own multi-domain model [12]. Plan a multi-domain table model for invoices, bank statements, insurance forms and scientific reports. When is a domain-specific fine-tune worth the cost of serving one more model?
    - How do you decide that a table failure is a data gap and not a model-capacity limit?
36. [Likely] When is something a table for accessibility purposes but not for data extraction, or the other way round (for example, a layout table or a form laid out as a grid)?
37. [Likely] How do you link figures to their captions, and how do you decide whether a figure is meaningful or decorative?
38. [Likely] Design alt-text generation for figures, charts and diagrams. How do you evaluate it, and when does a human have to write or approve it?
    - What are the risks of a vision-language model writing confident but wrong alt text for a chart in a financial filing?
    - How long should alt text be, and how do you use surrounding context?
39. [Likely] How do you detect key-value pairs on forms and semi-structured documents, and how do they map both to structure tags and to DU or IXP fields?
    - Should a key-value pair be one structure element or two? Who decides?
40. [Likely] Which inline spans matter (language changes, emphasis, links, abbreviations with expansions, formulas), and how do you detect and tag them?
    - How would you produce accessible output for mathematical formulas?
41. [Likely] How do you tag footnotes, endnotes, citations and cross-references (Note, Reference, Link) so that a screen-reader user can follow them and come back? How do you generate a table of contents (TOC/TOCI) and bookmarks that match the heading tree?
42. [Likely] Some incoming PDFs already have interactive form fields. How do you tag the existing widgets (Form elements, tooltips, tab order) so they agree with the visual labels your model detects? What do you do when the field names and the printed labels disagree?
43. [Likely] [16] Some pipelines add an optional LLM pass that merges tables across pages and pulls values out of forms [16]. Where in your cascade would an LLM "repair" pass sit? How do you stop it from changing or inventing text that is on the page?
    - What hard constraint would you put on its output (for example, every output token must map to an input token)?
44. [Likely] Would you use one large model that outputs the whole tree, or a cascade of specialised models? Defend the choice on accuracy, debuggability, cost and how fast you can iterate.
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

45. [Likely] Design the tag taxonomy and its schema. What is the canonical internal representation, and why not use PDF structure types directly?
    - How do you represent confidence, provenance (model, version, human) and bounding boxes on each node?
    - Docling, for example, uses one internal document model and exports it to Markdown, HTML, DocTags and lossless JSON [2]. What do you gain and lose with a single canonical tree?
46. [Likely] How do you version the taxonomy, and what happens to old labels, old models and customer outputs when you split or merge a class?
    - How do you migrate a label store from taxonomy v3 to v4 without relabelling everything?
47. [Likely] Customers want their own custom tags, such as "ClauseIndemnity" or "InvoiceLineItem". How do you support them without breaking accessibility output?
    - How does role mapping to standard types keep a custom tag valid for assistive technology?
48. [Likely] How strict should the schema be? Which rules are hard constraints (a TD must sit inside a TR) and which are soft (a document should have one H1)?
    - Where do you enforce each: in decoding, in post-processing, or in validation?
49. [Likely] How would you make the same tree serve accessibility, extraction and RAG chunking for agents? Where do these uses conflict?
50. [Likely] [24] Docling defines its canonical document as a Pydantic model [24]. Your training code is in Python, but the PDF writer and the review UI may run in other stacks. How do you keep the schema in one place and evolve it without breaking any service?
    - What is your compatibility rule: can an old reader read a new tree, and can a new reader read an old tree?
51. [Likely] [26] The PDF 2.0 and PDF/UA-2 specifications still receive errata and amendments through a public tracker [26]. How do you version the "standards profile" (role map, writer rules, validator profile) separately from the tag taxonomy and the models?
    - When a standard changes, which customer outputs must you re-validate or regenerate?

## G. Tagged-PDF and PDF/UA output

```
canonical tree ──> structure tree writer: StructTreeRoot, StructElems, parent tree,
                   marked-content IDs (MCIDs) linking tree <-> page content,
                   RoleMap for custom types, /Alt, /ActualText, /Lang, Artifacts
               ──> PDF/UA validator (machine-checkable rules) ──> report
               ──> residual checks needing human judgement (alt-text quality, order)
```

52. [Likely] Walk through how you write the tagged PDF: how the structure tree links to the page content, and how you mark headers and footers as artifacts.
    - What goes wrong if marked content is split, or if one content stream draws text for several elements?
    - Do you rewrite content streams or only add structure? What are the risks of each?
53. [Likely] [9] [10] Open-source validators such as veraPDF check PDF/UA-1, PDF/UA-2 and WTPDF rules, with WCAG 2.2 profiles in an experimental stage [9] [10]. What share of accessibility can a machine validator check, and what remains for human judgement?
    - How do you report "passes machine checks" versus "human-verified accessible" honestly to customers?
54. [Likely] How do you set /Lang, /Alt, /ActualText and expansion text, and which of them can you generate automatically?
55. [Likely] How do you handle PDF 2.0 and PDF/UA-2 versus PDF 1.7 and PDF/UA-1 outputs? Does the internal tree change?
56. [Likely] How do you test the output with real assistive technology (screen readers) and with accessibility experts, not only validators? How would you scale that testing?
    - How do you turn their findings into a repeatable metric that you can track release to release?
57. [Likely] Writing a tagged PDF must not change how the page looks. How do you guarantee visual fidelity and detect regressions in the writer?
    - What is your regression test corpus for the writer, and how is it different from the ML eval set?
58. [Likely] [22] veraPDF's PDF/UA-1 profile encodes about 106 machine rules, such as "the MarkInfo dictionary has Marked set to true" and "content marked as an artifact should not sit inside tagged content" [22]. How would you use the validator as a release gate for the writer and as a runtime check on every output?
    - Most legacy customer files fail some rules before you touch them. How do you report "our output fixed X of Y failures" honestly?
    - Which rule failures should block delivery, and which should only warn?
59. [Likely] A customer later edits a tagged PDF in a desktop editor (adds a page, redacts a paragraph). The structure tree goes stale. Do you support incremental re-tagging of only the changed pages? What must the canonical tree store to make that possible?

## H. Confidence, calibration and human review

```
per-element scores ──> calibrated confidence ──> document-level risk score
   ├─ all elements >= threshold and validator passes ──> auto-publish
   ├─ some elements low ──> targeted review (only the risky regions)
   └─ document very low / unsupported type ──> full manual remediation queue
corrections ──> label store (with reviewer id, time spent, taxonomy version)
```

60. [Likely] [1] UiPath's document APIs include validation actions that block until a person completes them in Action Center [1]. Design the human-review step for structure tags: what does the reviewer see, and what do they fix?
    - How do you show a tree, reading order and table structure so that review is fast?
    - What is the unit of review: a page, an element, or a document?
61. [Likely] How do you produce calibrated confidence for a structured output like a tree or a table, not just for a single class?
    - How do you combine element confidences into a document-level decision?
    - How do you check calibration after a model update?
62. [Likely] How do you set the auto-publish threshold? Who owns the trade-off between review cost and risk, and how does it vary by customer or document type?
    - What happens when the threshold sends 60% of a large customer's documents to review on day one?
63. [Likely] How do you measure and reduce reviewer effort (minutes per page, edits per page), not just model error?
    - How would you run an A/B test on the review UI itself?
64. [Likely] How do you prevent reviewers from rubber-stamping model output (automation bias), and how would you detect it in the data?
    - Would you inject known errors to audit reviewers? What are the ethical and practical concerns?
65. [Likely] Review corrections are a training signal but they are biased: reviewers only see low-confidence items. How do you use them without biasing the next model?
66. [Likely] A customer asks for a statistical guarantee, such as "at most 1% of auto-published documents have a wrong table header". How would you set a threshold that gives that guarantee (for example, with conformal or risk-control methods)? Which assumptions break when the document mix shifts?
67. [Likely] Reviewers often fix only part of a reading order or part of a table and leave the rest. How do you record a partial correction in the label store so that training does not treat the unfixed parts as confirmed?

## I. Annotation and labelling tooling

68. [Likely] Design the annotation tool for structure tags. How is it different from a bounding-box tool?
    - How do annotators label reading order, nesting and table spans efficiently?
    - How do you pre-fill with model output without anchoring annotators?
69. [Likely] Write the outline of the annotation guideline for headings, lists and tables. Which edge cases would you settle up front?
    - How do you measure inter-annotator agreement for trees and orders, not only boxes?
    - What process resolves disagreements (adjudication), and how do guideline changes flow back into old labels?
70. [Likely] [3] [6] [11] Public datasets use different labelling strategies: expert manual labelling [3] and weak supervision from Word or LaTeX source files [6] [11]. When would you use each for your product, and how would you combine them?
    - Could you build synthetic tagged PDFs from Word or HTML sources? What gap would remain to real scanned documents?
71. [Likely] Design an active-learning loop. Which documents should humans label next, and how do you avoid only labelling the hardest outliers?
    - How do you balance labelling budget across document types, languages and tag classes?
72. [Likely] Customers' documents are confidential. How do you get labelled data from them (consent, on-tenant labelling, federated or per-tenant fine-tuning), and how do you keep tenants separate?
73. [Likely] Would you use LLMs or VLMs as annotators or pre-labellers? How do you check the quality of machine labels before they enter training?
74. [Likely] [21] Tools such as cleanlab estimate annotator quality and consensus labels for multi-annotator data with confident learning [21]. Those methods assume class labels. How would you extend consensus and annotator-quality estimation to trees, reading orders and table structures?
75. [Likely] [18] Donut pre-trains partly on SynthDoG, a synthetic document generator, across several languages [18]. Design a synthetic generator for tagged PDFs with a known ground-truth tree. How would you measure and close the gap between synthetic and real scanned documents?
    - Which tags are easy to synthesise, and which (alt text, reading order of complex layouts) are not?
76. [Likely] Labelling a full structure tree costs far more per page than drawing boxes. Build a labelling budget for the first year: which label types, how many pages, which mix of experts and vendors, and what you buy with each extra dollar.

## J. Versioned label store and data management

```
label record = (doc_id, page, element_id, geometry, tag, parent, order_index,
                attributes, taxonomy_version, source{human|model|weak|synthetic},
                annotator_id, adjudicated?, created_at, supersedes)
snapshots: train/val/test frozen by hash  ──>  model registry links model <-> snapshot
```

77. [Likely] Design the versioned label store. What is the record schema, and how do you represent a tree, an order and a table so they can be diffed and queried?
    - How do you store several competing labels for the same element (multi-annotation, model vs human)?
78. [Likely] How do you make training and evaluation reproducible: frozen snapshots, dataset hashes, and links from each model version to the exact data it saw?
79. [Likely] How do you prevent leakage between train and test when the same template or vendor appears many times?
    - Would you split by document, by template, by customer or by time? Why?
80. [Likely] A customer asks you to delete all their data, including labels and anything derived from it. What does your label store and model lineage need to support?
81. [Likely] How do you track label quality over time and find bad labels in the store (for example, with model disagreement or confident-learning methods)?
82. [Likely] The same document comes back re-scanned, re-exported or with one page changed, so its hash is new. How do you keep document and element identity across versions so that labels, review history and customer outputs stay linked?

## K. Evaluation

```
offline:  layout mAP | tag F1 (per class, per level) | tree-edit distance (whole tree)
          reading-order accuracy (pairwise / BLEU / edit distance) | table TEDS (+structure-only)
          alt-text quality (human rubric) | PDF/UA machine-check pass rate
online:   auto-publish rate | review minutes per doc | post-publish defect rate
          customer-reported accessibility issues | cost per page | p95 latency
```

83. [Likely] Define tag F1 for a tree. How do you match predicted elements to ground-truth elements before you count true positives?
    - Do you require geometric overlap, text overlap or both? What IoU threshold, and why?
    - Do you report micro or macro F1? How do you weight rare but critical tags (table headers) against frequent ones (paragraphs)?
84. [Likely] Why is element-level F1 not enough for a tree? Define a tree-edit-distance metric for the whole structure tree.
    - What are the insert, delete and rename costs, and should a wrong heading level cost as much as a missing table?
    - How do you normalise it so that scores compare across documents of different sizes?
85. [Likely] [4] [5] TEDS scores tables as 1 minus the tree edit distance divided by the size of the larger tree. Renaming a cell costs the normalised Levenshtein distance of its content, and a structure-only mode ignores the content [4] [5]. What are the weaknesses of TEDS for your use case?
    - The reference code does not clamp the score, so in theory it can fall below 0 [5]. Does that matter in reporting?
    - When would you report structure-only TEDS, full TEDS and cell-level adjacency F1 side by side?
    - How would you score header-cell identification, which accessibility depends on?
86. [Likely] [6] How do you measure reading-order accuracy? Compare page-level BLEU, average relative distance [6], Kendall's tau and pairwise accuracy within regions.
    - Should you measure reading order on tokens, on lines or on regions? What does each hide?
    - How do you score order when the ground truth allows several valid orders?
87. [Likely] How do you build an evaluation set that reflects your customers and not public benchmarks? How large must it be per class to detect a 1-point regression?
    - How do you stratify (native vs scanned, language, document type, page complexity)?
88. [Likely] Offline metrics improved but customers report more accessibility complaints. How do you debug the gap?
89. [Likely] How do you evaluate alt text and other generated text without relying only on an LLM judge? If you do use an LLM judge, how do you validate it?
90. [Likely] [15] olmOCR-Bench checks simple, machine-checkable "facts" about each page, like unit tests, and avoids soft metrics such as edit distance because a correct parse can differ from the reference [15]. Would you build a fact-based test suite for structure tagging? What facts would you write, and what would it miss that tree-edit distance or TEDS catches?
91. [Likely] [4] [12] Compare TEDS [4] with GriTS, the grid table similarity metric from the Table Transformer work [12]. For accessibility, where header-to-cell relations matter most, which would you trust, and why?
92. [Likely] [13] OmniDocBench reports one "Overall" score as the average of (1 - text edit distance) x 100, table TEDS and formula CDM [13]. What can go wrong when you rank systems on a composite like that? Would you publish a single number to leadership?
    - The same benchmark changed its matching algorithm between versions [13]. How can the matching step alone change a leaderboard, and how do you stop teams from tuning to it?
93. [Likely] How do you compute confidence intervals and significance for document metrics when pages from one document, or documents from one template, are not independent?
    - Would you bootstrap by page, by document or by template? What does each choice do to the width of the interval?
94. [Likely] Define the release gate for a new model version. Which metrics block a release, and which are only monitored?
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

95. [Likely] Design serving for two workloads: an interactive API (one document, a few seconds) and a backlog of millions of pages. What is shared and what is separate?
    - Which stages run per page in parallel and which need the whole document?
    - How do you schedule GPU work across tenants fairly?
96. [Likely] Estimate the cost per 1,000 pages for your design. Which stage dominates, and what would you change to cut cost by 5x?
    - When is a smaller distilled model plus targeted review cheaper than a large VLM on every page?
    - One open VLM pipeline claims under $200 per million pages converted [14]. What does that figure leave out (review minutes, PDF writing, validation, retries, on-prem hardware)?
97. [Likely] How do you cache and deduplicate work (identical pages, repeated templates, re-runs after a model update)?
98. [Likely] How do you roll out a new model version safely: shadow mode, canary by tenant, rollback? How do customers pin a version [1]?
    - What does a customer expect when the same document gives a different tree after an upgrade?
99. [Likely] What do you monitor in production to detect drift (new templates, new scanners, new languages) before customers notice?
    - Which signals work without labels: confidence distribution, validator failure rate, review edit rate?
100. [Likely] Page cost for a VLM varies widely: a blank page and a dense 8-point financial table differ by an order of magnitude in tokens. How do you batch and schedule mixed pages on GPUs to meet a p95 latency target without wasting capacity?
101. [Likely] Large customers want models fine-tuned on their own documents. How would you serve hundreds of per-tenant variants (for example, adapters on a shared base model)? What breaks first: memory, cold starts, evaluation or release management?
102. [Likely] How does the design change for an on-prem, air-gapped deployment with no GPU?
103. [Likely] How do you make the pipeline fail safe? If a stage crashes or times out on a 500-page document, what does the customer get?
    - How do retries and idempotency work when outputs are written to the customer's storage?

## M. Integration with UiPath products and agents

104. [Likely] [1] How would robots and agents consume the tagging output, for example, to drive classification, extraction and validation steps [1]?
    - What do agents need from the tree that a screen reader does not (stable element IDs, provenance, confidence)?
105. [Likely] How would you use the structure tree to improve retrieval (chunking by section, keeping tables whole) for agent grounding over enterprise documents?
    - How would you measure whether structure-aware chunking actually improves answer quality?
106. [Likely] An agent asks a question about a table on page 47, and the answer is wrong because the table was split across pages. Whose bug is this, and how would your metrics have caught it?
107. [Likely] [25] UiPath's Python SDK exposes a Context Grounding service that agents use to search indexed content [25]. Define the contract between the tagging system and that index: which fields from the tree must the index keep (element IDs, section path, table cells, page boxes) so that an agent can cite the exact source?
108. [Likely] How would you govern this system for regulated customers: audit logs, explainability of each tag, data residency, and model risk management?

## N. Research depth and open problems

109. [Likely] What are the hardest open research problems in document structure understanding today, and which would you fund first?
    - Where do you expect vision-language models to replace the cascade within two years, and where not?
110. [Likely] How would you build a single model that predicts layout, reading order and hierarchy jointly? What would the training targets and the loss look like?
    - How do you represent a tree as a target sequence, and how do you keep the decoded output valid?
111. [Likely] How would you adapt the system to a new document domain with 50 labelled documents?
    - Which parts transfer, and where do you spend the labels?
112. [Likely] How would you use self-supervised or weakly supervised signals from native PDFs that already have good tags, or from Word and HTML sources, to train models for scanned documents?
113. [Likely] How would you estimate and report uncertainty over a whole tree rather than over single nodes?
114. [Likely] [14] One open document-OCR model reports gains from synthetic data and reinforcement learning [14]. Design a reinforcement-learning setup for structure tagging with verifiable rewards, for example PDF/UA rule passes, fact-based tests and table-structure checks. How would you stop the model from gaming the reward?
115. [Likely] A per-page model cannot see that "Section 4.2" on page 90 must sit under "Chapter 4" on page 61. How would you give a model document-level context for hierarchy and cross-page objects at an acceptable cost?
116. [Likely] What would make you stop using public benchmarks such as DocLayNet, PubTabNet or ReadingBank [3] [4] [6] for decisions? What would you publish instead?

## O. Principal-level scoping, roadmap and trade-offs

117. [Likely] Present a 4-quarter roadmap for this system. What ships in quarter 1 to deliver customer value while you build the learning loop?
    - What do you deliberately not build?
118. [Likely] Which teams own which parts (OCR, layout, tables, review UI, PDF writer, label operations, platform)? Where do you expect friction, and how do you resolve it?
119. [Likely] Product wants "100% PDF/UA compliant" in marketing. How do you respond, and what claim would you sign off on?
120. [Likely] A large customer escalates: 2% of tagged PDFs read tables in the wrong order to screen-reader users. Walk through your first week.
    - How do you separate a model problem from a writer bug or a taxonomy gap?
    - What process change stops this class of issue from recurring?
121. [Likely] How would you decide between investing in a better model and investing in a better review experience, given a fixed budget?
122. [Likely] Walk through a document-AI or structured-prediction system you shipped. Which design decision in this chapter would you make differently because of it?
    - What metric did you get wrong at first, and how did you find out?
    - What would you tell a new team on day one so they avoid the same mistake?

## References

1. UiPath, Documents service source, `uipath-platform` package of `uipath-python`: classify, extract (DU Modern and IXP), published project version and tag parameters, a separate digitization step (start, then poll for a result), and validation actions that block until a person completes them in Action Center; marked experimental (official; fetched in this pass). GitHub, 2025-2026. https://raw.githubusercontent.com/UiPath/uipath-python/main/packages/uipath-platform/src/uipath/platform/documents/_documents_service.py
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
12. Microsoft, "Table Transformer (TATR)" README: PubTables-1M (CVPR 2022), the GriTS grid table similarity metric, pre-trained weights for PubTables-1M and FinTabNet only, with guidance to train your own multi-domain model (fetched). GitHub, 2022-2023. https://raw.githubusercontent.com/microsoft/table-transformer/main/README.md
13. OpenDataLab, "OmniDocBench" README: 1,651 PDF pages, 10 document types, 28 block-level and 4 span-level element types; normalised edit distance, TEDS and CDM; Overall = ((1 - text edit distance) x 100 + table TEDS + formula CDM) / 3; matching algorithm updates across versions (fetched). GitHub, 2024-2025. https://raw.githubusercontent.com/opendatalab/OmniDocBench/main/README.md
14. Allen Institute for AI, "olmOCR" README: VLM-based PDF conversion, under $200 per million pages claimed; release notes on fixing hallucinations on blank documents and auto-rotation, and on synthetic data plus RL training (fetched). GitHub, 2025. https://raw.githubusercontent.com/allenai/olmocr/main/README.md
15. Allen Institute for AI, "olmOCR-Bench" README: machine-checkable per-page "facts" like unit tests, in place of soft metrics such as edit distance (fetched). GitHub, 2025. https://raw.githubusercontent.com/allenai/olmocr/main/olmocr/bench/README.md
16. Datalab, "marker" README: Apache 2.0 code with model weights under a modified OpenRAIL-M licence with revenue limits; optional LLM mode that merges tables across pages and extracts form values (fetched). GitHub, 2025-2026. https://raw.githubusercontent.com/datalab-to/marker/master/README.md
17. OCRmyPDF README: adds an OCR text layer to scanned PDFs, "lossless" where possible, and outputs PDF/A by default (fetched). GitHub. https://raw.githubusercontent.com/ocrmypdf/OCRmyPDF/main/README.md
18. NAVER Clova, "Donut" README: OCR-free document understanding transformer and SynthDoG synthetic document generator (fetched). GitHub, 2022. https://raw.githubusercontent.com/clovaai/donut/master/README.md
19. Microsoft / doc-analysis, "DocBank" README: 500K pages with token-level labels for 12 kinds of semantic unit, built by weak supervision (fetched). GitHub, 2020. https://raw.githubusercontent.com/doc-analysis/DocBank/master/README.md
20. IBM, "PubLayNet" README: layout annotations generated automatically by matching PDFs to their XML from the PubMed Central commercial-use subset (fetched). GitHub, 2019. https://raw.githubusercontent.com/ibm-aur-nlp/PubLayNet/master/README.md
21. Cleanlab, "cleanlab" README: confident-learning label-error detection, and consensus and annotator-quality estimation for multi-annotator data (fetched). GitHub. https://raw.githubusercontent.com/cleanlab/cleanlab/master/README.md
22. veraPDF, PDF/UA-1 validation profile `PDFUA-1.xml`: 106 machine-checkable rules, including MarkInfo Marked = true and no artifact content inside tagged content (fetched). GitHub. https://raw.githubusercontent.com/veraPDF/veraPDF-validation-profiles/integration/PDF_UA/PDFUA-1.xml
23. Meta AI, "Nougat" README: academic-document PDF parser that outputs LaTeX math and tables (fetched). GitHub, 2023. https://raw.githubusercontent.com/facebookresearch/nougat/main/README.md
24. Docling project, "docling-core" README: the DoclingDocument data model defined as a Pydantic model (fetched). GitHub. https://raw.githubusercontent.com/docling-project/docling-core/main/README.md
25. UiPath, "UiPath Python SDK" README: SDK for UiPath Cloud Platform services, including the Context Grounding service used to search indexed content (official; fetched). GitHub, 2025-2026. https://raw.githubusercontent.com/UiPath/uipath-python/main/README.md
26. PDF Association, "PDF Specification Issues" tracker README: public errata and amendments for ISO 32000-2 (PDF 2.0) and PDF/UA-2 (fetched). GitHub. https://raw.githubusercontent.com/pdf-association/pdf-issues/main/README.md
