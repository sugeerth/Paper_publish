# 13. ML System Design: Product Tagging, Catalog Attribute Extraction and PDF/Document Understanding

> Scope: design-round questions on two systems the rest of this pack does not cover in depth. (1) Product tagging and catalog attribute extraction. (2) PDF and document understanding pipelines (OCR and Textract-style layout parsing, key-value and table extraction, classification and routing, human review). The focus is the "Science Application / ML System Design (team-specific)" round of the Amazon L5 Applied Scientist (Applied Scientist II) loop, 2025-2026.
>
> Research note (please read): no new web research was possible for the first draft of this chapter. The run's shared web-search budget was already used up before it started. Every [Reported] tag in that draft came from sourced findings already gathered and cited in chapters 01-11 of this guide. A question is tagged [Reported] only where a source attributes it, or a close paraphrase, to an Amazon science or ML interview. No candidate report found so far names a "PDF pipeline" or "Textract" design prompt verbatim, so every document-understanding question is [Likely]. Those questions are grounded in Amazon's products (Textract, Augmented AI/A2I, Comprehend, Bedrock, the product catalog) and in the round's reported format.
>
> Review pass (2026-10-14): web search was again unavailable because the shared budget was used up, so no new Blind, Glassdoor or 1point3acres reports could be found. Fourteen cited or candidate URLs were fetched successfully, all on github.com: refs [6], [7], [9]-[12] and the new refs [15]-[22]. Each one supported the claims made for it. Refs [1]-[5], [8], [13] and [14] could not be re-fetched: amazon.jobs, igotanoffer.com, leetcode.com, medium.com, teamblind.com, amazon.science and enginebogie.com failed DNS resolution in this environment. A refused connection is not evidence that a page is dead, so those references are kept, with this caveat. Question 1 cited only a prep-vendor description of the round, so it was retagged [Likely]. The 26 new questions come from a dated Amazon question aggregator ([15], 2025-2026 entries), the Amazon MLE aggregate [7], and public AWS and Amazon Science repositories [16]-[22]. Where a reported prompt's role or round is not stated, the item says so. A later session with a search budget should look for first-person reports using the terms "attribute extraction", "catalog quality", "Textract" and "IDP".
>
> Additions beyond the questions-only format: the user asked for "ML system design diagrams" and "details about the tag/PDF system". For that reason this chapter includes two whiteboard reference diagrams and a checklist of the components, metrics and failure modes interviewers probe. These are scaffolds for drawing your own design. They are not model answers.

## Format notes for this round

- **Where it sits.** L5 AS loops have about 4-6 interviews of roughly 55-60 minutes. One 2025-2026 Applied Scientist II loop was reported as "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [2]. Amazon's official prep page describes about four 55-minute interviews with members of the science community. Each interviewer is assigned Leadership Principle competencies and asks 2-3 behavioral questions [1].
- **Interviewer.** Usually the hiring manager or a senior scientist on the hiring team. The prompt comes from the team's domain [2]. A tagging or document prompt is most likely on catalog, Selling Partner, Product Graph, AWS AI services (Textract, Comprehend, Bedrock), Finance or Payments, and Compliance teams. That part is an inference.
- **What is assessed.** IGotAnOffer describes the round as being given a high-level ML problem and proposing data sources, annotation strategy, modeling technique, evaluation metrics, and trade-offs and risks [3]. Candidates report being drilled on the data pipeline "from collection to deployment" [8], and on building with very little labeled data [14].
- **L5 bar.** Blind commenters say L5 loops usually include a design or application round and L4 loops may not [4]. At L5 you are expected to own the problem end to end: frame a vague problem, design the labels and taxonomy, choose models, set per-attribute or per-field metrics and thresholds, plan human review, launch, and monitor. You should also make the cost-accuracy trade-off calls yourself.
- **Closest reported prompts.** "Categorize 10k product listings across 100 categories, each with a title, description and photo" [5]. "Design an evaluation framework that checks whether a text-to-image system spells a brand name correctly, combining OCR accuracy, embedding checks and a human review protocol" [6][7]. "Design an LLM quality validation system" [6][15]. A dated Amazon aggregator also lists "Design Faceted Product Search at Large Scale" and "Design a Product Image Upload Pipeline for Sellers" (both August 2026, role not stated) and "Design an end-to-end spam detection system" (undated) [15]. These are the nearest catalog-side prompts on record.

Tag legend: **[Reported]** means a cited source attributes the question, or a close paraphrase, to an Amazon science or ML interview. **[Likely]** means a grounded inference from the role, team and Amazon products, not seen verbatim.

---

## Reference diagram 1: Product tagging and attribute extraction (whiteboard scaffold)

```
                         +-----------------------------+
 Seller listing  ------> |  Ingestion & normalization  |  title, bullets, description,
 (text, images,          |  (dedupe, lang-ID, unit     |  images, browse node, brand,
  feeds, PDFs)           |   normalization)            |  seller-provided attributes
                         +--------------+--------------+
                                        |
                  +---------------------+----------------------+
                  v                                            v
     +-------------------------+                 +-----------------------------+
     | Product-type / taxonomy |                 |  Attribute schema lookup    |
     | classifier (hierarchical|---- PT -------->|  (which attributes apply to |
     | multi-class, text+image)|                 |   this product type; enums) |
     +-------------------------+                 +--------------+--------------+
                                                                |
          +-----------------------------+-----------------------+-------------+
          v                             v                                     v
 +------------------+       +-------------------------+          +------------------------+
 | Closed-vocab     |       | Open-value extraction   |          | Image attribute models |
 | multi-label tags |       | (NER/sequence tagging,  |          | (color, pattern, style,|
 | (enum attrs)     |       |  LLM/VLM generative)    |          |  OCR on packaging)     |
 +--------+---------+       +------------+------------+          +-----------+------------+
          |                              |                                   |
          +---------------+--------------+-----------------------------------+
                          v
           +-------------------------------+
           | Normalization & verification  |  value canonicalization (units, synonyms),
           | - grounding check (is value   |  cross-attribute consistency rules,
           |   supported by source text?)  |  conflict resolution across sources,
           | - LLM-judge / rules / ensemble|  per-attribute calibrated confidence
           +---------------+---------------+
                           |
          conf >= T_auto   |  T_review <= conf < T_auto     conf < T_review
          +----------------+------------------+-------------------------+
          v                                   v                         v
   +-------------+                  +--------------------+       +-------------+
   | Auto-publish|                  | Human review queue |       | Drop / ask  |
   | to catalog  |                  | (prioritized by    |       | seller      |
   +------+------+                  |  traffic x risk)   |       +-------------+
          |                         +---------+----------+
          |                                   |  labels
          v                                   v
   +----------------------------------------------------------+
   | Feedback & monitoring: customer returns ("not as        |
   | described"), search/browse engagement, seller disputes, |
   | audit samples -> per-attribute P/R, coverage, drift     |
   | -> active learning -> retraining / distillation          |
   +----------------------------------------------------------+
```

Details interviewers usually expect you to cover, as a checklist rather than an answer:
- **Problem shape.** Product type is a hierarchical, single-path classification over a large, changing taxonomy. Tags are multi-label. Attribute values are either closed-vocabulary (enum) or open (free text, numeric with units).
- **Metrics.** Per-attribute precision, recall and F1. Coverage (fill rate) per product type. Precision at the auto-publish threshold. Traffic-weighted accuracy. Human review rate and cost per item. Downstream metrics: search relevance and conversion, "item not as described" returns.
- **Typical failure modes.** Hallucinated values. Values taken from the wrong product in a bundle or "compatible with" text. Unit and format errors. Taxonomy drift. Long-tail product types with no labels. Seller-provided values that are wrong or adversarial. Miscalibrated confidence on new categories.

## Reference diagram 2: PDF and document understanding pipeline (whiteboard scaffold)

```
  Upload (PDF, scan,      +------------------------+
  image, email, fax) ---> | Pre-processing         |  file type, page split, de-skew,
                          | (render, de-skew,      |  denoise, rotation, password/
                          |  quality score)        |  corrupt-file handling
                          +-----------+------------+
                                      v
                          +------------------------+
                          | Text layer present?    |-- yes (digital PDF) --> parse text + coords
                          +-----------+------------+
                                      | no (scanned)
                                      v
                          +------------------------+
                          | OCR + layout analysis  |  words + bounding boxes, reading order,
                          | (Textract-style:       |  blocks: LINE/WORD/TABLE/CELL/
                          |  text, forms, tables,  |  KEY_VALUE_SET/SELECTION_ELEMENT,
                          |  signatures, queries)  |  handwriting, checkboxes
                          +-----------+------------+
                                      v
                          +------------------------+
                          | Document classification|  invoice / receipt / W-2 / ID /
                          | & splitting (page-level|  contract / bank statement / other;
                          |  -> doc-level, packets)|  split multi-doc packets
                          +-----------+------------+
                                      | route by class
             +------------------------+-------------------------+
             v                        v                         v
   +-------------------+   +----------------------+   +--------------------------+
   | Template/rule-    |   | Layout-aware model   |   | VLM / LLM extraction     |
   | based extractor   |   | (LayoutLM-style KV + |   | (schema-guided prompts,  |
   | (stable forms)    |   |  table extraction)   |   |  long/unseen layouts)    |
   +---------+---------+   +----------+-----------+   +------------+-------------+
             +------------------------+----------------------------+
                                      v
                          +------------------------+
                          | Normalization &        |  dates, currency, addresses,
                          | validation             |  cross-field checks (line items sum
                          | (business rules, cross-|  to total), lookups vs master data,
                          |  field checks, ground- |  evidence bbox for every value,
                          |  ing to bbox)          |  per-field calibrated confidence
                          +-----------+------------+
                                      |
            all fields >= threshold   |   any critical field < threshold / rule fail
            +-------------------------+-------------------------+
            v                                                   v
   +--------------------+                         +------------------------------+
   | Straight-through   |                         | Human review (A2I-style loop)|
   | processing -> ERP, |                         | UI shows page + bbox overlay,|
   | search index, RAG  |                         | reviewer fixes fields; SLA,  |
   +---------+----------+                         | sampling for audit           |
             |                                    +---------------+--------------+
             +---------------------+------------------------------+
                                   v
                 +-----------------------------------------+
                 | Monitoring & learning: STP rate, field  |
                 | accuracy, review rate, latency, cost per|
                 | page, new-layout drift -> active        |
                 | learning, threshold re-tuning, retrain  |
                 +-----------------------------------------+
```

Details interviewers usually expect you to cover, as a checklist rather than an answer:
- **Problem shape.** OCR and layout, then classification and routing, then field, key-value and table extraction, then validation, then human-in-the-loop review. The pipeline is cascaded, so errors compound from stage to stage.
- **Metrics.** Character and word error rate (CER/WER) for OCR. Document classification accuracy and confusion. Per-field precision, recall and exact match. Table cell and structure accuracy (for example TEDS-style metrics). Straight-through-processing (STP) rate. Human review rate. Accuracy after review. Latency and cost per page.
- **Typical failure modes.** Low-quality scans, rotated or multi-column pages. Tables that span pages. Handwriting. Lookalike fields (ship-to versus bill-to). Multi-document packets. Unseen layouts. VLM hallucination of missing fields. Overconfident scores. PII exposure in logs and to reviewers. Reviewer fatigue and automation bias.

---

## A. Framing and scoping (both systems)

1. Here is a high-level ML problem from our team (for example, filling missing product attributes, or extracting fields from supplier invoices). Propose a solution covering data sources, annotation strategy, modeling techniques, evaluation metrics, and trade-offs and risks. [Likely] (round format as described by a prep vendor [3]; the same wording is cited as a candidate-reported prompt in chapters 03, 06 and 11)
   - Which parts need ML, and which can be rules, templates or seller-facing forms?
   - What is the v1 you would ship in one quarter?
2. Walk me through the full data pipeline for your design, from collection to deployment. [Reported] [8]
   - Where do labels come from, and how noisy are they?
   - How do you version data, taxonomy and models together?
3. How would you build this with very little labeled data? [Reported] [14]
   - Where would weak supervision, distant supervision from existing catalog values, or LLM pseudo-labels come in?
4. Who is the customer of the tagging system: shoppers, search ranking, sellers, or internal compliance? How does the answer change your metric? [Likely]
   - Which single metric would you report to leadership, and which guardrail metrics would sit next to it?
5. What scale are we designing for (items per day, pages per day, number of attributes and document types), and how do the answer and the cost budget change your architecture? [Likely]
   - Batch or real-time? What is the latency SLA at listing creation compared with backfill?
6. How would you write the 6-page or PR-FAQ document that proposes this system? [Likely]
   - What would the "customer FAQ" say about wrong tags or wrong extracted fields?

7. How would you decide that an LLM-based attribute or document extractor is ready to launch to millions of customers? [Likely] (adapted from a launch-gating prompt in a synthesized Amazon question bank [16])
   - Which per-attribute or per-field gates must pass, and on which slices?
   - What does shadow mode tell you that an offline gold set cannot?
8. Your extraction model performs worse in production than offline. Diagnose and fix it. [Reported for the general prompt "Diagnose and fix underperforming ML model" (Amazon MLE aggregate, project deep dive) [7]]; extraction application [Likely]
   - Check first for train-serve skew: a new OCR engine version, different HTML cleaning, or a different product-type mix.
   - How would you tell label noise in the gold set apart from real model regressions?

## B. Product taxonomy and product-type classification

9. You have 10k product listings across 100 categories, each with a title, description and photo. How would you categorize them? [Reported] [5]
   - How do you fuse text and image signals?
   - How do you handle a long tail of categories with very few examples?
   - How do you evaluate when the category taxonomy itself is noisy?
10. Scale that to a taxonomy of tens of thousands of nodes and hundreds of millions of products. What changes? [Likely]
   - Flat classifier, top-down hierarchical, or retrieval over node embeddings? Why?
   - How do you enforce hierarchy consistency in predictions?
11. The taxonomy changes: nodes are split, merged and added every month. How do you keep models and labels valid? [Likely]
   - How do you remap historical labels without full relabeling?
   - How would you detect that a node should be split?
12. How would you evaluate a zero-shot product-to-taxonomy matcher when you only have a few hundred human labels? [Likely] (CaML, Amazon Science [11])
    - How would you compute confidence intervals per node with so few labels?
13. Is a product-type error near the root (electronics versus apparel) worse than one near a leaf? How would you build that into the loss and the metric? [Likely]
14. How would you use an LLM to categorize millions of products into a large taxonomy, combining LLM labeling, a distilled classifier and human audit? [Likely] (LLM distillation work at Amazon Science [13])
    - When is the LLM in the serving path, and when only in the labeling path?
    - What is the cost per million items of each option?

15. Map every catalog product to an external standard taxonomy (for example NAICS industry codes, used to look up emission factors for carbon footprinting) with no training pairs. [Likely] (CaML, Amazon Science [11])
   - Zero-shot semantic similarity with top-k candidates and human verification: where does each fit?
   - How does a mapping error propagate into the downstream estimate, and which errors matter most?
16. For a search query, classify each retrieved product as Exact, Substitute, Complement or Irrelevant. How would better catalog attributes change this classifier, and how would you measure the effect? [Likely] (Shopping Queries / ESCI dataset, Amazon Science [18])
   - Which attribute mismatches (size, compatibility, brand) turn an Exact into a Substitute?

## C. Multi-label tagging and attribute schema

17. Design a multi-label tagging system for product listings (for example "waterproof", "vegan", "gift idea", "for kids"). [Likely]
    - Sigmoid per tag or a label-set model? How do you set per-tag thresholds?
    - How do you handle label co-occurrence and mutually exclusive tags?
18. How do you decide which attributes apply to which product type, and who owns that schema? [Likely]
19. Closed-vocabulary attributes (color family, material) compared with open values (model number, wattage): how does your modeling differ? [Likely]
    - When would you turn an open attribute into an enum?
20. How would you handle extreme label imbalance across thousands of tags? [Likely]
    - Is macro or micro averaging the right summary, and for whom?
21. Missing tags are not negatives. How does partial labeling (positive-unlabeled data) affect training and evaluation? [Likely]
    - How do you estimate true recall when most items have never been checked for a tag?
22. How do you canonicalize extracted values ("Navy", "navy blue", "dark blue"; "1.5 L" compared with "1500 ml")? [Likely]
    - Is normalization a model, a dictionary, or both?
23. Two attributes conflict (size "XL" with a chest measurement of 34 inches). How does your system detect and resolve cross-attribute inconsistency? [Likely]

24. Design faceted product search at large scale. [Reported] (Amazon aggregate, August 2026, role and round not stated) [15]
   - ML angle [Likely]: where do facet values come from when sellers leave attributes empty, and what extraction coverage does a facet need before you show it?
   - How do extraction precision and coverage show up as filter recall for shoppers?
   - Which facets would you show for a given query, and how would you learn that?
25. A shopper filters by "Material: cotton" and many cotton items are missing from the results. Is that a precision or a recall problem in your attribute system, and how would you measure it from logs alone? [Likely]

## D. Attribute value extraction (classical and LLM)

26. Design a system to extract attribute values (brand, color, size, material, wattage) from product titles and descriptions. [Likely] (attribute extraction at catalog scale also appears in chapters 06 and 11)
    - Sequence tagging (BIO), QA-style span extraction, or generative extraction? What drives the choice?
    - How do you scale to thousands of attributes without one model per attribute?
27. How would you detect and fix wrong attribute values (size, color, material) on listings at catalog scale? [Likely]
    - How would you use an LLM for attribute extraction, and how do you verify its outputs?
28. How would you extract product attributes from noisy seller descriptions without inventing facts that are not in the text? [Likely] (WebIE, faithful extraction, Amazon Science [10])
    - How do you check that each extracted value is grounded in a span of the source?
    - What is your policy when the value is implied but not stated?
29. Use an LLM to generate or enrich product listings (titles, bullet points, attributes) from seller input. How do you prevent made-up attributes? [Likely]
30. Text says "compatible with iPhone 15", but the product is a case, not a phone. How do you stop the model from assigning the wrong brand or model? [Likely]
31. A bundle listing contains three products. How do you attach attributes to the right component? [Likely]
32. Which attributes would you extract from images instead of text (color, pattern, neckline), and how do you resolve text-image disagreement? [Likely]
    - When would you OCR the packaging in product images?
33. How would you serve an LLM extractor over hundreds of millions of items within budget? [Likely]
    - Batching, caching, distillation into a small model, or extracting only for high-traffic items?
    - How do you decide which items to re-extract when the model improves?
34. How do you design prompts and output schemas for structured LLM extraction (JSON schema, enums, "unknown" option)? [Likely]
    - How do you measure and reduce schema-invalid outputs?
35. How would you get calibrated confidence from an LLM extractor? [Likely]
    - Token log-probabilities, self-consistency across samples, verifier models, or agreement with a classical model?

36. Attribute evidence is spread across the title, description, bullet features and images, and the sources sometimes disagree. How do you combine them, and which source wins a conflict? [Likely] (multi-source attribute value extraction, MAVE dataset built from Amazon product profiles [19])
37. Your extraction benchmark spans more than a thousand product categories. How would you report results so that a few large categories don't hide failures on the long tail? [Likely] [19]
38. Design a product image upload pipeline for sellers. [Reported] (Amazon aggregate, August 2026, role and round not stated) [15]
   - ML angle [Likely]: which image-quality and policy checks run at upload, and which attributes would you extract from the images at the same time?
   - How would you detect an image that does not match the listing's title or product type?

## E. LLM verification and LLM-as-labeler

39. Label (attribute, value) pairs as Correct, Incorrect or Unknown using a majority vote over many LLM-judge configurations (several model families and prompts). When would you trust such an "LLM arena" as a labeler? [Likely] (SynthAVE, Amazon Science [9])
    - How would you measure judge agreement with humans, and how many human labels do you need?
    - What correlated errors survive majority voting?
40. Design an LLM quality validation system for extracted attributes. [Reported for the general prompt "Design an LLM quality validation system"] [6]; attribute application [Likely]
    - Which checks are deterministic (schema, units, ranges), and which are model-based?
41. Generator and verifier are the same model family. What goes wrong, and how would you fix it? [Likely]
42. How would you use LLM-generated synthetic listings to train a small extractor, and how do you avoid training on the LLM's own biases? [Likely]
43. When should the verifier's "Unknown" send an item to humans, and when should it leave the attribute empty? [Likely]

44. Your LLM-arena majority vote disagrees with the label produced by the synthetic-data pipeline. Who breaks the tie, and how do you size the human adjudication budget? [Likely] (SynthAVE, Amazon Science [9])
45. A stratified audit of a few hundred LLM-arena labels shows high agreement with an expert (Cohen's kappa around 0.9). Is that enough to release the labels as a benchmark or training set? What would you still check? [Likely] [9]
   - How large must the audit be to bound the label error rate per language or per attribute?
46. One locale's labels were judged by a different judge-model version than the others. How does that affect cross-locale comparisons, and what would you do about it? [Likely] [9]

## F. Cold-start tagging

47. A new product type launches with zero labeled examples. How do you tag and extract attributes on day one? [Likely]
    - Zero-shot LLM, nearest-neighbor transfer from similar product types, or seller-provided values?
48. A new marketplace and language launches (for example a new EU locale). How do you reuse models trained on English data? [Likely]
    - Translate-train, translate-test, or multilingual encoders? How do you handle locale-specific units and sizes?
49. A brand-new attribute is requested by the search team ("pet-friendly"). How would you backfill it across the catalog? [Likely]
    - How do you estimate precision before a full human audit?
50. A new seller with no history uploads 50,000 listings. How does your system treat their provided attributes compared with established sellers? [Likely]
    - How do you avoid penalizing honest new sellers?
51. With a fixed annotation budget, how would you choose which examples to label to estimate a new tagger's accuracy? [Likely] (active testing, Amazon Science [12])

52. You are estimating a new tagger's accuracy with active testing. When would you stop labeling with an adaptive stopping rule instead of fixing the budget up front? [Likely] (active testing, Amazon Science [12])
   - How do you report the uncertainty of an estimate built from non-uniformly selected samples?

## G. Human-in-the-loop labeling and review queues

53. Design the human-in-the-loop component: who labels, what they see, and how labels flow back to training. [Likely]
    - How do you measure inter-annotator agreement and resolve disagreement?
    - How do you write labeling guidelines for subjective tags ("casual", "luxury")?
54. You can afford 5,000 human labels per month. How do you spend them between training data, evaluation audits and production review? [Likely]
55. How do you prioritize the review queue? [Likely]
    - Uncertainty, traffic, risk of harm (safety and compliance attributes), or revenue impact?
56. Reviewers accept model suggestions 98% of the time. Is that good news? [Likely]
    - How would you detect automation bias, and would you hide suggestions for a sample?
57. How do you use review outcomes for active learning without biasing your evaluation set? [Likely]
    - Why do you need a random audit sample separate from the uncertainty-sampled queue?
58. How do you set and maintain SLAs for review when volume spikes (Prime Day onboarding)? [Likely]
    - What degrades gracefully: thresholds, coverage, or latency?

59. One managed human-review service is shared by text classification, document extraction and image moderation. What should be shared across tasks (workforce, UI framework, audit sampling), and what must be specific to each task? [Likely] (Amazon Augmented AI sample workflows for Textract, Comprehend custom classification, Rekognition and custom SageMaker models [21])
   - How would reviewer corrections on a custom classification endpoint flow back into retraining?

## H. Metrics and evaluation for tagging

60. Which offline metrics would you use for attribute extraction, and why per attribute rather than a global F1? [Likely]
    - How would you weight attributes by business importance?
61. Define coverage and precision for a catalog attribute. How do you trade one against the other when setting the auto-publish threshold? [Likely]
    - Draw the precision-coverage curve. Where would you operate for "brand" compared with "color"?
62. How do you build a gold evaluation set that is representative of traffic and of the long tail? [Likely]
    - Stratified by product type, traffic tier and source?
63. How would you show that better attributes caused more sales or fewer returns? [Likely]
    - How would you design the A/B test when the treatment is catalog data that every surface reads?
    - What about interference between treatment and control through shared search ranking?
64. Which online signals indicate an attribute is wrong (filter abandonment, "not as described" returns, customer Q&A), and how noisy are they? [Likely]

## I. Document ingestion, OCR and layout

65. Design a system that ingests millions of PDFs per day (invoices, receipts, contracts, forms) and extracts structured data. [Likely] (Amazon Textract product context)
    - Which steps are synchronous and which are asynchronous?
    - How do you scale per page compared with per document?
66. How do you decide whether to OCR a PDF or use its embedded text layer? [Likely]
    - How do you detect a "fake" text layer, such as a bad OCR from a scanner?
67. Walk me through layout analysis: words and bounding boxes, reading order, blocks, tables and key-value pairs. [Likely]
    - How do you get reading order right on multi-column pages and pages with sidebars?
68. How would you handle handwriting, stamps, signatures and checkboxes? [Likely]
69. How do you measure OCR quality (CER/WER) without ground-truth transcriptions for production documents? [Likely]
    - Which image-quality signals predict OCR failure before you run it?
70. Design an evaluation framework that checks whether a text-to-image system spells a requested brand name correctly and places it naturally. [Reported] [6][7]
    - How would you combine OCR accuracy, embedding-based semantic checks and a human review protocol? [Reported] [6]
    - Where does OCR itself mislead your evaluation (stylized fonts, curved text)?

71. Design receipt and expense extraction (vendor, date, total, tax, line items) for an expense-reporting product. [Likely] (Textract AnalyzeExpense summary fields and line items [20])
   - How do you evaluate summary fields separately from line items?
   - How do you handle crumpled phone photos of receipts, or foreign currencies?
72. Design identity-document extraction (driver's licenses, passports) for an identity-verification flow. What accuracy, latency and fraud constraints change your design? [Likely] (Textract AnalyzeID [20])
   - Which errors are worse here: a false reject or a false accept?

## J. Key-value, table and field extraction

73. Design key-value extraction for forms (invoice number, date, total, vendor). [Likely]
    - Template-based, layout-aware model (text + position + image), or VLM? When would you use each?
74. How would you extract line items from invoice tables whose layout differs across thousands of vendors? [Likely]
    - How do you handle tables spanning pages, merged cells and tables without borders?
    - What metric would you use for table structure compared with cell content?
75. How do you tell lookalike fields apart (bill-to compared with ship-to, invoice date compared with due date)? [Likely]
76. How would you attach evidence (page and bounding box) to every extracted value, and why does it matter for review and audit? [Likely]
77. What cross-field validation rules would you add (line items sum to subtotal, tax rate in plausible range, dates ordered), and how do rules interact with model confidence? [Likely]
78. How would you add "queries" support, where a customer asks a natural-language question of a document ("What is the policy end date?")? [Likely]
    - How do you evaluate answers that are absent from the document?

79. Extracted tables must export cleanly to spreadsheets and data frames for downstream users. How would you evaluate table output for usability, not only cell accuracy? [Likely] [20]
   - Header detection, merged cells, numeric typing: which of these break downstream analysis most often?

## K. Document classification, splitting and routing

80. Design document classification for an intake pipeline (invoice, receipt, W-2, ID, bank statement, contract, other). [Likely]
    - Text only, image only, or multimodal? How do you treat "other"?
81. A single uploaded PDF contains several documents (a mortgage packet). How do you split it into documents? [Likely]
    - Page-pair boundary classification or sequence labeling?
82. How do you route documents to different extractors, and what happens when the classifier is wrong? [Likely]
    - How do you limit the damage of a misrouted document downstream?
83. A new document type appears in traffic that the classifier has never seen. How do you detect it (open-set or out-of-distribution detection)? [Likely]
84. Documents arrive in 20 languages. How do classification and extraction change? [Likely]

85. Lay out an intelligent document processing pipeline in five stages: classification, extraction, enrichment, validation and human review. What does enrichment add, and why does validation come before human review? [Likely] (AWS intelligent document processing guidance [22])
86. Design an agentic document-processing system in which an LLM agent decides, per document, which tools to call (OCR, table parser, master-data lookup, human review). How do you bound cost and latency, and how do you make each decision auditable? [Likely] [22]
   - When is a fixed pipeline better than an agent?
87. Orchestrate a document workflow that mixes LLM prompt steps, OCR, queues and human review. How do you handle retries, idempotency and partial failure across stages? [Likely] [22]

## L. Confidence thresholds, straight-through processing and human review

88. How would you set per-field confidence thresholds that send a document to human review, given a target accuracy after review? [Likely] (Textract with Augmented AI human-review product context)
    - Why per field and per document type, rather than one global threshold?
    - How do you recalibrate when the model is retrained?
89. Define straight-through-processing (STP) rate. How do you trade STP against error cost? [Likely]
    - Write the expected-cost expression you would minimize (error cost, review cost, latency).
90. Model confidence is badly calibrated on new layouts. How do you detect and fix it? [Likely]
    - Temperature scaling per document type, conformal prediction, or a separate error predictor?
91. Design the reviewer UI and workflow. What does the reviewer see, and what do you log? [Likely]
    - How do you capture reviewer corrections as training data with field-level granularity?
92. Some fields are high-stakes (payment amount, bank account number) and others low-stakes (memo). How does review policy differ? [Likely]
93. How do you sample documents that passed automatically for audit, and how big must the sample be to bound the error rate? [Likely]

## M. Multimodal VLM and LLM extraction for documents

94. When would you replace an OCR-plus-layout-model pipeline with a vision-language model that reads the page image directly? [Likely]
    - What are the cost, latency and accuracy trade-offs per page?
    - How do you get bounding-box evidence from a VLM?
95. A VLM fills in a field that is not in the document ("hallucinated invoice number"). How do you detect and prevent it? [Likely]
    - How would you cross-check VLM outputs against OCR tokens?
96. How do you handle 300-page contracts that exceed the model's context window? [Likely]
    - Chunking by layout section, retrieval, or a map-reduce extraction?
97. Design a hybrid: a cheap specialist model for common layouts and a VLM fallback for hard or unseen ones. How do you decide which path each document takes? [Likely]
98. How would you distill a large VLM extractor into a smaller model for high-volume document types? [Likely] (LLM distillation at Amazon Science [13])
99. Extracted documents feed a RAG assistant. How does extraction quality affect retrieval and answers, and how would you evaluate end to end? [Likely]
    - How do you represent tables and key-value pairs in the index?

100. Design an enterprise assistant that answers questions over a company's internal documents (PDFs, scans, spreadsheets) while respecting each user's document permissions. [Likely] (synthesized Amazon question bank, not a leaked question [16])
   - Where in ingestion and retrieval do you enforce permissions?
101. Evaluate that document assistant at the claim level, separating retriever errors (claim recall, context precision) from generator errors (faithfulness, noise sensitivity, hallucination). How would a badly extracted table show up in those metrics? [Likely] (RAGChecker, Amazon Science [17])
102. A customer's VLM-based document extraction costs too much per page. Cut the cost substantially without unacceptable quality loss. [Likely] (adapted from a cost-reduction prompt in a synthesized Amazon question bank [16])
   - Page triage, cheaper OCR-plus-text model on easy pages, caching, batching, smaller distilled models: in which order would you try them?

## N. Data, labeling and privacy for documents

103. How do you get labeled training data for document extraction when customer documents are confidential? [Likely]
    - Synthetic document generation, customer opt-in, on-premise labeling, or federated approaches?
104. How do you handle PII (SSNs, account numbers) in training data, logs and reviewer views? [Likely]
    - Which fields would you mask from reviewers, and how does that affect review quality?
105. How would you generate synthetic invoices or forms that actually help the model on real layouts? [Likely]
    - How do you measure the synthetic-to-real gap?
106. A customer wants a custom extractor for their own form with 20 examples (adapter-style fine-tuning). How would you design that feature? [Likely]
    - How do you stop one customer's adapter from leaking another customer's data?

## O. Failure modes, monitoring and drift (both systems)

107. What are the top five failure modes of your tagging or document system, and how would you detect each in production? [Likely]
108. A vendor changes its invoice template overnight and extraction accuracy for that vendor collapses. How do you notice and recover? [Likely]
109. Precision on "brand" dropped from 97% to 91% after a model update, but overall F1 went up. What do you do? [Likely]
    - How would per-attribute launch gates have caught it?
110. How do you monitor drift in input distribution (new product types, new document layouts, new scanners) without labels? [Likely]
111. Errors compound across the cascade (OCR, then classification, then extraction). How do you attribute an end-to-end error to a stage? [Likely]
    - Would you train the pipeline end to end, or keep stages separate? Why?
112. Sellers learn to game your tagger (keyword stuffing to get tagged "organic"). How do you make it robust? [Likely]
113. How would you roll back a bad batch of auto-published attributes that already reached search and detail pages? [Likely]
    - What lineage and versioning do you need for that?

114. Design an end-to-end spam detection system. [Reported] (Amazon aggregate, undated, role and round not stated) [15]
   - Listing application [Likely]: apply it to keyword-stuffed titles, fake attribute values and duplicate listings. Which signals come from your tagging system?
   - How do you keep adversarial sellers from probing your thresholds?

## P. Trade-offs, cost and the L5 bar

115. Classical extractor, fine-tuned small LLM, or a hosted frontier LLM: for each, what is the cost per million items or pages, the accuracy and the latency? How would you choose? [Likely]
116. Rules and templates still beat your model on the top 50 vendors. Do you keep them? [Likely]
117. Where would you invest the next quarter: better models, better labels, better taxonomy, or better review tooling? Defend the choice with data you would collect. [Likely]
118. How would you explain a precision-coverage trade-off to a product manager who wants "100% of attributes filled"? [Likely]
119. Tell me about a time you shipped a model whose labels were noisy or whose taxonomy was in flux. What would you do differently? [Likely] (LP blend: Dive Deep, Insist on the Highest Standards)
120. Tell me about a time you had to simplify an over-engineered extraction or labeling pipeline. [Likely] (LP: Invent and Simplify)

121. For a classification or extraction model you built, explain the backbone choice, how you handled class imbalance, and what you would do with more resources. [Reported] (Amazon MLE aggregate, project deep dive on a detection system) [7]
122. Write a symmetric, numerically stable InfoNCE loss in PyTorch for paired image and text embeddings, then explain how you would use such a model to check that a listing's images match its text attributes. [Reported for the coding prompt "Handwritten InfoNCE in PyTorch" (Amazon MLE aggregate and dated aggregator entry, February 2026) [7][15]]; tagging application [Likely]

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026 (includes the "1 Sys Design (Team-specific)" loop report). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer (prep vendor), 2025-2026. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
4. Amazon applied scientist interview loops, Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
5. "Amazon Applied Scientist | L6 Luxembourg | Passed, freeze | 2022", LeetCode Discuss, 2022 (10k listings across 100 categories). https://leetcode.com/discuss/interview-experience/4421594/
6. "Amazon Research Scientist Interview Process" (covers Research and Applied Scientist loops; aggregator adapted from trueinterview.io, CC BY 4.0), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
7. "Amazon Machine Learning Engineer Interview Process" (aggregator adapted from trueinterview.io, CC BY 4.0), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
8. "My interview experience for the Applied Scientist II role at Amazon" (Advertising Trust), Chaitanya Mundle, Medium, circa 2024. https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
9. amazon-science/SynthAVE, "SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation" (EMNLP 2026 Industry Track), GitHub. https://github.com/amazon-science/SynthAVE
10. amazon-science/webie, "WebIE: Faithful and Robust Information Extraction on the Web" (ACL 2023), GitHub. https://github.com/amazon-science/webie
11. amazon-science/carbon-assessment-with-ml, "CaML: Carbon Footprinting of Household Products with Zero-Shot Semantic Text Similarity" (WWW 2023), GitHub. https://github.com/amazon-science/carbon-assessment-with-ml
12. amazon-science/NLPActiveTesting, "Select, Label, Evaluate: Active Testing in NLP" (EMNLP 2026), GitHub. https://github.com/amazon-science/NLPActiveTesting
13. "Learning with less: Knowledge distillation from large language models via unlabeled data", Amazon Science. https://www.amazon.science/publications/learning-with-less-knowledge-distillation-from-large-language-models-via-unlabeled-data
14. "Amazon Interview Experience for Applied Scientist 2", Enginebogie, circa 2025 (building with very little labeled data, as recorded in chapter 03). https://enginebogie.com/interview/experience/amazon-applied-scientist-2/1004

15. "Amazon" company question listing (dated candidate-report entries, 2025-2026, including "Design Faceted Product Search at Large Scale", "Design a Product Image Upload Pipeline for Sellers", "Design an end-to-end spam detection system", "Design an LLM quality validation system" and "Handwritten InfoNCE in PyTorch"), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
16. "Amazon" AI engineer question bank (the page says its questions are synthesized, not leaked; reviewed October 2026), ombharatiya/AI-Engineer-Interview-Questions, GitHub. https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
17. amazon-science/RAGChecker, "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation" (NeurIPS Datasets and Benchmarks Track), GitHub. https://github.com/amazon-science/RAGChecker
18. amazon-science/esci-data, "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search" (arXiv 2206.06588, 2022), GitHub. https://github.com/amazon-science/esci-data
19. google-research-datasets/MAVE, "MAVE: A Product Dataset for Multi-source Attribute Value Extraction" (WSDM 2022; 3M attribute-value annotations across 1,257 categories from Amazon product profiles), GitHub. https://github.com/google-research-datasets/MAVE
20. aws-samples/amazon-textract-textractor, Textractor library for Amazon Textract (forms, tables, AnalyzeExpense, AnalyzeID, table export), GitHub. https://github.com/aws-samples/amazon-textract-textractor
21. aws-samples/amazon-a2i-sample-jupyter-notebooks, Amazon Augmented AI (A2I) human-review sample workflows (Textract, Comprehend, Rekognition, custom SageMaker models), GitHub. https://github.com/aws-samples/amazon-a2i-sample-jupyter-notebooks
22. aws-samples/aws-ai-intelligent-document-processing, AWS intelligent document processing guidance (classification, extraction, enrichment, validation, human review; agentic and prompt-flow variants), GitHub. https://github.com/aws-samples/aws-ai-intelligent-document-processing

Note on product context: Amazon Textract, Amazon Augmented AI (A2I), Amazon Comprehend custom classification and Amazon Bedrock are public AWS products that ground the [Likely] document questions. The official docs.aws.amazon.com pages could not be fetched in either pass. Refs [20]-[22] are AWS's own public sample repositories and stand in for them.
