# 02. Document Parsing and Chunking: From Pixels to Retrievable Units

## Overview

Every RAG system rests on its ingestion layer. Whether a retriever can find the right evidence depends on decisions made long before any query arrives: how a PDF page becomes text, whether a table keeps its header row, where one chunk ends and the next begins, and what metadata travels with each chunk into the index. Retrieval errors that look like embedding or ranking problems are often parsing and chunking problems that started upstream.

This chapter covers three layers of ingestion design.

1. **Layout-aware parsing.** This layer turns heterogeneous documents (born-digital PDFs, scans, Office files, HTML) into a structured, ordered representation. The systems covered are open-source toolkits (Docling, Unstructured, Marker, MinerU, olmOCR), commercial APIs (LlamaParse, Reducto, Azure AI Document Intelligence, Mistral OCR) and "parse-free" visual retrieval (ColPali).
2. **Chunking.** This layer segments the parsed representation into retrieval units. Strategies covered are fixed-size, recursive, structure- or element-aware, semantic (embedding-breakpoint and clustering), LLM-driven ("agentic"), hierarchical/multi-granularity, proposition-level, late chunking, and Anthropic's Contextual Retrieval.
3. **Metadata enrichment.** This layer attaches headings, provenance, bounding boxes, LLM-generated summaries, questions and situating context, so that chunks can be found, filtered, cited and expanded.

The canonical ingestion pipeline that most of the designs in this chapter instantiate looks like this:

```
 raw files                 parse                      structure                 index
+-----------+   +---------------------------+   +-------------------+   +----------------+
| PDF/scan  |-->| text layer / OCR / VLM    |-->| typed elements    |-->| chunker        |
| DOCX/PPTX |   | layout detection (boxes)  |   | (title, para,     |   | (size, struct, |
| HTML/XLSX |   | table structure recog.    |   |  table, figure,   |   |  semantic,     |
| images    |   | reading-order recovery    |   |  list, caption)   |   |  late, ctx)    |
+-----------+   +---------------------------+   | + hierarchy tree  |   +-------+--------+
                                                | + bbox/provenance |           |
                                                +-------------------+           v
                                       +-----------------------------------------------+
                                       | enrichment: heading path, page, bbox, doc     |
                                       | title, LLM context/summary/questions, entities|
                                       +-------------------------+---------------------+
                                                                 v
                                              dense index + BM25 index + doc store
```

The rest of the chapter goes through each stage, system by system, then compares them and ends with design guidance.

---

## Part I: Layout-aware parsing

### The problem

A PDF is a set of drawing instructions, not a semantic document. Characters are placed at coordinates. Columns, tables, headers, footers and reading order are implicit, and scanned pages have no text layer at all. Naive text extraction therefore interleaves columns, merges page furniture (running headers, page numbers) into body text, flattens tables into token soup and silently drops image-only content. The benchmark literature summarized below shows that the gap between a plain text dump and a layout-aware parser is very large. On olmOCR-Bench, a CPU text-layer extractor with no layout model scores about 20-22 points overall, while layout-aware pipelines and VLMs score 66-86 [37].

Two architectural families have emerged, plus a hybrid of the two.

- **Modular pipelines** use a layout detector (object detection over page images), then specialized models per region type (OCR, table structure, formula recognition), then rule-based or learned reading-order assembly. Examples are Docling, Unstructured hi_res, Marker, the MinerU pipeline backend and Azure Layout.
- **End-to-end VLM parsers** use one vision-language model that reads a page image and emits Markdown, HTML or a custom markup. Examples are olmOCR, Granite-Docling and SmolDocling, MinerU's VLM backend, Mistral OCR, and general-purpose models such as Gemini.
- **Hybrids** route regions or pages to different engines by difficulty and often add an LLM or VLM correction pass. Examples are Reducto, LlamaParse's agentic tiers and Marker's `--use_llm` mode.

### Docling (IBM Research)

**Problem and positioning.** Docling is an MIT-licensed, self-contained package for PDF and document conversion that runs on commodity hardware. Its specialized models are DocLayNet-trained layout analysis and TableFormer table-structure recognition [12]. Docling v2 shipped on 16 October 2024 and introduced a unified document model [12][15]. Today it accepts PDF, DOCX, PPTX, XLSX, HTML, EPUB, images, audio and several XML schemas, exports Markdown, HTML, DocTags and lossless JSON, and integrates with LangChain, LlamaIndex, CrewAI and Haystack and with agents through an MCP server [13].

**Architecture.** Docling's default PDF pipeline runs in stages, with a pluggable model at each stage [16]:

```
PDF page --> text-layer parse ----------------------------+
         \-> page image --> Layout (RT-DETR family) ------+--> assemble -> DoclingDocument
                        \-> OCR (auto/Tesseract/EasyOCR/  |    (reading     (body tree,
                             RapidOCR/macOS Vision/Surya)-+     order,       furniture,
                        \-> TableFormer (accurate/fast) --+     captions)    groups, prov)
                        \-> optional enrichers: picture classification,
                            chart extraction, formula/code
```

- **Layout.** The current default layout model is `docling-layout-heron`. Larger variants are `heron-101` and the `egret` medium, large and xlarge models, all object detectors from the RT-DETR family [16]. The September 2025 report on these models says heron-101 (RT-DETRv2 with a ResNet-101 backbone) reaches about 78% mAP on DocLayNet, which is a 23.9% improvement over Docling's previous layout model. Its runtime is about 0.028 s per image on one A100, on par with the old model, and the D-FINE-based egret-m is the fastest at about 0.024 s per image. The models were trained on about 150,000 documents [17]. The authors also argue that mAP may be an imperfect proxy for layout quality [17].
- **Tables.** TableFormer (TableModel04rs, which uses the OTSL representation) predicts table structure and cell bounding boxes. It was trained on datasets including PubTabNet (516k+ tables), FinTabNet (112k+) and TableBank (417k) [19].
- **Representation.** `DoclingDocument` is a Pydantic model. It has content lists (`texts`, `tables`, `pictures`, `key_value_items`) and structure trees: `body` for main content, `furniture` for headers and footers, and `groups` for lists and chapters. All nodes are linked by JSON pointers, and each item carries bounding boxes and provenance. Reading order is encoded as the order of children in the `body` tree [15]. Because furniture is kept separate from body, page headers and footers can be excluded from chunks without writing heuristics.
- **VLM path.** `VlmPipeline` converts pages end-to-end with a VLM that emits DocTags (preferred), Markdown or HTML [16]. IBM's Granite-Docling-258M was released in September 2025 under Apache 2.0. It succeeds the experimental SmolDocling-256M-preview, swaps the SmolLM-2 backbone for a Granite 3-based one and moves the vision encoder from SigLIP to SigLIP2. IBM says it addressed SmolDocling's repetition-loop instability through dataset filtering, and it positions Granite-Docling as complementary to the Docling pipeline rather than a replacement for it [18].

**Chunking integration.** Docling's chunkers operate on `DoclingDocument` directly, not on flattened text [14]:
- `HierarchicalChunker` emits one chunk per document element. It merges list items by default and attaches headings and captions as metadata.
- `HybridChunker` takes the hierarchical output and makes it tokenizer-aware. It splits only chunks that exceed the token limit, then merges successive undersized chunks, but only when they share the same headings and captions (`merge_peers=True`). For tables that span several chunks it repeats the header row in each (`repeat_table_header=True`).
- `LineBasedTokenChunker` keeps lines intact for tables, code and logs.
- `contextualize(chunk)` returns the chunk text enriched with metadata such as headings. This string is the one meant to be embedded [14].

**Failure modes and lessons.** In Datalab's olmOCR-Bench comparison, Docling's default pipeline scores 50.3 overall (64.0 on born-digital pages) at about 2.1 pages/s on a B200 host. That is well below Marker's balanced mode and MinerU's pipeline on the same bench [37]. The comparison is vendor-run, but it is consistent with the general pattern that text-layer pipelines lose most on math and scans. The design lesson from Docling is that the representation matters as much as the models. A typed tree with furniture separated out and provenance attached makes structure-aware chunking and citation almost free.

### Unstructured

**Problem.** Unstructured partitions dozens of file types into a flat list of typed *elements* (Title, NarrativeText, ListItem, Table, Image and others), each with metadata. It then chunks those elements.

**Architecture.**
- **Partition strategies.** The open-source library defines `auto`, `fast`, `ocr_only` and `hi_res` [20]. `fast` reads the PDF text layer. `hi_res` runs a document layout model, and the default model in `unstructured-inference` is YOLOX, with a Detectron2-ONNX alternative [23]. `ocr_only` OCRs the page image. Reading order for detected regions can be sorted with an XY-cut mode [20].
- **Chunking.** Chunking happens after partitioning, on elements. In the hosted API, `basic`, `by_title`, `by_page` and `by_similarity` are available, and if no strategy is specified no chunking occurs [22]. The `by_title` docstring describes the core rule: a new chunk starts when a Title element is detected or when metadata such as page or section changes, and sections are cut once they exceed `max_characters` (or `max_tokens`) [21]. The default hard maximum is 500 characters [21]. A soft maximum (`new_after_n_chars`) and small-section combining (`combine_text_under_n_chars`) let short sections merge. Overlap applies by default only when an oversized element has to be text-split. `overlap_all=True` extends it to normal chunks, but the docs warn this "pollutes" otherwise clean semantic boundaries [21]. Tables are isolated in their own chunks by default (`isolate_table=True`), and oversized tables become `TableChunk`s with repeated headers (`repeat_table_headers=True`). `orig_elements` keeps the original element metadata inside each chunk [21].

**Evidence.** Unstructured's own researchers studied element-type-based chunking on financial reports (FinanceBench). The paper's abstract reports that element-type chunking "largely improves" RAG results and yields the best chunk size without tuning [10]. The README of the open-source project compares its own OSS output against the hosted Transform v2 API on vendor benchmarks: table cell content 0.426 vs 0.866, adjusted text accuracy 0.715 vs 0.878, and so on. The OSS figures come from more than 1,000 pages and the hosted figures from a 224-page set [20]. That gap illustrates a recurring pattern: hosted parsers ship newer models than the open-source releases they grew from.

**Lesson.** An element-first design ("partition, then chunk elements") makes section-boundary-respecting chunking the default, and keeping `orig_elements` preserves provenance through the merge step.

### LlamaParse (LlamaIndex)

**Problem.** LlamaParse is a hosted parser that produces LLM-ready Markdown or JSON from complex documents. Its distinguishing feature is tiered LLM/VLM-assisted ("agentic") parsing.

**Architecture and cost.** LlamaParse v2 was announced in December 2025 with four tiers priced in credits per page: Fast at 1 credit, Cost-effective at 3, Agentic at 10 and Agentic Plus at 45. Layout extraction is a +3-credit add-on [24][25]. The v2 launch post claims Agentic Plus was cut 50% in price at comparable accuracy, which is consistent with an older 90-credit figure [24][25]. The docs list $1.25 per 1,000 credits in North America and Europe. Older pages show $1.00, so the rate appears to have changed and should be checked at the source [24]. At $1.25 per 1,000 credits, Agentic Plus works out to roughly $56 per 1,000 pages and Fast to about $1.25 per 1,000 pages (approximate, derived arithmetic). LlamaIndex documentation also describes per-page routing on the agentic modes, which reserves premium processing for complex pages [24]. The older `llama_parse` / `llama-cloud-services` client repository is deprecated in favor of `llama-cloud` 1.0+, with maintenance until 1 May 2026 [26].

**Lesson.** Price discrimination per page, routing cheap pages to fast parsers and hard pages to VLM-backed modes, is now a standard design pattern. Ingestion cost can vary by more than an order of magnitude depending on routing policy.

### Reducto

**Problem.** Reducto targets enterprise documents where table fidelity and citations matter, such as finance, insurance and healthcare.

**Architecture.** According to a 2025 write-up of Reducto's architecture, Reducto combines traditional OCR, layout detection and specialized VLMs in one pipeline, routing different models to different regions by complexity. The motivation it gives is that end-to-end LLMs fail systematically on document parsing, for example by hallucinating values in dense financial tables. The same write-up reports more than 1 billion documents processed [29]. Reducto's own blog says OCR provides character-level text accuracy while VLMs supply visual context for tables and multi-column layouts. Outputs include bounding boxes for citations, per-block layout labels and LLM-ready chunks. The blog also describes "Agentic OCR", a multi-pass self-correction step [28].

```
page --> layout detect --> region router --+--> classic OCR (dense text)
                                           +--> VLM (tables, charts, complex)
                                           +--> ...
                         merge + verify <--+
                               |
                     agentic multi-pass correction --> blocks{type,bbox,text} --> chunks
```

**RD-TableBench.** Reducto released RD-TableBench, 1,000 hand-labeled table images with grading code and a results viewer. In Reducto's original evaluation Reducto scored an average similarity of 0.902, against 0.646 for Google Document AI [27]. A rerun by Pulse, a competitor, using Reducto's ground truth and scorer, reports Reducto Agentic (2026) at 0.9308 and also notes that a Reducto audit found 43 ground-truth files (4.3%) with verifiable errors [30]. Third-party coverage of a February 2025 practitioner post, as reported by Gigazine, cites rd-tablebench scores of Gemini 2.0 Flash 0.84 ± 0.16, Claude 3.5 Sonnet 0.81 and GPT-4o 0.67, against Reducto's 0.90 [31].

**Failure modes and lessons.** The benchmark is vendor-owned and the vendor scores at the top, so results should be read as directional. The architectural lesson is still general. Pure generative parsing has a hallucination failure mode that is unacceptable for numeric tables, and hybrid systems use deterministic OCR as an anchor for character accuracy.

### Azure AI Document Intelligence (Layout)

**Problem.** Azure Document Intelligence is a managed OCR and layout service integrated with Azure AI Search and Azure OpenAI ingestion.

**Architecture.** `prebuilt-layout` (v4.0, API version 2024-11-30 GA) extracts pages, paragraphs, lines, words, selection marks, tables (with row and column spans and header detection), figures (with captions and optional cropped images) and hierarchical sections [32]. Paragraphs carry logical roles, which are useful signals for chunking: `title`, `sectionHeading`, `pageHeader`, `pageFooter`, `pageNumber` and `footnote`, with role support varying by file type [32]. Setting `outputContentFormat=markdown` emits Markdown in which tables are rendered as HTML tables and selection marks as ☒/☐ [32]. Limits include up to 2,000 pages per PDF or TIFF, 500 MB per file on the paid tier, and only the first two pages on the free tier [32].

**Chunking pattern.** Microsoft's RAG guidance is to emit Markdown from Layout and split on headings, for example with LangChain's `MarkdownHeaderTextSplitter` mapping `#`, `##` and `###` to header metadata [33]. For Azure AI Search, the chunking guide recommends starting fixed-size chunking at 512 tokens (about 2,000 characters) with 25% (128-token) overlap. It also shows how chunk count on a 200-page e-book varies with the Text Split skill's parameters: 172 chunks at 1,000 characters with no overlap, 216 with 200 characters of overlap, and 13,361 with sentence mode. It suggests "custom combinations" such as appending the document title to mid-document chunks to prevent context loss [34].

**Lesson.** A parser that emits semantic roles (headings vs page headers vs footnotes) lets downstream chunkers drop furniture and use heading paths as metadata. Microsoft's own docs effectively endorse "Markdown headings as chunk boundaries" as the default enterprise pattern.

### Marker (Datalab)

**Architecture.** Marker reads the PDF text layer and OCRs selectively. Layout, OCR and table recognition run through a single "surya" VLM served by a local inference server (vLLM or llama.cpp). Many thin CPU workers share that server [37]. An optional `--use_llm` mode adds an external LLM (Gemini by default, with Claude, OpenAI-compatible and Ollama also supported) to merge tables across pages, fix inline math and extract form values [37].

**Numbers (vendor-published, olmOCR-Bench, one B200 host).** Balanced mode scores 76.0 overall and 83.5 on born-digital pages at 2.9 pages/s sustained, about 341 ms effective per page. Fast mode scores 66.6 at 7.4 pages/s. CPU-only no-OCR mode scores 43.6 at 23.7 pages/s. In the same table the MinerU pipeline scores 72.7 at 0.54 pages/s and Docling 50.3 at 2.1 pages/s [37]. Marker states that balanced mode saturates only about 30% of the GPU because the ceiling is VLM decode throughput, and that concurrency rather than single-stream latency (about 0.3 pages/s) drives production throughput [37].

**Lesson.** At scale, parsers become inference-serving problems. The relevant design metric is sustained pages per second per GPU under concurrency, not single-document latency.

### MinerU (OpenDataLab / Shanghai AI Lab)

MinerU began as a pipeline system. In the description used by the OmniDocBench authors, it uses a layout model to segment regions and then applies task-specific models per region [39]. It has since added a VLM backend (MinerU 2.5, plus the 2026 MinerU 2.5 Pro report). MinerU 4.0 offers four parsing tiers: Flash for fast previews and indexing, Basic for OCR and model-based parsing, and Standard and Advanced for demanding layouts. It parses Office, EPUB, HTML and CSV formats natively and keeps "stable citation locators" for agent reading [38]. Datalab notes that MinerU's VLM backend scores higher than its pipeline backend but is "a different, full-page-VLM approach" [37].

### End-to-end VLM parsers: olmOCR, Mistral OCR, Gemini

- **olmOCR (Ai2).** olmOCR fine-tunes a 7B VLM (initially from Qwen2-VL-7B-Instruct). Its "document anchoring" technique injects coordinates of salient elements and raw text from the PDF's text layer into the prompt alongside the page image [36]. Ai2 quotes about $190 per million pages, roughly 1/32 the cost of GPT-4o batch processing [36]. The README says "less than $200 USD per million pages", requires a GPU with at least 12 GB, and lists v0.4.0 (October 2025) as adding synthetic data and RL training for about +4 points on olmOCR-Bench, a suite of more than 7,000 test cases across 1,400 documents [35]. On that bench olmOCR v0.4.0 scores 82.4. In the same README table, Mistral OCR API scores 72.0, Marker 1.10.1 76.1, MinerU 2.5.4 75.2, DeepSeek-OCR 75.7, PaddleOCR-VL 80.0 and Chandra OCR 0.1.0 83.1. The weakest category for nearly every system is "old scans", at 29-51 [35].
- **Mistral OCR.** Coverage of the March 2025 launch reported about 1,000 pages per dollar (2,000 with batch) and 2,000 pages per minute per node. Mistral OCR 3 (`mistral-ocr-2512`, December 2025) is reported at $2 per 1,000 pages, or $1 with the Batch API, with a vendor-claimed 74% win rate over OCR 2 [40]. These figures come from secondary sources.
- **General multimodal LLMs.** A widely discussed February 2025 practitioner post argued that Gemini 2.0 Flash made PDF-to-text for RAG dramatically cheaper. It contrasted this with NVIDIA's nv-ingest, which in that account required Kubernetes, eight services and two GPUs [31]. Google staff have clarified that the often-quoted 258 tokens per page is an approximate average, not a fixed cost [31].

**Failure modes of VLM parsers.** The known failure modes are repetition loops (which IBM explicitly addressed for Granite-Docling [18]), hallucinated output on blank pages (fixed in olmOCR v0.3.0 [35]), auto-rotation errors [35], and numeric hallucination in dense tables (Reducto's stated motivation for hybrid routing [29]). OmniDocBench's supplementary analysis finds that colored backgrounds hurt pipeline tools much more than VLMs, while pipeline tools hold up better on merged-cell tables. MinerU and Marker also lose accuracy on mixed Chinese and English content [39].

### Benchmarks for parsing

- **OmniDocBench (CVPR 2025).** 1,651 PDF pages across 10 document types, 5 layout types and 5 languages in the current version, with component-level evaluation of text, tables, formulas and reading order. The v1.5 Overall metric is ((1 - text edit distance) x 100 + table TEDS + formula CDM) / 3 [39]. Its leaderboard has been updated continuously through 2026 with dozens of pipeline and VLM systems [39].
- **olmOCR-Bench.** More than 7,000 unit-test-style checks over about 1,400 PDFs in categories such as arXiv math, old scans, tables, headers and footers, multi-column and long tiny text [35][37].
- **RD-TableBench.** 1,000 table images, vendor-owned (see above) [27].

### Parse-free retrieval: ColPali

ColPali feeds PaliGemma-3B ViT patch outputs through a linear projection to produce ColBERT-style multi-vector page embeddings, and is trained so queries match page images directly. The authors argue this "removes the need for potentially complex and brittle layout recognition and OCR pipelines" [41]. In 2026 the `colpali-engine` package was deprecated in favor of Sentence Transformers v6's `MultiVectorEncoder` [41]. In this design the page itself is the chunk. It suits visually rich corpora such as slides, charts and infographics, but answer generation still needs a VLM that can read the retrieved page, and per-page multi-vector storage is much larger than one vector per chunk.

### NVIDIA NeMo Retriever extraction (nv-ingest)

NVIDIA's open framework splits documents into pages, classifies artifacts (text, tables, charts, infographics), extracts them with NIM microservices or Hugging Face models, and then embeds and uploads the results. For production it is deployed on Kubernetes via Helm, with a lighter library mode for workloads under about 100 PDFs [42]. Its design treats charts and infographics as first-class extraction targets, a direction other parsers have followed with chart-to-table enrichers; Docling's chart extraction defaults to Granite Vision 4.1 4B [16].

### Parsing comparison

| System | Type | Layout / tables | Output | Notable numbers (source) | Main risks |
|---|---|---|---|---|---|
| Docling | OSS pipeline + optional VLM | RT-DETR Heron; TableFormer | DoclingDocument, MD, JSON, DocTags | heron-101 about 78% mAP, 0.028 s/img A100 [17]; 50.3 olmOCR-Bench [37] | weaker on math and scans in default mode |
| Unstructured | OSS + hosted API | YOLOX (hi_res) | typed elements | OSS vs hosted table content 0.43 vs 0.87 (vendor) [20] | OSS lags hosted models |
| LlamaParse | hosted, tiered | VLM/LLM tiers | MD, JSON | 1/3/10/45 credits per page [24] | cost varies with tier |
| Reducto | hosted hybrid | CV + OCR + VLM, agentic correction | blocks + bbox + chunks | RD-TableBench 0.90 (vendor) [27] | benchmark owned by vendor |
| Azure DI Layout | managed | roles, tables, sections, figures | JSON, MD (HTML tables) | 2,000-page and 500 MB limits [32] | HTML-in-MD tables need handling |
| Marker | OSS + hosted | surya VLM, selective OCR | MD, JSON, HTML | 76.0 at 2.9 pages/s per B200 (vendor) [37] | math needs balanced or LLM mode |
| MinerU | OSS pipeline + VLM | layout, then per-region models | MD, JSON, more | pipeline 72.7 at 0.54 pages/s [37] | mixed-language drop [39] |
| olmOCR | OSS VLM (7B) | end-to-end with anchoring | MD | 82.4 olmOCR-Bench; under $200 per M pages [35] | needs GPU; old scans weak |
| Mistral OCR 3 | hosted VLM | end-to-end | MD | $2 per 1k pages ($1 batch) [40] | vendor-only claims |
| ColPali | visual retriever | none (page images) | multi-vector | n/a | storage; VLM needed downstream |

---

## Part II: Chunking strategies

### Why chunking exists and what it trades off

Chunking is needed because embedding models have input limits (for example 8,191 tokens for text-embedding-3-small) and because a single vector represents a multi-topic page poorly [34]. Every strategy trades among three things:

- **Recall.** Is the evidence in some retrieved chunk?
- **Precision and token efficiency.** How much irrelevant text comes along with it?
- **Self-containedness.** Does the chunk carry enough context to be matched and understood on its own?

Chroma's technical report made this explicit with token-level metrics. An LLM generates queries and relevant excerpts from a corpus, and retrieval is scored by token-level precision, recall and IoU (Jaccard) between retrieved tokens and relevant tokens, by analogy to bounding boxes in computer vision [7]. Its main findings are that the choice of chunking strategy has a significant impact on retrieval accuracy and efficiency, and that heuristic splitters such as RecursiveCharacterTextSplitter "often perform well in practice when parametrized appropriately" [7]. The released `chunking_evaluation` toolkit shows example results with text-embedding-3-large: fixed 1,200-character splits reach IoU 0.177 and recall 0.809, while ClusterSemanticChunker with a 400-token maximum reaches IoU 0.183 and recall 0.897 [8].

### Fixed-size (token/character) chunking

**Mechanism.** Text is cut every N tokens or characters, optionally with overlap.
**Defaults seen in the wild.** Azure AI Search recommends starting at 512 tokens with 25% overlap, or 2,000 characters with 500 characters of overlap for its Text Split skill [34]. LlamaIndex's `SentenceSplitter` example uses 1,024 tokens with 20 overlap, and its `TokenTextSplitter` example uses 512 with 128 [45].
**Failure modes.** Sentences, tables and code get cut mid-unit. Anaphora breaks across boundaries ("its", "the city"). Overlap duplicates tokens in the index and can produce near-duplicate hits.
**Evidence it is not obsolete.** Vectara's NAACL 2025 Findings study compared breakpoint-based and clustering-based semantic chunking against fixed-size chunking across document retrieval, evidence retrieval and answer generation. It concluded that "the computational costs associated with semantic chunking are not justified by consistent performance gains" [9].

### Recursive character splitting

**Mechanism.** LangChain's `RecursiveCharacterTextSplitter` tries a ranked list of separators, `["\n\n", "\n", " ", ""]` by default, and recurses into any piece still over the size limit, so paragraph and sentence boundaries are preferred over hard cuts [44]. Language-specific separator sets (Markdown, Python, JS and others) apply the same idea to code and markup, which is Kamradt's "Level 3: document-specific splitting" [43].
**Status.** It is the de facto default. Chroma's report found it competitive when parametrized appropriately [7], and LumberChunker uses it as a baseline [48].

### Structure-aware (element/heading/page) chunking

**Mechanism.** Boundaries come from the parser's structure, such as headings, elements, pages or table rows, rather than from character counts. Examples are Unstructured `by_title` [21], Docling `HierarchicalChunker` and `HybridChunker` [14], and Azure's Markdown-header splitting [33].
**Evidence.** Element-type chunking improved RAG on FinanceBench financial reports and removed the need to tune chunk size [10]. NVIDIA compared token-based (several sizes), page-level and section-level chunking on its developer blog [11]. Secondary summaries report that page-level chunking had the highest average accuracy, about 0.648, with the most consistent performance across datasets, and that 256-512-token chunks did best for factoid queries. These numbers have not been checked against the original post and should be treated as approximate [11].
**Design notes.**
- Keep tables whole when possible, and when they must be split repeat the header row in every piece (Docling `repeat_table_header`, Unstructured `repeat_table_headers`) [14][21].
- Strip furniture (running headers, footers, page numbers) before chunking, using parser roles or trees (Azure `pageHeader`/`pageFooter`, Docling `furniture`) [15][32].
- Merge tiny sections only within the same heading scope (Docling `merge_peers`) [14].

### Semantic chunking (embedding breakpoints and clustering)

**Mechanism (Kamradt, "Level 4").** Split into sentences, embed each sentence together with a buffer of neighbors, compute the cosine distance between consecutive windows, and cut where the distance exceeds a threshold [43]. LangChain's experimental `SemanticChunker` exposes four threshold types with defaults of percentile 95, standard deviation 3, interquartile 1.5 and gradient 95, and a `buffer_size` of 1 [44]. LlamaIndex's `SemanticSplitterNodeParser` ports the same idea with `breakpoint_percentile_threshold=95`. It warns that its sentence regex is primarily English and that the threshold may need tuning [45].
**Clustering variant.** Chroma's ClusterSemanticChunker uses a global optimization over embedding similarity under a maximum chunk size [7][8].
**Costs and failure modes.** Every sentence must be embedded at ingestion. Thresholds are corpus-dependent. Chunk sizes are unbounded unless capped. Benefits over recursive splitting are inconsistent [9].

### LLM-driven ("agentic") chunking

- **Kamradt's Level 5** has an LLM build chunks the way a human would, sentence by sentence, deciding whether each sentence belongs to an existing chunk, often over propositions [43].
- **LumberChunker** iteratively prompts an LLM over a group of sequential passages to find where the content begins to shift. It was evaluated on GutenQA, built from 100 public-domain books with 30 QA pairs each, against paragraph, recursive, semantic and proposition baselines [48].
- **Chonkie** packages `SlumberChunker` (LLM-based, "AgenticChunker") alongside token, sentence, recursive, semantic, late, code, neural and table chunkers, and a SIMD byte chunker it advertises at 100+ GB/s [47].
- **Chroma's LLMChunker** is evaluated in the same framework as its other chunkers [8].
**Trade-off.** These methods cost LLM calls at ingestion time and their output is nondeterministic. They are most defensible for long narrative text with no structural markup.

### Proposition-level chunking (Dense X Retrieval)

Chen et al. propose propositions, atomic self-contained factoid statements, as the retrieval unit. They release FactoidWiki, an English Wikipedia index with 257M propositions vs 114M sentences vs 41M passages, and a fine-tuned Flan-T5-large "propositionizer" that rewrites passages into propositions using the title and section as context [49]. The design gains precision and decontextualized units at the cost of index size (roughly 6x more units than passages here) and generation cost at ingestion.

### Hierarchical and multi-granularity chunking

```
              doc
      +--------+--------+
   [2048]            [2048]            parent nodes (returned to LLM)
   /  |  \           /  |  \
 [512][512][512]  [512][512][512]      intermediate
  /|\                                  
[128][128][128] ...                    leaf nodes (embedded / searched)

AutoMergingRetriever: if most children of a parent are hit -> return parent
```

- **LlamaIndex `HierarchicalNodeParser`** produces nested chunk sizes, by default `[2048, 512, 128]`, with parent links. Paired with `AutoMergingRetriever`, it replaces retrieved leaves with their parent when a majority of a parent's children are retrieved [45].
- **Sentence-window retrieval** embeds single sentences, stores a window of surrounding sentences (for example ±3) in metadata, and swaps the window in at generation time via `MetadataReplacementNodePostProcessor` [45]. This decouples the unit you match on from the unit you read.
- **RAPTOR** recursively clusters chunks, summarizes each cluster with an LLM, and builds a tree of summaries, so retrieval can hit either leaf detail or higher-level abstractions [50].
**Lesson.** Small units give precise matching and large units give enough context to answer. Separating the matched unit from the returned unit is one of the most reliable design patterns in this chapter.

### Late chunking (Jina AI)

**Problem.** Chunks embedded independently lose cross-chunk references. In Jina's Berlin example, "Its more than 3.85 million inhabitants..." no longer mentions Berlin [4].

**Mechanism.** Run the long-context embedding model's transformer over the whole document (or as much as fits, for example 8,192 tokens for jina-embeddings-v2), then mean-pool the token embeddings within each chunk's span. Chunking happens *after* the transformer and just before pooling, so every chunk embedding is conditioned on the whole document [3][4].

```
naive:  [c1]->enc->pool   [c2]->enc->pool   [c3]->enc->pool      (no cross-context)
late:   [c1 c2 c3]->enc (full attention) -> token vecs -> pool(c1) pool(c2) pool(c3)
```

**Results (Jina repo, jina-embeddings-v2-small-en, 256-token chunks, nDCG@10).**

| Dataset | Avg doc length (chars) | Naive | Late | No chunking |
|---|---|---|---|---|
| SciFact | 1498 | 64.20 | 66.10 | 63.89 |
| TREC-COVID | 1117 | 63.36 | 64.70 | 65.18 |
| FiQA2018 | 767 | 33.25 | 33.84 | 33.43 |
| NFCorpus | 1590 | 23.46 | 29.98 | 30.40 |
| Quora | 62 | 87.19 | 87.19 | 87.19 |

Late chunking beat naive chunking on every dataset except Quora, where the documents are tiny and all three methods tie. The gains grew with document length [4]. It needs no extra training and works with any long-context embedding model. The paper's later version also proposes a fine-tuning method that strengthens the effect [3]. Jina exposes it in the jina-embeddings-v3 API, and Chonkie ships a `LateChunker` [5][47].

**Failure modes.** It requires a long-context, mean-pooling embedding model and self-hosting or a supporting API. Documents longer than the window need a macro-chunking workaround. The benefit depends on the model and the dataset [6].

### Anthropic Contextual Retrieval

**Problem.** The same as for late chunking: chunks lack situating context. A sentence like "The company's revenue grew by 3% over the previous quarter" does not say which company or which quarter [1].

**Mechanism.** For each chunk, prompt an LLM (Claude 3 Haiku in the original post) with the whole document plus the chunk and ask for a short, succinct context that situates the chunk within the document for search retrieval. The 50-100-token result is prepended to the chunk before both embedding (Contextual Embeddings) and BM25 indexing (Contextual BM25) [1].

```
for doc in corpus:
   cache(doc)                                 # prompt caching: write once
   for chunk in doc.chunks:                   # process sequentially to hit cache
       ctx = LLM("<document>{doc}</document> <chunk>{chunk}</chunk> situate it")
       text = ctx + "\n" + chunk
       dense.add(embed(text)); bm25.add(text)
query -> dense top-150 + bm25 top-150 -> rank fusion -> rerank -> top-20 -> LLM
```

**Results (Anthropic, 1 - recall@20 averaged across codebases, fiction, arXiv and science papers).** The baseline failure rate was 5.7%. Contextual Embeddings brought it to 3.7% (-35%), adding Contextual BM25 brought it to 2.9% (-49%), and adding reranking (Cohere; retrieve 150 and keep 20) brought it to 1.9% (-67%) [1]. Contextualization helped every embedding model tested, and Gemini Text 004 and Voyage were the most effective. Generic document summaries prepended to chunks gave "very limited gains" [1].

**Cost.** Assuming 800-token chunks, 8k-token documents, 50 tokens of instructions and 100 tokens of context, with prompt caching, the one-time cost was $1.02 per million document tokens [1]. In the companion cookbook (737 chunks from 9 codebases, Voyage embeddings), 61.83% of input tokens were cache reads. Input cost fell from about $9.20 to about $2.85, and Pass@5/10/20 rose from 80.92/87.15/90.06% to 88.12/92.34/94.29%. Hybrid BM25 (80/20 semantic/BM25 weighted RRF over the top 150 from each) reached 88.86/92.31/95.23%, and the notebook reports that Cohere reranking adds roughly 100-200 ms per query [2].

**Guidance from the source.** If the knowledge base is under about 200k tokens (about 500 pages), skip RAG and put the whole corpus in the prompt with caching [1]. Prompts can be specialized by domain (for example, including a glossary). Keep generated context and original text distinguishable for the generator. Evaluate on your own data [1].

**Contextual Retrieval vs late chunking.** Merola and Singh (ECIR 2025 workshop) compared the two. They found that contextual retrieval preserves semantic coherence better but needs more compute, while late chunking is more efficient but tends to sacrifice relevance and completeness. Their "ContextualRankFusion" configuration did best overall on an NFCorpus subset with Jina-v3, and late chunking's results varied with the model and dataset [6].

### Chunking strategies compared

| Strategy | Boundary signal | Ingest cost | Context in chunk | Evidence | Typical failure |
|---|---|---|---|---|---|
| Fixed token/char | count | lowest | none | competitive with semantic [9] | mid-sentence and table cuts |
| Recursive | separator hierarchy | very low | none | strong when tuned [7] | ignores document structure |
| Structure/element | parser headings, elements, pages | parse cost | heading path (metadata) | element chunking helps financial RAG [10] | bad parse gives bad chunks |
| Semantic (breakpoint) | embedding distance | embed every sentence | none | inconsistent gains [9] | threshold tuning; unbounded size |
| Semantic (cluster) | global similarity optimization | embeddings | none | small IoU/recall gains [8] | compute; parameters |
| LLM/agentic | LLM judgment | LLM calls | none | narrative text [48] | cost, nondeterminism |
| Propositions | LLM rewrite to atomic facts | seq2seq or LLM | decontextualized | index 6x larger than passages [49] | index size |
| Hierarchical/auto-merge | nested sizes | low | parent on demand | widely used pattern [45] | parent bloat in context |
| Late chunking | any, pooled post-encoder | one long forward pass | implicit (attention) | +0.6 to +6.5 nDCG@10 [4] | needs long-context model |
| Contextual Retrieval | any, plus LLM context | about $1.02 per M tokens [1] | explicit text | -35% to -67% failures [1] | LLM cost; stale on doc edits |

---

## Part III: Metadata enrichment

Metadata does four jobs: it improves matching (when embedded), filtering (when stored as structured fields), citation (provenance) and expansion (pointers to parents or neighbors).

**Structural metadata from parsers.**
- Heading paths and captions are attached by Docling's chunkers and serialized into the embedded text via `contextualize()` [14].
- Page number, section and the original elements are kept in Unstructured chunk metadata (`orig_elements`) [21].
- Bounding boxes and provenance come from DoclingDocument items [15], Reducto blocks (used for citations) [28] and Azure regions [32]. MinerU 4.0 advertises "stable citation locators" for agents that continue reading by page or block [38].
- Semantic roles (title, sectionHeading, footnote, pageHeader) come from Azure Layout [32].

**LLM-generated metadata.**
- LlamaIndex's extractors are `SummaryExtractor`, `QuestionsAnsweredExtractor` (questions each node can answer), `TitleExtractor` and `EntityExtractor`. They are chained in an `IngestionPipeline` after a splitter, for example a 512/128 `TokenTextSplitter`, then `TitleExtractor(nodes=5)`, then `QuestionsAnsweredExtractor(questions=3)` [46].
- Hypothetical questions and summaries as alternative representations are Kamradt's "bonus level". They are especially useful for sparse data such as chat logs, where raw messages embed poorly [43].
- Chunk-specific situating context is Contextual Retrieval [1]. Note Anthropic's finding that generic document-level summaries help far less than chunk-specific context [1].

**Cheap heuristics.** Azure's guidance suggests simply appending the document title to chunks from the middle of a document [34]. Docling and Unstructured repeat table headers in split table chunks [14][21].

**Enrichment pipeline pattern.**

```
chunk --> +heading path, page, bbox, doc title           (free, from parser)
      --> +entities, dates, doc type, ACL tags           (NER/rules; for filtering)
      --> +situating context / questions / summary       (LLM; for matching)
      --> embed( context + heading path + chunk )        (dense)
      --> bm25 ( context + chunk )                       (sparse)
      --> store( raw chunk, metadata, parent_id, prov )  (for generation + citation)
```

**Failure modes.** Embedded metadata can dominate short chunks and pull unrelated chunks together, for example when every chunk starts with the same boilerplate title. LLM-generated context can be wrong, and Anthropic advises clearly separating generated context from source text when passing it to the generator [1]. Enrichment is also tied to the document version: any edit invalidates the context generated for every chunk in that document, which is the main operational cost of Contextual Retrieval relative to late chunking.

---

## Cross-cutting lessons and failure taxonomy

1. **Garbage in, permanent garbage out.** Parse errors (merged columns, lost tables, OCR noise) cannot be fixed by later stages. Benchmarks show order-of-magnitude differences between text-dump and layout-aware parsers on hard categories, with scores near 0 vs 80+ on arXiv math [37].
2. **Tables are their own problem.** They need structure recognition (TableFormer, VLMs), isolation in their own chunks, header repetition on split, and HTML or Markdown serialization [14][19][21][32]. Numeric hallucination by generative parsers is the main risk and is the reason Reducto routes to hybrid OCR [29].
3. **Old scans remain unsolved.** Even the best systems score about 30-51 on olmOCR-Bench's old-scans category [35].
4. **Benchmarks are vendor-shaped.** RD-TableBench is Reducto's, Marker's comparison is Datalab's, and LlamaParse's claims are LlamaIndex's. A competitor rerun found errors in 4.3% of RD-TableBench ground truth [30]. Run evaluations on your own documents.
5. **Throughput is a serving problem.** VLM parsers are bounded by decode throughput, and concurrency through a shared inference server matters more than per-page latency [37]. Costs per million pages range from under $200 self-hosted (olmOCR [35]) to about $1,000-2,000 for hosted OCR APIs (Mistral OCR 3 batch and list prices [40]) to more for agentic tiers (approximate, derived from the listed prices).
6. **Chunk-size defaults are starting points, not answers.** Published defaults include 512 tokens with 25% overlap (Azure [34]), 500 characters hard maximum (Unstructured [21]), and 800-token chunks (Anthropic's cost example [1]). Chroma's and Vectara's results both say that tuned simple splitters are hard to beat [7][9].
7. **Context injection beats boundary cleverness.** The largest published gains come from giving chunks document context, either explicitly (Contextual Retrieval, -35% to -67% failures [1]) or implicitly (late chunking [3][4]), rather than from finding better boundaries (semantic chunking's gains are inconsistent [9]).

---

## Key takeaways for system designers

- **Pick the parser by document mix, not by leaderboard.** For born-digital text with a CPU budget, use text-layer plus layout models (Docling, Marker fast, Unstructured fast or hi_res). For scans, math and complex tables, use VLM parsers (olmOCR, MinerU VLM, Granite-Docling, Mistral OCR) or hybrids (Reducto, LlamaParse agentic, Marker with an LLM). Route per page by difficulty when the budget allows [24][29][37].
- **Keep a structured intermediate representation** (DoclingDocument, Unstructured elements, Azure JSON) with roles, hierarchy and bounding boxes. Generate Markdown from it, not instead of it [15][21][32].
- **Default to structure-aware chunking with a token cap**: heading or element boundaries, a tokenizer-aware maximum that matches your embedding model, merging only within the same heading scope, isolated tables with repeated headers, and furniture removed [14][21].
- **Treat semantic and agentic chunking as experiments, not defaults.** Their extra ingestion cost is not consistently repaid [9]. LLM chunking is most defensible for long narrative text without markup [48].
- **Separate the matched unit from the returned unit** with sentence windows, parent-child auto-merging or RAPTOR-style summaries [45][50].
- **Add document context to every chunk.** If you can afford LLM calls at ingestion, use Contextual Retrieval with prompt caching (about $1.02 per million document tokens), hybrid BM25 and reranking [1][2]. If you self-host a long-context embedder, late chunking is nearly free [3][4]. Merola and Singh suggest the explicit variant wins on quality and the implicit one on cost [6].
- **Always run hybrid lexical plus dense retrieval over the contextualized text.** Contextual BM25 took the relative failure reduction from 35% to 49% on top of contextual embeddings in Anthropic's study [1].
- **Store provenance (page, bounding box, element ids)** for citations and for agentic "read more" navigation [15][28][38].
- **Check whether you need RAG at all.** Below about 200k tokens, put the whole corpus in the context with caching [1].
- **Build a chunking evaluation harness** (token-level IoU and recall as in Chroma [7][8], Pass@k as in Anthropic's cookbook [2]) and rerun it whenever the parser, chunker or embedding model changes.

---

## References

1. "Introducing Contextual Retrieval." Anthropic, 2024. https://www.anthropic.com/news/contextual-retrieval
2. "Enhancing RAG with Contextual Retrieval" (cookbook notebook). Anthropic, 2024-2025. https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb
3. Günther, M., Mohr, I., Williams, D. J., Wang, B., Xiao, H. "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." Jina AI, arXiv:2409.04701, 2024. https://arxiv.org/abs/2409.04701
4. "Late Chunking of Short Chunks in Long-Context Embedding Models" (code and results). Jina AI, GitHub, 2024. https://github.com/jina-ai/late-chunking
5. "Late Chunking in Long-Context Embedding Models." Jina AI blog, 2024. https://jina.ai/news/late-chunking-in-long-context-embedding-models/
6. Merola, C., Singh, J. "Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation." arXiv:2504.19754 (ECIR 2025 KEIR workshop), 2025. https://arxiv.org/abs/2504.19754
7. Smith, B., Troynikov, A. "Evaluating Chunking Strategies for Retrieval." Chroma Technical Report, 2024. https://research.trychroma.com/evaluating-chunking
8. "chunking_evaluation" (ClusterSemanticChunker, LLMChunker, evaluation framework). Chroma / B. Smith, GitHub, 2024. https://github.com/brandonstarxel/chunking_evaluation
9. Qu, R., Bao, F., Tu, R. "Is Semantic Chunking Worth the Computational Cost?" Vectara, arXiv:2410.13070 (NAACL Findings 2025), 2024. https://arxiv.org/abs/2410.13070
10. Jimeno Yepes, A., et al. "Financial Report Chunking for Effective Retrieval Augmented Generation." Unstructured Technologies, arXiv:2402.05131, 2024. https://arxiv.org/abs/2402.05131
11. "Finding the Best Chunking Strategy for Accurate AI Responses." NVIDIA Technical Blog, c. 2025 (figures quoted via secondary summaries; approximate). https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/
12. Auer, C., Livathinos, N., et al. "Docling Technical Report." IBM Research, arXiv:2408.09869, 2024. https://arxiv.org/abs/2408.09869
13. "Docling" (repository README). Docling Project / IBM, GitHub, 2024-2026. https://github.com/docling-project/docling
14. "Chunking" (Docling concepts documentation). Docling Project, 2025-2026. https://github.com/docling-project/docling/blob/main/docs/concepts/chunking.md
15. "Docling Document" (DoclingDocument concepts documentation). Docling Project, 2024-2026. https://github.com/docling-project/docling/blob/main/docs/concepts/docling_document.md
16. "Model Catalog" and "Vision Models" (Docling usage docs). Docling Project, 2025-2026. https://github.com/docling-project/docling/blob/main/docs/usage/model_catalog.md
17. "Advanced Layout Analysis Models for Docling." IBM Research, arXiv:2509.11720, 2025. https://arxiv.org/abs/2509.11720
18. "IBM Granite-Docling: End-to-end document understanding with one tiny model." IBM, 2025. https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion
19. "Docling IBM models" (TableFormer). Docling Project / IBM, GitHub. https://github.com/docling-project/docling-ibm-models
20. "Unstructured" (repository README, partition strategies, benchmark table). Unstructured-IO, GitHub, 2023-2026. https://github.com/Unstructured-IO/unstructured
21. "chunking/title.py" and "chunking/base.py" (chunk_by_title implementation and options). Unstructured-IO, GitHub, 2024-2026. https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/chunking/title.py
22. "Chunking" (API reference). Unstructured documentation. https://docs.unstructured.io/api-reference/partition/chunking
23. "unstructured-inference models/base.py" (default YOLOX layout model). Unstructured-IO, GitHub. https://github.com/Unstructured-IO/unstructured-inference/blob/main/unstructured_inference/models/base.py
24. "LlamaParse Pricing." LlamaIndex developer documentation, 2025-2026. https://developers.llamaindex.ai/llamaparse/general/pricing/
25. "Introducing LlamaParse v2: Simpler, Better, Cheaper." LlamaIndex blog, 2025. https://www.llamaindex.ai/blog/introducing-llamaparse-v2-simpler-better-cheaper
26. "llama_parse / Llama Cloud Services" (deprecation notice). LlamaIndex, GitHub, 2026. https://github.com/run-llama/llama_parse
27. "RD-TableBench." Reducto blog, 2024. https://reducto.ai/blog/rd-tablebench
28. "How to use Reducto parsing with Elasticsearch for semantic search." Reducto blog, 2025. https://reducto.ai/blog/how-to-reducto-parsing-elasticsearch-semantic-search
29. "Production-scale document parsing with vision language models and specialized OCR" (Reducto case study). ZenML LLMOps Database, 2025. https://www.zenml.io/llmops-database/production-scale-document-parsing-with-vision-language-models-and-specialized-ocr
30. "RD-TableBench: Updated Results." Pulse, 2026. https://www.runpulse.com/blog/rd-tablebench-updated-results
31. News report (10 February 2025) on Sergey Filimonov's post about ingesting millions of PDFs with Gemini 2.0 Flash. Gigazine, 2025. https://www.gigazine.net/gsc_news/en/20250210-ingesting-pdf-gemini-2-0
32. "Document Intelligence layout model." Microsoft Learn, 2024-2026. https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout
33. "Retrieval-augmented generation with Azure AI Document Intelligence." Microsoft Learn, 2024-2026. https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept/retrieval-augmented-generation
34. "Chunk large documents for RAG and vector search in Azure AI Search." Microsoft Learn, updated 2026. https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents
35. "olmOCR" (repository README, olmOCR-Bench results, release notes). Allen Institute for AI, GitHub, 2025. https://github.com/allenai/olmocr
36. "olmOCR: Unlocking trillions of tokens in PDFs." Allen Institute for AI blog, 2025. https://allenai.org/blog/olmocr
37. "Marker" (repository README, olmOCR-Bench comparison and throughput). Datalab, GitHub, 2024-2026. https://github.com/datalab-to/marker
38. "MinerU" (repository README, MinerU 4.0). OpenDataLab, GitHub, 2024-2026. https://github.com/opendatalab/MinerU
39. Ouyang, L., et al. "OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations." CVPR 2025; repository and leaderboard. https://openaccess.thecvf.com/content/CVPR2025/html/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.html and https://github.com/opendatalab/OmniDocBench
40. "Mistral OCR 3 promises better, cheaper document analysis." The Decoder, 2025; and "Mistral OCR 3." InfoQ, 2026. https://the-decoder.com/mistral-ocr-3-promises-better-cheaper-document-analysis/ and https://www.infoq.com/news/2026/01/mistral-ocr3
41. Faysse, M., et al. "ColPali: Efficient Document Retrieval with Vision Language Models" (repository and paper arXiv:2407.01449). Illuin Technology, 2024-2026. https://github.com/illuin-tech/colpali
42. "NeMo Retriever Library (nv-ingest)." NVIDIA, GitHub, 2024-2026. https://github.com/NVIDIA/nv-ingest
43. Kamradt, G. "5 Levels of Text Splitting." FullStackRetrieval, 2024. https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
44. "langchain_text_splitters/character.py" (RecursiveCharacterTextSplitter) and "langchain_experimental/text_splitter.py" (SemanticChunker). LangChain, GitHub. https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py and https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py
45. "Node Parser Modules." LlamaIndex documentation. https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/loading/node_parsers/modules.md
46. "Metadata Extraction Usage Pattern." LlamaIndex documentation. https://github.com/run-llama/llama_index/blob/main/docs/src/content/docs/framework/module_guides/loading/documents_and_nodes/usage_metadata_extractor.md
47. "Chonkie" (chunking library README). Chonkie Inc., GitHub, 2024-2026. https://github.com/chonkie-inc/chonkie
48. Duarte, A. V., Marques, J. D. S., Graça, M., Freire, M., Li, L., Oliveira, A. L. "LumberChunker: Long-Form Narrative Document Segmentation." EMNLP Findings 2024; repository. https://github.com/joaodsmarques/LumberChunker
49. Chen, T., et al. "Dense X Retrieval: What Retrieval Granularity Should We Use?" arXiv:2312.06648, 2023; FactoidWiki repository. https://github.com/chentong00/factoid-wiki
50. Sarthi, P., et al. "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." arXiv:2401.18059 (ICLR 2024); repository. https://github.com/parthsarthi03/raptor
