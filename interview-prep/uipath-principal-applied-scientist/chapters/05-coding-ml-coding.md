# 05. Coding and ML Coding for System Design Components: Building the Pieces of Document AI, Agent and Retrieval Systems

> **How to read this chapter.** You asked for ML system design, so this chapter does not cover general LeetCode-style coding. It covers coding and ML coding questions where you implement one part of an ML system that you could be asked to design elsewhere in the loop: document and PDF processing, extraction and tagging, retrieval, agent tool loops, evaluation, and serving. Each section opens with a plain-text block diagram that shows which box of the system you are coding. Diagrams are plain text, so they render the same in Markdown and in a PDF export.
>
> **Tagging system.** Every question carries one of two tags:
> - **[Reported] [n]**: a community-maintained list of LeetCode problems tagged to UiPath links this problem to UiPath interviews. These lists [3][4][5][6] are compiled from LeetCode's company tags. They do not name the role or level, and only one of them [4] has a date window (the last six months before its snapshot). Treat them as evidence that UiPath interviews use the problem, not that a Principal Applied Scientist was asked it. The sub-bullets under each [Reported] problem are [Likely] follow-up probes that connect the algorithm to a box in the system design diagram.
> - **[Likely]**: an inference from the role, the level and UiPath's AI products (Document Understanding, Communications Mining, agentic automation, Autopilot, the AI Trust Layer, computer vision for UI automation), or from UiPath's public open-source agent SDKs [7]-[13]. No candidate has reported it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **Research limits (please read).** The shared web-search budget for this workflow was used up before this review, and DNS lookups failed for glassdoor.com, leetcode.com, uipath.com, docs.uipath.com, reddit.com, teamblind.com, ambitionbox.com, geeksforgeeks.org, interviewquery.com and arxiv.org. The only sources that could be fetched were raw files on GitHub and PyPI. So the [Reported] tags rest only on the UiPath-tagged LeetCode lists. No candidate write-up of a UiPath ML coding round could be read, and no ML-specific question below is tagged [Reported]. The format notes are inferred, not confirmed. Ask your recruiter whether the coding round is general algorithms, ML coding or both, and which language and environment to expect.

## Format notes (inferred, not confirmed)

- **Duration:** likely 45-60 minutes, in a shared editor or notebook, usually Python. There may be no execution environment, so you should be able to dry-run code by hand.
- **Interviewer:** likely a senior or staff scientist or ML engineer, not a principal.
- **What is assessed at principal level:** code that is correct and readable, without needing to be the fastest solution. The emphasis likely shifts toward whether you can turn a design box into a clean interface, pick sensible data structures, write tests and edge cases without being prompted, and explain how the component behaves at scale and how it fails. Expect the interviewer to connect the code to a system ("now this runs on 10 million pages a day").
- **Two common variants:** (a) implement an ML primitive from scratch (attention, a metric, a decoding loop); (b) implement a pipeline component with a small amount of ML logic (chunker, matcher, tagger, evaluator).
- **A third variant with some evidence:** a standard algorithm problem. The UiPath-tagged LeetCode lists [3][4][5][6] lean towards graph search (BFS, union-find), trees and dynamic programming at medium and hard difficulty. At principal level, expect the interviewer to ask how the algorithm maps to a real component (Section L).

### How the coding round connects to the system design diagram

```
   ML SYSTEM DESIGN ROUND                         CODING / ML CODING ROUND
   (whiteboard, 60 min)                           (editor, 45-60 min)

   ┌──────────┐  ┌──────────┐  ┌──────────┐       "Implement this box"
   │ Ingest & │─>│ Layout / │─>│ Extract/ │──┐        │
   │ OCR (A)  │  │ Tag (B)  │  │ Classify │  │        ├── A: PDF/OCR token handling, reading order
   └──────────┘  └──────────┘  │   (C)    │  │        ├── B: tagging, span/BIO decoding, tables
                               └──────────┘  │        ├── C: classifier, entity matching, validation
   ┌──────────┐  ┌──────────┐  ┌──────────┐  │        ├── D: chunking, BM25, vector top-k, rerank
   │ Retrieve │<─│ Agent /  │<─│  Human   │<─┘        ├── E: agent loop, tool schema, retries
   │   (D)    │  │ LLM (E)  │  │ review(F)│           ├── F: confidence routing, active learning
   └──────────┘  └──────────┘  └──────────┘           ├── G: metrics and evaluation harness
         └────────── Eval (G) ── Serve (H) ──┘        └── H: batching, caching, rate limits
```

## Questions

### A. PDF and document ingestion: tokens, boxes and reading order

```
   PDF / scan ──> [ text layer? ] ──yes──> native tokens (text, bbox, font)
                       │ no
                       └──> OCR ──> words + bbox + confidence
                                     │
                                     v
                        [ YOUR CODE: normalise, sort, group into lines/blocks ]
                                     │
                                     v
                        reading-ordered tokens ──> tagger / extractor
```

1. [Likely] Given a list of OCR words with bounding boxes `(x0, y0, x1, y1)` and text, write a function that groups them into lines and returns the lines in reading order.
   - How do you handle slight skew, where words on the same line have different y values?
   - What changes for a two-column page?
2. [Likely] Write a function that detects whether a page has two or more text columns from word bounding boxes alone.
   - How would you test it without labelled data?
3. [Likely] Implement intersection over union for two bounding boxes, then a function that merges overlapping boxes from two OCR engines into one set of words.
   - How do you decide which engine's text wins when the boxes overlap but the text differs?
4. [Likely] Given native PDF text tokens and OCR tokens for the same page, align them so each OCR word maps to its native token or to nothing.
   - What is the complexity of your alignment, and how would you make it faster for dense pages?
5. [Likely] Write a function that normalises bounding boxes from different page sizes and rotations (0, 90, 180, 270 degrees) into one coordinate system.
6. [Likely] Implement a de-hyphenation and line-joining step that rebuilds paragraphs from line-broken OCR text.
   - Which cases would wrongly join two words, and how would you guard against them?
7. [Likely] Write a streaming page iterator for a multi-thousand-page PDF batch that never holds more than N pages in memory.
   - How do you resume after a crash halfway through a file?
8. [Likely] Given per-word OCR confidences, compute a page-level quality score and decide whether to re-OCR the page with a slower engine.
   - What threshold policy would you choose, and how would you tune it?

### B. Tagging and structure: spans, key-value pairs and tables

```
   reading-ordered tokens ──> token classifier ──> per-token labels (B-/I-/O)
                                                        │
                                  [ YOUR CODE: decode labels -> spans/fields ]
                                                        │
                    ┌───────────────────────────────────┼───────────────────┐
                    v                                   v                   v
              key-value pairs                     table cells          document tags
```

9. [Likely] Given tokens and BIO labels, write a function that decodes them into typed entity spans with start and end offsets.
   - What do you do with an I- label that follows O or a different entity type?
10. [Likely] Implement Viterbi decoding for a linear-chain CRF given emission and transition score matrices.
    - How would you add a constraint that forbids illegal BIO transitions?
11. [Likely] Write span-level precision, recall and F1 for entity extraction, with exact match and with partial (overlap) match.
    - Which one would you report to a product team, and why?
12. [Likely] Given word boxes for a page, write a function that links each detected key ("Invoice No.", "Date") to its most likely value using spatial rules (right of, below).
    - How would you replace the rules with a learned scorer while keeping the same interface?
13. [Likely] Given detected horizontal and vertical ruling lines and word boxes, reconstruct a table as a 2D grid of cell strings.
    - How do you handle merged cells and tables with no ruling lines?
14. [Likely] Write a function that matches table header cells to a target schema (for example "Qty", "Quantity", "Units") using fuzzy matching.
    - Where would an embedding model beat string similarity here, and where would it be worse?
15. [Likely] Implement a multi-label document tagger head on top of a pooled encoder output in PyTorch, including the loss.
    - How would you pick a decision threshold per tag?
16. [Likely] Given a hierarchy of document tags (for example Finance > Invoice > Utility invoice), write a function that makes predictions consistent with the hierarchy.
17. [Likely] Write a function that splits a multi-document PDF (several invoices scanned together) into separate documents from per-page classifier scores.
    - How would you formulate this as a sequence problem instead of page-by-page thresholds?

### C. Extraction, normalisation and validation

18. [Likely] Write parsers that normalise extracted dates, currency amounts and numbers across locales ("1.234,56", "1,234.56", "12/03/2025").
    - How do you resolve an ambiguous date like 03/04/2025 using other fields on the same document?
19. [Likely] Implement a validator that checks invoice line items sum to the subtotal, and subtotal plus tax equals the total, within a tolerance, and reports which field is most likely wrong.
20. [Likely] Given extracted vendor names and a master vendor list of 1 million entries, write a matcher that returns the top 5 candidates quickly.
    - Which index would you build: n-gram, phonetic, or embedding? Why?
21. [Likely] Write a function that turns LLM output into a strict JSON object for a given schema, with repair for common errors (trailing commas, missing quotes, extra text).
    - When would you prefer constrained decoding over repairing output after generation?
22. [Likely] Implement a field-level confidence aggregator that combines OCR confidence, model probability and validation-rule results into one score.
    - How would you check that the combined score is calibrated?
23. [Likely] Write a regex-plus-model hybrid extractor for IBANs or tax IDs, including checksum validation.
24. [Likely] Given extraction results from three models for the same document, write an ensembling function that votes per field.
    - How do you break ties, and how do you log disagreement for later review?

### D. Retrieval and RAG components

```
   documents ──> [ YOUR CODE: chunk ] ──> embed ──> vector index
                                                        │
   query ──> embed ──> [ YOUR CODE: top-k + BM25 + fusion ] ──> [ rerank ] ──> LLM context
```

25. [Likely] Implement a chunker that splits a document into overlapping chunks by token count while not breaking sentences, and that keeps page numbers and section titles as metadata.
    - How would you chunk a long table?
26. [Likely] Implement BM25 scoring from scratch over a small corpus.
    - How do you update document frequencies when documents are added and deleted?
27. [Likely] Implement exact top-k cosine similarity search with NumPy over a matrix of embeddings, and then explain how you would scale it.
    - Write the version that processes queries in batches without running out of memory.
28. [Likely] Implement reciprocal rank fusion to merge BM25 and dense retrieval results.
29. [Likely] Implement maximal marginal relevance (MMR) to pick diverse chunks for an LLM context window under a token budget.
30. [Likely] Write recall@k, MRR and nDCG@k for a retrieval evaluation set.
    - How would you build the relevance labels for enterprise documents you cannot read?
31. [Likely] Implement a simple inverted file (IVF) index: k-means over embeddings, assign vectors to lists, search the nearest n lists.
    - How does recall change with the number of lists probed?
32. [Likely] Write a permission-aware retrieval filter so a user only gets chunks from documents they can access, applied before ranking.
    - Why is filtering after ranking a problem?
33. [Likely] Implement a deduplication step for near-duplicate chunks using MinHash or SimHash.

### E. LLM and agent components

```
   user goal ──> [ YOUR CODE: agent loop ] ──> LLM ──> tool call (JSON)
                       ^                                    │
                       │                  [ YOUR CODE: validate, execute, retry ]
                       └──────── observation <──────────────┘
                                         │
                            stop: answer / max steps / escalate to human
```

34. [Likely] Write a minimal agent loop: call a model, parse a tool call, run the tool, append the observation, and stop on a final answer or a step limit.
    - Where do you enforce budgets for tokens, time and money?
35. [Likely] Implement a tool registry that builds a JSON schema from Python function signatures and validates model arguments against it.
36. [Likely] Write retry logic for tool calls with exponential backoff and jitter, and separate retryable from non-retryable errors.
    - How do you avoid running a non-idempotent action, such as sending a payment, twice?
37. [Likely] Implement greedy, temperature, top-k and top-p (nucleus) sampling from a logits vector.
    - What breaks numerically with very low temperature, and how do you fix it?
38. [Likely] Implement beam search for a toy sequence model.
39. [Likely] Write a token-budget manager that fits a system prompt, retrieved chunks and conversation history into a context limit, dropping or summarising the least useful parts.
40. [Likely] Implement a simple planner that turns a task into a DAG of steps and runs independent steps in parallel.
    - How do you handle one failing branch?
41. [Likely] Write a PII redaction step (emails, phone numbers, card numbers, names) that runs before text is sent to an external LLM, and that can restore the values in the response.
    - How do you measure its recall?
42. [Likely] Implement a prompt-injection heuristic filter for content read from documents and emails before it reaches an agent.
    - Why is a filter alone not enough, and what structural defence would you code instead?
43. [Likely] Write a semantic cache for LLM calls keyed by embedding similarity, with a threshold and time-to-live.
    - What is the risk of serving a cached answer for a slightly different query?

### F. Human-in-the-loop, confidence routing and active learning

```
   prediction + confidence ──> [ YOUR CODE: router ] ──> auto-accept (straight-through)
                                       │
                                       └──> human validation queue ──> corrected labels
                                                                              │
                                    [ YOUR CODE: sample for retraining ] <────┘
```

44. [Likely] Write a router that sends each document to straight-through processing or human review based on field confidences and a target precision.
    - How would you choose the thresholds from a labelled validation set?
45. [Likely] Implement temperature scaling for calibration and compute expected calibration error (ECE).
46. [Likely] Implement uncertainty sampling (least confidence, margin, entropy) to choose which documents a human should label next.
    - How do you avoid choosing many near-identical documents?
47. [Likely] Write a function that merges human corrections into a training set while tracking label provenance and annotator disagreement.
48. [Likely] Compute Cohen's kappa for two annotators on a tagging task.
49. [Likely] Given a stream of human corrections, write a drift detector that raises an alert when the correction rate for a field moves beyond normal variation.

### G. Evaluation harnesses and metrics

50. [Likely] Write a bootstrap confidence interval for the F1 difference between two extraction models on the same test set.
    - Why should you resample documents rather than fields?
51. [Likely] Implement a field-level evaluation for document extraction that handles missing fields, extra fields and multi-valued fields such as line items.
52. [Likely] Implement character error rate and word error rate for OCR output using edit distance.
53. [Likely] Write an LLM-as-judge harness: run a judge prompt over model outputs, parse a score, and measure agreement with a small human-labelled set.
    - How would you detect position bias or length bias in the judge?
54. [Likely] Implement a task-success evaluator for agent trajectories that checks final state rather than the exact sequence of actions.
55. [Likely] Write a stratified sampler for an evaluation set across document types, languages and vendors with a fixed labelling budget.
56. [Likely] Implement a paired sign test or McNemar's test to compare two classifiers on the same examples.

### H. ML primitives from scratch

57. [Likely] Implement scaled dot-product attention and multi-head attention in NumPy or PyTorch, including a causal mask.
    - What is the memory cost in sequence length, and how does it affect long documents?
    - Make the softmax numerically stable and fuse it with a cross-entropy loss; what overflows first in half precision?
58. [Likely] Implement 2D sinusoidal or learned position embeddings for tokens with (x, y) layout coordinates, as in layout-aware document models.
    - How would you add bounding box information to a text transformer's input?
59. [Likely] Implement a LoRA adapter layer that wraps a linear layer, with merge and unmerge methods.
60. [Likely] Implement byte-pair encoding training on a small corpus and encoding of new text.
61. [Likely] Implement a contrastive (InfoNCE) loss for training a retrieval embedding model with in-batch negatives.
    - What goes wrong with in-batch negatives when the batch has duplicates?
62. [Likely] Implement non-maximum suppression for UI element or layout region detection.
63. [Likely] Write a training loop in PyTorch with gradient accumulation, mixed precision and checkpointing.

### I. Screen and UI understanding components

64. [Likely] Given a list of detected UI elements (type, box, text), write a function that finds the element matching a natural-language target such as "the Submit button under Billing".
    - How would you make it robust to small layout changes between application versions?
65. [Likely] Write a function that builds a containment tree from flat UI element boxes.
66. [Likely] Implement a selector-repair function: when a stored selector no longer matches, find the most similar element on the new screen using attributes and position.
67. [Likely] Given two screenshots' element lists, compute a diff of added, removed and moved elements.

### J. Serving, scaling and production code

```
   requests ──> [ YOUR CODE: queue + dynamic batcher ] ──> model worker(s)
                         │                                       │
                   rate limiter                        [ YOUR CODE: cache ]
                         │                                       │
                   tenant isolation ───────── metrics / tracing ─┘
```

68. [Likely] Implement a dynamic batcher that groups incoming inference requests up to a maximum batch size or maximum wait time.
    - How do you choose the wait time against a latency SLO?
69. [Likely] Implement a token-bucket rate limiter per tenant for calls to an external LLM provider.
70. [Likely] Implement an LRU cache with a size limit in bytes, then make it safe for concurrent use.
71. [Likely] Write an async client that sends many document pages to a model endpoint with bounded concurrency and collects results in order.
72. [Likely] Write a fallback router that sends a request to a smaller model first and escalates to a larger model when confidence is low.
    - How would you measure whether the router saves money without hurting quality?
73. [Likely] Write a function that shards a large batch of documents across workers so that each worker gets a similar total page count.
74. [Likely] Write structured logging for a model call that records inputs, outputs, latency and model version without logging customer PII.

### K. Code review and principal-level judgement in the coding round

75. [Likely] Here is a 60-line extraction pipeline written by a junior scientist. Review it: what is wrong, what is risky, and what would you change first?
    - Which of your comments block the merge and which are suggestions?
76. [Likely] Turn the notebook prototype you just wrote into a module others can use: what interface, tests and configuration would you add?
77. [Likely] Write unit tests for your reading-order or BIO-decoding function, including edge cases.
78. [Likely] Your code works on the sample. The interviewer says it now runs on 10 million pages a day across 500 tenants. What do you change in the code, not only in the architecture?

### L. UiPath-tagged algorithm problems, recast as system design components

```
   UiPath-tagged problem (graph / tree / DP)        Box in the ML system it maps to
   ─────────────────────────────────────────        ───────────────────────────────
   BFS / multi-source BFS  ───────────────────────> layout grids, UI state exploration (A, I)
   Union-find / connected components ─────────────> region merging, entity resolution (A, C)
   Sliding window / two pointers ─────────────────> OCR token streams, request batching (A, J)
   Tree DP / tree pruning ────────────────────────> UI element trees, agent plan trees (E, I)
   Interval DP / LIS ─────────────────────────────> document splitting, box nesting (B, I)
```

79. [Reported] [3][5][6] Escape a Large Maze (LeetCode 1036): decide whether a target cell can be reached on a 10^6 x 10^6 grid with up to 200 blocked cells.
   - Why does a BFS bounded by the number of blocked cells give the right answer, and what is the bound?
   - An agent explores a large application state graph to reach a target screen. How would you bound its exploration in the same way, and when does the bound stop being valid?
80. [Reported] [3][5] Sorting Three Groups (LeetCode 2826): find the minimum number of removals that make an array of values in {1, 2, 3} non-decreasing.
   - Recast it: per-page labels must follow the order cover page, body, appendix. Find the minimum number of relabels, and compare it with constrained Viterbi decoding.
81. [Reported] [3][5][6] As Far from Land as Possible (LeetCode 1162): return the largest distance from any water cell to the nearest land cell.
   - Use a multi-source BFS distance transform on a rasterised page to find column gutters and white-space separators. How do you handle scan noise?
82. [Reported] [3][5] Integer to Roman (LeetCode 12), then the reverse parser.
   - Write a robust parser for clause numbering in contracts ("IV.2(b)", "4.2.b", "Section iv") that returns a hierarchical section key. How do you test it on OCR errors such as "IV" read as "1V"?
83. [Reported] [3][5] Making A Large Island (LeetCode 827): find the largest island possible after changing at most one 0 to 1.
   - Map it to connected-component labelling of text regions where one small gap splits a paragraph. How do you find the merge that helps most without merging two columns by mistake?
84. [Reported] [3][5] Boats to Save People (LeetCode 881): use the fewest boats, each with a weight limit and at most two people.
   - Generalise it to packing documents of different token lengths into GPU batches under a memory limit. Why does the greedy two-pointer answer stop being optimal when a batch can hold more than two items?
85. [Reported] [3][5] Satisfiability of Equality Equations (LeetCode 990): decide whether a set of "==" and "!=" constraints can all hold.
   - Turn it into vendor entity resolution with must-link and cannot-link constraints from human reviewers. What do you return when the constraints conflict, and to whom?
86. [Reported] [3][5] Longest Repeating Character Replacement (LeetCode 424): find the longest substring that becomes one repeated character after at most k replacements.
   - On an OCR token stream with per-token confidence, find the longest span with at most k low-confidence tokens, as a candidate for one human review task.
87. [Reported] [3][5] Collect Coins in a Tree (LeetCode 2603): find the minimum number of edge traversals that collect all coins within distance 2 and return to the start.
   - Prune an accessibility or DOM tree of thousands of UI elements to the smallest subtree that keeps every element an agent may need, before it goes into an LLM prompt.
88. [Reported] [3][5] Burst Balloons (LeetCode 312): find the maximum coins from bursting balloons in the best order, using interval DP.
    - Use interval DP to split a scanned batch into documents so that the total segment score is highest. What is the complexity, and how would you reduce it for a 2,000-page batch?
89. [Reported] [3][4][5] Min Stack (LeetCode 155): design a stack that returns the minimum element in O(1) time. It is one of only two problems in the six-month window of [4], and the only problem in the six-month list in the same repository as [5].
    - Extend it into an agent checkpoint stack that supports undo and returns the lowest-cost recoverable state in O(1) time. What must be persisted for a long-running agent that resumes after a human approval?
90. [Reported] [3][5] Subsets II (LeetCode 90): return all subsets of a list that contains duplicates, without duplicate subsets.
    - Use it to enumerate tool-set or prompt-component ablations where some components are interchangeable. How do you cut the search when 2^n runs are too expensive to evaluate?
91. [Reported] [3][5] Reorder List (LeetCode 143): reorder a linked list in place to L0, Ln, L1, Ln-1, and so on.
    - Write an in-place interleaving of results from two ranked retrievers that removes duplicates and keeps the reordering stable. When does round-robin interleaving beat score fusion?
92. [Reported] [3][5] Russian Doll Envelopes (LeetCode 354): find the longest chain of envelopes that nest inside each other, in O(n log n) time.
    - Apply it to bounding boxes: find the deepest chain of nested layout regions (page, section, table, cell). Why does the trick of sorting by width ascending and height descending matter?
93. [Reported] [3][4] Binary Tree Cameras (LeetCode 968): find the minimum number of cameras that monitor every node of a binary tree. This problem is in the six-month window of [4].
    - Recast it: choose the fewest human-approval checkpoints in an agent's plan tree so that every irreversible action is covered by an approval at that step or an adjacent one.

### M. Agent runtime components on UiPath's public SDK contracts

```
   uipath.json / langgraph.json ──> [ YOUR CODE: entry points + JSON schema ]
                                              │
   input ──> Runtime.execute() / stream() ──> events (state, message) ──> result
                 │            │                                   │
          tools / MCP   context grounding (search)        structured error contract
                 │            │                                   │
          HITL: suspend ──> Action Center task ──> resume     traces (OpenTelemetry)
```

94. [Likely] [9] Implement a class that satisfies a runtime protocol with `get_schema`, `execute`, `stream` and `dispose`, for an agent that extracts fields from an invoice PDF. `stream` should yield state events before the final result.
    - What must stay true between `execute` and the last event of `stream`, and how would you test it?
95. [Likely] [9] Write a runtime factory that finds entry points from a config file, creates runtimes lazily, and reuses expensive resources (model clients, indexes) across executions without leaking state between tenants.
96. [Likely] [9] Implement a structured error model with a code, a title, a detail and a category (user or system), and map tool, model and validation exceptions onto it.
    - Which errors should an orchestrator retry, and which should go to a human?
97. [Likely] [8][11] Implement durable suspend and resume for a human-in-the-loop step: the agent creates a review task, saves its state, exits, and continues from the same point when the reviewer answers.
    - How do you make the resume idempotent if the callback arrives twice, or after the agent code has changed?
98. [Likely] [7] Write a wrapper around a context-grounding search call (`index name`, `query`, `number of results`) that adds query rewriting, deduplication and a citation check before chunks reach the LLM.
    - How would you test the wrapper offline without access to the real index?
99. [Likely] [10] Expose two document-processing functions (classify and extract) as tools on a small MCP server, with input and output schemas, timeouts and size limits for uploaded PDFs.
    - How does the agent side tell a tool failure apart from a tool result that says "no fields found"?
100. [Likely] [9][12] Wrap any runtime so that each execution's spans and logs are isolated when several runtimes run in the same process. Record model latency, token counts and tool calls as OpenTelemetry-style spans.
    - What breaks if you use a global logger, and how do you show the bug in a test?
101. [Likely] [7] Write a transaction-queue worker that takes documents from a queue, runs extraction, and marks each item as successful, failed with a retryable error, or failed with a business exception.
    - How do you avoid processing an item twice when a worker dies after the model call but before the status update?
102. [Likely] [7][9] Write a function that reads Python entry-point functions with type hints or Pydantic models and creates the input and output JSON schemas that a packaging step would need.
    - Which Python types have no clean JSON Schema equivalent, and what do you do with them?
103. [Likely] [11] Write an adapter layer so that one evaluation harness can run agents built with two different frameworks (for example LangGraph and LlamaIndex) behind one runtime interface.
    - What behaviour leaks through the adapter (streaming, interrupts, tool-call formats), and how do you keep the evaluation fair?

## References

Sources marked "fetched" were read in this review (October 2026). Sources marked "not retrieved" are official pages that could not be reached because DNS lookups failed. They are listed only for product context and support no question.

1. UiPath Careers, uipath.com (official careers page; not retrieved). https://www.uipath.com/company/careers
2. UiPath Blog, uipath.com (official product and AI announcements; not retrieved). https://www.uipath.com/blog
3. liquidslr, "leetcode-company-wise-problems", UiPath "All" list (15 problems with LeetCode frequency scores; fetched). https://raw.githubusercontent.com/liquidslr/leetcode-company-wise-problems/main/UiPath/5.%20All.csv
4. liquidslr, "leetcode-company-wise-problems", UiPath "Six Months" list (Min Stack and Binary Tree Cameras; the 30-day and 3-month lists were empty; fetched). https://raw.githubusercontent.com/liquidslr/leetcode-company-wise-problems/main/UiPath/3.%20Six%20Months.csv
5. snehasishroy, "leetcode-companywise-interview-questions", UiPath all-time list (14 problems; fetched). https://raw.githubusercontent.com/snehasishroy/leetcode-companywise-interview-questions/master/uipath/all.csv
6. krishnadey30, "LeetCode-Questions-CompanyWise", UiPath all-time list (older snapshot: Escape a Large Maze, As Far from Land as Possible; fetched). https://raw.githubusercontent.com/krishnadey30/LeetCode-Questions-CompanyWise/master/uipath_alltime.csv
7. UiPath, "uipath-python" README: Python SDK and CLI, with services for processes, assets, buckets, context grounding, jobs, queues and Action Center tasks (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-python/main/README.md
8. UiPath, "uipath-langchain-python" README: LangGraph agents on the UiPath platform, with human-in-the-loop through Action Center (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-langchain-python/main/README.md
9. UiPath, "uipath-runtime-python" README: runtime protocol (get_schema, execute, stream, dispose), factory, structured errors, execution context, tracing and durable execution (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-runtime-python/main/README.md
10. UiPath, "uipath-mcp-python" README: hosting MCP servers on the UiPath platform (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-mcp-python/main/README.md
11. UiPath, "uipath-integrations-python" README: agent runtime integrations for LangChain, LlamaIndex, Google ADK, OpenAI Agents and more, with human-in-the-loop through Action Center (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-integrations-python/main/README.md
12. UiPath, "uipath-dev-python" README: developer console for running and debugging runtimes, with OpenTelemetry traces (official; fetched). https://raw.githubusercontent.com/UiPath/uipath-dev-python/main/README.md
13. PyPI, "uipath" package metadata (version 2.14.37 at the time of this review; fetched). https://pypi.org/pypi/uipath/json
