# 03. ML/LLM Science Depth for System Design: Agents, Document Understanding, Screen Understanding, NLP and Evaluation

> **How to read this chapter.** You asked for ML system design, so this chapter covers the science-depth questions that sit underneath design decisions. At principal level, the depth round and the design round overlap: interviewers test whether you can justify each box on the diagram with modelling evidence. Each sub-topic opens with a plain-text reference diagram of the system the questions probe. The diagrams give you a shared picture to reason about. They are not model answers, and they render the same in Markdown and in a PDF export.
>
> **Tagging system (same as Chapter 01).**
> - **[Reported] [n]**: a candidate report, question bank or official page links this question, or this round type, to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role, the level and UiPath's AI products (agentic automation and Agent Builder, Document Understanding, Communications Mining, Autopilot, the AI Trust Layer, and computer vision for UI automation). No candidate has reported it.
> - Indented sub-bullets are follow-up probes an interviewer may use to push a strong first answer toward the principal bar.
>
> **Research limits (please read).** Two runs have now tried to research this chapter, and neither could reach a source. In the first run, the shared web-search budget (200 searches) was used up and fetches of glassdoor.com, uipath.com and arxiv.org failed DNS lookup. In the review run (2026-10-14), the search budget was still used up, and the network proxy rejected direct fetches of glassdoor.com, uipath.com, docs.uipath.com, arxiv.org and reddit.com. So the first two runs tagged no question [Reported]. None could be checked against a candidate write-up, and this chapter does not invent reports. Every question is **[Likely]**. The review run removed two duplicates and added 27 new principal-level [Likely] questions on gaps in the first draft: agent protocols and process conformance, credit assignment, line items and field normalisation, VLM image-token budgets, screen tokenisation and timing, taxonomy discovery, simulated-user evaluation, label noise and privacy-preserving training. The format notes are inferred from how principal-scientist loops usually run at enterprise AI companies; they are not a confirmed UiPath process. A later run with network access can upgrade tags to [Reported] where sources support them. **Update (October 2026 deepening pass):** three web searches found search-result snippets of UiPath ML interview reports (an ML Engineer, Agents team, 2026 report on prachub, and two Glassdoor Research Scientist, London, 2025 reviews). Questions 5, 12 and 93 are now tagged [Reported] on that basis, and new reported questions are in the deepening-pass section at the end. None of these reports is for the Principal Applied Scientist title, so treat them as signals of topic, not of level.

## Format notes (inferred, not confirmed)

| Item | Likely shape | Status |
|---|---|---|
| Duration | 45-60 min, one or two rounds (ML depth; sometimes merged into research-presentation Q&A) | Likely |
| Interviewer | Staff or principal applied scientist from the agents, Document Understanding, Communications Mining or CV/UI team | Likely |
| Style | Starts from your past work or a UiPath-shaped problem, then drills down 3-5 levels until you hit the edge of your knowledge | Likely |
| Principal bar | (1) You choose a method from first principles and evidence, not fashion. (2) You design evaluation that predicts customer outcomes. (3) You see failure modes before they happen. (4) You know when not to use an LLM. (5) You set direction for a research area across teams. | Likely |

### How depth questions attach to a design diagram

```
                ┌─────────────────────────── UiPath-shaped ML system ───────────────────────────┐
  Inputs        │  Documents ──> OCR + layout ──> extraction ──> validation ──> RPA / API sink  │
  (docs, email, │  Emails/msgs ─> NLP taxonomy ──> routing ──────────────────────> queue         │
   UI screens,  │  Screens ────> UI element detection ──> grounding ──> agent action ──> robot   │
   logs)        │                         ▲                    ▲                                │
                │                         └──── LLM planner / agent loop (tools, memory) ───────┘
                └──────────────┬────────────────────┬──────────────────────┬───────────────────┘
                               │                    │                      │
                   Depth probe A: Agents   Depth probe B/C: DU & CV   Depth probe D/E: NLP & Eval
                   "why this planner?"     "why this model?"          "how do you know it works?"
```

## Questions

### A. Agents and LLM reasoning

```
   Goal ──> ┌──────────┐   plan   ┌───────────┐  tool call  ┌──────────────────────┐
            │ Planner  │ ───────> │ Executor  │ ──────────> │ Tools: robots, APIs, │
            │  (LLM)   │ <─────── │ (LLM/code)│ <────────── │ DU, UI actions, RAG  │
            └────┬─────┘ observe  └─────┬─────┘  result     └──────────────────────┘
                 │                      │
                 ▼                      ▼
          ┌────────────┐        ┌──────────────┐        ┌────────────────────────┐
          │ Memory /   │        │ Guardrails / │ ─────> │ Human escalation (HITL)│
          │ state      │        │ policy check │        └────────────────────────┘
          └────────────┘        └──────────────┘
```

1. [Likely] Walk me through how you would decide between a single ReAct-style agent, a plan-then-execute agent, and a fixed workflow with LLM steps for an enterprise process such as invoice dispute handling.
   - What evidence would make you move from the agent design back to a deterministic workflow?
   - How does your answer change when each tool call triggers a real side effect in an ERP system?
2. [Likely] Why do LLM agents degrade on long-horizon tasks, and which of those causes can be fixed by better modelling rather than better orchestration?
   - How would you measure error compounding across steps?
   - What does a per-step success rate of 95% imply for a 20-step task, and is that model of failure realistic?
3. [Likely] How would you design the tool interface (schemas, descriptions, granularity) so an LLM selects the correct tool among 200 enterprise tools?
   - When would you use retrieval over tool descriptions instead of putting all tools in context?
   - How would you evaluate tool-selection accuracy separately from argument-filling accuracy?
4. [Likely] Compare chain-of-thought prompting, self-consistency, tree search and trained reasoning models for agent planning. Which would you ship under a fixed cost per run?
   - How would you show that extra test-time compute actually improves task success in your domain?
5. [Reported] [4] How do you make an agent recover from a failed tool call without looping or hallucinating success? (A 2026 UiPath MLE (Agents) report lists stopping a tool-calling loop that runs too long and fixing a model that calls tools badly.)
   - How would you detect "fake completion", where the agent says it finished but the system state disagrees?
6. [Likely] What memory architecture would you use for an agent that runs the same business process thousands of times a day?
   - What should be stored as episodic memory versus distilled into prompts, rules or fine-tuning data?
   - How do you stop memory from one tenant leaking into another?
7. [Likely] When would you fine-tune a smaller model for agent actions instead of prompting a frontier model? Describe the data you would need.
   - How would you build training trajectories from existing RPA logs and human corrections?
   - What failure modes do you expect from imitation learning on successful trajectories only?
8. [Likely] How would you apply reinforcement learning (for example, RL from task-level rewards) to improve an enterprise agent, given that real environments are expensive and have side effects?
   - How would you build a safe simulated environment, and how would you check the sim-to-real gap?
   - How would you prevent reward hacking when the reward is "process completed"?
9. [Likely] Describe a multi-agent design you would build for a cross-system process, and argue whether multiple agents are actually better than one.
   - How do you assign accountability for a failure in a multi-agent run?
10. [Likely] How would you defend an agent against prompt injection delivered through documents, emails or web pages it reads?
    - What can be done at the model level, and what must be done at the system and permission level?
    - How would you red-team this before release?
11. [Likely] How would you decide which agent actions need human approval, and how would you set that threshold from data?
    - How would you calibrate the agent's confidence so that the approval policy is trustworthy?
12. [Reported] [4] How do structured outputs, constrained decoding and function calling change error rates in agents? Where do they fail? (The same report lists keeping tool-call formats and calling stable.)
13. [Likely] How would you blend deterministic RPA robots with LLM agents so that the robot does the reliable parts and the agent handles exceptions?
    - How would the agent hand control back to a robot mid-process?
14. [Likely] What are the main open research problems in enterprise agents today? If you joined UiPath and could fund one science bet across agents, document understanding and screen understanding, which would it be?
    - How would you staff it, and what would you stop doing to pay for it?
    - How would you know after six months whether that bet was paying off?

15. [Likely] UiPath exposes robots and agents to other vendors' agents through open protocols such as the Model Context Protocol. How would you design and evaluate tool contracts so that an external agent cannot misuse a UiPath tool?
    - How would you version a tool schema without silently breaking agents that were tuned on the old one?
16. [Likely] How would you combine a BPMN-style process model with LLM agents so that each agent run can be checked for conformance against the model?
    - What do you do when the agent finds a better path than the modelled one: flag it, block it or learn from it?
17. [Likely] Given process-mining event logs from a customer, how would you estimate which steps an agent should take over, and the expected value, before anything is built?
    - How would you validate that estimate after deployment?
18. [Likely] How would you estimate uncertainty for an LLM agent's next action (for example, sampling-based agreement, semantic entropy or a trained verifier), and which would you trust for an approval gate?
19. [Likely] How would you assign credit to individual steps when training an agent from sparse, trajectory-level success signals?
    - Compare outcome reward models with process reward models for this setting.
20. [Likely] Agent latency is dominated by sequential LLM calls. Which modelling changes, not only infrastructure changes, would you make to cut end-to-end latency by half?

### B. Document understanding: OCR, layout models and extraction

```
  Scan/PDF/photo
      │
      ▼
 ┌──────────┐   ┌───────────────┐   ┌──────────────┐   ┌────────────────┐   ┌───────────┐
 │ Pre-proc │──>│ OCR / text    │──>│ Classify doc │──>│ Extract fields │──>│ Validate  │──> Export
 │ deskew,  │   │ + boxes, or   │   │ type, split  │   │ (layout model  │   │ rules +   │   to RPA /
 │ denoise  │   │ native PDF    │   │ multi-doc    │   │  or VLM / LLM) │   │ confidence│   ERP
 └──────────┘   └───────────────┘   └──────────────┘   └───────┬────────┘   └─────┬─────┘
                                                               │                  │ low conf.
                                                               │                  ▼
                                                               │          ┌────────────────┐
                                                               └──<───────│ Human validation│
                                                              retrain     │ station (labels)│
                                                                          └────────────────┘
```

21. [Likely] Compare an OCR-plus-layout-transformer pipeline (LayoutLM-style) with an OCR-free vision-language model (Donut or a general VLM) for invoice extraction. Which would you ship, and for which customers?
    - How does each behave on low-quality scans, handwriting and 30-page documents?
    - What is the cost per page of each at a million pages a day?
22. [Likely] How do layout-aware models encode 2D position, and why does that help over plain text sequences?
    - What breaks when the OCR reading order is wrong?
23. [Likely] How would you handle tables that span pages, have merged cells or have no ruling lines?
    - How would you evaluate table extraction: cell-level, row-level or by downstream task success?
24. [Likely] How would you produce well-calibrated per-field confidence scores from an extraction model, including an LLM that does not expose useful probabilities?
    - How would you set thresholds that trade straight-through processing rate against error rate per customer?
25. [Likely] A customer brings a new document type with 20 labelled examples. How would you get to production accuracy fastest?
    - Compare few-shot LLM prompting, fine-tuning a specialist model and active learning.
    - How would you decide when the specialist model should replace the LLM?
26. [Likely] How would you use human validation corrections as training data without amplifying annotator errors or bias?
    - How would you detect that validators are rubber-stamping low-confidence predictions?
27. [Likely] How do generative extractors hallucinate values that are not in the document, and how would you detect and prevent this?
    - How would you ground every extracted value back to a region on the page?
28. [Likely] How would you design a multilingual and multi-script document model, including right-to-left and CJK scripts?
    - Where would you expect the biggest accuracy drops, and how would you find them before customers do?
29. [Likely] How would you classify and split a 200-page PDF that contains many concatenated documents?
30. [Likely] How would you train a general-purpose extraction foundation model for business documents? Describe data, objectives and evaluation.
    - How would you get pretraining data while respecting customer data rights?
    - How would you show it beats a frontier VLM on your benchmark and not only on public ones such as DocVQA or FUNSD?
31. [Likely] What is the role of synthetic documents in training, and how do you check that synthetic gains transfer to real documents?
32. [Likely] How would you handle handwriting, checkboxes, signatures and stamps in forms?
33. [Likely] How would you detect drift when a vendor changes its invoice template, and what should happen automatically?
34. [Likely] How would you distil a large VLM extractor into a model that can run on-premises on CPU?
    - Which accuracy loss would you accept, and how would you explain it to a customer?

35. [Likely] How would you extract line items and other nested or repeated structures, and link each value to the right key and row?
    - How would you score a line-item prediction where one row is split into two?
36. [Likely] How do you define 'correct' for an extracted field when dates, amounts, names and addresses can be written many ways? Design the normalisation and the metric.
37. [Likely] How would you measure how much OCR error propagates into extraction error, and decide whether to spend the next quarter on OCR or on the extractor?
38. [Likely] A customer uploads documents with no schema. How would you propose a field schema automatically, and how would you evaluate the proposal?
39. [Likely] A VLM must read a 40-page document at enough resolution to see small print. How would you manage the image-token budget (page selection, tiling, cropping, multi-pass reading)?
    - How would you check that the model did not skip the page that held the answer?

### C. Computer vision for UI and screen understanding

```
  Screenshot (+ optional DOM / accessibility tree)
      │
      ▼
 ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
 │ UI element       │──>│ Element text /   │──>│ Grounding: map an     │──>│ Action: click,  │
 │ detection (boxes,│   │ icon semantics   │   │ instruction or step to│   │ type, select    │
 │ types)           │   │ (OCR + captions) │   │ a target element      │   │ + verify result │
 └──────────────────┘   └──────────────────┘   └──────────┬───────────┘   └────────┬────────┘
                                                          │ anchors / selectors     │ screen diff
                                                          ▼                         ▼
                                                 ┌───────────────────┐     ┌─────────────────┐
                                                 │ Robust re-locate  │<────│ Self-healing on │
                                                 │ on UI changes     │     │ failure         │
                                                 └───────────────────┘     └─────────────────┘
```

40. [Likely] How would you build a model that finds UI elements on screens of applications with no accessible DOM, such as Citrix or remote desktop sessions?
    - Detector plus OCR, or an end-to-end VLM? Argue both sides.
    - How do you handle tiny icons, dense tables and high-DPI screens?
41. [Likely] How would you ground a natural-language step ("click the Submit button in the payment dialog") to the correct element when there are several similar candidates?
    - How do you use spatial context, text anchors and visual similarity together?
42. [Likely] Selectors break when an application UI is updated. How would you design a self-healing targeting model, and how would you measure that it heals correctly rather than clicking the wrong element?
    - What is the cost of a false heal compared with a failed run, and how does that set your threshold?
43. [Likely] How would you build a training dataset for screen understanding across thousands of enterprise applications?
    - How would you use synthetic UI rendering, recorded robot runs and weak labels from accessibility trees?
    - How would you remove sensitive data from screenshots used for training?
44. [Likely] Compare pixel-only computer-use agents with agents that use the DOM or accessibility tree. When does each win?
    - How would you fuse both signals when they disagree?
45. [Likely] How would you evaluate a computer-use agent? Discuss benchmarks such as OSWorld, WebArena or ScreenSpot and their limits for enterprise apps.
    - What would an internal benchmark look like, and how would you keep it from leaking into training data?
46. [Likely] How would you verify that a UI action had the intended effect, using only the next screenshot?
47. [Likely] How would you make screen-understanding inference fast enough for interactive automation on a user's laptop?
    - Which parts would you cache across frames?
48. [Likely] How would you handle the long tail: custom controls, canvas-rendered apps, legacy terminals and mainframe green screens?
49. [Likely] How would you turn a recorded user session (task mining) into a reusable automation or agent skill?
    - How do you segment a long recording into steps, and how do you evaluate the segmentation?

50. [Likely] Screenshots are high-resolution and UI targets are tiny. How would you tokenise or crop screens for a VLM without losing small elements, and how would you test that choice?
51. [Likely] How would you get a screen model trained mostly on web and Windows apps to generalise to SAP GUI, Java Swing, mobile and terminal UIs?
    - How would you measure the domain gap before collecting new data?
52. [Likely] How would a UI agent know when the screen is ready to act on (loading spinners, delayed pop-ups, animations), and how would you evaluate timing errors separately from targeting errors?
53. [Likely] From a single human demonstration, how would you build an automation that generalises to new inputs and slightly different screens?
    - What should the system ask the user to clarify, and when?

### D. NLP: communications mining, classification and retrieval

```
  Emails / tickets / chats
      │
      ▼
 ┌────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌───────────────────┐
 │ Clean,     │──>│ Multi-label      │──>│ Entity / field    │──>│ Route / trigger    │
 │ thread,    │   │ intent taxonomy  │   │ extraction        │   │ automation, reply  │
 │ dedupe     │   │ (hierarchical)   │   │                   │   │ drafting (RAG)     │
 └────────────┘   └────────┬─────────┘   └───────────────────┘   └───────────────────┘
                           │
                  ┌────────▼─────────┐
                  │ Active learning: │  analysts label → taxonomy evolves → model retrains
                  │ discover, label, │
                  │ refine taxonomy  │
                  └──────────────────┘
```

54. [Likely] How would you build a multi-label, hierarchical intent classifier for customer emails where the taxonomy is defined by business users and changes every month?
    - How do you handle labels being split, merged or renamed without relabelling everything?
55. [Likely] Compare a fine-tuned encoder classifier with zero-shot or few-shot LLM classification on cost, latency, calibration and accuracy at 10 million messages a month.
56. [Likely] How would you design the active-learning loop so business analysts can build a taxonomy from unlabelled data in a few hours?
    - Which sampling strategy would you use for discovery versus refinement?
    - How do you avoid sampling bias in the resulting evaluation set?
57. [Likely] How would you handle email threads, quoted replies, signatures and attachments when modelling intent?
58. [Likely] How would you measure and improve calibration for multi-label predictions so thresholds work per label?
59. [Likely] How would you design retrieval-augmented generation over enterprise knowledge (process documents, SOPs, policies) for an agent?
    - Which chunking strategy suits SOPs with tables and numbered steps?
    - When do hybrid search, reranking or query rewriting pay for themselves?
    - How do you enforce document-level permissions in retrieval?
60. [Likely] How would you detect sentiment, urgency and churn risk in operational messages, and how would you prove the signal is useful to the business?
61. [Likely] How would you support many languages for communications mining with labels mostly in English?
62. [Likely] How would you mask PII in text before it reaches a third-party LLM while keeping enough context for the task?
    - How do you measure the accuracy loss caused by masking?
63. [Likely] How would you use LLMs to discover a taxonomy from unlabelled messages (clustering plus LLM-written label names), and how would you evaluate a taxonomy when there is no ground truth?
64. [Likely] How would you evaluate drafted email replies for factual grounding in the retrieved policy, tone and resolution rate, and which of these can be automated?

### E. Evaluation and measurement science

```
 ┌──────────────┐   ┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
 │ Golden sets  │──>│ Offline metrics     │──>│ Pre-release gates    │──>│ Online monitoring    │
 │ per task /   │   │ field F1, task      │   │ regression suite,    │   │ STP rate, HITL rate, │
 │ tenant slice │   │ success, LLM-judge  │   │ red-team, safety     │   │ cost/run, drift      │
 └──────▲───────┘   └─────────────────────┘   └──────────────────────┘   └──────────┬──────────┘
        │                                                                           │
        └────────────────── human corrections and incidents become new tests ───────┘
```

65. [Likely] Design an evaluation framework for enterprise agents that a dozen product teams can share.
    - What is the unit of evaluation: step, trajectory or business outcome?
    - How do you score partial success?
66. [Likely] How would you validate an LLM-as-judge before relying on it for release decisions?
    - How do you measure and correct position bias, verbosity bias and self-preference?
    - How much human agreement data is enough?
67. [Likely] Offline field-level F1 went up, but customers report more errors. List possible causes and how you would check each.
68. [Likely] How would you build test sets that represent customers you cannot see, because their data stays in their tenant?
    - What can privacy-preserving or federated evaluation offer here, and what does it cost?
69. [Likely] How would you compute confidence intervals and test significance for agent task success when runs are non-deterministic?
    - How many runs per task would you need to detect a 2-point improvement?
70. [Likely] How would you tie model metrics to business metrics such as straight-through processing rate, handling time saved and cost per transaction?
71. [Likely] How would you catch benchmark contamination when you use third-party foundation models?
72. [Likely] How would you design an A/B test or staged rollout for a model change in an automation product, where errors cause downstream financial actions?
    - When would you use shadow mode instead of an A/B test?
73. [Likely] How would you evaluate safety and policy compliance (for example, the agent must never approve a payment above a limit) in a way that gives strong guarantees, not only averages?
74. [Likely] How would you evaluate hallucination in extraction and in generated replies, and what metric would you report to a customer?
75. [Likely] How would you decide that a model is good enough to switch a customer from human validation of every document to sampled validation?
76. [Likely] How would you compare foundation-model providers for UiPath products when the AI Trust Layer lets customers choose their own model?
    - How do you keep evaluations stable when providers update models silently?

77. [Likely] How would you evaluate a conversational agent that needs several turns with a user, for example with simulated users as in tau-bench? What makes a simulated user trustworthy?
78. [Likely] Your golden set has label noise and annotators disagree. How would you measure inter-annotator agreement, fix the set and report metrics that account for the noise?
79. [Likely] How would you design an internal benchmark that stays useful as models improve, rather than saturating within a year?
80. [Likely] Which metric would you use to measure human effort saved (for example, time to validate, corrections per document) rather than model accuracy, and how would you collect it without slowing users down?
81. [Likely] How would you compare verbalised confidence, token log-probabilities and a separate verifier as confidence signals for LLM outputs? Which would you ship?

### F. Model strategy, efficiency and deployment science

82. [Likely] How would you decide between building UiPath-specific models (for documents, communications or UI) and relying on third-party frontier models?
    - What would change your decision in 12 months?
83. [Likely] How would you route requests between small and large models to minimise cost at fixed quality?
    - How would you train and evaluate the router?
84. [Likely] Which compression methods (quantisation, distillation, pruning, speculative decoding) would you apply for on-premises deployments, and how do you measure quality loss on the long tail?
85. [Likely] How would you fine-tune a model per customer (for example, with LoRA adapters) and serve thousands of adapters efficiently?
    - How do you stop one customer's adapter from learning another customer's data?
86. [Likely] How would you handle continual learning from customer feedback without catastrophic forgetting or silent regressions?
87. [Likely] How do context length and long-context models change document and agent design, and where do they still fail (for example, lost-in-the-middle)?
88. [Likely] How would you design a model that is robust to adversarial or malformed documents and screenshots?
89. [Likely] What is your approach to data governance for training: consent, retention, opt-out and the right to be forgotten in trained models?

90. [Likely] When would you train one generalist model across documents, communications and UI, and when separate specialists? How would you detect negative transfer between tasks?
91. [Likely] How would you use LLM-generated synthetic data to train smaller models without collapsing diversity or copying the teacher's errors?
92. [Likely] When would you apply differential privacy or other formal privacy methods when training on customer documents, and how would you explain the accuracy cost?

### G. Principal-level depth on your own work

93. [Reported] [4] Pick the most complex model you shipped. Draw the system, and tell me which design decision you would reverse today and why.
    - What experiment told you it was the wrong decision?
94. [Likely] Tell me about a time an offline evaluation misled you. What did you change in your evaluation practice afterwards?
95. [Likely] Describe a research result you chose not to ship. How did you make that call?
96. [Likely] Which paper from the last two years most changed how you would build agents or document models, and what would you build differently because of it?
    - What is the weakest part of that paper?
97. [Likely] How have you set the technical direction for a group of scientists working across several products?
    - How did you settle a disagreement between research ambitions and product deadlines?
98. [Likely] How do you decide whether a problem needs new research or a well-executed application of known methods?

99. [Likely] I will show you the abstract and main table of a recent agent or document-AI paper. Critique it as if you were deciding whether your team should build on it.
    - Which missing experiment would change your decision?
100. [Likely] What standards for reproducibility, baselines and ablations would you require before a result from your team reaches a product roadmap?

## Deepening pass (October 2026): newly reported questions

These come from search-result snippets only (the pages themselves could not be fetched). The reports are for adjacent UiPath roles (ML Engineer on the Agents team; Research Scientist in London), not for the Principal Applied Scientist title. At principal level, expect the same topics followed by deeper probes.

### Newly reported

101. [Reported] [4] [6] How does a transformer work? Walk through its internals in detail.
     - Why is attention scaled by the square root of the key dimension?
     - Where does the quadratic cost come from, and what would you do about it for 40-page documents?
102. [Reported] [4] How does beam search work, and when would you use it instead of greedy decoding or sampling?
     - Why can beam search make LLM outputs worse for open-ended generation but better for structured extraction?
103. [Reported] [4] How does LoRA work?
     - How do you choose the rank and the layers to adapt?
     - When does LoRA fall short of full fine-tuning?
104. [Reported] [4] How does an LLM call tools? Explain the mechanism from the training data to the decoded call.
     - How would you keep the tool-call format and calling behaviour stable across model versions?
     - A model calls tools badly. How would you diagnose and fix it: prompt, constrained decoding or fine-tuning?
     - How would you stop a tool-calling loop that runs too long?
105. [Reported] [6] How does FSDP (fully sharded data parallel) work?
     - What exactly is sharded, and what communication happens in the forward and backward passes?
     - When would you choose FSDP over tensor or pipeline parallelism?
106. [Reported] [5] What is the time complexity of matrix multiplication, and how would you parallelise it across GPUs?
     - How does that choice interact with memory bandwidth and inter-GPU communication?

### Gap-closing [Likely] questions for the principal bar

107. [Likely] Estimate the GPU memory needed to serve a 70B-parameter model with a 32k context at batch size 16. Where does the KV cache dominate, and which techniques (grouped-query attention, paged attention, quantised KV cache) would you use?
108. [Likely] Compare RLHF with PPO, DPO and RL from verifiable rewards for aligning an agent model to enterprise process outcomes. Which would you choose, and why?
     - How would you detect over-optimisation against the reward or preference model?
109. [Likely] Compare mixture-of-experts with dense models for a UiPath workload that mixes documents, emails and screens. What are the serving and fine-tuning consequences?
110. [Likely] How do tokenisation choices affect numbers, dates, codes and non-Latin scripts in documents, and how would you measure the damage?
111. [Likely] How would you choose between full fine-tuning, LoRA, continued pretraining and prompting for a new domain, given a fixed compute and data budget? Which scaling evidence would guide you?
112. [Likely] Explain how speculative decoding works and when it does not speed things up. Would it help an agent whose outputs are mostly short tool calls?
113. [Likely] Mixed-precision training diverged at step 40k. How would you debug it (loss scaling, bf16 versus fp16, learning-rate schedule, data issues)?
114. [Likely] How would you decide the training data mixture (documents, UI traces, code, general text) for a UiPath foundation model, and how would you test the mixture cheaply before a full run?
115. [Likely] How would you explain the difference between in-context learning and fine-tuning in terms of what the model learns, and what does that imply for per-customer adaptation?
116. [Likely] Take-home follow-up: you submitted a statistics-heavy solution. Defend your modelling assumptions, and show how your conclusion changes if one key assumption is wrong.

## References

The first two runs retrieved no source page. References 4-7 were added in the October 2026 deepening pass from search-result snippets. The official pages below are the right places to check product context (Agents, Maestro, Document Understanding, Communications Mining and the AI Trust Layer). On 2026-10-14 the network proxy blocked every fetch to them, so they are listed as unverified starting points. No question relies on them.

1. UiPath Careers, uipath.com (official careers page; unverified, fetch blocked). https://www.uipath.com/company/careers
2. UiPath Blog, uipath.com (official product and AI announcements; unverified, fetch blocked). https://www.uipath.com/blog
3. UiPath Documentation portal, docs.uipath.com (product documentation; unverified, fetch blocked). https://docs.uipath.com
4. Prachub, "UiPath Machine Learning Engineer Interview Experience — LLM Agent Q&A, a LeetCode 837 Variant, and a Coding Reject" (Agents team, 2026; seen as a search-result snippet only). https://prachub.com/interview-experiences/uipath-machine-learning-engineer-interview-experience-llm-agent-qanda-a-leetcode-837-variant-and-a-coding-reject
5. Glassdoor, UiPath interview review, Research Scientist, London, April 2025 (seen as a search-result snippet only). https://www.glassdoor.co.nz/Interview/UiPath-Interview-E1102519-RVW96319156.htm
6. Glassdoor, UiPath interview review, Research Scientist, London, June 2025 (seen as a search-result snippet only). https://www.glassdoor.co.nz/Interview/UiPath-Interview-E1102519-RVW98010961.htm
7. Principal Applied Scientist job posting, UiPath (via remotivated.com; product context only, not an interview report). https://remotivated.com/jobs/principal-applied-scientist-e786c87c
