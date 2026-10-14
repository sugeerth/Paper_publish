# 09. Cross-Portal ML System Design Bank: Peer-Company Prompts with Block Diagrams and UiPath Recasts

> **How to read this chapter.** You asked for ML system design diagrams and an explanation of the tag system, so this chapter is a bank of design prompts. Each prompt family has one plain-text block diagram, a source tag and a "UiPath recast" note that turns the generic prompt into a UiPath problem. The diagrams draw only the prompt's problem space: the inputs, the parts the interviewer expects you to fill in, and the outputs. They are scaffolds, not solutions. As with the rest of the pack, this chapter gives questions only, not answers.

## The tag system (read this first)

Each numbered question carries exactly one tag.

| Tag | Meaning | Evidence needed |
|---|---|---|
| **[Reported] [n]** | A candidate report, or an official company page, links this question to a named company's loop (UiPath or a peer). | A source [n] that we fetched and read in this run. |
| **[Bank] [n]** | The prompt appears in a public ML system design question bank or study guide [n], but that source does not tie it to a specific company's interview. | A source [n] that we fetched and read in this run. |
| **[Likely]** | An inference from the Principal Applied Scientist role, the level and UiPath-peer products. No source reports it. | None; it is judgement. |

Other markers:

- **Peer anchor:** a published engineering system at a peer company (title, company, year, as listed in the curated index [9]) that shows a real production version of the prompt. A peer anchor is context. It is **not** evidence that the company asks the prompt in interviews.
- **UiPath recast:** how to turn the prompt into a UiPath problem (agents, Document Understanding, Communications Mining, Autopilot, the AI Trust Layer, Process Mining, UI-automation vision). Recasts are always [Likely].
- Indented sub-bullets are follow-up probes, phrased as questions.

### Research limits (please read)

This run aimed to collect prompts that candidates report on Glassdoor, Blind, LeetCode Discuss, Exponent, interviewing.io, IGotAnOffer, Reddit, Interview Query and 1point3acres for Applied, Principal and Staff Scientist loops at Microsoft, Amazon, Google, ServiceNow, Automation Anywhere and Celonis. **That collection did not happen.** The workflow's shared web-search budget (200 searches per turn) was already used up before this chapter started, so no search ran. Direct fetches failed for igotanoffer.com, interviewquery.com, teamblind.com, leetcode.com, huyenchip.com, amazon.science and learn.microsoft.com (DNS errors), and for reddit.com (fetching blocked). Only github.com could be reached.

So:

- **No question in this chapter is tagged [Reported].** None could be verified, and this chapter does not invent candidate reports.
- **[Bank]** tags cite open-source interview-prep repositories on GitHub that were fetched and read [1]-[8].
- **Peer anchors** come from a curated GitHub index of company engineering papers and blog posts [9]. The index gives titles, companies and years. Its individual article URLs were not opened, so this chapter cites the index, not the articles.
- To upgrade tags to [Reported], run a follow-up with search budget against the portals named above.

## Format notes for this round

| Item | What the sources say | Source |
|---|---|---|
| Structure | A 9-step frame: problem formulation, metrics (offline and online), high-level architecture, data, features, modelling and offline evaluation, prediction service, online testing and deployment, scaling and monitoring | [1], [2] |
| Flexibility | "You may skip some of these components or spend more time for a deep dive in one or two components." | [1] |
| Duration | Not given by any fetched source. Peer loops usually run 45-60 minutes; confirm with your recruiter | [Likely] |
| Interviewer | Not given by any fetched source. Usually a staff or principal scientist or engineer | [Likely] |
| Answer style | Open-ended, "no single correct answer"; the goal is to "zoom out and design a production-level ML system" | [1] |
| GenAI round | "Increasingly common as a separate round"; interviewers "care more about cost, latency, guardrails, and monitoring than about the architecture diagram"; "evaluation methodology is the new system design" | [1] |
| Company spread | Prep authors find the components "very similar across FAANG companies"; startups tailor prompts to their own products | [10] |

**Principal-level bar (inferred, [Likely]).** At principal level, the bar is less about naming the right model and more about these:

- framing the business problem and its SLOs;
- choosing where ML belongs and where deterministic logic (rules, robots, workflows) should take over [1];
- designing the evaluation programme before the model;
- planning for cost, latency and multi-tenant governance;
- describing a rollout and a feedback loop that other teams can run;
- naming the two or three decisions that are hard to reverse.

### Canvas used by every diagram below

```
 [ Inputs / data ] --> [ ? stage A ] --> [ ? stage B ] --> [ Output / action ]
                             ^                                    |
                             +------- feedback / labels <---------+
 Boxes marked "?" are the parts the interviewer expects you to design.
```

---

## A. Search and ranking

### A1. Product / document search with retrieval and ranking

Tag basis: [Bank] [3], [7]. Peer anchors in [9]: Amazon "Semantic Product Search" (2019), Amazon "QUEEN: Neural query rewriting in e-commerce" (2021), Microsoft "AI at Scale in Bing" (2020), Google "A Guide to Google Search Ranking Systems" (2022).

```
 query --> [ ? query understanding ] --> [ ? candidate retrieval ] --> [ ? ranker ]
            (rewrite, intent, lang)       (lexical | dense | hybrid)    (LTR / x-enc)
                                                                         |
 catalogue / corpus --> [ ? indexing + embeddings ]                       v
 clicks, conversions, dwell --> [ ? labels + training ] ----------> ranked results
```

UiPath recast: search over the automation catalogue (Marketplace components, processes, agents, API workflows) or over Orchestrator assets, where a query such as "invoice approval for SAP" must return reusable automations. Labels come from reuse and deploy events, not clicks.

1. Design a search system for an e-commerce platform that returns a ranked list of products for a text query at about 50M daily active users. [Bank] [3]
   - Which of the clarifying questions in [3] (product-only versus general search, personalisation, languages, latency budget) would change your architecture most?
   - How do you split the latency budget between retrieval and ranking?
2. Design a document search system that supports full-text and semantic queries. [Bank] [5]
   - When does hybrid retrieval beat pure dense retrieval, and how would you show it offline?
3. Design search for an internal catalogue of automations and agents, where "relevance" means "the user deployed and reused it". [Likely]
   - What is your implicit-feedback signal when most users search once and never click?
   - How do you stop popular legacy automations from crowding out better new ones?
4. Design a learning-to-rank system for a marketplace where most items have little interaction data (cold start). [Likely]
   - How would you use content features or LLM-generated descriptions to rank new items?

### A2. Query autocompletion and related searches

Tag basis: [Bank] [5], [8].

```
 keystrokes --> [ ? prefix candidate gen ] --> [ ? ranker / LM ] --> top-k suggestions
 query logs --> [ ? mining + filtering ] ---^          ^
 user/tenant context ---------------------------------+
```

UiPath recast: prompt completion in Autopilot or in a natural-language-to-workflow box, where the completion must map to activities that actually exist in the tenant.

5. Design an autocompletion or typeahead suggestion system. [Bank] [5]
   - How would you meet a per-keystroke latency target with an LLM in the loop?
6. How would you generate a list of related searches for each query? [Bank] [8]
   - How do you keep related searches from leaking another tenant's query terms?
7. Design next-step suggestions for a workflow designer: given the steps a user has built so far, suggest the next activity. [Likely]
   - Is this ranking, sequence modelling or retrieval, and why?
   - How do you evaluate it when "accepted" suggestions may still be wrong?

### A3. Ranking a feed or a queue

Tag basis: [Bank] [5], [7], [8]. Peer anchors in [9]: LinkedIn "Understanding Dwell Time to Improve LinkedIn Feed Ranking" (2020); Airbnb "Managing Diversity in Airbnb Search" (2020).

```
 items (posts | tasks | cases) --> [ ? features ] --> [ ? multi-objective ranker ]
 user / agent state ------------------^                     |
                                                            v
                           [ ? re-rank: diversity, SLA, fairness ] --> ordered queue
```

UiPath recast: rank the human-in-the-loop queue in Action Center, or the exception queue for a process, by business urgency, SLA risk and which reviewer is best placed to handle each item.

8. Design a newsfeed ranking system. [Bank] [5], [7]
   - How do you combine several objectives (click, dwell, long-term retention) into one score?
9. How do you build a model that ranks all the answers to a question? [Bank] [8]
10. Design a ranking system for a queue of human-review tasks that a fleet of agents and robots generates, so that reviewers clear the items with the highest business risk first. [Likely]
    - What label tells you a ranking was "good" when every item eventually gets done?
    - How would you stop the ranker from starving low-value but SLA-bound items?
11. Rank the top 10 results for a rental-listing search. [Bank] [8], [7]
    - How would you add diversity without hurting conversion?

---

## B. Recommendation

### B1. Item recommendation (two-stage)

Tag basis: [Bank] [5], [7]. Peer anchors in [9]: Amazon "Item-to-Item Collaborative Filtering" (2003), Google/YouTube "Deep Neural Networks for YouTube Recommendations" (2016), Google "Self-supervised Learning for Large-scale Item Recommendations" (2021).

```
 user history + context --> [ ? candidate generation ] --> [ ? ranking model ]
                              (CF | two-tower | graph)        |
 item catalogue -------------------^                          v
                                 [ ? business rules / re-rank ] --> top-N recs
 impressions, accepts --> [ ? logging + training loop ] --------------^
```

UiPath recast: recommend which processes to automate next, or which agent or Marketplace component to adopt, for a customer tenant. Use process-mining signals and the tenant's existing automations as the history.

12. Design a video recommendation system. [Bank] [5], [7]
    - Why use two stages, and what does each stage optimise?
13. Design a recommendation system to suggest replacement items when the one ordered is unavailable. [Bank] [8], [5]
14. Design an event recommendation system. [Bank] [5]
15. Design a system that recommends the next automation opportunity to a customer, using task-mining and process-mining data from that customer. [Likely]
    - What is the label: automations built, automations still running after 90 days, or hours saved?
    - How do you recommend for a new tenant with no automation history?
16. Design "people you may know" or who-to-follow recommendations. [Bank] [5], [8]
    - Recast: how would the same graph approach recommend which colleague or Centre of Excellence member should review an automation? [Likely]

### B2. LLM-assisted recommendation

Tag basis: [Bank] [1].

```
 user profile + history --> [ ? retrieval (classic) ] --> candidates
                                                              |
                         [ ? LLM as ranker / feature generator / explainer ]
                                                              |
                                                              v
                                              ranked + explained recommendations
```

UiPath recast: Autopilot suggests which agent or automation to build, with a natural-language reason the Centre of Excellence lead can audit.

17. Design an LLM-based recommendation or personalisation system, using the LLM as a ranker or as a feature generator. [Bank] [1]
    - When is an LLM ranker worth its cost and latency compared with a gradient-boosted ranker?
    - How do you test that the LLM's explanations are faithful to the real ranking signal?
18. Design a system that recommends reusable workflow templates while a user describes a task in natural language. [Likely]

---

## C. Anomaly and fraud detection

### C1. Transaction fraud with humans in the loop

Tag basis: [Bank] [5], [8]. Peer anchor in [9]: Uber "Project RADAR: Intelligent Early Fraud Detection System with Humans in the Loop" (2022).

```
 events / transactions --> [ ? real-time features ] --> [ ? scorer ] --> allow
                                    ^                        |   \--> block
 entity history / graph ------------+                        v
                                           [ ? human review queue ] --> labels
                                                             |              |
                                    [ ? retraining + drift monitor ] <------+
```

UiPath recast: detect fraud or policy breaches in invoices and payments that Document Understanding and finance automations process, such as duplicate invoices, vendor-bank-detail changes and split purchase orders. Flagged items go to Action Center.

19. How would you build a fraud detection system? [Bank] [8], [5]
    - How do you train when confirmed fraud labels arrive weeks late and are biased by what was reviewed?
    - How do you set thresholds when the review team's capacity is fixed?
20. Design an anomaly detection system for invoices flowing through an accounts-payable automation, so that suspicious invoices are held before payment. [Likely]
    - Which signals come from the document itself, and which from the vendor's history?
    - How would you explain a hold to an auditor?
21. Design a system that detects when the automation itself is being misused, for example a robot account used to move money outside policy. [Likely]

### C2. Abuse and anomaly detection in behaviour logs

Peer anchors in [9]: LinkedIn "Detecting and Preventing Abuse on LinkedIn using Isolation Forests" (2019), LinkedIn "Preventing Abuse Using Unsupervised Learning" (2020), LinkedIn "Using deep learning to detect abusive sequences of member activity" (2021).

```
 activity / telemetry streams --> [ ? sessionise + featurise ] --> [ ? unsup. / seq model ]
                                                                         |
                                     [ ? alert triage + dedup ] <--------+
                                                |
                                                v
                                  investigations --> labels --> [ ? semi-sup. retrain ]
```

UiPath recast: anomaly detection on robot and agent run logs in Orchestrator. Flag runs whose step sequence, duration or error pattern drifts. These often signal that a target application's UI changed and the robots are about to break.

22. Design an unsupervised anomaly detection system for event sequences, with almost no labels at the start. [Likely]
    - How would you move from isolation-forest style scores to a supervised model as investigation labels arrive?
23. Design a system that predicts robot or agent run failures before they happen, from execution logs and screenshots. [Likely]
    - How do you tell "the target app changed" apart from "the robot logic is wrong"?
    - What alert precision do on-call operators need before they trust the system?
24. Design a monitoring system that finds process-level anomalies (bottlenecks, rework loops, unusual paths) in process-mining event logs across thousands of process variants. [Likely]

---

## D. RAG enterprise assistants

### D1. Chat with your docs / enterprise Q&A

Tag basis: [Bank] [1], [8].

```
 enterprise sources (wiki, tickets, PDFs, email) --> [ ? ingest: parse, chunk, ACL tag ]
                                                               |
                                                     [ ? embed + index ]
                                                               |
 user question --> [ ? query rewrite ] --> [ ? hybrid retrieve + re-rank ]
                                                               |
                         [ ? grounded generation + citations + "no answer" ]
                                                               |
                                    [ ? guardrails ] --> answer --> feedback
```

UiPath recast: an Autopilot or agent knowledge layer, grounded in a customer's SOPs, process documentation and past exception resolutions, that must respect per-document permissions and the AI Trust Layer's data controls.

25. Design a RAG document Q&A ("chat with your docs") system. [Bank] [1]
    - When would you pick RAG, fine-tuning, long context or tools and memory for this problem? [1]
    - How do you handle questions the corpus cannot answer?
26. Design a question-answering system that can extract an answer from a large collection of documents. [Bank] [8]
27. Design enterprise semantic search with LLM-generated answers and citations. [Bank] [1]
    - How do you enforce document-level access control at retrieval time, not after generation?
    - How do you keep the index fresh when source permissions change hourly?
28. Design a RAG assistant that answers employees' "how do I do X in system Y" questions and can hand off to an automation that does X. [Likely]
    - Where does the boundary sit between answering a question and executing an action?
29. Design retrieval over scanned and semi-structured documents (invoices, contracts, forms), where layout carries meaning. [Likely]
    - How do you chunk a table or a multi-page form without breaking the facts in it?

### D2. Customer-support assistant with human fallback

Tag basis: [Bank] [1], [8].

```
 customer message --> [ ? intent + risk triage ] --> [ ? RAG answerer | tool calls ]
                                |                              |
                                v                              v
                     [ ? human agent queue ] <-- low confidence / policy hit
                                |
                                v
                      resolution --> [ ? feedback + eval set ]
```

UiPath recast: a Communications Mining plus agent pipeline. Incoming customer emails are classified, entities are extracted, the agent drafts or completes the resolution through robots, and humans handle the rest.

30. Design an LLM-powered customer-support chatbot with guardrails and human fallback. [Bank] [1]
    - What confidence signal decides the handoff, and how do you calibrate it?
31. Build a chatbot to help people book hotels. [Bank] [8]
    - Recast: how does the design change when "booking" is an enterprise transaction a robot executes in SAP? [Likely]
32. Design a system that reads high-volume inbound email for a bank's operations team, routes each message, extracts fields and triggers the right automation. [Likely]
    - How do you add a new intent without retraining everything?
    - How do you measure automation rate without rewarding wrong automations?

### D3. Agentic assistant (planning, tools, memory)

Tag basis: [Bank] [1].

```
 goal --> [ ? planner ] --> [ ? tool / robot selection ] --> [ ? executor ]
             ^                                                    |
             |                 [ ? memory: short / long ] <-------+
             |                                                    v
             +------------ [ ? verifier + HITL checkpoint ] <-- results
```

UiPath recast: this is the core UiPath agentic-automation problem (agents that orchestrate robots, APIs and people). Chapter 04 covers it in depth, so this chapter keeps only the peer-bank versions.

33. Design an agentic workflow or AI assistant covering planning, tool use and memory. [Bank] [1]
    - How do you detect and stop loops and hallucinated tools? [1]
    - Where do you put human-in-the-loop checkpoints?
34. Design a code assistant or coding agent that uses RAG over a repository, tool use and verification. [Bank] [1]
    - Recast: how would the same verifier pattern apply to an agent that generates automation workflows? [Likely]

---

## E. Content moderation and safety

### E1. Harmful content detection

Tag basis: [Bank] [4], [5].

```
 post (text | image | video) --> [ ? per-modality encoders ] --> [ ? fusion ]
                                                                     |
                                     [ ? multi-label / multi-task heads ]
                                                                     |
                                     remove | demote | allow | send to review
 user reports + reviewer labels --> [ ? label pipeline + retrain ] ----^
```

UiPath recast: screen what agents send out (emails, chat replies, generated documents) and what they take in (inbound documents and emails that may carry prompt injection, PII or toxic content) inside the AI Trust Layer.

35. Design a system that monitors posts, detects harmful content and demotes or removes it (about 500M posts a day and about 10K labelled examples). [Bank] [4]
    - Which clarifying questions in [4] (harm types, languages, user reports, explainability, legal constraints) change the design most?
    - How do you get to a usable model from 10K labels?
36. Design a multimodal harmful-content detection system. [Bank] [5]
37. Design a spam detection system. [Bank] [5]
38. Design an output guardrail that checks every outbound message an enterprise agent writes for PII, policy breaches and hallucinated commitments before it is sent. [Likely]
    - What is the latency budget, and which checks can run asynchronously?
    - How do you measure the false-block rate customers will tolerate?

### E2. Prompt-injection and data-exfiltration defence

Tag basis: [Bank] [1] (topic list: "prompt-injection & jailbreak defense", PII redaction).

```
 untrusted inputs (email, web, docs) --> [ ? input scanner ] --> agent context
                                                                     |
 tool calls --> [ ? policy engine / allow-list ] --> [ ? output scanner ] --> world
                                    |
                                    v
                           audit log --> [ ? red-team + eval set ]
```

UiPath recast: agents that read customer emails and documents and then act in ERP systems give an injected instruction a direct path to a real transaction.

39. Design a defence against prompt injection for an agent that reads untrusted documents and can call tools. [Likely]
    - Which defences are ML classifiers, and which must be deterministic policy?
    - How do you red-team it continuously as attacks change?
40. Design a PII detection and redaction service that sits between enterprise data and third-party LLM APIs. [Likely]
    - How do you keep redaction from destroying the information the LLM needs to do the task?

---

## F. LLM evaluation platforms

### F1. Evaluation platform for LLM and RAG features

Tag basis: [Bank] [1] ("evaluation methodology is the new system design"; golden sets, LLM-as-judge, pairwise win rates, the RAG triad, recall@k, MRR, nDCG, regression testing).

```
 prod traces + curated tasks --> [ ? golden / regression sets ] --> [ ? runners ]
                                                                        |
         [ ? metrics: exact, retrieval, judge, pairwise, human ] <------+
                         |
                         v
           [ ? dashboards + release gates ] --> ship / block
                         ^
 judge calibration vs human labels -----+
```

UiPath recast: one evaluation platform for every agent, Document Understanding model and Autopilot feature, which product teams use to gate releases and customers use to check their own agents before deployment.

41. Design an evaluation system for a RAG product covering retrieval quality and answer faithfulness. [Bank] [1]
    - How do you build the golden set, and how do you keep it from going stale?
    - How do you show that your LLM judge agrees with human raters, and how do you track drift?
42. Design an offline-to-online evaluation pipeline for an LLM feature: offline evaluation, online A/B tests and regression testing before shipping. [Bank] [1]
43. Design an evaluation platform for multi-step agents, where success depends on the final state of external systems, not on the text. [Likely]
    - How do you build sandboxed replicas of enterprise apps to score agent runs?
    - How do you give credit or blame to individual steps in a 30-step trajectory?
44. Design a platform that lets enterprise customers evaluate their own agents on their own data without their data leaving their tenant. [Likely]
    - What evaluation data can you share across tenants, and what must never leave its tenant?
45. Design a continuous evaluation system that catches quality regressions when a foundation-model provider silently updates a model. [Likely]

### F2. Monitoring LLM systems in production

Tag basis: [Bank] [1] (monitoring list: quality drift, hallucination rate, retrieval hit rate, per-request latency and cost, user feedback, per-step tracing).

```
 requests --> [ ? tracing (per step) ] --> [ ? online metrics + sampled judging ]
                                                       |
                         [ ? drift / cost / latency alerts ] --> on-call
                                                       |
                         [ ? triage --> new eval cases ] --> eval set (F1)
```

UiPath recast: observability for agent runs in Orchestrator, with traces that join LLM calls, robot steps and human approvals.

46. Design production monitoring for an LLM application: hallucination rate, retrieval hit rate, latency, cost and user feedback. [Bank] [1]
    - Which metrics can you compute on every request, and which only on a sample?
47. Design tracing and root-cause analysis for agent failures that span LLM calls, robot actions and human approvals. [Likely]
    - How do you turn a production failure into a regression test automatically?

---

## G. Serving, cost and platform (cross-cutting)

### G1. Model routing and cost control

Tag basis: [Bank] [1] (KV caching, continuous batching, quantisation, speculative decoding, small and large model routing, self-hosting versus API, time to first token).

```
 request --> [ ? difficulty / risk classifier ] --> small model --+
                         |                                        +--> [ ? cache ] --> response
                         +-----------------------> large model --+
                                                     |
                               [ ? cost + quality feedback --> router retrain ]
```

UiPath recast: a gateway that serves many tenants' agents across several LLM providers, with per-tenant cost caps and data-residency rules.

48. Design a routing layer that sends simple requests to small models and hard ones to large models. [Bank] [1]
    - How do you train the router without labels for "this request needed the large model"?
49. Should you self-host or call an API for a given LLM workload? Design the decision and the serving stack for each option. [Bank] [1]
50. Design a multi-tenant LLM gateway with per-tenant quotas, data-residency constraints, provider failover and cost attribution. [Likely]
    - How do you degrade gracefully when one provider has an outage? [1]

### G2. Large-scale batch generation and summarisation

Tag basis: [Bank] [1].

```
 backlog of documents --> [ ? batch scheduler ] --> [ ? LLM workers ] --> [ ? safety filter ]
                                                                             |
                                                [ ? sampled QA ] --> outputs + audit
```

UiPath recast: overnight summarisation and field extraction over millions of archived documents during a customer's migration.

51. Design content generation or summarisation at scale, with batch processing and safety filtering. [Bank] [1]
    - How do you estimate cost and throughput before you run it?

---

## H. Classic NLP and vision prompts that recast directly

### H1. Text classification, language identification and entity linking

Tag basis: [Bank] [5], [8].

```
 text --> [ ? normalise + tokenise ] --> [ ? encoder ] --> [ ? task head ] --> label / entity
 labels (sparse, noisy) --> [ ? active learning + weak supervision ] ---^
```

UiPath recast: Communications Mining taxonomies (intents and fields defined per customer) and entity linking to master data such as vendor IDs and customer accounts.

52. Design a system to predict the language a text is written in. [Bank] [8], [5]
53. Design a named-entity linking system. [Bank] [5]
    - Recast: how would you link an extracted vendor name on an invoice to the right record in a customer's ERP? [Likely]
54. Design a sentiment analysis system. [Bank] [5]
55. How would you train a model to decide whether "jaguar" in a sentence means the animal or the car? [Bank] [8]
56. How would you map nicknames (Pete, Andy, Nick, Rob) to real names? [Bank] [8]
    - Recast: how would you resolve the same customer across email, CRM and invoice records? [Likely]
57. Design a text classifier where each enterprise customer defines its own label taxonomy with only a few dozen examples per label. [Likely]
    - Fine-tuned small model, LLM prompting or embeddings with a light head: how do you choose for each tenant?

### H2. OCR, object detection and image search

Tag basis: [Bank] [5].

```
 image / screenshot --> [ ? detector (regions / UI elements) ] --> [ ? recogniser (OCR) ]
                                                                         |
                                     [ ? layout / structure model ] --> fields | elements
```

UiPath recast: Document Understanding (scanned documents to fields) and UI-automation computer vision (screenshots to clickable elements for robots and computer-use agents).

58. Design an OCR / text recognition system. [Bank] [5]
59. Design an object detection system. [Bank] [5]
    - Recast: how would you detect and classify UI elements in screenshots of remote-desktop sessions (such as Citrix) where no DOM is available? [Likely]
60. Design an image search system, or a multimodal (text plus image) search system. [Bank] [5]
    - Recast: how would you find "the same screen" across app versions to make robot selectors robust? [Likely]
61. Design an end-to-end document extraction system for invoices from thousands of vendors with different layouts, where an extraction error has a financial cost. [Likely]
    - Where do you place confidence thresholds and human validation?
    - How do validated corrections flow back into training without one tenant's data leaking to another?

### H3. Prediction and estimation prompts

Tag basis: [Bank] [5], [7], [8].

```
 historical events + context --> [ ? features ] --> [ ? regressor / survival model ] --> estimate
                                                                       |
                                    [ ? calibration + uncertainty ] ---+
```

UiPath recast: predict how long a process instance will take, or when a case will breach its SLA, from process-mining event logs.

62. Design a delivery-time estimation system. [Bank] [5], [7]
    - Recast: how would you predict case completion time and SLA breach risk from process-mining data? [Likely]
63. If you built a Netflix clone, how would you predict when a user stops watching? [Bank] [8]
    - Recast: how would you predict when a customer will stop using an automation or churn from the platform? [Likely]
64. Design an ad click-prediction system. [Bank] [5], [7]
    - What transfers from CTR modelling to predicting whether a suggested automation will be accepted? [Likely]

---

## I. Principal-level stress probes (apply to any prompt above)

These are not standalone prompts. Use them after any family above to check your answer against the principal bar. All are [Likely].

65. Which two decisions in your design are hardest to reverse, and what evidence would you gather before committing to them? [Likely]
66. Where in this design should ML stop and deterministic logic (rules, robots, workflows) take over? [Likely]. The general principle appears in [1].
67. Estimate cost per 1,000 requests and the main cost driver. What would you cut first if the budget were halved? [Likely]
68. How does the design change when every customer is a separate tenant with its own data, labels, taxonomy and compliance regime? [Likely]
69. What would you build first in 90 days, what would you measure, and what would make you stop? [Likely]
70. Which parts of this system would you turn into a shared platform for other teams, and which should stay product-specific? [Likely]
71. How would you explain this system's failure modes to a regulated customer's auditor? [Likely]
72. If a competitor shipped a frontier-model-only version of this system tomorrow, which parts of your design would still be an advantage? [Likely]

---

## Coverage summary

| Topic family | Questions | [Bank] | [Likely] | [Reported] |
|---|---|---|---|---|
| A. Search and ranking | 1-11 | 7 | 4 | 0 |
| B. Recommendation | 12-18 | 5 | 2 | 0 |
| C. Anomaly and fraud | 19-24 | 1 | 5 | 0 |
| D. RAG and agents | 25-34 | 7 | 3 | 0 |
| E. Moderation and safety | 35-40 | 3 | 3 | 0 |
| F. LLM evaluation | 41-47 | 3 | 4 | 0 |
| G. Serving and cost | 48-51 | 3 | 1 | 0 |
| H. Classic NLP and vision | 52-64 | 11 | 2 | 0 |
| I. Principal stress probes | 65-72 | 0 | 8 | 0 |
| **Total** | **72** | **40** | **32** | **0** |

## References

1. "ML System Design" (9-step formula; GenAI/LLM and agentic system design section), alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
2. "ML System Design Template", alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-template.md
3. "Search System" (e-commerce search design), alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-search.md
4. "Harmful Content Detection" design, alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-harmful-content.md
5. ML system design sample questions and case-study directory (recommendation, search, ranking, NLP, CV, fraud, harmful content, typeahead, OCR, object detection, delivery time), alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/tree/main/src/MLSD
6. "ML Companies" reading list (LinkedIn, Google, AWS, Meta, Netflix, Airbnb, Uber systems), alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-companies.md
7. "machine-learning-interview" (ML design section: YouTube recommendation, LinkedIn feed ranking, ad click prediction, delivery time, Airbnb search ranking), khangich, GitHub, accessed 2026. https://github.com/khangich/machine-learning-interview
8. Chip Huyen, "Machine Learning Systems Design: Exercises" (27 open-ended design questions), chiphuyen/machine-learning-systems-design, GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
9. Eugene Yan, "applied-ml: Papers and tech blogs by companies sharing their work on data science and machine learning in production" (source of the peer-anchor titles), GitHub, accessed 2026. https://github.com/eugeneyan/applied-ml
10. alirezadir, "Machine-Learning-Interviews" (repository overview; notes on FAANG versus startup design interviews), GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews

Reference [6] was read for context, but no question cites it.
