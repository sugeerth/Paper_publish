# 09. Cross-Portal ML System Design Bank: Peer-Company Prompts with Block Diagrams and UiPath Recasts

> **How to read this chapter.** You asked for ML system design diagrams and an explanation of the tag system, so this chapter is a bank of design prompts. Each prompt family has one plain-text block diagram, a source tag and a "UiPath recast" note that turns the generic prompt into a UiPath problem. The diagrams draw only the prompt's problem space: the inputs, the parts the interviewer expects you to fill in, and the outputs. They are scaffolds, not solutions. As with the rest of the pack, this chapter gives questions only, not answers.

## The tag system (read this first)

Each numbered question carries exactly one tag.

| Tag | Meaning | Evidence needed |
|---|---|---|
| **[Reported] [n]** | A candidate report that we opened, or an official company page, links this question to a named company's loop (UiPath or a peer). | A primary source [n] that we fetched and read in this run. |
| **[Reported-2nd] [n]** | A candidate-report aggregator that we fetched and read [n] links the question to a named company's interview, with a report date or date range. We could not open the original candidate post. | A secondary source [n] that we fetched and read in this run. Treat the company link as probable, not certain. |
| **[Bank] [n]** | The prompt appears in a public ML system design question bank or study guide [n]. That source either ties it to no company, or carries only a maintainer-assigned company label with no per-question report behind it. | A source [n] that we fetched and read in this run. |
| **[Likely]** | An inference from the Principal Applied Scientist role, the level and UiPath-peer products. No source reports it. | None; it is judgement. |

Other markers:

- **Peer anchor:** a published engineering system at a peer company (title, company, year, as listed in the curated index [9]) that shows a real production version of the prompt. A peer anchor is context. It is **not** evidence that the company asks the prompt in interviews.
- **UiPath recast:** how to turn the prompt into a UiPath problem (agents, Document Understanding, Communications Mining, Autopilot, the AI Trust Layer, Process Mining, UI-automation vision). Recasts are always [Likely].
- Company and date notes in brackets after a question, such as "(Microsoft MLE loop; reports 2026 Q1-Q3)", come from the cited source.
- Indented sub-bullets are follow-up probes, phrased as questions. Probes are [Likely] unless they carry their own citation.

### Research limits (please read)

This chapter set out to collect prompts that candidates report on Glassdoor, Blind, LeetCode Discuss, Exponent, interviewing.io, IGotAnOffer, Reddit, Interview Query and 1point3acres for Applied, Principal and Staff Scientist loops at Microsoft, Amazon, Google, ServiceNow, Automation Anywhere and Celonis. Two passes ran in October 2026. In both, the workflow's shared web-search budget (200 searches per turn) was already used up, so no web search could run. Direct fetches failed with DNS errors for igotanoffer.com, tryexponent.com, interviewing.io, glassdoor.com, reddit.com, medium.com, interviewquery.com, 1point3acres.com, huyenchip.com and web.archive.org. Only github.com could be reached.

What the second pass did instead:

- It re-fetched and checked the original GitHub sources [1]-[10]. All resolved. Several questions had cited the MLSD directory listing [5] for prompts that appear only in the sample-question list of [1]. Those citations now point to [1].
- It used GitHub code search to find GitHub-hosted mirrors of candidate-report aggregators. These are TrueInterview's company guides and question pages, mirrored with a CC BY 4.0 licence [11]-[20], a curated list of Reddit, Blind and X interview threads [21], and an archived LinkedIn interview-experience post [24]. Questions taken from these sources are tagged **[Reported-2nd]**, because the original candidate posts could not be opened.
- Two company-tagged GitHub banks, [22] and [23], label questions with companies but give no per-question report. Their questions are tagged **[Bank]**.
- **No source found names UiPath, ServiceNow, Automation Anywhere or Celonis for an ML system design question.** The ServiceNow and Celonis code searches returned no interview content, and the UiPath search returned only unrelated pages. The [Reported-2nd] items therefore come from Microsoft, Amazon, Google, Meta, Databricks and other enterprise-AI or fraud-heavy peers. Treat them as calibration for the shape and level of the round, not as UiPath's own questions.
- To upgrade [Reported-2nd] to [Reported], open the original posts on the portals above in a run that has search budget.

## Format notes for this round

| Item | What the sources say | Source |
|---|---|---|
| Structure | A 9-step frame: problem formulation, metrics (offline and online), high-level architecture, data, features, modelling and offline evaluation, prediction service, online testing and deployment, scaling and monitoring | [1], [2] |
| Flexibility | Interviewers may want you to skip some components and go deep on one or two | [1] |
| GenAI round | "Increasingly common as a separate round"; interviewers "care more about cost, latency, guardrails, and monitoring" than about the diagram; "evaluation methodology is the new system design" | [1] |
| What an Amazon AS2 candidate saw (2026) | ML system design was "the most important round". It was "less about fancy models": scope, data, baseline then advanced model, offline versus online training, business-aligned evaluation and A/B tests, batch versus real-time serving, scale, and drift and feedback-loop monitoring | [24] |
| Prompt shape at peers (2025-2026) | Product-shaped prompts with a deliberate twist: position-bias correction, calibration, offline metrics that disagree with online ones, adversarial inputs, or "is the new model ready to replace the old one?" | [12], [13], [14], [21] |
| Company spread | Prep authors find interview components "very similar across FAANG companies"; startups tailor prompts to their own products | [10] |
| Duration and interviewer | Not given by any fetched source. Peer loops usually run 45-60 minutes with a staff or principal scientist or engineer; confirm with your recruiter | [Likely] |

**Principal-level bar (inferred, [Likely]).** At principal level, the bar is less about naming the right model and more about these:

- framing the business problem and its SLOs;
- choosing where ML belongs and where deterministic logic (rules, robots, workflows) should take over;
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

### A1. Product, document and enterprise search with retrieval and ranking

Tag basis: [Bank] [1], [3], [7], [23]; [Reported-2nd] [12], [21]. Peer anchors in [9]: Amazon "Semantic Product Search" (2019), Amazon "QUEEN: Neural query rewriting in e-commerce" (2021), Microsoft "AI at Scale in Bing" (2020), Google "A Guide to Google Search Ranking Systems" (2022).

```
 query --> [ ? query understanding ] --> [ ? candidate retrieval ] --> [ ? ranker ]
            (rewrite, intent, lang)       (lexical | dense | hybrid)    (LTR / x-enc)
                                                                         |
 catalogue / corpus --> [ ? indexing + embeddings ]                       v
 clicks, conversions, dwell --> [ ? labels + debiasing + training ] --> ranked results
```

UiPath recast: search over the automation catalogue (Marketplace components, processes, agents, API workflows) or over Orchestrator assets, where a query such as "invoice approval for SAP" must return reusable automations. Labels come from reuse and deploy events, not clicks.

1. Design a search system for an e-commerce platform that returns a ranked list of products for a text query at about 50M daily active users. [Bank] [3]
    - Which of the clarifying questions in [3] (product-only versus general search, personalisation, languages, latency budget) would change your architecture most?
    - How do you split the latency budget between retrieval and ranking?
2. Design a search retrieval and ranking pipeline, including online experiments and position-bias correction. (Amazon MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [12]
    - How do you estimate position bias without degrading the live experience too much?
    - Recast: the automation catalogue shows only a few results above the fold. How would you debias deploy and reuse signals there? [Likely]
3. Design a unified query engine over a user's data spread across several products, such as mail and file storage. (Google, AI role) [Reported-2nd] [21]
    - How do you rank results that come from sources with incomparable relevance scores?
    - Recast: one query engine over Orchestrator queues, process-mining logs, Communications Mining streams and DU outputs for one tenant. [Likely]
4. Design a document search system that supports full-text and semantic queries. [Bank] [1]
    - When does hybrid retrieval beat pure dense retrieval, and how would you show it offline?
5. Design search for an internal catalogue of automations and agents, where "relevance" means "the user deployed and reused it". [Likely]
    - What is your implicit-feedback signal when most users search once and never click?
    - How do you stop popular legacy automations from crowding out better new ones?
6. Design a learning-to-rank system for a marketplace where most items have little interaction data (cold start). [Likely]
    - How would you use content features or LLM-generated descriptions to rank new items?
7. Design a system that answers from content that appeared only 30 minutes ago. (Perplexity) [Bank] [23]
    - Recast: how do you keep an agent's knowledge layer current when a customer's SOPs change mid-day? [Likely]

### A2. Query autocompletion and related searches

Tag basis: [Bank] [1], [5], [8].

```
 keystrokes --> [ ? prefix candidate gen ] --> [ ? ranker / LM ] --> top-k suggestions
 query logs --> [ ? mining + filtering ] ---^          ^
 user/tenant context ---------------------------------+
```

UiPath recast: prompt completion in Autopilot or in a natural-language-to-workflow box, where the completion must map to activities that actually exist in the tenant.

8. Design an autocompletion or typeahead suggestion system. [Bank] [1], [5]
    - How would you meet a per-keystroke latency target with an LLM in the loop?
9. How would you generate a list of related searches for each query? [Bank] [8]
    - How do you keep related searches from leaking another tenant's query terms?
10. Design next-step suggestions for a workflow designer: given the steps a user has built so far, suggest the next activity. [Likely]
    - Is this ranking, sequence modelling or retrieval, and why?
    - How do you evaluate it when "accepted" suggestions may still be wrong?

### A3. Ranking a feed, a page or a queue

Tag basis: [Bank] [1], [5], [7], [8]; [Reported-2nd] [14], [18]. Peer anchors in [9]: LinkedIn "Understanding Dwell Time to Improve LinkedIn Feed Ranking" (2020); Airbnb "Managing Diversity in Airbnb Search" (2020).

```
 items (posts | modules | notifications | tasks) --> [ ? features ] --> [ ? multi-objective ranker ]
 user / agent state -------------------------------------^                     |
                                                                               v
                                    [ ? re-rank: diversity, SLA, fairness, caps ] --> ordered list
```

UiPath recast: rank the human-in-the-loop queue in Action Center, or the exception queue for a process, by business urgency, SLA risk and which reviewer is best placed to handle each item.

11. Design a newsfeed ranking system. [Bank] [1], [5], [7]
    - How do you combine several objectives (click, dwell, long-term retention) into one score?
12. Design a unified ranking model for notifications from several products, balancing engagement, latency and fairness across notification types. (Meta MLE loop; reports 2025 Q4-2026 Q3) [Reported-2nd] [14]
    - Recast: one ranker for alerts from robots, agents, process-mining monitors and DU validation tasks, sent to the same operations lead. How do you stop one source from flooding the others? [Likely]
13. Predict the probability that a user dislikes a feed post, and explain why offline and online metrics disagree. (Meta MLE loop) [Reported-2nd] [14]
14. Build a top-level ranking of the modules shown on an e-commerce landing page. (Wayfair; last reported Sep 2026) [Reported-2nd] [18]
    - Recast: rank the widgets on an Automation Cloud home page (failed jobs, pending approvals, suggested automations) per persona. [Likely]
15. How do you build a model that ranks all the answers to a question, and what does it cost to run? [Bank] [8]
16. Design a ranking system for a queue of human-review tasks that a fleet of agents and robots generates, so that reviewers clear the items with the highest business risk first. [Likely]
    - What label tells you a ranking was "good" when every item eventually gets done?
    - How would you stop the ranker from starving low-value but SLA-bound items?
17. Rank the top 10 results for a rental-listing search. [Bank] [8], [7]
    - How would you add diversity without hurting conversion?

---

## B. Recommendation

### B1. Item recommendation (two-stage) and model replacement

Tag basis: [Bank] [1], [5], [7], [8]; [Reported-2nd] [13], [21]. Peer anchors in [9]: Amazon "Item-to-Item Collaborative Filtering" (2003), YouTube "Deep Neural Networks for YouTube Recommendations" (2016), Google "Self-supervised Learning for Large-scale Item Recommendations" (2021).

```
 user history + context --> [ ? candidate generation ] --> [ ? ranking model ]
                              (CF | two-tower | graph)        |
 item catalogue -------------------^                          v
                                 [ ? business rules / re-rank ] --> top-N recs
 impressions, accepts --> [ ? logging + training loop ] --------------^
                                     |
                     [ ? champion / challenger gate ] --> replace old model?
```

UiPath recast: recommend which processes to automate next, or which agent or Marketplace component to adopt, for a customer tenant. Use process-mining signals and the tenant's existing automations as the history.

18. Design a video recommendation system. [Bank] [1], [5], [7]
    - Why use two stages, and what does each stage optimise?
19. Design a real-time news-feed recommender, covering cold start, latency budgets, drift monitoring and fallbacks. (Google MLE loop; reports 2025 Q4-2026 Q3) [Reported-2nd] [13]
    - What is your fallback when the feature store is down, and how do you test it?
20. You have trained a new recommendation model. How do you make sure it is ready to replace the old one? (Netflix MLE) [Reported-2nd] [21]
    - Which offline checks, shadow traffic and online guardrail metrics would you require?
    - Recast: a new DU extraction model or agent-planning model is ready for thousands of tenants. What is the gate? [Likely]
21. Design a recommendation system to suggest replacement items when the one ordered is unavailable. [Bank] [8], [1]
22. Design a system that recommends the next automation opportunity to a customer, using task-mining and process-mining data from that customer. [Likely]
    - What is the label: automations built, automations still running after 90 days, or hours saved?
    - How do you recommend for a new tenant with no automation history?
23. Design "people you may know" or who-to-follow recommendations. [Bank] [1], [5], [8]
    - Recast: how would the same graph approach recommend which colleague or Centre of Excellence member should review an automation? [Likely]

### B2. LLM-assisted recommendation and personalisation

Tag basis: [Bank] [1]; [Reported-2nd] [11].

```
 user profile + history + memory --> [ ? retrieval (classic) ] --> candidates
                                                                      |
                              [ ? LLM as ranker / feature generator / explainer ]
                                                                      |
                                                                      v
                                                   ranked + explained recommendations
```

UiPath recast: Autopilot suggests which agent or automation to build, with a natural-language reason the Centre of Excellence lead can audit.

24. Design an LLM-based recommendation or personalisation system, using the LLM as a ranker or as a feature generator. [Bank] [1]
    - When is an LLM ranker worth its cost and latency compared with a gradient-boosted ranker?
    - How do you test that the LLM's explanations are faithful to the real ranking signal?
25. Design the personalisation and memory system for a chatbot. (Microsoft MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [11]
    - What should be remembered, for how long, and who can delete it?
    - Recast: per-user and per-tenant memory for UiPath agents under the AI Trust Layer's retention rules. [Likely]
26. Design a system that recommends reusable workflow templates while a user describes a task in natural language. [Likely]

---

## C. Anomaly and fraud detection

### C1. Transaction fraud and account takeover with humans in the loop

Tag basis: [Bank] [1], [8]; [Reported-2nd] [15]. Peer anchor in [9]: Uber "Project RADAR: Intelligent Early Fraud Detection System with Humans in the Loop" (2022).

```
 events / transactions --> [ ? real-time features ] --> [ ? scorer ] --> allow
                                    ^                        |   \--> step-up / block
 entity history / graph ------------+                        v
                                           [ ? human review queue ] --> labels
                                                             |              |
                                    [ ? retraining + drift monitor ] <------+
```

UiPath recast: detect fraud or policy breaches in invoices and payments that Document Understanding and finance automations process, such as duplicate invoices, vendor-bank-detail changes and split purchase orders. Flagged items go to Action Center.

27. How would you build a fraud detection system from labelled purchase records? [Bank] [8], [1]
    - How do you train when confirmed fraud labels arrive weeks late and are biased by what was reviewed?
    - How do you set thresholds when the review team's capacity is fixed?
28. Design an account-takeover detection system that catches the takeover at the moment the attacker tries a hazardous action. (Stripe, Affirm, Robinhood, Roblox, Uber; last reported Oct 2025) [Reported-2nd] [15]
    - What latency budget does a "hazardous action" check have, and which features must be precomputed?
    - Recast: detect a hijacked robot or agent credential at the moment it tries to change vendor bank details. [Likely]
29. Design an anomaly detection system for invoices flowing through an accounts-payable automation, so that suspicious invoices are held before payment. [Likely]
    - Which signals come from the document itself, and which from the vendor's history?
    - How would you explain a hold to an auditor?
30. Design a system that detects when the automation itself is being misused, for example a robot account used to move money outside policy. [Likely]

### C2. Abuse and anomaly detection in behaviour logs

Tag basis: [Likely]. Peer anchors in [9]: LinkedIn "Detecting and Preventing Abuse on LinkedIn using Isolation Forests" (2019), LinkedIn "Preventing Abuse Using Unsupervised Learning" (2020), LinkedIn "Using deep learning to detect abusive sequences of member activity" (2021).

```
 activity / telemetry streams --> [ ? sessionise + featurise ] --> [ ? unsup. / seq model ]
                                                                         |
                                     [ ? alert triage + dedup ] <--------+
                                                |
                                                v
                                  investigations --> labels --> [ ? semi-sup. retrain ]
```

UiPath recast: anomaly detection on robot and agent run logs in Orchestrator. Flag runs whose step sequence, duration or error pattern drifts. These often signal that a target application's UI changed and the robots are about to break.

31. Design an unsupervised anomaly detection system for event sequences, with almost no labels at the start. [Likely]
    - How would you move from isolation-forest style scores to a supervised model as investigation labels arrive?
32. Design a system that predicts robot or agent run failures before they happen, from execution logs and screenshots. [Likely]
    - How do you tell "the target app changed" apart from "the robot logic is wrong"?
    - What alert precision do on-call operators need before they trust the system?
33. Design a monitoring system that finds process-level anomalies (bottlenecks, rework loops, unusual paths) in process-mining event logs across thousands of process variants. [Likely]

---

## D. RAG enterprise assistants and agents

### D1. Chat with your docs / enterprise Q&A with permissions

Tag basis: [Bank] [1], [8], [22]; [Reported-2nd] [11], [21].

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

34. Design a RAG document Q&A ("chat with your docs") system that can extract answers from a large collection of documents. [Bank] [1], [8]
    - When would you pick RAG, fine-tuning, long context or tools and memory for this problem? [1]
    - How do you handle questions the corpus cannot answer?
35. Design an enterprise RAG assistant over 10M documents with per-user permissions. (Labelled OpenAI, Microsoft, Amazon, Databricks, Scale AI by the bank's maintainer) [Bank] [22]
    - How do you enforce document-level access control at retrieval time, not after generation?
    - How do you keep the index fresh when source permissions change hourly?
36. Design a Copilot feature that answers questions over a user's work email, documents and meetings without leaking content the user cannot see. (Labelled Microsoft) [Bank] [22]
37. Design a RAG system with agentic tools. (Microsoft MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [11]
    - When should the agent call a tool instead of retrieving, and how do you evaluate that choice?
38. A RAG system is hallucinating in production. How do you tell whether the retriever or the generator is at fault? (Perplexity MLE) [Reported-2nd] [21]
    - Which traces would you need to have logged in advance to answer this within a day?
39. Design a RAG assistant that answers employees' "how do I do X in system Y" questions and can hand off to an automation that does X. [Likely]
    - Where does the boundary sit between answering a question and executing an action?
40. Design retrieval over scanned and semi-structured documents (invoices, contracts, forms), where layout carries meaning. [Likely]
    - How do you chunk a table or a multi-page form without breaking the facts in it?

### D2. Question answering over structured and unstructured data

Tag basis: [Reported-2nd] [13]; [Bank] [22].

```
 question --> [ ? router: SQL | docs | both ] --> [ ? schema linker ] --> [ ? SQL gen + validate ] --> rows
                         |                                                                           |
                         +--> [ ? doc retriever ] --> passages --> [ ? answer composer + citations ] <+
```

UiPath recast: an agent that answers "which vendors missed SLA last quarter, and what did their contracts say?" by joining process-mining tables with contract PDFs extracted by Document Understanding.

41. Design a chatbot that answers from both structured tables and unstructured documents. (Google MLE loop; reports 2025 Q4-2026 Q3) [Reported-2nd] [13]
    - How do you evaluate an answer that is half a SQL result and half a cited passage?
42. Design a Text-to-SQL system over a warehouse with thousands of tables. (Labelled Databricks, Palantir) [Bank] [22]
    - How do you find the right 5 tables out of 5,000 before generation?
    - Which generated queries must a human approve before they run?

### D3. Customer-support assistant with human fallback

Tag basis: [Bank] [1], [8], [22]; [Reported-2nd] [12].

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

43. Architect an AI chatbot, covering intent detection, dialogue state, generation, fallback and continuous learning. (Amazon MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [12]
    - What confidence signal decides the handoff, and how do you calibrate it?
44. Design a customer-support agent that can take real actions, with escalation to humans. (Labelled Sierra) [Bank] [22]
    - Which actions need a human approval step, and how do you decide?
45. Build a chatbot to help people book hotels. [Bank] [8]
    - Recast: how does the design change when "booking" is an enterprise transaction a robot executes in SAP? [Likely]
46. Design a system that reads high-volume inbound email for a bank's operations team, routes each message, extracts fields and triggers the right automation. [Likely]
    - How do you add a new intent without retraining everything?
    - How do you measure automation rate without rewarding wrong automations?

### D4. Agentic assistant (planning, tools, memory)

Tag basis: [Bank] [1], [22]; [Reported-2nd] [20].

```
 goal --> [ ? planner ] --> [ ? tool / robot selection ] --> [ ? executor ]
             ^                                                    |
             |                 [ ? memory: short / long ] <-------+
             |                                                    v
             +------------ [ ? verifier + HITL checkpoint ] <-- results
```

UiPath recast: this is the core UiPath agentic-automation problem (agents that orchestrate robots, APIs and people). Chapter 04 covers it in depth, so this chapter keeps only the peer versions.

47. Design an agentic workflow or AI assistant covering planning, tool use and memory. [Bank] [1]
    - How do you detect and stop loops and hallucinated tools? [1]
    - Where do you put human-in-the-loop checkpoints?
48. Design an agent that operates a web browser to complete multi-step tasks. How do you make it reliable enough to ship? (Labelled Amazon) [Bank] [22]
    - Recast: the same agent drives a legacy desktop app over Citrix with no DOM. What changes in perception, recovery and evaluation? [Likely]
49. Design a stock-trading agent system. (Databricks; last reported Apr 2026) [Reported-2nd] [20]
    - Which actions are irreversible, and what limits, approvals and kill switches sit around them?
50. Design a code assistant or coding agent that uses RAG over a repository, tool use and verification. [Bank] [1]
    - Recast: how would the same verifier pattern apply to an agent that generates automation workflows? [Likely]

---

## E. Content moderation and safety

### E1. Harmful and policy-violating content detection

Tag basis: [Bank] [1], [4], [22]; [Reported-2nd] [14], [16], [21].

```
 post / ad (text | image | video) --> [ ? per-modality encoders ] --> [ ? fusion ]
                                                                         |
                                         [ ? multi-label / multi-task heads ]
                                                                         |
                                         remove | demote | allow | send to review
 user reports + reviewer labels --> [ ? label pipeline + retrain ] -------^
```

UiPath recast: screen what agents send out (emails, chat replies, generated documents) and what they take in (inbound documents and emails that may carry prompt injection, PII or toxic content) inside the AI Trust Layer.

51. Design a system that monitors posts, detects harmful content and demotes or removes it (about 500M posts a day and about 10K labelled examples). [Bank] [4]
    - Which clarifying questions in [4] (harm types, languages, user reports, explainability, legal constraints) change the design most?
    - How do you get to a usable model from 10K labels?
52. Design a content moderation system, given traffic data. (Google L6 MLE) [Reported-2nd] [21]
53. Design detection of weapon sales in ads, from data collection and labelling to the enforcement pipeline. (Meta MLE loop) [Reported-2nd] [14]
    - How do you handle sellers who change wording and images to evade the classifier?
54. Design copyright detection for uploaded images, including multi-image posts, adversarial overlaid text and photos of copyrighted photos. (Meta MLE loop) [Reported-2nd] [14]
    - Recast: detect a re-photographed or re-scanned copy of an invoice that was already paid. [Likely]
55. Design a content-moderation system that combines classifiers and LLMs. (Labelled Meta) [Bank] [22]
    - Which decisions go to the cheap classifier, which to the LLM, and which to a human?
56. A spam classifier shows strong offline metrics but misses a large share of spam in live traffic. Investigate and redesign it for production. (Indeed; last reported Aug 2026) [Reported-2nd] [16]
    - Which of label shift, adversarial drift, sampling bias and feature skew would you check first, and how?
57. Design an output guardrail that checks every outbound message an enterprise agent writes for PII, policy breaches and hallucinated commitments before it is sent. [Likely]
    - What is the latency budget, and which checks can run asynchronously?
    - How do you measure the false-block rate customers will tolerate?

### E2. Prompt-injection, data-leak and memorisation defence

Tag basis: [Bank] [1], [22]; [Reported-2nd] [14].

```
 untrusted inputs (email, web, docs) --> [ ? input scanner ] --> agent context
                                                                     |
 tool calls --> [ ? policy engine / allow-list ] --> [ ? output scanner ] --> world
                                    |
                                    v
                           audit log --> [ ? red-team + eval set ]
```

UiPath recast: agents that read customer emails and documents and then act in ERP systems give an injected instruction a direct path to a real transaction.

58. Design guardrails for a consumer-facing assistant. Do you filter inputs, outputs or both? (Labelled Sierra, Character.AI) [Bank] [22]
59. Design a defence against prompt injection for an agent that reads untrusted documents and can call tools. [Likely]
    - Which defences are ML classifiers, and which must be deterministic policy?
    - How do you red-team it continuously as attacks change?
60. Design a way to stop a code-generation model from reproducing private-repository code it was trained on. (Meta MLE phone screen) [Reported-2nd] [14]
    - Recast: how do you stop a model fine-tuned on several tenants' documents from reproducing one tenant's data for another? [Likely]
61. Design a PII detection and redaction service that sits between enterprise data and third-party LLM APIs. [Likely]
    - How do you keep redaction from destroying the information the LLM needs to do the task?

---

## F. LLM and agent evaluation platforms

### F1. Evaluation platform for LLM, RAG and agent features

Tag basis: [Bank] [1], [22]; [Reported-2nd] [12], [19].

```
 prod traces + curated tasks --> [ ? golden / regression sets ] --> [ ? runners / sandboxes ]
                                                                        |
         [ ? metrics: exact, retrieval, judge, pairwise, human ] <------+
                         |
                         v
           [ ? dashboards + release gates ] --> ship / block
                         ^
 judge calibration vs human labels -----+
```

UiPath recast: one evaluation platform for every agent, Document Understanding model and Autopilot feature, which product teams use to gate releases and customers use to check their own agents before deployment.

62. Design an evaluation system for a RAG product covering retrieval quality and answer faithfulness. [Bank] [1]
    - How do you build the golden set, and how do you keep it from going stale?
63. Design an LLM-as-judge evaluation. What are its known biases, and how do you correct for them? (Labelled Perplexity) [Bank] [22]
    - How do you show that your judge agrees with human raters, and how do you track drift?
64. How would you evaluate an agent, as opposed to a single model response? (Labelled Moonshot AI, Zhipu AI, Scale AI, Cognition) [Bank] [22]
65. Design an evaluation framework that checks whether a text-to-image system renders a requested brand name correctly. (Amazon MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [12]
    - Recast: how would you check that a DU model or an agent reproduces exact strings (vendor names, IBANs, amounts) without silent edits? [Likely]
66. Evaluate agents that generate or optimise GPU-kernel code: check candidates against a trusted reference on test cases and pick the acceptable one with the fewest operators. (AMD; last reported Sep 2026) [Reported-2nd] [19]
    - Recast: how would you verify that an agent-generated automation workflow is equivalent to a reference workflow before it runs on production systems? [Likely]
67. Design an offline-to-online evaluation pipeline for an LLM feature: offline evaluation, online A/B tests and regression testing before shipping. [Bank] [1]
68. How would you evaluate a meeting-summarisation feature before shipping it to a hundred million users? (Labelled Microsoft) [Bank] [22]
69. Design an evaluation platform for multi-step agents, where success depends on the final state of external systems, not on the text. [Likely]
    - How do you build sandboxed replicas of enterprise apps to score agent runs?
    - How do you give credit or blame to individual steps in a 30-step trajectory?
70. Design a platform that lets enterprise customers evaluate their own agents on their own data without their data leaving their tenant. [Likely]
    - What evaluation data can you share across tenants, and what must never leave its tenant?

### F2. Monitoring LLM systems in production

Tag basis: [Bank] [1]; [Reported-2nd] [21].

```
 requests --> [ ? tracing (per step) ] --> [ ? online metrics + sampled judging ]
                                                       |
                         [ ? drift / cost / latency alerts ] --> on-call
                                                       |
                         [ ? triage --> new eval cases ] --> eval set (F1)
```

UiPath recast: observability for agent runs in Orchestrator, with traces that join LLM calls, robot steps and human approvals.

71. Design production monitoring for an LLM application: hallucination rate, retrieval hit rate, latency, cost and user feedback. [Bank] [1]
    - Which metrics can you compute on every request, and which only on a sample?
72. A chatbot's accuracy dropped from 95% to 80% in six weeks. What do you do? (Databricks MLE) [Reported-2nd] [21]
    - Recast: the drop coincides with a silent foundation-model update by the provider. How would your system have caught it on day one? [Likely]
73. Design tracing and root-cause analysis for agent failures that span LLM calls, robot actions and human approvals. [Likely]
    - How do you turn a production failure into a regression test automatically?

---

## G. Serving, cost, data and platform (cross-cutting)

### G1. Model routing, gateways and cost control

Tag basis: [Bank] [1], [22].

```
 request --> [ ? difficulty / risk classifier ] --> small model --+
                         |                                        +--> [ ? cache ] --> response
                         +-----------------------> large model --+
                                                     |
                               [ ? cost + quality feedback --> router retrain ]
```

UiPath recast: a gateway that serves many tenants' agents across several LLM providers, with per-tenant cost caps and data-residency rules.

74. Design a routing layer that sends simple requests to small models and hard ones to large models. [Bank] [1]
    - How do you train the router without labels for "this request needed the large model"?
75. Design the routing layer that decides whether a request is handled on-device or in the cloud. (Labelled Apple) [Bank] [22]
    - Recast: decide whether a step runs in a local robot model, a customer-hosted model or a cloud LLM, given data-residency rules. [Likely]
76. Should you self-host or call an API for a given LLM workload? Design the decision and the serving stack for each option. [Bank] [1]
77. Design an LLM gateway: routing across providers, failover, caching, budgets and rate limits. (Labelled Perplexity, Palantir) [Bank] [22]
    - How do you degrade gracefully when one provider has an outage? [1]
    - How do you attribute cost to tenants and enforce data-residency per tenant? [Likely]
78. Design a multi-tenant inference platform that serves many foundation models to thousands of customers. (Labelled Amazon) [Bank] [22]

### G2. Document-to-structure pipelines and batch generation

Tag basis: [Bank] [1], [22]; [Reported-2nd] [17].

```
 uploaded / archived docs --> [ ? batch + priority scheduler ] --> [ ? layout + OCR / VLM workers ]
                                                                            |
                                        [ ? structure / field output + confidence ]
                                                                            |
                              [ ? safety + sampled QA ] --> outputs + audit --> [ ? HITL fixes ]
```

UiPath recast: Document Understanding and IXP at scale, such as overnight extraction over millions of archived documents during a customer's migration. Chapter 08 covers the PDF-tagging version end to end.

79. Design a production service that converts uploaded PDF files into Markdown. (Mistral AI; last reported Apr 2026) [Reported-2nd] [17]
    - How do you handle reading order, tables and multi-column layouts, and how do you measure them?
    - How do you price and schedule a 2,000-page upload against many 2-page ones?
80. Design a document-intelligence pipeline that takes scanned PDFs in and returns structured fields, at 10M documents. (Labelled Palantir) [Bank] [22]
81. Design content generation or summarisation at scale, with batch processing and safety filtering. [Bank] [1]
    - How do you estimate cost and throughput before you run it?

### G3. Training-data quality and model-building pipelines

Tag basis: [Reported-2nd] [11], [21]; [Bank] [23].

```
 raw data (OCR text, logs, user tasks) --> [ ? filters: dedup, quality, PII, contamination ]
                                                       |
                              [ ? labelling / synthetic data ] --> [ ? train + eval ]
                                                       |
                         [ ? data-quality dashboards + audits ] <------+
```

UiPath recast: build training sets for DU, Communications Mining and agent models from customer corrections in Action Center, under tenant consent and data-use rules.

82. Design a pipeline to clean and filter OCR-extracted text for language-model training. (Microsoft MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [11]
    - Which quality signals can you compute without a human label?
83. Design a quality check for an LLM training dataset of spreadsheet-copilot tasks. (Microsoft MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [11]
    - Recast: quality-check a dataset of (natural-language request, generated workflow) pairs for an Autopilot workflow generator. [Likely]
84. Design a video vision-language model from data collection through evaluation. (Microsoft MLE loop; reports 2026 Q1-Q3) [Reported-2nd] [11]
    - Recast: a screen-recording model that turns a user's recorded task into an automation. [Likely]
85. A platform logs billions of assistant requests a month. How would you use this data to improve its speech recognition? (Apple, AI role) [Reported-2nd] [21]
    - Recast: how would you use millions of human validations in Action Center to improve DU models without breaching tenant boundaries? [Likely]
86. For a new domain, would you fine-tune or use a complex prompting strategy? Design the decision process. (Anthropic MLE) [Reported-2nd] [21]
87. Roll out a new embedding model over an index holding billions of vectors. (Labelled Glean) [Bank] [23]
    - How do you re-embed without downtime, and how do you compare old and new indices online?

---

## H. Classic NLP and vision prompts that recast directly

### H1. Text classification, language identification and entity linking

Tag basis: [Bank] [1], [8].

```
 text --> [ ? normalise + tokenise ] --> [ ? encoder ] --> [ ? task head ] --> label / entity
 labels (sparse, noisy) --> [ ? active learning + weak supervision ] ---^
```

UiPath recast: Communications Mining taxonomies (intents and fields defined per customer) and entity linking to master data such as vendor IDs and customer accounts.

88. Design a system to predict the language a text is written in. [Bank] [8], [1]
89. Design a named-entity linking system. [Bank] [1]
    - Recast: how would you link an extracted vendor name on an invoice to the right record in a customer's ERP? [Likely]
90. How would you map nicknames (Pete, Andy, Nick, Rob) to real names? [Bank] [8]
    - Recast: how would you resolve the same customer across email, CRM and invoice records? [Likely]
91. Design a text classifier where each enterprise customer defines its own label taxonomy with only a few dozen examples per label. [Likely]
    - Fine-tuned small model, LLM prompting or embeddings with a light head: how do you choose for each tenant?

### H2. OCR, object detection and image search

Tag basis: [Bank] [1], [5].

```
 image / screenshot --> [ ? detector (regions / UI elements) ] --> [ ? recogniser (OCR) ]
                                                                         |
                                     [ ? layout / structure model ] --> fields | elements
```

UiPath recast: Document Understanding (scanned documents to fields) and UI-automation computer vision (screenshots to clickable elements for robots and computer-use agents).

92. Design an OCR / text recognition system. [Bank] [1]
93. Design an object detection system. [Bank] [5]
    - Recast: how would you detect and classify UI elements in screenshots of remote-desktop sessions (such as Citrix) where no DOM is available? [Likely]
94. Design an image search system, or a multimodal (text plus image) search system. [Bank] [1], [5]
    - Recast: how would you find "the same screen" across app versions to make robot selectors robust? [Likely]
95. Design an end-to-end document extraction system for invoices from thousands of vendors with different layouts, where an extraction error has a financial cost. [Likely]
    - Where do you place confidence thresholds and human validation?
    - How do validated corrections flow back into training without one tenant's data leaking to another?

### H3. Prediction and estimation prompts

Tag basis: [Bank] [1], [7], [8]; [Reported-2nd] [14].

```
 historical events + context --> [ ? features ] --> [ ? regressor / survival model ] --> estimate
                                                                       |
                                    [ ? calibration + uncertainty ] ---+
```

UiPath recast: predict how long a process instance will take, or when a case will breach its SLA, from process-mining event logs.

96. Design a delivery-time estimation system. [Bank] [1], [7]
    - Recast: how would you predict case completion time and SLA breach risk from process-mining data? [Likely]
97. For a streaming service, predict whether a user stopped watching a show from fatigue or only to take a break. [Bank] [8]
    - Recast: how would you predict when a customer will stop using an automation or churn from the platform? [Likely]
98. Design an ads ranking system and explain how you calibrate predicted click and conversion probabilities. (Meta MLE loop) [Reported-2nd] [14]
    - What transfers from CTR calibration to calibrating the confidence of a DU extraction or an agent's "I am done" signal? [Likely]

---

## I. Principal-level stress probes (apply to any prompt above)

These are not standalone prompts. Use them after any family above to check your answer against the principal bar. All are [Likely].

99. Which two decisions in your design are hardest to reverse, and what evidence would you gather before committing to them? [Likely]
100. Where in this design should ML stop and deterministic logic (rules, robots, workflows) take over? [Likely]
101. Estimate cost per 1,000 requests and the main cost driver. What would you cut first if the budget were halved? [Likely]
102. How does the design change when every customer is a separate tenant with its own data, labels, taxonomy and compliance regime? [Likely]
103. What would you build first in 90 days, what would you measure, and what would make you stop? [Likely]
104. Which parts of this system would you turn into a shared platform for other teams, and which should stay product-specific? [Likely]
105. How would you explain this system's failure modes to a regulated customer's auditor? [Likely]
106. If a competitor shipped a frontier-model-only version of this system tomorrow, which parts of your design would still be an advantage? [Likely]

---

## Coverage summary

| Topic family | Questions | [Reported-2nd] | [Bank] | [Likely] | [Reported] |
|---|---|---|---|---|---|
| A. Search and ranking | 1-17 | 5 | 8 | 4 | 0 |
| B. Recommendation | 18-26 | 3 | 4 | 2 | 0 |
| C. Anomaly and fraud detection | 27-33 | 1 | 1 | 5 | 0 |
| D. RAG enterprise assistants and agents | 34-50 | 5 | 9 | 3 | 0 |
| E. Content moderation and safety | 51-61 | 5 | 3 | 3 | 0 |
| F. LLM and agent evaluation platforms | 62-73 | 3 | 6 | 3 | 0 |
| G. Serving, cost, data and platform | 74-87 | 6 | 8 | 0 | 0 |
| H. Classic NLP and vision prompts that recast directly | 88-98 | 1 | 8 | 2 | 0 |
| I. Principal-level stress probes | 99-106 | 0 | 0 | 8 | 0 |
| **Total** | **106** | **29** | **47** | **30** | **0** |

## References

1. "ML System Design" (9-step formula; Section 2 sample questions including GenAI/LLM systems; Section 3 GenAI/LLM system design topics), alirezadir/Machine-Learning-Interviews, GitHub, accessed October 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
2. "ML System Design Template" (9 steps), alirezadir/Machine-Learning-Interviews, GitHub, accessed October 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-template.md
3. "Search System" (e-commerce search design, 50M DAU, clarifying questions), alirezadir/Machine-Learning-Interviews, GitHub, accessed October 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-search.md
4. "Harmful Content Detection" design (500M posts a day, about 10K labelled examples, clarifying questions), alirezadir/Machine-Learning-Interviews, GitHub, accessed October 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-harmful-content.md
5. MLSD case-study directory (files for video and event recommendation, typeahead, newsfeed, PYMK, image search, object detection, ads ranking, harmful content), alirezadir/Machine-Learning-Interviews, GitHub, accessed October 2026. https://github.com/alirezadir/Machine-Learning-Interviews/tree/main/src/MLSD
6. "ML Companies" reading list, alirezadir/Machine-Learning-Interviews, GitHub, accessed 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-companies.md
7. "machine-learning-interview" (ML design section: YouTube recommendation, LinkedIn feed ranking, ad click prediction, delivery time, Airbnb search ranking), khangich, GitHub, accessed October 2026. https://github.com/khangich/machine-learning-interview
8. Chip Huyen, "Machine Learning Systems Design: Exercises" (27 open-ended design questions), chiphuyen/machine-learning-systems-design, GitHub, 2019, accessed October 2026. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md
9. Eugene Yan, "applied-ml: Papers and tech blogs by companies sharing their work on data science and machine learning in production" (source of the peer-anchor titles; all 13 cited titles confirmed), GitHub, accessed October 2026. https://github.com/eugeneyan/applied-ml
10. alirezadir, "Machine-Learning-Interviews" (repository overview; notes on FAANG versus startup interviews), GitHub, accessed October 2026. https://github.com/alirezadir/Machine-Learning-Interviews
11. TrueInterview (mirrored), "Microsoft Machine Learning Engineer interview process" (sample ML system design questions; "candidate-reported interview experiences, primarily from 2026 Q1 through 2026 Q3"), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/microsoft/interview-process-machine-learning-engineer.md
12. TrueInterview (mirrored), "Amazon Machine Learning Engineer interview process" (based on 20 candidate-reported experiences, 2026 Q1-Q3), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
13. TrueInterview (mirrored), "Google Machine Learning Engineer interview process" (based on 13 candidate-reported experiences, 2025 Q4-2026 Q3), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/google/interview-process-machine-learning-engineer.md
14. TrueInterview (mirrored), "Meta Machine Learning Engineer interview process" (based on 25 candidate-reported experiences, 2025 Q4-2026 Q3), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/meta/interview-process-machine-learning-engineer.md
15. TrueInterview (mirrored), "Design an Account Takeover Detection System" (asked at Stripe, Affirm, Robinhood, Roblox, Uber; last reported Oct 2025), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/system-design/account-takeover-prediction-system/README.md
16. TrueInterview (mirrored), "Redesign a spam classifier for production" (asked at Indeed; last reported Aug 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/system-design/redesign-a-spam-classifier-for-production/README.md
17. TrueInterview (mirrored), "Design a PDF-to-Markdown inference API" (asked at Mistral AI; last reported Apr 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/system-design/design-a-pdf-to-markdown-inference-api/README.md
18. TrueInterview (mirrored), "Rank homepage modules for an e-commerce product" (asked at Wayfair; last reported Sep 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/system-design/rank-homepage-modules-for-an-e-commerce-product/README.md
19. TrueInterview (mirrored), "Evaluate agents that generate or optimize kernels and compiler code" (asked at AMD; last reported Sep 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/system-design/evaluate-agents-that-generate-or-optimize-kernels-and-compiler-code/README.md
20. TrueInterview (mirrored), Databricks company page ("Stock Trading Agent System Design", reported Apr 2026), kevin-2023-code/Tech-Interview-Questions, GitHub, accessed October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/databricks/README.md
21. Alexey Grigorev, "AI Engineering Field Guide: discussion threads" (curated Reddit, TeamBlind and X interview threads with links to the original posts; the original posts were not opened in this run), alexeygrigorev/ai-engineering-field-guide, GitHub, accessed October 2026. https://github.com/alexeygrigorev/ai-engineering-field-guide/blob/main/interview/data/sources/discussion-threads.md
22. pallavi-shekhar, "AI engineering interview questions, company-wise" (says it is "compiled from publicly reported interview experiences"; the "Asked at" labels are the maintainer's, with no per-question source), GitHub, accessed October 2026. https://github.com/pallavi-shekhar/ai-engineering-interview-questions-company-wise
23. ombharatiya, "AI Companies Interview Questions" (says it is compiled from candidate reports, blogs and guides; only some items are labelled as reported), FAANG-Coding-Interview-Questions, GitHub, accessed October 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
24. Subrata Roy, "Interview experience: Amazon Applied Scientist 2 (ML + DSA)", LinkedIn post archived in snknitin/LLMwiki (bookmarked 2026-03-20), GitHub, accessed October 2026. https://github.com/snknitin/LLMwiki/blob/main/Bookmarks/LinkedIn/2026-03-20%20-%20subrata-roy%20-%20interview-experience-amazon-applied-scientist-2-ml-dsa%20-%20743936.md

Reference [6] was read for context, but no question cites it.
