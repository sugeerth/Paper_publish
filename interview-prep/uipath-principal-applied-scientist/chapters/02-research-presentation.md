# 02. Research Presentation Round: Presenting Your ML System, and the Design Probes Principal Scientists Get Afterwards

> **How to read this chapter.** You asked for ML system design, so this chapter treats the research or job-talk presentation as a system design round you lead yourself. It covers how to structure the talk around architecture diagrams, and the probing questions a panel is likely to ask about your system's design, evaluation, deployment and transfer to UiPath products. General research-talk questions (publication history, theory) are included only where they lead into design.
>
> **Tagging system** (the same as Chapter 01):
> - **[Reported] [n]**: a candidate report, question bank or official page links the question, or this round type, to UiPath. The number points to the References list.
> - **[Likely]**: an inference from the role, the principal level and UiPath's AI products (agentic automation, Document Understanding, Communications Mining, Autopilot, the AI Trust Layer, computer vision for UI automation, process mining). No candidate has reported it.
> - Indented sub-bullets are follow-up probes, phrased as questions.
>
> **Research limits (please read).** Two runs have now tried to source this chapter, and neither could do live research. In the first run the workflow's shared web-search budget (200 searches per turn) was already used up. In the second (review) run the search budget was again exhausted, and direct fetches of glassdoor.com, teamblind.com, uipath.com, docs.uipath.com and arxiv.org all failed with DNS errors (ENOTFOUND). No cited URL could be verified, and no new candidate-reported question could be found. So **no question below is tagged [Reported]**. Every question is **[Likely]**, and the format notes describe how principal-scientist job talks usually run at enterprise AI companies; they are not a confirmed UiPath process. Ask your recruiter for the presentation format, the length and who sits on the panel. A follow-up run with working search could upgrade tags to [Reported].
>
> **Review changes.** The review run removed generic and duplicate questions (for example, separate PII, mentoring and "what would you do differently" questions that repeated others). It also added principal-level stress tests that probe specific arrows on your architecture diagram (Section L) and moved general agent-design questions that Chapter 04 already covers out of this chapter.

## Format notes (inferred, not confirmed)

| Item | Typical for principal applied scientist job talks | Status |
|---|---|---|
| Duration | 60 min: about 30-40 min talk, 20-30 min Q&A; some companies interrupt during the talk | Likely |
| Topic | Your own past work, usually one or two projects that reached production; sometimes a "vision for X at our company" section | Likely |
| Audience | Panel of 3-8: scientists (staff/principal), ML or platform engineers, a product lead, often the hiring manager | Likely |
| Materials | Slides shared in advance or presented live; confidential details abstracted | Likely |
| Debrief weight | Often a heavy signal, because every panel member sees the same material | Likely |

**What the principal bar likely assesses in this round**

- **Depth and rigour.** Sound baselines, ablations and error analysis; you know where the method breaks.
- **Research-to-product conversion.** A clear path from a result to a system that others operate, with SLOs, costs and monitoring.
- **System design ownership.** You can draw the architecture, defend each box and say what you would change today.
- **Scope and influence.** The work changed a roadmap, a platform or several teams, not only one model.
- **Communication.** Mixed audiences (scientists, engineers, product) all follow the talk.
- **Transfer.** You can map the work onto UiPath problems: agents that call robots and tools, document and message understanding, UI vision, enterprise trust and governance.

### Suggested talk structure built around system diagrams

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 1. Problem & business stakes      (2-3 min)  who hurt, how much, SLOs     │
 │ 2. Why it was hard                (3 min)    data, scale, constraints      │
 │ 3. SYSTEM ARCHITECTURE DIAGRAM    (5-7 min)  boxes, data flow, owners      │
 │ 4. Key technical idea(s)          (8-10 min) the research contribution     │
 │ 5. Evaluation                     (5 min)    offline -> online -> business │
 │ 6. Production & failures          (4 min)    what broke, what you changed  │
 │ 7. Impact & scope                 (2 min)    metrics, teams, roadmap       │
 │ 8. If I did it at UiPath          (3 min)    mapping + next 12 months      │
 └───────────────────────────────────────────────────────────────────────────┘
```

### One architecture slide the panel will probe (template)

```
  Data sources ──> Ingest / labelling ──> Training & eval ──> Model registry
       │                  │                     │                   │
       │            (active learning)     (offline bench)           v
       │                  ^                                   Serving / agent runtime ──> Robots / tools / APIs
       │                  │                                         │
       └──── feedback ────┴──── human review / corrections <────────┤
                                                                    v
                                         Monitoring: quality, drift, cost, latency, audit log
```

Expect probes at every arrow: who owns it, what fails there, how you measure it, and what it costs.

### Second slide worth preparing: evaluation and feedback loop

```
  Production traffic ──> sampler (stratified by tenant, doc type, confidence)
          │                          │
          │                          v
          │                 human labels / reviewer corrections ──> golden set (versioned, frozen)
          │                          │                                    │
          v                          v                                    v
  shadow / canary model ──> offline + online metrics ──> release gate (thresholds, CIs, guardrails)
                                     │                                    │
                                     └──── error clusters ──> data / model / prompt fix ──┘
```

Panels at this level often skip the model and ask about this loop: how the sample is drawn, how reviewer corrections are de-biased, and what stops a release.

## Questions

### A. Framing and problem selection

1. [Likely] Why did you choose this project for your presentation, and what does it show about you at principal level?
   - What other project did you consider presenting, and why not that one?
2. [Likely] What was the business problem in one sentence, and who measured success?
   - What SLOs (accuracy, latency, cost per request, review rate) did you commit to?
3. [Likely] How did you decide this problem needed research rather than an off-the-shelf model?
   - What did the simplest baseline achieve, and how far was it from the target?
4. [Likely] What constraints (data access, privacy, latency, hardware, regulation) shaped the design most?
   - Which constraint did you push back on, and did you win?

### B. Architecture walk-through

5. [Likely] Walk us through the architecture diagram box by box. Which box did you design yourself?
   - Which box took the most engineering time, and was that expected?
6. [Likely] Where is the single point of failure in this design?
   - What happens to users when that component goes down?
7. [Likely] Which parts are learned models and which are deterministic code or rules, and why did you split them that way?
   - Did you ever move logic from a model into code, or the reverse?
8. [Likely] How does data flow from raw input to the final action, and where is it stored along the way?
    - Where does personally identifiable information appear, and how is it protected?
9. [Likely] What alternative architectures did you reject, and what evidence decided it?
10. [Likely] How did the system handle inputs far outside the training distribution?
    - How did it signal low confidence to the downstream system or user?
11. [Likely] What were the interfaces between your system and the teams upstream and downstream?
    - Who could change those interfaces, and how did you version them?
12. [Likely] Draw how the architecture would look if traffic grew 100x. What breaks first?
13. [Likely] How did you handle retries, idempotency and partial failure when the model triggered actions in other systems?

### C. Modelling and the core technical contribution

14. [Likely] What is the one technical idea in this work that someone else would not have tried?
    - Is it new in the literature, or new in how you applied it?
15. [Likely] Why this model family? What would a large general LLM have done on the same task?
    - Did you test prompting or fine-tuning a foundation model as a baseline?
16. [Likely] Walk us through the loss function or training objective, and why it fits the business metric.
17. [Likely] Which ablation surprised you most?
    - Which component could we remove with almost no loss?
18. [Likely] How did you get labels, and how noisy were they?
    - How did you measure label quality and annotator agreement?
19. [Likely] How much labelled data did you need, and how would that change for a new customer or domain?
    - Did you use active learning, weak supervision or synthetic data? How did you validate it?
20. [Likely] If you had to make the model ten times cheaper to serve, what would you do?
    - Distillation, quantisation, caching, smaller models per step: which did you try?

### D. Evaluation rigour

21. [Likely] What was your offline evaluation set, and how do you know it represented production traffic?
    - How often was it refreshed, and how did you stop leakage into training?
22. [Likely] Which baselines did you compare against, and why should we believe the comparison was fair?
    - Did baselines get the same tuning budget as your method?
23. [Likely] Report the confidence interval or variance on your headline number.
    - Was the improvement statistically and practically significant?
    - How much of the gain survives a change of seed, data ordering or hyperparameters?
24. [Likely] How did offline gains translate to online metrics, and where did they not?
25. [Likely] How did you evaluate the parts of the output that have no single correct answer (for example, generated text or agent plans)?
    - If you used an LLM as a judge, how did you calibrate it against humans?
26. [Likely] Show us the error analysis. What are the top three failure categories?
    - Which failure is most costly to the customer, rather than most frequent?
27. [Likely] How did you evaluate fairness or performance across subgroups, languages or customers?
28. [Likely] What did the A/B test or staged rollout look like, and what guardrail metrics did you watch?
    - What would have made you stop the rollout?

### E. Production, operations and failures

29. [Likely] What went wrong after launch that your offline evaluation did not predict?
    - How did you detect it, and how long did it take?
30. [Likely] How did you monitor model quality without ground-truth labels in production?
31. [Likely] How did you detect and handle data drift or concept drift?
    - Did you retrain on a schedule or on a trigger?
32. [Likely] What was the on-call or ownership model after launch, and who got paged?
33. [Likely] What did the system cost to run per month, and how did that compare to the value it created?
34. [Likely] How did you handle a rollback, and how fast could you do it?
35. [Likely] What did you hand over to the product or platform team: code, evaluation suite, runbooks, people?
    - Is the system still running without you?

### F. Agentic and LLM-system follow-ups

36. [Likely] If you rebuilt this system today with current LLMs and agents, what would you delete and what would you keep?
37. [Likely] How would you turn this system into an agent that calls tools, triggers robots and asks people for help?
    - Which actions would always need human approval?
38. [Likely] How would you evaluate an agent version of your system whose prompts, tools and underlying model change often?
    - How would you build a regression suite that runs on every change?
39. [Likely] How would you route between small, large, hosted and self-hosted models for different steps?
    - How would you stop quality dropping silently when the router changes?
40. [Likely] What part of your system would you replace with a vision-language model reading screens, and what risk does that add?

### G. Enterprise constraints and trust

41. [Likely] How would your system work for a multi-tenant enterprise product where each customer's data must stay isolated?
42. [Likely] How would you deploy this on-prem or in an air-gapped customer environment?
    - How would you update models and monitor quality there?
43. [Likely] How would the design change if customers bring their own LLM provider?
44. [Likely] How would you explain a decision your system made to an auditor or a regulator?
    - What would the audit log contain?
45. [Likely] How would you train or improve the model when some customers allow their data to be used and others forbid it?

### H. Mapping your work to UiPath products

46. [Likely] Which UiPath product area would benefit most from your work: agents, document understanding, communications mining, UI automation or process mining?
    - What is the first experiment you would run there in your first 90 days?
47. [Likely] How would your extraction or classification approach handle invoices, contracts and forms across many layouts and languages?
    - How would it produce field-level confidence for straight-through processing?
48. [Likely] How would your approach help customers build a new extractor or classifier with only a handful of examples?
49. [Likely] How would your work apply to a system that finds intents in high-volume email and then triggers automations?
    - How would you handle new intents nobody labelled?
50. [Likely] How would your work apply to UI automations that break when the target application changes?
51. [Likely] How would you decide whether UiPath should build this capability, buy it, or rely on a foundation-model provider?
52. [Likely] What would make your approach a mistake for UiPath's customers?

### I. Scope, influence and technical leadership

53. [Likely] What was your exact role, and what did others on the team do?
    - Which decisions would not have happened without you?
54. [Likely] How many teams adopted this work, and how did you get teams that did not report to you to use it?
55. [Likely] Describe a disagreement about this design with product or engineering. How was it resolved?
56. [Likely] What did this project change in your organisation's roadmap or platform?
57. [Likely] How did you balance publication goals with product deadlines on this project?
    - What did you publish, and what did you keep as product advantage?
58. [Likely] If you led this at UiPath, what would your 12-month roadmap look like, and how would you staff it?
59. [Likely] What would make you stop or pivot this line of work?

### J. Reflection and judgement

60. [Likely] What is the weakest part of your work, and how would a sceptical reviewer attack it?
61. [Likely] Which result in your talk do you trust least?
62. [Likely] Tell us about a design decision where you were wrong and a junior colleague or customer was right. (Humble)
63. [Likely] What was the boldest bet in this project, and how did you reduce its risk? (Bold)
64. [Likely] What detail did you only get right because you looked at raw data or sat with users? (Immersed)
65. [Likely] What did you leave out of version one to ship faster, and did it hurt later? (Fast)

### K. Presentation mechanics and clarification probes

66. [Likely] Can you go back to the architecture slide and explain the arrow between these two components?
67. [Likely] What does this metric mean in customer terms, for example hours saved or documents processed without review?
    - Can you explain the core idea to the product manager on the panel in two minutes, without jargon?
68. [Likely] Which numbers on your slides are from production, and which are from offline experiments?
69. [Likely] What part of this work is confidential, and can you describe it at a level that lets us judge it?

### L. Stress tests on your architecture diagram (principal-level)

70. [Likely] Point to the arrow on your diagram where training-serving skew is most likely. How did you detect it, and what contract stops it recurring?
    - Were features or embeddings computed by the same code path offline and online?
71. [Likely] Your confidence threshold decides what goes to straight-through processing and what goes to a human. Derive that threshold from the cost of a false accept, the cost of a review and the reviewer capacity.
    - What happens to the threshold when a new customer has a different cost of error?
72. [Likely] Human corrections feed back into training on your diagram. How do you stop that loop from only learning from the cases the model already flagged as uncertain (selection bias)?
    - How do you sample cases the model was confident about but got wrong?
73. [Likely] If you swap the embedding model behind a retrieval index, what must be rebuilt, how long does it take at your scale, and how do you serve traffic during the migration?
74. [Likely] Which box on your diagram dominates p99 latency, and which dominates cost? Are they the same box?
    - What would you trade to halve p99 without raising cost?
75. [Likely] Draw the GPU or accelerator capacity plan for peak load. How do you handle a customer whose batch job arrives at the same time as interactive traffic?
76. [Likely] Where on your diagram would you add a cascade (cheap model first, expensive model on escalation), and how do you set the escalation rule without hurting recall on rare, costly cases?
77. [Likely] Which component's output is consumed by another model downstream? How do you stop an upstream model update from silently breaking the downstream one?
    - Who signs off when the upstream team ships a new version?
78. [Likely] How is each artefact on your diagram versioned (data snapshot, labels, model, prompt, config, index), and can you reproduce last quarter's production output exactly?
79. [Likely] Your golden set was frozen at launch. How do you know it still represents traffic a year later, and how do you refresh it without making old and new scores incomparable?
80. [Likely] Show where a single customer with unusual data (one tenant, 40% of volume) would distort your aggregate metrics. How do you report quality so that this is visible?
81. [Likely] Which metric on your dashboard could improve while customers get worse outcomes? How did you guard against that?
82. [Likely] If one upstream data source silently changed its schema or encoding, which box would notice first, and how long would bad outputs flow before an alert?
83. [Likely] What is the blast radius of a bad model release on your diagram, and what mechanisms (canary, per-tenant rollout, kill switch) limit it?
84. [Likely] Which parts of your system would you keep if the core model were replaced tomorrow by a stronger general foundation model? Which parts are durable advantage?
85. [Likely] How would your diagram change if each customer needed its own fine-tuned variant? How do you train, store, serve and evaluate thousands of variants?
    - Adapters per tenant, prompt-only customisation, or shared model with retrieval: which and why?
86. [Likely] Where would an LLM step on your diagram need a deterministic validator (schema, business rule, cross-check against a system of record), and what happens when the validator rejects the output?
87. [Likely] Walk through one real incident on this system end to end: detection, diagnosis, mitigation and the permanent fix. Which box on the diagram changed afterwards?
88. [Likely] If you had to run this system with half the team, which boxes would you merge, outsource or delete?
89. [Likely] How would you prove to a sceptical panel that the research contribution, rather than more data or more compute, caused the headline improvement?
    - What is the compute-matched or data-matched baseline?
90. [Likely] What would a principal engineer on this panel say is over-engineered on your diagram, and how would you answer?
91. [Likely] Which design decision on this slide is hardest to reverse, and how much evidence did you have when you made it?

### M. Questions to ask the panel (reverse questions)

92. [Likely] Is the presentation expected to cover my own work, a given topic, or a vision for UiPath, and how is it weighted in the debrief?
93. [Likely] Who will be on the panel, and is Q&A held to the end or are interruptions expected?
94. [Likely] Which product area's problems would the person in this role own first?
95. [Likely] How does UiPath move applied science results into products today, and where does that stall?
96. [Likely] What does success look like for this role after one year?

## References

No source page could be retrieved in either run (see "Research limits" above), so no question cites a source. The official pages below are the right places to check the format notes and product context. They were not read in either run; the review run's fetch of uipath.com failed with a DNS error, so the links are given as unverified.

1. UiPath Careers, uipath.com (official careers page; unverified, fetch failed). https://www.uipath.com/company/careers
2. UiPath Blog, uipath.com (official product and AI announcements; unverified, fetch failed). https://www.uipath.com/blog
