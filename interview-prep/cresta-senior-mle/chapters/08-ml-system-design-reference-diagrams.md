# 08. ML System Design Reference Diagrams: Agent Assist, Voice AI Agent, Agent Evaluation, Model Deployment, Knowledge Assist RAG, Conversation Intelligence and Multi-Tenant LoRA Serving

> **Why this chapter exists.** Chapters 04 and 07 give a "Sketch" prompt for most design questions, but no drawn diagram. This chapter draws seven labeled box-and-arrow architectures for the core Cresta prompts. Under each diagram is a set of whiteboard questions an interviewer can ask about that drawing. The diagrams are reference skeletons to practice against. They are not Cresta's real architecture, and the questions have no written answers.
>
> **Tagging system (used on every diagram and question).**
> - **[Reported] [n]**: a candidate report or question bank attributes the prompt, or a close variant, to Cresta. [n] points to the References list. Where the report came from a different round or role, the tag says so.
> - **[Likely]**: inferred from Cresta's products, engineering blog and ML job postings. No candidate has reported it.
> - **Back-pointers**: "ch04 Q5" means question 5 in chapter 04 (ML System Design). "ch07 Q1" means question 1 in chapter 07 (Cresta Product and Research Context).
>
> **How to read the diagrams.**
> - Boxes are components. Arrows are numbered data flows: (1), (2) and so on.
> - The **budget** on an arrow is an illustrative latency target written for practice. It is an assumption, not a number published by Cresta, unless a citation is attached. Interviewers usually want you to state your own budget and defend it.
> - "async" marks a flow that is off the latency-critical path.
> - The diagrams are plain-text (ASCII) so that they render the same in Markdown, in a terminal and in a PDF export.
>
> **Research limits.** In the adversarial review of this chapter, 10 cited URLs were tried (PracHub ML system design list and eval-platform experience, three cresta.com blog posts, Fireworks Knowledge Assist story, Glassdoor, Blind, Built In Vancouver and 1point3acres). Every one failed: WebFetch returned DNS errors and the egress proxy refused the CONNECT (HTTP 403). The shared web-search budget was also used up, so no new candidate reports could be searched for. No URL could be shown to be dead, and none was shown to be live. Every citation is carried over from the reviewed excerpt lists in chapters 01, 04, 07 and 10. Only two Cresta ML system design prompts are on record: the agent evaluation platform and "How would you deploy a model in production?". Those two diagrams, and only their headline prompts, are tagged [Reported]. Follow-up questions under them are the guide's own extrapolation and are tagged [Likely]. All other diagrams are [Likely]. The Knowledge Assist diagram extends a [Reported] conceptual question from a Forward Deployed Engineer screen, and its tag says so.

## Format notes for this round

- **Length and slot.** Senior/Staff loop reports describe one hour of system design alongside an hour of coding and a hiring-manager chat [2]. A 2025-26 Senior MLE report lists a systems design round after a recruiter call, an LC-medium round and a "vibe coding" round [1].
- **Reported prompts.** "How would you deploy a model in production?" (Senior MLE) [1]. "Design an Evaluation Platform for AI Agents Across Prompt and Model Changes" (PracHub ML system design list, with a linked Cresta SWE interview experience) [4][5].
- **Recruiter-flagged scope.** A recruiter told one candidate the interviews would cover key concepts and fundamentals of LLMs, GenAI, RAG and AI agents [3]. A 2026 FDE screen asked "What is RAG, and how do you prevent hallucination?" [6].
- **Interviewer (inferred).** Most likely a senior or staff MLE from the target track: Agentic Assist, Agent and System Quality, Voice/ASR, Voice Experience or Insights [7][8][9][10][11].
- **Senior bar (from postings).** End-to-end ownership from research to production [11]. Evaluation of non-deterministic systems with offline benchmarks, online experiments and LLM-as-a-judge [8]. Diagnosis of hallucination, retrieval errors, tool misuse and context drift [7][8]. Design for latency, cost and real-time streaming inference in voice [9].
- **Where diagrams matter (inferred).** Expect to draw the architecture within the first 10-15 minutes, then defend one or two arrows in depth: the latency budget, the evaluation loop, multi-tenancy or the rollout path. Practice drawing each diagram below from memory in under five minutes.

## Questions

### Diagram 1. Real-time Agent Assist path [Likely]

Points back to: ch04 Q5 (Agent Assist), Q6 (moment detection), Q9 (inference graph), Q13 (delivery layer); ch07 Q1 (Agent Assist), Q8 (partial transcripts), Q46 (inference graphs), Q47 (voice stack). Sources for the product shape: [12][13].

```
 Customer <====== live call ======> Human agent
                     |
                     v
 +----------------------------------+
 | Telephony / CCaaS                |   SIP trunk, SIPREC fork or a
 | (carrier, contact-center vendor) |   vendor media-stream API
 +----------------------------------+
        | (1) dual-channel audio, 20 ms frames          budget <= 50 ms
        v
 +----------------------------------+
 | Audio ingest gateway             |   codec decode, jitter buffer,
 | (one session per call)           |   call metadata, tenant id
 +----------------------------------+
        | (2) PCM frames per channel                    budget <= 20 ms
        v
 +----------------------------------+      +---------------------------+
 | Streaming ASR                    | (3a) | Diarization / role labels |
 | partial + final hypotheses       |----->| channel map; mono: speaker|
 | per channel                      |      | change model; IVR / hold  |
 +----------------------------------+      +---------------------------+
        | (3) partials about once a second [12];        |
        |     final at endpoint                         | (3b) speaker-tagged
        |     budget 300-800 ms after speech            |      utterances
        v                                               v
 +-------------------------------------------------------------------+
 | Transcript event bus  (partitioned by conversation id;            |
 | events carry utterance id + revision id)          budget <= 30 ms |
 +-------------------------------------------------------------------+
        | (4) hot path                                  | (4b) async
        v                                               v
 +----------------------------------+      +---------------------------+
 | Inference graph executor  [13]   |      | Conversation store + logs |
 |  - keyword / moment detectors    |      | (transcripts, hints shown,|
 |  - intent + hint classifiers     |      |  agent feedback)          |
 |  - suggestion LLM / KB lookup    |      +---------------------------+
 +----------------------------------+                   ^
        | (5) classifiers 20-80 ms;                     |
        |     LLM suggestion 300-800 ms                 |
        v                                               |
 +----------------------------------+                   |
 | Guidance policy                  |   rank, dedupe,   |
 | (per-tenant config)              |   fire-once,      |
 +----------------------------------+   throttle        |
        | (6) push over WebSocket                       |
        |     budget <= 100 ms                          |
        v                                               |
 +----------------------------------+  (7) accept /     |
 | Agent desktop guidance UI        |  dismiss / edit   |
 | hint cards, scripts, KB answers  |------ async ------+
 +----------------------------------+

 End-to-end target (assumed): end of customer phrase -> hint on screen in about 1-2 s.
```

#### Drawing and data-flow questions

1. Draw the real-time Agent Assist path from live call audio to a hint on the agent's screen. Number every arrow and write a latency budget on each. [Likely]
   - Which arrow dominates end-to-end latency, and what evidence would show that?
   - Which arrows carry audio, which carry text events, and which carry UI pushes? How does the choice of transport differ for each?
   - Where would you place queues, and where must you avoid queueing altogether?
2. How do you get audio out of the customer's telephony or CCaaS platform (arrow 1) when each vendor exposes a different integration? [Likely] [12]
   - What breaks when a vendor gives you mixed mono audio instead of two channels?
   - How do you detect and recover from dropped or out-of-order media packets?
3. On arrow 3, which downstream nodes should consume partial transcripts, and which should wait for finals? [Likely] [12]
   - How does a detector avoid firing twice when a partial is revised?
   - What do you do when a final transcript contradicts a hint that is already on screen?
4. Where does diarization sit (arrow 3a), and what does each downstream model need from it? [Likely]
   - How would you label IVR prompts, hold music and transfers so they do not trigger hints?
   - What is the cost of a speaker-attribution error for a compliance hint versus a sales hint?
5. Why partition the transcript bus by conversation id, and what ordering guarantees does arrow 4 need? [Likely]
   - What happens to in-flight events when a consumer instance restarts mid-call?
   - How do you size the bus for thousands of concurrent calls?

#### Inference graph and guidance questions

6. Inside the inference graph executor, which nodes run on every utterance, which run on triggers, and which run only after the call? [Likely] [13]
   - How do per-tenant policies switch nodes on and off without separate deployments?
   - How do you stop one slow node from delaying all the others?
7. How would you meet a 20-80 ms budget for intent and hint classifiers while the suggestion LLM takes hundreds of milliseconds? [Likely]
   - Would you show a fast hint first and refine it later, or wait for the slow path?
   - What would you cache, and what is the cache key?
8. Design the guidance policy box: how does it decide which of several candidate hints to show, and when to stay silent? [Likely]
   - Which signals feed the alert-fatigue model?
   - How would a customer admin tune thresholds without breaking other tenants?
9. How does feedback (arrow 7) reach model training without slowing the hot path? [Likely]
   - Which feedback events are reliable labels, and which are biased by what was shown?
10. Which boxes in this diagram must be redundant for a live call, and what does the agent see when each one fails? [Likely]
    - What is the degradation order: which nodes do you shed first under load?
11. How would you change this diagram for chat Agent Assist, where there is no audio but an agent handles several conversations at once? [Likely]
    - Which boxes disappear, and which new ones (a conversation prioritizer, for example) appear?
12. Where would you add PII redaction in this diagram, and on which arrows must raw audio or text never travel? [Likely]

#### Session lifecycle and audit questions

13. Redraw Diagram 1 as a call-session state machine (ringing, connected, hold, transfer, conference, wrap-up). Which boxes hold per-call state, and what happens to that state on a warm transfer to another agent? [Likely] [12][13]
   - Which hints must be suppressed while the customer is on hold or in the IVR?
14. Where would you attach provenance (model version, input utterance ids, policy rule) to each hint so that a supervisor or customer can audit why it was shown? [Likely] [13]
15. How would you replay a recorded call through this diagram offline, with realistic partial-transcript timing, to reproduce a hint bug deterministically? [Likely] [12][13]
   - Which arrows must be recorded with timestamps for the replay to be faithful?
16. Add a supervisor branch to Diagram 1: one supervisor watches 30 live calls with real-time alerts. Which new arrows and fan-out does that add, and what is their latency budget? [Likely] [12]

### Diagram 2. Voice AI agent loop [Likely]

Points back to: ch04 Q19 (voice agent), Q20 (tool layer), Q21 (handoff), Q23 (guardrails), Q25 (peak latency); ch07 Q17 (AI Agent), Q18 (latency pipeline), Q20 (guardrails), Q26 (side-effecting tools), Q89 (turn-taking). Sources for the product shape: [9][14][18][19].

```
  Caller audio in                                         Agent audio out
        |                                                        ^
        v                                                        |
 +--------------------+                              +------------------------+
 | Telephony / SIP    |                              | Audio out + playback   |
 | media gateway      |                              | controller             |
 +--------------------+                              +------------------------+
        | (1) 20 ms frames   budget <= 40 ms                     ^
        v                                                        | (8) first audio
 +--------------------+   (2a) speech start while agent speaks   |     byte <= 50 ms
 | VAD + turn-taking  |------------------------------------------+ BARGE-IN:
 | (endpointing,      |   stop TTS, flush queued audio, mark the  stop playback
 |  backchannel model)|   unspoken text as "not said"
 +--------------------+
        | (2) end-of-turn decision    budget 200-500 ms of silence (assumed)
        v
 +--------------------+
 | Streaming ASR      |   final transcript ready about when endpoint fires
 +--------------------+
        | (3) final text + confidence     budget <= 150 ms after endpoint
        v
 +---------------------------------------------------------------+
 | LLM planner / dialogue orchestrator                           |
 |  state: conversation, slots, verified identity, flow step     |
 |                                                               |
 |   (4a) tool call ----> +----------------------+               |
 |        <--- result --- | Tool gateway         | policy checks,|
 |                        | (orders, refunds,    | idempotency,  |
 |                        |  account APIs)       | timeouts      |
 |                        +----------------------+               |
 |   (4b) retrieve -----> +----------------------+               |
 |        <--- passages - | RAG over tenant KB   |               |
 |                        +----------------------+               |
 +---------------------------------------------------------------+
        | (5) streamed tokens   time to first token budget 200-400 ms
        v
 +--------------------+     (5b) block / rewrite / escalate
 | Output guardrails  |-----------------------------------> +---------------+
 | (policy, PII,      |                                      | Human handoff |
 |  groundedness)     |                                      | (warm transfer|
 +--------------------+                                      |  + summary)   |
        | (6) approved sentence chunks   budget <= 50 ms     +---------------+
        v
 +--------------------+
 | Streaming TTS      |   (7) time to first byte budget 100-200 ms
 +--------------------+
        | (7) audio chunks --------------------------------> (8) playback

 Side channel (async): every turn writes a trace (audio refs, ASR, prompts,
 tool calls, guardrail verdicts, timings) to the trace store for evaluation.

 Target (assumed): caller stops speaking -> first agent audio in under ~1 s.
```

#### Turn-taking and latency questions

17. Draw the voice agent loop from caller audio to agent audio and back, with a latency budget on each arrow that adds up to your end-to-end target. [Likely] [14]
    - Which budget would you cut first if the total came to 1.6 s?
    - Which stages can overlap (streaming) and which must run in sequence?
    - How do you measure the gap the caller actually hears, as opposed to the sum of component timings?
18. How does the VAD and turn-taking box decide that the caller has finished (arrow 2)? [Likely] [9]
    - What do you do for slow speakers, people reading out account numbers, and long pauses mid-thought?
    - Could a semantic end-of-turn model replace a fixed silence threshold? What would it cost in latency?
19. Walk through barge-in (arrow 2a): the caller interrupts while the agent is speaking. [Likely]
    - How does the planner learn which part of its answer the caller actually heard?
    - How do you tell a real interruption from a backchannel ("uh-huh") or background noise?
20. Where would you speculate (start the LLM before the endpoint is final, pre-synthesize filler), and how do you discard wasted work? [Likely]
21. How does an ASR error on arrow 3 (for example, a wrong account digit) spread through the rest of the loop, and where would you catch it? [Likely] [10]
    - Which confirmations would you add, and what do they cost in call time?

#### Planner, tools and guardrail questions

22. What state does the LLM planner hold, and where does it live so a crashed worker can resume the call? [Likely]
23. Draw the tool gateway (arrow 4a) for side-effecting actions such as refunds. [Likely]
    - How do idempotency keys and confirmation turns fit into the diagram?
    - What does the agent say while a slow backend API is running?
    - How would you test tools without touching the customer's production systems?
24. When does the planner call RAG (arrow 4b), and how does that change time to first token? [Likely] [19]
25. Where do guardrails run (input, planning, output), and what latency does each add? [Likely] [19]
    - Which checks are small classifiers and which are LLM calls?
    - A sentence fails the output check after earlier sentences have already been spoken. What happens?
26. Which signals trigger the human handoff branch (5b), and what context travels with the call? [Likely]
    - How do you avoid escalating too early or too late?
27. How would you change the diagram to keep latency stable during a 5x call spike after an outage? [Likely]
    - Which box degrades first, and what is the fallback for each?
28. What goes into the per-turn trace, and how is it used by the evaluation platform in Diagram 3? [Likely] [15]
29. How would you redraw this loop for a structured flow (nodes, transitions, slots) with LLM steps inside, as in Cresta's agent-building examples? [Likely] [18]
    - Which box owns the flow state, and how do you stop the LLM from skipping a required step?

#### Speech front-end and context questions

30. Where in Diagram 2 would language identification sit, and what happens to ASR, the planner and TTS when the caller switches language mid-call? [Likely] [9][10]
31. A caller must read out a card number or press DTMF keys for payment. Redraw the loop so those digits bypass the LLM, the trace store and recordings, and show which box resumes control afterward. [Likely] [12][19]
32. Draw what goes into the planner's prompt on turn 40 of a long call. Where do summarization of older turns and retrieval of earlier facts happen, and what do they add to time to first token? [Likely] [14]
33. How would the planner pass expected-entity hints (for example, "the next answer is an order number") back to the ASR box for contextual biasing, and which arrow carries them? [Likely] [10]
   - How do you measure whether biasing helped entity accuracy without hurting general word error rate?

### Diagram 3. Agent evaluation platform for prompt and model changes [Reported] [4][5]

Points back to: ch04 Q1 (reported eval platform), Q33 (customer simulator), Q35 (LLM-as-a-judge), Q38 (release of prompt changes), Q42 (golden sets); ch07 Q29 (reported eval platform), Q30 (simulator), Q31 (evaluators), Q34 (release process), Q35 (expert labeling). Product context: [15][16][17].

```
 +---------------------+   (1) change request: new prompt, model,
 | Change sources      |       tool schema or KB snapshot
 | (prompt editor, CI, |----------------------------------------+
 |  model registry)    |                                        |
 +---------------------+                                        v
                                               +-----------------------------+
 +---------------------+  (2) select suites    | Run orchestrator            |
 | Test-suite store    |---------------------->| pins versions: prompt id,   |
 | - golden convos     |                       | model id, tools, KB snapshot|
 | - scenario specs    |                       | fans out N reruns/scenario  |
 |   (persona + goal)  |                       +-----------------------------+
 | - red-team attacks  |                           | (3) scenario jobs
 | - tags by tenant,   |                           v
 |   intent, risk      |     +-------------------------+   (4) turns    +---------------------+
 +---------------------+     | Customer simulator      |<-------------->| Agent under test    |
        ^                    | (LLM persona; optional  |   text or      | (exact config being |
        |                    |  TTS/noise for voice)   |   synthetic    |  evaluated, sandbox |
        |                    +-------------------------+   audio        |  tools, mock APIs)  |
        |                                 | (5) transcripts + traces    +---------------------+
        |                                 v
        |              +--------------------------------------------------+
        |              | Evaluator layer                                  |
        |              |  (5a) deterministic checks: tool args, policy    |
        |              |       rules, latency, schema                     |
        |              |  (5b) LLM-as-judge per criterion (rubric prompts)|
        |              |  (5c) sample routed to human labelers            |
        |              +--------------------------------------------------+
        |                     |                     | (6) judge vs human
        |                     | (7) scores          v     agreement
        |                     v              +--------------------------+
        |              +----------------+    | Labeling + calibration   |
        |              | Results store  |    | (gold labels, IAA,       |
        |              | (per run,      |    |  judge drift reports)    |
        |              |  per scenario) |    +--------------------------+
        |              +----------------+
        |                     | (8) compare with the baseline run
        |                     v
        |              +-------------------------------+     pass   +------------------+
        |              | Regression gate               |----------->| Deploy (see      |
        |              | stat test over reruns, per-   |            | Diagram 4)       |
        |              | slice thresholds, hard-fail   |   fail     +------------------+
        |              | rules (safety, compliance)    |-----------> block + report diff
        |              +-------------------------------+
        |                                                 (9) async
        +---- new failing production conversations are mined into the suite
```

#### Pipeline and reproducibility questions

34. Draw an evaluation platform that catches regressions when a prompt or model changes, and name what flows on each arrow. [Reported] [4][5]
    - Who triggers a run (arrow 1): an engineer, CI, or a model-registry event?
    - Who reads the result: ML engineers, forward-deployed engineers or the customer? How does that change the report?
35. What must the run orchestrator pin so that a run can be reproduced exactly a month later? [Likely] (follow-up to the reported prompt [4][5]; the follow-up itself is not reported)
    - How do you snapshot a knowledge base and mock tool responses that change over time?
36. How many reruns per scenario do you need, and which statistical test turns noisy scores into a pass or fail at the gate (arrow 8)? [Likely]
    - How do you keep a flaky scenario from blocking every release?
37. What is in the test-suite store, and how is each scenario tagged so you can report by tenant, intent and risk? [Likely] [17]
    - How does a failing production conversation become a new scenario (arrow 9) without leaking PII?

#### Simulator and evaluator questions

38. How do you make the customer simulator (arrow 4) realistic enough that passing the suite predicts production behavior? [Likely] [17]
    - How would you mine personas and goals from real conversations?
    - How do you stop the simulator from being too cooperative?
    - What changes when the agent under test is a voice agent and the simulator must speak?
39. Which checks belong in the deterministic box (5a), and which genuinely need an LLM judge (5b)? [Likely] [15]
    - How do you score the trajectory (tool calls, steps) separately from the final outcome?
40. How do you calibrate the LLM judge against human labels (arrows 5c and 6), and what agreement level is good enough to gate a release? [Likely] [15]
    - What do you do when the judge model itself is upgraded?
    - How do you catch a judge that prefers the style of its own model family?
41. Apply the Swiss-cheese model to this diagram: which failure types does each layer stop, and which pass through every layer? [Likely] [16]
42. How do you scale to thousands of scenarios across hundreds of customer-specific agents, and what does a full run cost? [Likely] (follow-up to the reported prompt [4][5])
    - Which runs are per-tenant, and which are shared across tenants?
    - How would you cut cost without losing the ability to detect a regression?
43. A change improves the average score but breaks one large customer's top intent. Where in the diagram is that caught, and who decides whether it ships? [Likely]
44. Where would red-team scenarios (prompt injection, social engineering) enter this diagram, and should they hard-fail the gate? [Likely]

#### Online evaluation and authoring-loop questions

45. Draw the online counterpart of Diagram 3: production conversations sampled to the same evaluators and results store. What do you do when offline suite scores and online judge scores disagree? [Likely] [15][16]
46. An agent-building assistant (as in Cresta Conductor) proposes prompt edits and runs the suite automatically. Where does it plug into Diagram 3, and how do you stop it from overfitting the test suite? [Likely] [17][18]
   - Which scenarios would you hold out from the assistant entirely?
47. Which evaluator boxes would you add for voice-only failures (response latency, talk-over, missed barge-in, mispronounced names), and where does their input come from? [Likely] [14][17]
48. When the gate fails, what does the diff report show the prompt author, and how does each failing scenario link back to its full trace? [Likely] [15]

### Diagram 4. Deploying a model in production [Reported] [1]

Points back to: ch04 Q2 (reported "deploy a model in production"), Q38 (prompt change release), Q70 (per-customer fine-tuning); ch07 Q34 (release process), Q63 (reported deploy question), Q64 (monitoring).

```
 +---------------+  (1) training run  +----------------------+
 | Training      |------------------->| Model registry       |
 | pipeline      |   artifact, data   | version, lineage,    |
 | (data+code    |   hash, metrics    | model card, owner,   |
 |  versioned)   |                    | per-tenant eligibility|
 +---------------+                    +----------------------+
                                             | (2) candidate
                                             v
                                      +----------------------+
                                      | Offline eval gate    |  <-- Diagram 3 for
                                      | held-out + per-slice |      agents/LLMs
                                      | + latency/cost check |
                                      +----------------------+
                                             | (3) pass
                                             v
 live traffic                         +----------------------+
 ----------+------------------------->| Shadow deployment    |  new model sees copies of
           |   (4) mirrored requests  | (responses not shown)|  real traffic; outputs
           |                          +----------------------+  logged and diffed
           |                                 | (5) diff OK
           |                                 v
           |                          +----------------------+
           +------------------------->| Canary               |  1% -> 5% -> 25% of
           |   (6) small slice        | (by tenant cohort or |  traffic, auto-advance
           |                          |  by agent group)     |  on healthy metrics
           |                          +----------------------+
           |                                 | (7)
           |                                 v
           |                          +----------------------+
           +------------------------->| A/B test             |  randomize by agent or
               (8) split traffic      | business metrics:    |  team; metrics may lag
                                      | AHT, CSAT, conversion|  by days
                                      +----------------------+
                                             | (9) ship decision
                                             v
                                      +----------------------+
                                      | Full rollout         |
                                      +----------------------+

 Cross-cutting (async, every stage):
 +--------------------------------------------------------------------------+
 | Monitoring: latency p50/p99, errors, input drift, output-score drift,     |
 | judge-sampled quality, user feedback (accept/dismiss), per-tenant slices  |
 +--------------------------------------------------------------------------+
        | (10) alert / SLO breach
        v
 +----------------------+   (11) pin previous version in the registry,
 | Rollback controller  |------------------------------------------------>
 | auto + manual        |   flip the traffic router, notify the owner
 +----------------------+
```

#### Rollout path questions

49. Draw how a model goes from training to full production, with every gate and the rollback path. [Reported] [1]
    - What does the registry entry contain, and who is allowed to promote a version?
    - Which gates are automatic and which need a human sign-off?
50. When is shadow mode (arrow 4) worth its GPU cost, and what do you compare between the shadow and production outputs? [Likely]
    - How do you evaluate shadow outputs for a generative model when there is no single correct answer?
51. How do you choose the canary unit (arrow 6) when a few large customers dominate traffic? [Likely]
    - Why might you canary by tenant cohort rather than by random request?
52. How do you randomize the A/B test (arrow 8) when agents work in teams and business metrics arrive days later? [Likely]
    - What is your guardrail metric, as opposed to your success metric?
    - How long must the test run, and how do you deal with novelty effects on agents?
53. How does this diagram differ for a real-time model (assist hints) and a batch model (post-call QA scoring)? [Likely] (follow-up to the reported prompt [1])
    - For the batch model, what replaces canary traffic?

#### Monitoring and rollback questions

54. Which monitoring signals (box 10) trigger an automatic rollback in the first 24 hours, and which only page a human? [Likely] (follow-up to the reported prompt [1])
    - How do you detect a quality regression when you have almost no labels?
55. How does the rollback controller (arrow 11) undo a model that also changed a prompt template, a feature schema or an index? [Likely]
    - What must be versioned together so that rollback is one operation?
56. How do you manage per-customer model versions, so that one tenant can stay pinned while others upgrade? [Likely] (follow-up to the reported prompt [1])
    - How do you stop the number of live versions from growing without limit?
57. Where in this diagram would a prompt-only change enter, and which gates can it skip? [Likely]
58. How would you prove to an enterprise customer, after the fact, which model version produced a given suggestion on a given call? [Likely]

#### Dependencies and change-management questions

59. Where in Diagram 4 do you log the features and inputs a model saw at serving time, so the next training run does not suffer training-serving skew? [Likely]
60. Diagram 1 chains ASR into intent, hint and suggestion models. How do you canary a new ASR model when every downstream model was trained on the old ASR's output? [Likely] [10][13]
   - Which downstream metrics would you gate on, in addition to word error rate?
61. An enterprise customer requires advance notice and approval before the model behind its agents changes. Where does that approval sit in Diagram 4, and how does it interact with shared base-model upgrades? [Likely] [24]

### Diagram 5. Knowledge Assist: RAG ingestion and query [Likely] (extends a [Reported] FDE question [6])

Points back to: ch04 Q3 (reported RAG and hallucination, FDE screen), Q58 (Knowledge Assist), Q59 (fine-tuned serving), Q60 (AI-ready ingestion), Q62 (retrieval evaluation); ch07 Q9 (Knowledge Assist), Q11 (fine-tuned migration), Q13 (grounding), Q14 (RAG and hallucination), Q15 (fresh KB updates). Product context: [19][20][21][22].

```
 ===================== INGESTION (offline / near-real-time) =====================

 +------------------+  (1) pull or   +---------------------+  (2) raw docs   +--------------------+
 | KB connectors    |  change feed   | Document normalizer |---------------->| Quality checks     |
 | (help center,    |--------------->| HTML/PDF/table      |                 | dedupe, stale,     |
 |  wiki, CRM, PDFs)|                | parsing, metadata   |                 | conflicts, PII     |
 +------------------+                +---------------------+                 +--------------------+
                                                                                    | (3) clean docs
                                                                                    v
 +------------------+  (5) vectors + +---------------------+  (4) chunks     +--------------------+
 | Per-tenant index |<---------------| Embedding workers   |<----------------| Chunker            |
 | dense + lexical, |   metadata,    | (batch, versioned   |                 | by section, with   |
 | doc version,     |   alias swap   |  embedding model)   |                 | titles + parents   |
 | ACL filters      |                +---------------------+                 +--------------------+
 +------------------+
        ^                                (6) async: conflicting or missing content -> KB owner queue
        |
 ======================== QUERY (live conversation) =================================
        |
 +------------------+  (a) last N turns   +--------------------+  (b) query  +---------------------+
 | Transcript bus / |-------------------->| Question detector  |------------>| Query rewriter      |
 | agent search box |   budget <= 30 ms   | + trigger policy   |  <= 50 ms   | (conversation ->    |
 +------------------+                     +--------------------+             |  standalone query)  |
                                                                             +---------------------+
                                                                                    | (c) 50-150 ms
                                                                                    v
 +------------------+  (f) answer +       +--------------------+  (d) top-k  +---------------------+
 | Guidance UI /    |   citations         | Grounded generator |<------------| Hybrid retrieval +  |
 | voice agent      |<--------------------| (fine-tuned model) |  80-150 ms  | reranker (tenant +  |
 | (channel format) |   budget <= 100 ms  | + groundedness /   |             |  ACL filtered)      |
 +------------------+                     | "no answer" check  |             +---------------------+
        |                                 +--------------------+
        |                                    (e) generation 300-700 ms
        | (g) async: click, copy, thumbs, "not helpful" -> eval + gap mining
        v

 Target (assumed): question detected -> cited answer on screen in about 1-1.5 s.
```

#### Ingestion questions

62. Draw the ingestion side of Knowledge Assist, from the customer's knowledge sources to a searchable per-tenant index. [Likely] [20]
    - How do you detect stale, duplicated and contradictory articles (box after arrow 2), and who fixes them?
    - How do you handle tables, PDFs and screenshots?
63. How do you keep the index fresh when an article changes during the business day (arrows 1 and 5)? [Likely]
    - What does the alias swap protect against, and what does it cost?
    - How do you record which article version an answer used?
64. How would you choose chunk boundaries for contact-center knowledge (procedures, eligibility rules, pricing tables)? [Likely]
65. Plan an embedding-model upgrade that re-embeds every tenant's index without downtime. Where in the diagram do the old and new indexes coexist? [Likely]

#### Query and grounding questions

66. Draw the query path from a live conversation to a cited answer, with a latency budget on each arrow. [Likely] [21][22]
    - Which step would you cache, and what is the cache key?
67. What is RAG, and where in this diagram do you prevent hallucination? [Reported] [6] (FDE technical screen; expect a senior MLE design round to ask for the diagram as well.)
    - At which box does each failure type start: retrieval miss, stale article, generator going beyond the passages?
    - What does the "no answer" path show the agent, or say to the caller?
68. How does the question detector (arrow a) decide when to answer proactively and when to wait for the agent to ask? [Likely]
69. How does the query rewriter turn a noisy, multi-turn, partly mis-transcribed conversation into a good retrieval query? [Likely]
70. How would you evaluate retrieval and answers when there are no labeled query-document pairs? [Likely]
    - How far can you trust LLM-generated relevance labels?
71. Fireworks reports that Cresta moved Knowledge Assist to fine-tuned models. Where in this diagram does that change cost and latency, and how would you prove quality held? [Likely] [21][22]
72. How does the diagram change when the consumer is a voice AI agent (Diagram 2) rather than a human agent's screen? [Likely]
73. How do you enforce document-level access control so an agent never sees an article meant for another team or tenant? [Likely]

#### Routing and safety questions

74. Some questions need account-specific facts from a customer API rather than KB articles. Where in Diagram 5 does a router decide between retrieval and a tool call, and how is that decision evaluated? [Likely] [19][20]
75. How does Diagram 5 change when the conversation is in Spanish but most KB articles are in English? Which boxes need multilingual models? [Likely] [20]
76. A KB article contains text that tries to instruct the model ("ignore previous instructions..."). At which boxes in Diagram 5 do you defend against this indirect prompt injection? [Likely] [19]

### Diagram 6. Conversation intelligence: batch and stream analytics [Likely]

Points back to: ch04 Q44 (topic discovery), Q45 (AI analyst), Q47 (emerging issues), Q48 (structured extraction), Q50 (event pipeline), Q51 (automated QA); ch07 Q38 (taxonomy), Q40 (AI Analyst), Q41 (QA scoring), Q42 (trend detector), Q44 (batch backend), Q90 (consistent extractions). Product context: [25].

```
 +-------------------+   (1) live events: utterances,     +---------------------------+
 | Real-time stack   |   hints, AI-agent turns           | Stream processor          |
 | (Diagrams 1 + 2)  |---------------------------------->| windowed counts per topic,|
 +-------------------+   seconds                         | sentiment, moment flags   |
        |                                                 +---------------------------+
        | (2) call-end event + final transcript                  | (3) anomaly detection
        |     (minutes)                                          v     (minutes)
        v                                                 +---------------------------+
 +-------------------+                                    | Emerging-issue alerts     |
 | Post-call queue   |  per-tenant priority, backfill     | (spike in topic X)        |
 | (durable, retry)  |  jobs at low priority              +---------------------------+
 +-------------------+
        | (4) batches of conversations
        v
 +-----------------------------------------------------------------+
 | Batch enrichment workers  (model router: small model vs LLM)    |
 |  - summary + disposition       - QA rubric scoring              |
 |  - structured extraction       - outcome / resolution inference |
 |    (reason, product, churn)    - embeddings                     |
 |  every output stamped with model + prompt version               |
 +-----------------------------------------------------------------+
        | (5) rows + vectors                         | (6) embeddings (daily/weekly)
        v                                            v
 +-------------------+                     +------------------------------+
 | Analytics store   |                     | Topic discovery job          |
 | (columnar tables, |<--------------------| cluster -> LLM label ->      |
 |  per tenant,      |  (7) topic ids      | hierarchy -> admin edits ->  |
 |  late-arrival     |                     | assignment model for new     |
 |  upserts)         |                     | conversations                |
 +-------------------+                     +------------------------------+
        | (8) SQL / metrics API
        v
 +-------------------+  (9) question  +--------------------------+  (10) answer + evidence links
 | Dashboards,       |<---------------| "AI analyst" agent       |----------------------------->
 | QA + coaching UI  |                | plan -> query tables ->  |
 +-------------------+                | sample convos -> LLM     |
                                      | read -> aggregate        |
                                      +--------------------------+
```

#### Pipeline questions

77. Draw the analytics system that turns every conversation into structured data, showing which parts are streaming and which are batch. [Likely] [25]
    - Why does the emerging-issue detector (arrow 3) need the streaming path?
    - What is the freshness target for each output: seconds, minutes, hours?
78. How do you schedule batch enrichment (arrow 4) fairly across thousands of tenants while live traffic shares the same GPUs? [Likely]
    - What happens when a tenant asks to backfill a year of calls with a new extractor?
79. Every output is stamped with a model and prompt version. What do you do when the extractor model changes and last month's numbers no longer match this month's? [Likely]
    - When do you backfill, and when do you show a break in the series?
80. How do you handle late-arriving and corrected events in the analytics store (arrow 5)? [Likely]
81. Which enrichments would you run with a small fine-tuned model and which with a frontier LLM? How does the router decide? [Likely]
    - How do you attribute cost per tenant and per feature?

#### Topic discovery and AI analyst questions

82. Draw the topic-discovery job (arrow 6): clustering, LLM labeling and a hierarchy up to three levels deep. [Likely] [25]
    - How do you keep topic ids stable when the job reruns next week?
    - How do admin edits survive re-clustering?
83. How does the AI analyst (arrows 9-10) answer "Why did cancellations rise last week?" without inventing a cause? [Likely] [25]
    - When does it query structured tables, and when does it read sampled conversations?
    - How do you show statistical confidence and link the evidence?
84. How would you evaluate automated QA scores (in the enrichment box) against human QA analysts? [Likely]
    - What agreement level would make a customer trust automatic scoring of every call?
85. Where in this diagram are PII redaction and data-retention rules enforced? [Likely]

#### Outcomes, custom extractors and search questions

86. Add a join between conversation data and CRM outcomes (sale closed, churn) to Diagram 6. How would you attribute outcomes to agent behaviors without confusing correlation with cause? [Likely] [25]
87. A customer admin defines a new extractor in plain language ("Did the agent offer the retention discount?"). Draw the path from that definition to scored historical calls, including validation before it runs on a full year of data. [Likely] [25]
88. Add semantic and keyword search over millions of conversations to Diagram 6. Which store serves it, and how does it respect per-tenant access control? [Likely] [25]

### Diagram 7. Multi-tenant LoRA / adapter LLM serving [Likely]

Points back to: ch04 Q65 (multi-tenant serving), Q66 (p99 serving stack), Q67 (model router), Q70 (per-customer fine-tuning), Q74 (per-tenant rate limiting), Q75 (real-time vs batch GPU sharing); ch07 Q54 (per-customer adaptation), Q55 (Fireworks multi-LoRA serving), Q56 (routing), Q57 (p99 serving). Product context: [22][23][24].

```
 Callers: Agent Assist, voice agent, KA, batch analytics
        |
        | (1) request: tenant id, feature, priority (real-time | batch)
        v
 +--------------------------------+
 | API gateway                    |  auth, tenant isolation, per-tenant
 | + rate limiter / quota         |  token quotas, real-time vs batch lanes
 +--------------------------------+        budget <= 10 ms
        | (2)
        v
 +--------------------------------+      +-------------------------------+
 | Request router                 |----->| Config + adapter registry     |
 | (tenant -> base model, adapter,| (2a) | tenant -> adapter id/version, |
 |  prompt template; small vs     |      | base-model compatibility,     |
 |  large model; region)          |      | eval status, rollout state    |
 +--------------------------------+      +-------------------------------+
        | (3) to the replica that already holds the adapter (affinity)   budget <= 5 ms
        v
 +---------------------------------------------------------------------------+
 | GPU serving pool  (one base model, e.g. a fine-tuned contact-center LLM)  |
 |                                                                           |
 |   continuous batching scheduler  --  mixes requests for many adapters     |
 |   in one batch; real-time lane preempts batch lane                        |
 |                                                                           |
 |   adapter cache tiers:  GPU HBM (hot) <- host RAM (warm) <- object store  |
 |                         (cold load: (4) budget 50-500 ms, assumed)        |
 |   KV cache, prefix cache per prompt template                              |
 +---------------------------------------------------------------------------+
        | (5) streamed tokens       TTFT budget 150-300 ms (real-time lane)
        v
 +--------------------------------+
 | Response path                  |  per-tenant metering (tokens, GPU-ms),
 | + logging (tenant-scoped)      |  traces, no cross-tenant caching of content
 +--------------------------------+

 Control plane (async):
 +-----------------+  (6) new adapter  +------------------+  (7) pass   +--------------------+
 | Adapter training|------------------>| Per-tenant eval  |------------>| Registry promote + |
 | jobs (consented |                   | gate (Diagram 3) |             | canary (Diagram 4) |
 |  tenant data)   |                   +------------------+             +--------------------+
 +-----------------+
 +----------------------------+
 | Autoscaler: scales on      |  queue depth, TTFT, tokens/s, adapter-miss rate
 | GPU signals, not CPU       |  (8) adds or drains replicas
 +----------------------------+
```

#### Serving questions

89. Draw a serving stack where one base model serves hundreds of customer-specific LoRA adapters on shared GPUs. [Likely] [23]
    - Why not one dedicated deployment per tenant? At what tenant count does that break?
    - How does mixed-adapter batching (in the GPU pool) affect throughput and latency?
90. How does the router (arrow 3) place requests so that adapter loads stay rare, without hot-spotting one replica? [Likely]
    - What happens on a cold adapter load (arrow 4) in the middle of a live call?
91. How do you keep real-time traffic within its p99 budget when a batch backfill from one tenant floods the pool? [Likely]
    - Where do preemption, separate lanes and quotas sit in the diagram?
92. Which signal should the autoscaler (arrow 8) use, and why is CPU utilization a poor fit for GPU inference? [Likely]
    - How do you keep scale-up time acceptable when a replica must load a large base model?
93. What happens to every tenant's adapter when the base model is upgraded? [Likely] [24]
    - How does the eval gate (arrow 7) catch one tenant's silent regression?
94. Where could data leak between tenants in this diagram (prefix cache, logs, metering, training data), and how do you prevent each? [Likely]
95. How would you choose between a small fine-tuned model and a frontier API for each request in the router box? [Likely]
    - How do you prove the router does not hurt quality?
96. How would you attribute GPU cost to each tenant when requests share batches? [Likely]
97. How does this diagram change for an EU tenant whose data and model calls must stay in the EU? [Likely]
    - What may the global control plane hold, and what must stay regional?
98. Fireworks describes serving many LoRA adapters on one base-model deployment for Cresta. Which box in your diagram does that remove from Cresta's own engineering scope, and which risks remain Cresta's? [Likely] [22][23]

#### Decoding and failover questions

99. Where in Diagram 7 would constrained decoding (a JSON schema or grammar) run for structured extraction, and what does it cost in throughput when mixed with free-text requests in one batch? [Likely] [23]
100. Would you add speculative decoding with a draft model to Diagram 7? How does it interact with per-tenant adapters and mixed-adapter batching? [Likely] [23]
101. During a GPU-capacity outage, could the router fail over to an external model provider? Show where that path sits, and which tenants' data and contract terms would forbid it. [Likely] [22]

### Cross-cutting whiteboard probes (apply to any diagram above)

102. Before drawing, what functional and non-functional requirements would you confirm, and which one shapes the diagram most? [Likely]
103. Give a back-of-envelope estimate for the diagram you drew: concurrent calls, events per second, tokens per day, GPUs. [Likely]
104. Which single box, if it failed, would hurt customers most, and how does the diagram change to protect it? [Likely]
105. Where are the tenant boundaries in your diagram, and which arrows cross them? [Likely]
106. Which arrows carry PII, and where is it redacted or encrypted? [Likely]
107. Which parts of your diagram are on the latency-critical path, and which could move to async without users noticing? [Likely]
108. Where does evaluation data come from in this diagram, and how does it flow back into training? [Likely]
109. How would the diagram change for 10 tenants compared with 1,000, and for chat compared with voice? [Likely]
110. If you had one quarter and three engineers, which boxes would you build first, which would you buy, and which would you defer? [Likely]
111. How do the seven diagrams connect: which boxes are shared platform services (trace store, model registry, eval gate, serving pool)? [Likely]
    - Draw the one-page map that links Diagrams 1-7.

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer review: recruiter call, LeetCode-medium, vibe-coding and system design rounds; "How would you deploy a model in production?"), 2025-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Blind, "Cresta interview process", 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
3. Blind, "Anyone interview with Cresta lately?", 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
4. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
5. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
6. Glassdoor, "Cresta (CA) Interview Experience and Questions" (Forward Deployed Engineer technical screen: "What is RAG, and how do you prevent hallucination?"), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
7. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
8. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
9. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
10. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
11. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
12. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
13. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
14. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
15. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
16. Cresta Blog, "Why AI Agent Evaluations Fail, and How the Swiss-Cheese Model Prevails". https://cresta.com/blog/why-ai-agent-evaluations-fail----and-how-the-swiss-cheese-model-prevails
17. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
18. Cresta Blog, "Cresta Conductor: The Agent for AI Agent Development". https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
19. Cresta Blog, "Grounding Reality: How Cresta Tackles LLM Hallucinations in Enterprise AI". https://cresta.com/blog/grounding-reality---how-cresta-tackles-llm-hallucinations-in-enterprise-ai
20. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
21. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
22. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
23. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
24. Cresta Blog, "Introducing Ocean-1: World's First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
25. Cresta Blog, "Introducing the Next Generation of Cresta Insights: A Complete System for Enterprises to Understand Their CX", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx

*Source note:* No page could be opened in review (DNS failures and proxy 403s on all 10 URLs tried; web-search budget used up). All 25 references are carried over from the reviewed reference lists in chapters 01, 04, 07 and 10, with the same attributions. *Changed in review:* the 1point3acres decoding thread (old reference 6) was removed because no question here cited it, and later references were renumbered. Five follow-up questions (Q35, Q42, Q53, Q54, Q56 in the new numbering) had carried [Reported] tags that their sources do not support, since the sources report only the headline prompt; they are now [Likely] and say which reported prompt they extend. 24 new [Likely] whiteboard questions were added, grouped under one new subsection per diagram (session lifecycle and audit; speech front-end and context; online evaluation and authoring loop; dependencies and change management; routing and safety; outcomes and search; decoding and failover). None is a new candidate report. [Reported] tags remain only on the headline prompts of Diagram 3 [4][5] and Diagram 4 [1] and on question 67 (RAG and hallucination, FDE screen [6]). All latency budgets on the diagrams are illustrative assumptions, except the roughly once-a-second partial transcript cadence attributed to [12] in earlier chapters.
