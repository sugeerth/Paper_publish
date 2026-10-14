# 12. ML Platform and Experimentation Pipeline Diagrams: Bighead, Zipline/Chronon, ERF/Minerva and Search Interleaving

> **Research note (read first).** Written on 2026-10-14 and revised the same day by an adversarial review. **No web search could run in either pass**: the shared search budget was used up, and both passes were refused. In the review pass, medium.com, arxiv.org, datacouncil.ai, youtube.com and other non-GitHub hosts did not resolve through the proxy. GitHub pages did. The review re-fetched and checked the interview-report pages [1][2][3], the 1point3acres mirrors [4][5] and the Chronon README and docs [6]-[11]. It also read the raw applied-ml index [12], which confirms the titles, years and URLs of the Airbnb posts [13]-[18] and [22]-[25]. It scanned about 100 Airbnb posts in the 1point3acres mirror for ML or experimentation content; only [4], [5] and [26] were relevant. The review made these changes: one [Reported] question (old Q67, which cited an "offline NDCG" wording that [1] does not contain) was rewritten and retagged [Likely]; one citation was moved from [5] to [4]; a round-length claim that [3] does not support was removed; one reference with a suspect URL (the old [22], Analytics India Magazine) was dropped; and Section E (Q101-Q130) was added. Section E maps the newly checked reported prompts onto the four diagrams. Rerun with web access (Glassdoor, Blind, LeetCode Discuss, Reddit, Prepfully) before relying on the [Reported] coverage.
>
> **What this chapter adds.** Chapters 04, 06, 09 and 10 describe these four systems in prose, and chapter 10 has one-box-per-stage sketches of the experiment funnel (Diagram 7) and the Chronon path (Diagram 8). This chapter draws each platform at whiteboard level, adds notes on how to draw and defend it in an MLE design round, and lists the questions an interviewer is likely to probe it with.
>
> **About the diagrams.** No source reports that an interviewer handed out or asked for these exact diagrams. They are **study scaffolds**. Bighead, ERF and Minerva internals are reconstructed from public talks and posts that were not reachable this session [14][19][20][21]; treat component names as "as publicly described, possibly outdated". The Chronon diagram rests on documentation fetched and read for this chapter [6]-[11]. Nothing here is Airbnb's current production architecture.

## How the tags work

- **[Reported] [n]**: a fetched source [n] reports this prompt, or a very close variant, from an Airbnb interview. Source [2] does not always name the role.
- **[Likely]**: an inference from the Senior MLE (experimentation) role, the reported round structure and Airbnb's public platforms. A citation next to [Likely] points to the system behind the topic. It does **not** mean a candidate was asked the question.
- **Follow-up probes** are the indented bullets. They inherit no tag unless a source is cited inline.
- **Reference status**: **fetched** = read on 2026-10-14; **URL confirmed** = title, year and URL confirmed by the index [12], but the page itself was not opened; **unverified** = host unreachable, carried over from earlier chapters.

## Format notes

- **Where these diagrams get used.** MLE candidates get "one or two ML system design rounds" in place of the standard system design round [3]. Senior (G9) loops often have two, with depth "close to staff level" [3]. The ML design rounds are "conversations without code" [1], and interviewers want the problem and its label defined before any model is discussed [3]. The fetched sources give no round length. The MLE guide draws on three candidate reports, "primarily from 2026 Q1 through 2026 Q2" [1]. No source reports a round devoted to ML platform design. These diagrams come up as zoom-ins inside product prompts such as search ranking, LTV, pricing, relisting and cover-photo selection [1][4][5], and inside the reported "Design an A/B Testing Platform" prompt (Apr 2026) [2].
- **What is probed.** Search-ranking rounds "pushed on latency, freshness and experimentation" [1]. Candidates are told to name "how an A/B test would confirm" each model, and role postings stress "training-serving skew, A/B testing, and feature and model selection" [1]. One rejection cited "jumping straight to model ideas" [3], so draw the label and data path before the model.
- **Interviewer.** "Interviewers steer firmly"; treat a redirection as a hint about the expected answer [1]. One LTV interviewer had no time for deployment and serving [4]. Be ready to collapse a platform diagram to its two or three boxes that matter, or to expand any single box on request.
- **Senior bar [Likely].** Owning a system end to end [3] here means you can (a) draw the offline/online split, (b) point at where skew, leakage and bias enter, and (c) say how you would detect each with a metric or a check.

---

## Diagram A. Bighead model lifecycle (Airbnb's end-to-end ML platform)

Publicly described components, from conference talks that were not reachable this session [20]: Redspot (hosted notebooks), Bighead Library (pipeline and transformer abstractions shared by training and serving), ML Automator (scheduled training on Airflow), Bighead Service and UI (model registry and versioning), Deep Thought (online inference) and Zipline (features). Treat the names as historical; Chronon later replaced Zipline [6].

```
  [Zipline / Chronon features] ----------------------------------+
        | point-in-time training set                             | online fetch
        v                                                        |
  Redspot notebook --> Bighead Library pipeline (fit) ---+       |
   (explore, prototype)   same transform code offline/online     |
        |                                                |       |
        v                                                v       |
  ML Automator (Airflow DAG: scheduled retrain) --> model artifact + metadata
                                                         |
                                                         v
                                      Bighead Service: registry, versions,
                                      lineage, eval reports, promote/rollback
                                                         |
                       +---------------------------------+------------------+
                       v                                                    v
            batch scoring (Airflow -> Hive)                Deep Thought: online inference
                                                            (container per model, k8s)
                                                                  |  <---------------+
                                                                  v
                                           product service -> ERF assignment -> logging
                                                                  |
                                                                  v
                             monitoring: feature drift, score drift, online/offline parity
```

**How to draw and defend it.** Draw the training path left to right and the serving path below it. Mark the **one shared transform library** with a double line, because that is your answer to training-serving skew. Put the registry in the middle as the single promote/rollback point. End at the experiment assignment so the interviewer sees the model is launched through an A/B test, not just deployed.

### A1. Lifecycle and stages

1. Draw the lifecycle of a search-ranking model from notebook prototype to 100% traffic. Which boxes are platform-owned, and which are owned by the product team? [Likely] [20]
    - Where in your diagram does the model first see production-like data?
    - Which stage takes the most calendar time at a company without a platform, and why?
2. Airbnb built Bighead because shipping a model took weeks to months. Which single component of your diagram would you build first for a team that has none of it? [Likely] [20]
    - How would you measure that the platform actually shortened time to production?
3. Your interviewer says "you have no time for serving". Which two boxes of the lifecycle stay on the board, and what do you say about the rest in one sentence? [Reported] [4]
4. Walk through the lifecycle of the model in your own past project using these same boxes: architecture, implementation, challenges and how you solved them. Where did it differ, and what broke? [Reported] [4]
    - Which box would you redesign today?
5. How does the training pipeline get a training set whose features match what Deep Thought will see at inference time? [Likely] [6][20]
6. Compare scheduled retraining on Airflow with event-triggered retraining (on drift or data arrival). What would make you move a pricing model from one to the other? [Likely] [20]

### A2. Training-serving skew

7. Role postings stress training-serving skew [1]. Name every arrow in the lifecycle diagram where skew can enter, and the check that catches each. [Likely] [1]
    - Which of those checks run before launch, and which only after?
8. Why put feature transforms in a shared library rather than re-implement them in the serving service? What does that cost you in language or runtime choice? [Likely] [20]
9. A model's offline AUC is 0.82 on a holdout but its online-logged predictions score 0.76 on the same period's labels. Draw where you would look first. [Likely] [1]
    - How do you tell skew from a genuine distribution shift?
10. How would you log features at inference time so the next model can train on exactly what was served? What does that do to your ability to add new features? [Likely] [9]

### A3. Registry, versioning and reproducibility

11. What metadata must the model registry store so you can reproduce any model that served traffic in the past 90 days? [Likely] [20]
    - Is the code commit enough? What about the feature definitions and the training-set snapshot?
12. How do you link a model version to the ERF experiment that launched it, so that a later regression can be traced to a launch? [Likely] [14][20]
13. Design the promote and rollback flow. Who can promote, what gates must pass, and how fast can you roll back? [Likely]
    - What is the gate for a model whose label arrives 30 days later (cancellations, LTV)?
14. Two teams train models on the same feature set but different snapshots. How does the registry stop them from comparing offline metrics that are not comparable? [Likely]

### A4. Serving and inference

15. Design the online inference tier for a search ranker that scores several hundred candidates per request inside a tight latency budget. Where do the features come from, and what happens if the feature fetch times out? [Likely] [1]
    - Do you fall back to default values, a cached score or a simpler model, and how does each choice show up in the A/B readout?
16. When would you choose batch scoring into a table over online inference, for example for listing quality, host LTV or relisting risk? [Likely] [1]
17. How do you serve a challenger model in shadow mode on the same traffic, and what do you compare before it gets real exposure? [Likely]
18. Container-per-model isolation versus a shared multi-model server: what does each buy you at Airbnb's scale of models and traffic? [Likely] [20]

### A5. Monitoring and feedback loops

19. What would page you at 3 a.m. for a production ranker, and what is merely a dashboard? [Likely]
20. Draw the feedback loop by which the ranker shapes its own training data. Where in the lifecycle do you break it (exploration, position-bias correction, randomized slots)? [Likely] [1]
21. How do you detect that an upstream table changed meaning (for example, a new booking status) before it silently degrades every model that uses it? [Likely] [6]

---

## Diagram B. Zipline/Chronon: point-in-time-correct backfill and online serving

Fetched documentation [6]-[11] describes: **GroupBy** (keyed, windowed aggregations over a source), **Join** (many GroupBys joined to a "left" table of keys and timestamps), **accuracy** either `SNAPSHOT` (midnight values, refreshed daily) or `TEMPORAL` (real time while serving, point-in-time correct in backfills) [7], **Label Join** for adding late-arriving labels to features [8], **Bootstrap** for reusing logged or precomputed features in a backfill [9], an **online/offline consistency** job that logs fetches and replays them as a backfill [10], and an optional **tiled** online architecture that stores pre-aggregated tiles instead of raw events [11]. Zipline was the earlier version, presented in 2018 and 2020 talks [13].

```
 SOURCES                      DEFINITIONS (Python, compiled)          OFFLINE (Spark)
 Hive event tables  ---+      GroupBy: key=listing_id                 Join backfill against LEFT
 Hive snapshot tables -+--->    count(bookings, 7d/30d)  ------------> left = search impressions
 Kafka topics  --------+        avg(review_rating, 365d)              (keys + ts per row)
                         |      accuracy = TEMPORAL | SNAPSHOT         -> feature value AS OF ts
                         |    GroupBy: key=guest_id                     (point-in-time correct)
                         |      last_k(viewed_listing_ids, 10)               |
                         |    Join: left + [GroupBys]                        v
                         |    LabelPart: booked_within_7d, offset       training table
                         |                                               + Label Join (labels land late)
                         |                                               + Bootstrap (reuse logged features)
                         |
                         |   ONLINE
                         +-> daily batch upload  ----> KV store <---- streaming (Spark/Flink)
                                (midnight snapshot)      ^  ^           events or 1h tiles
                                                         |  |
                         model server --- fetchJoin(keys) --+  (batch value + stream tail merged)
                               |
                               v
                          fetch log (keys, values, ts)
                               |
                               v
       CONSISTENCY: replay logged keys+ts as a Join backfill -> compare with logged values
                    -> per-feature consistency metrics -> alert / dashboard            [10]
```

**How to draw and defend it.** Draw one definition box feeding two outputs, offline and online, because "define once, compute twice" is the whole argument. Write the **left table with timestamps** explicitly: point-in-time correctness is computed against those timestamps. Draw the consistency loop as a closed arrow from the fetch log back into the backfill, and say it is how you *measure* skew rather than assume it away [10].

### B1. Definitions and point-in-time correctness

22. Draw how "number of bookings for this listing in the last 7 days" flows from raw booking events to both a training row and an online prediction. [Likely] [6][7]
    - What timestamp do you attach to a booking event: request, acceptance or payment capture? Why does it matter for leakage?
23. What exactly does point-in-time correctness mean for a search-ranking training row whose label is "booked within 7 days of the search"? [Likely] [7][8]
    - Which features would leak the label if they were computed as of the end of the day instead of the search timestamp?
24. How would you test that a backfill is point-in-time correct? Design the unit test or data check. [Likely] [10]
25. When is `SNAPSHOT` (midnight) accuracy good enough, and when must a feature be `TEMPORAL`? Give one Airbnb feature of each kind. [Likely] [7]
    - What does choosing TEMPORAL cost you in infrastructure and in backfill compute?
26. Design the feature set for the reported "model cumulative booking metrics and first and last booking times" prompt in GroupBy and Join terms. [Reported] [2]
    - Which aggregations are reversible within a sliding window, and which (first, last, unique count) are not?
    - How do you keep first-booking time correct when a booking is later cancelled or refunded?
27. Your left table is search impressions, with billions of rows a day. How do you keep the backfill tractable (sampling, key filtering, partition pruning)? [Likely] [6][8]
28. What goes wrong when an entity source (daily snapshot of a listings table) is joined to event-level timestamps? How do you explain the resulting midnight staleness to a modeler? [Likely] [8]

### B2. Labels and late-arriving outcomes

29. Bookings, cancellations and reviews arrive days or weeks after the search. Draw how Label Join attaches them to features computed at search time. [Likely] [8]
    - How do you version a label that keeps changing (a booking later cancelled), and which version trains the model?
30. For an LTV model, where is the cutoff between the feature window and the label window on your diagram, and how do you prevent overlap? [Likely] [1][4]
    - The interviewer moves you from regression to bucketized classification. Which part of the data diagram changes? [Reported] [1]
31. How do you build a training set for a model whose label needs 30 or 90 days to mature, without the newest month of data being dropped entirely? [Likely] [8]

### B3. Online serving path

32. Draw the online fetch for a search request that ranks hundreds of listings. How many KV lookups is that, and how do you meet the latency budget? [Likely] [1][6]
    - Batch the keys, cache hot listings, or precompute per-listing vectors: which, and why?
33. How does the fetcher combine a midnight batch value with the streaming events since midnight? What happens if the stream is lagging? [Likely] [7]
34. A host with thousands of bookings is a hot key. Why does the raw-event read path degrade, and how does the tiled architecture change the read and write paths? [Likely] [11]
    - What do tiles cost you in freshness and in storage?
35. Which features of a search ranker actually need second-level freshness? Defend streaming against the reported "latency and freshness" pushback. [Reported] [1]
    - What is the measurable gain from making one feature real time, and how would you run that experiment?
36. The KV store is down for 3 minutes. What does the model server do, and how is that period treated in experiment readouts and in the next training set? [Likely] [9]

### B4. Online/offline consistency

37. A feature's online value differs from its backfilled value 3% of the time. Using the consistency loop, how do you find the cause? [Likely] [10]
    - Which causes are expected (stream lag, midnight upload delay), and which are bugs?
38. Why does the consistency job sample *all fetches of a subset of keys* rather than a random subset of fetches? [Likely] [10]
39. Which consistency threshold would block a model launch, and who owns that decision, the platform or the model team? [Likely]
40. When would you switch from backfilling a feature to "log stitching", training on logged online values via Bootstrap? What do you lose? [Likely] [9]
    - How do you add a new feature to a model whose training data comes only from logs?

### B5. Feature experimentation and governance

41. You want to A/B test a model that adds 20 new features. How do you build its training set without disturbing the production Join? [Likely] [9]
42. Two teams define "listing booking rate" differently. How does a feature platform stop duplicate, inconsistent definitions, and how does that connect to Minerva's "define once" goal for metrics? [Likely] [6][21]
43. How do you deprecate a feature that 15 models depend on? [Likely]
44. What privacy and retention constraints apply to guest-level features and fetch logs, and where in the diagram do you enforce them? [Likely] [10]

---

## Diagram C. ERF/Minerva: assignment, exposure logging, metric computation and analysis

Public descriptions, from posts confirmed by the index but not opened this session [14][15][17], and a Minerva talk that could not be reached [21]. ERF (Experiment Reporting Framework) assigns subjects, logs exposures and computes experiment metrics daily. Minerva is the metric layer that defines a metric once for dashboards and experiments. Guardrails flag launches that hurt key metrics [15].

```
 EXPERIMENT CONFIG (owner, hypothesis, unit, allocation, layers/exclusions,
                    primary + guardrail metrics from Minerva, MDE, duration)
        |
        v
 ASSIGNMENT  hash(unit_id, experiment_salt) -> bucket -> arm
   unit: visitor/device id | user id | listing | market (cluster)
        |
        v
 PRODUCT / MODEL SERVICE  --> treatment applied (e.g. ranker v2)
        |                         |
        v                         v
 EXPOSURE LOG (unit, arm,     EVENT LOGS (searches, bookings, cancellations,
  ts, surface) - logged only   CS contacts, host earnings)   + identity map
  when the treatment could     (logged-out visitor -> user after login)
  change what the unit saw
        |                         |
        +-----------+-------------+
                    v
 METRIC COMPUTATION (daily, Airflow/Spark)
   Minerva metric definitions -> per-unit, per-day aggregates
   join exposures to events after first exposure (attribution window)
   pre-period covariates for CUPED
                    |
                    v
 ANALYSIS: SRM check -> deltas, CIs (delta method / cluster-robust)
           -> variance reduction -> sequential/fixed-horizon decision
           -> guardrails [15] -> heterogeneity cuts
                    |
                    v
 DECISION: ship / iterate / kill;  holdout; long-term value translation [17]
```

**How to draw and defend it.** Draw assignment and exposure as two separate boxes: an interviewer will ask why. Write the unit of randomization on the assignment box and make sure the analysis box uses the same unit for its variance. Put the SRM check first in analysis, because nothing downstream is trustworthy without it. Name where interference breaks the diagram (shared listing inventory) and redraw the unit as a market or cluster.

### C1. Platform design (the reported prompt)

45. Design an A/B testing platform. [Reported] [2]
    - Which parts are online (latency-critical) and which are offline (batch)?
    - How many concurrent experiments must it support, and how do they avoid colliding?
46. Design an A/B testing platform that ML teams at Airbnb would use to launch ranking, pricing and trust models. What would you add for ML that a generic platform lacks? [Likely] [1][2]
    - How does the platform link a model version to an experiment arm?
47. Draw ERF's end-to-end path from a config change to a readout. Where would you spend your first quarter of engineering effort if readouts were slow? [Likely] [14]

### C2. Assignment and randomization units

48. Assignment is a hash of unit id and salt. Why a hash rather than a stored random draw, and what must the salt guarantee across experiments? [Likely] [14]
49. A guest browses logged out on mobile, then logs in on desktop and books. How do assignment and exposure work, and what does the identity merge do to your estimate? [Likely]
50. Which randomization unit fits each case: a search-ranking model, Smart Pricing suggestions for hosts, a trust model that blocks bookings, a support-routing LLM? [Likely] [1]
    - Which of these violates SUTVA through shared inventory, and how do you redraw the assignment box?
51. Design a network-aware experiment for a "Wi-Fi badge" on listings, where treatment effects spill over between listings. [Reported] [2]
    - Cluster by market, by geohash or by listing graph? How do you trade bias against power?
52. How do you run layered or mutually exclusive experiments so that two ranking teams do not test on the same traffic? [Likely] [14]
53. Design an A/B test with causal inference for a change where randomization is imperfect or impossible (for example, a policy that applies to a whole city). [Reported] [2]
    - Where does a synthetic-control or ML counterfactual (ACE) box sit relative to the ERF diagram? [Likely] [18]

### C3. Exposure logging

54. Why log exposure separately from assignment? What goes wrong if every assigned user counts in the analysis? [Likely]
    - The ranker change only affects searches with more than 50 results. Where exactly do you fire the exposure event?
55. Exposure events are lost on 2% of mobile sessions, unevenly by app version. What does that do to the readout, and how would SRM catch it? [Likely]
56. How do you log exposure for a model whose effect is server-side and invisible (for example, a fraud model that blocks a few bookings)? [Likely] [1]

### C4. Metric computation (Minerva layer)

57. Design robust primary and guardrail metrics for a search-ranking launch. [Reported] [2]
    - Which metric is the decision metric, and which are guardrails that can only block?
58. Why define metrics once in a metric layer (Minerva) rather than per experiment? What breaks when the experiment readout and the finance dashboard disagree on "bookings"? [Likely] [21]
59. Compute treatment/control booking metrics from bookings and visits tables. What is the join key, and what is the attribution window? [Reported] [2]
    - A booking happened 9 days after first exposure on a different device. Does it count?
60. Build a user-level panel and run a causal regression on it. Draw where the panel is built in the pipeline and which covariates come from the pre-period. [Reported] [2]
61. Find users with seven consecutive impression days. Where would such a metric be used in experiment analysis (engagement, novelty decay)? [Reported] [2]
62. Bookings are rare and heavy-tailed in value. Which metrics would you precompute per unit-day so that analysis stays fast and variance stays honest? [Likely]
63. Metrics are recomputed daily for thousands of experiments. Design the computation so a late-arriving partition does not silently change last week's readouts. [Likely] [14]
    - How do you version metric definitions so an old readout can be reproduced?

### C5. Analysis and decision

64. Walk through the analysis box in order. Why must the SRM check come first? [Likely]
65. Users are randomized but the metric is bookings per search. How does the variance computation change, and what happens if you treat searches as independent? [Likely]
66. Where does CUPED or regression adjustment plug into the pipeline, and which pre-period data must already be logged? [Likely]
67. Your offline ranking metric went up but the A/B on bookings is flat. Interviewers grade "how an offline metric predicts the online result" [1]. Walk through this diagram (and Diagram B) to find why. [Likely] [1]
    - Which of the following would you check first: exposure dilution, skew, metric sensitivity or a real null?
68. Two guardrails disagree: bookings up 0.4%, cancellations up 0.3%. How should the platform present this, and who decides? [Likely] [15]
69. A two-week test cannot see repeat bookings or host churn. How do you translate a short-term delta into long-term value, and where does that model sit in the diagram? [Likely] [17]
    - How would you validate the long-term proxy against holdouts?
70. Design the decision layer to allow peeking without inflating false positives. Sequential test, fixed horizon or Bayesian, and why? [Likely]
71. How should the platform report heterogeneous effects (new versus returning guests, markets) without inviting p-hacking? [Likely]

### C6. Platform operations and scale

72. The daily ERF pipeline misses its SLA during a holiday peak. What degrades gracefully, and what must never be wrong? [Likely] [14]
73. How would you add near-real-time guardrail monitoring so a harmful model launch is stopped within hours, not days? [Likely] [15]
74. How do you audit that an experiment's ship decision was made on the pre-registered metric? [Likely]

---

## Diagram D. Interleaving pipeline for search ranking

From Airbnb's 2022 interleaving post (URL confirmed, not opened) [16] and a 2025 paper on interleaving and counterfactual evaluation for Airbnb search (unverified) [19]. Interleaving merges two rankers' results into one list per search and credits each engaged or booked listing to the ranker that contributed it. Airbnb reported it as a faster, more sensitive screen before A/B tests [16].

```
 search request (guest, query, dates)
        |
        +---> interleaving assignment (hash guest_id -> interleaving slice;
        |                              rest of traffic -> normal A/B layers)
        v
 candidate retrieval (shared)  ---> ranker A scores   ranker B scores
                                         \               /
                                          v             v
                                   TEAM-DRAFT MERGE
                                   (coin flip per round; each team picks its
                                    best remaining listing; record team label)
                                          |
                                          v
                         results page + map (dedupe; pagination keeps teams)
                                          |
                                          v
 LOG: search_id, guest_id, position, listing_id, team (A/B/both), ts
        |
        v
 ATTRIBUTION: join clicks, wishlists, booking requests, bookings (within window)
              -> credit to team of the listing; drop "both" or split
        |
        v
 PER-GUEST PREFERENCE: wins_A vs wins_B -> preference score
        |
        v
 STATS: sign test / bootstrap over guests -> direction, magnitude, CI
        |
        v
 GATE: winner goes to full A/B (Diagram C); track interleaving-vs-A/B agreement
```

**How to draw and defend it.** Show retrieval as shared and only the ordering as interleaved, so the interviewer sees interleaving compares rankers, not candidate sets. Write "team label logged per position" on the merge box, because attribution is impossible without it. Draw the gate into Diagram C explicitly and name what interleaving cannot measure (price display, result-set size, set-level diversity), which is why the A/B still runs.

### D1. Mechanics

75. Explain team-draft interleaving on this diagram. Why team draft rather than balanced interleaving? [Likely] [16]
76. Search ranking rounds have been reported to push on experimentation [1]. Where in your launch funnel does interleaving go, and what gates sit before and after it? [Likely] [1][16]
    - What does the offline gate check that interleaving cannot?
77. The two rankers share most of their top results. How does that affect sensitivity, and how do you handle listings both rankers placed? [Likely] [16]
78. How do you keep a guest's interleaved experience consistent across page 2, the map view and a repeated search the next day? [Likely] [16]

### D2. Attribution and metrics

79. Bookings happen days after a search, often after several searches. How do you attribute a booking to a team in an interleaved list? [Likely] [16]
    - Last search, first exposure or all exposures? Defend one.
80. Clicks are plentiful but bookings are the goal. Which interleaving metric would you trust for a go/no-go, and how would you show it agrees with A/B booking results? [Likely] [16][19]
81. Why is per-impression significance wrong here, and what is the unit of analysis for the interleaving statistics? [Likely]
82. How do you combine interleaving signals with counterfactual (off-policy) evaluation from logged data? Where does each box sit in the funnel? [Likely] [19]

### D3. Validity and limits

83. Name ranking changes that interleaving cannot evaluate and must go straight to A/B. Why does each break the merge? [Likely] [16]
84. A diversity-aware ranker optimizes the whole list. Why can interleaving give a misleading verdict for it? [Likely] [16]
85. Interleaving says B wins clearly; the A/B on bookings is flat. Using Diagrams C and D, list the hypotheses in the order you would test them. [Likely] [16]
86. Can interleaving and A/B tests share traffic? Draw how you would allocate slices so neither contaminates the other. [Likely] [16]
87. How would you build a historical agreement dataset of interleaving versus A/B outcomes, and what agreement rate would make you skip the A/B for small changes? [Likely] [16][19]

### D4. Infrastructure and cost

88. Interleaving doubles the scoring cost for the slice. How do you stay inside the latency budget? [Likely] [1]
89. What must be logged per position for the analysis to be reproducible, and how large is that log per day? [Likely]
90. How do you run 10 candidate rankers through interleaving in a week? Pairwise tournaments or multileaving? [Likely] [16]

---

## Cross-diagram probes (connecting A, B, C and D)

91. Draw all four systems on one board in five minutes: where do Bighead, Chronon, ERF and interleaving touch? [Likely]
    - Which arrow carries the most risk of silent error?
92. A new ranker passes offline evaluation, wins interleaving, and loses the A/B on cancellations. Trace the path backwards through the four diagrams. [Likely] [15][16]
93. Which logs must be joined to turn one search into (a) a training row, (b) a consistency check, (c) an interleaving credit and (d) an A/B metric? Can one logging design serve all four? [Likely] [9][10]
94. The experiment arm is a feature in the model (for example, "guest saw new pricing UI"). How do you stop the next model from learning the experiment instead of the user? [Likely]
95. Rank Home search results without a text query. Point to where Chronon features, the model lifecycle and the experiment pipeline enter your design. [Reported] [2]
96. Design user-embedding semantic search. How are embeddings versioned in the registry, served from the feature path, and A/B tested when the index changes? [Reported] [2]
97. Design Airbnb search ranking from candidate generation through offline and online evaluation and deployment. Which of the four diagrams would you draw in full, and which as a single box? [Reported] [1]
98. For a pricing model (reported as a design prompt [4][5]), how do the experiment unit and the feature freshness requirements differ from search ranking? [Likely] [4][5]
99. If you owned the ML platform for a year, which metric would you use to show that Bighead, Chronon and ERF together made ML teams faster? [Likely]
100. Which piece of these four systems would you buy or adopt from open source, and which must Airbnb build itself? [Likely] [6]

---

## Section E. Reported prompts mapped onto the four diagrams (added in review)

Each prompt below comes from a fetched source: the dated index [2], the MLE guide [1] or a 1point3acres mirror [4][26]. The sources give only the prompt titles. The diagram angle added to each is an inference for this role. The follow-up probes are [Likely] unless they cite a source.

### E1. Product ML prompts: which diagram to zoom into

101. Estimate listing lifetime value (reported May 2026). Draw the label window, the feature cutoff and the batch-scoring path from Diagram A. [Reported] [2]
    - "Agree on what lifetime value means here" first [1]. Which definition changes the Chronon Join's label offsets [8]?
    - The scores land in a Hive table nightly. Who consumes them, and how would you A/B test a model whose output is used by a pricing or sales team, not shown to guests?
102. Design a system to calculate customer lifetime value. Which part of the lifecycle do you collapse when the interviewer has no time for deployment and serving? [Reported] [4]
103. Design the home page: search, availability and ranking (reported Mar 2026). Where does the availability filter sit relative to retrieval and the ranker, and which features have to be fresher than the midnight batch upload? [Reported] [2][7]
    - Availability changes minute to minute. Is it a feature, a filter or both? What does each choice do to the training data?
104. Design a personalized search ranking system. Show where guest-level features come from (Diagram B) and how a new ranker is screened by interleaving (Diagram D) before an A/B test (Diagram C). [Reported] [1][2]
    - Real-time personalization from in-session clicks: which GroupBy is TEMPORAL, and what is its window? [Likely] [7][24]
105. Pick a listing's cover photo to raise its booking rate, with cold start for new listings. Which randomization unit do you use: listing, guest or search? Why does the choice change the exposure log and the analysis? [Reported] [1]
    - The treatment is a listing attribute every guest sees. How do you avoid hosts seeing their photo flip between arms?
106. Find high-quality listings (reported as a phone-screen design prompt). The model's output is a score used by search, by marketing and by trust teams. Draw it as a batch-scored entity feature. How do downstream rankers consume it point-in-time correctly? [Reported] [1][8]
107. Design a relisting system, including "how its outputs show up in online metrics and how geohashing helps". Where does geohash enter the Chronon keys, and where does it enter the experiment unit if you cluster by area? [Reported] [1][2]
108. Design a system to help hosts set prices. Draw the experiment: host-level randomization, market-level spillover through guest substitution, and the guardrails you watch. [Reported] [4]
    - Hosts can override the suggestion. Which metric captures adoption, and which captures the effect?
109. Design listing recommendation and price prediction (two ML design rounds in one loop). Which pieces of the platform diagrams do the two designs share, and what would you say once and refer back to in the second round? [Reported] [5]
110. Generate customer-support instructions from company policies (reported Sep 2026). Draw the lifecycle for an LLM-based system: offline evaluation set, registry entry for prompt plus model version, shadow traffic and an A/B on resolution and recontact rates. [Reported] [2]
    - What is the analogue of training-serving skew when the policy documents change weekly?

### E2. Data and infrastructure prompts that map onto the pipelines

111. Design an ad-click event aggregation system (reported Jan 2026). Redraw it as the exposure-and-event path of Diagram C: what changes when the counts feed an experiment readout instead of billing? [Reported] [2]
    - Exactly-once counts versus late events: which matters more for a daily experiment readout, and which for a real-time guardrail?
112. The same ad-click aggregator, but feeding online ranking features. Compare it with Chronon's streaming write path, untiled and tiled. [Likely] [2][11]
113. Design a job scheduler (reported Mar 2026; also reported in a G9 loop). Use it as the scheduler behind ML Automator retraining and the daily ERF metric pipeline. How do you express dependencies on upstream partitions landing, and how do you backfill a failed day? [Reported] [2][26]
    - A retrain depends on a label partition that is three days late. What does the scheduler do, and what does the registry show?
114. Design a query system for time- and geo-filtered user activity (reported Nov 2025). Which indexes would you build so an analyst can slice an experiment readout by market and date without rescanning raw logs? [Reported] [2]
115. Optimize host listings-page aggregations (reported Jan 2026). Draw the precomputed per-host aggregates as a Chronon GroupBy. Which aggregations can be updated incrementally, and which need a periodic batch correction? [Reported] [2][7]
116. Model cumulative booking metrics and first and last booking times (reported Sep 2026). Show where these tables sit: Minerva metric source, Chronon feature source, or both, and who owns the definition. [Reported] [2][21]

### E3. Metric-computation prompts behind the analysis box

117. Estimate page-visit duration from the next user event (SQL, reported Sep 2026). Where in the ERF metric pipeline would you compute it, and how do you treat the last event of a session, which has no next event? [Reported] [2]
    - Does a treatment that changes app navigation bias this duration metric? How would you detect that?
118. Compute browsing metrics in Python from raw logs. What would make you move this from an ad hoc script into the shared metric layer? [Reported] [2][21]
119. Compute treatment and control metrics from bookings and visits tables, then add a ratio metric (bookings per visit). How does the per-unit pre-aggregation change so the delta-method variance is computable? [Reported] [2]

### E4. Chronon depth probes grounded in the fetched documentation

120. Sliding windows are exact but must store every raw event; hopping windows bound memory but miss recent events. When would you accept Chronon's sawtooth window for a 30-day booking count in search ranking? [Likely] [7]
121. A GroupBy's batch upload runs weekly to save compute, but the KV store's TTL is five days. What happens at serving time, and how would the platform stop this configuration from being deployed? [Likely] [7]
122. A listings table arrives as a stream of inserts, updates and deletes. Which aggregations (average versus max) can be corrected in place when a row is deleted, and how does the online value become correct for the others? [Likely] [7]
123. You add a feature keyed by host_id, but your log-based left table only has guest_id and listing_id. Draw the key backfill needed before the Join can compute it. [Likely] [9]
124. Why does the online fetch need to carry a row_id if the training set will be built from fetch logs? What breaks in the label join without it? [Likely] [9]
125. A modeler renames the left table of a production Join to clean up naming. What happens to the historical training data, and how should the platform guard against this? [Likely] [8]
126. The consistency job reports distributions through quantile sketches, not only mismatch rates. Which skew does a distribution comparison catch that a per-row equality check misses, and the reverse? [Likely] [10]

### E5. Senior-bar framing

127. One rejection cited "jumping straight to model ideas" [3]. In a design round, how long do you spend on the label and data diagram before you name a model, and what do you draw first? [Likely] [3]
128. Senior loops are described as "close to staff level" [3]. Which platform-level trade-off (shared transforms, metric layer, interleaving gate) would you raise unprompted to show staff-level judgment, and how do you keep it short? [Likely] [3]
129. Airbnb has written about building a model to predict the value of homes [22]. If you were the first product team on a new ML platform, which platform gaps would you accept and which would you block on? [Likely] [22]
130. Airbnb's search team has written about lessons from moving search ranking to deep learning [23] and about ranking for diversity [25]. Pick one lesson, then show which box in Diagram A or D it would change. [Likely] [23][25]

---

## References

1. "Airbnb Machine Learning Engineer Interview Process" (MLE prompts; "conversations without code"; search ranking rounds "pushed on latency, freshness and experimentation"; postings stress training-serving skew and A/B testing). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb" question index with report dates (Design an A/B Testing Platform, Apr 21 2026; Model Cumulative Booking Metrics and First and Last Booking Times, Sep 2026; Rank Home Search Results Without a Text Query, Sep 2026; Design User Embedding Semantic Search, Apr 2026; Find Users with Seven Consecutive Impression Days, Sep 2026; undated: primary and guardrail metrics, A/B test with causal inference, network-aware Wi-Fi badge experiment, C/T metrics from bookings and visits, panel and causal regression). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
3. "Airbnb interview process" (MLE candidates get one or two ML system design rounds; 45-60 minutes; senior loops "close to staff level"; rejection for "jumping straight to model ideas"). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. Fetched. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. 1point3acres post #40, English mirror (MLE loop: project deep dive; ML design on customer LTV, with little time for deployment and serving; host pricing system). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
5. 1point3acres post #79, English mirror (MLE loop: ML design on listing recommendation and price prediction; "ML experience" round turned into an unplanned ML design question). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
6. "Chronon: a data platform for AI/ML" (README: GroupBy, Join, point-in-time backfills, KV-store serving, fetcher, consistency measurement). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon
7. Chronon documentation, "GroupBy" (aggregations, windows, SNAPSHOT versus TEMPORAL accuracy, online upload). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/GroupBy.md
8. Chronon documentation, "Join" (left table, entity versus event sources, Label Join and label offsets). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/Join.md
9. Chronon documentation, "Bootstrap" (log stitching, external feature backfills, production versus experimental Joins, data recovery). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon/blob/main/docs/source/authoring_features/Bootstrap.md
10. Chronon documentation, "Online Offline Consistency" (causes of inconsistency, query-log table, key-subset sampling, replay and compare). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon/blob/main/docs/source/test_deploy_serve/Online_Offline_Consistency.md
11. Chronon documentation, "The Tiled Architecture" (pre-aggregated tiles via Flink, hot keys, latency). airbnb/chronon, GitHub. Fetched. https://github.com/airbnb/chronon/blob/main/docs/source/Tiled_Architecture.md
12. Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & machine learning in production." eugeneyan/applied-ml, GitHub. Fetched (raw README re-read in review); used to confirm [13]-[18] and [22]-[25]. https://github.com/eugeneyan/applied-ml
13. Simha, "Zipline: Airbnb's Machine Learning Data Management Platform" (2018) and "Zipline - A Declarative Feature Engineering Framework" (2020), conference talks, YouTube. URL confirmed. https://www.youtube.com/watch?v=Tg5VEMEsC-0 ; https://www.youtube.com/watch?v=LjcKCm0G_OY
14. "Scaling Airbnb's Experimentation Platform" (ERF), Airbnb Tech Blog (Medium), 2017. URL confirmed. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
15. "Designing Experimentation Guardrails," Airbnb Tech Blog (Medium), 2021. URL confirmed. https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669
16. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving," Airbnb Tech Blog (Medium), 2022. URL confirmed. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
17. "How Airbnb Measures Future Value to Standardize Tradeoffs," Airbnb Tech Blog (Medium), 2021. URL confirmed. https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5
18. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb," Airbnb Tech Blog (Medium), 2022. URL confirmed. https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512
19. Zhang et al., "Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking," arXiv 2508.00751, 2025. Unverified (arxiv.org unreachable in both passes; not listed in [12]). https://arxiv.org/abs/2508.00751
20. "Bighead: Airbnb's End-to-End Machine Learning Platform," Data Council talk. Unverified (component names from prior knowledge). https://datacouncil.ai/talks/bighead-airbnbs-end-to-end-machine-learning-platform
21. "Ep 1: Robert Chang on Scaling Metrics @ Airbnb" (Minerva), dbt Roundup. Unverified. https://roundup.getdbt.com/p/ep-1-robert-chang-on-scaling-metrics
22. "Using Machine Learning to Predict Value of Homes On Airbnb," Airbnb Tech Blog (Medium), 2017. URL confirmed by [12]; page not opened. https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d
23. Haldar et al., "Applying Deep Learning To Airbnb Search," arXiv 1810.09591, 2018. URL confirmed by [12]; page not opened. https://arxiv.org/abs/1810.09591
24. "Listing Embeddings in Search Ranking" (real-time personalization using embeddings), Airbnb Tech Blog (Medium), 2018. URL confirmed by [12]; page not opened. https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e
25. "Learning To Rank Diversely," Airbnb Tech Blog (Medium), 2022. URL confirmed by [12]; page not opened. https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621
26. 1point3acres post #80, English mirror (G9 loop: system design on group chat and a job scheduler; the experience round decided the outcome). FlowSight/coding, GitHub. Fetched. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/80.md
