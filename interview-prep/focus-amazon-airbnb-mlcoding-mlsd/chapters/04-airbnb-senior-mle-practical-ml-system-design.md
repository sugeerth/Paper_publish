# 04. Airbnb Senior MLE: Practical ML System Design

> **Scope.** This chapter covers the ML system design rounds in the Airbnb Senior Machine Learning Engineer loop, including the experimentation and A/B testing half of those rounds, for 2024-2026. It lists questions only and gives no answers or solutions.
>
> **How it relates to the older chapters.** Four new search passes (October 2026) were merged and deduped with the prompts in `airbnb-senior-mle/chapters/03-ml-system-design.md`, `10-ml-system-design-architecture-diagrams.md` and `04-ab-testing-experimentation.md`. Those chapters keep the long tail: 140+ search, embedding, pricing and trust sub-questions, full architecture diagrams, and the ERF, CUPED, interleaving and sequential-testing deep dives. This chapter keeps the 24 strongest design prompts. Each one gets a skeleton diagram of the boxes a strong answer must cover. The boxes name components only; they do not say which choice to make.
>
> **Source caveat.** Almost every source is an anonymous self-report (PracHub [19][20][21], 1point3acres, Blind, or TrueInterview-derived GitHub indexes) or a prep-vendor page. Many were read only as search-result snippets or login-gated teasers. Read each [Reported] tag as "a candidate said this was asked", not as verified fact.

Tag legend:

- **[Reported]**: a cited candidate report says the prompt, or a close paraphrase, was asked in an Airbnb MLE loop.
- **[Likely]**: a grounded inference from Airbnb papers, the engineering blog, job postings or prep guides. Nobody was seen being asked it verbatim. A citation next to [Likely] points to the system or source that makes it plausible.

Follow-up bullets are untagged unless a source reports them. They are the probes interviewers typically pull on.

---

## How this round works

- **Where it sits in the loop.** Expect the design rounds to decide the outcome. Reported loops:
  - An April 2026 Senior MLE loop had **four rounds built around ML systems, ranking and marketplace scenarios**, with past-project deep dives mixed in. It ended in rejection "on calibration" [2].
  - A September 2026 Senior+ loop (Relevance and Personalization) covered document redaction, project decisions, search ranking without a text query, and retrieval-assisted support. Team matching followed [1].
  - A TrueInterview-derived summary says the loop is "filtered by design depth, not by code", with 1-2 decisive ML design rounds [9].
  - 1point3acres loops had two ML design rounds each: LTV plus host pricing [11]; listing recommendation plus price prediction [12]; property damage claims plus pricing [3].
  - An ML platform loop had a coding screen, then an onsite with code review, one general system design round, one ML system design round and behavioral [7].
- **Two flavors of design round.**
  - The **ML design** round is "typically a search or recommendation question" [6].
  - The **ML practitioner** round asks you to propose an ML solution to one of Airbnb's real, current engineering problems [6].
  - The **ML experience** (project) round can turn into an unplanned design question [8][12].
  - Prep guides also list a "debug a model with unexpected behavior" round [22].
- **Duration and tools.** Rounds last about 45-60 minutes over video, with a shared doc or whiteboard [9]. You walk through the design verbally; there is no code to run. Interviewers steer partway through, for example from regression to bucketized classification in the LTV round [9].
- **What the senior bar looks like.**
  - Drive the problem from business goal to target definition, labels, features, model, offline evaluation, online experiment, serving, and monitoring and debugging in production [1][9].
  - Defend every choice against a simpler and a more complex alternative. Reported probes ask why not a more complex model, how you would trade off precision, recall, latency and infrastructure cost, how you would tune thresholds, and how you would choose between factorization machines and an embedding plus DNN model [2].
  - Show business sense. The April 2026 write-up says "the team really value business sense and decision making" [2].
  - Show two-sided marketplace thinking, covering guests, hosts, supply and demand [6][24].
- **Experimentation weight for this role.** Assume every design ends with "how would you launch and measure it?". Prepare for interference-aware randomization (guest, listing, market or two-sided), guardrails, lead-day bias in pricing tests, and what to do when offline and online metrics disagree [2][53][54][56].

---

## Top 15 to practice first

All 15 are [Reported] prompts, ordered by number of independent reports, then by importance.

1. Search ranking with no text query (destination, dates, guests only). [Reported] [1][10][13] (see Q1)
2. Pricing prediction and Smart Pricing price recommendations for hosts. [Reported] [3][4][5][11][12] (see Q2)
3. Retrieval-assisted (RAG) customer support for guests, hosts and agents. [Reported] [1][9][17] (see Q3)
4. User embedding and two-stage semantic retrieval and ranking for booking conversion. [Reported] [2][10][14] (see Q4)
5. Offline metrics win but the online A/B test is flat or negative: diagnose it and decide whether to ship. [Reported] [2] (see Q5)
6. Guest or listing lifetime value (LTV). [Reported] [9][10][11][15] (see Q6)
7. Photo and listing quality models that feed search. [Reported] [9][19][20] (see Q7)
8. Property damage claims: triage, payout estimate and fraud. [Reported] [3] (see Q8)
9. "Family friendly" listing classifier with noisy host labels and cold start. [Reported] [2] (see Q9)
10. Threshold choice and model choice (FM vs embedding plus DNN; precision, recall, latency, cost). [Reported] [2] (see Q10)
11. Real-time content personalization (homepage and categories feed). [Reported] [2][22] (see Q11)
12. PII and sensitive-information detection and redaction in documents and messages. [Reported] [1] (see Q12)
13. Listing recommendation and similar-listing embeddings. [Reported] [12] (see Q13)
14. Relisting detection (the same property relisted under a new listing). [Reported] [9][16] (see Q14)
15. The ML parts of the accommodation booking funnel (booking likelihood from search to stay). [Reported] [2] (see Q15)

---

## 1. Search ranking without a text query

**Prompt.** "Design an ML ranker for Airbnb home search when the user gives no text query, only destination, dates and guest count. Separate the features available at ranking time from future outcomes, set product objectives and guardrails, then pick labels and an architecture. Cover data, evaluation and how you would debug it in production." [Reported] [1][10][13] (Senior+ Relevance and Personalization onsite, September 2026. The broader "design Airbnb search ranking / personalized search ranking" prompt is also reported [6][9][18].)

**Maps to:** Airbnb stays search ranking, which orders listings by estimated booking probability, evolved from GBDT to deep neural networks [44][45][46]. It also maps to the Relevance and Personalization team [60].

```
 SERVING
 [Query: dest, dates, guests] -> [Availability + geo filter] -> [Candidate retrieval]
          -> [First-pass ranker] -> [Second-pass ranker] -> [Re-rank: diversity,
             business rules] -> [Results list / map]
 [Feature store: listing | guest | query | real-time session] --> both rankers
 TRAINING LOOP
 [Impression + click + booking logs] -> [Label + position-bias handling]
 -> [Point-in-time feature join] -> [Train] -> [Offline NDCG eval]
 -> [Interleaving / A/B] -> [Monitoring + prod debugging] -> back to logs
```

Follow-ups:
- With no query text, what replaces query relevance: location intent, dates, price range, guest history, season?
- Which features exist at ranking time, and which are future outcomes that would leak into training (eventual booking, later reviews, host cancellation)?
- What is the primary objective (bookings, booking value, uncancelled stays, long-term guest and host value), and which guardrails protect hosts and marketplace health?
- Bookings are rare and many searches end without one. Would you train on bookings, clicks, long views, wishlists or contacts to the host? How would you combine them?
- How do you treat a booking request the host declined or later cancelled?
- Would you frame the model as pointwise, pairwise or listwise, and how do you build training pairs from a session with one booked listing?
- How do you correct for position bias in logged results, and what position value do you feed at inference time?
- How do you give cold-start listings controlled exposure without hurting conversion?
- When does moving from a GBDT to a neural ranker pay off, and what usually breaks (feature scaling, sparse IDs, price sensitivity)?
- What is the latency budget for each stage, and which features are batch versus real-time?
- How would ranking change for the map view, for Experiences, or for long-term stays? [Likely] [29][47]

## 2. Pricing prediction and Smart Pricing

**Prompt.** "Design a pricing prediction / price recommendation (Smart Pricing) model for Airbnb hosts. Cover the features (listing, seasonality, demand, comparable listings), the label when you only observe booked prices, and how you evaluate it." [Reported] [3][4][5][11][12] (1point3acres onsites 2022-2025: "ML round 1: a price prediction system" [5]; "design a system to help hosts set prices" [11]; "price prediction" [12]. Prep guides also list "dynamic pricing recommendation engine" [22][25][26].)

**Maps to:** Airbnb dynamic pricing (Smart Pricing), which combines a booking-probability model with a customized price regression [51]. It also maps to the host calendar's comparable-listing embeddings [42].

```
 SERVING
 [Listing + night + lead time] -> [Demand / booking-prob model] -> [Price strategy
          model] -> [Host constraints: min/max, rules] -> [Suggested price + reason]
 [Comps: similar listings] [Seasonality + events] [Market supply/demand] --> models
 TRAINING LOOP
 [Calendar + price + booking logs] -> [Label: booked / unbooked nights, censoring]
 -> [Train] -> [Offline eval: price regret, calibration] -> [Host adoption logs]
 -> [Pricing A/B with lead-day handling] -> [Monitoring] -> back to logs
```

Follow-ups:
- What exactly are you predicting: the market-clearing price, booking probability at a given price, or the host's revenue-maximizing price?
- You only see the price for booked nights and the host's asking price for unbooked ones. How do you build a label from that censored, host-chosen data?
- Observed prices are set by hosts, not randomized. How do you separate the price effect from listing quality (endogeneity)?
- What feature families matter (listing attributes, location, seasonality, local events, lead time, comps, market occupancy)?
- How do you pick comparable listings, and could listing embeddings do it better than hand-built rules?
- What offline metric fits when the "right price" is never observed?
- How do you respect host goals (occupancy versus nightly rate) and host-set minimums?
- How do you explain a suggestion so that hosts adopt it?
- Many hosts in one market adopt the suggestion. How do you account for that feedback loop?
- Would you randomize a pricing test by listing, by night or by market? Why does lead-day bias matter? [Likely] [56]
- How would you replace a rule-based pricing function (weekend, event and competitor bounds) with ML, and how would you prove it is better? [Likely] [27]
- How does the system behave for a brand-new listing with no booking history?
- How would damage or cancellation risk enter pricing or security-deposit decisions? (One reported loop paired a pricing round with a damage-claims round.) [Likely] [3]

## 3. Retrieval-assisted (RAG) customer support

**Prompt.** "Design a retrieval-assisted customer support system for Airbnb guests and hosts that answers questions or helps agents. Cover knowledge-base retrieval, LLM generation, evaluation, guardrails and escalation to humans." [Reported] [1][9][17] (Senior+ onsite, September 2026 [1]. 2026 senior loops added retrieval and ranking prompts for GenAI support tools [9]. A related prompt is "generate customer-support instructions from company policies" [17].)

**Maps to:** Airbnb Community Support chat and voice AI assistants. The team's postings list SFT, RL, distillation, RAG/search, LLM evaluation, feedback-based learning and guardrails [62][64][65]. An October 2025 paper covers LLMs in Airbnb support [67], and an earlier case study covers support text generation [68].

```
 SERVING
 [Guest/host message + reservation ctx] -> [Intent + risk classifier]
   -> [Retriever: help center, policies, past cases] -> [Re-ranker]
   -> [LLM generator (distilled)] -> [Guardrails: policy, PII, safety]
   -> [Answer | agent suggestion | escalate to human]
 TRAINING / EVAL LOOP
 [Tickets + resolutions + CSAT] -> [Human + LLM-judge labels] -> [SFT / pref tuning]
 -> [Offline eval: groundedness, resolution] -> [A/B] -> [Feedback logs] -> back
```

Follow-ups:
- Which intents should the bot answer itself, which should it draft for an agent, and which must go straight to a human (safety, emergencies, high-value refunds)?
- How do you chunk and index help-center and policy content that changes often and differs by country?
- How do you retrieve reservation-specific context (dates, cancellation policy, payment status) safely?
- Would you use RAG, SFT or both? When does distilling into a smaller model pay off on latency and cost?
- How do you build an offline evaluation set, and which metrics matter (groundedness, policy correctness, resolution, tone)?
- When can you trust an LLM-as-judge, and how do you calibrate it against human reviewers?
- Which guardrails stop the model promising refunds or exceptions that policy does not allow?
- How do you detect and hand off escalations without losing context?
- Which online metrics decide the launch (resolution rate, contact rate, CSAT, handle time), and would you randomize by ticket, by user or by agent?
- How do you learn from agent edits and user feedback without amplifying the model's own mistakes?
- How do you protect against prompt injection from user messages or listing text?
- What is the per-turn latency budget for chat versus voice?

## 4. User embeddings and two-stage semantic retrieval and ranking

**Prompt.** "Design a user embedding plus two-stage semantic retrieval and ranking system for Airbnb search, optimized for booking conversion. Walk through candidate generation, the ranker, features, labels and online serving." [Reported] [2][10][14] (April 2026 Senior MLE onsite [2]. "Design user-embedding semantic search", reported April 2026 [10][14].)

**Maps to:** Airbnb embedding-based retrieval (a two-tower model with listing embeddings precomputed daily, targeted negative sampling, IVF chosen over HNSW, and traffic-replay evaluation) [39][40][41]. It also maps to user-type embeddings [43] and BiListing multimodal listing vectors [52].

```
 SERVING
 [Guest ctx + query] -> [User / query tower] -> [ANN index (listing tower,
     precomputed)] + [Availability / geo / price filters] -> [Top-K candidates]
   -> [Ranker (booking objective)] -> [Results]
 [Batch: listing embeddings daily] [Stream: session events -> user embedding]
 TRAINING LOOP
 [Search sessions + bookings] -> [Positives + negative sampling] -> [Two-tower train]
 -> [Offline recall@K, traffic replay] -> [Index rebuild] -> [A/B] -> back
```

Follow-ups:
- How do you build a user embedding from sparse, long-gap travel histories, and how do you refresh it in-session?
- What counts as a positive (booked, wishlisted, long view), and which negatives do you sample (in-batch, same-market, hard negatives)?
- Why might random negatives produce embeddings that mostly learn geography?
- Price and availability change daily. What stays inside the embedding, and what goes into filters at query time?
- When would you pick IVF over HNSW, given index rebuild cost and frequent filtering?
- How do you evaluate retrieval recall separately from end-to-end ranking quality, when inventory changes every day?
- How do you share the latency budget between retrieval and ranking? Each stage scores about an order of magnitude fewer homes than the one before [40].
- How do you represent a new guest or a new listing in the embedding space?
- How do you keep retrieval from narrowing what the ranker can ever learn from (a feedback loop)?
- How would you fold photo and text embeddings into one listing vector, and how would you measure the NDCG and revenue impact? [Likely] [52]
- What do you A/B test first, retrieval or ranker, and how do you attribute the gain?

## 5. Offline wins, online flat: diagnose and decide

**Prompt.** "Your new ranking model wins on offline metrics such as NDCG/AUC but is flat or negative in the online A/B test. How do you diagnose the gap, and what do you ship?" [Reported] [2] (April 2026 Senior MLE: "What do you do when the offline metric and the online metric disagree?". A related prompt, "debug a model with unexpected behavior", appears in prep guides [22], and the Senior+ report advises preparing production debugging [1].)

**Maps to:** Airbnb's deep-learning search papers, which describe many offline wins that failed online [44][45], plus interleaving for faster ranking tests [58].

```
 [Offline eval set] --vs-- [Online A/B readout]
        |                          |
 [Label / objective check]  [Experiment health: SRM, exposure, power, novelty]
 [Leakage + split check]    [Interference between arms]
 [Candidate-set mismatch]   [Segment / market breakdown]
        \                         /
       [Training-serving skew: feature parity + latency]
                    |
         [Decision: ship / iterate / kill] -> [Post-launch monitoring]
```

Follow-ups:
- What is your first check, the experiment or the model?
- How do you test for training-serving skew, feature leakage and stale features?
- Was the offline candidate set the same as what the online ranker actually scored?
- Could position bias or a feedback loop (training on control's rankings) explain the gap?
- Does the offline metric measure the online objective (bookings versus clicks, uncancelled stays)?
- Did the new model add latency, and how would you separate the latency effect from the quality effect?
- Could interference between arms or novelty effects hide a real gain?
- Which segments or markets moved in opposite directions?
- Would interleaving have given a faster or clearer read? [Likely] [58]
- The result is +0.3% bookings with a wide interval, plus extra latency and infrastructure cost. Do you ship? [Likely]
- How would you build an offline metric that predicts online wins better next time?
- Conversion drops in one market the day after a new ranker ships. Walk through rollback criteria and root-cause analysis (data, features, serving skew, model). [Likely] [1][22]
- The interviewer pushes back on your launch call. How do you defend it on business grounds? [Reported] [2]

## 6. Guest or listing lifetime value (LTV)

**Prompt.** "Design a system to calculate customer (or listing) lifetime value." The interviewer first made the candidate define "lifetime value", then steered the design from regression to bucketized classification. [Reported] [9][10][11][15] (1point3acres loop [11]; listing LTV reported May 2026 [10][15].)

**Maps to:** Airbnb's future-value measurement, used to standardize trade-offs across teams (see the older chapter `airbnb-senior-mle/chapters/03-ml-system-design.md`).

```
 SERVING
 [Guest / listing entity] -> [Feature snapshot at time t] -> [LTV model]
   -> [Calibrated value or value bucket] -> [Consumers: marketing, ranking,
      host acquisition, incentives]
 TRAINING LOOP
 [Booking + revenue + churn history] -> [Target definition: horizon, margin]
 -> [Censoring-aware labels] -> [Train] -> [Offline eval: calibration, rank]
 -> [Backtest by cohort] -> [Downstream A/B] -> [Drift monitoring] -> back
```

Follow-ups:
- How do you define LTV: revenue or contribution margin, over what horizon, and for which entity?
- How do you handle right-censoring for recent users and listings?
- Why might bucketized classification beat regression on a heavy-tailed target?
- How do you predict LTV for brand-new guests or listings?
- Which features are allowed without leaking future behavior?
- How do you evaluate it: calibration by decile, rank correlation, or business backtests?
- Who consumes the score, and how would each consumer misuse it?
- How would you validate that decisions based on LTV (incentives, host acquisition spend) actually raise long-term value?
- How often do you retrain, and how do you detect drift after a macro shock (for example, a travel slowdown)?

## 7. Photo and listing quality models

**Prompt.** "Design photo and listing quality models for Airbnb. Cover the labeling strategy, feature engineering (image and text), model selection, and how quality scores feed into search." [Reported] [9][19][20] (PracHub-listed Airbnb prompt [19][20]; cover-photo and high-quality-listing prompts in a 2026 summary [9].)

**Maps to:** Airbnb Categories built with ML and human-in-the-loop review [70], and to listing-quality models used in merchandising and ranking [20].

```
 SERVING
 [Listing photos + text + amenities] -> [Image encoder] + [Text encoder]
   -> [Quality model / cover-photo selector] -> [Quality score + reasons]
   -> [Consumers: search ranking feature, host tips, merchandising]
 TRAINING LOOP
 [Human ratings] + [Engagement proxies] + [Review signals] -> [Label strategy]
 -> [Train + calibrate] -> [Offline eval: agreement with raters]
 -> [Ranking A/B + host-side metrics] -> [Rater audits] -> back
```

Follow-ups:
- What does "quality" mean: photo aesthetics, accuracy versus reality, completeness, or predicted guest satisfaction?
- Would you label with human raters, engagement proxies (CTR on a photo) or post-stay reviews? What bias does each bring?
- How do you keep the model from learning that "expensive-looking" equals "high quality"?
- How do you pick the cover photo per listing, and could it be personalized per guest?
- How would the score enter search: as a ranker feature, as a filter, or as a boost? What feedback loop does that create?
- How would you measure the model's effect on hosts (photo uploads, edits) as well as guests?
- How do you handle new listings with few photos, or photos that are not of the home?
- What inter-rater agreement do you need before you trust the labels?
- How do you serve image models cost-efficiently: batch on upload, or on demand?
- How do you audit quality scores for bias against particular regions or property types?

## 8. Property damage claims

**Prompt.** "Design an ML system for property damage claims, for example triaging and estimating claims or detecting fraudulent ones, using claim text, photos and host/guest history. Cover labels, features, class imbalance, human review and metrics." [Reported] [3] (1point3acres Airbnb MLE onsite, 2025.)

**Maps to:** Airbnb's support products. A staff MLE posting lists "automated property damage claim item entity extraction" and "agent tools to investigate claims and possible fraud through real-time image, text, and user similarity search" [63].

```
 SERVING
 [Claim: text + photos + receipts] -> [Entity extraction: items, amounts]
   -> [Similarity search: image / text / user] -> [Validity + fraud model]
   -> [Payout estimator] -> [Router: auto-approve | agent review | investigate]
 [Host / guest / reservation history features] --> models
 TRAINING LOOP
 [Resolved claims + agent decisions + appeals] -> [Labels (delayed)]
 -> [Train + calibrate] -> [Offline eval by cost] -> [Shadow / A/B] -> back
```

Follow-ups:
- What is the label: approved claim, amount paid, or claim confirmed fraudulent? How do you handle months of delay before resolution?
- Labels come from past agent decisions. How do you avoid learning the agents' biases?
- How do you detect reused or stock photos, and claims copied across listings?
- How would you estimate payout amounts from free-text item lists and receipts?
- Where do you set thresholds for auto-approval, agent review and investigation, given each error's cost?
- How do you handle heavy class imbalance for fraud?
- How do you keep the system fair to both the host filing and the guest being charged?
- What does the agent see, and how do you measure that the tool makes agents faster and more accurate?
- How would you test the system online without exposing guests to wrong charges?
- How would you monitor for new fraud patterns after launch?

## 9. "Family friendly" listing classifier

**Prompt.** "Design a classifier that tags listings as 'family friendly' (a filter for guests searching with children). How do you get labels (review keywords, post-trip ratings), which features do you use (listing metadata, review text embeddings, image features), which model do you pick, and how do you evaluate it?" Follow-up: "Some hosts mislabel their listings, and new listings have no reviews. How do you handle noisy host labels and cold start?" [Reported] [2] (April 2026 Senior MLE onsite; evaluated on precision, recall, F1 and calibration.)

**Maps to:** Airbnb Categories and attribute classification with human-in-the-loop review [70].

```
 SERVING
 [Listing metadata + amenities] [Photos] [Review text] -> [Feature encoders]
   -> [Classifier] -> [Calibrated prob] -> [Threshold] -> [Tag / filter in search]
 TRAINING LOOP
 [Weak labels: host tags, review keywords, ratings from families]
   + [Human-labeled gold set] -> [Label denoising] -> [Train]
 -> [Offline P / R / F1 / calibration] -> [Search A/B] -> [Complaint feedback] -> back
```

Follow-ups:
- Which label sources would you combine, and how do you estimate each one's noise rate?
- Hosts may tag listings "family friendly" to gain traffic. How do you treat host-provided labels?
- How do you classify a new listing with no reviews?
- Which errors are worse here, false positives (an unsafe listing shown to families) or false negatives (lost supply)?
- How do you pick the threshold? [Reported] [2] Would it differ by market, and what risk does that bring?
- Why does calibration matter if the output is only a binary tag?
- How would you evaluate with a human-labeled gold set, and how big must it be?
- How do you measure the tag's effect on bookings and post-stay satisfaction for families?
- How would you generalize to dozens of other attributes (pet friendly, accessible, workspace) without one model per tag?
- Why choose this model over a more complex multimodal one, given latency and cost? [Reported] [2]

## 10. Threshold choice and model choice under cost constraints

**Prompt.** "For a classifier in production at Airbnb, how would you pick the decision threshold, balancing precision, recall, latency and infrastructure cost, and how would you validate that threshold online? For the ranking or CTR model, would you use factorization machines or an embedding plus DNN model?" [Reported] [2] (April 2026 Senior MLE onsite. The candidate argued that the threshold choice needs an online A/B test, and the round also covered why not a more complex model.)

**Maps to:** Airbnb search ranking's evolution from GBDT to neural networks [44][46], and trust and classification thresholds generally.

```
 [Business costs: FP cost, FN cost, capacity of review queue]
            |
 [Candidate models: LR / GBDT / FM / Emb+DNN] -> [Offline PR / ROC / calibration]
            |                                         |
 [Latency + infra cost per 1k preds]        [Shortlist thresholds]
            \                                       /
             [Online A/B over thresholds / models] -> [Decision + monitoring]
```

Follow-ups:
- How do you turn business costs into a threshold rather than taking the default of 0.5?
- How would an A/B test choose between shortlisted thresholds?
- When do factorization machines win on sparse feature interactions and serving cost?
- What does an embedding plus DNN model buy you for long-term user behavior, and what does it cost?
- How do latency and infrastructure cost enter the model choice for a search-path model?
- Why does calibration matter when scores feed thresholds or a multi-objective blend?
- How do you re-tune the threshold after a retrain shifts the score distribution?
- What monitoring tells you that the threshold has drifted out of its intended operating point?

## 11. Real-time content personalization (homepage and categories)

**Prompt.** "Design a real-time content personalization system for Airbnb, for example a personalized homepage or a categories/experiences feed." [Reported] [2][22] (April 2026 Senior MLE report [2]; prep guide reports [22]. The ML design round is "usually a search or recommendation problem tied to Airbnb's real product problems", for example personalized homepage recommendations [6].)

**Maps to:** Airbnb real-time personalization with in-session embeddings [43]. It also maps to guest-journey sequence models (long-term and short-term history, decoupled from real-time ranking) [50] and the Relevance and Personalization team's low-latency, batch plus real-time pipelines [60].

```
 SERVING
 [App open: guest id, location, season] -> [Session event stream]
   -> [Candidate sources: categories, destinations, listings, experiences]
   -> [Sequence encoder (long + short term)] -> [Ranker] -> [Page layout / slots]
 [Batch features (daily)] + [Real-time features (seconds)] --> ranker
 TRAINING LOOP
 [Impressions + engagement + bookings] -> [Labels] -> [Train]
 -> [Offline eval] -> [A/B + holdout] -> [Freshness / drift monitors] -> back
```

Follow-ups:
- What is the objective for a homepage with no query: clicks, searches started, wishlists or bookings?
- How do you represent long-term booking history versus short-term views, and keep latency low?
- What freshness do in-session signals need, and what streaming infrastructure supports it?
- How do you rank heterogeneous items (destinations, categories, listings) on one page?
- How do you handle logged-out and brand-new guests?
- How do you avoid a filter bubble and still let guests explore?
- Which features are computed in batch and which in real time? Why?
- How do you keep training and serving features consistent?
- How would you A/B test a layout change when it shifts which searches guests run next?

## 12. Sensitive-information detection and redaction

**Prompt.** "Design an ML system that detects and redacts sensitive information, such as PII, contact details and payment info, in documents and messages on Airbnb. Cover labeling, model choice, the precision/recall threshold and serving." [Reported] [1] (Senior+ onsite, September 2026, "document redaction".)

**Maps to:** guest-host messaging ML (scam and off-platform payment detection) [66] and support document handling [62].

```
 SERVING
 [Message / uploaded doc (OCR if image)] -> [Rules + regex pre-pass]
   -> [NER / span tagger] -> [Context classifier (intent: off-platform pay?)]
   -> [Policy: redact | warn | block | send to review] -> [Delivered content]
 TRAINING LOOP
 [Human-annotated spans] + [Synthetic PII] + [Reviewer decisions]
 -> [Train] -> [Span-level P / R by entity type] -> [Shadow mode]
 -> [A/B on messaging metrics] -> [Missed-PII reports] -> back
```

Follow-ups:
- Which entity types matter (phone, email, address, payment, ID documents), and which are allowed at which booking stage?
- How do you get labeled spans without exposing annotators to real PII?
- When are regex rules enough, and when do you need a learned tagger or an LLM?
- How do you catch obfuscated contact details ("five five five, at gmail")?
- Which error is worse, a missed redaction or an over-redaction that breaks a legitimate message? How does that set the threshold?
- How do you handle many languages and scripts?
- What is the latency budget for real-time messaging versus batch document processing?
- How would you extend the system to detect scams and off-platform payment attempts? [Likely] [66]
- How do you measure success online (scam reports, off-platform leakage, message completion)?
- How do you audit the system and keep logs without storing the very data you redact?

## 13. Listing recommendation and similar listings

**Prompt.** "Design an Airbnb listing recommendation system" (1point3acres loop, alongside price prediction). [Reported] [12] The common variant is: "Design a similar-listings carousel on the listing page using listing embeddings learned from guest click sessions. How do you build negatives, correct for sampling bias, handle cold-start listings and the booking signal, and evaluate offline and online?" [Likely] [19][42]

**Maps to:** Airbnb listing embeddings for similar listings and real-time personalization (KDD 2018) [42][43]. The same embeddings feed host-side products such as host-calendar comparable listings [42].

```
 SERVING
 [Viewed listing + guest ctx] -> [Embedding lookup] -> [ANN neighbours]
   -> [Filters: dates, guests, market] -> [Re-ranker] -> [Similar-listings carousel]
 TRAINING LOOP
 [Click sessions + booked listing] -> [Training pairs from sessions]
 -> [Negative sampling] -> [Train embeddings]
 -> [Offline: rank of booked listing] -> [A/B on carousel CTR + bookings] -> back
```

Follow-ups:
- How do you define a session and its context window?
- How do you make the booked listing influence every listing in the session?
- Why add same-market negatives, and what goes wrong without them?
- How do you embed a brand-new listing with no sessions?
- How do you evaluate embeddings offline before shipping them?
- How do guest-side and host-side (comparable listings for pricing) objectives differ? [Likely] [42]
- How often do you retrain embeddings, and how do you keep downstream consumers consistent?
- How do you stop the carousel from cannibalizing search bookings, and how would you measure that?
- When would a sequence model replace skip-gram embeddings?

## 14. Relisting detection

**Prompt.** "Design Airbnb relisting detection": find hosts who take down a listing (often after bad reviews or enforcement) and relist the same property as new. The interviewer steered into geohashing. [Reported] [9][16]

**Maps to:** Airbnb trust and safety and listing integrity [26].

```
 SERVING
 [New / edited listing] -> [Candidate blocking: geohash, host graph, device]
   -> [Pairwise features: photo sim, text sim, address, amenities, host links]
   -> [Match model] -> [Score] -> [Action: auto-link | review queue | allow]
 TRAINING LOOP
 [Confirmed relists from investigations] + [Hard negatives: same building]
 -> [Train + calibrate] -> [Offline P / R at review capacity]
 -> [Shadow] -> [Reviewer feedback + appeals] -> back
```

Follow-ups:
- How do you generate candidate pairs cheaply among millions of listings (geohash cells, locality-sensitive hashing)?
- How do you separate a relisted home from a legitimate neighbor in the same building?
- Which photo and text similarity signals survive deliberate edits (crops, filters, rewording)?
- Where do labels come from, and how biased are investigation-driven labels?
- How do you set thresholds given a fixed human review capacity?
- What should happen to the reviews and history once a relist is confirmed?
- How do you measure success online without a clean ground truth?
- How does this connect to the fake-listing and review-fraud detectors?

## 15. ML across the booking funnel

**Prompt.** "Design the ML parts of an accommodation booking system for Airbnb, such as ranking and predicting booking likelihood across the guest-to-booking funnel." [Reported] [2] (April 2026 Senior MLE report. Prep sites list "design an accommodation booking system for Airbnb (availability, reservations, consistency) and explain where ML fits into it" [28][35].)

**Maps to:** the multi-stage funnel of search, listing view, contact or request, host acceptance, booking and stay, modeled by Journey Ranker [49].

```
 [Search] -> [Listing view] -> [Request / Instant Book] -> [Host accept]
     |            |                   |                         |
 [Rank model] [P(book | view)]  [P(host accepts)]       [P(cancel before stay)]
     \____________|___________________|_________________________/
                  [Shared features: guest, listing, calendar, price]
 TRAINING: [Funnel event logs] -> [Per-stage labels, delayed] -> [Multi-task train]
           -> [Offline per-stage + end-to-end eval] -> [A/B] -> back
```

Follow-ups:
- Where along the funnel does ML change a decision, and where would rules suffice?
- How do you combine per-stage probabilities into one ranking score?
- How do calendar accuracy and availability consistency affect your labels?
- How would you predict host acceptance, and use it in ranking without hurting guest experience? [Likely] [19]
- How would you predict guest or host cancellations, and what action would you take on the prediction? [Likely] [20][28]
- How do you handle the delay between booking and stay when labeling cancellations?
- Which latency budgets apply at each stage?
- How do you measure the end-to-end effect when one stage's model changes the next stage's inputs?

---

## 16. Two-sided marketplace A/B test for a ranking or pricing launch

**Prompt.** "You launch a new ranking or pricing model. How do you design the A/B test in Airbnb's two-sided marketplace, where treatment and control compete for the same inventory? Would you randomize by guest, by listing or by market, and when is each design unbiased?" [Likely] [33][34][53][54] (Experiment-design prompts appear in the Airbnb question index [10] and an Airbnb data-science prep guide [37], role unspecified. Interference is a core topic in Airbnb-coauthored research [53].)

**Maps to:** Johari, Li, Liskovich and Weintraub (Stanford and Airbnb): customer randomization is unbiased in demand-constrained markets, listing randomization in supply-constrained ones, and two-sided randomization (TSR) is proposed [53]. Holtz and Aral show that ignoring interference can inflate estimates by up to 100% [54]. Airbnb's own experimentation write-ups cover the platform side [59].

```
 [Hypothesis + primary metric + guardrails (guest AND host)]
      -> [Randomization unit: guest | listing | two-sided | market cluster]
      -> [Power calc with clustering] -> [Assignment + exposure logging]
      -> [Run: novelty window, SRM checks]
      -> [Analysis: variance reduction, interference diagnostics, segments]
      -> [Launch decision] -> [Holdout / long-term readout]
```

Follow-ups:
- How does interference bias a guest-randomized test when inventory is limited?
- When would you randomize listings instead, and what bias does that leave?
- How does two-sided randomization work, and what bias-variance trade-off remains?
- When is market-level or cluster randomization worth the loss in power?
- How do you compute power when randomizing clustered units?
- Which guest metrics and which host metrics do you track as primary and guardrail?
- A new search filter raised bookings but lowered host response rates. Would you launch? [Likely] [34]
- How would you detect novelty effects in a ranking test?
- How would you measure cannibalization of a new category, with holdout regions or synthetic controls? [Likely] [34]
- Which pitfalls (peeking, dilution, marketplace-level bias) should the platform guard against automatically? [Likely] [57]
- Walk through the experiment from hypothesis to launch call. [Likely] [38]

## 17. Pricing experiments on expiring inventory

**Prompt.** "How would you run a pricing (or fee) experiment on Airbnb, where nights are expiring inventory and bookings for a given night arrive over many lead days? How do you randomize, correct for lead-day bias, and avoid the standard estimator pointing you in the wrong price direction?" [Likely] [33][55][56][57]

**Maps to:** Airbnb's CIKM 2023 paper on lead-day bias in pricing experiments, where bias correction limits rollout to nights far enough from the lead day [56]. It also maps to price experimentation and interference, where canonical estimators can move prices the wrong way in Airbnb-like markets [55].

```
 [Price / fee change] -> [Unit: listing-night | listing | market]
   -> [Treatment window by stay date + lead day]
   -> [Booking curve per night (bookings arrive over lead days)]
   -> [Estimator + bias corrections]
   -> [Short-term vs long-term readout] -> [Decision + guardrails (host earnings)]
```

Follow-ups:
- Why can the short-term result move opposite to the long-term effect?
- What is lead-day bias, and how does restricting treatment by lead day correct it?
- How does interference between treated and control nights or listings bias the estimate?
- Should you randomize listings, nights or markets for a price change?
- Which host-side metrics (earnings, occupancy, churn) guard the launch?
- How long must the test run before the booking curves for treated nights are complete?
- How would you handle host overrides of suggested prices during the test?
- How do you communicate a pricing test to hosts, and does that communication bias it?

## 18. Fraud and trust: fake bookings, fake listings, account takeover and review fraud

**Prompt.** "Design a real-time fraud detection system for fake bookings or fake listings on Airbnb. Cover labels from chargebacks and investigations, delayed feedback, class imbalance, latency, and how you trade off the friction a false positive causes against fraud loss." [Likely] [25][26][27][30][31] Variants: account takeover and payment fraud with layered defenses [22][27]; review fraud using new accounts, shared IP or device between reviewer and host, and copied text [26].

**Maps to:** Airbnb trust and safety. Postings list payments, fraud prevention and anti-discrimination work [69]. A guide says trust prompts make up a large share of Airbnb MLE design rounds [31].

```
 SERVING
 [Event: signup | login | listing | booking | payment | review]
   -> [Identity + device + graph features] -> [Rules layer] -> [Risk model]
   -> [Action: allow | friction (verify) | hold | block | review]
 [Real-time feature store] [Entity graph: user-device-IP-card]
 TRAINING LOOP
 [Chargebacks + investigations + appeals (delayed)] -> [Labels]
 -> [Train with imbalance handling] -> [Offline cost-weighted eval]
 -> [Holdout / champion-challenger] -> [Drift + attack alerts] -> back
```

Follow-ups:
- Labels arrive weeks later (chargebacks). How do you train and evaluate on fresh data?
- The model's own blocks hide outcomes. How do you avoid that selection bias (holdouts, exploration)?
- How do you trade off fraud loss against the friction a false positive puts on a real guest?
- What latency budget applies at booking and payment time?
- Which graph signals (shared devices, cards, IPs) help most, and how do you serve them in real time?
- How do you respond in the first 48 hours after a new fraud pattern appears?
- How would you detect review fraud, and what does it mean for ranking features built on reviews?
- How do you audit for proxy discrimination in risk decisions?
- Is a random holdout from the fraud model ethical or practical? What would you propose instead?

## 19. Multi-task ranking that balances guests and hosts

**Prompt.** "Design a multi-task search ranker that uses intermediate guest actions (clicks, wishlists, contacting the host) as milestones toward booking, and balances guest relevance with host success and marketplace health. Which objective and which guardrail metrics would you use?" [Likely] [25][49]

**Maps to:** Journey Ranker, Airbnb's multi-task deep learning model over guest-journey milestones [49].

```
 [Shared features] -> [Shared towers] -> [Heads: click | long view | wishlist |
                                          contact | book | host accept | cancel]
                                      -> [Score combiner (weights)] -> [Ranking]
 TRAINING: [Journey logs across sessions/days] -> [Per-head labels] -> [Joint train]
           -> [Per-head offline eval] -> [Weight tuning] -> [A/B: guest + host]
```

Follow-ups:
- Why model the whole journey rather than a single search?
- How do you weight the heads, and how do you re-tune after one head is retrained?
- How do host-side outcomes (acceptance, cancellations, earnings) enter guest-facing ranking?
- Which guardrails protect new hosts from losing exposure?
- How do you handle conflicting gradients between heads?
- How do you evaluate a multi-task model offline when the heads disagree?
- How do you attribute an online win to a specific head?
- Searches span many days before one booking. How does that change labels and logging?

## 20. Diversity and map ranking

**Prompt.** "Search results are too homogeneous. How would you change the ranker to promote diversity, given that a listing's booking probability depends on the other listings shown? And how would you rank the map view differently from the list view, given that position-bias assumptions built for lists break down on maps?" [Likely] [47][48]

**Maps to:** "Learning To Rank Diversely at Airbnb" [48] and Airbnb's map-ranking material [47].

```
 [Ranked candidates] -> [Context-aware re-ranker]
   -> [Diversity objective: price, type, location] -> [List slots]
   -> [Map pin selection] -> [Map view]
 TRAINING: [Impression logs with full result set + map viewport]
           -> [Context-dependent labels] -> [Train] -> [Offline diversity + NDCG]
           -> [A/B: list and map separately]
```

Follow-ups:
- Why is a listing's booking probability not independent of the other results?
- How would you measure diversity, and how would you trade it off against relevance in the loss?
- How does user attention on a map differ from attention on a list?
- How do you discount position when there is no top-to-bottom order?
- Which pins do you show when hundreds of listings fit the viewport?
- How do you log impressions on a map?
- How would you A/B test list and map ranking without one confounding the other?
- How would a re-ranker that sees the whole result set fit within the latency budget?

## 21. Cold start for new hosts, listings and guests

**Prompt.** "Design an ML system for new hosts and new guests in Airbnb ranking or recommendations, with cold start as the main focus. Should features be computed in real time or in batch?" [Likely] [23][24]

**Maps to:** cold-start handling in Airbnb search ranking [45] and listing embeddings [42].

```
 [New listing / new guest] -> [Content features: attributes, photos, text, geo]
   -> [Prior from similar entities (embeddings, market averages)]
   -> [Exploration policy] -> [Ranker]
 [Batch features (daily)] | [Real-time features (session)] -> [Feature store]
 TRAINING: [Early engagement logs] -> [Warm-up model] -> [Eval on new-entity slice]
           -> [A/B with new-host guardrails] -> back
```

Follow-ups:
- How do you estimate engagement for a listing with no bookings or reviews?
- How much exposure do new listings get, and who pays for that exploration?
- When does a contextual bandit beat a supervised model plus an A/B test here?
- How do you personalize for a brand-new guest from session signals alone?
- Which features justify real-time computation, and which are fine in batch?
- How do you evaluate cold-start performance offline on a held-out slice of new entities?
- Which host-side metrics (time to first booking, new-host churn) would you guard?
- How do you stop exploration traffic from contaminating the training labels of the main ranker?

## 22. Natural-language query understanding and AI search

**Prompt.** "Design query understanding for Airbnb search: autocomplete, query tagging/NER, query expansion, intent modeling and LLM-powered natural-language search" (for example, "a quiet cabin near Lake Tahoe for 6 with a hot tub"). [Likely] [61]

**Maps to:** the Relevance and Personalization Query Intelligence team [61] and embedding-based retrieval [39].

```
 [Free-text query] -> [Autocomplete] -> [Parser: NER / LLM]
   -> [Hard filters | soft preferences | semantic query] -> [Retrieval + ranking]
   -> [Results + explanation]
 TRAINING: [Query logs + reformulations + bookings] -> [Labeled parses]
           -> [Train / distill] -> [Offline parse accuracy] -> [A/B] -> back
```

Follow-ups:
- How do you decide which parsed attributes become hard filters and which become soft preferences?
- How do you evaluate parsing errors that silently drop a hard constraint?
- Where would you use an LLM, and how would you bound its latency and cost on the search path?
- How do you handle ambiguous locations ("Portland")?
- How do you measure whether natural-language search increases bookings, rather than just novelty?
- How do you cache or distill the parser for common queries?
- How do you build labeled parses at scale, and when would you use LLM-generated labels?
- How do autocomplete suggestions shape the queries guests type, and how does that bias your training logs?

## 23. Monitoring, training-serving skew and the feature store

**Prompt.** "How do you minimize training-serving skew in a search ranking stack that has both batch and real-time features? Design real-time monitoring for production models: feature and prediction drift, label delay, alerting and rollback." [Likely] [36][60]

**Maps to:** the Relevance and Personalization posting, which lists skew minimization, A/B tests and feature and model selection [60].

```
 [Feature definitions (one source)] -> [Batch backfill] + [Streaming compute]
   -> [Online store] -> [Serving] -> [Logged features at serve time]
   -> [Training set = logged or point-in-time joins]
 [Monitors: feature drift | score drift | skew diff | label-delayed metrics]
   -> [Alerts] -> [Rollback / fallback model]
```

Follow-ups:
- Where does skew usually come from (different code paths, time travel, defaults for missing values)?
- When do you log features at serving time instead of recomputing them for training?
- How do you build point-in-time-correct training data?
- How do you monitor a model whose labels arrive days or weeks late?
- What triggers an automatic rollback, and what is the fallback?
- How often do you retrain, and do you A/B test every refresh?
- A predicted booking rate drops suddenly in one market. What do you check first? [Likely] [22]
- How do you tag experiment traffic so that one arm's logs do not contaminate the next model's training data?

## 24. Distributed recommendation system at Airbnb scale

**Prompt.** "Design a distributed recommendation system at Airbnb scale. Discuss data quality, the feature store, model serving, A/B testing, and the trade-offs between model complexity, interpretability and latency." [Likely] [32] (An ML platform loop paired an ML design round with a general system design round [7].)

**Maps to:** Airbnb's batch and real-time ranking infrastructure [60], and the end-to-end search stack from pipelines to experimentation [60].

```
 [Event ingestion] -> [Data quality checks] -> [Feature store: offline + online]
   -> [Training cluster] -> [Model registry] -> [Serving fleet (sharded, cached)]
   -> [Request path: candidates -> ranker -> response]
 [Experiment assignment] -> [Metrics pipeline] -> [Dashboards + alerts]
```

Follow-ups:
- Where do the latency and throughput bottlenecks sit, and how would you shard and cache?
- How do you catch upstream data-quality breaks before they reach training?
- How do you version models and features together?
- How do you roll out a model safely (shadow, canary, A/B)?
- How would you let other teams (Experiences, Services, long-term stays) reuse the stack?
- How do you serve the same model to several surfaces (search, email, homepage) with different latency budgets?
- What would you precompute in batch versus score on request, and what does each cost?

## Level calibration: senior probes

The bank above already includes many senior probes. Use these harder, cross-cutting ones to check that answers show end-to-end ownership and marketplace and experimentation judgment. [Likely] [2][53][60]

1. You own search ranking end to end. Which three metrics would you put on your team's quarterly scorecard, and which metric would you refuse to optimize directly?
2. A ranking change raises guest bookings 1% but shifts bookings from new hosts to top hosts. How do you decide, and who has to agree?
3. Your A/B test is guest-randomized and positive, but a market-level switchback shows no effect. Which result do you trust, and what do you ship?
4. Three teams (ranking, pricing, homepage) all claim the same booking lift. How would you design experiments and holdouts so that the gains add up?
5. A model needs two more quarters of infrastructure work to beat the GBDT baseline. How do you build the case to fund it, or to stop it?
6. Your pricing model's suggestions are adopted by 60% of hosts in a market. How do you tell whether it now sets the market price, and what would you change?
7. Labels for your model come from decisions made by an earlier version of the same model. How do you design logging and exploration so the next model can still learn?
8. A launch is positive on the 2-week readout. What evidence would you require before trusting the long-term effect, and how long would you keep a holdout?
9. Leadership wants an LLM in the search path next quarter. How would you scope the first launch, set the latency and cost budget, and define the kill criteria?
10. A model you shipped a year ago is now hurting a segment nobody monitored. How do you find it, and what process change prevents the next one?

---

## References

Sources were read as search-result snippets or as previously fetched pages, as noted in the source chapters. 1point3acres threads are login-gated. Confirm every link before relying on it.

1. "Airbnb Senior+ Machine Learning Engineer Interview Experience — Search Ranking and Team Matching", PracHub, 2026. https://prachub.com/interview-experiences/airbnb-seniorplus-machine-learning-engineer-interview-experience-search-ranking-and-team-matching
2. "Airbnb Senior Machine Learning Engineer Interview Experience — Four ML System Design Rounds, Rejected on Calibration", PracHub, 2026. https://prachub.com/interview-experiences/airbnb-senior-machine-learning-engineer-interview-experience-four-ml-system-design-rounds-rejected-on-calibration
3. "Airbnb Machine Learning Fulltime Onsite Interview Experience and Insights", 1point3acres, 2025. https://www.1point3acres.com/interview/thread/1162455
4. "Airbnb Fulltime Machine Learning Engineering Onsite Interview Overview", 1point3acres, 2025. https://www.1point3acres.com/interview/thread/1114733
5. Airbnb MLE onsite post ("ML round 1: price prediction system"; the exact thread match is uncertain), 1point3acres, 2022. https://www.1point3acres.com/bbs/thread-934565-1-1.html
6. "Airbnb senior MLE interview prep", Blind, 2024. https://www.teamblind.com/post/airbnb-senior-mle-interview-prep-n1mdsd3b
7. "Interviewing for Airbnb ML platform role", Blind, 2025. https://www.teamblind.com/post/interviewing-for-airbnb-ml-platform-role-8in7czso
8. "ML experience round at Airbnb", Blind, year not given. https://www.teamblind.com/post/ml-experience-round-at-airbnb-sorv1g3x
9. "Airbnb Machine Learning Engineer interview process" (TrueInterview-derived summary of 3 candidate reports), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
10. "Airbnb" question index with report dates, kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
11. 1point3acres post #40 (English mirror: LTV and host-pricing design rounds), FlowSight/coding, GitHub, year not given. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
12. 1point3acres post #79 (English mirror: listing recommendation and price prediction rounds), FlowSight/coding, GitHub, year not given. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
13. "Rank Home Search Results Without a Text Query", TrueInterview, 2026. https://trueinterview.io/questions/rank-home-search-results-without-a-text-query
14. "Design User Embedding Semantic Search", TrueInterview, 2026. https://trueinterview.io/questions/design-user-embedding-semantic-search
15. "Listing Lifetime Value — Estimation", TrueInterview, 2026. https://trueinterview.io/questions/listing-lifetime-value-ml-design
16. "Design Airbnb Relisting Detection", TrueInterview, 2026. https://trueinterview.io/questions/design-airbnb-relisting-detection
17. "Generate Customer-Support Instructions from Company Policies", TrueInterview, 2026. https://trueinterview.io/questions/generate-customer-support-instructions-from-company-policies
18. "Design A Personalized Search Ranking System", TrueInterview, 2026. https://trueinterview.io/questions/d6c8316c-c7a9-4074-bec9-f19e4252f21d
19. "Airbnb ML System Design Interview Questions", PracHub, 2025-2026. https://prachub.com/companies/airbnb/categories/ml-system-design
20. "Airbnb Machine Learning Engineer Interview Questions", PracHub, 2025-2026. https://prachub.com/companies/airbnb/positions/machine-learning-engineer
21. "Airbnb Interview Questions", PracHub, year not given. https://prachub.com/companies/airbnb
22. "Airbnb Machine Learning Engineer Interview Guide", Interview Query, 2025. https://interviewquery.com/interview-guides/airbnb-machine-learning-engineer
23. "Airbnb AI Engineer Interview Guide: Real Questions & Expectations", Interview Query, 2026. https://www.interviewquery.com/prep-guides/airbnb-ai-engineer
24. "Airbnb Machine Learning Engineer interview guide", Dataford, 2025. https://dataford.io/interview-guides/airbnb/machine-learning-engineer
25. "Airbnb Machine Learning Engineer", CleverPrep, 2025. https://www.cleverprep.com/companies/airbnb/machine-learning-engineer
26. "Airbnb Interview Guide 2026: Search Systems, Trust and Safety, and Full-Stack Engineering", techinterview.org, 2026. https://techinterview.org/airbnb-interview/
27. "Airbnb Interview Guide 2026: Search Systems, Trust and Safety, and Full-Stack Engineering" (post version), techinterview.org, 2026. https://www.techinterview.org/post/3233460279/airbnb-interview-guide-2026-search-systems-trust-and-safety-and-full-stack-engineering/
28. "Airbnb interview process and top questions", Educative, 2025. https://educative.io/blog/airbnb-interview-process-and-questions
29. "Get a Job at Airbnb: Interview Process and Top Questions", Exponent, 2025. https://tryexponent.com/blog/airbnb-interview-process
30. "Airbnb Machine Learning Engineer Applications: Resume Keywords and Interview Prep", Jobrise, 2026. https://jobrise.io/en/blog/airbnb-machine-learning-engineer-resume-interview-2026/
31. "Airbnb Machine Learning Engineer" guide, DataInterview, year not given. https://www.datainterview.com/blog/airbnb-machine-learning-engineer-interview
32. "Design a Recommendation Service", Codemia, 2025. https://codemia.io/practice/design-a-recommendation-service-12
33. "Design an A/B test for a pricing model", Codemia, year not given. https://codemia.io/practice/design-an-a-b-test-for-a-pricing-model-48
34. "Airbnb Data Scientist guide", Codemia, 2025. https://codemia.io/guides/airbnb-data-scientist
35. "Airbnb Machine Learning Engineer interview questions" (page 5), Prepfully, 2025. https://prepfully.com/interview-questions/airbnb/machine-learning-engineer?page=5
36. "Airbnb Machine Learning Engineer interview questions" (page 8), Prepfully, year not given. https://prepfully.com/interview-questions/airbnb/machine-learning-engineer?page=8
37. "Airbnb Data Scientist Interview Prep Guide", PracHub, year not given. https://prachub.com/interview-prep/9_Zq5otS4cjz
38. "Walk through an experiment from design to decision", PracHub, year not given. https://prachub.com/interview-questions/walk-through-an-experiment-from-design-to-decision
39. "Embedding-Based Retrieval for Airbnb Search", Airbnb Tech Blog (airbnb.tech), 2025. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/
40. "Applying Embedding-Based Retrieval to Airbnb Search", arXiv, 2026. https://arxiv.org/html/2601.06873v1
41. "Airbnb embedding retrieval system" (case study), Machine Learning at Scale, 2024. https://machinelearningatscale.com/blog/airbnb-embedding-retrieval-system/
42. "Learning and Applying Airbnb Listing Embeddings", airbnb.tech (PDF), 2024. https://airbnb.tech/wp-content/uploads/sites/19/2024/12/Learning-and-Applying-Airbnb-Listing-Embeddings-.pdf
43. Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD, 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
44. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD, 2019. https://arxiv.org/abs/1810.09591
45. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv / KDD, 2020. https://arxiv.org/abs/2002.05515
46. "The story of how Airbnb implemented neural networks to improve their search feature", Analytics India Magazine, year not given. https://analyticsindiamag.com/the-story-of-how-airbnb-implemented-neural-networks-to-improve-their-search-feature/
47. "Distilled: Rental Search Ranking", aman.ai, year not given. https://aman.ai/sysdes/airbnb/
48. Abdool, Haldar et al., "Learning To Rank Diversely At Airbnb", arXiv, 2022 (CIKM 2023). https://arxiv.org/abs/2210.07774
49. Tan et al., "Optimizing Airbnb Search Journey with Multi-task Learning", arXiv / KDD, 2023. https://arxiv.org/abs/2305.18431
50. "Personalizing Airbnb search by learning from the guest journey" (summary), thenote.app, year not given. https://thenote.app/post/en/personalizing-airbnb-search-by-learning-from-the-guest-journey-c4zzcdn3wz
51. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing", KDD, 2018. https://dl.acm.org/doi/10.1145/3219819.3219830
52. "Academic publications: Airbnb Tech 2025 year in review" (includes BiListing and pairwise learning-to-rank), airbnb.tech, 2025. https://airbnb.tech/infrastructure/academic-publications-airbnb-tech-2025-year-in-review/
53. Johari, Li, Liskovich and Weintraub, "Experimental Design in Two-Sided Platforms: An Analysis of Bias", arXiv, 2020. https://arxiv.org/pdf/2002.05670v3
54. Holtz and Aral, "Limiting Bias from Test-Control Interference in Online Marketplace Experiments", arXiv, 2020. https://arxiv.org/pdf/2004.12162
55. "Price Experimentation and Interference", arXiv, 2023. https://arxiv.org/pdf/2310.17165
56. Airbnb CIKM paper on lead-day bias in pricing experiments for expiring goods, airbnb.tech (PDF), 2023. https://airbnb.tech/wp-content/uploads/sites/19/2023/12/CIKM.pdf
57. "Experimentation" notes (pitfalls; lead-day bias in pricing tests), Notion, year not given. https://citrine-straw-ba5.notion.site/Experimentation-0990e9ed61db4d2299bea641c6ab7ffb
58. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Tech Blog (Medium), 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
59. "Experiments at Airbnb", Airbnb Engineering and Data Science (Medium), 2014. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
60. "Senior Staff Machine Learning Engineer, Relevance and Personalization", Airbnb Careers, 2025-2026. https://careers.airbnb.com/positions/7381973/
61. "Senior Machine Learning Engineer, Relevance and Personalization (Query Intelligence)", Airbnb Careers, 2026. https://careers.airbnb.com/positions/8065789/
62. "Senior Machine Learning Engineer, Customer Support Engineering", Airbnb Careers, 2026. https://careers.airbnb.com/positions/8014904/
63. "Staff Machine Learning Engineer, Support Products, Airbnb", Working Nomads, 2026. https://www.workingnomads.com/jobs/staff-machine-learning-engineer-support-products-airbnb-460586
64. "Machine Learning Engineer, Customer Support Engineering", Accel jobs board, 2026. https://jobs.accel.com/companies/hoteltonight-2/jobs/75694130-machine-learning-engineer-customer-support-engineering
65. "Senior Staff Machine Learning Engineer, Community Support Engineering (GenAI)", Built In, 2025. https://builtin.com/job/senior-staff-machine-learning-engineer-community-support-engineering/6560520
66. "Senior Staff Machine Learning Engineer, Communication & Connectivity", Built In, 2025. https://builtin.com/job/senior-staff-machine-learning-engineer-communication-connectivity/6495497
67. Paper on integrating LLMs into Airbnb customer support, arXiv, 2025. https://arxiv.org/abs/2510.10331
68. "LLM integration for customer support automation and enhancement" (Airbnb case study), ZenML LLMOps database, 2022. https://zenml.io/llmops-database/llm-integration-for-customer-support-automation-and-enhancement
69. "Machine Learning Engineering Intern (PhD), Airbnb", Working Nomads, 2026. https://www.workingnomads.com/jobs/machine-learning-engineering-intern-phd-airbnb
70. "Building Airbnb Categories with ML and Human-in-the-Loop", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb
