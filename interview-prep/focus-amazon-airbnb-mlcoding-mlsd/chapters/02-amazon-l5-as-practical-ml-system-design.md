# 02. Amazon L5 Applied Scientist: Practical ML System Design

> Scope: the ML system design round (also called "Science Application" or "Sys Design (Team-specific)") in the Amazon L5 Applied Scientist (Applied Scientist II) loop, 2024-2026. This chapter contains questions only. It gives no answers or solutions.
>
> How it relates to the older chapters: this chapter merges and dedupes four new search passes (October 2026) with the prompts in `amazon-l5-applied-scientist/chapters/06-ml-system-design.md`, `12-ml-system-design-reference-diagrams.md` and `13-tagging-and-document-understanding-system-design.md`. Those chapters keep the long tail: SCOT buying, payments fallback, Alexa accents, enterprise RAG, document understanding and the full attribute-extraction deep dive. Each prompt here gets a skeleton diagram listing the boxes a strong answer must cover. The boxes are deliberately left empty: they name components, not choices.
>
> Source caveat: almost every source is an anonymous self-report on Glassdoor, Blind, Reddit or LeetCode, or a prep-vendor page. Many were read only as search-result extracts. Some Glassdoor review IDs were matched to the right report only approximately (noted in the references). Read each [Reported] tag as "a candidate said this was asked", not as verified fact.

Tag legend: **[Reported]** means a cited source says the prompt, or a close paraphrase, was asked in an Amazon science interview. **[Likely]** means a grounded inference from Amazon job postings, prep guides or the hiring team's products, not seen verbatim in a candidate report. Follow-up bullets are untagged unless a source reports them. They are the probes interviewers typically pull on.

---

## How this round works

- **Where it sits in the loop.** Recent AS II loops are a phone screen plus 4-5 onsite interviews of about 55-60 minutes each. Seattle 2026 reports give the loop as "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [2][3][4]. A Luxembourg loop was a tech talk plus Coding, Science Depth, Science Breadth and Science Application [11]. An August 2024 AS2 loop had 2 phone screens and 5 onsites, one of them a business-problem ML design round [1]. Some recent loops had two practical ML design rounds "related to day to day ml implementations" rather than LeetCode [6][26].
- **What kind of design.** This is ML system design, not software architecture. A Berlin candidate found it "higher-level than classic software design", focused on recognizing and applying standard or recent ML approaches [12]. The Alexa round explicitly asked for "ML system design, not software design": datasets, task framing, model choice with justification, and evaluation metrics [1]. Interviewers still ask about latency, scaling, serving, drift monitoring and rollout [22][40].
- **Where the prompt comes from.** Usually from the hiring team's own work. A Blind AS2 thread describes the round as taking the team's business problem, framing it as a science problem, and defining data, model, metrics and the learning process [16]. In 2025-2026, GenAI and agentic prompts became common: agentic recommenders, a Copilot-style agent, LLM training and decoding [2][3][9][24][25].
- **Format and tools.** Expect a verbal walkthrough on a shared virtual whiteboard or document, with no code to run. The sources do not name a specific tool. The interviewer interrupts with follow-ups after the first end-to-end pass [10]. Plan for about 40 minutes of design and 10-15 minutes of Leadership Principle questions. Amazon interviewers each own one or two LPs, often Customer Obsession, Dive Deep, Invent and Simplify, and Think Big.
- **What the L5 bar is.** You must own the whole problem end to end: scope a vague business goal, choose data and labels, justify the model against alternatives, pair offline metrics with an online (Weblab) metric, and plan deployment, monitoring and iteration [16][29]. Depth probes test why you chose each component and whether you know the alternatives. One candidate was down-leveled partly for lacking production experience and for narrowing the problem too much [13]. Breadth questions such as bias-variance and bagging versus boosting can be woven into the design round [2][4].
- **Out of scope here.** ML coding (for example ROC AUC from scratch, or a CNN layer forward pass [2][15]) is covered in the ML coding chapter of this focus guide.

---

## Top 15 to practice first

Reported prompts come first, ordered by how many independent reports mention them.

1. Design an agentic recommendation system for Amazon shopping, and evaluate its quality and safety. [Reported] [2][3] (see Q1)
2. Team-specific design: turn the hiring team's business problem into a science problem, end to end. [Reported] [3][4][12][16] (see Q2)
3. Design an end-to-end pipeline that translates multimodal catalog objects into another language. [Reported] [5][6][7][8] (see Q3)
4. Design Alexa from scratch as an ML system. [Reported] [1] (see Q4)
5. Summarize product reviews into key concepts for the product page. [Reported] [3][11] (see Q5)
6. Design and evaluate a product recommendation model or engine for Amazon. [Reported] [10][20][21] (see Q6)
7. Frame an open-ended ML product problem: scope, data, domain adaptation, distillation, metrics. [Reported] [13] (see Q7)
8. Design a spam (or fake-review) detection system. [Reported] [17] (see Q8)
9. Classify short noisy text: the language of reviews, or tweets into categories. [Reported] [10][18][19][20] (see Q9)
10. Design a GitHub Copilot-style coding agent. [Reported] [9] (see Q10)
11. Design a system for global request detection and labeling. [Reported] [22] (see Q11)
12. Design product search ranking, including LLM-powered search. [Likely; LLM-search variant Reported] [14][24][35] (see Q12)
13. Design a Rufus-style LLM shopping assistant. [Likely; DPO and decoding probes Reported] [24][25] (see Q13)
14. Design CTR/CVR prediction and placement for Sponsored Products ads. [Likely] [42][43][46] (see Q14)
15. A production model is degrading: diagnose, retrain and safely redeploy it. [Likely] [26][40] (see Q22)

---

## 1. Design an agentic recommendation system

**Prompt.** "Design an agentic recommendation system (for example, for Amazon shopping). How would you structure the system, what models or components would you use, and how would you evaluate its quality and safety?" [Reported] [2][3] (Seattle, May-June 2026; the later rounds of that loop focused on agentic product thinking.)

**Maps to:** Amazon Stores personalization, Rufus, Alexa+ shopping.

```
 SERVING
 [Shopper msg + context] -> [Intent / planner LLM] -> [Tool router]
                                                       |-> [Catalog search tool]
                                                       |-> [Recs retrieval tool]
                                                       |-> [Cart / order tools]
                            [Ranker / re-ranker] <-----+
                            [Grounding + safety guard] -> [Response + product cards]
 TRAINING / EVAL LOOP
 [Session + tool-call logs] -> [Human + LLM-judge labels] -> [SFT / pref tuning]
 [Offline eval suite: task, safety, red-team] -> [Weblab A/B] -> [Monitoring] -> back
```

Follow-ups:
- What does "agentic" add over a standard retrieval-and-ranking recommender? What can the agent do that a ranked carousel cannot?
- Which decisions belong to the LLM planner, and which stay with classical retrieval and ranking models?
- How do you define tools (search, filters, compare, add to cart), and how do you stop the agent calling them in loops?
- How do you ground every recommended product in the live catalog so the agent never invents an ASIN, price or spec?
- How do you build an offline evaluation set for multi-turn shopping tasks, and what is "task success"?
- When would you trust an LLM-as-judge for quality, and how do you calibrate it against human raters?
- Which safety failures matter here (unsafe products, age-restricted items, biased steering, prompt injection from listing text), and how do you red-team them?
- Which online metrics decide the launch, which are guardrails, and how do you randomize a conversational A/B test?
- What is your latency budget per turn, and where would you use a small model or a cache instead of the large LLM?
- How do you personalize using purchase history without breaking privacy expectations?
- How do sponsored products enter the agent's answers without hurting trust?
- How do agent recommendations feed back into training data without amplifying the agent's own biases?

## 2. Team-specific ML system design: from business problem to science problem

**Prompt.** "Design the ML system behind the hiring team's product. Take their business problem (for example, improving a metric on their product), turn it into a science problem, then define the data, labels, model, metrics, and the learning or iteration loop." [Reported] [3][4][12][16] (AS II loops in Seattle 2026 and Berlin; the Alexa International Tech guide [23] and a senior AS's design guide [29] describe the same framing. At Seattle the round was followed by modeling-choice probes such as bagging versus boosting [4].)

**Maps to:** whichever team you interview for: Ads, Search, Alexa, Prime Video, SCOT, Catalog and others.

```
 [Business goal + constraints] -> [Science problem: decision, target, unit]
 SERVING
 [Request] -> [Features (batch + real-time)] -> [Model] -> [Policy / threshold]
                                                              -> [Action]
                                                                       |
 [Logged outcomes] <---------------------------------------------------+
 TRAINING LOOP
 [Data sources] -> [Labels / proxy label] -> [Train + offline eval] -> [Registry]
 [Shadow -> Weblab A/B] -> [Launch decision] -> [Monitoring + retrain trigger]
```

Follow-ups:
- What exactly is the decision the model makes, and who acts on it?
- What is your proxy label, and how does it diverge from the business outcome you actually care about?
- What is the simplest baseline (heuristic or rules) you would ship first, and what would make you move beyond it?
- Which standard and which recent ML approaches apply to this problem, and why do you prefer one?
- Why this model over the alternatives? Would you use bagging or boosting on the tabular features, and how does that tie to bias and variance? [Reported] [2][4]
- Which offline metric correlates with the online business metric, and how do you check that it does?
- How would you design the Weblab experiment: randomization unit, primary metric, guardrails, duration and power?
- What is the iteration loop after launch? How do new labels arrive, and how often do you retrain?
- What would v1 look like in one quarter, and what does v2 add?
- What is the single riskiest assumption, and how would you de-risk it in two weeks?
- What would make you kill the project after launch?

## 3. Design a multimodal translation pipeline for catalog objects

**Prompt.** "Model a pipeline end to end for translating one set of multimodal objects (for example, product listings with text, images and structured attributes) to another language. Cover data, model choice, training, quality evaluation and serving." [Reported] [5][6][7][8] (Applied Scientist II, Hyderabad, 2026; asked as the system design part of a data-science round.)

**Maps to:** Amazon catalog localization for new marketplaces, International Stores, the Machine Translation science team.

```
 SERVING / BATCH
 [Listing: title, bullets, attrs, images] -> [Segment + protect: brand, units, attrs]
     -> [Image encoder] --+
     -> [Text encoder] ---+--> [MT model / multimodal LLM] -> [Post-edit rules]
     -> [Quality estimation (QE) score] -> [Publish | Human translator queue]
 TRAINING LOOP
 [Parallel corpora + in-domain pairs] -> [Fine-tune] -> [Offline eval: auto + human]
 [Translator edits + customer signals] -> [Retrain] -> [Per-locale monitoring]
```

Follow-ups:
- What is the unit of translation: each field separately, or the whole multimodal object together?
- When does the image actually help disambiguate the text, and how would you measure whether it does?
- Where would you get in-domain parallel data for a new marketplace that has very little of it?
- How do you keep brand names, model numbers, sizes and units from being translated or mistranslated?
- Would you use a dedicated MT model, a multimodal LLM, or both? How do cost and latency at catalog scale drive that choice?
- How do you evaluate quality without enough human references (BLEU or chrF versus COMET versus human evaluation)? [Likely follow-up to [7]]
- How would you train a quality-estimation model to decide which items can publish automatically and which go to human translators?
- How do translator post-edits feed back into training?
- Which online metric shows the translation helped customers (conversion, returns, search recall in the target language)?
- How would you handle attributes that need conversion rather than translation (sizes, measurements, currency)?
- What changes for a right-to-left or low-resource language?

## 4. Design Alexa from scratch as an ML system

**Prompt.** "Design Alexa from scratch. This is ML system design, not software design. Which datasets would you need, how would you frame the ML tasks (wake word, ASR, NLU intent and slot, response selection), which models would you choose and why, and how would you evaluate them offline and online?" [Reported] [1] (Applied Scientist 2, August 2024, business-problem round.)

**Maps to:** Alexa AI: wake word, ASR, NLU, dialog, Alexa+.

```
 SERVING (on-device -> cloud)
 [Audio] -> [Wake word (device)] -> [ASR] -> [NLU: domain, intent, slots]
         -> [Skill / API router] -> [Dialog + response gen] -> [TTS] -> [User]
 TRAINING LOOP
 [Annotated utterances + transcripts] -> [Per-stage models] -> [Per-stage offline eval]
 [Implicit feedback: rephrase, barge-in, abandon] -> [Defect labels] -> [Retrain]
 [Weblab A/B on end-to-end task success] -> [Monitoring per locale / device]
```

Follow-ups:
- Which datasets do you need for each stage, and how do you collect and annotate them while respecting privacy?
- Would you build a cascaded pipeline or an end-to-end speech-to-intent (or speech LLM) model? What are the trade-offs?
- What runs on the device and what runs in the cloud, and why?
- Which offline metric fits each component (false accepts and false rejects per hour, WER, intent accuracy, slot F1)?
- How do you tie component metrics to online customer metrics such as task completion or user friction? [Likely follow-up to [1]]
- How would you detect intent misclassification from implicit feedback such as rephrases, barge-ins or abandons? [Likely follow-up to [1]]
- How do you add a new intent or skill that has no training data?
- How would you handle accents, code-switching and new locales? (See also Q62 in chapter 06.)
- What end-to-end latency budget would you set, and how would you split it across the stages?
- Which online metric shows that a new NLU model is better, and how do you run that Weblab test?
- How do you stop errors compounding from ASR into NLU?
- How would you justify each model choice against a simpler baseline?

## 5. Summarize product reviews into key concepts

**Prompt.** "Science Application: design a system that summarizes product reviews by key concepts or aspects (for example, 'battery life', 'fit') for display on the product page. How do you extract aspects, aggregate sentiment, generate summaries, and evaluate faithfulness?" [Reported] [3][11] (Luxembourg AS onsite, posted August 2025; also a May 2026 Seattle report.)

**Maps to:** "Customers say" and review highlights on the detail page; Community Shopping.

```
 BATCH / NEAR-REAL-TIME
 [New reviews] -> [Abuse / fake-review filter] -> [Aspect extraction]
   -> [Aspect sentiment] -> [Aggregate per ASIN: counts, polarity, quotes]
   -> [Summary generator (LLM)] -> [Faithfulness check] -> [Detail-page widget]
 TRAINING / EVAL LOOP
 [Annotated aspects + summaries] -> [Fine-tune / prompt] -> [Offline faithfulness eval]
 [Widget engagement + reports] -> [Weblab A/B] -> [Refresh + drift monitoring]
```

Follow-ups:
- Would you use a fixed aspect taxonomy per product type, open extraction, or both?
- How do you merge synonymous aspects ("battery", "charge lasts")?
- How do you aggregate sentiment so a summary reflects proportions and not a few loud reviews?
- Extractive or abstractive summaries? Where does LLM hallucination creep in?
- How do you measure faithfulness at scale, and when would you trust an automatic checker over human raters?
- How do you keep fake or incentivized reviews from steering the summary? [Likely follow-up to [11]]
- What do you show for long-tail products with only a handful of reviews?
- How often do you refresh summaries, and how do you avoid sudden flips?
- How do you represent minority opinions and safety complaints so they are not averaged away?
- Which online metric shows the widget helps customers (conversion, returns, time to decision)?
- How do you handle reviews in many languages for one ASIN?

## 6. Design an end-to-end product recommendation engine

**Prompt.** "How would you design and evaluate a model that improves product recommendations on Amazon?" [Reported] [10]. Variants: "design various models from Recommendation Systems", comparing two recommender models for an Amazon product [Reported] [20]; "a real-time product recommendation engine" [Likely; role and date unconfirmed] [21]; "customers who viewed or bought X" on product pages, covering retrieval, ranking, features, real-time personalization, cold start and A/B testing [Likely] [36][48]; "a large-scale shopping recommender: when does two-tower retrieval beat a traditional ranking stack?" [Likely] [30][37][38].

**Maps to:** Amazon Personalization and Recommendations (detail page, homepage, "customers also bought").

```
 SERVING
 [User + context + anchor ASIN] -> [Candidate gen: co-purchase | two-tower | trending]
   -> [Merge + dedupe] -> [Ranker] -> [Re-rank: diversity, business rules] -> [Widget]
                                                                    |
 [Impressions, clicks, purchases logged with positions] <-----------+
 TRAINING LOOP
 [Behavior logs + catalog] -> [Labels + negative sampling] -> [Retrieval + ranker train]
 [Offline eval] -> [Embedding index refresh] -> [Weblab A/B] -> [Monitoring]
 [Real-time feature stream] -> [Online feature store]
```

Follow-ups:
- Which recommendation surfaces would you design for, and does each need its own objective?
- How do you generate candidates: item-to-item co-occurrence, two-tower retrieval, sequence models, or a mix?
- When would a two-tower retrieval model beat a traditional ranking stack, and how would you combine them?
- How do you train the two-tower model (in-batch negatives, hard negatives, sampling bias correction), and serve it with ANN search?
- What does the ranker optimize: click, add-to-cart, purchase, or a multi-task combination?
- Which features need real-time freshness (the session so far), and which can be batch?
- How do you handle cold start for new users and new ASINs?
- How do you correct for position bias and feedback loops in the training data?
- Which offline metrics would you use (recall@k, NDCG), and why might they disagree with online conversion?
- How would you design the Weblab test, and how do you account for cannibalization between widgets?
- What is your latency budget for candidate generation versus ranking?
- How do you balance short-term conversion against diversity and long-term customer value?

## 7. Frame an open-ended ML product problem

**Prompt.** "Given an open-ended ML product problem, frame it: reduce the scope, describe how you would gather data, how you would handle domain adaptation, whether you would distill a large model for serving, and which metrics you would use. Justify each choice against the alternatives." [Reported] [13] (Applied Scientist loop, 2021; the candidate felt they narrowed the scope too much and was down-leveled partly for lacking production experience.)

**Maps to:** any science team; this is how many team-specific prompts open.

```
 [Vague goal] -> [Scoped v1 problem] -> [Data plan: existing | weak | human labels]
 TRAINING
 [Source-domain data] -> [Large teacher model] -> [Domain adaptation on target data]
     -> [Distill to student] -> [Offline eval vs baselines + ablations]
 SERVING
 [Request] -> [Student model] -> [Fallback / teacher for hard cases] -> [Output]
 [Online metric + guardrails] -> [Iteration on scope]
```

Follow-ups:
- How did you decide what to cut from the scope, and what happens if you widen it again? [Reported] [13]
- Why this model or loss over the alternatives? [Reported] [13]
- What would you do without production-scale labeled data? [Reported] [13]
- Which domain-adaptation technique fits (fine-tuning, continued pretraining, pseudo-labeling, adversarial adaptation), and why?
- When is distillation worth it, compared with quantization or simply serving a smaller model?
- How do you know the student kept the teacher's quality on the slices that matter?
- Which offline metric would you report, and which online metric would decide the launch?
- Which ablations would you run to prove each component earns its place?
- What would you need to see to move from v1 scope to the full problem?

## 8. Design a spam or fake-review detection system

**Prompt.** "Design a spam detection system (for example, spam product reviews or messages). Cover labels, features, model choice, class imbalance, thresholds and how you would measure success." [Reported] [17] (Amazon AS virtual onsite, August 2025; this was an L4 loop.) The L5 variant: "Design a system that detects fake or incentivized reviews and review-abuse rings: reviewer-seller-product graph features, text signals, label scarcity, adversarial adaptation, and the human investigator queue." [Likely] [14][40][53]

**Maps to:** Community Shopping and Reviews integrity, Selling Partner abuse prevention.

```
 SERVING
 [New review / message] -> [Fast rules + text model] -> [Real-time score]
     -> [Threshold policy] -> [Publish | Hold | Investigator queue]
 BATCH
 [Reviewer-seller-product graph] -> [Graph / ring detection] -> [Account-level score]
 TRAINING LOOP
 [Investigator decisions + takedowns] -> [Labels (weak + gold)] -> [Retrain]
 [Precision-at-capacity eval] -> [Adversarial drift monitoring] -> back
```

Follow-ups:
- Where do labels come from, and how biased are labels that come only from items investigators chose to look at?
- How would you use weak supervision or active learning when gold labels are scarce?
- Which signals matter besides text (timing bursts, reviewer history, graph links to sellers, verified purchase)?
- When would a graph model beat per-review classification?
- How do you handle extreme class imbalance in training and in evaluation?
- How do you pick the operating threshold given investigator capacity and the cost of false positives on honest reviewers?
- Is precision at fixed capacity, recall, or prevented harm the right success metric?
- How do you detect when spammers adapt, and how fast can you retrain?
- How do you evaluate a new model when the old model shaped which items got labeled?
- How do you run an A/B test on an abuse system without exposing customers to more spam?
- How do LLM-generated fake reviews change your design?

## 9. Classify short, noisy text (review language, tweets)

**Prompt.** "Design a model that classifies the language of customer reviews." Expect follow-ups after your solution. [Reported] [10] Related: "design an algorithm or ML system to classify tweets into categories (topic, sentiment, complaint or abuse)." [Reported] [18][19][20]

**Maps to:** Reviews and Community, Customer Service, Social and brand listening, multilingual marketplaces.

```
 SERVING
 [Raw text] -> [Normalize: emoji, URLs, script] -> [Fast model]
     -> [Heavier model for hard cases?] -> [Label + confidence]
     -> [Downstream routing / analytics]
 TRAINING LOOP
 [Weak labels: marketplace, hashtags, rules] + [Human gold set] -> [Train]
 [Per-class / per-length eval] -> [Deploy] -> [Drift + new-class monitoring]
```

Follow-ups:
- Where do the labels come from, and how noisy are weak labels such as marketplace or hashtags?
- Which features or representations would you use: character n-grams, subword embeddings, or a pretrained multilingual encoder? Why?
- How do you handle very short texts, code-mixed text (for example Hinglish) and transliteration?
- Is this multi-class or multi-label, and how does that change the loss and the metrics?
- How do you handle class imbalance and rare categories?
- How would you evaluate it, by class, by text length and by marketplace?
- When is a simple linear model good enough, and when would you pay for a transformer?
- How do you add a new category or language after launch?
- How do you set the confidence threshold for "unknown"?
- What are the throughput and latency requirements, and would you run it in batch or in real time?

## 10. Design a GitHub Copilot-style coding agent

**Prompt.** "Design a GitHub Copilot agent: model choice, retrieval of repository context, tool use, evaluation, and latency and cost constraints." [Reported] [9] (Amazon AS report, 2025-2026. Chapter 06 lists it from a Senior AS loop.)

**Maps to:** Amazon Q Developer, AWS AI coding tools, internal developer productivity.

```
 SERVING
 [Editor context + request] -> [Context retrieval: repo index, open files, symbols]
   -> [Prompt builder] -> [Code LLM] -> [Tools: run tests, lint, search]
   -> [Edit / completion] -> [Developer accepts / edits / rejects]
 TRAINING / EVAL LOOP
 [Repo + task benchmarks] -> [Offline eval: pass@k, test pass] -> [Fine-tune / route]
 [Acceptance + retention telemetry] -> [Weblab A/B] -> [Monitoring + cost tracking]
```

Follow-ups:
- Inline completion versus a multi-step agent: which product are you building, and how does that change the model and the latency budget?
- How do you index and retrieve repository context, and how do you rank what fits in the context window?
- Which tools would the agent use, and how do you sandbox code execution?
- Would you use one large model or route between a small fast model and a large one?
- How do you evaluate offline: pass@k, test pass rate on repository tasks, human review?
- How do you measure whether it helps developers online (acceptance rate, retained code, downstream bugs)?
- How do you prevent leaking secrets or licensed code?
- What is your cost per request, and how would you halve it?
- How would you fine-tune on internal code without overfitting to stale patterns?
- How do you detect when the agent is confidently wrong?

## 11. Design a system for global request detection and labeling

**Prompt.** "Design a system for global request detection and labeling. State your assumptions, propose an architecture, and discuss scaling, latency and reliability." [Reported] [22] (Listed on an Amazon ML system design question page; the role is not specified.)

**Maps to:** traffic classification, abuse and bot detection, Ads traffic quality, AWS WAF-style services.

```
 SERVING (multi-region)
 [Requests] -> [Edge feature extraction] -> [Fast detector (rules + light model)]
     -> [Label: benign | suspect | known class] -> [Action / routing]
     -> [Async deep model on sampled traffic] -> [Label store]
 TRAINING LOOP
 [Sampled requests] -> [Human / heuristic labels] -> [Train + per-region eval]
 [Model distribution to regions] -> [Drift + latency monitoring] -> back
```

Follow-ups:
- What exactly counts as a "request" and what labels do you need? Which assumptions would you state first?
- Which detection must happen inline within the latency budget, and which can run asynchronously?
- Where do training labels come from at global scale, and how do you keep label quality consistent across regions?
- One global model or per-region models? How do you handle regional distribution shift?
- How do you keep the system available when the model service fails (fail open or fail closed)?
- How would you sample traffic for labeling so rare classes are covered?
- How do you roll out a new model across regions safely?
- Which metrics define success: precision, recall, latency, cost per million requests?
- How do you detect new request patterns that match no known label?

## 12. Design Amazon product search ranking

**Prompt.** "Design a ranker for Amazon product search and explain how it fits into the product: query understanding, retrieval (lexical plus semantic), learning-to-rank features, labels from clicks and purchases, position bias, and offline (NDCG) versus online (conversion) metrics." [Likely] [14][35][39][47] (Blind advises studying how rankers fit into products [14]; Search Ranking postings describe early-stage efficient models, personalization and exploration [47].) Variant: "How would you design a search system with LLMs?" [Reported] [24] (AS II, LLM team, March 2025).

**Maps to:** Amazon Search (A9), Search Relevance, multimodal search.

```
 SERVING
 [Query + user] -> [Query understanding: spell, product type, attributes]
   -> [Retrieval: lexical | semantic two-tower ANN] -> [Pre-rank (light)]
   -> [Rank (LTR)] -> [Re-rank: diversity, sponsored blend, LLM rerank?] -> [SERP]
                                                                      |
 [Impressions + clicks + purchases with positions] <------------------+
 TRAINING LOOP
 [Logs + human relevance labels] -> [Debias] -> [Train retrieval + rankers]
 [Offline NDCG / recall] -> [Interleaving / Weblab A/B] -> [Drift monitoring]
```

Follow-ups:
- What is relevance on Amazon: textual match, purchase likelihood, or both? How do you combine them?
- Which labels would you use (clicks, add-to-cart, purchases, human judgments), and how do you weight them?
- How do you handle position bias in click training data?
- How do lexical and semantic retrieval complement each other, and how do you merge their candidates?
- Why have a pre-ranking stage, and what latency budget does each stage get?
- How do you personalize without hurting the relevance of head queries?
- How do you explore new products (bandits or RL) without hurting conversion? [Likely] [47]
- How would you add an image query (multimodal search) to this stack? [Likely] [33]
- Where would an LLM help (query rewriting, re-ranking, generated answers), and what does it cost in latency?
- How do you handle tail queries and queries with zero results?
- How do you monitor feature drift across batch and real-time features at high QPS? [Likely] [32]

## 13. Design a Rufus-style LLM shopping assistant

**Prompt.** "Design an LLM shopping assistant that answers product questions using catalog data, reviews and Q&A. Cover retrieval-augmented generation (RAG), grounding and hallucination control, latency and cost, safety guardrails, and offline (LLM-as-judge, human evaluation) and online evaluation." [Likely] [2][3][24][25] (Recent AS II design rounds moved toward LLM and agentic systems; no report quotes a Rufus prompt verbatim.)

**Maps to:** Rufus, Amazon Stores generative AI.

```
 SERVING
 [Question + page context + user] -> [Query rewrite / intent] -> [Retrieve: catalog,
   reviews, Q&A, web?] -> [Rerank chunks] -> [LLM generation] -> [Grounding +
   safety check] -> [Answer + cited products]
 TRAINING / EVAL LOOP
 [Pretrained LLM] -> [SFT on shopping dialogs] -> [Preference tuning (RLHF / DPO)]
 [Offline eval: faithfulness, helpfulness, safety] -> [Weblab A/B] -> [Monitoring]
 [Thumbs, follow-ups, purchases] -> [Preference + error data] -> back
```

Follow-ups:
- What do you retrieve, and how do you chunk and index catalog attributes versus free-text reviews?
- How do you stop the model inventing specs, prices or compatibility claims?
- When would you use DPO instead of RLHF to align the assistant, and what preference data would you collect? [Reported] [25]
- How do decoding choices (greedy, beam search, sampling) affect latency, diversity and factuality? [Reported] [24]
- How would you fine-tune on domain data (SFT, then RLHF or DPO) without losing general ability?
- How do you build an offline evaluation set, and when is LLM-as-judge trustworthy?
- Which online metric shows the assistant helps (engagement, purchase conversion, returns), and what are the guardrails?
- How do you meet a latency budget at peak traffic (distillation, caching, routing, speculative decoding)?
- How do you detect and block unsafe, biased or policy-violating answers? Which red-teaming process would you use?
- How do sponsored products appear in answers without misleading customers?
- How do you keep answers fresh when prices and stock change by the minute?

## 14. Design CTR/CVR prediction and ad placement for Sponsored Products

**Prompt.** "Design CTR and CVR prediction for Sponsored Products ads in Amazon search, and decide which ads to show, where to place them, how many, and to which customers, under low-latency, high-volume constraints. How do you combine query-ad relevance, predicted CTR and the advertiser's bid?" [Likely] [42][43][44][46] (AS II and L5 Sponsored Products postings describe deep-learning CTR/CVR models, NLP relevance, bandits and causal inference.)

**Maps to:** Sponsored Products, Marketplace Intelligence, Sponsored Sourcing.

```
 SERVING (strict p99)
 [Query + shopper] -> [Ad sourcing: lexical + semantic ANN] -> [Relevance filter]
   -> [pCTR / pCVR model] -> [Calibration] -> [Auction: bid x pCTR + quality]
   -> [Placement + ad-load policy] -> [SERP slots]
                                              |
 [Impressions, clicks, delayed conversions] <-+
 TRAINING LOOP
 [Logs] -> [Delayed-label handling + debias] -> [Train pCTR / pCVR]
 [Offline AUC, log-loss, calibration] -> [Weblab A/B + advertiser metrics] -> [Monitor]
```

Follow-ups:
- How do you index millions of ads and retrieve relevant candidates for a query (lexical plus semantic two-tower)? [Likely] [45]
- Which features matter (shopper, query, ad, context), and which must be real-time?
- Which model architecture would you choose (GBDT, DeepFM, DCN, a multi-task CTR/CVR model), and why?
- Why does calibration matter for auction pricing, and how do you calibrate and monitor it?
- How do you correct for position bias in the training labels?
- How do you handle conversions that arrive days after the click?
- How do you cold-start new ads and new advertisers?
- How would a contextual bandit choose placements or creatives instead of a static model? How do you define the reward and evaluate a new policy off-policy? [Likely] [42]
- How many ads per page? How do you trade ad revenue against shopper experience?
- How do you A/B test when advertiser budgets couple treatment and control (budget cannibalization)?
- How would you build GenAI-based shopper understanding and audience targeting, and measure its lift? [Likely] [43]
- What does a 1 ms increase in model latency cost you, and where would you cut?

## 15. Detect invalid or bot traffic on Amazon Ads

**Prompt.** "Design a system that detects invalid or non-human (bot) traffic on Amazon Ads at billions of events per day. Cover features from user, context and creative engagement, the lack of labels, adapting to new attack patterns, and self-supervised or clustering methods." [Likely] [41] (Ad Fraud Detection, Traffic Quality posting.)

**Maps to:** Amazon Ads Traffic Quality, Ad Fraud Detection.

```
 STREAMING
 [Ad events: impressions, clicks] -> [Sessionize + real-time features]
   -> [Rules + online model] -> [Invalid / valid] -> [Billing filter]
 BATCH
 [Event history] -> [Self-supervised embeddings] -> [Clustering / anomaly detection]
   -> [Analyst review] -> [Labels] -> [Supervised model retrain]
 [Advertiser refunds, audits] -> [Delayed ground truth] -> [Eval + drift monitoring]
```

Follow-ups:
- With almost no labels, where does ground truth come from (honeypots, known bot lists, analyst review)?
- Which signals separate bots from humans (timing, device, IP diversity, engagement depth, creative interaction)?
- How would self-supervised representations or clustering surface new attack patterns?
- What must be decided in real time before billing, and what can be corrected afterwards in batch?
- How do you set thresholds when false positives cost publishers and false negatives cost advertisers?
- How do you evaluate a detector when attackers adapt to it?
- How do you scale feature computation to billions of events a day?
- How would you explain a decision to an advertiser who disputes a refund?
- How do you detect sophisticated bots that mimic human behavior?

## 16. Design a fraud detection system for payments or orders

**Prompt.** "Build a real-time fraud detection pipeline for Amazon orders and payments. How do you trade off false positives (blocking good customers) against missed fraud? Cover delayed labels such as chargebacks, adversarial drift and real-time scoring latency." [Likely] [36][37]

**Maps to:** Transaction Risk Management, Amazon Payments, Buyer Risk Prevention.

```
 SERVING (checkout path)
 [Order + account + payment] -> [Real-time features: velocity, device, graph]
   -> [Risk model] -> [Decision policy: approve | challenge | review | block]
 BATCH
 [Account-device-address graph] -> [Graph features] -> [Feature store]
 TRAINING LOOP
 [Chargebacks (weeks later) + investigator labels] -> [Label maturity window]
   -> [Retrain] -> [Cost-weighted eval] -> [Shadow -> Weblab] -> [Drift monitoring]
```

Follow-ups:
- How do you define the label, and how do you train when chargebacks arrive 30-90 days later?
- How do you handle extreme class imbalance in training and evaluation?
- Which features need sub-second freshness (velocity counters), and how do you keep them consistent between training and serving?
- How do you choose operating thresholds from the dollar cost of fraud versus customer friction?
- Blocked orders never produce a label. How do you correct for that selection bias?
- How do you detect adversarial drift, and how fast can you retrain?
- What role do graph models play in catching fraud rings?
- How would you A/B test a fraud model without letting extra fraud through?
- How do you meet the latency budget of the checkout path?

## 17. Extract structured product attributes for catalog quality

**Prompt.** "Design a system that extracts and normalizes product attributes (brand, size, color, material) from noisy seller titles, descriptions and images to improve catalog quality. How would you get training labels at scale and measure precision and recall?" [Likely] [6][14][34] Related reported probe: "Describe the Segment Anything Model (SAM) and how it works. How would you use it in a product-image pipeline, for example for background removal or attribute extraction?" [Reported] [8]

**Maps to:** Amazon Catalog, Selling Partner listing quality. Chapter 13 covers this prompt in depth; this entry is the compact version.

```
 BATCH + ON LISTING CREATE
 [Title, bullets, description, images] -> [Product-type classifier]
   -> [Text extractor (NER / seq2seq / LLM)] + [Image model (VLM, SAM crops)]
   -> [Fuse + normalize to schema] -> [Confidence policy] -> [Auto-fill | Human review]
 TRAINING LOOP
 [Existing attributes (noisy)] + [LLM labels] + [Gold audits] -> [Train]
 [Per-attribute P/R eval] -> [Seller accept / reject feedback] -> [Retrain]
```

Follow-ups:
- How do you get labels at scale (existing seller attributes, weak supervision, LLM labeling, human audits)?
- NER, sequence-to-sequence, or an LLM? How does cost at catalog scale change the answer?
- How do you handle thousands of product types with different attribute schemas?
- How do you normalize values ("XL", "extra large", "42 EU")?
- When does the image add information the text lacks, and how do you fuse the two?
- How would you evaluate per attribute and per product type, and which metric do you report upward?
- How do you choose confidence thresholds for auto-filling versus human review?
- How do you measure the downstream impact on search and conversion?
- How do you handle sellers who game attributes to rank higher?

## 18. Detect duplicate listings in the catalog

**Prompt.** "Design a duplicate-listing detection system for the Amazon catalog that matches the same product across sellers, using text and image embeddings, blocking or ANN retrieval, and a pairwise matcher." [Likely] [38]

**Maps to:** Catalog quality, product matching, entity resolution.

```
 BATCH + INCREMENTAL
 [New / updated listing] -> [Text + image encoders] -> [Blocking: ANN | key fields]
   -> [Candidate pairs] -> [Pairwise matcher] -> [Clustering / merge decision]
   -> [Auto-merge | Human review | No match]
 TRAINING LOOP
 [Known merges + human-labeled pairs + hard negatives] -> [Train encoders + matcher]
 [Pair P/R + cluster purity eval] -> [Merge-reversal feedback] -> [Retrain]
```

Follow-ups:
- What is a "duplicate": the same product, the same variant, or the same product in a different pack size?
- How do you generate candidate pairs without comparing every pair of listings?
- Which hard negatives matter most (same brand, different model; near-identical images)?
- How do you go from pairwise scores to consistent clusters (transitivity)?
- Is a wrong merge or a missed merge more costly, and how does that set the threshold?
- How do you evaluate recall when you cannot label all pairs?
- How do you handle sellers who copy another seller's images?
- How do you run this incrementally for millions of new listings a day?

## 19. Design a demand forecasting system for millions of ASINs

**Prompt.** "Design a demand forecasting system for millions of ASINs across fulfillment centers. Cover hierarchical and probabilistic (quantile) forecasts, intermittent demand, cold-start items, promotions, and the metrics used for inventory decisions." [Likely] [37][39][40]

**Maps to:** Supply Chain Optimization Technologies (SCOT), Forecasting.

```
 BATCH (daily / weekly)
 [Sales, price, promos, events, catalog] -> [Feature pipeline]
   -> [Forecast model] -> [Forecast per ASIN x FC x horizon]
   -> [Hierarchical reconciliation] -> [Buying / placement optimizer]
 TRAINING LOOP
 [History with stockout censoring] -> [Train] -> [Backtest]
 [Error metrics by segment] -> [Forecast-to-cost eval] -> [Monitoring + retrain]
```

Follow-ups:
- One global model or per-ASIN models? What are the trade-offs?
- Why probabilistic forecasts, and which quantiles does the inventory decision actually consume?
- How do you handle intermittent demand for long-tail items?
- How do you forecast a brand-new ASIN with no history?
- How do you account for stockouts that censor observed demand?
- How do you model promotions, holidays and price changes?
- How do you keep forecasts consistent across the hierarchy (ASIN, category, FC, region)?
- Which metric would you choose (wQL, MAPE, bias), and how does forecast error translate into ordering cost?
- How do you backtest fairly over time?
- How would you detect and respond to a sudden demand shift?

## 20. Predict delivery ETA (promise date)

**Prompt.** "Design a delivery ETA (promise date) prediction system for Amazon orders. Which features would you use (carrier, node, warehouse load, traffic, weather), point versus quantile prediction, how would you produce calibrated uncertainty, and how would you trade off the cost of late versus early promises?" [Likely] [30][53] Variant: "Develop an algorithm or ML system to optimize delivery routes for last-mile delivery." [Likely]

**Maps to:** Last Mile, Delivery Experience, Fulfillment by Amazon promise.

```
 SERVING (at detail page and checkout)
 [Item + destination + options] -> [Real-time features: FC load, carrier capacity]
   -> [ETA model] -> [Promise policy] -> [Promise shown]
 TRAINING LOOP
 [Shipment scans + actual delivery] -> [Labels] -> [Train] -> [Calibration eval]
 [Late / early rate by lane] -> [Weblab A/B on conversion + on-time] -> [Monitoring]
```

Follow-ups:
- Which quantile becomes the promise, and how do you choose it from the costs of late versus early?
- How do you validate that predicted intervals are calibrated?
- Which features are available at the detail page versus after checkout?
- How do you handle peak season, weather events and carrier disruptions?
- The promise itself changes customer behavior (conversion). How does that feed back into training?
- How do you evaluate a new model online (on-time rate, conversion, contacts)?
- How do you handle new lanes or new delivery stations with no history?
- What latency budget does the ETA model have on the detail page?

## 21. Estimate price elasticity for pricing

**Prompt.** "Design a pricing or price-elasticity model for Amazon retail. How would you estimate the causal effect of price on demand from observational data, handle confounding, validate it, and run experiments safely?" [Likely] [38]

**Maps to:** Pricing, Retail Business Services, Deals.

```
 [Price + demand history] -> [Confounder features: promos, competitors, season]
   -> [Causal estimator] -> [Elasticity per ASIN / segment]
   -> [Price optimizer + business constraints] -> [Price change]
 VALIDATION LOOP
 [Randomized price tests (where allowed)] -> [Compare with causal estimates]
 [Outcome monitoring: units, margin, customer trust] -> [Re-estimate]
```

Follow-ups:
- Why is a naive regression of demand on price biased, and which confounders worry you most?
- Which identification strategy would you use (double ML, instrumental variables, natural experiments)?
- How do you get elasticity for long-tail items with little price variation?
- How do you validate an elasticity estimate when you cannot see the counterfactual?
- How do you account for cross-elasticity and cannibalization between similar items?
- What constraints keep a pricing system customer-trustworthy?
- How would you design a safe price experiment?

## 22. Diagnose and redeploy a degrading production model

**Prompt.** "You have a model in production whose performance is degrading. Walk through how you would diagnose it (data drift, label drift, pipeline bugs), retrain, and safely redeploy. Cover the feature store, batch versus real-time serving, drift monitoring, and the rollout or rollback strategy." [Likely] [26][40][54] (Recent reports describe practical design rounds tied to day-to-day ML work [6][26]; PracHub says Amazon rounds mix in feature stores, serving trade-offs, drift monitoring and rollout strategy [22].)

**Maps to:** any production science team.

```
 SERVING
 [Requests] -> [Online feature store] -> [Model vN] -> [Predictions] -> [Outcomes]
 MONITORING
 [Input drift] [Prediction drift] [Label / KPI drift] [Skew: train vs serve] [Latency]
   -> [Alert + slice analysis] -> [Root cause: data | pipeline | concept]
 RETRAIN + ROLLOUT
 [Fresh data] -> [Retrain vN+1] -> [Offline gate] -> [Shadow] -> [Canary] -> [Weblab]
   -> [Promote | Rollback to vN]
```

Follow-ups:
- What is your first hour of investigation: what do you check first, and why?
- How do you tell data drift, concept drift and a pipeline bug apart?
- How do you detect training/serving skew, and how does a feature store help?
- When would you choose batch inference over real-time inference, and how do you keep features consistent between them? [Likely] [54]
- Which metrics do you monitor when labels arrive weeks late?
- What triggers retraining: a schedule, a drift alarm, or a KPI drop?
- What offline gates must a retrained model pass before it sees traffic?
- How do you design the shadow and canary stages, and what triggers an automatic rollback?
- How do you avoid a feedback loop where the degraded model shaped its own training data?
- How would you explain the incident and the fix to a non-science stakeholder?

## 23. Design Prime Video homepage recommendations with cold start

**Prompt.** "Design Prime Video's homepage recommendation system: candidate generation, ranking for watch time versus engagement, row and page layout, and freshness for new releases. Include a strategy for both model cold start and new-user cold start, with serving, evaluation and safe rollout." [Likely] [31][52]

**Maps to:** Prime Video personalization.

```
 SERVING
 [Profile + device + time] -> [Candidates: CF | content | trending | new releases]
   -> [Title ranker] -> [Row generator + page layout optimizer] -> [Homepage]
 COLD START
 [New user: onboarding, context priors] | [New title: content embeddings, metadata]
 TRAINING LOOP
 [Plays, watch time, browse] -> [Labels] -> [Train] -> [Offline eval]
 [Weblab A/B on long-term engagement] -> [Monitoring]
```

Follow-ups:
- What is the objective: clicks, starts, watch time, or long-term retention? How do you avoid clickbait?
- How do you rank titles within rows and rows within the page?
- How do you recommend for a brand-new profile with no history?
- How do you surface a new release that has no interaction data?
- What does "model cold start" mean for a new model launch, and how do you roll it out safely?
- How do you handle shared household profiles?
- How do you measure long-term effects in a short A/B test?
- How do licensing windows and content expiry affect the system?

## 24. Design GenAI ad creative generation for sellers

**Prompt.** "Design an LLM system that automatically generates ad creatives (headlines, images) for sellers. How do you ensure brand safety, factual consistency with the catalog, and measure performance?" [Likely] [33] Related: "Design the prediction, ranking and bidding system for Amazon's programmatic (DSP) advertising." [Likely] [49]

**Maps to:** Amazon Ads creative generation, Amazon DSP.

```
 GENERATION
 [ASIN catalog data + brand guidelines] -> [Text LLM] + [Image generator]
   -> [Fact check vs catalog] -> [Brand-safety + policy classifiers]
   -> [Advertiser review] -> [Creative pool]
 SELECTION LOOP
 [Creative pool] -> [Bandit / CTR model selects creative] -> [Serve]
 [CTR, CVR, advertiser edits] -> [Preference data] -> [Tune generators] -> back
```

Follow-ups:
- How do you guarantee generated claims match the catalog (no invented features)?
- How do you check brand safety and ad-policy compliance before anything is served?
- How do you preserve product fidelity in generated lifestyle images?
- How do you measure creative quality offline before spending advertiser money?
- How would a bandit pick among generated creatives, and how do you evaluate it?
- How do advertiser edits become training signal?
- What is the cost per creative, and how does it scale to millions of ASINs?

## 25. Cross-cutting probes asked across any prompt

These are not design prompts on their own. They are the probes interviewers attach to whichever prompt you get.

- Your model improves offline AUC or NDCG but the A/B test shows no lift. What could explain this, and how would you design the experiment (randomization unit, guardrail metrics, sample size, novelty effects)? [Likely] [36][40][51]
- How would you handle severe class imbalance and label noise, and how would you choose an operating threshold from business costs? [Likely] [28]
- Explain the bias-variance trade-off and compare bagging with boosting. Which would you use for a tabular fraud or CTR model, and why? [Reported] [2][4]
- Pick a project on your resume. Why did you choose that model, loss and evaluation over the alternatives? What were the baselines and ablations, what failed, and what would you change to deploy it at Amazon scale? [Reported] [1] (a Senior AS prep thread describes the same probe [27])
- How would cold start (new users, new items, a new model with no logged history) change the design you just drew? [Likely] [50]
- What is the proxy label, what is the offline metric versus the business metric, and how will you launch? [Likely] [29]
- Which parts of your design would you build, and which would you reuse from existing Amazon or AWS services?
- How would you cut the cost of this system by 50% with minimal quality loss?
- If you had ten times the data, or one-tenth of it, how would the design change?

## Level calibration: harder L5 probes

The bank above leans on component-level follow-ups. These probes test the L5 bar: owning an ambiguous problem end to end, choosing metrics and experiments, and trading model complexity against latency and cost. Attach any of them to any prompt. They are untagged: they are how interviewers typically push past a junior answer, not quoted reports.

1. Before any model, what one-page problem statement would you write, and which open question would you take back to the product manager first?
2. Your proxy label and the business metric disagree on the launch decision. Which do you trust, and what evidence would change your mind?
3. You have p99 headroom for only a model 10x smaller than your best offline model. Where do you spend that budget, and how do you quantify the quality you give up?
4. Name a design choice you would reverse if serving cost doubled, and one you would keep regardless. Why?
5. Which metric would you refuse to optimize directly here, even though leadership asks for it, and what would you propose instead?
6. Your Weblab shows a statistically significant but tiny lift with a guardrail moving the wrong way. Do you launch? Who decides, and on what evidence?
7. The randomization unit you need is not the one the platform supports (for example, interference across sellers, households or advertisers). How do you still get a trustworthy estimate?
8. Which slice (new customers, a small marketplace, a long-tail category) could your launch quietly harm, and how would you detect it before the aggregate metric does?
9. What would you deliberately not build in v1, and how would you defend that cut to a skeptical senior engineer?
10. Six months after launch, the gain has eroded. How do you separate feedback-loop effects, novelty decay and real drift?
11. How do you size the team and timeline for v1, and which dependency on another team is the biggest schedule risk?
12. What offline experiment, run in a week, would most reduce your uncertainty about whether the full system is worth building?

---

## References

Glassdoor links with a review ID (RVW...) point to anonymous self-reports, often read only as search-result extracts. Where the search returned the same review on several country mirrors, the mirrors are listed together. "ID approximate" means the search could not confirm which of several returned review IDs held the quoted text.

1. r/leetcode post, Aug 27 2024 (Applied Scientist 2 loop; "design Alexa from scratch"; resume deep-dive depth round). Reddit via a redlib mirror, 2024. https://redlib.hbubli.cc/r/leetcode/s/PljCbbdntj
2. Amazon Interview review RVW102987402 (Seattle AS, May-June 2026; agentic recommendation design; loop structure; bagging vs boosting; AUC coding). Glassdoor, 2026. https://www.glassdoor.com.au/Interview/Amazon-Interview-E6036-RVW102987402.htm ; mirror https://www.glassdoor.ca/Interview/Amazon-Interview-E6036-RVW102987402.htm
3. Amazon Entretien review RVW104708794 (Seattle AS II, 2026; agentic recommender with quality and safety evaluation; review summarization; team-specific design). Glassdoor (French mirrors), 2026, ID approximate. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104708794.htm ; mirror https://fr.glassdoor.ca/Entretien/Amazon-Entretien-E6036-RVW104708794.htm
4. "J'ai passé un entretien chez Amazon (Seattle, WA) en juin 2026", review RVW104288349 (1 Science ML Breadth, 1 Depth, 1 Sys Design (Team-specific), 1 coding, 1 Bar Raiser; bagging vs boosting). Glassdoor, 2026. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104288349.htm
5. Amazon Entretien review RVW103457600 (multimodal translation pipeline design). Glassdoor, 2026. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW103457600.htm
6. Amazon Interview review RVW102945860 (Hyderabad AS, 2026; multimodal translation; day-to-day ML questions). Glassdoor India, 2026, ID approximate. https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW102945860.htm
7. Amazon Entrevista review RVW96903377 (Hyderabad, April; multimodal translation pipeline). Glassdoor Mexico, 2025-2026; the review ID suggests 2025, so the year is unverified. https://www.glassdoor.com.mx/Entrevista/Amazon-Entrevista-E6036-RVW96903377.htm
8. Amazon Entretien review RVW98336019 (AS II, 2026; multimodal translation; "Describe SAM, how it works"). Glassdoor (French mirrors), 2026, ID approximate. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW98336019.htm ; mirror https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW98336019.htm
9. Amazon Entretien review RVW102577344 ("Design a Github Copilot agent"). Glassdoor, 2025-2026. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW102577344.htm ; mirror https://www.glassdoor.co.nz/Interview/Amazon-Interview-E6036-RVW102577344.htm
10. Amazon Colloquio review RVW56678112 (classify the language of reviews; recommendation model design). Glassdoor Italy, year not stated, attribution approximate. https://www.glassdoor.it/Colloquio/Amazon-Colloquio-E6036-RVW56678112.htm
11. Amazon Interview review RVW99133644 (Luxembourg AS, posted Aug 2025; Science Application: summarize product reviews by key concepts; LLM training questions). Glassdoor Singapore, 2025. https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
12. Amazon Interview review RVW79375451 (Berlin; team-specific design focused on standard and recent ML approaches). Glassdoor Singapore, year not stated. https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW79375451.htm
13. "Amazon, MSFT, Google, Apple, FB, Zoom | Applied Scientist | Multiple | Sep 21 [OFFER, rest Reject]". LeetCode Discuss, 2021. https://leetcode.com/discuss/interview-experience/1585287/amazon-msft-google-apple-fb-zoom-applied-scientist-multiple-sep-21-offer-rest-reject ; alternate https://leetcode.com/discuss/post/1585287/
14. "Amazon MLE to AS transition". Blind, year not stated. https://www.teamblind.com/post/amazon-mle-to-as-transition-jzgcd2gq
15. "Amazon applied scientist loop". Blind, year not stated. https://www.teamblind.com/post/amazon-applied-scientist-loop-odlzgrp3
16. "Amazon AS2 interview loop". Blind, 2025. https://www.teamblind.com/post/amazon-as2-interview-loop-ry5amdu4
17. "Amazon New Grad Applied Scientist interview experience: five virtual onsite rounds, mostly behavioral, end in an L4 offer". PracHub, 2025. https://prachub.com/interview-experiences/amazon-new-grad-applied-scientist-interview-experience-five-virtual-onsite-rounds-mostly-behavioral-end-in-an-l4-offer
18. Amazon Interview review RVW30340277 (AS; "design an algorithm to classify tweets"). Glassdoor India, 2021, ID approximate. https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW30340277.htm
19. Amazon Entretien review RVW60311509 (designing an ML algorithm to classify tweets). Glassdoor France, year not stated. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW60311509.htm
20. Applied Scientist Interview Questions (listing page; tweets classification; "design various models from Recommendation Systems"; ML application round about an Amazon product). Glassdoor Spain, year not stated. https://static.glassdoor.es/Interview/applied-scientist-interview-questions-SRCH_KO0,17_IP81.htm
21. Amazon Interview review RVW81627346 (real-time product recommendation engine; role not confirmed as AS). Glassdoor Singapore; the extract said May 2026 but the review ID suggests an earlier year, so the date is unverified. https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW81627346.htm
22. Amazon ML System Design Interview Questions (global request detection and labeling; feature stores, drift, rollout). PracHub, 2025. https://prachub.com/companies/amazon/categories/ml-system-design
23. "Amazon Applied Scientist Interview Guide (Alexa International Tech)". Medium (Interview Preparation), 2025. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
24. Amazon Interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara, March 2025; beam search decoding for LLMs; "design a search system with LLMs"). Glassdoor Hong Kong, 2025. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
25. Amazon Entretien review RVW96863276 ("Differences between DPO and RLHF"). Glassdoor (French mirrors), 2025. https://fr.glassdoor.ca/Entretien/Amazon-Entretien-E6036-RVW96863276.htm ; mirror https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW96863276.htm
26. Amazon Interview review RVW47917252 (AS; two practical ML design rounds tied to day-to-day work). Glassdoor India; the extract said April 2026 but the review ID suggests an earlier year, so the date is unverified. https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW47917252.htm
27. "Amazon Senior Applied Scientist prep". Blind, year not stated. https://www.teamblind.com/post/amazon-senior-applied-scientist-prep-xyoamorv
28. "Amazon Applied Scientist 2 interview". Blind, 2025. https://www.teamblind.com/post/amazon-applied-scientist-2-interview-rdv5k0ef
29. "Nailing the Machine Learning Design Interview" (by a senior Amazon AS). Towards Data Science, year not stated. https://towardsdatascience.com/nailing-the-machine-learning-design-interview-6b91bc1d036c/
30. "What is an Applied Scientist at Amazon Services?" Dataford (prep vendor), 2025. https://dataford.io/interview-guides/amazon-services/applied-scientist
31. Prime Video Applied Scientist interview guide. Dataford (prep vendor), 2025. https://dataford.io/interview-guides/prime-video/applied-scientist
32. Amazon Lab126 Applied Scientist interview guide. Dataford (prep vendor), year not stated. https://dataford.io/interview-guides/amazon-lab126/applied-scientist
33. "What is an Applied Scientist at Amazon Advertising?" Dataford (prep vendor), 2025. https://dataford.io/interview-guides/amazon-advertising/applied-scientist
34. "What is an Applied Scientist at Amazon?" Dataford (prep vendor), 2025. https://dataford.io/interview-guides/amazon/applied-scientist
35. "How would you approach designing a search ranking algorithm for Amazon?" Dataford (prep vendor), year not stated. https://dataford.io/questions/how-would-you-approach-designing-a-search-ranking-algorithm-for-amazon
36. Amazon Applied Scientist interview prep. CleverPrep (prep vendor), 2025. https://www.cleverprep.com/companies/amazon/applied-scientist
37. Amazon Machine Learning Engineer guide. Codemia (prep vendor), 2025. https://codemia.io/guides/amazon-machine-learning-engineer
38. Amazon Applied Scientist interview questions. Interview Kickstart (prep vendor), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
39. Amazon Applied Scientist interview guide. Interview Kickstart (prep vendor), 2025. https://interviewkickstart.com/odio/amazon-applied-scientist-interview-guide
40. Amazon Applied Scientist Interview Questions. PracHub, 2025. https://prachub.com/companies/amazon/positions/applied-scientist
41. Applied Scientist, Ad Fraud Detection, Traffic Quality - ML (Job ID 2642812). amazon.jobs, 2025. https://amazon.jobs/en/jobs/2642812/applied-scientist-ad-fraud-detection-traffic-quality-ml
42. Applied Scientist, Sponsored Products Marketplace Intelligence (Job ID 2734206). amazon.jobs, year not stated. https://amazon.jobs/en/jobs/2734206/applied-scientist-sponsored-products-marketplace-intelligence
43. Applied Scientist II, Sponsored Products (Job ID 2919860). amazon.jobs, 2025. https://amazon.jobs/en/jobs/2919860/applied-scientist-ii-sponsored-products
44. Applied Scientist, Sponsored Products (Job ID 2828425). amazon.jobs (Japan), year not stated. https://amazon.jobs/jp/jobs/2828425/applied-scientist-sponsored-products
45. Applied Scientist, Sponsored Products (Sponsored Sourcing). amazon.dejobs.org, year not stated. https://amazon.dejobs.org/seattle-wa/applied-scientist-sponsored-products-amazon/D36BC87C80E24BD1A253ADB0E0940534/job
46. Applied Scientist (L5), Sponsored Products Marketplace Intelligence, New York. amazon.dejobs.org, 2025. https://amazon.dejobs.org/new-york-ny/applied-scientist-l5-sponsored-products-marketplace-intelligence/DAEEB404497146528424E91C0D070B6E/job
47. Amazon Applied Scientist, Search Ranking. myGwork (job mirror), 2025. https://mygwork.com/jobs/amazon-applied-scientist-search-ranking-6534939
48. Applied Scientist, MAPLE - Recommender System. myGwork (job mirror), year not stated. https://mygwork.com/jobs/amazon-applied-scientist-maple-recommender-system-6249539
49. Senior Applied Scientist, Amazon Ads, Demand Tech. AnitaB.org job board, 2025. https://jobs.anitab.org/companies/amazon-3-60ad394d-c673-4474-9694-344b0cae748f/jobs/84048208-senior-applied-scientist-amazon-ads-demand-tech-amazon-advertising-demand-tech
50. "What the ML System Design Interview Actually Tests in 2026". techinterview.org, 2026. https://www.techinterview.org/post/3233476380/ml-system-design-interview-what-it-tests-2026/
51. "ML System Design: Build a Search Ranking System". techinterview.org, 2026. https://www.techinterview.org/post/3233460105/ml-system-design-build-a-search-ranking-system/
52. Recommender systems (tag page). Amazon Science, year not stated. https://www.amazon.science/tag/recommender-systems?p=16
53. "Amazon System Design Interview". IGotAnOffer, year not stated. https://igotanoffer.com/en/advice/amazon-system-design-interview
54. "Amazon Software Engineer Interview Experience: Boomerang Loop With a Surprise ML Platform System Design" (an SDE loop, not AS). PracHub, 2025. https://prachub.com/interview-experiences/amazon-software-engineer-interview-experience-boomerang-loop-with-a-surprise-ml-platform-system-design
