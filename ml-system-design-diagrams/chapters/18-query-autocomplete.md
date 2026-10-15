# 18. Design Search Query Autocomplete / Typeahead with ML Ranking

> Source note: refs [1] [3] [5] [6] [14] [15] were fetched for this review. Several other
> domains (arXiv, ByteByteGo, Google, LinkedIn, Instacart, Algolia) could not be reached
> from this run; those references are well-known primary sources, and the Instacart and
> Algolia URLs are confirmed by the index in [15]. Rows marked "(assumption)" are typical
> interview defaults, not cited figures.

## Company tags

| Company / context | What links it to this problem | Source |
|---|---|---|
| Google (Search, Gmail) | Chip Huyen's exercise #5 uses Google's related searches. Gmail Smart Compose is the production neural-completion reference | [3] [9] [10] |
| Amazon (product search) | Session-aware QAC built as extreme multi-label ranking (KDD 2021) | [7] |
| Microsoft (Bing) | Personalized QAC learning-to-rank (SIGIR 2013) and rare-prefix QAC (CIKM 2015) | [5] [6] |
| LinkedIn | Partial, out-of-order, real-time typeahead over connections, companies, groups, skills (Cleo, open-sourced 2012) | [8] [14] |
| Instacart | ML-driven autocomplete for grocery search (2022) | [11] |
| Algolia (SaaS search) | Query Suggestions product built from search analytics | [12] |
| Meta, Uber, DoorDash, etc. | Often reported on forums as a search sub-problem. Not verified for this chapter | (unverified) |

**Portals that cover it**

- alirezadir / AIMLInterviews lists "Autocompletion / typeahead suggestion system" as an NLP sample MLSD question [1].
- Chip Huyen, ML Systems Design exercises #5 (related searches) and #10 ("finish your sentence when you text") [3] [4].
- ByteByteGo / Alex Xu, *System Design Interview* Vol. 1, Ch. 13 "Design a Search Autocomplete System". This is the trie/infra version that most ML answers build on [2].
- DesignGurus / Educative Grokking, "Designing Typeahead Suggestion" (infra version) [13].
- Engineer1999 curated case-study list (Instacart, Algolia entries) [15].

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Return the top-K (5-10) completions for a prefix on every keystroke. Personalized, fresh (trending), safe | [2] [5] |
| Optional | Spelling-tolerant prefixes, entity suggestions (people, products), multi-language | [8] (assumption) |
| ML objective | Rank candidate completions by P(user selects completion and the search succeeds) | [5] [7] |
| DAU | 10M users, 10 searches/day, about 20 chars/query, one request per keystroke | [2] |
| QPS | About 24k QPS average and about 48k peak in the Xu estimate. FAANG-scale is more than 100k (assumption) | [2] |
| Latency | Under 100 ms end-to-end, or the suggestions feel laggy. Model budget 5-20 ms (assumption) | [2] |
| Candidate pool | 10M-1B distinct historical queries. Top-N per prefix is precomputed | [2] (assumption) |
| Freshness | Trending queries surface within minutes. Base index rebuilt daily or weekly | [2] (assumption) |
| Non-functional | Highly available, safe (no offensive or defamatory completions), privacy (no PII leakage) | (assumption) |

## Diagram 1: Online serving path

**DIAGRAM 1: Online keystroke-to-suggestions path (synthesised from [2][5][6][7][8])**

```
 keystroke "how to b"
+--------------+  debounce   +-----------+  cache hit (popular prefix)
| Client / app |------------>| CDN /edge |------------------------------+
| local cache, |  ~50ms      | prefix    |                              |
| history      |             | cache     |                              |
+--------------+             +-----+-----+                              |
       ^                           | miss                               |
       |                           v                                    |
       |                  +-----------------+     +------------------+  |
       |                  | QAC API gateway |---->| Prefix normalize |  |
       |                  +--------+--------+     | lower, unicode,  |  |
       |                           |              | spell-fix prefix |  |
       |                           v              +------------------+  |
       |      +----------------------------------------------------+    |
       |      | CANDIDATE GENERATION  (~200-1000 candidates)       |    |
       |      |  +----------+  +-----------+  +----------------+   |    |
       |      |  | Trie/FST |  | Trending  |  | Personal hist. |   |    |
       |      |  | top-N per|  | stream    |  | + session ctx  |   |    |
       |      |  | prefix   |  | index     |  | (last queries) |   |    |
       |      |  +----------+  +-----------+  +----------------+   |    |
       |      |  +--------------------+  +-----------------------+ |    |
       |      |  | Suffix synth (rare |  | Neural LM / seq2seq   | |    |
       |      |  | prefix) [6]        |  | generator (optional)  | |    |
       |      |  +--------------------+  +-----------------------+ |    |
       |      +--------------------------+-------------------------+    |
       |                                 | union + dedup                |
       |                                 v                              |
       |      +----------------------------------------------------+    |
       |      | FILTERING: blocklist, offensive/PII classifier,    |    |
       |      | legal takedowns, zero-result queries, near-dupes   |    |
       |      +--------------------------+-------------------------+    |
       |                                 v                              |
       |  +-----------+  features  +-----------------------------+      |
       |  | Feature   |----------->| RANKER  (GBDT / small DNN)  |      |
       |  | store     |  prefix,   | score = P(select & success) |      |
       |  | (online)  |  query,    +--------------+--------------+      |
       |  +-----------+  user,          ^         |                     |
       |                   context      | model   v                     |
       |                      +---------+  +-------------------+        |
       |                      | Model   |  | RE-RANK / POLICY  |        |
       |                      | store   |  | diversity, intent |        |
       |                      +---------+  | mix, freshness,   |        |
       |                                   | semantic dedup    |        |
       |                                   +---------+---------+        |
       |          top-K suggestions                  |                  |
       +---------------------------------------------+------------------+
                                                     |
                                    log impressions, keystrokes,
                                    shown list, selection, final query
                                                     v
                                            +-------------------+
                                            | Logging (Kafka)   |
                                            +-------------------+
```

## Diagram 2: Offline training & data pipeline

**DIAGRAM 2: Logs to index and ranker, with feedback loop (synthesised from [2][5][7][11])**

```
+-----------------+    +-------------------+    +----------------------------+
| Search logs     |--->| Sessionize + join |--->| Label generation           |
| queries, clicks |    | keystroke stream, |    | per prefix impression:     |
| QAC impressions |    | shown list, pick, |    | pos = selected completion  |
| keystrokes      |    | final query,      |    |   or final submitted query |
+-----------------+    | downstream click/ |    | neg = shown, not selected  |
                       | purchase          |    | weight by search success   |
                       +---------+---------+    +-------------+--------------+
                                 |                            |
            aggregate counts     v                            v
                       +-------------------+    +----------------------------+
                       | Query stats       |    | Feature generation         |
                       | freq, time-decay, |--->| prefix-query, user, ctx    |
                       | geo, CTR, success |    | (offline = online parity)  |
                       +---------+---------+    +-------------+--------------+
                                 |                            |
            build top-N per      v                            v
            prefix      +-------------------+    +----------------------------+
                        | Trie / FST index  |    | Train ranker (LambdaMART / |
                        | builder (daily) + |    | DNN), hyper-param search   |
                        | trending (minutes)|    +-------------+--------------+
                        +---------+---------+                  v
                                  |              +----------------------------+
                                  |              | Offline eval: MRR, nDCG,   |
                                  |              | keystrokes saved, safety   |
                                  |              +-------------+--------------+
                                  |                  pass gate v
                                  |              +----------------------------+
                                  |              | Model registry -> shadow   |
                                  |              | -> canary -> A/B -> ramp   |
                                  v              +-------------+--------------+
                        +---------------------------------------+
                        | Serving: index shards + ranker        |
                        +-------------------+-------------------+
                                            | new impressions (biased by
                                            | current ranker: position bias)
                                            v
                                  back to Search logs (feedback loop)
```

## Diagram 3: Model architecture

**DIAGRAM 3: Two-stage QAC model, with LTR ranker and optional neural generator ([5][6][7][9])**

```
 INPUTS
+---------------+ +----------------+ +---------------+ +--------------------+
| Prefix p      | | Candidate q    | | User u        | | Context c          |
| chars, length | | text, lang     | | long-term hist| | session queries,   |
|               | |                | | geo, device   | | time, locale, app  |
+-------+-------+ +-------+--------+ +-------+-------+ +---------+----------+
        |                 |                  |                   |
        v                 v                  v                   v
+--------------------------------------------------------------------------+
| FEATURE LAYER                                                            |
|  popularity: freq(q), freq(q | p), time-decayed and trending velocity    |
|  match: |q|-|p|, word-boundary match, edit distance (typo prefix)        |
|  personal: u issued q before? recency; sim(q, u history)                 |
|  session: sim(q, previous queries) via embeddings (CLSM / BERT) [6][7]   |
|  quality: CTR(q | p, position), results count, downstream success        |
+-------------------------------------+------------------------------------+
                                      v
          +-----------------------------------------------------+
          | RANKER                                              |
          |  baseline: MostPopularCompletion (MPC)              |
          |  v1: GBDT LambdaMART, listwise nDCG / MRR           |
          |  v2: small DNN with text towers (prefix+ctx, query) |
          |       [ctx tower]--emb--+                           |
          |                         +--> dot / MLP --> score    |
          |       [query tower]-emb-+                           |
          +--------------------------+--------------------------+
                                     v
                    OUTPUT: ranked top-K completions + scores

 OPTIONAL GENERATOR (open-vocab, rare/unseen prefixes):
   prefix + context --> char/subword LM (LSTM or small Transformer)
                    --> beam search (width B) --> new candidates --> ranker
```

## Diagram 4: Real-system variants

**DIAGRAM 4a: Amazon session-aware QAC as extreme multi-label ranking [7]**

```
+------------------+   +---------------------+   +---------------------------+
| prefix +         |-->| text features (tf-  |-->| XMR tree over ~millions of|
| previous query   |   | idf n-grams/embeds) |   | label queries: beam down  |
| in session       |   +---------------------+   | clusters -> leaf rankers  |
+------------------+                             +-------------+-------------+
                                                               v
                                    top-K labels (queries), filtered by prefix
```

**DIAGRAM 4b: LinkedIn typeahead (Cleo) multi-source fan-out [8] [14]**

```
                 +---------------------------+
   prefix ------>| Typeahead broker          |
                 +---+----------+----------+-+
                     |          |          |   fan out in parallel
                     v          v          v
              +---------+ +----------+ +----------+
              | Network | | Generic  | | Entity   |
              | (1st/2nd| | (all     | | (company,|
              | degree) | | members) | | groups)  |
              +----+----+ +-----+----+ +-----+----+
                   |  bloom-filter prefix match, forward index
                   +------------+------------+
                                v
                    merge + rank (social + static score) --> UI
```

## What production systems add beyond the portal answers

| System | Addition often missed in interviews | Source |
|---|---|---|
| Bing (Shokouhi 2013) | Supervised personal ranker with a labeling strategy for offline data. Long-term history and location matter most. Up to 9% MRR gain over popularity | [5] |
| Bing (Mitra & Craswell 2015) | Synthetic candidates from mined query suffixes for unseen prefixes, ranked with n-gram and CLSM features | [6] |
| Amazon (Yadav et al. 2021) | Previous query in the session as input to an XMR model that retrieves completions directly | [7] |
| LinkedIn Cleo | Partial and out-of-order token matching, real-time index updates, one library across many entity types | [8] [14] |
| Gmail Smart Compose | Neural LM completion under a strict per-keystroke latency budget, with safety filtering of generated text | [9] [10] |

## Data & labels

| Item | Choice | Note |
|---|---|---|
| Unit of example | (prefix, candidate, user, context) at each keystroke impression | [5] |
| Positive | Shown completion the user selected | Selection is the direct signal |
| Weak positive | Final submitted query that starts with the prefix but was not shown | Recovers recall when the user kept typing [5] |
| Negative | Shown, not selected | Position bias: down-weight or use IPS |
| Label weight | Boost when the downstream search succeeded (click, purchase) | [11] (assumption) |
| Cold data | Synthetic prefixes from full queries (all prefixes of q) | Standard bootstrap |
| Public benchmark | AOL query log (used with Bing logs in Shokouhi 2013) | [5] |

## Features

| Group | Examples |
|---|---|
| Popularity | Global and per-geo frequency, time-decayed count, hourly trend velocity |
| Prefix-query | Completion length, prefix ends mid-word, fuzzy-match distance, out-of-order token match [14] |
| Suffix / semantic | N-gram suffix frequency, CLSM similarity of prefix and candidate (rare prefixes) [6] |
| Personal | Past issuance of q by user, recency, long-term history and location (the strongest personal signals in [5]) |
| Session | Sim to previous queries in session, reformulation patterns [7] |
| Quality | Historical CTR at position, zero-result rate, downstream conversion [11] |
| Context | Time of day, device, locale, surface (web, app, voice) |

## Model choices

| Stage | Option | Pros | Cons |
|---|---|---|---|
| Baseline | MostPopularCompletion from a trie | Simple, under 1 ms | Not personal, fails on rare prefixes |
| Retrieval | Trie/FST with precomputed top-N per node [2] | O(prefix len) lookup | Fixed vocab, rebuild cost |
| Retrieval | Suffix synthesis for rare prefixes [6] | Covers the long tail | Can produce nonsense |
| Retrieval | XMR / extreme classification [7] | Uses session context | Training complexity |
| Ranker | GBDT LambdaMART | Strong on tabular features, cheap | Weak on raw text |
| Ranker | Two-tower / cross DNN | Semantic and session aware | Latency, needs GPU or quantization |
| Generator | Char/subword LM + beam [9][10] | Open vocabulary | Safety risk, ms per token |

## Loss

| Model | Loss |
|---|---|
| LTR ranker | Pairwise (RankNet) or LambdaRank/LambdaMART optimizing nDCG / MRR |
| Pointwise alt | Binary cross-entropy on selected vs shown, IPS-weighted |
| Generator | Token-level cross-entropy (next-char / next-token) |
| Safety filter | Binary or multi-label cross-entropy |

## Offline metrics

| Metric | Meaning |
|---|---|
| MRR | Reciprocal rank of the eventually-submitted query [5] |
| Success@K / Recall@K | Target query appears in the top K |
| pSaved / eSaved | Probability-weighted keystrokes saved |
| nDCG@K | Graded by downstream success |
| Safety rate | Percent of prefixes that produce an offensive completion (red-team set) |
| Coverage | Percent of prefixes with at least 1 suggestion (rare-prefix tail) [6] |

## Online metrics / A/B

| Metric | Type |
|---|---|
| Suggestion CTR / acceptance rate | Primary |
| Keystrokes per search, time to search | Primary (effort) |
| Downstream search success (click, purchase), reformulation rate | North star |
| Zero-result searches | Guardrail |
| p99 latency, error rate | Guardrail |
| Offensive-suggestion reports | Safety guardrail |

## Serving & latency budget

| Step | Budget (assumption) |
|---|---|
| Client debounce + network | 30-50 ms |
| Edge/prefix cache lookup | 1-2 ms (high hit rate for short prefixes) |
| Candidate gen (trie + trending + personal, parallel) | 5-10 ms |
| Filtering | 1-2 ms |
| Feature fetch | 3-5 ms |
| Ranker on ~200 candidates | 5-15 ms (GBDT or a distilled DNN) |
| Re-rank / policy | 1 ms |
| Total server | Under 40 ms. End-to-end under 100 ms [2] |

## Monitoring

| What | Signal |
|---|---|
| Data | Log volume per keystroke, label join rate, feature null rate |
| Model | Score distribution drift, MRR on fresh holdout, CTR by position |
| Freshness | Lag from trending event to suggestion |
| Safety | Blocklist hits, user reports, sensitive-entity completions |
| System | Cache hit rate, p50/p99 latency, shard hotness |

## Failure modes

| Failure | Mitigation |
|---|---|
| Offensive, defamatory or PII completions | Blocklists, classifier, human escalation, takedown pipeline |
| Rich-get-richer / position-bias loop | IPS, randomized exploration slots, counterfactual LTR |
| Rare or new prefixes return nothing | Suffix synthesis [6], neural generator, fuzzy match |
| Trending spikes missed | Streaming counts with time decay |
| Manipulation (query spam to push a completion) | Unique-user thresholds, anomaly detection |
| Latency spikes on 1-char prefixes | Precompute and cache short prefixes at the edge |
| Personal leakage on shared devices | Per-account history, opt-out, no cross-user PII |

## Interviewer follow-ups

- How do you generate labels when the user never clicks a suggestion and keeps typing?
- How do you correct for position bias in the selection logs?
- How do you handle a prefix never seen before (long tail)? [6]
- Trie with precomputed top-N vs on-the-fly ranking: what goes where? [2]
- How do you surface a breaking-news query within 5 minutes?
- How would you personalize without hurting the cache hit rate? [5]
- How do you use the previous query in the session? [7]
- How do you prevent offensive or legally risky completions?
- Would you use an LLM to generate completions? What are the latency and safety costs? [9]
- What offline metric best predicts online keystrokes saved?
- How do you handle typos in the prefix ("iphnoe")?
- How do you match out-of-order tokens ("york new" to "new york")? [14]
- Personalized ranker vs popularity: what gain is realistic? (Up to 9% MRR on Bing logs) [5]
- How do you support multiple languages and scripts (CJK input methods)?
- How do you stop adversaries from gaming the suggestions?
- How do you shard the index and keep it consistent during rebuilds?

## Portal comparison

| Portal | Framing | Core solution | ML depth |
|---|---|---|---|
| ByteByteGo Vol. 1 [2] | Infra system design | Trie with cached top-K per node, data-gathering service, weekly rebuild, sharding | Popularity only |
| DesignGurus / Grokking [13] | Infra system design | Trie, offline aggregation, partitioning, client-side tricks | Popularity only |
| Chip Huyen exercises [3] | ML exercise | Open-ended: related searches and sentence completion | Expects LM + ranking |
| alirezadir [1] | NLP MLSD question | Use the 6-step MLSD template | Ranking + NLP |
| Research / company blogs [5][6][7][11] | Production QAC | Candidate generation + LTR ranker, personalization, session context | Deep (LTR, XMR, embeddings) |

## References

1. alirezadir, "ML System Design" (sample questions: Autocompletion / typeahead), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
2. A. Xu, "Design a Search Autocomplete System" (System Design Interview Vol. 1, Ch. 13), ByteByteGo, 2020. https://bytebytego.com/courses/system-design-interview/design-a-search-autocomplete-system
3. C. Huyen, "machine-learning-systems-design" exercises (#5 related searches, #10 autocompletion), GitHub, 2019. https://raw.githubusercontent.com/chiphuyen/machine-learning-systems-design/master/content/exercises.md
4. C. Huyen, "machine-learning-systems-design" booklet, GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design
5. M. Shokouhi, "Learning to Personalize Query Auto-Completion", SIGIR / Microsoft Research, 2013. https://www.microsoft.com/en-us/research/publication/learning-to-personalize-query-auto-completion/
6. B. Mitra & N. Craswell, "Query Auto-Completion for Rare Prefixes", CIKM / Microsoft Research, 2015. https://www.microsoft.com/en-us/research/publication/query-auto-completion-for-rare-prefixes/
7. N. Yadav et al., "Session-aware Query Auto-completion using Extreme Multi-label Ranking", KDD / arXiv, 2021. https://arxiv.org/abs/2012.07654
8. LinkedIn Engineering, "Cleo: the open source technology behind LinkedIn's typeahead search", linkedin.com, 2012. https://engineering.linkedin.com/open-source/cleo-open-source-technology-behind-linkedins-typeahead-search
9. Google AI Blog, "Smart Compose: Using Neural Networks to Help Write Emails", googleblog.com, 2018. https://ai.googleblog.com/2018/05/smart-compose-using-neural-networks-to.html
10. M. X. Chen et al., "Gmail Smart Compose: Real-Time Assisted Writing", KDD / arXiv, 2019. https://arxiv.org/abs/1906.00080
11. Instacart Tech, "How Instacart Uses Machine Learning-Driven Autocomplete to Help People Fill Their Carts", tech.instacart.com, 2022. https://tech.instacart.com/how-instacart-uses-machine-learning-driven-autocomplete-to-help-people-fill-their-carts-9bc56d22bafb
12. Algolia, "Feature Spotlight: Query Suggestions", algolia.com, 2023. https://www.algolia.com/blog/product/feature-spotlight-query-suggestions/
13. DesignGurus, "The Grokking series" (includes Designing Typeahead Suggestion), designgurus.io, 2025. https://www.designgurus.io/grokking
14. LinkedIn, "Cleo: partial, out-of-order and real-time typeahead search" (source repository), GitHub, 2012. https://github.com/linkedin/cleo
15. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies
