# 20. Design an ML-Driven Push Notification / Email System (What, When, How Often)

> Prompt as usually given: "We send push notifications and emails to re-engage users.
> Design an ML system that decides which notification to send, when to send it, and how
> many to send per user, without driving users to disable notifications or unsubscribe."
> Variants: "Rank candidate notifications for a social app", "LinkedIn notification
> volume optimization", "Duolingo practice reminders", "Email digest personalization",
> "Twitter/X push recommendations".

> Verification note: in this run the X/Twitter pushservice READMEs [6][7][8][9] and the
> alirezadir template [11] were fetched and checked. Web search was exhausted and most
> other hosts did not resolve, so papers [1]-[4] and portal pages marked (not re-fetched)
> rest on well-known publications or on links verified in earlier chapters (ch-ref).
> Paper entries point to a Google Scholar title search, so no URL is guessed.

## Company tags

| Company | Basis | Production system it mirrors |
|---|---|---|
| LinkedIn | Public notification-optimization papers [1][2] (not re-fetched) | Multi-objective volume optimization of activity notifications; survival model of user response state [1][2] |
| Pinterest | KDD 2018 paper on volume control [3] (not re-fetched) | Per-user weekly volume as the decision; models of long-term activity vs volume [3] |
| Duolingo | KDD 2020 bandit paper [4] (not re-fetched) | Picks a practice-reminder template with a "sleeping, recovering" bandit [4] |
| X / Twitter | Open-source release blog [5]; pushservice code fetched this run [6][7][8][9] | Target eligibility, candidate sources, light filter, light ranker, multi-task heavy ranker (ClemNet), take step, send [7][8][9] |

Meta and "generic big tech" tags were removed: no source fetched in this run or in
earlier chapters shows notification ranking as a reported interview question there.

**Portals that cover it, or cover the building blocks**

| Portal | What it gives you | Link |
|---|---|---|
| ByteByteGo System Design Interview Vol. 1, "Design a Notification System" | Delivery infrastructure (notification servers, per-channel queues, workers, APNs/FCM/SMS/email, retries, templates, settings, rate limits, event tracking). No ML | [12] (not re-fetched) |
| ByteByteGo ML System Design Interview (feed / event recsys chapters) | The multi-stage ranking template reused here | [13][14] (ch-ref) |
| Hello Interview, recsys pattern | Retrieval -> ranking -> re-ranking pattern | [15] (ch-ref) |
| alirezadir ML-Interviews | 9-step MLSD formula (problem, metrics, architecture, data, features, model, serving, online testing, scaling/monitoring). No notification question in its list | [11] (fetched) |
| Chip Huyen, "Designing ML Systems" | Feedback loops, delayed labels, continual learning | [16] |
| Patrick Halina, MLSD guide | Interview delivery framework and metric discussion | [17] (ch-ref) |
| Exponent ML system design course | Framework and question bank | [10] (ch-ref) |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | For each user and time slot, decide whether to send, which item/template, which channel (push, email, in-app) and when | [1][3][4] |
| Business goal | Long-term engagement (DAU/sessions) minus the cost of fatigue (disables, unsubscribes, churn) | [1][3] |
| Notification types | Activity (likes, comments, connections), recommendation (content, people), reminder (streak, practice), digest (email) | [1][4][7] |
| Two entry points | "Refresh" (should this user get a recommendation push now, pick best of many) vs "send" (a given candidate, e.g. a reply, is checked and delivered) | [7] |
| Users | 100M-1B notifiable users | typical assumption |
| Candidates | 10-100 eligible candidates per user per day; most are dropped | typical assumption |
| Decision QPS | Batch: ~1B users scored per day. Real-time events: ~50k-200k per second at peak | typical assumption |
| Latency | Real-time activity push: under 1 s for the decision, seconds end to end. Batch digests: hours | typical |
| Volume caps | Hard per-user caps (e.g. N per day / per week) plus quiet hours and local time zone | [3] + typical |
| Non-functional | Respect opt-outs and OS permission; dedupe across devices; privacy (GDPR, CAN-SPAM); idempotent delivery | [12] + typical |

---

## Diagram 1 - Online serving path

**Figure 20.1 - Target eligibility, aggregation, light filtering, two-stage ranking, a
volume policy with a take step, then scheduling and delivery. Mirrors X pushservice [7]
plus the volume policy of [1][3].**

```
+----------------+  +----------------+  +----------------+  +-----------------+
| Activity events|  | Scheduled jobs |  | Recsys outputs |  | Reminder rules  |
| like, comment, |  | daily digest,  |  | new content,   |  | streak at risk, |
| connect (Kafka)|  | weekly recap   |  | PYMK, trending |  | cart abandoned  |
+-------+--------+  +-------+--------+  +-------+--------+  +--------+--------+
        |                   |                   |                    |
        +---------+---------+---------+---------+--------------------+
                  | raw candidates (user_id, item, type, ts)
                  v
     +--------------------------------+      +--------------------------+
     | 0. TARGET ELIGIBILITY          |<-----| Push perm, opt-outs,     |
     |  user notifiable now? (perm,   |      | quiet hours, sends today |
     |  quiet hours, daily cap left)  |      +--------------------------+
     +---------------+----------------+
                     | eligible users only
                     v
     +--------------------------------+      +--------------------------+
     | 1. CANDIDATE AGGREGATOR        |<-----| Per-user pending buffer  |
     |  group by user, dedupe, merge  |      | (Redis, TTL = staleness) |
     |  ("5 people liked your post")  |      +--------------------------+
     +---------------+----------------+
                     | ~10-100 candidates / user / day
                     v
     +--------------------------------+      +--------------------------+
     | 2. LIGHT FILTER (pre-rank)     |<-----| Type prefs, block lists, |
     |  type muted, seen in-app,      |      | seen-item log, OS token  |
     |  expired, blocked actor        |      | registry                 |
     +---------------+----------------+      +--------------------------+
                     | eligible candidates
                     v
+---------------+    +--------------------------------+    +--------------------+
| Feature store |--->| 3. RANKING                     |<---| Model registry     |
| user: open    |    |  light ranker prunes, then     |    | (versioned, A/B)   |
| rate, last    |    |  multi-task heavy ranker:      |    +--------------------+
| active, tz    |    |  P(open), P(session|open),     |
| item: actor   |    |  P(disable|send), P(unsub)     |
| affinity, age |    +---------------+----------------+
+---------------+                    | scores per candidate x channel
                                     v
     +----------------------------------------------------------------+
     | 4. VOLUME / POLICY OPTIMIZER + TAKE STEP                       |
     |  utility = w1*P(open)*value - w2*P(disable) - w3*fatigue       |
     |  send iff utility > threshold_u (per user, from LP dual)       |
     |  walk ranked list; heavy filters until one passes; caps, mix   |
     +------------------------------+---------------------------------+
                                    | send decisions
                                    v
     +--------------------------------+      +--------------------------+
     | 5. TIMING / SCHEDULER          |<-----| Send-time model:         |
     |  send now (urgent activity) or |      | P(open | hour-of-week)   |
     |  delay to best slot in tz      |      | per user                 |
     +---------------+----------------+      +--------------------------+
                     | (user, payload, channel, send_at)
                     v
     +--------------------------------+  APNs / FCM / SMTP   +-----------+
     | Delivery service (queues,      |--------------------->| Device /  |
     | retries, rate limits, idempot.)|<---------------------| inbox     |
     +---------------+----------------+ delivered / bounced  +-----+-----+
                     | send log (decision id, scores, model ver)   | open,
                     v                                             | dismiss,
     +--------------------------------+                            | disable,
     | Event logger (Kafka)           |<---------------------------+ unsub
     +--------------------------------+
```

## Diagram 2 - Offline training & data pipeline

**Figure 20.2 - Labels are joined to send logs by decision id. Open labels close within
hours. Disable and churn labels need days, so they use a longer window and a separate
head.**

```
+--------------+  +--------------+  +--------------+  +--------------+
| Send log     |  | Delivery     |  | Engagement   |  | Negative     |
| (decision id,|  | receipts     |  | open, click, |  | dismiss,     |
| features,    |  | (APNs/FCM/   |  | session after|  | disable push,|
| model ver,   |  | SMTP bounce) |  | open         |  | unsubscribe, |
| propensity)  |  |              |  |              |  | uninstall    |
+------+-------+  +------+-------+  +------+-------+  +------+-------+
       |                 |                 |                 |
       +--------+--------+--------+--------+--------+--------+
                | join on decision_id within label windows
                v
   +----------------------------+     +-------------------------------+
   | LABEL BUILDER              |     | FEATURE PIPELINES             |
   |  y_open: open in 24 h      |     |  batch: open rate 7/28 d,     |
   |  y_sess: session >= 1 min  |     |   hour-of-week histogram,     |
   |  y_dis: disable in 7 d     |     |   actor affinity, tenure      |
   |  y_unsub: unsub in 7 d     |     |  stream: sends today, last    |
   |  holdout uplift labels     |     |   open ts, active in app now  |
   +-------------+--------------+     +---------------+---------------+
                 |                                    | point-in-time
                 +-----------------+------------------+
                                   v
                 +-----------------------------------+
                 | TRAINING DATASET                  |
                 |  time-split; IPS weights from     |
                 |  logged propensity; downsample    |
                 |  non-opens, recalibrate           |
                 +-----------------+-----------------+
                                   v
   +------------------+   +---------------------+   +----------------------+
   | Train rankers    |-->| Offline eval        |-->| Simulate volume      |
   | light: small MLP |   | AUC / log loss per  |   | policy on logs:      |
   | heavy: multi-task|   | head, calibration,  |   | opens vs sends curve,|
   | NN (or GBDT/head)|   | slice by type, tz   |   | off-policy estimate  |
   +------------------+   +---------------------+   +----------+-----------+
                                                               v
   +------------------+   +---------------------+   +----------------------+
   | Online A/B with  |<--| Shadow / canary     |<--| Model registry +     |
   | long holdout     |   | (log-only scoring)  |   | policy config (w, t) |
   +--------+---------+   +---------------------+   +----------------------+
            |
            +--> results update weights w1..w3 and thresholds; new logs
                 feed back into training (feedback loop: the model only
                 sees outcomes of what it chose to send, so keep exploration)
```

## Diagram 3 - Model architecture

**Figure 20.3 - Multi-task heavy ranker. Shared bottom with one head per outcome. A
separate send-time head scores hour-of-week slots. Head outputs are combined into a
utility. X's push heavy ranker is likewise multi-task (open and engagement) [8].**

```
 user features         candidate features        context features
 (tenure, open rate,   (type, actor affinity,    (local hour, day,
  sends last 24 h,      item age, text emb,       device, last app
  user embedding)       channel)                  session gap)
       |                      |                        |
       +----------+-----------+-----------+------------+
                  v                       v
        +-------------------+   +----------------------+
        | Embedding lookups |   | Dense normalization  |
        | (ids, type, chan) |   | (log counts, rates)  |
        +---------+---------+   +----------+-----------+
                  |                        |
                  +-----------+------------+
                              v
              +-------------------------------+
              | Shared bottom: MMoE / MLP     |
              | (feature crosses: DCN layer)  |
              +--+--------+--------+-------+--+
                 |        |        |       |
                 v        v        v       v
          +--------+ +--------+ +-------+ +----------------+
          | P(open)| | P(sess | | P(dis-| | P(open | slot) |
          |        | | given  | | able) | | 168 hour-of-wk |
          |        | | open)  | |       | | slots          |
          +---+----+ +---+----+ +---+---+ +-------+--------+
              |          |          |             |
              +-----+----+----------+             v
                    v                     best send time
     +--------------------------------+   (argmax, or now if
     | utility = P(open)*V(sess)      |    the item is urgent)
     |   - lambda * P(disable)        |
     |   - mu * fatigue(sends_recent) |
     +---------------+----------------+
                     v
           send if utility > t_u
```

## Diagram 4 - Real-system variants

**Figure 20.4 - Three production patterns: LinkedIn/Pinterest volume optimization, the
Duolingo bandit, and the X/Twitter pushservice funnel.**

```
(a) Volume optimization (LinkedIn [1], Pinterest [3])
    maximize  sum_i sessions_i(sends)   s.t.  sum_i disables_i(sends) <= C
    +-------------+   +----------------------+   +----------------------+
    | Per-user    |-->| Lagrangian dual:     |-->| Per-user threshold   |
    | response    |   | lambda found offline |   | or weekly volume k_u |
    | models      |   | by sweeping on logs  |   | pushed to serving    |
    +-------------+   +----------------------+   +----------------------+

(b) Sleeping, recovering bandit (Duolingo [4])
    +-------------+   +----------------------+   +----------------------+
    | Templates   |-->| Eligible arms only   |-->| Score = base reward  |
    | (arms)      |   | ("sleeping" arms are |   | - recency penalty if |
    |             |   |  ineligible today)   |   | arm used recently    |
    +-------------+   +----------------------+   +----------+-----------+
                                                            | softmax pick
                                                            v
                                                   reward = return within 2 h

(c) X / Twitter pushservice, RefreshForPushHandler ([6][7][8][9])
    +-------------+   +-------------+   +-------------+   +-------------+
    | Target      |-->| Candidate   |-->| Light filter|-->| Light ranker|
    | eligibility |   | sources +   |   | (cheap RPC  |   | (MLP, prunes|
    | filters     |   | hydration   |   |  checks)    |   |  pool)      |
    +-------------+   +-------------+   +-------------+   +------+------+
                                                                 |
    +-------------+   +-------------+   +-------------+          |
    | Send: push  |<--| Take step:  |<--| Heavy ranker|<---------+
    | + in-app    |   | heavy filter|   | (ClemNet,   |
    | notification|   | top-1 by 1  |   | multi-task: |
    |             |   | until pass  |   | P(open),    |
    |             |   |             |   | P(engage))  |
    +-------------+   +-------------+   +-------------+
```

---

## Data & labels

| Label | Definition | Window | Notes |
|---|---|---|---|
| Open / click | Notification tapped or email clicked | 1-24 h | Main positive. Apple Mail privacy proxy inflates email "opens", so prefer clicks |
| Engagement after open | Session of at least X sec, or a downstream action (like, reply) | 1-24 h | Fights clickbait templates; X's heavy ranker predicts open and engagement [8] |
| Dismiss | Swiped away without opening | 24 h | Weak negative |
| Disable / unsubscribe | Turned off push for app or type; email unsubscribe or spam report | 7 d | Rare, costly; separate head, class weighting |
| Uninstall / churn | App uninstall, 28-day inactivity | 7-28 d | Usually a guardrail metric, not a training label |
| Time to response | Time until user visits after a send (censored if never) | 1-7 d | Survival-style label used for timing [2] |
| Incremental lift | Activity(sent) minus activity(held out) | 1-7 d | Needs randomized holdout; for uplift models [18] |

## Features

| Group | Examples |
|---|---|
| User | Tenure, DAU/WAU state, push/email open rate 7/28 d, last active ts, time zone, OS, permission state |
| Fatigue | Sends in last 1/24/168 h, time since last send, consecutive unopened sends, type mix sent recently |
| Candidate | Type, actor-to-user affinity (messages, profile views), item age, item popularity, text/template embedding |
| Timing | Local hour/day, hour-of-week open histogram, currently active in app (then skip push, show in-app) |
| Channel | Push vs email vs in-app, device count, email domain |
| Cross | User x type open rate, user x template historical reward (bandit stats) |

## Model choices

| Stage | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Send / no-send | Fixed caps + rules | Multi-objective utility with per-user threshold from a Lagrangian [1][3] | Better long-term balance; needs reliable P(disable) |
| Candidate pruning | Heuristic top-N per source | Light ranker (small MLP) before the heavy ranker [9] | Cuts heavy-ranker cost; may drop good candidates |
| What (heavy ranking) | LR / GBDT on P(open) | Multi-task NN with open, engagement and disable heads (X uses ClemNet [8]; MMoE is a common choice) | More capacity; harder calibration |
| Final pick | Take top-1 | Take step: walk ranked list, apply heavy filters until one passes [7] | Expensive checks only on a few candidates |
| Template / copy | A/B test of templates | Contextual or sleeping-recovering bandit [4] | Adapts per user; needs exploration traffic |
| When | Send at a fixed hour | Per-user hour-of-week P(open) head, or survival model of response time [2]; delay queue | Delayed sends can arrive stale for activity |
| How often | Global cap | Per-user volume k_u by response curve or uplift model [3][18] | Uplift avoids credit for users who would return anyway |
| Email digest | Top-K by score | Listwise re-rank with diversity, subject-line generation (LLM) | LLM copy needs safety review and cost control |

## Loss

| Head | Loss | Note |
|---|---|---|
| P(open), P(engage) | Binary cross-entropy, IPS-weighted by logged propensity | Corrects selection bias: only sent items have labels |
| P(disable), P(unsub) | Weighted BCE or focal loss | Very rare positives |
| Send time | Softmax CE over 168 slots, BCE per slot, or survival likelihood [2] | Only sent slots are observed; smooth with priors |
| Total | sum_k alpha_k * L_k | Tune alphas; MMoE reduces task conflict |
| Bandit | Reward = return within window; softmax with recency penalty [4] | Online, no gradient training |

## Offline metrics

| Metric | Use |
|---|---|
| ROC-AUC / PR-AUC per head | Discrimination; PR-AUC for disable head |
| Log loss + calibration (ECE, reliability plot) | Utility uses raw probabilities, so calibration matters |
| Light-ranker recall@K of heavy-ranker top-1 | Checks pruning does not drop the winner [9] |
| Opens-vs-sends curve on logs | Simulates volume policy; pick operating point |
| Off-policy estimate (IPS / doubly robust) | Estimate new policy value before A/B |
| Sliced metrics | By notification type, tenure, time zone, platform |

## Online metrics / A-B

| Type | Metric |
|---|---|
| Primary | Sessions or DAU per user; incremental vs holdout [1] |
| Engagement | Open rate (CTR), click-through to session, time to open |
| Guardrails | Push disable rate, email unsubscribe and spam-report rate, uninstalls, sends per user |
| Experiment design | User-level randomization; long test (2-4 weeks) because fatigue builds slowly; long-term holdout group that receives no ML notifications |
| Pitfall | Higher volume always lifts short-term opens; judge on long-term DAU and disables |

## Serving & latency budget

| Step | Budget (real-time push) | Batch digest |
|---|---|---|
| Event ingest + aggregation window | 0.1-60 s (wait to batch "5 people liked") | n/a |
| Target eligibility + light filter + feature fetch | 20-30 ms | Bulk read |
| Light ranker + heavy ranker | 20-40 ms for under 100 candidates | Spark / batch inference over all users |
| Volume policy + take step + scheduling | Under 10 ms | Minutes |
| Delivery (APNs/FCM/SMTP) | Seconds (third party) | Spread over hours by time zone |

## Monitoring

| Signal | Alert on |
|---|---|
| Sends per user per day distribution | Spikes (bug sends 50 pushes) |
| Open rate, disable rate by type | Sudden drops or rises |
| Prediction drift | Mean predicted P(open) vs observed |
| Delivery health | APNs/FCM error rate, email bounce, spam-folder rate |
| Feature freshness | Stale "last active" feature causes pushes to users already in app |
| Dedupe | Same notification sent twice across devices or channels |

## Failure modes

| Failure | Mitigation |
|---|---|
| Notification fatigue: short-term opens up, long-term disables up | Disable head, per-user caps, long holdouts |
| Feedback loop: model only learns from what it sent | Epsilon exploration, propensity logging, IPS |
| Credit for users who would return anyway | Uplift model and randomized holdout [18] |
| Stale activity (a like from 3 days ago) | TTL on candidates; urgency-aware scheduler |
| Clickbait templates | Optimize engagement after open, not open alone |
| Thundering herd at 9:00 local time | Jitter send times; rate-limit delivery queues |
| Cold-start users | Priors by cohort; bandit exploration |
| Privacy / compliance | Respect consent, quiet hours, unsubscribe at once |

## Interviewer follow-ups

- How do you decide whether to send at all, not just which item to send?
- How do you trade off opens against disables and unsubscribes? How do you pick lambda? [1][3]
- How do you measure the incremental value of a notification vs a user who would return anyway? [18]
- How do you fix selection bias when labels exist only for sent notifications?
- How do you choose send time per user? What happens to time-sensitive activity? [2]
- How do you aggregate many small events into one notification?
- How do you pick among push, email and in-app?
- Why split a light ranker from a heavy ranker, and why run expensive filters only after ranking? [7][9]
- Why are disable labels hard, and how do you handle the long label delay?
- How would you use a bandit to choose the template, and why add a recency penalty? [4]
- Which online metric is primary, and why run the A/B test for weeks?
- How do you stop a bug from sending 50 pushes to one user?
- How do you handle new users with no history?
- Would you generate notification copy with an LLM? What are the risks?
- How do notifications interact with the feed ranker (do not notify about what the user just saw)?

## Portal comparison

| Source | Framing | Core solution | ML depth |
|---|---|---|---|
| ByteByteGo SDI Vol. 1 [12] | Infra system design | Notification servers, message queues, workers, APNs/FCM/SMTP, retries, opt-out settings, rate limits | None |
| ByteByteGo MLSDI feed/event chapters [13][14] | ML recsys | Multi-stage funnel and pointwise ranker; reused here as the "what" component | Ranking |
| Hello Interview / alirezadir / Exponent [15][11][10] | Framework | Generic retrieval -> rank -> re-rank template and 9-step MLSD formula | Template only |
| Chip Huyen [16] | ML systems | Delayed labels, feedback loops, continual learning, monitoring | Concepts |
| LinkedIn / Pinterest papers [1][2][3] | Production | Multi-objective volume optimization with constraints; per-user thresholds; survival timing | Deep (optimization) |
| Duolingo paper [4] | Production | Sleeping, recovering bandit for template choice | Deep (bandits) |
| X pushservice [6][7][8][9] | Production (open source) | Eligibility, light filter, light ranker, multi-task heavy ranker, take step | Deep (MTL ranking) |

## References

1. Y. Gao et al. (LinkedIn), "Near Real-time Optimization of Activity-based Notifications", KDD, 2018. Scholar search: https://scholar.google.com/scholar?q=%22Near+Real-time+Optimization+of+Activity-based+Notifications%22 (not re-fetched)
2. Y. Yuan et al. (LinkedIn), "A State Transition Model for Mobile Notifications via Survival Analysis", WSDM, 2019. Scholar search: https://scholar.google.com/scholar?q=%22A+State+Transition+Model+for+Mobile+Notifications+via+Survival+Analysis%22 (not re-fetched)
3. B. Zhao et al. (Pinterest), "Notification Volume Control and Optimization System at Pinterest", KDD, 2018. Scholar search: https://scholar.google.com/scholar?q=%22Notification+Volume+Control+and+Optimization+System+at+Pinterest%22 (not re-fetched)
4. K. P. Yancey & B. Settles (Duolingo), "A Sleeping, Recovering Bandit Algorithm for Optimizing Recurring Notifications", KDD, 2020. https://research.duolingo.com/papers/yancey.kdd20.pdf (host unreachable this run)
5. Twitter Engineering, "Twitter's Recommendation Algorithm", blog.x.com, 2023. https://blog.x.com/engineering/en_us/topics/open-source/2023/twitter-recommendation-algorithm (host unreachable this run)
6. Twitter, "the-algorithm" (README lists pushservice under Recommended Notifications), GitHub, 2023. https://github.com/twitter/the-algorithm (fetched)
7. Twitter, "pushservice" README (RefreshForPushHandler, SendHandler, light filter, take step), GitHub, 2023. https://github.com/twitter/the-algorithm/blob/main/pushservice/README.md (fetched)
8. Twitter, "Notification heavy ranker" (ClemNet, multi-task open and engagement), GitHub, 2023. https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/python/models/heavy_ranking (fetched)
9. Twitter, "Notification light ranker" (pre-selects candidates to cut heavy-ranking cost), GitHub, 2023. https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/python/models/light_ranking (fetched)
10. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
11. alirezadir, "ML System Design" (9-step formula), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
12. A. Xu, "Design a Notification System" (System Design Interview Vol. 1, ch. 10), ByteByteGo, 2020. https://bytebytego.com/courses/system-design-interview/design-a-notification-system (host unreachable this run)
13. A. Aminian & A. Xu, "Personalized News Feed" (Machine Learning System Design Interview, ch. 10), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/personalized-news-feed (ch-ref)
14. A. Aminian & A. Xu, "Event Recommendation System" (Machine Learning System Design Interview, ch. 7), ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/event-recommendation-system (ch-ref)
15. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
16. C. Huyen, "Designing Machine Learning Systems", O'Reilly, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/ (not re-fetched)
17. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
18. P. Gutierrez & J.-Y. Gerardy, "Causal Inference and Uplift Modelling: A Review of the Literature", PMLR 67, 2017. http://proceedings.mlr.press/v67/gutierrez17a.html (host unreachable this run)
