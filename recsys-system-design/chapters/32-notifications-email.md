# 32. Notification, Push and Email Recommendation: Volume, Fatigue and Timing

> Research note for this chapter: the session's shared web-search quota ran out before
> this chapter's research began, and most company blog and paper hosts (linkedin.com,
> medium.com, arxiv.org, research.duolingo.com) did not resolve from the research
> sandbox. The only sources fetched and read in this session are the open-sourced
> X/Twitter `pushservice` code and READMEs [1][2][3][4][5]. Every other section summarizes
> peer-reviewed papers the author already knew. Their headline results are given
> qualitatively or marked "approximately", and their URLs (mostly DOIs and arXiv IDs)
> should be checked again before print. Read those sections as a guide to the
> literature, not as freshly verified numbers.

## Overview

Notification recommendation is the "push" side of recommender systems. A feed ranker
answers "what should this user see now that they are here?" A notification system has to
answer three harder questions together:

1. **Whether** to interrupt the user at all (send or don't send).
2. **What** to send (which candidate item, template or channel).
3. **When** and **how often** to send (timing, volume, fatigue).

These differences shape the architecture:

- **The user is not present.** The system starts the interaction, so there is no request
  context. Systems are either event-driven (a friend posted, a job matched) or
  schedule-driven (a daily digest, a streak reminder).
- **Every action has a hidden cost.** A badly targeted push can lead to a disabled
  notification channel, an email unsubscribe or an app uninstall. These are near-permanent
  losses of the channel. Click-through rate (CTR) maximization alone pushes volume
  upward, and the cost shows up weeks later as churn. That is why most mature systems
  optimize a *long-term* objective, such as sessions or daily/weekly active users
  (DAU/WAU) over a horizon, against a negative-feedback penalty.
- **Strong counterfactual confounding.** Users who get more notifications are already
  more active. Observational CTR data overstates how much notifications cause activity,
  so production systems depend on randomized holdouts, volume-perturbation experiments
  and causal or uplift modeling.
- **Several channels share one attention budget.** Push, in-app badge/inbox ("NTab" in
  X's code), email and SMS all compete for it. A send on one channel changes the response
  on the others.

The chapter covers the open-sourced X (Twitter) push stack in depth (the only system here
whose code was read directly). It then covers the published designs from LinkedIn
(Concourse, Air Traffic Controller, survival-model timing, offline reinforcement
learning), Pinterest (notification volume control), Duolingo (sleeping, recovering
bandits), Twitter's "should I send?" model-the-future work, and the Meta (Facebook and
Instagram) notification setting. It ends with the shared patterns for fatigue modeling and
send-time optimization.

## 32.1 The canonical notification recommendation pipeline

Nearly every published system breaks down into the same stages. They differ mainly in
where the "send or not" decision sits and which objective trains it.

```
 events / schedules                           user state (fatigue, history,
 (new post, connection,                        channel prefs, timezone, opt-outs)
  job, streak, digest timer)                          |
        |                                              v
        v                                     +-----------------+
 +----------------+   candidates   +--------->| target building |
 | candidate gen  |--------------->|          | + eligibility   |
 | (fan-out,      |                |          +-----------------+
 |  graph, ANN,   |                v                   |
 |  trends, rules)|      +------------------+          |
 +----------------+      | hydration +      |<---------+
                         | light filters    |
                         +------------------+
                                  |
                                  v
                         +------------------+   p(open), p(engage),
                         | light ranker     |   p(negative), p(unsub)
                         | heavy ranker     |---------------------+
                         +------------------+                     |
                                  |                               v
                                  v                    +--------------------+
                         +------------------+          | volume / fatigue   |
                         | take step: heavy |<---------| policy, thresholds,|
                         | filters, dedupe, |          | budgets, timing    |
                         | caps, quiet hrs  |          +--------------------+
                         +------------------+
                                  |
                       send now / schedule / drop
                                  v
                    push | in-app inbox | email | SMS
                                  |
                                  v
            delivery + open/click/dismiss/disable logs --> training data
```

The design decisions that separate systems are:

| Decision | Options seen in practice |
|---|---|
| Trigger | event-driven near-real-time (LinkedIn Concourse), scheduled batch (digests), hybrid |
| Send decision | fixed threshold on p(open) or p(engage); per-user threshold from a volume model; RL policy |
| Objective | open/click CTR; sessions or DAU over a horizon; multi-objective with unsub/disable penalty |
| Volume control | global caps; per-user budgets from causal response curves (Pinterest); learned thresholds |
| Timing | send immediately; delay to the predicted best hour; survival/hazard model (LinkedIn) |
| Template choice | A/B tests; multi-armed bandit with novelty decay (Duolingo) |
| Fatigue features | recent send counts, time since last send, recent dismissals, "show less" feedback |

## 32.2 X / Twitter: `pushservice` (open-sourced 2023)

### Problem

X sends recommended pushes. Some are in-network (activity from accounts the user follows).
Many are out-of-network (OON): popular or relevant posts, trends, Spaces, topics and
"magic fanout" events such as news and sports. The service has to decide, per user,
whether any candidate is good enough to deliver as a push plus an in-app notification-tab
entry.

### Architecture (from the open-source code)

The `pushservice` README describes two entry points [2]:

**RefreshForPushHandler** is the recommendation path. For a given user ID it:

1. Builds a *target* object (the recipient) and applies target-level eligibility filters.
2. Queries several candidate sources using the target.
3. Hydrates candidates with batched downstream calls.
4. Applies lightweight, RPC-based "light" pre-rank filters.
5. Hydrates features for candidates and target, then runs a **light ranker** and a
   **heavy ranker**.
6. Runs a **take step**. It walks the ranked list from the top and applies "heavy" filters
   one candidate at a time until **one** candidate passes.
7. Sends that candidate as a push and in-app notification.

**SendHandler** is the direct-send path for a candidate supplied by the caller. It builds
the target, hydrates, runs heavy filtering and validation, and sends. It skips candidate
generation and ranking.

```
 user id --> Target(eligibility) --> candidate sources --> hydrate
        --> light filters --> LIGHT RANKER (MLP) --> HEAVY RANKER (ClemNet MTL)
        --> TAKE: for c in ranked: if heavy_predicates(c): send(c); break
```

The take step is a key design choice. Expensive checks (quality-model predicates, fatigue
and caret-feedback checks, health checks, engagement-ratio checks) run *after* ranking and
only until the first pass. Their cost therefore scales with the rejection rate, not the
candidate count.

### Candidate types

The `model` package lists the push candidate types [5]. They include
`F1FirstdegreeTweet` (in-network posts), `OutOfNetworkTweetPushCandidate`,
`TopTweetImpressionsPushCandidate`, `TrendTweetPushCandidate`, `TripTweetPushCandidate`,
`TopicProofTweetPushCandidate`, `SubscribedSearchTweetPushCandidate`,
`ListRecommendationPushCandidate`, `DiscoverTwitter`, scheduled-Space speaker and
subscriber candidates, and a family of "MagicFanout" candidates (news events, sports,
product launches, creator events). They also include social-proof actions such as
`TweetFavorite` and `TweetRetweet`.

### Ranking models

- **Light ranker.** It sits between candidate generation and heavy ranking and
  "pre-selects highly-relevant candidates from the initial huge candidate pool". The
  model is defined in `model_pools_mlp.py` [3]. The file name suggests a multilayer
  perceptron (MLP).
- **Heavy ranker.** The README calls it a multi-task model that estimates the probability
  that targets will open and engage with sent notifications [1][2]. The architecture is
  **ClemNet** [4]: a stack of blocks, each made of a `ChannelWiseDense` layer, batch norm
  (momentum 0.9999), an activation, then a 1-D convolution, batch norm and activation,
  with an optional residual connection. The output is flattened into a logits layer.
- **Default configuration** (from `params.py`, as read in this session): sparse features
  hashed to 2^18 buckets (`bits = 18`) with embedding size 50; four blocks with dense
  output sizes 1024, 512, 256 and 128; ReLU activations; conv kernel size 3 with 5
  filters; residual off by default.
- **Tasks and score weights.** Two default tasks:
  - `OONC` on `label`, weight 0.9. The acronym is not expanded in the files read; it is
    commonly read as "open or NTab click", which fits the open-focused task description.
  - `Engagement` on `label.engagement`, weight 0.1.

  The final score is therefore a weighted mix that leans heavily toward the open/click
  probability.
- **Negative-feedback label.** The feature config always adds `label.ntabDislike` [4].
  "Dislike" feedback in the notification tab is a first-class training label, the raw
  material for fatigue- and annoyance-aware scoring.

### Predicates: where fatigue and quality live

The `predicate` package [5] shows how much of the "send or not" logic sits in rule and
model predicates outside the ranker:

- **Fatigue:** `FatiguePredicate`, `TargetNtabCaretClickFatiguePredicate`,
  `CaretFeedbackHistoryFilter`, and an `ntab_caret_fatigue` subpackage. The "caret" is the
  per-notification "show less / see fewer like this" menu; recent caret clicks suppress
  similar future pushes.
- **Quality and health:** `BqmlQualityModelPredicates`, `JointDauAndQualityModelPredicate`,
  `OutOfNetworkCandidatesQualityPredicates`, `BqmlHealthModelPredicates`,
  `HealthPredicates`, and `PNegMultimodalPredicates` (a predicted-negative model).
  `JointDauAndQualityModelPredicate` suggests that gating combines a DAU-impact model with
  a quality model rather than relying on CTR alone.
- **Spread and exploration:** `OONSpreadControlPredicate` limits how widely an
  out-of-network post fans out. `BigFilteringEpsilonGreedyExplorationPredicate` adds
  epsilon-greedy exploration to filtering, so the training data includes some sends the
  filters would have blocked. `MlModelsHoldbackExperimentPredicate` keeps an ML holdback
  for causal measurement.
- **Content safety and other gates:** withheld content, language, engagement ratio,
  negative-feedback-based OON filtering, CAS locks (to stop concurrent duplicate sends),
  and decider (feature-flag) gates.

### Lessons

- One candidate per refresh. The take step turns "rank many, send at most one" into the
  basic unit of work.
- Annoyance is modeled explicitly, both as a label (`ntabDislike`) and as stateful
  fatigue predicates.
- Exploration and holdbacks are built into the serving path, because notification
  training data is heavily biased by past send policies.

## 32.3 Twitter: "Should I send this notification?" (modeling the future)

Twitter researchers published work (circa 2022) on framing the send decision as a
sequential decision rather than a one-shot CTR threshold [6]. The core argument is that the
value of sending now depends on what the user will do later. A user who would open the app
anyway gains little from a push. A push sent now may also displace a more valuable one
later or speed up fatigue. The paper models the future with long-horizon user-activity
targets and uses those predictions to set the send policy, reporting online gains over a
myopic open-probability policy. The exact metrics were not re-verified in this session.

Design lesson: train the send/no-send decision on *incremental* long-term engagement
(with send minus without send). Training on the probability of an open conditional on
sending is not enough.

## 32.4 LinkedIn

LinkedIn has published one of the most complete series of notification papers. Its
notifications span activity-based notifications ("X shared a post", "Y mentioned you"),
network updates, job alerts, and email digests.

### 32.4.1 Concourse: near-real-time activity-based notifications

**Problem.** When a member's connection or followed entity creates content, LinkedIn must
decide within minutes which of that creator's followers (possibly millions) should get a
notification. Batch, offline scoring was too stale for timely content.

**Architecture** (from the KDD 2018 paper [7]; details from prior knowledge):

```
 content creation event (Kafka)
          |
          v
 +-------------------+     follower / connection lists
 | fan-out           |<--------------------------------------+
 | (Samza stream     |                                       |
 |  processors)      |---> per-(creator,content,recipient) candidates
 +-------------------+
          |
          v
 +-------------------+  features: recipient profile, affinity to creator,
 | scoring (online   |  content features, notification history, fatigue
 |  logistic/GLMix)  |
 +-------------------+
          |
          v
 +-------------------+     decision events
 | Air Traffic       |----------------------> push / in-app / email channels
 | Controller (ATC): |
 | volume caps,      |
 | dedupe, channel,  |
 | timing            |
 +-------------------+
```

- The fan-out is stream-based (Apache Samza over Kafka). Each (content, recipient) pair is
  scored with a response-prediction model. Following the GLMix/Photon-ML lineage, the
  model has global and per-member components.
- The decision is a threshold on a utility that combines the predicted click/engagement
  probability with penalties for volume and for negative actions such as disables.
- **Air Traffic Controller (ATC)** is the centralized downstream "member-first"
  arbitration layer. It applies frequency caps, merges or deduplicates related
  notifications, picks the channel, and can delay delivery. Every upstream producer
  (activity notifications, jobs, recommendations) goes through ATC, so the global volume
  budget is enforced in one place.

**Lessons.** Separate per-item relevance (in the producers) from per-member volume control
(centralized in ATC). Scoring in near real time made notifications for fresh content more
relevant; the paper reports engagement lifts over the earlier offline pipeline.

### 32.4.2 Volume optimization as constrained multi-objective optimization

LinkedIn's notification and email volume work [7][8] frames the problem as maximizing
sessions/engagement subject to an upper limit on negative outcomes such as unsubscribes,
disables and complaints, or the reverse framing. Each send gets a utility like:

```
 utility(send) = p(click) * w_click + E[delta sessions]
                 - lambda * p(disable or unsubscribe)
 send iff utility(send) > threshold_m        (threshold per member segment)
```

The Lagrangian dual of the constraint gives the multiplier lambda. It and the thresholds
are tuned on randomized volume experiments that send more or fewer notifications to random
groups of members, so the response curve can be estimated causally.

### 32.4.3 Timing: a state-transition / survival model

LinkedIn published a state-transition model of mobile notifications based on survival
analysis (WSDM 2019) [9]. A member moves between states: notification received, visit,
idle, and so on. Hazard models estimate the time to the next visit with and without a
notification. The send decision, including whether to delay a notification, uses the
predicted *change in time to visit*. If the member is likely to visit organically soon,
the notification can be held, folded into the in-app inbox, or dropped. This is one of the
clearest public examples of **send-time optimization as a causal hazard problem**, not a
"best hour of day" lookup.

### 32.4.4 Offline reinforcement learning for notifications

Later LinkedIn work (KDD 2022) [10] treats the notification decision as a Markov decision
process. The state is member activity plus notification history, the actions are
send/no-send (and threshold choices), and the reward is long-term sessions minus negative
feedback. A policy is learned with **offline RL** (a double-DQN-style value-based method)
on logged data, because online exploration on real members is expensive. The paper
reports that the RL policy beat the hand-tuned multi-objective threshold system in online
A/B tests.

Lesson: once the multi-objective threshold system is mature, the next step is to make the
threshold state-dependent and forward-looking. RL does this by learning the value of
"saving" a member's attention for a later, better notification.

## 32.5 Pinterest: notification volume control and optimization

### Problem

Pinterest sends large volumes of email and push: boards and Pins you might like, activity
on your Pins, digests. Its 2018 KDD paper [11] describes a central finding. Optimizing
per-notification CTR, and sending whenever CTR is above a threshold, did not maximize
long-term engagement. More notifications raised short-term activity but also raised
unsubscribes and "notification blindness".

### Design

```
 per-user features (activity level, history, past response)
                 |
                 v
 +-------------------------------------------+
 | volume response model:                    |
 | predict E[weekly activity | k sends/week] |
 | and p(unsubscribe | k sends/week)         |
 +-------------------------------------------+
                 |
                 v
 +-------------------------------------------+
 | optimizer: choose k_u per user maximizing |
 | sum_u activity_u(k_u)                     |
 | s.t. unsubscribe / total-volume budget    |
 +-------------------------------------------+
                 |  weekly budget k_u
                 v
 +-------------------------------------------+
 | per-notification ranker picks the best    |
 | k_u candidates; scheduler spreads them    |
 +-------------------------------------------+
```

- **Model the long-term effect of volume, not single sends.** Pinterest trained models
  that predict a user's activity over a time window as a function of the number of
  notifications they receive. These were trained on data from randomized volume
  experiments in which users were randomly assigned different volume levels.
- **Budget then rank.** A per-user weekly budget is set first by marginal analysis: send
  more to users whose activity curve is still rising and fewer to users who are saturated
  or at risk of unsubscribing. The per-item ranker then fills the budget.
- **Unsubscribe as a hard constraint.** Unsubscribing removes the channel permanently, so
  it enters as a constraint or a heavy penalty, not a small negative weight.
- **Results** (approximate, from prior knowledge): the paper reports gains in weekly active
  users together with *fewer* notifications sent and fewer unsubscribes than the earlier
  threshold system. This is the main reason "send less, better" became industry consensus.

Pinterest has also written in its engineering blog about multi-channel notification
infrastructure and email ranking (for example, digest composition and ranking of Pins
inside emails). Those posts could not be retrieved in this session, so no figures from
them appear here.

## 32.6 Duolingo: sleeping, recovering bandits for practice reminders

### Problem

Duolingo sends a daily practice reminder ("Hi! It's Duo...") to keep learners' streaks
going. The *content*, meaning which of many templates to use, strongly affects whether the
learner comes back. Two properties break standard bandits:

1. **Recovering arms.** A template's effect wears off when it is repeated: novelty decays.
   An arm's reward depends on how recently that user last saw it, and it recovers after a
   rest period.
2. **Sleeping arms.** Not every template is eligible for every user on every day. Some
   apply only to users with a streak, to certain languages, or to certain contexts. The
   available arm set changes from round to round.

### Algorithm

Yancey and Settles (KDD 2020) [12] propose a **recovering difference softmax** bandit:

- Each arm's value is estimated as the *difference* between the reward when the arm is
  chosen and the reward when it is eligible but not chosen. This is a built-in
  counterfactual baseline that removes the effect of eligibility-correlated user
  populations.
- A **recency penalty** lowers an arm's score when the user saw it recently. The penalty
  decays as the time since the last exposure grows, which models recovery.
- Arms are sampled with a **softmax** over the adjusted scores. This gives continual
  exploration and avoids sending the same "best" template every day, which would destroy
  its novelty.
- The reward is whether the learner completes a lesson within a short window, about two
  hours, after the notification.

### Scale and results

The paper describes deployment to Duolingo's whole notification population, then
hundreds of millions of reminders. It reports statistically significant gains in daily
active users and new-user retention over the previous random or round-robin template
policy (gains in the low single-digit percent range, approximate). The authors also
released a large dataset of logged practice-reminder notifications for bandit research.
Duolingo's later public talks and blog posts describe the same bandit infrastructure as
the base for its notification experimentation; those posts could not be fetched in this
session.

### Lessons

- In repeated-message channels the *novelty* of an item is part of its value. Model
  per-user exposure recency explicitly.
- Under eligibility constraints, compare chosen against eligible-but-not-chosen to avoid
  selection bias.
- Softmax/Thompson-style stochastic policies are a feature here, not a compromise, because
  variety itself raises reward.

## 32.7 Meta: Facebook and Instagram notifications

Facebook and Instagram notifications are among the highest-volume recommendation surfaces
in the world, but primary technical sources specific to notifications could not be fetched
in this session. From Meta's public practice and published ranking material (prior
knowledge, unverified here), the general shape is:

- Notifications are ranked by multi-task models predicting click/open, downstream
  engagement after the open, and negative signals (turning off a notification type,
  "hide"). A value model combines these predictions, as in feed ranking.
- Notification candidates (friend activity, group posts, Reels and creator updates,
  "memories", birthdays) come from many producers into a central ranking and filtering
  layer with per-type and global volume caps. This is architecturally the same split as
  LinkedIn's producers-plus-ATC.
- In-app notification tabs (the "jewel" or activity tab) get the same ranking with a
  looser volume constraint, because they do not interrupt the user.
- Meta publishes "system cards" or ranking explainers for some surfaces, which name the
  main prediction targets. Readers should check the current transparency-center pages for
  Facebook and Instagram notifications for the authoritative list.

Treat this section as a structural sketch. The chapter has no verified scale numbers for
Meta notifications.

## 32.8 Fatigue modeling: patterns across systems

Across the designs above, fatigue appears in four complementary forms:

| Mechanism | Example | Strength | Weakness |
|---|---|---|---|
| Hard frequency caps | per-day/per-week caps in ATC [7]; X `FatiguePredicate` [5] | simple, safe, auditable | ignores heterogeneity |
| Fatigue features in ranker | recent sends, time since last open, recent dismissals | personalized | still myopic per send |
| Negative labels / penalties | X `label.ntabDislike` [4]; LinkedIn disable penalty [8] | trains against annoyance directly | rare, noisy labels |
| Volume response curves / RL | Pinterest per-user budgets [11]; LinkedIn offline RL [10] | optimizes long-term causal effect | needs randomized volume data |

Practical recipe:

1. Log *every* decision, including drops and their reasons, so that "not sent" exists in
   the data.
2. Keep a long-running **notification holdback**, a small random population with reduced
   or zero notifications. It is the only way to measure the true incremental value of the
   channel.
3. Run **volume perturbation** experiments (randomized higher or lower thresholds) to
   estimate response curves. This is the training data for budget or RL policies.
4. Treat channel-loss events (push disable, email unsubscribe, uninstall) as constraint
   terms with their own dual multipliers.
5. Add stateful predicates for explicit feedback ("show less", the X caret) that suppress
   similar candidates for a cooling period.

## 32.9 Send-time optimization (STO)

Send-time optimization chooses *when* within a window to deliver. Published approaches,
from simple to sophisticated:

- **Historical best hour.** Per user, pick the hour of day with the highest historical
  open rate, smoothed with segment priors. Widely used in email platforms. Cheap, but
  confounded by past send times.
- **Activity-aligned delivery.** Send shortly before the user's predicted next active
  period. In Duolingo's setting the reminder is tied to the time the learner usually
  practises [12].
- **Hazard / survival models.** LinkedIn's state-transition model [9] predicts time to the
  next visit with and without a notification and sends when the *difference* is largest.
  This is the causally correct version: avoid notifying users who were about to come back
  anyway.
- **Sequential decision.** Twitter's model-the-future approach [6] and LinkedIn's offline
  RL [10] fold timing into a policy that can "wait" as an action.

Timing interacts with volume. Delaying a notification often lets the system batch several
events into one richer notification (digest-ification), which lowers volume and raises
per-notification value. LinkedIn's ATC arbitration and email digests make this trade-off
explicitly [7].

## 32.10 Email recommendation specifics

Email differs from push in several ways that matter to the designer:

- **Composite items.** An email (a digest) contains several recommended items. The
  problem is slate ranking plus layout, and the "send" decision covers the whole slate.
- **Delayed and weak open signals.** Privacy features such as image-proxy prefetching
  inflate measured opens, so clicks and downstream sessions are more reliable labels.
- **Deliverability is part of the objective.** Spam complaints and low engagement damage
  sender reputation for *all* users, a cross-user externality that per-user optimizers miss.
  Pinterest's and LinkedIn's volume systems [8][11] effectively add a global constraint for
  this.
- **Unsubscribe is close to permanent.** This favors conservative budgets for low-activity
  users, where the marginal gain is small and the unsubscribe risk is high.

## 32.11 Comparison of published designs

| System | Decision unit | Model(s) | Objective | Volume / fatigue control | Timing |
|---|---|---|---|---|---|
| X pushservice [1-5] | best candidate per refresh | MLP light ranker; ClemNet MTL heavy ranker (OONC 0.9, engagement 0.1) | open/NTab click + engagement; dislike label | fatigue, caret, quality, health predicates; epsilon-greedy exploration | refresh-driven |
| Twitter "should I send" [6] | send/no-send | long-horizon activity models | incremental future activity | decision-theoretic threshold | implicit |
| LinkedIn Concourse + ATC [7] | (content, recipient) pair | GLMix-style response models, streaming (Samza) | engagement with negative penalties | central ATC caps, dedupe, channel choice | near real time, can delay |
| LinkedIn MOO [8] | per send | response + negative-outcome models | sessions s.t. disable/unsub limits | Lagrangian thresholds | n/a |
| LinkedIn survival [9] | send/delay | hazard (time-to-visit) models | reduce time to visit causally | implicit via incrementality | core focus |
| LinkedIn offline RL [10] | per-step policy | double-DQN-style offline RL | long-term sessions minus negatives | learned, state-dependent | wait as action |
| Pinterest [11] | weekly per-user budget | volume response curves | long-term weekly activity | budget s.t. unsubscribes | scheduler |
| Duolingo [12] | template per reminder | recovering difference softmax bandit | lesson completion within about 2h | recency penalty (novelty) | daily, user-habit-aligned |

## Key takeaways for system designers

1. **Optimize incremental, long-term value, not CTR.** Every mature system (Pinterest,
   LinkedIn, Twitter) moved from "send if p(open) > t" to models of *causal* lift in
   activity over a horizon.
2. **Separate relevance from arbitration.** Producers score items. A central arbiter
   (LinkedIn ATC, X's take step and predicates) enforces per-user budgets, deduplication,
   quiet hours and channel choice. This keeps global volume coherent as teams add new
   notification types.
3. **Make "don't send" a first-class outcome.** Log drops, keep holdbacks, add exploration
   (X's epsilon-greedy filtering predicate) and run randomized volume experiments. Without
   them the training data reflects only the old policy.
4. **Model annoyance explicitly.** Use dislike, caret and "show less", disable and
   unsubscribe as labels or constraints. Channel loss is near-permanent, so price it high.
5. **Rank many, send one.** A light/heavy cascade followed by a sequential take step keeps
   expensive safety and quality checks cheap.
6. **Novelty decays.** For recurring messages, put exposure recency in the policy
   (Duolingo's recovering bandit) and keep stochastic exploration.
7. **Timing is a causal question.** The best time to notify is when the notification
   changes behavior most, not when the user is most likely to open anyway. Survival models
   and RL policies with a "wait" action capture this.
8. **Budget across channels.** Push, in-app, email and SMS share one attention budget.
   Coordinate them centrally and use digests to turn volume into value.

## References

1. "the-algorithm" README (Recommended Notifications: pushservice, light and heavy
   rankers), X Corp (Twitter), 2023. https://github.com/twitter/the-algorithm
2. "pushservice" README (RefreshForPushHandler and SendHandler), X Corp (Twitter), 2023.
   https://github.com/twitter/the-algorithm/tree/main/pushservice
3. "Light ranking" model README for pushservice, X Corp (Twitter), 2023.
   https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/python/models/light_ranking
4. "Heavy ranking" model (ClemNet; `model.py`, `params.py`, `features.py`), X Corp
   (Twitter), 2023.
   https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/python/models/heavy_ranking
   ; https://github.com/twitter/the-algorithm/blob/main/pushservice/src/main/python/models/heavy_ranking/params.py
5. pushservice Scala packages (`predicate`, `model`), X Corp (Twitter), 2023.
   https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/scala/com/twitter/frigate/pushservice/predicate
   ; https://github.com/twitter/the-algorithm/tree/main/pushservice/src/main/scala/com/twitter/frigate/pushservice/model
6. C. O'Brien et al., "Should I send this notification? Optimizing push notifications
   decision making by modeling the future", Twitter, 2022. https://arxiv.org/abs/2202.08812
   (not re-verified in this session)
7. Y. Gao et al., "Near Real-time Optimization of Activity-based Notifications", LinkedIn,
   KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219880 (DOI not re-verified in this
   session)
8. LinkedIn notification/email volume optimization work (multi-objective optimization of
   sessions vs. unsubscribes/disables), as summarized in [7] and [10]; LinkedIn
   Engineering Blog, https://www.linkedin.com/blog/engineering (index page; specific post
   URL not retrievable in this session)
9. Y. Yuan, J. Zhang, S. Chatterjee, S. Yu, R. Rosales, "A State Transition Model for
   Mobile Notifications via Survival Analysis", LinkedIn, WSDM 2019.
   https://dl.acm.org/doi/10.1145/3289600.3290981 (DOI not re-verified in this session)
10. P. Prabhakar, Y. Yuan, G. Yang, W. Sun, A. Muralidharan, "Multi-objective Optimization
    of Notifications Using Offline Reinforcement Learning", LinkedIn, KDD 2022.
    https://arxiv.org/abs/2207.03029 (not re-verified in this session)
11. B. Zhao, K. Narita, B. Orten, J. Egan, "Notification Volume Control and Optimization
    System at Pinterest", Pinterest, KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219906
    (DOI not re-verified in this session)
12. K. P. Yancey, B. Settles, "A Sleeping, Recovering Bandit Algorithm for Optimizing
    Recurring Notifications", Duolingo, KDD 2020.
    https://research.duolingo.com/papers/yancey.kdd20.pdf (host unreachable from this
    session's sandbox)
