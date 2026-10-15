# 16. Design Email Spam / Abuse / Bot Detection (Gmail Spam Filter, Twitter/X Bots)

> Prompt as usually given: "Design a system that classifies incoming email as spam /
> phishing / ham" or "Detect spam bots and fake accounts on a social network (Twitter/X,
> LinkedIn, Facebook)." Variants: comment / DM spam, invite spam (Slack), phone-lead spam
> (Zillow), viral spam posts (LinkedIn), fake-account registration, coordinated
> inauthentic behaviour.
>
> Sourcing note: at review time only github.com was reachable from the build sandbox
> (arxiv.org, medium.com, company blogs failed DNS) and web search was exhausted. Each
> reference carries a tag. (fetched) = GitHub page re-read during review.
> (repo-link) = URL appears verbatim in a fetched GitHub index (ByteByteGo ml-bytebytego,
> Engineer1999 case-study list, Botometer README, RETVec README). (ch-ref) = URL verified in
> another chapter of this atlas. (known) = canonical page cited from memory, not re-fetched.
> "Asked by" tags without a cited interview report were removed. Numbers marked "typical
> assumption" are standard interview estimates.

## Company tags

| Company | Evidence | Notes | Source |
|---|---|---|---|
| Google (Gmail) | Built it: RETVec robust text vectorizer (~200K params) deployed for Gmail spam per Google's blog [1]; RETVec repo itself does not name Gmail [2] | Canonical "email spam" system; [1] reports ~38% higher spam detection, ~19% fewer false positives and ~83% less TPU use vs prior vectorizer (not re-fetched) | [1][2][3] |
| X / Twitter | Built it: open-sourced trust-and-safety models (pNSFWMedia, pNSFWText, pToxicity, pAbuse) and visibility filters; no spam/bot model published | README: "several more models and rules" withheld "because of the adversarial nature of this area" | [4][5] |
| Meta (Facebook, Instagram) | Built it: inauthentic-behaviour policy, integrity survey, TIES temporal interaction embeddings, abusive-account detection framework | Actor-level (account) detection; ByteByteGo ch. 5 cites these as follow-ups | [6][7][8][9][10] |
| LinkedIn | Built it: fake-account detection, isolation forest for abuse, viral spam content detection | Two-stage: at registration + post-registration activity | [11][12][13] |
| Pinterest, Slack, Zillow | Built it: spam clustering + automated rule creation; invite-spam ML; SpectroBrain phone-spam (semi-supervised) | Good "variant" systems | [14][15][16] |
| Academic (Indiana Univ. OSoMe) | Botometer: public Twitter bot-score service (v1-v4, BotometerLite); Botometer X is archival, scores from data collected before June 2023 | Canonical bot-detection features and paper trail | [17][18][19] |

**Portals that cover it**

| Portal | Treatment | Link |
|---|---|---|
| alirezadir Machine-Learning-Interviews | Lists "Harmful content / Spam detection system" and "Fraud detection system" under "Other"; harmful-content notes list "detect fake accounts" as other topic | [20][21] |
| ByteByteGo (Aminian & Xu, MLSDI) ch. 5 | Harmful content detection; harmful actors / fake accounts / abusive-account framework as follow-ups | [22][23] |
| Educative "Grokking the ML Interview" / Khang Pham MLSD | Classification template; "Fraud detection (TBD)" heading, empty, in Khang Pham's repo | [24][25][26] |
| Chip Huyen MLSD exercises | Closest: exercise 2, fraud detection from labelled credit-card transactions | [27] |
| Exponent, Hello Interview, IGotAnOffer, Patrick Halina | General MLSD frameworks; integrity/spam frequently used as example classification prompt | [28][29][30][31] |
| Engineer1999 curated case studies | Index of spam / abuse / fraud systems from company blogs | [32] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional (email) | For each inbound message: spam / phishing / malware / promo / ham; action = reject at SMTP, send to Spam folder, warn banner, deliver | typical assumption |
| Functional (bots) | For each account: bot / fake / compromised score; action = challenge (CAPTCHA, phone verify), rate-limit, shadow-limit, suspend | [11][17] + typical assumption |
| Entities scored | Message, sender (IP, domain), account, URL, cluster (campaign) | typical assumption |
| Email volume | Billions of messages/day; ~100K-300K msg/s peak (interview assumption) | typical assumption |
| Account volume | ~1M sign-ups/day; ~500M daily actions to score (posts, follows, DMs) | typical assumption |
| Latency | Email: p99 < 100-200 ms inside SMTP accept path; account sign-up: < 100 ms; batch account sweeps: hourly/daily | typical assumption |
| Precision | False positives (ham in spam, real user banned) are costly; target FPR < 0.1% on ham | typical assumption |
| Adversarial | Attackers adapt in hours: obfuscation (homoglyphs, LEET, typos), image spam, account farms | [1][2][4] |
| Languages | 100+ languages; vocabulary-free text encoder preferred | [2][3] |
| Privacy | Email content is private: limited human labeling, on-device or aggregate features, data retention limits | typical assumption |
| Sender policy (Gmail, 2024) | Bulk senders (>5K msgs/day to Gmail) must pass SPF + DKIM + DMARC, offer one-click unsubscribe, keep Postmaster spam rate < 0.3%: authentication results become hard features / gates | [34] |

## Diagram 1: end-to-end serving architecture

**Diagram 1. Online email-spam path: SMTP -> reputation/rules -> feature fetch -> layered
models -> policy engine -> deliver / spam folder / reject (with feature & model stores,
logging).**

```
 +-------------+  SMTP conn   +-----------------+ IP/domain  +---------------------+
 | Sending MTA |------------->|  Edge / MX gate |----------->| Reputation service  |
 | (or client) |              | (TLS, rate lim.)|<-----------| IP, domain, ASN,    |
 +-------------+              +--------+--------+  rep score | SPF/DKIM/DMARC      |
                                       | message + headers   +---------------------+
                                       v
                          +------------------------+  block   +------------------+
                          | Stage 0: hard rules    |--------->| Reject at SMTP   |
                          | blocklists, known-bad  |          | (5xx) / drop     |
                          | URL & attachment hashes|          +------------------+
                          +-----------+------------+
                                      | pass
                                      v
 +--------------------+   features    +-----------------------+
 | Feature store      |<--------------| Feature assembler     |
 | online: sender rep,|-------------->| parse MIME, URLs,     |
 | recip. history,    |  counters,    | images, headers;      |
 | campaign counters  |  embeddings   | join recipient context|
 +--------------------+               +-----------+-----------+
          ^                                       | feature vector
          | stream updates                        v
          |                          +---------------------------+   +-----------------+
          |                          | Stage 1: cheap model      |   | Model store     |
          |                      +---| GBDT / logistic on rep +  |<--| (versioned,     |
          |                      |   | header features (~2 ms)   |   |  shadow models) |
          |                      |   +------------+--------------+   +--------+--------+
          |                      |                | uncertain band            |
          |                      |                v                           |
          |                      |   +---------------------------+            |
          |          clear ham / |   | Stage 2: deep classifier  |<-----------+
          |          clear spam: |   | text (RETVec/BERT-like) + |
          |         skip stage 2 |   | URL + image + sender emb. |
          |                      |   | multi-head: spam/phish/   |
          |                      |   | malware/promo (~10-30 ms) |
          |                      |   +------------+--------------+
          |                      |                | p(spam), p(phish), ...
          |                      |                v
          |                      |   +---------------------------+
          |                      |   | Policy engine             |
          |                      |   | per-class thresholds,     |
          |                      +-->| per-user allowlists,      |
          |                          | campaign-cluster overrides|
          |                          +--+---------+---------+----+
          |                    deliver  |  spam   |  warn   |
          |                             v         v         v
          |                     +------------+ +---------+ +-------------+
          |                     | Inbox      | | Spam    | | Inbox +     |
          |                     |            | | folder  | | phish banner|
          |                     +-----+------+ +----+----+ +------+------+
          |                           | user actions: "Report spam",
          |                           | "Not spam", open, reply, delete
          |                           v
 +--------+------------+     +---------------------+
 | Stream processor    |<----| Event log (Kafka)   |<--- decisions + scores + features
 | (campaign counters, |     | impressions, labels |
 |  sender rep update) |     +----------+----------+
 +---------------------+                | to offline pipeline (Diagram 2)
                                        v
```

## Diagram 2: offline training & data pipeline

**Diagram 2. Logging -> label sources (user reports, honeypots, analysts, rules) ->
point-in-time feature join -> training -> adversarial eval -> registry -> shadow/canary ->
deploy, with feedback loop.**

```
 +------------------+  +-------------------+  +-----------------+  +-----------------+
 | Decision logs    |  | User feedback     |  | Honeypot / trap |  | Analyst labels  |
 | scores, features |  | report spam /     |  | inboxes, seeded |  | (sampled, priv.-|
 | at decision time |  | not spam, unsub.  |  | bot accounts    |  |  reviewed)      |
 +--------+---------+  +---------+---------+  +--------+--------+  +--------+--------+
          |                      |                     |                    |
          +----------------------+----------+----------+--------------------+
                                            v
                         +--------------------------------------+
                         | Label builder (weak supervision)     |
                         | - combine noisy sources (Snorkel-    |
                         |   style label model)                 |
                         | - de-dup by campaign cluster         |
                         | - label delay window (reports later) |
                         +-------------------+------------------+
                                             | (entity, ts, label, weight)
                                             v
 +------------------------+     point-in-time join   +---------------------------+
 | Offline feature store  |------------------------->| Training set builder      |
 | (snapshot at decision  |                          | - time-based split        |
 |  time: no leakage)     |                          | - neg. downsampling +     |
 +------------------------+                          |   reweighting             |
                                                     | - campaign-grouped split  |
                                                     +-------------+-------------+
                                                                   |
                    +----------------------------------------------+
                    v
 +-----------------------------+   +-----------------------------+
 | Training                    |   | Unsupervised side models    |
 | GBDT (stage 1) + deep multi-|   | clustering / LSH campaigns, |
 | head model (stage 2);       |   | isolation forest, graph     |
 | adversarial augmentation    |   | (bot rings) -> features +   |
 | (typos, homoglyphs, LEET)   |   | auto-rules                  |
 +--------------+--------------+   +--------------+--------------+
                |                                 |
                v                                 v
 +---------------------------------------------------------------+
 | Evaluation gate                                               |
 | PR-AUC, recall @ FPR<=0.1%, per-language & per-segment slices,|
 | adversarial test set, latest-week holdout, calibration        |
 +------------------------------+--------------------------------+
                                | pass
                                v
 +-------------------+   +------------------+   +-------------------------------+
 | Model registry    |-->| Shadow scoring   |-->| Canary 1% -> ramp -> 100%     |
 | (version, data    |   | (log only, diff  |   | threshold re-tuning; rollback |
 |  snapshot, thres.)|   |  vs prod)        |   | on FP spike                   |
 +-------------------+   +------------------+   +---------------+---------------+
                                                                |
            feedback loop: new decisions + user reports         |
  <-------------------------------------------------------------+
  (retrain daily for stage 1; weekly for deep model; hourly rule/cluster refresh)
```

## Diagram 3: model architecture

**Diagram 3. Multi-signal spam/phish classifier: robust char-level text encoder (RETVec-
style) + URL/attachment + sender/reputation + recipient-context towers -> fusion ->
multi-task heads.**

```
 +--------------------+ +------------------+ +--------------------+ +------------------+
 | Subject + body     | | URLs, domains,   | | Sender features    | | Recipient context|
 | text (raw UTF-8)   | | attachments,     | | IP/ASN rep, domain | | contact? replied |
 |                    | | image OCR text   | | age, auth results, | | before? user-    |
 |                    | |                  | | volume, bounce rate| | level spam rate  |
 +---------+----------+ +--------+---------+ +---------+----------+ +--------+---------+
           |                     |                     |                     |
           v                     v                     v                     v
 +--------------------+ +------------------+ +--------------------+ +------------------+
 | RETVec char-       | | URL char-CNN /   | | Dense + bucketized | | Dense; user      |
 | encoder (~200K     | | domain embedding;| | numeric; sender-ID | | embedding        |
 | params, vocab-free)| | image CNN        | | embedding (hashed) | |                  |
 | + small transformer| |                  | |                    | |                  |
 +---------+----------+ +--------+---------+ +---------+----------+ +--------+---------+
           | 256-d               | 128-d               | 64-d                | 32-d
           +----------+----------+----------+----------+----------+----------+
                      v                                           v
              +---------------------------------------------------------+
              | Fusion: concat -> MLP (3 x 512, ReLU, dropout)          |
              +------------------------------+--------------------------+
                                             |
          +----------------+-----------------+-----------------+---------------+
          v                v                 v                 v               v
   +------------+   +------------+   +---------------+   +------------+  +------------+
   | p(spam)    |   | p(phish)   |   | p(malware)    |   | p(promo)   |  | campaign   |
   | sigmoid    |   | sigmoid    |   | sigmoid       |   | sigmoid    |  | embedding  |
   +------------+   +------------+   +---------------+   +------------+  | (for LSH)  |
                                                                         +------------+
   Loss = sum_k w_k * BCE_k (focal for rare phish/malware) + optional contrastive term
```

## Diagram 4: account-level bot / fake-account detection (Twitter/X, LinkedIn, Meta)

**Diagram 4. Two-checkpoint bot detection: registration-time scoring + post-registration
behavioural / graph models + campaign clustering, feeding challenges and enforcement.**

```
 +-------------+  sign-up   +--------------------------+  high risk   +---------------+
 | New account |----------->| Checkpoint A: register   |------------->| Challenge:    |
 | request     |            | GBDT on IP, device FP,   |              | CAPTCHA/phone |
 +-------------+            | email domain, name, UA,  |  med risk    | verify, or    |
                            | sign-up velocity per IP  |------------->| block         |
                            +------------+-------------+              +---------------+
                                         | low risk: allow
                                         v
 +------------------------------------------------------------------------------+
 | Activity stream: posts, follows, likes, DMs, logins, profile edits           |
 +-----------+---------------------------+-----------------------------+--------+
             |                           |                             |
             v                           v                             v
 +-----------------------+  +---------------------------+  +---------------------------+
 | Behaviour model       |  | Graph model               |  | Campaign clustering       |
 | sequence of actions   |  | follower/friend graph,    |  | MinHash/LSH on content,   |
 | (TIES-style temporal  |  | shared IP/device edges;   |  | same URL / template;      |
 | embeddings), timing,  |  | GNN / label propagation   |  | cluster size, burstiness  |
 | ratios (Botometer-like|  | (bot rings, Sybil)        |  | -> auto-rule creation     |
 | features)             |  |                           |  |                           |
 +-----------+-----------+  +-------------+-------------+  +-------------+-------------+
             | p(bot)                     | p(ring)                      | cluster score
             +----------------------------+------------------------------+
                                          v
                         +-----------------------------------+
                         | Account risk aggregator           |
                         | (stacked GBDT / max with weights) |
                         +----------------+------------------+
                                          v
                         +-----------------------------------+   appeals /
                         | Enforcement policy                |---> human review
                         | rate-limit, shadow-limit, label   |   (labels back to
                         | "automated", suspend, mass-action |    training)
                         | on whole cluster                  |
                         +-----------------------------------+
```

## Data & labels

| Source | Type | Pros | Cons |
|---|---|---|---|
| "Report spam" / "Not spam" clicks | Explicit, user | Huge volume, fresh | Noisy (users report newsletters), biased to visible mail |
| Implicit (delete unread, never opened, unsubscribe) | Weak | Free | Very noisy; use as features or low-weight labels |
| Honeypot / spam-trap inboxes, seeded accounts | Ground truth spam | High precision, early | Not representative of targeted spam |
| Analyst review (sampled, privacy-reviewed) | Gold | Accurate eval set | Expensive, privacy constraints |
| Enforcement outcomes (suspended, appealed) | Account labels | Strong for bots | Delayed, feedback-loop bias |
| Rule / blocklist hits | Heuristic | Cheap | Model learns the rule; keep held-out |
| Semi-supervised / clustering | Propagated | Scales labels to campaigns | Error propagation [16][14] |

## Features

| Group | Examples |
|---|---|
| Sender reputation | IP / ASN / domain age, SPF/DKIM/DMARC pass, historical spam rate, volume spikes, bounce rate |
| Content | Raw text via robust char encoder, subject length, ALL CAPS ratio, HTML/text ratio, hidden text, OCR from images |
| URL / attachment | Domain age, redirect chain, URL shortener, known-bad hash, file type, macro presence; async sandbox detonation verdict for unknown attachments (feeds post-delivery re-scan) |
| Recipient context | Sender in contacts, prior replies, user's own report history, # recipients |
| Campaign | Near-duplicate cluster size (MinHash), burst rate, # distinct senders sharing template |
| Account behaviour (bots) | Account age, followers/following ratio, posts per hour, inter-action time entropy, default profile, username randomness [17][18] |
| Graph (bots) | Shared device/IP edges, clustering coefficient, reciprocity, neighbour bot scores [7][8] |

## Model choices (baseline -> advanced)

| Stage | Model | Tradeoffs |
|---|---|---|
| 0 | Rules + blocklists + hash match | Instant, explainable; brittle, easily evaded |
| 1 | Naive Bayes / logistic on bag-of-words | Classic baseline; cheap; obfuscation breaks it |
| 2 | GBDT (XGBoost/LightGBM) on reputation + engineered features | Strong on tabular signals; fast; weak on raw text |
| 3 | Deep text model (char-CNN / RETVec + transformer, multilingual) | Robust to typos/homoglyphs; costlier; needs GPU/TPU budget [2][3] |
| 4 | Multimodal multi-task (text + URL + image + sender) | Shares signal across spam/phish/malware; harder to debug |
| 5 | Graph / temporal models for actors (GNN, TIES) | Catches coordinated bots; offline/near-line [8] |
| 6 | Unsupervised (isolation forest, clustering + auto-rules) | Finds new attacks without labels [12][14] |
| 7 | LLM-as-judge for borderline / analyst assist | Rich reasoning; too slow/expensive for every message |
| 8 | Few-shot adaptable large model (Meta Few-Shot Learner style) | New abuse policy enforced from a handful of examples in weeks, not months; heavy, used near-line [35] |

## Loss

| Choice | When |
|---|---|
| Weighted BCE per head | Default multi-label (spam, phish, malware, promo) |
| Focal loss | Rare classes (phish, malware) with extreme imbalance [33] |
| Sample weights by label source | Honeypot > analyst > user report > implicit |
| Cost-sensitive / asymmetric | FP (ham to spam) costs more than FN |
| Contrastive / triplet on campaign pairs | Learn campaign embeddings for clustering |

## Offline metrics

| Metric | Why |
|---|---|
| Recall at fixed FPR (e.g. FPR <= 0.1% on ham) | Operating-point metric that matches the business |
| PR-AUC | Imbalanced data; better than ROC-AUC |
| Per-slice FPR (language, sender type, new domains) | Fairness to legit small senders |
| Adversarial robustness | Accuracy on perturbed (typo/homoglyph) test set [2] |
| Calibration (ECE) | Thresholds and policy engine depend on calibrated scores |
| Time-split holdout | Detect decay against newest attacks |

## Online metrics / A-B

| Metric | Direction |
|---|---|
| Spam-in-inbox rate (user "report spam" per 1K inbox mails) | Down |
| Ham-in-spam rate ("not spam" clicks per 1K spam-folder mails) | Down (guardrail) |
| Phishing click-through on delivered mail | Down |
| Bot prevalence (sampled audit) / fake sign-ups per day | Down |
| Prevalence (violating views / all views) and proactive rate (caught by system / all actioned) | Down / Up [21][22] |
| False suspensions, appeal overturn rate | Down (guardrail) |
| Time-to-detect new campaign | Down |
| Engagement of legit senders / legit new users | Neutral (guardrail) |

- A/B caveat: adversaries see both arms; randomize by recipient, run short tests, and favour
  shadow mode + back-testing.

## Serving & latency budget (email, p99 ~150 ms)

| Step | Budget |
|---|---|
| SMTP edge + reputation lookup | 10 ms |
| Rules / hash / blocklist | 5 ms |
| Parse MIME, URLs, OCR (async for heavy images) | 25 ms |
| Feature store fetch | 15 ms |
| Stage-1 GBDT | 2-5 ms |
| Stage-2 deep model (only uncertain ~10-20%) | 20-40 ms on accelerator |
| Policy engine + write decision | 10 ms |
| Headroom | ~40 ms |

- Post-delivery re-scan: messages already delivered can be moved to Spam when a campaign is
  later detected ("zero-hour" retro-actions).

## Monitoring

| Signal | Action |
|---|---|
| Score distribution shift per sender segment | Alert, re-calibrate thresholds |
| Sudden rise in "not spam" clicks | Auto-rollback model / threshold |
| New cluster growth rate | Analyst queue, auto-rule |
| Feature freshness (reputation lag) | Page on-call |
| Label pipeline delay | Pause retraining |
| Per-language precision from sampled audits | Retrain / add data |

## Failure modes

| Failure | Mitigation |
|---|---|
| Adversarial obfuscation (homoglyphs, LEET, zero-width chars) | Robust char encoder (RETVec), Unicode normalization, adversarial augmentation [2][3] |
| Image-only spam | OCR + image embeddings |
| Account warm-up (bots act human first) | Long-horizon behaviour models, graph signals |
| Feedback loop (only mail in inbox gets reported) | Exploration sample, honeypots, analyst audits |
| Legit bulk senders (newsletters) flagged | Promo class separate from spam; sender reputation; one-click unsubscribe |
| Label noise from user reports | Label model, per-user reliability weights |
| Model leaks via probing | Randomized thresholds, delayed actions, no fine-grained error messages |
| Compromised legit accounts | Account-takeover detection (login anomalies), not only content |
| Cold start new domains/IPs | Prior from ASN/registrar; conservative throttling |

## Interviewer follow-ups

- How do you get labels when email content is private and cannot be freely reviewed?
- How do you handle "report spam" noise (users reporting newsletters they subscribed to)?
- Spammers adapt in hours: how do you detect a brand-new campaign with zero labels?
- How would you make the text model robust to homoglyphs, typos and zero-width characters?
- Rules vs ML: when do you keep a hand-written rule, and how do rules feed the model?
- Choose the threshold: what is the cost of a false positive vs a false negative here?
- Why not one model for spam, phishing and malware? Multi-task vs separate models.
- How do you evaluate offline when attackers react to your deployed model?
- Bots: what features separate a bot from a power user or a legit automated account?
- How would you use the graph (shared IPs/devices, follow graph) to find bot rings?
- How do you avoid feedback loops where only inboxed mail can be reported?
- How do you serve billions of messages/day under 150 ms? What runs in the cascade?
- How do you A/B test when the adversary observes both arms?
- How would you add an LLM, and where in the cascade would it sit?
- How do you handle appeals and false suspensions, and use them as labels?
- How do sender-authentication rules (SPF/DKIM/DMARC, bulk-sender spam-rate caps) change what the model must learn? [34]
- How do you measure prevalence when most spam is never seen by users? (sampled audits, proactive rate) [21][22]

## Portal comparison

| Portal | Framing | Model emphasis | Distinctive point |
|---|---|---|---|
| ByteByteGo ch. 5 [22][23] | Harmful content on social platform; actors as follow-up | Multimodal multi-task NN, BCE per task, focal loss | Prevalence, proactive rate, appeals; references Meta abusive-account framework |
| alirezadir notes [20][21] | "Harmful content / Spam detection" in question list | DistilmBERT text, CLIP / SimCLR image, VideoMoCo video, multi-task NN | 500M posts/day, natural labels for train, manual for eval |
| Educative / Khang Pham [24][25][26] | Generic binary classification template | GBDT -> deep; imbalance handling | Fraud/spam section is TBD in Khang Pham repo |
| Chip Huyen [27] | Fraud-classification exercise (closest analogue) | Supervised on labelled records | Focus on data/labels and metrics |
| Exponent / Hello Interview / Halina [28][29][31] | Framework-driven; spam is an example classifier prompt | Baseline -> advanced, cascade serving | Delivery framework, tradeoff discussion |
| Real systems [1]-[19] | Email: Gmail RETVec; social: Twitter T&S models, LinkedIn fake accounts, Botometer | Robust text encoders, GBDT + graph + clustering | Adversarial robustness and campaign clustering dominate |

## References

1. Google Security Blog, "Improving Text Classification Resilience and Efficiency with RETVec", security.googleblog.com, 2023. https://security.googleblog.com/2023/11/improving-text-classification.html (known)
2. Google Research, "RETVec: Resilient and Efficient Text Vectorizer" (repository), GitHub, 2023. https://github.com/google-research/retvec (fetched)
3. E. Bursztein et al., "RETVec: Resilient and Efficient Text Vectorizer", arXiv, 2023. https://arxiv.org/abs/2302.09207 (repo-link)
4. X / Twitter, "the-algorithm: trust_and_safety_models" (pNSFWMedia, pNSFWText, pToxicity, pAbuse), GitHub, 2023. https://github.com/twitter/the-algorithm/tree/main/trust_and_safety_models (fetched)
5. X / Twitter, "the-algorithm" (visibility filters), GitHub, 2023. https://github.com/twitter/the-algorithm (fetched)
6. Meta Transparency Center, "Inauthentic behavior" policy, transparency.fb.com. https://transparency.fb.com/policies/community-standards/inauthentic-behavior/ (repo-link)
7. A. Halevy et al., "Preserving Integrity in Online Social Networks", arXiv, 2020. https://arxiv.org/pdf/2009.10311.pdf (repo-link)
8. N. Noorshams, S. Verma, A. Hofleitner, "TIES: Temporal Interaction Embeddings for Enhancing Social Media Integrity at Facebook", arXiv, 2020. https://arxiv.org/pdf/2002.07917.pdf (repo-link)
9. Meta @Scale, "Temporal interaction embedding" (talk), facebook.com. https://www.facebook.com/atscaleevents/videos/730968530723238/ (repo-link)
10. "Abusive account detection framework" (talk), YouTube. https://www.youtube.com/watch?v=YeX4MdU0JNk (repo-link)
11. LinkedIn Engineering, "Automated Fake Account Detection at LinkedIn", engineering.linkedin.com, 2018. https://engineering.linkedin.com/blog/2018/09/automated-fake-account-detection-at-linkedin (known)
12. LinkedIn Engineering, "Detecting and preventing abuse on LinkedIn using isolation forests", 2019. https://engineering.linkedin.com/blog/2019/isolation-forest (repo-link)
13. LinkedIn Engineering, "Viral spam content detection at LinkedIn", 2023. https://engineering.linkedin.com/blog/2023/viral-spam-content-detection-at-linkedin (repo-link)
14. Pinterest Engineering, "Fighting Spam using Clustering and Automated Rule Creation", Medium, 2021. https://medium.com/pinterest-engineering/fighting-spam-using-clustering-and-automated-rule-creation-1c01d8c11a05 (repo-link)
15. Slack Engineering, "Blocking Slack Invite Spam With Machine Learning", 2021. https://slack.engineering/blocking-slack-invite-spam-with-machine-learning/ (repo-link)
16. Zillow Tech, "SpectroBrain: Detecting Phone Spam with Semi-Supervised Learning", zillow.com, 2023. https://www.zillow.com/tech/spectrobrain-detecting-phone-spam-with-semi-supervised-learning/ (repo-link)
17. OSoMe, Indiana University, "botometer-python" (Botometer X client), GitHub. https://github.com/osome-iu/botometer-python (fetched)
18. K.-C. Yang et al., "Scalable and Generalizable Social Bot Detection through Data Selection" (BotometerLite), arXiv, 2019. https://arxiv.org/abs/1911.09179 (repo-link)
19. M. Sayyadiharikandeh et al., "Detection of Novel Social Bots by Ensembles of Specialized Classifiers" (Botometer v4), arXiv, 2020. https://arxiv.org/abs/2006.06867 (repo-link)
20. alirezadir, "ML System Design" (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
21. alirezadir, "Harmful content detection" (mlsd-harmful-content.md), GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/mlsd-harmful-content.md (fetched)
22. A. Aminian, A. Xu, "Machine Learning System Design Interview", ch. 5 "Harmful Content Detection", ByteByteGo, 2023. https://bytebytego.com/courses/machine-learning-system-design-interview/harmful-content-detection (ch-ref)
23. ByteByteGo, "ml-bytebytego: references for Machine Learning System Design Interview", GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
24. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
25. Educative / K. Pham, "Machine Learning System Design", educative.io, 2021. https://educative.io/courses/machine-learning-system-design/introduction (ch-ref)
26. K. Pham, "machine-learning-interview", GitHub. https://github.com/khangich/machine-learning-interview (fetched)
27. C. Huyen, "Machine Learning Systems Design - Exercises", GitHub. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
28. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
29. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
30. IGotAnOffer, "Meta Machine Learning Engineer Interview", igotanoffer.com, 2025. https://igotanoffer.com/blogs/tech/facebook-machine-learning-engineer-interview (ch-ref)
31. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
32. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (fetched)
33. T.-Y. Lin et al., "Focal Loss for Dense Object Detection", arXiv, 2017. https://arxiv.org/abs/1708.02002 (known)
34. Google Workspace Admin Help, "Email sender guidelines", support.google.com, 2024. https://support.google.com/a/answer/81126 (known)
35. Meta, "Meta's New AI System Helps Tackle Harmful Content" (Few-Shot Learner), about.fb.com, 2021. https://about.fb.com/news/2021/12/metas-new-ai-system-tackles-harmful-content/ (repo-link)
