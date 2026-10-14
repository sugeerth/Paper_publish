# 22. Design Job Recommendation / Candidate Matching (LinkedIn Jobs, Indeed)

> Prompt as usually given: "Design the 'Jobs you may be interested in' module for LinkedIn
> (or Indeed's home-page job feed). Given a member, recommend job postings they are likely
> to apply to and be a good fit for." Variants: "Rank candidates for a recruiter"
> (the reverse direction: talent search / Recruiter), "Job search ranking", "Job alert
> emails", "Match resumes to job descriptions with LLMs".

> Compilation note: arxiv.org, linkedin.com and the portal sites were blocked by the
> network proxy in both the writing and review runs, and the web-search budget was used up.
> All 12 GitHub sources were re-checked in review by cloning them and searching the cited
> text. Sources marked (fetched) were read this way. Sources marked (ch-ref) were verified
> in earlier chapters of this atlas. Papers that could not be checked point to a Google
> Scholar search for the exact title, so no URL is guessed. Figures marked "approx."
> should be checked against the cited paper.

## Company tags

| Company | Basis | Production system it mirrors |
|---|---|---|
| LinkedIn | Khang Pham names LinkedIn as a source of real questions [17] (fetched); LinkedIn has published its job-recsys stack for nearly a decade [1][2][6][8] | JYMBII ("Jobs You May Be Interested In"): GLMix, then GDMix, then LiRank, then 360Brew [2][3][4][6][8] |
| LinkedIn (reverse direction) | Recruiter / talent search is LinkedIn's other matching product [14][13] | Recruiter search and recommendations; fairness-aware re-ranking [13][14] |

Indeed appears in the title only as a product analogue (a two-sided job marketplace). No
source checked here shows Indeed, ZipRecruiter, Glassdoor or the big generic-recsys
companies asking this exact prompt, so they get no company tag.

**Portal coverage.** No major portal checked in this run has a dedicated "job recommendation"
chapter. The building blocks come from these:

| Portal / source | What it gives you for this problem | Link |
|---|---|---|
| ByteByteGo MLSDI (Aminian & Xu) | No jobs chapter [19] (fetched). The Event Recommendation and PYMK chapters are the closest templates | [19] |
| alirezadir ML-Interviews | No jobs question [16] (fetched). Lists "ride matching" and "friend / follower recommendation" (tagged Facebook, Twitter, LinkedIn) as the closest two-sided problems. Its "ML at Companies" page links LinkedIn Recruiter and two-tower posts [15] | [15][16] |
| Khang Pham, "Machine Learning Design Interview" | "LinkedIn Feed Ranking" case and LinkedIn named as a question source [17] (fetched) | [17] |
| Hello Interview recsys pattern | Retrieval -> ranking -> re-ranking pattern | [20] (ch-ref) |
| shafaypro CrackingMachineLearningInterview | Generic recsys stage latency targets (retrieval ~1000 candidates in < 20 ms; ranking < 50 ms) | [26] (fetched) |
| Exponent / Educative / IGotAnOffer / Patrick Halina | MLSD framework and recsys question banks | [21][22][24][23] (ch-ref) |
| Chip Huyen | Delayed labels, feedback loops, continual learning. No jobs exercise in her booklet [18] (fetched) | [18][25] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Given a member (and an optional query or context), return the top-K active jobs. Support "Jobs you may be interested in", job alerts (email/push) and "similar jobs" | [1] + typical |
| Two-sided goal | Member applies AND the employer is interested (recruiter replies, "hearing back"). Optimize for qualified applications, not clicks | [6][7] |
| Hard constraints | Job is open, location / remote match, visa, seniority, language. Dedupe reposts. Respect employer targeting | typical assumption |
| Members | ~1B registered, ~100M+ weekly job seekers | typical assumption |
| Catalog | ~10-50M active postings. High churn: new jobs every second, most expire in 30-60 days | typical assumption |
| QPS | ~10k-50k recommendation requests/s at peak, plus batch job-alert generation for ~100M members per day | typical assumption |
| Latency | p99 ~150-200 ms end to end. Retrieval ~20-30 ms, ranking ~50 ms | typical assumption; [26] |
| Freshness | New job is retrievable within minutes. Member activity features within seconds to minutes | typical assumption |
| Fairness / legal | No use of protected attributes. Monitor exposure parity (gender, region). Explainability for "why this job" | [13] |
| Per-job capacity | Popular jobs saturate (hundreds of applicants). Spread applications across postings | [1] |

## Diagram 1 - Online serving path

**DIAGRAM 1: End-to-end serving for "Jobs you may be interested in" (synthesised from [1][6][10][12][26])**

```
 member opens Jobs tab / alert job fires
              |
              v
 +-------------------------+   member_id, geo,     +------------------------------+
 | API gateway / JYMBII    |---device, query?----->| Online feature store         |
 | orchestrator            |<--profile, skills,----| (Venice-style KV): profile,  |
 +------------+------------+   recent activity     | skills, prefs, last 50 acts  |
              |                                    +------------------------------+
              | fan out (parallel, ~20-30 ms)
     +--------+-----------+-------------------+--------------------+
     v                    v                   v                    v
 +----------------+ +----------------+ +----------------+ +-----------------+
 | Structured     | | Embedding ANN  | | Graph / collab | | Fresh / context |
 | search index   | | (two-tower,    | | "viewers of X  | | new jobs near   |
 | (Galene-like): | | member tower   | | also applied", | | member, job     |
 | title, skills, | | -> job index)  | | company follows| | alert saved     |
 | location query | | top ~500       | | top ~300       | | searches ~200   |
 +-------+--------+ +-------+--------+ +-------+--------+ +--------+--------+
         +------------------+---------+--------+-------------------+
                                      | ~1-2k job ids (union, dedupe)
                                      v
 +--------------------------------------------------------------------------+
 | FILTER: closed/expired, location & remote, visa, already applied/hidden, |
 | employer targeting, spam/scam jobs, duplicate reposts                    |
 +------------------------------------+-------------------------------------+
                                      | ~500-1000 candidates
                                      v
 +------------------------+  job features  +-----------------------------------+
 | L1 light ranker        |<---------------| Job feature store: title emb,     |
 | (GBDT / small DNN /    |                | skills, seniority, company, apps  |
 | GLMix), keep top ~200  |                | so far, age, salary, location     |
 +-----------+------------+                +-----------------------------------+
             | ~200
             v
 +--------------------------+  model   +---------------------------------------+
 | L2 heavy ranker          |<---------| Model store / registry (versioned,    |
 | multi-task DNN (LiRank-  |  weights | canary, shadow) + per-member random-  |
 | style): P(view), P(apply)|          | effect coefficients (GDMix)           |
 | P(recruiter reply|apply) |          +---------------------------------------+
 +-----------+--------------+
             | scored list
             v
 +--------------------------------------------------------------------------+
 | RE-RANK / POLICY: value = P(apply) x P(reply|apply) x quality;           |
 | per-job exposure cap (avoid flooding popular jobs), company diversity,   |
 | freshness boost, fairness constraints, promoted-job slots, explanations  |
 +------------------------------------+-------------------------------------+
                                      | top 25
                                      v
                        response: jobs + "why" (skills match, network)
                                      |
                                      v
 +--------------------------------------------------------------------------+
 | Impression & action logging (Kafka): request_id, position, model version,|
 | features snapshot -> views, saves, applies, dismisses, recruiter replies |
 +--------------------------------------------------------------------------+
```

## Diagram 2 - Offline training & data pipeline

**DIAGRAM 2: Training data, delayed employer labels and the feedback loop (synthesised from [1][2][4][11][25])**

```
 +------------------+  +------------------+  +------------------+  +----------------+
 | Impression log   |  | Member actions   |  | Employer / ATS   |  | Entity data    |
 | (served list,    |  | view, save,      |  | events: viewed   |  | profiles, job  |
 | position, model) |  | apply, dismiss   |  | applicant, reply,|  | posts, company,|
 +--------+---------+  +--------+---------+  | interview, hire  |  | skill graph    |
          |                     |            | (days-weeks late)|  +-------+--------+
          |                     |            +--------+---------+          |
          +----------+----------+---------------------+                    |
                     v                                                     |
 +----------------------------------------------+                          |
 | LABEL JOIN (by request_id + job_id)          |                          |
 | view=1 | apply=1 | qualified apply=1 |        |                          |
 | dismiss = explicit negative;                 |                          |
 | impressed-no-action = weak negative;         |                          |
 | wait window (e.g. 14 d) for recruiter reply  |                          |
 +----------------------+-----------------------+                          |
                        |                                                  |
                        v                                                  v
 +----------------------------------------------+  +-----------------------------------+
 | POINT-IN-TIME FEATURE JOIN (Feathr-style)    |<-| Feature pipelines: text encoders  |
 | features as of impression time; no leakage   |  | (title/skill embeddings), counters|
 | from later applies or job closure            |  | (applies/job, CTR by title), GNN  |
 +----------------------+-----------------------+  | embeddings (LiGNN), nearline aggs |
                        |                          +-----------------------------------+
                        v
 +---------------------------+   +-----------------------------+  +--------------------+
 | Sampling: negatives from  |-->| TRAIN                       |->| OFFLINE EVAL       |
 | impressions + in-batch /  |   | retrieval: two-tower        |  | AUC per task, NDCG,|
 | random jobs for retrieval;|   | ranker: multi-task DNN      |  | recall@K, calib.,  |
 | downsample, IPS weights   |   | personalization: per-member |  | slice by country,  |
 +---------------------------+   | random effects (GDMix)      |  | seniority, gender  |
                                 +-------------+---------------+  +---------+----------+
                                               |                            | pass gate
                                               v                            v
 +----------------------------------+   +-------------------------------------------+
 | Batch: re-embed all active jobs, |<--| MODEL REGISTRY -> shadow -> canary -> A/B |
 | rebuild ANN index (hourly);      |   | (ramp 1% -> 50%), then full deploy        |
 | streaming: embed new jobs in min |   +---------------------+---------------------+
 +----------------------------------+                         |
                                                              v
                                        online serving (Diagram 1) --> new logs
                                        ^                                  |
                                        +-------- feedback loop -----------+
           (exposure bias: only shown jobs get labels -> add exploration slots)
```

## Diagram 3 - Model architecture

**DIAGRAM 3: Two-tower retrieval + multi-task ranker with mixed-effects personalization (synthesised from [3][4][5][6][7])**

```
 RETRIEVAL (two-tower, trained with in-batch softmax on apply/view pairs)

 member tower                                  job tower
 +------------------------------+             +------------------------------+
 | title, seniority, skills,    |             | title, description text,     |
 | industry, location, history  |             | required skills, seniority,  |
 | of viewed/applied job embs,  |             | company, location, remote,   |
 | GNN member emb               |             | GNN job emb (cold start)     |
 +--------------+---------------+             +--------------+---------------+
                v                                            v
        +---------------+                            +---------------+
        | text enc +    |                            | text enc +    |
        | MLP -> u (128)|                            | MLP -> v (128)|
        +-------+-------+                            +-------+-------+
                +------------> dot(u, v) <-------------------+
                               ANN top-k (jobs pre-indexed)

 RANKING (L2, ~200 candidates per request)

 +-----------------+ +------------------+ +-------------------+ +------------------+
 | member features | | job features     | | member x job      | | context          |
 | profile, skills,| | title, skills,   | | skill overlap,    | | surface, device, |
 | activity seq    | | age, #applicants,| | title-sim (DeText)| | time, position   |
 | (last N jobs)   | | salary, company  | | distance, network | | (train only)     |
 +--------+--------+ +--------+---------+ +---------+---------+ +--------+---------+
          +-------------------+----------+----------+--------------------+
                                         v
                    +-----------------------------------------+
                    | embeddings + dense -> DCN cross layers  |
                    | + transformer over member's recent jobs |
                    | (target-attention to candidate job)     |
                    +--------------------+--------------------+
                                         | shared representation
              +--------------------------+------------------------+
              v                          v                        v
      +---------------+         +---------------+        +-------------------+
      | head: P(view) |         | head: P(apply)|        | head: P(reply     |
      | logit, BCE    |         | logit, BCE    |        | given apply) logit|
      +-------+-------+         +-------+-------+        +---------+---------+
              +-------------------------+--------------------------+
                                        v
        logit_task = f_global(x)  +   g_member(x)   +   h_job(x)
                     deep fixed       per-member LR     per-job LR
                     effect (DNN)     random effect     random effect
                     (GDMix / GLMix style, one set per task head)
                                        |
                          sigmoid per task -> P(view), P(apply), P(reply|apply)
                                        |
                                        v
            score = w1*P(view) + w2*P(apply) + w3*P(apply)*P(reply|apply)
                           -> calibration (isotonic) -> re-rank
```

## Diagram 4 - The real system: LinkedIn's job-recsys lineage and the LLM variant

**DIAGRAM 4: How LinkedIn's JYMBII stack evolved, and the 360Brew prompt-as-ranker variant (from [1][2][3][4][6][7][8])**

```
 2016-17             2020               2024                   2025+
 +--------------+   +--------------+   +------------------+   +--------------------+
 | GLMix        |   | GDMix        |   | LiRank           |   | 360Brew (150B      |
 | (Photon ML)  |-->| deep fixed   |-->| Residual DCN,    |-->| decoder, Mixtral   |
 | global LR +  |   | effect       |   | isotonic calib., |   | 8x22B base). One   |
 | per-member + |   | (DeText) +   |   | multi-task,      |   | model, 30+ tasks,  |
 | per-job      |   | per-member LR|   | quantization     |   | text prompts       |
 | random eff.  |   |              |   | +1.76% qualified |   | (pre-production)   |
 +--------------+   +--------------+   | applies (approx.)|   +---------+----------+
                                       +--------+---------+             |
                    LiGNN graph embeddings -----+                       |
                    (members, jobs, skills): LiGNN reports              |
                    ~+1% job-apply hearing-back rate (approx.)          |
                                                                        v
 +----------------------------------------------------------------------------------+
 | 360Brew prompt for one (member, job) pair                                        |
 |   Instruction: predict apply / view / dismiss for the Question job               |
 |   Member profile: current position, company, location                            |
 |   Past job interactions: applied [title, location, company, description] ...     |
 |                          viewed [...]  dismissed [...]                           |
 |   Question: Will the member apply to the following job: [...]                    |
 |   Answer: "The member will apply"  -> logit of yes-token = ranking score         |
 +----------------------------------------------------------------------------------+
   Serving reality: too costly per candidate -> use as teacher (distill into the
   L2 ranker) or as a final re-ranker over the top ~20-50.
```

## Data & labels

| Label | Definition | Notes |
|---|---|---|
| View / click | Member opens the job detail page | Abundant, noisy. Title clickbait inflates it |
| Save | Bookmark | Mid-strength positive |
| Apply | Clicked apply (on-site "Easy Apply" or offsite redirect) | Primary member-side label. Offsite applies are only partly observed |
| Qualified apply | Apply that meets the job's requirements, or that the employer rates as a fit | LinkedIn reports this metric [6] |
| Recruiter reply / "hearing back" | Employer views, messages or advances the applicant | Delayed by days to weeks; sparse [7] |
| Dismiss / "not interested" | Explicit negative | Strong negative. Also a filter signal |
| Impressed, no action | Weak negative | Position-biased; downweight or use IPS |

## Features

| Group | Examples |
|---|---|
| Member | Standardized title and seniority, skills (from the LinkedIn Knowledge Graph taxonomy [32]), industry, location, years of experience, open-to-work preferences (titles, locations, remote) |
| Member activity | Last N viewed / applied / dismissed jobs (as embeddings), searches, saved alerts, application velocity |
| Job | Standardized title, required skills, seniority, company, industry, location, remote flag, salary, posting age, #applicants so far, employer response rate |
| Member x job | Skill overlap (Jaccard, weighted), title similarity (DeText-style text match [5]), seniority gap, distance / commute, has connections at company, past apply to same company |
| Graph | GNN embeddings of member, job, company and skill nodes (LiGNN [7]). Key for cold-start jobs |
| Marketplace | Job's current application count vs. its target, applicant quality so far (for exposure capping) |
| Context | Surface (feed, email, push), device, hour, position (training only) |
| Standardization (pre-step) | Map free-text titles, skills, companies and locations to canonical entity ids before any feature is built; both towers and the search index depend on it [32] |

## Model choices (baseline -> advanced)

| Stage | Option | Pros | Cons |
|---|---|---|---|
| Baseline | Content match: skills / title TF-IDF or BM25 + rules | No training data needed. Explainable | Ignores behavior. Poor ranking |
| Retrieval | Structured search query built from profile (title, skills, location) [12] | Hard constraints become index filters | Low recall on adjacent titles |
| Retrieval | Two-tower embeddings + ANN [30] | Semantic recall. Handles new jobs via content tower | Loses fine-grained interactions |
| Retrieval | Learned job-matching retrieval and GNN job embeddings (LinkedIn 2024) [28][29] | Learns from apply / hire outcomes; helps cold-start jobs | Graph and training infra cost |
| Retrieval | GPU exhaustive / model-based retrieval (LiNR) [10] | Exact top-k with attribute pre-filters | Needs GPU index infrastructure |
| Ranking | GLMix: global LR + per-member + per-job random effects [2][3] | Strong personalization. Cheap. Interpretable | Linear. Manual crosses |
| Ranking | GDMix: deep fixed effect + per-entity LR [4] | Deep text matching plus personalization | Two-stage training complexity |
| Ranking | Multi-task DCN / transformer ranker (LiRank) [6] | Learns crosses and sequences. Multiple objectives | Heavier serving; calibration needed |
| Re-rank / teacher | LLM ranker (360Brew: 150B decoder on Mixtral 8x22B, 30+ tasks across 8+ surfaces) [8][9] | Zero-shot cold start; reads full text; one model for many tasks | Very costly. Pre-production |
| Employer preference | Per-job / per-recruiter model of which applicants get hearing-back ("learning hiring preferences") built on the Feathr feature store [11] | Adds the employer side to the value function | Sparse, delayed labels |
| Employer side | Recruiter / talent search ranker + fairness-aware re-rank [13][14] | Covers the reverse match | Separate system and labels |

## Loss

| Component | Loss |
|---|---|
| Two-tower retrieval | Sampled / in-batch softmax with log-Q correction for popularity, or contrastive loss on (member, applied job) pairs |
| Ranker heads | Sum of weighted BCE per task: L = a*BCE(view) + b*BCE(apply) + c*BCE(reply on applied rows) |
| Mixed effects | Logistic loss with L2 on each random-effect block, fit by coordinate descent with offsets [2][4] |
| Bias handling | Inverse propensity weights by position, or a position feature set to a constant at serving |
| LLM ranker | Next-token cross-entropy on the verbalized answer ("will apply") [8] |

## Offline metrics

| Metric | Use |
|---|---|
| Recall@K (retrieval) | Share of later-applied jobs found in the top 500-1000 candidates |
| AUC / log loss per head | Ranker quality per task. LinkedIn and 360Brew report AUC [6][8] |
| NDCG@K / MAP@K | List quality with graded labels (apply > save > view) |
| Calibration (ECE, predicted / observed) | Needed because heads are multiplied in the value function |
| Slice metrics | By country, seniority, new members, new jobs, and for fairness by gender where legally allowed [13] |
| Coverage / Gini of applies per job | Marketplace health: does the model concentrate applies on a few jobs? |

## Online metrics / A-B

| Type | Metric |
|---|---|
| Primary (two-sided) | Qualified applications per member [6]; hearing-back / recruiter-reply rate [7] |
| Member side | Apply rate, saves, sessions with job activity, job-alert email CTR |
| Employer side | Applicants per job (median, not mean), share of jobs with at least N qualified applicants, recruiter InMail / reply rate |
| Guardrails | Dismiss rate, unsubscribes from alerts, latency, fairness exposure ratios [13], scam-job reports |
| Long-term | Confirmed hires (sparse; use as a validation, not as a training target) |
| Experiment design | Marketplace interference: treatment members compete for the same jobs. Use job-side or market (region) randomization, or ego-cluster designs [31] |

## Serving & latency budget (~180 ms p99, typical assumption)

| Step | Budget | Notes |
|---|---|---|
| Feature fetch (member) | 10 ms | KV store (Venice-style [27]) |
| Candidate generation (parallel) | 25 ms | Search index + ANN + graph + fresh sources |
| Filtering | 5 ms | Bloom filters for applied / hidden; job status cache |
| L1 light ranker (~1000) | 20 ms | GBDT or small DNN on CPU |
| Job feature fetch (~200) | 15 ms | Batched multi-get |
| L2 heavy ranker (~200) | 50 ms | GPU or optimized CPU; quantized [6] |
| Re-rank / policy | 10 ms | Exposure caps, diversity, explanations |
| Network + overhead | 45 ms | |
| Batch path (alerts) | Hours | Precompute top-K for ~100M members nightly, then refresh with fresh jobs |

## Monitoring

| What | Signal |
|---|---|
| Data | Feature null rates, schema drift, title-standardization failures, embedding drift for new job families |
| Model | Online vs. offline AUC per head, calibration drift, score distribution shift |
| Marketplace | Applicant concentration (Gini), jobs with zero impressions, new-job time-to-first-impression |
| Fairness | Exposure and apply-rate parity across groups; audits of protected attributes leaking through proxies [13] |
| Business | Qualified applies, reply rate, alert unsubscribes; alarms on sudden drops |
| Trust | Scam / fraudulent job reports, spammy reposts slipping through filters |

## Failure modes

| Failure | Mitigation |
|---|---|
| Popular jobs flooded with applicants, so employer reply rate falls | Per-job exposure caps; optimize P(reply given apply); downweight jobs near capacity [1] |
| Cold-start jobs never shown (no interactions) | Content / GNN job tower, freshness boost, exploration slots |
| Cold-start members (empty profile) | Use location + title from onboarding, popular-near-you, ask preferences. 360Brew's gain over the production model is largest for members with few interactions, so an LLM ranker or teacher helps here [9] |
| Delayed recruiter labels make training stale | Train the apply head daily; train the reply head on a lagged window; use survival-style or delayed-feedback correction |
| Clickbait titles optimize views, not applies | Optimize apply and qualified-apply heads, not view |
| Bias / discrimination (gender, age proxies) | Drop protected attributes and proxies; fairness-aware re-ranking [13]; audits |
| Expired or filled jobs still recommended | Real-time job status stream into the filter stage |
| Feedback loop narrows the member's job types | Exploration, diversity on title / company, "adjacent role" retrieval |
| Duplicate / scam jobs | Dedup by text hash and embedding; trust and safety classifier before indexing |
| A/B results biased by marketplace interference | Market-level or job-side randomization |

## Interviewer follow-ups

- What exactly is a positive label: view, apply, or the recruiter replying? How do you handle a label that arrives two weeks late?
- How do you avoid sending 2,000 applicants to one popular job while other jobs get none?
- A job was posted five minutes ago. How does it get into the candidate set and get a score?
- A new member has an empty profile. What do you recommend on day one?
- How would you put hard constraints (location, visa, remote) into retrieval instead of filtering afterwards?
- Why multi-task instead of one P(apply) model? How do you set the weights in the value function?
- How do you randomize an A/B test when treatment and control members compete for the same jobs?
- How do you make sure the system does not discriminate by gender or age, even through proxies?
- Offsite applies are not observed. How does that bias training, and what do you do about it?
- How do you build the reverse system (rank candidates for a recruiter)? What changes?
- How would you use an LLM here: features, retrieval embeddings, a re-ranker, or a teacher? What is the cost per request?
- How do you generate "why you are seeing this job" explanations?
- What features would leak the label (for example applicant count updated after the impression)?
- How do you serve job-alert emails for 100M members every day?
- How would you detect and remove scam job postings before they reach recommendations?
- Millions of postings use free-text titles ("Rockstar Dev II"). How do you standardize titles and skills so retrieval and features work [32]?
- How do you learn what each employer actually hires for (hiring preferences) and use it in ranking [11]?

## Portal comparison

| Source | Framing | Core solution | What it misses for jobs |
|---|---|---|---|
| ByteByteGo MLSDI (event recsys, PYMK chapters) [19] | Generic user-to-item recsys | Candidate generation + pointwise LightGBM / DNN ranker; offline and online metrics | No two-sided labels, no exposure capping |
| alirezadir ML-Interviews [15][16] | 6-step template; question list | Points to ride matching and friend recommendation as analogues; links LinkedIn Recruiter and two-tower posts | No worked jobs design |
| Khang Pham [17] | LinkedIn Feed Ranking case | Same LinkedIn-style ranking with logistic regression to deep models, AUC and engagement metrics | Feed, not jobs; no employer label |
| Hello Interview / Exponent / Educative [20][21][22] | Recsys pattern and framework | Retrieval -> ranking -> re-ranking, two-tower and multi-task | Marketplace and fairness only briefly |
| LinkedIn papers and repos [1]-[10] | Production | GLMix to GDMix to LiRank to 360Brew; qualified-apply and hearing-back metrics; per-job capacity | Interview-level time limits (too deep for 45 min) |
| Chip Huyen [18][25] | ML systems concepts | Delayed labels, feedback loops, continual learning | No jobs case |

## References

1. K. Kenthapadi, B. Le, G. Venkataraman (LinkedIn), "Personalized Job Recommendation System at LinkedIn: Practical Challenges and Lessons Learned", RecSys, 2017. Scholar search: https://scholar.google.com/scholar?q=%22Personalized+Job+Recommendation+System+at+LinkedIn%3A+Practical+Challenges+and+Lessons+Learned%22 (known; not fetched)
2. LinkedIn, "Photon ML" repository README (GLMix adopted by "jobs search and recommendation"), GitHub, 2016-2020. https://github.com/linkedin/photon-ml (fetched)
3. X. Zhang, Y. Zhou, Y. Ma, B.-C. Chen, L. Zhang, D. Agarwal (LinkedIn), "GLMix: Generalized Linear Mixed Models For Large-Scale Response Prediction", KDD, 2016. http://www.kdd.org/kdd2016/papers/files/adf0562-zhangA.pdf (ch-ref)
4. LinkedIn, "GDMix: Generalized Deep Mixed Model" repository README (job recommendation example), GitHub, 2020. https://github.com/linkedin/gdmix (fetched)
5. LinkedIn, "DeText: A Deep Neural Text Understanding Framework for Ranking and Classification Tasks", GitHub, 2020. https://github.com/linkedin/detext (fetched)
6. F. Borisyuk et al. (LinkedIn), "LiRank: Industrial Large Scale Ranking Models at LinkedIn", KDD / arXiv, 2024. https://arxiv.org/abs/2402.06859 (ch-ref)
7. F. Borisyuk et al. (LinkedIn), "LiGNN: Graph Neural Networks at LinkedIn", KDD / arXiv, 2024. https://arxiv.org/abs/2402.11139 (ch-ref)
8. 360Brew Team (LinkedIn), "360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation", arXiv, 2025. https://arxiv.org/abs/2501.16450 (ch-ref)
9. LLMPersonalization (LinkedIn), "360Brew-V1" repository and report, GitHub, 2025. https://github.com/LLMPersonalization/360Brew-V1 (fetched: README and 360brew_v1.pdf, which state 150B parameters, Mixtral 8x22 base, 30+ tasks, pre-production)
10. F. Borisyuk et al. (LinkedIn), "LiNR: Model Based Neural Retrieval on GPUs at LinkedIn", arXiv, 2024. https://arxiv.org/pdf/2407.13218 (ch-ref)
11. Feathr project (LinkedIn), talk slides incl. "Learning Hiring Preferences: The AI Behind LinkedIn Jobs" use case, GitHub, 2022. https://github.com/feathr-ai/feathr/tree/main/docs/talks (fetched: both talk PDFs list this use case)
12. LinkedIn Engineering, "Did you mean 'Galene'?", LinkedIn Engineering Blog, 2014. https://engineering.linkedin.com/blog/2014/03/did-you-mean-galene (ch-ref)
13. S. C. Geyik, S. Ambler, K. Kenthapadi (LinkedIn), "Fairness-Aware Ranking in Search and Recommendation Systems with Application to LinkedIn Talent Search", KDD / arXiv, 2019. https://arxiv.org/abs/1905.01989 (ch-ref)
14. LinkedIn Engineering, "The AI Behind LinkedIn Recruiter search and recommendation systems", LinkedIn Engineering Blog, 2019. https://engineering.linkedin.com/blog/2019/04/ai-behind-linkedin-recruiter-search-and-recommendation-systems (URL from [15]; not fetched)
15. alirezadir, "ML at Companies" (LinkedIn section), AIMLInterviews, GitHub, 2025. https://github.com/alirezadir/AIMLInterviews/blob/main/src/MLSD/ml-companies.md (fetched)
16. alirezadir, "ML System Design" (sample questions; no jobs question), Machine-Learning-Interviews, GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
17. K. Pham, "machine-learning-interview" (LinkedIn Feed Ranking case; LinkedIn named as question source), GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched)
18. C. Huyen, "Machine Learning Systems Design: Exercises" (checked: no jobs exercise), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
19. ByteByteGo, "ml-bytebytego: Reference materials for ML System Design Interview" (chapter list; no jobs chapter), GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
20. Hello Interview, "Recommendation Systems (ML System Design pattern)", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/patterns/recommendation-systems (ch-ref)
21. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
22. Educative, "Grokking the Machine Learning Interview", educative.io, n.d. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
23. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
24. IGotAnOffer, "Machine Learning System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/machine-learning-system-design-interview (ch-ref)
25. C. Huyen, "Designing Machine Learning Systems", O'Reilly, 2022. https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/ (ch-ref)
26. shafaypro, "Recommendation System" (system_design/recommendation_system.md; stage latency targets), CrackingMachineLearningInterview, GitHub, n.d. https://github.com/shafaypro/CrackingMachineLearningInterview/blob/master/system_design/recommendation_system.md (fetched)
27. LinkedIn, "Venice: Derived Data Platform for Planet-Scale Workloads", GitHub, 2022-2026. https://github.com/linkedin/venice (repo existence checked in review)
28. LinkedIn, "LinkSAGE: Optimizing Job Matching Using Graph Neural Networks", 2024. Scholar search: https://scholar.google.com/scholar?q=%22LinkSAGE%3A+Optimizing+Job+Matching+Using+Graph+Neural+Networks%22 (known; not fetched)
29. LinkedIn, "Learning to Retrieve for Job Matching", KDD, 2024. Scholar search: https://scholar.google.com/scholar?q=%22Learning+to+Retrieve+for+Job+Matching%22 (known; not fetched)
30. G. Chakravorty, "Personalized Recommendations IV: Two Tower Models for Retrieval", LinkedIn Pulse, n.d. https://www.linkedin.com/pulse/personalized-recommendations-iv-two-tower-models-gaurav-chakravorty/ (URL from [15]; not fetched)
31. G. Saint-Jacques et al. (LinkedIn), "Using Ego-Clusters to Measure Network Effects at LinkedIn", arXiv, 2019. https://arxiv.org/abs/1903.08755 (ch-ref)
32. LinkedIn Engineering, "Building The LinkedIn Knowledge Graph", LinkedIn Engineering Blog, 2016. https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph (URL from [15]; not fetched)
