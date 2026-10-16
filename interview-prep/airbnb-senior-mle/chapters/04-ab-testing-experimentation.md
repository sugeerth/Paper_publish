# 04. A/B Testing and Experimentation Deep Dive: ERF, Marketplace Interference, Variance Reduction, Interleaving and Sequential Testing

> Research note (read first). This chapter went through an adversarial review on 2026-10-14. During that review the egress proxy blocked medium.com, arxiv.org, igotanoffer.com, glassdoor.com, dl.acm.org, airbnb.tech, teamblind.com, 1point3acres.com, leetcode.com and reddit.com, and the shared web-search budget was already used up. Only raw.githubusercontent.com was reachable. Five GitHub-hosted sources were fetched and read: two curated indexes of company engineering posts [23][24], which confirm the title, URL and year of the Airbnb posts cited here; the community Airbnb question list [22]; and two general ML/DS interview question banks. None of them contains a candidate report of an Airbnb experimentation or A/B-testing question. **Every question is therefore tagged [Likely].** No question is tagged [Reported], and none could honestly be. The 2024-2026 candidate reports the review set out to find (Glassdoor, Blind, 1point3acres, LeetCode Discuss) could not be reached. Citations such as "[Likely; 3]" point to the publication that makes a topic a plausible probe. They do not mean the question was asked. The References section marks each entry as **confirmed via index** (title, URL and year confirmed from [23] or [24], page itself not opened), **verified** (page fetched and read) or **unverified** (from prior knowledge only). To close the gap, rerun with web access and look for real candidate reports. **Update (2026-10-16):** a deepening pass found aggregator snippets of candidate reports. Questions 47, 65 and 112 are now tagged [Reported], and new reported questions are in the "Deepening pass" section near the end; see refs [41]-[45].

## Format notes

- **Where this material shows up.** Public descriptions of Airbnb's engineering loop (coding x2, system design, ML design, past-project deep dive, core values x2) do not list a separate "A/B testing" round for MLEs [9][10]. For an MLE role that names experimentation or A/B testing, expect it in three places: (a) the evaluation and launch half of the ML design round; (b) the past-project deep dive, with questions like "how did you know it worked?"; and (c) possibly a dedicated experimentation or analytics conversation with a data scientist or experimentation-platform engineer. Ask your recruiter which of these your loop includes. [Likely]
- **Duration and interviewer.** Usually 45-60 minutes with a senior or staff MLE, a data scientist (Airbnb's DS org has an Inference/Experimentation track) or an engineer from the experimentation platform team (ERF) [1][3]. [Likely]
- **The senior (L5) bar.** It goes beyond knowing the t-test. Expect to show you can:
  - spot when the standard A/B setup is biased (interference, novelty, selection, peeking);
  - choose the randomization unit and the metrics, and defend that choice;
  - quantify trade-offs between bias and variance and between speed and rigor;
  - turn ambiguous or conflicting results into a launch decision that Product and leadership accept.
  
  Airbnb's own writing stresses that experiments run on a two-sided marketplace with long, rare conversion funnels (search, then contact or book, then stay, then review). Expect probes on exactly that [1][2][4][5]. [Likely]
- **Format of the questions.** Expect open-ended scenarios ("we changed X in search ranking; design the test; results came back like this; what now?"). Some rounds may add light whiteboard math, such as sample size, standard errors or CUPED adjustment. Expect few closed-form trivia questions. [Likely]

## Questions

### A. Airbnb experimentation platform (ERF) and experiment fundamentals

1. Walk me through how you would design an online A/B test for a new search-ranking model at Airbnb, from hypothesis to launch decision. [Likely; 1][2]
    - What is your primary metric, and why not just click-through rate?
    - What is the randomization unit, and what are you assuming by choosing it?
    - How long would you run it, and what determines that?
2. Airbnb's experimentation platform (ERF, the Experiment Reporting Framework) computes metrics for hundreds of concurrent experiments. If you were designing such a platform, what are its core components? [Likely; 3]
    - How would you implement deterministic, salted hashing for assignment so that concurrent experiments are independent?
    - How do you log exposures versus assignments, and why does the difference matter?
    - How would you make metric definitions consistent across teams (for example, through a central metrics layer)? [Likely; 11]
3. Airbnb's 2017 post is about scaling ERF as the number of experiments grew. If concurrent experiments and metrics grew 10x again, where would the pipeline break first, and how would you restructure it? [Likely; 3]
    - What would you precompute per user-day versus compute on demand?
    - How would you keep results reproducible when metric definitions change mid-experiment?
4. What is the difference between assignment and exposure (trigger) analysis, and when is an intent-to-treat analysis the right choice? [Likely]
    - The treatment only changes the page when a listing has more than 10 photos. Who should be in your analysis population?
    - How does triggering affect your power, and how do you translate a triggered effect back to a site-wide impact?
5. Your exposure log fires on page load for every visitor, including those who never reach the component you changed (overtracking). How does this dilute the measured effect and the power, and how would you fix both the logging and the analysis? [Likely; 29]
6. For a search-ranking test, treatment and control produce identical results on many queries. How would you define a triggered population of queries where the rankings actually differ, and what does computing the counterfactual control ranking cost in serving? [Likely; 29][8]
    - How do you avoid bias when the trigger condition itself depends on the treatment model's output?
7. Many Airbnb users browse logged out and on several devices before booking. How does this affect randomization and metric attribution? [Likely; 1]
    - Would you randomize by cookie, device, user ID or something else? What breaks in each case?
    - How would you handle a user who is assigned on a logged-out device and then logs in?
8. Request-to-book listings need host acceptance, and Instant Book listings do not. How do you define 'conversion' for a guest-side experiment when part of the outcome is the host's decision? [Likely]
    - What happens to your metric if treatment sends more requests to hosts with low acceptance rates?
9. What is a sample ratio mismatch (SRM)? How would you detect it, and what are its common causes on a web/mobile booking site? [Likely]
    - The treatment loads more slowly, so fewer treatment users fire the exposure log. What does that do to your results?
    - Would you ever analyze an experiment that fails the SRM check?
10. How would you run an A/A test, and what would you use it for on a mature platform? [Likely]
    - Your A/A tests show 9% significant results at alpha = 0.05 on one metric. What do you investigate?
11. Explain the p-value and the confidence interval of a difference in means to a product manager who wants a yes/no answer. [Likely]
12. How do you compute the sample size and minimum detectable effect (MDE) for a booking-conversion experiment? [Likely; 1]
    - Booking conversion is low and bookings are lumpy. How does that change the calculation?
    - The PM wants a two-week answer, but power analysis says six weeks. What options do you propose?
13. Your metric is a ratio, such as bookings per search or revenue per booking, where the analysis unit differs from the randomization unit. How do you compute its variance correctly? [Likely]
    - When would you use the delta method, and when a bootstrap?
    - What goes wrong if you treat each search as independent?
14. How do you handle heavy-tailed metrics like gross booking value, where a few long or luxury stays dominate? [Likely]
    - Would you winsorize/cap, log-transform or use a different estimator? What does each do to interpretation?
15. A treatment raises bookings per user but lowers booking conversion per search. How can both be true, and which metric do you trust for the launch decision? [Likely]
16. Airbnb's 2014 post describes an experiment whose result flipped depending on when it was checked, and a case where a browser-specific bug drove the result. What general lessons do you take from such cases? [Likely; 1]
    - How would you build a platform guard against "result depends on the day you looked"?
    - How would you segment by platform/browser without going on a fishing expedition?
17. How long should a booking experiment run given weekly seasonality and a lead time between search and check-in of days to months? [Likely; 1]
    - What do you lose if you stop after full weeks but before guests actually stay?
18. How would you design guardrails that block a launch automatically? What would the guardrail metrics be for a search change at Airbnb? [Likely; 4]
    - How do you set the guardrail threshold so it is neither too noisy nor too lax?
    - Who should be able to override a guardrail, and how should overrides be logged?
19. Airbnb has written about designing experimentation guardrails. Beyond a guardrail that fires on a large negative impact, why might a platform also need a guardrail that checks whether each guardrail metric is adequately powered, and how does it change how long teams must run tests? [Likely; 4]

### B. Metric choice: north stars, proxies and long-term value

20. What should the primary success metric be for a search-ranking experiment at Airbnb: bookings, booking value, nights booked, uncancelled bookings or something else? Defend your choice. [Likely; 2][5]
    - Why might "uncancelled bookings" or "booking with no host cancellation" be better than raw bookings?
    - How would you weigh guest-side gains against host-side outcomes?
21. Bookings are slow and rare, so the team wants a faster proxy metric (listing clicks, wishlist saves, contacts). How do you validate that a proxy is aligned with the north star? [Likely; 2]
    - Show me a scenario where the proxy improves and bookings drop.
    - How would you use historical experiments to measure proxy-to-north-star correlation, and what are the pitfalls?
22. Airbnb has written about putting a monetary future value on actions so that teams can compare trade-offs. How would you use such a metric in an experiment, and how could it mislead you? [Likely; 6]
    - How do you validate a model-based "future value" metric before letting it decide launches?
23. How would you measure long-term effects such as repeat booking, host churn or review quality that appear months after a test ends? [Likely]
    - What are the trade-offs of long-running holdouts?
    - How would you use surrogate indices or a long-term holdback group?
    - After launch, how would you use a long-term holdback to measure whether the feature's effect decays? How large should it be and how long should it run? [Likely]
24. Design an overall evaluation criterion (OEC) for a pricing-suggestion feature shown to hosts. [Likely; 7]
    - What are the host-side metrics, the guest-side metrics and the marketplace-level metrics?
    - Host earnings rise but occupancy falls. Is that a win?
25. How do you choose between a mean metric (average bookings per user) and a proportion metric (percentage of users who book)? [Likely]
26. How would you test whether a change shifts the 90th percentile of a metric, such as search latency or booking lead time, rather than the mean? How do you get valid confidence intervals for a quantile difference at platform scale? [Likely; 30]
27. You track 50 metrics, and three moved significantly in the wrong direction. How do you decide whether to worry? [Likely]
    - How do you control for multiple comparisons without destroying power?
    - Which metrics would you pre-register as primary, secondary and guardrail?
28. How would you measure the quality of search results beyond conversion, for example relevance, diversity or "guest found what they wanted"? [Likely; 8]
    - How would you test a diversity objective whose value is not captured by short-term bookings?
29. Airbnb has published work on managing diversity in search and on learning to rank diversely. How would you run an experiment on a diversity change whose benefit is that some guests find a match they would otherwise miss, while per-search booking rate for the majority may dip slightly? [Likely; 25][26]
    - Which metric captures 'guests who would otherwise have booked nothing'?
30. Cancellation and refund outcomes happen after the experiment window closes. How do you build a metric that accounts for them without waiting months? [Likely]

### C. Marketplace interference and two-sided experiments

31. Why can a standard user-randomized A/B test give a biased estimate in a two-sided marketplace like Airbnb? Explain it with an example. [Likely; 12][13]
    - In which direction is the bias usually, for a ranking change that helps treatment guests book popular listings?
    - When is the interference negligible?
32. A treatment ranking model steers guests toward high-quality listings. Treatment and control guests compete for the same limited inventory on the same dates. What does the A/B test overstate, and why? [Likely; 12]
    - How would you estimate the size of this cannibalization bias?
33. Compare guest-side (demand) randomization with listing/host-side (supply) randomization. When would you use each? [Likely; 13]
    - Which side do you randomize for a host pricing tool, and what interference does that leave?
    - What is the interference pattern when you randomize listings for a ranking boost?
34. Explain two-sided randomization (randomizing both guests and listings, then analyzing the cells). What bias does it reduce, and at what cost? [Likely; 13]
35. Write down the estimand you actually care about for a marketplace launch (the global treatment effect: everyone treated versus no one treated). How does it differ from the difference in means in a user-randomized test under interference? [Likely; 13]
36. Airbnb collaborated on research about reducing interference bias in marketplace pricing experiments. How would you design a pricing experiment that limits interference? [Likely; 12]
    - How would cluster randomization by market or by "similar listings" help, and what does it do to variance?
    - How would you build listing clusters (for example, by geography and listing similarity) so that most competition stays within a cluster?
37. Compare switchback (time-based) designs, geo/market-level randomization and cluster randomization for an Airbnb search or pricing change. [Likely]
    - Why are switchbacks awkward when there is a long lead time between booking and stay?
    - With only a few hundred markets, how do you get enough power from a market-level experiment?
38. How would you detect interference empirically, for example by running the same treatment at two allocation levels (10% versus 50%)? [Likely; 13]
    - What would you conclude if the effect at 50% allocation is half the effect at 10%?
39. Describe an 'A/B test of A/B tests': randomly assigning clusters to either individual-level or cluster-level randomization and comparing the two estimates to detect interference. How would you adapt this for Airbnb listings clustered by market and date? [Likely; 27]
    - How many clusters would you need for the comparison to have any power?
40. Market conditions matter. When supply is tight (peak dates), interference bias grows. How would you account for supply/demand balance in the analysis? [Likely; 13]
    - Would you report the effect separately for supply-constrained and demand-constrained markets?
41. A change makes guests book earlier. Total bookings over the quarter may not change, but the A/B test shows a lift. How would you detect this kind of intertemporal substitution (pull-forward)? [Likely]
42. A host-side change (for example, better calendar sync) increases the number of available nights. How would you measure its marketplace effect, given that extra supply in treatment can take bookings from control listings? [Likely]
43. How would you experiment on a host-side product, such as a new listing-quality nudge, when the outcome you care about (bookings) is driven by guests who are not in your experiment? [Likely]
44. Network effects among guests are weak on Airbnb, but interference through inventory is strong. Contrast this with social networks, and explain how the remedies differ. [Likely]
45. Airbnb Experiences has far smaller inventory than Homes, with fixed time slots that sell out. How would interference and power differ from a Homes ranking test, and how would you design an Experiences ranking experiment? [Likely; 38]
46. How would you use a market-level holdout ("launch everywhere except these N cities") to validate the global effect of a ranking system after launch? [Likely]
    - How do you choose holdout markets, and how do you analyze them (difference-in-differences, synthetic control)?
47. When randomization at the right level is impossible, how would you estimate the causal effect of a marketplace-wide change? [Reported (Data Scientist loop: "estimate the impact of a global launch without a holdout"); 43][14]
    - How does a counterfactual model (predict what would have happened without the change) work, and how do you validate it?
    - What assumptions does synthetic control rely on, and how would you check them?

### D. Variance reduction: CUPED, stratification and regression adjustment

48. Explain CUPED. What is it, why does it reduce variance, and why does it not bias the treatment effect estimate? [Likely; 15]
    - How do you choose the covariate, and what is the optimal theta?
    - Why must the covariate be measured before the experiment starts?
49. Booking is rare, and many users have no pre-period activity (new or logged-out users). How does CUPED perform for them, and what would you do instead? [Likely]
    - Could you use a model-predicted covariate, for example a predicted booking probability (CUPAC-style)? What are the risks?
50. DoorDash describes CUPAC, which uses an ML model's prediction as the CUPED covariate. What must be true of that model (training window, features, version freezing) for the adjusted estimator to stay unbiased? What happens if the model was trained on data that overlaps the experiment period? [Likely; 28]
51. How do CUPED and regression adjustment with several covariates relate? When would you use one over the other? [Likely]
52. Explain stratified sampling versus post-stratification for variance reduction. Which strata would you use for Airbnb (market, platform, new versus returning guests)? [Likely]
53. You applied CUPED and the variance fell by only 3%. Why might that happen for a booking metric, and what else would you try? [Likely]
54. How does variance reduction change your sample-size calculation and your experiment duration? [Likely]
55. Can you combine CUPED with ratio metrics and with the delta method? What do you need to be careful about? [Likely]
56. You cap gross booking value at the 99th percentile and also apply CUPED. Should you cap before or after the adjustment? What does each order do to bias, variance and interpretation? [Likely]
57. When would you choose a more sensitive design over variance reduction in the analysis? Examples include interleaving, paired designs and within-user designs. [Likely; 8]

### E. Interleaving for search ranking

58. What is interleaving, and why is it much more sensitive than an A/B test for comparing rankers? [Likely; 8]
    - Compare team-draft interleaving with balanced interleaving.
    - How do you assign credit for a booking, as opposed to a click, in an interleaved result list?
59. Airbnb has described using interleaving to speed up search-ranking experiments. How would you set it up so that the interleaving verdict agrees with A/B results? [Likely; 8]
    - How would you measure agreement between interleaving and A/B tests over a set of historical launches?
    - What would you do if interleaving says ranker B wins but the A/B test is flat?
60. Airbnb search is not a single click-and-done session. Guests search many times over days, and listings appear on map views and across pages. What complications does this create for interleaving? [Likely; 8]
    - How would you handle the same listing appearing in both rankers' lists, and duplicates across searches?
    - How do you attribute a booking that happens days later?
61. Interleaving is a within-user design. How would you compute its significance (for example, a per-user preference sign test or a bootstrap over users), and why is per-impression analysis wrong? [Likely; 8]
62. What can interleaving not tell you? Give examples of ranking changes that must still be A/B tested (for example, changes to price display, diversity or result-set size). [Likely; 8]
63. Can you run interleaving and A/B tests on the same search traffic at the same time? How would you allocate traffic so that neither contaminates the other? [Likely; 8]
64. Design a two-stage experimentation funnel: offline evaluation, then interleaving, then A/B test. What are the gates between stages? [Likely; 2][8]
    - Offline NDCG improves but interleaving is flat. What is your hypothesis?

### F. Offline-online gap and ML-specific experimentation

65. Your new ranking model improves offline NDCG by 2%, but the A/B test shows no booking lift. What are the possible reasons, and how do you debug? [Reported (Senior MLE onsite, April 2026: pushed on offline versus online metric disagreement); 41][2][5]
    - Could position bias or feedback loops in the training data explain it?
    - How would you tell a modeling problem apart from an experiment-sensitivity problem?
66. Airbnb's deep-learning search papers describe many launches that improved offline metrics but failed online. How would you build an offline metric that better predicts online wins? [Likely; 2][5]
67. When training data come from the control model's own rankings, the treatment model is evaluated on data biased toward control. How does this affect A/B results, and how would you mitigate it? [Likely]
    - What role do exploration traffic or randomized-result buckets play?
68. How would you use off-policy evaluation (counterfactual estimators such as inverse propensity scoring) to pre-screen ranking models before an online test? What does it require from logging? [Likely]
69. Treatment and control models share a training pipeline. After launch, the new model's data feeds back into retraining. How would you measure that feedback-loop effect? [Likely]
70. Your ranking model is retrained weekly with the same architecture on fresh data. Do you A/B test every refresh? Design a policy for automated model refreshes with lighter-weight validation and a rollback trigger. [Likely; 2][5]
71. How would you A/B test a model whose outputs are cached or precomputed, such as listing embeddings or nightly batch scores? [Likely; 16]
    - What contamination can occur if the treatment's embeddings are also used by control's similar-listings carousel?
72. How would you run an experiment on an ML model that serves several surfaces (search, similar listings, email recommendations) at once? [Likely]
73. A new home-page browsing surface (such as Airbnb Categories) changes which searches guests run next. How would you separate its effect on search metrics from changes in ranking quality, and how would you test a ranking change that runs alongside it? [Likely; 39]
74. How would you test a latency-increasing model (larger neural net) when latency itself hurts conversion? [Likely]
    - How would you separate the model-quality effect from the latency effect?
75. How would you A/B test an LLM-powered customer-support feature, such as suggested replies for agents, where outcomes include resolution time, CSAT, escalations and policy errors? [Likely; 40]
    - Do you randomize by agent, by ticket or by guest, and what interference does each choice leave?
    - How do you combine offline LLM evaluation with the online test?
76. How would you experiment on a fraud or trust model, where false positives block real guests and the true outcome (fraud, chargebacks) arrives weeks later? [Likely; 17]
    - Is holding out a random set of transactions from the fraud model ethical or practical? What would you propose?
77. Treatment and control both generate logs that feed the next model's training data through the feature store. How do you tag and filter experiment traffic in feature and training pipelines so that one arm does not contaminate the next model? [Likely; 37]

### G. Sequential testing, peeking and decision rules

78. What goes wrong when a team peeks at an A/B test every day and stops as soon as p < 0.05? Quantify it roughly. [Likely; 1]
79. Explain sequential testing approaches (group sequential with alpha spending, always-valid p-values or mSPRT). Which would you build into an experimentation platform, and why? [Likely; 18]
    - What do you give up in power compared with a fixed-horizon test?
    - How would you explain an "always-valid" confidence interval on a dashboard?
80. Airbnb's 2014 post proposed a dynamic p-value threshold that is stricter early on. How does that compare with modern sequential methods? [Likely; 1]
81. When is early stopping for harm appropriate, versus early stopping for benefit? [Likely]
82. A test has run for its planned duration, and the p-value is 0.07. The PM wants two more weeks. What do you advise? [Likely]
83. A bug in treatment is found and fixed on day 6 of a 21-day test. Do you restart, analyze only the post-fix period or analyze everything? What are the trade-offs? [Likely]
84. How would you apply a Bayesian A/B testing approach to booking conversion? When does it help decision-making, and when does it mislead? [Likely]
    - How would you set priors from Airbnb's history of past experiments?
85. When would you replace an A/B test with a multi-armed bandit, for example for email subject lines or the order of home-page categories? What do you lose for inference and for long-term measurement? [Likely; 34]
86. Explain the winner's curse in experimentation. Airbnb has written about correcting for selection bias when summing the impact of launched winners. How would you correct it? [Likely; 19]
    - Leadership adds up the measured lifts of 40 launched experiments, and the total far exceeds the actual business growth. Why?
87. How would you control the false discovery rate across hundreds of experiments and metrics on a platform? [Likely]
88. Your platform runs about 1,000 experiments a year at alpha = 0.05 and 80% power, and only about 10% of ideas truly work. Roughly what fraction of 'significant' wins are false positives? How should that change launch criteria? [Likely; 35]

### H. Novelty, primacy, learning and time effects

89. How do you detect novelty or primacy effects in an Airbnb experiment, for example a redesigned listing page? [Likely]
    - How would you plot and test the treatment effect over time or by days since first exposure?
    - Why are novelty effects harder to see when most guests visit Airbnb only a few times a year?
90. How do hosts learn and adapt to a new host-side feature, such as changing prices after a pricing tip? How would you design the experiment to capture the post-adaptation equilibrium? [Likely; 7]
91. How would you handle seasonality and holidays, for example a test that spans the summer peak or the New Year period? [Likely]
    - Would you pause, extend or reweight?
92. A city regulation forces a change (for example, a new fee display) on every listing in that market at once. How would you estimate its impact with a quasi-experiment, and what are the main threats (few units, parallel trends, spillover into nearby markets)? [Likely; 31][14]
93. Experiments run during an external shock (for example a pandemic-scale demand drop or a major event in a city). How do you decide if the results still generalize? [Likely]
94. Should Airbnb keep a universal holdout group that sees none of a year's launches? What would you measure, how large should it be, and what does it cost the business and the users in it? [Likely; 32]

### I. Heterogeneous effects, segments and the launch decision

95. The overall effect is flat, but new guests show +3% and returning guests show -2%. How do you decide whether to launch, and whether to personalize? [Likely]
    - How do you avoid Simpson's paradox in this reading?
    - How do you know the segment effect is not a multiple-testing artifact?
96. How would you estimate heterogeneous treatment effects with ML (for example causal forests or uplift models), and how would you validate them? [Likely]
97. You estimated heterogeneous treatment effects and want to show the feature only to segments that benefit. How do you turn the estimates into a targeting policy, and how do you validate the policy itself with a new experiment? [Likely; 33]
98. A treatment helps in large markets and hurts in small markets. What could explain this in a marketplace, and what would you do? [Likely]
99. The experiment improves bookings but reduces revenue per booking because guests choose cheaper stays. How do you make the launch call? [Likely; 6]
100. Results are significant on iOS and web but not on Android. How do you reason about it? [Likely; 1]
101. Walk me through the launch-review discussion you would lead when the primary metric is up, one guardrail is slightly down and the result is borderline significant. [Likely; 4]
    - What would you write in the experiment review document?
    - How do you communicate uncertainty to a VP who wants a single number?
102. How would you size the actual business impact of a launch, as opposed to the experiment's measured lift? Consider trigger rate, ramp-up, interference and decay. [Likely; 19]

### J. Experiment infrastructure and ML-platform design for experiments

103. Design the assignment service for an experimentation platform that serves search, mobile apps and email, with consistent bucketing across surfaces. [Likely; 3]
    - How do layers or mutually exclusive groups work, and when do you need them?
    - How do you handle ramping from 1% to 50% without reshuffling users?
104. Airbnb's mobile apps fetch experiment configuration at startup, and many users run older app versions or open the app offline. How do you handle version eligibility, cached assignments and late exposure logging in the analysis? [Likely; 36]
105. Design the metrics computation pipeline behind a system like ERF: daily aggregates for thousands of metrics across hundreds of experiments. [Likely; 3][11]
    - How do you make it incremental, backfill-able and consistent with the company's metric definitions?
    - How do you make CUPED and the delta method efficient at that scale?
106. Two notifications experiments (email and push) run at once, and both are subject to a global frequency cap. How does the cap create interference between the experiments and between arms, and how would you design around it? [Likely]
107. How do you detect and handle interactions between concurrent experiments, for example two ranking experiments that both touch the same feature? [Likely]
108. How would you design an experiment-quality monitoring system that automatically checks SRM, logging gaps, pre-period imbalance and metric anomalies? [Likely; 4]
109. How would you architect shadow and canary deployment for a new ranking model before it enters an A/B test? [Likely]
    - What would you compare in shadow mode, and what kinds of failure would make you abort?
110. How would you store and expose historical experiment results so that teams can run meta-analyses, such as proxy validation or calibrating priors? [Likely]
111. How would you build self-serve power analysis into the platform, given that metric variances differ by market and season? [Likely]

### K. Pricing, Smart Pricing and host-side experiments

112. Design an experiment for a change to a host price-suggestion model. What is randomized, and what is measured? [Reported (Data Scientist question list: "A/B test a pricing suggestion tool for hosts"); 43][7][12]
    - Hosts may ignore suggestions. How does adoption affect your estimand (intent-to-treat versus compliers)?
    - How do you prevent treatment hosts' price changes from affecting control hosts' bookings?
113. How would you estimate the price elasticity of demand from experiments without running illegal or unfair price discrimination on guests? [Likely]
    - Why do companies like Airbnb generally avoid randomizing the price shown to guests, and what do they do instead?
114. Design an experiment for a weekly or monthly-stay discount nudge shown to hosts, where guests book long stays months ahead and the stay itself spans weeks. When can you read results, and what interim metrics would you trust? [Likely; 7]
115. A pricing experiment shows host revenue up 4% in treatment, but marketplace-wide bookings are flat. How do you interpret this? [Likely; 12]
116. How would you test a change in guest fees or the display of total price (for example, showing all-inclusive prices)? Which interference and regulatory concerns apply? [Likely]

### L. Fairness, ethics and trust in experimentation

117. How would you check that an experiment does not harm a protected group of guests or hosts, given that you may not observe protected attributes directly? [Likely; 20]
    - How might perceived race or gender be measured under privacy constraints (for example, in Airbnb's Project Lighthouse work), and how would that feed into experiment analysis?
118. How would you add a fairness guardrail, such as a gap in booking acceptance rates between groups, to every experiment's scorecard? What power do you realistically have to detect a small change in that gap? [Likely; 20]
119. Which experiments should not be run at all, or need special review, on a travel/housing marketplace? [Likely]
120. How would you experiment on a trust and safety intervention, such as party prevention or booking restrictions, where the main outcome is a rare, severe harm? [Likely; 17]

### M. Past-project deep dive: experimentation probes

121. Tell me about an experiment you ran where the result surprised you. What did you do next? [Likely]
122. Tell me about a time an A/B result was wrong or misleading, for example because of a bug, SRM, interference or novelty. How did you find out? [Likely]
123. Describe a time you pushed back on launching a feature that had a "statistically significant" win. [Likely]
124. Describe a time you changed how your team or org runs experiments, such as a new metric, a new method or new tooling. What was the impact? [Likely]
125. What was the smallest effect you ever detected reliably, and what did you do to get enough sensitivity? [Likely]
126. How did you decide between running an experiment and shipping on offline evidence? Give a time you chose not to A/B test. [Likely]
127. Tell me about a time a limitation of your experimentation platform forced a workaround. What did you build, and what would you change if you owned the platform? [Likely]
128. How do you work with data scientists, PMs and engineers when you disagree about how to read an experiment? [Likely]

### N. Quick-fire concept checks (likely as follow-ups)

129. What is the stable unit treatment value assumption (SUTVA), and where does it fail at Airbnb? [Likely; 13]
130. Why is the t-test usually fine for non-normal metrics at Airbnb scale, and when does it break? [Likely]
131. What is the difference between relative and absolute lift, and how do you build a confidence interval for relative lift? [Likely]
132. What is the difference between a cluster-randomized design's unit of randomization and its unit of analysis, and how do you compute the standard errors? [Likely]
133. What is Twyman's law, and how does it apply to experiment review? [Likely]

## Deepening pass (October 2026): newly reported questions

Five web searches were run on 2026-10-16. Direct Glassdoor, Blind and LeetCode Discuss threads still did not surface; the hits were interview-experience aggregators (PracHub, Codemia) and search snippets of their pages. Pages were not opened, so wording is paraphrased from snippets. Most reported experimentation items come from Airbnb **Data Scientist** loops, and the role is named on each tag; treat them as strong signals for an MLE role that names experimentation, not as MLE reports. Three existing questions were retagged in place: 47, 65 and 112.

### Newly reported

134. A model scores bookings or listings, and the business needs a decision threshold. How do you tune that threshold: what do you choose offline from ROC/PR curves, and what must you settle with an online A/B test on business metrics? [Reported (Senior MLE onsite, April 2026: probed on threshold tuning); 41]
    - Offline precision improves at the new threshold, but the A/B test shows a guardrail drop. Which do you trust?
    - Would you test several thresholds as arms at once, and how does that change the power and multiple-testing picture?
135. Design an experiment to test new pricing tiers. Include the sample-size calculation, the metrics and the analysis plan. [Reported (Data Scientist onsite, aggregator); 42]
    - What are your primary and guardrail metrics?
    - How do novelty, network effects and interference change the design?
136. A new guest booking feature will run as a 14-day test. Choose the primary and guardrail metrics, and work out whether 14 days gives enough power. [Reported (Data Scientist question list, posted October 2025); 43]
    - If 14 days is underpowered, what would you change: the metric, the allocation, variance reduction or the duration?
137. Walk me through an experiment you designed or analyzed, from design to decision: the unit, the metrics, the result and a trade-off you managed. [Reported (Data Scientist HR screen); 44]
    - How did you check for sample-ratio mismatch, broken instrumentation, novelty and interference?
    - Which segments did you look at, and how did you avoid fishing?
138. Design ranking for a search surface with no text query, such as map browsing, flexible-destination or home-page browsing. How would you evaluate it offline, and how would you design the online test? [Reported (Senior+ MLE loop, September 2026: "search ranking without text queries"; the experimentation follow-ups are [Likely]); 45]
    - With no query, what is the analysis unit and what counts as a "successful" session?
    - How would you debug a production regression found after launch?

### Gap-closing [Likely] questions

139. Plan the ramp for a new ranking model from 1% to 100%. What do you check at each stage, and when can you reuse an earlier stage's data in the final analysis? [Likely]
140. Users in this experiment's buckets were in another experiment last month. How could carryover from that earlier experiment bias your result, and how does the platform prevent it (re-salting, cool-down periods)? [Likely; 3]
141. You find a meaningful pre-period imbalance between arms on bookings after the test has started. Do you re-randomize, adjust with CUPED or regression, or carry on? [Likely]
142. You randomize by market-and-date clusters. Show how the intra-cluster correlation (design effect) enters the sample-size calculation, and what that means for duration. [Likely; 13]
143. Several candidate ranking models are ready, but search traffic is limited. How do you allocate traffic and prioritize experiments to maximize learning velocity without underpowering each test? [Likely; 8]
144. How would you measure the online impact of a listing-photo-quality or listing-quality model whose outputs only change ranking indirectly and affect hosts' behaviour over months? [Likely; 43]
145. A team proposes a new metric for the experiment scorecard. How do you validate its sensitivity and directionality against a corpus of past experiments before adopting it? [Likely]
146. How would you A/B test an LLM-powered search or trip-planning feature where answers are non-deterministic and quality is hard to log? Which online metrics and which offline evaluations would gate the launch? [Likely]
147. A model update changes calibration but not ranking order. Which downstream experiments and systems (pricing, fraud thresholds, guardrails) could still be affected, and how would you test for that? [Likely]
148. Treatment hosts receive a feature, and some never adopt it. How do you estimate the effect on adopters (complier average causal effect), and when is that the number leadership should see instead of the intent-to-treat effect? [Likely; 7]

## References

Status key: **verified** = fetched and read during the 2026-10-14 review. **Confirmed via index** = title, URL and year match the verified curated lists [23] or [24], but the page itself could not be opened (medium.com and arxiv.org were blocked). **Unverified** = listed from prior knowledge; confirm before relying on it. No source below is a candidate report of an Airbnb experimentation question.

1. "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7 (unverified: medium.com blocked)
2. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv 2018 (KDD 2019). https://arxiv.org/abs/1810.09591 (confirmed via index [23])
3. "Scaling Airbnb's Experimentation Platform" (ERF), Airbnb Engineering (Medium), 2017. https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166 (confirmed via index [23])
4. "Designing Experimentation Guardrails", Airbnb Engineering (Medium), 2021. https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669 (confirmed via index [23])
5. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv 2020 (KDD 2020). https://arxiv.org/abs/2002.05515 (confirmed via index [23]). Companion post: "Improving Deep Learning for Ranking Stays at Airbnb", https://medium.com/airbnb-engineering/improving-deep-learning-for-ranking-stays-at-airbnb-959097638bde (confirmed via index [24])
6. "How Airbnb Measures Future Value to Standardize Tradeoffs", Airbnb Engineering (Medium), 2021. https://medium.com/airbnb-engineering/how-airbnb-measures-future-value-to-standardize-tradeoffs-3aa99a941ba5 (confirmed via index [23])
7. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219830 (unverified: dl.acm.org blocked)
8. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e (confirmed via index [23])
9. "Airbnb Interview Process & Questions", IGotAnOffer. https://www.igotanoffer.com/blogs/tech/airbnb-interview-process (unverified: domain blocked; used only for loop structure in the Format notes)
10. "Airbnb Machine Learning Engineer Interview Questions", Glassdoor (candidate reports). https://www.glassdoor.com/Interview/Airbnb-Machine-Learning-Engineer-Interview-Questions-EI_IE391850.0,6_KO7,32.htm (unverified: domain blocked; used only for loop structure)
11. "How Airbnb Achieved Metric Consistency at Scale" (Minerva metrics platform), Airbnb Engineering, 2021. No URL given because none could be confirmed. Search the title on the Airbnb engineering blog. (unverified)
12. Holtz, Lobel, Liskovich and Aral, "Reducing Interference Bias in Online Marketplace Pricing Experiments", working paper, 2020. Search the title on arXiv or SSRN. (unverified)
13. Johari, Li, Liskovich and Weintraub, "Experimental Design in Two-Sided Platforms: An Analysis of Bias", Management Science, 2022 (arXiv preprint 2020). Search the title on arXiv. (unverified)
14. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/artificial-counterfactual-estimation-ace-machine-learning-based-causal-inference-at-airbnb-ee32ee4d0512 (confirmed via index [23])
15. Deng, Xu, Kohavi and Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED), WSDM 2013. Search the title in the ACM Digital Library. (unverified)
16. Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD 2018. https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb (confirmed via index [23]; the earlier dl.acm.org link was replaced with the URL the index gives)
17. "Fighting Financial Fraud with Targeted Friction", Airbnb Engineering, 2018. Search the title on the Airbnb engineering blog. (unverified)
18. Johari, Pekelis and Walsh, "Always Valid Inference: Bringing Sequential Analysis to A/B Testing", arXiv 2015 / Operations Research 2022. Search the title on arXiv. (unverified)
19. "Selection Bias in Online Experimentation" (winner's curse correction), Airbnb Engineering, around 2018-2019. Search the title on the Airbnb engineering blog. (unverified)
20. Airbnb "Project Lighthouse" technical paper on measuring discrimination using perceived race under privacy constraints, 2020. Search via the Airbnb Newsroom. (unverified)
21. Kohavi, Tang and Xu, "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing", Cambridge University Press, 2020. General background for SRM, A/A tests, Twyman's law and OEC questions. https://www.researchgate.net/publication/339914315_Trustworthy_Online_Controlled_Experiments_A_Practical_Guide_to_AB_Testing (link as listed in the verified khangich/machine-learning-interview README; page not opened)
22. allaboutjst, "AirBnB Interview Questions" (community-collected from Glassdoor and GeeksForGeeks, last updated 2020). https://github.com/allaboutjst/airbnb (verified 2026-10-14). Contains only coding questions and no experimentation questions, which is why no question here is tagged [Reported].
23. Eugene Yan, "applied-ml: Papers & tech blogs by companies sharing their work on data science & machine learning in production", GitHub. https://github.com/eugeneyan/applied-ml (verified 2026-10-14; its "Validation and A/B Testing" section is the index used to confirm references above and below)
24. Engineer1999, "A Curated List of ML System Design Case Studies", GitHub. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (verified 2026-10-14)
25. Abdool et al., "Managing Diversity in Airbnb Search", arXiv, 2020. https://arxiv.org/abs/2004.02621 (confirmed via index [23])
26. "Learning To Rank Diversely", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621 (confirmed via index [23][24])
27. "Detecting Interference: An A/B Test of A/B Tests", LinkedIn Engineering, 2019. https://engineering.linkedin.com/blog/2019/06/detecting-interference--an-a-b-test-of-a-b-tests (confirmed via index [23])
28. "Improving Experimental Power through Control Using Predictions as Covariate (CUPAC)", DoorDash Engineering, 2020. https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/ (confirmed via index [23])
29. "Overtracking and Trigger Analysis: Reducing sample sizes while INCREASING sensitivity", Booking.com (booking.ai), 2022. https://booking.ai/overtracking-and-trigger-analysis-how-to-reduce-sample-sizes-and-increase-the-sensitivity-of-71755bad0e5f (confirmed via index [23])
30. "Comparing quantiles at scale in online A/B-testing", Spotify Engineering, 2022. https://engineering.atspotify.com/2022/03/comparing-quantiles-at-scale-in-online-a-b-testing (confirmed via index [23])
31. "Key Challenges with Quasi Experiments at Netflix", Netflix Tech Blog, 2020. https://netflixtechblog.com/key-challenges-with-quasi-experiments-at-netflix-89b4f234b852 (confirmed via index [23])
32. "Universal Holdout Groups at Disney Streaming", Disney Streaming (Medium), 2021. https://medium.com/disney-streaming/universal-holdout-groups-at-disney-streaming-2043360def4f (confirmed via index [23])
33. "Analyzing Experiment Outcomes: Beyond Average Treatment Effects", Uber Engineering, 2018. https://eng.uber.com/analyzing-experiment-outcomes/ (confirmed via index [23])
34. "Multi-Armed Bandits and the Stitch Fix Experimentation Platform", Stitch Fix, 2020. https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/ (confirmed via index [23])
35. "Interpreting A/B Test Results: False Positives and Statistical Significance", Netflix Tech Blog, 2021. https://netflixtechblog.com/interpreting-a-b-test-results-false-positives-and-statistical-significance-c1522d0db27a (confirmed via index [23])
36. "Reliable and Scalable Feature Toggles and A/B Testing SDK at Grab", Grab Engineering, 2018. https://engineering.grab.com/feature-toggles-ab-testing (confirmed via index [23])
37. "Zipline: Airbnb's Machine Learning Data Management Platform" (talk), Airbnb, 2018. https://www.youtube.com/watch?v=Tg5VEMEsC-0 (confirmed via index [23])
38. "Machine Learning-Powered Search Ranking of Airbnb Experiences", Airbnb Engineering (Medium), 2019. https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789 (confirmed via index [23][24])
39. "Building Airbnb Categories with ML and Human-in-the-Loop", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb (confirmed via index [23])
40. "How AI Text Generation Models Are Reshaping Customer Support at Airbnb", Airbnb Engineering (Medium), 2022. https://medium.com/airbnb-engineering/how-ai-text-generation-models-are-reshaping-customer-support-at-airbnb-a851db0b4fa3 (confirmed via index [24])
41. "Airbnb Senior Machine Learning Engineer Interview Experience: Four ML System Design Rounds, Rejected on Calibration", PracHub (candidate report, April 2026 onsite). https://prachub.com/interview-experiences/airbnb-senior-machine-learning-engineer-interview-experience-four-ml-system-design-rounds-rejected-on-calibration (search snippet only, 2026-10-16)
42. "Design an A/B test for a pricing model", Codemia (Airbnb Data Scientist onsite practice item). https://codemia.io/practice/design-an-a-b-test-for-a-pricing-model-48 (search snippet only, 2026-10-16)
43. "Airbnb Interview Questions", PracHub company question list (pages 2-3). https://prachub.com/companies/airbnb?page=2 and https://prachub.com/companies/airbnb?page=3 (search snippets only, 2026-10-16; the snippets did not say which page holds each question; much of the content is paywalled)
44. "Walk through an experiment from design to decision", PracHub (tagged Airbnb Data Scientist HR screen). https://prachub.com/interview-questions/walk-through-an-experiment-from-design-to-decision (search snippet only, 2026-10-16)
45. "Airbnb Senior+ Machine Learning Engineer Interview Experience: Search Ranking and Team Matching", PracHub (candidate report, September 2026). https://prachub.com/interview-experiences/airbnb-seniorplus-machine-learning-engineer-interview-experience-search-ranking-and-team-matching (search snippet only, 2026-10-16)
