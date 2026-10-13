# 04. A/B Testing and Experimentation Deep Dive: ERF, Marketplace Interference, Variance Reduction, Interleaving and Sequential Testing

> Research note (read first). No live research was possible for this chapter. The shared web-search budget for this run was already used up (0 of the planned 30-45 searches ran), and every WebFetch attempt failed DNS resolution (medium.com, igotanoffer.com). The questions below are therefore built from prior knowledge of Airbnb's published experimentation work and of its widely reported loop structure (see Chapter 01). **Every question is tagged [Likely].** None is tagged [Reported], because no candidate report could be opened and checked during this run. Citations such as "[Likely; 3]" point to the Airbnb publication that makes the topic a plausible probe. They do not mean the question was asked. Confirm each reference before relying on it. To fill the gap, rerun this chapter's research with a fresh search budget and look for real candidate reports on Glassdoor, Blind, 1point3acres and LeetCode Discuss.

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
3. What is the difference between assignment and exposure (trigger) analysis, and when is an intent-to-treat analysis the right choice? [Likely]
   - The treatment only changes the page when a listing has more than 10 photos. Who should be in your analysis population?
   - How does triggering affect your power, and how do you translate a triggered effect back to a site-wide impact?
4. Many Airbnb users browse logged out and on several devices before booking. How does this affect randomization and metric attribution? [Likely; 1]
   - Would you randomize by cookie, device, user ID or something else? What breaks in each case?
   - How would you handle a user who is assigned on a logged-out device and then logs in?
5. What is a sample ratio mismatch (SRM)? How would you detect it, and what are its common causes on a web/mobile booking site? [Likely]
   - The treatment loads more slowly, so fewer treatment users fire the exposure log. What does that do to your results?
   - Would you ever analyze an experiment that fails the SRM check?
6. How would you run an A/A test, and what would you use it for on a mature platform? [Likely]
   - Your A/A tests show 9% significant results at alpha = 0.05 on one metric. What do you investigate?
7. Explain the p-value and the confidence interval of a difference in means to a product manager who wants a yes/no answer. [Likely]
8. How do you compute the sample size and minimum detectable effect (MDE) for a booking-conversion experiment? [Likely; 1]
   - Booking conversion is low and bookings are lumpy. How does that change the calculation?
   - The PM wants a two-week answer, but power analysis says six weeks. What options do you propose?
9. Your metric is a ratio, such as bookings per search or revenue per booking, where the analysis unit differs from the randomization unit. How do you compute its variance correctly? [Likely]
   - When would you use the delta method, and when a bootstrap?
   - What goes wrong if you treat each search as independent?
10. How do you handle heavy-tailed metrics like gross booking value, where a few long or luxury stays dominate? [Likely]
    - Would you winsorize/cap, log-transform or use a different estimator? What does each do to interpretation?
11. Airbnb's 2014 post describes an experiment whose result flipped depending on when it was checked, and a case where a browser-specific bug drove the result. What general lessons do you take from such cases? [Likely; 1]
    - How would you build a platform guard against "result depends on the day you looked"?
    - How would you segment by platform/browser without going on a fishing expedition?
12. How long should a booking experiment run given weekly seasonality and a lead time between search and check-in of days to months? [Likely; 1]
    - What do you lose if you stop after full weeks but before guests actually stay?
13. How would you design guardrails that block a launch automatically? What would the guardrail metrics be for a search change at Airbnb? [Likely; 4]
    - How do you set the guardrail threshold so it is neither too noisy nor too lax?
    - Who should be able to override a guardrail, and how should overrides be logged?

### B. Metric choice: north stars, proxies and long-term value

14. What should the primary success metric be for a search-ranking experiment at Airbnb: bookings, booking value, nights booked, uncancelled bookings or something else? Defend your choice. [Likely; 2][5]
    - Why might "uncancelled bookings" or "booking with no host cancellation" be better than raw bookings?
    - How would you weigh guest-side gains against host-side outcomes?
15. Bookings are slow and rare, so the team wants a faster proxy metric (listing clicks, wishlist saves, contacts). How do you validate that a proxy is aligned with the north star? [Likely; 2]
    - Show me a scenario where the proxy improves and bookings drop.
    - How would you use historical experiments to measure proxy-to-north-star correlation, and what are the pitfalls?
16. Airbnb has written about putting a monetary future value on actions so that teams can compare trade-offs. How would you use such a metric in an experiment, and how could it mislead you? [Likely; 6]
    - How do you validate a model-based "future value" metric before letting it decide launches?
17. How would you measure long-term effects such as repeat booking, host churn or review quality that appear months after a test ends? [Likely]
    - What are the trade-offs of long-running holdouts?
    - How would you use surrogate indices or a long-term holdback group?
18. Design an overall evaluation criterion (OEC) for a pricing-suggestion feature shown to hosts. [Likely; 7]
    - What are the host-side metrics, the guest-side metrics and the marketplace-level metrics?
    - Host earnings rise but occupancy falls. Is that a win?
19. How do you choose between a mean metric (average bookings per user) and a proportion metric (percentage of users who book)? [Likely]
20. You track 50 metrics, and three moved significantly in the wrong direction. How do you decide whether to worry? [Likely]
    - How do you control for multiple comparisons without destroying power?
    - Which metrics would you pre-register as primary, secondary and guardrail?
21. How would you measure the quality of search results beyond conversion, for example relevance, diversity or "guest found what they wanted"? [Likely; 8]
    - How would you test a diversity objective whose value is not captured by short-term bookings?
22. Cancellation and refund outcomes happen after the experiment window closes. How do you build a metric that accounts for them without waiting months? [Likely]

### C. Marketplace interference and two-sided experiments

23. Why can a standard user-randomized A/B test give a biased estimate in a two-sided marketplace like Airbnb? Explain it with an example. [Likely; 12][13]
    - In which direction is the bias usually, for a ranking change that helps treatment guests book popular listings?
    - When is the interference negligible?
24. A treatment ranking model steers guests toward high-quality listings. Treatment and control guests compete for the same limited inventory on the same dates. What does the A/B test overstate, and why? [Likely; 12]
    - How would you estimate the size of this cannibalization bias?
25. Compare guest-side (demand) randomization with listing/host-side (supply) randomization. When would you use each? [Likely; 13]
    - Which side do you randomize for a host pricing tool, and what interference does that leave?
    - What is the interference pattern when you randomize listings for a ranking boost?
26. Explain two-sided randomization (randomizing both guests and listings, then analyzing the cells). What bias does it reduce, and at what cost? [Likely; 13]
27. Airbnb collaborated on research about reducing interference bias in marketplace pricing experiments. How would you design a pricing experiment that limits interference? [Likely; 12]
    - How would cluster randomization by market or by "similar listings" help, and what does it do to variance?
    - How would you build listing clusters (for example, by geography and listing similarity) so that most competition stays within a cluster?
28. Compare switchback (time-based) designs, geo/market-level randomization and cluster randomization for an Airbnb search or pricing change. [Likely]
    - Why are switchbacks awkward when there is a long lead time between booking and stay?
    - With only a few hundred markets, how do you get enough power from a market-level experiment?
29. How would you detect interference empirically, for example by running the same treatment at two allocation levels (10% versus 50%)? [Likely; 13]
    - What would you conclude if the effect at 50% allocation is half the effect at 10%?
30. Market conditions matter. When supply is tight (peak dates), interference bias grows. How would you account for supply/demand balance in the analysis? [Likely; 13]
    - Would you report the effect separately for supply-constrained and demand-constrained markets?
31. A change makes guests book earlier. Total bookings over the quarter may not change, but the A/B test shows a lift. How would you detect this kind of intertemporal substitution (pull-forward)? [Likely]
32. How would you experiment on a host-side product, such as a new listing-quality nudge, when the outcome you care about (bookings) is driven by guests who are not in your experiment? [Likely]
33. Network effects among guests are weak on Airbnb, but interference through inventory is strong. Contrast this with social networks, and explain how the remedies differ. [Likely]
34. How would you use a market-level holdout ("launch everywhere except these N cities") to validate the global effect of a ranking system after launch? [Likely]
    - How do you choose holdout markets, and how do you analyze them (difference-in-differences, synthetic control)?
35. When randomization at the right level is impossible, how would you estimate the causal effect of a marketplace-wide change? [Likely; 14]
    - How does a counterfactual model (predict what would have happened without the change) work, and how do you validate it?
    - What assumptions does synthetic control rely on, and how would you check them?

### D. Variance reduction: CUPED, stratification and regression adjustment

36. Explain CUPED. What is it, why does it reduce variance, and why does it not bias the treatment effect estimate? [Likely; 15]
    - How do you choose the covariate, and what is the optimal theta?
    - Why must the covariate be measured before the experiment starts?
37. Booking is rare, and many users have no pre-period activity (new or logged-out users). How does CUPED perform for them, and what would you do instead? [Likely]
    - Could you use a model-predicted covariate, for example a predicted booking probability (CUPAC-style)? What are the risks?
38. How do CUPED and regression adjustment with several covariates relate? When would you use one over the other? [Likely]
39. Explain stratified sampling versus post-stratification for variance reduction. Which strata would you use for Airbnb (market, platform, new versus returning guests)? [Likely]
40. You applied CUPED and the variance fell by only 3%. Why might that happen for a booking metric, and what else would you try? [Likely]
41. How does variance reduction change your sample-size calculation and your experiment duration? [Likely]
42. Can you combine CUPED with ratio metrics and with the delta method? What do you need to be careful about? [Likely]
43. When would you choose a more sensitive design over variance reduction in the analysis? Examples include interleaving, paired designs and within-user designs. [Likely; 8]

### E. Interleaving for search ranking

44. What is interleaving, and why is it much more sensitive than an A/B test for comparing rankers? [Likely; 8]
    - Compare team-draft interleaving with balanced interleaving.
    - How do you assign credit for a booking, as opposed to a click, in an interleaved result list?
45. Airbnb has described using interleaving to speed up search-ranking experiments. How would you set it up so that the interleaving verdict agrees with A/B results? [Likely; 8]
    - How would you measure agreement between interleaving and A/B tests over a set of historical launches?
    - What would you do if interleaving says ranker B wins but the A/B test is flat?
46. Airbnb search is not a single click-and-done session. Guests search many times over days, and listings appear on map views and across pages. What complications does this create for interleaving? [Likely; 8]
    - How would you handle the same listing appearing in both rankers' lists, and duplicates across searches?
    - How do you attribute a booking that happens days later?
47. What can interleaving not tell you? Give examples of ranking changes that must still be A/B tested (for example, changes to price display, diversity or result-set size). [Likely; 8]
48. Design a two-stage experimentation funnel: offline evaluation, then interleaving, then A/B test. What are the gates between stages? [Likely; 2][8]
    - Offline NDCG improves but interleaving is flat. What is your hypothesis?

### F. Offline-online gap and ML-specific experimentation

49. Your new ranking model improves offline NDCG by 2%, but the A/B test shows no booking lift. What are the possible reasons, and how do you debug? [Likely; 2][5]
    - Could position bias or feedback loops in the training data explain it?
    - How would you tell a modeling problem apart from an experiment-sensitivity problem?
50. Airbnb's deep-learning search papers describe many launches that improved offline metrics but failed online. How would you build an offline metric that better predicts online wins? [Likely; 2][5]
51. When training data come from the control model's own rankings, the treatment model is evaluated on data biased toward control. How does this affect A/B results, and how would you mitigate it? [Likely]
    - What role do exploration traffic or randomized-result buckets play?
52. How would you use off-policy evaluation (counterfactual estimators such as inverse propensity scoring) to pre-screen ranking models before an online test? What does it require from logging? [Likely]
53. Treatment and control models share a training pipeline. After launch, the new model's data feeds back into retraining. How would you measure that feedback-loop effect? [Likely]
54. How would you A/B test a model whose outputs are cached or precomputed, such as listing embeddings or nightly batch scores? [Likely; 16]
    - What contamination can occur if the treatment's embeddings are also used by control's similar-listings carousel?
55. How would you run an experiment on an ML model that serves several surfaces (search, similar listings, email recommendations) at once? [Likely]
56. How would you test a latency-increasing model (larger neural net) when latency itself hurts conversion? [Likely]
    - How would you separate the model-quality effect from the latency effect?
57. How would you experiment on a fraud or trust model, where false positives block real guests and the true outcome (fraud, chargebacks) arrives weeks later? [Likely; 17]
    - Is holding out a random set of transactions from the fraud model ethical or practical? What would you propose?

### G. Sequential testing, peeking and decision rules

58. What goes wrong when a team peeks at an A/B test every day and stops as soon as p < 0.05? Quantify it roughly. [Likely; 1]
59. Explain sequential testing approaches (group sequential with alpha spending, always-valid p-values or mSPRT). Which would you build into an experimentation platform, and why? [Likely; 18]
    - What do you give up in power compared with a fixed-horizon test?
    - How would you explain an "always-valid" confidence interval on a dashboard?
60. Airbnb's 2014 post proposed a dynamic p-value threshold that is stricter early on. How does that compare with modern sequential methods? [Likely; 1]
61. When is early stopping for harm appropriate, versus early stopping for benefit? [Likely]
62. A test has run for its planned duration, and the p-value is 0.07. The PM wants two more weeks. What do you advise? [Likely]
63. How would you apply a Bayesian A/B testing approach to booking conversion? When does it help decision-making, and when does it mislead? [Likely]
    - How would you set priors from Airbnb's history of past experiments?
64. Explain the winner's curse in experimentation. Airbnb has written about correcting for selection bias when summing the impact of launched winners. How would you correct it? [Likely; 19]
    - Leadership adds up the measured lifts of 40 launched experiments, and the total far exceeds the actual business growth. Why?
65. How would you control the false discovery rate across hundreds of experiments and metrics on a platform? [Likely]

### H. Novelty, primacy, learning and time effects

66. How do you detect novelty or primacy effects in an Airbnb experiment, for example a redesigned listing page? [Likely]
    - How would you plot and test the treatment effect over time or by days since first exposure?
    - Why are novelty effects harder to see when most guests visit Airbnb only a few times a year?
67. Guests take days to weeks to book. How would you distinguish a real long-term effect from a novelty spike in such a sparse-visit product? [Likely]
68. How do hosts learn and adapt to a new host-side feature, such as changing prices after a pricing tip? How would you design the experiment to capture the post-adaptation equilibrium? [Likely; 7]
69. How would you handle seasonality and holidays, for example a test that spans the summer peak or the New Year period? [Likely]
    - Would you pause, extend or reweight?
70. Experiments run during an external shock (for example a pandemic-scale demand drop or a major event in a city). How do you decide if the results still generalize? [Likely]
71. How would you use a long-term holdback after launch to measure decay in a feature's effect? [Likely]

### I. Heterogeneous effects, segments and the launch decision

72. The overall effect is flat, but new guests show +3% and returning guests show -2%. How do you decide whether to launch, and whether to personalize? [Likely]
    - How do you avoid Simpson's paradox in this reading?
    - How do you know the segment effect is not a multiple-testing artifact?
73. How would you estimate heterogeneous treatment effects with ML (for example causal forests or uplift models), and how would you validate them? [Likely]
74. A treatment helps in large markets and hurts in small markets. What could explain this in a marketplace, and what would you do? [Likely]
75. The experiment improves bookings but reduces revenue per booking because guests choose cheaper stays. How do you make the launch call? [Likely; 6]
76. Results are significant on iOS and web but not on Android. How do you reason about it? [Likely; 1]
77. Walk me through the launch-review discussion you would lead when the primary metric is up, one guardrail is slightly down and the result is borderline significant. [Likely; 4]
    - What would you write in the experiment review document?
    - How do you communicate uncertainty to a VP who wants a single number?
78. How would you size the actual business impact of a launch, as opposed to the experiment's measured lift? Consider trigger rate, ramp-up, interference and decay. [Likely; 19]

### J. Experiment infrastructure and ML-platform design for experiments

79. Design the assignment service for an experimentation platform that serves search, mobile apps and email, with consistent bucketing across surfaces. [Likely; 3]
    - How do layers or mutually exclusive groups work, and when do you need them?
    - How do you handle ramping from 1% to 50% without reshuffling users?
80. Design the metrics computation pipeline behind a system like ERF: daily aggregates for thousands of metrics across hundreds of experiments. [Likely; 3][11]
    - How do you make it incremental, backfill-able and consistent with the company's metric definitions?
    - How do you make CUPED and the delta method efficient at that scale?
81. How do you detect and handle interactions between concurrent experiments, for example two ranking experiments that both touch the same feature? [Likely]
82. How would you design an experiment-quality monitoring system that automatically checks SRM, logging gaps, pre-period imbalance and metric anomalies? [Likely; 4]
83. How would you architect shadow and canary deployment for a new ranking model before it enters an A/B test? [Likely]
    - What would you compare in shadow mode, and what kinds of failure would make you abort?
84. How would you store and expose historical experiment results so that teams can run meta-analyses, such as proxy validation or calibrating priors? [Likely]
85. How would you build self-serve power analysis into the platform, given that metric variances differ by market and season? [Likely]

### K. Pricing, Smart Pricing and host-side experiments

86. Design an experiment for a change to a host price-suggestion model. What is randomized, and what is measured? [Likely; 7][12]
    - Hosts may ignore suggestions. How does adoption affect your estimand (intent-to-treat versus compliers)?
    - How do you prevent treatment hosts' price changes from affecting control hosts' bookings?
87. How would you estimate the price elasticity of demand from experiments without running illegal or unfair price discrimination on guests? [Likely]
    - Why do companies like Airbnb generally avoid randomizing the price shown to guests, and what do they do instead?
88. A pricing experiment shows host revenue up 4% in treatment, but marketplace-wide bookings are flat. How do you interpret this? [Likely; 12]
89. How would you test a change in guest fees or the display of total price (for example, showing all-inclusive prices)? Which interference and regulatory concerns apply? [Likely]

### L. Fairness, ethics and trust in experimentation

90. How would you check that an experiment does not harm a protected group of guests or hosts, given that you may not observe protected attributes directly? [Likely; 20]
    - How might perceived race or gender be measured under privacy constraints (for example, in Airbnb's Project Lighthouse work), and how would that feed into experiment analysis?
91. Which experiments should not be run at all, or need special review, on a travel/housing marketplace? [Likely]
92. How would you experiment on a trust and safety intervention, such as party prevention or booking restrictions, where the main outcome is a rare, severe harm? [Likely; 17]

### M. Past-project deep dive: experimentation probes

93. Tell me about an experiment you ran where the result surprised you. What did you do next? [Likely]
94. Tell me about a time an A/B result was wrong or misleading, for example because of a bug, SRM, interference or novelty. How did you find out? [Likely]
95. Describe a time you pushed back on launching a feature that had a "statistically significant" win. [Likely]
96. Describe a time you changed how your team or org runs experiments, such as a new metric, a new method or new tooling. What was the impact? [Likely]
97. What was the smallest effect you ever detected reliably, and what did you do to get enough sensitivity? [Likely]
98. How did you decide between running an experiment and shipping on offline evidence? Give a time you chose not to A/B test. [Likely]
99. How do you work with data scientists, PMs and engineers when you disagree about how to read an experiment? [Likely]

### N. Quick-fire concept checks (likely as follow-ups)

100. What is the difference between statistical significance and practical significance? How do you set the latter? [Likely]
101. Explain Type I and Type II errors, and the cost of each for a ranking launch at Airbnb. [Likely]
102. When would you use a one-sided test? [Likely]
103. What is the stable unit treatment value assumption (SUTVA), and where does it fail at Airbnb? [Likely; 13]
104. Why is the t-test usually fine for non-normal metrics at Airbnb scale, and when does it break? [Likely]
105. What is the difference between relative and absolute lift, and how do you build a confidence interval for relative lift? [Likely]
106. What is a holdout group versus a control group? [Likely]
107. What is the difference between a cluster-randomized design's unit of randomization and its unit of analysis, and how do you compute the standard errors? [Likely]
108. What is Twyman's law, and how does it apply to experiment review? [Likely]

## References

None of the sources below could be opened while writing this chapter. The web-search budget was used up and WebFetch could not resolve medium.com or igotanoffer.com. Each entry is listed from prior knowledge. Where the exact article URL was not known with confidence, the entry gives the title and points to the publication's index instead of a guessed link. Confirm every reference before relying on it.

1. "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
2. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD 2019. https://arxiv.org/abs/1810.09591
3. "Scaling Airbnb's Experimentation Platform" (describes ERF, the Experiment Reporting Framework), Airbnb Tech Blog, around 2017. URL not verified. Search via https://medium.com/airbnb-engineering
4. "Designing Experimentation Guardrails", Airbnb Tech Blog, around 2021-2022. URL not verified. Search via https://medium.com/airbnb-engineering
5. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv / KDD 2020. https://arxiv.org/abs/2002.05515
6. "How Airbnb Measures Future Value to Standardize Tradeoffs", Airbnb Tech Blog, 2022. URL not verified. Search via https://medium.com/airbnb-engineering
7. Ye et al., "Customized Regression Model for Airbnb Dynamic Pricing", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219830
8. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Tech Blog (Medium), 2022. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
9. "Airbnb Interview Process & Questions", IGotAnOffer, 2023-2024. https://www.igotanoffer.com/blogs/tech/airbnb-interview-process
10. "Airbnb Machine Learning Engineer Interview Questions", Glassdoor (candidate reports), ongoing. https://www.glassdoor.com/Interview/Airbnb-Machine-Learning-Engineer-Interview-Questions-EI_IE391850.0,6_KO7,32.htm
11. "How Airbnb Achieved Metric Consistency at Scale" (Minerva metrics platform), Airbnb Tech Blog, 2021. URL not verified. Search via https://medium.com/airbnb-engineering
12. Holtz, Lobel, Liskovich and Aral, "Reducing Interference Bias in Online Marketplace Pricing Experiments" (Airbnb collaboration), working paper / arXiv, 2020. URL not verified. Search the title on arXiv or SSRN.
13. Johari, Li, Liskovich and Weintraub, "Experimental Design in Two-Sided Platforms: An Analysis of Bias", Management Science, 2022 (arXiv preprint 2020). URL not verified. Search the title on arXiv.
14. "Artificial Counterfactual Estimation (ACE): Machine Learning-Based Causal Inference at Airbnb", Airbnb Tech Blog, around 2021-2022. URL not verified. Search via https://medium.com/airbnb-engineering
15. Deng, Xu, Kohavi and Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED), WSDM 2013. URL not verified. Search the title in the ACM Digital Library.
16. Grbovic and Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
17. "Fighting Financial Fraud with Targeted Friction", Airbnb Tech Blog, 2018. URL not verified. Search via https://medium.com/airbnb-engineering
18. Johari, Pekelis and Walsh, "Always Valid Inference: Bringing Sequential Analysis to A/B Testing", arXiv 2015 / Operations Research 2022. URL not verified. Search the title on arXiv.
19. "Selection Bias in Online Experimentation" (winner's curse correction), Airbnb Tech Blog, around 2019. URL not verified. Search via https://medium.com/airbnb-engineering
20. Airbnb "Project Lighthouse" technical paper and posts on measuring discrimination with perceived race under privacy constraints, Airbnb, 2020. URL not verified. Search via https://medium.com/airbnb-engineering or the Airbnb Newsroom.
21. Kohavi, Tang and Xu, "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing", Cambridge University Press, 2020. General background for SRM, A/A tests, Twyman's law and OEC questions. URL not verified.
