# 08. Statistics, Causal Inference and A/B Testing: Hypothesis Tests, Power, Variance Reduction, Weblab, Bandits and Forecasting

> Scope: the statistics and experimentation content an L5 Applied Scientist (Applied Scientist II) candidate meets in the Amazon onsite loop. This chapter covers hypothesis testing, power and sample size, metric design, variance reduction, experiment pitfalls at Amazon scale, causal inference without clean randomization, bandits, and forecasting. Questions only; no answers.

> Sourcing note: the live web search and fetch budget for this run was exhausted, and outbound fetches were blocked by network policy. Every [Reported] tag below cites a source that earlier chapters of this bank (01-06) verified and summarized. Questions without a direct candidate report are tagged [Likely], with the reasoning given. No question is presented as reported unless a cited source reports it.

## Format notes for this round

- **There is no dedicated "stats round" in most L5 AS loops.** Amazon's official prep page describes the onsite as four 55-minute interviews with members of the science community. Each interviewer is assigned competencies and typically asks 2-3 Leadership Principle questions [1]. Candidates more often report 5 rounds of 55-60 minutes for AS II: ML breadth, science depth, science application or ML design, coding, and a bar raiser [2][3][15].
- **Where statistics shows up:**
  - **ML/Science Breadth round:** p-values, statistical tests, CLT, and "design an A/B test" are reported as part of the breadth survey [4][8][9].
  - **Science Application / ML design round** (often the hiring manager, and usually present at L5 [20]): candidates are asked how they would A/B test the launch, choose metrics and handle conflicting metrics [5][6][10].
  - **Phone screen:** an L5 phone screen mixed coding, statistics and LPs [12], and a 2025 first-round AS interview covered ML and statistics alongside LLM theory [21].
  - **Bar raiser:** a Principal TPM bar raiser asked a few stats checks, such as interpreting a p-value or a confidence interval [7].
  - **Team-specific depth:** supply-chain, pricing and forecasting teams (SCOT) ask forecasting and inventory questions such as the newsvendor problem [16][17][18].
- **The L5 bar.** L5 candidates are expected to own a problem end to end. That includes framing a vague business goal, defining offline and online metrics, designing the launch experiment and making the ship or no-ship call when metrics disagree [6][20]. In stats questions, interviewers probe past definitions into judgement. Expect follow-ups like "the test is underpowered, now what?", "the primary metric is up but a guardrail is down", and "you can't randomize, now what?" [5][6][10].
- **Experimentation at Amazon (context).** Amazon's internal A/B testing platform is widely known as Weblab. Teams treat online experiments as the launch gate for ranking, recommendation, ads, pricing and Alexa/Rufus changes. Interviewers outside retail may phrase this generically as "how would you A/B test this?" Questions that mention Weblab are tagged [Likely], because no source in this bank reports a candidate being asked about Weblab by name.

---

## A. Probability and statistical foundations

1. State the Central Limit Theorem. Why does it matter for A/B tests? [Reported] [8]
   - When does the CLT approximation break down for a revenue-per-visitor metric with a heavy right tail?
   - How large must n be before you trust a normal approximation for a skewed metric?
2. What is Bayes' rule? Explain the prior, likelihood and posterior. [Reported] [13]
   - Walk through a base-rate example: a fraud model flags 1% of orders, fraud prevalence is 0.1%, and recall is 90%. What is the precision?
3. Work through a coin-flip probability question on the whiteboard. [Reported] [14] (the exact wording was not captured)
   - Derive the MLE for p after k heads in n tosses. What is its variance?
   - How would you build a confidence interval for p when k = 0?
4. State Hoeffding's inequality and use it. [Reported] [11]
   - How many samples do you need to estimate a click-through rate to within plus or minus 0.1% with 95% confidence?
   - How does a Hoeffding bound compare with a CLT-based interval in practice?
5. What is the difference between a frequentist confidence interval and a Bayesian credible interval? [Likely] (follows from the reported CI interpretation probe [7])
6. What is the law of large numbers, and how does it differ from the CLT? [Likely]
7. Explain the delta method. When do you need it for an A/B test metric? [Likely] (ratio metrics like CTR or revenue per session are standard in Amazon retail and ads)
   - Derive the variance of a ratio metric such as clicks / impressions when the randomization unit is the customer.
8. What is the bootstrap? When would you use it instead of an analytic standard error? [Likely]
   - How do you bootstrap correctly when observations are clustered by customer?
9. Compare the Poisson, negative binomial and zero-inflated distributions for modeling orders per customer. [Likely] (retail count data)

## B. Hypothesis testing and p-values

10. What is a p-value? Interpret it in the context of an A/B test. [Reported] [4][9]
    - What does a p-value of 0.03 NOT tell you?
    - Explain it to a product manager with no statistics background.
11. Interpret a p-value or a confidence interval. [Reported] [7] (asked by a Principal TPM bar raiser)
    - What is a common misinterpretation of a 95% confidence interval?
12. Walk through hypothesis testing end to end: which statistical tests would you use, and when? [Reported] [8][9]
    - When would you use a t-test, a z-test, a chi-squared test or a non-parametric test?
    - Welch's t-test or Student's t-test: which is the safer default, and why?
13. Explain hypothesis testing, then design an A/B test for a model change. [Reported] [5][6]
14. Define Type I and Type II errors. Which is more costly for a checkout-flow change compared with a recommendation widget? [Likely]
15. Compare one-sided and two-sided tests. When, if ever, is a one-sided test justified for a launch decision? [Likely]
16. How would you test whether conversion rates differ between two arms? Compare a two-proportion z-test, a chi-squared test and Fisher's exact test. [Likely] (conversion is the core retail metric)
17. When would you use Mann-Whitney U or a permutation test instead of a t-test? [Likely]
    - Does Mann-Whitney test a difference in means? What does it actually test?
18. You observe a statistically significant 0.05% lift on a site with hundreds of millions of sessions. How do you think about statistical versus practical significance? [Likely] (Amazon traffic scale)
19. What is a paired test? When could an Amazon experiment use a paired design? [Likely]
20. How do you test for equivalence or non-inferiority, for example to show that a cheaper model is "no worse"? [Likely] (cost-reduction launches such as model distillation)
    - How do you choose the non-inferiority margin?

## C. A/B test design, power and sample size

21. How would you design an A/B test for a model launch? [Reported] [9][10]
    - How do you choose the randomization unit, primary and guardrail metrics, and sample size or power?
    - What do you do if the test is underpowered or the metric is heavy-tailed?
22. Treatment improves the target metric, but a population-level metric drops. What do you do? [Reported] [6]
    - How would you decide whether the drop is real, a segment effect, or noise?
    - Who do you bring into the decision, and what do you write in the launch doc?
23. Derive the sample-size formula for a two-sample test of means. Which inputs drive it the most? [Likely]
    - How does required sample size scale with the minimum detectable effect (MDE)?
24. How do you choose the MDE for an experiment? [Likely]
    - Who should own that number, science or the business?
25. Your power analysis says you need 8 weeks, but the team wants an answer in 2. What are your options? [Likely]
26. How do you choose the randomization unit (customer, session, page view, product, seller, marketplace)? [Likely] (the Amazon marketplace has customers, sellers and products as distinct units)
    - What goes wrong if the analysis unit differs from the randomization unit?
27. How long should an experiment run? How do you account for day-of-week effects and paydays? [Likely]
28. What is an A/A test, and what would you learn from one? [Likely]
    - Your A/A test shows a significant difference. Walk through your debugging.
29. What is a sample ratio mismatch (SRM)? How do you detect and diagnose it? [Likely]
    - The split is 50.4/49.6 on 20 million users. Is that a problem?
30. How do you handle triggering and dilution, for example when only 3% of users ever see the changed widget? [Likely] (retail page widgets)
    - How do you translate a triggered-population effect into a site-wide impact estimate?
31. How would you run an experiment for a change that affects only Prime members, or only a single marketplace? [Likely]

## D. Metrics: primary, guardrail and long-term

32. Choose an overall evaluation criterion for a change to product search ranking. [Likely] (search is a core AS team)
    - Why might clicks be a poor primary metric? What about revenue?
33. What guardrail metrics would you add for a recommendation change? [Likely]
    - Consider latency, returns, customer-service contacts, and seller-side metrics.
34. How do you define a metric that captures long-term customer value when the experiment runs for only two weeks? [Likely]
    - How would you build and validate a surrogate or proxy metric?
35. Your offline metric (NDCG or AUC) improved, but the online A/B test is neutral. What could explain it? [Likely] (ranking and recommendation teams launch through online tests)
36. How do you evaluate a metric's sensitivity, or directionality, before using it as a launch criterion? [Likely]
37. How would you measure a change whose effect shows up as fewer returns 30 days later? [Likely] (delayed outcomes in retail)
38. What is Simpson's paradox? Give an Amazon-flavoured example where it could reverse an experiment readout. [Likely]
39. A metric moved by +2% in the US and -1% in the UK. How do you decide whether to launch globally? [Likely]

## E. Variance reduction and sensitivity

40. What is CUPED, and why does it reduce variance? [Likely] (CUPED appears as a probe on the reported "design an A/B test" question in this bank's design chapter)
    - Which pre-period covariate would you choose, and what happens for new customers with no pre-period data?
    - How does CUPED relate to regression adjustment (ANCOVA)?
41. Compare stratified sampling, post-stratification and regression adjustment as variance reduction methods. [Likely]
42. How do you handle heavy-tailed metrics like revenue per customer? [Likely] (the heavy-tail probe is in [10])
    - Compare winsorization, capping, log transforms and switching to a quantile metric. What do you trade off with each?
    - Does capping introduce bias, and when is that acceptable?
43. When would you use interleaving instead of an A/B test for a ranking change? [Likely] (search and ads ranking teams)
    - What can interleaving tell you, and what can it not?
44. How can you use ML predictions as control variates for variance reduction? [Likely]
45. Your experiment is underpowered and you cannot get more traffic. List five ways to increase sensitivity. [Likely]

## F. Multiple testing, peeking and sequential testing

46. Your dashboard tracks 40 metrics. Three are significant at 0.05. What do you conclude? [Likely]
    - Compare Bonferroni, Holm and Benjamini-Hochberg. When is each appropriate?
47. How would you handle many simultaneous variants (multiple testing)? [Likely] (follow-up to a reported A/B design question; see [6])
48. A PM checks results every day and wants to stop when p < 0.05. What is wrong with that? [Likely]
    - How would you design a sequential test (alpha spending, group sequential, mSPRT or always-valid p-values) to allow safe early stopping?
49. You slice results by 20 segments and one shows a large effect. How do you decide whether to believe it? [Likely]
    - How would you estimate heterogeneous treatment effects in a principled way?
50. Many teams run experiments on the same pages at the same time. How do you handle interactions between concurrent experiments? [Likely] (Amazon runs many concurrent experiments across teams)

## G. Pitfalls at Amazon scale: interference, novelty and marketplaces

51. How do you measure a recommender when network or interference effects break user-level randomization? [Likely] (follow-up to a reported design question; see [6])
52. How would you design an experiment to measure the effect of a pricing algorithm when prices interact across products? [Likely] (pricing science teams; see [18])
    - When would you use switchback, cluster or marketplace-level randomization?
53. A new seller-facing feature changes how sellers price, which affects buyers in control. How do you get an unbiased estimate? [Likely] (two-sided marketplace)
54. How would you experiment on a fulfillment or inventory-placement policy when products share warehouses and trucks? [Likely] (SCOT; see [18][19])
55. What are novelty and primacy effects? How do you detect them in a two-week test? [Likely]
56. The test wins in week 1 and is flat in week 3. What do you report? [Likely]
57. How do you A/B test an ads model when advertiser budgets are shared between treatment and control? [Likely] (ads teams; see [8])
    - What is budget cannibalization, and how do budget-split designs address it?
58. How would you A/B test a change to Alexa or Rufus, where the conversation spans many turns and outcomes are sparse? [Likely] (LLM product teams)
    - What would your primary metric be, and how would you combine it with human or LLM-judge evaluation?
59. When is an A/B test the wrong tool, and what would you do instead? [Likely]

## H. Weblab and the launch decision

60. Walk me through how you would set up, monitor and close out an experiment on Amazon's internal experimentation platform (Weblab) for a model change. [Likely] (Weblab is Amazon's internal A/B platform; no source here reports it being named in an interview)
    - What would you check on day 1, and what before the launch decision?
61. How would you ramp a risky change (1%, 5%, 50%), and what triggers a rollback? [Likely]
62. What should go in a launch recommendation, a narrative-style write-up for leadership, after an experiment? [Likely] (Amazon's written narrative culture)
    - How do you present a result with a wide confidence interval that includes zero?
63. The result is significant on the primary metric, but the effect is smaller than the cost of serving the new model. How do you decide? [Likely]
64. Leadership wants to launch despite a non-significant result. How do you respond? [Likely] (also a Leadership Principle probe for "Have Backbone; Disagree and Commit")
65. After launch, how do you check that the measured lift persists? [Likely]
    - What is a holdback (long-term holdout) group, and how big should it be?
66. How would you estimate the cumulative impact of 30 launched experiments in a year? Why might the sum of individual lifts overstate it? [Likely]
    - What are the winner's curse and regression to the mean, and how do you correct for them?

## I. Causal inference without clean randomization

67. How would you estimate the causal effect of a promotion or price change without a clean experiment? [Likely] (pricing and promotion teams; see [18])
    - When would you use difference-in-differences, synthetic control, instrumental variables or double ML?
68. Explain difference-in-differences. What is the parallel-trends assumption, and how would you check it? [Likely]
    - What goes wrong with staggered adoption, for example a feature rolled out marketplace by marketplace?
69. Explain synthetic control. When would you use it to measure a launch in a single country? [Likely]
70. What is an instrumental variable? Give a plausible instrument for estimating price elasticity on Amazon. [Likely]
    - How do you defend the exclusion restriction?
71. How do you estimate price elasticity when prices are set by your own algorithm (endogeneity)? [Likely] (pricing system design; see [18])
72. Explain propensity score matching and inverse propensity weighting. What assumptions do they need? [Likely]
    - How do you check overlap (positivity)?
    - What does doubly robust estimation buy you?
73. What is regression discontinuity? Give an example from a free-shipping threshold. [Likely] (free-shipping thresholds are a familiar Amazon retail mechanic)
74. How would you measure the effect of joining Prime on customer spending? [Likely]
    - Why is a naive comparison of Prime and non-Prime members biased, and in which direction?
75. What is double or debiased machine learning, and why use cross-fitting? [Likely]
76. Draw a causal DAG for "recommendation exposure leads to purchase". Name the confounders, colliders and mediators. [Likely]
77. How would you measure the incremental impact (lift) of an ad campaign for an advertiser? [Likely] (ads teams; see [8])
    - What are the trade-offs between ghost ads, geo experiments and observational methods?
78. What is uplift modeling? How does it differ from response modeling? [Likely]
    - How would you evaluate an uplift model (Qini or uplift curves)?
79. How do you validate a causal estimate when you can never observe the counterfactual? [Likely]

## J. Bandits and adaptive experimentation

80. When would you use a multi-armed bandit instead of an A/B test? [Likely] (ads creative and widget selection)
    - What do you give up in inference when you use a bandit?
81. Explain epsilon-greedy, UCB and Thompson sampling. How do their exploration strategies differ? [Likely]
    - Write the Beta-Bernoulli posterior update for Thompson sampling.
82. Define regret. What regret bounds do UCB and Thompson sampling achieve? [Likely] (Hoeffding bounds are reported [11] and underpin UCB)
83. How would you design a contextual bandit for recommending widgets on the homepage? [Likely]
    - How do you handle delayed rewards, such as purchases that happen hours later?
    - How do you handle non-stationarity, for example during Prime Day?
84. How do you evaluate a new policy offline from logged bandit data? [Likely]
    - Compare inverse propensity scoring, self-normalized IPS and doubly robust estimators. When does each have high variance?
    - What must the logging policy record for off-policy evaluation to be possible?
85. A bandit is optimizing CTR on deal-page creatives and converges to clickbait. How do you fix the reward? [Likely]
86. How would you frame budget pacing or bidding as a bandit problem or as constrained optimization? [Likely] (ads teams; see [8])

## K. Forecasting and time series

87. What is the newsvendor problem, and what is its solution? [Reported] [16]
    - How does a quantile forecast map to the newsvendor critical ratio?
88. How would you forecast demand for millions of products, including new ones with no history? [Likely] (SCOT forecasting teams; see [17][18][19])
    - What does a global model trained across products buy you over per-product models?
89. Why use probabilistic or quantile forecasts rather than point forecasts for inventory decisions? [Likely] (see [17])
    - What is quantile (pinball) loss, and how do you evaluate calibration of P50 and P90 forecasts?
90. Compare MAPE, sMAPE, WAPE, RMSE and quantile loss. Which would you use for intermittent demand? [Likely]
    - Why does MAPE behave badly for low-volume products?
91. How do you backtest a forecasting model correctly? [Likely]
    - What is an expanding-window or rolling-origin evaluation, and how does leakage creep in?
92. How do you forecast demand during Prime Day or other promotions, when history is sparse and distorted? [Likely] (see [17])
    - How do you separate a promotion lift from the baseline?
93. Demand is censored when products are out of stock. How do you correct for it? [Likely] (in-stock is a core SCOT metric)
94. Compare ARIMA, exponential smoothing, and neural forecasters such as DeepAR- or transformer-style models. When does each win? [Likely] (Amazon publishes neural forecasting research; see [17])
95. Explain stationarity, autocorrelation and seasonality. How do you test for them? [Likely]
96. How would you forecast a hierarchy (product to category to total) so the forecasts add up? [Likely]
97. How do you evaluate a forecasting system against a business outcome (in-stock rate, holding cost) rather than accuracy alone? [Likely] (see [18])
    - How would you A/B test a new forecast when buying decisions in treatment and control share inventory?
98. How would you detect anomalies or structural breaks in a demand time series, such as the start of a pandemic? [Likely]

## L. Applied stats case questions (mixed, interview-style)

99. Orders dropped 5% week over week. Walk me through how you would diagnose it. [Likely]
100. A product page change increased add-to-cart but not purchases. What hypotheses would you test, and how? [Likely]
101. You are asked to estimate how many customers would buy a new Echo device at a given price. What data and statistical approach would you use? [Likely]
102. Customer reviews have an average rating of 4.6 from 5 reviews and 4.3 from 2,000 reviews. How should ranking treat them? [Likely] (product reviews ranking)
    - How would you apply Bayesian shrinkage, or a lower confidence bound, to the rating?
103. How would you estimate the false-positive rate of a fraud model when you only see labels for orders you blocked? [Likely] (selective labels in fraud and trust teams)
104. Two models have AUCs of 0.812 and 0.815 on the same test set. Is the difference meaningful? [Likely]
    - How would you get a confidence interval for the AUC difference (DeLong or bootstrap)?
105. How would you design a survey or human-labelling study to estimate the defect rate of search results, with a confidence interval? [Likely]
    - How many labelled samples do you need, and how do you handle disagreement between labelers?

---

## References

1. "Applied Scientist Interview Prep", amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. "Amazon Applied Scientist II Interview Experience & Questions", Glassdoor, reports from 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
3. "Amazon Applied Scientist Interview Experience & Questions", Glassdoor, reports from 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
4. "Amazon Web Services Applied Scientist Interview Experience & Questions", Glassdoor (explain transformers, p-value, LoRA). https://www.glassdoor.com/Interview/Amazon-Web-Services-Applied-Scientist-Interview-Questions-EI_IE7470741.0,19_KO20,37.htm
5. Amazon interview review RVW99272421 (London Applied Scientist, Jul 2024; hypothesis testing and A/B test design), Glassdoor, 2024. https://static.glassdoor.it/Interview/Amazon-Interview-E6036-RVW99272421.htm
6. "The 2026 Amazon Applied Scientist Interview Guide", Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
7. "My final interview for Amazon Applied Scientist-II will be taken by Principal TPM", Blind. https://www.teamblind.com/post/my-final-interview-for-amazon-applied-scientist-ii-will-be-taken-by-principal-tpm-cant-understand-what-tpm-will-ask-ryaa3tjo
8. "Applied scientist interview at Amazon Ads", Blind. https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
9. "What to expect in the Amazon Applied Scientist interview?", Grapevine, circa 2024 (AS2 breadth: statistical tests, p-value, A/B testing). https://www.grapevine.in/post/what-to-expect-in-the-amazon-applied-scientist-interview-a649970d-b602-4e78-ac34-9d5fe491a7cd
10. "Amazon applied scientist interview-onsite", Blind, circa 2024. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
11. "Amazon Applied Scientist Interview" (process, questions, prep), IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
12. "Amazon L5 Applied Scientist Fulltime Tech Phone Screen Overview", 1point3acres. https://www.1point3acres.com/interview/thread/1170176
13. Akanksha Paul, "Interview Experience for Applied Scientist at Amazon", Medium, 2020. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
14. "Science Breadth Interview @Amazon", Blind, 2024. https://www.teamblind.com/post/science-breadth-interview-amazon-hfwqznrb
15. "Amazon Applied Scientist (L5) Virtual Onsite", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
16. "Amazon Supply Chain Interview Experience & Questions" (newsvendor problem), Glassdoor. https://www.glassdoor.com/Interview/Amazon-Supply-Chain-Interview-Questions-EI_IE6036.0,6_KO7,19.htm
17. "Predicting what Amazon's customers will need tomorrow, next week, and beyond", Amazon Science. https://www.amazon.science/working-at-amazon/predicting-what-amazons-customers-will-need-tomorrow-next-week-and-beyond
18. "Applied Scientist II, SCOT Optimal Sourcing Systems, Buying Science Team" (Job ID 10468056), amazon.jobs. https://www.amazon.jobs/en/jobs/10468056/applied-scientist-ii-scot-optimal-sourcing-systems-buying-science-team
19. "reMARS revisited: Amazon's supply chain optimization", Amazon Science. https://www.amazon.science/latest-news/remars-revisited-amazons-supply-chain-optimization
20. "Amazon applied scientist interview loops", Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
21. "Amazon online 1st round, Bengaluru, Applied Scientist", LeetCode Discuss, May 2025 (LLM theory, transformer internals, ML and stats). https://leetcode.com/discuss/post/6871782/amazon-online-1st-round-bengaluru-applie-uiwk/
