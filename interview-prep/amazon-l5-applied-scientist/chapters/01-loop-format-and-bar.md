# 01. Loop Format and Hiring Bar: Amazon L5 Applied Scientist (Applied Scientist II)

> Scope: how the L5 Applied Scientist (AS II) process is structured, who sits in each seat, how the bar raiser and debrief work, how L5 is calibrated against L4 and L6, and the questions each seat tends to ask. Later chapters cover the technical content (ML breadth, depth, coding, ML design, LPs) in full. This chapter covers the format-defining questions and probes.
>
> Verification status (2026-10-14 review): a re-check of the cited URLs could not be completed. The egress proxy rejected every one of the 43 reference hosts, and no web search was available. The [Reported] tags and citations are therefore unchanged from the original research pass and have not been re-verified. No new questions were added, because none could be sourced.
>
> Research note: everything here comes from Amazon's own pages, candidate self-reports (Glassdoor, Blind, LeetCode, 1point3acres, Medium) and prep-vendor guides (IGotAnOffer, Interview Kickstart, Dataford and others). The live page fetcher could not reach these sites from this environment, so the content was read through search-engine excerpts of each page. Vendor guides are marketing material and are weighted below Amazon's own pages and first-person reports.

## Format notes (what to expect, 2025-2026)

### Pipeline

| Stage | What's reported | Sources |
|---|---|---|
| Recruiter call | Logistics, level target, team, and sometimes a light background check on your ML experience. | [22][24] |
| Technical phone screen(s) | Amazon says there are one or two screens of about 60 minutes each, run by "a senior leader on our team" (in practice usually a senior or principal scientist, sometimes the hiring manager). Candidates report a mix of ML fundamentals, a project walk-through, 1-2 LP questions, and sometimes one LeetCode easy/medium or a pandas/data-manipulation task. Some screens have no coding at all. | [1][9][20][27][33] |
| Onsite "loop" (now usually virtual on Chime) | Amazon's official page says "four 55-minute interviews ... with members of our science community." Candidate reports for AS II more often describe 5 rounds of 55-60 minutes, sometimes spread over 1-3 days. Some teams add a job talk of about 45-50 minutes plus Q&A. | [1][6][10][17][28] |
| Debrief | Interviewers submit written feedback and a vote independently. The bar raiser chairs the debrief, and the hiring manager and bar raiser make the final decision. | [3][4][21][30] |
| Team match (sometimes) | Some candidates pass the loop and are then told they need a team match before an offer letter. Others are "recycled" to another team's opening without re-interviewing. Some are asked to do another loop. | [11][18][34] |

### Typical L5 AS loop composition (as candidates report it)

- **ML/Science Breadth** (about 1 round): fast survey of classical ML, deep learning, statistics, and, increasingly, transformers, LLMs and GenAI. Reported examples include bias-variance, bagging vs boosting, transformer attention, BERT vs GPT, DPO vs RLHF, and how SAM works. [6][7][26]
- **ML/Science Depth** (about 1 round): a deep dive on one project or publication from your resume, where the interviewer keeps drilling down on your choices. [10][22][28]
- **Science Application / ML System Design** (about 1 round, often the hiring manager): a team-specific, open-ended problem. Blind and vendor sources say L5 loops usually include it, while L4 loops may not. [6][12][27]
- **Coding** (1 round, sometimes 2): usually LeetCode easy to medium, sometimes pandas/numpy, ML-from-scratch, or OOD. A Blind commenter says the bar is about SDE I level. [10][12][20][27]
- **Bar Raiser** (1 round): mostly LP/behavioral, run by a trained interviewer from outside the hiring org. It may also include technical probing. Candidates report BRs from non-tech orgs and Principal TPMs. [6][11][13]
- **Leadership Principles in every round:** Amazon's prep page says each interviewer typically asks 2-3 behavioral questions. Candidates report about 15 minutes of each round, and one L5 report put behavioral questions at over 60% of the loop. [1][19][27]

### Who owns what

- Each interviewer is pre-assigned a competency (breadth, depth, application, coding) plus about 2-3 Leadership Principles, so the loop covers the full LP set without repeating. Amazon's page confirms that interviewers are assigned competencies "based off of our Leadership Principles". The per-LP split comes from prep guides, not from Amazon. [1][36][23]
- **Hiring manager:** usually owns the Science Application / design round and team fit. With the BR, the HM makes the final hire/no-hire call. [3][37]
- **Bar raiser:** a volunteer interviewer who is nominated, trained, and shadows veteran BRs for several months. The BR comes from outside the hiring team, chairs the debrief, and holds a de facto veto, which former Amazonians say is rarely used explicitly. The BR's job is to judge whether the candidate is better than at least half of current employees at that level. That 50% bar is folklore quoted by one BR, not a published standard. [3][4][5][29][37]

### What the L5 bar is (calibration)

- **Level map:** AS I = L4, AS II = L5, Senior AS = L6. A PhD is often credited as about 2-3 years of experience, and new PhDs are frequently interviewed at L5. One example L5 posting asks for a Master's, or a PhD plus 5+ years of practical ML. Levels.fyi data puts typical L5 experience at 5-7 years. [15][31][32]
- **L5 vs L4:** L5 is expected to own projects end to end, including design, to make the technical calls, and to influence peers on the team, with some spillover to sister teams. Vendor debriefs say L5 LP answers face 2-3 layers of follow-up, against 1 layer at L4. [15][27]
- **L5 vs L6:** L6 needs proof of large scope, influence outside the team, mechanisms, and multiple strong examples. Blind commenters say the coding bar is similar at L5 and L6, and that the LP narratives are what differentiate them. [15][16][25]
- **Leveling outcomes:** downleveling from L6 to L5 and from L5 to L4 is common. A BR can push the level down even when the HM is happy (one L5 candidate's BR thought they were "at the cusp of L4 and L5"). One Blind comment says that a weak coding round with strong science can lead to a Research Scientist offer instead. [11][12][16]

---

## Questions

Tag legend:

- **[Reported]**: a candidate or source attributes this question, or a close paraphrase, to Amazon AS (or, where noted, Amazon DS or Senior AS) interviews.
- **[Likely]**: an inference grounded in the format, the LP rubric and the team's domain. It has not been seen verbatim.

### A. Recruiter call and technical phone screen

#### A1. Recruiter / logistics screen

1. Walk me through your background and why Amazon, why Applied Science, and why this team. [Likely]
   - Which Amazon products or science areas have you followed, and why?
   - Are you open to other teams if this one isn't the right fit (team match / recycle)?
2. How many years of hands-on applied ML experience do you have post-degree, and how much of it shipped to production? [Likely]
   - Which of your projects would you consider "L5 scope," and why?
3. Are you interviewing at other companies, and what is your timeline? [Likely]

#### A2. Technical phone screen (senior scientist or HM, about 60 min)

4. Walk me through the research or project on your resume you're most proud of. [Reported] [33][39]
   - Why that modeling approach over the obvious baseline?
   - What was the business metric, and how did the offline metric map onto it?
5. Explain the bias-variance trade-off and how it shows up in a model you've built. [Reported] [6][20]
   - How did you diagnose which side you were on?
6. How does hyperparameter tuning work, and how do you evaluate models? [Reported] [20]
   - How do you avoid leakage between tuning and evaluation?
7. Explain decision trees, batch normalization and word embeddings. [Reported] [39]
   - Why does batch norm behave differently at train and inference time?
8. Write out the matrix form of OLS and derive the closed-form solution. [Reported] [20]
   - What happens when X^T X is singular, and how does ridge fix it?
9. Load this dataset in pandas, clean it, engineer features, and fit a model. Which breadth questions does each step raise? [Reported] [9][20]
   - How would you handle missing values and class imbalance here?
10. Solve one LeetCode-style easy/medium problem (one candidate reported a graph problem, others arrays or trees). [Reported] [10][20]
    - What is the time and space complexity, and can you do better?
11. Tell me about a time you made a technical decision without complete data. This is also reported as a bar raiser question. [Reported] [26]
    - What would have changed your mind?
12. (Ads, 2024-25 report) Explain how transformers and attention work. One candidate with strong classical ML answers was rejected after not preparing for this. [Reported] [43]
    - How does a decoder-only LLM generate text at inference time?

### B. Job talk / research presentation (team-dependent)

13. Present a recent piece of your work, covering only your own contributions, in about 45-50 minutes with about 10 minutes of Q&A. [Reported] [17][34]
    - Which parts were yours, and which were your collaborators'?
    - What are the impact numbers, and how were they measured?
14. Why did this problem matter to the business or customer, and how did you know? [Likely]
    - If you had to ship a version in four weeks, what would you cut?
15. What baselines did you compare against, and why are they the right ones? [Likely]
    - Which ablation would most likely break your conclusion?
16. Which paper references does this build on, and what is novel versus engineering? [Reported] [17]
17. How would this work transfer to one of our team's problems? [Likely]
    - What data would you need from us that you don't see yet?

### C. ML/Science Breadth round

18. Compare bagging and boosting. When would each fail? [Reported] [6][7]
    - Why does boosting overfit noisy labels more readily?
19. Explain L1 and L2 regularization and their effect on the learned model. [Reported] [24]
20. Explain the transformer attention mechanism. [Reported] [7]
    - Why is standard self-attention expensive for long contexts, and how does Longformer-style attention help? [Reported] [7]
21. How does BERT differ from GPT? [Reported] [7]
22. Compare DPO and RLHF. [Reported] [7]
    - What failure modes of PPO-based RLHF does DPO avoid, and which does it introduce?
23. How does KV caching work during autoregressive inference? [Reported] [7]
24. Describe how SAM (Segment Anything Model) works and its key architectural components. [Reported] [6][26]
    - How would you improve CLIP for a product-image retrieval use case? [Reported] [26]
25. Compare gradient descent with Adam. When might a second-order optimizer help? [Reported] [22]
26. State and use Hoeffding's inequality. [Reported] [22]
27. Explain hypothesis testing and design an A/B test for a model change. [Reported] [9][24]
    - If treatment improves the target metric but a population-level metric drops, what do you do? [Reported] [24]
28. Cover supervised vs unsupervised learning, dimensionality reduction, neural network basics and evaluation metrics, in rapid fire. [Reported] [28]
29. Pick a technique from a field adjacent to your PhD and explain it at grad-course depth. [Likely]
    - Where does your knowledge stop, and how would you find out more?

### D. ML/Science Depth round

30. Walk through a project or publication from your resume, end to end. [Reported] [10][22]
    - Why this loss and this architecture? What did you try that didn't work?
    - What did you personally build, versus what the team built?
31. You proposed an algorithmic fix just now. Go one level deeper: derive it or justify it mathematically. [Reported] [7]
32. Which alternatives did you consider, and what trade-offs drove the final choice? [Reported] [22]
33. What was the business impact, and how did you attribute it to the model rather than to confounders? [Reported] [28]
34. If you restarted this project today with LLMs and foundation models available, what would change? [Likely]
35. What was the hardest unknown problem in that project, and how did you get unstuck? [Likely]
    - How did you know your solution was correct?
36. What did the data look like? How were labels produced, and how noisy were they? [Likely]
    - How did label noise bound your achievable accuracy?
37. Did the model hold up after launch? How did you monitor drift? [Likely]
    - Did the gains hold six months later? [Likely]

### E. Science Application / ML system design (often the hiring manager)

38. Design an agentic recommendation system for Amazon, including component architecture, model choices, and quality and safety evaluation. [Reported] [26]
39. Design a pipeline to improve personalized product recommendations on the Amazon homepage. [Reported] [26]
40. Design a GitHub Copilot-style coding agent (Senior AS, Boston report). [Reported] [7]
41. Design an enterprise RAG service. [Reported] [35]
42. Use computer vision to detect defective products in fulfillment-center warehouses. [Reported] [24]
43. Here's a vague business goal from our team. Frame it as an ML problem, choose metrics, and outline v1. [Likely]
    - What's your online/offline metric pair, and how do they disagree?
    - What's the cheapest baseline you would ship first?
44. Evaluate an LLM agent with metrics beyond accuracy: tool use, hallucination, calibration. [Reported] [7]
45. Walk me through how you'd scale this from a prototype to Amazon traffic. [Likely]
    - Where do latency and cost bite first?
46. What would make you kill this project? [Likely]
47. (Product-thinking variant, Seattle 2026 report) Pick an agentic product we could build. Who's the customer, and how do you measure success? [Reported] [6]

### F. Coding round

48. String compression (in place, run-length). [Reported] [6][26]
49. Top-K elements in a stream or array, in Python. [Reported] [27]
    - Heap versus quickselect: which would you choose, and when?
50. Design a rental system (object-oriented design). [Reported] [27]
51. A LeetCode medium on trees, followed by ML questions in the same round. [Reported] [7]
52. Implement a simple ML algorithm or training loop from scratch (e.g., logistic regression with gradient descent, k-means). [Reported] [12]
53. A dynamic programming problem, such as an edit-distance-style or knapsack-style problem. [Reported] [7]
54. Compute or minimize a mathematical function in code (e.g., numerically). [Likely]
55. Write code on LiveCode without running it, then test it by hand. [Likely]
    - What edge cases would break this, and what tests would you write?

### G. Bar raiser round (LP-heavy, outside interviewer)

56. Tell me about a time you had a conflict with a coworker or manager and how you approached it (Have Backbone; Disagree and Commit). [Reported] [23][41][42]
    - What did you do once the decision went against you?
57. Tell me about a time you had a major conflict with a Principal Engineer. How did you resolve it? [Reported] [25]
58. Tell me about a project failure and how you managed the aftermath. [Reported] [25]
    - What would you do differently? [Reported] [22]
59. Tell me about a time your model underperformed and you had to diagnose the issue. [Reported] [24]
    - What did you learn, and how did you pivot? [Reported] [26]
60. (Pushback probe) I think your approach was wrong. Convince me otherwise. Candidates report BRs deliberately telling them they were wrong. [Reported] [41][42]
    - What data would make you concede?
61. How was that measured? [Reported] [41][42]
62. What was your exact individual contribution, stated without "we"? [Reported] [41][42]
63. Tell me about a time you took initiative without waiting for permission. One rejection was reportedly explained as "seemed to wait for permission." [Likely]
64. Tell me about a time you raised the quality bar on a deliverable even though it delayed you (Insist on the Highest Standards). [Likely]
65. Tell me about a time you simplified a complex scientific approach so it could ship (Invent and Simplify). [Likely]
66. Tell me about a time you had to decide quickly with incomplete information and the decision turned out wrong (Bias for Action / Are Right, A Lot). [Likely]
67. Tell me about a time you dove into data or code to find a root cause that others missed (Dive Deep). [Likely]
68. Tell me about the most ambitious science bet you have proposed (Think Big). [Likely]
69. Tell me about a time you did more with less: compute, labels or headcount (Frugality). [Likely]
70. Tell me about a time you mentored someone or raised the bar of your team's science (Hire and Develop the Best). [Likely]
71. Tell me about a time you considered the downstream harm or fairness impact of a model (Success and Scale Bring Broad Responsibility). [Likely]
72. Tell me about a time you made your team's work environment better (Strive to be Earth's Best Employer). [Likely]
73. (TPM- or non-science BR) How do you set ETAs, map dependencies and communicate next steps on a science project? [Reported] [13]
    - When did a science timeline slip, and how did you tell stakeholders?
74. (TPM BR) A few stats/math checks, such as interpreting a p-value or confidence interval. [Reported] [13]

### H. LP questions embedded in technical rounds (each interviewer owns about 2-3 LPs)

75. Tell me about a time you started from the customer and worked backwards to the science problem (Customer Obsession). [Likely]
    - Who was the customer, and how did you get their signal?
76. Tell me about a time you took ownership of something outside your formal scope (Ownership). [Likely]
77. Tell me about a time you delivered a model under a hard deadline (Deliver Results). [Likely]
    - What did you descope, and what did that cost?
78. Tell me about a time you learned a new area fast to unblock a project (Learn and Be Curious). [Likely]
79. Tell me about a time you earned the trust of an engineering or product partner who doubted your science (Earn Trust). [Likely]
80. Tell me about a time you were wrong about a technical call. How did you find out? (Are Right, A Lot). [Likely]
81. Tell me about a time you pushed back on a stakeholder's metric or launch decision using data. [Likely]
    - Did you commit after being overruled?

### I. Level-calibration probes (L5 vs L4 vs L6 signal)

82. What was the largest scope you have owned end to end, from problem framing to production? [Likely]
    - Who else depended on it, beyond your own team?
83. Tell me about a time your work influenced a sister team's roadmap or technical direction. [Likely]
84. Tell me about a time you set the technical direction for other scientists or engineers, not just executed it. [Likely]
    - How did you get buy-in without authority?
85. What mechanism did you build (review, monitoring, tooling) that kept working without you in the room? [Likely]
86. Give me a second example: a different project where you showed the same thing. L6 rejections often cite a lack of multiple strong examples. [Likely]
87. How do you balance publishing or novelty against shipping on your current team? [Likely]
88. (Depth round, applied to leveling) Where did you have to make a judgment call that your manager didn't make for you? [Likely]
89. (HM) What would you expect to own in your first six months here? [Likely]
    - What would you need from me to get there?

### J. Questions to put to the recruiter or HM about format (candidate-side calibration)

90. Is the loop four or five rounds? Is there a job talk, and how long is it? [Likely]
91. Which rounds are breadth, depth, application/design and coding, and which one is the bar raiser? [Likely]
92. Is the coding round DSA (LeetCode), ML-from-scratch, or data manipulation (pandas/SQL)? [Likely]
93. Is the loop calibrated to L5, or is it an L5/L6 "up-level-able" loop? Can an L6 be considered if signal is strong? [Likely]
94. If the loop is positive but this team's headcount closes, does the result carry to other teams (recycle/team match)? [Likely]
95. How soon after the loop does the debrief happen, and when should I expect feedback? [Likely]
96. If I'm not successful, what is the cool-down period, and does it apply org-wide? Reports range from none ("recycle") to 6-24 months. [Likely]
97. Who on the panel is likely to be from outside the team? [Likely]
98. Does the team expect GenAI/LLM depth even if the role is not LLM-titled? [Likely]

### K. Debrief-driven follow-up (second loops, extra rounds)

99. (Extra round after a split debrief) Let's revisit coding or design. Can you solve this follow-up problem? [Reported] [14]
100. (New HM after team switch) Tell me about your project again, framed for our domain. [Reported] [12]
101. (Down-level conversation) Would you accept an L4 / AS I offer? What would you need to see to reach L5 quickly? [Likely]

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. SDE II Interview Prep, amazon.jobs (Amazon, official; used for the SDE comparison). https://amazon.jobs/content/en/how-we-hire/sde-ii-interview-prep
3. "Amazon Bar Raiser", About Amazon (Amazon, official). https://aboutamazon.com/news/workplace/amazon-bar-raiser
4. "Hire power: How Amazonians raise the bar with every interview", About Amazon (Amazon, official). https://www.aboutamazon.com/news/workplace/hire-power-how-amazonians-raise-the-bar-with-every-interview
5. "Amazon bar raisers, entry-level interview hiring procedure", Fortune, 2024. https://fortune.com/2024/08/07/amazon-bar-raisers-entry-level-interview-hiring-procedure
6. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor (reports incl. 2025-2026). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
7. Amazon Applied Scientist Interview Experience & Questions, Glassdoor (reports incl. 2025-2026). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
8. Amazon Senior Applied Scientist Interview Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Senior-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
9. Amazon interview review RVW99272421 (London Applied Scientist, Jul 2024), Glassdoor. https://static.glassdoor.it/Interview/Amazon-Interview-E6036-RVW99272421.htm
10. "Amazon Applied Scientist (L5) Virtual Onsite", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
11. "Amazon applied scientist interview feedback" (leveling dispute with bar raiser), Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-feedback-tvc4yec6
12. "Amazon applied scientist interview loops", Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
13. "My final interview for Amazon Applied Scientist-II will be taken by Principal TPM", Blind. https://www.teamblind.com/post/my-final-interview-for-amazon-applied-scientist-ii-will-be-taken-by-principal-tpm-cant-understand-what-tpm-will-ask-ryaa3tjo
14. "Amazon (AGI team) Applied Scientist II interview", Blind. https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT
15. "What are the expectations of Amazon L5 vs L6 for Applied Scientists? At Amazon, what level are fresh PhDs hired at?", Blind. https://www.teamblind.com/post/amazon-applied-scientists-l5-vs-l6-expectations-kegmp4rx
16. "Amazon applied scientist L6 to L5 downlevel", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l6-to-l5-downlevel-qvqtn0a8
17. "Amazon Science Sr Applied Scientist interview presentation requirement", Blind. https://www.teamblind.com/post/amazon-science-sr-applied-scientist-interview-presentation-requirement-tgrgowoo
18. "Amazon post-interview team match experience", Blind. https://www.teamblind.com/post/amazon-post-interview-team-match-experience-cqdnnskg
19. "Amazon Applied Scientist L5 Offer Eval", Blind, 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-offer-eval-c03bg14s
20. "Amazon applied scientist interview-onsite", Blind (also covers phone screen, data manipulation, OLS). https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
21. "Amazon bar raiser veto", Blind. https://www.teamblind.com/post/amazon-bar-raiser-veto-ybkwpkzv
22. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
23. Amazon Bar Raiser Interview (questions, prep tips), IGotAnOffer. https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview
24. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
25. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
26. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford, 2026. https://dataford.io/interview-guides/amazon/applied-scientist
27. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (vendor-published candidate debrief). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
28. "My interview experience for the Applied Scientist II role at Amazon" (Advertising Trust), Chaitanya Mundle, Medium. https://chaitanya001.medium.com/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
29. "Memoirs of an Amazon Bar Raiser", Carlos Arguelles, Medium / Geek Culture. https://medium.com/geekculture/memoirs-of-an-amazon-bar-raiser-718e36241310
30. "Working Backwards, by Colin Bryar and Bill Carr", Evan's Notes, Substack (book summary of the debrief and BR process). https://evansamek.substack.com/p/working-backwards-by-colin-bryar
31. Applied Scientist II, Amazon Payment Products (L5), Job ID 3168758, amazon.jobs. https://www.amazon.jobs/en/jobs/3168758/applied-scientist-ii-amazon-payment-products-l5
32. "The meaning of numbers in Applied Scientist 1/2/3 at Amazon", Levels.fyi community. https://www.levels.fyi/community/thread/g6lIB5/the-meaning-of-numbers-in--applied-scientist-123----amazon
33. "Amazon | Applied Scientist | Phone Screen | USA", LeetCode Discuss. https://leetcode.com/discuss/interview-experience/985312/amazon-applied-scientist-phone-screen-usa
34. "Amazon, MSFT, Google, Apple, FB, Zoom | Applied Scientist | Multiple | Sep 21 [OFFER, rest Reject]", LeetCode Discuss, 2021. https://leetcode.com/discuss/interview-experience/1585287/amazon-msft-google-apple-fb-zoom-applied-scientist-multiple-sep-21-offer-rest-reject
35. "Amazon Onsite Interview Experience for Fulltime Machine Learning Role", 1point3acres. https://www.1point3acres.com/interview/thread/1157086
36. Amazon Leadership Principles Interview (questions + tips), interviewing.io. https://interviewing.io/guides/amazon-leadership-principles
37. "Amazon Bar Raiser: Who They Are & How to Pass the Loop", Carrus. https://www.carrus.io/blog/all-about-bar-raisers-amazons-essential-element-to-the-hiring-process
38. "My Amazon Applied Science Interview Journey", Ajinkya Pahinkar, Medium. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
39. "Interview Experience for Applied Scientist at Amazon", Akanksha Paul, Medium. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
40. Amazon AGI Applied Scientist Interview Experience and Insights, 1point3acres. https://www.1point3acres.com/interview/thread/1130091
41. "Mastering the Amazon Bar Raiser Interview", 4 Day Week. https://4dayweek.io/interview-process/amazon-bar-raiser
42. "The Amazon Bar Raiser Interview: Who They Are and What They Ask", BarRaiser AI. https://barraiserai.com/guides/amazon-bar-raiser-interview
43. "Applied scientist interview at Amazon Ads", Blind. https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
