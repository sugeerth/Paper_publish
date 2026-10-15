# 07. Past Project / Technical Experience Deep Dive: Reported Prompts and Probing Follow-ups for Senior MLEs

> Research note (read first). This chapter was re-verified on 2026-10-14. The shared web-search budget was used up, so no new search queries could run. The network proxy could not resolve arxiv.org, medium.com, airbnb.tech, careers.airbnb.com or trueinterview.io, and the Glassdoor, Blind, LeetCode, IGotAnOffer and InterviewQuery hosts were unreachable earlier. GitHub was reachable, so this chapter is built from pages fetched live there: a mirror of about 100 Airbnb interview posts from 1point3acres (the FlowSight mirror, in Chinese and English), the TrueInterview Airbnb guides (CC BY 4.0), which are based on 3 MLE reports and 41 SWE reports from 2025 Q4 to 2026 Q2, and the TrueInterview Airbnb question list (dated entries to September 2026). In this review, [1]-[4], [5], [6]-[10], [14], [15], [17], [18], [19], [21], [23], [24], [28] and [34]-[37] were re-fetched and checked against the questions that cite them. Three tags were corrected: the whiteboard prompt (now [Likely]), the API and database probe (now [Likely], because that report came from a system design round), and the time-zone and "code to project" prompts (still [Reported], but from the core values round). [30]-[33] could not be fetched (DNS failure) and are cited from prior knowledge. Tags: **[Reported]** means a cited source says a candidate was asked this, or something very close to it, at Airbnb. **[Likely]** means an inference from the role, the senior bar and Airbnb's products; a citation on a [Likely] question only supplies context. Most deep-dive reports come from SWE and infrastructure loops, but the MLE guide says the MLE track runs the same round format [1].

## Format notes

| Aspect | What sources report |
|---|---|
| Name on the schedule | "Experience deep dive", "Technical Experience Interview", "Project Deep Dive", "Resume deep dive", "Project presentation", or for ML roles "ML Experience" [1][2][6][7][9][16][17] |
| Length | 45-60 min. One candidate's round was booked for 1 hour and ran about 90 minutes [1][8][9][14] |
| Interviewer | Usually a senior IC or an engineering manager, and sometimes a senior manager from a neighbouring team, who walks through the project step by step [16][21][25][36]. A separate hiring-manager (HM) call often repeats the deep dive and adds behavioral questions [9][22][23] |
| Shape | One project, taken apart: architecture, the decisions that were yours, trade-offs, measured impact. Interviewers keep asking "until they heard what they wanted to hear" [1][6][16]. In at least one loop the deep dive and a coding exercise shared a single session [26]. Behavioral and values questions are scored in separate rounds run by employees from any function [5] |
| Senior / L5 bar | **The perceived scope of your work is scored on its own.** Lead with problem size, ownership and outcome, not model details [1]. The round "also drives leveling" [2]. Scope judgements here can overturn strong technical rounds [2][3] |
| Weight | One recruiter told a candidate "the deep dive round is the most important", and Airbnb would not re-run it [18]. Several senior rejections were pinned on this round alone [8][18][20] |
| Reported failure modes | Opening walkthrough too long: keep it near 10 minutes, and one candidate's 20+ minutes ate the follow-up time [2]. A narrow project reads as below level, and a polished success story with no real mistakes is marked down [3]. A project too complex to explain or abstract quickly [8][14]. Generic answers without technical detail [18]. "No technical challenge visible" [10]. Background judged a poor match for the team [20] |
| MLE-specific twist | One MLE candidate's "ML Experience" round turned into an unexpected ML design question [7]. Prepare to whiteboard your project's system the way you would a design round |
| Prep advice from reports | Know every line on your resume: one candidate was grilled on a project that took one line [12]. Bring metrics, a conflict and a rejected alternative [2][3]. At staff level, compare your system with how Airbnb solved a similar problem on its tech blog, and say why you chose your approach [24] |

How to use this chapter: choose **one** flagship ML project that touched an experiment or online metric, with clear cross-team reach. Practise a 10-minute arc, then drill every question below against that project. Prepare a backup project for questions that ask for a different story, such as a failure or a conflict.

## Questions

### A. Opening, project selection and the 10-minute arc

1. Walk me through the project you are proudest of. [Reported] [16][11][29]
   - Why this project and not another one on your resume?
   - Can you give me the two-minute version before we go deep? [Likely] [3]
2. Pick one recent project and walk me through it from start to finish. [Reported] [19][25]
   - Where do you want to start: the problem, the system or the result?
3. Tell me about the most interesting project you have worked on. [Reported] [22]
4. Show me a project you are proud of that was cross-organizational or cross-functional. [Reported] [11]
   - Which teams outside yours had to change something for it to ship?
5. Walk me through the projects on your resume. What were you responsible for on each? [Reported] [13]
   - You wrote one line about this older project. Tell me about it in detail. [Reported] [12]
6. Walk through one ML project end to end, covering the decisions that were yours and the impact you measured. [Reported] [1]
   - Who used its output, and how many users, listings or dollars did it touch? [Likely] [1]
   - Which metric moved, and which teams did you have to align with? [Likely] [1]
7. If you had to describe the project's scope in one sentence (problem size, your ownership, outcome), what would it be? [Likely] [1]
8. Can you draw the overall system on the whiteboard while you explain? [Likely] [14]
9. Walk me through your background, and why Airbnb specifically? Which project best shows why you fit this team? [Reported] [3]

### B. Problem framing and business context

10. What problem was the project solving, and why did it matter to the business at that time? [Reported] [17][6]
    - What would have happened if nobody had done this project?
11. Why did the project start, and how did you get stakeholders to commit to it? [Reported] [2][3]
    - Which stakeholders outside your team had to commit, and what did they give up? [Reported] [3]
12. Set the implementation aside: what was the user problem, and how did the product change for users? [Reported] [2]
    - How would you explain the impact to a product manager who never looked at the model? [Likely]
13. What decision or product surface did the model's output feed into? [Likely] [1]
    - Who consumed the score: a ranker, a rule engine, an ops team or a user-facing UI?
14. How did you turn a vague product goal into a measurable prediction problem? [Likely] [1]
    - What was the label, and why that label rather than a nearby proxy?
15. What was the scale: traffic, QPS, data volume, number of entities? [Reported] [10]
    - What about that scale made the problem hard?
16. Why did you split the work into a short-term plan and a long-term plan? Why was the long-term plan needed at all? [Reported] [10]
17. Was ML actually necessary here? What was the heuristic or rules baseline? [Likely]
    - By how much did ML beat the baseline, and was the extra complexity worth it?
18. How did you size the opportunity before you started? [Likely]
19. How did you choose the training objective or loss, and how closely did it track the business metric you cared about? [Likely]
    - In a two-sided marketplace, whose outcome did the objective favour, guests or hosts, and how did you balance the two? [Likely]

### C. Ownership: what was yours versus the team's

20. What was your specific role on the project? [Reported] [15][13]
    - Which decisions did you personally make, and which did you inherit?
21. How was the work divided across the team? [Reported] [9]
    - Who else was on the project, and what did each person own?
22. Did you lead the project, or contribute to it? How many people did you direct or influence? [Likely] [1]
23. Were the metrics you analysed already there, or did you add them? [Reported] [10]
24. What did you do that nobody asked you to do? [Likely]
25. If you had left halfway through, what would not have happened? [Likely]
26. Whom did you influence outside your own group, and how? [Likely] [2]
27. After launch, who owned the model in production (on-call, retraining, deprecation)? Did ownership move to another team? [Likely]

### D. Data, labels and features

28. Where did the training data come from, and what were its biggest quality problems? [Likely] [1]
    - How did you detect and handle label leakage?
    - How did you handle delayed or censored labels (for example a booking that happens weeks after a search)?
29. How did you build the label, and how noisy was it? [Likely] [1]
    - Did you consider regression versus a bucketed classification framing? [Likely] [1]
30. Which features mattered most, and how do you know? [Likely]
    - Did any "obviously useful" feature turn out to hurt?
31. How did you keep training and serving features consistent? [Likely]
    - Did you ever ship with training-serving skew? How was it caught?
32. How did you handle cold start for new users or new items? [Likely] [1]
33. Did you use spatial or geographic signals? How did you encode them? [Likely] [1]
34. Did your training data come from logs produced by an earlier model? How did you deal with feedback-loop and position bias? [Likely] [30]
35. For a ranking or recommendation project, how did you build the training examples: impressions, negatives, sampling? [Likely] [34]
36. Did the project use personal or sensitive data? What privacy, retention or fairness constraints shaped the features? [Likely]

### E. Modeling decisions and trade-offs

37. Walk me through the architecture and the key design points. [Reported] [6][23]
    - Why this design over the obvious alternative?
38. What were the main trade-offs in your ML project? [Reported] [1][9][37]
    - What did you give up, and how did you decide it was acceptable?
39. What alternatives did you consider and reject? [Reported] [2][3]
    - What evidence made you reject each one?
40. Why this model family (for example GBDT versus a neural network)? [Likely] [1][30]
    - When did you decide to move to something more complex, and what convinced you?
    - Why this feature set and this serving path rather than the alternatives? [Likely] [1]
41. What was the single modeling change that gave the biggest gain? [Likely]
    - What did you try that did not work? [Likely] [30]
42. How did you tune the model, and how did you avoid overfitting to your validation set? [Likely]
43. How did you trade model quality against latency, cost or interpretability? [Likely]
44. Did downstream consumers depend on calibrated scores? How did you check calibration, and did it hold after launch? [Likely]
45. If the project used embeddings or an LLM, how did you evaluate output quality, and how did you control inference cost? [Likely] [34]
46. If the interviewer cannot follow your domain, how would you simplify the core idea in two minutes? [Reported] [8][14]

### F. Architecture, serving and production

47. Walk me through the implementation details end to end, from data pipeline to serving. [Reported] [6]
48. What were the latency and throughput requirements at serving time, and how did you meet them? [Likely]
49. Was inference batch, online or both? Why? [Likely]
50. How did you deploy new model versions safely (shadow, canary, rollback)? [Likely]
51. How did you make sure of system quality after deployment? [Reported] [17]
    - What monitoring and alerting did you set up for data drift and model decay?
52. What broke in production, and how did you find out? [Likely]
53. Which part of the system would break first at 10x scale? [Likely]
54. Did you build or reuse a platform (feature store, training pipeline, experiment framework)? Why? [Likely]
55. Walk me through an incident on this project. What was the root cause, and what was the fix? [Reported] [2]
56. How often was the model retrained, and how did you choose that cadence? How were data and model versions tracked so that a run could be reproduced? [Likely]
57. What did the project cost in compute and infrastructure, and how did you justify that cost? [Likely]

### G. Offline-to-online evaluation and experimentation (A/B testing focus)

58. How did you evaluate the model offline, and which metric did you trust? [Likely] [1]
    - How well did the offline metric predict the online result?
59. How did an offline metric gain become an online metric gain? [Likely] [1]
60. How did you know it worked? Was there an A/B test? [Likely] [31]
    - What was the primary metric, and what were the guardrails? [Likely] [34]
    - What was the randomization unit, and why?
    - How long did it run, and how did you choose the sample size?
61. What were the confidence intervals on the main result? Were any segments surprising? [Likely]
62. Did offline gains ever fail to show up online? How did you debug it? [Likely] [30]
63. Did your experiment face interference effects (for example a two-sided marketplace or shared inventory)? How did you handle it? [Likely] [31][34]
    - Would cluster-level or market-level randomization have changed the answer? [Likely] [34]
64. Did you consider faster methods such as interleaving for a ranking change? Why or why not? [Likely] [32]
65. Did you use variance reduction such as CUPED? What did it buy you? [Likely]
66. How did you handle novelty effects, seasonality or ramp-up effects when you read the result? [Likely]
67. Did you check for sample ratio mismatch or logging bugs before you trusted the result? [Likely]
68. Did you keep a long-term holdout? What did it show compared with the initial test? [Likely]
69. Tell me about an experiment result that surprised you. What did you do next? [Likely]
70. Who made the launch decision, and what would have made you not launch? [Likely]
71. How did you choose the minimum detectable effect, and did you check beforehand that the metric was sensitive enough to detect it? [Likely]
72. How did you handle many metrics and segments at once, and how did you avoid peeking or stopping the test early? [Likely]
73. The true outcome (for example a completed stay, a review or long-term value) arrives weeks later. Which proxy or surrogate metric did you use in the test, and how did you validate it? [Likely] [6][34]
74. If a clean A/B test was not possible (for example for pricing or a policy change), which quasi-experimental method did you use, and what assumptions did it rest on? [Likely] [34]
75. The primary metric was flat but a secondary metric improved. What did you recommend, and why? [Likely]

### H. Impact and metrics

76. What was the measured impact, in business terms? [Reported] [2][3]
    - How did you attribute the impact to your work and not to other launches?
77. How did you ensure the users actually benefited after deployment? [Reported] [17]
78. What metrics did you use to define success, and who agreed to them? [Reported] [2]
    - How did you align Product and Data Science on the success metric? [Likely]
79. Were there negative side effects on another metric or another group (for example hosts versus guests)? [Likely]
80. If the impact was smaller than expected, why? [Likely]
81. Did you measure segment-level harm, for example on new hosts, small markets or new users? What did you find? [Likely]

### I. Challenges, failures and debugging

82. What was the most challenging part of the project? [Reported] [16][15]
    - Was the challenge technical, organizational or about reliability?
83. What technical challenges did you hit, and how did you solve them? [Reported] [6][14]
    - Where exactly is the technical challenge in this project? [Reported] [10]
84. Tell me about a project or plan that missed its ETA, or whose scope had to shrink. [Reported] [9]
    - How did you identify the root cause? [Reported] [9]
    - Did you have a fallback plan to unblock the team? [Reported] [9]
85. Tell me about a model that worked offline but failed in production. [Likely]
86. Tell me about a bug or data problem that invalidated results. How was it found? [Likely]
87. What was the hardest debugging problem on the project? [Likely]
88. Tell me about a failure on this project and what you did next. [Reported] [3][4]
    - Tell me about a real mistake you made, not the polished version of the story. [Reported] [3]

### J. Planning, milestones, collaboration and conflict

89. How did you plan the project? [Reported] [15]
    - How did you set and deliver milestones? [Reported] [17][11][1]
90. Describe the project's milestones, a conflict over scope, and how it was resolved. [Reported] [1]
91. Describe a conflict during the project, and which of the rejected alternatives you would choose now. [Reported] [2]
92. How did you scope the project to fit a deadline? [Reported] [2]
    - When requirements changed mid-project, how did you re-plan and rescope? [Reported] [2][3]
93. How was the engineering effort structured? [Reported] [11][29]
94. How did you persuade stakeholders to back the project? [Reported] [2]
    - Who opposed it, and what changed their mind?
95. Did you have a conflict on the project? How did you resolve it? [Reported] [15][16][9]
    - What would the other person say about how it was resolved?
96. How did you work with your teammates? [Reported] [16]
97. Tell me about collaborating with another team where their delay affected you. [Reported] [9]
98. How did you keep teams in different time zones or cultures aligned from design through implementation? What problems came up? (asked in the core values round) [Reported] [9]
99. How did you handle a tight deadline or a complex project you owned? [Reported] [4]
100. How did you work with data scientists on metric definitions and experiment reading? [Likely]
101. How did you influence the technical direction without formal authority? [Likely]

### K. Retrospective, learning and next steps

102. If you redid the project, what would you change? [Reported] [15][17][3]
     - What would you change if you had to redesign it from scratch? [Reported] [17]
103. What did you learn from the project? [Reported] [16]
104. What would the next iteration of the system look like? [Likely]
105. If you had half the time, what would you cut? [Likely]
106. What technical debt did you leave behind, and why was it acceptable? [Likely]
107. How did you decide the project was finished, and what did you stop doing? [Likely]
108. What was the biggest obstacle in moving from writing code to owning a whole project? (asked in the core values round) [Reported] [9]

### L. Leadership and senior/staff calibration

109. Walk me through your experience, a past project and your leadership on it. [Reported] [21][36]
110. How did you grow or mentor others on this project? [Likely]
111. What is your current scope at work, and what scope are you looking for? [Reported] [10]
     - Are you looking to be a tech lead, or a strong IC? [Reported] [10]
112. How did this project change how your team or organization works? [Likely]
113. Which decision on this project would you defend most strongly to a skeptical staff engineer? [Likely]
114. What was the strongest pushback in design review, and did it change the design? [Likely]
115. How did you decide what not to build? [Likely]

### M. Mapping your project to Airbnb

116. How does your system compare with how Airbnb solved a similar problem, and why did you choose your approach? [Reported] [24]
     - Have you read Airbnb's engineering blog posts on search ranking, pricing or experimentation? How does your work differ? [Likely] [24][30][31]
117. How would your project transfer to a two-sided marketplace like Airbnb's (guests and hosts)? [Likely]
118. What in your background maps directly onto this team's problems? [Reported] [20][2]
     - You have also worked in an unrelated area (for example hardware). How is that relevant here? [Reported] [20]
119. If you joined, what would you build in your first six months, based on this experience? [Likely]
120. Tell me about a project connected to your personal or professional values. How did it embody or challenge them? [Reported] [19][33]

### N. Pressure probes reported in the room

These are the push-back styles candidates describe. Practise staying concise and concrete under each one.

121. "I still don't understand. Can you explain the task again, more simply?" [Reported] [14]
122. "Why did you do that at all?" (repeated challenges to the rationale for a phase of the project) [Reported] [10]
123. "Can you be more specific? What were the actual technical details?" (asked because answers sounded generic) [Reported] [18]
124. Interruptions and nitpicking mid-answer: how do you regain the thread? [Reported] [27]
125. "Let's switch: how would you design an ML system for X?" (the experience round turns into ML design) [Reported] [7]
126. "Give me the specific API and database design of your system." (a detail-level probe instead of scaling talk; reported in an Airbnb system design round, not the deep dive) [Likely] [28]

## References

[1]-[29] and [34]-[37] were fetched from GitHub; the subset listed in the research note was re-fetched on 2026-10-14. [6]-[29] and [35]-[37] are English or Chinese 1point3acres posts mirrored on GitHub. The mirror does not keep original post dates, though several posts carry 2025 timestamps. [30]-[33] could **not** be fetched (DNS failure on 2026-10-14) and are cited from prior knowledge. Verify them before relying on them.

1. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview, CC BY 4.0; based on 3 candidate reports, 2026 Q2). GitHub mirror kevin-2023-code/Tech-Interview-Questions, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb Software Engineer Interview Process" (TrueInterview, CC BY 4.0; 41 candidate reports, 2025 Q4 to 2026 Q2). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
3. "Airbnb interview process" (TrueInterview general guide). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. "Airbnb" company guide (Technical Experience round, behavioral list; updated October 2026). jugaldb/faang-interview-prep, GitHub, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
5. "Interview formats of top companies" (Airbnb section). Tech Interview Handbook, GitHub. https://github.com/yangshun/tech-interview-handbook/blob/main/apps/website/contents/interview-formats-top-companies.md
6. 1p3a post #40 (MLE loop): project deep dive covering architecture, implementation, challenges and solutions; ML design on LTV and host pricing. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
7. 1p3a post #79 (MLE loop): "ML Experience" round expected as a project deep dive but turned into an ML design question; two ML design rounds. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
8. 1p3a post #10 (senior, rejected on design and resume-project rounds): 1-hour round ran 90 min on one project; advice to keep it under 20 min and simplify. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/10.md
9. 1p3a post #11: VO3 "Tech Experience Interview" (45 min; "trade-offs, conflicts"); HM round with resume deep dive plus BQ (missed ETA or reduced scope, root cause, division of work, fallback plan); core values round (cross-time-zone design-to-implementation, moving from writing code to owning a project). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/11.md
10. 1p3a post #13 (senior): HM scope call (TL or IC); project deep dive challenged on the long-term plan, traffic level, new versus existing metrics, and the lack of a technical challenge. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/13.md
11. 1p3a post #8: "project experience: most proud project, milestone, eng structure, conflict"; "BQ: show a proud project (cross organizational/functional)". https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/8.md
12. 1p3a post #3: experience BQ round asked about a project described in a single resume line. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/3.md
13. 1p3a post #30: phone round with a 30-minute resume deep dive on responsibilities. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/30.md
14. 1p3a post #39: 1-hour resume round felt like a stress interview ("couldn't understand"); advice to focus on role, decisions, technical challenges and conflicts. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/39.md
15. 1p3a post #58: project deep dive covering role, planning, challenges, conflicts and redo improvements. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/58.md
16. 1p3a post #69: Technical Experience Interview covering proudest project, most challenging part, teammates, conflicts and lessons. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/69.md
17. 1p3a post #84 (senior backend): project presentation covering scope, milestones, post-deploy quality and user benefit, and redesign. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/84.md
18. 1p3a post #86: failed the deep dive for "generic answers without any technical details"; recruiter said the deep dive is the most important round. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/86.md
19. 1p3a post #93: experience round as a deep dive into a recent project; values-linked project prompt. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/93.md
20. 1p3a post #7: deep-dive feedback said the candidate's background did not fit the team. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/7.md
21. 1p3a post #78 (staff SWE, L6): senior manager from another group ran experience, past projects and leadership. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/78.md
22. 1p3a post #29 (senior): HM went over the most interesting project in detail; VO behavior round focused on one project. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/29.md
23. 1p3a post #62 (infrastructure senior): HM interview on resume projects, key design points and challenges. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/62.md.md
24. 1p3a post #94 (staff): HM round deep on the resume; advice to read Airbnb's tech blog and compare your system with Airbnb's. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/94.md
25. 1p3a post #99: experience interview "dug into a project from start to finish". https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/99.md.md
26. 1p3a post #104: project deep dive and coding combined in one round. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/104.md
27. 1p3a post #31: a manager in the behavioral and in-depth project round kept interrupting and nitpicking. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/31.md
28. 1p3a post #6: "BQ and past experience deep dive" round (no questions listed); the API and database detail probe came from the system design rounds. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/6.md
29. 1p3a post #18: "most proud project, milestone, eng structure, conflict", asked in fine detail. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/18.md
30. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD 2019 (includes lessons on failed attempts and offline/online gaps). Not fetched. https://arxiv.org/abs/1810.09591
31. "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014. Not fetched. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
32. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Tech Blog (Medium), 2022. Not fetched. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
33. "Life at Airbnb" (core values), Airbnb Careers. Linked from [4]; not fetched. https://careers.airbnb.com/life-at-airbnb/
34. "Airbnb" company question list (TrueInterview mirror; experiment design prompts including primary and guardrail metrics, a network-aware experiment, causal regression, and LTV and embedding-search ML design; dated entries to September 2026). kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
35. 1p3a post #16: onsite round 2 "BQ and past experience deep dive"; notes that senior candidates are usually pushed on scaling. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/16.md
36. 1p3a post #75: a senior manager walked through the candidate's project experience step by step and gave feedback afterwards. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/75.md
37. 1p3a post #21: experience round on the candidate's project, mostly trade-offs, conflicts and standard behavioral questions. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/21.md
