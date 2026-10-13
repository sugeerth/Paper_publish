# 07. Past Project / Technical Experience Deep Dive: Reported Prompts and Probing Follow-ups for Senior MLEs

> Research note (read first). The shared web-search budget was already used up when this chapter was written, so no new search queries could be run. The egress proxy also could not resolve Glassdoor, Blind, LeetCode, IGotAnOffer, InterviewQuery, Hello Interview, PracHub, TrueInterview and Medium. GitHub could be reached, so this chapter is built mainly from pages fetched live from GitHub: a mirror of about 100 Airbnb interview posts from 1point3acres (the FlowSight mirror, in Chinese and English) and the TrueInterview Airbnb guides (CC BY 4.0), which are based on 3 MLE reports and 41 SWE reports from 2025 Q4 to 2026 Q2. References [1]-[29] were fetched and read. [30]-[33] are cited from prior knowledge and were not re-fetched. Tags: **[Reported]** means a cited source says a candidate was asked this, or something very close to it, at Airbnb. **[Likely]** means an inference from the role, the senior bar and Airbnb's products. It does not mean anyone reported it. Most of the deep-dive reports come from SWE and infrastructure loops, but the MLE guide says the MLE track runs the same round format [1].

## Format notes

| Aspect | What sources report |
|---|---|
| Name on the schedule | "Experience deep dive", "Technical Experience Interview", "Project Deep Dive", "Resume deep dive", "Project presentation", or for ML roles "ML Experience" [1][2][6][7][9][16][17] |
| Length | 45-60 min. One candidate's round was booked for 1 hour and ran about 90 minutes [1][8][9][14] |
| Interviewer | Usually a senior IC or an engineering manager, and sometimes a senior manager from a neighbouring team (staff loops) [16][21][25]. A separate hiring-manager (HM) call often repeats the deep dive and adds behavioral questions [9][22][23] |
| Shape | One project, taken apart: architecture, the decisions that were yours, trade-offs, measured impact. Interviewers keep asking "until they heard what they wanted to hear" [1][6][16]. In at least one loop the deep dive and a coding exercise shared a single session [26]. Behavioral and values questions are scored in separate rounds run by employees from any function [5] |
| Senior / L5 bar | **The perceived scope of your work is scored on its own.** Lead with problem size, ownership and outcome, not model details [1]. The round "also drives leveling" [2]. Scope judgements here can overturn strong technical rounds [2][3] |
| Weight | One recruiter told a candidate "the deep dive round is the most important", and Airbnb would not re-run it [18]. Several senior rejections were pinned on this round alone [8][18][20] |
| Reported failure modes | Opening walkthrough too long: keep it near 10 minutes, and one candidate's 20+ minutes ate the follow-up time [2]. A project too complex to explain or abstract quickly [8][14]. Generic answers without technical detail [18]. "No technical challenge visible" [10]. Background judged a poor match for the team [20] |
| MLE-specific twist | One MLE candidate's "ML Experience" round turned into an unexpected ML design question [7]. Prepare to whiteboard your project's system the way you would a design round |
| Prep advice from reports | Know every line on your resume: one candidate was grilled on a project that took one line [12]. Bring metrics, a conflict and a rejected alternative [2][3]. At staff level, compare your system with how Airbnb solved a similar problem on its tech blog, and say why you chose your approach [24] |

How to use this chapter: choose **one** flagship ML project that touched an experiment or online metric, with clear cross-team reach. Practise a 10-minute arc, then drill every question below against that project. Prepare a backup project for questions that ask for a different story, such as a failure or a conflict.

## Questions

### A. Opening, project selection and the 10-minute arc

1. Walk me through the project you are proudest of. [Reported] [16][11][29]
   - Why this project and not another one on your resume?
   - Can you give me the two-minute version before we go deep?
2. Pick one recent project and walk me through it from start to finish. [Reported] [19][25]
   - Where do you want to start: the problem, the system or the result?
3. Tell me about the most interesting project you have worked on. [Reported] [22]
4. Show me a project you are proud of that was cross-organizational or cross-functional. [Reported] [11]
   - Which teams outside yours had to change something for it to ship?
5. Walk me through the projects on your resume. What were you responsible for on each? [Reported] [13]
   - You wrote one line about this older project. Tell me about it in detail. [Reported] [12]
6. Tell me about the ML project with the biggest business impact that you led end to end. [Likely] [1]
   - Who used its output, and how many users, listings or dollars did it touch?
7. If you had to describe the project's scope in one sentence (problem size, your ownership, outcome), what would it be? [Likely] [1]
8. Can you draw the overall system on the whiteboard while you explain? [Reported] [14][6]

### B. Problem framing and business context

9. What problem was the project solving, and why did it matter to the business at that time? [Reported] [17][6]
   - What would have happened if nobody had done this project?
10. What decision or product surface did the model's output feed into? [Likely] [1]
    - Who consumed the score: a ranker, a rule engine, an ops team or a user-facing UI?
11. How did you turn a vague product goal into a measurable prediction problem? [Likely] [1]
    - What was the label, and why that label rather than a nearby proxy?
12. What was the scale: traffic, QPS, data volume, number of entities? [Reported] [10]
    - What about that scale made the problem hard?
13. Why did you split the work into a short-term plan and a long-term plan? Why was the long-term plan needed at all? [Reported] [10]
14. Was ML actually necessary here? What was the heuristic or rules baseline? [Likely]
    - By how much did ML beat the baseline, and was the extra complexity worth it?
15. How did you size the opportunity before you started? [Likely]

### C. Ownership: what was yours versus the team's

16. What was your specific role on the project? [Reported] [15][13]
    - Which decisions did you personally make, and which did you inherit?
17. How was the work divided across the team? [Reported] [9]
    - Who else was on the project, and what did each person own?
18. Did you lead the project, or contribute to it? How many people did you direct or influence? [Likely] [1]
19. Were the metrics you analysed already there, or did you add them? [Reported] [10]
20. What did you do that nobody asked you to do? [Likely]
21. If you had left halfway through, what would not have happened? [Likely]

### D. Data, labels and features

22. Where did the training data come from, and what were its biggest quality problems? [Likely] [1]
    - How did you detect and handle label leakage?
    - How did you handle delayed or censored labels (for example a booking that happens weeks after a search)?
23. How did you build the label, and how noisy was it? [Likely] [1]
    - Did you consider regression versus a bucketed classification framing? [Likely] [1]
24. Which features mattered most, and how do you know? [Likely]
    - Did any "obviously useful" feature turn out to hurt?
25. How did you keep training and serving features consistent? [Likely]
    - Did you ever ship with training-serving skew? How was it caught?
26. How did you handle cold start for new users or new items? [Likely] [1]
27. Did you use spatial or geographic signals? How did you encode them? [Likely] [1]
28. Did your training data come from logs produced by an earlier model? How did you deal with feedback-loop and position bias? [Likely] [30]

### E. Modeling decisions and trade-offs

29. Walk me through the architecture and the key design points. [Reported] [6][23]
    - Why this design over the obvious alternative?
30. What were the main trade-offs you made? [Reported] [9][21]
    - What did you give up, and how did you decide it was acceptable?
31. What alternatives did you consider and reject? [Reported] [2][3]
    - What evidence made you reject each one?
32. Why this model family (for example GBDT versus a neural network)? [Likely] [1][30]
    - When did you decide to move to something more complex, and what convinced you?
33. What was the single modeling change that gave the biggest gain? [Likely]
    - What did you try that did not work? [Likely] [30]
34. How did you tune the model, and how did you avoid overfitting to your validation set? [Likely]
35. How did you trade model quality against latency, cost or interpretability? [Likely]
36. If the interviewer cannot follow your domain, how would you simplify the core idea in two minutes? [Reported] [8][14]

### F. Architecture, serving and production

37. Walk me through the implementation details end to end, from data pipeline to serving. [Reported] [6]
38. What were the latency and throughput requirements at serving time, and how did you meet them? [Likely]
39. Was inference batch, online or both? Why? [Likely]
40. How did you deploy new model versions safely (shadow, canary, rollback)? [Likely]
41. How did you make sure of system quality after deployment? [Reported] [17]
    - What monitoring and alerting did you set up for data drift and model decay?
42. What broke in production, and how did you find out? [Likely]
43. Which part of the system would break first at 10x scale? [Likely]
44. Did you build or reuse a platform (feature store, training pipeline, experiment framework)? Why? [Likely]

### G. Offline-to-online evaluation and experimentation (A/B testing focus)

45. How did you evaluate the model offline, and which metric did you trust? [Likely] [1]
    - How well did the offline metric predict the online result?
46. How did an offline metric gain become an online metric gain? [Likely] [1]
47. How did you know it worked? Was there an A/B test? [Likely] [31]
    - What was the primary metric, and what were the guardrails?
    - What was the randomization unit, and why?
    - How long did it run, and how did you choose the sample size?
48. What were the confidence intervals on the main result? Were any segments surprising? [Likely]
49. Did offline gains ever fail to show up online? How did you debug it? [Likely] [30]
50. Did your experiment face interference effects (for example a two-sided marketplace or shared inventory)? How did you handle it? [Likely] [31]
51. Did you consider faster methods such as interleaving for a ranking change? Why or why not? [Likely] [32]
52. Did you use variance reduction such as CUPED? What did it buy you? [Likely]
53. How did you handle novelty effects, seasonality or ramp-up effects when you read the result? [Likely]
54. Did you check for sample ratio mismatch or logging bugs before you trusted the result? [Likely]
55. Did you keep a long-term holdout? What did it show compared with the initial test? [Likely]
56. Tell me about an experiment result that surprised you. What did you do next? [Likely]
57. Who made the launch decision, and what would have made you not launch? [Likely]

### H. Impact and metrics

58. What was the measured impact, in business terms? [Reported] [2][3]
    - How did you attribute the impact to your work and not to other launches?
59. How did you ensure the users actually benefited after deployment? [Reported] [17]
60. What metrics did you use to define success, and who agreed to them? [Reported] [2]
    - How did you align Product and Data Science on the success metric? [Likely]
61. Were there negative side effects on another metric or another group (for example hosts versus guests)? [Likely]
62. If the impact was smaller than expected, why? [Likely]

### I. Challenges, failures and debugging

63. What was the most challenging part of the project? [Reported] [16][15]
    - Was the challenge technical, organizational or about reliability?
64. What technical challenges did you hit, and how did you solve them? [Reported] [6][14]
    - Where exactly is the technical challenge in this project? [Reported] [10]
65. Tell me about a project or plan that missed its ETA, or whose scope had to shrink. [Reported] [9]
    - How did you identify the root cause? [Reported] [9]
    - Did you have a fallback plan to unblock the team? [Reported] [9]
66. Tell me about a model that worked offline but failed in production. [Likely]
67. Tell me about a bug or data problem that invalidated results. How was it found? [Likely]
68. What was the hardest debugging problem on the project? [Likely]
69. Tell me about a failure on this project and what you did next. [Reported] [3][4]

### J. Planning, milestones, collaboration and conflict

70. How did you plan the project? [Reported] [15]
    - How did you set and deliver milestones? [Reported] [17][11]
71. How was the engineering effort structured? [Reported] [11][29]
72. How did you persuade stakeholders to back the project? [Reported] [2]
    - Who opposed it, and what changed their mind?
73. Did you have a conflict on the project? How did you resolve it? [Reported] [15][16][9]
    - What would the other person say about how it was resolved?
74. How did you work with your teammates? [Reported] [16]
75. Tell me about collaborating with another team where their delay affected you. [Reported] [9]
76. How did you keep teams in different time zones or cultures aligned from design through launch? [Reported] [9]
77. How did you handle a tight deadline or a complex project you owned? [Reported] [4]
78. How did you work with data scientists on metric definitions and experiment reading? [Likely]
79. How did you influence the technical direction without formal authority? [Likely]

### K. Retrospective, learning and next steps

80. If you redid the project, what would you change? [Reported] [15][17][3]
    - What would you change if you had to redesign it from scratch? [Reported] [17]
81. What did you learn from the project? [Reported] [16]
82. What would the next iteration of the system look like? [Likely]
83. If you had half the time, what would you cut? [Likely]
84. What technical debt did you leave behind, and why was it acceptable? [Likely]
85. What was the biggest obstacle in moving from writing code to owning a whole project? [Reported] [9]

### L. Leadership and senior/staff calibration

86. Walk me through your experience, a past project and your leadership on it. [Reported] [21]
87. How did you grow or mentor others on this project? [Likely]
88. What is your current scope at work, and what scope are you looking for? [Reported] [10]
    - Are you looking to be a tech lead, or a strong IC? [Reported] [10]
89. How did this project change how your team or organization works? [Likely]
90. Which decision on this project would you defend most strongly to a skeptical staff engineer? [Likely]
91. How did you decide what not to build? [Likely]

### M. Mapping your project to Airbnb

92. How does your system compare with how Airbnb solved a similar problem, and why did you choose your approach? [Reported] [24]
    - Have you read Airbnb's engineering blog posts on search ranking, pricing or experimentation? How does your work differ? [Likely] [24][30][31]
93. How would your project transfer to a two-sided marketplace like Airbnb's (guests and hosts)? [Likely]
94. What in your background maps directly onto this team's problems? [Reported] [20][2]
    - You have also worked in an unrelated area (for example hardware). How is that relevant here? [Reported] [20]
95. If you joined, what would you build in your first six months, based on this experience? [Likely]
96. Tell me about a project connected to your personal or professional values. How did it embody or challenge them? [Reported] [19][33]

### N. Pressure probes reported in the room

These are the push-back styles candidates describe. Practise staying concise and concrete under each one.

97. "I still don't understand. Can you explain the task again, more simply?" [Reported] [14]
98. "Why did you do that at all?" (repeated challenges to the rationale for a phase of the project) [Reported] [10]
99. "Can you be more specific? What were the actual technical details?" (asked because answers sounded generic) [Reported] [18]
100. Interruptions and nitpicking mid-answer: how do you regain the thread? [Reported] [27]
101. "Let's switch: how would you design an ML system for X?" (the experience round turns into ML design) [Reported] [7]
102. "Give me the specific API and database design of your system." (detail-level probe rather than scaling talk) [Reported] [28]

## References

[1]-[29] were fetched and read for this chapter. [6]-[29] are English or Chinese 1point3acres posts mirrored on GitHub. The mirror does not keep original post dates, though several posts carry 2025 timestamps. [30]-[33] were **not** fetched and are cited from prior knowledge. Verify them before relying on them.

1. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview, CC BY 4.0; based on 3 candidate reports, 2026 Q2). GitHub mirror kevin-2023-code/Tech-Interview-Questions, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb Software Engineer Interview Process" (TrueInterview, CC BY 4.0; 41 candidate reports, 2025 Q4 to 2026 Q2). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
3. "Airbnb interview process" (TrueInterview general guide). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process.md
4. "Airbnb" company guide (Technical Experience round, behavioral list; updated October 2026). jugaldb/faang-interview-prep, GitHub, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
5. "Interview formats of top companies" (Airbnb section). Tech Interview Handbook, GitHub. https://github.com/yangshun/tech-interview-handbook/blob/main/apps/website/contents/interview-formats-top-companies.md
6. 1p3a post #40 (MLE loop): project deep dive covering architecture, implementation, challenges and solutions; ML design on LTV and host pricing. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
7. 1p3a post #79 (MLE loop): "ML Experience" round expected as a project deep dive but turned into an ML design question; two ML design rounds. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
8. 1p3a post #10 (senior, rejected on design and resume-project rounds): 1-hour round ran 90 min on one project; advice to keep it under 20 min and simplify. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/10.md
9. 1p3a post #11: VO3 "Tech Experience Interview" (45 min); HM round with deep dive plus BQ (missed ETA or reduced scope, root cause, division of work, fallback plan, cross-timezone collaboration); experience round "trade-offs, conflicts". https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/11.md
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
28. 1p3a post #6: "BQ and past experience deep dive" round; interviewers wanted API and database detail. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/6.md
29. 1p3a post #18: "most proud project, milestone, eng structure, conflict", asked in fine detail. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/18.md
30. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD 2019 (includes lessons on failed attempts and offline/online gaps). Not fetched. https://arxiv.org/abs/1810.09591
31. "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014. Not fetched. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
32. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving", Airbnb Tech Blog (Medium), 2022. Not fetched. https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e
33. "Life at Airbnb" (core values), Airbnb Careers. Linked from [4]; not fetched. https://careers.airbnb.com/life-at-airbnb/
