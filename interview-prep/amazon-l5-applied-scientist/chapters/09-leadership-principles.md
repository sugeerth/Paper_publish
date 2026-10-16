# 09. Leadership Principles Bank: Behavioral Questions for the L5 Applied Scientist Loop, with L5 Probes and Bar Raiser Favorites

> Scope: behavioral (Leadership Principle, "LP") questions as asked of Amazon Applied Scientists, grouped by all 16 LPs, plus the follow-up probes used to test the L5 (Applied Scientist II) bar, and the bar raiser's favorite questions and pressure tactics. Questions only, no model answers.
>
> Research note (read this first): this chapter was built under tight limits. The session's web-search budget had already been used by earlier chapters, and this environment's egress policy blocked direct fetches of Glassdoor, Blind, LeetCode, Medium, IGotAnOffer, interviewing.io and amazon.jobs. Two kinds of evidence were used:
> (a) Sourced claims and reported questions already recorded, with citations, in chapters 01-08 of this bank. Those were gathered from search-engine excerpts of the cited pages and are carried over here with the same sources.
> (b) Pages read in full for this chapter: GitHub-hosted material, including a candidate's notes that repost 1point3acres Amazon AS interview reports [21][22] and a curated Amazon AI-role question guide [20].
> [Reported] means a source attributes the question, or a close paraphrase, to Amazon interviews (AS where possible). [Likely] means an inference from the LP text, the AS role and Amazon's products. It has not been seen verbatim. Treat vendor guides as weaker than first-person reports.

## Format notes for the behavioral portion

- **No single "LP round"; LPs are in every round.** Amazon's AS prep page says interviewers are assigned competencies "based off of our Leadership Principles" and each typically asks 2-3 behavioral questions [1]. Candidates report 1-2 LP questions opening or closing every technical round. Estimates of LP time per round range from 5-15 minutes [15] to "at least 30 minutes" of each hour-long interview [21]. One L5 candidate put behavioral content at over 60% of the loop [23].
- **The bar raiser (BR) round is mostly or entirely LP.** One 1point3acres AS report said the BR round was "pure LPs" [21]. The BR comes from outside the hiring org, has been trained and shadowed into the role, chairs the debrief, and holds a de facto veto [3][19]. Glassdoor AS and AS II reports describe the BR round as LP-heavy [4][5]. AS candidates have reported BRs who were non-science, including a Principal TPM who also asked about ETAs, dependencies and a few stats checks [7].
- **Assigned LPs.** Each interviewer usually owns 2-3 LPs, so the loop covers the set without overlap [1][16]. Some interviewers name the LP before asking [21]. Candidates recommend asking the recruiter in the prep call which LPs each interviewer is assigned [21]. A candidate's compiled notes say an Amazon recruiter advised against reusing one story with more than two interviewers [22].
- **Fresh-PhD adaptation.** Interviewers prompt candidates without industry projects to draw on research experience. Reported examples: "what did you do when you didn't like a project your advisor assigned?" and "have you worked in a research area you weren't familiar with?" [21]
- **What the L5 bar looks like in LP answers.** L5 means owning a project end to end, making the technical calls, and influencing peers, with some spill-over to sister teams [8]. A vendor debrief says L5 stories face 2-3 layers of follow-up, against about 1 layer at L4 [15]. Blind commenters say LP narratives, not coding, are what separate L5 from L6, and that L6 needs multiple strong examples and influence beyond the team [8][9]. A BR can pull the level down even when the hiring manager is positive. One AS candidate's BR thought they were "at the cusp of L4 and L5" [6].
- **Grading habits reported for BRs.** BRs push on metrics ("how was that measured?"), on individual contribution ("I", not "we"), and sometimes say outright that your approach was wrong to see how you hold your position [17][18][20]. Weak or rehearsed stories "collapse" under several "why" layers [20].
- **The 16 LPs** are published on amazon.jobs [2]. The last two, Strive to be Earth's Best Employer and Success and Scale Bring Broad Responsibility, were added in 2021 and are asked less often in reports, but they are in scope.

---

## Questions

Tag legend: [Reported] = a source attributes it to Amazon interviews (cite). [Likely] = a grounded inference. Indented bullets are follow-up probes, phrased as interviewers ask them.

### 1. Customer Obsession

1. Tell me about a time you started from the customer and worked backwards to define the science problem. [Likely]
   - Who exactly was the customer: the end shopper, an internal team, or a seller?
   - How did you get the customer signal: data, anecdotes, or a PR/FAQ?
   - What metric did you choose as the customer proxy, and where did it diverge from what customers actually felt?
2. Tell me about a time you used data to anticipate a customer need nobody had asked for. [Likely]
   - How did you validate that the need was real before you built it?
3. Tell me about a time you traded scientific elegance or novelty for customer impact, or refused to. [Likely]
   - Looking back, was that the right call for the customer?
4. Tell me about a time a model's offline metric improved but the customer experience got worse. [Likely]
   - How did you find out, and what did you change in how you evaluate?
5. Tell me about a time you dealt with a difficult or demanding internal customer (a product, ops or business team consuming your model). [Reported] [22]
   - What did they want that you wouldn't give them, and why?
6. Tell me about a time you had to say no to a stakeholder request because it would hurt end customers. [Likely]
   - Did the stakeholder agree in the end? If not, what happened?

### 2. Ownership

7. Tell me about a time you went above and beyond your job responsibilities. [Reported] [21]
   - Why was it your job to do it, and who else could have?
8. Tell me about a time you went well beyond the scope of a project and still delivered. [Reported] [21]
   - What did you stop doing to make room for it?
9. Tell me about a time you took on something significant outside your area of responsibility. [Reported] [22]
   - How did the owning team react?
10. Tell me about a time you worked on a project with unclear responsibilities. [Reported] [22]
    - How did you decide what you would own?
11. Tell me about a time a model you shipped failed or underperformed in production. What did you do? [Reported] [12] (model underperformance reported in BR-style questioning; production framing [Likely])
    - Who found it first, you or someone else?
    - What did you put in place so it couldn't happen silently again?
12. Tell me about a time you sacrificed a short-term result for long-term value, for example paying down technical or science debt instead of shipping another model. [Likely]
    - How did you justify it to your manager?
13. Tell me about a time you took initiative without waiting for permission. [Likely] (chapter 01 notes a rejection reportedly explained as "seemed to wait for permission")
    - What would have happened if you had waited?
14. Tell me about something you own today that will keep working after you leave the team. [Likely]
    - What mechanism, such as a runbook, monitor or review, makes it independent of you?

### 3. Invent and Simplify

15. Tell me about a time you gave a simple solution to a complex problem, or found a new way to do something. [Reported] [22]
    - What did the complex alternative look like, and why did you reject it?
16. Tell me about the most innovative project you have worked on. [Reported] [20] (Exponent-style "invented a solution to a problem that didn't have an obvious answer"; quoted in a prep source, so weak)
    - What, specifically, was your invention versus prior art?
17. Tell me about a time you simplified a complex ML approach into something that could ship. [Likely]
    - What did you give up in accuracy, and how did you decide it was worth it?
18. Tell me about a time you started from a heuristic or simple baseline and it beat the sophisticated model. [Likely]
19. How do you solve a problem when the capability you need isn't in your programming tools or libraries? [Reported] [21] (AS NLP phone screen)
    - Give me a specific time you built that missing piece yourself.
20. Tell me about a time you reused an idea from outside your field or from another team rather than inventing from scratch. [Likely]
    - How did you know it would transfer?

### 4. Are Right, A Lot

21. Tell me about a time you made a critical scientific or technical decision without full data. [Reported] [14]
    - What did you learn, and how did you pivot? [Reported] [14]
22. Tell me about a time you realized your initial modeling hypothesis was wrong. [Likely]
    - What evidence made you change your mind, and how fast did you act on it?
23. Tell me about a time you were wrong about a technical call. How did you find out? [Likely]
24. Tell me about a time your judgment went against the data, or the data went against your intuition. Which did you trust? [Likely]
25. Tell me about a time you sought out a view that disagreed with yours before making a decision. [Likely]
    - Whose view did you seek, and did it change the decision?
26. Tell me about a time you had to choose between two model architectures or approaches with no clear winner. [Likely]
    - What was the decision rule, and in hindsight was it right?

### 5. Learn and Be Curious

27. Tell me about a time you learned a new area quickly to solve a problem. [Likely] (fresh-PhD variant reported: "worked in a research area you weren't familiar with" [21])
    - How did you know you had learned enough to make decisions?
28. Tell me about a time you worked in a research or technical area you were not familiar with. [Reported] [21]
29. What is something you taught yourself in the last year (for example LLM fine-tuning, RL from human feedback, or a new framework), and how have you applied it? [Likely]
    - Where did you get stuck?
30. Tell me about a time curiosity about an anomaly in the data led to a bigger finding. [Likely]
31. How do you stay current with the literature, and tell me about a paper that changed how you work. [Likely]

### 6. Hire and Develop the Best

32. Tell me about a time you mentored a junior scientist, engineer or intern. [Likely]
    - How did you measure their growth?
    - What did you do when they struggled?
33. Tell me about a time you raised the science bar on your team through reviews, standards or reproducibility. [Likely]
34. Tell me about a time you gave someone difficult feedback. [Likely]
35. Tell me about a time you were involved in hiring or interviewing. What bar did you hold? [Likely]
    - Did you ever vote no on someone others liked?
36. Tell me about a time you helped a teammate move to a role or team where they could grow, even though your team lost them. [Likely]

### 7. Insist on the Highest Standards

37. Tell me about a time you raised the quality bar on a deliverable even though it delayed you. [Likely]
    - Who disagreed with the delay, and how did you handle it?
38. Tell me about a time you had to deliver a model under a tight deadline. What corners did you refuse to cut? [Likely]
39. Tell me about a time you found a bug or leak in someone else's experiment, such as data leakage or a bad A/B setup. [Likely]
    - How did you tell them?
40. Tell me about a time you were not satisfied with the quality of your own shipped work. What did you do about it? [Likely]
41. Tell me about a time insisting on high standards turned out to be the wrong call. [Likely]
    - How do you now decide when "good enough" is good enough?

### 8. Think Big

42. Tell me about the most ambitious science bet you have proposed. [Likely]
    - How did you sell it, and how far did it get?
43. Tell me about a time you turned a one-off model into a platform or capability other teams use. [Likely]
44. Where do you think your field will be in 3-5 years, and what would you build at Amazon toward that? [Likely] (prep sources describe a BR "bold vision" prompt; not seen verbatim in first-person reports)
45. Tell me about a time you challenged a team to aim for a much bigger goal than planned. [Likely]
    - What was the inspiring direction, in one sentence?
46. Tell me about a time your vision for a project was larger than your manager's. What happened? [Likely]

### 9. Bias for Action

47. Tell me about a time you had to make a decision with incomplete information. [Reported] [14] [20]
    - Was it a one-way or a two-way door?
    - What was your backup plan if you were wrong?
48. Tell me about a time you had to take a risk, or had very little time to decide. [Reported] [22]
49. Tell me about a time you took a calculated risk and it failed. [Reported] [22]
    - What did you learn, and would you take the same risk again?
50. Tell me about a time you did something without asking for your manager's approval. [Reported] [22]
    - How did your manager react afterwards?
51. Tell me about a time you shipped a quick baseline or experiment to unblock a decision instead of waiting for the full model. [Likely]
52. Tell me about a time you moved too fast. What broke? [Likely]

### 10. Frugality

53. Tell me about a time you did more with less: compute, labels or headcount. [Likely]
    - What specific constraint forced you to be creative?
54. Tell me about a time you reduced the training or inference cost of a model significantly. [Likely]
    - What did it cost in quality?
55. Tell me about a time you dealt with limited or no labeled data. [Likely]
    - Why not just buy more annotation?
56. Tell me about a time you chose not to use a large or expensive model (for example an LLM) because a cheaper one was sufficient. [Likely]

### 11. Earn Trust

57. Tell me about a time you failed, or the biggest mistake you made. [Reported] [20][22]
    - What signal existed before the failure that you missed? [Reported] [20]
    - How did you tell your customers or leadership? [Reported] [20]
58. Tell me about a time you received negative feedback. [Reported] [22]
    - What did you change, and how would the person who gave it know?
59. Tell me about a time you earned the trust of an engineering or product partner who doubted your science. [Likely]
60. Tell me about a time you had to deliver bad news, such as a model not working or a launch slipping. [Likely]
    - When did you know, and when did you tell them?
61. Tell me about a time you explained a complex model trade-off to a non-scientist stakeholder. [Likely]
62. Tell me about a time you were vocally self-critical about your own team's work. [Likely]

### 12. Dive Deep

63. Tell me about a time you had to dive deep into data or code to find a root cause. [Reported] [15] (asked opening or closing an AS coding round)
    - What was the first hypothesis, and how did you rule it out?
    - At what level of detail did you personally look at the raw data?
64. Tell me about a time your model underperformed and you had to diagnose the issue. [Reported] [12]
65. Tell me about a time a metric and anecdotes disagreed. Which did you believe? [Likely] (taken from the LP text)
66. Tell me about a time you found a data quality or pipeline problem that was silently degrading a model. [Likely]
    - How long had it been broken, and why didn't monitoring catch it?
67. Tell me about a time an A/B test result looked too good to be true. [Likely]
68. Walk me through one number in your resume. How exactly was it computed? [Likely] (BR "how was that measured?" probe reported [17][18])

### 13. Have Backbone; Disagree and Commit

69. Tell me about a time you had a conflict with a coworker or manager, and how you approached it. [Reported] [10][17][18]
    - What did you do once the decision went against you?
70. Tell me about a time you had a major conflict with a Principal Engineer (or senior scientist). How did you resolve it? [Reported] [13]
71. Tell me about a time you disagreed with your team's technical direction. What did you do? [Reported] [20]
    - What would you have done if you had been overruled and then proven right? [Reported] [20]
    - Tell me about a time you were the one who was wrong. [Reported] [20]
72. Tell me about a time you disagreed with a teammate or manager. [Reported] [22]
73. Tell me about a time you pushed back on a stakeholder's metric or launch decision using data. [Likely]
    - Did you commit after being overruled? How would we know you really committed?
74. Tell me about a time you disagreed with a senior scientist's technical approach. [Likely]
75. Tell me about a time you committed to a decision you disagreed with. How did it turn out? [Likely]
76. Tell me about a time you stopped, or tried to stop, a model launch. [Likely]

### 14. Deliver Results

77. Tell me about a time you had a tight deadline. [Reported] [21] (AWS AS phone screen)
    - If you did it again, what would you do better? [Reported] [21]
78. Tell me about a time you missed a deadline. [Reported] [22]
    - When did you realize it would slip, and who did you tell?
79. Tell me about your most challenging or proudest project, or a time you solved a complex problem. [Reported] [22]
    - What was the business result, in numbers?
80. Tell me about a time you delivered a model under a hard deadline. What did you descope? [Likely]
    - What did that descoping cost?
81. Tell me about a project failure and how you managed the aftermath. [Reported] [13]
    - What would you do differently? [Reported] [11]
82. Tell me about a time you hit a major setback mid-project, such as data being unavailable or a dependency slipping, and still shipped. [Likely]
83. Tell me about a time you had to choose between shipping something imperfect on time and slipping for quality. [Likely]

### 15. Strive to be Earth's Best Employer

84. Tell me about a time you made your team's work environment better. [Likely]
85. Tell me about a time you noticed a teammate struggling (workload, morale, inclusion) and acted. [Likely]
86. A senior manager assigned you work when you already had a full load. How did you handle it? [Likely] (seen in consolidated MLE behavioral banks; not verified as Amazon-specific)
87. Tell me about a time you made a process or on-call rotation less painful for others. [Likely]

### 16. Success and Scale Bring Broad Responsibility

88. Tell me about a time you considered the downstream harm or fairness impact of a model. [Likely]
    - What did you measure, and what did you change?
89. Tell me about a time you identified a secondary effect of your model on sellers, customers or another team that others had missed. [Likely]
90. Tell me about a time you handled privacy, safety or responsible-AI constraints on a model, such as hallucination in a customer-facing LLM feature. [Likely]
91. Tell me about a time you decided not to launch because the risk to customers outweighed the gain. [Likely]

---

## Bar raiser favorites and pressure probes

These are the questions and tactics most often attributed to the BR seat. The BR may come from any org, so expect plain-language questions, not science ones.

92. Tell me about your most significant failure. What happened, and what did you change afterwards? [Reported] [20] (described as "a Bar Raiser staple")
93. Tell me about a time you had a conflict with a coworker or manager (Have Backbone; Disagree and Commit). [Reported] [10][17][18]
94. (Pushback) "I think your approach was wrong. Convince me otherwise." [Reported] [17][18]
    - What data would make you concede?
95. How was that measured? [Reported] [17][18]
96. What was your exact individual contribution? Say it without "we." [Reported] [17][18]
97. (TPM or non-science BR) How do you set ETAs, map dependencies and communicate next steps on a science project? [Reported] [7]
    - When did a science timeline slip, and how did you tell stakeholders?
98. (TPM BR) Interpret this p-value or confidence interval for me. [Reported] [7]
99. Give me a second, different example where you showed the same principle. [Likely] (L6 rejections often cite a lack of multiple strong examples [8][9])
100. Why are you leaving your current role, and why Amazon now? [Likely]
101. What is the hardest feedback you have received from a manager, and what did you do with it? [Likely]
102. Tell me about a time you were not proud of how you behaved in a disagreement. [Likely]

## L5 calibration probes (any interviewer, layered on any story)

These follow-ups test whether a story is L5 scope rather than L4 execution. Expect 2-3 layers per story [15].

103. What was the largest scope you have owned end to end, from problem framing to production? [Likely]
     - Who else depended on it, beyond your own team?
     - What were you accountable for that your manager was not?
104. Tell me about a time your work changed a sister team's roadmap or technical direction. [Likely]
105. Tell me about a time you set the technical direction for other scientists or engineers, not just carried it out. [Likely]
     - How did you get buy-in without authority?
106. Where did you make a judgment call that your manager did not make for you? [Likely]
107. What mechanism did you build (a review, monitor or tool) that kept working without you in the room? [Likely]
108. If I called your skip-level manager, how would they describe your contribution to this project? [Likely]
109. What were the alternatives you rejected, and what data did you use to reject them? [Likely]
110. What would you do differently if you ran this project again today? [Reported] [11][21]

## Generic deepening probes interviewers attach to any LP story

These are not standalone questions. Expect a few of them after each story. [Likely, except where cited]

- What was the situation in one sentence, and what was your specific task?
- What did you personally do, as opposed to your team? [Reported] [17][18]
- What was the measurable result, and how was it measured? [Reported] [17][18]
- What was the hardest moment, and what options did you consider?
- Who disagreed with you, and what was their best argument?
- What did you learn, and where have you applied it since? [Reported] [14]
- How long ago was this? Do you have a more recent example?
- What would have happened if you had not acted?

## Deepening pass (October 2026): newly reported questions

Method: five web searches (Blind/Reddit, 1point3acres/Chinese "面经", LeetCode Discuss, Glassdoor, vendor guides). Direct page fetches remained blocked, so everything below rests on search-engine excerpts only. Several Glassdoor excerpts did not state the role; those items are marked "role unstated" and should be treated as Amazon-wide LP prompts, not AS-specific ones. Where a summary did not tie a quote cleanly to one review page, the citation names the review page that dominated that result set.

Format signals from this pass (not questions):
- Recent reports (Seattle 2025 and London 2024 science-track loops per the search summaries; Vancouver April 2025 explicitly AS) again describe a Bar Raiser round made up only of LP questions, with about 2 LP questions inside each other round [24][25][26].
- An AS II candidate (2026) said behavioral content took "maybe half the time or more" of each round and that interviewers pushed for specifics on their direct involvement [27].
- A 2025-grad Blind timeline describes a loop containing a separate "Gen AI Fluency" round alongside a Bar Raiser round (role not AS-specific) [28].

### Newly reported

111. Tell me about a difficult conversation you had with your manager. [Reported] [27] (AS II behavioral phone screen, Luxembourg, asked in STAR format)
     - What did you say first, and how did your manager react?
     - What changed in your working relationship afterwards?
112. Tell me about a time you had to compromise your standards in order to achieve a goal. [Reported] [29] (Amazon, role unstated; filed under Insist on the Highest Standards / Earn Trust)
     - Which standard did you relax, and who did you tell that you were relaxing it?
     - Would you make the same compromise today?
113. Tell me about a time you had a plan for what to propose to a customer, and changed your approach after a deep dive with that customer. [Reported] [29] (Amazon, role unstated; Dive Deep / Customer Obsession)
     - What did the deep dive show that your original plan had missed?
     - How did you explain the change of plan to your own team?
114. What feedback has your manager given you, and what did you do with it? [Reported] [25] (a candidate summary says many LP questions come back to the "feedback" part, for example feedback from your manager; this adds an AS source to Q58 and Q101)
     - Give me an example where you disagreed with that feedback.

### Gap-closing [Likely] questions for the L5 AS bar

115. Tell me how you use generative-AI tools (coding assistants, LLM APIs) in your own day-to-day science work, and a time one led you astray. [Likely] (prompted by the reported "Gen AI Fluency" round [28])
     - How did you verify the output before you relied on it?
116. Tell me about a time you wrote a science document (a design doc, PR/FAQ or 6-pager) that changed a decision. [Likely]
     - What was the strongest objection in the review, and how did you answer it?
117. Tell me about a time you had to explain why an experiment showed no significant effect, and what you recommended next. [Likely] (Are Right, A Lot / Earn Trust)
118. Tell me about a time you killed or deprioritized your own project because the data said it would not pay off. [Likely] (Ownership / Frugality)
     - How long did it take you to admit it, and what did you salvage?
119. Tell me about a time you had to reproduce or productionize research code written by someone else, and found it did not hold up. [Likely] (Dive Deep / Insist on the Highest Standards)
120. Tell me about a time you worked across time zones or with a remote partner team to land a launch. [Likely] (Deliver Results / Earn Trust)
121. Tell me about a time you had to choose between publishing a result and keeping it in-house or shipping it first. [Likely] (Think Big / Success and Scale)
122. Tell me about a time an LLM-based or generative feature you worked on produced harmful, biased or wrong output for customers. What did you do? [Likely] (Success and Scale Bring Broad Responsibility)
     - What guardrail or evaluation did you add afterwards?
123. Tell me about a time you balanced a research-quality goal against an operational metric such as latency, cost or on-call load. [Likely] (Frugality / Customer Obsession)
124. Tell me about a time you onboarded quickly to a team or codebase and delivered within your first few months. [Likely] (Learn and Be Curious / Deliver Results)

---

## References
1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Leadership Principles, amazon.jobs (Amazon, official). https://www.amazon.jobs/content/en/our-workplace/leadership-principles
3. "Amazon Bar Raiser", About Amazon (Amazon, official). https://aboutamazon.com/news/workplace/amazon-bar-raiser
4. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor (reports incl. 2025-2026). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
5. Amazon Applied Scientist Interview Experience & Questions, Glassdoor (reports incl. 2025-2026). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
6. "Amazon applied scientist interview feedback" (BR leveling at the L4/L5 cusp), Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-feedback-tvc4yec6
7. "My final interview for Amazon Applied Scientist-II will be taken by Principal TPM", Blind. https://www.teamblind.com/post/my-final-interview-for-amazon-applied-scientist-ii-will-be-taken-by-principal-tpm-cant-understand-what-tpm-will-ask-ryaa3tjo
8. "What are the expectations of Amazon L5 vs L6 for Applied Scientists?", Blind. https://www.teamblind.com/post/amazon-applied-scientists-l5-vs-l6-expectations-kegmp4rx
9. "Amazon applied scientist L6 to L5 downlevel", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l6-to-l5-downlevel-qvqtn0a8
10. Amazon Bar Raiser Interview (questions, prep tips), IGotAnOffer. https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview
11. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
12. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
13. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
14. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford, 2026. https://dataford.io/interview-guides/amazon/applied-scientist
15. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (vendor-published candidate debrief). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
16. Amazon Leadership Principles Interview (questions + tips), interviewing.io. https://interviewing.io/guides/amazon-leadership-principles
17. "Mastering the Amazon Bar Raiser Interview", 4 Day Week. https://4dayweek.io/interview-process/amazon-bar-raiser
18. "The Amazon Bar Raiser Interview: Who They Are and What They Ask", BarRaiser AI. https://barraiserai.com/guides/amazon-bar-raiser-interview
19. "Memoirs of an Amazon Bar Raiser", Carlos Arguelles, Medium / Geek Culture. https://medium.com/geekculture/memoirs-of-an-amazon-bar-raiser-718e36241310
20. AI Engineer Interview Questions: Amazon (curated guide to Amazon AI/science loops, incl. LP questions and BR follow-ups), GitHub (ombharatiya), 2026. https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/amazon.md
21. "interview_ds.md": Amazon AS/DS interview reports reposted from 1point3acres (AS loop with 30+ min LP per round and a "pure LP" BR; AWS AS phone screen tight-deadline BQ; "went above and beyond" BQs; fresh-PhD LP prompts), GitHub (goldin2008/Note). https://github.com/goldin2008/Note/blob/HEAD/interview_ds.md
22. "interview_bq_ans.md": a candidate's compiled list of frequently asked Amazon LP questions, mapped to LPs, with recruiter advice on story reuse, GitHub (goldin2008/Note). https://github.com/goldin2008/Note/blob/HEAD/interview_bq_ans.md
23. "Amazon Applied Scientist L5 Offer Eval", Blind, 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-offer-eval-c03bg14s
24. Amazon interview review (Seattle, 2025: phone screen, LeetCode medium, six-round virtual onsite with an LP-only Bar Raiser), Glassdoor (search excerpt only). https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
25. Amazon Applied Scientist interview review (Vancouver, April 2025: Bar Raiser "heavy on examples for the leadership principles"), Glassdoor France (search excerpt only). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW97603690.htm
26. Amazon interview review (London, July 2024, referral: Bar Raiser round fully behavioral/LP, about 2 LP questions in each other round), Glassdoor Netherlands (search excerpt only). https://static.glassdoor.nl/Interview/Amazon-Interview-E6036-RVW99272421.htm
27. Amazon Applied Scientist II interview review (Luxembourg STAR behavioral screen, "difficult conversation with a manager"; "half the time or more" behavioral), Glassdoor (search excerpt only; attribution to this review page inferred from the result set). https://www.glassdoor.com.ar/Entrevista/Amazon-Entrevista-E6036-RVW96627863.htm
28. "Amazon interview experience timeline: almost a 1 year process (2025 grad)", Blind (search excerpt only). https://www.teamblind.com/post/amazon-interview-experience-timeline-almost-a-1-year-process-2025-grad-0fbl4cvy
29. Amazon interview review ("compromise your standards to achieve a goal"; "changed the approach after deep dive with customer"; role unstated), Glassdoor (search excerpt only). https://www.glassdoor.co.uk/Interview/Amazon-Interview-E6036-RVW25065841.htm
