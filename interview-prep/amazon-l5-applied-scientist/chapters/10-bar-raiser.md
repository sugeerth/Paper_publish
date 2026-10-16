# 10. Bar Raiser Round: Reported Questions, Mixed Technical and Behavioral Probing, and Follow-up Patterns

> Scope: the one interview in the L5 Applied Scientist (Applied Scientist II) onsite run by the Bar Raiser (BR). It covers who the BR is, what they assess at L5, the questions candidates report, the questions the format makes likely, and the follow-up patterns BRs use to test whether a story holds up.
>
> Research note (please read): in this review pass, the web-search budget was already used up, and page fetches to the cited sites (4dayweek.io, igotanoffer.com, interviewkickstart.com, dataford.io, barraiserai.com, amazon.jobs) failed with DNS errors or proxy refusals. The one source family that could be read in full was the GitHub question-bank aggregator [28]-[32]. It was cloned and read directly; it says its content is based on candidate reports, mostly from Q1-Q3 2026. Every question tagged [Reported] that cites [28]-[32] was checked against that text. References [1]-[27] carry over from chapters 01-08 of this guide and were **not re-fetched in this pass**. Their [Reported] tags rest on search-result extracts gathered earlier, so treat them as unverified until someone re-opens the pages.

## Format notes

| Aspect | What is known | Sources |
|---|---|---|
| Length and slot | One of the onsite loop interviews. Amazon's official AS page describes the loop as four 55-minute interviews with members of the science community. Candidates more often report 5 rounds of 55-60 minutes, one of them labeled "Bar Raiser". The BR round is sometimes held on a separate day, from the day before the loop to weeks after it. | [1][8][9][13][28] |
| Who the BR is | A trained interviewer from outside the hiring team. BRs are nominated, trained, and shadow veteran BRs before they run debriefs. Science candidates report BRs from non-tech orgs, and one AS II candidate was told the final interview would be a Principal TPM. | [2][3][5][6][10][29] |
| Power | The BR chairs the debrief and, with the hiring manager, makes the hire/no-hire call. The BR holds an effective veto that former Amazonians say is rarely used explicitly. | [2][3][7][12][29] |
| What is assessed | For scientists the round is usually entirely behavioral: 3-4 LP questions, each followed many layers down. Customer Obsession, Learn and Be Curious and Insist on the Highest Standards are the LPs named for science BRs. In one L6 loop the BR also asked for coding (a priority queue from scratch). The "better than 50% of current employees at that level" line is BR folklore, not a published Amazon standard. | [4][5][6][28][29][30] |
| ML content in stories | Science loops assign each round 2-3 LPs; one five-round loop covered ten different LPs. Interviewers expect stories to contain an ML decision about models, data or experiments, and one team required ML work in every story. "A disagreement about an evaluation metric beats one about meeting schedules." | [28][29][30] |
| L5-specific bar | L5 means owning a project end to end, making the technical calls, and influencing peers and sometimes sister teams. A vendor-published debrief says L5 LP answers face 2-3 layers of follow-up, against 1 layer at L4. Blind commenters say LP narratives, not coding, separate L5 from L6. Scope is judged against the target level, so solid work can still be marked down for insufficient impact. | [14][15][18][29][30] |
| Leveling | A BR can push the level down even when the HM is positive. One L5 candidate's BR saw them "at the cusp of L4 and L5". Downleveling from L6 to L5 and from L5 to L4 is commonly reported. | [11][15] |
| Behavioral weight overall | Amazon says each interviewer typically asks 2-3 behavioral questions. One candidate counted 17 LP questions in a four-hour loop. In one five-round science loop, LP questions left only about 15 minutes for coding in the fourth hour. | [1][16][18][28][29] |
| Split decisions and cooldowns | When the debrief splits, candidates report an extra round, for example one more coding round for an AGI-team AS II. Recent science and ML rejections carried cooldowns of 6-12 months. | [17][29][30] |

### Observed probing patterns (what makes the BR round different)

- **Layered "why" and "how do you know":** 3-4 layers on one story: why that path, which alternatives, what data, what you would change. [18][19][21][22][28]
- **"I" not "we":** BRs separate your individual contribution from the team's, down to dates, data and your exact role. [21][22][28][30]
- **Deliberate pushback:** candidates report BRs saying the approach was wrong, or repeatedly saying "that is not what I'm looking for", to see whether the candidate defends with data or switches to a better story. [21][22][29][30]
- **After-commit focus:** on disagree-and-commit stories, the grade turns on what you did after committing, for example quietly reducing the risk you had raised. [28][29][31]
- **Metric interrogation:** "How was that measured?", including baselines, attribution and counterfactuals. [21][22]
- **Lay-audience translation:** a non-science BR (for example a Principal TPM, or a backend engineer BR in another track) asks for project-management mechanics, basic stats interpretation, or technical trade-offs in plain language. [10][32]
- **Closing sweep:** some BRs end with an open invitation to share anything earlier interviewers missed. [32]

---

## Questions

Tag legend:

- **[Reported]**: a candidate or source attributes this question, or a close paraphrase, to an Amazon BR round or to Amazon AS/science LP questioning. When the report comes from another round or track, the item says so in parentheses.
- **[Likely]**: an inference grounded in the BR format, the 16 LPs, the L5 bar and Amazon's science products. It has not been seen verbatim.

Indented bullets are follow-up probes. In a BR round, expect at least two of them per story.

### A. Opening and scope calibration (sets the level the BR will argue in the debrief)

1. Walk me through your career and the one project that best shows the level you are operating at today. [Likely]
   - What part of that project would not have happened without you?
   - Who else could have done what you did on that project?
   - If I called your manager, what would they say was your scope?
2. Tell me about the largest-scope science project you have owned end to end, from problem framing to production. [Likely]
   - Where did the problem come from: did you find it or were you handed it?
   - What was the decision you made that most changed the outcome?
   - How many people or teams depended on your output?
3. Describe your exact individual contribution to that project, without saying "we". [Reported] [21][22][28]
   - Which design document or analysis did you write yourself?
   - What did you delegate, and why?
   - That sounds like a team result. What specifically did you do? [Reported] [21][22]
4. Tell me about a time your work influenced a team other than your own. [Likely]
   - How did they adopt it: mechanism, code, paper or persuasion?
   - What would have happened if they had said no?
5. What is the hardest technical problem you have solved in the last two years? [Likely]
   - Why was it hard for you specifically, and not just for the field?
   - Explain it so someone outside ML understands why it mattered.
6. Give me your flagship project in two minutes. Now take ten minutes and go deeper on the part you are proudest of. [Likely] [29]
   - What did you leave out of the two-minute version that a skeptic would ask about?

### B. Conflict, backbone and disagree-and-commit

7. Tell me about a time you had a conflict with a coworker or manager, and how you handled it. [Reported] [19][21][22][29][30]
   - What did you do once the decision went against you?
   - Looking back, who was right?
   - How is your relationship with that person now?
8. Tell me about a time you had a major conflict with a Principal Engineer (or a more senior scientist). How did you resolve it? [Reported] [24]
   - What data did you bring to the conversation?
   - Did you escalate? Why or why not?
9. Tell me about a time a client or decision maker proposed a plan that was far too aggressive. What did you do? [Reported] [29][30]
   - What evidence showed you the plan was unrealistic?
   - What did you counter-propose, and what did they accept?
10. Tell me about a time you disagreed with a product manager's launch decision based on your model's results. [Likely]
    - What metric were they looking at that you thought was wrong?
    - What happened after launch, and were you proven right?
    - What would have convinced you to commit to their plan?
11. Tell me about a time you disagreed with a manager or challenged a team decision, committed anyway, and what happened next. [Reported] [28][30][31]
    - After you committed, how did you reduce the specific risk you had raised? [Reported] [28][31]
    - How did you make sure your doubts did not slow the team down?
    - Did you ever revisit the decision? What triggered that?
12. Tell me about a disagreement with a colleague over which evaluation metric or baseline to use. [Likely] [29]
    - How did you settle it: argument, experiment or escalation?
    - Which metric does the team use today?
13. Tell me about a time an engineer said your model could not ship in its current form. [Likely]
    - What did you give up to get it into production?
    - Who made the final call?
14. Tell me about a time you had to tell a senior leader that their idea would not work scientifically. [Likely]
    - How did you frame it?
    - What did they do next?
15. Describe a time you saw your team heading in the wrong direction. What did you do? [Reported] [32] (front-end BR round, same format)
    - How did you know before the results showed it?
    - Whose decision was it to change course?
16. Tell me about a time you raised a risk, were overruled, and the risk then happened. [Likely]
    - What did you say afterwards, and to whom?

### C. Failure, being wrong and learning

17. Tell me about a project failure and how you managed the aftermath. [Reported] [24]
    - What would you do differently? [Reported] [20][28]
    - What mechanism did you put in place so it does not happen again?
18. Tell me about a time your model underperformed in production and you had to diagnose the issue. [Reported] [23][30]
    - What did you learn, and how did you pivot? [Reported] [25]
    - How long did it take you to notice, and why that long?
19. Tell me about a time you were wrong about a technical call. How did you find out? [Likely]
    - Who told you, or did you find it yourself?
    - What did it cost the team?
    - Did you have to admit it in front of the team or stakeholders?
20. Tell me about a time you got critical feedback that surprised you. [Reported] [28] (listed as a recurring BR theme)
    - What did you change afterwards, and how would someone see that change?
21. What do you most need to improve, and what have you changed so far? [Reported] [31] (hiring-manager round)
    - What evidence would your manager point to that you have improved?
22. Tell me about an experiment or research direction you killed. [Likely]
    - How did you decide it was dead rather than under-tuned?
    - How much effort had gone in, and how did you tell the team?
23. Describe a deadline you were going to miss: when you realized it, who you told, and what you proposed. [Reported] [28]
    - Why didn't you see it earlier?
24. Tell me about a time AI gave you a wrong answer. How did you catch it? [Reported] [28][31]
    - What do you now check by hand that you used to trust?
    - How did you realize the output of a generative AI tool was not good enough, and what did you do? [Reported] [31] (hiring-manager round)
25. How would you test code or analysis that an AI assistant generated for one of your experiments? [Reported] [31] (hiring-manager round; adapted from "AI-generated code")
    - What failure would slip past your unit tests but break the experiment?

### D. Decisions under ambiguity (Bias for Action, Are Right a Lot)

26. Describe a decision you made with incomplete information and how you later validated it. [Reported] [25][29]
    - What data would you have wanted, and why couldn't you wait for it?
    - How did you limit the downside if you were wrong?
    - Was it a one-way or two-way door? What would you have done if it had turned out wrong?
27. Describe a calculated risk you took: what you knew at the time and how it turned out. [Reported] [28][29][30]
    - What was the worst case you planned for?
    - Who else knew about the risk before you took it?
28. Tell me about a time you took initiative without waiting for permission. [Likely]
    - Whose permission would you normally have needed?
    - What did your manager say afterwards?
29. Tell me about a time you had to choose between a quick heuristic and a principled model. [Likely]
    - What did you ship first, and did the principled model ever replace it?
30. Tell me about a time two offline metrics disagreed and you had to make a ship call. [Likely]
    - Which metric did you trust, and why?
31. Tell me about a time the online A/B result fell short of, or contradicted, your offline gain. How did you find out why? [Reported] [29][30] (science-depth and project-dive rounds)
    - What was the root cause?
    - Did you change your offline evaluation afterwards?
32. Tell me about a time you settled a technical debate with a quick experiment instead of more argument. [Likely]
    - How did you make sure both sides would accept the result?

### E. Ownership and Deliver Results (the core L5 signal)

33. Tell me about a time you took ownership of something outside your formal scope. [Reported] [28] (listed as a recurring BR theme)
    - Why didn't the actual owner handle it?
    - What did you stop doing to make room?
34. Tell me about a time you delivered a model under a hard deadline. [Likely]
    - What did you descope, and what did that cost?
    - What corners did you refuse to cut?
35. Tell me about the most significant business result your science has driven. [Likely]
    - How was that measured? [Reported] [21][22]
    - How do you know it was the model and not seasonality, a UI change or a marketing push? What was the counterfactual?
    - What was the baseline?
36. Tell me about a time a project was about to fail and you turned it around. [Likely]
    - What was the first thing you did differently?
37. Tell me about a time you had to keep a model working long after launch (monitoring, retraining, on-call). [Likely]
    - Who owned it when it broke at 2am?
38. Tell me about a time you were blocked by another team. [Likely]
    - What did you do in the first week?
    - Did you find a way around them, or did you wait?
39. Tell me about a time two partner teams wanted your model to optimize different metrics. [Likely]
    - Whose metric won, and how was that decided?

### F. Customer Obsession and Working Backwards, for a scientist

40. Tell me about a time you went above and beyond for a customer. [Reported] [29][30]
    - Was the customer internal or external?
    - What did going beyond cost you or the team?
41. Tell me about a time you started from the customer and worked backwards to the science problem. [Likely]
    - Who was the customer, and how did you get their signal?
42. Tell me about a time you chose a simpler model because it was better for the customer. [Likely]
    - What accuracy did you give up, and how did you justify it?
43. Tell me about a time a model you built hurt some customers even though the average metric improved. [Likely]
    - How did you find that segment?
    - What did you do about it?
44. Tell me about a time you used customer anecdotes to overturn what the data said. [Likely]
    - How many anecdotes was enough?
45. If you wrote a press release for your best project, what would the headline and customer quote be? [Likely]
    - What would the FAQ's hardest question be?

### G. Dive Deep and scientific rigor (the technical side of the BR round)

46. Describe a time you found an issue nobody else had noticed: how you isolated it, and why your first hypothesis was wrong. [Reported] [18][27][31][32]
    - What was the first anomaly you noticed?
    - How did you rule out the other explanations?
47. Tell me about a time you found a data-quality or label-leakage problem in your own work. [Likely]
    - How long had it been in production?
    - Which numbers did you have to retract?
48. Walk through a major technical decision from your resume, the alternatives you rejected, and whether you would make the same choice again. [Reported] [28][31]
    - What experiment told you the alternative was worse?
    - Who did you have to convince?
49. For your main project or publication, what was your specific contribution, and how did you choose the baselines? [Reported] [29] (science-depth round)
    - Which strong baseline did you leave out, and why?
50. Walk me through how you validated that your evaluation set reflected production traffic. [Likely]
    - What distribution shift did you see after launch?
51. Tell me about a time a result looked too good to be true. [Likely]
    - What did you check first?
52. Tell me about a time you had to defend a statistical result to a skeptical audience. [Likely]
    - What was the effect size, and was it practically significant?
53. You mentioned an improvement of X%. What was the confidence interval? [Likely]
    - How many samples, and how long did the test run?
54. Tell me about a time you reproduced a paper's results and could not match them. [Likely]
    - What did you conclude, and what did you ship instead?
55. What is the biggest weakness of your flagship model that I have not asked about yet? [Likely] [29]
    - What would it take to fix it?

### H. Technical checks from a non-science Bar Raiser (TPM, SDE or business BR)

56. How do you set ETAs, map dependencies and communicate next steps on a science project? [Reported] [10]
    - When did a science timeline slip, and how did you tell stakeholders?
    - How do you estimate a project when you do not know if the idea will work?
57. Interpret a p-value or a confidence interval for me. [Reported] [10]
    - What would you tell a business leader who reads p = 0.06 as "no effect"?
58. Explain your most recent model to me as if I were a business leader with no ML background. [Likely]
    - What decision does it let someone make that they could not make before?
59. Tell me about a technique you learned on your own and applied. How did it hold up across the conditions it had to handle? [Likely] [32] (asked in a front-end BR about a front-end technology across devices; adapted here to ML)
    - Explain the trade-off to me as a non-specialist.
60. How do you decide when a research prototype is ready to hand to engineering? [Likely]
    - What does the handoff document contain?
61. Your science project needs six months but the business wants results in six weeks. What do you do? [Likely]
    - What would a six-week version look like?
62. How do you know a model is still working a year after launch? [Likely]
    - Who gets paged, and what is the rollback plan?
63. Tell me about a time you had to explain a model's mistake to a customer-facing or operations team. [Likely]

### I. Invent and Simplify, Think Big, Frugality, Learn and Be Curious

64. Tell me about a time you simplified a complex scientific approach so it could ship. [Likely]
    - What did the complex version get right that the simple one missed?
65. Tell me about the most ambitious science bet you have proposed. [Likely]
    - Who funded it, and what did you have to prove first?
    - If it had been fully funded, what would the three-year version be?
66. Tell me about a time you did more with less: compute, labels or headcount. [Likely]
    - What was the cost of the cheaper approach in model quality?
67. Tell me about something you invented, such as a method, tool or mechanism, that others now use. [Likely]
    - How did you get people to adopt it?
68. Tell me about a time you used an LLM or foundation model where a custom model would have been the default. [Likely]
    - How did you weigh inference cost against development time?
69. Tell me about a time curiosity about an odd pattern in data led you to a new direction for the project. [Likely] [29] (Learn and Be Curious, named for science BRs)
    - How much time did you spend before you told anyone?
70. Tell me about a time you learned a new area fast to unblock a project. [Likely]
    - How did you know you had learned enough?

### J. Earn Trust, Hire and Develop the Best, influence without authority

71. Tell me about a time you earned the trust of an engineering or product partner who doubted your science. [Likely]
    - What was the turning point?
72. Tell me about a time a teammate's work did not meet the bar, and what you did. [Reported] [29]
    - Did you fix it yourself or help them fix it? Why?
    - How did they react, and what changed afterwards?
73. Tell me about a time you mentored someone or raised the bar of your team's science. [Likely]
    - How do you know they improved?
74. Tell me about a time you influenced a decision without having authority over the people involved. [Likely]
    - Who was the hardest person to convince?
75. Tell me about a time you reviewed someone else's experiment and found a serious flaw. [Likely]
    - How did you raise it?

### K. Insist on the Highest Standards, Broad Responsibility

76. Tell me about a time you raised the quality bar on a deliverable even though it delayed you. [Likely] [29]
    - Who disagreed with the delay?
77. Tell me about a time you refused to launch something that was not ready. [Likely]
    - What was the minimum bar, and who set it?
78. Tell me about a time you insisted on better labels or data quality when the team wanted to train on what they had. [Likely] [29]
    - How did you show the cost of the noisy data?
79. Tell me about a time you considered the downstream harm or fairness impact of a model. [Likely]
    - How did you measure it?
    - What trade-off did you accept?
80. Tell me about a time you handled sensitive customer data in a modeling project. [Likely]
    - What would you not do even if it improved the model?

### L. Deliberate pushback and stress probes (BR-specific)

81. I think your approach was wrong. Convince me otherwise. [Reported] [21][22]
    - What data would make you concede?
    - What would the alternative have cost?
82. That is not what I'm looking for. Give me a different example for the same principle. [Reported] [29][30]
83. That example is from three years ago. Do you have a more recent one? [Likely]
84. Your manager made that call. Where was your judgment in that? [Likely]
85. If you had not been on the project, what would have been different? [Likely]
86. What was the worst decision you made on that project? [Likely]
87. What would your harshest critic on that team say about how you handled it? [Likely]
88. You said the result was a success. Who would disagree? [Likely]
89. Stop the story there. What exactly did you say in that meeting? [Likely]

### M. Leveling probes: L4 vs L5 vs L6

90. Tell me about a time you set the technical direction for a project, not just executed it. [Likely]
    - Who was in the room when the direction was decided?
91. Tell me about a time you identified a science problem nobody had asked you to solve. [Likely]
    - How did you get time and resources for it?
92. Tell me about a mechanism (process, tool, review) you created that outlived your involvement. [Likely]
    - Who runs it now?
93. Tell me about a time you had to coordinate science work across several people or teams. [Likely]
    - How did you divide the work, and what broke?
94. What is the biggest ambiguity you have resolved, from vague goal to concrete plan? [Likely]
    - What was the first artifact you produced?
95. How would your current manager describe the gap between you and the next level? [Likely]
    - What are you doing about it?
96. What was the largest measured business effect of your work? [Likely] [29][30] (scope is judged against the target level)
    - If it were twice as big, what would you have done differently?

### N. Mixed technical-behavioral prompts (BR who is a scientist)

97. Walk through a ranking (or similar) project: the pain point, why this model, the metrics, and how you diagnosed a disappointing A/B result. [Reported] [30] (project deep-dive round)
    - What would you change in the experiment design today?
98. For a model you built, why did you choose that backbone, how did you handle class imbalance, and what would you do with more resources? [Reported] [29][30] (science-depth and project-dive rounds)
    - Which of those choices would you defend to a skeptical Principal Scientist?
99. Explain an AI-agent or LLM project you built and how it got adopted. [Reported] [30] (project deep-dive round)
    - Who used it in the first month, and who stopped using it?
100. Tell me about a time a technical trade-off (latency, cost, accuracy) forced a business compromise. [Likely]
     - Who signed off on the compromise?
101. Tell me about a time you moved a model from research to production at scale. [Likely]
     - What failed first in production that never failed offline?
102. Tell me about a time you had to choose between publishing and shipping. [Likely]
     - What did the business get out of the choice?
103. Tell me about a GenAI or LLM feature you built or evaluated. How did you decide it was safe to launch? [Likely]
     - How did you measure hallucination, and what threshold did you accept?
104. Tell me about a time an evaluation metric you chose turned out to be the wrong proxy. [Likely]
     - How did you detect the mismatch?
105. (Senior BR adding coding) Implement a priority queue from scratch. [Reported] [28] (reported in an L6 loop; possible but less likely at L5)
     - What is the cost of each operation, and how would you support decrease-key?

### O. Post-loop, split-debrief and closing scenarios

106. (Extra round after a split debrief) Let's revisit coding with a follow-up problem. [Reported] [17]
107. (Downlevel conversation) We think your scope maps to L4. Would you consider that level? [Likely] (BR-driven downleveling reported in [11][15])
     - What evidence from your past work would change our assessment?
108. Why Amazon, and why this team? [Reported] [28][33]
     - Which LP do you find hardest to live by?
109. Is there anything the earlier interviewers did not ask that you want me to know? [Reported] [32]
     - Which LP does that story show?
110. What questions do you have for me about how Amazon hires and develops scientists? [Likely]

## Deepening pass (October 2026): newly reported questions

> Method: five web searches (Blind/Reddit, 1point3acres in Chinese, Glassdoor/Prepfully, LP-phrase search, PracHub). Page fetches were not attempted; every tag below rests only on the search-result title or snippet. Blind, Reddit and 1point3acres returned almost nothing specific to the AS II Bar Raiser. Format signals from the snippets: a 2025 Seattle PhD candidate said their Bar Raiser "only focused on Leadership Principles" while the other rounds were technical, alongside a separate tech-talk round [35]. A 2025 Vancouver AS candidate summarized the BR as "a lot of examples for the leadership principles" [36]. A 2025 new-grad Blind post (role not stated) reported a BR of about 22 minutes [33]. Existing question 108 (Why Amazon) now also cites [33].

### New [Reported] questions

111. Introduce yourself. [Reported] [33] (BR opener in a 2025 new-grad loop, role not stated)
     - Which part of that background is most relevant to this team?
112. Give me an example that shows Dive Deep. [Reported] [33] (the LP was named in the snippet; exact wording not given)
     - What was the lowest level of detail you personally checked?
113. Give me an example that shows Learn and Be Curious. [Reported] [33] (LP named only; compare question 69)
     - What did you learn that changed how you work today?
114. Give me an example that shows Earn Trust. [Reported] [33] (LP named only; compare question 71)
     - How did you rebuild trust after it had been damaged?
115. Tell me about a time you showed leadership in a challenging situation and had to make a key decision. [Reported] [34] (Data Scientist onsite, Bar Raiser behavioral loop)
     - What options did you reject, and why?
     - How did you communicate the decision to people who disagreed?
116. How did you keep a team's morale up while it was under heavy pressure? [Reported] [34] (Data Scientist onsite, Bar Raiser behavioral loop)
     - How did you know morale was slipping?
     - What did you do that cost you something personally?

### New [Likely] questions (gap closers for the L5 bar)

117. Tell me about a time new data made you change your mind on a science decision you had argued for. [Likely] (Are Right a Lot)
     - How quickly did you change course, and who did you tell?
118. Tell me about a time a customer or partner team asked for a model feature you thought was the wrong solution. [Likely] (Customer Obsession vs. what the customer asked for)
     - What did you build instead, and did they agree afterwards?
119. Tell me about a time you had to deliver after a key teammate or collaborator left mid-project. [Likely] (Deliver Results)
     - What did you drop, and who agreed to that?
120. Tell me about a time you wrote a six-pager or science document that changed a team's plan. [Likely] (written narrative culture)
     - What was the hardest question in the review, and how did you answer it?
121. Tell me about a time you deliberately chose not to pursue a promising research idea because of cost or scale. [Likely] (Frugality, Think Big trade-off)
     - How did you estimate the cost?
122. Tell me about a time you made a fast, reversible decision on a model or experiment and later had to undo it. [Likely] (Bias for Action, two-way doors)
     - What did undoing it cost?
123. Tell me about a time you interviewed or helped hire a scientist, and how you judged the bar. [Likely] (Hire and Develop the Best)
     - What signal made you vote no?
124. Tell me about a time a launch you owned created an unintended effect for customers or employees outside your team. [Likely] (Success and Scale Bring Broad Responsibility)
     - How did you find out, and what did you change?
125. Tell me about a time you made your team's work easier or more enjoyable, beyond your own deliverables. [Likely] (Strive to be Earth's Best Employer)
     - How did you know it helped?
126. Pick the story you would least like me to probe. Walk me through it now. [Likely] (BR stress probe on story consistency)
     - Which detail in it would you most like to restate?

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep (not re-fetched in this pass)
2. "Amazon Bar Raiser", About Amazon (Amazon, official). https://aboutamazon.com/news/workplace/amazon-bar-raiser (not re-fetched)
3. "Hire power: How Amazonians raise the bar with every interview", About Amazon (Amazon, official). https://www.aboutamazon.com/news/workplace/hire-power-how-amazonians-raise-the-bar-with-every-interview (not re-fetched)
4. "Amazon bar raisers, entry-level interview hiring procedure", Fortune, 2024. https://fortune.com/2024/08/07/amazon-bar-raisers-entry-level-interview-hiring-procedure (not re-fetched)
5. "Memoirs of an Amazon Bar Raiser", Carlos Arguelles, Medium / Geek Culture. https://medium.com/geekculture/memoirs-of-an-amazon-bar-raiser-718e36241310 (not re-fetched)
6. "Amazon Bar Raiser: Who They Are & How to Pass the Loop", Carrus. https://www.carrus.io/blog/all-about-bar-raisers-amazons-essential-element-to-the-hiring-process (not re-fetched)
7. "Working Backwards, by Colin Bryar and Bill Carr", Evan's Notes, Substack (book summary covering the debrief and BR process). https://evansamek.substack.com/p/working-backwards-by-colin-bryar (not re-fetched)
8. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm (not re-fetched)
9. Amazon Applied Scientist Interview Experience & Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm (not re-fetched)
10. "My final interview for Amazon Applied Scientist-II will be taken by Principal TPM", Blind. https://www.teamblind.com/post/my-final-interview-for-amazon-applied-scientist-ii-will-be-taken-by-principal-tpm-cant-understand-what-tpm-will-ask-ryaa3tjo (not re-fetched)
11. "Amazon applied scientist interview feedback" (leveling dispute with bar raiser), Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-feedback-tvc4yec6 (not re-fetched)
12. "Amazon bar raiser veto", Blind. https://www.teamblind.com/post/amazon-bar-raiser-veto-ybkwpkzv (not re-fetched)
13. "Amazon Applied Scientist (L5) Virtual Onsite", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml (not re-fetched)
14. "What are the expectations of Amazon L5 vs L6 for Applied Scientists?", Blind. https://www.teamblind.com/post/amazon-applied-scientists-l5-vs-l6-expectations-kegmp4rx (not re-fetched)
15. "Amazon applied scientist L6 to L5 downlevel", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l6-to-l5-downlevel-qvqtn0a8 (not re-fetched)
16. "Amazon Applied Scientist L5 Offer Eval", Blind, 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-offer-eval-c03bg14s (not re-fetched)
17. "Amazon (AGI team) Applied Scientist II interview", Blind (extra round after split debrief). https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT (not re-fetched)
18. "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP", OA VO Service (vendor-published candidate debrief, promotional). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding (not re-fetched)
19. "Amazon Bar Raiser Interview (questions, prep tips)", IGotAnOffer. https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview (fetch failed in this pass: DNS/proxy refusal)
20. "Amazon Applied Scientist Interview (process, questions, prep)", IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview (not re-fetched)
21. "Mastering the Amazon Bar Raiser Interview", 4 Day Week. https://4dayweek.io/interview-process/amazon-bar-raiser (fetch failed in this pass: DNS/proxy refusal)
22. "The Amazon Bar Raiser Interview: Who They Are and What They Ask", BarRaiser AI. https://barraiserai.com/guides/amazon-bar-raiser-interview (fetch failed in this pass: DNS refusal)
23. "The 2026 Amazon Applied Scientist Interview Guide", Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist (fetch failed in this pass: DNS refusal)
24. "The 2026 Amazon Senior Applied Scientist Interview Guide", Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist (fetch failed in this pass: DNS refusal)
25. "Amazon Applied Scientist Interview Questions & Guide 2026", Dataford, 2026. https://dataford.io/interview-guides/amazon/applied-scientist (fetch failed in this pass: DNS refusal)
26. "Amazon Leadership Principles Interview (questions + tips)", interviewing.io. https://interviewing.io/guides/amazon-leadership-principles (not re-fetched)
27. "Senior Engineer's Guide to Amazon Interviews + Questions", interviewing.io. https://interviewing.io/guides/hiring-process/amazon (not re-fetched)
28. "Amazon Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator mirrored from TrueInterview; Bar Raiser section with recurring themes, sample questions, the L6 BR coding report and the GenAI-wrong-answer question). Read in full in this pass, October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process.md
29. "Amazon Research Scientist Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (covers Research and Applied Scientist loops; says it is based on 20 candidate reports, mostly from Q1-Q3 2026; Bar Raiser and science-depth sections). Read in full in this pass. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
30. "Amazon Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (MLE and AS loops; Bar Raiser and project deep-dive sections). Read in full in this pass. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
31. "Amazon Software Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (Bar Raiser and hiring-manager sections; disagree-and-commit grading, GenAI questions). Read in full in this pass. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-software-engineer.md
32. "Amazon Frontend Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (Bar Raiser section: separate-day BR, backend-background BR, closing "anything else" invitation). Read in full in this pass. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-frontend-engineer.md
33. "Amazon interview experience timeline, almost a 1 year process, 2025 grad", Blind (BR about 22 minutes: introduction, Why Amazon, Dive Deep, Learn and Be Curious, Earn Trust; role not stated). Search snippet only, October 2026. https://www.teamblind.com/post/amazon-interview-experience-timeline-almost-a-1-year-process-2025-grad-0fbl4cvy
34. Amazon Interview Questions (page 14), PracHub ("Demonstrate Leadership in Challenging Situations and Decision-Making", Amazon Data Scientist Onsite, Bar-Raiser Behavioral Loop; morale-under-pressure prompt). Search snippet only, October 2026. https://prachub.com/companies/amazon?page=14
35. Amazon interview review RVW96903377, Glassdoor (2025 Seattle PhD Applied Scientist candidate: tech talk round plus a Bar Raiser that "only focused on Leadership Principles"). Search snippet only, October 2026. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW96903377.htm
36. Amazon interview review RVW97603690, Glassdoor (2025 Vancouver Applied Scientist: "Bar-Raiser: A lot of examples for the leadership principles"). Search snippet only, October 2026. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW97603690.htm
