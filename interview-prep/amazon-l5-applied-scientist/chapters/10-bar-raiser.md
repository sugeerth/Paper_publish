# 10. Bar Raiser Round: Reported Questions, Mixed Technical and Behavioral Probing, and Follow-up Patterns

> Scope: the one interview in the L5 Applied Scientist (Applied Scientist II) onsite run by the Bar Raiser (BR). It covers who the BR is, what they assess at L5, the questions candidates report, the questions the format makes likely, and the follow-up patterns BRs use to test whether a story holds up.
>
> Research note (please read): no live web research worked for this chapter. The run's shared web-search budget was already used up, and direct page fetches (igotanoffer.com, medium.com, aboutamazon.com, teamblind.com) failed with DNS errors in this environment. Every [Reported] tag below therefore comes from findings already gathered and cited in chapters 01-08 of this guide. Those findings were themselves built from search-result extracts of the cited pages, not full reads. Each [Reported] item cites the original page, and anything those chapters did not attribute to an Amazon interview is tagged [Likely]. A follow-up session with a fresh search budget could add more first-person BR reports, especially from Blind, 1point3acres and Glassdoor individual reviews.

## Format notes

| Aspect | What is known | Sources |
|---|---|---|
| Length and slot | One of the onsite loop interviews. Amazon's official AS page describes the loop as four 55-minute interviews with members of the science community. Candidates more often report 5 rounds of 55-60 minutes, one of them labeled "Bar Raiser". | [1][8][9][13] |
| Who the BR is | A trained interviewer from outside the hiring team. BRs are nominated, trained, and shadow veteran BRs before they run debriefs. Science candidates report BRs from non-tech orgs, and one AS II candidate was told the final interview would be a Principal TPM. | [2][3][5][6][10] |
| Power | The BR chairs the debrief and, with the hiring manager, makes the hire/no-hire call. The BR holds a de facto veto that former Amazonians say is rarely used explicitly. | [2][3][7][12] |
| What is assessed | Mostly Leadership Principles (LPs), with technical probing possible. The BR asks whether the hire raises the bar for the role and level. The "better than 50% of current employees at that level" line is BR folklore, not a published Amazon standard. | [4][5][6][8] |
| L5-specific bar | L5 means owning a project end to end, making the technical calls, and influencing peers and sometimes sister teams. A vendor-published debrief says L5 LP answers face 2-3 layers of follow-up, against 1 layer at L4. Blind commenters say LP narratives, not coding, separate L5 from L6. | [14][15][18] |
| Leveling | A BR can push the level down even when the HM is positive. One L5 candidate's BR saw them "at the cusp of L4 and L5". Downleveling from L6 to L5 and from L5 to L4 is commonly reported. | [11][15] |
| Behavioral weight overall | Amazon says each interviewer typically asks 2-3 behavioral questions. One L5 report put behavioral content above 60% of the loop. Vendor debriefs put about 15 minutes of each technical round on LPs. The BR round is where LPs get the most time. | [1][16][18] |
| Split decisions | When the debrief splits, candidates report an extra round, for example one more coding round for an AGI-team AS II. | [17] |

### Observed probing patterns (what makes the BR round different)

- **Layered "why" and "how do you know":** 2-3 levels of follow-up on one story, ending in metrics, mechanisms and your personal decisions. [18][19][21][22]
- **"I" not "we":** BRs separate your individual contribution from the team's. [21][22]
- **Deliberate pushback:** candidates report BRs telling them their approach was wrong to see whether they defend it with data or fold. [21][22]
- **Metric interrogation:** "How was that measured?", including baselines, attribution and counterfactuals. [21][22]
- **Lay-audience translation:** a non-science BR (for example a Principal TPM) asks for project-management mechanics and basic stats interpretation. [10]
- **Story reuse checks:** because each interviewer owns about 2-3 LPs, a BR may return to an LP that another interviewer covered weakly, or ask for a second, different example. This comes from prep guides and is [Likely], not confirmed. [23][26]

---

## Questions

Tag legend:

- **[Reported]**: a candidate or source attributes this question, or a close paraphrase, to an Amazon BR round or Amazon AS/science LP questioning. Citations point to the References.
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
3. Describe your exact individual contribution to that project, without saying "we". [Reported] [21][22]
   - Which design document or analysis did you write yourself?
   - What did you delegate, and why?
4. Tell me about a time your work influenced a team other than your own. [Likely]
   - How did they adopt it: mechanism, code, paper or persuasion?
   - What would have happened if they had said no?
5. What is the hardest technical problem you have solved in the last two years? [Likely]
   - Why was it hard for you specifically, and not just for the field?
   - Explain it so someone outside ML understands why it mattered.

### B. Conflict, backbone and disagree-and-commit

6. Tell me about a time you had a conflict with a coworker or manager, and how you handled it. [Reported] [19][21][22]
   - What did you do once the decision went against you?
   - Looking back, who was right?
   - How is your relationship with that person now?
7. Tell me about a time you had a major conflict with a Principal Engineer (or a more senior scientist). How did you resolve it? [Reported] [24]
   - What data did you bring to the conversation?
   - Did you escalate? Why or why not?
8. Tell me about a time you disagreed with a product manager's launch decision based on your model's results. [Likely]
   - What metric were they looking at that you thought was wrong?
   - What would have convinced you to commit to their plan?
9. Tell me about a time you pushed back on a stakeholder's metric or launch decision using data. [Likely]
   - What happened after launch, and were you proven right?
10. Tell me about a time you committed to a decision you disagreed with. [Likely]
    - How did you make sure your doubts did not slow the team down?
    - Did you ever revisit the decision? What triggered that?
11. Tell me about a time an engineer said your model could not ship in its current form. [Likely]
    - What did you give up to get it into production?
    - Who made the final call?
12. Tell me about a time you had to tell a senior leader that their idea would not work scientifically. [Likely]
    - How did you frame it?
    - What did they do next?

### C. Failure, being wrong and learning

13. Tell me about a project failure and how you managed the aftermath. [Reported] [24]
    - What would you do differently? [Reported] [20]
    - What mechanism did you put in place so it does not happen again?
14. Tell me about a time your model underperformed in production and you had to diagnose the issue. [Reported] [23]
    - What did you learn, and how did you pivot? [Reported] [25]
    - How long did it take you to notice, and why that long?
15. Tell me about a time you were wrong about a technical call. How did you find out? [Likely]
    - Who told you, or did you find it yourself?
    - What did it cost the team?
16. Tell me about a time you got critical feedback that surprised you. [Likely]
    - What did you change afterwards, and how would someone see that change?
17. Tell me about an experiment or research direction you killed. [Likely]
    - How did you decide it was dead rather than under-tuned?
    - How much effort had gone in, and how did you tell the team?
18. Tell me about a time you missed a deadline. [Likely]
    - When did you first know you would miss it, and when did you tell people?
19. How do you use AI tools in your work, and when did one give you a wrong answer? How did you catch it? [Reported] [28]
    - What do you now check by hand that you used to trust?

### D. Decisions under ambiguity (Bias for Action, Are Right a Lot)

20. Tell me about a time you made a critical scientific or technical decision without complete data. [Reported] [25]
    - What data would you have wanted, and why couldn't you wait for it?
    - How did you limit the downside if you were wrong?
21. Tell me about a time you had to decide quickly with incomplete information and the decision turned out to be wrong. [Likely]
    - Was it a one-way or two-way door?
22. Tell me about a time you took initiative without waiting for permission. [Likely]
    - Whose permission would you normally have needed?
    - What did your manager say afterwards?
23. Tell me about a time you had to choose between a quick heuristic and a principled model. [Likely]
    - What did you ship first, and did the principled model ever replace it?
24. Tell me about a time two offline metrics disagreed and you had to make a ship call. [Likely]
    - Which metric did you trust, and why?
25. Tell me about a time the online A/B result contradicted your offline evaluation. [Likely]
    - What was the root cause?
    - Did you change your offline evaluation afterwards?

### E. Ownership and Deliver Results (the core L5 signal)

26. Tell me about a time you took ownership of something outside your formal scope. [Likely]
    - Why didn't the actual owner handle it?
    - What did you stop doing to make room?
27. Tell me about a time you delivered a model under a hard deadline. [Likely]
    - What did you descope, and what did that cost?
    - What corners did you refuse to cut?
28. Tell me about the most significant business result your science has driven. [Likely]
    - How was that measured? [Reported] [21][22]
    - How do you know it was the model and not seasonality, a UI change or a marketing push?
    - What was the baseline?
29. Tell me about a time a project was about to fail and you turned it around. [Likely]
    - What was the first thing you did differently?
30. Tell me about a time you had to keep a model working long after launch (monitoring, retraining, on-call). [Likely]
    - Who owned it when it broke at 2am?
31. Tell me about a time you were blocked by another team. [Likely]
    - What did you do in the first week?
    - Did you find a way around them, or did you wait?

### F. Customer Obsession and Working Backwards, for a scientist

32. Tell me about a time you started from the customer and worked backwards to the science problem. [Likely]
    - Who was the customer, and how did you get their signal?
    - Was the customer internal or external, and did that change how you worked?
33. Tell me about a time you chose a simpler model because it was better for the customer. [Likely]
    - What accuracy did you give up, and how did you justify it?
34. Tell me about a time a model you built hurt some customers even though the average metric improved. [Likely]
    - How did you find that segment?
    - What did you do about it?
35. Tell me about a time you used customer anecdotes to overturn what the data said. [Likely]
    - How many anecdotes was enough?
36. If you wrote a press release for your best project, what would the headline and customer quote be? [Likely]
    - What would the FAQ's hardest question be?

### G. Dive Deep and scientific rigor (the technical side of the BR round)

37. Tell me about a time you dove into data or code to find a root cause that others missed. [Reported] [18][27]
    - What was the first anomaly you noticed?
    - How did you rule out the other explanations?
38. Tell me about a time you found a data-quality or label-leakage problem in your own work. [Likely]
    - How long had it been in production?
    - Which numbers did you have to retract?
39. Pick a model from your resume. Explain why you chose that architecture over the obvious alternative. [Likely]
    - What experiment told you the alternative was worse?
    - Would you make the same choice today?
40. Walk me through how you validated that your evaluation set reflected production traffic. [Likely]
    - What distribution shift did you see after launch?
41. Tell me about a time a result looked too good to be true. [Likely]
    - What did you check first?
42. Tell me about a time you had to defend a statistical result to a skeptical audience. [Likely]
    - What was the effect size, and was it practically significant?
43. You mentioned an improvement of X%. What was the confidence interval? [Likely]
    - How many samples, and how long did the test run?
44. Tell me about a time you reproduced a paper's results and could not match them. [Likely]
    - What did you conclude, and what did you ship instead?

### H. Technical checks from a non-science Bar Raiser (TPM, SDE or business BR)

45. How do you set ETAs, map dependencies and communicate next steps on a science project? [Reported] [10]
    - When did a science timeline slip, and how did you tell stakeholders?
    - How do you estimate a project when you do not know if the idea will work?
46. Interpret a p-value or a confidence interval for me. [Reported] [10]
    - What would you tell a business leader who reads p = 0.06 as "no effect"?
47. Explain your most recent model to me as if I were a business leader with no ML background. [Likely]
    - What decision does it let someone make that they could not make before?
48. How do you decide when a research prototype is ready to hand to engineering? [Likely]
    - What does the handoff document contain?
49. Your science project needs six months but the business wants results in six weeks. What do you do? [Likely]
    - What would a six-week version look like?
50. How do you know a model is still working a year after launch? [Likely]
    - Who gets paged, and what is the rollback plan?
51. Tell me about a time you had to explain a model's mistake to a customer-facing or operations team. [Likely]

### I. Invent and Simplify, Think Big, Frugality

52. Tell me about a time you simplified a complex scientific approach so it could ship. [Likely]
    - What did the complex version get right that the simple one missed?
53. Tell me about the most ambitious science bet you have proposed. [Likely]
    - Who funded it, and what did you have to prove first?
    - If it had been fully funded, what would the three-year version be?
54. Tell me about a time you did more with less: compute, labels or headcount. [Likely]
    - What was the cost of the cheaper approach in model quality?
55. Tell me about something you invented, such as a method, tool or mechanism, that others now use. [Likely]
    - How did you get people to adopt it?
56. Tell me about a time you used an LLM or foundation model where a custom model would have been the default. [Likely]
    - How did you weigh inference cost against development time?

### J. Earn Trust, Hire and Develop the Best, influence without authority

57. Tell me about a time you earned the trust of an engineering or product partner who doubted your science. [Likely]
    - What was the turning point?
58. Tell me about a time you mentored someone or raised the bar of your team's science. [Likely]
    - How do you know they improved?
59. Tell me about a time you gave hard feedback to a peer. [Likely]
    - How did they react, and what changed?
60. Tell me about a time you had to admit a mistake publicly. [Likely]
61. Tell me about a time you influenced a decision without having authority over the people involved. [Likely]
    - Who was the hardest person to convince?
62. Tell me about a time you reviewed someone else's experiment and found a serious flaw. [Likely]
    - How did you raise it?

### K. Insist on the Highest Standards, Broad Responsibility, Earth's Best Employer

63. Tell me about a time you raised the quality bar on a deliverable even though it delayed you. [Likely]
    - Who disagreed with the delay?
64. Tell me about a time you refused to launch something that was not ready. [Likely]
    - What was the minimum bar, and who set it?
65. Tell me about a time you considered the downstream harm or fairness impact of a model. [Likely]
    - How did you measure it?
    - What trade-off did you accept?
66. Tell me about a time you handled sensitive customer data in a modeling project. [Likely]
    - What would you not do even if it improved the model?
67. Tell me about a time you made your team's work environment better. [Likely]
68. Tell me about a time you learned a new area fast to unblock a project. [Likely]
    - How did you know you had learned enough?

### L. Deliberate pushback and stress probes (BR-specific)

69. I think your approach was wrong. Convince me otherwise. [Reported] [21][22]
    - What data would make you concede?
    - What would the alternative have cost?
70. That sounds like a team result. What specifically did you do? [Reported] [21][22]
71. How was that measured? [Reported] [21][22]
    - What was the counterfactual?
72. That example is from three years ago. Do you have a more recent one? [Likely]
73. You used that story already. Give me a different example for the same principle. [Likely]
74. Your manager made that call. Where was your judgment in that? [Likely]
75. If you had not been on the project, what would have been different? [Likely]
76. What was the worst decision you made on that project? [Likely]
77. What would your harshest critic on that team say about how you handled it? [Likely]
78. You said the result was a success. Who would disagree? [Likely]
79. Stop the story there. What exactly did you say in that meeting? [Likely]

### M. Leveling probes: L4 vs L5 vs L6

80. Tell me about a time you set the technical direction for a project, not just executed it. [Likely]
    - Who was in the room when the direction was decided?
81. Tell me about a time you identified a science problem nobody had asked you to solve. [Likely]
    - How did you get time and resources for it?
82. Tell me about a mechanism (process, tool, review) you created that outlived your involvement. [Likely]
    - Who runs it now?
83. Tell me about a time you had to coordinate science work across several people or teams. [Likely]
    - How did you divide the work, and what broke?
84. What is the biggest ambiguity you have resolved, from vague goal to concrete plan? [Likely]
    - What was the first artifact you produced?
85. How would your current manager describe the gap between you and the next level? [Likely]
    - What are you doing about it?

### N. Mixed technical-behavioral prompts (BR who is a scientist)

86. Pick one modeling decision from your resume and walk me through the alternatives, the evidence and who you had to convince. [Likely]
    - What experiment would have changed your mind?
87. Tell me about a time a technical trade-off (latency, cost, accuracy) forced a business compromise. [Likely]
    - Who signed off on the compromise?
88. Tell me about a time you moved a model from research to production at scale. [Likely]
    - What failed first in production that never failed offline?
89. Tell me about a time you had to choose between publishing and shipping. [Likely]
    - What did the business get out of the choice?
90. Tell me about a GenAI or LLM feature you built or evaluated. How did you decide it was safe to launch? [Likely]
    - How did you measure hallucination, and what threshold did you accept?
91. Tell me about a time an evaluation metric you chose turned out to be the wrong proxy. [Likely]
    - How did you detect the mismatch?

### O. Post-loop and split-debrief scenarios

92. (Extra round after a split debrief) Let's revisit coding with a follow-up problem. [Reported] [17]
93. (Downlevel conversation) We think your scope maps to L4. Would you consider that level? [Likely] (BR-driven downleveling reported in [11][15])
    - What evidence from your past work would change our assessment?
94. Why Amazon, and why this team, over other offers you have? [Likely]
    - Which LP do you find hardest to live by?
95. What questions do you have for me about how Amazon hires and develops scientists? [Likely]

---

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. "Amazon Bar Raiser", About Amazon (Amazon, official). https://aboutamazon.com/news/workplace/amazon-bar-raiser
3. "Hire power: How Amazonians raise the bar with every interview", About Amazon (Amazon, official). https://www.aboutamazon.com/news/workplace/hire-power-how-amazonians-raise-the-bar-with-every-interview
4. "Amazon bar raisers, entry-level interview hiring procedure", Fortune, 2024. https://fortune.com/2024/08/07/amazon-bar-raisers-entry-level-interview-hiring-procedure
5. "Memoirs of an Amazon Bar Raiser", Carlos Arguelles, Medium / Geek Culture. https://medium.com/geekculture/memoirs-of-an-amazon-bar-raiser-718e36241310
6. "Amazon Bar Raiser: Who They Are & How to Pass the Loop", Carrus. https://www.carrus.io/blog/all-about-bar-raisers-amazons-essential-element-to-the-hiring-process
7. "Working Backwards, by Colin Bryar and Bill Carr", Evan's Notes, Substack (book summary covering the debrief and BR process). https://evansamek.substack.com/p/working-backwards-by-colin-bryar
8. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor (reports incl. 2025-2026). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
9. Amazon Applied Scientist Interview Experience & Questions, Glassdoor (reports incl. 2025-2026). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
10. "My final interview for Amazon Applied Scientist-II will be taken by Principal TPM", Blind. https://www.teamblind.com/post/my-final-interview-for-amazon-applied-scientist-ii-will-be-taken-by-principal-tpm-cant-understand-what-tpm-will-ask-ryaa3tjo
11. "Amazon applied scientist interview feedback" (leveling dispute with bar raiser), Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-feedback-tvc4yec6
12. "Amazon bar raiser veto", Blind. https://www.teamblind.com/post/amazon-bar-raiser-veto-ybkwpkzv
13. "Amazon Applied Scientist (L5) Virtual Onsite", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
14. "What are the expectations of Amazon L5 vs L6 for Applied Scientists?", Blind. https://www.teamblind.com/post/amazon-applied-scientists-l5-vs-l6-expectations-kegmp4rx
15. "Amazon applied scientist L6 to L5 downlevel", Blind. https://www.teamblind.com/post/amazon-applied-scientist-l6-to-l5-downlevel-qvqtn0a8
16. "Amazon Applied Scientist L5 Offer Eval", Blind, 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-offer-eval-c03bg14s
17. "Amazon (AGI team) Applied Scientist II interview", Blind (extra round after split debrief). https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT
18. "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP", OA VO Service (vendor-published candidate debrief, promotional; L5 vs L4 follow-up depth; LP time per round). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
19. "Amazon Bar Raiser Interview (questions, prep tips)", IGotAnOffer. https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview
20. "Amazon Applied Scientist Interview (process, questions, prep)", IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
21. "Mastering the Amazon Bar Raiser Interview", 4 Day Week. https://4dayweek.io/interview-process/amazon-bar-raiser
22. "The Amazon Bar Raiser Interview: Who They Are and What They Ask", BarRaiser AI. https://barraiserai.com/guides/amazon-bar-raiser-interview
23. "The 2026 Amazon Applied Scientist Interview Guide", Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
24. "The 2026 Amazon Senior Applied Scientist Interview Guide", Interview Kickstart, 2026. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
25. "Amazon Applied Scientist Interview Questions & Guide 2026", Dataford, 2026. https://dataford.io/interview-guides/amazon/applied-scientist
26. "Amazon Leadership Principles Interview (questions + tips)", interviewing.io. https://interviewing.io/guides/amazon-leadership-principles
27. "Senior Engineer's Guide to Amazon Interviews + Questions", interviewing.io. https://interviewing.io/guides/hiring-process/amazon
28. "Amazon Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator; GenAI-usage behavioral question), 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process.md
