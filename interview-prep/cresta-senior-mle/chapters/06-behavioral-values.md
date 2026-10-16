# 06. Behavioral and Values: Ownership, Customer Obsession, Speed, Ambiguity, Conflict and Mentoring for Cresta Senior MLE

> **How to read this chapter.** This chapter covers the hiring-manager chat and any behavioral or culture rounds in the Cresta Senior MLE loop. It is organized around Cresta's publicly cited operating principles and the behaviors a senior MLE at a Series-D-scale AI startup is expected to show. It lists questions only, with no model answers.
>
> **Tagging system** (same as chapters 01-05):
> - **[Reported] [n]**: a candidate report attributes this question, or this round type, to Cresta. The tag names the role when the report is for another role.
> - **[Likely]**: an inference from Cresta's stated principles, job postings and products. No candidate has reported it. A citation on a [Likely] item points to the principle or posting it is based on, not to a candidate report.

## Format notes

- **Behavioral rounds exist, but their questions are not public.** A mid-level SWE who received an offer described a loop of one technical screen, three coding interviews and **three behavioral rounds** [1]. A Senior/Staff SWE onsite was described as an hour of coding, an hour of system design and a **hiring-manager chat** [1]. The reported Senior MLE loop (recruiter call, LeetCode-medium round, "vibe coding" round, system design) does not list a separate behavioral round, so expect values questions inside the recruiter and hiring-manager conversations [2].
- **Leadership round for senior customer-facing roles.** A Senior FDE loop included a separate leadership round and a customer-delivery round [3]. Senior MLEs work closely with forward-deployed teams, so customer-delivery stories are worth preparing.
- **Culture fit comes late.** A third-party culture write-up says Cresta runs a four-round loop for most roles, hires for craft depth first, and assesses culture fit at the end [6]. This is not a candidate report.
- **Operating principles.** Third-party summaries cite "Mission is the boss", "Do too much", "Have the courage to be disliked" and "Patient in strategy, impatient in execution". They also describe high ownership and high talent density [4]. These do not come from Cresta's own careers page in any source seen for this bank, so check them against cresta.com/careers before quoting them back to an interviewer.
- **What the senior bar looks like (inferred from postings).** You own an ML system end to end. You turn vague customer asks into measurable requirements. You ship fast without breaking enterprise trust. You raise the bar of the engineers around you [5].
- **Research limits.** Every Cresta-specific search in this pass returned only format-level evidence (round counts and round names). No source quoted a specific behavioral question asked at Cresta. That is why every question below is [Likely] except where a round format is reported.

## A. Motivation and fit

1. Walk me through your background, focusing on the ML systems you shipped and what you personally owned. [Likely] (asked in the hiring-manager chat [1])
   - Which one of those systems are you proudest of, and why that one?
2. Why Cresta, and why contact-center AI rather than a general-purpose LLM lab or a big-tech ML team? [Likely]
   - What do you think is hardest about building AI for contact centers?
3. Why are you leaving your current role now? [Likely]
4. Which team appeals to you most: Agentic Assist, Agent and System Quality, Voice/ASR, or Insights? What would you want to own in your first six months? [Likely] [5]
5. Cresta's mission is often summarized as making every customer conversation better through AI. Tell me about a time you chose work because of the mission, not the technology. [Likely] [4]
6. What kind of environment do you do your worst work in? How would you know early that it was happening? [Likely]

## B. "Mission is the boss": ownership beyond your job description

7. Tell me about a time you owned a problem end to end that nobody had assigned to you. [Likely] [4]
   - How did you decide it was worth your time over your assigned work?
   - Who did you have to convince, and how?
8. Tell me about a time you put the company's goal ahead of your team's goal or your own preference. [Likely] [4]
9. Describe a time you were on call for, or personally responsible for, an ML system that failed in production. What did you do in the first hour? [Likely]
   - What did you change afterwards so it would not happen again?
10. Tell me about a time you picked up unglamorous work (data cleaning, labeling tooling, eval infrastructure, on-call) because it was the real bottleneck. [Likely]
11. Tell me about a time you noticed a problem outside your area (for example, in data, infra or product) and fixed it, or got it fixed. [Likely]
12. Tell me about something you shipped that failed or was rolled back. What was your part in the failure? [Likely]
    - What would you do differently with what you know now?

## C. "Do too much" and "impatient in execution": speed and bias for action

13. Tell me about a time you went far beyond the original scope to get something shipped. [Likely] [4]
    - What did doing "too much" cost you or the team, and was it worth it?
14. Describe the fastest you have taken an ML idea from prototype to production. What did you skip, and what did you refuse to skip? [Likely]
15. Tell me about a time you shipped something you knew was imperfect. How did you decide it was good enough? [Likely]
    - What guardrails, monitoring or rollback plan did you put in place?
16. Tell me about a time you were too slow: you over-researched or over-engineered. How did you notice? [Likely]
17. How do you balance speed against quality when the model output goes in front of an enterprise customer's agents or end customers? Give an example. [Likely]
18. Tell me about a time you unblocked yourself when a dependency (another team, a vendor, a data source) was holding you up. [Likely]
19. Describe a week where you had more urgent work than you could finish. How did you choose what to drop? [Likely]

## D. "Patient in strategy": long-term thinking

20. Tell me about a time you invested in something with no short-term payoff (an eval suite, a data flywheel, a platform) and it paid off later. [Likely] [4]
    - How did you justify it to a manager who wanted features?
21. Tell me about a time you said no to a quick fix because it would hurt the long-term architecture. [Likely]
22. Describe a technical bet you made that turned out wrong. How long did you take to change course, and what told you to? [Likely]
23. How do you decide which new LLM research or model release is worth adopting, and which is noise? Give a recent example. [Likely]

## E. "Have the courage to be disliked": disagreement and conflict

24. Tell me about a time you pushed back on a PM, a leader or a customer and turned out to be right. [Likely] [4]
    - What data did you bring?
25. Tell me about a time you pushed back and turned out to be wrong. How did you handle it? [Likely]
26. Describe a disagreement with a peer engineer over a technical design. How was it resolved? [Likely]
    - What did you do once the decision went against you?
27. Tell me about a time you gave a senior person difficult feedback. [Likely]
28. Tell me about a time you had to tell a customer, or a customer-facing colleague, that the model could not do what they wanted. [Likely]
29. Describe a conflict between research quality and shipping speed on your team. Which side were you on, and what happened? [Likely]
30. Tell me about a time you raised an ethical, privacy or safety concern about an ML system. [Likely]
    - Contact-center transcripts contain PII and payment data. How would you raise a concern about how they are used for training?

## F. Customer obsession

31. Tell me about a time you worked directly with a customer, or a forward-deployed team, to define what an ML feature should do. [Likely] [3]
    - How did you turn a vague ask into measurable requirements?
32. Tell me about a time a customer's real usage differed sharply from what you designed for. What did you change? [Likely]
33. Describe a time you used customer feedback or production data to reverse a decision your team had already made. [Likely]
34. Tell me about a time you had to explain a model regression or a bad output to a non-technical stakeholder or customer. [Likely]
35. Tell me about a time you had to choose between a custom fix for one large customer and a general solution for all customers. [Likely]
    - How would you make that call at Cresta, where each enterprise has its own workflows and knowledge base?
36. How do you measure whether an ML feature actually helped the end user, beyond offline metrics? Give an example from your work. [Likely]

## G. Ambiguity

37. Tell me about the most ambiguous project you have led. How did you decide what to do first? [Likely]
38. Describe a time the success metric was undefined or contested. How did you settle on one? [Likely]
39. Tell me about a time priorities changed mid-project. What did you salvage? [Likely]
40. Tell me about a time you had to make an important decision with incomplete data. [Likely]
    - What would have made you change your mind?
41. You join a team where evaluation for a key model is weak and nobody owns it. What do you do in your first 90 days? [Likely]

## H. Mentoring, leadership and raising the bar

42. Tell me about an engineer you mentored. What changed in how they worked? [Likely]
    - How did you adapt your approach to them?
43. Tell me about a time you raised the technical bar of your team: code review norms, eval practice, design docs or on-call. [Likely]
44. Tell me about a time you changed another team's roadmap or technical direction without authority over them. [Likely]
45. Describe a time you had to give critical feedback on a junior engineer's work under deadline pressure. [Likely]
46. How do you run a design review for an ML system when people on the team disagree? [Likely]
47. Tell me about a time you helped hire someone, or helped decide not to. What signal did you look for? [Likely]

## I. Learning, self-awareness and working in an AI-native team

48. What is a piece of feedback you received that changed how you work? [Likely]
49. Tell me about a skill you had to learn fast for a project. How did you learn it? [Likely]
50. How do you use AI coding tools in your daily work, and where do you not trust them? [Likely] (the loop includes an AI-assisted "vibe coding" round [2])
51. What is your biggest weakness as an ML engineer right now, and what are you doing about it? [Likely]

## Deepening pass (October 2026): newly reported questions

Five web searches were run for Cresta-specific behavioral and values reports: Blind, Reddit, 1point3acres, Glassdoor, Prepfully, igotanoffer and InterviewQuery, plus Cresta's operating principles. None of the result snippets quoted a specific behavioral question asked at Cresta. The searches confirmed only the round formats already cited in the Format notes: three behavioral rounds and a hiring-manager chat [1], and a third-party note that culture fit is assessed late in a four-round loop [6]. **No new [Reported] questions were added**, and no existing question was retagged.

The [Likely] questions below close gaps for the senior level that sections A-I do not cover.

52. Tell me about a time you owned the business outcome of an ML system (for example, revenue, retention or cost), not just its model metrics. [Likely] [5]
    - How did you connect a model metric to the number the business cared about?
53. Tell me about a time you killed a project or a model you had built. How did you decide, and how did you tell the team? [Likely]
54. Describe a time an enterprise customer escalated an AI failure (a hallucination, a wrong compliance flag or a bad summary). What did you do on the day, and what did you change afterwards? [Likely]
55. Tell me about a time you worked across the research-to-production boundary: you took a research result and made it production-grade. [Likely] [5]
    - What broke when it met real data?
56. Startups change direction often. Tell me about a time your company pivoted and your work was deprioritized. How did you respond? [Likely]
57. Tell me about a time you had to decide between building in-house and using a third-party model provider, and how you persuaded others. [Likely]
58. Describe a time you reduced cost or latency significantly without hurting quality. Who had to agree, and how did you get them to? [Likely]
59. Tell me about a time you had to work across time zones or with a remote or distributed team to ship something urgent. [Likely]
60. Which of Cresta's operating principles do you find hardest to live by, and why? Give an example from your past. [Likely] [4]
61. What would your last manager say is the one thing you should improve before becoming a Staff engineer? [Likely]

## References

1. Blind, "Cresta interview process" (mid-level SWE loop with three behavioral rounds; Senior/Staff onsite with coding, system design and a hiring-manager chat), 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
2. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer loop with vibe-coding and system design rounds), 2024-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
3. Glassdoor, "Cresta (CA) Interview Questions" (Senior FDE loop with customer-delivery and leadership rounds), 2025-2026. https://www.glassdoor.ca/Interview/Cresta-CA-Interview-Questions-E3180909.htm
4. JobsByCulture, "Cresta: Frontend Engineer" (third-party summary of operating principles: "Mission is the boss", "Do too much", "Have the courage to be disliked", "patient in strategy, impatient in execution"; high ownership and talent density). https://jobsbyculture.com/jobs-at/cresta/frontend-engineer
5. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
6. ZeroG Talent, "Company culture at Cresta" (third-party; four-round loop for most roles, craft depth first, culture fit last). Added in the October 2026 deepening pass. https://zerogtalent.com/blog/company-culture-at-cresta
