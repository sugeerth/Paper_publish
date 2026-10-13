# 01. Loop Format and the Senior Bar: Airbnb Senior MLE (Experimentation / A/B Testing)

> Research note (read first). Live web research was not possible when this chapter was written. The shared web-search budget was used up, and the egress proxy refused connections to igotanoffer.com, interviewing.io, careers.airbnb.com and airbnb.tech. So this chapter draws on what was already known about Airbnb's widely reported interview process and on the sources listed under References. None of those sources were re-fetched for this chapter. Before relying on a URL or a "[Reported]" tag, open the source and confirm it. A recruiter's prep packet always overrides this chapter. Tags: [Reported] means a public source or the LeetCode Airbnb company tag attributes the item to Airbnb. [Likely] means an inference from the role, the level and Airbnb's products. It does not mean anyone reported it.

## Format notes

| Stage | Typical length | Interviewer | What is assessed (Senior / L5 bar) |
|---|---|---|---|
| Recruiter call | 20-30 min | Recruiter | Motivation ("Why Airbnb?"), level calibration (L5 = Senior), team fit (Search/Ranking, Pricing, Trust, Payments, Relevance, Experimentation platform), comp, timeline [1][2] |
| Technical phone screen | 45-60 min | Engineer | One or two practical coding problems on a shared editor (CoderPad-style). The code is expected to **compile and run against test cases**. Pseudocode does not pass [1][2][3] |
| Onsite: Coding x2 | 45-60 min each | Engineers | Runnable, clean, tested code. One round is often algorithmic (LeetCode-medium/hard, many Airbnb-tagged problems). The other is a longer "practical" problem with evolving requirements (parsing, pagination, scheduling, simulation) [1][3][4] |
| Onsite: ML design | 45-60 min | Senior/Staff MLE | End-to-end ML system for an Airbnb product (search ranking, pricing, fraud, recommendations), including offline/online evaluation and **experiment design** [Likely; 6-10] |
| Onsite: System design | 45-60 min | Senior/Staff engineer | Scalable services and data pipelines (for example a booking system, an experimentation/assignment service, a feature store) [1][2] |
| Onsite: Past project deep dive | 45-60 min | Hiring manager or senior IC | Ownership, technical depth, trade-offs and measured impact of a project you led. Sometimes presented with slides [1][2] |
| Onsite: Core values x2 | 30-45 min each | Employees **outside** the hiring team (often non-engineers) | Airbnb's four core values: *Champion the Mission, Be a Host, Embrace the Adventure, Be a "Cereal" Entrepreneur*. A strong fail here can veto an otherwise strong loop [1][2][5] |
| Debrief / hiring committee | n/a | Panel + leadership review | Calibrated on level. Senior candidates are expected to drive ambiguity, lead cross-functional work and show measured impact [1][2] |

Key expectations that candidates consistently report:
- **Runnable code is mandatory.** Interviewers run your code. Syntax errors, unhandled edge cases or no tests count against you even if the logic is right [1][3].
- **Practical over puzzle.** Problems are often framed as real product scenarios: listings pagination, host search results, price rounding, travel itineraries, calendar availability [3][4].
- **Core values are scored as seriously as the technical rounds** and are run by people with no stake in the team [1][5].
- **Senior bar (L5):** you independently own an ambiguous problem end to end. You make clear trade-offs (latency vs. quality, bias vs. variance in experiment estimates). You influence partners such as Product, Data Science and Design, and you mentor others. Coding and design are expected to be done without heavy hints [Likely; 1][2].
- **Experimentation-flavored MLE roles:** Airbnb runs a mature in-house experimentation platform (ERF) and has written publicly about A/B testing pitfalls, interleaving for search ranking and two-sided-marketplace effects. Expect experimentation probes inside the ML design and past-project rounds [6][7][8][9].

## Questions

### A. Recruiter call and process logistics

1. Why Airbnb, and why now? [Reported] [1][2]
   - Which Airbnb product surface (Search, Pricing, Trust, Experiences, Payments) would you most want to work on, and why?
   - Have you used Airbnb as a guest or as a host? What would you change?
2. Walk me through your background in under three minutes, focusing on ML in production. [Reported] [1]
3. What level are you targeting, and what scope of work did you own at your current level? [Likely]
   - How many engineers or scientists did you lead or influence on your largest project?
4. Which programming language will you use in coding rounds, and are you fluent enough to run code without IDE help? [Likely] [3]
5. How much of your recent work was modeling, and how much was infrastructure or experimentation? [Likely]
6. Have you designed or analyzed online A/B tests yourself, or did a data scientist own that? [Likely]
7. What are your compensation expectations and timeline, and do you have competing processes? [Reported] [2]
8. Are you open to team matching across Search/Relevance, Pricing, Trust and the Experimentation platform? [Likely]

### B. Technical phone screen (practical coding, must run)

9. Given a list of listings (host_id, listing_id, score, city) and a page size, return paginated results so that no host appears twice on the same page if possible, preserving score order. [Reported] [3][4]
   - What happens on the last page when only one host's listings remain?
   - What is the time complexity, and can you do it in a single pass?
10. Minimize rounding error: round a list of prices to integers so the sum equals a target, minimizing the total absolute rounding error. [Reported] [4]
    - What if the target is unreachable? How do you signal that?
    - How would you handle currency with two decimal places?
11. Pour water: simulate water droplets falling onto an elevation map at index K, with the leftward-first flow rule. [Reported] [4]
    - How would you print the final terrain and water visually?
    - How do you test the boundary indices?
12. Text justification: fully justify lines of words to a fixed width. [Reported] [4]
    - How do you handle a single word on a line, and the last line?
13. Flatten a 2D vector into an iterator with `next`, `hasNext` and (as a follow-up) `remove`. [Reported] [4]
14. Implement a CSV parser that handles quoted fields, escaped quotes and commas inside quotes. [Reported] [3]
    - How do you handle multi-line quoted fields?
15. Given an IP address and a count, return the minimal list of CIDR blocks covering exactly that range. [Reported] [4]
16. Write tests for your solution right now. Which cases did you choose, and why? [Likely] [3]

### C. Onsite coding round 1 (algorithmic, Airbnb-tagged style)

17. Alien dictionary: derive a character ordering from a sorted list of words. [Reported] [4]
    - How do you detect an invalid ordering, such as a prefix appearing after a longer word?
18. Palindrome pairs: find all index pairs whose concatenation forms a palindrome. [Reported] [4]
    - Can you do better than O(n^2 * k)?
19. Cheapest flights within K stops (framed as a travel-itinerary cost minimizer). [Reported] [4]
    - What changes if edges carry both price and duration and you must optimize a weighted objective?
20. Sliding puzzle: minimum moves to solve a 2x3 board. [Reported] [4]
21. Pyramid transition: can you build a pyramid from a bottom row given allowed triples? [Reported] [4]
22. Employee free time: given each person's busy intervals, return the common free intervals (calendar availability). [Reported] [4]
    - How would you adapt this to host calendars with blocked dates and minimum-night rules?
23. Smallest common region: find the lowest common region in a region hierarchy. [Reported] [4]
24. Combination sum (framed as "choose menu items that exactly use a budget"). [Reported] [4]
    - How do you handle float prices without precision bugs?
25. Regular expression matching with `.` and `*`. [Reported] [4]
26. Word search II on a Boggle-style board. [Reported] [4]
27. Basic calculator with `+ - * /` and parentheses. [Reported] [4]
28. Contains duplicate within distance k and within value difference t. [Reported] [4]
29. Flatten a nested list iterator / mini parser for nested lists. [Reported] [4]
30. House robber (framed as non-adjacent booking selection maximizing revenue). [Reported] [4]
    - What if bookings are intervals with arbitrary overlaps (weighted interval scheduling)?

### D. Onsite coding round 2 (practical, evolving requirements)

31. Design and implement a booking calendar that supports `book(start, end)`, rejects overlaps and answers `is_available(start, end)`. [Likely] [3]
    - Now add per-night pricing and return the total price for a stay.
    - Now support split stays across two listings when no single listing is available.
32. Implement a simple key-value store with transactions (`begin`, `commit`, `rollback`), nested. [Likely] [3]
33. Parse a log of experiment exposures (user_id, experiment, variant, timestamp) and output per-variant unique user counts, de-duplicating users exposed to multiple variants. [Likely]
    - How would you flag and count users with conflicting assignments?
34. Implement deterministic hash-based bucketing: given user_id, experiment salt and traffic split, return a variant. [Likely] [6]
    - How do you guarantee independence between experiments run on the same users?
    - How would you ramp traffic from 1% to 50% without reshuffling already-exposed users?
35. Given streaming events, compute a rolling 7-day conversion rate per listing with bounded memory. [Likely]
36. Implement a file system with `create(path, value)` and `get(path)`, including parent-existence checks. [Reported] [4]
37. Given guest reviews with ratings and timestamps, return the top-K listings by a time-decayed average rating. [Likely]
    - How do you handle listings with very few reviews (shrinkage)?
38. Implement an LRU or LFU cache for search results, then extend it with a TTL. [Likely]
39. Refactor your solution so that a new requirement (for example multi-currency support) fits in with minimal change. How do you structure the code? [Likely] [3]

### E. ML design round (Airbnb product + experimentation)

40. Design the search ranking system for Airbnb stays. [Likely] [7][8][9]
    - What are your labels (bookings, clicks, long clicks, host acceptances), and how do you weight them?
    - How do you handle position bias in training data?
    - How do you evaluate offline (NDCG on booked listings) versus online?
41. Design real-time personalization using listing embeddings from in-session clicks. [Likely] [8]
    - How would you inject booking or market signals into the embedding objective?
    - How do you handle cold-start listings with no sessions?
42. Design a dynamic pricing suggestion model for hosts (Smart Pricing). [Likely]
    - How do you estimate demand elasticity when price is confounded with seasonality and quality?
    - How do you A/B test a pricing model when hosts can override it?
43. Design a fraud or risk model for bookings (Trust and Safety). [Likely]
    - How do you evaluate a model whose positive labels arrive weeks late and are censored by your own interventions?
44. Design "similar listings" or "recommended experiences" recommendations. [Likely]
45. Design a model that predicts host acceptance probability and use it in ranking. [Likely] [7]
46. Design an ML system to detect low-quality or misleading listing photos and text. [Likely]
47. How would you design the online experiment that launches your new ranking model? [Likely] [6][9]
    - What is the primary metric, and what are your guardrail metrics?
    - What unit of randomization would you use (user, session, listing, market), and why?
    - How do you estimate the required sample size and duration?
48. In a two-sided marketplace, why can a user-randomized test of a ranking change be biased? [Likely] [6]
    - How would listing-side or market-level (cluster) randomization change your estimates?
    - What is interference, or a SUTVA violation, in this setting?
49. When would you use interleaving instead of A/B testing for search ranking? [Likely] [9]
    - How does interleaving sensitivity compare with A/B, and what can interleaving not measure?
50. Offline NDCG improved but the online A/B showed no booking lift. How do you debug this? [Likely] [7]
51. Bookings happen weeks after search. How do you choose an experiment metric that is sensitive enough but still aligned with long-term value? [Likely] [6]
    - How would you build and validate a proxy or surrogate metric?
52. How would you apply variance reduction (CUPED or regression adjustment) to booking-conversion experiments? [Likely] [6]
53. How do you handle novelty effects and long-tail seasonality (holidays, events) when reading test results? [Likely]
54. How do you detect and correct sample ratio mismatch? [Likely] [6]
55. How do you control false discoveries when a team runs dozens of concurrent experiments and checks results daily? [Likely] [6]
    - What is the peeking problem, and how do sequential tests address it?
56. How would you measure the long-term effect of a model change after the A/B test has ended (holdouts)? [Likely]
57. Your model improves guest conversion but reduces host earnings in some markets. How do you decide whether to ship? [Likely]

### F. System design round

58. Design an experimentation assignment and logging service used by every Airbnb client (web, iOS, Android, backend). [Likely] [6]
    - How do you keep assignment consistent across logged-out and logged-in states?
    - How do you guarantee exposure logging is not lost or double-counted?
59. Design the metrics computation pipeline for an experimentation platform (daily scorecards for thousands of metrics). [Likely] [6]
    - How do you backfill when a metric definition changes?
60. Design a booking and reservation system that prevents double-booking under high concurrency. [Likely] [1]
61. Design a feature store that serves both offline training and low-latency online ranking. [Likely]
    - How do you prevent training-serving skew?
62. Design a search service that retrieves and ranks listings within 200 ms p99 across millions of listings. [Likely] [7]
    - Where do you place candidate retrieval, filtering (availability, price, map bounds) and ML re-ranking?
63. Design a model serving and rollout system with shadow mode, canary and automatic rollback. [Likely]
64. Design a notification system for hosts and guests that respects time zones and rate limits. [Likely]
65. Design a calendar availability service for millions of listings with frequent updates and iCal sync. [Likely]

### G. Past project deep dive

66. Walk me through the ML project you are proudest of, from problem framing to measured impact. [Reported] [1][2]
    - What was your specific contribution compared with the team's?
    - What alternatives did you reject, and why?
67. How did you measure the impact of that project? Was there an A/B test, and what did it show? [Likely]
    - What were the confidence intervals, and were there any surprising segment-level results?
68. What was the hardest technical trade-off you made on that project? [Likely]
69. If you had to redo the project with half the time, what would you cut? [Likely]
70. Tell me about a model that worked offline but failed in production. What did you learn? [Likely]
71. How did you get alignment from Product and Data Science on the success metric? [Likely]
72. What would the next iteration of that system look like? [Likely]
73. How did you handle data quality or labeling problems in that project? [Likely]
74. Which part of the system would break first at 10x scale? [Likely]

### H. Core values interviews (two rounds, cross-functional)

75. Which of Airbnb's core values resonates with you most, and why? [Reported] [1][5]
76. Tell me about a time you went above and beyond for a customer, user or teammate ("Be a Host"). [Reported] [1][5]
77. Tell me about a time you took a big risk or tried something no one had done before ("Embrace the Adventure"). [Reported] [1][5]
78. Tell me about a time you had to be scrappy and resourceful with limited resources ("Be a Cereal Entrepreneur"). [Reported] [1][5]
79. What does "belonging" mean to you, and when have you created it for someone else ("Champion the Mission")? [Reported] [1][5]
80. Describe a memorable travel or hosting experience and what it taught you. [Reported] [1]
81. Tell me about a time you failed. What did you do next? [Reported] [1][2]
82. Tell me about a time you disagreed with a teammate or manager. How did you resolve it? [Reported] [2]
83. Tell me about a time you received difficult feedback. [Reported] [2]
84. Describe a time you helped someone grow or mentored a junior colleague. [Likely]
85. Tell me about a time you made a decision that was right for users but unpopular internally. [Likely]
86. How do you build inclusive teams or include people with different perspectives? [Likely] [5]
87. Tell me about a time you had to adapt to a big change you did not choose (a reorg, layoffs, a pivot). [Likely]
88. What would your previous teammates say is the one thing you should improve? [Likely]
89. Why do you want to work on a two-sided marketplace product? [Likely]

### I. Senior-level calibration probes (appear across rounds)

90. How do you decide when a simpler heuristic should ship instead of an ML model? [Likely]
91. Tell me about a time you set the technical direction for a team without formal authority. [Likely]
92. How do you prioritize between model quality, latency and engineering cost? [Likely]
93. Describe a time you stopped or reversed a launch because the experiment results were ambiguous. [Likely]
94. How do you communicate a negative or inconclusive A/B result to leadership? [Likely]
95. How do you make sure your experiment results are reproducible and trusted by other teams? [Likely]
96. What metrics would you put on a dashboard to know your production model is healthy? [Likely]
97. How do you balance short-term booking metrics against long-term guest and host trust? [Likely]
98. What questions do you have for us about the team, the experimentation culture or the roadmap? [Reported] [1]

## References

The sources below were **not re-fetched** for this chapter because live web access was blocked. Each is listed because it was recalled from earlier knowledge as relevant. Confirm that each URL loads before you rely on it.

1. "Airbnb Interview Process & Questions (+ how to prepare)", IGotAnOffer, 2023-2024. https://www.igotanoffer.com/blogs/tech/airbnb-interview-process
2. "Airbnb Machine Learning Engineer Interview Questions", Glassdoor (candidate reports), ongoing. https://www.glassdoor.com/Interview/Airbnb-Machine-Learning-Engineer-Interview-Questions-EI_IE391850.0,6_KO7,32.htm
3. "Airbnb Interview Discussions", LeetCode Discuss (search tag "Airbnb"), ongoing. https://leetcode.com/discuss/interview-question?currentPage=1&orderBy=hot&query=airbnb
4. "Airbnb company-tagged problems", LeetCode (Premium company tag list), ongoing. https://leetcode.com/company/airbnb/
5. "Airbnb Careers: Life at Airbnb / core values", Airbnb, ongoing. https://careers.airbnb.com/
6. "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014. https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
7. Haldar et al., "Applying Deep Learning to Airbnb Search", arXiv / KDD 2019. https://arxiv.org/abs/1810.09591
8. Grbovic & Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219885
9. Haldar et al., "Improving Deep Learning for Airbnb Search", arXiv / KDD 2020. https://arxiv.org/abs/2002.05515
10. "The Airbnb Tech Blog", Airbnb Engineering (Medium), ongoing. https://medium.com/airbnb-engineering
