# 01. Loop Format and the Senior Bar: Airbnb Senior MLE (Experimentation / A/B Testing)

> Research note (read first). This chapter went through an adversarial review on 2026-10-14. During that review the egress proxy blocked igotanoffer.com, glassdoor.com, leetcode.com, medium.com, arxiv.org, dl.acm.org and careers.airbnb.com, and the shared web-search budget was already used up. Only one source could be fetched and checked line by line: the community-collected Airbnb question list on GitHub [11]. Every question tagged **[Reported]** below is supported by that verified source. Questions tagged **[Reported, unverified]** are attributed to Airbnb by sources [1]-[5], which could not be re-fetched in this review. Treat them as plausible but unconfirmed. **[Likely]** means an inference from the role, the level and Airbnb's products, not a candidate report. Source [11] was compiled around 2017-2020 and says Airbnb may have rotated its question pool since, so the 2024-2026 loop could not be confirmed here. A recruiter's prep packet always overrides this chapter.

## Format notes

| Stage | Typical length | Interviewer | What is assessed (Senior / L5 bar) |
|---|---|---|---|
| Recruiter call | 20-30 min | Recruiter | Motivation ("Why Airbnb?"), level calibration (L5 = Senior), team fit (Search/Ranking, Pricing, Trust, Payments, Relevance, Experimentation platform), comp, timeline [1][2] |
| Technical phone screen | 45-60 min | Engineer | One or two practical coding problems on a shared editor (CoderPad-style). The code is expected to **compile and run against test cases**. Pseudocode does not pass [1][2][3] |
| Onsite: Coding x2 | 45-60 min each | Engineers | Runnable, clean, tested code. One round is often algorithmic (LeetCode-medium/hard, many Airbnb-tagged problems). The other is a longer "practical" problem with evolving requirements (parsing, pagination, scheduling, simulation) [1][3][4][11] |
| Onsite: ML design | 45-60 min | Senior/Staff MLE | End-to-end ML system for an Airbnb product (search ranking, pricing, fraud, recommendations), including offline/online evaluation and **experiment design** [Likely; 6-10] |
| Onsite: System design | 45-60 min | Senior/Staff engineer | Scalable services and data pipelines (for example a booking system, an experimentation/assignment service, a feature store) [1][2] |
| Onsite: Past project deep dive | 45-60 min | Hiring manager or senior IC | Ownership, technical depth, trade-offs and measured impact of a project you led. Sometimes presented with slides [1][2] |
| Onsite: Core values x2 | 30-45 min each | Employees **outside** the hiring team (often non-engineers) | Airbnb's four core values: *Champion the Mission, Be a Host, Embrace the Adventure, Be a "Cereal" Entrepreneur*. A strong fail here can veto an otherwise strong loop [1][2][5] |
| Debrief / hiring committee | n/a | Panel + leadership review | Calibrated on level. Senior candidates are expected to drive ambiguity, lead cross-functional work and show measured impact [1][2] |

Key expectations that candidates consistently report:
- **Runnable code is mandatory.** Interviewers run your code. Syntax errors, unhandled edge cases or no tests count against you even if the logic is right [1][3].
- **Practical over puzzle.** Problems are often framed as real product scenarios: listings pagination, host search results, price rounding, travel itineraries, calendar availability [3][4][11].
- **Core values are scored as seriously as the technical rounds** and are run by people with no stake in the team [1][5].
- **Senior bar (L5):** you independently own an ambiguous problem end to end. You make clear trade-offs (latency vs. quality, bias vs. variance in experiment estimates). You influence partners such as Product, Data Science and Design, and you mentor others. Coding and design are expected to be done without heavy hints [Likely; 1][2].
- **Experimentation-flavored MLE roles:** Airbnb runs a mature in-house experimentation platform (ERF) and has written publicly about A/B testing pitfalls, interleaving for search ranking and two-sided-marketplace effects. Expect experimentation probes inside the ML design and past-project rounds [6][7][8][9].

## Questions

### A. Recruiter call and process logistics

1. Why Airbnb, and why now? [Reported, unverified] [1][2]
   - Which Airbnb product surface (Search, Pricing, Trust, Experiences, Payments) would you most want to work on, and why?
   - Have you used Airbnb as a guest or as a host? What would you change?
2. Walk me through your background in under three minutes, focusing on ML in production. [Reported, unverified] [1]
3. What level are you targeting, and what scope of work did you own at your current level? [Likely]
   - How many engineers or scientists did you lead or influence on your largest project?
4. Which programming language will you use in coding rounds, and are you fluent enough to run code without IDE help? [Likely] [3]
5. How much of your recent work was modeling, and how much was infrastructure or experimentation? [Likely]
6. Have you designed or analyzed online A/B tests yourself, or did a data scientist own that? [Likely]
7. What are your compensation expectations and timeline, and do you have competing processes? [Reported, unverified] [2]
8. Are you open to team matching across Search/Relevance, Pricing, Trust and the Experimentation platform? [Likely]

### B. Technical phone screen (practical coding, must run)

9. Display page: given CSV strings of search results (host_id, listing_id, score, city) sorted by score and a page size of 12, reorder them so a host appears at most once per page if possible, otherwise preserving order, and print the pages separated by a blank line. [Reported] [11]
   - What happens on the last page when only one host's listings remain?
   - What is the time complexity, and can you do it in a single pass?
10. Round prices: round each price to its floor or ceiling so the rounded sum equals Round(sum of prices), minimizing the total absolute rounding error. [Reported] [11]
    - What if the target is unreachable? How do you signal that?
    - How would you handle currency with two decimal places?
11. Pour water: given terrain heights, a water quantity and a pour position, simulate the droplets and print the final terrain with water. [Reported] [11]
    - How do you test the boundary indices?
12. Text justification: fully justify lines of words to a fixed width. [Reported] [11]
    - How do you handle a single word on a line, and the last line?
13. List-of-lists iterator with `hasNext`, `next` and `remove` (remove the last returned element; calling it twice is an error). [Reported] [11]
14. Implement a CSV parser that handles quoted fields, escaped quotes and commas inside quotes. [Reported] [11]
    - How do you handle multi-line quoted fields?
15. Given an IPv4 address and a count, return the minimal list of CIDR blocks covering exactly that range. [Reported] [11]
16. Write tests for your solution right now. Which cases did you choose, and why? [Likely] [3]
17. Collatz conjecture: for every start value in [1, n], find the maximum number of steps to reach 1. [Reported] [11]
    - How would memoization change the runtime, and how do you bound memory for large n?
18. Implement a queue backed by a set of fixed-size arrays (no dynamic array resizing). [Reported] [11]
19. Travel buddy: a friend whose city wishlist overlaps yours by more than 50% (of their list) is a buddy. Return buddies sorted by similarity. [Reported] [11]
    - Extend it to recommend up to N cities from buddies' wishlists.
20. Find case combinations of a string: generate all upper/lower case variants and test each against a hidden-string checker API. [Reported] [11]
    - How do you limit API calls when the checker is expensive?
21. Finding ocean: given a map of L (land) and W (water) and a starting ocean cell, mark all connected ocean cells as O and print the map. [Reported] [11]
    - When does recursive DFS fail on large maps, and what do you replace it with?
22. Guess the number: a secret 4-digit code with digits 1-6 sits behind an API that returns the count of correctly placed digits. Find the code with as few calls as possible. [Reported] [11]

### C. Onsite coding round 1 (algorithmic, Airbnb-tagged style)

23. Alien dictionary: derive a character ordering from a sorted list of words. [Reported, unverified] [4]
    - How do you detect an invalid ordering, such as a prefix appearing after a longer word?
24. Palindrome pairs: find all index pairs whose concatenation forms a palindrome. [Reported] [11]
    - Can you do better than O(n^2 * k)?
25. Minimum cost with at most K stops: given flights (from, to, price), find the cheapest itinerary from source to destination. [Reported] [11]
    - What changes if edges carry both price and duration and you must optimize a weighted objective?
26. Sliding game: find a sequence of moves that solves a 3x3 eight-tile puzzle. [Reported] [11]
    - How does your search change for an unsolvable board, and how do you detect it up front?
27. String pyramid transition: given the bottom row and a map of allowed parents for each (left, right) pair, can the pyramid reach a root? [Reported] [11]
28. Meeting time: given everyone's meeting intervals, return the common free intervals. [Reported] [11]
    - How would you adapt this to host calendars with blocked dates and minimum-night rules?
29. Smallest common region: find the lowest common region in a region hierarchy. [Likely] [4]
30. Menu combination sum: find all combinations of menu item prices that sum to a budget K. [Reported] [11]
    - How do you handle float prices without precision bugs?
31. Simple regex matching supporting `.`, `*` and `+`. [Reported] [11]
32. Boggle game: on a letter grid with a dictionary, find the path that contains the largest number of dictionary words. [Reported] [11]
33. Basic calculator with `+ - * /` and parentheses. [Likely] [4]
34. Contains duplicate within distance k and within value difference t. [Likely] [4]
35. Mini parser for nested lists (nested-list iterator). [Likely] [4]
36. Maximum nights you can accommodate: given consecutive-night booking requests, pick bookings that maximize occupied nights, leaving at least one day gap between bookings for cleaning. [Reported] [11]
    - What if bookings are intervals with arbitrary overlaps (weighted interval scheduling)?
37. K edit distance: find all dictionary words within edit distance k of a target word. [Reported] [11]
    - How would a trie cut the work compared with checking each word?
38. Preference list: merge everyone's ordered city preferences into one list consistent with all of them. [Reported] [11]
    - What do you return when the preferences contain a cycle?
39. Ten wizards: each wizard knows a list of others, and a hop from i to j costs (i - j)^2. Find the minimum-cost path from wizard 0 to wizard 9. [Reported] [11]
40. Minimum vertices to traverse a directed graph: return the smallest set of start vertices from which every vertex is reachable. [Reported] [11]
41. Find the median of a file of integers too large for memory, with only sequential access. [Reported] [11]
42. Hilbert curve: given (x, y) and an iteration count, return the step index of that point along the curve. [Reported] [11]
43. Number of intersecting rectangles: given many axis-aligned rectangles, count how many groups of them intersect. [Reported] [11]

### D. Onsite coding round 2 (practical, evolving requirements)

44. Design and implement a booking calendar that supports `book(start, end)`, rejects overlaps and answers `is_available(start, end)`. [Likely] [3]
    - Now add per-night pricing and return the total price for a stay.
    - Now support split stays across two listings when no single listing is available.
45. Implement a simple key-value store with nested transactions (`begin`, `commit`, `rollback`). [Likely] [3]
46. Parse a log of experiment exposures (user_id, experiment, variant, timestamp) and output per-variant unique user counts, de-duplicating users exposed to multiple variants. [Likely]
    - How would you flag and count users with conflicting assignments?
47. Implement deterministic hash-based bucketing: given user_id, experiment salt and traffic split, return a variant. [Likely] [6]
    - How do you guarantee independence between experiments run on the same users?
    - How would you ramp traffic from 1% to 50% without reshuffling already-exposed users?
48. Given streaming events, compute a rolling 7-day conversion rate per listing with bounded memory. [Likely]
49. Implement a file system with `create(path, value)` and `get(path)`, returning an error when a parent path does not exist. [Reported] [11]
    - Add a `watch(path, callback)` that fires on changes to the path or its children.
50. Given guest reviews with ratings and timestamps, return the top-K listings by a time-decayed average rating. [Likely]
    - How do you handle listings with very few reviews (shrinkage)?
51. Implement an LRU or LFU cache for search results, then extend it with a TTL. [Likely]
52. Refactor your solution so that a new requirement (for example multi-currency support) fits in with minimal change. How do you structure the code? [Likely] [3]
53. Simulate Diplomacy: given orders per army (move, hold, support), resolve the board, where stronger armies win a contested location and ties eliminate both. [Reported] [11]
    - A new rule arrives mid-interview: a support is cut if the supporting army is attacked. How do you add it without rewriting the resolver?
54. Given per-user rows (variant, converted, pre-period bookings), write runnable code that returns the lift and a 95% confidence interval for each treatment arm. [Likely] [6]
    - Add a CUPED adjustment using the pre-period covariate and show the variance reduction on the test data.
55. Given daily assignment counts per variant, write a function that flags sample ratio mismatch against the intended split. [Likely] [6]

### E. ML design round (Airbnb product + experimentation)

56. Design the search ranking system for Airbnb stays. [Likely] [7][8][9]
    - What are your labels (bookings, clicks, long clicks, host acceptances), and how do you weight them?
    - How do you handle position bias in training data?
    - How do you evaluate offline (NDCG on booked listings) versus online?
57. Design real-time personalization using listing embeddings from in-session clicks. [Likely] [8]
    - How would you inject booking or market signals into the embedding objective?
    - How do you handle cold-start listings with no sessions?
58. Design a dynamic pricing suggestion model for hosts (Smart Pricing). [Likely]
    - How do you estimate demand elasticity when price is confounded with seasonality and quality?
    - How do you A/B test a pricing model when hosts can override it?
59. Design a fraud or risk model for bookings (Trust and Safety). [Likely]
    - How do you evaluate a model whose positive labels arrive weeks late and are censored by your own interventions?
60. Design "similar listings" or "recommended experiences" recommendations. [Likely]
61. Design a model that predicts host acceptance probability and use it in ranking. [Likely] [7]
62. Design an ML system to detect low-quality or misleading listing photos and text. [Likely]
63. How would you design the online experiment that launches your new ranking model? [Likely] [6][9]
    - What is the primary metric, and what are your guardrail metrics?
    - What unit of randomization would you use (user, session, listing, market), and why?
    - How do you estimate the required sample size and duration?
64. In a two-sided marketplace, why can a user-randomized test of a ranking change be biased? [Likely] [6]
    - How would listing-side or market-level (cluster) randomization change your estimates?
    - What is interference, or a SUTVA violation, in this setting?
65. When would you use interleaving instead of A/B testing for search ranking? [Likely] [9]
    - How does interleaving sensitivity compare with A/B, and what can interleaving not measure?
66. Offline NDCG improved but the online A/B showed no booking lift. How do you debug this? [Likely] [7]
67. Bookings happen weeks after search. How do you choose an experiment metric that is sensitive enough but still aligned with long-term value? [Likely] [6]
    - How would you build and validate a proxy or surrogate metric?
68. How would you apply variance reduction (CUPED or regression adjustment) to booking-conversion experiments? [Likely] [6]
69. How do you handle novelty effects and seasonality (holidays, events) when reading test results? [Likely]
70. How do you control false discoveries when a team runs dozens of concurrent experiments and checks results daily? [Likely] [6]
    - What is the peeking problem, and how do sequential tests or dynamic p-value thresholds address it?
71. How would you measure the long-term effect of a model change after the A/B test has ended (holdouts)? [Likely]
72. Your model improves guest conversion but reduces host earnings in some markets. How do you decide whether to ship? [Likely]
73. The ranking team wants to ship a model every two weeks, but each booking-metric A/B test needs four weeks. How do you redesign the evaluation pipeline? [Likely] [6][9]
74. Design an ML model that predicts which experiments are likely to produce a false positive (for example, from metric variance, traffic and peeking history) so the platform can warn owners. [Likely] [6]

### F. System design round

75. Design an experimentation assignment and logging service used by every Airbnb client (web, iOS, Android, backend). [Likely] [6]
    - How do you keep assignment consistent across logged-out and logged-in states?
    - How do you guarantee exposure logging is not lost or double-counted?
76. Design the metrics computation pipeline for an experimentation platform (daily scorecards for thousands of metrics). [Likely] [6]
    - How do you backfill when a metric definition changes?
77. Design a booking and reservation system that prevents double-booking under high concurrency. [Likely] [1]
78. Design a feature store that serves both offline training and low-latency online ranking. [Likely]
    - How do you prevent training-serving skew?
79. Design a search service that retrieves and ranks listings within 200 ms p99 across millions of listings. [Likely] [7]
    - Where do you place candidate retrieval, filtering (availability, price, map bounds) and ML re-ranking?
80. Design a model serving and rollout system with shadow mode, canary and automatic rollback. [Likely]
81. Design a notification system for hosts and guests that respects time zones and rate limits. [Likely]
82. Design a calendar availability service for millions of listings with frequent updates and iCal sync. [Likely]
83. Design a feature-flag and kill-switch service that lets an on-call engineer turn off a misbehaving model in every region within one minute. [Likely]

### G. Past project deep dive

84. Walk me through the ML project you are proudest of, from problem framing to measured impact. [Reported, unverified] [1][2]
    - What was your specific contribution compared with the team's?
    - What alternatives did you reject, and why?
85. How did you measure the impact of that project? Was there an A/B test, and what did it show? [Likely]
    - What were the confidence intervals, and were there any surprising segment-level results?
86. What was the hardest technical trade-off you made on that project? [Likely]
87. If you had to redo the project with half the time, what would you cut? [Likely]
88. Tell me about a model that worked offline but failed in production. What did you learn? [Likely]
89. How did you get alignment from Product and Data Science on the success metric? [Likely]
90. What would the next iteration of that system look like? [Likely]
91. How did you handle data quality or labeling problems in that project? [Likely]
92. Which part of the system would break first at 10x scale? [Likely]
93. Your project's A/B test showed a lift that later disappeared in the holdout. Walk me through how you found the cause. [Likely]

### H. Core values interviews (two rounds, cross-functional)

94. Which of Airbnb's core values resonates with you most, and why? [Reported, unverified] [1][5]
95. Tell me about a time you went above and beyond for a customer, user or teammate ("Be a Host"). [Reported, unverified] [1][5]
96. Tell me about a time you took a big risk or tried something no one had done before ("Embrace the Adventure"). [Reported, unverified] [1][5]
97. Tell me about a time you had to be scrappy and resourceful with limited resources ("Be a Cereal Entrepreneur"). [Reported, unverified] [1][5]
98. What does "belonging" mean to you, and when have you created it for someone else ("Champion the Mission")? [Reported, unverified] [1][5]
99. Describe a memorable travel or hosting experience and what it taught you. [Reported, unverified] [1]
100. Tell me about a time you failed. What did you do next? [Reported, unverified] [1][2]
101. Tell me about a time you disagreed with a teammate or manager. How did you resolve it? [Reported, unverified] [2]
102. Tell me about a time you received difficult feedback. [Reported, unverified] [2]
103. Describe a time you helped someone grow or mentored a junior colleague. [Likely]
104. Tell me about a time you made a decision that was right for users but unpopular internally. [Likely]
105. How do you build inclusive teams or include people with different perspectives? [Likely] [5]
106. Tell me about a time you had to adapt to a big change you did not choose (a reorg, layoffs, a pivot). [Likely]

### I. Senior-level calibration probes (appear across rounds)

107. How do you decide when a simpler heuristic should ship instead of an ML model? [Likely]
108. Tell me about a time you set the technical direction for a team without formal authority. [Likely]
109. How do you prioritize between model quality, latency and engineering cost? [Likely]
110. Describe a time you stopped or reversed a launch because the experiment results were ambiguous. [Likely]
111. How do you communicate a negative or inconclusive A/B result to leadership? [Likely]
112. How do you make sure your experiment results are reproducible and trusted by other teams? [Likely]
113. What metrics would you put on a dashboard to know your production model is healthy? [Likely]
114. How do you balance short-term booking metrics against long-term guest and host trust? [Likely]
115. A product manager wants to ship a feature whose test was stopped early at p = 0.04 after daily peeking. What do you tell them, and what do you change in the process? [Likely] [6]
116. What questions do you have for us about the team, the experimentation culture or the roadmap? [Reported, unverified] [1]

## References

Only [11] was fetched and checked during the 2026-10-14 review. Sources [1]-[10] could not be reached because the proxy blocked their domains. They are kept because they are well-known and their bibliographic details match prior knowledge, but each one is marked "not re-verified". Open each before relying on it.

1. "Airbnb Interview Process & Questions (+ how to prepare)", IGotAnOffer (not re-verified). https://www.igotanoffer.com/blogs/tech/airbnb-interview-process
2. "Airbnb Machine Learning Engineer Interview Questions", Glassdoor candidate reports (not re-verified). https://www.glassdoor.com/Interview/Airbnb-Machine-Learning-Engineer-Interview-Questions-EI_IE391850.0,6_KO7,32.htm
3. "Airbnb" interview discussions, LeetCode Discuss search (not re-verified). https://leetcode.com/discuss/interview-question?currentPage=1&orderBy=hot&query=airbnb
4. Airbnb company-tagged problems, LeetCode (Premium; not re-verified). https://leetcode.com/company/airbnb/
5. Airbnb Careers, core values (not re-verified). https://careers.airbnb.com/
6. Overgoor, "Experiments at Airbnb", Airbnb Engineering & Data Science (Medium), 2014 (not re-verified). https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7
7. Haldar et al., "Applying Deep Learning to Airbnb Search", KDD 2019 (not re-verified). https://arxiv.org/abs/1810.09591
8. Grbovic & Cheng, "Real-time Personalization using Embeddings for Search Ranking at Airbnb", KDD 2018 (not re-verified). https://dl.acm.org/doi/10.1145/3219819.3219885
9. Haldar et al., "Improving Deep Learning for Airbnb Search", KDD 2020 (not re-verified). https://arxiv.org/abs/2002.05515
10. The Airbnb Tech Blog, Airbnb Engineering (Medium) (not re-verified). https://medium.com/airbnb-engineering
11. allaboutjst, "AirBnB Interview Questions" (community-collected from Glassdoor and GeeksForGeeks; 31 questions; v1.0 2017, last updated 2020-08-26). Verified 2026-10-14. https://github.com/allaboutjst/airbnb (raw: https://raw.githubusercontent.com/allaboutjst/airbnb/master/README.md)
