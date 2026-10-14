# 02. Coding Rounds: Airbnb-Tagged LeetCode and Reported Practical Coding Questions

> Research note (read first). First researched 2026-10-13; adversarially reviewed 2026-10-14. On review, web search was unavailable (the shared search budget was exhausted), and direct HTTP to GitHub returned proxy 403s, but pages could still be fetched through a fetch tool. About 50 cited pages were refetched and read on review; a few returned transient 503s and were left as cited. Glassdoor, Blind, LeetCode, Reddit, 1point3acres, Prepfully and Airbnb's own sites were not reachable. **All evidence below comes from GitHub-hosted pages:**
> - two open mirrors of LeetCode's Airbnb company tag [5][6];
> - two mirrored TrueInterview candidate-report guides, one for the MLE track and one for the SWE track [1][2];
> - a dated question tracker (Sep 2025 to Sep 2026) and a 2025-2026 company guide [3][4];
> - classic Airbnb question collections [7][8][9][10];
> - a GitHub mirror of about 100 1point3acres (一亩三分地) Airbnb interview posts [13]. Posts that carry a specific question are cited individually [14]-[58] and [59]-[82].
>
> All of this is third-party, user-submitted material, so treat every [Reported] tag as "a candidate or aggregator says so", not as Airbnb confirmation. Several mirror files carry no role, level or date, and some do not name the company; they are attributed to Airbnb only because the mirror files them under an Airbnb folder. Where that is the only link to Airbnb, the question says so. [Likely] marks an inference from the MLE/experimentation role. It does not mean anyone reported it.

## Format notes

- **How many coding rounds an MLE gets.** The one public MLE-track summary (3 candidate reports, mainly 2026 Q2) gives MLEs one technical phone screen and **typically one onsite coding round** (45-60 min). It says the track is "filtered by design depth, not by code": the decisive rounds are 1-2 ML system design rounds [1]. One 1point3acres MLE-type loop had one coding round (LC 1298) plus two ML design rounds (LTV and host pricing) [14]. Another had two ML design rounds plus a "ConnectN" OOD coding round [15]. SWE-track loops usually have 1-2 coding rounds plus a separate **code review** round (about 3 PRs) [2][4]. One aggregator says code review replaced the second coding round around 2024, while some 2026 India loops still had two 45-minute coding rounds [4]. In several 2025-2026 loops, a weak code review round led to an added coding round or to rejection [59][72][77]. Ask your recruiter whether the MLE loop includes code review.
- **Runnable code with tests is the bar.** Candidates report that Airbnb requires "bug free" code on the spot with all cases passing [20]. Other reports: code must run and be tested within 45 minutes [2]; there were "about 10 test cases" for Connect 4 [32]; problems arrived with stdin-style test strings [17]; one interviewer gave no interface, so the candidate had to agree the API and then write tests that pass [36]. Airbnb has long stressed "compilable, runnable code" [11]. One MLE screen expected the candidate to explain pruning or memoization *before* coding, then write test cases [1].
- **Shape of a round.** A round is one practical problem in 45 min, often in two parts. Interviewers add constraints once part 1 runs, so leave 10-15 minutes for follow-ups [4][33]. One candidate instead got three short LeetCode problems in 60 minutes [47]. Several posts say the interviewer said nothing and just pasted the problem [18][20]. Communication still counts: a promotional prep guide claims that silent candidates score lower than ones who think aloud [12]. Switching algorithms mid-round and polishing helpers too early are cited as reasons for rejection [2].
- **Flavor.** Problems are framed as product stories: split stays, group lodging, menus and combos, refunds, layovers, ski runs, host pagination, review highlighting. They test DP (often bitmask), graph search, simulation, parsing and small-object design rather than exotic tricks [1][3][4]. LeetCode's Airbnb tag over the last 6 months is led by Design Excel Sum Formula, Maximum Profit in Job Scheduling, Shortest Uncommon Substring, Maximum Candies from Boxes, Shopping Offers, Accounts Merge, Text Justification and Smallest Sufficient Team [5].
- **Senior bar.** Seniors are expected to clarify ambiguous specs themselves, for example tie-breaking rules, whether the graph has cycles, and whether to return the path or only the cost [16]. They must also reach working code fast enough to discuss complexity and extensions. One candidate passed with a suboptimal solution that ran and passed every test, and only *described* the optimization [12][19]. Another failed because they could not beat O(N^2) when asked [19]. Several senior rejections were attributed to design or deep-dive rounds even when coding was strong [16][28].

## Questions

### A. Airbnb "classic" practical problems (long-running question bank)

1. Display Page / pagination: given CSV search results (host_id, listing_id, score, city) sorted by score and a page size, reorder them so that a host appears at most once per page when possible, while otherwise preserving order. Print the pages as blocks. [Reported] [7][8][52]
   - What should happen when the remaining results all belong to one host?
   - Can you do it in one pass without repeatedly rescanning the list?
   - How would you test that the relative order inside a page is preserved?
2. Text Justification: pack words greedily into lines of exactly width L, spreading extra spaces left-first, with the last line left-justified. [Reported] [5][7][8]
   - How do you handle a line that holds a single word?
   - What changes if words can be longer than L?
3. Fixed-width text formatting (simpler variant: left-align and pad with trailing spaces), then lay the lines out into k side-by-side columns separated by one space. [Reported] [1][40][50]
   - When one column runs out of lines, what do you print in its place?
   - What is the complexity for n up to 1e5 words?
4. Menu combination sum: given menu item prices, return every combination of items whose total is exactly the budget. Items may repeat, and duplicate combinations are not allowed. [Reported] [7][49]
   - Prices are floats like 2.40. How do you avoid floating-point equality bugs? [Reported] [17][78]
   - Now return only the count of combinations. Does order matter? [Reported] [10][48]
5. Palindrome Pairs: find all pairs (i, j) such that words[i] + words[j] is a palindrome. [Reported] [5][7][39]
   - The interviewer wants O(n*k) rather than O(n*k^2). Which requirement do you clarify first? [Reported] [16]
   - How do the empty string and duplicate words affect correctness?
6. IP range to CIDR: given a starting IPv4 address and a count n, return the shortest list of CIDR blocks that covers exactly that range. [Reported] [5][7][8]
   - How do you test the block boundaries, for example a range starting at an odd address?
7. List-of-list (2D) iterator with `hasNext`, `next` and a `remove` that may be called once per `next` (LeetCode: Flatten 2D Vector; reported Oct 2025 as "Flatten 2D Array With Remove"). [Reported] [3][5][7][8]
   - What should `remove` do when it is called twice, or before `next`?
   - How would you make the iterator lazy over a very large nested input?
8. Flatten Nested List Iterator. [Reported] [5][6]
9. Cheapest flights / minimum cost with at most K stops. Return both the cost **and** the route. [Reported] [5][7][27]
   - If the route must be returned, which algorithms still work?
   - How would you handle ties, or a cycle of zero-cost edges?
10. Sliding puzzle: find the minimum moves to solve the 2x3 board (LC 773), or produce a move sequence for the 3x3 (8-puzzle) variant. [Reported] [5][7][8]
    - How do you detect that a board is unsolvable before searching?
    - Would A* help, and with which heuristic?
11. Terrain rendering plus water simulation: Part 1 prints the terrain as ASCII from a heights array. Part 2 drops water at a column and prints the terrain with W for water. The classic form is Pour Water (LC 755): pour V units at index K and return the final heights. [Reported] [3][5][7][35]
    - Which way should water flow when both sides are equally low?
    - How do you test the printed output exactly, including trailing whitespace?
12. Round prices: round each price up or down so that the rounded values sum to the rounded total, minimizing the total rounding error (LC 1058). [Reported] [5][7][10]
    - What do you return when the target is unreachable?
13. CSV parser: parse a CSV line with quoted fields, escaped quotes and commas inside quotes. [Reported] [7][8]
    - How do you handle a quoted field that spans multiple lines?
14. Design File System: `create(path, value)` fails if the parent is missing, and `get(path)` returns the value. [Reported] [5][7]
    - How would you add `delete` with recursive children?
    - Add `watch(path, callback)`, which fires when a `create` succeeds on that path or any path below it. [Reported] [81]
15. Alien Dictionary: derive the letter order from a sorted word list. [Reported] [5][6][8]
    - How do you detect an invalid input, such as a longer word appearing before its own prefix?
16. Regular expression matching with `.`, `*` and also `+`. [Reported] [5][7]
17. Boggle game: find the path on a letter grid that spells the **largest number** of dictionary words (not just any word). Word Search II is the tagged LeetCode version. [Reported] [5][7]
18. Pyramid transition matrix: can the bottom row be built up to a single root using the allowed triples? [Reported] [5][7][9]
19. Ten wizards: the cost of a link is the squared difference of wizard IDs. Find the minimum-cost path from wizard 0 to wizard 9. [Reported] [7][8][9]
20. Travel buddy list: friends whose wishlists overlap more than half with yours are "buddies". Recommend up to `max` cities from buddies' wishlists, ordered by similarity. [Reported] [7][57]
    - How do you break ties between buddies with equal overlap?
21. Collatz conjecture: find the longest step count for any start value in [1, n]. [Reported] [3][7][65]
    - How would you memoize, and how do you bound memory for large n?
22. Find the median of a file of integers too large for memory, with only sequential access. [Reported] [3][7][8]
23. Guess the number: find a hidden 4-digit code (digits 1-6) using an API that returns how many digits are in the correct position. Minimize the number of API calls, and supply your own test cases. [Reported] [3][7][54][69]
    - Variant: talk to a server over a socket with a START/GUESS protocol. How do you structure the I/O? [Reported] [54]
24. Meeting time / Employee Free Time: return the common free intervals across everyone's schedules. [Reported] [5][7]
25. Maximum number of nights you can accommodate: choose non-adjacent booking requests to maximize nights (House Robber framing). [Reported] [5][7][9]
26. Generate all upper- and lower-case combinations of a string and test each against a hidden string. [Reported] [7]
27. K edit distance: find all dictionary words within edit distance k of a target. [Reported] [7][8]
    - How would a trie change the complexity?
28. Preference list: merge everyone's ranked lists into one ordering that respects every person's order. [Reported] [7][8]
29. Minimum vertices from which every node of a directed graph is reachable. [Reported] [3][5][7]
    - Does the graph contain cycles, and if so, how does that change the answer? [Reported] [16]
30. Finding ocean (flood fill from a water cell), number of intersecting rectangles, and Hilbert curve step count. [Reported] [7][8][10]
31. Implement a queue backed by a chain of fixed-size arrays. [Reported] [7][8]

### B. Practical problems reported in 2025-2026 (current question bank)

32. Split stay: given listings with lists of available day numbers and a date range, return every pair of listings [X, Y] where X covers a prefix of the range and Y covers the rest (one split only). [Reported] [3][18][19][20]
    - Your O(N^2 * D) solution works. The interviewer says something better exists. Where is the redundant work?
    - What if a split stay may use up to three listings?
    - How do you deduplicate symmetric pairs, and must the order be preserved?
    - Must the first listing cover a contiguous prefix and the second a contiguous suffix, or is plain set coverage enough? (The two readings give different answers on the reported example.) [Reported] [59]
    - One candidate wrote brute force with `combinations` and earned extra credit for a bitmask encoding of availability. How would you use one? [Reported] [59]
33. Group lodging / select properties: for a target neighborhood and group size, choose properties with total capacity >= group size, minimizing the number of properties and then the total capacity. Return the IDs, or -1 or [] if impossible. [Reported] [4][21][41]
    - Which tie-break do you confirm first: fewer properties or less wasted capacity? [Reported] [21]
    - Is this subset-sum-like DP or backtracking, and what is the worst-case cost?
34. Minimum-cost menu order: given single items and combos with prices plus a list of wanted items (at most 3 unique), return the lowest cost **and** the items to order. [Reported] [1][16][22][23]
    - You wrote DFS. Now convert it to DP. How do you add bitmask state? [Reported] [22]
    - Which menu entries can you prune before searching? [Reported] [16][56]
    - Return every optimal combination, not just one. [Reported] [16][62][63]
    - Variant II: order *at least* the required dish types using combos or singles. Why discard combos containing no required item before the DP? [Reported] [68]
    - Variant: rental "services" such as wifi, parking and TV instead of food. [Reported] [23]
35. Menu bundles closest to budget: choose up to 3 bundles whose total is closest to, but not above, the budget. Write at least 5 test cases. [Reported] [35]
36. Layover: the experience durations can each be booked repeatedly. Can you exactly fill the layover hours? Follow-up: what is the minimum number of bookings? [Reported] [1][3][51]
37. Connect Four simulation: take a list of column moves, stop on a win or an invalid move, ignore the remaining moves, and print the grid. [Reported] [3][17][32][37][76]
    - How do you avoid rescanning the column for gravity on every move? [Reported] [32]
    - Check the diagonals without off-by-one errors in the limited time. [Reported] [16]
    - Determine win, draw or ongoing from a static board alone. [Reported] [3][73][74]
    - One interviewer asked for logically clear code over a clever algorithm, yet the candidate expected to fail for not running it. How do you budget 45 minutes so the diagonal check and a test run both fit? [Reported] [74]
38. ConnectN object model: configurable width, height and N in a row to win, alternating players, and draw detection. [Reported] [15]
    - Which classes and interfaces would you expose so that a new rule, such as an irregular board, is easy to add?
39. Retryer: wrap a function with maximum attempts, an interval, and exponential backoff (for example capped at 16 s). Rethrow the last error. Make it extensible. [Reported] [2][10][32][75][78]
    - How would you plug in jitter, retry-on-specific-exceptions or a time budget?
    - How do you unit-test timing without real sleeps?
40. Key store with sum dependencies: `set_value(k, v)`, `set_sum(k, [keys])` (keys may repeat) and `get_value(k)`. Updates must cascade to dependents. [Reported] [3][24][77][82]
    - What happens on a cycle?
    - Choose between eager propagation and lazy recompute for a workload with frequent writes and rare reads. [Reported] [2][16]
41. Design Excel Sum Formula (LC 631). [Reported] [4][5]
42. Banking class: deposit, withdraw, transfer, balance, and balance change over (start, end], where the boundary convention matters. [Reported] [2][3][34][62][63][64]
    - Your tests fail on a boundary. Which spec detail did you assume?
    - Online-assessment version: extend in levels to top spenders, scheduled payments and account merging, where later levels reuse earlier data structures. [Reported] [4]
43. Refund allocation: given payments (method, date, amount) and existing refunds, allocate a new refund in full to one payment before the next. Prioritize by method (CREDIT, then CREDIT_CARD, then PAYPAL), and within a method by most recent payment. [Reported] [2][3][42]
    - How do earlier partial refunds change the remaining refundable amount per payment?
    - How do you design the data structures when the interviewer gives only examples? [Reported] [42]
44. URL query parser: parse `?k=v&...` into a map. Duplicate keys become lists, a bare key means true, and percent-encoded separators (%26, %3D) are decoded *after* splitting. [Reported] [2][3][26]
    - In a typed language, how do you return values that may be a string or a list?
    - Variant: a valueless key maps to an empty string rather than true. Which spec detail do you confirm first? [Reported] [2]
45. Ski tournament: in a DAG of checkpoints with point values and edge travel times, find the path from start to any end node that maximizes points minus time, and print the path. [Reported] [3][17][42]
    - Is the input guaranteed acyclic? What if it is not?
    - Is Dijkstra correct here, and why or why not?
    - Extend to multiple skiers. How should the first version be structured so this follow-up is cheap? [Reported] [2][61]
46. Task scheduling with deadlines: unit-time tasks with a deadline and a reward. Return the order that maximizes total reward (for example, (profit, deadline) = (100,2), (19,1), (27,2), (25,1), (15,3) gives 142). [Reported] [4][28][67][82]
    - Why does the naive greedy fail, and what is your optimal-runtime approach? [Reported] [28]
    - How do you find the latest free slot before a deadline quickly? [Reported] [2]
47. Maximum Profit in Job Scheduling (LC 1235). Follow-up: output the chosen jobs. [Reported] [5][38]
48. Account registration deduplication: accounts share email, phone or government ID. Return all duplicates of a given account. [Reported] [2][3][4]
    - Generalize from two attributes to an arbitrary attribute set. [Reported] [2]
    - What changes if new accounts stream in continuously?
49. Build the smallest number from a list of digits (leading zeros dropped). Part 2: the smallest permutation that is >= a lower bound. [Reported] [2][30][55][64]
50. Intersection of two linked lists where either list may contain a cycle (four structural cases). [Reported] [2][3][5]
51. Multi-stream round-robin iterator: read bytes from several character streams in round-robin order with `has_next` and `next`. [Reported] [2][37]
52. Calculate board score: for every cell, add the length of the same-character run going right and going down. R and C go up to 2000. [Reported] [3][53]
53. Run-length encoding of an int array. Follow-ups: arithmetic-sequence RLE, and an iterator-based decoder for each encoding. [Reported] [25]
54. Sum tree: leaves hold numbers and parents hold the sum of their children. Return any node's value, then support mutations (leaf to subtree, subtree to leaf) while keeping reads correct. [Reported] [29]
55. Payment data store: `insert(ts, value, id)`, `get(id)` and `getValueInRange(start, end)`. [Reported] [33]
56. Keyword tagging in reviews: wrap keywords as `[tag]{keyword}` case-insensitively. Overlapping matches go to the longer keyword. [Reported] [3][10]
57. Replace text in a review using a replacement dictionary. [Reported] [3][10]
58. Shortest uncommon substring: for each string, find the shortest substring that appears in no other string (LC 3076). [Reported] [4][5][31]
59. Maximum Candies You Can Get from Boxes (LC 1298). It was given with no interface, so you define the API and write passing tests. [Reported] [5][14][36]
    - One solution passed every test but was rejected because its class model did not match the interviewer's intent. Which entities and methods would you model? [Reported] [2]
60. Office layout grid (-1 wall, 0 open): find the maximum reachable open area and count the unreachable rooms. [Reported] [36]
61. Two water jugs: return the minimum operations to measure a target volume **and** the sequence of states. [Reported] [10][27][64]
62. Given three numbers, list every way the first two combine with + - * / to equal the third. Follow-up: allow parentheses. [Reported] [45][64]
63. Card pile: by moving the top card to the bottom, can you reach N identical consecutive cards? [Reported] [44]
64. Parse strings of letter-score tokens and compute per-letter totals. [Reported] [43]
65. Find Median from Data Stream (LC 295). [Reported] [5][46]
66. Chain booking: find the maximum number of cascading reservations. [Reported] [2][3]
67. Compute browsing metrics in Python from event logs. [Reported] [3]
    - Which metrics would you compute per session versus per user, and how do you define a session?
68. Durable key-value store serialization, an in-memory database with SQL operations, and an in-memory object manager with TTL. [Reported] [2][3]
69. Car-speed controller: through a socket API (`getStatus`, `setStatus(acceleration)`), hold a simulated car at 29 mph. [Reported] [58] (The mirror files it under Airbnb; the page itself names no company.)

### C. LeetCode Airbnb-tag problems (company tag; frequency per the mirrors)

70. Smallest Sufficient Team (bitmask DP). It ranks near the top of the Airbnb tag frequency lists. [Reported] [4][5]
71. Shopping Offers (DP and backtracking with bundles, the LeetCode cousin of the menu problem). [Reported] [5][16]
72. Accounts Merge. [Reported] [4][5]
73. Smallest Common Region. [Reported] [5][6]
74. Detonate the Maximum Bombs. [Reported] [4][5]
75. Simple Bank System (LC 2043). [Reported] [5][6]
76. Trapping Rain Water. [Reported] [5][6]
77. Shortest Path to Get All Keys (LC 864). It was asked in the same round as Palindrome Pairs. [Reported] [5][39]
78. Minimum Number of Flips to Convert Binary Matrix to Zero Matrix. [Reported] [5][6]
79. Mini Parser and Basic Calculator II. [Reported] [5][6]
80. Strings Differ by One Character. [Reported] [5][6]
81. Number of Ways to Build House of Cards. [Reported] [5][6]
82. Minimum Window Substring and Minimum Window Subsequence. [Reported] [5][6]
83. Making a Large Island. [Reported] [5][6]
84. All O`one Data Structure. [Reported] [5][6]
85. Minimum Cost to Reach Destination in Time. [Reported] [5][6]
86. Time Based Key-Value Store. [Reported] [6][12]
87. Coin Change and Combination Sum. [Reported] [5][17]

### C2. Additional practical problems reported 2025-2026 (added on review)

88. Task build order: given tasks and prerequisite pairs (Course Schedule / Airflow-style DAG), decide whether every task can finish, then return a valid order. [Reported] [3][4][59][60]
    - One candidate mixed up in-degree and out-degree. Which test would have caught it? [Reported] [59]
    - How do you report which tasks are stuck in a cycle rather than just returning false?
89. Two-hotel coverage with interval input: hotels are given as (start, end) availability ranges. Can at most two hotels cover a whole stay? The reported example marks (Apr 1-10), (Apr 5-15), (Apr 20-30) as covering Apr 5-25 (True), even though Apr 16-19 is uncovered. What would you ask the interviewer before coding? [Reported] [79]
    - Are dates nights or days? How does that change the boundary test?
90. Maximum ski score on a height grid (n, m up to 100): each move goes to a strictly lower neighbor. Maximize the total height drop along a path. [Reported] [68]
    - Memoized DFS or processing cells in height order: which is easier to test in 45 minutes?
91. Accommodation allocation: requests and segments each have a start and an end, and segments cannot overlap. Return the maximum number of fulfilled requests and the allocation itself. [Reported] [66]
    - The reporter passed with brute force. When does the input size force something better?
92. Wallet class: create a wallet, deposit, withdraw and check the balance, with one wallet per user. A withdrawal that would overdraw must fail and leave the balance unchanged. No `main` or test harness is provided, so you write it. [Reported] [70]
    - Which invariants would you assert after every operation?
93. Single-source distances in a weighted undirected graph (phone-screen problem-solving part). [Reported] [64]
    - The reporter proposed topological sort. Why does that not work here, and what does?
94. Product of Array Except Self, without division and in O(n). [Reported] [72]
    - How do zeros in the input change your edge-case tests?
95. Maximum Product Subarray. [Likely] [71] (The mirror files it under Airbnb, but the page has no interview context.)
96. Search in a rotated sorted array, as a "classic with a twist" onsite problem. [Reported] [2]
    - How do duplicates break the O(log n) guarantee?
97. Remove digits to form the smallest possible number (Remove K Digits). [Reported] [2]
98. Print sentences inside a bordered, fixed-width table, wrapping words across lines. [Reported] [2]
    - What happens when one word is longer than the width?
99. Gift-card spend: find the fewest items whose prices exactly equal the card value, and return the items. [Reported] [2][3]
    - What changes if each item can be bought at most once?
100. Board-game region scoring: score every connected region of same-type cells on a grid. [Reported] [2]
101. Maze score: given a board-game maze state, compute its score via shortest path. The reporter warns to expect an OOD angle, not just an algorithm. [Reported] [4][61]
102. Infection spread time on a grid: return the time until every reachable cell is infected, or -1. [Reported] [2][3]
    - Follow-up: infection also spreads diagonally. How much of your BFS changes? [Reported] [2]
103. Diplomacy battle resolution: parse `<Army> <Location> <Hold | Support X | Move L>` actions. Each army has strength 1 and each valid supporter adds 1. In a contested location the strongest army wins and ties kill all tied armies. A supporter whose own location is attacked loses its support. Return each army's final location or `[dead]`. [Reported] [7][80]
    - Can an army support a move into the location it is supporting from? Which ambiguities do you raise before coding?
104. MLE phone screen: implement a key-value store, then state the test cases you would write against it. Onsite, extend it with additional constraints and show how the tests change. [Reported] [1]
105. Document redaction exercise, reported in recent senior MLE loops: given text and a set of sensitive terms or patterns, return the redacted text. [Reported] [1]
    - How do you handle overlapping matches, case and word boundaries? Which tests prove the redaction is complete?
106. Code review round: review about three pull requests in about 60 minutes and leave written comments. Reported planted defects include wrong results without a crash, swallowed exceptions, unchecked nulls, a race on shared state, missing error handling around an external API, and a hidden quadratic loop. [Reported] [2][4][59][60]
    - The third PR is long and time is short. How do you triage which comments to write first?
107. Multi-level online assessment: cloud storage or file store with add, copy, prefix and suffix search, per-user quotas, and compression and decompression. The next level unlocks only when the current tests pass. [Reported] [3][4]
    - How do you choose level-1 data structures that will not break at level 4?
108. Multi-level online assessment: task management system with add, update, get, search, priority sort, per-user quotas and an overdue report. [Reported] [4]

### C3. SQL and data-analysis coding reported for Airbnb (relevant to experimentation roles)

109. Find users with impressions on seven consecutive days. [Reported] [3]
    - How do you handle several impressions on the same day and time-zone boundaries?
110. Find hosts who meet listing-count, rating and review-coverage rules. [Reported] [3]
111. Estimate page-visit duration as the gap to the user's next event. [Reported] [3]
    - What do you do with the last event of a session, which has no next event?
112. Calculate the median house price in each city in SQL. [Reported] [3]
113. Compute the average vacant days per listing. [Reported] [3]
114. Match rentals to required amenities (listings that have all of a given amenity set). [Reported] [3]
115. Compute conversion (C/T) metrics from booking and visit tables. [Reported] [3]
    - Is the denominator visits, unique visitors or searches? How does the choice interact with the randomization unit?
116. Build a user-by-period panel in SQL, then run a causal regression on it. [Reported] [3]
    - Which fixed effects would you include, and how do you cluster the standard errors?
117. Model cumulative booking metrics with first and last booking times per user. [Reported] [3]

### D. ML/experimentation-flavored practical coding (role-specific, not yet reported)

None of these were reported for Airbnb. They follow the same practical, "make it run with tests" format and fit an MLE role on experimentation, search ranking or pricing.

118. Deterministic experiment bucketing: given user_id, an experiment salt and variant weights, return the assigned variant. [Likely]
    - How do you show that assignment is uniform and independent across two experiments?
    - How would you ramp from 5% to 50% without reassigning users already exposed?
119. From raw exposure and conversion logs (user, experiment, variant, ts, event), compute per-variant conversion, the absolute and relative lift, and a 95% confidence interval. [Likely]
    - How do you handle a user exposed to both variants, and events that happen before exposure?
120. Implement a sample-ratio-mismatch check for an expected 50/50 split and decide whether to alert. [Likely]
121. Implement CUPED or regression adjustment on a per-user metric with a pre-period covariate, and report the variance reduction. [Likely]
122. Implement a bootstrap confidence interval for the difference in a ratio metric (for example bookings per search). [Likely]
    - Why is resampling individual searches wrong when the unit of randomization is the user?
123. Implement NDCG@k for ranked search results with graded relevance (booking > click > impression). [Likely]
    - How do you handle queries with no relevant results?
124. Implement team-draft interleaving of two rankers' result lists and credit clicks to each ranker. [Likely]
125. Re-rank a scored list of listings to enforce host diversity: at most one listing per host in each window of k, as in the pagination problem. [Likely] [52]
126. Compute a rolling 7-day booking rate per listing from an event stream with bounded memory. [Likely]
127. Given availability calendars and nightly prices, compute the total price of a stay, including weekly-discount and cleaning-fee rules, and test the boundary nights. [Likely]
128. Reservoir-sample k events from a stream of unknown length, then make it weighted. [Likely]
129. Implement streaming AUC, or approximate quantiles, over a large prediction log without loading it all in memory. [Likely]
130. Write a sequential-testing guard: given daily cumulative results, decide whether a pre-registered stopping boundary has been crossed. [Likely]

### E. Cross-cutting probes interviewers use in every coding round

131. Before you code: what are the input format, the output format and the tie-breaking rules? Is any of them ambiguous? [Reported] [16][21]
132. Write your own test cases now. Which edge cases did you choose, and why? [Reported] [1][2][35]
133. What are the time and space complexity, and where is the bottleneck for the largest stated input? [Reported] [1][56]
134. Your code fails one hidden test. How do you debug it in the remaining five minutes? [Likely] [17][34]
135. A new requirement arrives, such as returning the path instead of the cost, or k columns instead of one. How much of your code has to change? [Reported] [3][4][27]
136. If this ran in production behind an API endpoint, what would you validate, log or cache? [Likely] [18]

## References

1. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview guide, mirrored on GitHub; based on 3 candidate reports, mainly 2026 Q2). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
2. "Airbnb Software Engineer Interview Process" (TrueInterview guide, CC BY 4.0, mirrored; based on 41 candidate reports, 2025 Q4 to 2026 Q2). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
3. "Airbnb" question list with report dates (algorithm, OOP and SQL questions, Sep 2025 to Sep 2026). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
4. "Airbnb" company guide (2025-2026 reports; merges LeetCode tag data, LeetCode Discuss, PracHub and Hello Interview). jugaldb/faang-interview-prep, GitHub, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
5. LeetCode company-tag mirror: Airbnb (All, Six Months and Three Months CSVs; updated June 2025). liquidslr/leetcode-company-wise-problems, GitHub, 2025. https://github.com/liquidslr/leetcode-company-wise-problems/tree/main/Airbnb
6. LeetCode company-tag mirror: airbnb/all.csv. snehasishroy/leetcode-companywise-interview-questions, GitHub. https://github.com/snehasishroy/leetcode-companywise-interview-questions/blob/master/airbnb/all.csv
7. "AirBnB Interview Questions" (31 classic problems with Java solutions). allaboutjst/airbnb, GitHub. https://github.com/allaboutjst/airbnb
8. "Airbnb_Interview" (Airbnb Beijing prep, 36 problems). hbsun2113, GitHub, 2019-2020. https://github.com/hbsun2113/Airbnb_Interview
9. "airbnb-interview-problems". shua2018ti, GitHub, 2018. https://github.com/shua2018ti/airbnb-interview-problems
10. "airbnb_interview_questions.md". warrenzhu25/system-design, GitHub. https://github.com/warrenzhu25/system-design/blob/master/airbnb_interview_questions.md
11. "Interview formats of top companies" (Airbnb section). Tech Interview Handbook (yangshun), GitHub. https://github.com/yangshun/tech-interview-handbook/blob/main/apps/website/contents/interview-formats-top-companies.md
12. "Airbnb coding interview questions: the problems that actually show up...". yhzl350/airbnb-dp-interview, GitHub, 2026. Promotional page with unverified claims. https://github.com/yhzl350/airbnb-dp-interview
13. Airbnb 1point3acres interview-post mirror (folder index). FlowSight/coding, GitHub. https://github.com/FlowSight/coding/tree/main/company%20wise/airbnb/1p3a
14. 1p3a post #40: project deep dive, coding LC 1298, ML design (LTV, host pricing). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
15. 1p3a post #79: two ML design rounds (listing recommendation, price prediction), coding ConnectN OOD, ML experience round. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
16. 1p3a post #37: compilation of 16 algorithm questions with clarification tips (menu, Connect 4, select property; LC 336/864/284/1298/1557/1235). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/37.md.md
17. 1p3a post #72: phone screen Connect 4 with print, Combination Sum, onsite ski DAG problem. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/72.md
18. 1p3a post #34: split stay problem statement. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/34.md
19. 1p3a post #44: split stay; O(N^2) judged not good enough. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/44.md
20. 1p3a post #15: hotel split stay; "bug free, all cases must run". https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/15.md
21. 1p3a post #38: group-size property selection DP. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/38.md
22. 1p3a post #10: menu min cost with bitmask; DFS then DP follow-up. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/10.md
23. 1p3a post #53: rental services min-cost variant. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/53.md
24. 1p3a post #65: set_value / set_sum / get_value key store. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/65.md
25. 1p3a post #87: RLE with arithmetic-sequence and iterator follow-ups. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/87.md
26. 1p3a post #100: URL query parser with percent decoding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/100.md
27. 1p3a post #101: cheapest itinerary with max layovers returning the route; two-jug problem. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/101.md
28. 1p3a post #104: task deadlines and rewards; greedy rejected. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/104.md
29. 1p3a post #55: G10 full-stack; sum-tree with mutations; retry higher-order function. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/55.md
30. 1p3a post #33: smallest number from digits; lower-bound follow-up. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/33.md
31. 1p3a post #27: shortest uniquely-mapping substring. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/27.md
32. 1p3a post #69: Connect 4 phone screen (about 10 tests, column-height optimization); onsite retryer with increasing interval. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/69.md
33. 1p3a post #64: payment data insert/get/getValueInRange. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/64.md
34. 1p3a post #98: banking system with timestamped balance changes. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/98.md
35. 1p3a post #12: menu bundles up to 3 with 5+ tests; terrain print and water drop. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/12.md
36. 1p3a post #24: LC 1298 with no interface given, tests required; office layout grid. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/24.md
37. 1p3a post #18: multi-stream iterator. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/18.md
38. 1p3a post #97: LC 1235 variant; running code on the spot. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/97.md
39. 1p3a post #50: coding round with LC 336 and LC 864. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/50.md
40. 1p3a post #85: simplified text formatting with column follow-up; task-score scheduling. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/85.md
41. 1p3a post #75: room allocation (set cover). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/75.md
42. 1p3a post #4: refund rules with examples; ski problem input format. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/4.md
43. 1p3a post #82: per-letter score aggregation. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/82.md
44. 1p3a post #90: card pile N-consecutive problem. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/90.md
45. 1p3a post #17: three-number arithmetic combinations with parentheses follow-up. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/17.md
46. 1p3a post #81: LC 295 in a 45-minute coding round. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/81.md
47. 1p3a post #36: three LeetCode problems in 60 minutes (incl. Happy Number). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/36.md.md
48. 1p3a post #95: count of meal orders summing to budget. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/95.md
49. 1p3a post #68: menu exact-target combinations. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/68.md
50. "Fixed-Width Text Formatting (Multi-Column Follow-up)" problem statement. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/Fixed-Width%20Text%20Formatting%20(Multi-Column%20Follow-up)/qs.md
51. "Fill Layover Hours by Booking Experiences" problem statement. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/Fill%20Layover%20Hours%20by%20Booking%20Experiences/qs.md
52. "pagination" problem statement (host_id, listing_id, score, city). FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/pagination/qs.md
53. "Calculate Board Score" problem statement. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/Calculate%20Board%20Score/qs.md
54. "guess word" (guess-number variants). FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/guess%20word/qs.md
55. "perm_qs 1" (smallest permutation >= lower bound). FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/perm_qs%201/qs.md
56. 1p3a post #56: menu DP, with the follow-up to optimize for the at-most-3-items constraint. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/56.md
57. "buddy list" follow-up statement. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/buddy%20list/qs.md
58. "car speed" (socket-based car speed controller) problem statement. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/car%20speed/qs.md
59. 1p3a post #13: Beijing loop, two coding rounds (topological sort; split stay with bitmask bonus), three-PR code review, extra design round. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/13.md
60. 1p3a post #21: Course Schedule-style topological sort; three-PR code review in Python. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/21.md
61. 1p3a post #3: board-game / maze-score phone screen; ski problem with multiple-skier follow-up; code review (updated 2026-03). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/3.md
62. 1p3a post #6: bank-account phone screen; menu min cost with DP plus bitmask and listing the combination. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/6.md
63. 1p3a post #16: SWE loop Oct-Dec 2025; bank account with time queries; menu combo with combination-listing follow-up. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/16.md
64. 1p3a post #7: three-number arithmetic; weighted-graph distances; bank account; water pouring; smallest number. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/7.md
65. 1p3a post #25: Collatz longest steps. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/25.md
66. 1p3a post #28: 2025 loop; accommodation segment allocation; code review. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/28.md
67. 1p3a post #29: job sequencing with deadlines (phone screen, worked example). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/29.md
68. 1p3a post #31: Minimum Menu Order Cost II; max ski score on a height grid (updated 2025-09). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/31.md
69. 1p3a post #39: guess the four-digit number with test cases. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/39.md
70. 1p3a post #42: wallet class with a self-written test harness. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/42.md
71. 1p3a mirror file #49: Maximum Product Subarray (no interview context). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/49.md
72. 1p3a post #58: Product of Array Except Self; weak code review led to an extra coding round. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/58.md
73. 1p3a mirror file #60: Connect 4 static-board result (win, draw, ongoing). https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/60.md
74. 1p3a post #103: Connect 4 static-board check; candidate did not run code. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/103.md
75. 1p3a post #63: `retry(func, max_retries)` signature. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/63.md
76. 1p3a post #67: column-drop game with a memoization follow-up; three-PR code review. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/67.md
77. 1p3a post #71: setSum/getValue/setValue coding; code review feedback. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/71.md
78. 1p3a post #73: senior booking-team loop; float Combination Sum phone screen; Retryer onsite. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/73.md
79. 1p3a post #76: two-hotel interval coverage; ski scoring. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/76.md
80. "Resolve Battles" (Diplomacy) problem statement. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/Resolve%20Battles/qs.md
81. "file_creation" problem statement with the `watch` follow-up. FlowSight/coding. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/file_creation/qs.md
82. 1p3a post #11: senior IC loop; task scheduling and get/set/getSum coding rounds. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/11.md
