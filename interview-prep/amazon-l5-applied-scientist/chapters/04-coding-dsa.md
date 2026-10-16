# 04. Coding Round for Applied Scientists: DSA Questions Reported at Amazon

> Scope: the live-coding (data structures and algorithms) round in the Amazon L5 Applied Scientist (Applied Scientist II) onsite or virtual loop, 2025-2026. Questions only, with no solutions. Tags: **[Reported]** means a candidate report or prep source ties the question to Amazon (citation [n]). Where the report comes from an SDE, intern or Data Scientist loop, or the source does not name the role, the line says so. **[Likely]** means a reasoned guess based on the role, team or Amazon's products. No source reported it.
>
> Research note: in this environment the egress proxy blocked Glassdoor, Blind, LeetCode, 1point3acres, Medium, amazon.jobs and the vendor guides, so those sources were read through search-result extracts and not as full pages. The GitHub-hosted sources (the TrueInterview mirrors and ombharatiya question lists, marked "fetched and read in full" in the References) were fetched and read in full. Treat the details of each report as what the candidate said happened, not as verified fact. ML-from-scratch coding (AUC, IoU, k-means, k-NN, attention, softmax, gradient descent and similar) is covered in Chapter 05 and is not repeated here.

## Format notes for this round

- **Where coding appears.** Amazon's official AS prep page says a technical phone screen comes first, and some teams add a second one [1]. Candidate reports describe the L5 loop as 4 to 6 back-to-back 60-minute interviews: ML breadth, ML depth, live coding, science application or ML system design, and a Bar Raiser [2][3][4][5]. An L5 poster said their recruiter described **one coding interview, usually a LeetCode medium**, with no coding in the other rounds [4]. One Seattle 2026 report listed "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [6]. A 2026 aggregator of scientist reports says coding appears in one or two loop rounds, sometimes the hiring manager's, and shares the hour with Leadership Principle (LP) questions [7].
- **Difficulty.** Most reports say easy to medium, and often an "easier medium" [4][8][9][10]. Some reports say medium/hard, with "clean production level, optimal code" and edge cases expected [11]. Others got one medium and one hard, with brute force explained before the optimal solution [12], or a LeetCode Medium/Hard such as median of a data stream [13]. One April 2026 AS phone screen set an original dynamic program that was not on any standard list [7][14]. Some AS candidates say they got almost no LeetCode at all (one poster: "roughly 4/5 of the AS I asked") [4][9]. **The bar depends on the team and on who interviews you.** A Reddit commenter said an SDE interviewer usually gives medium DSA, while an AS interviewer often gives ML or data-science coding [2].
- **Time pressure.** LP questions often take the first 20 to 30 minutes of a technical hour [15][16][10]. One 2026 report says LP questions left about 15 minutes for the coding task [7]. Plan to finish a medium in about 25 minutes [10].
- **Environment.** Amazon's prep pages ask for syntactically correct code with no pseudo code, written in its Livecode shared editor with no IDE, and they say to check edge cases and bad input [1][17]. Reports say the editor does not run code, so you trace through it by hand [18][10]. Python is the norm for AS candidates, but any language is accepted [19]. The 2026 SDE online assessment added an AI-assisted repository-debugging section in some regions. Live coding rounds still ban AI, and no source reports the AI-assisted section for AS loops [20].
- **What is assessed at L5.** Problem framing and clarifying questions, choice of data structure, stating time and space complexity, moving from brute force to optimal, testing and edge cases, and readable, maintainable code [17][19][21][18]. Scientist interviewers also want edge cases and complexity stated without being asked, and they want you to narrate decisions as your own: one candidate who solved course schedule was told too many hints had been given [7][10].
- **Prep signals from candidates.** Candidates say Blind 75 or 150 at medium level is usually enough, and that you should know arrays, hash maps, trees, graphs, heaps and basic DP well [22][23][24]. L5 phone screens mix coding, statistics and LPs [25]. Alexa, AGI and LLM teams lean toward ML-implementation coding [26][27]. Amazon AGI phone screens were reported as "standard LC coding plus a backend task" [28]. Amazon SDE-tagged frequency lists (islands, LRU, top-k, intervals) are useful backup practice, because AS coding interviewers are often SDEs [29][30][31][32][20].
- **Reports too vague to list as questions.** Several AS reports name only the topic: "implement a data structure from scratch" (4 or 5 small prompts) [2], a sliding-window variant with "rigorous follow-ups" [9], a standard tree medium followed by Transformer questions [2], a DFS exercise and a binary-search question in Senior AS loops [33], a DP problem from the Blind 150 list asked by a hiring manager [2][34], two string DP and hash problems [35], and simple Python data wrangling or a medium SQL question [3][11].
- **Common follow-ups.** Scaling to data that does not fit on one machine (MapReduce) [36], streaming versions [14], and "now make it production-grade" [11].

## A. Arrays, strings and hashing

1. [Reported] Given an array of integers and a target, return the indices of the two numbers that add up to the target. AS reports use it as a warm-up before a harder problem [34][7]. A 2026 onsite used an in-flight-movie framing: pick two movie lengths that fit the flight time [37].
   - What changes if the array is sorted, or if there are many queries against the same array?
   - How would you return all unique pairs without duplicates?
   - How would you solve it at scale, when the array does not fit in memory? [14]
2. [Reported] Find the maximum subarray sum in an integer array. [11][37]
   - How would you also return the start and end indices?
   - What if the array is circular?
3. [Reported] Compress a string in place by run-length encoding (for example "aabccc" becomes "a2bc3"), and analyze time and space complexity. [6][5]
   - How do you handle counts of 10 or more?
   - What if the compressed string would be longer than the original?
4. [Reported] For each position in a list, return the product of all other elements. A fraud-detection team screen asked this without allowing division. It was also reported in an April 2026 Amazon phone screen (role not stated). [38][37]
   - How do you handle zeros?
   - Can you do it in O(1) extra space, not counting the output?
5. [Reported] Find the longest substring without repeating characters. [5][19][37]
   - How would you return the substring itself, not just its length?
6. [Reported] Divide two integers without using multiplication, division or modulo. [19][14]
   - How do you handle overflow and negative numbers?
7. [Reported] In a sorted array where every element appears twice except one, find the single element. [19][14]
   - Can you beat O(n)?
8. [Reported] For each element, find the nearest smaller element to its left. This was reported in an Amazon loop, but the role is not specified. [39]
   - Which data structure gives you amortized O(n)?
9. [Reported] Merge two sorted arrays in place when the first array has spare capacity at the end. A third-party guide lists it as a sample AS question, and it was reported at Amazon in February 2026. [40][14]
   - What changes if you must merge three sorted arrays and remove duplicates? [14]
10. [Reported] A hash-map counting problem, followed by: what if the data is too big for one machine? [36]
    - How would you split it into map and reduce stages?
    - How do you handle a skewed key that overloads one reducer?
11. [Reported] Rotate an n x n matrix by 90 degrees in place. Reported in an AS loop (2024) and in Amazon screens and onsites through May 2026. [41][14][37]
    - How do you rotate counterclockwise, or by 180 degrees, with the same code?
12. [Reported] Count the subarrays whose sum equals k. This is on an Amazon-tagged list. [20]
    - Why does a sliding window fail when values can be negative?
13. [Reported] Group anagrams. Reported at Amazon in an August 2026 report and on Amazon-tagged lists (role not stated). [14][20][37]
    - How does your key choice change if the input is Unicode?
14. [Reported] Find all triplets that sum to zero (3Sum). This is on an Amazon-tagged list. [20]
    - How do you skip duplicate triplets without a set?
15. [Likely] Given customer reviews as strings, return the k most frequent words, ignoring stop words.
    - How do you break ties?
    - What if the vocabulary is far larger than k?
16. [Likely] Check whether two strings are one edit apart (insert, delete or replace).
    - How would you generalize this to a full edit-distance threshold k?

## B. Intervals and sorting

17. [Reported] Merge overlapping intervals. Reported in AS loops (Sunnyvale 2020, a June 2026 AS technical screen) and as the coding question in a scientist online assessment. [2][42][43][7][37]
    - Should touching intervals such as [1,3] and [3,5] merge? How does your code change if they should not?
    - How would you insert one new interval into an already merged list?
18. [Reported] Merge overlapping time ranges for room reservations given as epoch-second pairs. This was tagged Amazon technical screen. [43]
    - How do you handle intervals where start equals end?
19. [Reported] Given meeting time intervals, find the minimum number of rooms needed. A scientist loop asked it with a min-heap after a two-sum warm-up. [7][37]
    - Can you do it with a sweep line and no heap?
    - How would you also return which meeting goes to which room?
20. [Reported] Given process start and end times, find the maximum number of processes running at once. Reported at Amazon in July 2026 (role not stated). [14]
    - How would you also return the time window when the peak occurred?
21. [Reported] Support adding non-overlapping intervals and querying whether a point is covered. This was reported at Amazon (role not stated). [14]
    - Which structure keeps both operations at O(log n)?
22. [Reported] Given a list of calendar events, find all pairs that conflict. Reported at Amazon in June 2026 (role not stated). [14]
23. [Likely] Sort a large file of (customer_id, timestamp) records that does not fit in memory.
    - How many passes does external merge sort need, given memory size M?

## C. Two pointers and sliding window

24. [Reported] Design a counter that returns the number of clicks on an ad in the last k minutes. Reported in Amazon ML Engineer and scientist screens; the expected structure was a deque plus a hash map. [7][10][14]
    - How does memory grow with k and with the number of ads?
    - What changes if events can arrive out of order?
25. [Reported] Find the k closest elements to a target x in a sorted array. [44]
    - Can you do it in O(log n + k)?
26. [Reported] Return the maximum of every window of size k (sliding window maximum). Reported at Amazon in June 2026 and on Amazon-tagged lists. [14][20]
    - Why does a monotonic deque give O(n)?
27. [Reported] Find the shortest substring of a string that contains all characters of a pattern (minimum window substring). This is on an Amazon-tagged list. [20]
28. [Reported] Find the longest substring you can get by replacing at most k characters so that all characters are the same. This is on an Amazon-tagged list. [20]
29. [Reported] Trapping rain water. Reported in an April 2026 Amazon phone screen and on Amazon-tagged lists. [45][20][37]
    - Can you do it in O(1) extra space?
30. [Likely] Given daily sales, find the longest run of days whose total is at most a budget B, assuming all values are non-negative.
    - What breaks if negative values are allowed?

## D. Linked lists and stacks

31. [Reported] Swap nodes in pairs (pairwise swap) in a linked list. A Research/Applied Scientist phone screen asked for this as a medium problem in about 25 to 30 minutes. [46][47]
    - Can you write both the recursive and iterative versions?
    - How would you generalize it to reverse every k nodes?
32. [Reported] Asteroid collision. Reported in an Applied Scientist intern DSA round (Bengaluru, 2026) and in an August 2026 Amazon report. [12][48][14]
    - Which edge cases involve equal sizes or asteroids all moving the same way?
33. [Reported] Check whether a string of brackets is valid. Reported in Amazon ML Engineer phone screens in 2026. [10][37]
    - How would you return the minimum number of insertions that make it valid?
34. [Reported] Evaluate an arithmetic expression. A July 2026 Amazon report used postfix notation. [14]
    - How would you extend it to infix with +, -, *, / and parentheses?
35. [Reported] Merge k sorted lists. This is on Amazon-tagged lists. [45][20]
    - What is the complexity with a heap compared with pairwise merging?
36. [Reported] Merge k sorted event streams behind an iterator with next() and hasNext(), without loading all events into memory. Reported in a September 2026 Amazon onsite. [14][20]
    - What happens when one stream is much slower than the others?
37. [Reported] Detect a cycle in a linked list, return where it starts, and break it. Reported at Amazon in June 2026. [14]
38. [Reported] Copy a linked list where each node has a next pointer and a random pointer. Reported at Amazon in March 2026. [14]
    - Can you do it in O(1) extra space?
39. [Reported] Implement a stack that pops the most frequent element, breaking ties by most recent (maximum frequency stack). Reported at Amazon in September 2026. [14]

## E. Trees and BSTs

40. [Reported] Given an org tree with salaries, compute the average salary of each manager's direct reports. Reported in Amazon scientist and ML Engineer loops in 2026. [7][10][14]
    - How would you compute the average over all reports, not just direct ones, in one pass?
    - What if the input is a flat list of (employee, manager, salary) rows?
41. [Reported] Serialize and deserialize a binary tree. Reported at Amazon in February 2026 and on Amazon-tagged lists. [14][20]
    - How does your format handle null children compactly?
42. [Reported] Find the lowest common ancestor of two nodes in a binary tree. Reported in a February 2026 Amazon phone screen. [20][37]
    - What if one node might not be in the tree?
    - How does it get easier in a BST?
43. [Reported] Validate a binary search tree. Reported at Amazon in June 2026. [14]
44. [Reported] Find the kth smallest element in a BST. An August 2026 report included follow-ups. [14]
    - What if the tree is modified often and you need kth smallest many times?
45. [Reported] Return the right-side view of a binary tree. Reported at Amazon in February 2026. [14]
46. [Reported] Return the vertical order traversal of a binary tree. Reported at Amazon in June 2026. [14]
47. [Reported] Build a binary tree from a list of (parent, child, isLeft) descriptions and return its root. Reported at Amazon in August 2026. [14]
48. [Likely] Given a product category tree (Electronics > Audio > Headphones), return the path from the root to a category and the depth of the deepest leaf.

## F. Graphs and grids

49. [Reported] Count the number of islands in a 2D grid. Reported in AS loops in 2020, 2025 and 2026, in a London 2026 screen, and in Amazon screens with a streaming follow-up in April 2026. [2][12][49][50][14][37]
    - Should diagonals count as connected?
    - What if the grid is streamed row by row and does not fit in memory?
    - Can you solve it with union-find?
50. [Reported] A BFS problem on a grid (for example, the minimum number of steps from start to target). [51]
    - How do you reconstruct the path?
51. [Reported] Shortest path with Dijkstra's algorithm where the grid or graph has obstacles. A 2026 source says weighted shortest-path problems are rising at Amazon. [52][20]
    - Why do negative edge weights break Dijkstra?
    - When would A* be the better choice?
52. [Reported] Explain how you would implement Dijkstra: which data structures, and what runtime? [53]
    - Which data structure gives O((V+E) log V), and which gives O(V^2)?
53. [Reported] Course schedule: given prerequisite pairs, return a valid order or report a cycle. Reported in 2026 Amazon ML Engineer screens, including a version framed around ads. [10][14][37]
    - How do you return all nodes that are part of a cycle?
54. [Reported] Given dependent tasks and a task that failed, order the tasks and find which tasks must be rerun. Reported in a September 2026 Amazon onsite. [14]
55. [Reported] Check whether adding an edge to a directed graph creates a cycle. Reported at Amazon in October 2025. [14]
56. [Reported] Word ladder (BFS on implicit graph). This is on Amazon-tagged lists. [18][20]
    - How do you build neighbors efficiently using wildcard patterns?
57. [Reported] Rotting oranges and multi-source BFS: how many minutes until every reachable cell is rotten? Reported at Amazon in April 2026 and on Amazon-tagged lists. [14][20]
58. [Reported] Search a grid of letters for a word, then for a list of words with a trie (word search I and II). Reported at Amazon in January and February 2026. [14][20]
59. [Reported] Given equations such as a / b = 2.0, answer division queries (evaluate division). This is on an Amazon-tagged list. [20]
60. [Reported] Count connected components with union-find. Reported at Amazon in May 2026 (role not stated). [14]
    - How would you use it to group users who share a payment method or address for fraud detection?
61. [Likely] Clone a graph, for example a knowledge-graph subgraph used by Alexa.
62. [Likely] Given warehouse aisles as a grid with blocked cells, find the shortest route for a robot from a start to multiple pick locations. This is related to Amazon Robotics.
    - How does the problem change with 2 pick locations compared with k?

## G. Heaps, top-k and streaming

63. [Reported] Find the median of a data stream. Reported as a LeetCode Medium/Hard in an AS loop (Luxembourg, 2023) and at Amazon in January 2026. [13][14]
    - What if most values are small integers in a known range?
    - How would you support a sliding-window median?
64. [Reported] Top K frequent elements. A September 2026 Amazon report added a streaming follow-up. [18][45][14]
    - Can you compare heap and bucket sort?
    - How would you approximate this with Count-Min Sketch when there are billions of items?
65. [Reported] K closest points to the origin. This is Amazon SDE-tagged. [45]
    - Can you use quickselect, and what is its worst case?
66. [Reported] Find the kth largest element in an unsorted array. This is on an Amazon-tagged list. [20]
67. [Reported] Rearrange a string so that no two adjacent characters are the same, or report that it is impossible. Reported at Amazon in May 2026. [14][20]
68. [Reported] Given tasks and a cooldown n between identical tasks, find the minimum time to run them all (task scheduler). This is on an Amazon-tagged list. [20]
69. [Likely] Return the top k products by click-through rate in a sliding one-hour window over a click stream.
    - What state must you keep per product?

## H. Binary search

70. [Reported] Find the minimum of a rotated sorted array. Reported in Amazon scientist and ML Engineer phone screens. [7][10]
    - How do duplicates change the worst case?
    - How would you then search for a target in the same array?
71. [Reported] Search a 2D matrix whose rows and columns are both sorted, and find the optimal approach. [5]
    - Can you compare the staircase search with binary search on each row?
72. [Reported] Implement binary search and state its time complexity. [40]
    - Can you write it to find the first and the last occurrence of a value? [20]
73. [Reported] Koko eating bananas: find the minimum eating speed to finish within h hours. Reported in an April 2026 Amazon phone screen. [14][37]
    - How would you reframe it as "ship packages within D days"?
74. [Reported] Design a key-value store that keeps a time-stamped history of values and returns the value at a given time. Reported in a March 2026 Amazon onsite and a September 2026 phone screen. [14][37]
    - How would you use it to look up point-in-time feature values for training data?
75. [Reported] Find the kth smallest element in two sorted arrays. This was reported at Amazon (role not stated). [14]
76. [Likely] Given a monotonic model-score threshold function, find the smallest threshold that keeps precision at or above a target. This is binary search on the answer.

## I. Dynamic programming

77. [Reported] Batch items into buckets so that total padding cost is as low as possible. This original DP was reported in an April 2026 Applied Scientist phone screen. [7][10][14]
    - What is your DP state, and what is the recurrence?
    - How does the answer change if the number of buckets is fixed?
78. [Reported] A minimum-coins variant solved with 1-D DP (AS intern, 2025), also reported at Amazon as "minimum bills and coins" in June 2026. [51][14][20]
    - Can you return the actual coins used, not just the count?
    - How do you count the number of ways instead?
79. [Reported] Edit distance between two strings, with a follow-up variant (July 2026). [14]
    - Can you reduce memory to O(min(m, n))?
80. [Reported] Word break: can a string be split into dictionary words? A July 2026 report gave a dictionary helper function. [14][20]
    - How does a trie help?
    - How would you return all valid segmentations?
81. [Reported] House robber and its variants (circular street, binary tree). Reported in Amazon screens and onsites in November 2025. [14][37]
82. [Reported] Best time to buy and sell stock, including the versions with unlimited trades and at most k trades. Reported at Amazon in March and July 2026. [14][20][37]
83. [Reported] Decode ways. This is on an Amazon-tagged list. [20]
84. [Reported] Find the longest palindromic substring. This is on an Amazon-tagged list. [20]
85. [Reported] Regular expression matching with . and *. This was reported at Amazon (role not stated). [14]
86. [Likely] Longest increasing subsequence.
    - Can you do it in O(n log n)?
87. [Likely] Viterbi-style DP: given emission and transition scores, find the highest-scoring tag sequence.

## J. Randomization and sampling

88. [Reported] Random pick with weight: build a weighted random number generator from a probability array. Reported in AS and Senior AS screens in 2026. [33][5][54][7][10]
    - How do you handle all-zero weights or a single element?
    - What does the alias method give you, and what does it cost?
89. [Reported] Design a set with insert, delete and getRandom in O(1). This is on an Amazon-tagged list. [20]
    - How do you support duplicates?
90. [Likely] Reservoir sampling: choose k items uniformly from a stream of unknown length.
    - What changes for weighted reservoir sampling?
91. [Likely] Shuffle an array uniformly at random (Fisher-Yates), and explain why the naive swap-with-any-index version is biased.

## K. Design-style data structures

92. [Reported] Implement an LRU cache. Reported in Amazon loops, including a basic-DS AS-adjacent report and an April 2026 phone screen. [49][18][37]
    - Can every operation be O(1)?
    - How would you make it thread-safe?
    - What about LFU, or TTL-based eviction? [20]
93. [Reported] Implement a trie that supports adding words and searching with a wildcard character. This was reported at Amazon, and autocomplete design is on an Amazon-tagged list. [14][20]
    - How would you return the top-3 completions by query frequency?
94. [Reported] Given a stream of (timestamp, message), decide whether each message may be printed, so that the same message prints at most once every 10 seconds. This was reported for Amazon new grads in August 2026. [20]
95. [Likely] Implement a min-stack or a max-queue that supports O(1) min or max.

## L. Practical data-processing algorithms

96. [Reported] Given order events and ad-impression events, find for each user the pair with the smallest time difference. Reported at Amazon in February 2026. [14]
    - What is the complexity if each list is sorted by time?
97. [Reported] Given an Audible reading-event stream, flag books that receive abusive activity. Reported at Amazon in February 2026. [14]
98. [Reported] Parse log lines and aggregate them by a key (group-by). Reported at Amazon in March 2026. [14]
    - What if the logs do not fit in memory?
99. [Reported] Given send and receive CSV logs, compute each message's latency. Reported at Amazon in May 2026. [14]
    - How do you handle a receive with no matching send?
100. [Reported] Return the top k slowest requests from a set of logs. This was reported at Amazon (role not stated). [14]
101. [Likely] Given a log of (user_id, session_start, session_end), compute each user's total active time, merging overlapping sessions.

## Deepening pass (October 2026): newly reported questions

> This pass used five web searches, and only the search-result extracts could be read. Most hits were Glassdoor AS reviews that confirm questions already in the chapter: number of islands in a 2026 LLM-focused AS loop, string compression in a 2026 AS II loop, and median from a data stream. They add no new questions. A December 2025 Senior AS tech screen on PracHub involved a graph problem the candidate did not finish, but the extract does not name the problem [56]. The three new reported items below come from an Amazon **L5 virtual onsite** coding report. The role is most likely SDE, not AS, but AS coding interviewers are often SDEs.

102. [Reported] Generate all subsets of a set of distinct integers (power set). Reported in an Amazon L5 virtual onsite (USA, remote; role not stated as AS). [55]
    - How would you change it if the input contains duplicates?
    - Can you write both the backtracking and the bitmask versions, and give the complexity of each?
103. [Reported] Convert a sorted array to a height-balanced binary search tree, then answer follow-ups on tree traversals. Reported in an Amazon L5 virtual onsite (role not stated as AS). [55]
    - How would you do the same for a sorted linked list without converting it to an array?
    - Can you write the in-order traversal iteratively, and explain why it returns the original array?
104. [Reported] Given birth and death years for a list of people, find the earliest year with the maximum population. Reported in an Amazon L5 virtual onsite (role not stated as AS). [55]
    - Can you do it with a difference array or sweep line instead of checking every year?
    - What changes if the year range is very large and the list is sparse?

### Gap-closing [Likely] questions

105. [Likely] Compute the dot product of two very sparse vectors, such as TF-IDF or one-hot feature vectors.
    - Which representation would you choose when one vector is much sparser than the other?
106. [Likely] Multiply two sparse matrices, for example a user-item matrix by an item-embedding matrix.
    - How does the cost depend on the number of non-zeros instead of the matrix dimensions?
107. [Likely] Intersect k sorted posting lists of document IDs from a product-search inverted index.
    - When would you use galloping (exponential) search instead of a linear merge?
108. [Likely] Given a list of candidate feature names, return all combinations of size k whose total cost stays within a budget (a combination-sum or backtracking variant).
    - How do you prune the search early?
109. [Likely] Find the longest common subsequence of two users' browsing sessions, given as sequences of product IDs.
    - How would you reconstruct the subsequence itself?
110. [Likely] Given each employee's busy intervals, return the common free time across all employees.
    - How does a heap over the per-employee lists compare with flattening and sorting everything?
111. [Likely] Find the kth smallest value in an n x n matrix whose rows and columns are sorted, such as a matrix of pairwise scores.
    - Can you compare the heap approach with binary search on the value range?
112. [Likely] Implement a binary min-heap from scratch with push, pop and decrease-key.
    - Why does decrease-key need a position index, and how does that help Dijkstra?
113. [Likely] In an array where every element appears three times except one, find the single element in O(n) time and O(1) extra space.
    - How would you generalize it to "every element appears k times"?
114. [Likely] Given a metric time series, flag every index whose value is more than three standard deviations from the rolling mean of the previous w points, in O(n) overall.
    - How do you keep the rolling variance numerically stable?

## References

1. Applied Scientist Interview Prep, amazon.jobs (official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Amazon Applied Scientist Interview Experience & Questions, Glassdoor (2020-2026 reviews). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
3. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
4. Amazon Applied Scientist (L5) Virtual Onsite, Blind. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
5. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford. https://dataford.io/interview-guides/amazon/applied-scientist
6. Glassdoor review RVW91266671, Amazon Applied Scientist, Seattle (2026; string compression). https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW91266671.htm
7. Amazon Research Scientist Interview Process, TrueInterview (CC BY 4.0 mirror on GitHub, kevin-2023-code/Tech-Interview-Questions; covers Research and Applied Scientist loops), 2026; fetched and read in full. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
8. Amazon applied scientist loop, Blind. https://www.teamblind.com/post/amazon-applied-scientist-loop-odlzgrp3
9. Applied scientist interview Amazon, Blind. https://www.teamblind.com/post/applied-scientist-interview-amazon-4udsobcp
10. Amazon Machine Learning Engineer Interview Process, TrueInterview (GitHub mirror, kevin-2023-code/Tech-Interview-Questions; includes applied-science loops), 2026; fetched and read in full. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
11. Tips for coding interview, Applied Scientist at Amazon, Blind. https://www.teamblind.com/post/tips-for-coding-interview-applied-scientist-at-amazon-txexmtsb
12. Amazon Applied Scientist Intern Interview Experience & Questions, Glassdoor (includes 2026 Bengaluru and New Delhi reports). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Intern-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
13. Glassdoor review RVW99133644, Amazon Applied Scientist (Luxembourg, 2023; median from data stream). https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
14. Amazon: every question reported (320 questions, with round and report date; role usually not stated), TrueInterview (GitHub mirror, kevin-2023-code/Tech-Interview-Questions), synced October 2026; fetched and read in full. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
15. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (commercial prep service). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
16. Senior Engineer's Guide to Amazon Interviews + Questions, interviewing.io. https://interviewing.io/guides/hiring-process/amazon
17. SDE II Interview Prep, amazon.jobs (official). https://amazon.jobs/content/en/how-we-hire/sde-ii-interview-prep
18. Amazon Coding Interview Questions: The Real Bar (2026 Guide), copilotinterview.com. https://copilotinterview.com/blog/amazon-coding-interview-questions
19. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
20. FAANG Recent Questions: Amazon section (Amazon-tagged LeetCode lists, 2025-2026 trends and OA reports; mostly SDE), ombharatiya/FAANG-Coding-Interview-Questions, GitHub, last verified October 2026; fetched and read in full. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/FAANG-Recent-Questions.md
21. Amazon Applied Scientist Interview: A Complete Guide, codinginterview.com. https://www.codinginterview.com/guide/amazon-applied-scientist-interview/
22. Amazon applied scientist interview-onsite, Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
23. Amazon Applied Scientist interview discussion, 1point3acres. https://www.1point3acres.com/bbs/thread-655973-1-1.html
24. My Amazon Applied Science Interview Journey, Medium. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
25. Amazon L5 Applied Scientist Fulltime Tech Phone Screen Overview, 1point3acres. https://www.1point3acres.com/interview/thread/1170176
26. Amazon Applied Scientist Interview Guide, Alexa International Tech, Medium. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
27. Amazon AGI Info Applied Scientist Interview Experience and Insights, 1point3acres. https://www.1point3acres.com/interview/thread/1130091
28. AI Companies Interview Questions: Amazon AGI section, ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator; underlying reports not named), 2026; fetched and read in full. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
29. Amazon recruiters share 6 job interview tips for software development engineers, aboutamazon.com. https://www.aboutamazon.com/news/workplace/amazon-job-interview-software-development-engineer-recruiter-tips
30. Amazon SDE-1/2 interview questions 2025, LeetCode Discuss. https://leetcode.com/discuss/post/6752237/amazon-interview-questions-2025-by-anony-qgzt/
31. 14 Most Popular Amazon Coding Interview Questions, Design Gurus. https://www.designgurus.io/blog/amazon-14-question
32. 30 Amazon LeetCode Interview Questions for 2026, Verve Copilot. https://www.vervecopilot.com/blog/amazon-leetcode-interview-questions
33. Glassdoor review RVW102577344, Amazon Senior Applied Scientist, Seattle (2026; weighted random number generator), plus related Senior AS reports. https://www.glassdoor.co.nz/Interview/Amazon-Interview-E6036-RVW102577344.htm
34. Glassdoor review RVW47917252, Amazon Applied Scientist (2020-2022 era; TwoSum, DP, ML classifier). https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW47917252.htm
35. Amazon Applied Scientist Intern Interview 2027: Coding, ML Depth, and Behavioral Questions, PracHub. https://prachub.com/resources/amazon-applied-scientist-intern-interview-2027-coding-ml-depth-and-behavioral-questions
36. Amazon Applied Scientist phone screen (hashmap plus MapReduce follow-up), 1point3acres (2020). https://www.1point3acres.com/bbs/thread-731301-1-1.html
37. Amazon interview questions, free list with round and month reported, TrueInterview (GitHub mirror, kevin-2023-code/Tech-Interview-Questions), 2026; fetched and read in full. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/README.md
38. Amazon applied scientist phone screen (product of other elements), 1point3acres. https://www.1point3acres.com/bbs/interview/amazon-data-science-506632.html
39. Amazon interview questions from phone screens (nearest smaller element to the left), Glassdoor review RVW67853279. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW67853279.htm
40. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
41. Glassdoor review RVW89381532, Amazon Applied Scientist (2024; rotate matrix 90 degrees). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW89381532.htm
42. Amazon applied scientist phone interview (merge intervals), 1point3acres (2020). https://www.1point3acres.com/bbs/interview/amazon-machine-learning-677329.html
43. Merge Overlapping Time Ranges (Amazon Applied Scientist technical screen, 2026), PracHub. https://prachub.com/interview-questions/merge-overlapping-time-ranges
44. Amazon Interview Experience for Applied Scientist Internship, GeeksforGeeks (updated 2024). https://www.geeksforgeeks.org/interview-experiences/amazon-interview-experience-for-applied-scientist-internship/
45. Top 100 LeetCode Problems for MAANG India 2025: Frequency-Ranked, Impacteers. https://futurejobs.impacteers.com/blogs/top-100-leetcode-problems-for-maang-interviews-frequency-ranked-with-patterns-and-solutions-india-2025
46. Amazon Research-Applied Scientist position, LeetCode Discuss. https://leetcode.com/discuss/interview-experience/444789/amazon-research-applied-scientist-position/
47. Amazon | Applied Scientist | Phone Screen | USA, LeetCode Discuss. https://leetcode.com/discuss/interview-experience/985312/amazon-applied-scientist-phone-screen-usa/
48. Asteroid Collision (Amazon-tagged community question), Hello Interview. https://www.hellointerview.com/community/questions/asteroid-collision/cm5eguhac02kt838o4qw0x2kf
49. Amazon Interview Question: Basic DS questions: LRU cache, trees, count the islands, Glassdoor. https://www.glassdoor.com/Interview/Basic-DS-questions-LRU-cache-trees-count-the-islands-QTN_4447065.htm
50. Applied scientist intern @Amazon (count islands), 1point3acres (2020). https://www.1point3acres.com/bbs/interview/amazon-machine-learning-613476.html
51. My Amazon Applied Scientist Intern Interview Experience (Off-Campus), Medium (2025). https://medium.com/@pranavpandeyghq123/my-amazon-applied-scientist-intern-interview-experience-off-campus-7163c5b7c9ae
52. Amazon new grad Applied Scientist interview experience: five virtual onsite rounds, L4 offer, PracHub (2025). https://prachub.com/interview-experiences/amazon-new-grad-applied-scientist-interview-experience-five-virtual-onsite-rounds-mostly-behavioral-end-in-an-l4-offer
53. Glassdoor review RVW74547087, Amazon (Luxembourg, 2022; implementing Dijkstra). https://www.glassdoor.com.au/Interview/Amazon-Interview-E6036-RVW74547087.htm
54. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
55. Amazon | L5 | Virtual Onsite - Coding questions | USA (remote), LeetCode Discuss (subsets, sorted array to BST, maximum population year, meeting rooms II variant; role not stated, likely SDE); read from search extract only. https://leetcode.com/discuss/interview-question/1921047/amazon-l5-virtual-onsite-coding-questions-usa-remote
56. Amazon Senior Applied Scientist Interview Experience: Passed the Tech Screen Despite an Unfinished Graph Question, PracHub (December 2025); read from search extract only. https://prachub.com/interview-experiences/amazon-senior-applied-scientist-interview-experience-passed-the-tech-screen-despite-an-unfinished-graph-question
