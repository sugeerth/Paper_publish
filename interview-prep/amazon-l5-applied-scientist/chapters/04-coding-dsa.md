# 04. Coding Round for Applied Scientists: DSA Questions Reported at Amazon

> Scope: the live-coding (data structures and algorithms) round in the Amazon L5 Applied Scientist (Applied Scientist II) onsite or virtual loop, 2025-2026. Questions only, with no solutions. Tags: **[Reported]** means a candidate report or prep source ties the question to Amazon (citation [n]). Where the report comes from an SDE, intern or Data Scientist loop and not an AS loop, the line says so. **[Likely]** means a reasoned guess based on the role, team or Amazon's products. No source reported it.
>
> Research note: page fetching was blocked in this environment, so sources were read through search-result extracts and not as full pages. Treat the details of each report as what the candidate said happened, not as verified fact.

## Format notes for this round

- **Where coding appears.** Amazon's official AS prep page says a technical phone screen comes first, and some teams add a second one [1]. Candidate reports describe the L5 loop as 4 to 6 back-to-back 60-minute interviews: ML breadth, ML depth, live coding, science application or ML system design, and a Bar Raiser [5][6][18][41]. An L5 poster said their recruiter described **one coding interview, usually a LeetCode medium**, with no coding in the other rounds [18]. One Seattle 2026 report listed "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [9].
- **Difficulty.** Most reports say easy to medium, and often an "easier medium" [18][19][22]. Some reports say medium/hard, with "clean production level, optimal code" and edge cases expected [20]. Others got one medium and one hard, with brute force explained before the optimal solution [7], or a LeetCode Medium/Hard such as median of a data stream [12]. Some AS candidates say they got almost no LeetCode at all (one poster: "roughly 4/5 of the AS I asked") [18][22]. **The bar depends on the team and on who interviews you.** A Reddit commenter said an SDE interviewer usually gives medium DSA, while an AS interviewer often gives ML or data-science coding [5].
- **Environment.** Amazon's prep pages ask for syntactically correct code with no pseudo code, written in its Livecode shared editor with no IDE, and they say to check edge cases and bad input [1][2]. Third-party guides add that you usually cannot run the code, so you are expected to trace through it by hand [47]. Python is the norm for AS candidates, but any language is accepted [4].
- **What is assessed at L5.** Problem framing and clarifying questions, choice of data structure, stating time and space complexity, moving from brute force to optimal, testing and edge cases, and readable, maintainable code [2][4][46][47]. Expect 1 or 2 Leadership Principle (LP) questions to open or close the coding round. Reports describe 20 to 30 minutes of LPs inside technical rounds [45][48]. The ML-coding variant (for example AUC, k-means, IoU or k-NN from scratch) bans libraries to test whether you really understand the method [6][43][45][50].
- **Prep signals from candidates.** Candidates say Blind 75 or 150 at medium level is usually enough, and that you should know arrays, hash maps, trees, graphs, heaps and basic DP well [21][30][34]. L5 phone screens mix coding, statistics and LPs [31]. Alexa, AGI and LLM teams lean toward ML-implementation coding [35][55]. Amazon SDE-tagged frequency lists (islands, LRU, top-k, intervals) are useful backup practice, because AS coding interviewers are often SDEs [3][49][51][52].
- **Common follow-ups.** Scaling to data that does not fit on one machine (MapReduce) [26], streaming versions, and "now make it production-grade" [20].

## A. Arrays, strings and hashing

1. [Reported] Given an array of integers and a target, return the indices of the two numbers that add up to the target. [17]
   - What changes if the array is sorted, or if there are many queries against the same array?
   - How would you return all unique pairs without duplicates?
2. [Reported] Find the maximum subarray sum in an integer array. [20]
   - How would you also return the start and end indices?
   - What if the array is circular?
3. [Reported] Compress a string in place by run-length encoding (for example "aabccc" becomes "a2bc3"), and analyze time and space complexity. [9][41]
   - How do you handle counts of 10 or more?
   - What if the compressed string would be longer than the original?
4. [Reported] For each position in a list, return the product of all other elements. A fraud-detection team screen asked this without allowing division. [27]
   - How do you handle zeros?
   - Can you do it in O(1) extra space, not counting the output?
5. [Reported] Find the longest substring without repeating characters. [41][4]
   - How would you return the substring itself, not just its length?
6. [Reported] Divide two integers without using multiplication, division or modulo. [4]
   - How do you handle overflow and negative numbers?
7. [Reported] In a sorted array where every element appears twice except one, find the single element. [4]
   - Can you beat O(n)?
8. [Reported] For each element, find the nearest smaller element to its left. This was reported in an Amazon loop, but the role is not specified. [53]
   - Which data structure gives you amortized O(n)?
9. [Reported] Merge two sorted arrays. This appears as a sample AS coding question in a third-party guide. [44]
   - How would you merge in place if the first array has spare capacity at the end?
10. [Reported] Implement a data structure from scratch (one 2022 report says the round was 4 or 5 small "implement a data structure" prompts). [5]
    - Which operations must be O(1), and what trade-off does that force?
11. [Reported] A hash-map counting problem, followed by: what if the data is too big for one machine? [26]
    - How would you split it into map and reduce stages?
    - How do you handle a skewed key that overloads one reducer?
12. [Likely] Group anagrams from a list of product search queries.
    - How does your key choice change if the input is Unicode?
13. [Likely] Given customer reviews as strings, return the k most frequent words, ignoring stop words.
    - How do you break ties?
    - What if the vocabulary is far larger than k?
14. [Likely] Check whether two strings are one edit apart (insert, delete or replace).
    - How would you generalize this to a full edit-distance threshold k?

## B. Intervals and sorting

15. [Reported] Merge overlapping intervals. Reported in AS loops (Sunnyvale, 2020, and a June 2026 AS technical screen). [5][28][39]
    - Should touching intervals such as [1,3] and [3,5] merge? How does your code change if they should not?
    - How would you insert one new interval into an already merged list?
16. [Reported] Merge overlapping time ranges for room reservations given as epoch-second pairs. This was tagged Amazon technical screen. [39]
    - How do you handle intervals where start equals end?
17. [Likely] Given delivery time windows for drivers, find the minimum number of vans needed so that no two overlapping windows share a van.
    - Can you do it with a sweep line and no heap?
18. [Likely] Given Prime Video viewing sessions, find the peak number of concurrent viewers and when it happened.
19. [Likely] Sort a large file of (customer_id, timestamp) records that does not fit in memory.
    - How many passes does external merge sort need, given memory size M?

## C. Two pointers and sliding window

20. [Reported] A sliding-window variant with "rigorous follow-ups". [22]
    - What invariant does your window maintain?
21. [Reported] Find the k closest elements to a target x in a sorted array. [36]
    - Can you do it in O(log n + k)?
22. [Likely] Find the shortest substring of a string that contains all characters of a pattern (minimum window substring).
23. [Likely] Given daily sales, find the longest run of days whose total is at most a budget B, assuming all values are non-negative.
    - What breaks if negative values are allowed?
24. [Likely] Trapping rain water. This is Amazon SDE-tagged. [54]
    - Can you do it in O(1) extra space?

## D. Linked lists and stacks

25. [Reported] Swap nodes in pairs (pairwise swap) in a linked list. A Research/Applied Scientist phone screen asked for this as a medium problem in about 25 to 30 minutes. [23][24]
    - Can you write both the recursive and iterative versions?
    - How would you generalize it to reverse every k nodes?
26. [Reported] Asteroid collision. This was reported in an Applied Scientist intern DSA round (Bengaluru, 2026) and is also common in SDE loops. [7][56]
    - Which edge cases involve equal sizes or asteroids all moving the same way?
27. [Likely] Evaluate an arithmetic expression string with +, -, *, / and parentheses.
28. [Likely] Merge k sorted lists (for example, merging ranked result lists from k retrieval shards). This is Amazon SDE-tagged. [54]
    - What is the complexity with a heap compared with pairwise merging?
29. [Likely] Detect a cycle in a linked list and return the node where the cycle starts.

## E. Trees and BSTs

30. [Reported] A standard LeetCode medium on trees, followed by Transformer and long-context questions (AS, 2025). [5]
31. [Reported] An easy tree-traversal question. [41]
    - Can you do level-order, and iterative in-order without recursion?
32. [Reported] Basic DS questions covering trees, an LRU cache and counting islands. [14]
33. [Likely] Serialize and deserialize a binary tree. In an ML context, this could be a decision tree learned by a model.
    - How does your format handle null children compactly?
34. [Likely] Find the lowest common ancestor of two nodes in a binary tree, and then in a BST.
    - What if one node might not be in the tree?
35. [Likely] Validate a binary search tree.
36. [Likely] Given a product category tree (Electronics > Audio > Headphones), return the path from the root to a category and the depth of the deepest leaf.
37. [Likely] Implement prediction for a trained decision tree stored as an array of nodes (feature, threshold, left, right, leaf value).
    - How would you batch-predict efficiently for a million rows?
38. [Likely] Return the right-side view of a binary tree.

## F. Graphs and grids

39. [Reported] Count the number of islands in a 2D grid. Reported in AS loops in 2020, 2025 and 2026, and in a London 2026 screen. [5][7][14][29]
    - Should diagonals count as connected?
    - What if the grid is streamed row by row and does not fit in memory?
    - Can you solve it with union-find?
40. [Reported] A BFS problem on a grid (for example, the minimum number of steps from start to target). [32]
    - How do you reconstruct the path?
41. [Reported] Shortest path with Dijkstra's algorithm where the grid or graph has obstacles. [38]
    - Why do negative edge weights break Dijkstra?
    - When would A* be the better choice?
42. [Reported] Explain how you would implement Dijkstra: which data structures, and what runtime? [13]
    - Which data structure gives O((V+E) log V), and which gives O(V^2)?
43. [Reported] A DFS-based coding exercise in a Senior AS onsite (Boston, 2025). [11]
44. [Reported] Word ladder (BFS on implicit graph). This is Amazon SDE-tagged and frequently reported. [47]
    - How do you build neighbors efficiently using wildcard patterns?
45. [Likely] Course schedule: given prerequisite pairs, return a valid order or report that there is a cycle. This could be scheduling ML pipeline steps in a DAG.
46. [Likely] Clone a graph, for example a knowledge-graph subgraph used by Alexa.
47. [Likely] Given warehouse aisles as a grid with blocked cells, find the shortest route for a robot from a start to multiple pick locations. This is related to Amazon Robotics.
    - How does the problem change with 2 pick locations compared with k?
48. [Likely] Rotting oranges and multi-source BFS: how many minutes until every reachable cell is infected?
49. [Likely] Find the connected components of users who share a payment method or address, which is a fraud-detection framing.
    - How would union-find with path compression help at scale?

## G. Heaps, top-k and streaming

50. [Reported] Find the median of a data stream. Reported as a LeetCode Medium/Hard in an AS loop (Luxembourg, 2023). [12]
    - What if most values are small integers in a known range?
    - How would you support a sliding-window median?
51. [Reported] Top K frequent elements. This is a high-frequency Amazon-tagged problem. [47][54]
    - Can you compare heap and bucket sort?
    - How would you approximate this with Count-Min Sketch when there are billions of items?
52. [Reported] K closest points to the origin. This is Amazon SDE-tagged. [54]
    - Can you use quickselect, and what is its worst case?
53. [Likely] Find the kth largest element in an unsorted array, such as the kth highest model score in a batch.
54. [Likely] Return the top k products by click-through rate in a sliding one-hour window over a click stream.
    - What state must you keep per product?
55. [Likely] Merge k sorted model-score streams from different shards into a global top-N ranking.

## H. Binary search

56. [Reported] A binary search coding question in a Senior AS loop (New York, 2026). [11]
57. [Reported] Search a 2D matrix whose rows and columns are both sorted, and find the optimal approach. [41]
    - Can you compare the staircase search with binary search on each row?
58. [Reported] Implement binary search and state its time complexity. [44]
    - Can you write it to find the first and the last occurrence of a value?
59. [Likely] Search in a rotated sorted array.
60. [Likely] Given a monotonic model-score threshold function, find the smallest threshold that keeps precision at or above a target. This is binary search on the answer.
61. [Likely] Koko eating bananas or ship packages within D days. This is binary search on capacity, in a package-delivery framing.

## I. Dynamic programming

62. [Reported] A minimum-coins variant solved with 1-D DP (AS intern, 2025). [32]
    - Can you return the actual coins used, not just the count?
    - How do you count the number of ways instead?
63. [Reported] A dynamic programming problem from the Blind 150 list, asked by the hiring manager in a phone screen. [5][17]
64. [Reported] Two string DP and hash problems, from a PracHub Amazon listing. [40]
65. [Likely] Edit distance between two strings, for example to correct misspelled Alexa transcriptions.
    - Can you reduce memory to O(min(m, n))?
66. [Likely] Longest increasing subsequence.
    - Can you do it in O(n log n)?
67. [Likely] Word break: can a query string be split into dictionary words? This is a framing for search-query segmentation.
    - How does a trie help?
68. [Likely] Decode ways, or count paths in a grid with obstacles.
69. [Likely] Viterbi-style DP: given emission and transition scores, find the highest-scoring tag sequence.

## J. Randomization and sampling

70. [Reported] Random pick with weight: build a weighted random number generator from a probability array. Reported in AS and Senior AS screens in 2026. [11][41][43]
    - How do you handle weights that change frequently?
    - What does the alias method give you, and what does it cost?
71. [Likely] Reservoir sampling: choose k items uniformly from a stream of unknown length.
    - What changes for weighted reservoir sampling?
72. [Likely] Shuffle an array uniformly at random (Fisher-Yates), and explain why the naive swap-with-any-index version is biased.
73. [Likely] Implement rand7() using rand5().

## K. Design-style data structures

74. [Reported] Implement an LRU cache. Reported in Amazon loops, including a basic-DS AS-adjacent report. [14][47]
    - Can every operation be O(1)?
    - How would you make it thread-safe?
    - What about LFU, or TTL-based eviction?
75. [Likely] Design a key-value store that keeps a time-stamped history of values, for features stored at a point in time.
76. [Likely] Implement a trie with insert, search and startsWith, used for query autocomplete. Tries appear in AS candidates' prep lists. [33]
    - How would you return the top-3 completions by query frequency?
77. [Likely] Design a hit counter that returns the number of events in the past 5 minutes.
78. [Likely] Implement a min-stack or a max-queue that supports O(1) min or max.

## L. ML-flavored coding (often in the same round for AS)

79. [Reported] Write a program to calculate AUC (Applied Scientist II). [6][16]
    - What do you do with tied scores?
    - Can you do it in O(n log n) without building the full ROC curve?
80. [Reported] Write a function that computes Intersection over Union (IoU) for two bounding boxes. [43]
    - What edge cases come from boxes that do not overlap or have zero area?
    - How would you vectorize it for N x M boxes?
81. [Reported] Implement k-means from scratch in Python. [41][4]
    - How do you initialize the centroids (k-means++)?
    - When should it stop, and what happens with empty clusters?
82. [Reported] Implement k-nearest neighbors from scratch. [50]
    - How do you scale features, choose k, and deal with high dimensionality?
83. [Reported] Compute the Chamfer distance between two point clouds efficiently, using an exact nearest-neighbor structure such as a KD-tree instead of computing all pairwise distances. [42]
    - How does a KD-tree perform as dimensionality increases?
84. [Reported] Code a traditional ML classifier from scratch. [17]
85. [Reported] Write gradient descent without libraries. [45]
    - How would you change it to mini-batch SGD with momentum?
86. [Likely] Implement logistic regression training with gradient descent and the loss function, using NumPy only.
87. [Likely] Implement non-maximum suppression for object detection, for example for Amazon Go or Lab126 vision teams.
88. [Likely] Implement a numerically stable softmax and cross-entropy.
89. [Likely] Implement scaled dot-product attention with a causal mask (for AGI or LLM teams).
    - What does a KV cache change at inference time?
90. [Likely] Implement beam search decoding over a toy next-token probability function.
91. [Likely] Implement byte-pair encoding (BPE) merges on a small corpus.
92. [Likely] Compute precision@k and NDCG@k for a ranked list of search results.

## M. Data manipulation (AS and Data Scientist-style coding)

93. [Reported] Compute a rolling average when some dates are missing. A Blind commenter advised preparing "leetcode medium with focus on data manipulation". [18]
    - Should missing days count as zeros or be skipped, and how do you defend that choice?
94. [Reported] A simple Python question plus a medium SQL question. [20]
95. [Reported] Simple data-wrangling questions in Python alongside arrays, hash maps, trees and linked lists. [6]
96. [Likely] Given a log of (user_id, session_start, session_end), compute each user's total active time, merging overlapping sessions.
97. [Likely] Deduplicate near-identical product titles using normalized string keys.

## N. LP probes typically embedded in the coding round

98. [Reported] Tell me about a time you had to dive deep into data or code to find a root cause. LP questions open or close most technical rounds. [45][48]
99. [Likely] You found a better algorithm after you had already shipped. What did you do (Insist on the Highest Standards)?
100. [Likely] Tell me about a time you simplified a complex piece of code or a pipeline (Invent and Simplify).

## References

1. Applied Scientist Interview Prep, amazon.jobs (official). https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. SDE II Interview Prep, amazon.jobs (official). https://amazon.jobs/content/en/how-we-hire/sde-ii-interview-prep
3. Amazon recruiters share 6 job interview tips for software development engineers, aboutamazon.com. https://www.aboutamazon.com/news/workplace/amazon-job-interview-software-development-engineer-recruiter-tips
4. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer. https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
5. Amazon Applied Scientist Interview Experience & Questions, Glassdoor (2020-2026 reviews). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
6. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
7. Amazon Applied Scientist Intern Interview Experience & Questions, Glassdoor (includes 2026 Bengaluru and New Delhi reports). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Intern-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
8. Amazon Senior Applied Scientist Interview Questions, Glassdoor. https://www.glassdoor.ca/Interview/Amazon-Senior-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
9. Glassdoor review RVW91266671, Amazon Applied Scientist, Seattle (2026; string compression). https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW91266671.htm
10. Glassdoor review RVW89381532, Amazon Applied Scientist (2024; rotate matrix 90 degrees). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW89381532.htm
11. Glassdoor review RVW102577344, Amazon Senior Applied Scientist, Seattle (2026; weighted random number generator), plus related Senior AS reports. https://www.glassdoor.co.nz/Interview/Amazon-Interview-E6036-RVW102577344.htm
12. Glassdoor review RVW99133644, Amazon Applied Scientist (Luxembourg, 2023; median from data stream). https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
13. Glassdoor review RVW74547087, Amazon (Luxembourg, 2022; implementing Dijkstra). https://www.glassdoor.com.au/Interview/Amazon-Interview-E6036-RVW74547087.htm
14. Amazon Interview Question: Basic DS questions: LRU cache, trees, count the islands, Glassdoor. https://www.glassdoor.com/Interview/Basic-DS-questions-LRU-cache-trees-count-the-islands-QTN_4447065.htm
15. Amazon Lab126 Applied Scientist Interview Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Lab126-Applied-Scientist-Interview-Questions-EI_IE267709.0,13_KO14,31.htm
16. Amazon Web Services Applied Scientist Interview Questions, Glassdoor. https://www.glassdoor.com/Interview/Amazon-Web-Services-Applied-Scientist-Interview-Questions-EI_IE7470741.0,19_KO20,37.htm
17. Glassdoor review RVW47917252, Amazon Applied Scientist (2020-2022 era; TwoSum, DP, ML classifier). https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW47917252.htm
18. Amazon Applied Scientist (L5) Virtual Onsite, Blind. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
19. Amazon applied scientist loop, Blind. https://www.teamblind.com/post/amazon-applied-scientist-loop-odlzgrp3
20. Tips for coding interview, Applied Scientist at Amazon, Blind. https://www.teamblind.com/post/tips-for-coding-interview-applied-scientist-at-amazon-txexmtsb
21. Amazon applied scientist interview-onsite, Blind. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
22. Applied scientist interview Amazon, Blind. https://www.teamblind.com/post/applied-scientist-interview-amazon-4udsobcp
23. Amazon Research-Applied Scientist position, LeetCode Discuss. https://leetcode.com/discuss/interview-experience/444789/amazon-research-applied-scientist-position/
24. Amazon | Applied Scientist | Phone Screen | USA, LeetCode Discuss. https://leetcode.com/discuss/interview-experience/985312/amazon-applied-scientist-phone-screen-usa/
25. Amazon online 1st round, Bengaluru, Applied Scientist (2025), LeetCode Discuss. https://leetcode.com/discuss/post/6871782/amazon-online-1st-round-bengaluru-applie-uiwk/
26. Amazon Applied Scientist phone screen (hashmap plus MapReduce follow-up), 1point3acres (2020). https://www.1point3acres.com/bbs/thread-731301-1-1.html
27. Amazon applied scientist phone screen (product of other elements), 1point3acres. https://www.1point3acres.com/bbs/interview/amazon-data-science-506632.html
28. Amazon applied scientist phone interview (merge intervals), 1point3acres (2020). https://www.1point3acres.com/bbs/interview/amazon-machine-learning-677329.html
29. Applied scientist intern @Amazon (count islands), 1point3acres (2020). https://www.1point3acres.com/bbs/interview/amazon-machine-learning-613476.html
30. Amazon Applied Scientist interview discussion, 1point3acres. https://www.1point3acres.com/bbs/thread-655973-1-1.html
31. Amazon L5 Applied Scientist Fulltime Tech Phone Screen Overview, 1point3acres. https://www.1point3acres.com/interview/thread/1170176
32. My Amazon Applied Scientist Intern Interview Experience (Off-Campus), Medium (2025). https://medium.com/@pranavpandeyghq123/my-amazon-applied-scientist-intern-interview-experience-off-campus-7163c5b7c9ae
33. My Amazon Applied Scientist Internship Interview Experience, Medium (2025). https://medium.com/@brintodibyendu/my-amazon-applied-scientist-internship-interview-experience-783736ba0ed4
34. My Amazon Applied Science Interview Journey, Medium. https://medium.com/@ajinkya.pahinkar98/my-amazon-applied-science-interview-journey-f3eb1ed748a6
35. Amazon Applied Scientist Interview Guide, Alexa International Tech, Medium. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
36. Amazon Interview Experience for Applied Scientist Internship, GeeksforGeeks (updated 2024). https://www.geeksforgeeks.org/interview-experiences/amazon-interview-experience-for-applied-scientist-internship/
37. Amazon Interview Experience for Applied Scientist, GeeksforGeeks. https://www.geeksforgeeks.org/interview-experiences/amazon-interview-experience-for-applied-scientist/
38. Amazon new grad Applied Scientist interview experience: five virtual onsite rounds, L4 offer, PracHub (2025). https://prachub.com/interview-experiences/amazon-new-grad-applied-scientist-interview-experience-five-virtual-onsite-rounds-mostly-behavioral-end-in-an-l4-offer
39. Merge Overlapping Time Ranges (Amazon Applied Scientist technical screen, 2026), PracHub. https://prachub.com/interview-questions/merge-overlapping-time-ranges
40. Amazon Applied Scientist Intern Interview 2027: Coding, ML Depth, and Behavioral Questions, PracHub. https://prachub.com/resources/amazon-applied-scientist-intern-interview-2027-coding-ml-depth-and-behavioral-questions
41. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford. https://dataford.io/interview-guides/amazon/applied-scientist
42. Efficient Chamfer Distance (Amazon-framed practice question), Dataford. https://dataford.io/questions/efficient-chamfer-distance
43. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart. https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
44. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
45. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (commercial prep service). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
46. Amazon Applied Scientist Interview: A Complete Guide, codinginterview.com. https://www.codinginterview.com/guide/amazon-applied-scientist-interview/
47. Amazon Coding Interview Questions: The Real Bar (2026 Guide), copilotinterview.com. https://copilotinterview.com/blog/amazon-coding-interview-questions
48. Senior Engineer's Guide to Amazon Interviews + Questions, interviewing.io. https://interviewing.io/guides/hiring-process/amazon
49. Amazon SDE-1/2 interview questions 2025, LeetCode Discuss. https://leetcode.com/discuss/post/6752237/amazon-interview-questions-2025-by-anony-qgzt/
50. Amazon ML Scientist Interview Question: Program k-Nearest Neighbour from Scratch, Substack. https://jaspinders30.substack.com/p/amazon-ml-scientist-interview-question
51. 14 Most Popular Amazon Coding Interview Questions, Design Gurus. https://www.designgurus.io/blog/amazon-14-question
52. 30 Amazon LeetCode Interview Questions for 2026, Verve Copilot. https://www.vervecopilot.com/blog/amazon-leetcode-interview-questions
53. Amazon interview questions from phone screens (nearest smaller element to the left), Glassdoor review RVW67853279. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW67853279.htm
54. Top 100 LeetCode Problems for MAANG India 2025: Frequency-Ranked, Impacteers. https://futurejobs.impacteers.com/blogs/top-100-leetcode-problems-for-maang-interviews-frequency-ranked-with-patterns-and-solutions-india-2025
55. Amazon AGI Info Applied Scientist Interview Experience and Insights, 1point3acres. https://www.1point3acres.com/interview/thread/1130091
56. Asteroid Collision (Amazon-tagged community question), Hello Interview. https://www.hellointerview.com/community/questions/asteroid-collision/cm5eguhac02kt838o4qw0x2kf
