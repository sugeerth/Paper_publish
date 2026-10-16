# 03. Airbnb Senior MLE: ML Coding and Practical Coding

> Scope: the coding rounds of the Airbnb Senior (G9) and Senior+ Machine Learning Engineer loop: the technical phone screen, the onsite coding round, the HackerRank-style ML/data screen and the code review round. This chapter contains questions only. It gives no answers or solutions.
>
> How it relates to the older chapters: this chapter merges and dedupes four October 2026 search passes with `airbnb-senior-mle/chapters/02-coding-practical.md` (practical coding) and `05-ml-fundamentals.md` (ML depth). Ch. 02 keeps the long tail of SWE-track problems (banking, refunds, ski DAG, Diplomacy, multi-level online assessments). Ch. 05 keeps verbal ML-theory probes. This chapter keeps the problems most likely to reach an MLE candidate, written as runnable practice prompts.
>
> Source caveat: almost every source is an anonymous candidate self-report (Blind, LeetCode Discuss, Glassdoor, 1point3acres, PracHub) or a prep-vendor page. Many were read only as search-result extracts, and some 1point3acres and PracHub bodies need a login. Several reports come from SWE, data science or data engineering candidates rather than MLEs; the role is noted where it matters. Read **[Reported]** as "a cited source says this, or a close paraphrase, was asked at Airbnb". **[Likely]** means a grounded inference from the MLE loop structure, prep guides or Airbnb's published ML work; nobody reported it verbatim.

---

## How this round works

- **Where coding sits in the MLE loop.** MLE candidates describe the loop as "LeetCode + standard ML design + an ML experience round" [48][49]. The public MLE-track summary gives one technical phone screen and typically one onsite coding round of 45-60 minutes, and says the track is filtered mainly by design depth [80]. A 2022 senior ML/DS loop instead had two "DS coding" rounds of Airbnb-tagged LeetCode hard and medium problems in Python, plus two ML rounds [47]. A July 2025 ML platform loop had a coding screen and then an onsite with a code review round [53]. Ask your recruiter which mix you will get.
- **Two flavors of coding.** (1) *Practical algorithmic coding*: an Airbnb-framed product problem (split stays, menus, group lodging, pagination, text layout, query parsing) solved with runnable code [69]. A 2026 Senior+ MLE in the search-ranking org got a "format book records as a table" screen and a "document redaction" onsite [43]. (2) *ML/data coding*: a 45-minute HackerRank screen with pandas data manipulation plus ML concepts [54][62], or implement-from-scratch tasks (kNN, k-means, IoU, a small neural net, convolution, z-scores) reported for a data scientist algorithms round [45]. A Staff MLE candidate asked whether coding has an ML component and got no clear answer [50], so prepare both. One prep site describes a 60-minute screen mixing ML fundamentals, a coding problem and introductory system design [78].
- **Difficulty.** Phone screens are reported as easy to medium. An MLE onsite coding round was LeetCode-hard and the candidate failed it [46]. Interviewers push for an optimal solution: one senior SWE passed with a suboptimal version that ran and passed every test, then was pushed to optimize [1]; another was rejected on split stay over live debugging and deduplication [4].
- **Language and tools.** Python is the norm for MLEs. Expect a shared editor or HackerRank with code you actually run, often with no starter interface, so you define the API and write your own tests [80][81][86]. Candidates stress "bug free, all cases must run" [9].
- **What the senior bar looks like.** Clarify ambiguous specs before coding (tie-breaks, half-open date ranges, return path versus cost). Reach working, tested code with 10-15 minutes left for follow-ups. State complexity, then extend the solution (k listings instead of two, return all optima, stream the input) without a rewrite. For ML coding, interviewers also probe numerical stability, vectorization, leakage-safe splits and how the code would run in production [43][62].
- **Code review.** Some loops add a code review round: about three pull requests in about 60 minutes. Do not spend too long on the first one [52][7][81].

---

## Top 15 to practice first

Reported prompts first, ordered by how many independent reports mention them.

1. Split stay: every pair of listings that covers a date range back to back; then k listings and the cheapest split. [Reported] (Q1)
2. Minimum-cost menu order with combos (DFS, then bitmask DP). [Reported] (Q2)
3. Menu combination sum with float prices: every combination that hits the budget exactly. [Reported] (Q3)
4. Group lodging / property booking optimizer: fewest properties, then least capacity. [Reported] (Q4)
5. Paginate listings so no host repeats on a page. [Reported] (Q5)
6. Text justification, text formatter and the book-records table (the 2026 Senior+ MLE screen). [Reported] (Q6, Q7, Q8)
7. Document redaction with overlap, case and word-boundary rules (2026 Senior+ MLE onsite). [Reported] (Q9)
8. Palindrome Pairs in better than O(n^2 * k). [Reported] (Q10)
9. Cheapest flights within K stops, returning the route. [Reported] (Q11)
10. Sliding puzzle and grid BFS with keys and doors. [Reported] (Q12, Q13)
11. 2D iterator with `remove`. [Reported] (Q14)
12. HackerRank ML screen: parse a string column into an integer average; write a CV loop that returns the variance of a metric. [Reported] (Q52, Q41)
13. Threshold tuning for a fraud/risk classifier plus a reliability curve (calibration topic reported; coding framing inferred). [Likely] (Q42)
14. kNN and k-means from scratch in NumPy. [Reported] (Q35, Q36)
15. Code review round: three PRs with planted bugs. [Reported] (Q33)

Then work through the Level calibration list (Q76-Q85) to check you can hit the Senior bar under time pressure.

---

## A. Airbnb-tagged practical coding problems (most reported first)

1. **Split stay.** Input: a requested range `[start, end)` and a map `listing_id -> available days` (a list of day numbers or date ranges). Output: every unordered pair of distinct listings (A, B) and split day k such that A covers `[start, k)` and B covers `[k, end)`, with no duplicate pairs. Constraints: up to N listings and D days; availability may be unsorted or have gaps. Edge cases: a single listing covering the whole range, one-day stays, `start == end`, listings with no availability. The code must run bug-free live against your own tests. [Reported] [1][4][5][7][8][9][82]
   - Your O(N^2 * D) solution passes. The interviewer says something better exists. Where is the redundant work? [Reported] [8]
   - Must A cover a contiguous prefix and B a contiguous suffix, or is plain set coverage enough? Which reading do you confirm before coding? [Reported] [7]
   - How would a bitmask encoding of each listing's availability change the check? [Reported] [7]
   - Generalize to at most k listings and return the best split by price or by fewest moves. What is the DP state? [Reported] [1][4]
   - Expose this as an HTTP endpoint that takes a date range and listings with availability. What do you validate, paginate and cache? [Reported] [5][6]
   - Are the day numbers nights (check-in day) or calendar days, and how does that change your boundary tests?

2. **Minimum-cost menu order.** Input: menu entries, each a single item or a combo of several items with a price, and a list of wanted items (small, for example at most 3 unique, possibly with repeats). Output: the minimum total cost to get at least the wanted items, and the entries to order. Edge cases: a combo cheaper than its parts, combos containing unwanted items, wanted items not on the menu (return -1 or an error). Write tests first. [Reported] [1][11][80][92]
   - You wrote DFS. Convert it to DP with a bitmask over wanted items. What does the state hold? [Reported] [11]
   - Which menu entries can you prune before searching? [Reported] [92]
   - Return every optimal order, not just one. How does that change memory? [Reported] [92]
   - Variant: rental "services" (wifi, parking, TV) instead of dishes. What changes in the model? [Reported] [91]

3. **Menu combination sum with float prices.** Input: menu items with decimal prices (for example 2.40, 3.15, 10.02) or a `name -> price` map, and a budget (for example 15.05). Output: every combination of items whose total equals the budget exactly, in a deterministic order, with no duplicate combinations. Items may repeat. Edge cases: float precision, a zero or negative price, an empty menu, a budget no combination reaches. [Reported] [12][13][14][15][16][17][85]
   - How do you avoid floating-point equality bugs? [Reported] [12][16]
   - Each item may now be used at most once and the input contains duplicates. How do you avoid duplicate results? [Reported] [17]
   - Return only the count of combinations. Does order matter? [Reported] [17]
   - What pruning do you add, and what is the worst-case complexity? [Reported] [15]

4. **Group lodging / property booking optimizer.** Input: properties in a neighborhood with capacities, a target neighborhood and a group size. Output: the property IDs whose total capacity is at least the group size, minimizing the number of properties and then total capacity (some versions reverse the tie-break). Return `[]` or -1 if impossible. Edge cases: one property fits exactly, properties in other neighborhoods, group size 0, equal-cost ties. [Reported] [2][3][10][83]
   - Which tie-break do you confirm first: fewer properties or less wasted capacity? [Reported] [10]
   - Is this subset-sum DP or backtracking? What is the worst case for 50 properties?
   - How would you return every optimal set?

5. **Paginate listings.** Input: CSV strings `host_id,listing_id,score,city` sorted by score descending, and a page size (12 in the original). Output: pages printed as blocks separated by a blank line. Within a page a host appears at most once while unused listings from other hosts remain; if a page cannot be filled with distinct hosts, backfill with the remaining listings in their original order. Keep score order otherwise and handle a final partial page. [Reported] [18][19][20][85]
   - What happens when every remaining listing belongs to one host?
   - Can you do it in one pass without rescanning the list for each page?
   - How do you test that relative order inside a page is preserved?
   - The page size changes per request and results stream in. How would you adapt this for a search API?

6. **Text justification.** Input: words and `maxWidth`. Output: lines of exactly `maxWidth` characters, fully justified, with extra spaces going to the leftmost gaps, and the last line left-justified. Edge cases: a single-word line, a word exactly `maxWidth` long. [Reported] [13][40][72][84][85]
   - What changes if a word can be longer than `maxWidth`?
   - Lay the output out in k side-by-side columns. What do you print when one column runs out?

7. **Text formatter (45-minute phone screen).** Input: a raw paragraph and a line width. Output: wrapped lines with a chosen alignment (left, right, center). Words longer than the width must be handled (hyphenate, split or overflow). The code must run with tests. [Reported] [41][80]
   - How do you handle repeated whitespace, tabs and newlines in the input?
   - What is the complexity for 1e5 words, and where is the bottleneck?

8. **Book records table (Senior+ MLE technical screen, 2026).** Input: a list of book records (title, author, year and similar fields), some with missing fields. Output: a formatted text table with a header row and separators, each column sized to its longest value, and cells padded and aligned. Practice spec: records are dicts with keys `title`, `author`, `year`; columns appear in that order; a missing field prints as an empty cell; separators use `+`, `-` and `|`. Signature: `format_table(records: list[dict]) -> str`. Edge cases: an empty list, very long titles, missing fields, non-ASCII characters. Write production-quality code with tests. [Reported] [43]
   - Truncate or wrap titles beyond a maximum column width. Which is easier to test?
   - How would you right-align numeric columns and left-align text without hardcoding columns?
   - Records arrive as a stream too large to hold. Can you still size columns?

9. **Document redaction (Senior+ MLE onsite, 2026).** Input: a document string and a list of sensitive terms or patterns (names, emails, phone numbers). Output: the document with each match replaced by a mask. Matching is case-insensitive, respects word boundaries and handles overlapping matches. Practice spec (confirm with the interviewer): terms are plain strings, each match is replaced by `[REDACTED]`, the original casing and all non-matched text are preserved, and when matches overlap the longest match starting at the earliest position wins, so a run covered by overlapping matches is redacted once. Signature: `redact(doc: str, terms: list[str]) -> str`. [Reported] [43][80]
   - Two terms overlap ("John" and "John Smith"). Which wins, and how do you test it?
   - Thousands of patterns: how would a trie or Aho-Corasick change the complexity?
   - How would you replace the rule-based matcher with an NER model, and how would you measure missed redactions? [Likely] [43]
   - Which tests prove the redaction is complete, including matches that span punctuation?

10. **Palindrome Pairs.** Input: unique words. Output: all index pairs (i, j), i != j, where `words[i] + words[j]` is a palindrome. Target better than O(n^2 * k) with a hash map of reversed words or a trie. [Reported] [20][21][84][85]
    - How do the empty string and single-character words affect correctness?
    - Solve it with a trie of reversed words and give the time and space for n words of length k. [Reported] [21]

11. **Cheapest flights within K stops, with the route.** Input: flights `(from, to, price)`, src, dst and K. Output: the cheapest price with at most K stops, or -1, and the list of stops on that route. [Reported] [26][27][28][73][84]
    - Which algorithms still work once the route must be returned?
    - How do you handle ties and zero-cost cycles?
    - Replace "at most K stops" with "layover between min and max hours". What changes? [Likely] [65]

12. **Sliding puzzle.** Input: a 2x3 board with tiles 1-5 and 0 as the blank (or the 3x3 variant). Output: the minimum moves to reach the solved state, or -1. BFS over board states is expected; DFS was judged suboptimal. [Reported] [24][25][84]
    - How do you detect an unsolvable board before searching?
    - Would A* help, and with which heuristic?
    - Return the move sequence, not only its length.

13. **Grid maze shortest path, including keys and doors.** Input: a 2D grid of strings where `#` is a wall, `.` is open, `@` is the start, `$` is the target, lowercase `a-f` are keys and uppercase `A-F` are doors that need the matching key; moves are 4-directional. Output: the minimum steps, or -1. Write complete, runnable BFS. [Reported] [30][31][32][84]
    - With up to 6 keys, how do you encode the key state, and how large is the state space?
    - Variant: a ball rolls until it hits a wall. What changes in neighbor generation?

14. **2D iterator with remove.** Input: a list of lists (inner lists may be empty). Implement `has_next()`, `next()` and `remove()`, where `remove()` deletes the element last returned by `next()` from the underlying structure. Calling `remove()` twice in a row, or before any `next()`, must raise. [Reported] [22][23][85]
    - How do you make the iterator lazy over a very large nested input?
    - What happens if the caller mutates the lists outside the iterator?

15. **IP to CIDR.** Input: a start IPv4 address and a count n. Output: the shortest list of CIDR blocks (for example `255.0.0.7/32`) that covers exactly `[ip, ip + n - 1]`. [Reported] [20][73][85]
    - How do you test block boundaries, such as a range starting at an odd address or crossing an octet?

16. **Minimum window substring.** Input: strings s and t. Output: the smallest substring of s containing every character of t with multiplicity, or "". An optimal solution was expected with few hints. [Reported] [24][84]
    - Follow-up: minimum window *subsequence*. Why does the sliding window no longer work?

17. **Ten wizards.** Wizard i knows a list of other wizards; travelling from i to j costs (j - i)^2. Return the minimum-cost path from wizard 0 to wizard 9 and the path itself. [Reported] [29][85]
    - Why is BFS wrong here, and what do you use instead?

18. **Smallest common region.** Input: region hierarchies as lists `[parent, child1, child2, ...]` and two regions. Output: their smallest common ancestor region. [Reported] [29][84]
    - The hierarchy is huge and queries are many. What do you precompute?

19. **Alien dictionary.** Input: words sorted in an unknown alphabet. Output: a valid letter order, or "" if invalid (a cycle, or a longer word before its own prefix). [Reported] [39][40][84]
    - How do you report which letters form the cycle?

20. **City preference merge.** Each user gives an ordered list of preferred cities. Merge them into one global ordering consistent with every list, or report that none exists. [Reported] [42][85]
    - Several valid orders exist. How do you make the output deterministic?

21. **Calculate board score.** Input: a grid of tiles (R and C up to 2000). Output: the total score of the board. Sources give the rule only loosely; practice this Kingdomino-style version: each cell has a terrain letter and a crown count `0-3`; a region is a maximal 4-connected group of cells with the same terrain; the board score is the sum over regions of `(region size) * (crowns in region)`. Signature: `board_score(terrain: list[str], crowns: list[list[int]]) -> int`. Reported in a full-time Airbnb ML tech phone screen. [Reported] [35][36]
    - What is the time and memory for a 2000 x 2000 grid, and can you do it in one pass?

22. **ML phone screen graph problem (45 minutes).** A graph problem solved with DFS and connected components. The exact statement is not public; practice this version: given `n` nodes labelled `0..n-1` and an undirected edge list, return the number of connected components and the sorted list of nodes in the largest one (ties broken by smallest minimum node). [Reported] [55]
    - Would recursive DFS overflow the stack on the largest input? When do you switch to iterative DFS or union-find?
    - Edges arrive as a stream. What changes?

23. **Strings that differ by one character.** Input: n strings of equal length m. Output: true if any two differ in exactly one position. Beat O(n^2 * m). [Reported] [33][84]
    - How do you avoid false positives from hash collisions?

24. **Queue from fixed-length arrays.** Implement a queue using only fixed-size arrays (for example size 5) as building blocks, with `push`, `pop` and `size` in O(1) amortized. [Reported] [34][85]
    - How do you reclaim arrays after many pops?

25. **Rooms and keys / boxes with keys.** Rooms (or boxes) contain keys to other rooms and possibly items. Decide whether every room can be visited, in how few steps, or (LC 1298) the maximum candies collectable. LC 1298 spec: `status[i]` is 1 if box i is open, `candies[i]`, `keys[i]` (boxes whose keys are inside box i), `containedBoxes[i]` (boxes inside box i) and `initialBoxes`; you may open any box you hold that is open or for which you hold a key; return the total candies collected. In one MLE loop, LC 1298 arrived with no interface, so the candidate defined the API and tests. [Reported] [37][86]
    - Which classes and methods would you model so the solution matches the interviewer's intent?

26. **Pour water on terrain.** Input: terrain heights, V water units and a drop index K. Each droplet moves left if it can eventually fall lower, otherwise right, otherwise stays at K. Output: final heights, and the terrain printed as ASCII with water marked. Part 1 may be printing the terrain alone. [Reported] [64][70][82][90]
    - Which way should water go when both sides are equally low?
    - How do you test exact printed output, including trailing whitespace?

27. **Trapping rain water.** Input: a list of non-negative bar heights of width 1. Output: total units of water trapped. Then the 2D version: an `m x n` height map, water can drain off any border cell. [Reported] [38][84]
    - Why does the 1D two-pointer idea not carry over to 2D, and what does?

28. **Connect Four / ConnectN engine.** A configurable board (6x7 default), two players, `drop(col)` returning the result, with win detection horizontally, vertically and diagonally, plus draws and invalid moves. ConnectN with configurable N was the coding round in one MLE loop. [Reported] [63][87][88]
    - How do you avoid rescanning a column for gravity on every move? [Reported] [88]
    - Determine win, draw or ongoing from a static board alone.
    - Which interfaces make a new rule (irregular board, three players) cheap to add?

29. **Key-value store.** Implement an in-memory store with `get`, `set` and `delete`, then state the tests you would write. Onsite, extend it with transactions (`begin`, `commit`, `rollback`, nested) or a versioned `get(key, timestamp)`, and show how the tests change. [Reported] [4][70][80]
    - Variant: `set_sum(k, [keys])` where updates cascade to dependents. Eager propagation or lazy recompute? What happens on a cycle? [Reported] [89]
    - How would you make it safe under concurrent readers and writers?

30. **URL query parser.** Input: the query portion of a GET URL (or a whole URL). Output: a map from keys to values. Repeated keys become lists, percent-encoding and `+` are decoded after splitting, and empty values and fragments are handled, without `urllib`. [Reported] [66][81][82]
    - A valueless key: true, empty string or null? Which spec detail do you confirm first?
    - Support nested keys such as `a[b]=1`. How does the return type change?

31. **CSV parser.** Implement `parse_csv_line(s)` with double-quoted fields containing commas and escaped quotes (`""`). Follow-up: write the inverse, escaping a list of fields into one line. [Reported] [71][85]
    - How do you handle a quoted field spanning multiple lines?

32. **Retryer.** Implement `retry(fn, max_attempts, base_interval, max_interval=16.0)` that calls `fn()`, sleeps `min(base_interval * 2**attempt, max_interval)` seconds after each failure, returns the first successful result and rethrows the last error once attempts run out. [Reported] [81][88]
    - How would you plug in jitter, retry-on-specific-exceptions or a time budget?
    - How do you unit-test timing without real sleeps?

33. **Code review round.** Practice by writing or finding three 40-80-line Python diffs and reviewing each in 15-20 minutes with prioritized written comments. Review about three short pull requests (Python service, data pipeline or helper code such as booking or pagination logic) in about 60 minutes. Planted defects include wrong results without a crash, swallowed exceptions, unchecked nulls, races on shared state, missing error handling around external calls, hidden quadratic loops and poor naming. [Reported] [7][52][53][81]
    - The third PR is long and time is short. How do you triage which comments to write first?
    - Rewrite the worst function correctly and add the tests that would have caught the bug.

34. **Warm-up and listed DP problems.** (a) Minimum cost climbing stairs: given `cost[i]` per step, start at step 0 or 1, climb 1 or 2 steps, return the minimum cost to pass the top. (b) Work sessions: given task durations (n <= 14) and `session_time` >= the longest task, return the minimum number of sessions, where tasks are not split across sessions (bitmask DP). (c) Booking requests: given a list of request lengths in nights, accept no two adjacent requests, maximize total nights, and return both the total and the accepted indices (House Robber framing). [Reported] [1][74][85]
    - For work sessions, what is the state space for 14 tasks, and how do you prune it?

---

## B. ML-specific coding

35. **k-nearest neighbors from scratch.** Implement `fit(X, y)` and `predict(X_test, k)` in Python/NumPy with Euclidean distance and majority vote. Handle ties in the vote, k larger than n, and mixed-scale features. Reported for an Airbnb data scientist algorithms round, where the recruiter said coding was ML-implementation focused, not LeetCode. [Reported] [45]
    - Vectorize the distance computation without an n x m Python loop. What is the memory cost?
    - How would you serve kNN over millions of listing embeddings?

36. **k-means from scratch.** Input: X (n x d), k, max_iter, tol. Output: labels and centroids. Cover initialization, the assignment and update steps, convergence, and empty clusters. [Reported] [45]
    - Implement k-means++ initialization. Why does it help?
    - How do you pick k, and how do you test the function deterministically?

37. **Intersection over Union.** Compute IoU for two axis-aligned boxes `(x1, y1, x2, y2)`, including no overlap, touching edges and degenerate boxes. [Reported] [45]
    - Vectorize it for N x M boxes, then use it to write non-maximum suppression.

38. **2D convolution.** Implement valid-padding 2D convolution with stride s on a NumPy array without library conv functions. [Reported] [45]
    - Add "same" padding and multiple channels. What output shape do you expect?

39. **Small neural network.** Implement a one-hidden-layer feed-forward network for binary classification in NumPy: input `X` (n x d), hidden layer of size h with ReLU, sigmoid output, binary cross-entropy loss; write the forward pass, backprop and mini-batch SGD, and show the loss falling on a toy dataset such as two moons. [Reported] [45]
    - How do you check your gradients numerically?
    - Your loss is NaN after a few steps. What do you check first?

40. **Z-score normalization and outliers.** For a feature column, compute mean and std, z-score it, handle zero variance and NaNs, and return rows with |z| > 3. [Reported] [45]
    - Why must the mean and std come from the training split only?
    - When would you use median and MAD instead?

41. **Cross-validated metric variance (ML screen).** Write `cv_metric(X, y, preprocessor, model, metric, k=5, seed=0) -> (mean, variance)` that, for each of k stratified folds, fits the preprocessing step and the classification model on the training part only, scores the held-out part with `metric(y_true, y_pred_or_score)`, and returns the mean and variance of the k scores. [Reported] [54]
    - Where must preprocessing be fit so that no information leaks across folds?
    - The data is grouped by user and ordered in time. How do you change the splitter?

42. **Threshold tuning and calibration.** Input: `y_true` and `y_score` from a fraud or booking-risk classifier. Output: the threshold that maximizes F1, and the threshold that maximizes recall subject to precision >= P; then a reliability (calibration) curve with expected calibration error. Reported from a 2026 senior MLE onsite rejected "on calibration". [Reported] [44]
    - How would you recalibrate (Platt scaling, isotonic regression), and on which data?
    - Negatives were down-sampled 10:1 in training. How do you correct the probabilities?
    - Explain ties in scores and the case where no threshold meets precision P.

43. **Practical ML notebook (take-home or onsite).** On an Airbnb-like dataset, write Python to clean the data, build and evaluate a classification or regression model, justify the modeling choices, and summarize results in a short tech report. Reported for data science and full-time ML onsite loops. [Reported] [59][60][76]
    - Which baseline do you show first, and why?
    - How do you show that the evaluation split matches how the model will be used?

44. **Nightly price prediction pipeline.** Given a listings DataFrame (location, room type, bedrooms, amenities, seasonality, historical nightly price, occupancy), build a pandas/scikit-learn pipeline with a time-based train/validation split, feature engineering, a baseline and a GBDT, and MAE/MAPE by market. Pricing prediction is reported in Airbnb MLE onsites; the exact coding framing is inferred. [Likely] (topic reported [57])
    - Why can MAPE mislead for cheap listings, and what would you report instead?
    - How would you deploy and monitor this model?

45. **Search ranking without a text query.** For a guest who enters only location and dates, build per (user, listing) features from a log DataFrame (price relative to market, past clicks and bookings, distance, review score). Train a pointwise or pairwise ranker and evaluate offline with NDCG@k on held-out searches. The topic was a 2026 Senior+ MLE onsite round, and a recruiter told another senior MLE candidate the practitioner round targets Airbnb's own engineering problems; the coding framing is inferred. [Likely] (topic reported [43][51])
    - How do you split searches into train and test so the same user's later searches do not leak?

46. **Ranking metrics.** Input: per search, a ranked list of listing IDs and graded relevance (booked = 2, clicked = 1, else 0). Implement NDCG@k, MRR and precision@k, averaged across searches, and compare two rankers. Then implement a "discounted cumulative booking" metric that counts only bookings. Handle searches with no positives. [Likely] [43][48][93]
    - Should searches with no positives be dropped or scored as zero? How does that bias the comparison?
    - How would you get a confidence interval on the NDCG difference between two rankers?

47. **Logistic regression and AUC from scratch.** Implement logistic regression with mini-batch gradient descent and L2 regularization in NumPy, then compute ROC AUC from scratch, including ties. Check both against a library on a toy dataset. [Likely] [50]
    - How do you keep the loss numerically stable for large logits?
    - Compute AUC in O(n log n) rather than over all pairs.

48. **Numerically stable losses.** Implement `bce_from_logits(y, z)` and multiclass `softmax_cross_entropy(Y, Z)` with the log-sum-exp trick, plus their gradients with respect to the logits, and verify the gradients with finite differences. Inputs with `|z| > 1000` must not produce NaN or inf. Explain why log loss beats accuracy for a booking-probability model. [Likely] [79]

49. **Factorization machine scoring.** Given sparse features (indices and values), bias w0, weights w and latent matrix V (n x k), compute the FM prediction in O(nk) using the reformulation, then write one SGD update for logistic loss. FM versus embedding + DNN came up in a 2026 senior MLE onsite. [Likely] [44]
    - How does the cost scale with the number of non-zero features rather than n?

50. **Pairwise learning-to-rank loss.** Given per-search scores and labels (booked or not), implement a vectorized pairwise logistic (RankNet) loss over all positive-negative pairs, then add LambdaRank-style weighting by |delta NDCG|. [Likely] [97]
    - How do you avoid the O(n^2) memory blow-up for long result lists?

51. **Two-tower retrieval and top-k.** Implement a two-tower model in PyTorch (query: location, dates, guests; listing: attributes) trained with in-batch softmax negatives. Then write a function returning the top-k listings for a query from precomputed embeddings with brute-force cosine similarity and an optional geo filter, and report recall@k. [Likely] [96]
    - Why do in-batch negatives over-penalize popular listings, and how do you correct it?
    - Pre-filter or post-filter by geography: what happens to recall?

---

## C. Data, pandas and SQL coding

52. **String column to integer average (HackerRank ML screen).** Given a pandas DataFrame with a string column of comma-separated numbers (for example `"3,5,10"`), add a column holding the integer average of each row. Handle empty strings, whitespace, malformed tokens and NaN. Reported alongside the CV-variance question (Q41) in a live two-question HackerRank screen. [Reported] [54]
    - Do it vectorized rather than with a Python loop over rows. Which is faster here, and why?
    - What rounding does "integer average" mean, and how do you confirm it?

53. **Raw event processing (ML/DS phone screen).** Given raw event rows `(user_id, ts, event_type)` with `event_type` in {search, view, wishlist, book}, unsorted and with exact duplicates, drop duplicates, sessionize per user so that a gap of more than 30 minutes starts a new session, and output one row per session (user_id, session_id, start, end, n_events, n_views, booked) plus per-user totals (sessions, median session length, booking rate). [Reported] [56][82]
    - How do you validate that your transformation did not silently drop or double-count users?
    - The file does not fit in memory. How do you restructure the code?

54. **Consecutive-day activity in SQL.** Given `impressions(user_id, listing_id, ts)`, return the users who have at least one impression on each of seven or more consecutive calendar days (UTC). [Reported] [82]
    - How do you handle several impressions per day and time-zone boundaries?

55. **Listing metrics in SQL.** Given `listings(listing_id, city, price)` and `calendar(listing_id, date, status)` with status in {booked, available, blocked}, compute the average vacant (available, not booked) days per listing over a given month, and the median price per city. [Reported] [82]
    - How do you define "vacant" for a night the host blocked?

56. **Cumulative booking metrics.** Given `bookings(guest_id, booking_id, ts)` and `events(guest_id, event_id, ts)`, for each event compute the guest's number of bookings strictly before the event, and the first and last booking times before it (NULL if none), so the columns are point-in-time correct model features. [Reported] [82]

57. **Pandas cleaning + GBDT screen.** Given bookings and listings DataFrames, clean them (nulls, dtypes), aggregate (booking rate per city and week), join with reviews, engineer features (groupby aggregates, date features, one-hot encoding), train a gradient-boosting model and report suitable metrics. Explain the split and the metric. [Likely] [62]
    - Which of your aggregates would leak the label if computed over the whole table?

58. **Point-in-time listing features.** From raw search-log and booking tables, compute per-listing impressions, CTR, booking rate and average price over the trailing 30 days as of each search, with no leakage, using pandas or SQL window functions. [Likely] [62]
    - How do you test that no feature used information from after the label timestamp?

59. **Search-to-booking attribution.** Given `searches(search_id, user_id, ts, market)`, `impressions(search_id, listing_id, position)` and `bookings(user_id, listing_id, ts)`, attribute each booking to the user's most recent search within 7 days that impressed that listing. Output the per-market search-to-booking conversion rate and the average position of booked listings, in vectorized pandas. [Likely] [62]
    - Use `merge_asof` or a window join? What happens with ties in timestamps?

60. **Booking-records practical coding.** Given `listings(listing_id, host_id)` and `bookings(booking_id, listing_id, check_in, check_out, updated_at)` where a booking may appear several times (keep the latest `updated_at`), output bookings per host, and occupancy per listing per month as booked nights divided by nights in the month, with `[check_in, check_out)` half-open. [Likely] [61]
    - A booking spans a month boundary. How do you split its nights?

61. **Calendar availability.** Given existing bookings `[check_in, check_out)` for a listing, decide whether a requested stay can be accepted, processing requests in arrival order. Follow-up: return the minimum number of listings needed so that no guest is double-booked. [Likely] [70][75]
    - Are dates half-open nights, and how does that change your boundary tests?

---

## D. Experimentation and statistics in code

62. **A/B test analysis.** Given per-user rows (variant, sessions, bookings, revenue), compute absolute and relative lift in booking conversion, a two-proportion z-test p-value and a 95% confidence interval; a bootstrap CI for revenue per user; and a chi-square sample-ratio-mismatch check against the expected split. State whether you would ship. [Likely] [1][67][68]
    - A user appears in both variants. What do you do with them?
    - Revenue is heavy-tailed. Do you cap, log-transform or switch tests?

63. **Conversion metrics from raw tables.** Given `assignments(user_id, variant, assigned_at)`, `visits(user_id, ts)` and `bookings(user_id, ts)`, compute per variant the share of assigned users who visited and the share who booked after assignment, counting only activity at or after `assigned_at`. [Reported] [82]
    - Is the denominator visits, unique visitors or searches, and how must that match the randomization unit?

64. **Panel regression.** From `users(user_id, signup_date)` and `bookings(user_id, ts, amount)`, build a user-by-week panel in SQL (zero rows included), then in Python regress weekly bookings on a treatment indicator with user and week fixed effects. [Reported] [82]
    - Which fixed effects do you include, and how do you cluster the standard errors?

65. **Team-draft interleaving.** Given ranked lists A (control) and B (treatment), produce one merged list of length k with no duplicates, recording which ranker contributed each item. Given clicks or bookings on the merged list, compute the per-query winner, the overall preference and a sign-test p-value. [Likely] [94]
    - What can interleaving not tell you that an A/B test can?

66. **Counterfactual offline evaluation.** Given logged impressions with logging-policy propensities, observed bookings and a new ranker's scores, implement IPS and self-normalized IPS estimates of the new ranker's expected bookings at top-k, and compare to the logged policy with bootstrap CIs. [Likely] [95]
    - What happens to variance when the new ranker places items the old one rarely showed?

67. **Ratio metrics.** Given per-user rows `(variant, searches, bookings)`, implement the delta-method variance of bookings per search (sum of bookings over sum of searches) per variant with user-level randomization, compute a 95% CI on the difference, and compare it with a user-level bootstrap. [Likely]
    - Why is resampling individual searches wrong here?

68. **Sample size and power.** Write `sample_size(p0, rel_lift, alpha=0.05, power=0.8, two_sided=True) -> int` returning the per-arm sample size to detect a relative lift `rel_lift` on a baseline conversion rate `p0` with a two-proportion z-test. [Likely]
    - How does the answer change for a one-sided test or a ratio metric?

69. **CUPED.** Given per-user arrays `y` (in-experiment metric), `x` (same metric in the pre-period) and `variant`, compute `theta = cov(x, y) / var(x)` pooled across variants, the adjusted metric `y - theta * (x - mean(x))`, the treatment effect and CI before and after adjustment, and the variance reduction. [Likely]
    - What happens for new users with no pre-period data?

70. **Deterministic bucketing.** Write `assign(user_id: str, salt: str, weights: dict[str, float]) -> str` that hashes `salt + user_id` into `[0, 1)` and returns the variant whose cumulative weight interval contains it; the same inputs always give the same variant. [Likely]
    - How do you show assignment is uniform and independent across two experiments?
    - How do you ramp from 5% to 50% without reassigning already-exposed users?

71. **Multiple testing.** Write `benjamini_hochberg(pvals: list[float], q=0.05) -> list[bool]` that returns, in input order, which metric p-values from one experiment stay significant at FDR q. [Likely]
    - When would you prefer Bonferroni or a single pre-declared primary metric?

---

## E. Debugging and production-readiness probes

72. **Debug a model.** You get a training script for a booking-probability classifier whose validation AUC is suspiciously high, or which trains but underperforms online. Find and fix the bugs (target leakage from post-booking features, a random split instead of a time split, unscaled features, misaligned labels or a wrong loss), rerun and report corrected metrics. A process guide lists "debugging a model" among the ML loop's rounds. [Likely] [62][43]
    - Which single check would you run first, and why?
    - How do you tell a data-pipeline bug from a genuine modeling problem?

73. **Listing or photo quality classifier.** Given listing metadata and labels, write a training and evaluation script for an imbalanced binary quality classifier and report precision and recall at a chosen threshold. [Likely] [63][67]
    - How do you choose the threshold if removing a good listing is far costlier than missing a bad one?

74. **Fraud, risky-booking or damage-claim model in code.** Train and evaluate a trust classifier (fraud, risky booking, or property-damage claim triage) with rare, delayed labels and a human-review threshold. Damage-claim and pricing ML problems are reported in a full-time ML onsite; the coding framing is inferred. [Likely] [58][77]
    - How do you build a validation set when labels mature 60 days after the booking?

75. **Cross-cutting probes in every coding round.** [Reported] [1][4][80][81]
    - Before you code: what are the input format, output format and tie-breaking rules?
    - Write your own tests now. Which edge cases did you choose, and why?
    - What are the time and space complexity at the largest stated input?
    - A new requirement arrives (return the path, k listings, stream the input). How much code changes?
    - This runs behind an API endpoint tomorrow. What do you validate, log or cache?

---

## F. Level calibration (Senior bar)

Harder probes to run on yourself after the core bank. Each assumes 45-60 minutes, runnable code and your own tests.

76. **Split stay at scale.** 10^5 listings, a 365-day horizon and 10^4 range queries per second. Precompute so that each query `[start, end)` returns every valid two-listing split in output-sensitive time, and state memory use. Then support availability updates without a full rebuild. [Likely]
77. **Menu order with quantities.** Extend Q2 so wanted items have counts up to 5 and up to 6 distinct items, so the bitmask becomes a mixed-radix state. Return the cheapest order and prove the state count fits in memory. Add a property-based test that compares against brute force on random small menus. [Likely]
78. **Streaming pagination.** Redo Q5 as a generator that consumes listings from an iterator in score order and yields pages lazily with O(page_size x hosts on page) memory, keeping the same output as the batch version. Write a test that checks equivalence on 1,000 random inputs. [Likely]
79. **Redaction with patterns.** Extend Q9 so terms can be regexes (emails, phone numbers) mixed with plain strings, matches can span line breaks, and the function returns both the redacted text and a list of `(start, end, term)` spans in the original text. Keep it linear in document length for 10^4 plain terms. [Likely]
80. **Vectorized kNN under a memory cap.** Extend Q35 to n = 10^6 training rows, m = 10^4 queries and d = 64 within 2 GB of RAM: chunked distance computation, `argpartition` for top-k and a distance-weighted vote. Report time and memory, and test against a brute-force reference on a small sample. [Likely]
81. **Mini-batch k-means for streams.** Implement mini-batch k-means with k-means++ seeding on a first batch, per-centroid learning rates and empty-cluster reseeding. Make results reproducible with a seed and test convergence on synthetic blobs with known centers. [Likely]
82. **Exact AUC and NDCG with ties and weights.** Implement ROC AUC in O(n log n) with tied scores (counting ties as half) and per-row sample weights, and NDCG@k where tied scores are ordered by expected gain over all tie permutations. Validate against scikit-learn on 100 random cases. [Likely]
83. **Leakage-safe time-series CV.** Write a splitter for booking data that is grouped by `guest_id`, ordered by `ts`, and has a label that matures 30 days later. Each fold trains on data before a cutoff minus a 30-day embargo and validates on the next window with no guest in both. Add tests that fail if any guest or future row leaks. [Likely]
84. **Calibrated threshold service.** Package Q42 as a class `ThresholdPolicy.fit(y_true, y_score, sample_weight)` with Platt or isotonic recalibration, a precision-constrained threshold, `predict(scores)`, JSON (de)serialization and input validation (NaNs, a single class, empty input). Write unit tests for each failure mode. [Likely]
85. **IPS evaluator with clipping.** Extend Q66 with weight clipping at a configurable cap, doubly robust estimation using a supplied reward model, and a bootstrap CI that resamples whole searches. Show on a simulation where the true value is known that clipping trades bias for variance. [Likely]

---

## References

1. "Airbnb SWE Interview Breakdown: The DP questions they asked + How I passed." Blind, 2026. https://www.teamblind.com/post/airbnb-swe-interview-breakdown-the-dp-questions-they-asked-how-i-passed-stpegz67
2. "Airbnb" (technical screen, senior; Property Booking Optimizer). LeetCode Discuss, 2025. https://leetcode.com/discuss/post/6717917/airbnb-technical-screen-interview-senior-fzjk/comments/2978747/
3. "Airbnb" (senior loop; hotel allocation). LeetCode Discuss, 2026. https://leetcode.com/discuss/post/7517460/
4. "Airbnb Software Engineer Interview Experience — Bombed the Live Bug-Free Hotel Split-Stay Problem." PracHub, 2025. https://prachub.com/interview-experiences/airbnb-software-engineer-interview-experience-bombed-the-live-bug-free-hotel-split-stay-problem
5. "airbnb Tech Phone Screen: API Design for Split Stay Availability." 1point3acres, 2025. https://www.1point3acres.com/interview/thread/1144028
6. "Airbnb Interview Questions" (page 3; split-stay HTTP API, graph and shortest-path puzzles). PracHub. https://prachub.com/companies/airbnb?page=3
7. 1point3acres post #13 (English mirror): two coding rounds, split stay with bitmask bonus, three-PR code review. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/13.md
8. 1point3acres post #44 (mirror): split stay; O(N^2) judged not good enough. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/44.md
9. 1point3acres post #15 (mirror): hotel split stay; "bug free, all cases must run". FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/15.md
10. 1point3acres post #38 (mirror): group-size property selection DP. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/38.md
11. 1point3acres post #10 (mirror): menu minimum cost; DFS then bitmask DP. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/10.md
12. "Airbnb | Phone screen | Combination Sum" (floating-point menu). LeetCode Discuss, 2017. https://leetcode.com/discuss/interview-question/124782/Airbnb-or-Phone-screen-or-Combination-Sum/
13. "airbnb phone interview" (text justification; spend all of X money on menu items). LeetCode Discuss, 2020. https://leetcode.com/discuss/interview-question/914253/airbnb-phone-interview
14. "Airbnb Software Engineer Fulltime Tech Phone Screen Interview: Menu Combination Question." 1point3acres. https://www.1point3acres.com/interview/thread/917068
15. "Airbnb Technical Phone Interview: Menu Combination DFS Question." 1point3acres. https://www.1point3acres.com/interview/thread/1058797
16. "Airbnb Tech Phone Screen: Menu Combination Problem Explained." 1point3acres. https://www.1point3acres.com/interview/thread/707167
17. "Airbnb Technical Phone Screen Interview Experience: Combination Sum Variation." 1point3acres. https://www.1point3acres.com/interview/thread/1084220
18. "Airbnb | Phone screen | Paginate Listings." LeetCode Discuss, 2018. https://leetcode.com/discuss/interview-question/125518/1st-telephone-interview-paginate-listings/ (also at https://leetcode.com/discuss/interview-experience/125518/)
19. Glassdoor review RVW17324807 (host-listing pagination; SWE intern). Glassdoor. https://fr.glassdoor.ca/Entretien/Airbnb-Entretien-E391850-RVW17324807.htm
20. Glassdoor review RVW17284596 (pagination, Palindrome Pairs, IP to CIDR). Glassdoor, 2017. https://www.glassdoor.sg/Interview/Airbnb-Interview-E391850-RVW17284596.htm
21. "Airbnb interview experience" (Nov 2022; Palindrome Pairs via trie). Naukri Code360, 2022. https://www.naukri.com/code360/interview-experiences/airbnb/airbnb-interview-experience-nov-2022-exp-0-2-years
22. "Airbnb | Phone screen | Iterator over list of lists." LeetCode Discuss, 2019. https://leetcode.com/discuss/interview-question/352704/airbnb-phone-screen-iterator-over-list-of-lists
23. "Airbnb Fulltime Software Engineer Tech Phone Screen: 2D Array Iterator Implementation." 1point3acres. https://www.1point3acres.com/interview/thread/714458
24. "AirBnb SSE rejected" (minimum window substring; sliding puzzle). LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/2133358/airbnb-sse-rejected/
25. "Airbnb | SDE Intern | India | July 2022 [Offer]" (sliding puzzle, DFS then BFS). LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/2345847/
26. "AirBnb Software Engineer L4 Bangalore" (cheapest flights within K stops with the stops). LeetCode Discuss, 2022. https://leetcode.com/discuss/post/1775088/airbnb-software-engineer-l4-bangalore-by-yxg5/
27. Airbnb online assessment post (cheapest flights within K stops). LeetCode Discuss. https://leetcode.com/discuss/interview-experience/1632963/
28. Glassdoor review RVW13237989 (cheapest itinerary between two cities). Glassdoor, 2016. https://fr.glassdoor.ch/Entretien/Airbnb-Entretien-E391850-RVW13237989.htm
29. "Airbnb phone - Smallest Common Region & 10 Wizards" (Senior SE). LeetCode Discuss, 2020. https://leetcode.com/discuss/interview-question/983456/airbnb-phone-smallest-common-region-10-wizards
30. "Airbnb phone interview" (grid with walls, keys and doors). LeetCode Discuss, 2019. https://leetcode.com/discuss/interview-question/418142/Airbnb-phone-interview
31. Glassdoor review RVW16192336 (Senior SWE; LeetCode-hard maze with BFS). Glassdoor. https://www.glassdoor.com.au/Interview/Airbnb-Interview-E391850-RVW16192336.htm
32. Glassdoor review RVW91124632 (senior SWE; hard maze problem). Glassdoor, 2025. https://www.glassdoor.sg/Interview/Airbnb-Interview-E391850-RVW91124632.htm
33. "Airbnb | Phone Screen | Find Strings That Differ by 1 Character." LeetCode Discuss, 2019. https://leetcode.com/discuss/interview-question/382955/Airbnb-or-Phone-Screen-or-Find-Strings-That-Differ-by-1-Character
34. "Airbnb phone screen: fixed-length array queue." LeetCode Discuss, 2019. https://leetcode.com/discuss/interview-question/341250/airbnb-phone-screen-fixed-length-array-queue
35. "Airbnb | Phone Screen | Calculate Board Score." LeetCode Discuss, 2019. https://leetcode.com/discuss/interview-question/341292/Airbnb-or-Phone-Screen-or-Calculate-Board-Score
36. "Airbnb Fulltime Machine Learning Tech Phone Screen Experience" (board score). 1point3acres. https://www.1point3acres.com/interview/thread/1168518
37. "Airbnb onsite - rooms and keys II." LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-question/2160577/airbnb-onsite-rooms-and-keys-ii
38. "AirBnB onsite similar to trapping rain water, found it too hard." LeetCode Discuss, 2021. https://leetcode.com/discuss/interview-question/1193518/airbnb-onsite-similar-to-trapping-rain-water-found-it-too-hard
39. "Interviewing at Airbnb." Blind. https://www.teamblind.com/post/interviewing-at-airbnb-g58oa2au
40. Glassdoor review RVW9496934 (Text Justification, Alien Dictionary; San Francisco). Glassdoor, 2015. https://fr.glassdoor.ch/Entretien/Airbnb-Entretien-E391850-RVW9496934.htm
41. "Airbnb Software Engineer Tech Phone Screen Interview" (text formatter). 1point3acres. https://www.1point3acres.com/interview/thread/728849
42. 1point3acres thread 197945 (text justification; city ranking by topological sort). 1point3acres. https://www.1point3acres.com/bbs/thread-197945-1-1.html
43. "Airbnb Senior+ Machine Learning Engineer Interview Experience — Search Ranking and Team Matching." PracHub, 2026. https://prachub.com/interview-experiences/airbnb-seniorplus-machine-learning-engineer-interview-experience-search-ranking-and-team-matching
44. "Airbnb Senior Machine Learning Engineer Interview Experience — Four ML System Design Rounds, Rejected on Calibration." PracHub, 2026. https://prachub.com/interview-experiences/airbnb-senior-machine-learning-engineer-interview-experience-four-ml-system-design-rounds-rejected-on-calibration
45. "Airbnb Data Scientist Algorithms interview" (kNN, k-means, IoU, ANN, convolution, z-score from scratch). Blind, 2023. https://www.teamblind.com/post/airbnb-data-scientist-algorithms-interview-n5rfjn3t
46. "Failed Airbnb on-site coding interview for an MLE position, how much hope do I have?" Blind, 2023. https://www.teamblind.com/post/failed-airbnb-on-site-coding-interview-for-an-mle-position-how-much-hope-do-i-have-avk5p8ly
47. "Airbnb on site interview" (senior ML/DS: two ML rounds, two DS coding rounds). Blind, 2022. https://www.teamblind.com/post/airbnb-on-site-interview-iyxjf46j
48. "ML Experience round at Airbnb." Blind. https://www.teamblind.com/post/ml-experience-round-at-airbnb-sorv1g3x
49. "Airbnb MLE design rounds." Blind. https://www.teamblind.com/post/airbnb-mle-design-rounds-pulb6mpe
50. "Airbnb Staff MLE interview." Blind. https://www.teamblind.com/post/airbnb-staff-mle-interview-0cfhvln7
51. "Airbnb senior MLE interview prep." Blind. https://www.teamblind.com/post/airbnb-senior-mle-interview-prep-n1mdsd3b
52. "Airbnb Code Review Round." Blind. https://www.teamblind.com/post/Airbnb-Code-Review-Round-OnE7Ex7H
53. "Interviewing for Airbnb ML platform role." Blind, 2025. https://www.teamblind.com/post/interviewing-for-airbnb-ml-platform-role-8in7czso
54. Glassdoor review RVW38727859 (live HackerRank: string-column average; preprocessing + classifier + metric variance). Glassdoor. https://www.glassdoor.com.ar/Entrevista/Airbnb-Entrevista-E391850-RVW38727859.htm
55. "airbnb Machine Learning Tech Phone Screen Review" (graph, DFS, connected components). 1point3acres. https://www.1point3acres.com/interview/thread/1169233
56. "Airbnb Machine Learning Data Science Interview Experience - Tech Phone Screen" (Python data processing). 1point3acres. https://www.1point3acres.com/interview/thread/1124986
57. "Airbnb Fulltime Machine Learning Engineering Onsite Interview Overview" (ML design, pricing prediction, coding). 1point3acres, 2025. https://www.1point3acres.com/interview/thread/1114733
58. "Airbnb Machine Learning Fulltime Onsite Interview Experience and Insights" (coding tests; property damage claims and pricing). 1point3acres. https://www.1point3acres.com/interview/thread/1162455
59. "My Fulltime Machine Learning Onsite Interview Experience at Airbnb" (practical ML rounds and coding challenges). 1point3acres. https://www.1point3acres.com/interview/thread/797269
60. "Airbnb Onsite Interview, Data Science" (take-home: Python, ML, tech report). LeetCode Discuss, 2019. https://leetcode.com/discuss/post/357222/airbnb-onsite-interview-data-science/
61. "Airbnb Senior Data Engineer Bangalore Reject." LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/1752517/airbnb-senior-data-engineer-bangalore-reject/
62. "Airbnb Machine Learning Engineer Interview Guide." Interview Query, 2025. https://www.interviewquery.com/interview-guides/airbnb-machine-learning-engineer
63. "Airbnb Machine Learning Engineer Interview Questions." PracHub. https://prachub.com/companies/airbnb/positions/machine-learning-engineer
64. "Airbnb Interview Questions" (simulate pouring water onto terrain). PracHub, 2025. https://prachub.com/companies/airbnb
65. "Airbnb Interview Questions" (page 4; split-stay / exact-layover booking logic). PracHub. https://prachub.com/companies/airbnb?page=4
66. "Airbnb Interview Questions" (page 5; query-string parser). PracHub. https://prachub.com/companies/airbnb?page=5
67. "Airbnb Interview Questions" (page 6; SQL/Python experiment analyses; photo/listing quality). PracHub. https://prachub.com/companies/airbnb?page=6
68. "Airbnb Machine Learning Interview Questions." PracHub, 2026. https://prachub.com/companies/airbnb/categories/machine-learning
69. "Airbnb Software Engineer Coding & Algorithms Interview Questions." PracHub. https://prachub.com/companies/airbnb/positions/software-engineer/categories/coding-and-algorithms
70. Airbnb problem list (water pouring, text justification, KV store, code review, lodging/booking, URL parsing). 1point3acres, 2025. https://www.1point3acres.com/interview/problems/company/Airbnb
71. Glassdoor review RVW5079102 (CSV parser in 45 minutes). Glassdoor, 2014. https://www.glassdoor.sg/Interview/Airbnb-Interview-E391850-RVW5079102.htm
72. "Airbnb interview questions" (57 questions). Taro. https://www.jointaro.com/interviews/companies/airbnb
73. "Airbnb LeetCode" (2019 company-tag list). Scribd, 2019. https://www.scribd.com/document/486147929/Airbnb-LeetCode
74. "Airbnb interview process and top questions." Educative, 2025. https://educative.io/blog/airbnb-interview-process-and-questions
75. "Blind 75 vs. LeetCode patterns: Effective for Airbnb interviews?" Educative, 2025. https://www.educative.io/blog/blind-75-vs-leetcode-patterns-for-airbnb-coding-interviews
76. "Get a Job at Airbnb: Interview Process and Top Questions." Exponent. https://tryexponent.com/blog/airbnb-interview-process
77. "Airbnb Machine Learning Engineer at a Glance." DataInterview. https://www.datainterview.com/blog/airbnb-machine-learning-engineer-interview
78. "Airbnb Machine Learning Engineer." CleverPrep. https://www.cleverprep.com/companies/airbnb/machine-learning-engineer
79. Airbnb Machine Learning Engineer interview questions (page 6). Prepfully. https://prepfully.com/interview-questions/airbnb/machine-learning-engineer?page=6
80. "Airbnb Machine Learning Engineer Interview Process" (TrueInterview guide, mirrored; 3 candidate reports, mainly 2026 Q2). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-machine-learning-engineer.md
81. "Airbnb Software Engineer Interview Process" (TrueInterview guide, mirrored; 41 candidate reports, 2025 Q4 to 2026 Q2). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb/interview-process-software-engineer.md
82. "Airbnb" question list with report dates (algorithm, OOP and SQL, Sep 2025 to Sep 2026). GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/airbnb.md
83. "Airbnb" company guide (2025-2026 reports). jugaldb/faang-interview-prep, GitHub, 2026. https://github.com/jugaldb/faang-interview-prep/blob/main/docs/companies/airbnb.md
84. LeetCode company-tag mirror: Airbnb. liquidslr/leetcode-company-wise-problems, GitHub, 2025. https://github.com/liquidslr/leetcode-company-wise-problems/tree/main/Airbnb
85. "AirBnB Interview Questions" (classic problems). allaboutjst/airbnb, GitHub. https://github.com/allaboutjst/airbnb
86. 1point3acres post #40 (mirror): MLE-type loop; coding LC 1298; ML design (LTV, host pricing). FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/40.md
87. 1point3acres post #79 (mirror): two ML design rounds, ConnectN OOD coding, ML experience round. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/79.md
88. 1point3acres post #69 (mirror): Connect 4 phone screen with about 10 tests; onsite retryer. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/69.md
89. 1point3acres post #65 (mirror): set_value / set_sum / get_value key store. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/65.md
90. 1point3acres post #12 (mirror): menu bundles; terrain print and water drop. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/12.md
91. 1point3acres post #53 (mirror): rental-services minimum-cost variant. FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/53.md
92. 1point3acres post #37 (mirror): 16 algorithm questions with clarification tips (menu pruning and all optima, Connect 4, select property). FlowSight/coding, GitHub. https://github.com/FlowSight/coding/blob/main/company%20wise/airbnb/1p3a/37.md.md
93. "Academic publications: Airbnb Tech 2025 year in review" (NDCB metric). Airbnb Tech, 2025. https://airbnb.tech/infrastructure/academic-publications-airbnb-tech-2025-year-in-review/
94. "Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving." Airbnb Tech, 2021. https://airbnb.tech/?p=251
95. "Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking." arXiv 2508.00751, 2025. https://arxiv.org/html/2508.00751
96. "Embedding-Based Retrieval for Airbnb Search." Airbnb Tech, 2025. https://airbnb.tech/ai-ml/embedding-based-retrieval-for-airbnb-search/
97. "Beyond Pairwise Learning-To-Rank At Airbnb." Airbnb Tech, 2025. https://airbnb.tech/?p=653
