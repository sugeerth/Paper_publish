# 05. ML Coding: Implementing Algorithms and Metrics from Scratch

> Scope: the "ML coding" variant of the coding round in the Amazon L5 Applied Scientist (Applied Scientist II) onsite or virtual loop, 2025-2026. This round asks you to implement an ML algorithm, metric or training loop in plain Python or NumPy, usually with no scikit-learn or PyTorch. Questions only, with no solutions.
>
> Tags: **[Reported]** means a candidate report or an Amazon-specific prep source ties this coding task to Amazon (citation [n]). Where the source is a vendor guide or prep blog and not a first-person report, the reference list says so. **[Likely]** means an inference from the role, Amazon's teams and products, or a concept that was reported in a breadth or depth round, turned into a coding task. A [Likely] item was not reported as a coding question.
>
> Research note (read this): in this run the web-search budget was already used up and direct page fetches failed with DNS errors. This chapter therefore uses the sources collected during the research for chapters 01-04 of this bank. Those sources were mostly read through search-result extracts, not full pages. Every [Reported] tag below matches something those sources recorded. No new reports were added without a source. Treat what each source describes as the candidate's account of what happened.

## Format notes for this round

- **Where it sits in the loop.** Amazon's official AS prep page describes the onsite as four 55-minute interviews with members of the science community. Candidates for AS II usually report 5 rounds: ML breadth, ML depth, coding, science application or system design, and a Bar Raiser [1][3][11]. Most L5 loops have **one coding round**. Depending on the team and the interviewer, it can be LeetCode-style DSA, data manipulation in pandas or NumPy, or ML from scratch [11][12][13]. One Reddit-sourced comment, summarized on Glassdoor, says SDE interviewers tend to give medium DSA, while AS interviewers often give ML or data-science coding [6].
- **Which teams lean toward ML coding.** Reports from Alexa, AGI and LLM teams lean toward implementing ML from scratch [24][25]. A Blind loop summary lists "implement a simple ML algorithm or training loop from scratch (e.g., logistic regression with gradient descent, k-means)" as one coding format [12]. Vendor guides say libraries are banned so the interviewer can test whether you truly understand the method [19][21].
- **Environment.** You code in Amazon's shared editor (Livecode), with no IDE and usually no way to run the code. Amazon expects syntactically correct code, not pseudo code, and expects you to check edge cases [1][2][30]. Python is the norm for AS candidates.
- **Bar at L5.** You are expected to write correct, readable, vectorized code. Interviewers also check that you state time and space complexity, handle edge cases such as ties, empty clusters, division by zero and numerical overflow, and can test your code by tracing a small example [14][19][20][30]. Expect follow-ups that scale the solution up (batching, streaming, distributed) or tie it to an Amazon use case. Each technical round typically also spends 1-2 questions, about 5-15 minutes (some reports say longer), on Leadership Principles (LPs) [1][21].
- **Typical pacing (inferred).** About 5 minutes to clarify the task and the input/output contract, 25-35 minutes to implement, 5-10 minutes on tests and complexity, then extensions. Not every report confirms this split.

## A. Evaluation metrics from scratch

### A1. Classification metrics

1. Write a program that computes ROC-AUC from a list of scores and binary labels. [Reported] [3][4][5][17]
   - How do you handle tied scores?
   - Can you do it in O(n log n) using ranks (Mann-Whitney U) instead of building the full ROC curve?
   - What do you return if all labels are the same class?
   - How would you compute AUC over a dataset too large for one machine?
2. Compute precision, recall and F1 from predicted and true labels, without libraries. [Likely] (metric choice was reported in breadth and depth rounds [21])
   - How do you extend it to multi-class with macro, micro and weighted averaging?
   - What should the function return when there are no predicted positives?
3. Build a confusion matrix for K classes from two integer arrays, vectorized in NumPy. [Likely]
4. Given scores and labels, find the threshold that maximizes F1, or that maximizes recall while precision stays at or above a target. [Likely]
   - Can you do it in a single pass after sorting?
5. Compute the precision-recall curve and the average precision (PR-AUC). [Likely]
   - Why can ROC-AUC look optimistic under heavy class imbalance, for example fraud or abuse detection?
6. Implement binary and multi-class log loss (cross-entropy) with numerically safe clipping. [Likely] (the cross-entropy versus MSE question was reported in breadth [7])
7. Compute expected calibration error (ECE) with equal-width bins, and the data for a reliability diagram. [Likely] (ad click-through-rate (CTR) prediction and auction teams)
   - Equal-width or equal-mass bins: which would you use, and why?

### A2. Ranking and retrieval metrics

8. Compute precision@k, recall@k and NDCG@k for a ranked list of search results with graded relevance. [Likely] (Amazon search, ads and recommendations)
   - What is IDCG, and how do you handle queries with no relevant items?
   - How do you average across queries?
9. Compute mean reciprocal rank (MRR) and mean average precision (MAP) for a batch of queries. [Likely]
10. Given impressions with click logs, compute CTR per position, and suggest a simple correction for position bias. [Likely] (position bias in clickstream relevance was reported in a depth round)

### A3. Vision, NLP and forecasting metrics

11. Write a function that computes Intersection over Union (IoU) for two bounding boxes. [Reported] [19]
    - What edge cases come up when boxes do not overlap or have zero area?
    - How would you vectorize it for N x M boxes?
12. Implement non-maximum suppression (NMS) using your IoU function. [Likely] (vision teams such as Lab126, Amazon Go and robotics)
    - What is the complexity, and how would you speed it up?
13. Compute mean average precision (mAP) at an IoU threshold for an object detector over a dataset. [Likely]
14. Implement BLEU (n-gram precision with brevity penalty) or ROUGE-L for a candidate and a reference. [Likely] (Alexa, translation and LLM teams)
15. Compute perplexity from per-token log-probabilities, handling padding masks. [Likely]
16. Compute word error rate (WER) with edit distance between a hypothesis and a reference transcript. [Likely] (Alexa ASR; accented-speech ASR was reported as a science-application problem)
    - How would you also return the counts of substitutions, insertions and deletions?
17. Implement forecasting metrics: MAPE, sMAPE, and weighted quantile (pinball) loss at P50 and P90. [Likely] (Amazon supply-chain forecasting teams)
    - What goes wrong with MAPE when actuals are zero?

## B. Clustering, neighbors and similarity search

18. Implement k-means from scratch in Python or NumPy. [Reported] [12][16][17]
    - How do you initialize centroids? Can you implement k-means++?
    - When should it stop, and what do you do with an empty cluster?
    - What is the per-iteration complexity? Can you vectorize the distance computation?
    - How would you choose k?
    - How would you run it on 1 billion product embeddings (mini-batch, distributed)?
19. Implement k-means++ initialization on its own. [Likely] (follow-up to the reported k-means task)
20. Implement k-nearest neighbors classification from scratch. [Reported] [22] (prep blog labeled as an Amazon ML Scientist question)
    - How do you choose k and break ties in voting?
    - Why must features be scaled, and how do you avoid leaking test statistics into the scaling?
    - What is the query cost, and how does it change in high dimensions?
21. Extend kNN to distance-weighted regression. [Likely]
22. Compute the Chamfer distance between two point clouds efficiently, using an exact nearest-neighbor structure such as a KD-tree instead of computing all pairwise distances. [Reported] [18] (Amazon-framed practice question from a prep site)
    - How does a KD-tree perform as dimensionality grows?
23. Build a KD-tree and run a nearest-neighbor query. [Likely] (follow-up to the Chamfer task)
24. Given a query embedding and a matrix of N item embeddings, return the top-k items by cosine similarity, using a heap rather than a full sort. [Likely] (semantic search, Rufus retrieval [33])
    - What is the complexity of the full sort versus the heap?
    - What changes for 100 million items (approximate nearest neighbor: IVF, HNSW, product quantization)?
25. Implement random-hyperplane locality-sensitive hashing for near-duplicate product detection. [Likely]
26. Compute pairwise Euclidean distances between two matrices without Python loops, using the expansion ||a||^2 - 2ab + ||b||^2. [Likely] (core subroutine of the reported k-means and kNN tasks)
    - What numerical issue can produce small negative values, and how do you fix it?

## C. Linear models and optimization

27. Implement a simple ML training loop from scratch, for example logistic regression with gradient descent. [Reported] [12]
    - Derive the gradient of the log loss before you code it.
    - How do you add L2 (and L1) regularization? Should the bias be regularized?
    - How do you avoid overflow in the sigmoid?
    - How do you check convergence?
28. Write gradient descent without libraries. [Reported] [21]
    - How would you change it to mini-batch SGD with momentum?
29. Code a traditional ML classifier from scratch (the candidate chose the model). [Reported] [8]
    - Why did you choose that model, and what are its assumptions?
30. Implement linear regression with the closed-form normal equation and with gradient descent, and compare them. [Likely] (closed form versus GD and OLS were reported in breadth [13][23])
    - When is the matrix X^T X singular, and what do you do about it (ridge, pseudo-inverse)?
31. Implement multinomial (softmax) regression for K classes with a vectorized gradient. [Likely]
32. Implement the Adam optimizer update for a parameter vector. [Likely] (Adam versus SGD was reported in breadth)
    - Why is bias correction needed in the early steps?
33. Implement a numerical gradient check to verify an analytic gradient. [Likely]
34. Implement L1-regularized regression with proximal gradient (soft-thresholding) or coordinate descent. [Likely] (L1 versus L2 sparsity was reported in breadth [10][16])
35. Implement a Newton-Raphson (IRLS) step for logistic regression. [Likely]
    - When does Newton's method beat gradient descent, and what does it cost per step?
36. Implement stratified k-fold cross-validation and use it to tune the regularization strength of your logistic regression. [Likely]

## D. Trees and ensembles

37. Implement the split search for a decision-tree node: find the feature and threshold that maximize information gain or Gini reduction. [Likely] (how splits are chosen was reported in breadth [36])
    - What is the complexity per node, and how does pre-sorting help?
    - How do you handle categorical features and missing values?
38. Implement a full decision-tree classifier with recursive building, max-depth and min-samples stopping rules, and predict. [Likely]
39. Implement prediction for a trained tree stored as arrays of nodes (feature, threshold, left, right, value). [Likely]
40. Implement a random forest on top of your tree: bootstrap sampling, feature subsampling and majority voting. [Likely] (bagging versus boosting is the most-reported breadth topic)
    - How would you compute out-of-bag error?
41. Implement gradient boosting for regression with depth-1 stumps and squared loss. [Likely]
    - What changes for log loss, and what are the "pseudo-residuals"?
42. Compute permutation feature importance for any trained model. [Likely]

## E. Neural networks and backpropagation

43. Implement forward and backward passes for a 2-layer MLP in NumPy, and train it on a toy dataset. [Likely] (backpropagation for an MLP was reported in breadth [35])
    - What are the shapes of every gradient?
    - How would you initialize the weights, and why?
44. Implement a numerically stable softmax and softmax-plus-cross-entropy, with its gradient with respect to the logits. [Likely]
    - Why does the combined gradient simplify to p - y?
45. Implement ReLU, sigmoid, tanh and GELU with their derivatives. [Likely]
46. Implement dropout (inverted dropout) for training and inference. [Likely]
47. Implement the batch normalization forward pass with running statistics, and explain or implement the backward pass. [Likely] (batch norm was reported in breadth)
    - How does it behave differently at train and inference time?
48. Implement a 2D convolution (single channel, then multi-channel) with stride and padding, using loops and then im2col. [Likely] (vision teams)
49. Implement a single LSTM or GRU cell forward step in matrix form. [Likely] (LSTM math and recurrent shapes in matrix form were reported in breadth [26][32])
    - How does backpropagation through time work, and where do vanishing gradients come from?
50. Build a minimal scalar autograd engine (a Value class with +, *, tanh and backward via topological sort). [Likely]
51. Implement a simple training loop with early stopping on validation loss and a learning-rate schedule (warmup plus cosine). [Likely]

## F. Attention, transformers and LLM components

52. Implement scaled dot-product attention in NumPy. [Likely] (attention "at root level" and the math of self-attention were reported in depth and breadth rounds [27][29])
    - Why divide by sqrt(d_k)?
    - What changes if Wq = Wk = Wv? (reported as a breadth question [7])
53. Add a causal mask and a padding mask to your attention function. [Likely]
    - Where exactly is the mask applied, and why use -inf instead of 0?
54. Implement multi-head self-attention, including the reshape, split heads and output projection. [Likely]
    - What is the time and memory complexity in sequence length?
55. Implement sinusoidal positional encoding, or rotary embeddings (RoPE). [Likely]
56. Implement layer normalization and RMSNorm. [Likely]
    - Pre-norm or post-norm: what is the difference in training stability?
57. Implement a decoder step with a KV cache: given cached K and V and a new token, return the output and the updated cache. [Likely] (KV caching and decoder-only inference were reported in a 2026 breadth round [6])
    - How much memory does the cache use for a given model size and context length?
    - How would multi-query or grouped-query attention change your code?
58. Implement a sliding-window (local) attention mask, as in Longformer. [Likely] (Longformer and long-context attention were reported [6])
59. Implement beam search decoding over a toy next-token probability function. [Likely] (beam search decoding was reported as a conceptual question for an LLM-team AS II [28])
    - How do you handle end-of-sequence tokens and length normalization?
60. Implement temperature, top-k and top-p (nucleus) sampling from a logits vector. [Likely]
61. Implement byte-pair encoding (BPE): learn merges on a small corpus, then tokenize a new word. [Likely]
62. Implement a LoRA linear layer forward pass and explain which parameters are trained. [Likely] (LoRA was reported in AWS AS breadth [4])
    - How do you merge the adapter weights for inference?
63. Implement the SFT loss for a causal LM, masking prompt tokens so only response tokens count. [Likely] (SFT pipelines were reported in a depth round [21])
64. Implement the DPO loss given log-probabilities from the policy and reference models for chosen and rejected responses. [Likely] (Alexa and AGI job descriptions mention SFT, DPO and RLHF [37])
65. Implement InfoNCE (in-batch negatives) contrastive loss for a bi-encoder retriever. [Likely] (cross-encoder versus bi-encoder was reported in a 2026 round [6])

## G. Probability, sampling and statistics in code

66. Build a weighted random number generator from an array of probabilities or weights (random pick with weight). [Reported] [9] (Senior AS screen, 2026)
    - What is the complexity per sample with prefix sums plus binary search? Can you do O(1) with the alias method?
67. Implement reservoir sampling to pick k items uniformly from a stream of unknown length. [Likely]
    - How would you make it weighted?
68. Compute a bootstrap confidence interval for a metric difference between two models, such as AUC. [Likely]
69. Implement a two-sample t-test (Welch) or a two-proportion z-test for an A/B test, from scratch. [Likely] (A/B testing and p-values were reported in breadth)
    - How would you compute the required sample size for a given minimum detectable effect?
70. Implement Gaussian or multinomial Naive Bayes for text classification, with Laplace smoothing and log-space computation. [Likely] (Naive Bayes was reported in breadth [38])
71. Implement PCA via SVD, or via eigendecomposition of the covariance matrix, and project data onto the top k components. [Likely] (PCA and covariance were reported in breadth)
    - How would you implement power iteration for the top component only?
72. Implement EM for a 1-D Gaussian mixture model with two components. [Likely]
73. Implement Thompson sampling or UCB for a multi-armed bandit over ad creatives or widgets, and simulate regret. [Likely] (ads and personalization)
74. Estimate pi, or an expected value, by Monte Carlo simulation, and report its standard error. [Likely]

## H. Recommendation, search and NLP building blocks

75. Implement matrix factorization for implicit or explicit ratings with SGD or ALS. [Likely] (Amazon recommender-systems roles list collaborative filtering [34])
    - How do you handle cold-start users and items?
76. Implement item-to-item collaborative filtering: compute cosine similarity between item vectors from a sparse user-item matrix, and recommend the top N. [Likely] (Amazon's well-known item-to-item recommendations)
    - How would you scale this computation to millions of items?
77. Implement TF-IDF and BM25 scoring for a query against a small document collection. [Likely] (product search)
78. Implement a pointwise or pairwise learning-to-rank loss (for example, RankNet) and its gradient. [Likely]
79. Implement n-gram language-model counting with add-k smoothing, and score a sentence. [Likely]
80. Implement a simple word2vec skip-gram negative-sampling update for one (center, context) pair. [Likely]
81. Deduplicate near-identical product titles using normalized string keys or character shingles plus Jaccard similarity. [Likely]

## I. Data manipulation and ML pipeline coding

82. Compute a rolling average when some dates are missing. [Reported] [11]
    - Should missing days count as zero or be skipped? Defend your choice.
83. Load a dataset in pandas, clean it, engineer features, and fit a model, explaining each step. [Reported] [13][31]
    - Which steps can leak information from the test set?
84. Answer simple data-wrangling questions in Python (group-by, joins, aggregations) alongside standard DSA. [Reported] [3]
85. Return the top-K most frequent elements in a stream or array, in Python. [Reported] [21]
86. Implement a train, validation and test split that is stratified by label and grouped by user, so no user appears in two splits. [Likely]
87. Implement a time-based split and an expanding-window backtest for a demand-forecasting model. [Likely] (supply-chain forecasting)
88. Implement standard scaling and one-hot encoding as fit/transform classes that handle unseen categories. [Likely]
89. Implement mini-batch iteration over a dataset that does not fit in memory, with shuffling. [Likely]
90. Implement random oversampling, or class-weighted loss, for an imbalanced dataset, and show the effect on your metric function. [Likely] (class imbalance was reported in depth rounds [21])

## J. Round-level probes (how the interviewer pushes at L5)

91. Before writing code: what are the inputs, outputs, data types and sizes? What should happen on empty or degenerate input? [Likely] (Amazon's prep pages stress edge cases and bad input [1][2])
92. What is the time and space complexity of your implementation, and where is the bottleneck? [Likely]
93. You cannot run the code. Walk me through it on a tiny example by hand. [Likely] (third-party guides note that code is usually not executed [30])
94. How would you test this function? Which unit tests and invariants would you write? [Likely]
95. How would this change if the data were 1000x larger, or arrived as a stream? [Likely]
96. Now apply it to our team's problem. For example, where would k-means or kNN fit in product categorization or Rufus retrieval? [Likely]
97. An extra round after a split debrief: let's revisit coding with a follow-up problem. [Reported] [15]
98. Tell me about a time you dove deep into data or code to find a root cause (an LP question opening or closing the coding round). [Reported] [21]
99. Tell me about a time you had to deliver a model under a tight deadline. What corners did you refuse to cut? [Likely] (LPs "Deliver Results" and "Insist on the Highest Standards")
100. Tell me about a time you simplified a complex ML approach into something you could ship. [Likely] (LP "Invent and Simplify")

## Prep priorities (derived from the tags above)

- **Most-reported ML coding tasks:** AUC from scores and labels (1), k-means (18), logistic regression or a gradient-descent training loop (27, 28), kNN (20), IoU (11), and a generic classifier from scratch (29).
- **Most likely extensions for 2025-2026 L5 loops on LLM, Alexa or AGI teams:** attention with masks and a KV cache, sampling and beam search, and LLM losses (52-65). These are [Likely] as coding tasks, but the concepts were reported often in breadth and depth rounds.
- **Do not neglect:** data-manipulation coding (82-90). It shows up in screens and loops where the interviewer is a scientist, not an SDE.

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), accessed 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. SDE II Interview Prep, amazon.jobs (Amazon, official; Livecode editor and coding expectations). https://amazon.jobs/content/en/how-we-hire/sde-ii-interview-prep
3. Amazon Applied Scientist II Interview Experience & Questions, Glassdoor, 2025-2026 reports (AUC program; loop structure; Python data-wrangling). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
4. Amazon Web Services Applied Scientist Interview Questions, Glassdoor (AUC; transformers; LoRA). https://www.glassdoor.com/Interview/Amazon-Web-Services-Applied-Scientist-Interview-Questions-EI_IE7470741.0,19_KO20,37.htm
5. Amazon interview review RVW95575263 (Applied Scientist II), Glassdoor, 2026 (code to compute AUC from scores and labels). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW95575263.htm
6. Amazon Applied Scientist Interview Experience & Questions, Glassdoor, 2026 (decoder-only inference, KV caching, Longformer, cross-encoder vs bi-encoder; SDE vs AS interviewer coding style). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
7. Amazon Applied Scientist Interview Questions, page 3, Glassdoor, circa 2024 (Wq = Wk = Wv; cross-entropy vs MSE). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24_IP3.htm
8. Amazon interview review RVW47917252 (Applied Scientist), Glassdoor, circa 2020-2022 (code a traditional ML classifier from scratch). https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW47917252.htm
9. Amazon interview review RVW102577344 (Senior Applied Scientist, Seattle), Glassdoor, 2026 (weighted random number generator). https://www.glassdoor.co.nz/Interview/Amazon-Interview-E6036-RVW102577344.htm
10. Amazon Applied Science Intern Interview Experience & Questions, Glassdoor (logistic regression formula; L1 vs L2). https://www.glassdoor.com/Interview/Amazon-Applied-Science-Intern-Interview-Questions-EI_IE6036.0,6_KO7,29.htm
11. "Amazon Applied Scientist (L5) Virtual Onsite", Blind, circa 2025 (one coding interview, LeetCode medium; rolling average with missing dates; data-manipulation focus). https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
12. "Amazon applied scientist interview loops", Blind (coding may be ML from scratch, e.g. logistic regression with gradient descent, k-means). https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
13. "Amazon applied scientist interview-onsite", Blind, circa 2024 (data manipulation; OLS). https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
14. "Tips for coding interview, Applied Scientist at Amazon", Blind (medium/hard; clean, production-level, optimal code). https://www.teamblind.com/post/tips-for-coding-interview-applied-scientist-at-amazon-txexmtsb
15. "Amazon (AGI team) Applied Scientist II interview", Blind (extra coding round after split debrief). https://www.teamblind.com/post/Amazon-AGI-team-Applied-Scientist-II-interview-2eqBXGtT
16. Amazon Applied Scientist Interview (process, questions, prep), IGotAnOffer, 2025-2026 (vendor guide; k-means from scratch). https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
17. Amazon Applied Scientist Interview Questions & Guide 2026, Dataford, 2026 (vendor guide; k-means; "write a program to calculate AUC"). https://dataford.io/interview-guides/amazon/applied-scientist
18. Efficient Chamfer Distance (Amazon-framed practice question), Dataford. https://dataford.io/questions/efficient-chamfer-distance
19. The 2026 Amazon Senior Applied Scientist Interview Guide, Interview Kickstart, 2026 (vendor guide; IoU; library-free ML coding). https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
20. The 2026 Amazon Applied Scientist Interview Guide, Interview Kickstart, 2026 (vendor guide). https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
21. Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP, OA VO Service (vendor-published candidate debrief; gradient descent without libraries; top-K; class imbalance; SFT; LP time in technical rounds). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
22. "Amazon ML Scientist Interview Question: Program k-Nearest Neighbour from Scratch", Jaspinder Singh, Substack, 2026 (prep post, not a verified candidate report). https://jaspinders30.substack.com/p/amazon-ml-scientist-interview-question
23. Jaspinder Singh, Substack series "Amazon Applied Scientist Interview Question" (closed form vs GD; logistic regression), 2026 (prep posts). https://jaspinders30.substack.com/p/amazon-applied-scientist-interview-7cd ; https://jaspinders30.substack.com/p/how-does-logistic-regression-work
24. Amazon AGI Applied Scientist Interview Experience and Insights, 1point3acres. https://www.1point3acres.com/interview/thread/1130091
25. Amazon Applied Scientist Interview Guide, Alexa International Tech, Medium, 2025. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
26. "Amazon Applied Scientist Phone Interview", LeetCode Discuss (LSTM vs Transformers mathematically; derive losses). https://leetcode.com/discuss/post/543684/amazon-applied-scientist-phone-interview/
27. "Amazon online 1st round, Bengaluru, Applied Scientist", LeetCode Discuss, May 2025 (transformer internals, LLM theory). https://leetcode.com/discuss/post/6871782/amazon-online-1st-round-bengaluru-applie-uiwk/
28. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara), Glassdoor, 2025 (beam search decoding). https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
29. Amazon interview review RVW96903377 (Applied Scientist II, Seattle), Glassdoor, 2026 (attention at root level). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW96903377.htm
30. Amazon Coding Interview Questions: The Real Bar (2026 Guide), copilotinterview.com, 2026 (code usually not executed; trace by hand). https://copilotinterview.com/blog/amazon-coding-interview-questions
31. Amazon interview review RVW99272421 (Applied Scientist, London, July 2024), Glassdoor (pandas load, clean, feature-engineer and fit). https://static.glassdoor.it/Interview/Amazon-Interview-E6036-RVW99272421.htm
32. theprogclub, "Amazon Applied Scientist Interview Experience", Medium, circa 2023 (LSTM; backpropagation through time). https://medium.com/@theprogclub/amazon-applied-scientist-interview-experience-e38e3fb2ec48
33. Applied Scientist, Rufus Experiences Science (Job ID 2855826), amazon.jobs (team context for [Likely] retrieval and LLM items). https://amazon.jobs/en/jobs/2855826/applied-scientist-rufus-experiences-science
34. 2027 Applied Science Internship, Recommender Systems / Information Retrieval (Job ID 10564598), amazon.jobs (collaborative filtering, learning to rank). https://www.amazon.jobs/en/jobs/10564598/2027-applied-science-internship-recommender-systems-information-retrieval-machine-learning-united-states-phd-student-science-recruiting
35. "Science Breadth Interview @Amazon", Blind, 2024 (backpropagation for an MLP; write concepts mathematically). https://www.teamblind.com/post/science-breadth-interview-amazon-hfwqznrb
36. Akanksha Paul, "Interview Experience for Applied Scientist at Amazon", Medium, 2020 (how decision-tree splits are chosen). https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
37. Applied Scientist, Alexa International Tech (Job ID 10490052), amazon.jobs (SFT, DPO, RLHF). https://www.amazon.jobs/en/jobs/10490052/applied-scientist-alexa-international-tech
38. Chaitanya Mundle, "My interview experience for the Applied Scientist II role at Amazon", Medium, circa 2024 (Advertising Trust; KNN, Naive Bayes, SVM; metrics). https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
