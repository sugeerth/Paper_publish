# 05. ML Coding: Implementing Algorithms and Metrics from Scratch

> Scope: the "ML coding" variant of the coding round in the Amazon L5 Applied Scientist (Applied Scientist II) onsite or virtual loop, 2025-2026. This round asks you to implement an ML algorithm, metric, loss or training loop in plain Python, NumPy or (on LLM teams) PyTorch, usually with no scikit-learn or high-level library calls. Questions only, with no solutions.
>
> Tags: **[Reported]** means a first-person candidate report, or an aggregator that compiles candidate reports, ties this coding task to an Amazon science or ML loop (citation [n]). Where the source is an aggregator or a vendor-published debrief, the reference list says so. **[Likely]** means an inference from the role, Amazon's teams and products, or a concept that was reported in a breadth or depth round, turned into a coding task. A [Likely] item was not reported as a coding question. Items listed only by a vendor prep guide, a prep blog or a practice site, and not tied to a candidate report, are tagged [Likely] with the source noted.
>
> Research note (read this): in the review pass for this chapter (October 2026), web search was unavailable and the egress proxy refused every non-GitHub host, so the 38 references carried over from the first draft could not be re-fetched. They were not removed, because a refused connection is not evidence that a page is dead; they reflect what the chapter 01-04 research recorded, mostly from search-result extracts. The new [Reported] items below come from GitHub-hosted aggregators that were fetched and read in full during this pass [39]-[43]. Two of those pages cover Amazon's Research Scientist and Machine Learning Engineer tracks, which share the science loop format with Applied Scientist; each item says which track it came from. Treat what each source describes as the candidate's account of what happened.

## Format notes for this round

- **Where it sits in the loop.** Amazon's official AS prep page describes the onsite as four 55-minute interviews with members of the science community. Candidates for AS II usually report 5 rounds: ML breadth, ML depth, coding, science application or system design, and a Bar Raiser [1][3][11]. Most L5 loops have **one coding round**, sometimes two, and the hiring manager's round may add a short coding task [40][41]. Depending on the team and the interviewer, it can be LeetCode-style DSA, data manipulation in pandas or NumPy, or ML from scratch [11][12][13]. One Reddit-sourced comment, summarized on Glassdoor, says SDE interviewers tend to give medium DSA, while AS interviewers often give ML or data-science coding [6]. One aggregator says some applied-science loops had no algorithm problems at all, only hand-written sigmoid, softmax and attention during the technical discussion [41].
- **Which teams lean toward ML coding.** Reports from Alexa, AGI and LLM teams lean toward implementing ML from scratch [24][25]. Amazon AGI reports include "debug a broken Transformer implementation", "build/train a small classifier live" and "reinforcement learning fundamentals + coding" [43]. A Blind loop summary lists "implement a simple ML algorithm or training loop from scratch (e.g., logistic regression with gradient descent, k-means)" as one coding format [12]. Vendor guides say libraries are banned so the interviewer can test whether you truly understand the method [19][21]; an Amazon online-assessment softmax task explicitly says "do not rely on any third-party libraries" [42].
- **Environment.** You code in Amazon's shared editor (Livecode), with no IDE and usually no way to run the code. Amazon expects syntactically correct code, not pseudo code, and expects you to check edge cases [1][2][30]. Python is the default; NumPy and PyTorch are expected for numerical tasks [40].
- **Bar at L5.** You are expected to write correct, readable, vectorized code. Aggregated reports list what interviewers check: numerical stability (subtract the max before exponentiating, stable variance formulas), vectorization when asked, linking statistics to their maximum-likelihood estimators, and edge cases such as all-zero weights or a single element, stated without prompting [40][41]. Expect follow-ups that scale the solution up (batching, streaming, distributed) or tie it to an Amazon use case.
- **Time pressure from LPs.** Each technical round typically opens with Leadership Principle (LP) questions, often 20-30 minutes. Several reports describe only about 15 minutes left for the coding task after long LP answers, and a rejection that followed [1][21][40][41]. Keep each LP story near five minutes.
- **Typical pacing (inferred).** About 5 minutes to clarify the task and the input/output contract, 15-30 minutes to implement depending on how long LPs ran, 5 minutes on tests and complexity, then extensions. Not every report confirms this split.

## A. Evaluation metrics from scratch

### A1. Classification metrics

1. Write a program that computes ROC-AUC from a list of scores and binary labels. [Reported] [3][4][5][17]
   - How do you handle tied scores?
   - Can you do it in O(n log n) using ranks (Mann-Whitney U) instead of building the full ROC curve?
   - What do you return if all labels are the same class?
   - How would you compute AUC over a dataset too large for one machine, for example with a fixed-bin histogram of scores per class?
2. Compute precision, recall and F1 from predicted and true labels, without libraries. [Likely] (metric choice was reported in breadth and depth rounds [21]; precision, recall and AUC for imbalanced fraud data were reported in Amazon science screens [40])
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
8. Fit temperature scaling for a trained classifier: find the single scalar T that minimizes validation negative log-likelihood, by gradient descent or a 1-D search, and apply it to the logits. [Likely] (temperature scaling is named among the "numerical care" points interviewers look for [41])
   - Why does temperature scaling leave accuracy and AUC unchanged?
9. Implement isotonic regression with the pool-adjacent-violators algorithm to calibrate CTR scores. [Likely] (ads calibration)
   - When would you prefer it to Platt or temperature scaling, and what is its overfitting risk?

### A2. Ranking and retrieval metrics

10. Compute precision@k, recall@k and NDCG@k for a ranked list of search results with graded relevance. [Likely] (Amazon search, ads and recommendations)
    - What is IDCG, and how do you handle queries with no relevant items?
    - How do you average across queries?
11. Compute mean reciprocal rank (MRR) and mean average precision (MAP) for a batch of queries. [Likely]
12. Given impressions with click logs, compute CTR per position, and suggest a simple correction for position bias. [Likely] (position bias in clickstream relevance was reported in a depth round)

### A3. Vision, NLP and forecasting metrics

13. Write a function that computes Intersection over Union (IoU) for two bounding boxes. [Likely] (listed as an Amazon Senior AS coding question by a vendor guide [19], not by a candidate report)
    - What edge cases come up when boxes do not overlap or have zero area?
    - How would you vectorize it for N x M boxes?
14. Implement non-maximum suppression (NMS) using your IoU function. [Likely] (vision teams such as Lab126, Amazon Go and robotics)
    - What is the complexity, and how would you speed it up?
15. Compute mean average precision (mAP) at an IoU threshold for an object detector over a dataset. [Likely]
16. Implement BLEU (n-gram precision with brevity penalty) or ROUGE-L for a candidate and a reference. [Likely] (Alexa, translation and LLM teams)
17. Compute perplexity from per-token log-probabilities, handling padding masks. [Likely]
18. Compute word error rate (WER) with edit distance between a hypothesis and a reference transcript. [Likely] (Alexa ASR; accented-speech ASR was reported as a science-application problem)
    - How would you also return the counts of substitutions, insertions and deletions?
19. Implement forecasting metrics: MAPE, sMAPE, and weighted quantile (pinball) loss at P50 and P90. [Likely] (Amazon supply-chain forecasting teams)
    - What goes wrong with MAPE when actuals are zero?

## B. Clustering, neighbors and similarity search

20. Implement k-means from scratch in Python or NumPy. [Reported] [12][16][17][39] (the aggregator listing records a December 2025 Amazon sighting of "Implement K-means and solve interval/frequency tasks" [39])
    - How do you initialize centroids? Implement k-means++ seeding as a separate function.
    - When should it stop, and what do you do with an empty cluster?
    - What is the per-iteration complexity? Can you vectorize the distance computation?
    - How would you choose k?
    - How would you run it on 1 billion product embeddings (mini-batch, distributed)?
21. Implement k-nearest neighbors classification from scratch. [Likely] (titled as an Amazon ML Scientist question by a prep blog [22], not a verified candidate report; kNN appeared in an AS II breadth report [38])
    - How do you choose k and break ties in voting?
    - Why must features be scaled, and how do you avoid leaking test statistics into the scaling?
    - What is the query cost, and how does it change in high dimensions?
    - Extend it to distance-weighted kNN regression.
22. Compute the Chamfer distance between two point clouds efficiently, using an exact nearest-neighbor structure such as a KD-tree instead of computing all pairwise distances. [Likely] (Amazon-framed practice question on a prep site [18], not a candidate report)
    - How does a KD-tree perform as dimensionality grows?
23. Build a KD-tree and run a nearest-neighbor query. [Likely] (follow-up to the Chamfer task)
24. Given a query embedding and a matrix of N item embeddings, return the top-k items by cosine similarity, using a heap rather than a full sort. [Likely] (semantic search, Rufus retrieval [33])
    - What is the complexity of the full sort versus the heap?
    - What changes for 100 million items (approximate nearest neighbor: IVF, HNSW, product quantization)?
25. Implement random-hyperplane locality-sensitive hashing for near-duplicate product detection. [Likely]
26. Compute pairwise Euclidean distances between two matrices without Python loops, using the expansion ||a||^2 - 2ab + ||b||^2. [Likely] (core subroutine of the reported k-means task)
    - What numerical issue can produce small negative values, and how do you fix it?
27. Implement greedy maximal marginal relevance (MMR) re-ranking: from N candidates with relevance scores and embeddings, pick k items that trade off relevance against similarity to items already picked. [Likely] (Amazon Science's MUSS work targets relevance-plus-diversity subset selection for recommendations [44])
    - What is the complexity of the greedy loop, and how would you scale it to millions of candidates?

## C. Linear models, optimization and estimation

28. Implement a simple ML training loop from scratch, for example logistic regression with gradient descent. [Reported] [12]
    - Derive the gradient of the log loss before you code it.
    - How do you add L2 (and L1) regularization? Should the bias be regularized?
    - How do you avoid overflow in the sigmoid?
    - How do you check convergence?
29. Write gradient descent without libraries, and explain each line. [Reported] [21][39] (vendor-published debrief [21]; "Write and explain gradient descent pseudocode" in an Amazon aggregator listing, undated [39])
    - How would you change it to mini-batch SGD with momentum?
30. Code a traditional ML classifier from scratch, or build and train a small classifier live (the candidate chose the model). [Reported] [8][43] (the live-classifier task is from an Amazon AGI aggregator entry [43])
    - Why did you choose that model, and what are its assumptions?
    - How do you show the interviewer it is learning, with no plotting tools?
31. Implement linear regression with the closed-form normal equation and with gradient descent, and compare them. [Likely] (closed form versus GD and OLS were reported in breadth [13][23]; least-squares assumptions were reported in Amazon science screens [40])
    - When is the matrix X^T X singular, and what do you do about it (ridge, pseudo-inverse)?
32. Implement multinomial (softmax) regression for K classes with a vectorized gradient. [Likely]
33. Implement the Adam optimizer update for a parameter vector. [Likely] (Adam versus momentum SGD, and Adam versus AdamW, were reported in Amazon screens and depth rounds [40][41])
    - Why is bias correction needed in the early steps?
    - Change it to AdamW: where does decoupled weight decay enter the update, and why does it differ from L2 in the loss?
34. Implement a numerical gradient check to verify an analytic gradient. [Likely]
35. Implement L1-regularized regression with proximal gradient (soft-thresholding) or coordinate descent. [Likely] (L1 versus L2 sparsity was reported in breadth [10][16])
36. Implement a Newton-Raphson (IRLS) step for logistic regression. [Likely]
    - When does Newton's method beat gradient descent, and what does it cost per step?
37. Implement stratified k-fold cross-validation and use it to tune the regularization strength of your logistic regression. [Likely]
38. Write a NumPy function for the mean and variance of an array, then relate them to maximum-likelihood estimation. [Reported] [39][40][41] (Research Scientist and MLE loops; aggregator sighting dated September 2026 [39])
    - Why do the sample variance and the Gaussian MLE of the variance differ, and which is biased?
    - How do you compute variance in a numerically stable way, and along a chosen axis without loops?
39. Write the negative log-likelihood of a Bernoulli model and of a Gaussian model in code, and show that they reduce to log loss and squared error. [Likely] (squared loss as a Gaussian likelihood and log loss from the Bernoulli likelihood were reported in Amazon science screens [40])
40. Implement Welford's online algorithm for a running mean and variance over a stream, and merge the statistics from two shards. [Likely] (follow-up to item 38; "stable variance formulas" are named as an evaluation point [40])

## D. Trees and ensembles

41. Implement the split search for a decision-tree node: find the feature and threshold that maximize information gain or Gini reduction. [Likely] (how splits are chosen was reported in breadth [36])
    - What is the complexity per node, and how does pre-sorting help?
    - How do you handle categorical features and missing values?
42. Implement a full decision-tree classifier with recursive building, max-depth and min-samples stopping rules, and predict. [Likely]
43. Implement prediction for a trained tree stored as arrays of nodes (feature, threshold, left, right, value). [Likely]
44. Implement a random forest on top of your tree: bootstrap sampling, feature subsampling and majority voting. [Likely] (bagging versus boosting is the most-reported breadth topic)
    - How would you compute out-of-bag error?
45. Implement gradient boosting for regression with depth-1 stumps and squared loss. [Likely]
    - What changes for log loss, and what are the "pseudo-residuals"?
46. Compute permutation feature importance for any trained model. [Likely] (impurity versus permutation importance was reported in Amazon science screens [40])

## E. Neural networks and backpropagation

47. Implement forward and backward passes for a 2-layer MLP in NumPy, and train it on a toy dataset. [Likely] (backpropagation for an MLP was reported in breadth [35])
    - What are the shapes of every gradient?
    - How would you initialize the weights, and why?
48. Implement a numerically stable softmax with no third-party libraries, then softmax-plus-cross-entropy with its gradient with respect to the logits. [Reported] [39][41][42] (an Amazon online-assessment task requires the max-subtraction form for inputs in [-1000, 1000] [42]; hand-written softmax was also reported in AS loops [41])
    - Why does the combined gradient simplify to p - y?
49. Implement sigmoid by hand without overflow for large negative inputs, then ReLU, tanh and GELU with their derivatives. [Reported] (sigmoid [39][41]); the other activations are [Likely]
50. Implement dropout (inverted dropout) for training and inference. [Likely]
51. Implement the batch normalization forward pass with running statistics, and explain or implement the backward pass. [Likely] (batch norm was reported in breadth)
    - How does it behave differently at train and inference time?
52. Implement a 2D convolution (single channel, then multi-channel) with stride and padding, using loops and then im2col. [Likely] (vision teams; convolution and receptive fields were reported in breadth [41])
53. Implement a single LSTM or GRU cell forward step in matrix form. [Likely] (LSTM math and recurrent shapes in matrix form were reported in breadth [26][32])
    - How does backpropagation through time work, and where do vanishing gradients come from?
54. Build a minimal scalar autograd engine (a Value class with +, *, tanh and backward via topological sort). [Likely]
55. Implement a PyTorch training loop for a small model: data loader, forward, loss, backward, optimizer step, evaluation. [Reported] [39] (Amazon aggregator listing, undated; "practice a training loop from memory" is advised for the coding round [41])
    - Where do zero_grad, model.train() and model.eval() go, and what breaks if you forget each?
    - Add gradient accumulation and gradient clipping by global norm.
    - Add early stopping on validation loss and a warmup-plus-cosine learning-rate schedule.
56. Implement focal loss for binary classification and its gradient. [Likely] (class imbalance was reported in depth rounds [21])
    - How do the alpha and gamma parameters change the loss for easy and hard examples?
57. Implement cross-entropy with label smoothing. [Likely]
58. Implement symmetric per-channel int8 quantization of a weight matrix, dequantize it, and measure the error on a matrix multiply. [Likely] (Amazon Science publishes low-precision LLM training work [45])
    - Why does per-channel scaling beat per-tensor scaling when a few weights are outliers?

## F. Attention, transformers and LLM components

59. Implement scaled dot-product attention by hand in NumPy or PyTorch. [Reported] [39][41] (hand-written attention reported in AS loops, aggregator sighting dated July 2026 [39]; attention "at root level" and the math of self-attention were also reported in depth and breadth rounds [27][29])
    - Why divide by sqrt(d_k)?
    - What changes if Wq = Wk = Wv? (reported as a breadth question [7])
60. Add a causal mask and a padding mask to your attention function. [Likely]
    - Where exactly is the mask applied, and why use -inf instead of 0?
61. Implement multi-head self-attention, including the reshape, split heads and output projection. [Likely]
    - What is the time and memory complexity in sequence length?
62. Implement sinusoidal positional encoding, or rotary embeddings (RoPE). [Likely] (writing the RoPE formula was reported in Amazon science depth rounds [40][41])
63. Implement layer normalization and RMSNorm. [Likely]
    - Pre-norm or post-norm: what is the difference in training stability?
64. Implement a decoder step with a KV cache: given cached K and V and a new token, return the output and the updated cache. [Likely] (KV caching and decoder-only inference were reported in a 2026 breadth round [6])
    - How much memory does the cache use for a given model size and context length?
    - How would multi-query or grouped-query attention change your code? (grouped-query attention was reported in screens [40])
65. Implement a sliding-window (local) attention mask, as in Longformer. [Likely] (Longformer and long-context attention were reported [6])
66. Implement beam search decoding over a toy next-token probability function. [Likely] (beam search decoding was reported as a conceptual question for an LLM-team AS II [28])
    - How do you handle end-of-sequence tokens and length normalization?
67. Implement temperature, top-k and top-p (nucleus) sampling from a logits vector. [Likely]
68. Implement byte-pair encoding (BPE): learn merges on a small corpus, then tokenize a new word. [Likely]
69. Implement a LoRA linear layer forward pass and explain which parameters are trained. [Likely] (LoRA was reported in AWS AS breadth [4])
    - How do you merge the adapter weights for inference?
70. Implement the SFT loss for a causal LM, masking prompt tokens so only response tokens count. [Likely] (SFT pipelines were reported in a depth round [21])
71. Implement the DPO loss given log-probabilities from the policy and reference models for chosen and rejected responses. [Likely] (Alexa and AGI job descriptions mention SFT, DPO and RLHF [37])
72. Implement a numerically stable, symmetric InfoNCE loss in PyTorch for a batch of paired image and text embeddings (image-to-text and text-to-image). [Reported] [39][40][41] (Research Scientist and MLE loops; aggregator sighting dated February 2026 [39])
    - Where does the temperature go, and why use log-sum-exp in both directions?
    - Why are in-batch negatives a biased sample, and how does batch size change the loss?
73. You are handed a Transformer implementation that trains but performs badly. Find and fix the bugs. [Reported] [43] (Amazon AGI, aggregator; underlying reports not individually named)
    - Which bugs would you check first (mask direction, missing scaling, wrong softmax axis, residual or norm placement, positional encoding added twice)?
    - What small sanity tests (overfit one batch, compare with a reference shape) would you run?
74. Implement attention with an online (streaming) softmax over key blocks, keeping a running max and running sum, so the full N x N score matrix is never stored. [Likely] (flash attention was reported in Amazon depth rounds [41])
    - Show that your blocked result matches the naive result.
75. Implement a mixture-of-experts layer with top-k gating and an auxiliary load-balancing loss. [Likely] (dense versus mixture-of-experts was reported in an Amazon "LLM depth" round; see chapter 07)
    - What happens to the experts if you drop the load-balancing term?
76. Implement the acceptance step of speculative decoding: given draft and target probabilities for k draft tokens, decide how many to accept and sample the correction token. [Likely] (inference-optimization questions on LLM teams)
77. Implement the PPO clipped surrogate loss given log-probabilities, old log-probabilities and advantages. [Likely] (PPO's clipped objective was reported in Amazon depth rounds [41])
78. Given G sampled responses per prompt with scalar rewards, compute GRPO group-relative advantages and the policy loss. [Likely] (why GRPO drops the critic, and how to design its reward, were reported in Amazon science screens [40][41])
    - What happens when every response in a group gets the same reward?
79. Code a reinforcement-learning fundamental from scratch, for example tabular Q-learning on a grid world or REINFORCE with a baseline. [Reported] (Amazon AGI, aggregator: "reinforcement learning fundamentals + coding" [43]); the specific algorithm is [Likely]
80. Fit a Bradley-Terry model from pairwise preference judgments (for example LLM-judge or human A/B comparisons between model outputs) to get a score per model. [Likely] (LLM evaluation was reported in 2025-2026 science-application rounds; see chapter 07)
81. Implement a triplet or margin ranking loss with in-batch hard-negative mining for a product-retrieval embedding model. [Likely]

## G. Probability, sampling and statistics in code

82. Build a weighted random number generator from an array of weights (random pick with weight). [Reported] [9][39][40][41] (Senior AS screen, 2026 [9]; Research Scientist and MLE loops, aggregator sighting dated April 2026 [39])
    - What is the complexity per sample with prefix sums plus binary search? Can you do O(1) with the alias method?
    - What should happen if all weights are zero or there is a single element?
83. Implement reservoir sampling to pick k items uniformly from a stream of unknown length. [Likely]
    - How would you make it weighted?
84. Implement bootstrap sampling for the mean and its confidence interval, then remove the explicit Python loop. [Reported] [39][40] (Research Scientist and MLE screens; aggregator sighting dated August 2026 [39])
    - Which statistics does the bootstrap estimate poorly?
    - Extend it to a confidence interval for the AUC difference between two models scored on the same examples.
85. Implement a two-sample t-test (Welch) or a two-proportion z-test for an A/B test, from scratch. [Likely] (A/B testing, p-values and power were reported in breadth and screens [40])
    - How would you compute the required sample size for a given minimum detectable effect?
86. Implement a Bayesian (Naive Bayes) classifier from scratch, with Laplace smoothing and log-space computation, then apply it to scenarios the interviewer gives. [Reported] [40][41] (Research Scientist and MLE loops, sometimes in the hiring manager's round; Naive Bayes also appeared in an AS II breadth report [38])
87. Implement PCA via SVD, or via eigendecomposition of the covariance matrix, and project data onto the top k components. [Likely] (PCA and covariance were reported in breadth)
    - How would you implement power iteration for the top component only?
88. Implement EM for a 1-D Gaussian mixture model with two components. [Likely]
89. Implement Thompson sampling or UCB for a multi-armed bandit over ad creatives or widgets, and simulate regret. [Likely] (ads and personalization)

## H. Recommendation, search and NLP building blocks

90. Implement matrix factorization for implicit or explicit ratings with SGD or ALS. [Likely] (Amazon recommender-systems roles list collaborative filtering [34])
    - How do you handle cold-start users and items?
91. Implement item-to-item collaborative filtering: compute cosine similarity between item vectors from a sparse user-item matrix, and recommend the top N. [Likely] (Amazon's well-known item-to-item recommendations)
    - How would you scale this computation to millions of items?
92. Implement TF-IDF and BM25 scoring for a query against a small document collection. [Likely] (product search)
93. Implement a pointwise or pairwise learning-to-rank loss (for example, RankNet) and its gradient. [Likely]
94. Implement n-gram language-model counting with add-k smoothing, and score a sentence. [Likely]
95. Implement a simple word2vec skip-gram negative-sampling update for one (center, context) pair. [Likely]
96. Deduplicate near-identical product titles using normalized string keys or character shingles plus Jaccard similarity. [Likely]

## I. Data manipulation and ML pipeline coding

97. Compute a rolling average when some dates are missing. [Reported] [11]
    - Should missing days count as zero or be skipped? Defend your choice.
98. Load a dataset in pandas, clean it, engineer features, and fit a model, explaining each step. [Reported] [13][31]
    - Which steps can leak information from the test set?
99. Answer simple data-wrangling questions in Python (group-by, joins, aggregations) alongside standard DSA. [Reported] [3]
100. Return the top-K most frequent elements in an array, then in a stream. [Reported] [21][39] (aggregator sighting of a streaming follow-up variant dated September 2026 [39])
101. Given a company hierarchy as a tree with salaries, compute the average salary of each manager's direct reports. [Reported] [40][41] (Research Scientist and MLE loop coding)
     - What do you return for a manager with no direct reports?
102. Count an ad's clicks in the last k minutes from a stream of click events, using a deque and a hash map. [Reported] [40][41] (MLE and Research Scientist screens)
     - How does memory grow with the number of ads, and how would you expire idle ads?
103. Split an ordered list of sequences into contiguous batches, where each batch costs a fixed overhead plus padding up to its longest item, so total cost is minimized. Give the DP state, the recurrence and a worked example. [Reported] [39][40] (Applied Scientist phone screen, aggregator sighting dated April 2026)
     - What is the complexity, and can you bound the batch size to speed it up?
     - How would a greedy bucketing heuristic compare in practice for an inference server?
104. Implement a train, validation and test split that is stratified by label and grouped by user, so no user appears in two splits. [Likely]
105. Implement a time-based split and an expanding-window backtest for a demand-forecasting model. [Likely] (supply-chain forecasting)
106. Implement standard scaling and one-hot encoding as fit/transform classes that handle unseen categories. [Likely]
107. Implement mini-batch iteration over a dataset that does not fit in memory, with shuffling. [Likely]
108. Implement random oversampling, or class-weighted loss, for an imbalanced dataset, and show the effect on your metric function. [Likely] (class imbalance was reported in depth rounds [21])
109. Simulate data-parallel training: given gradient arrays from N workers, implement a ring all-reduce (reduce-scatter then all-gather) and check that it matches a plain average. [Likely] (inter-node communication and pipeline-parallel schedules were reported in the most specialized Amazon screens [41])
     - How much data does each worker send, as a function of N and the gradient size?

## J. Round-level probes (how the interviewer pushes at L5)

110. Before writing code: what are the inputs, outputs, data types and sizes? What should happen on empty or degenerate input? [Likely] (Amazon's prep pages stress edge cases and bad input [1][2])
111. What is the time and space complexity of your implementation, and where is the bottleneck? [Likely]
112. You cannot run the code. Walk me through it on a tiny example by hand. [Likely] (third-party guides note that code is usually not executed [30]; aggregators say coding is "discussed rather than run" in screens [40])
113. How would you test this function? Which unit tests and invariants would you write? [Likely]
114. How would this change if the data were 1000x larger, or arrived as a stream? [Likely]
115. Now apply it to our team's problem. For example, where would k-means or kNN fit in product categorization or Rufus retrieval? [Likely]
116. Estimate the compute time and memory-transfer time of a residual MLP block at batch size 1 and batch size 256, given memory bandwidth, on-chip memory and peak throughput. Which case is memory-bound? [Reported] [39][40][41] (manager-level screen, aggregator sighting dated August 2026)
117. Tell me about a time you dove deep into data or code to find a root cause (an LP question opening or closing the coding round). [Reported] [21]

## Prep priorities (derived from the tags above)

- **Most-reported ML coding tasks:** AUC from scores and labels (1), k-means (20), logistic regression or a gradient-descent training loop (28, 29, 55), a classifier from scratch or trained live (30), stable softmax, sigmoid and attention by hand (48, 49, 59), stable symmetric InfoNCE (72), NumPy mean and variance with MLE (38), bootstrap CI (84), a Bayesian classifier (86) and weighted random pick (82).
- **Team-specific:** AGI and LLM teams add debugging a broken Transformer (73) and RL coding (79); the [Likely] LLM items (60-71, 74-78) are the natural follow-ups.
- **Do not neglect:** data-manipulation and ML-flavored DSA coding (97-109), including the batching DP (103). It shows up in screens and loops where the interviewer is a scientist, not an SDE.
- **Time management:** practice each core primitive so it fits in 15 minutes, because long LP answers squeeze the coding time [40][41].

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
39. "Amazon interview process, OA & interview questions" (question listing with dated sightings, synced from trueinterview.io candidate reports), kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator), 2026; fetched and read October 2026 (k-means Dec 2025; InfoNCE Feb 2026; random pick with weight and Applied Scientist bucket-batching DP Apr 2026; sigmoid/softmax/attention Jul 2026; MLP compute-vs-memory and bootstrap CI Aug 2026; NumPy mean/variance with MLE and streaming top-K Sep 2026; undated gradient-descent and PyTorch training-loop entries). https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
40. "Amazon Research Scientist Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io, CC BY 4.0; Research Scientist track, same science loop format as AS), 2026; fetched and read October 2026 (coding-round families, InfoNCE, NumPy mean/variance with MLE, Bayesian classifier, bootstrap CI, bucket-batching DP, org-tree salary, click counter, LP time squeeze). https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
41. "Amazon Machine Learning Engineer Interview Process", kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator adapted from trueinterview.io; MLE track, includes notes on applied-science loops with only hand-written sigmoid, softmax and attention), 2026; fetched and read October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
42. "Implement Softmax" (asked at Amazon, online assessment; no third-party libraries, max-subtraction form required), kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator), 2026; fetched and read October 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/algorithm/implement-softmax/README.md
43. "AI Companies Interview Questions", Amazon AGI section ("Debug a broken Transformer implementation", "Build/train a small classifier live", "Reinforcement learning fundamentals + coding"), ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator; underlying reports not individually named), 2026; fetched and read October 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
44. amazon-science/muss, "MUSS: Multilevel Subset Selection for Relevance and Diversity" (UAI 2026), GitHub (team context for item 27); fetched and read October 2026. https://github.com/amazon-science/muss
45. amazon-science/mxfp4-llm, "Training LLMs with MXFP4" (AISTATS 2025), GitHub (team context for item 58); fetched and read October 2026. https://github.com/amazon-science/mxfp4-llm
