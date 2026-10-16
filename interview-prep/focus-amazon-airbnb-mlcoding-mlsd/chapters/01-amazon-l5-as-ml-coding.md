# 01. Amazon L5 Applied Scientist: ML Coding Round

> Scope: the coding round(s) in the Amazon L5 Applied Scientist (Applied Scientist II) phone screen and onsite loop, with the focus on **ML coding**: implementing algorithms, losses, metrics, sampling and data pipelines from scratch in Python, NumPy or PyTorch. A short DSA section at the end covers the LeetCode-style problems reported in AS loops. This chapter merges and dedupes four independent search passes with the earlier chapters `amazon-l5-applied-scientist/chapters/05-ml-coding.md` and `04-coding-dsa.md`. **Questions only. No answers or solutions.**
>
> Tags: **[Reported]** means a candidate report, or an aggregator of candidate reports, ties the coding task to an Amazon science loop (citation [n]). Where the report is from a junior, intern, Senior, Data Scientist, Research Scientist or MLE loop, the line says so. **[Likely]** means the task is inferred: the concept was asked verbally in a breadth or depth round, or the prompt comes from a practice site or prep guide and no candidate report says it was coded. Many Glassdoor sources were read only as search-result extracts, and several extracts could not be tied to one exact review ID; the reference list flags these. Treat each report as the candidate's account.
>
> Ordering: within each section, questions are ordered by how many independent reports name them, then by importance for an L5 loop. Numbering runs across the whole chapter.

## How this round works

- **Where it sits.** Amazon's AS prep page describes the onsite as 55-minute interviews with scientists [1]. AS II candidates in 2025-2026 report five rounds: ML breadth, ML depth, team-specific system design, coding, and a Bar Raiser [6]. A 2026 loop was "1 phone screen, 1 coding, 2 ML design and 2 LP rounds" [23]. A London 2026 loop had live coding plus ML depth and breadth rounds, 60 minutes each [32]. L5 phone screens mix ML concepts, statistics, LPs and a short coding task [9][74].
- **What the coding task looks like.** It depends on who interviews you [73]. An SDE-style interviewer gives "leetcode at L4 SWE bar. Correctness is top priority, explain your answer, give complexities" [17]. A scientist interviewer "tends to ask ML questions such as PCA and least squares" [11]; Blind commenters call OLS, gradient descent and batch normalization fair game, but not bagging, boosting or a vanilla recommender [10][14]. Reported mixes: "two coding questions, one ML related and the other an easy to medium leetcode question" (Melbourne, Nov 2025) [23]; "two easy Leetcode problems, some ML coding problems" (Santa Monica AS II, 2026) [3]; an ML coding round separate from ML design and LP rounds in a two-day onsite [51]; "most questions were non-leetcode questions more related to day to day ml implementations" (2026) [23]. LLM teams ask decoding code (beam search) in the screen [8]; AGI teams report debugging a broken Transformer, training a small classifier live, and RL coding [56].
- **Duration and pacing.** 45 to 60 minutes [28][32]. LP questions often take the first 20 to 30 minutes, and several reports describe only about 15 minutes left for code [53][54][70]. Practice each core primitive so it fits in 15 minutes.
- **Language and tools.** Python. NumPy for from-scratch numerics, PyTorch on deep-learning and LLM teams, and no scikit-learn estimators unless the task is a dataset round, where pandas, NumPy and sklearn are allowed [13]. You write in Amazon's shared editor (Livecode), with no IDE, and the code is usually not run, so you trace it by hand [1][2][71]. Amazon asks for syntactically correct code, not pseudo code [2]; a Berlin candidate who wrote k-means in "a mix of Python and pseudo-code" shows how this gets tested [7].
- **What is evaluated at L5.** Correct, readable, vectorized code; complexity stated without prompting; edge cases (empty input, ties, a single class, all-zero weights) raised before you are asked; numerical stability (max-subtraction, log-space, stable variance); and an answer to the scale-up follow-up (streaming, mini-batch, distributed) [17][53][54]. Preparation matters: an LLM-team candidate could not deliver working beam search "as I hadn't prepared" [8], while a Senior AS candidate passed with an unfinished graph problem by narrating the approach clearly [49].

## Top 15 to practice first

1. ROC AUC from scores and labels, with ties, in O(n log n) (Q44) [Reported]
2. k-means with k-means++ and empty-cluster handling (Q1) [Reported]
3. Logistic regression or "a traditional ML classifier" from scratch (Q2) [Reported]
4. Feed-forward neural network with backprop, extended to any number of layers (Q16) [Reported]
5. Gradient descent, then mini-batch SGD with momentum (Q3) [Reported]
6. OLS linear regression: closed form vs gradient descent (Q4) [Reported]
7. PCA from scratch with explained variance (Q5) [Reported]
8. Numerically stable softmax + cross-entropy and its gradient (Q17) [Reported]
9. Scaled dot-product attention with a causal mask (Q27) [Reported]
10. Beam search decoding for an LLM (Q29) [Reported]
11. Law-of-large-numbers simulation (Q55) [Reported]
12. Weighted random pick (Q56) [Reported]
13. Bootstrap confidence interval for a statistic (Q58) [Reported]
14. Dataset round: pandas cleaning, encoding, split, train, metrics (Q68) [Reported]
15. String compression, then number of islands, as the DSA warm-ups (Q79, Q80) [Reported]

## A. Classical ML from scratch

1. **k-means.** Implement `kmeans(X: np.ndarray, k: int, max_iters: int = 300, tol: float = 1e-4, init: str = "k-means++", seed: int | None = None) -> tuple[np.ndarray, np.ndarray]` returning `centroids` (k, d) and `labels` (n,). Alternate assignment and update steps, stop when the centroid shift is below `tol` or at `max_iters`. Handle: empty clusters, k > n, duplicate points, k = 1. No sklearn. [Reported] [7][15][39][52] (Berlin AS, Feb 2025, coded in Python [7]; a 2024 phone screen and a distance-metric follow-up [39]; Amazon aggregator sighting Dec 2025 [52]; also on Amazon-tagged practice lists [58][69])
   - What is the cost per iteration, and can you compute all point-to-centroid distances with broadcasting and no Python loop over points?
   - Can you implement k-means++ seeding as its own function, and why does it help?
   - What do you do when a cluster goes empty?
   - Which distance metric would you use, and what goes wrong when features have very different dynamic ranges (price vs rating)? [39]
   - How would you choose k?
   - How would you run it on data that arrives as a generator of (b, d) batches, or on 1 billion product embeddings (mini-batch, distributed)?
   - How would you port it to PyTorch tensors on a GPU? [68]
   - Does it always converge, and to what?

2. **Logistic regression / "a traditional ML classifier" from scratch.** Implement `class LogisticRegression` with `fit(X, y, lr=0.1, epochs=1000, l2=0.0)`, `predict_proba(X)` and `predict(X, threshold=0.5)`, trained with batch or mini-batch gradient descent, with a numerically stable sigmoid and log-loss. Then train on a train split, predict on a test split and report accuracy, precision and recall. Handle: a constant feature, a single class in `y`, separable data. [Reported] [13][15][22][24][56] (phone-screen "traditional ML classifier" [22][24]; "code a ML model from scratch", junior AS [13]; Blind loop summary names LR with gradient descent [15]; Amazon AGI "build/train a small classifier live" [56]; the conceptual loss and MLE were asked in [4]; also a practice prompt [60])
   - Can you derive the gradient of the log loss before you code it?
   - Why log loss and not MSE for this model? What are the model's assumptions? [38]
   - How do you avoid overflow in the sigmoid and `log(0)` in the loss?
   - Should the bias be regularized? How would you add L1 instead of L2?
   - The classes are 1:100. How do you add class weights to the loss and gradient, and which metric do you report (AUC vs PR-AUC)? [26]
   - How do you show it is learning when you cannot plot?
   - How would you check convergence, and what happens on perfectly separable data?

3. **Gradient descent, then SGD with momentum.** Implement `gradient_descent(X, y, lr, epochs) -> (w, b)` for linear regression with MSE, with the gradient written by hand and vectorized. Then generalize to `sgd(params, grad_fn, data, batch_size, lr, momentum, epochs)` that shuffles each epoch and applies momentum. Handle: a last batch smaller than `batch_size`, divergence (loss becomes NaN). [Reported] [19][21][52][70] ("implement gradient descent from scratch", AS intern [19]; "implement SGD for a problem (derive gradients, update, momentum etc.)" [21]; "write and explain gradient descent" [52][70])
   - Why does the learning rate that works for batch GD fail for SGD?
   - What does momentum change about the trajectory on an ill-conditioned problem?
   - Why does feature scaling matter here?
   - How would you add a learning-rate schedule and early stopping?

4. **Ordinary least squares, two ways.** Implement `fit_ols_closed_form(X, y)` with a bias column, and `fit_ols_gd(X, y, lr, epochs)`, and compare their weights on the same data. [Reported] [11][14][27] (Blind: scientist interviewers ask least squares [11][14]; an older review asked for the closed form, when GD is preferable, and the cost of the inversion [27])
   - What is the complexity of the closed form, and when does gradient descent win?
   - When is `X^T X` singular (collinearity, d > n), and what do you use instead (`lstsq`, pseudo-inverse, ridge)?
   - Why should you avoid calling `np.linalg.inv` directly?
   - How would you fit this when X has 1 billion rows and does not fit in memory?

5. **PCA.** Implement `pca(X: np.ndarray, k: int) -> tuple[components (k, d), projected (n, k), explained_variance_ratio (k,)]`, with centering and either the covariance eigendecomposition or the SVD. Handle: k > d, n < d, a zero-variance feature. [Reported] [11] (Blind: AS interviewers "tend to ask ML questions such as PCA and least squares")
   - SVD or eigendecomposition: which is more numerically stable, and which is cheaper when d is much larger than n?
   - Why are component signs arbitrary, and how would you make the output deterministic?
   - Should you standardize features first?
   - How would you get only the top component with power iteration?
   - How would you do PCA on data that streams in (incremental PCA)?

6. **Naive Bayes.** Implement multinomial Naive Bayes for text: `fit(docs: list[list[str]], labels)` estimates class priors and per-class word likelihoods with Laplace smoothing; `predict(docs)` uses log-probabilities. Then do the Gaussian variant for numeric features. Handle: unseen words, a class with no examples of a word, zero variance. [Reported] [53][54] (a Bayesian classifier from scratch, applied to interviewer scenarios, in Amazon Research Scientist and MLE loops; practice prompt [59])
   - Why compute in log space?
   - What does the smoothing parameter do, and how would you pick it?
   - What does the independence assumption break, and when does Naive Bayes still rank well?

7. **Decision tree (CART).** Implement `DecisionTreeClassifier(max_depth, min_samples_split)` with `fit(X, y)` and `predict(X)`: search every feature and candidate threshold for the best Gini or entropy split, then recurse. Handle: all labels equal, no valid split, duplicate feature values. [Likely] [57] (Amazon-tagged MLE onsite on a practice site, not an AS candidate report; how splits are chosen is asked verbally)
   - What is the cost per node, and how does pre-sorting each feature help?
   - How do you handle categorical features and missing values?
   - How would you store the trained tree as flat arrays for fast prediction?
   - How do you turn it into a random forest, and how do you compute out-of-bag error?

8. **k-nearest neighbors.** Implement `knn_predict(X_train, y_train, X_query, k) -> labels` with fully vectorized pairwise Euclidean distances (no Python loop over n x m pairs), majority vote, and distance-weighted tie-breaking. [Likely] [65]
   - The expansion `||a||^2 - 2ab + ||b||^2` can return small negative values. Why, and how do you fix it?
   - How do you pick k with cross-validation, and why must scaling be fit on training folds only?
   - What is the query cost, and what changes at millions of items (KD-tree, approximate nearest neighbors such as FAISS/HNSW)?

9. **Gradient boosting with stumps.** Implement `fit_gbm(X, y, n_estimators, learning_rate)` that fits depth-1 regression trees to the residuals of squared loss, plus `predict`. [Likely] (bagging vs boosting asked verbally in AS II loops [6][25]; Blind says boosting is a stretch for live coding [10])
   - What changes for log loss, and what are the pseudo-residuals?
   - How does this differ from bagging in what it does to bias and variance?

10. **Matrix-factorization recommender.** Implement `train_mf(ratings: list[tuple[user, item, rating]], k, lr, reg, epochs)` with SGD on regularized MSE, then `predict(u, i)` and `recommend(u, n)` that excludes items the user has already seen. [Likely] [38] (a recommendation-system question followed logistic regression)
    - How do you handle a cold-start user or item?
    - How would you adapt it to implicit feedback (clicks, no ratings)?

11. **Cross-validation and grid search from scratch.** Implement `stratified_kfold(y, k, seed)` and `grid_search(model_factory, param_grid, X, y, k) -> (best_params, results)` reporting mean and std per config, with preprocessing fit inside each training fold only. [Likely] [22]
    - Where can leakage creep in, and how does your code prevent it?
    - How do you split when rows from the same user must stay together?

12. **L1-regularized regression.** Implement `lasso(X: np.ndarray (n, d), y: np.ndarray (n,), alpha: float, max_iters: int = 1000, tol: float = 1e-6) -> np.ndarray (d,)` with cyclic coordinate descent and soft-thresholding (unpenalized intercept, objective `(1/2n)||y - Xw||^2 + alpha ||w||_1`). Stop when the max coordinate change is below `tol`. Handle: an all-zero column, alpha large enough that every weight is zero. [Likely]
    - Why does L1 produce exact zeros and L2 does not?
    - How do you keep the residual updated in O(n) per coordinate instead of recomputing `Xw`, and how would you compute a warm-started regularization path?

13. **EM for a Gaussian mixture.** Implement `gmm_em(x: np.ndarray (n,), k: int = 2, max_iters: int = 200, tol: float = 1e-6, seed: int | None = None) -> (means (k,), variances (k,), weights (k,), log_likelihood: float)`. Compute responsibilities in log space, stop when the log-likelihood gain is below `tol`, and floor variances at a small epsilon. Handle: a component that collapses onto one point, identical data points. [Likely]
    - How is this related to k-means, and how do you keep the likelihood computation stable?

14. **Linear SVM.** Implement `fit_linear_svm(X (n, d), y in {-1, +1} (n,), lam: float, epochs: int, batch_size: int, seed) -> (w (d,), b)` minimizing `lam/2 ||w||^2 + mean(max(0, 1 - y(Xw + b)))` by mini-batch subgradient descent with step size `1 / (lam * t)` (Pegasos). Handle: labels given as {0, 1}, a batch with no margin violations. [Likely]
    - What does the subgradient do at the margin?

15. **Item-to-item collaborative filtering.** Given `interactions: scipy.sparse.csr_matrix` of shape (n_users, n_items) with implicit counts, implement `top_n_similar(interactions, n: int) -> np.ndarray (n_items, n)` returning, for each item, the indices of the n most cosine-similar other items (exclude the item itself; items with no interactions return -1 padding). [Likely] (Amazon's item-to-item recommendations)
    - How would you scale this to millions of items without materializing the full similarity matrix?

## B. Deep learning from scratch (NumPy / PyTorch)

16. **Feed-forward network with backpropagation, then any number of layers.** In NumPy only, implement a network of `Linear` and `ReLU` layers with a softmax output: forward pass, cross-entropy loss, backward pass for every weight and bias, and SGD updates. Then refactor so the user can stack an arbitrary number of layers (`Layer.forward`, `Layer.backward`, `Network.add`). Input: `X` (N, D), `y` (N,), layer sizes, lr, epochs. Output: trained weights, the loss curve and final accuracy. [Reported] [12][13][20] ("write a neural network and add layers" [12]; "implement back propagation without using any additional libraries" [20]; "code a ML model from scratch", junior AS [13]; prep-thread variants [61][62])
    - What is the shape of every gradient?
    - How do you verify your gradients with finite differences, and which epsilon do you choose?
    - How do you initialize weights, and what goes wrong with all-zero initialization?
    - What changes if you swap ReLU for sigmoid in a deep stack?
    - How would you add L2 weight decay and mini-batching to your abstraction?

17. **Stable softmax and cross-entropy with gradient.** Implement `softmax(z)` with no third-party libraries for inputs in [-1000, 1000], then `cross_entropy(logits: (N, C), labels: (N,)) -> (loss, dlogits)` using log-sum-exp. [Reported] [52][54][55] (Amazon online assessment requiring the max-subtraction form [55]; hand-written softmax in applied-science loops [54]; the cross-entropy formula was asked verbally [37])
    - Why does the combined gradient simplify to `(softmax - onehot) / N`?
    - How would you support an ignore index for padding, and label smoothing?
    - What happens to the loss if a predicted probability is exactly zero?

18. **Activations by hand.** Implement elementwise `sigmoid(x: np.ndarray) -> np.ndarray` with no overflow warnings for x in [-1000, 1000], then `relu`, `tanh` and `gelu` (tanh approximation), each with a `*_grad(x)` derivative checked against finite differences. [Reported] (sigmoid [52][54]); the other activations are [Likely]
    - Why is the naive `1 / (1 + exp(-x))` unsafe, and for which inputs?

19. **Training and evaluation loop.** Given a PyTorch model, a `DataLoader`, a loss function and an optimizer, implement `train_one_epoch(...)` and `evaluate(...) -> dict` (loss, accuracy, AUC), with early stopping on validation loss and best-checkpoint saving. [Reported] [52][54] (PyTorch training loop listed for Amazon; "practice a training loop from memory"); framed as "day to day ML implementation" in [23][7]
    - Where do `zero_grad`, `model.train()` and `model.eval()` go, and what breaks if you forget each?
    - Add gradient accumulation and gradient clipping by global norm.
    - Add a warmup-plus-cosine learning-rate schedule.
    - How do you make the run reproducible?

20. **2D convolution layer.** Implement the forward pass of `conv2d(x: (N, C_in, H, W), w: (C_out, C_in, kH, kW), b, stride, padding)` first with loops, then vectorized (im2col). [Reported] [18] (Blind: some AS teams ask for "a convolutional layer in a CNN")
    - What is the output shape formula?
    - What does im2col cost in memory?
    - How would you write the backward pass for the weights?

21. **Batch normalization.** Implement `batchnorm_forward(x: (N, D), gamma, beta, eps, momentum, running_mean, running_var, training: bool) -> (out, cache)` and `batchnorm_backward(dout, cache) -> (dx, dgamma, dbeta)`. [Likely] [10][14] (Blind commenters call it fair game for Amazon AS; not a first-person report)
    - How does it behave at train time vs inference time?
    - What happens with batch size 1?
    - How does it differ from layer normalization, and why do Transformers use the latter?

22. **Optimizers.** Implement plain gradient descent, momentum and Adam as `step(w, grad, state)` functions, run them on an ill-conditioned quadratic, and return each trajectory. [Likely] [32] (GD vs Adam asked verbally in a June 2026 London AS loop)
    - Why does Adam need bias correction in the early steps?
    - Change Adam to AdamW. Where does decoupled weight decay enter, and why is it not the same as L2 in the loss?

23. **Dropout.** Implement `dropout_forward(x: np.ndarray, p: float, training: bool, rng) -> (out, mask)` and `dropout_backward(dout, mask) -> dx` using inverted dropout, where `p` is the drop probability. Handle: `p = 0`, `p = 1`. [Likely]
    - Why scale at training time and not at inference?

24. **Recurrent cell.** Implement `lstm_step(x_t (B, D), h_prev (B, H), c_prev (B, H), W (D + H, 4H), b (4H,)) -> (h_t, c_t)` with one fused matmul for the four gates, then `lstm_forward(x (B, T, D), h0, c0, W, b) -> (h_all (B, T, H), (h_T, c_T))` that respects a per-sequence `lengths` vector. [Likely]
    - Where do vanishing gradients come from in backpropagation through time?

25. **Tiny autograd.** Build a scalar `Value(data: float)` class supporting `+`, `*`, `-`, `**` (constant exponent), `tanh` and `backward()` that fills `.grad` for every ancestor via topological sort. Test it by fitting a 2-layer MLP on four points and comparing gradients against finite differences. [Likely]
    - How do you handle a node that is used twice in the graph?

26. **Losses for imbalance.** Implement `focal_loss(logits: (N,), y: (N,), alpha: float = 0.25, gamma: float = 2.0) -> (loss, dlogits)` and `weighted_bce(logits, y, pos_weight) -> (loss, dlogits)`, computed from logits with log-sigmoid for stability (no `log(sigmoid(x))`). [Likely] [26] (imbalanced binary classification asked verbally)
    - How do alpha and gamma change the loss on easy vs hard examples?

## C. NLP and LLM components

27. **Scaled dot-product attention.** Implement `attention(Q, K, V, mask=None) -> (out, weights)` for shapes (B, T, d) in NumPy or PyTorch, with a stable softmax and an optional causal or padding mask. [Reported] [52][54] (hand-written attention reported in applied-science loops; sighting July 2026); attention "at root level" and how a Transformer works were asked verbally in AS II loops [4][6]
    - Why divide by sqrt(d_k)?
    - Where exactly is the mask applied, and why use a large negative value or -inf instead of 0?
    - What are the time and memory costs in sequence length T?
    - What changes if Wq = Wk = Wv?

28. **Multi-head self-attention.** Implement `MultiHeadSelfAttention(d_model, num_heads)` with Q/K/V projections, head split, parallel attention across heads, merge and output projection. Input `x` (B, T, d_model), optional mask; output (B, T, d_model). [Likely] [37] (how heads are computed in parallel and merged was asked verbally on a personalization team)
    - What happens if `d_model` is not divisible by `num_heads`?
    - How would grouped-query or multi-query attention change your code?

29. **Beam search decoding.** Given `next_token_logprobs(prefix: list[int]) -> np.ndarray` of shape (V,), a beam width k, `max_len` and `eos_id`, implement `beam_search(...) -> list[tuple[list[int], float]]` returning the top-k finished sequences with cumulative log-probabilities. Stop beams that emit EOS. [Reported] [8] (Santa Clara LLM-team AS screen, 2025; the candidate did not finish working code)
    - How and why do you apply length normalization?
    - When do you stop: all beams finished, or `max_len`?
    - What is the complexity per step, and how do you avoid sorting the full k x V candidate set?
    - How does it compare with greedy decoding and with top-k/top-p sampling for an assistant?

30. **Debug a broken Transformer.** You are handed a ~100-line PyTorch decoder-only Transformer that trains without errors but whose loss plateaus far above a reference. Find and fix the bugs. (Practice: plant three bugs from the list below in your own implementation from Q27/Q28 and find them by test.) [Reported] [56] (Amazon AGI, aggregator; underlying reports not named)
    - Which bugs would you look for first (mask direction, missing scaling, wrong softmax axis, residual or norm placement, positional encoding added twice)?
    - Which sanity checks would you run (overfit a single batch, compare shapes against a reference, check that loss starts near log V)?

31. **Contrastive loss.** Implement a numerically stable, symmetric InfoNCE loss in PyTorch for a batch of paired image and text embeddings (image-to-text and text-to-image), with a temperature. [Reported] [52][53][54] (Research Scientist and MLE loops; sighting Feb 2026)
    - Where does the temperature go, and why use log-sum-exp in both directions?
    - How does batch size change the loss, and why are in-batch negatives a biased sample?

32. **RL fundamentals in code.** Implement `q_learning(env, n_episodes, alpha, gamma, epsilon, seed) -> Q (n_states, n_actions)` for a small grid world whose `env.reset() -> s` and `env.step(a) -> (s', r, done)`, with epsilon-greedy exploration; or `reinforce_loss(log_probs: (T,), rewards: (T,), gamma, baseline: (T,)) -> scalar` with discounted returns. [Reported] [56] (Amazon AGI: "reinforcement learning fundamentals + coding"); the specific algorithm is [Likely]
    - Why does a baseline reduce variance without adding bias?

33. **Sampling for generation.** Implement `sample_next(logits: (V,), temperature, top_k, top_p, rng) -> int`. Keep at least one token for top-p and handle `top_k > V`. [Likely] (sibling of Q29)
    - What do temperature 0 and very large temperature reduce to?

34. **BPE tokenizer.** Implement `train_bpe(word_freqs: dict[str, int], num_merges) -> list[tuple[str, str]]` and `encode(word, merges) -> list[str]`. [Likely] [33] (GenAI pre-training and post-training fundamentals advised for AS loops)
    - How would you make pair counting incremental instead of recounting after every merge?

35. **Word-level tokenizer class.** Implement `Tokenizer(min_freq: int, specials=("<pad>", "<unk>", "<bos>", "<eos>"))` with `fit(texts: list[str])`, `encode(text, add_special=True) -> list[int]` and `decode(ids, skip_special=True) -> str`: lowercase and NFKC-normalize, split on whitespace and punctuation, keep words with count >= `min_freq`, map the rest to `<unk>`, and make vocabulary ids deterministic under ties. [Likely]
    - How do you handle casing and Unicode normalization?

36. **KV-cache decode step.** Implement `decode_step(h_t: (B, d_model), W_q, W_k, W_v, W_o, k_cache: (B, H, T_past, d_head), v_cache: (B, H, T_past, d_head)) -> (out (B, d_model), k_cache', v_cache')` for one multi-head attention layer, and show that running it token by token reproduces full causal attention on the whole sequence. [Likely]
    - How much memory does the cache need for a given model size, batch and context length?

37. **Positional encodings.** Implement `sinusoidal_pe(T: int, d: int) -> (T, d)`, then `apply_rope(x: (B, H, T, d_head), positions: (T,), base: float = 10000.0) -> same shape` applied to Q and K (d_head even). Verify that `q_m . k_n` after RoPE depends only on `m - n`. [Likely] [53][54] (the RoPE formula was asked in Amazon science depth rounds)
    - Why does RoPE encode relative position?

38. **Normalization layers.** Implement `layernorm(x: (..., d), gamma, beta, eps=1e-5)` and `rmsnorm(x: (..., d), gamma, eps=1e-6)` normalizing over the last axis, plus the LayerNorm backward pass `-> (dx, dgamma, dbeta)`. [Likely]
    - Pre-norm or post-norm: which trains more stably, and why?

39. **LoRA layer.** Implement `LoRALinear(base: nn.Linear, r: int, alpha: float, dropout: float)` in PyTorch with frozen base weights, `A` (r, in) Gaussian-initialized and `B` (out, r) zero-initialized, forward `base(x) + (alpha / r) * B(A(dropout(x)))`, and `merge()` / `unmerge()` that fold the adapter into the base weight for inference. [Likely]
    - Which parameters are trained, and how many are there for rank r?

40. **SFT loss and collate function.** Implement `collate(batch: list[tuple[prompt_ids: list[int], response_ids: list[int]]], pad_id: int) -> dict(input_ids (B, T), attention_mask (B, T), labels (B, T))` with labels set to -100 on prompt and padding tokens, then `sft_loss(logits: (B, T, V), labels) -> scalar` with the one-position shift, averaged over response tokens in the batch. [Likely]
    - Does left or right padding matter for decoder-only generation?

41. **DPO loss.** Implement `dpo_loss(pi_chosen: (B,), pi_rejected: (B,), ref_chosen: (B,), ref_rejected: (B,), beta: float) -> (loss, chosen_reward, rejected_reward)` where each input is the summed response log-probability, using `logsigmoid` for stability. [Likely] (DPO vs RLHF asked verbally in AS II loops [6])
    - What does beta control?

42. **PPO and GRPO.** Implement `ppo_clip_loss(logp_new: (N,), logp_old: (N,), advantages: (N,), clip_eps: float = 0.2) -> scalar`, then `grpo_advantages(rewards: (P, G), eps: float = 1e-8) -> (P, G)` that normalizes each prompt's G rewards by the group mean and std. [Likely] [53][54]
    - What happens when every response in a group gets the same reward?

43. **Blocked attention with online softmax.** Implement `blocked_attention(Q: (T, d), K: (T, d), V: (T, d), block: int, causal: bool) -> (T, d)` that iterates over key blocks with a running max and running sum, so the full T x T score matrix is never stored, and assert `np.allclose` against the naive result. [Likely] [54] (flash attention asked in depth rounds)
    - What extra memory does it use, and how would you write the backward pass without storing the attention matrix?

## D. Metrics and evaluation code

44. **ROC AUC from scratch.** Implement `roc_auc(y_true: list[int], y_score: list[float]) -> float` without sklearn, in O(n log n) by sorting (rank / Mann-Whitney formulation or a threshold sweep with trapezoids). Tied scores count as 0.5. [Reported] [3][4][5][6][69] ("Write a program to calculate AUC", AS II, several 2025-2026 reports; "What are ROC and AUC?" asked separately [4])
    - What does your function return if all labels are one class?
    - Why is the O(n^2) pair count too slow, and how do ties break a naive sort-based version?
    - What do ROC and AUC mean, and when is PR-AUC the better choice?
    - How would you compute AUC over a stream or a dataset that does not fit on one machine (fixed-bin score histograms per class)?
    - How would you get a confidence interval for the AUC difference between two models on the same examples?

45. **Chamfer distance.** Implement `chamfer(A: (n, 3), B: (m, 3)) -> float`: the mean squared distance from each point in A to its nearest point in B, plus the same from B to A. Start with vectorized NumPy, then make it efficient. [Reported] [8] (extract "compute the chamfer distance efficiently" from Amazon AS reports; exact review not certain)
    - The O(nm) distance matrix does not fit in memory. How do you chunk it?
    - How does a KD-tree help, and how does it degrade as dimensionality grows?

46. **Percentage RMSE with a bootstrap CI.** Implement `prmse(y_true, y_pred)` and `bootstrap_ci(y_true, y_pred, metric, B=1000, alpha=0.05, seed) -> (point, low, high)` in NumPy only. [Reported] [50] (Amazon Data Scientist coding list, not AS)
    - What do you do when `y_true` contains zeros?
    - Can you vectorize the B resamples without a Python loop, and what does that cost in memory?

47. **Classification metrics.** Implement `confusion_matrix(y_true, y_pred, num_classes)` vectorized, then precision, recall and F1 with macro, micro and weighted averaging; then `best_f1_threshold(y_true, y_score)`. [Likely] [8][24] (precision, recall and AUC asked verbally in AS phone screens)
    - What do you return when a class has no predicted positives?
    - Can you find the best threshold in a single pass after sorting?

48. **PR curve and average precision.** Implement `pr_curve(y_true: (n,), y_score: (n,)) -> (precision, recall, thresholds)` with one point per distinct score (ties grouped) in O(n log n), and `average_precision(y_true, y_score) -> float` as the step-wise sum `sum((R_k - R_{k-1}) * P_k)`. Handle: no positives. [Likely]
    - Why can ROC AUC look optimistic under heavy class imbalance (fraud, abuse)?

49. **Ranking metrics.** Implement precision@k, recall@k, MAP, MRR and NDCG@k given, per query, a ranked list of item ids and a dict of graded relevance; return the mean over queries. Handle k larger than the list and queries with no relevant items (IDCG = 0). [Likely] [26][67] ("metrics for information retrieval" asked verbally)
    - Why does NDCG fit graded relevance in Amazon search better than MAP?

50. **Perplexity.** Given per-token log-probabilities (B, T) and a padding mask, return corpus-level perplexity. Then do it from a (T, V) probability array and target ids. [Likely] [8][24] (perplexity asked verbally in AS screens)
    - Why average over tokens rather than sequences?

51. **Log loss and calibration.** Implement `log_loss(y_true: (N,), probs: (N, C), eps=1e-15) -> float`, `ece(y_true: (N,), probs: (N, C), n_bins: int = 15) -> float` with equal-width confidence bins, and `fit_temperature(val_logits: (N, C), val_labels: (N,)) -> float` minimizing NLL by gradient descent on log T. [Likely]
    - Why does temperature scaling leave accuracy and AUC unchanged?

52. **IoU and NMS.** Implement `iou_matrix(a: (N, 4), b: (M, 4)) -> (N, M)` for boxes in `(x1, y1, x2, y2)` format with broadcasting, then `nms(boxes: (N, 4), scores: (N,), iou_threshold: float) -> list[int]` returning kept indices in descending score order. [Likely]
    - Which edge cases come from zero-area or non-overlapping boxes?

53. **Sequence metrics.** Implement `wer(ref: list[str], hyp: list[str]) -> (wer: float, subs: int, ins: int, dels: int)` with an edit-distance DP and backtrace, then `corpus_bleu(refs: list[list[str]], hyps: list[list[str]], max_n: int = 4) -> float` with clipped n-gram precision and brevity penalty. Handle: an empty reference. [Likely] (Alexa ASR and translation teams)

54. **Forecasting metrics.** Implement `mape(y, yhat)`, `smape(y, yhat)` and `pinball_loss(y: (n,), yhat_q: (n,), q: float) -> float`, then evaluate P50 and P90 forecasts from a (n, 2) array, all vectorized. [Likely] (supply-chain forecasting teams)
    - What goes wrong with MAPE when actuals are zero?

## E. Probability, statistics and sampling code

55. **Simulate the law of large numbers.** Implement `running_means(sampler: Callable[[int], np.ndarray], n: int, seed) -> np.ndarray (n,)` that draws n i.i.d. samples (a fair die, or Bernoulli(p)) and returns the running sample mean without a Python loop, then a check that the final mean is within a stated tolerance of the true expectation. [Reported] [9][18] (L5 AS tech phone screen [9]; "a simulation" as an applied ML-coding variant [18])
    - How does the error shrink with n, and can you show it empirically?
    - How many samples do you need for a given precision, and how do you report the uncertainty?
    - How do you seed it so the result is reproducible?
    - What happens with a heavy-tailed distribution such as Cauchy?

56. **Weighted random pick.** Implement `WeightedSampler(weights: list[float])` with `sample() -> int`, each index returned with probability proportional to its weight. [Reported] [43][52][53] (Senior AS screen, Seattle 2026 [43]; Research Scientist and MLE loops, sighting Apr 2026)
    - What is the cost per sample with prefix sums plus binary search, and can you reach O(1) (alias method)?
    - What should happen with all-zero weights, negative weights or a single element?
    - How would you test that the empirical frequencies are right?

57. **Mean and variance with MLE.** Implement `mean(x: np.ndarray, axis: int | None)` and `var(x, axis, ddof: int = 0)` without `np.mean`/`np.var` and without Python loops, ignoring NaNs if `skipna=True`, then relate them to maximum-likelihood estimation for a Gaussian. [Reported] [52][53][54] (sighting Sep 2026)
    - Which variance estimator is biased, and why do the sample variance and the Gaussian MLE differ?
    - How do you compute variance in a numerically stable way?

58. **Bootstrap CI for a statistic.** Implement `bootstrap_ci(x: np.ndarray, stat_fn, B: int = 1000, alpha: float = 0.05, seed) -> (point, low, high)` with the percentile method for the mean, then remove the explicit Python loop. (Q46 applies the same routine to a paired metric.) [Reported] [52][53] (Research Scientist and MLE screens, sighting Aug 2026)
    - Which statistics does the bootstrap estimate poorly?
    - Percentile interval or another variant: when does the choice matter?

59. **Weighted sampling without replacement.** Implement `weighted_sample_without_replacement(items, weights, k, rng) -> list` where each draw is proportional to the remaining weights, in O(n log k), and write a test that checks empirical frequencies. [Reported] [50] (Amazon Data Scientist coding list, not AS)
    - How would you do it on a stream of unknown length?

60. **Grid walk with probabilistic moves.** Implement `p_on_grid(m: int, n: int, start: tuple[int, int], k: int, probs: dict[str, float]) -> float`: an agent moves up/down/left/right with the given probabilities (summing to 1) and stops forever once it leaves the grid; return the probability it is still on the grid after k steps. Variant: probability of reaching a target cell within k steps. [Reported] [34] (2026 AS reports, "grid search variant with probability-weighted moves"; exact review not certain)
    - What are the time and space complexity, and can you reduce space to two layers?
    - How would you check your answer by Monte Carlo?

61. **Reservoir sampling.** Implement `reservoir_sample(stream: Iterable[T], k: int, rng) -> list[T]` returning k items sampled uniformly from a stream of unknown length in O(k) memory (all items if the stream has fewer than k). [Likely] [64]
    - Can you argue why every item ends up with probability k/n?
    - How would you extend it to weighted sampling?

62. **A/B test analysis.** Given control and treatment conversions and visitor counts, compute the lift, a two-proportion z-test p-value and a 95% CI; implement a bootstrap CI for the difference in mean revenue; then `sample_size(baseline, mde, alpha, power)` per arm. [Likely] [41][63] (statistics questions reported in depth rounds)
    - How does the required sample size change for a 1% relative lift vs a 10% one?

63. **Streaming statistics.** Implement `class RunningStats` with `update(x: float)`, `update_batch(xs: np.ndarray)`, `mean`, `variance(ddof=1)` using Welford's algorithm, and `merge(a, b) -> RunningStats` combining two shards' (count, mean, M2). [Likely]
    - Why does the naive `E[x^2] - E[x]^2` fail for values near 1e9 with small spread?

64. **Monte Carlo vs closed form.** Implement `p_collision_mc(k: int, n: int, trials: int, rng) -> (estimate, std_error)` for the probability that at least two of k shoppers pick the same item out of n uniformly, vectorized over trials, and `p_collision_exact(k, n)` computed in log space; compare them. [Likely]

65. **Bandits.** Implement `simulate(policy: "thompson" | "ucb1", true_ctr: np.ndarray (K,), horizon: int, seed) -> np.ndarray (horizon,)` returning cumulative regret for Bernoulli arms (ad creatives), with Beta(1, 1) priors for Thompson sampling. [Likely]
    - How do you handle non-stationary click-through rates?

66. **CUPED.** Implement `cuped(y: (n,), x_pre: (n,), group: (n,) in {0, 1}) -> (theta, y_adj, var_reduction)` with theta estimated on pooled data, then the treatment effect and its standard error before and after adjustment. [Likely]
    - What do you do for new users with no pre-period data?

67. **Drift metrics.** Implement `psi(train: np.ndarray, serve: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float` with bin edges from training quantiles, and `kl_divergence(p, q, eps)` on the same bins. Handle: serving values outside the training range, a constant feature. [Likely]
    - Why are both sensitive to the binning?

## F. Data wrangling (pandas / SQL / pipelines)

68. **ML dataset coding round.** Given a CSV of labeled tabular data (numeric and categorical columns, missing values, a binary target with about 10% positives), use pandas, NumPy and sklearn to clean it (missing values, categorical encoding), engineer features, split train/test or cross-validate, train a baseline, and report precision, recall and F1, explaining each choice as you go. [Reported] [13][42] (junior AS loop with "ML dataset coding (pandas, numpy, sklearn)" [13]; London AS 2024: load, clean, feature-engineer and fit [42])
    - Which of your steps can leak information from the test set?
    - Why that imputation and that encoding for a high-cardinality categorical?
    - How would you pick a decision threshold for the business metric?

69. **Python data wrangling.** Given `orders: list[dict(order_id, customer_id, amount, ts)]` and `customers: list[dict(customer_id, region)]`, return revenue per region and the top-3 customers by spend per region, once in plain Python (no pandas) and once in pandas. [Reported] [72]
    - How would you do the same join if neither table fits in memory?

70. **Rolling average with missing dates.** Implement `rolling_avg(rows: list[tuple[date, float]], window: int = 7) -> list[tuple[date, float]]` over a calendar window (not a row window), returning one value per calendar day from the first to the last date, given unsorted input with gaps and possible duplicate dates. [Reported] [16] (L5 AS virtual onsite)
    - Should missing days count as zero or be skipped? Defend your choice.

71. **Org-tree aggregation.** Given `rows: list[tuple[employee_id, manager_id | None, salary]]`, implement `avg_direct_report_salary(rows) -> dict[manager_id, float]`. [Reported] [53][54] (Research Scientist and MLE loops)
    - What do you return for a manager with no reports, and how would you average over all (not only direct) reports in one pass?
    - What if the input is a nested `Employee(id, salary, reports: list[Employee])` tree instead, and how do you avoid recursion-depth limits?

72. **Ad clicks in the last k minutes.** Implement `ClickCounter(window_seconds: int)` with `record(ad_id: str, ts: int)` (timestamps non-decreasing) and `count(ad_id, now: int) -> int` returning clicks in `(now - window, now]`, amortized O(1) per call. [Reported] [53][54] (MLE and Research Scientist screens)
    - How does memory grow with the number of ads, and how do you expire idle ads?
    - What if events arrive out of order?

73. **Day-to-day feature pipeline.** Given a DataFrame of events (`customer_id`, `item_id`, `timestamp`, `event_type`, `price`), build per-customer features as of a label date (purchases in the last 30 days, average price, days since last event, rolling 7-day activity), compute the label "purchased within the next N days", and make a time-based train/test split with no leakage. Return a feature DataFrame keyed by `customer_id`. [Likely] [7][23] (the "day to day ML implementations" round was reported; the prompt is inferred)
    - Where exactly can future data leak in, and how does your code prevent it?
    - How would you make it run on a year of events across all customers (Spark, incremental)?

74. **Crowdsourced label quality.** Given a DataFrame (`item_id`, `annotator_id`, `label`, `timestamp`) with categorical labels and a variable number of annotators per item, compute inter-annotator agreement, each annotator's accuracy against the majority vote, and label entropy per item, then filter a high-quality subset. [Likely] [8] (asked as a design question in a Bellevue 2025 onsite)

75. **SQL.** Given `orders(order_id, customer_id, order_date, amount)` and `returns(order_id, return_date)`, return per month the active customers, the return rate and a 7-day rolling average revenue; then each customer's second purchase date with window functions. [Likely] [65]

76. **Splits.** Implement `group_stratified_split(df, group_col: str, label_col: str, fractions=(0.8, 0.1, 0.1), seed) -> (train_idx, val_idx, test_idx)` so no group appears in two splits and each split's label rate is within a tolerance of the overall rate. [Likely]

77. **Preprocessing classes.** Implement `StandardScaler` with `fit(X)`, `transform(X)`, `inverse_transform(X)` (zero-variance columns scale by 1) and `OneHotEncoder(handle_unknown="ignore" | "error", min_frequency: int)` with `fit(cols: list[list[str]])` and `transform -> np.ndarray` (unseen categories become all-zero rows). [Likely]

78. **Out-of-core batching.** Implement `batch_iter(shard_paths: list[str], batch_size: int, buffer_size: int, seed) -> Iterator[np.ndarray]` that shuffles shard order each epoch, streams rows through a bounded shuffle buffer and yields fixed-size batches (last batch may be short). [Likely]
    - How do you approximate a full shuffle with bounded memory?

## G. DSA problems seen in AS loops

This section lists problems tied to Amazon AS loops. For the wider Amazon DSA list, see `amazon-l5-applied-scientist/chapters/04-coding-dsa.md`.

79. **String compression.** Run-length encode a string (`"aabcccccaaa"` becomes `"a2b1c5a3"`), returning the original if the result is not shorter; variant: compress a char array in place and return the new length. O(n) time. [Reported] [6][22][25] (AS II Seattle 2026; Hyderabad AS 2025-2026; AS II June 2026)
    - How do you handle counts of 10 or more in the in-place variant?
    - Can you do it in O(1) extra space?

80. **Number of islands.** Count 4-connected islands in a grid of `'1'` and `'0'`. [Reported] [27][28] (London AS phone screen; 45-minute Python coding round, Jan 2025)
    - What changes for a very large grid (iterative BFS, union-find), or one streamed row by row?

81. **Two Sum.** Return the indices of two numbers that sum to a target, in O(n). [Reported] [24][26][40] (AS II NLP phone screen [26]; AS online coding [40])
    - What if the array is sorted, or you must return all unique pairs?

82. **Median of a data stream.** Implement `addNum(int)` and `findMedian()` with O(log n) insert. [Reported] [29] (Luxembourg AS; "Leetcode Medium or Hard")
    - What if all numbers are in [0, 100]?
    - How would you support a sliding-window median?

83. **2D binary search.** Search a target in a matrix where each row is sorted and each row starts after the previous row ends, in O(log(mn)); follow-up: rows and columns sorted independently. [Reported] [8][24] (AS phone screens, with ML breadth on precision, recall, AUC and perplexity)
    - Can you compare the staircase search with binary search per row?

84. **Rotate a matrix.** Rotate an n x n matrix 90 degrees clockwise in place with O(1) extra space. [Reported] [35][36] (Cambridge AS screen; 2024 AS report)
    - How would you rotate a non-square image tensor with NumPy?

85. **Merge intervals.** Merge overlapping [start, end] intervals. [Reported] [30]
    - Should touching intervals merge? How would you insert one interval into an already merged list?

86. **Batching to minimize padding cost.** Split an ordered list of sequences into contiguous batches, where each batch costs a fixed overhead plus padding up to its longest item, so total cost is minimal. Give the DP state and recurrence. [Reported] [52][53] (Applied Scientist phone screen, sighting Apr 2026)
    - What is the complexity, and can bounding the batch size speed it up?
    - How would a greedy bucketing heuristic compare for an inference server?

87. **Edit distance.** Compute the Levenshtein distance between two strings. [Reported] [44] (Cambridge AS, 2020: "a coding problem similar to edit distance")
    - Can you reduce memory to O(min(m, n))?

88. **Search insert position.** Return the index of the target in a sorted array, or where it would be inserted, in O(log n) without `bisect`. [Reported] [47] (AS live coding, PracHub)

89. **A DP medium.** A Blind-150-style DP problem; practice prompt: `coin_change(coins: list[int], amount: int) -> int` returning the minimum number of coins (-1 if impossible), then `count_ways(coins, amount) -> int` counting combinations (order does not matter). [Reported] [23][33] (the specific problem was not given)
    - Can you write the recurrence first, then the bottom-up version, then O(B) memory?

90. **Count inversions.** Implement `count_inversions(a: list[int]) -> int`, the number of pairs i < j with a[i] > a[j], in O(n log n) (merge sort or a Fenwick tree over ranks). [Reported] [26]

91. **A graph problem in the screen.** Practice prompt: given `n` nodes and `edges: list[tuple[int, int]]`, implement `count_components(n, edges) -> int`, `has_cycle_directed(n, edges) -> bool` and `shortest_path_unweighted(n, edges, src, dst) -> list[int]`. [Reported] [49] (Senior AS tech screen, Dec 2025; the exact problem was not given; the candidate did not finish and still passed)
    - Which traversal fits, and how do you show the rest of the solution if you run out of time?

92. **Two-pointer problem, then 3Sum.** Implement `two_sum_sorted(a: list[int], target) -> tuple[int, int] | None` with two pointers, then `three_sum(a: list[int], target: int) -> list[tuple[int, int, int]]` returning all unique value triplets in O(n^2). [Reported] [45] (Sr. AS L6 phone screen, 2022)

93. **Implement a data structure.** `LRUCache(capacity)` with O(1) `get(key) -> int` (-1 if missing) and `put(key, value)`, then a `MinHeap` with `push`, `pop` and `heapify(list)` without `heapq`; state the cost of each operation. (Trie: see Q94.) [Reported] [31] (2021 AS: "4-5 questions on implementation of some data structures"; the specific structures are illustrative)

94. **Trie autocomplete.** Implement a trie with `insert`, `search` and `starts_with`, then `suggest(products: list[str], query: str) -> list[list[str]]` returning the three lexicographically smallest product names for each typed prefix of `query`. [Likely] [66] (a cross-company Applied Scientist results page mentions a medium trie problem; it could not be tied to one Amazon review)

95. **LCA in a BST.** Return the lowest common ancestor of two nodes in O(h). [Reported] [46] (new-grad AS loop, L4)

96. **Course Schedule III.** Given courses as [duration, lastDay], return the maximum number you can take. [Reported] [46] (new-grad AS loop, L4, with a disguised description)

97. **Fruits Into Baskets III.** Place each fruit type into the leftmost unused basket with enough capacity and count unplaced fruits, faster than O(n^2). [Reported] [46] (new-grad AS loop, L4)

98. **Basic Calculator II.** Implement `calculate(s: str) -> int` for non-negative integers, `+ - * /` (division truncates toward zero) and spaces, without `eval`, in O(n). [Likely] [48] (Amazon-tagged on PracHub, but the role is not shown, so not tied to an AS loop)

99. **My Calendar III.** Implement `book(start, end)` returning the maximum overlap so far, then extend it to cancellations. [Reported] [50] (Amazon Data Scientist coding list, not AS)

## H. Level calibration: L5 probes

The bank above is broad, but many prompts stop at a junior bar (write the loop, get it right). An L5 Applied Scientist is expected to go further unprompted: vectorize, prove numerical safety, state complexity, test, and extend. Use these harder variants after the base version of each prompt. All are [Likely]: they are inferred from the L5 evaluation criteria above [17][53][54], not from a specific report.

100. **AUC at scale (extends Q44).** Implement `streaming_auc(batches: Iterable[tuple[y (b,), score (b,)]], n_bins: int) -> float` using per-class score histograms, bound its error versus exact AUC as a function of `n_bins`, then `merge` histograms from many workers. Write a property test against your exact O(n log n) version on random data with heavy ties. [Likely]
101. **DeLong-free AUC comparison (extends Q44, Q58).** Implement `paired_bootstrap_auc_diff(y, s_a, s_b, B, seed) -> (diff, low, high)` vectorized over B with a single argsort per resample, and explain why resampling rows independently is wrong when rows cluster by user. [Likely]
102. **Fused, stable softmax-cross-entropy (extends Q17).** Implement forward and backward for logits (N, C) with C = 1e5 in float16 inputs: accumulate in float32, never materialize the full softmax for the loss, support `ignore_index` and label smoothing, and write a finite-difference gradient check that passes in float64. [Likely]
103. **k-means on 1e9 rows (extends Q1).** Implement `minibatch_kmeans(batch_iter, k, n_steps, seed)` with per-centroid learning rates `1/count`, k-means|| style seeding from a sample, and empty-cluster reseeding from the farthest points; state the memory bound and how you would shard it across workers. [Likely]
104. **Logistic regression that cannot diverge (extends Q2).** Add L2, class weights and a Newton/IRLS solver using `np.linalg.solve` (no `inv`); detect separable data and show what happens to `||w||` with and without L2. Compare iterations-to-converge with GD. [Likely]
105. **Attention with memory bounds (extends Q27, Q43).** Implement causal attention for T = 32k with a padding mask in which some rows are entirely masked: return zeros, not NaN, for those rows, never allocate T x T, and assert equivalence with the naive version on small T. [Likely]
106. **Batched beam search (extends Q29).** Vectorize beam search over a batch of B prompts with a `(B, k)` beam state, using `argpartition`/`topk` on `(B, k*V)`, length penalty, early stopping per batch element, and a test that k = 1 reproduces greedy decoding. [Likely]
107. **Weighted sampler, production version (extends Q56).** Implement the alias method in O(n) build and O(1) sample, support `update(i, w)` in O(log n) with a Fenwick tree, and write a chi-square test of the empirical frequencies with a fixed seed. [Likely]
108. **Gradient check harness (extends Q16, Q20, Q21).** Implement `grad_check(f, params: dict[str, np.ndarray], eps=1e-6) -> dict[str, float]` returning relative error per parameter using central differences on a random subset of entries; use it to catch a planted bug in your batchnorm backward. [Likely]
109. **Leak-proof feature pipeline (extends Q73, Q76).** Given the Q73 events table, write unit tests that fail if any feature uses an event with `timestamp >= label_date`, and make the pipeline produce identical features whether computed in one batch or incrementally day by day. [Likely]
110. **Variance at extremes (extends Q57, Q63).** Show with a test that the naive two-pass variance on float32 data around 1e8 loses all precision, then fix it with Welford or a shifted two-pass and with pairwise summation; state the error bounds informally. [Likely]
111. **Profile and speed up (any prompt).** Take your pure-Python k-NN or Chamfer solution (Q8, Q45), state its complexity, then rewrite it to be chunked and vectorized with memory capped at M bytes, and give the chunk size as a function of M, n, m and d. [Likely]

## References

1. Applied Scientist Interview Prep, amazon.jobs (Amazon, official), n.d. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. SDE II Interview Prep (Livecode editor, syntactically correct code), amazon.jobs (Amazon, official), n.d. https://amazon.jobs/content/en/how-we-hire/sde-ii-interview-prep
3. Amazon interview review RVW102987402 (Applied Scientist II, Santa Monica; AUC; two easy LeetCode plus ML coding), Glassdoor, 2026 (read from search extract; snippet-to-review mapping approximate). https://www.glassdoor.ie/Interview/Amazon-Interview-E6036-RVW102987402.htm
4. Amazon interview review RVW66719970 (Applied Scientist II; AUC; logistic regression loss and MLE; attention), Glassdoor, 2024-2025 (search extract). https://www.glassdoor.com.au/Interview/Amazon-Interview-E6036-RVW66719970.htm
5. Amazon interview review RVW95575263 (Applied Scientist II; code AUC from scores and labels), Glassdoor, 2026. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW95575263.htm
6. Amazon interview review RVW96903377 (Applied Scientist II, Seattle; AUC; string compression; attention; bagging vs boosting; DPO vs RLHF), Glassdoor, 2026 (search extract). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW96903377.htm
7. Amazon interview review RVW95289060 (Applied Scientist, Berlin; k-means in Python; also cited for a 2026 "day to day ML implementations" loop), Glassdoor, 2025-2026 (search extract; mapping approximate). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW95289060.htm
8. Amazon interview review RVW96068026 (Applied Scientist II, LLM team, Santa Clara: beam search; Bellevue: 2D binary search, precision/recall/AUC/perplexity, crowdsourced-data metrics, Chamfer distance), Glassdoor, 2025-2026 (search extract; mapping approximate). https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW96068026.htm
9. "Amazon L5 Applied Scientist Machine Learning Engineering Tech Phone Screen Experience" (law-of-large-numbers simulation), 1point3acres, 2024-2025. https://www.1point3acres.com/interview/thread/1098917
10. "Tips for coding interview, Applied Scientist at Amazon" (OLS, gradient descent, batch norm fair game; not bagging/boosting), Blind, 2021-2024. https://www.teamblind.com/post/tips-for-coding-interview-applied-scientist-at-amazon-txexmtsb
11. "Amazon applied science coding interview" (AS interviewers ask PCA and least squares), Blind, 2021-2024. https://www.teamblind.com/post/amazon-applied-science-coding-interview-onryzukx
12. "Amazon applied scientist" (asked to write a neural network and add layers), Blind, 2021-2024. https://www.teamblind.com/post/amazon-applied-scientist-6m0hwyih
13. "Amazon Jr Applied Scientist" (ML dataset coding with pandas, NumPy, sklearn; code an ML model from scratch), Blind, 2022. https://www.teamblind.com/post/amazon-jr-applied-scientist-zan6avr8
14. "Amazon applied scientist interview" (OLS, gradient descent, batch normalization fair to ask), Blind, n.d. https://www.teamblind.com/post/amazon-applied-scientist-interview-hn6cwts0
15. "Amazon applied scientist interview loops" (logistic regression with gradient descent, k-means), Blind, n.d. https://www.teamblind.com/post/amazon-applied-scientist-interview-loops-lucdk0uq
16. "Amazon Applied Scientist (L5) Virtual Onsite" (one coding round; rolling average with missing dates), Blind, circa 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
17. "Applied scientist interview Amazon" (LeetCode at L4 SWE bar; correctness first, complexities), Blind, 2021-2025. https://www.teamblind.com/post/applied-scientist-interview-amazon-4udsobcp
18. "Amazon applied scientist loop" (simulation or a CNN convolutional layer as applied ML coding), Blind, n.d. (search extract). https://www.teamblind.com/post/amazon-applied-scientist-loop-odlzgrp3
19. Amazon interview review RVW81277838 (Applied Scientist Intern; implement gradient descent from scratch), Glassdoor, n.d. (search extract; mapping approximate). https://fr.glassdoor.be/Entretien/Amazon-Entretien-E6036-RVW81277838.htm
20. Amazon interview review RVW79806572 (implement backpropagation without libraries), Glassdoor, n.d. (search extract; mapping approximate). https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW79806572.htm
21. Amazon interview review RVW40731696 (implement SGD: derive gradients, update, momentum), Glassdoor, n.d. (search extract; mapping approximate). https://fr.glassdoor.ca/Entretien/Amazon-Entretien-E6036-RVW40731696.htm
22. Amazon interview review RVW98336019 (Applied Scientist, Hyderabad; phone screen coding a traditional ML classifier; string compression; four technical rounds), Glassdoor, 2025-2026 (search extract; mapping approximate). https://www.glassdoor.es/Entrevista/Amazon-Entrevista-E6036-RVW98336019.htm
23. Amazon interview review RVW101665167 (Applied Scientist; Melbourne Nov 2025 one ML plus one LeetCode coding question; DP medium; 2026 "day to day ML implementations" loop), Glassdoor, 2025-2026 (search extract; mapping approximate). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW101665167.htm
24. Amazon interview review RVW47917252 (Applied Scientist; code a traditional ML classifier; 2D binary search; precision, recall, AUC, perplexity), Glassdoor, 2020-2026 (search extract). https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW47917252.htm
25. Amazon interview review RVW101785177 (Applied Scientist II; bias-variance, bagging vs boosting, string compression), Glassdoor, 2026 (search extract). https://www.glassdoor.com.mx/Entrevista/Amazon-Entrevista-E6036-RVW101785177.htm
26. Amazon interview review RVW32314669 (Applied Scientist II NLP phone screen: Two Sum; counting inversions; IR metrics; imbalanced logistic regression), Glassdoor, n.d. https://www.glassdoor.com/Interview/Amazon-Interview-E6036-RVW32314669.htm
27. Amazon interview review RVW75899476 (Applied Scientist, London; number of islands; linear regression closed form and inversion cost), Glassdoor, n.d. https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW75899476.htm
28. Amazon interview review RVW96647342 (45-minute Python coding round; number of islands), Glassdoor, 2025. https://www.glassdoor.co.in/Interview/Amazon-Interview-E6036-RVW96647342.htm
29. Amazon interview review RVW99133644 (Applied Scientist, Luxembourg; median from data stream), Glassdoor, 2023-2026. https://www.glassdoor.sg/Interview/Amazon-Interview-E6036-RVW99133644.htm
30. Amazon interview review RVW58753536 (merge intervals as the medium question), Glassdoor, n.d. (search extract). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW58753536.htm
31. Amazon interview review RVW36269508 (Applied Scientist; implementation of data structures), Glassdoor, 2021. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW36269508.htm
32. Amazon interview review RVW104768247 (Applied Scientist, London area; gradient descent vs Adam; 60-minute rounds), Glassdoor, 2026 (search extract). https://fr.glassdoor.ca/Entretien/Amazon-Entretien-E6036-RVW104768247.htm
33. Amazon interview review RVW94715488 (DP problem from Blind 150; GenAI fundamentals), Glassdoor, 2025 (search extract). https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW94715488.htm
34. Amazon interview review RVW91266671 (Applied Scientist, Seattle; grid problem with probability-weighted moves), Glassdoor, 2026 (search extract; mapping approximate). https://www.glassdoor.com.hk/Interview/Amazon-Interview-E6036-RVW91266671.htm
35. Amazon interview review RVW65272345 (Applied Scientist, Cambridge UK; matrix rotation), Glassdoor, 2024-2025 (search extract). https://www.glassdoor.es/Entrevista/Amazon-Entrevista-E6036-RVW65272345.htm
36. Amazon interview review RVW89381532 (Applied Scientist; rotate matrix 90 degrees), Glassdoor, 2024. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW89381532.htm
37. Amazon interview review RVW87352113 (Applied Scientist, personalization; multi-head attention in parallel; cross-entropy formula), Glassdoor, n.d. (search extract; mapping approximate). https://www.glassdoor.es/Entrevista/Amazon-Entrevista-E6036-RVW87352113.htm
38. Amazon interview review RVW9853746 (logistic regression assumptions and loss; recommendation system), Glassdoor, n.d. https://fr.glassdoor.ca/Entretien/Amazon-Entretien-E6036-RVW9853746.htm
39. Amazon interview review RVW92098279 (k-means distance metric; features with different dynamic ranges), Glassdoor, n.d. https://fr.glassdoor.ca/Entretien/Amazon-Entretien-E6036-RVW92098279.htm
40. Amazon interview review RVW22038746 (Applied Scientist online coding: Two Sum), Glassdoor, 2018-2019. https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW22038746.htm
41. Amazon interview review RVW71908933 (statistics questions in depth rounds), Glassdoor, n.d. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW71908933.htm
42. Amazon interview review RVW99272421 (Applied Scientist, London, July 2024; pandas load, clean, feature-engineer and fit), Glassdoor, 2024. https://static.glassdoor.it/Interview/Amazon-Interview-E6036-RVW99272421.htm
43. Amazon interview review RVW102577344 (Senior Applied Scientist, Seattle; weighted random number generator), Glassdoor, 2026. https://www.glassdoor.co.nz/Interview/Amazon-Interview-E6036-RVW102577344.htm
44. "Amazon | Applied/Research Scientist | Cambridge UK | May 2020 [Reject]" (coding problem similar to edit distance), LeetCode Discuss, 2020. https://leetcode.com/discuss/interview-experience/658771/amazon-appliedresearch-scientist-cambridge-uk-may-2020-reject
45. "Amazon | Sr. Applied Scientist L6 | Phone Screen | June 2022" (three-sum follow-up), LeetCode Discuss, 2022. https://leetcode.com/discuss/interview-experience/2221480/
46. "Amazon new grad Applied Scientist interview experience: five virtual onsite rounds, L4 offer" (LCA in BST, LC 3479, Course Schedule III), PracHub, 2025. https://prachub.com/interview-experiences/amazon-new-grad-applied-scientist-interview-experience-five-virtual-onsite-rounds-mostly-behavioral-end-in-an-l4-offer
47. Amazon Interview Questions, page 15 (AS live coding: Search Insert Position without helper libraries), PracHub, 2025. https://prachub.com/companies/amazon?page=15
48. Amazon Interview Questions, page 28 (Basic Calculator II; role not shown), PracHub, n.d. https://prachub.com/companies/amazon?page=28
49. "Amazon Senior Applied Scientist Interview Experience: Passed the Tech Screen Despite an Unfinished Graph Question", PracHub, December 2025 (search extract). https://prachub.com/interview-experiences/amazon-senior-applied-scientist-interview-experience-passed-the-tech-screen-despite-an-unfinished-graph-question
50. Amazon Data Scientist Coding & Algorithms Interview Questions (weighted sampling without replacement; percentage RMSE with bootstrap CI; My Calendar III), PracHub, 2025. https://prachub.com/companies/amazon/positions/data-scientist/categories/coding-and-algorithms
51. "Amazon Applied Scientist Onsite Interview Experience: Machine Learning Focus" (two-day onsite: coding, ML system design, LPs), 1point3acres, n.d. (title and summary only). https://www.1point3acres.com/interview/thread/1020600
52. "Amazon interview process, OA & interview questions" (dated sightings synced from trueinterview.io candidate reports), kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator), 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon.md
53. "Amazon Research Scientist Interview Process" (aggregator adapted from trueinterview.io; science loop format shared with AS), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-research-scientist.md
54. "Amazon Machine Learning Engineer Interview Process" (aggregator; includes applied-science loops with hand-written sigmoid, softmax and attention), kevin-2023-code/Tech-Interview-Questions, GitHub, 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/companies/amazon/interview-process-machine-learning-engineer.md
55. "Implement Softmax" (Amazon online assessment; no third-party libraries), kevin-2023-code/Tech-Interview-Questions, GitHub (aggregator), 2026. https://github.com/kevin-2023-code/Tech-Interview-Questions/blob/main/questions/algorithm/implement-softmax/README.md
56. "AI Companies Interview Questions", Amazon AGI section (debug a broken Transformer; build/train a small classifier live; RL fundamentals + coding), ombharatiya/FAANG-Coding-Interview-Questions, GitHub (aggregator), 2026. https://github.com/ombharatiya/FAANG-Coding-Interview-Questions/blob/main/AI-Companies-Interview-Questions.md
57. "Implement decision tree from scratch" (Amazon-tagged MLE onsite practice listing), Codemia, 2025. https://codemia.io/practice/implement-decision-tree-from-scratch-10
58. "Implement k-means from scratch" (Amazon-tagged MLE onsite practice listing), Codemia, 2025. https://codemia.io/practice/implement-k-means-from-scratch
59. "Implement naive Bayes from scratch" (practice listing; not Amazon-attributed), Codemia, 2025. https://codemia.io/practice/implement-naive-bayes-from-scratch-13
60. "Implement logistic regression from scratch" (practice listing; attributed to other companies), Codemia, n.d. https://codemia.io/practice/implement-logistic-regression-from-scratch-8
61. "What to expect in ML coding interview?" (#amazon), Blind, n.d. https://www.teamblind.com/post/ml-coding-interview-kq506p3t
62. "Amazon applied scientist role interview prep", Blind, n.d. https://www.teamblind.com/post/amazon-applied-scientist-role-interview-prep-tz8wfbdk
63. "Amazon Applied Scientist Interview Guide, Alexa International Tech", Medium, n.d. https://medium.com/interview-preparation/amazon-applied-scientist-interview-guide-alexa-international-tech-022d3a056feb
64. theprogclub, "Amazon Applied Scientist Interview Experience", Medium, n.d. https://medium.com/@theprogclub/amazon-applied-scientist-interview-experience-e38e3fb2ec48
65. "The 2026 Amazon Applied Scientist Interview Guide", Interview Kickstart (vendor guide), 2026. https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
66. Applied Scientist Interview Questions, page 67 (medium Trie problem among Amazon reports), Glassdoor, 2025. https://static.glassdoor.com.mx/Interview/applied-scientist-interview-questions-SRCH_KO0,17_IP67.htm
67. Applied Scientist Interview Questions, page 81 (metrics for information retrieval), Glassdoor, n.d. https://static.glassdoor.es/Interview/applied-scientist-interview-questions-SRCH_KO0,17_IP81.htm
68. "Implement K-Means in NumPy or PyTorch" (vectorized, GPU; tied to another company), 1point3acres, 2025. https://www.1point3acres.com/interview/post/7100456
69. "Amazon Applied Scientist Interview Questions & Guide 2026" (vendor guide; k-means; "write a program to calculate AUC"), Dataford, 2026. https://dataford.io/interview-guides/amazon/applied-scientist
70. "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP" (vendor-published debrief; gradient descent without libraries; LP time in technical rounds), OA VO Service, n.d. https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
71. "Amazon Coding Interview Questions: The Real Bar (2026 Guide)" (code usually not executed), copilotinterview.com, 2026. https://copilotinterview.com/blog/amazon-coding-interview-questions
72. Amazon Applied Scientist II Interview Experience & Questions (Python data wrangling alongside DSA), Glassdoor, 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
73. Amazon Applied Scientist Interview Experience & Questions (SDE vs AS interviewer coding style), Glassdoor, 2020-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
74. "Amazon L5 Applied Scientist Fulltime Tech Phone Screen Overview", 1point3acres, n.d. https://www.1point3acres.com/interview/thread/1170176
