# 02. ML Breadth Round: Classical ML, Statistics, Probability, Optimization, Deep Learning, NLP/CV

## Format notes

- **Official loop shape.** Amazon's Applied Scientist interview-prep page describes a 60-minute technical phone screen with a senior scientist or leader, then a loop of four 55-minute interviews with members of the science community. Each loop interview covers technical areas plus Leadership Principles (LPs), and each interviewer is assigned different areas [1].
- **What candidates actually see.** Reported loops usually include one dedicated "Science / ML Breadth" interview next to Science Depth (your own project or paper), coding, a team-specific science application or ML design round, and a Bar Raiser. One Applied Scientist II (L5) loop was reported as "1 Science ML Breadth, 1 Science Depth, 1 Sys Design (Team-specific), 1 coding round and 1 Bar Raiser" [7]. Other loops had five 60-minute interviews over three days [18], or five rounds including a separate "Scalable ML" round [19]. Interviews are about 55-60 minutes, and LP questions are usually built into each one [1][11][26].
- **Who interviews.** Usually an Applied Scientist, Senior Applied Scientist or Principal Applied Scientist from the hiring org or a partner team [1][15]. The breadth round often starts from one of your projects or from something the team works on, then branches out. A Blind commenter on an L5 virtual onsite said breadth "can go anywhere the interviewer wants to take it" [2].
- **Style.** Expect rapid, quiz-like questions with "why" follow-ups. Blind commenters describe it as a quiz-style series of questions on general ML that tests how deep you can go, and say interviewers care about the intuition behind basic concepts, not just the formula [4]. Candidates report being asked to write things out mathematically: losses, regularization, and recurrent shapes in matrix form [13][16].
- **Scope at L5 (2025-2026).** The round reliably covers classical ML (from linear regression to ensembles), bias-variance and regularization, metrics, and probability and statistics [3][4][7][18]. **Transformers and attention are now treated as core breadth even when the team does not use LLMs.** An L5 candidate reported this [2], and an Ads candidate was rejected at the phone screen after being unprepared for attention, transformers and LLMs [5]. Questions from 2025-2026 go into decoder-only inference, KV caching, long-context attention, retrieval encoders and LoRA [6][10][14].
- **L5-specific bar.** Blind commenters say L5 questions resemble L4 questions, but L5 candidates are vetted more for whether they "raise the bar" [27]. A prep debrief says L5 answers should hold up through two or three layers of follow-up, against one layer at L4 [26]. In practice, the L5 signal is depth on demand: deriving results, naming trade-offs and failure modes, and tying a technique to a product decision. The official page and prep guides also stress being able to explain *why* algorithms work [1][23].
- **Method note.** This chapter comes from about 45 web searches. Direct page fetches to Glassdoor, Blind, LeetCode, Medium and similar sites were blocked by this environment's egress policy, so every question marked [Reported] rests on search-result extracts of the cited page. Several sources are prep-site aggregators or older posts (2020-2022). Where a question appears only in a prep guide, a promotional prep service or an unattributed review, and not in a clearly attributable candidate account, it is tagged [Likely] with a "see [n]" pointer.
- **Review note (October 2026).** An adversarial review pass could not re-fetch any cited URL: every fetch failed at DNS resolution, and the session's web-search budget was exhausted. Because of that, the review (a) downgraded [Reported] tags whose only support is an aggregator, a promotional prep service or a review that could not be matched to the question; (b) merged duplicate items; and (c) added 20 new L5-depth questions. All new questions are tagged [Likely] because no new candidate report could be verified. The URLs below have not been re-checked in this pass and should be re-verified when network access is available.

**Tag legend:** [Reported] means a candidate or source reports it for Amazon Applied Scientist or a closely related Amazon science role (cited). [Likely] means a well-grounded inference from the role, recent loop trends or Amazon products.

---

## A. Linear models

### A1. Linear regression

1. What is the closed-form (normal-equation) solution for linear regression, and when would you use gradient descent instead? [Likely] (see [28])
   - What breaks in the closed form when features are highly collinear or when n and d are very large?
   - What is the computational complexity of each approach?
2. What are the assumptions of linear regression, and what happens to your estimates and inference when each is violated? [Reported] [4][5]
   - How would you detect heteroscedasticity or autocorrelated residuals?
3. What is the difference between MLE and MAP for linear regression? [Reported] [15]
   - Under what noise assumption does least squares equal MLE?
   - When does MLE equal MAP?
4. What is the probabilistic interpretation of Bayesian linear regression, and how would you model uncertainty in its predictions? [Reported] [9]
   - How does the posterior predictive variance behave far from the training data?

### A2. Logistic regression

5. How does logistic regression work? Write down the model and its loss. [Reported] [9][21]
   - Why is the decision boundary linear in the original feature space?
   - What does maximizing the likelihood mean for logistic regression? [Reported] [15]
6. What does logistic regression assume about P(y|x)? [Reported] [15]
7. What happens if you put a prior over the weights of logistic regression, and why would you assume one? [Reported] [15]
   - Which prior gives you L1 and which gives L2?
8. What are the differences between L1 and L2 regularization in logistic regression? [Reported] [9][21]
   - Why does L1 produce sparse solutions, geometrically and from a gradient or subgradient view?
9. What is the difference between logistic regression and linear regression, and why not use linear regression for classification? [Reported] [15]
10. Write the loss function for logistic regression and show that it is convex. [Likely] (see [13], which reports "derive loss functions for each model")
    - Derive the gradient. Why does it have such a simple form?
    - What happens to the weights on linearly separable data with no regularization?
11. How would you classify Amazon product reviews when you do not have a well-labeled dataset? [Likely] (Amazon reviews are a core data source; a similar older Glassdoor report is described in search extracts, but its page could not be verified.)

---

## B. Trees, ensembles and other classical models

12. How is a decision tree built? How are splits chosen? [Reported] [15][19]
    - Compare Gini impurity, entropy or information gain, and variance reduction.
    - How do you stop a tree from overfitting (pruning, depth, min-samples)?
13. How is a random forest different from a single decision tree, and why is the forest better? [Reported] [19][11]
    - The interviewer pressed repeatedly: *why* does averaging decorrelated trees reduce variance? [Reported] [19]
14. What is the difference between bagging and boosting? [Reported] [7][15][26]
    - How does the bias-variance decomposition apply to bagging versus boosting? [Reported] [12]
15. How does gradient boosting (GBDT) work? [Reported] [15]
    - What exactly is being fit at each stage (pseudo-residuals or negative gradients)?
    - Why do the learning rate (shrinkage) and the number of trees interact?
16. What is the computational difference between XGBoost and random forest? [Reported] [21][11]
    - What makes XGBoost fast (histogram splits, sparsity-aware splits, second-order terms)?
    - How does it handle missing values?
17. Compare random forest and gradient-boosted trees in terms of bias and variance. When would you pick each? [Likely] (see [29], a prep article whose author calls this common in Google, Amazon and TikTok interviews; not an Amazon candidate report)
18. Walk through KNN, Naive Bayes and SVM: the assumptions behind each, its complexity, and when it fails. [Reported] [18]
    - What is the "naive" assumption, and why does Naive Bayes still classify well when it is violated?
    - What do the C parameter and the kernel choice do in an SVM? [Likely] (see [18])
    - What are the failure modes of k-nearest neighbors (scale sensitivity, curse of dimensionality, query cost), and how would you make it fast at scale? [Likely] (see [28])
19. How would you handle high-cardinality categorical features (for example, seller ID or ASIN) in a tree ensemble? What are the leakage risks of target encoding, and how do you avoid them? [Likely]
20. What is your favorite ML algorithm, and what are its drawbacks? [Reported] [12]

---

## C. Bias-variance, overfitting, regularization and model selection

21. Explain the bias-variance trade-off and how it affects model performance. [Reported] [3][7][11]
    - Write the decomposition of expected squared error.
22. How do you detect and reduce overfitting? [Reported] [3][9][11]
    - What do you do about underfitting?
23. You have two models, each with fit and eval functions. How would you tell which one is overfitting, especially on genuinely new test data? [Reported] [19]
    - Why is "train error low, test error high" not a sufficient answer?
24. Explain regularization mathematically. [Reported] [13]
    - Show that L2 corresponds to a Gaussian prior and L1 to a Laplace prior (MAP view). [Likely] (see [15])
25. Is dropout a form of regularization? Why? [Reported] [19]
    - How are weights handled at inference time with dropout? [Likely] (see [21], a prep aggregator)
26. How do you handle data imbalance, collinearity, feature selection and regularization in one model? [Likely] (see [28])
27. How do you approach feature selection? [Reported] [3][20]
    - Compare filter, wrapper and embedded methods.
28. How do you tune hyperparameters, and how do you avoid leaking validation data into model selection? [Reported] [3]
29. Why is standardization of features preferred, and which models need it? [Reported] [15]
30. When would you use k-fold, stratified, grouped or time-based cross-validation? What goes wrong with a random split on time-series data or on data with many rows per customer? [Likely]

---

## D. Evaluation metrics, calibration and class imbalance

31. How do you evaluate a model? Which metrics would you choose, and why? [Reported] [3][18]
32. How do you handle severe class imbalance? Would you evaluate with precision, recall and F1, or with ROC-AUC? [Likely] (see [26], a promotional prep-service debrief)
    - Why can ROC-AUC look optimistic under heavy imbalance, and when is PR-AUC more informative?
    - Compare resampling, class weighting and threshold tuning. Which do you try first?
33. Explain ROC-AUC. What does it measure, and how would you compute it by hand? [Likely] (see [24], which describes a "write a program to calculate AUC" item)
    - Why does AUC equal the probability that a random positive outranks a random negative?
34. How would you train and evaluate a real-time fraud or abuse model with severe imbalance? [Likely] (see [25]; an Ads Trust loop is reported in [18])
35. How would you pick a decision threshold when false positives and false negatives have different business costs? [Likely]
36. What does it mean for a classifier to be calibrated? How would you measure calibration (reliability diagram, expected calibration error) and fix it (Platt scaling, isotonic regression, temperature scaling)? [Likely]
    - Why does downsampling negatives miscalibrate predicted probabilities, and how do you correct it?
37. Which ranking metrics would you use for search or recommendations (NDCG, MRR, recall@k), and how do they relate to online metrics? [Likely] (Amazon search, ads and recommendation teams; see [32])
38. Your training labels are clicks logged by the current production ranker. What biases does that introduce (position bias, selection bias), and how would you evaluate a new policy offline (inverse propensity scoring, counterfactual evaluation)? [Likely] (see [32])

---

## E. Unsupervised learning and dimensionality reduction

39. Explain k-means clustering. [Reported] [19]
    - How do you choose k? What objective does it minimize, and does it converge to the global optimum?
    - When would you use a GMM or DBSCAN instead?
40. Derive the EM algorithm for a Gaussian mixture model. Why does each iteration never decrease the likelihood, and how does k-means arise as a limiting case? [Likely]
41. What methods can reduce dimensionality, and how do they work? [Reported] [18] (see also [28])
    - Distinguish feature selection from feature extraction.
42. Explain PCA. How does covariance relate to PCA? [Reported] [12]
    - Does PCA keep the "most important features"? (Expect a correction: it builds new axes.)
43. What is the difference between eigenvalue decomposition (EVD) and SVD? [Reported] [15]
    - What is the computational complexity of SVD? [Reported] [12]
    - Would you compute PCA by eigendecomposing the covariance matrix or by SVD of the data matrix? Why?
44. What is the difference between Pearson correlation and covariance? [Reported] [12]
45. What is the difference between a VAE and a plain autoencoder? [Reported] [15]
    - Where does the KL term come from in the ELBO?
46. What is mutual information, and how would you use it? [Reported] [15]
47. What are the differences among supervised, unsupervised and reinforcement learning? [Reported] [9][12]
    - How does reinforcement learning differ from supervised learning in what feedback the learner gets? [Reported] [12]
    - What is a policy gradient, at a high level? [Likely]

---

## F. Probability and statistics

48. What is Bayes' rule? Explain posterior, likelihood and prior. [Reported] [15]
49. What is maximum likelihood estimation, and how does it differ from Bayesian inference? [Likely] (see [23])
    - Derive the MLE for the probability of heads from k heads in n tosses. [Likely] (see [21])
    - What is the difference between a frequentist confidence interval and a Bayesian credible interval?
50. Work a coin-flip probability problem on the whiteboard. [Reported] [4] (the exact wording was not captured; practice variants such as the expected number of tosses until two consecutive heads)
    - How would you get a fair bit from a biased coin? [Likely]
51. A rare disease has prevalence 1 in 1,000, and a test is 98% sensitive with a 1% false-positive rate. Given a positive result, what is P(disease)? [Likely] (attributed to Amazon on prep sites; unverified)
52. State the Central Limit Theorem. Why does it matter for A/B tests? [Reported] [5]
53. What is a p-value? Interpret it in the context of an A/B test. [Reported] [10][20]
    - What is a common misinterpretation of a 95% confidence interval?
54. Walk through hypothesis testing: which statistical tests would you use, and when? [Reported] [5][20]
    - When would you use a t-test, a z-test, chi-squared, or a non-parametric test?
55. How would you design an A/B test for a model launch? [Reported] [20]
    - How do you choose the randomization unit, primary and guardrail metrics, and sample size or power?
    - How would you handle many simultaneous variants (multiple testing)? [Likely] (see [24])
    - How do you measure a recommender when network or interference effects break user-level randomization? [Likely] (see [24])
56. How can you reduce the variance of an A/B-test estimate so the test needs less traffic (for example, CUPED or stratification using pre-experiment covariates)? [Likely]
57. Explain the bootstrap. How would you build a confidence interval for a metric with no simple variance formula, such as AUC or a median order value? [Likely]
58. Give an example of Simpson's paradox. When you only have observational data, how would you estimate the causal effect of a feature or a promotion (propensity scores, difference-in-differences), and what assumptions does each method need? [Likely]
59. What is KL divergence, and how does it relate to cross-entropy loss in practice? [Reported] [9][20][21]
    - Is KL symmetric? When does forward versus reverse KL matter?
    - Relate entropy, cross-entropy and KL to information gain in decision trees. [Likely]

---

## G. Optimization

60. Describe SGD and Adam, and say when you would use each. [Reported] [3] (see also [24][25])
    - What do momentum and the second-moment estimate each fix?
    - Why might Adam generalize worse than SGD in some settings? What does AdamW change?
61. Which loss functions and optimizers did you use in your project, and why those, given the alternatives? [Reported] [16][3]
62. What changes if you swap cross-entropy for MSE in a classification loss? [Reported] [8]
    - Why do sigmoid or softmax with MSE give weak gradients when the prediction is confidently wrong?
63. What is the role of the learning rate, and how do warmup and decay schedules help? [Likely]
64. What happens to convergence with very large batch sizes, and how do you compensate? [Likely] (see [23])
65. How would you diagnose a training loss that diverges or goes NaN? [Likely] (gradient explosion is reported in [11])
    - What does gradient clipping do?
66. Define convexity. Which common ML objectives are convex and which are not? Why do deep nets still train well? [Likely]
67. Compare first-order methods with second-order or quasi-Newton methods (Newton's method, L-BFGS). Why are second-order methods common for logistic regression but rare for deep networks? [Likely]

---

## H. Deep learning fundamentals

68. Explain backpropagation for an MLP. [Reported] [4][5]
    - Derive the gradient for one hidden layer with a sigmoid or ReLU.
69. Why use a sigmoid activation, and what are its problems in deep networks? [Reported] [12]
    - Compare ReLU, GELU and tanh. What is the dying-ReLU problem?
70. Explain the vanishing and exploding gradient problems and how to address them. [Reported] [16][26]
71. Explain residual connections. Why do they make very deep networks trainable, and how do they interact with normalization (pre-norm versus post-norm)? [Likely]
72. Explain batch normalization. Why does it help, and how does it behave differently at train and inference time? [Reported] [17]
    - Why do Transformers use LayerNorm instead? What is RMSNorm? [Likely] (a 2024 Glassdoor review mentions "LLM architecture and various normalization, regularization etc methods"; see [6])
73. What is label smoothing, and what is the intuition behind it? [Likely] (see [23])
    - How does it affect calibration?
74. How do you initialize weights, and why does it matter (Xavier or He)? [Likely]
75. How does knowledge distillation work? What do the soft targets and the temperature contribute, and when does a distilled student beat a student trained on hard labels? [Likely]
76. A model is too slow or too expensive to serve at Amazon scale. Compare post-training quantization, quantization-aware training, pruning and distillation. What accuracy risks does each carry? [Likely] (see [22])

---

## I. Sequence models and classic NLP

77. Explain LSTMs and how they differ from a vanilla neural network or RNN. [Reported] [16]
    - Why is an LSTM better than a plain RNN? [Reported] [16]
    - How does backpropagation through time work? [Reported] [16]
78. How does the shape of the input change as data moves through a recurrent network? Write it in matrix form. [Reported] [16]
79. What is the difference between LSTMs and GRUs? [Reported] [15]
    - How do GRU cells address the vanishing gradient problem? How does a BiLSTM work? [Likely] (see [21])
80. Explain the difference between LSTMs and Transformers mathematically. [Reported] [13]
    - What are the compute and parallelism trade-offs? How do they handle long-term dependencies?
81. Which text representations are there? What is the difference between Word2Vec, GloVe and FastText? [Reported] [15]
82. What is Word2Vec, how is it trained, what are the alternatives, and what are its drawbacks? [Reported] [17]
    - Compare skip-gram and CBOW, and explain negative sampling.
83. How is BERT trained? What are its constraints and advantages, and which other members of the BERT family do you know? [Reported] [17]
    - Why might BERT not work in a given use case? [Reported] [17]
84. What are embeddings, and how do you get sentence or token embeddings from BERT? [Reported] [12]
    - CLS token or mean pooling? Why do raw BERT embeddings make poor sentence similarity features?
85. Compare subword tokenizers (BPE, WordPiece, SentencePiece or unigram). How does the choice of tokenizer affect multilingual text, product titles full of model numbers, and sequence length? [Likely] (multilingual Alexa work; see [31])
86. What is statistical language modeling, and what does perplexity measure? [Likely] (see [21]; language modeling is reported in Glassdoor extracts summarized in [7])

---

## J. Transformers and LLMs (now a breadth staple)

87. Explain the Transformer architecture: encoder, decoder, and what each component does. [Reported] [3][8][10]
    - Why scale the dot product by sqrt(d_k)?
    - Why multi-head attention?
88. Why do Transformers need positional encoding? Compare sinusoidal, learned and rotary (RoPE). [Likely] (positional encoding appears in candidate-report extracts for 2025 transformer-heavy screens; see [14])
89. What happens if you set Wq = Wk = Wv in a Transformer? [Reported] [8]
90. How does Transformer inference work for decoder-only models? [Reported] [6]
91. How does KV caching work during autoregressive inference? [Reported] [6]
    - What is the memory cost? How do multi-query and grouped-query attention reduce it? [Likely]
92. What makes standard self-attention expensive for long contexts? [Reported] [6]
    - What is the time and memory complexity in sequence length?
    - How does FlashAttention speed attention up without changing the result? [Likely]
93. Explain Longformer and how its attention mechanism scales to long sequences. [Reported] [6]
    - What are the trade-offs between full attention, sparse or local attention, and long-context architectures? [Reported] [6]
94. What is the difference between a cross-encoder and a bi-encoder for retrieval? [Reported] [6]
    - When would you use each in a retrieve-then-rerank pipeline?
95. Explain LoRA. [Reported] [10]
    - Compare parameter-efficient fine-tuning with full fine-tuning. Which parameters and optimizer states are trained? [Likely] (see [22])
96. Which modern LLM training techniques do you know (pretraining, SFT, preference optimization)? [Likely] (see [34]: a 2026 Glassdoor review for an AGI-team role, not confirmed as an Applied Scientist report)
    - Compare RLHF and DPO. What does the DPO beta control? [Likely] (Alexa postings list SFT, DPO, RLHF and RLAIF [31])
97. How do decoding strategies differ (greedy, beam, top-k, top-p, temperature)? [Likely]
98. What is an encoder-decoder model such as T5, and how do you optimize it for inference? [Reported] [11]
99. What is a mixture-of-experts layer? What do you gain in compute per token, and what problems does routing introduce (load balancing, expert collapse, serving memory)? [Likely]
100. What do scaling laws say about splitting a fixed compute budget between model size and training tokens? When would you deliberately train a smaller model on more data? [Likely]
101. How would you reduce hallucination in an LLM-powered assistant? Where does retrieval-augmented generation (RAG) help, and where does it not? [Likely] (Amazon's shopping assistant uses LLMs including Amazon Nova, built on Bedrock [30])
102. How would you evaluate an LLM-based feature before launch? Compare reference-based metrics, LLM-as-a-judge and human evaluation, and name the biases of an LLM judge (position, length, self-preference). [Likely] (see [30])

---

## K. Computer vision fundamentals

103. How does a CNN work, and how is it different from a feed-forward network? [Reported] [17]
     - What are parameter sharing and translation equivariance?
104. What is a receptive field, and how does it grow with depth, stride and dilation? [Reported] [17]
105. How do you reduce overfitting in CNNs (augmentation, dropout, weight decay, transfer learning)? [Likely] (see [17])
106. How does a Vision Transformer turn an image into tokens? When does a ViT beat a CNN, and why does it need more data or stronger augmentation? [Likely]
107. Describe how the Segment Anything Model (SAM) works. [Likely] (see [34]; the review could not be matched to the question or confirmed as an Applied Scientist role)
108. Compare two-stage and one-stage object detectors. What are IoU and non-maximum suppression? [Likely]
109. Explain contrastive multimodal learning such as CLIP, and diffusion models at a high level. [Likely] (see [22])
     - Write the InfoNCE loss. What do the temperature and the number of negatives control?

---

## L. Applied breadth: production, ranking, forecasting and data issues

110. A model performs well offline but poorly online. What do you check? [Likely] (see [22])
     - How do you detect data drift or feature drift, and what would you monitor?
111. How would you handle noisy labels or non-stationary data? [Likely] (see [22])
112. Implement a rolling average when some dates are missing. What data-manipulation pitfalls arise? [Reported] [3][5] (this sometimes appears inside the breadth or screen round)
113. How does collaborative filtering work, and what are its cold-start problems? [Likely] (Amazon item-to-item collaborative filtering [37]; recommender and learning-to-rank skills are listed in [32]; a recommender design question is reported in [33])
     - Why does item-to-item collaborative filtering scale better than user-based collaborative filtering?
     - Compare two-tower retrieval with a ranking model. Why have separate stages?
114. When would you use a multi-armed bandit instead of a fixed A/B test? Compare epsilon-greedy, UCB and Thompson sampling, and explain how a contextual bandit changes the problem. [Likely]
115. How would you forecast demand for millions of products, including new products with no sales history? Compare per-series models (ARIMA, exponential smoothing) with a global deep model such as DeepAR or a pretrained time-series model such as Chronos. [Likely] (Amazon forecasting research [35][36])
     - Why would you produce a probabilistic forecast and train with a quantile (pinball) loss instead of MSE?
116. How would you measure whether a new ranking or recommendation model improved the customer experience, beyond offline metrics? [Likely] (see [24])
117. How would you choose between a simple interpretable model and a deep model for a business-critical launch? [Likely] (tests L5 judgment; the official prep page asks candidates to explain trade-offs [1])

---

## Prioritization for an L5 loop (based on report frequency)

- **Highest frequency in reports:** bias-variance (21), bagging vs boosting (14), regularization L1/L2 (8, 24), overfitting (22), decision tree and random forest (12, 13), logistic regression (5-9), Transformer internals (87-94), loss functions (61, 62), and p-values or A/B tests (53, 55).
- **Rising in 2025-2026:** KV caching, long-context attention, cross-encoder vs bi-encoder, LoRA, and LLM training techniques (90-96).
- **Expect derivations:** loss functions, regularization as MAP, EM, LSTM vs Transformer in math, and RNN shapes in matrix form (10, 24, 40, 78, 80).
- **Team-dependent L5 extras (unverified, [Likely]):** calibration and counterfactual evaluation for Ads and Search (36, 38), forecasting for supply chain (115), and distillation or quantization for serving cost (75, 76).

---

## References

1. Amazon.jobs, "Applied Scientist Interview Prep" (official), current as of 2026. https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep
2. Blind (teamblind.com), "Amazon Applied Scientist (L5) Virtual Onsite", circa 2025. https://www.teamblind.com/post/amazon-applied-scientist-l5-virtual-onsite-navwesml
3. Blind, "Amazon applied scientist interview-onsite", circa 2024. https://www.teamblind.com/post/amazon-applied-scientist-interview-onsite-t01hpnc4
4. Blind, "Science Breadth Interview @Amazon", 2024, plus related Blind AS prep threads summarized in the same search. https://www.teamblind.com/post/science-breadth-interview-amazon-hfwqznrb
5. Blind, "Applied Scientist Interview at Amazon Ads". https://www.teamblind.com/post/applied-scientist-interview-at-amazon-ads-zoerec5t
6. Glassdoor, "Amazon Applied Scientist Interview Experience & Questions" (includes an April 2026 report on decoder-only inference, KV caching, Longformer, cross-encoder vs bi-encoder), 2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24.htm
7. Glassdoor, "Amazon Applied Scientist II Interview Experience & Questions" (loop: 1 Science ML Breadth, 1 Science Depth, 1 Sys Design, 1 coding, 1 Bar Raiser; bias-variance; bagging vs boosting), 2025-2026. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-II-Interview-Questions-EI_IE6036.0,6_KO7,27.htm
8. Glassdoor, "Amazon Applied Scientist Interview Questions" page 3 (Bangalore report: cross-entropy vs MSE; Wq = Wk = Wv; transformer basics), circa 2024. https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Interview-Questions-EI_IE6036.0,6_KO7,24_IP3.htm
9. Glassdoor, "Amazon Applied Science Intern Interview Experience & Questions" (logistic regression formula, L1 vs L2, KL vs cross-entropy, Bayesian linear regression, overfitting). https://www.glassdoor.com/Interview/Amazon-Applied-Science-Intern-Interview-Questions-EI_IE6036.0,6_KO7,29.htm
10. Glassdoor, "Amazon Web Services Applied Scientist Interview Experience & Questions" (explain transformers, p-value, LoRA). https://www.glassdoor.com/Interview/Amazon-Web-Services-Applied-Scientist-Interview-Questions-EI_IE7470741.0,19_KO20,37.htm
11. Glassdoor, "Amazon Applied Scientist Intern Interview Experience & Questions" (Sunnyvale 2025: transformers at root level, T5, optimization; RF/XGBoost; bias-variance; LPs in every round). https://www.glassdoor.com/Interview/Amazon-Applied-Scientist-Intern-Interview-Questions-EI_IE6036.0,6_KO7,31.htm
12. LeetCode Discuss, "Amazon Applied Scientist | L6 Luxembourg | Passed, freeze | 2022". https://leetcode.com/discuss/interview-question/4421594/Amazon-Applied-Scientist-or-L6-Luxembourg-or-Passed-freeze-or-2022/
13. LeetCode Discuss, "Amazon Applied Scientist Phone Interview" (LSTM vs Transformers mathematically; derive losses; regularization mathematically). https://leetcode.com/discuss/post/543684/amazon-applied-scientist-phone-interview/
14. LeetCode Discuss, "Amazon online 1st round, Bengaluru, Applied Scientist", May 2025 (LLM theory, transformer internals, ML and stats). https://leetcode.com/discuss/post/6871782/amazon-online-1st-round-bengaluru-applie-uiwk/
15. Akanksha Paul, "Interview Experience for Applied Scientist at Amazon", Medium, 2020. https://akanksha-paul.medium.com/interview-experience-for-applied-scientist-at-amazon-5ff5003375dd
16. theprogclub, "Amazon Applied Scientist Interview Experience", Medium, circa 2023. https://medium.com/@theprogclub/amazon-applied-scientist-interview-experience-e38e3fb2ec48
17. ultronmaster, "Amazon Applied Scientist Interview Experience as Fresher Undergrad", Medium, circa 2021 (Amazon Ads team). https://medium.com/@ultronmaster/amazon-applied-scientist-interview-experience-as-fresher-undergrad-b9a2c5b40a63
18. Chaitanya Mundle, "My interview experience for the Applied Scientist II role at Amazon", Medium (Advertising Trust team). https://medium.com/@mundle1997.cm/my-interview-experience-for-the-applied-scientist-ii-role-at-amazon-738aba21f611
19. GeeksforGeeks, "Amazon Interview Experience for Applied Scientist" (Amazon India campus, October 2023). https://www.geeksforgeeks.org/interview-experiences/amazon-interview-experience-for-applied-scientist/
20. Grapevine, "What to expect in the Amazon Applied Scientist interview?" (AS2 breadth: KL divergence, statistical tests, p-value, feature selection, A/B testing), circa 2024. https://www.grapevine.in/post/what-to-expect-in-the-amazon-applied-scientist-interview-a649970d-b602-4e78-ac34-9d5fe491a7cd
21. IGotAnOffer, "Amazon Applied Scientist Interview (process, questions, prep)", 2026 (prep aggregator). https://igotanoffer.com/en/advice/amazon-applied-scientist-interview
22. Interview Kickstart, "The 2026 Amazon Applied Scientist Interview Guide", 2026 (prep guide). https://interviewkickstart.com/interview-questions/companies/amazon/applied-scientist
23. Interview Kickstart, "The 2026 Amazon Senior Applied Scientist Interview Guide", 2026 (prep guide). https://interviewkickstart.com/interview-questions/companies/amazon/senior-applied-scientist
24. Dataford, "Amazon Applied Scientist Interview Questions & Guide 2026", 2026 (prep guide). https://dataford.io/interview-guides/amazon/applied-scientist
25. Dataford, "Amazon DSP Applied Scientist Interview Questions 2026", 2026 (prep guide). https://dataford.io/interview-guides/amazon-dsp/applied-scientist
26. OA VO Service, "Amazon Applied Scientist Interview Debrief: Five-Round Loop + LP" (prep-service write-up; promotional). https://oavoservice.com/en/articles/amazon-applied-scientist-vo-loop-lp-ml-coding
27. Blind, "Amazon AS L4 vs L5" and "Amazon Applied Scientist L4 vs Applied Scientist II L5" (level-bar discussion). https://www.teamblind.com/post/amazon-as-l4-vs-l5-n0zunqgt and https://www.teamblind.com/post/amazon-applied-scientist-l4-vs-applied-scientist-ii-l5-aasmppdi
28. Jaspinder Singh, Substack series "Amazon Applied Scientist Interview Question: ..." (closed form vs GD; dimensionality reduction; logistic regression; imbalance and collinearity; kNN from scratch), 2026; author prep posts, not verified candidate reports. https://jaspinders30.substack.com/p/amazon-applied-scientist-interview-7cd ; https://jaspinders30.substack.com/p/amazon-applied-scientist-interview ; https://jaspinders30.substack.com/p/how-does-logistic-regression-work ; https://jaspinders30.substack.com/p/amazon-applied-scientist-interview-f15 ; https://jaspinders30.substack.com/p/amazon-ml-scientist-interview-question
29. penggongting, "In Interview: How to answer 'Compare Random Forest and Gradient Boosting Decision Tree'?", Medium (author reports it as common across Google, Amazon and TikTok interviews). https://medium.com/@penggongting/in-interview-how-to-answer-compare-random-forest-and-gradient-boosting-decision-tree-105de35cff3b
30. About Amazon, "Amazon Rufus AI assistant personalized shopping features" (LLMs including Amazon Nova on Bedrock; renamed Alexa for Shopping in May 2026). https://aboutamazon.com/news/retail/amazon-rufus-ai-assistant-personalized-shopping-features
31. Amazon.jobs, "Applied Scientist, Alexa International Tech", Job ID 10490052 (SFT, DPO, RLHF, RLAIF). https://www.amazon.jobs/en/jobs/10490052/applied-scientist-alexa-international-tech
32. Amazon.jobs, "2027 Applied Science Internship - Recommender Systems / Information Retrieval", Job ID 10564598 (collaborative filtering, learning to rank). https://www.amazon.jobs/en/jobs/10564598/2027-applied-science-internship-recommender-systems-information-retrieval-machine-learning-united-states-phd-student-science-recruiting
33. Blind, "Interview Experience Amazon Applied Scientist" (deep learning questions; recommender design round; LP rounds). https://www.teamblind.com/post/Interview-Experience-Amazon-Applied-Scientist-hWgaGsmO
34. Glassdoor, individual Amazon interview reviews, 2026, surfaced in search (one reports many "modern LLM training technique" questions for an AGI-team role; one reports a Segment Anything Model question). Which review contains which question could not be verified because page fetches were blocked. https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104288349.htm ; https://fr.glassdoor.ch/Entretien/Amazon-Entretien-E6036-RVW104864395.htm ; https://www.glassdoor.fr/Entretien/Amazon-Entretien-E6036-RVW101665167.htm
35. Salinas, Flunkert, Gasthaus and Januschowski (Amazon), "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", arXiv:1704.04110, 2017 (published in International Journal of Forecasting, 2020). https://arxiv.org/abs/1704.04110
36. Ansari et al. (Amazon), "Chronos: Learning the Language of Time Series", arXiv:2403.07815, 2024. https://arxiv.org/abs/2403.07815
37. Linden, Smith and York, "Amazon.com Recommendations: Item-to-Item Collaborative Filtering", IEEE Internet Computing 7(1), 2003. https://doi.org/10.1109/MIC.2003.1167344
