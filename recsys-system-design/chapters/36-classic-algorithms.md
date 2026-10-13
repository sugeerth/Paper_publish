# 36. Classic Algorithms Foundations: Neighborhoods, Factorization, Linear Autoencoders, Embeddings, and Why Baselines Still Win

## Overview

Every modern recommender stack, from two-tower retrieval to generative sequence models, sits on top of a small set of ideas that were mostly worked out between 1994 and 2019:

- **Neighborhood collaborative filtering (CF).** User-kNN, and later the item-to-item CF that Amazon made famous [1][2].
- **Latent-factor matrix factorization (MF).** Popularised by the Netflix Prize (Funk-SVD, SVD++, timeSVD++) [3][4], then adapted to implicit feedback with confidence-weighted ALS (Hu, Koren and Volinsky) [5]. It was later industrialised on Spark and Giraph [6][7].
- **Pairwise ranking objectives.** BPR [8], plus the WARP and BPR losses in hybrid factorization libraries such as LightFM [9].
- **Sparse and dense linear item-item models.** SLIM [10] and its closed-form descendant EASE [11].
- **Skip-gram item embeddings.** item2vec and prod2vec, later Airbnb's listing embeddings [12][13][14].
- **Content-based and hybrid models** [9][15].

These models matter for more than history. Between 2019 and 2022 a series of reproducibility studies showed a pattern [16][17][18][19][20]. Many neural recommenders published at top venues could not be reproduced. Of those that could, most were beaten by properly tuned item-kNN, graph random walks (P3alpha/RP3beta), SLIM, EASE or iALS. This chapter covers each classic family as a system design: what problem it solves, what it computes and how it scales. It then covers what the reproducibility crisis means for anyone building or judging a recommender in 2026.

A note on sourcing. Live web search was unavailable for most of this chapter's research session, and the paper hosts (arXiv, ACM, company blogs) could not be reached. The project repositories on GitHub were read directly: Dacrema et al. evaluation code [16], Google Research iALS and NCF-vs-MF code [18][19], LightFM [9], SLIM [10], an EASE implementation [11b], Microsoft Recommenders/SAR [15][21], implicit [22], Elliot [23], RecBole [24] and Anelli et al. [20]. Numbers that come from the canonical papers but could not be re-checked in this session are marked "approximately".

```
             The classic recommender family tree (simplified)

  Explicit ratings (Netflix Prize era)       Implicit feedback (clicks, plays, buys)
  ------------------------------------       ---------------------------------------
  user-kNN (GroupLens, 1994)                 item-to-item CF (Amazon, 1998-2003)
        |                                          |
  item-kNN (Sarwar et al., 2001) ---------------+  +--> SAR (co-occurrence + decay)
        |                                       |  |
  Funk-SVD / biased MF (2006)                   |  +--> SLIM (2011) --> EASE (2019)
        |                                       |               \        /
  SVD++ / timeSVD++ (Koren 2008-09)             |             linear autoencoders
        |                                       |
        +--> WALS / iALS (Hu-Koren-Volinsky 2008) --> iALS++ / revisited iALS (2021-22)
        +--> BPR-MF (Rendle 2009) --> WARP / LightFM hybrid (2015)
                                         |
  word2vec (2013) --> item2vec / prod2vec (2015-16) --> Airbnb listing emb. (2018)
                                         |
                       two-tower / sequential / generative models (later chapters)
```

## 1. Neighborhood collaborative filtering

### 1.1 User-based kNN

**Problem.** Predict a user's interest in an item from the behaviour of similar users.

**Mechanism.** Build user-user similarities from co-rated items, usually with Pearson correlation or cosine similarity. Pick the k most similar neighbours who interacted with the target item. Predict a similarity-weighted average of their (mean-centred) ratings.

**Why it fell out of favour at scale.** The user space is far larger and changes far faster than the item space. A user's neighbourhood changes with every new interaction, so the similarity matrix has to be recomputed constantly. Sparse overlap between users also makes similarities noisy. Amazon's engineers gave exactly these reasons for moving away from user-based CF [1][2].

### 1.2 Item-to-item CF (Amazon)

**Problem.** Amazon needed recommendations for tens of millions of customers and millions of products. Results had to be computed in real time on every page view, and had to react to what the customer had just put in the cart [1].

**Architecture.** The key move was to swap the roles of users and items. Item-item relatedness is much more stable than user-user similarity, so it can be computed offline. The online step is then a cheap lookup and merge:

```
 OFFLINE (batch, periodic)                          ONLINE (per request, ms)
 ------------------------------------------------   -------------------------------
 for each item i1 in catalog:                        user context:
   for each customer c who bought i1:                  recent purchases / cart / views
     for each item i2 bought by c:                           |
        record co-purchase(i1, i2)                           v
   for each i2: sim(i1,i2) = cosine over customers    lookup top-N similar items
 store top-N similar items per item  ------------->   for each seed item
   (item -> [(item, score), ...])                            |
                                                             v
                                                      merge, de-dup, remove owned,
                                                      re-rank (business rules)
```

**Scale properties.** The worst-case offline cost is O(N^2 M), where N is the number of items and M the number of customers. In practice it is closer to O(NM), because most customers buy very few items. Online cost depends only on how many items the user has interacted with, not on catalogue size or user count [1]. The 2003 IEEE Internet Computing paper later received the journal's "test of time" recognition. Its 2017 retrospective, "Two Decades of Recommender Systems at Amazon.com", describes how the basic item-to-item idea was refined: correcting relatedness for purchase base rates, handling time and context, and using it far beyond the "customers who bought" widget [2].

**Academic formalisation.** Sarwar, Karypis, Konstan and Riedl (WWW 2001) studied item-based CF systematically. They compared cosine, adjusted-cosine and correlation similarities, and showed that item-based models could be precomputed and gave better quality than user-based kNN on sparse data [25].

### 1.3 SAR: item-kNN with time decay as a production recipe

Microsoft's open-source Recommenders repository ships **SAR (Simple Algorithm for Recommendation)**, a cleaned-up item-kNN that works as a strong first baseline [15][21]:

1. **Co-occurrence matrix C (m x m).** c_ij counts the users who interacted with both item i and item j.
2. **Similarity S by rescaling C.** Jaccard is s_ij = c_ij / (c_ii + c_jj - c_ij), a "compromise between popularity and discovery". Lift is s_ij = c_ij / (c_ii * c_jj), which favours niche and serendipitous items. Raw counts favour popular items.
3. **Affinity matrix A (n x m) with event weights and exponential time decay.** a_ij = sum over events k of w_k * (1/2)^((t0 - t_k)/T), where T is a half-life.
4. **Scores = A x S.** Previously seen items can optionally be filtered out [21].

The example notebook uses Jaccard similarity, a 30-day half-life and top-10 on MovieLens. It reports NDCG@10 of about 0.385 and Precision@10 of about 0.33 [21]. The notebook also says plainly that the single-node implementation is not meant for very large data, because the dense item-item work becomes expensive [21]. That trade-off holds for every item-item model: quality is strong and the model is simple, but memory grows as O(m^2) unless similarities are sparsified to top-k per item.

### 1.4 Graph random walks: P3alpha and RP3beta

The Dacrema et al. evaluation suite includes two graph baselines next to kNN: **P3alpha** and **RP3beta** [16]. Both score items by the probability of a three-step random walk user -> item -> user -> item on the bipartite interaction graph. P3alpha raises transition probabilities to a power alpha. RP3beta also divides by item popularity raised to beta, which pushes the ranking towards the long tail. They are item-item models in disguise: the walk collapses into a sparse item-item matrix, so serving is the same as kNN. In the reproducibility studies these cheap models often matched or beat neural baselines [16][17].

## 2. Matrix factorization for explicit feedback: Funk-SVD, SVD++, timeSVD++

### 2.1 The Netflix Prize context

The Netflix Prize (2006-2009) offered USD 1M for a 10% RMSE improvement over Cinematch. It released about 100M ratings from about 480K users on about 17,770 movies. Latent-factor models became the mainstream approach as a result [3][4]. The summary of the winning techniques by Koren, Bell and Volinsky, "Matrix Factorization Techniques for Recommender Systems" (IEEE Computer, 2009), is the standard reference [3].

**Biased MF (Funk-SVD).** Predict r_ui = mu + b_u + b_i + q_i^T p_u. Here mu is the global mean, b_u and b_i are user and item biases, and p_u and q_i are f-dimensional factors. Fit by stochastic gradient descent (SGD) on observed ratings only, with L2 regularisation. The bias terms explain much of the variance in ratings. Leaving them out is still one of the most common reasons a "baseline" looks weak [3][26].

**SVD++.** In "Factorization Meets the Neighborhood" (KDD 2008), Koren added implicit information: the set N(u) of items the user rated, whatever the ratings were. The user representation becomes p_u + |N(u)|^(-1/2) * sum over j in N(u) of y_j. The fact that a user chose to rate an item carries signal. The same paper combined factor and neighbourhood models in one objective [4].

**timeSVD++.** This variant lets biases and user factors drift over time, because user taste and item popularity are not stationary. It was one of the strongest single models in the winning ensemble [3].

**Lesson from production.** The Netflix Prize also showed the cost of ensembles. The grand-prize blend combined on the order of a hundred models. Netflix said publicly that it adopted the core MF and RBM components, but did not put the full blend into production, because the extra accuracy did not justify the engineering cost. The company was also moving from DVD ratings to streaming implicit signals [3]. That shift, from rating prediction (RMSE) to ranking over implicit feedback, is what the rest of this chapter covers.

## 3. Implicit-feedback MF: confidence-weighted ALS (Hu, Koren, Volinsky)

### 3.1 Problem

Most real systems never see ratings. They see views, plays, purchases and dwell time. Such data has several properties that break rating-style MF [5]:

- There is no negative feedback. A missing entry may mean dislike, or may mean the user never saw the item.
- The data is noisy. A watched show may have been playing in the background.
- The numeric value shows confidence, not preference. Watching a show many times makes it more likely the user likes it, but the count is not a rating.
- Evaluation must be ranking-based.

### 3.2 Model

Hu, Koren and Volinsky (ICDM 2008) built the method on TV-viewing data from a digital set-top box service: roughly 300K households and about 17K programmes [5]. They split each observation r_ui into:

- **Preference** p_ui = 1 if r_ui > 0, else 0.
- **Confidence** c_ui = 1 + alpha * r_ui. A log variant, c_ui = 1 + alpha * log(1 + r_ui / epsilon), is also proposed. The paper reports alpha of about 40 working well on their data.

The objective covers **all** user-item pairs, observed or not:

```
 min over x_*, y_*:  sum_{u,i} c_ui (p_ui - x_u^T y_i)^2
                     + lambda ( sum_u ||x_u||^2 + sum_i ||y_i||^2 )
```

Summing over all m*n cells is the point of the method: unobserved cells act as low-confidence negatives. The engineering problem is that this dense sum is intractable if done naively.

### 3.3 The ALS trick that makes it scale

With item factors Y fixed, each user's factor has a closed form:

```
 x_u = (Y^T C^u Y + lambda I)^(-1) Y^T C^u p(u)
     where  Y^T C^u Y = Y^T Y  +  Y^T (C^u - I) Y
                        ^^^^^     ^^^^^^^^^^^^^^^^
                    precompute     only n_u nonzero rows (items u touched)
                    once, O(f^2 n)
```

Y^T Y is shared by all users. The correction term only touches the n_u items the user interacted with, so one sweep over users costs O(f^2 N + f^3 m). N is the number of non-zeros, f the number of factors and m the number of users. The cost is linear in the data, not in the full matrix [5]. Items are handled the same way. Every user (and every item) solves independently, so the problem is embarrassingly parallel. That is why ALS rather than SGD became the default for distributed implicit MF.

The same paper also gives **explanations**. Because the solve is linear, a user's score for an item can be written as a weighted sum over the items they consumed. "Because you watched X" falls out of the algebra [5].

### 3.4 Industrial implementations

| Implementation | Algorithm(s) | Execution model | Notes |
|---|---|---|---|
| Spark MLlib ALS [6] | explicit ALS, implicit ALS (Hu et al.) | distributed, block-partitioned | parameters rank, regParam, alpha, implicitPrefs; coldStartStrategy to drop NaN predictions |
| Facebook Giraph CF [7] | SGD and ALS on Apache Giraph | graph-parallel "rotational" scheme | reported on graphs of roughly 100B ratings (approx.) |
| benfred/implicit [22] | iALS (incl. conjugate gradient), BPR, logistic MF, item-kNN (cosine/TF-IDF/BM25) | multi-threaded Cython/OpenMP, CUDA kernels for ALS and BPR | integrates Annoy, NMSLIB and Faiss for ANN; advises disabling BLAS threading for big speedups |
| Google iALS / iALS++ [18] | iALS, block (subspace) iALS++ | single-machine research code | embedding dims up to 2048 (ML-20M) and 8192 (MSD) |
| TensorFlow WALS (historical) | weighted ALS | distributed TF | basis of earlier Google Cloud recommendation tutorials (approx.) |

**Facebook's rotational hybrid.** Facebook's 2015 engineering post "Recommending items to more than a billion people" describes running MF on Giraph [7]. Workers hold partitions of users, and item factors rotate between workers in a ring. This avoids broadcasting the full item matrix, which standard Giraph and Spark ALS would do. Facebook reported that the approach handled datasets of around 100 billion ratings, which was much larger than the open-source alternatives of the time could handle. The post also covers approximate top-k serving by grouping items. Figures are approximate here because the post could not be re-read during this session.

**The implicit library's practical tips.** Set OPENBLAS_NUM_THREADS=1 or MKL_NUM_THREADS=1 so BLAS threading does not fight the library's own OpenMP parallelism. Use GPU kernels for ALS and BPR. Serve with an ANN index (Annoy, NMSLIB, Faiss) rather than brute-force scoring [22].

### 3.5 iALS revisited (2021-2022)

Rendle, Krichene, Zhang and Koren re-ran iALS on the benchmarks that neural papers use. These were the NCF setups (ML-1M, Pinterest) and the Mult-VAE setups (ML-20M, Million Song Dataset) [18]. The released configurations show what "properly tuned" means [18]:

| Benchmark | Dim | Reg | Unobserved weight | Epochs |
|---|---|---|---|---|
| ML-1M (NCF protocol) | 192 | 0.007 | 0.3 | 12 |
| Pinterest (NCF protocol) | 192 | 0.02 | 0.007 | 16 |
| ML-20M (VAE protocol) | 256 (2048 in headline table) | 0.003 | 0.1 | 16 |
| MSD (VAE protocol) | 256 (8192 in headline table) | 0.002 | 0.02 | 16 |

Three details carried most of the gain [18]:

1. A **separate weight on unobserved entries**, rather than the implicit weight of 1 in the original formulation.
2. **Frequency-scaled regularisation** (the "reg exp" exponent).
3. **Much larger embedding dimensions than papers usually allow**, up to 2048 and 8192.

The companion **iALS++** paper uses block (subspace) coordinate descent so that these large dimensions are cheap enough to train [18]. The conclusion of the revisiting paper was that a well-tuned iALS is competitive with the strongest published results on these benchmarks, including Mult-VAE and EASE. The numbers are in the paper's Tables 3 and 4 and are not reproduced here.

## 4. Pairwise learning to rank: BPR and WARP

### 4.1 BPR (Rendle et al., UAI 2009)

**Problem.** Pointwise regression on 0/1 implicit data optimises the wrong thing. What matters is that observed items rank above unobserved ones for each user.

**Model.** BPR assumes that for user u, an observed item i should be preferred to an unobserved item j. It maximises the posterior

```
 sum over (u,i,j) in D_S of  ln sigma( xhat_uij )  -  lambda ||Theta||^2
   with xhat_uij = xhat_ui - xhat_uj
```

This is a smooth surrogate for AUC [8]. BPR is a **criterion**, not a model. xhat can come from MF (BPR-MF), item-kNN (BPR-kNN) or anything differentiable.

**Training.** The full set of (u, i, j) triples is huge, so BPR uses **LearnBPR**: SGD with bootstrap sampling of triples, drawn uniformly with replacement. The paper argues this converges much faster than user-wise sweeps [8].

**Lessons from later practice.**
- Uniform negative sampling gives mostly easy negatives late in training, so gradients vanish. This led to popularity-aware, adaptive and hard negative sampling, and to WARP.
- BPR-MF is in every evaluation suite. In Dacrema et al., BPRMF sits next to iALS, PureSVD, FunkSVD and SLIM-BPR [16].
- It needs careful tuning: learning rate, regularisation and the number of sampled negatives all matter a lot.

### 4.2 WARP and LightFM

**WARP (Weighted Approximate-Rank Pairwise).** For each positive, sample negatives until one violates the margin. The number of draws needed estimates the positive's rank, and the update is weighted by a function of that rank. This focuses learning on the top of the list. LightFM implements both BPR and WARP losses, and its quickstart uses LightFM(loss='warp') [9].

## 5. Sparse and dense linear item-item models: SLIM and EASE

### 5.1 SLIM (Ning and Karypis, ICDM 2011)

**Idea.** Learn the item-item weight matrix W directly rather than computing it from a similarity heuristic:

```
 min_W  1/2 ||A - A W||_F^2 + beta/2 ||W||_F^2 + lambda ||W||_1
 s.t.   W >= 0,  diag(W) = 0
```

A is the user-item matrix. The L1 + L2 (elastic net) penalty makes W sparse, so serving is as cheap as item-kNN. The zero-diagonal constraint stops the trivial solution of each item predicting itself. The non-negativity keeps W interpretable. Each column of W is its own regression, so training splits by item.

**System notes.** The KarypisLab SLIM library is a multi-threaded C implementation with command-line tools (slim_learn, slim_predict, slim_mselect for model selection) and a Python interface [10]. It offers coordinate descent and an ADMM solver (which needs Intel MKL). The README says ADMM usually wins when there are few items relative to users and many non-zeros [10]. In the Dacrema et al. suite, **SLIM ElasticNet** and **SLIM BPR** are among the strongest baselines [16].

### 5.2 EASE (Steck, WWW 2019)

**Idea.** Drop the L1 penalty and the non-negativity constraint from SLIM and keep only L2 and the zero diagonal. The problem then has a **closed-form solution**:

```
 min_B ||X - X B||_F^2 + lambda ||B||_F^2   s.t. diag(B) = 0

 P   = (X^T X + lambda I)^(-1)                 # item-item, m x m
 B_ij = - P_ij / P_jj   for i != j ,  B_jj = 0
 scores(u) = x_u B                             # dense item-item "autoencoder"
```

An open-source implementation [11b] follows exactly this recipe: build the sparse user-item matrix, form the regularised Gram matrix X^T X + lambda I, invert it, take B with a zeroed diagonal, score with X B, filter seen items and take the top-k.

**Why it matters.** EASE is a linear autoencoder with no hidden layer, which is why the title calls it "embarrassingly shallow". It matched or beat deep autoencoders such as Mult-VAE on the standard ML-20M, Netflix and MSD benchmarks [11]. Training is one m x m inversion, with cost roughly O(m^3) in the number of items. That cost does not depend on the number of users, and m in the low tens of thousands fits comfortably in memory. Its two limits are the m^2 memory of a dense B and the cubic inversion. Beyond about 10^5 items you need item pruning, sparse approximations or blockwise methods. Steck also observed that the learned weights include many **negative** item-item weights, which carry useful information. SLIM's non-negativity constraint throws that information away [11].

**Follow-ups.** The main deep model EASE was measured against was Mult-VAE [27]. Steck's later work extended the linear family, with higher-order relations and denoising views of the linear autoencoder (not re-verified this session). It also pointed out pitfalls of using cosine similarity on learned embeddings [28]. In 2026 EASE is still the default "is your fancy model actually better?" check in the major benchmarking frameworks.

```
  Linear item-item models: what each one learns

               similarity source           constraints            training cost
  item-kNN     cosine / Jaccard heuristic  top-k sparsify         O(nnz * avg deg)
  SAR          co-occurrence + rescale     time-decayed A         O(nnz + m^2)
  RP3beta      3-step random walk          popularity penalty     sparse mat-mults
  SLIM         learned (elastic net)       W >= 0, diag 0         m regressions
  EASE         learned (ridge)             diag 0                 one m x m inverse
```

## 6. Word2vec-style item embeddings: item2vec, prod2vec, Airbnb

### 6.1 item2vec (Barkan and Koenigstein, 2016)

Apply skip-gram with negative sampling (SGNS) to item sets, such as the items in one user's basket or listening list. Each set is treated as a "sentence" in which every pair of items is context, because order is ignored. The learned vectors give item-item similarity that is competitive with SVD-based item similarity, and it was evaluated on Microsoft Store and music data [12]. The appeal for systems is that SGNS is cheap, scales linearly with the number of co-occurrences, and gives dense vectors that drop straight into an ANN index.

### 6.2 prod2vec and bagged-prod2vec (Yahoo, KDD 2015)

Grbovic et al. ("E-commerce in Your Inbox") learned product embeddings from purchase sequences mined from e-mail receipts. Receipts were ordered in time per user, and products in the same receipt formed context. They added user embeddings (user2vec), and a "bagged" variant that treats each receipt as an unordered bag. This was one of the first web-scale industrial uses of word2vec for recommendation, serving product ads in Yahoo Mail. It reported clear CTR gains over popularity-based targeting (exact figures not re-verified here) [13].

### 6.3 Airbnb listing embeddings (KDD 2018 best paper)

Airbnb trained listing embeddings on **click sessions**, roughly 800 million of them. A session is split when consecutive clicks are more than 30 minutes apart. Approximate details from the KDD paper and the Airbnb engineering post [14]:

- 32-dimensional embeddings for about 4.5M active listings.
- **Booked listing as global context.** In booked sessions, the booked listing is kept in every context window. Every click in the session is then pulled towards the conversion.
- **Market-aware negatives.** Extra negatives are sampled from the same market, so embeddings separate listings within a city instead of just learning geography.
- **Cold start.** A new listing's embedding is the average of its three nearest listings by type, price and location (approx.).
- **User-type and listing-type embeddings** trained on bookings to handle sparsity for long-horizon personalisation.
- Production use: similar-listing carousels, and real-time personalisation features in search ranking. They reported a meaningful CTR lift on similar-listing recommendations (approx. 21%, not re-verified this session).

```
 Airbnb-style session embedding pipeline
 --------------------------------------
  click logs --> sessionize (30-min gap) --> filter (dwell >= threshold)
       |
       v
  SGNS training (window, negatives = random + same-market, global ctx = booking)
       |
       v
  listing vectors (d=32) --> ANN index --> "similar listings" candidates
       |
       +--> real-time features: sim(candidate, recently clicked / skipped)
                                  --> fed to GBDT/NN search ranker
```

**Lesson.** Each of these word2vec variants is the same algorithm. What made them work was domain-specific sessionisation, context design and negative sampling. Treat embedding quality as a data-engineering problem first.

## 7. Content-based and hybrid recommenders

### 7.1 Content-based filtering

Content-based recommenders represent items by features: text, tags, categories, audio or image embeddings. They represent users by aggregating the features of items they liked, then score candidates by similarity or a learned classifier. Microsoft Recommenders includes a TF-IDF content baseline for text datasets [15]. The Dacrema et al. suite includes CBF-kNN and a CF+CBF hybrid kNN [16]. Strengths: cold start for new items, transparency, and no need for other users' data. Weaknesses: over-specialisation (the "more of the same" filter bubble), and quality capped by feature quality. In 2026 that cap has been raised a lot by pretrained text and image encoders.

### 7.2 Hybrids: LightFM as the canonical design

LightFM (Kula, RecSys 2015 workshop, built at Lyst) represents each user and item as **the sum of the latent vectors of its features** [9]. Feature sets usually include a per-entity indicator feature plus metadata such as tags, brand and category. The model:

```
 user u  -> features f_u (one-hot id + metadata)  -> e_u = sum_{j in f_u} E_U[j]
 item i  -> features f_i (one-hot id + metadata)  -> e_i = sum_{j in f_i} E_I[j]
 score(u,i) = e_u . e_i + b_u + b_i     trained with BPR or WARP (or logistic)
```

With only indicator features, LightFM is plain MF. With only metadata features, it is a content model. With both, a new item with no interactions still gets a meaningful embedding from its tags [9]. This structure is conceptually a two-tower model with linear, bag-of-features towers. It is the direct ancestor of the neural two-tower retrieval in Chapter 2. Factorization machines (xLearn FM/FFM in Microsoft Recommenders [15]) generalise the same idea to arbitrary feature crosses.

### 7.3 Hybridisation patterns in production

- **Feature augmentation.** CF embeddings (iALS, item2vec) become features in a GBDT or neural ranker. Airbnb does this [14].
- **Switching.** Use content and popularity for cold users and items, and CF once there are enough interactions.
- **Candidate-source union.** Item-kNN, MF, content and trending sources each supply candidates. A learned ranker merges them. This is the "classic models as retrieval sources" pattern that most large systems still use (see Chapter 1).

## 8. The reproducibility crisis: why classic baselines still matter

### 8.1 Dacrema, Cremonesi and Jannach (RecSys 2019, best paper)

"Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches" examined 18 top-N recommendation algorithms. All were neural and published at major venues (KDD, SIGIR, WWW, RecSys) in the preceding years. The findings [16][17]:

- Only **7 of 18** could be reproduced with reasonable effort, using the authors' code and data.
- Of those 7, **6 were often outperformed by comparatively simple heuristics**: nearest-neighbour or graph methods tuned properly.
- The remaining one beat the simple baselines but **did not consistently beat a well-tuned non-neural linear ranking method**.

### 8.2 The TOIS extension (2021)

"A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research" widened the study [17]. Roughly **12 of 26** examined neural methods could be reproduced, and **11 of those 12** were outperformed by simple, well-tuned baselines (approximate counts as stated in the paper's abstract). The accompanying code base [16] ported the neural methods:

- CollaborativeDL, CollaborativeVAE, MCRec, DELF, DMF, ConvNCF, CoupledCF, NeuRec, CFM, NeuMF, Mult-VAE, SpectralCF and CMN.

It evaluated them against 17 baselines, all with systematic hyperparameter search:

- Random and TopPop.
- UserKNN and ItemKNN.
- CBF and CF+CBF kNN.
- P3alpha and RP3beta.
- PureSVD, NMF, iALS, BPRMF and FunkSVD.
- EASE-R, SLIM-BPR and SLIM-ElasticNet.

A 2021 maintenance note fixed an HR/NDCG implementation issue that affected a few NDCG numbers, and stated that "the analysis and conclusions of the study are not affected" [16].

A companion CIKM 2020 paper went after a specific architectural claim. It showed that the reported benefit of applying CNNs to the outer-product "embedding map" of user and item vectors (ConvNCF and relatives) did not come from the convolutions [16].

### 8.3 Converging evidence from other groups

- **Rendle, Krichene, Zhang and Anderson (RecSys 2020), "Neural Collaborative Filtering vs. Matrix Factorization Revisited."** On NCF's own datasets and protocol (ML-1M, Pinterest; embedding sizes swept from 16 to 192, 8 repeats), a properly set-up dot-product MF beat NeuMF. A second experiment showed that an MLP needs very large capacity and data to learn even a plain dot product [19]. The systems corollary: dot-product scoring is also what makes ANN retrieval possible, which an MLP scoring function rules out.
- **Rendle, Zhang and Koren (2019), "On the Difficulty of Evaluating Baselines."** On MovieLens-10M and the Netflix Prize data, carefully tuned classic methods (Bayesian timeSVD++ flipped and similar) beat many recently published results. The conclusion was that the reported baseline numbers had been far too weak [26].
- **Rendle et al., iALS revisited (2021-2022).** Tuned iALS matched EASE and Mult-VAE on ML-20M and MSD [18].
- **Anelli et al. (UMAP 2022), "Top-N Recommendation Algorithms: A Quest for the State of the Art."** A large Elliot-based comparison on MovieLens-1M (about 1M interactions), Amazon Digital Music (about 1.58M interactions, 840K users) and Epinions (about 300K interactions) [20]. It again found classic linear and neighbourhood models highly competitive. It also examined beyond-accuracy dimensions such as novelty, diversity and popularity bias, where rankings differ.
- **Krichene and Rendle (KDD 2020), "On Sampled Metrics for Item Recommendation."** Evaluating with 1 positive vs. 100 sampled negatives, the protocol inherited from NCF, can change the ranking of algorithms compared with full ranking [29]. That puts many published comparisons in doubt.
- **Session-based recommendation.** Ludewig, Jannach and colleagues found that session-kNN variants are often competitive with GRU4Rec and later neural session models [30].

### 8.4 Root causes, and the fixes the community adopted

| Failure mode | Typical symptom | Fix adopted by benchmark frameworks |
|---|---|---|
| Untuned baselines | baseline uses defaults; proposed model tuned for weeks | equal hyperparameter budget (Bayesian or random search) for all models [16][23] |
| Weak baseline choice | only BPR-MF / NeuMF as baselines | always include ItemKNN, RP3beta, SLIM/EASE, iALS, TopPop [16][20] |
| Sampled metrics | 1 positive + 100 negatives | full-ranking evaluation [29] |
| Data leakage, inconsistent splits | test info used for early stopping or tuning | explicit train/validation/test splits; temporal splits |
| Missing code or data | cannot reproduce | released configs and seeds (Elliot, RecBole, Dacrema repo) [16][23][24] |
| Single-metric reporting | only HR@10 | accuracy + beyond-accuracy + significance tests (Elliot has Wilcoxon and paired t-test) [23] |

Frameworks now encode these practices:

- **Elliot (SIGIR 2021)** runs the whole pipeline, from loading and splitting through hyperparameter optimisation to statistical testing. It covers BPRMF, FunkSVD, SVD++, kNN, SLIM, VAE, NeuMF and more [23].
- **RecBole** offers 94 algorithms across general, sequential, context-aware and knowledge-based tasks, with BPR as its quick-start default [24].
- **Microsoft Recommenders** wraps ALS (PySpark), BPR (Cornac), SAR, LightFM, Surprise SVD, xLearn FM/FFM and TF-IDF next to neural models [15].

## 9. Comparison: classic algorithms as a system designer sees them

| Algorithm | Feedback | Training cost (approx.) | Serving pattern | Cold start | Typical role in 2026 stacks |
|---|---|---|---|---|---|
| User-kNN | explicit or implicit | O(n^2) user sims or on-the-fly | neighbour lookup per request | poor | rarely used at scale |
| Item-kNN / item-to-item [1][25] | implicit | O(nnz * deg), offline batch | precomputed top-N per item, merge | poor for new items | "similar items" widgets, candidate source |
| SAR [21] | implicit + time | co-occurrence O(m^2) memory | A x S, or top-N per item | poor | quick strong baseline |
| P3alpha / RP3beta [16] | implicit | sparse matrix products | precomputed item-item | poor | strong baseline, long-tail boost |
| Funk-SVD / SVD++ [3][4] | explicit ratings | SGD, O(nnz * f) per epoch | dot product | poor | rating prediction, legacy |
| iALS [5][18] | implicit | O(f^2 nnz + f^3 (n+m)) per sweep | dot product + ANN | poor (fold-in possible) | retrieval source, embedding features |
| BPR-MF [8] | implicit | SGD over sampled triples | dot product + ANN | poor | baseline; sampling research |
| SLIM [10] | implicit | m elastic-net regressions | sparse x_u W | poor | strong accuracy, sparse serving |
| EASE [11] | implicit | one m x m inverse, O(m^3) | dense x_u B | poor | strongest simple baseline for m < ~1e5 |
| item2vec / prod2vec [12][13] | sequences / baskets | SGNS, linear in co-occurrences | ANN on item vectors | poor (fixable via averaging) | similar items, session features |
| Content-based (TF-IDF etc.) [15] | item features | feature extraction | similarity search | good | cold start, explanations |
| LightFM hybrid [9] | implicit/explicit + metadata | SGD (BPR/WARP) | dot product + ANN | good | small/mid-scale hybrid, cold-start retrieval |

## 10. Reference architecture: classic models in a modern pipeline

```
                              +----------------------------+
   interaction logs --------> | batch training (daily)     |
   (views, plays, buys)       |  - item-kNN / RP3beta      |----> item->topN table (KV)
                              |  - iALS / BPR (Spark/GPU)  |----> item & user factors
                              |  - EASE (if m small)       |----> B matrix / top-N
                              |  - item2vec on sessions    |----> item vectors
                              +-------------+--------------+
                                            |
                                    ANN index build (Faiss/ScaNN/Annoy)
                                            |
 request --> context (recent items, user id) v
         --> candidate generation: union of
               [kNN lookups of recent items] + [ANN on user factor]
               + [content / trending for cold users]          (~100s-1000s)
         --> feature join (incl. CF scores, embedding sims as features)
         --> learned ranker (GBDT or DNN)                           (~10s)
         --> business rules, diversity, dedup  --> response
```

Here the classic models do three jobs:

1. **Candidate sources.** They are cheap and high-recall.
2. **Features for the ranker.** Examples: an iALS dot product, an item2vec similarity to the last clicked item, a co-occurrence count.
3. **Baselines and guardrails.** A new model has to beat them offline on full-ranking metrics, and online in A/B tests, before it ships.

## Key takeaways for system designers

1. **Tune baselines before building new models.** In the reproducibility studies, ItemKNN, RP3beta, SLIM, EASE and iALS beat most published neural recommenders once they were tuned [16][17][18][19]. Give every baseline the same hyperparameter-search budget as the new model.
2. **Pick item-item over user-user for online serving.** Item relationships are stable and can be precomputed. Online cost then depends on the user's history length, not on catalogue size [1][2].
3. **For implicit data, weight by confidence and count the unobserved entries.** The Hu-Koren-Volinsky formulation, made tractable by the Y^T Y precompute trick, is still the right mental model. Modern tuning adds a separate weight for unobserved entries, frequency-scaled regularisation and large dimensions [5][18].
4. **Prefer dot products.** They are as accurate as MLP scorers in fair comparisons, and they are what makes ANN retrieval and caching possible [19].
5. **EASE is the cheapest strong baseline when the catalogue is small enough.** One matrix inverse, no SGD and no sampling. Move to SLIM (sparse), item-kNN, iALS or blockwise approximations as m grows past about 1e5 [10][11].
6. **Embedding quality depends on the training data you construct.** item2vec-style models succeed or fail on sessionisation, context windows and negative sampling, as with Airbnb's booked-listing global context and same-market negatives [14].
7. **Hybrids solve cold start.** Building entity embeddings as sums of feature embeddings (LightFM) is the simplest way to embed new items. Today's two-tower models are the neural version of the same idea [9].
8. **Evaluate with full ranking, fixed splits and significance tests.** Sampled metrics can reorder algorithms [29]. Use a framework such as Elliot, RecBole or the Dacrema code base, and release configs [16][23][24].
9. **Retire classic models with care.** Many production stacks still use kNN and MF as candidate sources and ranker features even after adding deep models, because they are cheap, interpretable and robust.

## References

1. G. Linden, B. Smith, J. York. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering." IEEE Internet Computing, 2003. https://doi.org/10.1109/MIC.2003.1167344
2. B. Smith, G. Linden. "Two Decades of Recommender Systems at Amazon.com." IEEE Internet Computing, 2017. https://doi.org/10.1109/MIC.2017.72
3. Y. Koren, R. Bell, C. Volinsky. "Matrix Factorization Techniques for Recommender Systems." IEEE Computer, 2009. https://doi.org/10.1109/MC.2009.263
4. Y. Koren. "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model." KDD, 2008. https://doi.org/10.1145/1401890.1401944
5. Y. Hu, Y. Koren, C. Volinsky. "Collaborative Filtering for Implicit Feedback Datasets." IEEE ICDM, 2008. https://doi.org/10.1109/ICDM.2008.22
6. Apache Spark. "Collaborative Filtering (ALS) - MLlib Guide." Apache Software Foundation. https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
7. M. Kabiljo, A. Ilic (Facebook Engineering). "Recommending items to more than a billion people." 2015. https://engineering.fb.com/2015/06/02/core-infra/recommending-items-to-more-than-a-billion-people/
8. S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme. "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI, 2009. https://arxiv.org/abs/1205.2618
9. M. Kula (Lyst). "Metadata Embeddings for User and Item Cold-start Recommendations" (LightFM). CBRecSys@RecSys, 2015. Code: https://github.com/lyst/lightfm ; paper: https://arxiv.org/abs/1507.08439
10. X. Ning, G. Karypis. "SLIM: Sparse Linear Methods for Top-N Recommender Systems." ICDM, 2011; SLIM library (Ning, Nikolakopoulos, Shui, Sharma, Karypis, 2019). https://github.com/KarypisLab/SLIM
11. H. Steck. "Embarrassingly Shallow Autoencoders for Sparse Data." WWW, 2019. https://arxiv.org/abs/1905.03375
    (11b) Open-source EASE implementation (Darel13712/ease_rec). https://github.com/Darel13712/ease_rec
12. O. Barkan, N. Koenigstein. "Item2Vec: Neural Item Embedding for Collaborative Filtering." IEEE MLSP, 2016. https://arxiv.org/abs/1603.04259
13. M. Grbovic et al. (Yahoo Labs). "E-commerce in Your Inbox: Product Recommendations at Scale." KDD, 2015. https://arxiv.org/abs/1606.07154
14. M. Grbovic, H. Cheng (Airbnb). "Real-time Personalization using Embeddings for Search Ranking at Airbnb." KDD, 2018. https://doi.org/10.1145/3219819.3219885
15. Microsoft / Recommenders team (LF AI & Data). "Recommenders: Best Practices on Recommendation Systems." GitHub. https://github.com/recommenders-team/recommenders
16. M. Ferrari Dacrema, P. Cremonesi, D. Jannach. "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches." RecSys, 2019; code and TOIS/CIKM/IJCAI material. Paper: https://arxiv.org/abs/1907.06902 ; code: https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
17. M. Ferrari Dacrema, S. Boglio, P. Cremonesi, D. Jannach. "A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research." ACM TOIS, 2021. https://arxiv.org/abs/1911.07698
18. S. Rendle, W. Krichene, L. Zhang, Y. Koren. "Revisiting the Performance of iALS on Item Recommendation Benchmarks" (RecSys 2022) and "iALS++: Speeding up Matrix Factorization with Subspace Optimization" (2021). Google Research code: https://github.com/google-research/google-research/tree/master/ials ; paper: https://arxiv.org/abs/2110.14037
19. S. Rendle, W. Krichene, L. Zhang, J. Anderson. "Neural Collaborative Filtering vs. Matrix Factorization Revisited." RecSys, 2020. https://arxiv.org/abs/2005.09683 ; code: https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity
20. V. W. Anelli, A. Bellogin, T. Di Noia, D. Jannach, C. Pomo. "Top-N Recommendation Algorithms: A Quest for the State-of-the-Art." UMAP, 2022. https://github.com/sisinflab/Top-N-Recommendation-Algorithms-A-Quest-for-the-State-of-the-Art
21. Recommenders team. "SAR Single Node on MovieLens (Python, CPU) - deep dive notebook." GitHub. https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb
22. B. Frederickson. "implicit: Fast Python Collaborative Filtering for Implicit Feedback Datasets." GitHub. https://github.com/benfred/implicit
23. V. W. Anelli et al. "Elliot: A Comprehensive and Rigorous Framework for Reproducible Recommender Systems Evaluation." SIGIR, 2021. https://github.com/sisinflab/elliot ; https://arxiv.org/abs/2103.02590
24. W. X. Zhao et al. (RUCAIBox). "RecBole: A Unified, Comprehensive and Efficient Recommendation Library." GitHub. https://github.com/RUCAIBox/RecBole
25. B. Sarwar, G. Karypis, J. Konstan, J. Riedl. "Item-based Collaborative Filtering Recommendation Algorithms." WWW, 2001. https://doi.org/10.1145/371920.372071
26. S. Rendle, L. Zhang, Y. Koren. "On the Difficulty of Evaluating Baselines: A Study on Recommender Systems." arXiv, 2019. https://arxiv.org/abs/1905.01395
27. D. Liang, R. G. Krishnan, M. D. Hoffman, T. Jebara. "Variational Autoencoders for Collaborative Filtering" (Mult-VAE). WWW, 2018. https://arxiv.org/abs/1802.05814 ; code: https://github.com/dawenl/vae_cf
28. H. Steck, C. Ekanadham, N. Kallus. "Is Cosine-Similarity of Embeddings Really About Similarity?" WWW Companion, 2024. https://arxiv.org/abs/2403.05440
29. W. Krichene, S. Rendle. "On Sampled Metrics for Item Recommendation." KDD, 2020. https://doi.org/10.1145/3394486.3403226
30. M. Ludewig, N. Mauro, S. Latifi, D. Jannach. "Performance Comparison of Neural and Non-Neural Approaches to Session-based Recommendation." RecSys, 2019. https://arxiv.org/abs/1910.12781
