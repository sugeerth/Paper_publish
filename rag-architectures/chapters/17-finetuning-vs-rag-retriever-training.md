# 17. Fine-tuning vs RAG and Training the Retrieval Stack

A RAG system has four places where learning can happen. Each one fixes a different failure:

1. **Generator knowledge.** Facts go into the LLM's weights through continued pre-training or supervised fine-tuning (SFT).
2. **Generator behaviour.** The LLM learns how to read retrieved context: find the relevant passage, ignore distractors, quote and reason. RAFT and the LM side of RA-DIT train this.
3. **The retriever.** The bi-encoder or late-interaction model that produces candidates. It is trained with synthetic queries (InPars, Promptagator, GPL, E5-Mistral), mined hard negatives (ANCE, RocketQA, NV-Retriever), or the LLM's own preferences (REPLUG-LSR, Atlas, RA-DIT).
4. **The reranker.** A cross-encoder or LLM reranker. It is often a distillation target or a distillation teacher (RankGPT, RankZephyr, Rank1).

The chapter starts with the evidence on the first-order question, "should knowledge go into the weights or into the index?". It then covers generator-side training (RAFT, RA-DIT, ChatQA, RankRAG), LM-supervised and RL-trained retrieval (REPLUG-LSR, Atlas, reader-to-retriever distillation, Search-R1, DeepRetrieval, ZeroSearch), synthetic-query pipelines (InPars, Promptagator, GPL, UDAPDR and their LLM-scale successors such as E5-Mistral, Gecko and Qwen3-Embedding), hard-negative mining, and distillation into rerankers and retrievers (MarginMSE, TAS-B, ColBERTv2, RankGPT, RankZephyr, Rank1, ReasonRank). Section 17.1 also covers Cartridges, a trained per-corpus KV cache that sits between fine-tuning and putting the corpus in context. It closes with a domain-adaptation playbook. Chapter 3 covers embedding-model loss functions and the Sentence Transformers tooling in detail. This chapter is about where the training signal comes from and which component it should update.

*Source note.* arXiv was not directly reachable while this chapter was researched. Papers were read in full text from public mirrors of the arXiv PDFs. They are cited by their canonical arXiv URLs. Where a fact comes only from a secondary summary, the text says so. In a later fact-check pass (October 2026), arXiv was again unreachable. Numbers were re-checked against the official code repositories (Gorilla RAFT, GPL, InPars, rank1, RankLLM, DeepRetrieval, Qwen3-Embedding, Sentence Transformers), the Hugging Face and Weaviate blog sources, and the Microsoft Research publication pages. Claims those sources could not confirm were softened.

---

## 17.1 The first-order decision: knowledge in weights or knowledge in the index?

### 17.1.1 Ovadia et al. (Microsoft, 2023): unsupervised fine-tuning vs RAG

**Problem.** Compare two ways to inject knowledge into an LLM: unsupervised fine-tuning (continued training on the raw text) and RAG [5].

**Setup.** Mistral-7B, Llama2-7B and Orca2-7B were tested on five MMLU subjects (anatomy, astronomy, college biology, college chemistry, prehistory). They were also tested on a new "current events" task: 910 multiple-choice questions built with GPT-4 from US events of August to November 2023, after the models' training cutoffs. RAG used `bge-large-en` with a FAISS index. Fine-tuning ran on 4 A100s for at most 5 epochs, with learning rates between 1e-6 and 5e-5 [5].

**Results.** RAG won across the board. On current events with Mistral-7B, the base model scored 0.481. Base + RAG scored 0.875. Fine-tuning on the raw chunks scored 0.504, and fine-tuning on ten paraphrases per chunk scored 0.588. Combining fine-tuning with RAG (0.810 and 0.830) was *worse* than RAG alone. For Llama2-7B, plain fine-tuning on new facts lowered accuracy from 0.353 to 0.219 [5].

**Lesson: repetition.** Accuracy rose monotonically with the number of paraphrases. The authors read this as evidence that a model must see a new fact many times, in varied wording, to absorb it [5]. They also found the RAG hyperparameter K (number of retrieved chunks) unstable: no single K was best across models, tasks or shot settings [5].

### 17.1.2 Soudani et al. (2024): less popular knowledge

**Problem.** Long-tail entities are where LLMs know least. Is RAG or fine-tuning on synthetic QA the better fix [6]?

**Setup.** PopQA (about 14,000 templated questions), bucketed by Wikipedia pageviews. FlanT5 small, base and large were fine-tuned on synthetic QA, with both full fine-tuning and QLoRA. Two QA generators were compared: an end-to-end T5 generator (E2E) and prompting an LLM [6].

**Findings** [6]:

- Fine-tuning helped across popularity buckets. The largest gains came in the most and least popular buckets.
- RAG beat fine-tuning overall. FT plus RAG helped most for small models, and that benefit "diminishes in base models and is non-existent in larger models".
- **Quality beat quantity.** The E2E generator produced more than 12 times as many QA pairs as prompting. Models trained on the smaller, prompt-generated set still did better.
- PEFT gave smaller gains than full FT without retrieval, but larger gains when combined with RAG. The authors read this as PEFT preserving the model's ability to reason over the prompt.

### 17.1.3 Gekhman et al. (Google / Technion, 2024): new facts via SFT encourage hallucination

**Problem.** Does SFT on facts the model does not already know teach it to hallucinate [7]?

**Method.** The SliCK taxonomy sorts each (question, answer) example by how often the base model (PaLM 2-M) already answers it correctly: HighlyKnown, MaybeKnown, WeaklyKnown or Unknown. The study then varied the share of Unknown examples in a fixed-size closed-book QA fine-tuning set [7].

**Findings** [7]:

- Unknown examples are fitted much more slowly than Known ones.
- Once they are fitted, they *linearly* increase the model's tendency to hallucinate on facts it previously knew.
- Early stopping, or filtering Unknown examples out, mitigates the effect.
- Training only on HighlyKnown examples was not optimal. MaybeKnown examples helped the model handle uncertain facts at test time.

The authors conclude that LLMs mostly acquire facts in pre-training, and that fine-tuning mainly teaches them to use those facts.

### 17.1.4 Microsoft agriculture case study (2024): fine-tuning and RAG are cumulative

**Setup.** A Microsoft Research pipeline extracted information from agricultural PDFs, generated Q&A pairs, fine-tuned Llama2-13B, GPT-3.5 and GPT-4, and used GPT-4 as the judge [8].

**Results.** Fine-tuning raised accuracy by over 6 percentage points, and RAG added about 5 more. The effects were cumulative. In one experiment, the fine-tuned model drew on information across geographies, and answer similarity rose from 47% to 72% [8].

**Follow-up on data design.** A later Microsoft paper on SFT for new knowledge (recent sporting events, GPT-4) compared two ways of building the training set. "Token-based scaling" generates more text overall. "Fact-based scaling" guarantees even coverage of every fact. Fact-based scaling covered the new knowledge more systematically [9].

### 17.1.5 EntiGraph / synthetic continued pre-training (Stanford, 2024)

**Problem.** A small private corpus contains each fact only once or twice. That is far too little repetition for continued pre-training to work [10].

**Method.** EntiGraph prompts GPT-4-turbo to extract the salient entities in each document and write text about the relations between them. This turned 1.3M real tokens (265 QuALITY books and articles) into 455M synthetic tokens. Llama 3 8B was then continually pre-trained on the result for 2 epochs, with replay of RedPajama data [10].

**Results** [10]:

- QA accuracy scaled log-linearly with synthetic token count up to 455M tokens.
- At the endpoint, closed-book synthetic continued pre-training recovered about 80% of the accuracy gain of giving the model the source documents at inference time.
- The two were complementary. EntiGraph CPT + RAG scored 62.60, against 60.35 for Llama 3 8B Base + RAG, with retrieval recall@8 at 99.63 for both.

### 17.1.6 Cartridges and self-study (Stanford / HazyResearch, 2025): a third option between weights and the prompt

**Problem.** Long-context "put the whole corpus in the prompt" is the simplest alternative to RAG. Its serving cost grows with KV-cache size, though, and the cache is rebuilt for every request [52].

**Method.** Cartridges train a small KV cache *offline* for a specific corpus, and the base model's weights stay frozen. Training data comes from **self-study**: two instances of the model hold synthetic conversations about the corpus, one asking questions or making requests and the other answering with the corpus in context. The cartridge is then trained with a *context-distillation* objective, so that the model with the cartridge matches the model with the full corpus in context. The cartridge size (`max_tokens`) trades memory against quality [52].

**Results.** The authors claim quality comparable to in-context learning with up to about 26x higher serving throughput. The repository states the throughput figure without a detailed baseline, so treat it as the authors' headline number [52].

**Where it fits.** A cartridge is closer to EntiGraph-style synthetic training [10] than to RAG: it needs a training run per corpus, and it must be retrained when the corpus changes. It does not touch the base weights, though, so it avoids the hallucination risk of SFT on unknown facts [7]. It suits small, static, heavily queried corpora such as a manual or a codebase. It is not a replacement for an index over large or fast-changing collections.

### 17.1.7 Synthesis

| Study | Knowledge type | Weights-only result | RAG result | Combined |
|---|---|---|---|---|
| Ovadia et al. [5] | Post-cutoff events (MCQ) | small gain; paraphrases help; Llama2 got worse | large gain (0.481 to 0.875, Mistral) | worse than RAG alone |
| Soudani et al. [6] | Long-tail entities (PopQA) | gains, largest at tails | better than FT | helps small models only |
| Gekhman et al. [7] | Unknown facts via SFT | slow to learn; raises hallucination | n/a | n/a |
| MS agriculture [8] | Regional domain docs | +6 pp | +5 pp | cumulative |
| EntiGraph [10] | Niche books, 1.3M tokens | about 80% of RAG gain with 455M synthetic tokens | strong | CPT + RAG > base + RAG |

These studies share a pattern. Fine-tuning on raw facts is a weak and risky way to add knowledge unless the data is heavily amplified: paraphrases [5], fact-level coverage [9], or entity graphs [10]. RAG is the default for adding facts. Fine-tuning pays off when it teaches *behaviour*: answer style, domain vocabulary, reading retrieved context, ignoring distractors. In those cases it adds to RAG instead of replacing it. The rest of this chapter is about that kind of training.

```
                Does the model need NEW FACTS, or a NEW SKILL?
                               |
           +-------------------+---------------------+
           | facts (change often / need             | skill (format, reading,
           | citations / access control)            | domain jargon, ranking)
           v                                         v
   RAG + better retrieval                    fine-tune a component
   (train retriever/reranker,                  |
    sec. 17.3-17.6)                            +-- generator reads docs badly
           |                                   |      -> RAFT / RA-DIT LM-ft
           | facts static, corpus small,       +-- retriever misses domain intent
           | latency-critical, offline?        |      -> synthetic queries + negatives
           v                                   +-- ranking quality / latency
   consider synthetic CPT (EntiGraph)          |      -> distil LLM -> cross-encoder
   AND keep RAG on top [10]                    +-- retriever != what the LM needs
                                                      -> LM-supervised retrieval (LSR)
```

---

## 17.2 Generator-side training: teaching the LLM to read retrieved context

### 17.2.1 RAFT: Retrieval-Augmented Fine-Tuning (UC Berkeley, 2024)

**Problem.** Standard domain fine-tuning (DSF) trains Q to A with no documents in the prompt. This aligns answer style, but the model never learns to read imperfect retrievals. Off-the-shelf RAG, for its part, wastes the fact that the test domain is known in advance. The authors' analogy is an open-book exam taken without studying [1].

**Data recipe.** Each training example contains a question Q, a set of documents, and a chain-of-thought answer A* derived from the "oracle" document(s) D*. The documents are mixed as follows [1]:

```
 P fraction of examples:      Q + D* + D_1 ... D_{k-1}  -->  A* (CoT with quotes)
 (1-P) fraction of examples:  Q + D_1 ... D_k (no oracle) -->  A*

 A* = reasoning that quotes the oracle verbatim between
      ##begin_quote## ... ##end_quote## markers, then the final answer
 default: 1 oracle + 4 distractors; CoT generated by GPT-4-1106
```

The open-source generator (`raft.py` in the Gorilla repository) automates this. It chunks a PDF, TXT, JSON or API-doc corpus (512-token chunks in the examples), has GPT-4 write N questions per chunk with CoT answers, adds `--distractors` random chunks, and sets the oracle-inclusion rate with `--p`. It can emit Hugging Face, chat, completion or eval formats (JSONL or Parquet) and can call Azure OpenAI deployments. A separate script, `raft_local.py`, runs the pipeline on local Hugging Face models instead [2].

**Results (Llama-2-7B)** [1]:

| Model | PubMed | HotpotQA | HuggingFace | Torch Hub | TF Hub |
|---|---|---|---|---|---|
| GPT-3.5 + RAG | 71.60 | 41.5 | 29.08 | 60.21 | 65.59 |
| Llama2-7B + RAG | 58.8 | 0.03 | 26.43 | 8.60 | 43.06 |
| DSF | 59.7 | 6.38 | 61.06 | 84.94 | 86.56 |
| DSF + RAG | 71.6 | 4.41 | 42.59 | 82.80 | 60.29 |
| RAFT | 73.30 | 35.28 | 74.00 | 84.95 | 86.86 |
| RAFT without CoT | 68.30 | 25.62 | 59.07 | 86.56 | 83.21 |

**Key findings** [1]:

- **Adding RAG to a DSF model often hurts** (HuggingFace drops from 61.06 to 42.59). A model fine-tuned without context has not learned to use context.
- **CoT matters.** Removing the reasoning chain cost 9.66 points on HotpotQA and 14.93 on HuggingFace. Answer-only targets made the loss fall quickly and training diverge.
- **Do not always include the oracle.** The best P varied by dataset (40%, 60% and 100% are all reported). Leaving the oracle out of some examples helped in-domain RAG.
- **Train with distractors.** Oracle-only training was consistently worse. The best training mix was D* + 3 distractors for NQ and D* + 1 for HotpotQA. Models trained with distractors were more robust when the number of test-time documents changed.

**Failure modes and limits.** RAFT targets the case where the test documents are known at training time (in-domain RAG), so it is not a general RAG improver [1]. Distractors are sampled randomly, not mined, so they are easier than real retrieval errors. On PubMedQA's yes/no format, gains over DSF + RAG were small [1].

### 17.2.2 RA-DIT: Retrieval-Augmented Dual Instruction Tuning (Meta FAIR, ICLR 2024)

**Problem.** Retrieval-augmented pre-training (REALM, RETRO, Atlas) is expensive. Bolting an off-the-shelf LLM onto an off-the-shelf retriever (REPLUG, In-Context RALM) leaves accuracy on the table. RA-DIT is a third option: two light fine-tuning passes that "retrofit" any LLM [3].

**Architecture.** LLaMA (7B, 13B, 65B) is paired with a DRAGON+ dual encoder over 399M chunks: 37M Wikipedia chunks of under 200 words, plus 362M Common Crawl chunks. Each retrieved chunk is prepended to the prompt separately, the k forward passes run in parallel, and their outputs are mixed, weighted by renormalised retrieval scores (REPLUG-style ensembling) [3].

```
            +--------------------- LM-ft ----------------------+
 instruction|  for each (x, y) and each of top-3 chunks c:      |
 dataset    |    minimise -log p_LM(y | "Background: c" + x)    |
 (20 sets)  |  one training instance per chunk                  |
            +---------------------------------------------------+
            +--------------------- R-ft -----------------------+
            |  LSR target: p_LSR(c|x,y) ~ exp(p_LM(y|c+x)/tau)  |
            |  minimise KL(p_R(c|x) || p_LSR(c|x,y)) over top-10 |
            |  update QUERY encoder only (index stays valid)    |
            +---------------------------------------------------+
 inference: top-10 chunks -> 10 parallel prompts -> weighted ensemble
```

**LM fine-tuning (LM-ft)** [3]:

- 20 datasets in five groups: dialogue, open-domain QA, reading comprehension, summarisation, and chain-of-thought reasoning.
- Each example is paired with its top-3 retrieved chunks. Each chunk becomes a separate training instance. Gold context is used where the task provides it.
- The mixture adds 10% unsupervised text and 5% OASST-1, with at most 7,500 examples per dataset.
- Training used 8, 16 and 64 A100s for 7B, 13B and 65B, with only a few hundred steps and early stopping.
- SQuAD 2.0 in the mix teaches the model to answer "I don't know" when the passage is irrelevant.

**Retriever fine-tuning (R-ft)** [3]:

- A generalised LM-Supervised Retrieval (LSR) objective. Data is 95% "corpus" self-supervision (900k chunks: first 50 tokens as input, last 50 as target) and 5% multi-task QA data.
- The top-10 chunks and their LSR scores are precomputed once with LLaMA 65B and not refreshed.
- Only the query encoder is updated, on 16 A100s for one epoch, with lr 1e-5, temperature 0.01 and per-GPU batch size 32.
- Because the document encoder is frozen, the index never needs re-embedding.

**Results** [3]:

- RA-DIT 65B beat REPLUG (LLaMA 65B + DRAGON+) by +8.9 points 0-shot and +1.4 points 5-shot, averaged over MMLU, NQ, TriviaQA and ELI5. 0-shot NQ rose from 28.8 to 35.2.
- In a 64-shot multi-task setting it averaged 60.9 against Atlas's 56.8, using one model for all tasks where Atlas used task-specific models.
- Commonsense-reasoning scores *without* retrieval rose slightly (average 72.1 to 74.5), so parametric ability was kept.

**Ablations worth copying** [3]:

- LM-ft alone and R-ft alone each beat the REPLUG baseline (5-shot dev averages 53.9 and 54.4 against 53.8). Together they reached 54.6.
- Updating both retriever encoders hurt (5-shot dev average 54.0 against 56.6 for the base retriever). Freezing the document tower was strictly better.
- Corpus-only LSR data generalised better than task-only data. The 95/5 mix was best.
- Iterating the two stages (re-scoring with the tuned LM, or re-retrieving with the tuned retriever) gave no further gain in one or two rounds.

**Lessons.** Most of the gain comes from teaching the LM to use and ignore retrieved text. Retriever alignment adds a smaller increment, and it is cheap when restricted to the query tower. The repository releases the data-prep scripts and a DRAGON index of Wikipedia 2021 [4].

### 17.2.3 ChatQA (NVIDIA, 2024): context-enhanced instruction tuning plus a tuned conversational retriever

**Problem.** A model that has only had general instruction tuning can hold a conversation but answers poorly when it must ground answers in supplied context. This is worst in multi-turn RAG, where the latest user turn ("what about the second one?") is not a usable search query [39].

**Architecture** [39]:

```
 Stage 1: general SFT (about 128K instruction + dialog samples)
          -> conversational, but weak at context-grounded QA
 Stage 2: context-enhanced instruction tuning on a blend of
          - HumanAnnotatedConvQA (about 7k dialogues)
          - SyntheticConvQA (about 7k dialogues from GPT-3.5-Turbo)
          - single-turn context QA (DROP, NarrativeQA, ...)
          - a slice of stage-1 data (to keep instruction following)
          - about 1.5k turns annotated "unanswerable"
 Retriever: dense retriever fine-tuned on multi-turn conversational QA,
            taking dialogue history + current turn as the query
```

**Key techniques.**

- **Unanswerable turns.** These teach the model to say "cannot find the answer in the context" rather than invent one. This is the generator-side counterpart to RAFT's distractor-only training examples [39][1].
- **Tuned conversational retriever.** The authors report that it performs comparably to state-of-the-art LLM query-rewriting approaches while "substantially reducing deployment costs", because it removes the extra LLM call per turn [39].

**Results.** The ChatQA-70B model is reported to outperform GPT-4 on several conversational QA datasets in the authors' ChatRAG Bench. The training data and benchmark were released [39]. ChatQA 2 moves to a Llama 3 base with 128K context and a three-stage recipe, adding long-context tuning [40].

**Lesson.** Collapsing query rewriting into the retriever's weights is a latency optimisation that pays off when conversational traffic is high. The cost is that the retriever must be retrained whenever dialogue patterns shift.

### 17.2.4 RankRAG (NVIDIA / Georgia Tech, NeurIPS 2024): one LLM for reranking and generation

**Problem.** A typical RAG stack uses a separate reranker to pick the top-k contexts before generation. Small rerankers generalise poorly, and the generator never learns to judge relevance itself [53].

**Method.** RankRAG extends ChatQA-style context-enhanced instruction tuning (section 17.2.3) with ranking tasks. The instruction-tuning blend adds a small share of context-ranking data: given a question and a passage, output whether the passage is relevant, plus retrieval-augmented ranking over passage sets. At inference the same LLM first reranks the retrieved candidates (for example the top 100), then generates from the top-k it selected [53].

**Results.** The authors report that Llama3-RankRAG (8B and 70B) outperforms ChatQA-1.5 and GPT-4-class baselines on nine knowledge-intensive benchmarks. It also does well on biomedical RAG benchmarks without domain fine-tuning. They also report that adding only a small fraction of ranking data to the blend was enough for ranking quality comparable to, or better than, dedicated rerankers trained on far more ranking data [53].

**Trade-off.** Scoring 100 candidates with a 70B model is much more expensive than a cross-encoder pass, so the reranking step needs batching or a smaller candidate set. The design matters mainly as evidence that ranking and reading are *one* skill that can be trained jointly, which is the same intuition behind RAFT's distractor training [1].

---

## 17.3 LM-supervised retriever training: REPLUG-LSR and Atlas

RA-DIT's R-ft step descends from two earlier systems. Both use the same signal: a passage is relevant if conditioning on it raises the LM's likelihood of the correct output.

### 17.3.1 REPLUG-LSR (UW / Meta, 2023)

REPLUG prepends each retrieved document separately to a *frozen, black-box* LM and ensembles the outputs [12]. REPLUG-LSR then trains the retriever:

1. Retrieve the top 20 documents.
2. Score each by the LM's likelihood of the ground-truth continuation.
3. Minimise the KL divergence between the retriever's softmax and the LM-likelihood softmax, with temperature 0.1 on both.

The retriever started from Contriever, and GPT-3 Curie supplied the supervision. Because the document encoder is trained too, embeddings go stale. REPLUG re-embeds the corpus and rebuilds the FAISS index every 3k steps, over 25k steps total [12]. On Pile language modelling, LSR roughly doubled REPLUG's gain for some models: GPT-3 Davinci's bits-per-byte improvement went from 3.8% (REPLUG) to 6.3% (LSR) [12].

The design matters for API-only LLMs. The generator needs only to return log-probabilities; it is never updated.

### 17.3.2 Atlas (Meta, 2022): retriever losses and the index-staleness problem

Atlas jointly pre-trains a Contriever retriever and a T5 Fusion-in-Decoder reader. Its main contributions for system designers are a careful comparison of retriever training signals and a cost model for keeping the index fresh [11].

| Retriever loss | Signal from the reader | Atlas finding |
|---|---|---|
| Attention Distillation (ADist) | cross-attention mass on each passage | works; less stable |
| EMDR^2 | passages as latent variables (EM-style) | works; less stable |
| Perplexity Distillation (PDist) | how much each passage alone raises output likelihood | chosen: stable and cheap |
| Leave-one-out (LOOP) | likelihood drop when a passage is removed | strong but costly |

There were no large systematic differences between the losses, so Atlas adopted PDist for stability and efficiency [11]. The authors also observed that joint retriever training helped the pre-training metric a lot, but helped less in 64-shot tasks and hardly at all with 1,024 examples. They read this as evidence that "the biggest impact of pre-training is on the language model" [11].

**Keeping the index fresh during retriever training** [11]:

| Strategy | Overhead | Quality |
|---|---|---|
| Full re-index | about 30% (37M docs, batch 64, 20 docs per query, refresh every 1,000 steps) | best for large data |
| Over-retrieve L and re-embed/re-rank with the current retriever | about 10% when L = 10K | close to full re-index |
| Query-side fine-tuning (freeze document encoder) | zero | best at 64-shot; weaker at 1,024-shot |
| Fixed retriever | zero | clearly worse |

**Headline result.** Atlas reached over 42% on Natural Questions with 64 examples. It beat a 540B model by 3 points with 50x fewer parameters [11].

**Lesson.** RA-DIT [3] and Atlas [11] both found query-side-only training to be the best trade-off when labelled data is scarce. That should be the default in production, where re-embedding a corpus of hundreds of millions of chunks is the dominant cost (see Chapters 3 and 4).

### 17.3.3 Reader-to-retriever distillation (Izacard and Grave, ICLR 2021)

Atlas's ADist loss comes from an earlier Meta paper [41]. It trains a retriever for open-domain QA *without* annotated query-passage pairs. A Fusion-in-Decoder reader is trained on (question, retrieved passages, answer). The reader's cross-attention mass over each passage is then used as a soft relevance label, and the retriever is trained to match it. Secondary summaries report that the reader and retriever were updated in alternating rounds, and that KL divergence between the normalised score distributions worked best among the losses tested [41]. This is the conceptual root of "the generator is the teacher": REPLUG-LSR, Atlas PDist and RA-DIT R-ft replace attention with likelihood, which also works for decoder-only and black-box LMs.

### 17.3.4 Reinforcement learning on retrieval outcomes (2025)

A newer line of work skips relevance labels and teacher scores entirely. It rewards a model for *retrieval outcomes* or *final-answer correctness*, with the search engine treated as part of the environment.

**Search-R1 (UIUC et al., COLM 2025).** An LLM is trained with RL to interleave reasoning with self-generated search calls. Retrieved text is spliced into the rollout [42]. Two design choices matter:

1. **Retrieved-token masking.** The policy-gradient loss is applied only to tokens the LLM generated, not to retrieved passages. The authors report that this stabilises training [42].
2. **Outcome-only reward.** The reward is exact-match correctness of the final answer, with no process supervision [42].

The reported relative gains over RAG baselines across seven QA datasets differ between paper versions. Later versions report about 24% for Qwen2.5-7B and about 20% for Qwen2.5-3B; an earlier version reported 41% for 7B [42]. Treat these as approximate.

**DeepRetrieval (UIUC, 2025).** The model rewrites queries inside a `<think>`/`<answer>` format, and the reward is the retrieval metric itself (recall, nDCG, or SQL execution accuracy). It needs no labelled query-rewrite pairs [43]. Against *real* search engines, the project reports 65.07% recall for publication search (prior best 24.68%) and 63.18% for clinical-trial search (prior best 32.11%), both with a 3B model (DeepRetrieval-3B). Earlier preprint versions reported different figures [43].

```
  question --> policy LLM --(query)--> frozen search engine / retriever
                   ^                           |
                   |                       top-k docs
                   |                           v
             PPO / GRPO update <-- reward: answer EM (Search-R1)
                                   or recall / nDCG (DeepRetrieval)
```

**Design implication.** These methods train the *query side* (the LLM that writes queries) and leave the index untouched, which is the same zero-re-index property that makes query-tower tuning attractive [11][3]. The risk is reward hacking against a specific engine. DeepRetrieval's own v3 title ("Hacking Real Search Engines...") hints that learned queries can be engine-specific and may not transfer when the backend changes [43]. Chapter 8 covers agentic search loops from the serving side.

### 17.3.5 ZeroSearch (Alibaba Tongyi Lab, 2025): RL without a live search engine

Search-R1-style training issues hundreds of thousands of search calls during rollouts. That is costly with commercial APIs, and the quality of returned documents is uncontrolled. ZeroSearch replaces the live engine with a **simulated search LLM** [54]:

- An LLM (released in 3B, 7B and 14B versions) is fine-tuned to act as a retrieval module. Given a query, it generates either relevant or deliberately noisy documents.
- **Curriculum rollout.** The share of noisy documents rises during training (configured with start and end thresholds), so the policy faces harder retrieval conditions over time.
- The policy (Qwen2.5 3B and 7B, or Llama-3.2-3B, base or instruct) is trained with PPO, GRPO or REINFORCE, as in Search-R1 [54][42].

The authors claim that policies trained this way match or beat those trained against a real search engine, with zero API cost. The 14B simulator is recommended for stable training [54]. The design point is that the *environment* in retrieval RL can itself be a trained model. That gives control over document difficulty, but it risks teaching the policy the simulator's quirks instead of the production engine's, which is the reward-hacking concern above seen from the other side.

---

## 17.4 Synthetic queries: training retrievers and rerankers without labels

Most corpora have documents but no queries. The common move is to have a generator invert the problem: write the query that this document answers. Designs differ in four places:

- the generator (fine-tuned T5 or a prompted LLM),
- the prompt (generic or task-specific few-shot),
- the filter (LM log-probability, a reranker, or round-trip consistency),
- the student (reranker or bi-encoder, with hard or soft labels).

### 17.4.1 InPars v1 and v2 (NeuralMind / Zeta Alpha, 2022-2023)

**InPars-v1** prompted GPT-3 Curie with 3 MS MARCO examples to write one query per sampled document. It kept the 10k pairs with the highest generation log-probability and fine-tuned a monoT5 reranker on them [13][14].

**InPars-v2** made the pipeline fully open and changed the filter [14]:

```
 corpus (BEIR dataset) --sample 100k docs--> GPT-J 6B, 3-shot MS MARCO "gbq" prompt,
                                             greedy, 1 query/doc   (~30 h on one A100)
        |
        v
 monoT5-3B (MS MARCO-tuned) scores all 100k (q, d) pairs  (~1.5 h on TPU v3-8)
        |  keep top 10k as positives
        v
 negatives: 1 random doc from BM25 top-1000 for each synthetic query
        |
        v
 fine-tune monoT5-3B (already MS MARCO-tuned) 1 epoch on 10k pos + 10k neg
 (Adafactor, lr 1e-3, batch 64+64; < 10 min on TPU v3-8)  -> one reranker per dataset
        |
 serve: BM25 top-1000 (Pyserini, k1=0.9, b=0.4) -> InPars-v2 monoT5-3B rerank
```

**Results (BEIR nDCG@10)** [14]:

| System | Average |
|---|---|
| BM25 | 0.424 |
| monoT5-3B (MS MARCO) | 0.533 |
| + InPars-v1 data | 0.539 |
| + InPars-v2 data | 0.545 (then state of the art) |

On the subset Promptagator reported, InPars-v2 scored 0.533 against Promptagator's 0.531 and RankT5's 0.536. InPars-v2 was clearly better than v1 on TREC-News, Climate-FEVER, Robust04 and Touche. It still could not beat BM25 on the argument-retrieval sets (ArguAna, Touche), where Promptagator's dataset-specific prompts helped. The authors saw more than 10 nDCG points on ArguAna from a dataset-specific prompt in preliminary runs [14].

**Tooling.** The `inpars` package wraps the steps (`generate`, `filter`, `generate_triples`, `train`, `rerank`, `evaluate`). It supports two filters, LLM scores and an auxiliary reranker, and publishes the v1 and v2 synthetic datasets and fine-tuned models [15].

### 17.4.2 Promptagator (Google, 2022)

**Problem.** Retrieval tasks have different *search intents*: argument retrieval, fact checking, duplicate-question finding. A single MS MARCO-trained model transfers poorly across them [16].

**Method** [16]:

- A task description and at most 8 labelled examples prompt FLAN (137B) to generate queries over the target corpus.
- **Round-trip consistency filtering.** A retriever trained only on the synthetic data must rank the source passage first (K=1 in the secondary description) for the pair to survive [16][19].
- A dual encoder is trained on the filtered data. Promptagator++ adds a cross-attention reranker trained on the same data. The InPars-v2 authors note that both are T5 models of about 110M parameters, and that no MS MARCO data was used [14].

**Results.** The abstract-level claim is that the few-shot dual encoder beat ColBERTv2 and SPLADEv2, which were trained on MS MARCO's 500k+ labelled pairs, across 11 BEIR datasets. Overall it improved over prior work by more than 1.2 nDCG points [16]. Its reranker is strongest on intent-heavy sets: ArguAna 0.630 and Touche 0.381 nDCG@10, against InPars-v2's 0.406 and 0.291 [14].

**Follow-up.** A secondary source reports that a Vespa follow-up repeated the recipe with FLAN-T5-XL (3B) as the generator and reached 80.2 nDCG@10 on TREC-COVID, against Promptagator's 76.2. That points to prompt design and filtering mattering more than generator scale. It is one dataset only [19].

**Lessons.** Task-specific prompts carry the intent definition. Round-trip filtering using only synthetic data removes the need for an external labelled set. The cost is a very large generator run over the whole corpus.

### 17.4.3 GPL: Generative Pseudo Labeling (UKP / Hugging Face, NAACL 2022)

**Problem.** Dense retrievers degrade badly under domain shift. Plain synthetic-query training (QGen) uses hard 0/1 labels, so bad queries and false negatives both inject noise [17].

**Method** [17][18]:

1. A T5 doc2query model (`BeIR/query-gen-msmarco-t5-base-v1`) writes queries for target passages. The total is capped at about 250K queries: at least 3 per passage, with the corpus downsampled if needed.
2. Negatives are mined with two general-domain bi-encoders. The README stresses that the miners "do not need to be strong for the target task/domain".
3. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores both the (q, pos) and (q, neg) pairs.
4. The student learns the teacher's *margin* CE(q, pos) - CE(q, neg) with MarginMSE. A bad query gets a small margin and therefore little weight. A false negative gets a high CE score, so the student is not pushed to separate it.

**Scale and cost.** The defaults are 140K steps at batch 32, and gains saturate around 100K steps [17]. The Sentence Transformers documentation reports about one day on one V100 [18].

**Results (nDCG@10, six BEIR domains)** [17]:

| Model | Average |
|---|---|
| Zero-shot MS MARCO model | 45.2 |
| QGen | 48.8 |
| QGen with hard negatives | 46.3 (2.5 below QGen) |
| GPL | 51.5 (up to +9.3 on a single dataset) |
| TSDAE pre-training + GPL | 52.9 (+7.7 over zero-shot) |
| BM25 + cross-encoder rerank | 54.7 |

The QGen-with-hard-negatives row is the key lesson. Hard negatives with hard labels made things *worse* because of false negatives. Soft teacher margins fixed this. A cross-encoder reranker on BM25 still beat every single-stage dense model, so GPL narrows the gap to reranking but does not close it [17].

### 17.4.4 UDAPDR (Stanford et al., 2023)

UDAPDR, as described in a secondary survey, targets zero-shot domain shift with no in-domain queries [20][19]:

- An expensive LLM (GPT-3 text-davinci-002, five prompting strategies) writes queries for a few sampled passages.
- Those pairs become few-shot demonstrations for a *cheaper* LLM, which generates at scale over 10k to 100k passages.
- Queries are kept only if a zero-shot ColBERTv2 returns the gold passage in its top 20.
- A DeBERTa-v3-large reranker is trained on the filtered data and then distilled into a ColBERTv2 retriever.

The pattern is "pay for quality once, then amplify cheaply" at the generation step, and "rerank, then distil" at the training step. It recurs throughout this chapter.

### 17.4.5 LLM-scale synthetic data for general-purpose embedders (2022-2025)

The same idea now drives the frontier embedding models, at much larger scale:

- **E5 (Microsoft, 2022).** Built 1.3B weakly paired texts (Reddit, Stack Exchange, Wikipedia, papers, Common Crawl). A *consistency-based filter* then kept a pair only if a model trained on the noisy data ranked it in the top 2 against 1M random passages. That left about 270M pairs [31].
- **E5-Mistral (Microsoft, 2024).** A two-step prompt has GPT-4 (with about 25% GPT-3.5-Turbo) first brainstorm retrieval tasks, then write (query, positive, *hard negative*) triplets for each. The output was 500k examples across 150k instructions in 93 languages, using about 180M tokens. Mistral-7B was fine-tuned with LoRA (rank 16) for one epoch, under 1k steps. Training on synthetic data alone was competitive on BEIR and MTEB. Synthetic plus labelled data took first place on MTEB, 2.4 points above the previous best [32].
- **Qwen3-Embedding (Alibaba, 2025).** About 150M synthetic pairs generated by Qwen3-32B for weak-supervision pre-training. Then about 12M filtered synthetic pairs plus about 7M labelled pairs for SFT, with false-negative masking, and finally checkpoint merging. In the 0.6B ablation, removing the synthetic stage dropped MMTEB from 64.33 to 61.21 [33] (also summarised in Chapter 3).
- **Gecko (Google DeepMind, 2024).** A two-step LLM distillation. (1) A few-shot-prompted LLM reads a sampled web passage and writes a *task description* and a matching query. (2) A pre-trained embedder retrieves nearest-neighbour candidates for that query, and the same LLM *relabels* them, choosing a positive (often not the seed passage) and a hard negative. The resulting dataset is FRet. Training is pre-finetuning on web pairs, then fine-tuning on FRet mixed with academic sets (NQ, HotpotQA and others), with a Matryoshka loss. On MTEB, the 768-dimension model reported an average of 66.31, and the 256-dimension version beat all existing 768-dimension entries at the time [44]. The relabelling step is the important idea. It recognises that the passage a query was generated from is not always the *best* answer to that query, which is the same false-positive problem round-trip filtering tries to catch [16].
- **Gemini Embedding (Google, 2025).** Gemini itself generates synthetic examples, filters bad training pairs and helps mine hard negatives [34].
- **ReasonIR (Meta / UW et al., 2025).** Existing training sets pair short factual queries with documents that answer them directly, so retrievers do poorly on reasoning-heavy tasks. ReasonIR-Synthesizer writes a hard, reasoning-intensive query for each real seed document, plus varied-length query/document pairs. For each query it synthesises a "plausibly related but ultimately unhelpful" hard negative over several turns. The authors found that standard hard-negative mining does not work for such queries. ReasonIR-8B reached 24.4 nDCG@10 on BRIGHT with original queries, 29.9 with GPT-4-rewritten queries, and 36.9 with an LLM reranker. In RAG it improved MMLU by 6.4% and GPQA by 22.6% [35].

### 17.4.6 Later refinements of the InPars family and enterprise practice (2025)

- **InPars+ (2025).** This paper re-evaluates InPars, InPars-v2 and Promptagator on SciFact and adds two generator-side changes. The query generator is fine-tuned with Contrastive Preference Optimization, and the static prompts are replaced with DSPy-optimised prompts. The authors report that both changes reduce the need for aggressive post-hoc filtering while improving retrieval [45]. The direction is to move quality control *upstream*, into the generator, instead of generating ten times too many queries and discarding most of them.
- **Small open generators.** A 2025 study of Promptagator-style training reports that open LLMs as small as about 3B parameters can serve as effective query generators. That removes the original dependency on a proprietary 137B model [46].
- **Databricks (2025).** In a public write-up, Databricks fine-tuned `gte-large-en-v1.5` and `e5-mistral-7b-instruct` on *synthetic* queries generated over three enterprise corpora (FinanceBench, ManufactQA, Databricks DocsQA). It compared them with OpenAI `text-embedding-3-large` on Recall@10 and end-to-end RAG quality [47]. Reported findings [47]:
  - Fine-tuning improved Recall@10 across datasets, often significantly.
  - On FinanceBench the retrieval gain carried through to better RAG answers.
  - Fine-tuned embedders were as good as or better than adding a reranker in many cases.
  - The win was not universal: `text-embedding-3-large` still had slightly higher Recall@10 than every `gte` variant in at least one setting.
  
  Databricks' retrieval-quality guide points to this synthetic-data recipe but describes fine-tuning as a last resort, after cheaper fixes have been tried [48].

### 17.4.7 Where synthetic queries fail

- **Graded relevance.** Google researchers studied three e-commerce benchmarks with fine-grained relevance labels. They found that, "contrary to claims from prior works", QGen approaches fell short of conventional cross-domain transfer learning. QGen models could not tell relevance grades apart. Label-conditioned generation helped, but the generated queries were still not faithful to the intended label [21].
- **Quantity versus quality.** Twelve times more E2E-generated QA pairs lost to fewer prompt-generated ones [6].
- **Intent mismatch.** Generic prompts produce MS MARCO-style factoid questions. Argument retrieval needs task-specific prompts [14][16].
- **Lexical copying.** Generated queries tend to reuse the passage's words. Always evaluate on held-out *real* queries (Chapter 3 discusses this further).

---

## 17.5 Hard-negative mining

In-batch negatives are mostly easy. Contrastive training learns fastest from negatives that the current model confuses with positives. The danger is that, in sparsely labelled corpora, many of those "negatives" are in fact relevant.

### 17.5.1 ANCE (Microsoft, 2020): global negatives from an asynchronously refreshed index

ANCE argues on theoretical grounds that local in-batch negatives are uninformative. They yield near-zero loss and small gradient norms, which slows convergence. ANCE therefore samples negatives from the *whole corpus*, using an ANN index built from a recent checkpoint [22].

```
 Trainer (GPUs) --------- checkpoint k ---------> Inferencer (GPUs)
    ^                                              re-encode entire corpus
    |                                              rebuild ANN index (Faiss)
    +---- ANCE negatives = top-ranked docs <-------+  (swap in when done)
          from the (slightly stale) index
```

A 1:1 Trainer:Inferencer GPU split was enough to keep the staleness gap small. Re-encoding the corpus is the bulk of the cost: on the order of 10 hours per refresh in the authors' document-retrieval setup (approximate, read from their efficiency table) [22]. At serving time the dense retriever was about 100x faster than BERT reranking [22].

### 17.5.2 RocketQA (Baidu, 2021): cross-batch negatives, denoising and augmentation

RocketQA starts from a measurement. MS MARCO has about 1.1 labelled positives per query in an 8.8M-passage collection. When the authors manually checked top-retrieved passages that were labelled non-relevant, about 70% were actually positive or highly relevant [23]. The four-step pipeline [23]:

1. Train a dual encoder with **cross-batch negatives**: passages from all GPUs' batches are shared through all-gather, at almost no extra cost on multi-GPU setups.
2. Train a cross-encoder.
3. Retrain the dual encoder with **denoised hard negatives**. Retrieved passages are kept as negatives only if the cross-encoder scores them below 0.1.
4. **Data augmentation.** The cross-encoder labels unlabelled questions: scores above 0.9 become positives and below 0.1 become negatives. Manual checks put this labelling at over 90% accuracy.

**MS MARCO ablation (MRR@10)** [23]:

| Training strategy | MRR@10 |
|---|---|
| In-batch negatives | 32.39 |
| Cross-batch negatives | 33.32 |
| Hard negatives *without* denoising | 26.03 |
| Hard negatives with denoising | 36.38 |
| + data augmentation | 37.02 |

Undenoised hard negatives cost about 7 points. This is the same false-negative effect that GPL saw with QGen and hard negatives [17].

### 17.5.3 NV-Retriever (NVIDIA, 2024): positive-aware mining

NV-Retriever argues that absolute score thresholds and fixed rank windows are crude. It sets the threshold relative to each query's *positive* score [24]:

- **TopK-MarginPos:** drop candidates scoring within a fixed margin of the positive.
- **TopK-PercPos:** drop candidates scoring above a percentage of the positive's score. 95% was the best setting.

Secondary summaries of the paper add two points. Negatives mined by different teacher models overlap little (Jaccard similarity under 30%), so ensembling teachers diversifies the negative set. The resulting NV-Retriever-v1 (Mistral-7B based) reported 60.9 average nDCG@10 on the MTEB/BEIR retrieval tasks and ranked first on that leaderboard in July 2024 [24].

The Sentence Transformers `mine_hard_negatives` utility encodes this recipe directly. It recommends `relative_margin=0.05`, `num_negatives` of 10 or fewer and `sampling_strategy="top"`, and labels this as the "TopK-PercPos (95%)" method from the paper [25]. Its other options are the standard practitioner knobs [25]:

- `range_min` skips the top few ranks, where false negatives cluster.
- `range_max` limits the candidate pool.
- `max_score` drops near-duplicates.
- `absolute_margin` and `relative_margin` set positive-aware thresholds.
- An optional cross-encoder rescores candidates.

In the docstring's NQ example, a 0.05 relative margin removed 8.36% of candidate negatives [25].

### 17.5.4 Comparison of negative strategies

| Strategy | Source of negatives | False-negative control | Cost | Evidence |
|---|---|---|---|---|
| In-batch random | other queries' positives | none needed (mostly easy) | free | baseline [23] |
| Cross-batch | all GPUs' in-batch passages | none | about free on multi-GPU | +0.9 MRR@10 [23] |
| BM25 top-k | lexical retriever | none | cheap | InPars negatives [14] |
| Static dense top-k | frozen bi-encoder | rank window, max score | one corpus encode | GPL miners [17] |
| ANCE | current model, async ANN | none (relies on volume) | periodic re-encode | [22] |
| Denoised (RocketQA) | dense top-k | cross-encoder score < 0.1 | cross-encoder pass | +10 MRR@10 vs undenoised [23] |
| Soft labels (GPL, MarginMSE) | dense top-k | teacher margin, no hard label | cross-encoder pass | +5.2 nDCG vs QGen + hard negatives [17] |
| Positive-aware (NV-Retriever) | teacher top-k, ensembles | score relative to positive (95%) | teacher encode | [24][25] |
| LLM-synthesised (E5-Mistral, ReasonIR) | generated text | generator instructed to make it unhelpful | LLM tokens | [32][35] |

---

## 17.6 Distillation into retrievers and rerankers

Distillation runs down a quality/latency ladder: LLM listwise ranker, then cross-encoder, then late-interaction model, then bi-encoder. Each rung trains the one below it.

```
  GPT-4 / R1-class LLM  --permutations / reasoning traces-->  7B-32B LLM reranker
          |                                                   (RankZephyr, Rank1)
          | permutation distillation (RankNet)                        |
          v                                                           v
  cross-encoder (DeBERTa / monoT5 / ModernBERT)  <--- scores --- (teacher for)
          |
          | MarginMSE / KL over top-k (GPL, ColBERTv2)
          v
  late-interaction (ColBERTv2)  /  bi-encoder (GPL student, E5)
          |
          | query-side-only or adapters to avoid re-indexing
          v
  production first stage
```

### 17.6.1 Cross-encoder to retriever

- **MarginMSE (TU Wien, 2020).** Hofstätter et al. observed that different ranking architectures produce scores on incompatible scales. They therefore trained efficient students (TK, ColBERT, PreTT and a BERT dot-product bi-encoder) to match the *margin* between a teacher BERT cross-encoder's positive and negative scores, using MSE. In their MS MARCO ablation, Margin-MSE beat pointwise MSE and weighted RankNet for every student architecture. The authors released the teacher-score files, so later work could reuse the supervision without rerunning the teacher [49]. GPL later reused the same loss for unsupervised domain adaptation, where it also absorbs label noise (section 17.4.3) [17].
- **TAS-Balanced (TU Wien, SIGIR 2021).** TAS-B makes distillation cheap. Training queries are clustered once, before training, and each batch draws its queries from a single topic cluster, so in-batch negatives are topically hard. Negatives are also balanced across margin levels. Supervision mixes a pairwise cross-encoder teacher with an in-batch ColBERT teacher. The 6-layer DistilBERT student is reported to train on a single consumer GPU in under 48 hours, against a common setup of 8 V100s, and it served TREC-DL queries at about 64 ms [50]. TAS-B checkpoints became a common base retriever for later domain-adaptation work.
- **ColBERTv2 (Stanford, 2022).** For each training query, ColBERTv2 retrieves the top-k passages and scores them with a 22M-parameter MiniLM cross-encoder. It distils those scores with KL divergence, because ColBERT's sum-of-cosines scores have a restricted scale that does not align with cross-encoder logits. In-batch negatives are added per GPU. Residual compression then cuts the index 6-10x [26]. The paper credits "denoised supervision" for much of ColBERTv2's quality jump over ColBERT [26].
- **BGE-M3 self-knowledge distillation.** The combined score of the dense, sparse and multi-vector heads acts as teacher for each single head [36] (see Chapter 3).

### 17.6.2 LLM to cross-encoder: RankGPT permutation distillation

RankGPT showed that GPT-4 prompted to output a *permutation* of candidate passages beat supervised rerankers. It reranked the top 100 BM25 results with a sliding window of size 20 and step 10, and averaged +2.7 and +2.3 nDCG@10 over monoT5-3B on TREC and BEIR [27].

To make that affordable, the authors sampled 10K MS MARCO queries, retrieved 20 BM25 candidates for each, and had ChatGPT rank them. A DeBERTa-large cross-encoder was trained on these permutations with RankNet loss, which yields M(M-1)/2 pairs per list. The resulting **DeBERTa-large student (about 440M parameters) outperformed the 3B monoT5 on BEIR**, with no human relevance labels [27]. The margin is small, under 2 nDCG@10 points on average. The public RankGPT repository's training command uses DeBERTa-v3-base, so check which size a released checkpoint is before reusing the numbers. Castorini's RankLLM toolkit packages the open successors: listwise RankZephyr and RankVicuna, LiT5-Distill (a T5 listwise model that can rerank 100 documents in one pass), and FirstMistral (first-token logits only) [28].

**RankZephyr (Waterloo, 2023)** shows the listwise version of the same recipe. A 7B Zephyr model was instruction-tuned to output permutations matching RankGPT-3.5 and then RankGPT-4 orderings of BM25 candidate lists. RankVicuna's training data came from RankGPT-3.5 reorderings over 100K MS MARCO training queries [51]. The authors report that RankZephyr closes most of the gap to GPT-4 and outperforms it on NovelEval, a set of queries and passages newer than the models' training data, which addresses contamination concerns [51]. The advantage is metric-dependent: the same paper reports Judged@10 2-4 points below RankGPT-4 [51].

### 17.6.3 Reasoning-trace distillation: Rank1 (JHU, 2025)

Rank1 distils test-time reasoning into a pointwise reranker. A secondary survey reports that it sampled 635,000 DeepSeek-R1 reasoning chains on MS MARCO [29]. Qwen2.5 bases from 0.5B to 32B, plus Mistral-Small-24B and Llama-3.1-8B variants, were fine-tuned to "think" before giving a relevance judgment. The repository publishes the R1 outputs and the training data, and offers AWQ-quantised variants that fit in 24GB GPUs [30]. The README notes that tasks whose relevance definition differs from MS MARCO need a custom prompt [30].

There is a cost-quality trade-off. ReasonIR reports that its retriever with query rewriting beats recent LLM-reranker baselines while needing "over 200x less compute" [35].

### 17.6.4 SFT then RL for listwise reasoning rerankers: ReasonRank (2025)

Rank1 is pointwise and trained only by imitation. ReasonRank (Renmin University / Baidu, ACL 2026) moves the reasoning-trace recipe to the *listwise* setting and adds an RL stage [55]:

1. **Data synthesis.** DeepSeek-R1 performs listwise reranking over queries from several domains, producing both reasoning chains and relevance labels. A self-consistency filter keeps only examples whose labels reach NDCG@10 above 0.4. The released training set has about 13K examples.
2. **Cold-start SFT** on the reasoning traces.
3. **RL with GRPO**, using a "multi-view" ranking reward based on ranking metrics over the sliding-window output, in place of a plain exact-match reward.

ReasonRank-7B and -32B are built on Qwen2.5-Instruct, and an 8B model on Qwen3 followed in 2026. The authors report that, with ReasonIR as first stage, ReasonRank beats Rank1 on BRIGHT and is more efficient. With a stronger hybrid first stage, the 32B model reached 40.8 nDCG@10 on BRIGHT, which they describe as state of the art at release [55]. RankLLM later added custom prompt templates so that such models can be served from the same toolkit [28]. The pattern matches the Search-R1 shift on the generator side (section 17.3.4): distil a strong teacher first, then optimise the target metric directly with RL.

### 17.6.5 Domain fine-tuning of small rerankers (2024-2025 practice)

- **Hugging Face (Aarsen, 2025).** A 150M ModernBERT-base cross-encoder was trained in 30 minutes on one RTX 3090 with binary cross-entropy. The data was 99k of GooAQ's roughly 3M pairs. A static embedding model mined 5 hard negatives per pair (`num_negatives=5`, `range_max=100`, `sampling_strategy="top"` in the training script), giving 578k labelled pairs: 99k positive and 479k negative. The post's illustrative mining example also shows a `max_score` cap, but the training script does not use one. Reranking the top 30 from a static retriever (59.12 nDCG@10 with no reranking), it reached 77.14. That beat all 13 general-purpose rerankers tested, including `bge-reranker-v2-m3` (568M, 73.56) and `mxbai-rerank-large-v2` (1.54B, 75.40). A 396M ModernBERT-large version trained in under an hour reached 79.42 [29b]. The author notes that the result is domain-specific: the model is "simply the strongest in *our* domain".
- **Managed fine-tuning (Cohere Rerank via Weaviate tutorial, 2024).** Weaviate's walkthrough generated synthetic queries over its blog corpus with DSPy, then fine-tuned Cohere's hosted reranker. The service required at least 256 unique training queries and at least 64 non-overlapping validation queries. Hard negatives were optional but had to differ from the positives. The tutorial states that the fine-tuned model was priced the same as the zero-shot one at the time [30b].

### 17.6.6 Instruction-following retrievers

Promptriever trains bi-encoders (LLaMA-2 7B, LLaMA-3.1 8B, Mistral 7B) so that a per-query natural-language instruction changes what counts as relevant. This lets a retriever be "prompted like a language model" instead of retrained for each intent [37]. It is a complement to Promptagator: one adapts the *data* to the intent, the other makes the *model* accept the intent at query time.

---

## 17.7 End-to-end domain adaptation playbook

The pieces above combine into a standard pipeline. Steps marked (G) touch the generator; the rest touch retrieval.

```
 0. Baseline: BM25 + off-the-shelf embedder + off-the-shelf reranker; build an eval set
    of REAL user queries with graded labels (LLM-judged + human spot checks)
 1. Synthetic queries per chunk
      - generic LLM prompt  -> fast, MS-MARCO-like intents
      - few-shot task prompt (Promptagator) -> intent-specific
      - filter: reranker score (InPars-v2) or round-trip top-1 (Promptagator)
 2. Mine negatives
      - dense top-k with rank window + positive-aware margin (NV-Retriever)
      - denoise with cross-encoder (RocketQA) or keep soft scores (GPL)
 3. Train reranker first (cheapest big win; no re-index) -- BCE / listwise
      - optionally distil an LLM ranker's permutations (RankGPT) into it
 4. Train retriever
      - query-side only / adapter if re-embedding is expensive (Atlas, RA-DIT)
      - else full fine-tune with MarginMSE from the step-3 reranker (GPL)
      - optionally LM-supervised (LSR) if a target generator is fixed (REPLUG, RA-DIT)
      - merge with base weights to limit forgetting (Qwen3 / Gemini model soups)
 5. (G) If the generator misreads context: RAFT-style SFT with oracle + distractors,
      CoT with verbatim quotes, P < 100%; or RA-DIT LM-ft over top-k chunks
 6. (G) If facts must live in weights (offline / latency): synthetic CPT (EntiGraph),
      fact-level coverage, early stopping; keep RAG on top
 7. Re-evaluate end-to-end; track hallucination rate, not only retrieval metrics
```

**Ordering rationale.**

- **Reranker first.** It needs no re-indexing, a few hundred labelled queries are enough for managed services [30b], and small domain rerankers can beat models several times their size [29b].
- **Retriever second.** Any change to the document tower forces a full corpus re-embed. Atlas puts periodic re-indexing at about 30% training overhead [11], and serving systems pay that cost again on every model update.
- **Generator last.** It is the most expensive component to train and the riskiest to change [7]. The evidence also says RAG already supplies the facts [5][6].

**Typical failure modes across the playbook:**

| Symptom | Likely cause | Fix |
|---|---|---|
| Retriever got worse after training on mined negatives | false negatives | denoise with a cross-encoder or use soft margins [17][23] |
| Great synthetic-eval scores, flat real-query scores | synthetic queries copy passage wording or miss intents | task-specific prompts; real-query eval [14][16] |
| Grades of relevance collapse | QGen cannot express graded labels | label-conditioned generation, or transfer learning [21] |
| Fine-tuned generator ignores context | trained without documents (DSF) | RAFT / RA-DIT LM-ft with retrieved context [1][3] |
| Fine-tuned generator invents facts | SFT on Unknown facts | filter Unknown examples, early stop [7] |
| Tuned retriever hurt other domains | catastrophic forgetting | merge with base weights; mix general data [33] |
| Index refresh dominates cost | document tower trained | query-side tuning or re-rank-L strategy [3][11] |

---

## 17.8 Comparison of designs

| Design | Org, year | Component trained | Training signal | Data needed | Cost (public numbers) | Headline result |
|---|---|---|---|---|---|---|
| RAFT [1] | UC Berkeley, 2024 | generator | SFT on Q + oracle + distractors -> CoT with quotes | domain docs; GPT-4 for Q and CoT | Llama-2-7B SFT | HuggingFace 74.00 vs DSF+RAG 42.59 |
| RA-DIT [3] | Meta, 2023 | generator + query encoder | RA instruction tuning + LSR KL | 20 instruction sets; 900k corpus chunks | 64 A100 (65B LM), 16 A100 (retriever) | +8.9 0-shot vs REPLUG |
| REPLUG-LSR [12] | UW/Meta, 2023 | retriever | black-box LM likelihood KL | unlabeled text | re-index every 3k steps | Davinci BPB gain 3.8% to 6.3% |
| Atlas [11] | Meta, 2022 | retriever + FiD reader | PDist (also ADist, EMDR^2, LOOP) | pre-training text + few-shot | about 30% re-index overhead | 42% NQ with 64 examples |
| InPars-v2 [14] | NeuralMind/Zeta Alpha, 2023 | reranker | GPT-J queries filtered by monoT5-3B | 100k docs per dataset | about 30 h A100 generation | BEIR 0.545 |
| Promptagator [16] | Google, 2022 | retriever + reranker | FLAN-137B few-shot queries + round-trip filter | 8 examples per task | 137B-model corpus pass | beats ColBERTv2/SPLADEv2 on 11 BEIR sets |
| GPL [17] | UKP/HF, 2022 | retriever | T5 queries + cross-encoder MarginMSE | unlabeled corpus | about 1 V100-day | +7.7 nDCG (with TSDAE) |
| UDAPDR [20] | Stanford et al., 2023 | reranker -> ColBERTv2 | two-stage LLM queries, distillation | in-domain passages | GPT-3 + small LLM | long-tail domain gains |
| ANCE [22] | Microsoft, 2020 | retriever | global ANN negatives | labelled pairs | periodic corpus re-encode | - |
| RocketQA [23] | Baidu, 2021 | retriever | cross-batch + denoised negatives + CE augmentation | labelled + unlabelled queries | up to 8 V100 | MS MARCO MRR@10 37.0 |
| NV-Retriever [24] | NVIDIA, 2024 | retriever | positive-aware negatives (95%) | labelled pairs + teachers | 7B model training | MTEB retrieval 60.9 (reported) |
| RankGPT distil [27] | Shandong/Baidu, 2023 | cross-encoder | ChatGPT permutations, RankNet | 10K queries x 20 docs | ChatGPT calls once | about 440M student beats monoT5-3B on BEIR |
| Rank1 [29][30] | JHU, 2025 | LLM reranker | R1 reasoning traces | 635K traces (reported) | 0.5B-32B SFT | reasoning reranking |
| E5-Mistral [32] | Microsoft, 2024 | embedder | GPT-4 synthetic triplets | 500k synthetic + labelled | about 180M generated tokens; under 1k steps | MTEB first (+2.4) |
| ReasonIR [35] | Meta/UW, 2025 | embedder | synthetic reasoning queries + synthetic negatives | seed docs | 8B model | BRIGHT 29.9 / 36.9 |
| ChatQA [39] | NVIDIA, 2024 | generator + conversational retriever | two-stage SFT incl. unanswerable turns | about 7k human + 7k synthetic dialogues | 70B SFT | beats GPT-4 on several ConvQA sets (reported) |
| MarginMSE [49] / TAS-B [50] | TU Wien, 2020-2021 | bi-encoder / late interaction | cross-encoder margin; topic-clustered batches | MS MARCO + teacher scores | 1 consumer GPU, under 48 h (TAS-B) | about 64 ms/query on TREC-DL |
| Gecko [44] | Google DeepMind, 2024 | embedder | LLM-generated task + query, LLM-relabelled pos/neg | web passages | not disclosed | MTEB 66.31 (768-d) |
| RankZephyr [51] | Waterloo, 2023 | 7B listwise reranker | GPT-3.5/4 permutation distillation | 100K MS MARCO queries (GPT-3.5 stage) | 7B SFT | beats GPT-4 on NovelEval |
| Search-R1 [42] | UIUC et al., 2025 | generator's search policy | RL, answer-EM reward, retrieved-token masking | QA pairs only | RL rollouts with live search | about 20-24% relative over RAG |
| DeepRetrieval [43] | UIUC, 2025 | query rewriter | RL, recall/nDCG reward | queries + engine | 3B RL | 65.07% recall pub. search |
| EntiGraph CPT [10] | Stanford, 2024 | generator weights | entity-graph synthetic corpus | 1.3M source tokens | 455M generated tokens | about 80% of RAG gain closed-book |
| Cartridges [52] | Stanford, 2025 | per-corpus trained KV cache (weights frozen) | self-study conversations + context distillation | the corpus only | offline training per corpus | ICL-level quality, up to about 26x throughput (reported) |
| RankRAG [53] | NVIDIA, 2024 | generator as reranker + reader | instruction tuning with a small share of ranking data | ChatQA-style blend + ranking data | 8B/70B SFT | beats ChatQA-1.5 on nine benchmarks (reported) |
| ZeroSearch [54] | Alibaba, 2025 | search policy | RL against a simulated search LLM with a noise curriculum | QA pairs | 3B-14B simulator; no API calls | matches or beats real-engine RL (reported) |
| ReasonRank [55] | Renmin/Baidu, 2025 | listwise LLM reranker | R1 listwise traces (SFT) + GRPO ranking reward | about 13K synthesised examples | 7B/32B | BRIGHT 40.8 with a hybrid first stage |

---

## Key takeaways for system designers

1. **Put facts in the index and skills in the weights.** Controlled studies agree that RAG beats fine-tuning for injecting facts [5][6]. SFT on unknown facts can raise hallucination [7]. Fine-tuning adds to RAG when it teaches behaviour or domain fluency [8][1].
2. **If you must put facts in weights, amplify and diversify.** Paraphrases [5], fact-level coverage [9] and entity-graph synthesis [10] are what make continued training absorb knowledge. Keep RAG on top: the effects compound [10].
3. **Fine-tune the generator *with* retrieved context, including bad context.** RAFT's oracle + distractor mix with P below 100% [1], and RA-DIT's one-chunk-per-instance tuning [3], both teach the model to extract from good passages and ignore bad ones. Fine-tuning without context can make RAG worse [1].
4. **The reranker is the highest-leverage, lowest-risk component to tune.** It needs no re-index. A 150M domain model trained in 30 minutes beat general rerankers up to 10x its size on its domain [29b], and LLM rankers distil well into small cross-encoders [27].
5. **Train only the query tower when data is small or the corpus is large.** Atlas and RA-DIT both found query-side tuning competitive or better at low data, and it costs zero re-indexing [11][3]. Fine-tuning both towers hurt in RA-DIT [3].
6. **False negatives are the main hazard in mining.** About 70% of "unlabelled" top-retrieved MS MARCO passages were relevant [23]. Undenoised hard negatives cost 7 MRR points [23], and hard-labelled QGen negatives cost 2.5 nDCG [17]. Use cross-encoder denoising, soft margins, or positive-relative thresholds [24][25].
7. **Synthetic-query quality depends on intent and filtering more than on generator size.** Task-specific few-shot prompts [16], reranker filtering [14] and round-trip consistency matter. One reported follow-up matched Promptagator with a 45x smaller generator [19]. Quality beats volume [6].
8. **Know where synthetic queries break.** They struggle with graded relevance [21] and with reasoning-intensive queries, where synthesised hard negatives work better than mined ones [35]. Always evaluate on real user queries.
9. **Let the generator supervise the retriever when the generator is fixed.** LSR-style KL objectives (REPLUG, RA-DIT, Atlas PDist) align retrieval with what actually helps the LM answer, and they work with black-box APIs that expose log-probabilities [12][3][11].
10. **Consider training the query writer instead of the index.** RL on retrieval outcomes (DeepRetrieval) or answer correctness (Search-R1) improves retrieval with zero re-embedding [42][43]. Distilling conversational rewriting into the retriever (ChatQA) removes an LLM call per turn [39]. Guard against policies that overfit one search backend.
11. **Relabel, don't just filter.** When an LLM generates a query from a passage, the source passage is not always the best positive. Gecko's LLM relabelling of retrieved neighbours [44] and InPars+'s generator-side optimisation [45] improve data quality more efficiently than discarding most generated queries.
12. **Budget for the hidden costs.** Corpus re-embedding (about 30% overhead during joint training [11]), teacher inference (cross-encoder or LLM passes over every candidate [14][27]) and synthetic-token bills (about 180M tokens for E5-Mistral [32]) are usually larger than the gradient steps themselves.

---

## References

1. T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica, J. E. Gonzalez. "RAFT: Adapting Language Model to Domain Specific RAG." UC Berkeley, 2024. https://arxiv.org/abs/2403.10131
2. Gorilla project. "RAFT" dataset-generation README (`raft.py`). UC Berkeley, 2024-2025. https://github.com/ShishirPatil/gorilla/tree/main/raft
3. X. V. Lin, X. Chen, M. Chen, W. Shi, M. Lomeli, R. James, P. Rodriguez, J. Kahn, G. Szilvasy, M. Lewis, L. Zettlemoyer, W. Yih. "RA-DIT: Retrieval-Augmented Dual Instruction Tuning." Meta FAIR, ICLR 2024. https://arxiv.org/abs/2310.01352
4. Facebook Research. "RA-DIT" artifact repository README. Meta, 2024. https://github.com/facebookresearch/RA-DIT
5. O. Ovadia, M. Brief, M. Mishaeli, O. Elisha. "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs." Microsoft, 2023. https://arxiv.org/abs/2312.05934
6. H. Soudani, E. Kanoulas, F. Hasibi. "Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge." Radboud University / University of Amsterdam, 2024. https://arxiv.org/abs/2403.01432
7. Z. Gekhman, G. Yona, R. Aharoni, M. Eyal, A. Feder, R. Reichart, J. Herzig. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" Technion / Google Research, 2024. https://arxiv.org/abs/2405.05904
8. A. de Luis Balaguer et al. "RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture." Microsoft Research, 2024. https://www.microsoft.com/en-us/research/publication/rag-vs-fine-tuning-pipelines-tradeoffs-and-a-case-study-on-agriculture/
9. N. Mecklenburg, Y. Lin, X. Li, D. Holstein, L. Nunes, S. Malvar, B. Silva, R. Chandra, V. Aski, P. K. R. Yannam, T. Aktas. "Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning." Microsoft Research, 2024. https://www.microsoft.com/en-us/research/publication/injecting-new-knowledge-into-large-language-models-via-supervised-fine-tuning/
10. Z. Yang, N. Band, S. Li, E. Candès, T. Hashimoto. "Synthetic continued pretraining." Stanford University, 2024. https://arxiv.org/abs/2409.07431
11. G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, E. Grave. "Atlas: Few-shot Learning with Retrieval Augmented Language Models." Meta, 2022. https://arxiv.org/abs/2208.03299
12. W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, W. Yih. "REPLUG: Retrieval-Augmented Black-Box Language Models." UW / Meta, 2023. https://arxiv.org/abs/2301.12652
13. L. Bonifacio, H. Abonizio, M. Fadaee, R. Nogueira. "InPars: Data Augmentation for Information Retrieval using Large Language Models." NeuralMind / Zeta Alpha, 2022. https://arxiv.org/abs/2202.05144
14. V. Jeronymo, L. Bonifacio, H. Abonizio, M. Fadaee, R. Lotufo, J. Zavrel, R. Nogueira. "InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval." NeuralMind / Zeta Alpha, 2023. https://arxiv.org/abs/2301.01820
15. Zeta Alpha. "InPars: Inquisitive Parrots for Search" toolkit README. 2023-2024. https://github.com/zetaalphavector/InPars
16. Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. B. Hall, M.-W. Chang. "Promptagator: Few-shot Dense Retrieval From 8 Examples." Google Research, 2022. https://arxiv.org/abs/2209.11755
17. K. Wang, N. Thakur, N. Reimers, I. Gurevych. "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval." UKP Lab / Hugging Face, NAACL 2022. https://arxiv.org/abs/2112.07577 (code: https://github.com/UKPLab/gpl)
18. Sentence Transformers. "Domain Adaptation" example README (TSDAE, GPL). UKP Lab / Hugging Face, 2022-2026. https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/domain_adaptation/README.md
19. frutik (awesome-search). "PROMPTAGATOR" concept note summarising Vespa's zero-shot and few-shot comparisons (secondary source). 2026. https://github.com/frutik/awesome-search/blob/master/obsidian/vault/Awesome%20Search/Concepts/PROMPTAGATOR.md
20. J. Saad-Falcon, O. Khattab, K. Santhanam, R. Florian, M. Franz, S. Roukos, A. Sil, M. Sultan, C. Potts. "UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers." 2023. https://arxiv.org/abs/2303.00807 (described in this chapter via secondary summary)
21. A. Chaudhary, K. Raman, K. Srinivasan, K. Hashimoto, M. Bendersky, M. Najork. "Exploring the Viability of Synthetic Query Generation for Relevance Prediction." Google Research / Google DeepMind, SIGIR eCom 2023. https://arxiv.org/abs/2305.11944
22. L. Xiong, C. Xiong, Y. Li, K.-F. Tang, J. Liu, P. Bennett, J. Ahmed, A. Overwijk. "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval" (ANCE). Microsoft, 2020. https://arxiv.org/abs/2007.00808
23. Y. Qu, Y. Ding, J. Liu, K. Liu, R. Ren, W. X. Zhao, D. Dong, H. Wu, H. Wang. "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering." Baidu / Renmin University, NAACL 2021. https://arxiv.org/abs/2010.08191
24. G. de Souza P. Moreira et al. "NV-Retriever: Improving text embedding models with effective hard-negative mining." NVIDIA, 2024. https://arxiv.org/abs/2407.15831
25. Sentence Transformers. `mine_hard_negatives` utility (source and docstring). Hugging Face, 2024-2026. https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/util/hard_negatives.py
26. K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, M. Zaharia. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." Stanford, NAACL 2022. https://arxiv.org/abs/2112.01488
27. W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, Z. Ren. "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" (RankGPT). Shandong University / Baidu / Leiden University, EMNLP 2023. https://arxiv.org/abs/2304.09542
28. Castorini (University of Waterloo). "RankLLM" toolkit README (RankZephyr, RankVicuna, LiT5-Distill, FirstMistral). 2023-2026. https://github.com/castorini/rank_llm
29. O. Weller, K. Ricci, E. Yang, A. Yates, et al. "Rank1: Test-Time Compute for Reranking in Information Retrieval." Johns Hopkins University, 2025. https://arxiv.org/abs/2502.18418 (training-set size from the secondary listing https://github.com/gomate-community/awesome-papers-for-rag/blob/main/ranker.md)
29b. T. Aarsen. "Training and Finetuning Reranker Models with Sentence Transformers." Hugging Face Blog, 2025. https://huggingface.co/blog/train-reranker
30. O. Weller et al. "rank1" repository README (models, datasets, quantised variants). JHU CLSP, 2025. https://github.com/orionw/rank1
30b. Weaviate (Erika). "Fine-Tuning Cohere's Reranker." Weaviate Blog, 2024. https://github.com/weaviate/weaviate-io/blob/main/blog/2024-02-13-fine-tuning-cohere-reranker/index.mdx
31. L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, F. Wei. "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5). Microsoft, 2022. https://arxiv.org/abs/2212.03533
32. L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, F. Wei. "Improving Text Embeddings with Large Language Models" (E5-Mistral). Microsoft, 2024. https://arxiv.org/abs/2401.00368
33. Y. Zhang, M. Li, D. Long, et al. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." Alibaba Qwen, 2025. https://arxiv.org/abs/2506.05176 (repository: https://github.com/QwenLM/Qwen3-Embedding)
34. J. Lee, F. Chen, S. Dua, D. Cer, M. Shanbhogue, et al. "Gemini Embedding: Generalizable Embeddings from Gemini." Google, 2025. https://arxiv.org/abs/2503.07891
35. R. Shao, R. Qiao, V. Kishore, N. Muennighoff, X. V. Lin, D. Rus, B. K. H. Low, S. Min, W. Yih, P. W. Koh, L. Zettlemoyer. "ReasonIR: Training Retrievers for Reasoning Tasks." Meta FAIR / UW et al., 2025. https://arxiv.org/abs/2504.20595 (code: https://github.com/facebookresearch/ReasonIR)
36. J. Chen et al. "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." BAAI, 2024. https://arxiv.org/abs/2402.03216
37. O. Weller et al. "Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models." 2024. https://arxiv.org/abs/2409.11136 (code: https://github.com/orionw/promptriever)
38. T. Aarsen. "Training and Finetuning Embedding Models with Sentence Transformers v3." Hugging Face Blog, 2024. https://huggingface.co/blog/train-sentence-transformers
39. Z. Liu, W. Ping, R. Roy, P. Xu, C. Lee, M. Shoeybi, B. Catanzaro. "ChatQA: Surpassing GPT-4 on Conversational QA and RAG." NVIDIA, NeurIPS 2024. https://arxiv.org/abs/2401.10225
40. P. Xu, W. Ping, X. Wu, et al. "ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities." NVIDIA, 2024. https://arxiv.org/abs/2407.14482
41. G. Izacard, E. Grave. "Distilling Knowledge from Reader to Retriever for Question Answering." Meta / Inria, ICLR 2021. https://arxiv.org/abs/2012.04584
42. B. Jin, H. Zeng, Z. Yue, et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." COLM 2025. https://arxiv.org/abs/2503.09516 (code: https://github.com/PeterGriffinJin/Search-R1)
43. P. Jiang et al. "DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning." 2025. https://arxiv.org/abs/2503.00223 (code: https://github.com/pat-jj/DeepRetrieval)
44. J. Lee, Z. Dai, X. Ren, et al. "Gecko: Versatile Text Embeddings Distilled from Large Language Models." Google DeepMind, 2024. https://arxiv.org/abs/2403.20327
45. "InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems." 2025. https://arxiv.org/abs/2508.13930
46. D. Gwon et al. "Study on LLMs for Promptagator-Style Dense Retriever Training." 2025. https://arxiv.org/abs/2510.02241
47. V. Chen, J. Portes, et al. "Improving Retrieval and RAG with Embedding Model Finetuning." Databricks Blog, 2025. https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning
48. Databricks. "AI Search retrieval quality guide." Databricks documentation, 2025-2026. https://docs.databricks.com/gcp/en/ai-search/retrieval-quality
49. S. Hofstätter, S. Althammer, M. Schröder, M. Sertkan, A. Hanbury. "Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation" (Margin-MSE). TU Wien, 2020. https://arxiv.org/abs/2010.02666
50. S. Hofstätter, S.-C. Lin, J.-H. Yang, J. Lin, A. Hanbury. "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling" (TAS-B). TU Wien / Waterloo, SIGIR 2021. https://arxiv.org/abs/2104.06967
51. R. Pradeep, S. Sharifymoghaddam, J. Lin. "RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!" University of Waterloo, 2023. https://arxiv.org/abs/2312.02724
52. S. Eyuboglu, R. Ehrlich, S. Arora, N. Guha, D. Zinsley, E. Liu, W. Tennien, A. Rudra, J. Zou, A. Mirhoseini, et al. "Cartridges: Lightweight and general-purpose long context representations via self-study." Stanford / HazyResearch, 2025. https://arxiv.org/abs/2506.06266 (code: https://github.com/HazyResearch/cartridges)
53. Y. Yu, W. Ping, Z. Liu, B. Wang, J. You, C. Zhang, M. Shoeybi, B. Catanzaro. "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs." NVIDIA / Georgia Tech, NeurIPS 2024. https://arxiv.org/abs/2407.02485
54. H. Sun, Z. Qiao, J. Guo, X. Fan, Y. Hou, Y. Jiang, P. Xie, Y. Zhang, F. Huang, J. Zhou. "ZeroSearch: Incentivize the Search Capability of LLMs without Searching." Tongyi Lab, Alibaba Group, 2025. https://arxiv.org/abs/2505.04588 (code: https://github.com/Alibaba-nlp/ZeroSearch)
55. W. Liu, X. Ma, W. Sun, Y. Zhu, Y. Li, D. Yin, Z. Dou. "ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability." Renmin University / Baidu, 2025 (ACL 2026). https://arxiv.org/abs/2508.07050 (code: https://github.com/8421BCD/ReasonRank)
