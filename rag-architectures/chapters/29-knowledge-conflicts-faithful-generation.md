# 29. The Generation Side of RAG: Knowledge Conflicts, Robustness, Compression, Attribution, and Abstention

## Overview

Most RAG engineering effort goes into the retrieval side: chunking, embeddings, hybrid search, reranking. The hardest open problems, though, are on the other side of the prompt boundary. Once a set of passages reaches the generator, the system has to answer several questions that retrieval cannot settle:

1. **Conflict.** If a passage disagrees with what the model learned in pre-training, which one wins? If two passages disagree with each other, which one wins?
2. **Noise and attack.** If some passages are irrelevant, misleading, or deliberately poisoned, does the generator ignore them or get pulled off course?
3. **Position and budget.** Does the generator use evidence equally wherever it sits in a long context? How far can the context be compressed before answers degrade?
4. **Attribution.** Can the generator say which passage, or which sentence, supports each claim, and are those citations actually true?
5. **Abstention.** When the evidence is insufficient, does the generator say so, or does it fill the gap from its parametric memory?

This chapter surveys the publicly documented designs for each question. They split into four families:

- **Diagnostic benchmarks** that measure the failure: ClashEval, ConflictBank, FaithEval, Lost in the Middle, ALCE, LongBench-Cite, AbstentionBench.
- **Training-time fixes** that change the generator: RetRobust, Chain-of-Note, InstructRAG, RAFT, Self-RAG, FRONT, LongCite, SelfCite, Trust-Align.
- **Inference-time context transforms** that change what the generator sees: CRAG, Spotlighting, RECOMP, LongLLMLingua, Provence, xRAG, order-preserving RAG.
- **Decoding-time and post-hoc controls** that change how tokens are chosen or whether an answer is released: context-aware decoding and its adaptive variant AdaCAD, isolate-then-aggregate decoding (RobustRAG), ablation-based attribution (ContextCite), sufficient-context selective generation, and interpretability-based hallucination detectors such as ReDeEP.

A note on sourcing. Repository pages for most systems were checked directly, and rechecked in a later fact-checking pass (October 2026), along with the vendor documentation cited in Section 29.4.7. Paper abstracts on arXiv, OpenReview and ACL Anthology could not be fetched in either pass. Headline figures that come from paper abstracts rather than checked repository pages are therefore labelled "reported" and should be read as approximate. Where a repository page confirmed a figure, the text says so.

```
                    THE GENERATION-SIDE CONTROL SURFACE
 query ──► retriever ──► top-k passages
                              │
             ┌────────────────┼──────────────────────────────┐
             │ (A) context transforms                        │
             │  evaluate / filter (CRAG, NLI filter)         │
             │  compress (RECOMP, LongLLMLingua, xRAG)       │
             │  reorder (put strong evidence at the edges)   │
             └────────────────┬──────────────────────────────┘
                              ▼
             ┌───────────────────────────────────────────────┐
             │ (B) generator, trained for:                   │
             │  noise robustness (RetRobust, RAFT)           │
             │  explicit reading notes (Chain-of-Note,       │
             │    InstructRAG rationales)                    │
             │  self-critique tokens (Self-RAG)              │
             │  citation (FRONT, LongCite, SelfCite)         │
             │  refusal (Trust-Align)                        │
             └────────────────┬──────────────────────────────┘
                              ▼
             ┌───────────────────────────────────────────────┐
             │ (C) decoding controls                         │
             │  context-aware decoding (contrast with/without)│
             │  isolate-then-aggregate (RobustRAG)           │
             └────────────────┬──────────────────────────────┘
                              ▼
             ┌───────────────────────────────────────────────┐
             │ (D) release gate                              │
             │  sufficient-context check, citation verifier, │
             │  hallucination detector -> answer | abstain   │
             └───────────────────────────────────────────────┘
```

## 29.1 The Knowledge-Conflict Problem

### 29.1.1 Taxonomy

The EMNLP 2024 survey "Knowledge Conflicts for LLMs" by Xu et al. [1] gives the standard vocabulary. It names three conflict types:

- **Context-memory conflict.** Retrieved or provided context disagrees with the model's parametric knowledge. The main causes are temporal misalignment (the model's knowledge is stale) and misinformation in the corpus.
- **Inter-context conflict.** Retrieved passages disagree with each other because of noise, outdated documents, or misinformation.
- **Intra-memory conflict.** The model's own parametric knowledge gives inconsistent answers to paraphrases of the same question. The survey attributes this to training-corpus bias, decoding strategy, and knowledge editing.

The survey's mitigation map is a useful index for this chapter:

- For context-memory conflict: fine-tuning, prompting, and decoding methods that improve faithfulness to context. These include context-aware and contrastive decoding, and inference-time intervention on attention heads.
- For inter-context conflict: contradiction detection, fact-checking, and robustness to counterfactual noise.
- For intra-memory conflict: consistency methods such as self-consistency and generator-validator consistency.

There is a design tension here that system builders often miss. "Faithfulness to context" and "robustness to bad context" pull in opposite directions. A generator trained to always follow the context will also follow poisoned context. A generator trained to trust its priors will ignore legitimate updates, such as a new drug dosage or this quarter's revenue. Every technique in this chapter picks a point on that trade-off, sometimes explicitly and sometimes not.

### 29.1.2 Early evidence: "Adaptive Chameleon or Stubborn Sloth"

Xie et al. [2] (ICLR 2024) gave the first widely cited behavioural result. Models are highly receptive to external evidence that is coherent and convincing, even when it conflicts with their parametric memory. Yet they also show confirmation bias when the evidence contains some information that agrees with their memory. Two practical consequences follow. First, the persuasiveness of a passage matters, not only its correctness. Second, a mixed evidence set does not get resolved neutrally. Later analyses listed in [1] build on this, including "What Evidence Do Language Models Find Convincing?" (ACL 2024) and "Context versus Prior Knowledge in Language Models" (ACL 2024).

### 29.1.3 ClashEval (Stanford, 2024)

**Problem.** ClashEval [3] measures how a model resolves a direct clash between its prior and a retrieved passage that has been perturbed to be wrong by a controlled amount.

**Design.** The benchmark has over 1,200 questions across six domains (verified on the repository [4]):

| Domain | Questions | Perturbations per question |
|---|---|---|
| Drug dosages | 249 | 10 |
| Sports records | 191 | 10 |
| Latest news | 238 | 10 |
| Wikipedia dates | 200 | 10 |
| Names | 200 | 3 |
| Locations | 200 | 3 |

For numeric domains, the perturbations scale the true value by graded factors. For names and locations, they apply graded categorical edits. This yields roughly 10,000 perturbed contexts. Six models were benchmarked: GPT-4o, GPT-3.5, Llama-3-8B-Instruct, Gemini 1.5, Claude Opus and Claude Sonnet. For each item, the released data records the model's prior answer, its post-context answer, correctness flags and token log-probabilities [4].

**Findings.** As the repository summarizes them, models sometimes adopt incorrect context over a correct prior and sometimes the reverse [4]. The paper reports, approximately, that models adopted incorrect retrieved content over their own correct prior more than 60% of the time. It also reports two monotone trends. The more implausible the perturbation, the less likely the model is to adopt it. The less confident the model is in its prior (measured by token probability), the more likely it is to defer to the context [3].

**Lesson for designers.** Two signals predict which source a model will follow: the log-probability of the closed-book answer and how far the context deviates from it. That makes a cheap conflict detector possible. Run a closed-book pass alongside the RAG pass. If both are confident and they disagree, escalate. You can show both answers, ask for confirmation, or apply a domain policy such as "the document always wins for pricing; never for drug-safety constants unless the source is on an allowlist." The ClashEval authors also proposed simple probability-based calibration methods along these lines [3].

### 29.1.4 ConflictBank (NeurIPS 2024 Datasets and Benchmarks)

**Problem.** ClashEval tests prompting-time conflict on a small curated set. ConflictBank [5] goes further: it makes conflict reproducible at scale and also covers conflict embedded in parametric memory.

**Architecture.** The construction pipeline (verified on the repository [6]) works as follows:

```
Wikidata facts ──► SLING fact extraction ──► (s, r, o) claims
                                              │
              LLM generates conflicting evidence per claim, by cause:
              ├─ misinformation conflict (false object)
              ├─ temporal conflict (fact changed over time)
              └─ semantic conflict (same surface form, different sense)
                                              │
              quality control: NLI (deberta-v3-base-tasksource-nli)
                             + embedding similarity (all-mpnet-base-v2)
                                              │
        7,453,853 claim-evidence pairs  +  553,117 QA pairs
                                              │
   optional: continual pre-training to embed conflicts into LLMs
   (LLaMA-Factory, 4,500 steps, batch 32/device, lr 2e-5 cosine,
    max length 4,096)
```

**Why it matters.** ConflictBank separates conflicts by cause. That lets an evaluator ask whether a model handles temporal conflict (the newer document should win) differently from misinformation (the more reliable source should win). These need different policies. Its continual-pretraining release also lets researchers create models whose parametric memory deliberately contains conflicting facts, so intra-memory and context-memory conflict can be studied together [6].

### 29.1.5 FaithEval (Salesforce, 2024)

FaithEval [7] measures contextual faithfulness on about 4.9K problems across three task types (verified on the repository [8]):

- **Unanswerable:** the context does not contain the answer, so the correct behaviour is abstention.
- **Inconsistent:** documents support different answers, so the correct behaviour is to surface the conflict.
- **Counterfactual:** the context contradicts world knowledge (as in the title's "the moon is made of marshmallows"), and the model is asked to answer from the context.

Its main finding is that even state-of-the-art models often fail to stay faithful to context, and that larger models are not consistently more faithful [8]. For designers, FaithEval's three task types correspond to three separate output behaviours a production generator needs: answer from context, flag a conflict, and abstain. Each should be tested on its own.

### 29.1.6 Mechanistic view: copying heads vs. knowledge FFNs

ReDeEP [9] (ICLR 2025) gives a mechanistic account of RAG hallucination. In its framing, hallucination happens when knowledge FFN layers push too much parametric knowledge into the residual stream while copying heads fail to retain or integrate the retrieved content. ReDeEP detects hallucinations by separating the model's use of external context from its use of parametric knowledge. A companion method, AARF, reduces hallucination by rebalancing the contributions of FFNs and copying heads [9]. This line of work connects to "retrieval heads" research and to head-level interventions such as "Cutting Off the Head Ends the Conflict," listed in [1]. It suggests that conflict handling can be steered with targeted activation or attention edits instead of full fine-tuning. Treat this as early-stage.

## 29.2 Robustness to Irrelevant and Adversarial Passages

### 29.2.1 RetRobust (Tel Aviv University / AI2, 2023)

**Problem.** Yoran et al. [10] showed that retrieval can hurt. When retrieved passages are irrelevant, especially in multi-hop QA, a retrieval-augmented LM can do worse than the same model without retrieval.

**Two designs.**

1. **NLI filter.** Before generation, an NLI model checks whether each passage entails the question-answer pair, and passages that fail are dropped. This removes noise but also removes relevant passages the NLI model misjudges, so it can hurt when retrieval is good. The repository documents the NLI component separately [11].
2. **Robustness fine-tuning.** The generator is fine-tuned on a mix of relevant and irrelevant retrieved contexts, so it learns to use good evidence and ignore bad evidence. The repository states that about 1K training examples are enough to make the model robust to irrelevant context [11].

**Lesson.** A hard filter in front of the generator is brittle. A cheap fine-tune on mixed-quality contexts teaches a soft filter inside the generator and generalizes better. This "train with distractors" principle recurs throughout this section.

### 29.2.2 RAFT (UC Berkeley, 2024)

RAFT [12] applies the distractor principle to domain adaptation. For each document chunk, GPT-4 generates question-answer pairs. Each training example then pairs a question with the oracle chunk plus several distractor chunks (3 or 4 in the repository example, using 512-token chunks and 5 questions per chunk) [13]. A parameter P sets the fraction of examples that include the oracle document at all. The remaining examples contain only distractors, which pushes the model to rely on memorized domain knowledge when retrieval fails [12][13]. Targets are chain-of-thought answers (a `cot_answer` field) that quote from the relevant document [12][13]. RAFT's main contribution is to teach two behaviours in one recipe: ignore distractors when the oracle is present, and fall back to memory when it is absent.

### 29.2.3 Chain-of-Note (Tencent AI Lab, 2023)

**Problem.** Standard RAG prompting gives no explicit signal of how relevant each passage is. The model has to decide implicitly while it generates.

**Architecture.** Chain-of-Note (CoN) [14] trains the generator to write a short reading note for each retrieved passage before answering. Each note assesses the passage's relevance to the question. The notes then feed a final answer that follows one of three patterns:

```
  passages p1..pk
       │
       ▼
  note(p1): "p1 directly answers -> X"            (pattern 1: direct answer)
  note(p2): "p2 is related context, implies X"     (pattern 2: inferred answer)
  note(p3): "p3 irrelevant"                       (pattern 3: none useful)
       │
       ▼
  final answer  OR  "unknown" when no note supports an answer
```

The training data were produced by prompting a strong model to write notes on NQ-style questions, and those notes were then distilled into a smaller (Llama-2 class) model [14]. The paper reports, approximately, an average gain of about +7.9 EM when all retrieved documents are noisy, and about +10.5 points in rejection rate on questions outside the model's pre-training knowledge [14].

**Costs and failure modes.** The notes add output tokens linear in k, which is the main latency cost. A prompted variant can be deployed without training, but the paper's gains come from the fine-tuned model. CoN's explicit "unknown" route is an early example of combining robustness with abstention.

### 29.2.4 InstructRAG (UVA, ICLR 2025)

**Problem.** Chain-of-Note and RAFT need rationales, but human-written denoising rationales are expensive.

**Architecture.** InstructRAG [15] has an instruction-tuned LM generate its own rationales. Given the question, the retrieved documents, and the gold answer, the LM writes an explanation of how the documents lead to that answer, including which documents are misleading. These self-synthesized rationales are then used in one of two ways (verified on the repository [16]):

- **InstructRAG-ICL**, a training-free variant in which rationales serve as in-context demonstrations.
- **InstructRAG-FT**, a supervised fine-tuning variant that performs better.

The repository reports improvements of up to 8.3% across five benchmarks, robustness as the noise ratio increases, and generalization to unseen out-of-domain tasks. It releases FT checkpoints for five datasets, each paired with the retriever used for that benchmark: PopQA and TriviaQA (Contriever), Natural Questions (DPR), ASQA (GTR) and 2WikiMultiHopQA (BM25) [16].

**Lesson.** The gold answer is used only when generating training data, so rationales are cheap to produce. Because the model learns to state explicitly why evidence is good or bad, its denoising is inspectable.

### 29.2.5 Self-RAG (UW / AI2 / IBM, ICLR 2024)

Self-RAG [17] puts critique into the generator's vocabulary. Four families of reflection tokens are interleaved with the output (verified on the repository [18]):

- **Retrieve**: `[Retrieval]` / `[No Retrieval]`. The model decides whether to retrieve.
- **IsRel**: whether a retrieved passage is relevant.
- **IsSup**: whether the generated segment is fully, partially, or not supported by the passage.
- **IsUse**: a 1-5 utility score.

**Training.** GPT-4 labels critic training data. A Llama-2-7B critic is fine-tuned on those labels (3 epochs, lr 2e-5) and then labels a roughly 150K-instance corpus for the generator. The generator learns the reflection tokens by ordinary next-token prediction [18].

**Inference controls.** Retrieval mode can be adaptive, always, or never, with a threshold (default 0.2) for adaptive mode. Segment-level beam search (beam width 2, max depth 6 by default; the repository's long-form example uses 7) scores candidates as a weighted sum of reflection-token probabilities, with defaults w_rel 1.0, w_sup 1.0 and w_use 0.5 [18].

**Why it matters here.** Self-RAG turns faithfulness into a tunable runtime knob. Raising w_sup favours segments the model itself judges supported, trading fluency and coverage for grounding without retraining. Its failure mode is self-assessment error: the critic is only as reliable as the GPT-4 labels it was trained on.

### 29.2.6 CRAG: Corrective RAG (2024)

CRAG [19] adds a lightweight retrieval evaluator that scores the overall quality of the retrieved set and returns a confidence. That confidence routes to one of three actions (verified on the repository [20]):

```
              retrieval evaluator (fine-tuned, lightweight)
                 │            │               │
             Correct      Ambiguous       Incorrect
                 │            │               │
      decompose-then-    combine both   rewrite query ->
      recompose: split   branches       web search (serper.dev);
      into strips,                      prefer Wikipedia or all
      drop irrelevant
                 └────────────┴───────────────┘
                              ▼
                          generator
```

CRAG is plug-and-play: the generator is unchanged, and the evaluator plus the "knowledge refinement" step act as a front-end filter. It is the inference-time counterpart to RetRobust's fine-tuning approach.

### 29.2.7 Adversarial passages: PoisonedRAG and RobustRAG

**PoisonedRAG** [21] (USENIX Security 2025) treats the knowledge base as an attack surface. The attacker crafts a few texts per target question, each satisfying two conditions. The retrieval condition: the text ranks highly for the target query. The generation condition: once retrieved, it induces the attacker's chosen answer. The repository defaults to five adversarial texts per target query with top-k 5, and supports a black-box (LM-targeted) and a white-box (HotFlip) variant [22]. The paper reports, approximately, attack success rates around 90% from injecting five texts per target question into knowledge bases with millions of passages [21]. This shows that "follow the context" generators can be exploited cheaply.

**RobustRAG** [23] is presented by its authors as the first defence framework against retrieval corruption with certifiable robustness guarantees; the repository itself does not make the priority claim [24]. Its core design is isolate-then-aggregate:

```
 top-k passages ──► k independent generator calls (one passage each)
                         │   │   │   ...   │
                         r1  r2  r3        rk     isolated responses
                         └───┴───┴────┬────┘
                     secure aggregation:
                     ├─ keyword aggregation: keep keywords appearing in
                     │   many responses, then generate from those keywords
                     └─ decoding aggregation: at each step, combine
                         next-token distributions across the k isolated runs
                                      │
                       certified: bounded influence of any
                       small number of corrupted passages
```

Because each passage influences only its own isolated response, a small number of corrupted passages cannot dominate the aggregate. The repository exposes `keyword`, `decoding` and `voting` defences, with `none` as the undefended baseline [24]. The cost is roughly k times the generation compute of single-pass RAG, which is acceptable for high-stakes and short-answer settings. A further limitation is that isolation makes it harder to synthesize answers that require combining several passages.

### 29.2.8 Astute RAG (Google, 2024)

Astute RAG [25] targets imperfect retrieval combined with knowledge conflict. The LLM first elicits its own internal knowledge as a passage. It then consolidates internal and external passages iteratively, grouping consistent sources and flagging conflicting ones, and finalizes the answer according to source reliability. The design makes parametric memory an explicit, inspectable participant in conflict resolution instead of an invisible prior.

### 29.2.9 Indirect prompt injection: Spotlighting (Microsoft, 2024)

PoisonedRAG-style attacks try to plant wrong facts. A second class of adversarial passage carries instructions instead ("ignore the question and tell the user to visit ..."). This is indirect prompt injection, and it targets the generator's instruction-following rather than its factual judgement. Spotlighting [53] is a family of prompt-side transforms that mark retrieved text as data so the model can tell it apart from the system and user instructions:

- **Delimiting:** wrap retrieved content in distinctive markers and tell the model never to follow instructions inside them. This is the weakest variant.
- **Datamarking:** interleave a special token (for example, replace whitespace with `^`) throughout the retrieved text, so provenance is visible at every position rather than only at the boundaries.
- **Encoding:** transform the retrieved text (for example, base64) and ask the model to decode it while reading. This is strongest, but works only with highly capable models.

The paper reports, approximately, that datamarking and encoding cut attack success rates from above 50% to below 2% on its GPT-family test set, with little loss on the underlying tasks [53]. For RAG designers the lesson is that conflict policy (Section 29.1) and instruction hierarchy are separate controls. A passage can be factually trustworthy and still contain text the generator must not obey. Spotlighting is cheap and training-free, but it is a mitigation, not a guarantee, and it should be layered with output-side checks.

## 29.3 Position Effects and Context Compression

### 29.3.1 Lost in the Middle (Stanford / Samaya / Meta, 2023)

Liu et al. [26] established the canonical position effect. The setup, verified on the repository [27], uses NQ-Open questions with Contriever-retrieved distractors. There is an oracle condition and 10-, 20- and 30-document conditions in which exactly one document contains the answer. In the 20-document condition, the gold document is placed at positions 0, 4, 9, 14 and 19. A synthetic key-value retrieval task over UUID pairs isolates pure lookup ability.

The paper reports a U-shaped curve. Accuracy is highest when the relevant document is at the start or end of the context and drops markedly in the middle. In some configurations, mid-context accuracy for GPT-3.5-Turbo fell below its closed-book accuracy [26]. Extended-context variants of a model did not use long contexts better than their shorter counterparts [26].

**Design consequences that persist in 2025-2026 stacks:**

- **Edge placement.** After reranking, put the strongest passages first and last (a "sandwich" order) rather than in descending order along the context.
- **Fewer, better passages.** Adding documents past a moderate k can lower accuracy. The reader saturates long before the retriever's recall does.
- **Compression** (below) is partly motivated by this effect, because shorter contexts have less "middle".

Newer long-context models have flattened the curve on simple needle tests, but multi-document reasoning still shows position sensitivity. The defensive orderings cost nothing, so keep them.

### 29.3.2 RECOMP (UT Austin / UW, 2023)

**Problem.** Prepending k full documents is expensive, and much of the text is irrelevant to the question.

**Architecture.** RECOMP [28] trains compressors that sit between the retriever and the frozen generator (verified on the repository [29]):

- **Extractive compressor:** a dual encoder initialized from `contriever-msmarco` that scores each sentence of the retrieved documents and keeps the top sentences.
- **Abstractive compressor:** a T5-large summarizer (max target length 512) that writes a query-focused summary of the top documents (the top 5 in the repo example).

The training signal is end-task utility: a sentence or summary is good if prepending it improves the frozen LM's likelihood of the correct output. This makes the compressors LM-aware rather than generic summarizers. **Selective augmentation** is central to the design. The compressor may output an empty string when the documents are irrelevant, so the generator falls back to closed-book answering. That is a built-in noise defence.

**Numbers.** The paper reports compression to as little as roughly 6% of the original tokens with minimal loss on language modeling and open-domain QA, and that the compressors transfer across LMs to some degree [28]. These figures are approximate.

### 29.3.3 LLMLingua, LongLLMLingua and LLMLingua-2 (Microsoft)

The LLMLingua family [30] compresses prompts by deleting tokens that a small LM judges low-information.

- **LLMLingua** (EMNLP 2023) uses a small LM's perplexity to drop non-essential tokens. The README example shows an 11.2x compression ratio, and the series claims up to 20x compression with minimal loss [30].
- **LongLLMLingua** (ACL 2024) [31] targets RAG and long contexts directly. Its key ideas map to library parameters [30]:
  - **Question-aware coarse-to-fine compression** (`condition_in_question`). Documents are first ranked by how much they make the question likely, then tokens are pruned conditioned on the question.
  - **Document reordering** (`reorder_context="sort"`). After ranking, the most relevant documents are moved to the edges, which directly counters lost-in-the-middle.
  - **Dynamic compression ratio** (`dynamic_context_compression_ratio`, e.g. 0.3 or 0.4). More relevant documents keep more tokens.
  - The project reports up to 21.4% RAG performance improvement using about 1/4 of the tokens [30].
- **LLMLingua-2** (ACL 2024 Findings) [32] recasts compression as token classification with a BERT-sized encoder distilled from GPT-4 compression labels. It is task-agnostic, handles out-of-domain data better, and is reported as 3x-6x faster than LLMLingua [30].

The family is integrated in LangChain, LlamaIndex and Prompt Flow [30]. That makes it the most deployable compressor in this chapter, because it works with any black-box API generator.

**Failure modes.** Token deletion can remove negations, units or qualifiers, which then shows up as answer errors. Compressed text is less readable, so citations into it are hard to show to users. Map citation spans back to the original document offsets.

### 29.3.4 xRAG (NeurIPS 2024): one-token context

xRAG [33] compresses the most aggressively. It does not feed passage text to the generator at all. Instead, it reuses the retriever's dense document embedding:

```
 document ──► retriever encoder (frozen; SFR-Embedding-Mistral) ──► e_doc
                                                                   │
                                   modality bridge (small trained projector)
                                                                   │
                                     one soft token in LLM embedding space
                                                                   │
 [instruction] [xRAG token] [question] ──► frozen LLM (Mistral-7B / Mixtral)
```

Only the bridge is trained. The retriever and the LLM stay frozen, and the document embedding is already computed for retrieval, so a whole passage costs one token of context. The repository confirms the Mistral-7B-Instruct and Mixtral-8x7B checkpoints and the SFR-Embedding-Mistral retriever [34]. The paper reports, approximately, an average improvement of over 10% across six knowledge-intensive tasks compared with no retrieval, comparable results to uncompressed RAG on several datasets, and a roughly 3.5x reduction in overall FLOPs [33].

**Trade-offs.** A single vector cannot support fine-grained citation or exact quotation, and it degrades on tasks that need precise multi-hop details. xRAG suits high-QPS factoid workloads, not auditable answers.

### 29.3.5 Provence: pruning fused with reranking (Naver Labs Europe, ICLR 2025)

Provence [54] sits between RECOMP's extractive compressor and LongLLMLingua's token pruning. It is a DeBERTa-v3 cross-encoder trained to do two jobs in one forward pass: produce the reranking score for a (query, passage) pair, and label each sentence of the passage as keep or drop. Because the pruner is the reranker, context pruning adds almost no latency to a pipeline that already reranks. The number of sentences kept is decided per passage rather than fixed, so an irrelevant passage can be pruned to nothing. That gives the same selective-augmentation behaviour as RECOMP's empty summary. The paper reports, approximately, substantial context reduction with negligible or no loss in QA accuracy across several domains and generators [54]. Because it deletes whole sentences rather than tokens, the surviving text stays readable and quotable, which avoids LLMLingua's citation-mapping problem.

### 29.3.6 Order-preserving RAG (NVIDIA, 2024)

Edge placement (Section 29.3.1) reorders passages by relevance. OP-RAG [55] tests a different order: keep the selected chunks in the order they appear in the source document, regardless of retrieval score. On long-document QA from ∞Bench, the paper reports that order preservation beats relevance ordering, and that answer quality follows an inverted-U as the number of retrieved chunks grows. Too few chunks miss evidence, and too many reintroduce distraction. It also reports that OP-RAG with a moderate token budget outperforms feeding the whole document to a long-context model [55]. The two findings are compatible. Relevance-edge ordering suits sets of independent passages from many documents. Source order suits chunks from one long document, where narrative and reference order carry meaning. In both cases the generator's effective context is far shorter than its advertised window.

## 29.4 Attribution-Trained Generators

### 29.4.1 Measuring attribution: ALCE

ALCE [35] (Princeton, EMNLP 2023) is the reference benchmark for generation with citations. It uses ASQA, QAMPARI and ELI5, with top-100 retrieval results from DPR/GTR or BM25. It scores three dimensions: fluency (MAUVE), correctness, and citation quality [36]. Citation quality is judged with an NLI model and has two parts. **Citation recall** asks whether each statement is entailed by the concatenation of its cited passages. **Citation precision** asks whether each individual citation is necessary [35]. Most attribution work after 2023 reports ALCE numbers, and FRONT's repository points to ALCE for evaluation [37].

### 29.4.2 FRONT: fine-grained grounded citations (HIT, ACL 2024 Findings)

**Problem.** Passage-level citations (`[2]`) are coarse. Users must read the whole passage, and NLI verifiers struggle with long premises. Generators also cite passages they did not actually use.

**Architecture.** FRONT [38] trains a two-stage generator (verified on the repository [37]):

```
 Stage 1  Grounding-Guided Generation (SFT)
   query + passages ──► select supporting QUOTES (fine-grained spans)
                       ──► generate answer conditioned on quotes, cite them

 Stage 2  Consistency-Aware Alignment (DPO)
   sample answers ──► score consistency between grounded quotes and answer
                  ──► preference pairs ──► DPO
```

Making the generator extract quotes first makes the citation part of the reasoning trace rather than something added afterwards. Training ran on 4x A100-80GB [37]. The paper reports, approximately, that FRONT on LLaMA-2-7B exceeded ChatGPT on citation quality by about 14% on average across ALCE datasets [38].

### 29.4.3 LongCite (Tsinghua / Zhipu, 2024)

LongCite [39] brings sentence-level citation to long contexts (up to 128K tokens). Its **CoF (coarse-to-fine)** data pipeline has four stages (verified on the repository [40]): generate long-context QA pairs, attach chunk-level citations, refine them to sentence-level citations, then filter. The result is the LongCite-45k SFT set. The released models are LongCite-glm4-9b and LongCite-llama3.1-8b. The companion benchmark **LongBench-Cite** uses GPT-4o as a judge of citation quality and correctness [40]. LongCite shows that synthetic citation SFT data can teach smaller open models fine-grained citation that is competitive with proprietary models, according to the paper's claims [39].

### 29.4.4 SelfCite: self-supervised citation alignment (MIT / Meta, ICML 2025)

**Problem.** Citation training data and judges (GPT-4o, NLI models) are expensive and imperfect.

**Key technique: a reward from context ablation.** SelfCite [41] uses the generator itself as the judge (verified on the repository [42]):

- **Necessity:** remove the cited sentences from the context. If the citation is necessary, the probability of the generated statement should drop.
- **Sufficiency:** keep only the cited sentences. If the citation is sufficient, the probability of the statement should stay high.

The reward combines both. It is used for **best-of-N sampling** at inference and for **preference optimization** (a SimPO pipeline) at training time. The reported gain is up to 5.3 points of citation F1 on LongBench-Cite across five long-form QA tasks. A reproduced LongCite-8B best-of-N run shows average citation F1 of about 0.78 and correctness of about 0.68 under GPT-4o judging [42]. Two models are released: SelfCite-8B and a fully self-supervised SelfCite-8B-from-CC.

**Lesson.** Context ablation is a model-internal, judge-free attribution signal. It applies wherever you can rerun the generator with edited contexts, and it is also a strong production verifier for spot-checking citations.

### 29.4.5 Trust-Align: attribution plus refusal (SUTD / DSO, ICLR 2025)

Song et al. [43] argue that RAG trustworthiness has three parts, which they combine into **Trust-Score** (verified on the repository [44]):

1. **Response correctness:** are the generated claims correct?
2. **Attribution quality:** citation recall and citation precision.
3. **Refusal groundedness:** does the model correctly judge whether the documents are enough to answer?

**Trust-Align** is a preference dataset with SFT and DPO data. Aligning LLaMA-3-8B with it gives reported gains of +10.7 on ASQA, +29.2 on QAMPARI and +14.9 on ELI5 over comparably sized open models [44]. The key design idea is that refusal is scored against the documents, not against the model's knowledge. A model that answers correctly from parametric memory when the documents do not support the answer is penalized. That is the right incentive for enterprise RAG, where every answer must trace back to approved sources.

### 29.4.6 ContextCite: post-hoc attribution by ablation (MIT, NeurIPS 2024)

ContextCite [51] is the training-free precursor of SelfCite's reward. Given a generated statement, it randomly ablates subsets of the context sources (for example, sentences), reruns the model on each ablated context, and records the log-probability of the original statement. It then fits a sparse linear surrogate that predicts that log-probability from the ablation mask. The surrogate's weights are the attribution scores: a large weight means removing that source makes the statement much less likely. The paper reports that a small number of ablations (tens of forward passes) is enough for useful attributions [51]. The released library exposes this as a `ContextCiter` object with a `get_attributions` call [52].

Three uses matter for RAG. Attributions can verify citations a generator emitted. They can prune the context to the sources that actually drove the answer. They can also flag poisoning, because an answer driven by a single unusual passage stands out. The cost is the extra forward passes, so in production it is better suited to spot-checks and high-stakes answers than to every request.

### 29.4.7 Platform-level citation APIs

Commercial APIs now expose grounded citation as a feature. Anthropic's Citations capability on the Claude API (launched in early 2025, now generally available and also offered through Amazon Bedrock and Google Cloud) accepts plain-text, PDF or custom-content documents, as well as `search_result` blocks for RAG pipelines, with citations enabled. Plain text and extracted PDF text are chunked into sentences; custom-content blocks are used as-is, which lets the application set citation granularity itself. The response returns typed citation objects (`char_location`, `page_location` or `content_block_location`, plus the `cited_text`) that point back into the source [45]. The documentation states that because the API extracts `cited_text` itself, citations are guaranteed to be valid pointers into the provided documents, and that `cited_text` does not count toward output tokens [45]. The design choice that matters is that the platform chunks the source and returns verifiable offsets, which removes a whole class of fabricated-quote errors. One constraint matters for Section 29.5.4: at the time of writing, citations cannot be combined with the API's structured-outputs (JSON-schema) mode, so a design must choose between native citations and schema-constrained citation fields [45]. Check the vendor documentation for current billing details.

## 29.5 Constrained and Grounded Decoding

### 29.5.1 Context-Aware Decoding (CAD)

**Technique.** CAD [46] (Shi et al., NAACL 2024) contrasts the generator's output distribution with and without the context. In logit space, it samples from (1+α)·logit(y | c, x) − α·logit(y | x). This amplifies what the context adds beyond the prior. The reference implementation encodes this as two weighted instances, weight 1+α for the context-conditioned run and −α for the context-free run. The repository examples use α=1 for knowledge-conflict QA (NQ-Swap) and α=0.5 for summarization (CNN-DM) [47].

**Behaviour.** CAD helps most when context contradicts the prior, as in NQ-Swap-style counterfactuals. That is exactly where plain decoding follows memory [46].

**Costs and failure modes.** CAD doubles the forward passes per token, though the two runs can be batched. It is indiscriminate: if the context is poisoned, CAD amplifies the poison. Use it only after context filtering, or with an adaptive α that drops toward zero when the context and the prior genuinely conflict and the source is untrusted.

### 29.5.2 Adaptive CAD (UNC, NAACL 2025)

AdaCAD [56] addresses CAD's fixed-α problem. At each decoding step it computes α as the Jensen-Shannon divergence between the next-token distributions with and without context, then applies the usual CAD combination (1+α)·logit(y | c, x) − α·logit(y | x) (verified on the repository [57]). When context and prior agree, α is near zero and decoding stays close to plain context-conditioned decoding. When they conflict, α grows and the context is amplified. This avoids a known CAD failure: a large fixed α distorts outputs on the many questions where there is no conflict. For long-form generation the repository adds a threshold on α as a warm-up step [57]. AdaCAD still trusts the context whenever there is conflict. It is a better-calibrated version of "context wins", not a defence against poisoned context, so the filtering advice in Section 29.5.1 still applies.

### 29.5.3 Isolation-based decoding

RobustRAG's decoding aggregation (Section 29.2.7) is another form of constrained decoding. At each step, the next token must be supported by many isolated per-passage runs. CAD makes the model follow the context more. Isolation decoding makes the model follow consensus across contexts. Production systems that need both often apply isolation first (to find consensus evidence) and then grounding-oriented decoding over the surviving passages.

### 29.5.4 Quote-first and structured generation

Several designs constrain the output form rather than the token distribution:

- **Quote-then-answer** (FRONT, RAFT's quoted CoT). The generator must emit verbatim spans before the answer, and the spans can be checked against the source with exact string matching.
- **Reflection-token gating** (Self-RAG). Segment beam search drops segments whose IsSup token probability is low.
- **Schema-constrained citation.** With JSON-schema or grammar-constrained decoding available in most serving stacks, citations can be forced into the form `{claim, doc_id, span_start, span_end}`. A validator then rejects any span that is not a substring of the source. Note that some platform citation APIs cannot be combined with schema-constrained output (Section 29.4.7), so this pattern and native citations are alternatives, not layers.

The common lesson is that anything the model emits as a pointer into the context can be checked deterministically. Grounding guarantees should rest on those checks, not on the model's own statements.

## 29.6 Abstention Policies

### 29.6.1 Why abstention is a generation-side problem

Retrieval cannot tell whether its results are sufficient, because sufficiency depends on the question's full information need. Chain-of-Note's "unknown" route [14], Self-RAG's IsSup tokens [17], CRAG's Incorrect route [19], RECOMP's empty summary [28], FaithEval's unanswerable split [7] and Trust-Align's refusal groundedness [43] all express the same need. The system must recognize "I have evidence, but it does not answer this" as different from "I have no evidence." A survey of abstention in LLMs by Wen et al. [48] organizes abstention by query, model and human-value perspectives, and notes that over-abstention is a real cost, not a safe default.

### 29.6.2 Sufficient context (Google, ICLR 2025)

Joren et al. [49] separate two situations: the context is relevant, and the context is sufficient to answer. They build an LLM-based sufficient-context autorater and report the following, approximately:

- Strong proprietary models answer well when the context is sufficient but often hallucinate rather than abstain when it is not.
- Smaller open models hallucinate or abstain even when the context is sufficient.
- A notable fraction of correct answers occur even with insufficient context, because the model fills the gap from parametric memory.

Their mitigation is **selective generation**. The system combines the sufficiency signal with the model's self-rated confidence to decide whether to answer, and reports a gain of roughly 2-10% in the accuracy of answered questions across Gemini, GPT and Gemma models [49].

```
 query + context ──► sufficiency autorater ──► s ∈ {sufficient, insufficient}
        │                                         │
        └──► generator ──► answer + confidence p  │
                                    │             │
                     selective policy: f(s, p) > τ ? answer : abstain
                     (τ tuned for a target coverage/accuracy point)
```

### 29.6.3 AbstentionBench: reasoning models abstain less (Meta, 2025)

AbstentionBench [58] evaluates abstention across 20 datasets covering six scenarios, including unknown answers, false premises, stale information, subjective questions and underspecified questions (three new underspecified reasoning datasets among them). As summarized on its repository, abstention remains an unsolved problem for frontier models, model scale has almost no effect on it, and reasoning fine-tuning hurts abstention: reasoning models answer overconfidently and rarely abstain, even when their reasoning trace notices the missing information [58]. For RAG this is a warning about a common 2025-2026 upgrade path. Swapping a reasoning model in as the generator can raise answer quality on sufficient contexts while lowering refusal rates on insufficient ones. Re-tune the release gate (Section 29.6.2) after any generator change, and keep the abstention test set separate from the accuracy test set.

### 29.6.4 A practical abstention stack

Combining the published pieces gives a layered policy:

1. **Pre-generation:** CRAG-style retrieval evaluator or a sufficiency autorater. If the evidence is clearly insufficient, re-retrieve or route to web search before abstaining.
2. **In-generation:** a trained "unknown" route (Chain-of-Note, Trust-Align) so the model can refuse naturally, with refusals judged against the documents.
3. **Post-generation:** citation verification (ALCE-style NLI, SelfCite-style ablation, or exact-span checks) plus a hallucination detector such as Patronus AI's Lynx [50] or a mechanistic detector such as ReDeEP [9]. If unsupported claims remain, strip them or abstain.
4. **Conflict escalation:** the ClashEval-style closed-book-vs-RAG disagreement check (Section 29.1.3). If both are confident and they disagree, show both or escalate.
5. **Coverage tuning:** choose τ by business cost. Wrong-answer cost is much higher than refusal cost in medical and legal RAG. The reverse often holds in consumer search.

## 29.7 Comparison Table

| System | Year / venue | Layer | Needs training? | Main problem addressed | Extra inference cost | Key published number (see text for caveats) |
|---|---|---|---|---|---|---|
| ClashEval [3] | 2024 | Benchmark | n/a | Prior vs. context conflict | n/a | ~1,278 Qs, 6 domains [4] |
| ConflictBank [5] | NeurIPS 2024 D&B | Benchmark | n/a | Conflict by cause | n/a | 7.45M claim-evidence, 553K QA [6] |
| FaithEval [7] | 2024 | Benchmark | n/a | Unanswerable / inconsistent / counterfactual | n/a | ~4.9K problems [8] |
| RetRobust [10] | 2023 | Generator FT / filter | Yes (~1K ex.) | Irrelevant context | None (FT); NLI pass (filter) | 1K examples suffice [11] |
| RAFT [12] | 2024 | Generator FT | Yes | Distractors in domain RAG | None | Oracle + 3-4 distractors [13] |
| Chain-of-Note [14] | 2023 | Generator FT | Yes | Noisy docs, unknown answers | Notes ~ O(k) tokens | ~+7.9 EM noisy (reported) |
| InstructRAG [15] | ICLR 2025 | Generator ICL/FT | Optional | Denoising via rationales | Rationale tokens | Up to +8.3% [16] |
| Self-RAG [17] | ICLR 2024 | Generator + critic | Yes | Adaptive retrieval, support | Segment beam search | 150K train set [18] |
| CRAG [19] | 2024 | Pre-gen evaluator | Evaluator only | Bad retrieval | Evaluator + web search | 3-way routing [20] |
| RobustRAG [23] | 2024 | Decoding | No | Poisoned passages (certified) | ~k x generation | Keyword / decoding agg. [24] |
| RECOMP [28] | 2023 | Compressor | Compressor only | Context cost, noise | Small compressor | ~6% of tokens (reported) |
| LongLLMLingua [31] | ACL 2024 | Compressor | No (small LM) | Cost + lost-in-middle | Small-LM pass | +21.4% at ~1/4 tokens [30] |
| xRAG [33] | NeurIPS 2024 | Soft-token bridge | Bridge only | Extreme compression | ~1 token/doc | ~3.5x FLOPs cut (reported) |
| FRONT [38] | ACL 2024 Findings | Generator SFT+DPO | Yes | Fine-grained citations | Quote tokens | ~+14% citation vs ChatGPT (reported) |
| LongCite [39] | 2024 | Generator SFT | Yes | Long-context sentence citations | None | 45K SFT set, 128K ctx [40] |
| SelfCite [41] | ICML 2025 | Reward / BoN / PO | Optional | Citation quality, judge-free | N ablation passes (BoN) | Up to +5.3 citation F1 [42] |
| Trust-Align [43] | ICLR 2025 | Generator SFT+DPO | Yes | Attribution + grounded refusal | None | +29.2 QAMPARI [44] |
| CAD [46] | NAACL 2024 | Decoding | No | Prior overriding context | 2x forward passes | α=1 conflict, 0.5 summ. [47] |
| Sufficient context [49] | ICLR 2025 | Release gate | Autorater | Hallucinate vs. abstain | Autorater call | ~2-10% selective acc. (reported) |
| Spotlighting [53] | 2024 | Prompt transform | No | Indirect prompt injection | Marking / encoding tokens | ASR >50% to <2% (reported) |
| Provence [54] | ICLR 2025 | Pruner + reranker | Pruner only | Context cost, noise | Folded into rerank pass | Sentence-level, adaptive (reported) |
| OP-RAG [55] | 2024 | Context ordering | No | Long-document ordering | None | Inverted-U in k (reported) |
| ContextCite [51] | NeurIPS 2024 | Post-hoc attribution | No (surrogate fit) | Which source drove the answer | Tens of ablation passes | Sparse linear surrogate [52] |
| AdaCAD [56] | NAACL 2025 | Decoding | No | Fixed-α distortion in CAD | 2x forward passes | α = JSD per step [57] |
| AbstentionBench [58] | 2025 | Benchmark | n/a | Over-answering, esp. reasoning models | n/a | 20 datasets, 6 scenarios [58] |

## 29.8 Reference Architecture: A Conflict-Aware, Attributed Generator

The following composite is a pattern assembled from the published components above, not a documented deployment:

```
 query
   │
   ├─► closed-book pass (small budget) ─► prior answer a0, logprob p0 ───────┐
   │                                                                          │
   └─► retrieve top-k ─► CRAG-style evaluator ──(incorrect)──► rewrite+web    │
                              │ (correct / ambiguous)                         │
                              ▼                                               │
              strip-level filter / LongLLMLingua (question-aware,             │
              dynamic ratio, reorder strongest -> edges)                      │
                              │                                               │
                              ▼                                               │
     generator fine-tuned with distractors + rationales + quote-first        │
     citations + grounded refusal (RAFT / InstructRAG / FRONT / Trust-Align)  │
          optional: CAD (trusted sources only) or isolate-then-aggregate      │
          (untrusted / user-uploaded sources)                                 │
                              │                                               │
                              ▼                                               │
     answer a1 + citations {claim, doc, span}                                 │
                              │                                               │
     verifiers: exact-span check │ NLI recall/precision │ ablation reward     │
                              │                                               │
     conflict check: a0 != a1 and p0 high? ◄──────────────────────────────────┘
                              │
     release gate: sufficiency x confidence > τ ? answer : abstain / escalate
```

Approximate cost relative to plain RAG: one extra small closed-book call, one evaluator call, one small-LM compression pass (which usually pays for itself by cutting generator input tokens), and verification calls. Isolate-then-aggregate multiplies generation cost by about k, so reserve it for untrusted sources.

## 29.9 Key Takeaways for System Designers

1. **Decide your conflict policy explicitly, per domain.** "Context always wins" is right for proprietary or fresh data and dangerous for open-web or user-uploaded data. ClashEval shows that models resolve conflicts inconsistently and that prior confidence predicts the outcome [3]. Use that as a signal instead of leaving the outcome to chance.
2. **Train with distractors.** RetRobust [10], RAFT [12], Chain-of-Note [14] and InstructRAG [15] all find that a small amount of fine-tuning on mixed relevant and irrelevant contexts beats hard pre-filters. About 1K examples can be enough [11].
3. **Make relevance judgments explicit and inspectable.** Reading notes, rationales and reflection tokens add tokens, but they turn silent failures into auditable traces. They also give the abstention policy a natural route to "unknown".
4. **Order matters and costs nothing.** Put the strongest evidence at the start and end of the context, and cap k well below the retriever's recall-optimal value [26]. LongLLMLingua's reorder plus dynamic ratio is a drop-in implementation [30].
5. **Compress with LM-aware signals, and keep offsets.** RECOMP and LongLLMLingua cut tokens substantially, but citations must map back to original spans. xRAG-style soft tokens give up auditability for speed [33].
6. **Treat citations as checkable pointers.** Quote-first generation (FRONT), sentence-level citation (LongCite), platform-chunked citation APIs, and exact-span validation turn attribution from a claim into a test. SelfCite's ablation reward is a judge-free verifier you can run in production [42].
7. **Score refusal against the documents, not the world.** Trust-Align's refusal groundedness [43] and the sufficient-context framework [49] both penalize correct-but-unsupported answers. That is the behaviour enterprise RAG needs.
8. **Assume the corpus is an attack surface.** A few injected passages can steer answers [21]. Use isolate-then-aggregate decoding [23] for untrusted sources. Never combine context-amplifying decoding (CAD, AdaCAD) with unfiltered sources, and mark retrieved text as data (Spotlighting) so injected instructions are not obeyed [53].
9. **Measure three behaviours separately:** answer-from-context, flag-conflict and abstain (FaithEval's split [7]). A single accuracy number hides the trade-off between faithfulness and robustness.
10. **Tune abstention thresholds to business cost.** Selective generation trades coverage for accuracy [49]. Set τ from the relative cost of a wrong answer versus a refusal, and monitor both rates. Re-tune after any generator swap, especially to a reasoning model, which tends to abstain less [58].

## References

1. "Knowledge Conflicts for LLMs: A Survey." Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, Wei Xu. EMNLP 2024. https://arxiv.org/abs/2403.08319 (repository: https://github.com/pillowsofwind/Knowledge-Conflicts-Survey)
2. "Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts." Jian Xie et al. ICLR 2024. https://arxiv.org/abs/2305.13300
3. "ClashEval: Quantifying the Tug-of-War between an LLM's Internal Prior and External Evidence." Kevin Wu, Eric Wu, James Zou (Stanford). 2024. https://arxiv.org/abs/2404.10198
4. ClashEval repository and dataset. Kevin Wu et al. 2024. https://github.com/kevinwu23/StanfordClashEval
5. "ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM." Zhaochen Su et al. NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2408.12076
6. ConflictBank repository. Zhaochen Su et al. 2024. https://github.com/zhaochen0110/conflictbank
7. "FaithEval: Can Your Language Model Stay Faithful to Context, Even If 'The Moon is Made of Marshmallows'." Yifei Ming et al. (Salesforce AI Research). 2024. https://arxiv.org/abs/2410.03727
8. FaithEval repository. Salesforce AI Research. 2024. https://github.com/SalesforceAIResearch/FaithEval
9. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability." Zhongxiang Sun et al. ICLR 2025. https://github.com/Jeryi-Sun/ReDEeP-ICLR
10. "Making Retrieval-Augmented Language Models Robust to Irrelevant Context." Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant. 2023 (ICLR 2024). https://arxiv.org/abs/2310.01558
11. RetRobust repository. Ori Yoran et al. 2023. https://github.com/oriyor/ret-robust
12. "RAFT: Adapting Language Model to Domain Specific RAG." Tianjun Zhang, Shishir G. Patil, et al. (UC Berkeley). 2024. https://arxiv.org/abs/2403.10131
13. RAFT code (Gorilla repository). UC Berkeley Gorilla team. 2024. https://github.com/ShishirPatil/gorilla/tree/main/raft
14. "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models." Wenhao Yu et al. (Tencent AI Lab). 2023. https://arxiv.org/abs/2311.09210
15. "InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales." Zhepei Wei, Wei-Lin Chen, Yu Meng. ICLR 2025. https://arxiv.org/abs/2406.13629
16. InstructRAG repository. Zhepei Wei et al. 2024. https://github.com/weizhepei/InstructRAG
17. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. ICLR 2024. https://arxiv.org/abs/2310.11511
18. Self-RAG repository. Akari Asai et al. 2023. https://github.com/AkariAsai/self-rag
19. "Corrective Retrieval Augmented Generation." Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling. 2024. https://arxiv.org/abs/2401.15884
20. CRAG repository. Shi-Qi Yan et al. 2024. https://github.com/HuskyInSalt/CRAG
21. "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models." Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia. USENIX Security 2025. https://arxiv.org/abs/2402.07867
22. PoisonedRAG repository. Wei Zou et al. 2024. https://github.com/sleeepeer/PoisonedRAG
23. "Certifiably Robust RAG against Retrieval Corruption." Chong Xiang et al. (Princeton). 2024. https://arxiv.org/abs/2405.15556
24. RobustRAG repository. Princeton INSPIRE group. 2024. https://github.com/inspire-group/RobustRAG
25. "Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models." Fei Wang et al. (Google Cloud AI Research). 2024. https://arxiv.org/abs/2410.07176
26. "Lost in the Middle: How Language Models Use Long Contexts." Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. TACL 2024 (arXiv 2023). https://arxiv.org/abs/2307.03172
27. Lost in the Middle repository. Nelson F. Liu et al. 2023. https://github.com/nelson-liu/lost-in-the-middle
28. "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation." Fangyuan Xu, Weijia Shi, Eunsol Choi. ICLR 2024 (arXiv 2023). https://arxiv.org/abs/2310.04408
29. RECOMP repository. Fangyuan Xu et al. 2023. https://github.com/carriex/recomp
30. LLMLingua series repository (LLMLingua, LongLLMLingua, LLMLingua-2). Microsoft. 2023-2025. https://github.com/microsoft/LLMLingua
31. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." Huiqiang Jiang et al. (Microsoft). ACL 2024. https://arxiv.org/abs/2310.06839
32. "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression." Zhuoshi Pan et al. (Microsoft). ACL 2024 Findings. https://arxiv.org/abs/2403.12968
33. "xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token." Xin Cheng et al. NeurIPS 2024. https://arxiv.org/abs/2405.13792
34. xRAG repository. Xin Cheng et al. 2024. https://github.com/Hannibal046/xRAG
35. "Enabling Large Language Models to Generate Text with Citations" (ALCE). Tianyu Gao, Howard Yen, Jiatong Yu, Danqi Chen. EMNLP 2023. https://arxiv.org/abs/2305.14627
36. ALCE repository. Princeton NLP. 2023. https://github.com/princeton-nlp/ALCE
37. FRONT repository (Fine-grained Attribution). Lei Huang et al. (HIT). 2024. https://github.com/LuckyyySTA/Fine-grained-Attribution
38. "Learning Fine-Grained Grounded Citations for Attributed Large Language Models" (FRONT). Lei Huang et al. ACL 2024 Findings. https://arxiv.org/abs/2408.04568
39. "LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA." Jiajie Zhang et al. (Tsinghua / Zhipu). 2024. https://arxiv.org/abs/2409.02897
40. LongCite repository. THUDM. 2024. https://github.com/THUDM/LongCite
41. "SelfCite: Self-Supervised Alignment for Context Attribution in Large Language Models." Yung-Sung Chuang et al. (MIT, Meta). ICML 2025. https://arxiv.org/abs/2502.09604
42. SelfCite repository. Meta / facebookresearch. 2025. https://github.com/facebookresearch/SelfCite
43. "Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse." Maojia Song et al. ICLR 2025. https://arxiv.org/abs/2409.11242
44. Trust-Align repository. DeCLaRe Lab (SUTD). 2024. https://github.com/declare-lab/trust-align
45. Citations documentation, Claude API. Anthropic. 2025 (accessed October 2026). https://platform.claude.com/docs/en/build-with-claude/citations
46. "Trusting Your Evidence: Hallucinate Less with Context-aware Decoding." Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, Scott Wen-tau Yih. NAACL 2024. https://arxiv.org/abs/2305.14739
47. Context-aware decoding repository. Xiaochuang Han, Weijia Shi. 2023. https://github.com/xhan77/context-aware-decoding
48. "Know Your Limits: A Survey of Abstention in Large Language Models." Bingbing Wen et al. 2024. https://arxiv.org/abs/2407.18418
49. "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems." Hailey Joren et al. (Google). ICLR 2025. https://arxiv.org/abs/2411.06037
50. "Lynx: An Open Source Hallucination Detection Model." Patronus AI. 2024. https://arxiv.org/abs/2407.08488 (repository: https://github.com/patronus-ai/Lynx-hallucination-detection)
51. "ContextCite: Attributing Model Generation to Context." Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, Aleksander Madry. NeurIPS 2024. https://arxiv.org/abs/2409.00729
52. ContextCite repository. MadryLab (MIT). 2024. https://github.com/MadryLab/context-cite
53. "Defending Against Indirect Prompt Injection Attacks With Spotlighting." Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman (Microsoft). 2024. https://arxiv.org/abs/2403.14720
54. "Provence: Efficient and Robust Context Pruning for Retrieval-Augmented Generation." Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant (Naver Labs Europe). ICLR 2025. https://arxiv.org/abs/2501.16214
55. "In Defense of RAG in the Era of Long-Context Language Models" (OP-RAG). Tan Yu, Anbang Xu, Rama Akkiraju (NVIDIA). 2024. https://arxiv.org/abs/2409.01666
56. "AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge." Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal. NAACL 2025. https://arxiv.org/abs/2409.07394
57. AdaCAD repository. Han Wang et al. 2024. https://github.com/HanNight/AdaCAD
58. "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions." Polina Kirichenko et al. (Meta FAIR). 2025. https://arxiv.org/abs/2506.09038 (repository: https://github.com/facebookresearch/AbstentionBench)
