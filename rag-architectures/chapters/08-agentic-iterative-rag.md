# 08. Agentic and Iterative RAG: Self-RAG, CRAG, FLARE, IRCoT, Adaptive-RAG, ReAct, RL-Trained Search Agents and Agentic Search

## Overview

Classic RAG (Chapter 01) is a fixed pipeline: embed the query, retrieve top-k once, put the
passages in the prompt, generate. Every design in this chapter breaks one of the assumptions
in that sentence.

- **Retrieve once.** Multi-hop questions need evidence that can only be found after part of
  the answer is known. DSP's example is "How many storeys are in the castle David Gregory
  inherited?". A retrieve-then-read pipeline fetches a passage about the St. Gregory Hotel
  and answers "nine". The right approach first asks which castle David Gregory inherited [14].
  IRCoT, Self-Ask, DSP and ITER-RETGEN turn retrieval into a loop [7][13][14][15].
- **Always retrieve.** Many queries need no retrieval, and irrelevant passages can make
  answers worse. FLARE, DRAGIN, Self-RAG and Adaptive-RAG decide *when* to retrieve
  [1][5][9][16].
- **Trust what came back.** Retrievers fail silently. Self-RAG and CRAG add an explicit
  judgment of whether retrieved text is relevant and whether the output is supported. CRAG
  can then throw the retrieval away and search the web instead [1][3].
- **A human-designed control flow.** ReAct hands the loop to the model: think, act (search),
  observe, repeat [11]. Search-R1 and its successors go further and train that policy with
  reinforcement learning (RL) against a search engine [21][23][29][30].
- **A pre-built index.** Coding agents such as Claude Code mostly skip the vector index.
  The model searches the live filesystem with `grep`/`find` and reads files on demand,
  which Anthropic calls "just in time" retrieval [36][37][38].

The result is a spectrum. At one end, retrieval is a fixed pre-processing step. At the
other, retrieval is one tool among many inside an autonomous agent that decides how many
searches to run, what to read and when to stop. This chapter goes along that spectrum in
roughly historical order: interleaved multi-hop loops (2022-23), triggered and self-reflective
retrieval (2023-24), routing by query complexity (2024), ReAct-style agents, RL-trained search
policies (2025-26), multi-agent research systems, and the agentic-search versus indexing
debate in coding agents. Query decomposition and learned query rewriting are covered in
Chapter 06. Graph-based multi-hop retrieval (HippoRAG) is covered in Chapter 07.

```
  FIXED PIPELINE                                                      AUTONOMOUS AGENT
  |---------------|----------------|----------------|----------------|---------------|
  Naive RAG     IRCoT / DSP      FLARE / DRAGIN   Self-RAG / CRAG   ReAct, Search-R1,
  (1 retrieval) (loop fixed by   (retrieval fires (model or small   Claude Code,
                 the program)     on uncertainty)  critic grades    multi-agent research
                                                   and corrects)    (model owns the loop)
  control: code ------------------------------------------------------> control: policy
  cost:    1 LLM call + 1 search -----------------------------> 10s-100s of calls/searches
```

## 1. A control-theoretic view: four decisions

Every iterative or agentic RAG system makes four decisions, explicitly or implicitly. The
designs differ in *who* makes each decision and *what signal* they use.

| Decision | Question | Signal used | Representative designs |
|---|---|---|---|
| Trigger | Retrieve now, or not at all? | token probability, attention entropy, learned token, classifier, policy | FLARE [5], DRAGIN [16], Self-RAG [1], Adaptive-RAG [9], Search-R1 [21] |
| Query | What to search for? | last CoT sentence, predicted next sentence, sub-question, attention-weighted terms, free-form policy output | IRCoT [7], FLARE [5], Self-Ask [13], DRAGIN [16], ReAct [11] |
| Verify | Is what came back useful and supporting? | critic tokens, trained evaluator, LLM grader | Self-RAG [1], CRAG [3], LangGraph graders [18][19] |
| Stop | Enough evidence to answer? | answer emitted, budget, max turns, evaluator verdict | IRCoT [7], Search-R1 `max_turns` [21], Anthropic lead agent [33] |

The survey by Gao et al. groups the loop variants into *iterative* retrieval (alternate
retrieval and generation, as in ITER-RETGEN), *recursive* retrieval (refine the query step by
step, as in IRCoT) and *adaptive* retrieval (the model decides when and whether to retrieve,
as in FLARE and Self-RAG) [15]. FlashRAG uses a similar split: *sequential*, *conditional*
(route by query type), *branching* (parallel paths, merged) and *loop* pipelines, plus a newer
*reasoning* class for RL-trained search models [17].

## 2. Interleaved retrieval for multi-hop questions

### 2.1 DSP and Self-Ask: programmatic and prompted decomposition

**Problem.** A frozen LM plus a retriever, wired as retrieve-then-read, cannot answer a
question whose second hop depends on the first hop's answer [14].

**Self-Ask** (Press et al., Findings of EMNLP 2023) names the *compositionality gap*: models
can often answer each sub-question correctly but still fail on the composed question. It
prompts the model to ask explicit follow-up questions, each of which can be sent to a search
engine, before it answers. The paper also introduced the Bamboogle dataset, which later
became a standard out-of-domain test for search agents [13][23].

**DSP (Demonstrate-Search-Predict)** (Khattab et al., Stanford, 2022) writes the pipeline as
a program. DEMONSTRATE bootstraps intermediate annotations (hop-1 query, hop-1 passage,
hop-2 query) on a few training examples using weak supervision. SEARCH generates hop queries
conditioned on earlier passages. PREDICT produces a grounded answer. The authors report
37-120% relative gains over vanilla GPT-3.5, 8-39% over retrieve-then-read and 80-290% over
a contemporaneous self-ask pipeline across open-domain, multi-hop and conversational
settings [14]. DSP later became DSPy. Its main lesson for designers is that multi-hop
control flow can be *compiled*: demonstrations for every intermediate step are generated
instead of hand-written.

### 2.2 IRCoT: chain-of-thought as the query generator

**Problem.** For multi-step questions, "what to retrieve depends on what has already been
derived", so one retrieval with the original question misses later-hop evidence [7].

**Architecture.** IRCoT (Trivedi et al., Stony Brook NLP, ACL 2023) alternates two
steps until the reasoning chain emits an answer or a step limit is reached:

```
  question q ---> BM25 (Elasticsearch) ---> paragraphs P0
       |
       v
  +---------------------------------------------------------------+
  | REASON: LLM(q, P_all, CoT so far) -> next CoT sentence s_i      |
  |   if s_i contains "answer is" or i == max_steps: stop          |
  | RETRIEVE: BM25(s_i) -> top-K paragraphs, add to P_all          |
  +---------------------------------------------------------------+
       |  (loop)
       v
  READER: answer from q + collected paragraphs (or read off CoT)
```

The latest chain-of-thought sentence becomes the next query. It usually names the bridge
entity that the original question did not contain. The released code supports Codex and
Flan-T5 (base to XXL) as reasoners over BM25 indexes for HotpotQA, 2WikiMultihopQA, MuSiQue
and IIRC. It also includes baselines for no retrieval (`nor`) and one-step retrieval
(`oner`) [8].

**Results.** The paper's abstract reports improvements of up to about 21 points in retrieval
recall and up to about 15 points in downstream QA over one-step retrieval (approximate,
abstract-level figures) [7]. In FlashRAG's uniform re-implementation (Llama3-8B-Instruct,
e5-base-v2, 5 documents), IRCoT gets the best HotpotQA F1 among non-RL "loop" methods:
41.5, against 35.3 for standard RAG. It costs several retrieve-reason rounds per question
[17].

**Failure modes.** (a) Error compounding. A wrong bridge entity in sentence *i* sends every
later retrieval off course. (b) Latency grows linearly with hops, because each round is a
sequential LLM call plus a search. (c) The loop runs even on single-hop questions, which
motivated Adaptive-RAG (Section 5). Chapter 07 reports that HippoRAG matched or beat IRCoT
at roughly 10-30x lower cost by moving the hop into a graph walk.

### 2.3 ITER-RETGEN and branching variants

ITER-RETGEN (Shao et al., 2023) runs a fixed number of retrieve-read rounds and uses the
whole previous generation as the next query. It is "iterative" rather than "recursive" in
the survey's terms [15]. Under FlashRAG's settings it reaches 38.3 HotpotQA F1, but only 21.6
on 2WikiMultihopQA, where naive generation without retrieval scores 33.9 [17]. The
2WikiMultihopQA column is a useful warning: several RAG methods score *below* no-retrieval
generation on that dataset in that setup. That is evidence that badly targeted retrieval
can hurt reasoning-heavy questions.

## 3. Uncertainty-triggered retrieval: FLARE and DRAGIN

### 3.1 FLARE (Forward-Looking Active REtrieval)

**Problem.** In long-form generation, information needs appear *during* generation.
Retrieving only with the input misses them. Retrieving at a fixed interval (every *n*
tokens) wastes calls and uses stale queries.

**Architecture.** FLARE (Jiang et al., EMNLP 2023) "actively decides when and what to
retrieve using a prediction of the upcoming sentence to anticipate future content". If the
tentative next sentence contains low-confidence tokens, that sentence is used as the query.
The sentence is then regenerated with the retrieved documents [5][6]. The survey summarizes
the trigger as retrieval firing "when the probability falls below a certain threshold" [15].

```
  context y_<t ---> LLM drafts next sentence s'_t (no retrieval)
                         |
         all token probs >= theta ?  ---- yes ---> accept s'_t, continue
                         | no
                         v
        query = s'_t (the tentative sentence itself)
                         |
                retrieve (BM25 over Wikipedia / Bing)
                         |
        regenerate s_t conditioned on retrieved docs ---> continue
```

**Implementation details.** The reference code runs on `text-davinci-003`, with
Elasticsearch over the DPR Wikipedia dump (`psgs_w100`) and a cached Bing search server for
WikiASP. The README warns that "experiments are relatively expensive because FLARE calls
OpenAI API multiple times for a single example" [6]. FLARE depends on token log-probabilities.
Many hosted chat APIs now expose these only partially or not at all, which limits the
design to self-hosted or log-prob-capable models.

**Reproduced behaviour.** In FlashRAG's uniform setting, FLARE scores almost exactly the same
as naive generation (NQ EM 22.5 vs 22.6; HotpotQA F1 28.0 vs 28.4) [17]. This does not
contradict the original paper, which used a different model and long-form tasks. It does
show that a confidence threshold tuned for one model can leave retrieval almost never
triggered on another. The threshold has to be recalibrated per model.

### 3.2 DRAGIN

DRAGIN (ACL 2024) refines both FLARE decisions. **RIND** (Real-time Information Needs
Detection) decides when to retrieve from the LLM's uncertainty, each token's importance and
its semantic significance. **QFS** (Query Formulation based on Self-attention) builds the
query from the tokens the model attends to across the whole context, not just the last
sentence [16]. One of its configurations is named `attn_entropy`. The repository warns about
an operational trap common to all these Wikipedia-based reproductions: the Elasticsearch
index must hold about 21 million passages (about 11.2 GB). If disk space runs low, indexing
stops "silently, with no errors and no logs", and the incomplete index then produces only a
"subtle drop in performance" [16]. In production, index-completeness checks belong in the
health checks of any iterative RAG system. The iteration hides recall loss because the model
can partly compensate from parametric memory.

## 4. Self-reflective retrieval: Self-RAG

**Problem.** Indiscriminate retrieval of a fixed number of passages "diminishes LM
versatility or can lead to unhelpful response generation", and the output is not guaranteed
to be consistent with the passages [1].

**Architecture.** Self-RAG (Asai et al., UW / AI2 / IBM; arXiv 2023, ICLR 2024) trains *one*
LM to emit four kinds of **reflection tokens** as ordinary next-token predictions from an
extended vocabulary. The 7B and 13B models are released on Hugging Face [1][2]:

| Token | Input | Values | Meaning |
|---|---|---|---|
| Retrieve | x, or x and y | yes, no, continue | whether to call the retriever |
| ISREL | x, d | relevant, irrelevant | passage d helps answer x |
| ISSUP | x, d, y | fully, partially, no support | output y is supported by d |
| ISUSE | x, y | 5..1 | overall utility of y |

Inference works segment by segment (one sentence per segment in the experiments). At each
segment the model first predicts Retrieve. If retrieval is needed, it retrieves *K*
passages, generates *K* candidate continuations in parallel (one per passage), critiques each
with ISREL/ISSUP/ISUSE, and keeps the best candidates by segment-level beam search [1].

```
  x, y_<t --> [Retrieve?] --no--> generate y_t --> [ISUSE]
                  | yes (or P(yes) > threshold delta)
                  v
          Contriever-MS MARCO: top-K passages d1..dK
                  |
     +------------+------------+-------------+
     v            v            v             v      (parallel, one per passage)
   d1 -> [ISREL] y_t^1 [ISSUP] [ISUSE]   ...   dK -> ... y_t^K
                  |
     score_k = log p(y_t^k) + sum_G w_G * s_G(critique tokens)
                  |
     segment-level beam search (keep top-B) --> next segment
```

**Training pipeline.**

1. *Critic data.* GPT-4 is prompted with a type-specific instruction to label reflection
   tokens. The authors collect 4k-20k examples per token type [1].
2. *Critic model C.* Llama 2-7B, fine-tuned on these labels. It agrees with the GPT-4
   predictions more than 90% of the time on most categories [1].
3. *Generator corpus.* C runs offline over 150k instruction-output pairs (Open-Instruct
   subsets plus knowledge-intensive sets such as NQ, Wizard of Wikipedia, FEVER, ASQA). It
   inserts Retrieve tokens, retrieved passages and critique tokens [1].
4. *Generator M.* Llama 2 7B/13B, trained with the plain LM objective. Retrieved passages are
   masked out of the loss. Training used 4x A100-80GB, 3 epochs, batch size 128, learning
   rate 2e-5 [1].

Inserting critiques offline avoids hosting a reward model during training, which the
authors say "significantly reduces training costs compared to PPO" [1].

**Inference knobs.** By default ISREL, ISSUP and ISUSE are weighted 1.0, 1.0 and 0.5. The
retrieval threshold is 0.2 for most tasks and 0 for ALCE-ASQA (always retrieve, because
citations are required). Inference runs on vLLM [1]. Raising the ISSUP weight raises citation
precision but lowers MAUVE fluency. The authors present this as test-time customization
without retraining [1].

**Results.** Self-RAG 13B reaches 55.8 accuracy on PopQA (long-tail subset), 69.3 on TriviaQA,
74.5 on PubHealth, 73.1 on ARC-Challenge, and 70.3/71.3 citation precision/recall on ASQA.
Retrieval-augmented Llama2-13B gets 2.3/3.6 citation precision/recall on ASQA, and Ret-ChatGPT
gets 65.1/76.6 [1]. Ablations show that removing the retriever or the critic during training
causes large drops [1].

**Failure modes and lessons.**

- The model *is* the controller. Changing the retriever, the corpus or the base model means
  retraining, or at least re-validating that the reflection tokens are still calibrated.
- In FlashRAG's uniform benchmark, the released `selfrag-llama2-7B` scores 38.2 EM on
  TriviaQA. Naive Llama3-8B generation scores 55.7 [17]. A trained controller is tied to its
  base model, and newer base models can overtake it.
- The cost is variable. CRAG's authors estimate Self-RAG generation at 26.5-132.4 TFLOPs per
  token, against a fixed 26.5 for plain RAG, because of the parallel per-passage candidates
  [3].
- The lasting contribution is the *vocabulary*: retrieve / relevant / supported / useful.
  LangGraph's Self-RAG tutorial re-implements these four decisions as prompted LLM "graders"
  wired into a state graph, with no fine-tuning [18]. Most production "self-RAG" today means
  this prompted variant.

## 5. Corrective retrieval: CRAG

**Problem.** Self-RAG asks whether retrieval is needed. CRAG (Yan et al., USTC / UCLA /
Google DeepMind, 2024) asks a different question: what should the system do when the
retriever returns wrong documents? A generator coupled to a weak retriever has "low risk
tolerance" [3].

**Architecture.**

```
  query x --> retriever --> ~10 docs
                  |
     T5-large retrieval evaluator scores each (x, d_i) pair
                  |
     +------------+--------------------+-------------------------+
     | any score > upper threshold      | all scores < lower      | otherwise
     v                                  v                         v
  CORRECT                            INCORRECT                 AMBIGUOUS
  knowledge refinement:              discard retrieval;        both: refined internal
  decompose docs -> strips,          ChatGPT rewrites x to     + web external
  score strips, drop irrelevant,     keyword query -> web      knowledge
  recompose in order                 search API (prefers
                                     Wikipedia pages) ->
                                     fetch pages -> refine
                  \__________________________|_________________________/
                                             v
                                         generator
```

**Key techniques.**

- *Evaluator, not generator, does the judging.* A fine-tuned T5-large model scores each
  question-document pair. On PopQA its accuracy at judging retrieval quality is 84.3%.
  ChatGPT scores 58.0% zero-shot, 62.4% with CoT and 64.7% few-shot [3]. A small trained
  classifier beat a large prompted model at this narrow task.
- *Three actions, not two.* In preliminary experiments with only Correct and Incorrect,
  CRAG's efficacy "was easily affected by the accuracy of the retrieval evaluator". The
  Ambiguous action hedges by combining both knowledge sources [3].
- *Decompose-then-recompose.* Documents are split into "knowledge strips" of a few
  sentences. The same evaluator scores each strip, and only relevant strips are
  concatenated. This is passage-level filtering before generation [3].
- *Web search as fallback corpus.* Queries are rewritten into keywords "to mimic the daily
  usage of search engine". A commercial web search API (Google Search API in the paper) is
  used, and authoritative pages such as Wikipedia are preferred, to limit the bias and
  unreliability of web content [3].

**Results and cost.** Applied to SelfRAG-LLaMA2-7b, CRAG improves over plain RAG by 7.0
points on PopQA, 14.9 FactScore on Biography, 36.6 on PubHealth and 15.4 on ARC-Challenge
[3]. When plain RAG and Self-RAG are also given web search, CRAG still wins in all cases.
The authors therefore attribute the gain to the correction mechanism, not to the extra
corpus [3]. Measured on PopQA generation only (retrieval excluded), the overhead is small:

| Method | TFLOPs per token | Execution time per instance (s) |
|---|---|---|
| RAG | 26.5 | 0.363 |
| CRAG | 27.2 | 0.512 |
| Self-RAG | 26.5 to 132.4 | 0.741 |
| Self-CRAG | 27.2 to 80.2 | 0.908 |

Source: [3], Table 6. These are rough estimates of the generation phase. Web search
latency, often seconds, is not included.

**Failure modes.** The whole system depends on the evaluator. The authors released the
evaluator weights and the knowledge-preparation code, but the evaluator was fine-tuned on
PopQA [4]. A judge trained on PopQA will not transfer to a legal or code corpus without new
labels. The web fallback also brings in freshness and trust problems that the internal corpus did not have. LangGraph's CRAG
tutorial makes the usual production simplification: it skips knowledge refinement, uses an
LLM grader for relevance, and runs a web search tool (Tavily) when any document is graded
irrelevant [19].

## 6. Routing by query complexity: Adaptive-RAG

**Problem.** Single-step RAG is cheap but fails on multi-hop questions. IRCoT-style
iteration handles multi-hop but wastes calls on simple ones. Real traffic mixes both [9].

**Architecture.** Adaptive-RAG (Jeong et al., NAACL 2024) puts a classifier in front
of three strategies, all built on the IRCoT codebase: `nor_qa` (no retrieval), `oner_qa`
(single-step) and `ircot_qa` (multi-step) [10].

```
  query --> T5-large complexity classifier --> A: no retrieval   (LLM only)
                                          \--> B: single-step    (BM25 once + LLM)
                                          \--> C: multi-step     (IRCoT loop)
```

**Label generation without annotation.** Labels come from two sources. *Silver* labels
record which strategy actually answered each training question correctly; the simplest
successful one wins. *Binary* labels come from the dataset's inductive bias: questions from
single-hop datasets (NQ, TriviaQA, SQuAD) are labelled simpler, and questions from multi-hop
datasets (MuSiQue, HotpotQA, 2WikiMultiHopQA) are labelled complex. A separate classifier is
trained for each reader LLM (FLAN-T5-XL, FLAN-T5-XXL, GPT) [10]. The classifier learns which
pipeline *this* LLM needs, not an abstract notion of difficulty.

**Results.** The authors report better overall efficiency and accuracy than the baselines,
including other adaptive retrieval approaches [9]. In FlashRAG's re-implementation,
Adaptive-RAG falls between standard RAG and IRCoT on HotpotQA (39.1 vs 35.3 and 41.5 F1)
[17]. That is the expected trade-off: it gives up some multi-hop accuracy in exchange for
skipping the loop on queries routed to A or B.

**Practitioner adoption.** LangGraph, LlamaIndex and Cohere published Adaptive-RAG
cookbooks [10]. LangGraph's version replaces the trained classifier with an LLM router
between web search (recent events) and a self-corrective RAG branch over the index [20].
That combination (router, then graders, then fallback) is the most common "agentic RAG"
blueprint in framework documentation.

## 7. ReAct-style retrieval agents

**ReAct** (Yao et al., ICLR 2023) interleaves free-form *thoughts* with
*actions* and *observations*. On HotpotQA and FEVER the actions are calls to a simple
Wikipedia API (search an entity, look up a string in the current page, finish with an
answer) [11]. The released prompts report, on 500 random dev questions, HotpotQA EM of 29.4
(PaLM-540B) and 30.4 (GPT-3 `davinci-002`), and FEVER EM of 62.2 and 54 [12]. These numbers
are low by 2026 standards. ReAct matters for its *interface*, not its accuracy. Every
current search agent, including Tongyi DeepResearch's evaluation mode and the LangGraph
agents used to benchmark code-search tools, still runs a ReAct-shaped loop [31][40].

```
  Thought: I need the castle David Gregory inherited.
  Action:  search[David Gregory (physician)]
  Obs:     "... inherited Kinnairdy Castle in 1664 ..."
  Thought: Now find how many storeys Kinnairdy Castle has.
  Action:  search[Kinnairdy Castle]
  Obs:     "... a tower house of five storeys ..."
  Action:  finish[five]
```

(Illustrative trace in ReAct format. The passages are the ones shown in DSP's Figure 1
for this running example [14].)

**Anthropic's framing.** Anthropic's "Building effective agents" (December 2024) separates
*workflows*, where LLMs and tools are orchestrated "through predefined code paths", from
*agents*, where LLMs direct their own processes and tool use. The basic building block is the
"augmented LLM", which generates its own search queries. In the *evaluator-optimizer* pattern,
an evaluator decides whether more searches are needed, and the post gives "complex search
tasks that require multiple rounds of searching and analysis" as a fitting example [35]. In
these terms IRCoT, FLARE, CRAG and Adaptive-RAG are workflows, and ReAct, Search-R1 and
Claude Code are agents.

## 8. Reasoning models that search: Search-o1 and WebThinker

Long-chain reasoning models (o1, R1, QwQ) brought a new failure mode: knowledge gaps in the
middle of a long chain of thought, which show up as hedging and uncertainty [27].

- **Search-o1** (Renmin University, EMNLP 2025) adds an agentic RAG step to the reasoning
  process. The model emits a search query whenever it hits uncertain knowledge. A
  **Reason-in-Documents** module then analyzes the retrieved documents and injects only a
  refined result back into the chain, to avoid flooding the trace with raw text. Inference is
  batched: all sequences generate together, search queries are detected and served in
  batches, and refined documents are spliced back in [27]. It was evaluated on GPQA, MATH500,
  AMC2023, AIME2024 and LiveCodeBench as well as open-domain QA [27].
- **WebThinker** (NeurIPS 2025) adds a **Deep Web Explorer** that searches and *navigates*:
  it clicks links and buttons and follows deeper pages. It also adds a
  **Think-Search-and-Draft** mode with tools to draft a section, check the report and edit
  it, all within the model's thinking. Training uses online DPO on preference pairs built
  from trajectory correctness, tool usage and output quality [28].

The design point both share is *compression at the boundary*. Retrieved text is distilled
by a separate step before it enters the reasoning context. CRAG's strip filtering and
Anthropic's subagents, which "return only the summary" (Section 10), do the same thing at
different layers.

## 9. RL-trained search agents

In 2025 the field moved from *prompting* a model to search toward *training* a search
policy with outcome-based RL, in the DeepSeek-R1 style. The environment is a search engine.
The reward is usually answer correctness alone.

### 9.1 Search-R1

**Architecture.** Search-R1 (Jin et al., 2025) extends
DeepSeek-R1(-Zero)-style RL with interleaved search calls. It is built on veRL and supports
PPO, GRPO and REINFORCE, with BM25, dense (flat GPU or ANN CPU) and online search engines
behind an HTTP retrieval server [21]. The base-model prompt is short [21]:

```
Answer the given question. You must conduct reasoning inside <think> and </think>
first every time you get new information. After reasoning, if you find you lack some
knowledge, you can call a search engine by <search> query </search> and it will return
the top searched results between <information> and </information>. You can search as
many times as your want. If you find no further external knowledge needed, you can
directly provide the answer inside <answer> and </answer> ...
```

**Rollout and loss masking.** During rollout, generation stops at `</search>`. The query goes
to the retriever (top-3 passages in the published scripts), and the results are spliced back
in as an `<information>` block. Generation then resumes, for up to `max_turns` rounds (4 in
the v0.1 scripts) [21]. The important implementation detail is the **information mask**. The
retrieved tokens appear in the context, but `_info_masked_concatenate_with_padding` masks
them out of the policy-gradient loss. The model is not trained to *generate* retrieved text,
only to reason around it [21].

```
  prompt | <think>..</think><search>q1</search> | <information>docs</information> | ...
         |<------- policy tokens: in loss ------>|<--- retrieved: masked, no grad ->|
  ... <think>..</think><answer>a</answer>   --> reward = EM(a, gold)  (outcome only)
```

**Reward.** Exact match on the final answer (`compute_score_em`), optionally with a format
score. There is no process supervision of individual searches [21].

**Results and ecosystem.** In FlashRAG's reasoning-pipeline table, the Search-R1
Qwen2.5-7B PPO checkpoint scores 45.2 NQ EM, 54.5 HotpotQA F1 and 59.9 Bamboogle F1. The best
non-RL loop method in that benchmark scores 41.5 HotpotQA F1, with a different (Llama3-8B)
backbone, so the comparison is only indicative [17]. A follow-up empirical study by the same
authors examines reward design, base versus instruct models and search engine choice for
these agents [22]. By late 2025 Search-R1 was integrated into veRL, SkyRL and Thinking
Machines' Tinker cookbook. The README lists more than 20 derived works, including
ZeroSearch, AutoRefine, ParallelSearch, StepSearch and Tree-GRPO [21].

### 9.2 R1-Searcher and R1-Searcher++

R1-Searcher (Renmin University, March 2025) uses a **two-stage outcome-supervised RL**
schedule with Reinforce++ [23]:

- *Stage 1* teaches the model to *invoke* search. A retrieve-reward of 0.5 is granted when
  the model searches and its output is well-formatted.
- *Stage 2* teaches the model to *use* search. The retrieval requirement is dropped, the
  answer reward is F1-based, and a format violation costs 2 points.

Training data comes from HotpotQA and 2WikiMultiHopQA, bucketed by difficulty according to
how many Qwen-2.5-7B-Instruct rollouts were needed to answer correctly (<10, 10-20, >20). It
is trained against a local KILT Wikipedia index, yet the authors report that it generalizes
to Bamboogle with online Google search [23]. Their insights include: outcome reward alone is
enough; no SFT cold start is needed for a capable base model; and "LongCoT reasoning after RL
is a more effective and efficient test time scaling method than existing tree-search based
methods" [23]. R1-Searcher++ adds an SFT cold start, a reward for using *internal* knowledge
when it suffices, and a memorization mechanism that absorbs retrieved facts into the model
[24]. That reward is aimed directly at the over-retrieval habit of pure search agents.

### 9.3 ReSearch / ReCall and ZeroSearch

**ReSearch**, later renamed **ReCall** (Agent-RL, 2025), trains reasoning with tool calls via
RL "without requiring any supervised data on tool use trajectories or reasoning steps". It
generalizes from a search tool to arbitrary Python-defined tools run in a sandbox, trained
on MuSiQue plus a synthetic tool-use set [25].

**ZeroSearch** (Alibaba-NLP, 2025) targets the *cost* of RL against real search APIs:
every rollout fires many paid queries. It first fine-tunes an LLM into a **simulated search
engine** that can return either relevant or deliberately noisy documents. A **curriculum
rollout** then makes the simulated results gradually harder. The authors report that the
policy "outperforms real search engine-based models while incurring zero API cost". They
release 3B, 7B and 14B simulators and recommend the 14B for stable simulation [26].

### 9.4 Real-web RL: DeepResearcher, ASearcher, Tongyi DeepResearch, MiroThinker

| System | Org, year | Environment | Notable mechanism | Reported results (self-reported) |
|---|---|---|---|---|
| DeepResearcher [29] | GAIR-NLP, 2025 | live web search | end-to-end RL in real web environments | up to +28.9 pts over prompt baselines, +7.2 over RAG-based RL agents |
| ASearcher [30] | inclusionAI, 2025 | live web | fully asynchronous RL; LLM agent that synthesizes QA data | >100 tool calls and >400k tokens per trajectory in training; Avg@4 58.7 GAIA, 51.1 xBench-DeepSearch, 74.5 Frames (ASearcher-Web-QwQ, 32B) |
| Tongyi DeepResearch [31] | Alibaba-NLP, Sept 2025 | web, Scholar, Python, file parsing | agentic continual pre-training + SFT + on-policy GRPO variant; ReAct mode and IterResearch "Heavy" mode | 30B-A3B MoE, 128K context |
| MiroThinker [32] | MiroMind, 2025-26 | web tools | "interactive scaling" (more, deeper tool interactions as a third scaling axis) | v1.0 (8B/30B/72B): 256K context, up to 600 tool calls, 47.1% BrowseComp; v1.7: 74.0% BrowseComp |

Three engineering lessons recur in these reports:

1. **Trajectory length variance dominates RL systems cost.** ASearcher found that long
   trajectories take dozens more turns, and up to two orders of magnitude more tokens, than
   short ones. In batch-synchronous RL every batch waits for its longest rollout. Decoupling
   generation from training removes that idle time and allows turn limits of 128 per
   trajectory [30].
2. **Data must be hard enough to need search.** ASearcher's synthesis agent makes questions
   harder by *fuzzing* (obscuring key details) and *context injection* (adding retrieved
   facts). It then validates each question for quality, difficulty against a strong reasoning
   model, and answer uniqueness [30].
3. **Behaviours emerge from the reward.** DeepResearcher reports plan formulation,
   cross-validation across sources, self-reflection, and "honesty when unable to find
   definitive answers" without these being explicitly supervised [29].

Leaderboard numbers on BrowseComp, GAIA and HLE come from each vendor's own harness, with
different tool sets and budgets. Treat cross-system comparisons as approximate.

## 10. Multi-agent research: Anthropic's Research system

**Architecture.** Anthropic's Research feature (engineering post, June 2025) uses an
orchestrator-worker design [33]:

```
                         user query
                             |
                  LeadResearcher (Opus 4) --- saves plan to memory
                 /           |            \
         subagent 1     subagent 2  ...  subagent N   (Sonnet 4, own context window,
         web search     web search       web search    3+ parallel tool calls each)
                 \           |            /
                  condensed findings only
                             |
           lead synthesizes; loops if more research is needed
                             |
              CitationAgent attributes claims to sources
                             |
                       final report
```

**Numbers.** On an internal research evaluation, Opus 4 as lead with Sonnet 4 subagents beat
single-agent Opus 4 by 90.2%. On BrowseComp, token usage alone explained about 80% of the
performance variance; tool-call count and model choice were the other two factors. Agents
use about 4x the tokens of chat, and multi-agent systems about 15x. Running 3-5 subagents in
parallel, each making 3 or more parallel tool calls, cut research time "by up to 90%" for
complex queries [33].

**Effort scaling in the prompts.** The published lead-agent prompt encodes the scaling rules.
A simple fact-finding query gets 1 subagent, a standard query 2-3, a medium one 3-5 and a
high-complexity one 5-10, with "maximum 20". Each subagent sets a "research budget" of tool
calls and must stay under 20 tool calls and about 100 sources [34]. This is Adaptive-RAG's
idea of routing compute by complexity, expressed in natural language rather than as a
trained classifier.

**Search-strategy lessons.** Start with short, broad queries, then narrow, because agents
default to over-specific queries that return little [33]. Observed failure modes include
spawning many subagents for simple queries, "endless searching for nonexistent sources",
duplicated work from vague delegation, continuing to search after results were already
sufficient, and preferring SEO content farms over authoritative sources [33]. Evaluation
began with about 20 representative queries and an LLM judge scoring factual accuracy,
citation accuracy, completeness, source quality and tool efficiency. It judged end states,
not prescribed paths [33]. Operationally, agents resume from checkpoints, and rainbow
deployments keep long-running agents on their original version [33].

Open implementations follow the same pattern. LangChain's `open_deep_research` reported an
overall score of 0.4344 (rank 6) on the Deep Research Bench leaderboard in August 2025 [43].
Jina's `node-DeepResearch` reduces the loop to "keep searching, reading webpages, reasoning
until an answer is found (or the token budget is exceeded)". It deliberately targets concise
answers, not long reports [42].

## 11. Agentic search vs. indexing: Claude Code and the grep debate

### 11.1 Claude Code's design

**Problem.** A codebase changes on every edit. A pre-built embedding index of it is stale
by construction. It needs an ingestion pipeline, has to respect `.gitignore` and
permissions, and returns chunks that cut across the units a programmer cares about
(functions, call sites).

**Approach.** Claude Code runs an agentic loop with three phases (gather context, take action,
verify results) that "blend together" [37]. Context gathering is plain tool use:

- **Search tools.** `Grep` is built on ripgrep, uses ripgrep regex syntax and respects
  `.gitignore`. `Glob` finds files by name pattern [38]. On macOS, Linux and WSL these two
  tools are now left out of the default tool set. Claude instead runs `find` and `grep`
  through the Bash tool, which inside Claude's shell are embedded versions of `bfs` and
  `ugrep` [38]. Reading is done with `Read` (oversized files must be read in ranges or
  searched) or with `cat`/`head`/`tail`/`sed -n` [38].
- **Up-front context.** `CLAUDE.md` project instructions and the first 200 lines (or 25 KB)
  of auto-memory load at session start [37].
- **Code intelligence (optional).** Plugins add jump-to-definition, find-references and
  type errors through language servers [37].
- **Explore subagent.** A built-in read-only subagent (Write and Edit denied) for "file
  discovery, code search, codebase exploration". It is called with a thoroughness level
  (*quick*, *medium* or *very thorough*) and returns only a summary, which "keeps
  exploration results out of your main conversation context" [39].

Anthropic's context-engineering post (September 2025) states the rationale. Many AI
applications use "embedding-based pre-inference time retrieval". Agents increasingly use
"just in time" retrieval instead: they keep lightweight references (file paths, stored
queries, links) and load data at runtime with tools, relying on folder hierarchy, naming
conventions and timestamps as relevance signals ("progressive disclosure"). Claude Code is
described as a *hybrid*: CLAUDE.md is loaded up front, and glob and grep retrieve just in
time, which avoids "stale indexing and complex syntax trees" [36].

```
  INDEX-FIRST (classic code RAG)          AGENTIC SEARCH (Claude Code)
  repo --chunk--> embed --> vector DB      repo (live filesystem, always current)
          (re-index on change)                    ^          ^          ^
  query --embed--> top-k chunks             grep/rg     find/glob    read ranges
          |                                       \          |          /
          v                                    model plans next probe from results
  LLM answers from k chunks                         (loop; subagent may summarize)
  + 1 round-trip, cheap                     + exact, fresh, no ingestion pipeline
  - stale, chunk boundaries, recall cap     - more tokens, more turns, slower
```

The same post lists the costs honestly. Runtime exploration "is slower than retrieving
pre-computed data". It needs deliberate tool design. Without guidance, agents waste context
by misusing tools or chasing dead ends [36]. Anthropic recommends hybrids for less dynamic
domains such as legal or finance, and expects designs to rely less on human curation as
models improve [36]. The same just-in-time principle applies to tools: Claude Code defers
MCP tool definitions and loads them on demand through tool search [37].

### 11.2 The counter-evidence: semantic search as an added tool

Vector-search vendors have measured the cost side. Zilliz's `claude-context` is an MCP server
that adds hybrid (BM25 + dense) code search, with AST-based chunking and Merkle-tree
incremental re-indexing [40]. In its controlled evaluation, a LangGraph ReAct agent on 30
SWE-bench Verified instances (2-file fixes, 15-60 minute difficulty) with GPT-4o-mini, 3 runs
per arm, the results were [40]:

| Metric | grep-only agent | + claude-context MCP | Change |
|---|---|---|---|
| Retrieval F1 (file localization) | 0.40 | 0.40 | none |
| Tokens per instance | 73,373 | 44,449 | -39.4% |
| Tool calls per instance | 8.3 | 5.3 | -36.3% |

Retrieval quality was the same. The index saved tokens and turns. This is consistent with
Anthropic's own statement that agentic search trades speed and tokens for freshness and
simplicity. It is a vendor benchmark with a small model and a small sample. Serena takes a
third route: symbol-level retrieval and editing through LSP language servers (40+ languages),
or a JetBrains plugin. It aims to replace "fragile text surgery" with reference-aware
navigation [41].

**Reading the debate.** The evidence points to a layered design, not a winner:

- grep/find is *exact and fresh*. It is ideal for identifiers, error strings and config keys,
  and it needs no infrastructure.
- Embeddings help with *conceptual* queries ("where do we handle auth?") and lower token
  cost on very large repositories.
- LSP gives *structural* answers (definitions, references) that neither text method gives
  reliably.
- Subagents with summaries bound the context cost of whichever method is used [39].

## 12. Head-to-head under one harness

Most papers in this chapter use different models, corpora and metrics. FlashRAG re-ran many
of them with one generator (Llama3-8B-Instruct, 2048 input tokens), one retriever
(e5-base-v2, 5 documents) and one prompt [17]. A subset:

| Method | Type | NQ EM | TriviaQA EM | HotpotQA F1 | 2Wiki F1 | PopQA F1 |
|---|---|---|---|---|---|---|
| Naive generation | sequential | 22.6 | 55.7 | 28.4 | 33.9 | 21.7 |
| Standard RAG | sequential | 35.1 | 58.9 | 35.3 | 21.0 | 36.7 |
| Adaptive-RAG | conditional | 35.1 | 56.6 | 39.1 | 28.4 | 40.4 |
| Self-RAG (selfrag-llama2-7B) | loop | 36.4 | 38.2 | 29.6 | 25.1 | 32.7 |
| FLARE | loop | 22.5 | 55.8 | 28.0 | 33.9 | 20.7 |
| ITER-RETGEN | loop | 36.8 | 60.1 | 38.3 | 21.6 | 37.9 |
| IRCoT | loop | 33.3 | 56.9 | 41.5 | 32.4 | 45.6 |

And from FlashRAG's "reasoning pipeline" table, which uses each method's released checkpoint
(PopQA reported as EM here) [17]:

| Method (checkpoint) | NQ EM | HotpotQA F1 | 2Wiki F1 | MuSiQue F1 | Bamboogle F1 |
|---|---|---|---|---|---|
| Search-R1 (Qwen2.5-7B, PPO) | 45.2 | 54.5 | 42.6 | 29.2 | 59.9 |
| R1-Searcher (Qwen2.5-7B base) | 36.9 | 49.0 | 49.1 | 24.7 | 57.7 |
| CoRAG (Llama3.1-8B) | 40.9 | 56.6 | 60.7 | 31.9 | 54.1 |
| ReaRAG (9B) | 26.3 | 42.9 | 41.6 | 21.2 | 41.9 |

What the numbers say:

1. On single-hop QA (NQ, TriviaQA), iteration adds little over standard RAG. The gains are
   concentrated in multi-hop sets.
2. Prompted loops (IRCoT) beat trained-controller designs from 2023 (Self-RAG) on a newer
   base model. RL-trained 7-8B search policies beat both by about 13 HotpotQA F1 points. The
   backbones differ, so this is indicative only.
3. FLARE's trigger and Self-RAG's tokens did not transfer cleanly to a new setup. Methods
   whose control signal depends on model-internal calibration are the most fragile to swap.

## 13. Master comparison

| Design | Year | Who controls the loop | Trigger / stop signal | Training required | Retrieval calls per query | Main failure mode |
|---|---|---|---|---|---|---|
| Self-Ask [13] | 2022-23 | prompt | follow-up questions | none | one per sub-question | decomposition errors |
| DSP [14] | 2022 | program | fixed hops | bootstrapped demos | fixed hops | program/task mismatch |
| IRCoT [7] | 2023 | program | each CoT sentence; "answer is" | none | one per reasoning step | bridge-entity error compounding |
| FLARE [5] | 2023 | program + log-probs | low-confidence token | none | 0 to one per sentence | threshold calibration; needs log-probs |
| DRAGIN [16] | 2024 | program + attention | RIND uncertainty score | none | variable | white-box model access |
| Self-RAG [1] | 2023-24 | fine-tuned LM | Retrieve token, threshold | critic + generator SFT | 0 to one per segment, K parallel | tied to base model; cost variance |
| CRAG [3] | 2024 | evaluator + rules | T5 confidence, 2 thresholds | T5 evaluator | 1 + optional web search | evaluator domain shift; web trust |
| Adaptive-RAG [9] | 2024 | classifier | complexity class A/B/C | T5 classifier (auto labels) | 0, 1 or IRCoT | misrouted complex queries |
| ReAct [11] | 2022-23 | LLM policy (prompted) | model emits finish | none | open-ended | loops, premature finish |
| Search-o1 [27] | 2025 | reasoning model | model emits search tags | none | open-ended | noisy docs in long CoT (mitigated) |
| Search-R1 / R1-Searcher [21][23] | 2025 | RL-trained policy | learned; max turns | outcome RL | few (4-turn cap in Search-R1 v0.1 scripts) | reward hacking on EM; over-search |
| ASearcher / MiroThinker [30][32] | 2025-26 | RL-trained policy | learned; up to 100s of turns | large-scale async RL | tens to hundreds | cost; benchmark-specific harnesses |
| Anthropic Research [33] | 2025 | lead LLM + subagents | evaluator judgment; budgets | none (prompted) | many, in parallel | 15x tokens; over-spawning |
| Claude Code agentic search [36][37] | 2025-26 | LLM agent | task completion | none | open-ended grep/read | token cost; dead-end exploration |

## 14. Cross-cutting failure modes

- **Over-retrieval.** Agents search when parametric knowledge would do, and keep searching
  after the evidence is sufficient. Anthropic lists both [33]. R1-Searcher++ adds an
  internal-knowledge reward for this reason [24], and Adaptive-RAG routes simple queries
  away from retrieval [9].
- **Under-retrieval.** Confidence triggers (FLARE) and learned Retrieve tokens (Self-RAG)
  can fire too rarely when moved to a new model [17].
- **Error compounding across hops.** A wrong intermediate entity poisons later queries
  (IRCoT, ReAct). Verifiers (CRAG, ISSUP) and cross-source validation (DeepResearcher's
  emergent behaviour) are the main defences [3][29].
- **Context pollution.** Raw retrieved text accumulates in long loops. Search-o1's
  Reason-in-Documents step, CRAG's strip filtering and subagent summaries all compress at the
  boundary [3][27][39].
- **Source quality.** Web fallbacks bring in SEO content. CRAG prefers Wikipedia [3], and
  Anthropic had to steer its agents toward authoritative sources [33].
- **Silent recall loss in the substrate.** Incomplete indexes degrade results subtly
  because the model fills gaps from memory [16].
- **Non-determinism and evaluation.** Agents take different valid paths. Evaluate end
  states with rubric-based LLM judges, not step sequences, and start with a small set of
  real queries [33].
- **Cost and latency.** Every loop iteration adds a sequential LLM call plus a search.
  Multi-agent research uses about 15x chat tokens [33]. Parallel subagents and parallel tool
  calls are the main latency lever [33].

## Key takeaways for system designers

1. **Gate iteration by need.** Use a complexity router (a trained classifier as in
   Adaptive-RAG, or effort rules in the prompt as in Anthropic's lead agent) so that simple
   queries take the single-shot path. Iteration pays off on multi-hop queries and is mostly
   overhead on single-hop ones [9][17][34].
2. **Prefer explicit, inspectable graders over model-internal calibration.** CRAG's small
   trained evaluator beat prompted ChatGPT at judging retrieval (84.3% vs at most 64.7%), and
   it survives a generator swap. Log-prob thresholds and fine-tuned reflection tokens have to
   be recalibrated when the model changes [3][17].
3. **Have a fallback when retrieval fails.** "Retrieval was bad" should be a detectable state
   with an action attached: rewrite and re-search, switch corpus (web), or abstain. Do not
   generate from irrelevant context [3][19].
4. **Use the chain of thought as the query generator for multi-hop.** IRCoT-style
   "reason one step, retrieve on that step" remains a strong prompt-only baseline. Cap the
   number of steps and verify the bridge entities [7][17].
5. **Compress at the boundary.** Whether through strip filtering, Reason-in-Documents or
   subagents that return summaries, keep raw retrieved text out of the main reasoning
   context [3][27][39].
6. **RL-trained search policies are now the accuracy frontier for open models.** The recipe
   is simple: tags for think/search/answer, a retrieval server, outcome reward, and masking
   of retrieved tokens from the loss. Budget for rollout cost (ZeroSearch's simulator,
   asynchronous RL) and for hard, verified training questions [21][26][30].
7. **For live, exact corpora such as code, start with agentic search and add an index only
   if measurements justify it.** grep/find plus subagents avoids staleness and ingestion
   infrastructure. Add hybrid semantic search or LSP tools when token cost or conceptual
   queries dominate. Measured savings are about 40% of tokens at equal retrieval quality in
   one vendor study [36][38][40][41].
8. **Spend tokens deliberately.** In Anthropic's BrowseComp analysis, token budget explained
   about 80% of performance variance. Multi-agent parallelism buys quality and wall-clock time
   at about 15x chat cost, so reserve it for tasks worth that much [33].
9. **Benchmark under one harness before choosing.** Published gains often disappear when
   methods are re-run with a common model and retriever. FLARE matched naive generation and
   Self-RAG fell below it on TriviaQA in FlashRAG's re-implementation [17].

## References

1. A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi. "Self-RAG: Learning to Retrieve, Generate,
   and Critique through Self-Reflection." University of Washington / AI2 / IBM, arXiv 2023;
   ICLR 2024. https://arxiv.org/abs/2310.11511
2. A. Asai et al. Self-RAG official repository (models `selfrag_llama2_7b`/`13b`, inference
   examples). GitHub, 2023. https://github.com/AkariAsai/self-rag
3. S.-Q. Yan, J.-C. Gu, Y. Zhu, Z.-H. Ling. "Corrective Retrieval Augmented Generation."
   USTC / UCLA / Google DeepMind, arXiv 2024. https://arxiv.org/abs/2401.15884
4. S.-Q. Yan et al. CRAG official repository (evaluator weights, knowledge refinement and
   search). GitHub, 2024. https://github.com/HuskyInSalt/CRAG
5. Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, G. Neubig.
   "Active Retrieval Augmented Generation" (FLARE). CMU et al., EMNLP 2023.
   https://arxiv.org/abs/2305.06983
6. Z. Jiang et al. FLARE official repository (README). GitHub, 2023.
   https://github.com/jzbjyb/FLARE
7. H. Trivedi, N. Balasubramanian, T. Khot, A. Sabharwal. "Interleaving Retrieval with
   Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" (IRCoT). ACL
   2023. https://arxiv.org/abs/2212.10509
8. StonyBrookNLP. IRCoT official repository (systems `ircot`, `oner`, `nor`; Codex and
   Flan-T5). GitHub, 2023. https://github.com/StonyBrookNLP/ircot
9. S. Jeong, J. Baek, S. Cho, S. J. Hwang, J. C. Park. "Adaptive-RAG: Learning to Adapt
   Retrieval-Augmented Large Language Models through Question Complexity." NAACL 2024.
   https://arxiv.org/abs/2403.14403
10. S. Jeong et al. Adaptive-RAG official repository (classifier training, silver and binary
    labels). GitHub, 2024. https://github.com/starsuzi/Adaptive-RAG
11. S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, Y. Cao. "ReAct: Synergizing
    Reasoning and Acting in Language Models." ICLR 2023. https://arxiv.org/abs/2210.03629
12. S. Yao. ReAct prompting repository (README results table). GitHub, 2022-2023.
    https://github.com/ysymyth/ReAct
13. O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, M. Lewis. "Measuring and Narrowing
    the Compositionality Gap in Language Models" (Self-Ask, Bamboogle). Findings of EMNLP
    2023. https://arxiv.org/abs/2210.03350 (code: https://github.com/ofirpress/self-ask)
14. O. Khattab, K. Santhanam, X. L. Li, D. Hall, P. Liang, C. Potts, M. Zaharia.
    "Demonstrate-Search-Predict: Composing retrieval and language models for
    knowledge-intensive NLP." Stanford, arXiv 2022. https://arxiv.org/abs/2212.14024
15. Y. Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey."
    Tongji / Fudan, arXiv 2023-2024. https://arxiv.org/abs/2312.10997
16. DRAGIN authors. "DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time
    Information Needs of LLMs" (RIND, QFS), ACL 2024; official repository and reproduction
    notes. GitHub, 2024-2025. https://github.com/oneal2000/DRAGIN
17. J. Jin et al. FlashRAG: A Python toolkit for RAG research (unified benchmark tables for
    23 methods and reasoning pipelines). RUC NLPIR, GitHub, 2024-2025.
    https://github.com/RUC-NLPIR/FlashRAG
18. LangChain. "Self-RAG" LangGraph example notebook (archived). GitHub, 2024.
    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb
19. LangChain. "Corrective RAG (CRAG)" LangGraph example notebook (archived). GitHub, 2024.
    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb
20. LangChain. "Adaptive RAG" LangGraph example notebook (archived). GitHub, 2024.
    https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb
21. B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang, H. Zamani, J. Han. "Search-R1:
    Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning." arXiv
    2025, https://arxiv.org/abs/2503.09516 ; code (prompt template, info masking, EM reward,
    training scripts): https://github.com/PeterGriffinJin/Search-R1
22. B. Jin, J. Yoon, P. Kargupta, S. O. Arik, J. Han. "An Empirical Study on Reinforcement
    Learning for Reasoning-Search Interleaved LLM Agents." arXiv 2025.
    https://arxiv.org/abs/2505.15117
23. H. Song et al. "R1-Searcher: Incentivizing the Search Capability in LLMs via
    Reinforcement Learning." RUC, arXiv 2025, https://arxiv.org/abs/2503.05592 ; code:
    https://github.com/RUCAIBox/R1-Searcher
24. H. Song et al. "R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs
    via Reinforcement Learning." RUC, arXiv 2025, https://arxiv.org/abs/2505.17005 ; code:
    https://github.com/RUCAIBox/R1-Searcher-plus
25. Agent-RL. ReSearch / ReCall: "Learning to Reason with Tool Call for LLMs via
    Reinforcement Learning." arXiv 2025, https://arxiv.org/abs/2503.19470 ; code:
    https://github.com/Agent-RL/ReCall
26. Alibaba-NLP. ZeroSearch: "Incentivize the Search Capability of LLMs without
    Searching" (simulation LLMs, curriculum rollout). GitHub, 2025.
    https://github.com/Alibaba-NLP/ZeroSearch
27. X. Li, G. Dong, J. Jin, Y. Zhang, Y. Zhou, Y. Zhu, P. Zhang, Z. Dou. "Search-o1: Agentic
    Search-Enhanced Large Reasoning Models." EMNLP 2025. https://arxiv.org/abs/2501.05366 ;
    code: https://github.com/sunnynexus/Search-o1
28. X. Li, J. Jin, G. Dong et al. "WebThinker: Empowering Large Reasoning Models with Deep
    Research Capability." NeurIPS 2025. https://arxiv.org/abs/2504.21776 ; code:
    https://github.com/RUC-NLPIR/WebThinker
29. GAIR-NLP. "DeepResearcher: Scaling Deep Research via Reinforcement Learning in
    Real-world Environments." arXiv 2025. https://arxiv.org/abs/2504.03160 ; code:
    https://github.com/GAIR-NLP/DeepResearcher
30. inclusionAI. ASearcher: "Large-Scale RL for Search Agents" (fully asynchronous agentic
    RL, data synthesis agent). arXiv 2025, https://arxiv.org/abs/2508.07976 ; code:
    https://github.com/inclusionAI/ASearcher
31. Alibaba-NLP (Tongyi Lab). Tongyi DeepResearch (Tongyi-DeepResearch-30B-A3B) repository.
    GitHub, 2025. https://github.com/Alibaba-NLP/DeepResearch
32. MiroMind AI. MiroThinker repository (v0.2 to v1.7 release notes; interactive scaling);
    technical report arXiv 2025, https://arxiv.org/abs/2511.11793 ; code:
    https://github.com/MiroMindAI/MiroThinker
33. J. Hadfield, B. Zhang, K. Lien, F. Scholz, J. Fox, D. Ford. "How we built our
    multi-agent research system." Anthropic Engineering, June 13, 2025.
    https://www.anthropic.com/engineering/multi-agent-research-system
34. Anthropic. Research lead agent and research subagent prompts (subagent count guidelines,
    tool-call budgets). Claude Cookbooks, GitHub, 2025.
    https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/prompts/research_lead_agent.md
    and
    https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/prompts/research_subagent.md
35. E. Schluntz, B. Zhang. "Building effective agents." Anthropic, December 19, 2024.
    https://www.anthropic.com/engineering/building-effective-agents
36. P. Rajasekaran, E. Dixon, C. Ryan, J. Hadfield. "Effective context engineering for AI
    agents." Anthropic Engineering, September 29, 2025.
    https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
37. Anthropic. "How Claude Code works" (agentic loop, tools, context window). Claude Code
    documentation, 2026. https://code.claude.com/docs/en/how-claude-code-works
38. Anthropic. "Tools reference" (Grep, Glob, Read behaviour). Claude Code documentation,
    2026. https://code.claude.com/docs/en/tools-reference
39. Anthropic. "Subagents" (built-in Explore subagent, context isolation). Claude Code
    documentation, 2026. https://code.claude.com/docs/en/sub-agents
40. Zilliz. claude-context: semantic code search MCP for coding agents, with evaluation
    against grep-only baseline. GitHub, 2025. https://github.com/zilliztech/claude-context
    (evaluation: https://github.com/zilliztech/claude-context/tree/master/evaluation)
41. Oraios AI. Serena: semantic (LSP-based) code retrieval and editing toolkit for coding
    agents. GitHub, 2025-2026. https://github.com/oraios/serena
42. Jina AI. node-DeepResearch (DeepSearch search-read-reason loop). GitHub, 2025.
    https://github.com/jina-ai/node-DeepResearch
43. LangChain. open_deep_research. GitHub, 2025.
    https://github.com/langchain-ai/open_deep_research
