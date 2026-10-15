# 28. Design a Machine Translation System (Google Translate)

> Prompt as usually given: "Design Google Translate: translate text between any two of
> N languages", or "Design a system that auto-translates user-generated content (reviews,
> posts, messages) into the viewer's language". Variants: "Translate a whole web page or
> document while keeping its formatting", "Add a low-resource language with almost no
> parallel data", "Real-time speech-to-speech translation", "Translate product listings
> for an e-commerce marketplace".

> Compilation note: in both the drafting and review runs the shared web-search budget was
> used up and only github.com resolved (arxiv.org, research.google, blog hosts and portal
> sites did not). Sources marked (fetched) were read directly; the review run re-read
> [10][15][16][21][22][24][25][26][27].
> Sources marked (ch-ref) were verified in earlier chapters of this atlas. Sources marked
> (known; not fetched) are well-known papers or posts cited from prior knowledge; check
> them before relying on exact numbers. Scale numbers marked "typical assumption" are
> interview estimates, not published figures.

## Company tags

| Company | Basis | Production system it mirrors |
|---|---|---|
| Google | Builds Google Translate; published GNMT [1], zero-shot multilingual NMT [3], massively multilingual NMT [4] and the "next thousand languages" work [5] | Single multilingual Transformer/hybrid model, wordpieces, TPU serving, quantized inference |
| Meta | Builds feed/message translation; published M2M-100 [7] and NLLB-200 (200 languages, 54.5B MoE + distilled 1.3B/600M) [6][21] | Many-to-many model, distilled for serving, used for Facebook/Instagram/Wikipedia content |
| Amazon (AWS Translate), Microsoft (Translator), Apple (Translate app, on-device), DeepL | Ship MT products (tag reflects the product, not verified interview reports) | Same cascade: lang-ID -> segment -> NMT -> post-process |
| Airbnb, LinkedIn, Booking.com | Translate user-generated listings/reviews/posts in product (tag reflects the product; interview reports not verified) | Batch + on-demand translation of UGC with caching |

**Portal coverage.**

| Portal / source | What it covers for this problem | Link |
|---|---|---|
| ByteByteGo / Aminian & Sheng, "Generative AI System Design Interview" | Has a dedicated "Google Translate" chapter (encoder-decoder Transformer, tokenization, beam search, BLEU/METEOR, language detection) - from prior knowledge, not fetched [23] | [23] |
| alirezadir ML-Interviews | Lists "Machine Translation" / "Seq2seq models, NMT, Transformers" under NLP seq2seq tasks; links "MT for 1000 languages" [5] and NLLB under "Scaling data collection" [24] | [24] |
| Chip Huyen, ML Systems Design exercises | No MT exercise; closest is ex. 17 "predict the language a text is written in" (the lang-ID stage of this design) [25] | [25] |
| ByteByteGo "ML System Design Interview" (2023) | No MT chapter (11 chapters, all ranking/vision) [26] | [26] |
| Khang Pham, machine-learning-interview | No MT case study; RNN/seq2seq fundamentals only [27] | [27] |
| Hello Interview / Exponent / Educative | Generic MLSD/GenAI delivery framework applied to MT (requirements -> data -> model -> eval -> serving) [28][29][30] | [28][29][30] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Translate text (and docs/web pages) between any pair of N languages; auto-detect source language; alternatives per phrase; user feedback ("suggest an edit") | [1][23] |
| Languages | 100-250 languages -> 10k-60k directions; one model must cover all of them | Google 133 (2022) + 110 via PaLM 2 (2024); NLLB 200 [5][6][31] |
| Traffic | ~100B+ words/day reported for Google Translate (2016); assume ~50k-200k QPS peak of short segments | known; not fetched / typical assumption |
| Latency | p95 < 200-300 ms for a short sentence (interactive); documents can be async | typical assumption |
| Quality | Beat previous model on BLEU/chrF/COMET and human MQM/SxS per direction; no regressions on high-traffic pairs | [1][16][17] |
| Robustness | Keep named entities, numbers, markup, formatting; no hallucination; handle code-switching, typos | [1][4] |
| Safety | No added toxicity; gender-bias handling; profanity masking optional | typical assumption |
| Cost | GPU/TPU minutes dominate; cache hit rate 20-40% on popular segments | typical assumption |
| Freshness | New model monthly/quarterly; glossary/terminology updates live | typical assumption |

## Diagram 1 - Online serving path

**Diagram 1. Online request path: lang-ID -> segment -> cache -> route -> NMT decode
(candidates) -> QE re-rank -> post-process/policy -> response.**

```
 +---------------+   text, tgt_lang, ctx   +------------------------------+
 | Client (web,  |------------------------>| API gateway / auth / quota   |
 | app, API, OCR,|                         +---------------+--------------+
 | ASR front-end)|                                         |
 +-------^-------+                                         v
         |                                 +------------------------------+
         |                                 | Pre-process                  |
         |                                 |  - language ID (fastText/CLD)|
         |                                 |  - sentence segmentation     |
         |                                 |  - markup/entity/number mask |
         |                                 +---------------+--------------+
         |                                                 | segments[]
         |                                                 v
         |                                   +------------------------------+
         |        +<----------- hit ---------| Translation memory / cache   |
         |        |                          | key=(src,tgt,seg,model_ver)  |
         |        |                          +---------------+--------------+
         |        |                                          | miss
         |        |                                          v
         |        |  +-----------------+     +------------------------------+
         |        |  | Model registry  |---->| Router: pair -> model, tier  |
         |        |  | (versions, A/B) |     | (multilingual / distilled /  |
         |        |  +-----------------+     |  pair-specific / LLM)        |
         |        |                          +---------------+--------------+
         |        |                                          | batched tokens
         |        |                                          v
         |        |  +-----------------+     +------------------------------+
         |        |  | Glossary / user |---->| NMT inference server         |
         |        |  | terminology     |     | SentencePiece -> enc-dec     |
         |        |  +-----------------+     | beam search k=4 -> n-best    |
         |        |                          | (GPU/TPU, int8, dyn. batch)  |
         |        |                          +---------------+--------------+
         |        |                                          | n-best hyps
         |        |                                          v
         |        |                          +------------------------------+
         |        |                          | QE re-ranker (CometKiwi-like)|
         |        |                          | + length/copy/hallucination  |
         |        |                          |   checks                     |
         |        |                          +---------------+--------------+
         |        |                                          | best hyp
         |        v                                          v
         |  +-----------------------------------------------------------+
         +--| Post-process & policy: detokenize, restore markup/entities|
            | casing, gender alternatives, toxicity filter, write cache |
            +-----------------------------+-----------------------------+
                                          | log (src hash, hyp, QE, latency)
                                          v
                           +------------------------------+
                           | Logging / feedback store     |
                           | edits, copies, ratings       |
                           +------------------------------+
```

## Diagram 2 - Offline data, training & evaluation loop

**Diagram 2. Offline pipeline: mined + licensed + back-translated data -> filtering ->
multilingual training -> distillation -> eval (auto + human) -> registry -> canary.**

```
 +----------------+  +----------------+  +----------------+  +----------------+
 | Licensed/human |  | Web-mined      |  | Monolingual    |  | User feedback  |
 | parallel data  |  | bitext (LASER/ |  | text per lang  |  | edits, ratings |
 | (UN, EU, books)|  | LaBSE align)   |  |                |  | (logged)       |
 +-------+--------+  +-------+--------+  +-------+--------+  +-------+--------+
         |                   |                   |                   |
         |                   |                   v                   |
         |                   |         +-------------------+         |
         |                   |         | Back-translation  |         |
         |                   |         | (tgt->src model,  |         |
         |                   |         |  sampled) [9]     |         |
         |                   |         +---------+---------+         |
         v                   v                   v                   v
 +---------------------------------------------------------------------------+
 | Cleaning & filtering: lang-ID, dedup, length ratio, alignment score,      |
 | toxicity, test-set decontamination, temperature sampling over pairs (T=5) |
 +-------------------------------------+-------------------------------------+
                                       | tagged pairs: <2xx> src -> tgt
                                       v
 +-------------------------------+     +---------------------------------------+
 | SentencePiece vocab (32k-256k |---->| Train multilingual teacher            |
 | shared, char coverage 0.9995) |     | (Transformer / MoE, label smoothing)  |
 +-------------------------------+     +-------------------+-------------------+
                                                           | teacher outputs
                                                           v
                                       +---------------------------------------+
                                       | Sequence-level distillation -> small  |
                                       | student (deep enc / shallow dec), int8|
                                       +-------------------+-------------------+
                                                           |
                                                           v
 +-------------------------------+     +---------------------------------------+
 | Human eval: MQM / side-by-    |<----| Offline eval per direction: sacreBLEU,|
 | side on sampled directions    |     | chrF, COMET, BLEURT; FLORES-200, WMT  |
 +---------------+---------------+     +-------------------+-------------------+
                 | pass gates (no regression on top pairs) |
                 v                                         v
        +-------------------------------------------------------------+
        | Model registry -> shadow -> canary -> A/B per language pair |
        +------------------------------+------------------------------+
                                       | serving logs, QE scores, edits
                                       +----> back to "User feedback" (loop)
```

## Diagram 3 - Model architecture

**Diagram 3. Multilingual encoder-decoder Transformer with a target-language token
(the Google/NLLB design), plus the decode loop.**

```
  source: "<2fr> How are you?"          target so far: "<s> Comment allez"
            |                                         |
            v                                         v
 +-------------------------+             +---------------------------+
 | SentencePiece tokenize  |             | SentencePiece tokens      |
 | shared vocab (~256k)    |             | (shifted right)           |
 +------------+------------+             +-------------+-------------+
              | token ids                              | token ids
              v                                        v
 +-------------------------+             +---------------------------+
 | Shared embeddings       |             | Shared embeddings         |
 | + positional encoding   |             | + positional encoding     |
 +------------+------------+             +-------------+-------------+
              |                                        |
              v                                        v
 +-------------------------+             +---------------------------+
 | ENCODER  x N (12-24)    |             | DECODER  x M (3-12)       |
 |  self-attention         |   memory    |  masked self-attention    |
 |  FFN (or MoE FFN:       |------------>|  cross-attention to enc   |
 |   top-2 of E experts,   |  (K, V)     |  FFN / MoE FFN            |
 |   routed per token)     |             |  KV cache across steps    |
 +-------------------------+             +-------------+-------------+
                                                       |
                                                       v
                                         +---------------------------+
                                         | Linear (tied) + softmax   |
                                         | P(y_t | y_<t, x, <2fr>)   |
                                         +-------------+-------------+
                                                       | logits
                                                       v
                                         +---------------------------+
                                         | Beam search k=4-5,        |
                                         | length penalty a~0.6,     |
                                         | constraints (glossary)    |
                                         +-------------+-------------+
                                                       | n-best
                                                       v
                                            "Comment allez-vous ?"

  Loss: token cross-entropy with label smoothing 0.1 (+ MoE load-balance loss)
```

## Diagram 4 - Variant: LLM-based translation and low-resource expansion

**Diagram 4. Hybrid system: fast NMT for most traffic; LLM path for hard/long/document
context; low-resource languages bootstrapped from monolingual data (Google 1000-language
recipe [5], NLLB [6]).**

```
                     +----------------------------+
   request --------->| Difficulty / tier router   |
                     | (pair, length, domain, QE) |
                     +------+--------------+------+
               easy, short  |              | doc-level, rare pair,
               high-traffic |              | low QE on first pass
                            v              v
            +--------------------+   +--------------------------------+
            | Distilled NMT      |   | LLM translator (prompted or    |
            | (~0.6-1.3B, int8)  |   | fine-tuned), doc context, few- |
            | p50 ~30-60 ms      |   | shot glossary, style controls  |
            +---------+----------+   +---------------+----------------+
                      |                              |
                      +--------------+---------------+
                                     v
                       +----------------------------+
                       | QE gate + MBR / re-rank    |
                       +-------------+--------------+
                                     v
                                  response

 Low-resource bootstrap (offline):
 +-------------+   +--------------+   +----------------+   +--------------------+
 | Web crawl   |-->| Lang-ID with |-->| Monolingual    |-->| Self-supervised    |
 | (1000+ lang)|   | precision    |   | corpora +      |   | (MASS/denoise) +   |
 |             |   | filters      |   | mined bitext   |   | BT + multilingual  |
 +-------------+   +--------------+   +----------------+   | transfer; eval with|
                                                           | native speakers    |
                                                           +--------------------+
```

## Data & labels

| Source | Volume (typical) | Label quality | Notes |
|---|---|---|---|
| Licensed / curated parallel (UN, EU, subtitles, books) | 10M-1B pairs for high-resource pairs | High | Domain skew (legal, formal) |
| Web-mined bitext (sentence-embedding alignment) | Billions of pairs | Medium/noisy | Needs alignment-score + lang-ID filtering [4][6] |
| Back-translated monolingual | Unlimited | Synthetic source, real target | Sampled BT beats beam BT; tag `<BT>` [9] |
| Human post-edits / "suggest an edit" | Small | High but adversarial risk | Use for eval + fine-tune after vetting |
| Implicit signals (copy, share, re-query, edit distance) | Huge | Weak | Proxy labels for QE / online metrics |
| Human ratings (MQM, side-by-side) | Thousands per direction | Gold | Eval only; expensive [17] |
| Benchmark sets (FLORES-200, NLLB-MD, Toxicity-200) | ~1-3k sentences per language | Gold | Held-out eval; decontaminate web data against them [21] |

## Features / inputs

| Input | Use |
|---|---|
| Source subword tokens (BPE / SentencePiece) | Encoder input [8][10] |
| Target-language token `<2xx>` | Selects output language in one shared model [3] |
| Detected source language + confidence | Routing; ambiguity handling |
| Domain / formality / glossary tags | Controlled generation, terminology constraints |
| Document context (previous sentences) | Pronouns, consistency (doc-level variant) |
| Markup / entity placeholders | Preserved and restored after decode |
| User locale, app surface | Routing, gender/formality defaults |

## Model choices (baseline -> advanced)

| Stage | Model | Pros | Cons |
|---|---|---|---|
| Baseline 0 | Phrase-based SMT | Interpretable, small data ok | Poor fluency; replaced by NMT [1] |
| Baseline 1 | Seq2seq LSTM + attention (GNMT: 8+8 layers, wordpieces) [1] | Big quality jump (~60% fewer errors vs PBMT on some pairs) | Sequential, slow training |
| Standard | Transformer encoder-decoder [2] | Parallel training, SOTA quality; hybrids (Transformer enc + RNN dec) used in production [19] | Quadratic attention on long docs |
| Scale | One multilingual model with `<2xx>` token [3][4] | Transfer to low-resource, zero-shot, one deployment | Capacity dilution ("curse of multilinguality") for high-resource pairs |
| Capacity | Sparse MoE (NLLB 54.5B) [6] | More capacity at same FLOPs | Serving memory, load balancing |
| Serving | Distilled student, deep-encoder / shallow-decoder, int8 [11][20][22] | 3-10x cheaper, small quality loss | Extra training stage |
| Frontier | LLM translation (prompted / fine-tuned); Google used PaLM 2 to add 110 languages [31] | Doc context, style control, rare phenomena | Cost, latency, hallucination/omission |
| Decoding | QE re-rank or MBR over n-best/samples (COMET-MBR with CometKiwi QE) [15] | Higher human-rated quality than beam top-1 | N x scoring cost; metric overfitting |

## Loss / objectives

| Objective | Where | Notes |
|---|---|---|
| Token cross-entropy, label smoothing 0.1 | Main NMT training | Teacher forcing [2] |
| MoE load-balance auxiliary loss | MoE layers | Prevents expert collapse [6] |
| Sequence-level KD (train on teacher beam output) | Student | Simpler targets, enables greedy decode [11] |
| Denoising / MASS-style objective | Monolingual data, low-resource | [5] |
| Minimum-risk training / RL on BLEU or COMET | Optional fine-tune | Watch metric gaming |
| QE regression (MSE vs DA/MQM scores) | QE re-ranker | CometKiwi-style [15] |

## Offline metrics

| Metric | Type | Use / caveat |
|---|---|---|
| BLEU (sacreBLEU) | n-gram overlap | Standard, reproducible signatures; weak correlation at high quality [12][13] |
| chrF / chrF++ | char n-gram | Better for morphologically rich languages; FLORES/NLLB headline [6] |
| COMET | Neural, reference-based | Best correlation with humans in WMT; ship/no-ship [14][18] |
| BLEURT | Neural, reference-based | Google-trained metric [16] |
| CometKiwi / QE | Reference-free | Runtime re-rank, monitoring of live traffic [15] |
| MQM / side-by-side human eval | Human | Final gate on top directions [17] |
| Added-toxicity rate (Toxicity-200 word lists) | Rules | Catches toxicity introduced by the model, per language [21] |
| Targeted test suites | Rules | Numbers, named entities, gender, toxicity, hallucination, omission |

## Online metrics / A-B

| Metric | Signal |
|---|---|
| Copy / share rate of output | Usefulness |
| Re-query / edit rate, "suggest edit" rate | Dissatisfaction proxy |
| Thumbs up/down rate | Explicit quality |
| Session length, follow-up searches | Engagement |
| p50/p95 latency, error rate, cost per 1k chars | Guardrails |
| Live QE score distribution per pair | Quality drift without references |
| UGC variant: engagement on translated posts/reviews | Downstream value |

## Serving & latency budget (p95 ~250 ms short segment; typical assumption)

| Step | Budget | Technique |
|---|---|---|
| Gateway + auth | 10 ms | |
| Lang-ID + segmentation + masking | 5-10 ms | fastText-style classifier, rules |
| Cache / translation memory lookup | 5 ms | Key includes model version |
| Queue + dynamic batching | 10-30 ms | Bucket by length and pair |
| Encoder | 10-20 ms | int8/fp16/bf16 or AWQ int4, layer fusion, padding removal, batch reordering [22] |
| Beam decode (k=4, ~20 tokens) | 60-120 ms | KV cache, shallow decoder, speculative decoding |
| QE re-rank n-best | 15-30 ms | Small QE model; skip for cached/high-confidence |
| Post-process + policy | 5-10 ms | Detok, restore entities, filters |

Other: per-pair model tiers; on-device small models for offline mode (Apple/Google
offline packs); async batch pipeline for documents and UGC backfills.

## Monitoring

| What | How |
|---|---|
| Quality drift per language pair | Live QE scores, sampled human eval weekly |
| Input drift | Lang-ID mix, length, script, new slang/entities |
| Hallucination / omission | Length-ratio alarms, attention coverage, QE outliers |
| Toxicity / bias incidents | Classifier on outputs, user reports |
| Latency / cost | p95 by pair and tier, GPU utilization, cache hit rate |
| Feedback poisoning | Rate-limit and cluster suspicious "suggest edit" submissions |

## Failure modes

| Failure | Mitigation |
|---|---|
| Hallucination on short/odd input (repeats, unrelated text) | QE gate, copy-through for untranslatable strings, length-ratio check |
| Omission of clauses | Coverage checks, QE, MBR decoding |
| Wrong source language (code-switching, short text) | Lang-ID confidence threshold, show "Detected X" with override |
| Named entities / numbers mistranslated | Placeholder masking, constrained decoding |
| Gender bias ("the doctor" -> masculine) | Gender-specific alternatives shown to user |
| Low-resource pairs poor quality | Pivot through English, BT, multilingual transfer [4][5] |
| Off-target language (zero-shot outputs wrong language) | Target-lang check on output, more direct-pair data [3][7] |
| Test-set contamination from web mining | Decontaminate against FLORES/WMT sets |
| Toxic or adversarial training data | Toxicity filtering in data pipeline [6] |

## Interviewer follow-ups

- One multilingual model vs one model per pair: what breaks at 200 languages (N^2 models)?
- How does the target-language token enable zero-shot translation, and when does it fail?
- How do you get training data for a language with almost no parallel text?
- Why sample (not beam) when generating back-translation data?
- BLEU went up but users complain: which metrics and human evals do you trust and why?
- How do you evaluate quality on live traffic with no reference translations?
- How do you hit p95 < 250 ms at 100k QPS? Batching, distillation, quantization, caching.
- How do you preserve HTML/markup, numbers and named entities?
- How do you handle gender ambiguity and bias in the output?
- How do you detect and prevent hallucinations and omissions?
- When would you route to an LLM instead of the NMT model, and how do you cap cost?
- How do you use "suggest an edit" feedback safely (spam, poisoning)?
- How would you add document-level context or formality control?
- How do you roll out a new model without regressing any of 10k+ directions?
- Speech-to-speech variant: cascade (ASR -> MT -> TTS) vs direct model tradeoffs?
- Offline mode: how do you fit a pair model on a phone (size, int8, vocab) and keep it fresh?
- When is MBR / QE re-ranking worth N-times scoring cost, and how do you avoid metric gaming? [15]
- How do you measure added toxicity across 200 languages? [21]

## Portal comparison

| Portal / source | Framing | Emphasis | Gap |
|---|---|---|---|
| ByteByteGo GenAI book, "Google Translate" chapter [23] | GenAI system: encoder-decoder Transformer | Tokenization, training stages, beam search, BLEU/METEOR, language detection, serving | Light on multilingual data mining and MoE |
| alirezadir ML-Interviews [24] | MT as a core NLP seq2seq topic | Seq2seq, NMT, Transformers; links 1000-language scaling paper [5] | Topic list only, no worked design |
| Chip Huyen exercises [25] | Language identification exercise | The lang-ID front-end only | No MT design |
| Google papers/blogs [1][3][4][5] | Real system | GNMT, zero-shot `<2xx>` token, massively multilingual, 1000 languages | Research framing, little serving infra |
| Meta NLLB [6][21] | Real system, open models | Data mining, MoE, distillation, FLORES-200, NLLB-MD, Toxicity-200 | Not an interview framework |
| Generic MLSD frameworks (Hello Interview, Exponent, Educative) [28][29][30] | 6-step framework | Requirements -> data -> model -> eval -> serving -> monitoring | No MT-specific depth |

## References

1. Y. Wu et al. (Google), "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation", arXiv, 2016. https://arxiv.org/abs/1609.08144 (known; not fetched)
2. A. Vaswani et al., "Attention Is All You Need", NeurIPS / arXiv, 2017. https://arxiv.org/abs/1706.03762 (known; not fetched)
3. M. Johnson et al. (Google), "Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation", TACL / arXiv, 2016. https://arxiv.org/abs/1611.04558 (known; not fetched)
4. N. Arivazhagan et al. (Google), "Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges", arXiv, 2019. https://arxiv.org/abs/1907.05019 (known; not fetched)
5. A. Bapna et al. (Google), "Building Machine Translation Systems for the Next Thousand Languages", arXiv, 2022. https://arxiv.org/abs/2205.03983 (link verified via alirezadir repo, fetched)
6. NLLB Team (Meta), "No Language Left Behind: Scaling Human-Centered Machine Translation", arXiv, 2022. https://arxiv.org/abs/2207.04672 (known; not fetched; model facts confirmed in [21])
7. A. Fan et al. (Meta), "Beyond English-Centric Multilingual Machine Translation" (M2M-100), JMLR / arXiv, 2020. https://arxiv.org/abs/2010.11125 (known; not fetched)
8. R. Sennrich, B. Haddow, A. Birch, "Neural Machine Translation of Rare Words with Subword Units" (BPE), ACL / arXiv, 2016. https://arxiv.org/abs/1508.07909 (known; not fetched)
9. S. Edunov et al., "Understanding Back-Translation at Scale", EMNLP / arXiv, 2018. https://arxiv.org/abs/1808.09381 (arXiv id confirmed in fairseq README, fetched: https://github.com/facebookresearch/fairseq/blob/main/examples/backtranslation/README.md)
10. T. Kudo, J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer", EMNLP / arXiv, 2018. https://arxiv.org/abs/1808.06226 (known; not fetched); code: https://github.com/google/sentencepiece (fetched)
11. Y. Kim, A. M. Rush, "Sequence-Level Knowledge Distillation", EMNLP / arXiv, 2016. https://arxiv.org/abs/1606.07947 (known; not fetched)
12. K. Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation", ACL, 2002. https://aclanthology.org/P02-1040/ (known; not fetched)
13. M. Post, "A Call for Clarity in Reporting BLEU Scores" (sacreBLEU), WMT / arXiv, 2018. https://arxiv.org/abs/1804.08771 (known; not fetched)
14. R. Rei et al., "COMET: A Neural Framework for MT Evaluation", EMNLP / arXiv, 2020. https://arxiv.org/abs/2009.09025 (known; not fetched)
15. Unbabel, "COMET" repository (CometKiwi reference-free QE: wmt22-cometkiwi-da, wmt23-cometkiwi-da-xl/xxl; comet-mbr re-ranking), GitHub, 2020-2024. https://github.com/Unbabel/COMET (fetched)
16. Google Research, "BLEURT: Learning Robust Metrics for Text Generation" repository, GitHub, 2020. https://github.com/google-research/bleurt (fetched)
17. M. Freitag et al. (Google), "Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation" (MQM), TACL / arXiv, 2021. https://arxiv.org/abs/2104.14478 (known; not fetched)
18. T. Kocmi et al. (Microsoft), "To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation", WMT / arXiv, 2021. https://arxiv.org/abs/2107.10821 (known; not fetched)
19. M. X. Chen et al. (Google), "The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation" (RNMT+, hybrid), ACL / arXiv, 2018. https://arxiv.org/abs/1804.09849 (known; not fetched)
20. J. Kasai et al., "Deep Encoder, Shallow Decoder: Reevaluating Non-autoregressive Machine Translation", ICLR / arXiv, 2020. https://arxiv.org/abs/2006.10369 (known; not fetched)
21. Meta AI, "fairseq - No Language Left Behind (nllb branch)": NLLB-200 MoE 54.5B, dense 3.3B/1.3B, distilled 1.3B/600M; FLORES-200, NLLB-MD, Toxicity-200 benchmarks, GitHub, 2022. https://github.com/facebookresearch/fairseq/tree/nllb (fetched)
22. OpenNMT, "CTranslate2: fast inference engine for Transformer models" (FP16/BF16/INT16/INT8, AWQ INT4, layer fusion, padding removal, batch reordering), GitHub, 2019-2025. https://github.com/OpenNMT/CTranslate2 (fetched)
23. A. Aminian, H. Sheng, "Generative AI System Design Interview" (incl. "Google Translate" chapter), ByteByteGo, 2024. https://bytebytego.com/ (site root; chapter content from prior knowledge, not fetched)
24. alirezadir, "ML System Design" (NLP topics: "Machine Translation - Seq2seq models, NMT, Transformers"; "MT for 1000 languages" link), Machine-Learning-Interviews, GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
25. C. Huyen, "Machine Learning Systems Design: Exercises" (ex. 17 language identification), GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (fetched)
26. ByteByteGo, "ml-bytebytego: Reference materials for ML System Design Interview" (chapter list; no MT chapter), GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (fetched)
27. K. Pham, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview (fetched; no MT case study)
28. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
29. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
30. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework (ch-ref)
31. Google, "110 new languages are coming to Google Translate" (PaLM 2), The Keyword blog, 2024. https://blog.google/products/translate/google-translate-new-languages-2024/ (known; not fetched)
