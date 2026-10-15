# 35. Design a ChatGPT-like LLM Chatbot Platform: Serving, Safety, Evaluation & Feedback Loop

> Prompt as usually given: "Design ChatGPT", "Design a personal-assistant chatbot", "Design an
> LLM chatbot at scale (1M+ DAU)", "Design a large-scale LLM inference / serving platform",
> "Design a multi-tenant LLM inference API". Deep-dive variants: KV cache and continuous
> batching, small/large model routing, guardrails, RLHF/DPO feedback loop, eval harness.

> Compilation note (re-reviewed 2026-10-14): web search was exhausted and arxiv.org,
> openai.com, bytebytego.com, huyenchip.com and igotanoffer.com did not resolve from the
> build environment. GitHub pages marked "fetched" were re-read in this review. (ch-ref) =
> verified in an earlier chapter; (known; not fetched) = standard paper cited from prior
> knowledge. Numbers marked "typical assumption" are interview-style estimates.

## Company tags

| Company | Basis | Production system it mirrors |
|---|---|---|
| OpenAI | "ChatGPT" (message tree, resumable stream, stop cancels backend work) credited to OpenAI by a question bank; team, date and round unknown [5] | ChatGPT: InstructGPT-style SFT + RLHF model behind a streaming chat service [15][23] |
| Anthropic | "LLM Inference API" (token-bucket admission, 429 + Retry-After, batching, sticky routing) from self-reported, unverified candidate reports [5] | Claude API; RLHF + Constitutional AI training [16][17] |

Reference systems (not interview tags): Meta PurpleLlama safety stack (Llama Guard 1-4, Prompt
Guard 2, Code Shield, LlamaFirewall) [11][22] and Llama 2-Chat RLHF recipe [24]; LMSYS FastChat
(controller + model workers) and Chatbot Arena (1.5M+ votes) [9][37].

**Portals covering this problem.**

| Portal / source | Coverage | Link |
|---|---|---|
| ByteByteGo - Aminian & Sheng, "Generative AI System Design Interview", Ch. 4 "ChatGPT: Personal Assistant Chatbot" | Full chapter: data (Common Crawl, C4), pretraining, SFT, RLHF/DPO, sampling, benchmark + safety eval, Chatbot Arena (80 refs) | [1][2] |
| alirezadir GenAI / Agentic interview prep | Template 1 "LLM Chatbot at Scale" and Template 3 "LLM Inference and Serving Platform" (must-cover + bar-raiser lists) [3]; MLSD question bank lists "Chatbot system" and "LLM-powered customer-support chatbot" [4] | [3][4] |
| evgenyvinnik llm-driven-system-design | "ChatGPT" (planned; OpenAI attribution) and "LLM Inference API" (Anthropic attribution) problem specs and traps | [5] |
| Chip Huyen, "AI Engineering" + "Building a GenAI platform" | Eval methodology, inference optimization, guardrails, user feedback chapters | [13][14] |
| Educative "Grokking the Generative AI System Design", "LLMOps" | Training, inference, scaling, evaluation; observability and cost | [30][31] |
| IGotAnOffer, Hello Interview, Exponent | Generic OpenAI / GenAI / MLSD framework guides; coverage of this exact problem not verified (sites unreachable) | [32][33][34][35] |

## Requirements & scale

| Dimension | Requirement / number | Source |
|---|---|---|
| Functional | Multi-turn chat, streaming tokens, stop / regenerate / edit (branching message tree), history + multi-device sync, file/tool use optional | [5] |
| Functional | Safety: refuse disallowed content, resist jailbreak / prompt injection, PII handling | [3][11][12] |
| Functional | Feedback: thumbs up/down, regenerate, A/B side-by-side; used for evals and preference data | [3][9] |
| Non-functional | TTFT p50 < 0.5-1 s, TPOT 20-50 ms (20-50 tok/s), p99 tracked separately | typical assumption; metrics named in [3] |
| Non-functional | Availability 99.9%+ (typical assumption); multi-region + provider failover, GPU-scarcity fallback | [3] |
| Users | 100M WAU, 10M DAU-concurrent peak sessions ~1M | typical assumption |
| QPS | ~10 msgs/user/day x 50M DAU = 5e8/day -> ~6k avg, ~20k peak req/s | typical assumption |
| Tokens | ~1.5k input (history+system) + ~400 output per turn -> ~8M output tok/s peak | typical assumption |
| GPU math | ~2.5k-5k output tok/s per 8-GPU node (70B, batched) -> 2k-3k nodes at peak | typical assumption |
| Context | 8k-128k window (typical assumption); token budgeting + history compression | [3] |
| Cost | Cost per request / per 1M tokens; route by query difficulty | [3] |
| Tenancy | Charge est. input + max_tokens to per-org token bucket; 429 + Retry-After; separate interactive vs bulk queues | [5] |

## Diagram 1 - Online serving path

**DIAGRAM 1: End-to-end request path (gateway -> context -> input safety -> router ->
inference cluster -> output safety -> SSE stream), with stores and logging**

```text
 +--------+  HTTPS/SSE  +----------------+ auth, quota +-------------------------+
 | Client |------------>|  API Gateway   |------------>|  Conversation Service   |
 | web /  |<------------| rate limit by  |             |  msg tree, branch leaf, |
 | mobile |  token SSE  | tokens/org/tier|             |  idempotency key        |
 +--------+  (resumable)+----------------+             +-----------+-------------+
                                                                   | turn + history
                                                                   v
 +-------------------+  memory, prefs  +-------------------------------------------+
 | User memory / KV  |---------------->|  Context Builder ("candidate gen")        |
 | profile store     |                 |  system prompt + history (truncate /      |
 +-------------------+                 |  summarise) + memory + RAG/web/tool docs  |
 +-------------------+  top-k chunks   |  -> token budget fit                      |
 | Retrieval / tools |---------------->|                                           |
 +-------------------+                 +---------------------+---------------------+
                                                             | prompt
                                                             v
                                       +-------------------------------------------+
                                       | INPUT GUARDRAILS ("filtering")            |
                                       | moderation clf, Prompt-Guard jailbreak,   |
                                       | PII mask  --block--> canned refusal       |
                                       +---------------------+---------------------+
                                                             | allowed
                                                             v
 +-------------------+  version,flags  +-------------------------------------------+
 | Model registry +  |---------------->| ROUTER ("ranking/policy")                 |
 | flag/canary cfg   |                 | difficulty clf: small vs large model,     |
 +-------------------+                 | canary %, region, prefix-cache affinity   |
                                       +-----+-------------------------+-----------+
                                             | easy                    | hard
                                             v                         v
                                  +------------------+     +------------------------+
                                  | Small-model pool |     | Large-model pool       |
                                  | (8B, INT8/FP8)   |     | (70B+/MoE, TP x 8 GPU) |
                                  +--------+---------+     +-----------+------------+
                                           |   per pool: continuous    |
                                           |  batching, PagedAttn KV,  |
                                           |  prefix cache, spec-dec   |
                                           +-------------+-------------+
                                                         | token stream
                                                         v
                                       +-------------------------------------------+
                                       | OUTPUT GUARDRAILS (streaming, per chunk)  |
                                       | Llama-Guard-style clf, PII/regex, code    |
                                       | shield --violation--> stop + replace      |
                                       +---------------------+---------------------+
                                                             | safe tokens
                                                             v
                                                   back to Gateway --> SSE to Client
          all stages --async--> +-----------------------------------------------+
                                | Log bus: prompt, ctx ids, model ver, tokens,  |
                                | TTFT/TPOT, safety scores, thumbs, regenerate  |
                                +-----------------------------------------------+
```

## Diagram 2 - Offline training, evaluation & feedback loop

**DIAGRAM 2: Pretrain -> SFT -> preference tuning -> eval gates -> registry -> canary,
closed by the production feedback loop (thumbs, regenerations, red-team, arena)**

```text
 +-------------+  crawl,dedup,  +-------------+  next-token   +--------------------+
 | Web / code /|  lang-id, PII, | Pretraining |-------------->| Base model ckpt    |
 | books corpus|--quality clf-->| corpus      |   (CE loss)   |                    |
 +-------------+                +-------------+               +---------+----------+
                                                                        |
 +--------------------+ demos (prompt,ideal)  +---------------+         |
 | Human labelers +   |---------------------->| SFT / instr.  |<--------+
 | synthetic data     |                       | tuning        |
 +--------------------+                       +-------+-------+
          | ranked pairs (A > B)                      | SFT model
          v                                           v
 +--------+-----------+   pairs   +-----------------+   +----------------------------+
 | Preference data    |---------->| Reward model    |-->| RLHF (PPO) or DPO          |
 | labelers + prod    |           | (BT pairwise)   |   | + safety RM / CAI critique |
 | thumbs + side-by-  |           +-----------------+   | KL penalty to SFT          |
 | side picks         |--------- DPO directly (no RM) ->|                            |
 +--------+-----------+                                 +-------------+--------------+
                                                                      | candidate
          ^                                                           v
          |                        +----------------------------------------------+
          |                        | EVAL GATES                                   |
          |                        | capability: MMLU, GSM8K, HumanEval, MT-Bench |
          |                        | safety: refusal / over-refusal, toxicity,    |
          |                        |   jailbreak + red-team suites                |
          |                        | LLM-as-judge pairwise win-rate vs prod       |
          |                        | perf: TTFT/TPOT, tok/s/GPU after quantize    |
          |                        +-----------------------------------+----------+
          |                                                            | pass
          |                                                            v
          |    +-----------+            +-------------+       +------------------+
          |    | Full      |<-1-10-50%--| Canary /    |<------| Model registry   |
          |    | rollout   | if metrics | A-B test    |shadow | versioned ckpt   |
          |    |           | hold       |             | first | + eval report    |
          |    +-----+-----+            +------+------+       +------------------+
          |          |                         |
          |          v                         v
          |   +--------------------------------------------------------------------+
          +---| PROD LOGS: thumbs, regenerate, edits, copy, abandonment, reports,  |
              | safety flags, arena votes -> sample, de-identify, label, dedup     |
              | -> new SFT demos / preference pairs / red-team cases / eval sets   |
              +--------------------------------------------------------------------+
```

## Diagram 3 - Model architecture

**DIAGRAM 3: Decoder-only chat LLM (inference view) plus the reward / safety classifier
heads that share the same backbone family**

```text
  chat template:  <sys>...</sys> <user>...</user> <asst>  (token ids, len T)
                                   |
                                   v
                  +-------------------------------+
                  | Token embedding (vocab ~100k+)|
                  +---------------+---------------+
                                  v
          +---------------------------------------------------+
          |  N x Transformer decoder block                    |
          |  +---------------------------------------------+  |
          |  | RMSNorm -> causal self-attention            |  |
          |  |   RoPE positions, GQA heads,                |  |
          |  |   FlashAttention kernel                     |  |
          |  |   K,V --write/read--> [ paged KV cache ]    |  |
          |  | + residual                                  |  |
          |  | RMSNorm -> FFN (SwiGLU) or MoE (top-2 of E) |  |
          |  | + residual                                  |  |
          |  +---------------------------------------------+  |
          +-------------------------+-------------------------+
                                    v
                     +--------------+--------------+
                     | final norm                  |
                     +------+---------------+------+
                            |               |
            LM head (vocab) v               v scalar head
          +--------------------+     +-------------------------------+
          | logits -> temp,    |     | Reward model: r(prompt, resp) |
          | top-p sampling ->  |     | Safety clf: P(unsafe | cat_i) |
          | next token         |     | (separate small models, e.g.  |
          +---------+----------+     |  8B Llama-Guard-style)        |
                    |                +-------------------------------+
                    | append token, reuse KV (decode step, 1 tok)
                    +-----> loop until <eos> or max_tokens

  phases:  PREFILL  = all T prompt tokens in parallel  (compute-bound, sets TTFT)
           DECODE   = 1 token / step / sequence        (memory-bw bound, sets TPOT)
```

## Diagram 4 - Variant: LLM inference platform internals (vLLM / DistServe style)

**DIAGRAM 4: Inside one model pool - scheduler, KV-cache manager, prefill/decode
disaggregation and autoscaling (synthesised from [6][7][20][21][39])**

```text
  requests (prompt, max_tokens, priority, tenant)
        |
        v
 +----------------------+ admit if KV blocks free +-----------------------------+
 | Per-pool queue       |------------------------>| Iteration-level scheduler   |
 | interactive vs batch |<--preempt / swap KV-----| (Orca-style): sequences     |
 | (priority, deadline) |                         | join/leave every step       |
 +----------------------+                         +----+-----------------+------+
                                                       |                 |
                              prompt (chunked prefill) | decode batch    |
                                                       v                 v
                                       +----------------+  KV  +----------------+
                                       | PREFILL workers|----->| DECODE workers |
                                       | high FLOPs     | xfer | high mem-bw    |
                                       | prefix-cache   |      | big batch,     |
                                       | hit -> skip    |      | spec-decoding  |
                                       +-------+--------+      +-------+--------+
                                               |                       |
                                               v                       v
                                     +-----------------------------------------+
                                     | KV-cache manager: fixed-size blocks,    |
                                     | block table per seq, copy-on-write for  |
                                     | n>1 samples, radix/prefix sharing of    |
                                     | system prompt + history                 |
                                     +-----------------------------------------+
  metrics: queue depth, KV util %, TTFT, TPOT, tok/s/GPU --> autoscaler (+/- nodes)
```

## Data & labels

| Data | Use | Label / signal | Notes |
|---|---|---|---|
| Web crawl (Common Crawl, C4), code, books | Pretraining | Next token (self-supervised) | Dedup, lang-id (fastText/langid), quality + toxicity filters [1] |
| Human demonstrations | SFT | Ideal response per prompt | ~10k-100k high-quality; plus Alpaca/Dolly/FLAN-style sets [1][15] |
| Pairwise comparisons | Reward model / DPO | A > B (+ ties) | Labeler rankings of K samples [15][16][18] |
| Production implicit feedback | Preference mining, eval sets | Thumbs, regenerate (= reject), edit, copy, abandon | Noisy, biased toward vocal users; needs consent + de-id [13] |
| Arena / side-by-side votes | Model ranking | Elo / Bradley-Terry | 1.5M+ votes in Chatbot Arena [9][37] |
| Red-team prompts | Safety RM, refusal SFT, eval | Harm category, should-refuse | Human + automated red-teaming [25] |
| Moderation dataset | Input/output classifiers | Hazard taxonomy (MLCommons) | Llama Guard 3 categories [11][22] |

## Features / context signals

| Group | Examples | Where used |
|---|---|---|
| Conversation | System prompt, last N turns, rolling summary, current leaf of branch | Context builder [5] |
| User | Memory facts, language, locale, custom instructions, tier | Context builder, router, rate limit |
| Retrieved | Web search / file / RAG chunks, tool results | Context builder [12] |
| Request | Est. input tokens + max_tokens, priority, tenant | Gateway token bucket, scheduler [5] |
| Router signals | Query length, domain, difficulty-clf score, prefix-cache locality | Small vs large routing [3] |
| Safety | Moderation scores per category, jailbreak score, PII spans | Input/output rails [11][12] |

## Model choices (baseline -> advanced)

| Component | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Chat model | Hosted API (one model) | Own SFT + RLHF/DPO model family (small/large, MoE) | Control + cost at scale vs time-to-market |
| Alignment | SFT only | RLHF (RM + PPO) [15] -> DPO [18] / Constitutional AI [17] | PPO: strong but unstable/costly; DPO: simple, offline |
| Routing | Single model | Difficulty classifier / cascade small->large | Cost down 2-5x; risk of misroutes |
| Serving engine | HF generate, static batch | vLLM / SGLang: continuous batching, PagedAttention / RadixAttention, prefix cache, chunked prefill (TGI archived 2026, points to vLLM/SGLang) [6][8][38] | 2-4x+ throughput from memory efficiency [7] |
| Caching | None | Prefix/KV cache for shared system prompt; exact + semantic response cache for repeated queries [6][13][14] | Semantic-cache false hits; never cache personalised answers |
| Model gateway | Direct calls to one backend | Unified gateway: auth, quotas, model-version routing, canary, fallback to other model/provider [3][13][14] | Extra hop; fallback model may differ in safety behaviour |
| Latency tricks | FP16 | GQA, FlashAttention, FP8/INT4 quant, speculative decoding, P/D disaggregation [19][21][27][28] | Quality regressions from quant must be gated |
| Input safety | Keyword list + moderation API | Prompt Guard 2 + Llama-Guard-style clf + NeMo input/dialog rails; retrieval + execution rails for RAG/tools [11][12] | Latency + over-refusal vs coverage |
| Output safety | Post-hoc full-response check | Streaming chunk check + stop/replace | Must not add per-token latency |
| Eval | Static benchmarks | LLM-as-judge win-rate, MT-Bench, arena, red-team suites [10][37] | Judge bias (position, verbosity) |

## Loss / objectives

| Stage | Objective |
|---|---|
| Pretraining | Cross-entropy next-token: -sum log p(x_t \| x_<t) |
| SFT | Same CE, masked to assistant tokens only |
| Reward model | Bradley-Terry: -log sigmoid(r(x, y_w) - r(x, y_l)) [15] |
| RLHF (PPO) | max E[r(x, y)] - beta * KL(pi \|\| pi_SFT) (+ pretrain mix) [15] |
| DPO | -log sigmoid(beta * [log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)]) [18] |
| Safety classifier | Multi-label BCE over hazard categories [22] |
| Router | BCE: P(small model answer is acceptable) from judge labels |

## Offline metrics

| Area | Metrics |
|---|---|
| Capability | MMLU, GSM8K/MATH, HumanEval/MBPP, TriviaQA, HellaSwag, MT-Bench [1][10] |
| Preference | LLM-as-judge pairwise win-rate vs prod model; arena Elo [10][37] |
| Safety | Attack success rate on jailbreak suites (AdvBench), toxicity (RealToxicityPrompts), bias (BBQ), TruthfulQA, over-refusal rate [1] |
| Classifiers | Precision / recall / F1 per harm category, latency |
| Serving | TTFT, TPOT, p50/p95/p99, tokens/s/GPU, cost per 1M tokens [3] |

## Online metrics / A-B

| Metric | Type | Note |
|---|---|---|
| Thumbs-up rate, thumbs-down rate | Quality (explicit) | Sparse (<1-2% of turns rate); typical assumption |
| Regenerate rate, edit-and-resend rate | Quality (implicit) | Strong negative signal |
| Conversation length, D1/D7 retention, sessions/user | Engagement | Long convo can also mean failure |
| Side-by-side pick rate | Preference | Shown occasionally ("which response do you prefer?") |
| Safety incident rate, refusal rate, user reports | Guardrail | Both directions: harm and over-refusal |
| TTFT/TPOT p95, error + fallback rate, cost/conv | System | Guardrail metrics for every A-B [3] |

## Serving & latency budget (typical assumption, p95)

| Step | Budget |
|---|---|
| Gateway auth + rate limit | 10 ms |
| Load history + memory | 20-40 ms |
| Retrieval / web tool (optional) | 100-800 ms |
| Input guardrails (parallel clf) | 20-50 ms |
| Router | 5-10 ms |
| Queue + prefill (2k tokens) | 100-400 ms -> TTFT ~0.3-1 s |
| Decode | 20-50 ms / token (TPOT) |
| Output guardrail per chunk (async, lagging window) | ~0 added TTFT, small buffer |

## Monitoring

| What | Signal / alert |
|---|---|
| Capacity | Queue depth, KV-cache utilisation, preemption rate, GPU util |
| Latency | TTFT / TPOT p95/p99 per model, per region |
| Quality drift | Daily judge win-rate on sampled prod traffic, thumbs-down spike, regenerate spike |
| Safety | Block rate per category, new jailbreak clusters, user reports |
| Cost | $/1M tokens, router small-model share, cache hit rate |
| Reliability | Error rate, stream disconnects, fallback-to-other-provider count |

## Failure modes

| Failure | Mitigation |
|---|---|
| KV-cache OOM under long contexts | Paged KV, admission control, preempt/swap, max context per tier [7] |
| Head-of-line blocking by long prompts | Chunked prefill, P/D disaggregation, separate interactive/batch queues [21][39] |
| Jailbreak / prompt injection via tools | Prompt Guard, retrieval/execution rails, least-privilege tools [11][12] |
| Hallucination | RAG with citations, refusal training, judge-based monitoring |
| Over-refusal after safety tuning | Track over-refusal eval; safety RM separate from helpfulness RM [24] |
| Reward hacking (verbosity, sycophancy) | KL penalty, length-controlled judges, fresh human evals |
| Feedback loop bias | Sample beyond thumbs; stratify; human review before training |
| Stream drop mid-generation | Resumable SSE: client resends Last-Event-ID and reattaches to the same generation; persist partial output [5] |
| User presses Stop but GPU keeps decoding | Abort upstream request so the sequence leaves the batch and frees KV blocks [5] |
| Cancel races completion | One atomic state transition decides outcome; edge-assigned request id owns the work [5] |
| Bad rollout (quality/latency regression) | Shadow + canary + automatic rollback on guardrail metrics |

## Interviewer follow-ups

- How does the KV cache work, and how much GPU memory does it need for 70B at 32k context?
- When would you use static vs continuous batching, quantization, or speculative decoding? [3]
- How would you route between a small and a large model, and how do you label the router? [3]
- How do you reduce p99 latency without hurting answer quality? [3]
- Why separate prefill and decode? What metric does each one control?
- How do you rate-limit fairly by tokens per org when output length is unknown? [5]
- How do you handle stop, regenerate, edit (branching), and a client reconnecting mid-stream? [5]
- How do you keep a conversation sticky to a replica (consistent hashing with bounded load) and
  what happens once output has started? [5]
- Where would you cache (prefix KV, exact, semantic), and what must never be cached? [13][14]
- How do you manage context beyond the window (truncate, summarise, memory)?
- RLHF vs DPO: tradeoffs, and how do you prevent reward hacking?
- How do you turn thumbs and regenerations into training data without bias or privacy issues?
- How do you evaluate a new model before launch? Benchmarks vs LLM-as-judge vs arena vs A-B.
- Where do guardrails sit, and how do you moderate a stream without adding latency?
- How do you defend against jailbreaks and prompt injection from retrieved pages?
- Capacity math: how many GPUs for 20k req/s peak, and how do you autoscale?
- Multi-region and provider failover: what state must be replicated?

## Portal comparison

| Portal | Framing | Emphasis | Gaps |
|---|---|---|---|
| ByteByteGo GenAI book Ch. 4 [1][2] | "ChatGPT: personal assistant chatbot" | Model lifecycle: data, pretrain, SFT, RLHF/DPO, sampling, benchmark + safety + arena eval | Light on serving infra (batching, KV, routing) |
| alirezadir GenAI prep [3] | "LLM Chatbot at Scale" + "LLM Inference and Serving Platform" templates | Pipeline gateway -> context -> safety -> inference -> moderation -> stream; TTFT/TPOT; routing; KV reuse; capacity math | Checklist style, no worked diagrams |
| evgenyvinnik [5] | ChatGPT product (OpenAI) / Inference API (Anthropic) | Product + backend: message tree, resumable streams, cancel, token-bucket, batch dispatch, drain | Mock model; no training/alignment |
| Chip Huyen AIE / GenAI platform [13][14] | Generic GenAI platform | Context construction, guardrails, router, cache, eval pipeline, user feedback | Not interview-formatted |
| Educative Grokking GenAI SD / LLMOps [30][31] | Generative AI system design course | Training, inference, scaling, evaluation, observability | Paywalled; not fetched |

## References

1. ByteByteGo, "generativeAI" reference list, Chapter 4 "ChatGPT: Personal Assistant Chatbot", GitHub, 2024. https://github.com/ByteByteGoHq/generativeAI/blob/main/README.md#chapter-4-chatgpt-personal-assistant-chatbot (fetched)
2. A. Aminian, H. Sheng, "Generative AI System Design Interview", ByteByteGo, 2024. https://bytebytego.com/ (site root; book page not fetched)
3. alirezadir, "GenAI and Agentic System Design Interview Prep", Agentic-AI-Systems, GitHub, 2026. https://github.com/alirezadir/Agentic-AI-Systems/blob/main/06_interview_prep/genai-agentic-system-design.md (fetched)
4. alirezadir, "ML System Design" (Chatbot system; Generative AI / LLM Systems (2026) sample questions), Machine-Learning-Interviews, GitHub, 2026. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (fetched)
5. E. Vinnik, "llm-driven-system-design" ROADMAP (ChatGPT; LLM Inference API), GitHub, 2026. https://github.com/evgenyvinnik/llm-driven-system-design/blob/main/ROADMAP.md (fetched)
6. vLLM project, "vllm" repository README, GitHub, 2026. https://github.com/vllm-project/vllm (fetched)
7. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP / arXiv, 2023. https://arxiv.org/abs/2309.06180 (known; not fetched)
8. SGLang project, "sglang" repository README, GitHub, 2026. https://github.com/sgl-project/sglang (fetched)
9. LMSYS, "FastChat" repository (Vicuna, Chatbot Arena, MT-Bench, serving), GitHub, 2024. https://github.com/lm-sys/FastChat (fetched)
10. L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS / arXiv, 2023. https://arxiv.org/abs/2306.05685 (known; not fetched)
11. Meta, "PurpleLlama" (Llama Guard 1-4, Prompt Guard 2, Code Shield, LlamaFirewall, CyberSecEval), GitHub, 2026. https://github.com/meta-llama/PurpleLlama (fetched)
12. NVIDIA, "NeMo-Guardrails" (input, dialog, retrieval, execution, output rails), GitHub, 2025. https://github.com/NVIDIA/NeMo-Guardrails (fetched)
13. C. Huyen, "AI Engineering" table of contents (Ch. 3-4 evaluation, Ch. 9 inference optimization, Ch. 10 guardrails and user feedback), GitHub, 2025. https://github.com/chiphuyen/aie-book/blob/main/ToC.md (fetched; Ch. 10 guardrails, router + gateway, caches, user feedback)
14. C. Huyen, "Building A Generative AI Platform", huyenchip.com, 2024. https://huyenchip.com/2024/07/25/genai-platform.html (ch-ref)
15. L. Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), NeurIPS / arXiv, 2022. https://arxiv.org/abs/2203.02155 (known; not fetched)
16. Y. Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", arXiv, 2022. https://arxiv.org/abs/2204.05862 (known; not fetched)
17. Y. Bai et al., "Constitutional AI: Harmlessness from AI Feedback", arXiv, 2022. https://arxiv.org/abs/2212.08073 (known; not fetched)
18. R. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS / arXiv, 2023. https://arxiv.org/abs/2305.18290 (known; not fetched)
19. Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", ICML / arXiv, 2022. https://arxiv.org/abs/2211.17192 (known; not fetched)
20. G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI, 2022. https://www.usenix.org/conference/osdi22/presentation/yu (known; not fetched)
21. Y. Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI / arXiv, 2024. https://arxiv.org/abs/2401.09670 (known; not fetched)
22. H. Inan et al., "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations", arXiv, 2023. https://arxiv.org/abs/2312.06674 (known; not fetched)
23. OpenAI, "Introducing ChatGPT", openai.com, 2022. https://openai.com/index/chatgpt/ (known; not fetched)
24. H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", arXiv, 2023. https://arxiv.org/abs/2307.09288 (known; not fetched)
25. D. Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned", arXiv, 2022. https://arxiv.org/abs/2209.07858 (known; not fetched)
26. A. Holtzman et al., "The Curious Case of Neural Text Degeneration" (nucleus sampling), ICLR / arXiv, 2019. https://arxiv.org/abs/1904.09751 (known; not fetched)
27. T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS / arXiv, 2022. https://arxiv.org/abs/2205.14135 (known; not fetched)
28. J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP / arXiv, 2023. https://arxiv.org/abs/2305.13245 (known; not fetched)
29. R. Pope et al., "Efficiently Scaling Transformer Inference", MLSys / arXiv, 2022. https://arxiv.org/abs/2211.05102 (known; not fetched)
30. Educative, "Grokking the Generative AI System Design", educative.io, 2025. https://www.educative.io/courses/generative-ai-system-design (link from alirezadir genai-resources; not fetched)
31. Educative, "LLMOps", educative.io, 2025. https://www.educative.io/courses/llmops (link from alirezadir genai-resources; not fetched)
32. IGotAnOffer, "OpenAI System Design Interviews", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/openai-system-design-interview (ch-ref)
33. IGotAnOffer, "Generative AI System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/generative-ai-system-design-interview (ch-ref)
34. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
35. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
36. E. Yan et al., "Applied LLMs: What We've Learned From A Year of Building with LLMs", applied-llms.org, 2024. https://applied-llms.org/ (ch-ref)
37. W.-L. Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference", ICML / arXiv, 2024. https://arxiv.org/abs/2403.04132 (known; not fetched)
38. Hugging Face, "text-generation-inference" repository (maintenance mode; archived 21 Mar 2026), GitHub, 2026. https://github.com/huggingface/text-generation-inference (fetched)
39. A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (chunked prefill), OSDI / arXiv, 2024. https://arxiv.org/abs/2403.02310 (known; not fetched)
