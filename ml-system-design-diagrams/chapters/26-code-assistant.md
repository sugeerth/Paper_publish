# 26. Design an AI Code Completion / Coding Assistant (GitHub Copilot, Cursor)

> Prompt as usually given: "Design GitHub Copilot: suggest the next line(s) of code as the
> developer types in the IDE", or "Design Cursor Tab: predict the next edit, not only the
> next token". Variants: "Design inline code completion for an internal monorepo" (Google,
> Meta CodeCompose), "Design a chat-with-your-codebase assistant", "Design an agent that
> fixes a GitHub issue end to end" (SWE-bench style), "Design Gmail Smart Compose" (the
> text analog in the ByteByteGo GenAI book).

> Verification note: in both the compile and the review pass, outbound fetches
> (arxiv.org, github.blog, cursor.com, research blogs, portal sites) failed DNS or proxy
> policy, and the web-search budget was spent. No URL was fetched live. (ch-ref) = verified
> in an earlier chapter; (known; not fetched) = well-known paper or post cited from prior
> knowledge, so check exact numbers before quoting them. "Typical assumption" = interview
> estimate.

## Company tags

| Company | Basis | Production system it mirrors |
|---|---|---|
| GitHub / Microsoft | Builds Copilot; published prompt-construction, neighboring-tabs and FIM details [4][5]; productivity study [3] | Copilot: IDE client builds prompt -> proxy -> Codex/GPT FIM model |
| Cursor (Anysphere) | Builds Cursor Tab (next-edit prediction), online RL for Tab [7], fast-apply model [31], codebase indexing [32] | Small fast edit model + policy-gradient updates from accept/reject; separate apply model |
| Google | Published hybrid semantic-engine + transformer completion for internal IDEs [6][21] | ~0.5B model; ~3% of new code and 6% shorter iteration time in 2022 [6]; later reports a large share of characters AI-completed [21] |
| Meta | Published CodeCompose deployment for internal monorepo [15] | InCoder-based [20] 1.3B FIM model fine-tuned on first-party code; ~22% accept rate, ~8% of code [15] |
| Sourcegraph | Published lifecycle of a Cody completion (context, debounce, post-processing) [16] | Retrieval of repo context + LLM + truncation heuristics |
| OpenAI | Codex paper is the origin of the problem framing (HumanEval, pass@k) [1][2] | Codex model family behind early Copilot |

**Portal coverage.**

| Portal / source | What it covers for this problem | Link |
|---|---|---|
| ByteByteGo / Aminian & Sheng (GenAI System Design Interview) | No Copilot chapter; "Gmail Smart Compose" chapter is the closest template (prefix -> LM -> beam search -> triggering, latency) [23][33]; the first ML SD book has no code chapter either [24] | [23][24] |
| IGotAnOffer | Generic GenAI system design guide (LLM-product skeleton); no code-assistant walkthrough confirmed [25] | [25] |
| Chip Huyen, "Building a GenAI platform" / AI Engineering | Context construction, cache, router, guardrails, eval - reused for code assistants [26] | [26] |
| Hello Interview / Exponent / Educative | Generic MLSD framework; GenAI questions use the same skeleton [27][28][29] | [27][28][29] |
| alirezadir ML-Interviews | MLSD template and question list (LLM/GenAI section) [30] | [30] |
| Copilot reverse engineering (Thakkar) | Most-cited "how Copilot really works" write-up used in interview prep [8] | [8] |

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional (inline) | Ghost-text completion of the rest of the line or a block at the cursor; accept with Tab, partial accept, cycle alternatives | [4][8] |
| Functional (next edit) | Predict the next edit anywhere nearby (rename propagation, cursor jump); Copilot "next edit suggestions" is the same product shape | [7] |
| Functional (apply) | Merge an LLM-written edit sketch into the full file quickly (fast apply) | [31] |
| Functional (chat / agent) | Explain, refactor, generate tests, multi-file edits, run tools (terminal, tests) | [12] typical assumption |
| Context | Prefix + suffix of current file, open tabs, imports, repo symbols, recent edits | [4][8][16] |
| Users | ~1-20M daily developers; ~1-3 completion requests per developer-minute while typing | typical assumption |
| QPS | 10M devs x ~200 requests/day -> ~2B/day, ~25k avg, ~75k peak QPS | typical assumption |
| Requests (Cursor) | Cursor reports 400M+ Tab requests per day [7] | [7] |
| Latency | Inline: p50 < 200-300 ms, p95 < 500 ms end to end (else user types past it); chat: first token < 1 s | [6] typical assumption |
| Quality | Acceptance rate (shown -> accepted) and retained characters after N seconds are north-star | [3][6] |
| Safety | No secrets / PII in suggestions; public-code match filter (Copilot checks ~150-char matches and can block or cite them) [34]; block insecure patterns; respect content exclusions | [34] typical assumption |
| Privacy | Enterprise: no training on customer code, zero retention, regional processing | typical assumption |
| Cost | GPU cost per request must be tiny: small model for inline, large model for chat | typical assumption |

## Diagram 1 - Online serving path

**Inline completion: keystroke -> context retrieval -> prompt -> FIM model -> filters -> ghost text.**

```
+------------------------------------------------------------------------------------+
| IDE CLIENT  (VS Code / JetBrains / Cursor extension)                               |
|                                                                                    |
|  keystroke --> trigger check --> debounce ------> client cache hit? --yes--> show  |
|               (contextual filter:  ~50-150 ms,          | no                       |
|                mid-word? comment?) cancel stale req     v                          |
|  +----------------------------+     +----------------------------------------+     |
|  | context gatherer           |---->| prompt builder (token budget packing)  |     |
|  | prefix, suffix, open tabs, |     | path + lang, snippets, imports, then   |     |
|  | recent edits, local index  |     | <PRE> prefix <SUF> suffix <MID>        |     |
|  +----------------------------+     +-------------------+--------------------+     |
|                                                         |                          |
|  telemetry: shown / accepted / partial / rejected;      |                          |
|  retained-text checks at 15s, 30s, 2m, 5m, 10m [8]      |                          |
+----------+----------------------------------------------|--------------------------+
           |                                              | HTTPS/2 stream,
           | events (opt-in)                              | cancel on next keypress
           v                                              v
+---------------------+     +------------------------------------------------------+
| telemetry pipeline  |     | API gateway / proxy: auth, rate limit, tenant policy,|
| event log -> joins  |     | content exclusion, secret redaction                  |
| -> Diagram 2        |     +---------------------------+--------------------------+
+---------------------+                                 |
                                                        v
+--------------------------+  top-k   +------------------------------------------+
| repo context service     |  chunks  | router: inline -> small FIM model        |
| server-side index:       |--------->|         next-edit -> edit model          |
| symbols, embeddings,     |          |         chat / agent -> frontier LLM     |
| incremental re-index     |          +---------------------+--------------------+
+--------------------------+                                |
                                                            v
                              +-----------------------------------------------------+
                              | inference cluster (GPU)                             |
                              | prefix/KV cache reuse across keystrokes, continuous |
                              | batching, speculative decoding, n=1-3 samples       |
                              +--------------------------+--------------------------+
                                                         | k candidates + logprobs
                                                         v
                              +-----------------------------------------------------+
                              | post-processing: stop at block end / suffix overlap,|
                              | dedupe, syntax check, mean-logprob rank, public-    |
                              | code match filter, security / secret filter         |
                              +--------------------------+--------------------------+
                                                         |
                                     best 1 (+ alternates) streamed back to client
```

## Diagram 2 - Offline data, training & evaluation loop

**From public + permitted code to a deployed FIM model, with an accept/reject feedback loop.**

```
 +----------------+   +-----------------+   +-------------------+   +-----------------+
 | public code    |-->| license filter, |-->| dedup (exact +    |-->| pretraining     |
 | (permissive),  |   | PII/secret scrub|   | MinHash near-dup),|   | corpus (tokens, |
 | docs, issues   |   | opt-out removal |   | quality filters   |   | FIM transform)  |
 +----------------+   +-----------------+   +-------------------+   +--------+--------+
                                                                             |
                                                                             v
 +--------------------+    +----------------------+    +----------------------------+
 | first-party / opt- |--->| SFT data: realistic  |--->| train / fine-tune          |
 | in repos (enterp.) |    | cursor positions,    |    | base LM: next token + FIM  |
 +--------------------+    | prompt format = prod |    | then SFT on prod-format    |
                           +----------------------+    | then RL / DPO on accepts   |
                                      ^                +-------------+--------------+
                                      |                              |
 +--------------------+    +----------+-----------+                  v
 | IDE telemetry      |--->| label join:          |    +----------------------------+
 | shown / accepted / |    | accepted+retained=+1 |    | offline eval               |
 | rejected / edits   |    | rejected/deleted=-1  |    | HumanEval/MBPP pass@k,     |
 | (opt-in only)      |    | edit-after = target  |    | repo-level (CrossCodeEval),|
 +---------+----------+    +----------------------+    | replay acceptance proxy,   |
           ^                                           | latency + size budget      |
           |                                           +-------------+--------------+
           |                                                         | pass gates
           |                                                         v
           |       +--------------------+    +-------------------------------+
           |       | online A/B: accept |<---| model registry -> distill /   |
           +-------| rate, retained chr |    | quantize -> canary 1% -> ramp |
                   | latency, cancels   |    +-------------------------------+
                   +--------------------+
                     feedback loop (Cursor: new Tab policy shipped multiple times
                     per day from fresh on-policy accept/reject data [7])
```

## Diagram 3 - Model architecture

**Decoder-only transformer with fill-in-the-middle prompt; retrieval augments the prefix.**

```
  INPUTS (packed into ~2k-8k token budget, priority order)
  +-------------------------------------------------------------------------------+
  | 1 file path + language   2 retrieved snippets (neighbor tabs / repo index)    |
  | 3 imports / symbol defs  4 recent edits diff    5 PREFIX   6 SUFFIX           |
  +---------------------------------------+---------------------------------------+
                                          |
                                          v
       PSM format:  <PRE> ctx + prefix <SUF> suffix <MID>  -->  generate middle
                                          |
                                          v
  +-------------------------------------------------------------------------------+
  | tokenizer (code BPE, whitespace-aware)                                        |
  +---------------------------------------+---------------------------------------+
                                          v
  +-------------------------------------------------------------------------------+
  | decoder-only transformer  (inline: 1-15B, distilled; chat: frontier LLM)      |
  | causal self-attn x N layers, MQA/GQA for small KV cache, RoPE long context    |
  +---------------------------------------+---------------------------------------+
                                          v
  +-------------------------------------------------------------------------------+
  | decoding: greedy / low-temp sampling, n=1-3 candidates, max ~64-256 tokens,   |
  | stop on <EOT>, block end, or suffix overlap; speculative draft model          |
  +---------------------------------------+---------------------------------------+
                                          v
  OUTPUTS: middle text + per-token logprobs --> show/no-show decision (threshold on
           mean logprob or a small "will-accept" classifier on context features)

  Training objective: next-token CE on [PRE..SUF..MID] (FIM rate ~50-90%) [2]
  Optional: reward model / policy gradient on accept vs reject (Cursor Tab) [7]
```

## Diagram 4 - Variant: next-edit prediction and agentic coding (Cursor-style)

**Two products on one platform: low-latency edit model + high-capability agent loop.**

```
+---------------------- NEXT-EDIT (Cursor Tab, Copilot NES) ----------------------+
| recent edit history + cursor position + nearby code                             |
|   --> small fast edit model --> predicted diff (+ cursor jump location)         |
|   --> show only if p(accept) high (online RL: fewer suggestions shown,          |
|       higher accept rate [7])                                                   |
+---------------------------------------------------------------------------------+

+---------------------- AGENT (chat / multi-file edits) --------------------------+
|  user task                                                                      |
|     |                                                                           |
|     v                                                                           |
|  +--------------+  plan   +---------------+  tool calls  +-------------------+  |
|  | frontier LLM |-------->| tool router   |------------->| codebase search   |  |
|  | + system     |<--------| (read, grep,  |<-- results --| (embeddings index,|  |
|  |   prompt     | observe |  edit, run)   |              | grep), run tests, |  |
|  +------+-------+         +-------+-------+              | lint, terminal    |  |
|         |                         | edit intent          +-------------------+  |
|         |                         v                                             |
|         |                 +---------------------+                               |
|         |                 | apply model: merge  |                               |
|         |                 | sketch into file,   |                               |
|         |                 | speculative edits   |                               |
|         |                 | (file = draft) [31] |                               |
|         |                 +---------------------+                               |
|         | loop until tests pass / step or token budget hit                      |
|         v                                                                       |
|  proposed diff --> human review in IDE --> accept / revert (label)              |
|  sandbox: confirm destructive commands; untrusted repo text = prompt injection  |
|  eval: SWE-bench (Verified) "resolve issue, hidden tests pass" [12]             |
+---------------------------------------------------------------------------------+
```

## Data & labels

| Source | Use | Label / signal | Caveat |
|---|---|---|---|
| Public permissive code (GitHub) | Pretraining | Next token / FIM middle (self-supervised) | License, dedup, test-set contamination [1][9] |
| Synthetic FIM splits | Infill training | Random span = target middle | Split at realistic points (line, block, AST node) |
| First-party monorepo | Fine-tune (Google, Meta) | Next token on internal code | Internal APIs; leakage across teams [6][15] |
| IDE telemetry (opt-in) | Online metrics, RL | Accepted & retained = positive; rejected / deleted = negative | Position bias, users typing past; strong privacy limits |
| Accepted-then-edited | SFT target | Final code after N seconds | Noisy; edit may be unrelated |
| Unit tests (HumanEval, MBPP) | Offline eval | pass@k | Function-level only, contaminated [1][22] |
| Repo-level benchmarks | Offline eval | Exact match / edit sim with cross-file context | [11] |

## Features / context signals

| Signal | Where | Why |
|---|---|---|
| Prefix and suffix around cursor | Client | Core FIM input [2] |
| File path, language, imports | Client | Disambiguates APIs; cheap and high value [4][8] |
| Neighboring tabs snippets (sliding window, Jaccard sim) | Client | Copilot's "neighboring tabs" improved acceptance [4][8] |
| Repo retrieval (symbols, embeddings, BM25) | Server / local index | Cross-file APIs; RepoCoder iterative retrieval [10] |
| Recent edits diff | Client | Next-edit prediction [7] |
| Cursor position type (mid-line, after comment, empty line) | Client | Trigger / show decision (contextual filter) [8] |
| User history accept rate, typing speed | Client | Show threshold, debounce tuning [8] |
| Semantic engine results (types, members) | Language server | Google hybrid: filter/rank by compiler info [6] |
| Codebase embeddings index, synced by file-hash Merkle tree | Server, refreshed from client | Chat / agent retrieval without re-uploading the whole repo [32] |

## Model choices (baseline -> advanced)

| Stage | Option | Pros | Cons |
|---|---|---|---|
| Baseline | N-gram / language-server symbol completion | Instant, exact | Single tokens only |
| v1 | Pretrained code LM, left-to-right only | Simple | Ignores suffix; poor mid-file |
| v2 | FIM-trained decoder (Codex/StarCoder/Code Llama/DeepSeek-Coder/Qwen-Coder) [2][9][13][17][18] | Uses suffix; standard today | Needs realistic FIM splits |
| v3 | + retrieval of repo context [4][10] | Cross-file correctness | Token budget, retrieval latency |
| v4 | Distill / quantize large -> small for inline | Meets <300 ms | Some quality loss |
| v5 | Fine-tune on accept/reject (DPO / online RL) [7] | Optimizes what users value | Feedback loop bias, reward hacking (show less) |
| Chat / agent | Frontier LLM + tools + agent loop [12] | Multi-file tasks | Seconds-minutes latency, cost |

## Loss / objectives

| Objective | Formula (words) | Notes |
|---|---|---|
| Causal LM | Cross-entropy of next token | Pretraining |
| FIM | Same CE on permuted PRE/SUF/MID document; FIM rate up to ~90% without hurting left-to-right [2] | Key for mid-file completion |
| SFT on prod prompts | CE on middle only, prompt format equals production | Closes train/serve skew |
| Show/no-show classifier | Binary log loss on accepted vs not | Controls noise and cost |
| Preference / RL | DPO on (accepted, rejected) pairs, or policy gradient with reward = accept - penalty for rejected show [7] | Must penalize showing bad suggestions |
| Distillation | KL between teacher and student token distributions | For small inline model |

## Offline metrics

| Metric | Measures | Note |
|---|---|---|
| pass@k (HumanEval, MBPP) | Functional correctness of k samples | Unbiased estimator in [1]; contaminated |
| Exact match / edit similarity of middle | Token-level fidelity on held-out repos | Cheap; misses equivalent code |
| Cross-file EM (CrossCodeEval) [11] | Retrieval usefulness | Repo-level |
| Replay acceptance proxy | Predict accept on logged contexts | Off-policy, biased |
| Syntax / compile rate | Parse and type-check success | Uses language server |
| Latency, tokens/s, model size | Serving fit | Hard gate |
| SWE-bench resolve rate [12] | Agent end-to-end | For agent variant; use the human-validated Verified subset |
| Contamination-free eval (LiveCodeBench) [35] | Problems released after model cutoff | Fixes HumanEval leakage |

## Online metrics / A-B

| Metric | Definition | Direction |
|---|---|---|
| Acceptance rate | accepted / shown | Up; strongest predictor of perceived productivity [3] |
| Retained characters | Chars of accepted code still present after 30s-10m | Up; guards against accept-then-delete [3][8] |
| Fraction of code written by AI | AI chars / total chars committed | Up [6] |
| Shown rate / suggestions per hour | Volume | Not blindly up (noise) [7] |
| Cancel / timeout rate | Request cancelled by keystroke before response | Down |
| p50/p95 latency | Keystroke -> ghost text | Down |
| Guardrails | Secret leaks, public-code matches, vuln patterns | ~0 |
| Long-term | Retention, DAU, survey (SPACE), PR throughput | Up |

## Serving & latency budget (inline, p95 ~400 ms; typical assumption)

| Step | Budget | Technique |
|---|---|---|
| Debounce + trigger | 50-150 ms (not counted in model time) | Skip mid-word, cancel stale requests |
| Client context + prompt build | 10-20 ms | Precomputed local index, Jaccard windows [8] |
| Network to region | 30-60 ms | Edge regions, HTTP/2 streaming, keep-alive |
| Gateway, auth, filters | 10 ms | In-memory policy cache |
| Repo retrieval (optional) | 20-40 ms | Precomputed index; can be client-side |
| Prefill (2-8k tokens) | 30-80 ms | Prefix/KV cache reuse across keystrokes, GQA |
| Decode (~20-60 tokens) | 50-150 ms | Small model, speculative decoding [14], int8/fp8, continuous batching (vLLM) [19] |
| Post-processing + filters | 10-20 ms | Truncation at block end, public-code filter |
| Apply edit (chat/agent only) | ~1 s for a few hundred lines | Speculative edits: the existing file is the draft, so most tokens are verified, not generated [31] |

## Monitoring

| What | Signal | Action |
|---|---|---|
| Quality | Accept rate, retained chars by language / IDE / tenant | Alert on drop; roll back model |
| Latency | p50/p95/p99, timeout and cancel rate | Autoscale GPUs, shrink context |
| Cost | GPU-sec per accepted char | Route, cache, tune show threshold |
| Safety | Secret detector hits, public-code match rate, insecure patterns | Block, retrain filter |
| Drift | New languages / frameworks, library version changes | Refresh fine-tune data |
| Client health | Extension crash rate, version skew of prompt format | Force-upgrade, server-side compat |

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| Suggestion arrives too late | Long prompt, cold KV cache | Cancel-on-keystroke, prefix cache, smaller model |
| Repeats suffix / overlaps existing code | No FIM or bad truncation | FIM training, suffix-overlap trimming [16] |
| Hallucinated APIs | Missing cross-file context | Repo retrieval, language-server validation [6][10] |
| Verbatim license-sensitive code | Memorization | Dedup training data, public-code match filter |
| Secret / PII leakage | Secrets in training data or context | Scrub data, redact prompts, output detector |
| Insecure code (SQLi, weak crypto) | Learned from public code | Security filter, static analysis on suggestion |
| Distracting noise | Showing low-confidence suggestions | Show/no-show model; RL penalizing rejects [7] |
| Accept-rate gaming | Optimizing short trivial completions | Track retained chars and AI-written fraction |
| Train/serve skew | Prompt format in training differs from client | SFT on logged production prompts |
| Prompt injection (agent) | Malicious repo files / README | Sandbox tools, confirm destructive actions |

## Interviewer follow-ups

- How do you decide when to trigger a request and when to show a suggestion at all?
- How do you build the prompt under a fixed token budget? What gets priority?
- Why fill-in-the-middle, and how do you create FIM training examples? [2]
- How do you bring in cross-file context with <50 ms extra latency? Client vs server index?
- Acceptance rate vs retained characters vs pass@k: which is your north star and why? [3]
- How would you hit p95 < 500 ms at ~75k QPS? Batching, KV cache reuse, speculative decoding?
- One model or two (small inline + large chat)? How do you route?
- How do you use accept/reject telemetry to improve the model without bias or reward hacking? [7]
- How do you avoid regurgitating licensed code or leaking secrets?
- How do you support enterprise customers who forbid training on their code?
- How would you evaluate a new model offline when HumanEval is contaminated?
- How do you handle a newly released library the model has never seen?
- How would you extend this to next-edit prediction or an autonomous coding agent? [7][12]
- How do you keep client prompt format and server model in sync across extension versions?
- What do you log, and what is the privacy story?
- How do you index a large repo for retrieval and keep it fresh as files change? [32]
- How do you apply a model-written edit to a 1,000-line file in about a second? [31]
- How do you detect and handle suggestions that match public code (block vs cite)? [34]

## Portal comparison

| Portal / source | Framing | Emphasis | Gap |
|---|---|---|---|
| ByteByteGo GenAI book (Smart Compose chapter) [23] | Prefix-only text completion | Triggering, beam search, latency, perplexity/ExactMatch | No suffix/FIM, no repo context |
| Copilot reverse-engineering (Thakkar) [8] | Real system teardown | Prompt wishlist, neighboring tabs, contextual filter, retained-text telemetry | Client-side only; no training |
| GitHub engineering blog [4][5] | Vendor view | Prompt library, FIM, neighbor tabs, model iteration | Light on numbers and infra |
| Google Research blog [6] | Internal monorepo | Hybrid semantic engine + small LM, acceptance and % code written | Single company setting |
| Cursor blog [7][31][32] | Next-edit + online RL, fast apply, indexing | Policy gradient on accept/reject, speculative edits, Merkle-tree sync | Few infra numbers |
| Meta CodeCompose paper [15] | Internal monorepo | Fine-tuning on first-party code, rollout, mixed-methods eval | Single company setting |
| Chip Huyen / generic GenAI frameworks [26][27][28] | Generic LLM app | Context construction, caching, guardrails, eval | Not code-specific |

## References

1. M. Chen et al. (OpenAI), "Evaluating Large Language Models Trained on Code" (Codex, HumanEval, pass@k), arXiv, 2021. https://arxiv.org/abs/2107.03374 (known; not fetched)
2. M. Bavarian et al. (OpenAI), "Efficient Training of Language Models to Fill in the Middle", arXiv, 2022. https://arxiv.org/abs/2207.14255 (known; not fetched)
3. A. Ziegler et al. (GitHub), "Productivity Assessment of Neural Code Completion", MAPS / arXiv, 2022. https://arxiv.org/abs/2205.06537 (known; not fetched)
4. GitHub Blog, "How GitHub Copilot is getting better at understanding your code", github.blog, 2023. https://github.blog/2023-05-17-how-github-copilot-is-getting-better-at-understanding-your-code/ (known; not fetched)
5. GitHub Blog, "Inside GitHub: Working with the LLMs behind GitHub Copilot", github.blog, 2023. https://github.blog/2023-05-17-inside-github-working-with-the-llms-behind-github-copilot/ (known; not fetched)
6. Google Research, "ML-Enhanced Code Completion Improves Developer Productivity", Google AI Blog, 2022. https://research.google/blog/ml-enhanced-code-completion-improves-developer-productivity/ (known; not fetched; old ai.googleblog.com URL redirects here)
7. Cursor, "Improving Cursor Tab with online RL", cursor.com blog, Sep 2025. https://cursor.com/blog/tab-rl (known; not fetched)
8. P. Thakkar, "Copilot Internals", copilot-explorer (GitHub Pages), 2022. https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals (known; not fetched)
9. R. Li et al. (BigCode), "StarCoder: may the source be with you!", arXiv, 2023. https://arxiv.org/abs/2305.06161 (known; not fetched)
10. F. Zhang et al., "RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation", EMNLP / arXiv, 2023. https://arxiv.org/abs/2303.12570 (known; not fetched)
11. Y. Ding et al., "CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion", NeurIPS / arXiv, 2023. https://arxiv.org/abs/2310.11248 (known; not fetched)
12. C. E. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", ICLR / arXiv, 2023. https://arxiv.org/abs/2310.06770 (known; not fetched)
13. B. Roziere et al. (Meta), "Code Llama: Open Foundation Models for Code", arXiv, 2023. https://arxiv.org/abs/2308.12950 (known; not fetched)
14. Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", ICML / arXiv, 2022. https://arxiv.org/abs/2211.17192 (known; not fetched)
15. V. Murali et al. (Meta), "AI-assisted Code Authoring at Scale: Fine-tuning, deploying, and mixed methods evaluation" (CodeCompose), arXiv, 2023. https://arxiv.org/abs/2305.12050 (known; not fetched)
16. Sourcegraph, "The lifecycle of a code AI completion", sourcegraph.com blog, 2023. https://sourcegraph.com/blog/the-lifecycle-of-a-code-ai-completion (known; not fetched)
17. D. Guo et al., "DeepSeek-Coder: When the Large Language Model Meets Programming", arXiv, 2024. https://arxiv.org/abs/2401.14196 (known; not fetched)
18. B. Hui et al., "Qwen2.5-Coder Technical Report", arXiv, 2024. https://arxiv.org/abs/2409.12186 (known; not fetched)
19. W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM), SOSP / arXiv, 2023. https://arxiv.org/abs/2309.06180 (known; not fetched)
20. D. Fried et al., "InCoder: A Generative Model for Code Infilling and Synthesis", ICLR / arXiv, 2022. https://arxiv.org/abs/2204.05999 (known; not fetched)
21. Google Research, "AI in software engineering at Google: Progress and the path ahead", research.google blog, 2024. https://research.google/blog/ai-in-software-engineering-at-google-progress-and-the-path-ahead/ (known; not fetched)
22. J. Austin et al., "Program Synthesis with Large Language Models" (MBPP), arXiv, 2021. https://arxiv.org/abs/2108.07732 (known; not fetched)
23. A. Aminian, H. Sheng, "Generative AI System Design Interview" (incl. Gmail Smart Compose chapter), ByteByteGo, 2024. https://bytebytego.com/ (site root; book page not fetched)
24. ByteByteGo, "ml-bytebytego: Reference materials for ML System Design Interview" (chapter list; no code-assistant chapter), GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (ch-ref)
25. IGotAnOffer, "Generative AI System Design Interview", igotanoffer.com, 2025. https://igotanoffer.com/en/advice/generative-ai-system-design-interview (ch-ref)
26. C. Huyen, "Building A Generative AI Platform", huyenchip.com, 2024. https://huyenchip.com/2024/07/25/genai-platform.html (known; not fetched)
27. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
28. Exponent, "Introduction to the ML System Design Interview" (course), tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
29. Educative, "The 6-step ML SD framework" (Grokking the ML System Design Interview), educative.io, 2025. https://www.educative.io/courses/grokking-the-machine-learning-system-design-interview/the-6-step-ml-sd-framework (ch-ref)
30. alirezadir, "ML System Design" (template and sample questions), Machine-Learning-Interviews, GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (ch-ref)
31. Cursor, "Editing Files at 1000 Tokens per Second" (fast apply, speculative edits), cursor.com blog, 2024. https://cursor.com/blog/instant-apply (known; not fetched)
32. Cursor, "Security" (codebase indexing: embeddings, Merkle tree of file hashes), cursor.com, 2025. https://cursor.com/security (known; not fetched)
33. M. X. Chen et al. (Google), "Gmail Smart Compose: Real-Time Assisted Writing", KDD / arXiv, 2019. https://arxiv.org/abs/1906.00080 (known; not fetched)
34. GitHub Docs, "Finding public code that matches GitHub Copilot suggestions" (code referencing / duplication filter), docs.github.com, 2025. https://docs.github.com/en/copilot (docs root; page title known, deep path not verified)
35. N. Jain et al., "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", arXiv, 2024. https://arxiv.org/abs/2403.07974 (known; not fetched)
