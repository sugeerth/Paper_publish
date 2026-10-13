# 18. Agent System Design Interview Playbook: Coding, Customer Support, Research and Browser Agents

> **Sourcing note.** The search engine budget ran out before this chapter's research began, and several hosts could not be reached from the research environment (openai.com, cdn.openai.com, arxiv.org, cognition.ai, manus.im, sierra.ai, stripe.dev, builders.ramp.com, huyenchip.com). The following were read directly for this chapter: Anthropic engineering and research posts, the Claude Platform documentation (pricing, prompt caching, task budgets, the support-agent guide), Claude Code documentation, claude.com blog posts, and the SWE-agent repository. Claims about OpenAI, Sierra, Decagon, Fin, Klarna, Salesforce, Stripe, Ramp, Manus, Cognition, Browser Use, Stagehand and the open deep-research projects come from sources that earlier chapters of this Atlas read through GitHub mirrors or repositories (Chapters 04, 05, 06, 07, 08, 15 and 16). They are cited by original URL. Vendor-reported numbers are marked as such. All prices are list prices as published in October 2026 and will change.

## Overview

"Design an agent that..." has become a standard system design interview prompt, and the same question gets asked in real design reviews. It is harder than a classic distributed-systems question for three reasons:

- **The core component is stochastic.** An LLM call can be correct 95% of the time and still sink a 10-step task. If each step succeeds independently with probability 0.95, ten steps all succeed only about 60% of the time (0.95^10 is about 0.60).
- **The workload is token-bound, not request-bound.** Anthropic reports that agents use about 4x the tokens of chat, and multi-agent systems about 15x [2]. Manus reports an input-to-output token ratio of about 100:1 in its agent [29]. Capacity planning therefore runs on tokens, prefix-cache hit rate and tool latency, not QPS alone.
- **Outputs cause side effects.** Agents edit code, issue refunds, submit forms and spend money. Failure handling must cover bad *actions*, not only bad answers.

This chapter is a working playbook for four canonical prompts: a **coding agent**, a **customer support agent**, a **research agent** and a **browser agent**. Each design starts from requirements and works through components, data flow, failure handling and cost/latency sizing, grounded in what production teams have published. Section 18.1 gives a general method. Section 18.2 gives a sizing toolkit with current prices. Sections 18.3 to 18.6 cover the four designs. Section 18.7 collects failure handling into one playbook, and Section 18.8 compares the designs.

## 18.1 A method for the interview

### Step 1: Clarify the problem before drawing boxes

Strong candidates ask questions that change the architecture:

1. **Who triggers the agent, and who watches?** An attended IDE agent, an unattended background agent and a customer-facing chat agent have very different latency and safety budgets. Anthropic measured that the median Claude Code turn lasts about 45 seconds, while the 99.9th percentile grew from under 25 minutes to over 45 minutes between late 2025 and early 2026 [13]. A design must cover both ends of that distribution.
2. **What side effects are allowed, and are they reversible?** In Anthropic's analysis of about one million API tool calls, 0.8% appeared irreversible [13]. Those are the calls that need gates.
3. **What does success mean, and can it be checked by a program?** Coding agents suit agentic designs because "solutions can be verified with automated tests" [1]. Support success is "measurable through user-defined resolutions" [1]. Research quality needs rubric grading. Browser success can usually be checked from URL, page or backend state [14].
4. **What is the volume, the latency SLO and the cost ceiling per task?**

### Step 2: Pick the lowest rung on the complexity ladder

Anthropic's "Building effective agents" separates **workflows**, where LLMs and tools follow predefined code paths, from **agents**, where the LLM directs its own process and tool use. It advises starting with the simplest solution, noting that agentic systems "trade latency and cost for better task performance" [1]. Name the rung you choose and justify it:

| Rung | Control flow | Typical cost multiple | Use when | Example |
|---|---|---|---|---|
| Single augmented LLM call | None | 1x | Retrieval plus examples suffice | FAQ answer with RAG [19] |
| Workflow (chain, router, parallel, evaluator-optimizer) | Code | Low multiple | Steps are known in advance | Agentless: localize, repair, validate [24] |
| Single agent loop | Model | About 4x chat tokens [2] | Open-ended, verifiable tasks | Claude Code [11], mini-SWE-agent [25] |
| Orchestrator plus parallel workers | Model, hierarchical | About 15x chat tokens [2] | Breadth-first, read-heavy, parallelizable | Claude Research [2] |
| Hybrid: deterministic blueprint with agent nodes | Code plus model | Bounded | High volume, known shape, needs flexibility in steps | Stripe Minions [26][27] |

### Step 3: Draw the universal skeleton, then specialize

Almost every production agent in this Atlas fits one skeleton. Anthropic's Managed Agents post gives the cleanest decomposition: a **session** (an append-only event log outside the context window), a **harness** or "brain" (a stateless loop that calls the model and routes tool calls), and **sandboxes** or "hands" (disposable execution environments behind a uniform `execute(name, input) -> string` interface) [8].

```text
            +-------------------- control plane --------------------------+
 trigger -->| auth, quotas, budgets, routing, versioning, rollout (canary) |
            +------------------------------+------------------------------+
                                           |
   +------------------+          +---------v---------+         +------------------+
   | session log      |<-------->| harness (brain)   |-------->| model gateway    |
   | append-only      | events   | loop, context     |  calls  | routing,failover |
   | durable, resume  |          | policy, compaction|<--------| prefix cache     |
   +------------------+          +---------+---------+         +------------------+
                                           | tool calls
                     +---------------------+----------------------+
                     |                     |                      |
             +-------v------+     +--------v--------+    +--------v---------+
             | policy gate  |     | tools / MCP     |    | sandbox (hands)  |
             | allow, deny, |---->| actionable      |    | fs + network     |
             | ask human    |     | errors, paging  |    | isolation, creds |
             +--------------+     +-----------------+    | via proxy        |
                                                         +------------------+
      observability: traces, transcripts, cost per task, eval hooks, monitors
```

### Step 4: Deep-dive the parts the interviewer cares about

For each design, cover the six questions that separate senior answers: **(a)** what goes into the context and how it is kept small, **(b)** what the tool interface looks like, **(c)** where the agent runs and what it can reach, **(d)** how results are verified, **(e)** how failures are detected and recovered, and **(f)** how much one task costs and how long it takes.

## 18.2 Sizing toolkit: tokens, dollars and seconds

### The cost model

An agent loop re-sends its whole context on every step. If the stable prefix (system prompt plus tool definitions) is `P` tokens and each step adds `d` tokens of tool results and model output, step `i` reads about `P + d*i` input tokens. Total input over `n` steps is about `n*P + d*n^2/2`. Input cost grows **quadratically** with step count, which is why prefix caching, compaction and sub-agent isolation dominate agent economics.

Published price points (USD per million tokens, October 2026) [16][17]:

| Model | Input | 5-min cache write | 1-h cache write | Cache read | Output |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 4.00 | 5.00 | 8.00 | 0.20 | 20.00 |
| Claude Sonnet 5.5 | 2.00 | 2.50 | 4.00 | 0.10 | 10.00 |
| Claude Haiku 5.5 (prompt up to 100k) | 0.10 | 0.125 | 0.20 | 0.01 | 0.50 |

Other sizing facts worth memorizing:

- **Cache mechanics.** Writes cost 1.25x base (5-minute TTL) or 2x (1-hour TTL). Reads cost 0.1x in general and 0.05x on Opus 5.5 and Sonnet 5.5. The TTL is measured from the *start* of the request, so a response that streams for 4 minutes leaves about 1 minute to reuse the prefix. Prefixes are built in the order `tools -> system -> messages`, and changing tool definitions invalidates all three [16]. Manus calls KV-cache hit rate "the single most important metric for a production-stage AI agent" [29].
- **Batch API:** 50% off input and output for work that is not time-sensitive [17].
- **Web search:** $10 per 1,000 searches, plus the tokens of the results [17].
- **Managed agent runtime:** $0.08 per session-hour of `running` time [17].
- **Tool definitions are not free.** Anthropic counted about 55K tokens for 58 tools across five MCP servers, and an internal peak of 134K. Deferred loading through a tool search tool cut one setup from about 77K to 8.7K tokens (about 85%) [5].
- **Screenshots** cost about 1,000 to 1,800 tokens each, so a 200K window fills in under 100 of them [20].

### Worked example: one 50-step coding session

The assumptions are illustrative and only the prices are sourced: Opus 5.5, a 20K-token prefix, 2K new tokens per step, 300 output tokens per step, 50 steps.

- Input tokens: 50 x 20K + 2K x (50 x 51 / 2) = 1.0M + 2.55M, or about **3.55M**.
- Uncached: 3.55M x $4 = **about $14.20** input.
- Cached, with every step inside the TTL: about 3.43M read at $0.20 (about $0.69) plus about 0.12M written at $5 (about $0.60), or **about $1.29** input.
- Output: 15K tokens x $20 = about $0.30.

Caching cuts the session from about $14.50 to about $1.60, roughly a 9x reduction. The model is still the right lever on top of that: the same session on Sonnet 5.5 costs about half as much. Two lessons follow for the interview. First, any design that mutates the prefix mid-session, such as by adding or removing tools, reordering JSON keys or putting timestamps in the system prompt, quietly multiplies cost. Manus therefore keeps every tool defined and masks token logits instead of removing tools [29]. Second, if a tool call can take longer than 5 minutes (a CI run, say), choose the 1-hour TTL or expect cache misses.

### The latency model

Per-step latency is roughly time-to-first-token (driven by uncached prefill) plus output tokens divided by decode speed plus tool execution time. Task latency is that sum over the critical path of steps. The published levers are:

| Lever | Evidence |
|---|---|
| Fewer model round-trips | Programmatic tool calling: 20+ tool calls in one code block remove 19+ inference passes; tokens fell 37% on complex research tasks [5] |
| Parallelism | Parallel subagents and tool calls cut research time by up to 90% [2] |
| Remove LLM hops | Salesforce: a routing-only subagent "adds an extra LLM hop (~3-5s latency)" [41]; cutting pre-first-token LLM calls from four to two, among 30+ changes, cut latency by about 70% (secondary) [42] |
| Small classifiers for routing and guards | HyperClassifier, a single-token SLM, was 30x faster at accuracy parity (secondary) [42] |
| Guards off the critical path | Sierra and Decagon run safety checks in parallel with generation and only gate the release of the reply [33][35] |
| Lazy provisioning | Provisioning containers only when needed cut p50 TTFT about 60% and p95 over 90% [8] |
| Cache pre-warming | A `max_tokens: 0` request removes the cache-miss penalty on the first user turn [16] |
| Hedging and failover | Sierra hedges requests across providers, and "the fastest valid response wins" [32] |
| Custom inference | Fin serves a reasoning model at 3.1 ms per token (from 11.35 ms) at about $0.01 per request, using FP8, 4-GPU tensor parallelism and disaggregated prefill/decode (vendor-reported) [37] |

### Budgets as first-class configuration

Give the agent explicit budgets and enforce them in the harness. Claude's task budgets (beta) give the model a running token countdown for a whole agentic loop, so it can "finish gracefully" instead of being cut off. They are advisory, with `max_tokens` as the hard per-request cap. The minimum is 20,000 tokens, and a budget that is clearly too small "can cause refusal-like behavior" [18]. Size budgets from the measured p99 of per-task spend, not a fixed default [18].

## 18.3 Design 1: A coding agent

**Prompt:** "Design an agent that takes a GitHub issue or a Slack request and produces a mergeable pull request."

### Requirements to pin down

- Attended (IDE or terminal) or unattended (background, many in parallel)? Stripe built unattended agents because "developer attention" was its scarce resource [26].
- Is there a test suite and CI, and how long do they take? Verification quality bounds agent quality.
- Repository scale. Stripe has hundreds of millions of lines of code [26].
- Blast radius: may the agent push to main, call production services or reach the internet?

### Architecture

```text
 issue / Slack / CLI
       |
       v
 [context hydration: fetch linked docs/tickets via MCP, deterministic]   (Stripe)
       |
       v
 +--------------------------- sandbox: devbox / microVM -----------------------+
 |  warm pool (~10 s Stripe, <5 s Ramp snapshots); repo cloned, caches warm     |
 |                                                                              |
 |   harness loop:  gather context --> act (edit) --> verify (lint/test) --+    |
 |        ^                                                                |    |
 |        +-------------------------- feedback ----------------------------+    |
 |   tools: read/view, grep/glob, str_replace edit, bash, LSP diagnostics       |
 |   memory: AGENTS.md / CLAUDE.md rules scoped per directory                   |
 |   network: egress via allowlist proxy; git creds held by proxy, not agent    |
 +------------------------------------------------------------------------------+
       |
       v
 [CI: selected tests] --fail--> agent fix (max 2 rounds) --> PR --> human review
```

### Components and key techniques

**The loop.** Claude Code describes three blended phases: gather context, take action and verify results. It chains "dozens of actions" and course-corrects as it goes [11]. The loop itself is trivial. mini-SWE-agent is about 100 lines and reports above 74% on SWE-bench Verified with frontier models [25]. Interviewers want to hear that **the harness, not the loop, is the product**.

**Agent-computer interface (ACI).** SWE-agent showed that tool design moves results substantially [23]:

- a file viewer showing 100 lines per turn instead of `cat`;
- search that lists only the matching files, because more detail "confused the model";
- edits rejected by a linter when the result is syntactically invalid;
- an explicit message when a command produces no output.

Anthropic's SWE-bench scaffold used only a prompt, bash and a `str_replace` edit tool. The edit applies only if `old_str` matches *exactly once*, and zero or multiple matches return an error so the model retries. The tool requires absolute paths because models "mangled relative paths" after changing directory. This scaffold scored 49% on SWE-bench Verified with Claude 3.5 Sonnet (new), and successful runs sometimes took "hundreds of turns" and more than 100K tokens [12]. General tool rules from Anthropic [4]:

- cap tool responses (Claude Code defaults to 25,000 tokens);
- offer concise and detailed response formats (one example went from about 206 tokens to about 72);
- make error messages actionable;
- namespace tools;
- avoid thin wrappers over API endpoints.

**Context strategy.** Claude Code uses a hybrid approach: project rule files (CLAUDE.md) load up front, and glob and grep fetch everything else "just in time" [3]. As the window fills, it clears old tool outputs first, then summarizes. MCP tool definitions are deferred and loaded through tool search [11]. Subagents explore in a fresh context and return only a summary [11]. Stripe scopes rule files to subdirectories and attaches them as the agent walks the tree, keeping almost no global rules so the window does not fill before work begins [27].

**Isolation and permissions.** This is the decision interviewers probe hardest.

- *OS sandbox.* Claude Code's sandbox limits writes to the working directory and routes network traffic through a proxy that enforces a domain allowlist. It uses bubblewrap on Linux and Seatbelt on macOS. Anthropic's reasoning is that both layers are needed: network isolation alone still allows exfiltration of local files, and filesystem isolation alone allows escape to the network. Sandboxing cut permission prompts by 84% internally [10].
- *Credential proxy.* In cloud sessions, git credentials stay outside the sandbox. A proxy checks each operation, for example that pushes go only to the configured branch, and attaches the real token [10]. Managed Agents applies the same idea to MCP: a session-scoped token fetches credentials from a vault, and "the harness never sees them" [8].
- *Classifier-gated autonomy.* Users approved 93% of permission prompts, which is evidence of approval fatigue [9]. Claude Code's auto mode keeps in-project edits outside the classifier and sends risky actions to a two-stage transcript classifier: a fast single-token filter, then chain-of-thought review of anything flagged. After 3 consecutive or 20 total denials, it escalates to a human [9]. The published error rates are a good example of honest tradeoffs:

| Dataset | Stage 1 only | Full pipeline |
|---|---|---|
| Real internal traffic (n=10,000), false-positive rate | 8.5% | 0.4% |
| Real overeager actions (n=52), false-negative rate | 6.6% | 17% |
| Synthetic exfiltration (n=1,000), false-negative rate | 1.8% | 5.7% |

- *Full permissions in a disposable box.* Stripe runs minions "with full permissions" and no prompts. This is safe only because devboxes sit in a QA environment with no real user data, no production services and no arbitrary egress [27].

**Verification.** Treat tests as the reward signal and bound the loop:

- Stripe runs lint in under 5 seconds before pushing, selects CI tests from more than 3 million, and caps agents at **two CI rounds** because of "diminishing marginal returns" [26][27].
- Agentless shows that test-based selection among sampled patches is a strong component on its own. It resolved 27.3% of SWE-bench Lite at about $0.34 per issue [24].
- For subjective quality, a separate evaluator works better than self-review. Agents "tend to praise their own work", and tuning a skeptical evaluator is "more tractable" than making a generator self-critical [7].

**Long-horizon work.** For multi-session builds, Anthropic splits the work [6]:

- an *initializer* agent writes a JSON feature list (over 200 features in one example), with every entry failing by default;
- *coding* agents each implement one feature per session;
- each session starts by reading `claude-progress.txt` and the git log, runs `init.sh`, and does a quick end-to-end check;
- each session ends with a commit.

JSON was chosen because models are less likely to overwrite it inappropriately than Markdown [6].

### Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Tries to one-shot the whole task; declares victory early | Feature list failing by default; one feature per session; browser-based self-verification [6] |
| Believes it passed when hidden tests fail | Run real tests; reproduction tests; test-based patch selection [12][24] |
| Edit misapplied or ambiguous | Exactly-once `str_replace`; lint-gated edits [12][23] |
| Context exhaustion or thrashing | Tool-result clearing, compaction, subagents; Claude Code stops auto-compacting after repeated attempts rather than looping [11] |
| Prompt injection from repo, web or tools | Input probe on tool outputs; OS sandbox; credentials outside the sandbox [9][10] |
| Runaway CI loops | Hard cap on CI rounds [27]; task budgets [18] |
| Container dies mid-task | Durable session log; `wake(sessionId)` resumes from last event; sandboxes are cattle, not pets [8] |

### Sizing and scale numbers

- Stripe: more than 1,300 merged PRs per week fully written by minions, all human-reviewed; a Toolshed MCP server with nearly 500 tools, from which each minion gets a small subset [27].
- Ramp Inspect: about 30% of merged PRs at publication; a later newsletter reports about 75% by May 2026 and more than one million sessions (secondary) [28].
- Harness cost tradeoff: Anthropic's retro game builder ran 20 minutes for $9 solo versus 6 hours for $200 with a planner, generator and evaluator. The solo app's core feature did not work [7]. A later digital audio workstation (DAW) build cost about $125 over about 3 hours 50 minutes [7]. Be ready to say when 20x cost is justified.

**Lesson.** Harness assumptions go stale. Context resets that helped Sonnet 4.5 became "dead weight" on Opus 4.5 [8]. Remove one harness component at a time when models improve [7].

## 18.4 Design 2: A customer support agent

**Prompt:** "Design an AI agent that resolves customer support conversations for an e-commerce or fintech company, across chat and voice."

### Requirements and metrics

The defining tension is that the agent speaks to an unauthenticated public on behalf of a brand, while writing to systems of record (refunds, cancellations, payments). Sierra frames the scale problem: "At scale, a one-in-ten-thousand hallucination rate is a daily occurrence" [33]. Anthropic's support guide gives concrete target metrics that make good interview anchors [19]:

- 95% or higher query-comprehension accuracy;
- 90% or higher response relevance;
- 100% accuracy on company facts;
- 95% topic adherence;
- 95% or higher escalation accuracy;
- 70-80% deflection;
- CSAT of 4 out of 5 or higher.

Grade consistency, not just best-case success. With pass^k (all k trials succeed), a 75% per-trial success rate gives only about 42% over three trials [14]. Customer-facing agents need pass^k, a metric popularized by Sierra's τ-bench [43].

### Architecture

```text
 chat | email | voice (VAD -> STT ... TTS)
            |
            v
 +----------------------+     in parallel      +--------------------------+
 | input supervisor     |--------------------->| abuse/injection/off-topic|
 | (small model)        |                      | gates the reply release  |
 +----------+-----------+                      +--------------------------+
            v
 [router: single-token SLM classifier -> procedure / FAQ / human]
            |
            v
 +---------------------------------------------------------------+
 | procedure layer (AOP / Journey / Agent Script)                |
 |  NL steps + code conditions; deterministic identity gate      |
 |  tools: get_customer, get_order, refund(amount<=policy) ...   |
 |  RAG: rewrite -> embed -> retrieve -> rerank -> generate      |
 +----------------------------+----------------------------------+
                              v
 [output supervisor: grounding, policy, brand] --ok--> reply
                              |
                              +--fail/low confidence--> handoff (summary -> queue)
 traces --> monitors, LLM-judge QA, simulations, versioned releases, canary
```

### Key techniques

- **A procedure layer between prompts and decision trees.** Sierra Journeys, Decagon AOPs, Fin Procedures and Agentforce Agent Script all express business procedures in natural language with embedded code conditions [35][41]. The Salesforce guide warns that a prose "if" can be ignored by the model, so identity verification uses "deterministic routing ... so the gate cannot be bypassed by LLM conversational drift" [41].
- **Cheap routing.** Put routing in the entry node with a small classifier instead of a routing LLM hop of about 3-5 seconds [41]. Anthropic's guide also suggests an intent classifier in front of specialized conversations, noting that it adds a call and therefore latency [19].
- **Supervisors.** Sierra manages each production agent with several supervisory agents. Input filtering catches threats from "multi-turn context poisoning to advanced jailbreaking". Output interception audits actions and replies and can escalate or end the conversation if a fix would add too much latency [33]. Decagon runs bad-actor detection in parallel with generation and checks grounding and brand "just before delivery" [35].
- **Retrieval latency.** Decagon found reranking to be the largest pre-generation cost in its pipeline [36]. Sierra's Linnaeus retrieval model works on the full transcript to skip a query-rewrite step, saving "up to ~800ms per search". Sierra reports P90 latency and cost each down by more than 75% (vendor-reported) [34].
- **Model reliability.** Sierra's Multi-Model Router keeps an ordered, pre-validated fallback list per task. It refuses fallback when a user-visible stream has already begun or when only one model has the needed capability. An AIMD admission controller, "similar to TCP congestion control", spreads load across providers so a 429 on one provider does not stampede another [31].
- **Voice.** Sierra measures time to first *relevant* audio. Its techniques are a custom VAD, a "concurrent graph, not a sequential pipeline" (abuse detection, retrieval and API calls in parallel), prefetching a known caller's orders, hedged requests, cached common phrases and sentence-by-sentence TTS. It rejects pure voice-to-voice models for most enterprise flows because "you still can't make an API call to a system of record with voice tokens" [32].

### Failure handling

- **Handoff is a first-class path.** Decagon's 2026 post, citing Klarna's reported rehiring for high-judgment cases, argues that "the handoff is every bit as important as the automation" and asks what happens "in the next 30 seconds" when the AI does not know [40]. Klarna's first-month figures were 2.3 million conversations, two-thirds of service chats, and resolution time cut from 11 minutes to under 2 (vendor-reported) [39]. They show that automation rates say little about the residual hard cases.
- **Model text is not proof.** An agent saying "I'm transferring you" is not evidence that a transfer happened. Verify the chain "configured -> available -> invoked -> executed -> effected" [41].
- **Metric hygiene.** Fin bills $0.99 per outcome and treats a conversation as resolved after 24 hours of silence following a real answer (secondary) [38]. Normalize such definitions before comparing vendors.
- **Grade end state.** For support agents, check verifiable outcomes such as ticket status or a processed refund, use rubrics for tone, and use a second LLM to simulate the user [14].

### Sizing

Use Haiku-class models for latency-critical multi-call flows and larger models for complex reasoning. Anthropic's guide recommends Haiku 5.5 when a flow combines several prompts with RAG and tools, with `low` effort as the speed control [19]. At the prices in 18.2, a 10-turn conversation with a 6K-token cached prefix and 1K new tokens per turn reads about 0.1M tokens. That is about a cent on Haiku 5.5 before caching (illustrative arithmetic). The dominant cost at volume is often retrieval, supervisors and human handoffs, not the main model. Fin's self-hosted reasoning model at about $0.01 per request is a reference point for teams that serve their own models (vendor-reported) [37].

## 18.5 Design 3: A research agent

**Prompt:** "Design a deep research agent that answers open-ended questions with a cited report."

### Requirements

- Latency tolerance: Perplexity Deep Research completed most tasks in under three minutes, and OpenAI's took roughly 5 to 30 minutes (both per launch posts) [52][49]. That sets the product category: synchronous chat, or a background job with notification.
- Source scope: public web, internal documents, or both (via MCP connectors).
- Citation guarantees: claims mapped to sources, verifiable.

### Architecture

```text
 user query
    |
    v
 [clarifier / scoping: questions or editable plan] --> research brief
    |
    v
 +-------------------------- lead researcher --------------------------+
 | plan -> saved to external memory (context > 200K would truncate)    |
 | effort rules: simple 1 agent / 3-10 calls; compare 2-4 subagents;   |
 |               complex 10+ subagents                                 |
 +-----+------------------+------------------+---------------------+---+
       | spawn 3-5 in parallel, each with objective, format, tools, bounds
   +---v----+        +----v---+         +----v---+
   |subagent|        |subagent|   ...   |subagent|  own context; search->read->reflect
   +---+----+        +----+---+         +----+---+  write big outputs to files
       | 1-2K-token summaries or file references |
       +----------------+------------------------+
                        v
           lead synthesizes; loop if gaps remain
                        v
           [citation agent: map claims to sources, text unchanged]
                        v
                 report (background job; resumable stream)
```

### Key techniques

- **Orchestrator-worker with explicit delegation contracts.** Anthropic's Research system uses an Opus 4 lead and Sonnet 4 subagents. It beat single-agent Opus 4 by 90.2% on an internal eval. Vague delegation caused duplicated work, so each subagent gets an objective, output format, tool and source guidance, and clear boundaries [2].
- **Effort scaling in the prompt.** Without it, early versions spawned 50 subagents for simple queries [2].
- **Token budget is the main quality lever.** On BrowseComp, token usage alone explained 80% of performance variance, and tokens, tool calls and model choice together explained 95%. A better model beat doubling the budget [2]. Expose effort tiers, as Gemini does with a standard and a Max Deep Research agent [55].
- **Scoping before research.** In ChatGPT, a lighter model asks clarifying questions and rewrites the prompt before the research model starts. The API research models expect "fully-formed prompts up front", and under-specified prompts cause "verbosity, higher latency, and increased token usage" [50][51]. LangChain's Open Deep Research uses clarify, then brief, then supervisor, then researchers [53].
- **Compression and filesystem handoff.** Subagents return condensed summaries of about 1,000 to 2,000 tokens [3], or write outputs to files and pass references to avoid a "game of telephone" [2].
- **Citation as a separate pass.** A dedicated CitationAgent attributes claims after writing [2]. OpenAI's API returns character-span citation annotations [50].
- **Long-running job semantics.** Run in background mode and poll or stream [50][55]. Resume from checkpoints instead of restarting, because errors compound. Ship updates as rainbow deployments so in-flight agents are not broken [2].

### The single-agent counterargument

Cognition's "Don't Build Multi-Agents" argues that agents should share full context, because "actions carry implicit decisions". Its Flappy Bird example shows two subagents producing stylistically incompatible parts [30]. The reconciliation interviewers expect: parallel subagents fit **read-heavy, breadth-first** work like research, where workers gather and one agent decides. A single thread fits **write-heavy** work like coding, where decisions must stay consistent. Anthropic itself notes that its lead waits synchronously for each subagent batch and cannot steer subagents mid-flight [2].

### Failure modes

Observed failures include endless searching for nonexistent sources, overly long and specific queries that return little, continuing after results were already sufficient, and a preference for SEO content farms over authoritative sources. Human testers caught the last one where automated evals did not [2]. Mitigations: "start wide, then narrow" query heuristics, source-quality rules, stop conditions, and an LLM judge rubric covering factual accuracy, citation accuracy, completeness, source quality and tool efficiency, starting from about 20 real queries [2].

### Sizing

- LangChain Open Deep Research cost $45.98 for the 100-task DeepResearch Bench run (about 58M tokens) in its default configuration, and $187.09 (about 139M tokens) with Claude Sonnet 4. That is roughly $0.46 to $1.87 per report [53].
- GPT Researcher reports about $0.115 per standard report, and about $0.40 and about 5 minutes for deep mode [54].
- Add search fees: at $10 per 1,000 searches, a run of 60 searches adds $0.60 [17] (illustrative arithmetic).

## 18.6 Design 4: A browser agent

**Prompt:** "Design an agent that completes tasks on arbitrary websites for a user, such as booking, form filling and data extraction."

### The central choice: what the model observes

| Observation | Strengths | Weaknesses | Systems |
|---|---|---|---|
| Pixels (screenshots) | Works on any UI, canvas or desktop | Coordinate grounding; small targets; about 1,000-1,800 tokens per image [20] | OpenAI CUA [44], Claude computer use [20] |
| DOM / accessibility tree | Exact element IDs; can strip passwords before the LLM | Missing ARIA, canvas UIs, hidden-text injection | Stagehand `observe` [48], WebArena baselines |
| Hybrid | IDs where available, pixels as fallback | Two views can disagree | Browser Use (DOM plus vision) [47] |

The broader production lesson comes from ChatGPT agent, which combined a visual browser, a text browser for fast reading, a terminal and API connectors [46]. **Pick the cheapest interface per step**: an API before a text browser, a text browser before the DOM, the DOM before pixels.

### Architecture

```text
 task --> planner/actor loop -------------------------------------------+
            | observe: URL, tabs, indexed elements (+ screenshot)       |
            | act: click(id) | type | scroll | navigate | extract       |
            v                                                           |
   +------------------------- isolated cloud browser ---------------+   |
   | per-session profile; domain allowlist enforced at network layer|   |
   | re-checked after redirects; secrets typed by user (takeover)   |   |
   +----------------------------------------------------------------+   |
            |                                                           |
   guards: injection classifier on page content; confirm before submit/ |
           purchase/send; blocked site categories; step budget; watch --+
   context: keep last ~3 screenshots, prune in batches, compact ~150K
   done: only after verifying every requirement against tool output
```

### Key techniques

- **A compact, change-annotated DOM view.** Browser Use builds an enhanced tree from DOM, layout snapshot and accessibility data, indexes interactive elements and marks elements new since the last step. Two runtime rules carry most of its reliability. Page-changing actions must come last in a multi-action step. `done` must be the only action in its step, and success is allowed only after verification [47].
- **AI only where the page is unknown.** Stagehand adds `act`, `observe` and `extract` primitives to Playwright code, with self-healing and server-side caching so repeated identical calls cost no tokens [48]. For high-volume, repeated flows this hybrid beats an autonomous agent on cost and determinism.
- **Screenshot hygiene** [20]:
  - downscale before sending (1280x720 to start on 4.6-class models), and scale returned coordinates back;
  - use zoom for dense UIs;
  - use up to four cache breakpoints;
  - prune old screenshots in batches (example: keep 3, prune every 25 steps) so the cached prefix stays stable;
  - compact at about 150K tokens.

### Failure modes

- **Perception.** Dropdowns drawn by the OS outside the captured viewport, tiny targets after downscaling (a 16-pixel checkbox becomes about 5 pixels when 4K is scaled to 720p), and invisible browser-native alert modals [20][6]. Fall back to keyboard navigation, JavaScript or the DOM.
- **Prompt injection, the defining risk.** In Anthropic's Claude for Chrome pilot, attack success across 123 test cases fell from 23.6% to 11.2% with mitigations. On browser-specific attacks such as hidden DOM fields and injections through URL or tab-title text, it fell from 35.7% to 0% [21]. A later report cites about 1% attack success against an adaptive best-of-N attacker and states that "no browser agent is immune to prompt injection" [22].
- **Unsafe side effects.** Operator's layered design is the template [45]:
  - *takeover mode*, so the model never sees credentials or payment details;
  - confirmations before consequential actions;
  - task refusals for disallowed categories;
  - *watch mode* on sensitive sites;
  - a prompt-injection monitor.

  Claude for Chrome adds site-level permissions and blocks high-risk categories such as financial services [21].

### Sizing

At about 1,000 to 1,800 tokens per screenshot plus a toolset overhead of a few thousand tokens [20], a 40-step task with full screenshots each step sends roughly 40K to 70K tokens of images. The cumulative re-sent context is far larger unless old screenshots are pruned (illustrative arithmetic). This is why pruning, caching and text-first observation matter more than model price.

## 18.7 Cross-cutting failure handling playbook

| Failure class | Detection | Mitigation | Sources |
|---|---|---|---|
| Compounding step errors | Trajectory traces; end-state checks | Verify each step with tools (tests, URL state, DB state); checkpoints; resume, do not restart | [2][14] |
| Infrastructure loss (container, harness crash) | Heartbeats; tool errors | Append-only session log; stateless harness; `wake(sessionId)`; disposable sandboxes | [8] |
| Provider outage or rate limits | 429s and latency SLO breaches | Ordered fallback lists; AIMD admission; hedging; no fallback mid-stream | [31][32] |
| Context rot or overflow | Token counters; `/context`-style accounting | Tool-result clearing; compaction; memory files; subagents. Memory plus context editing gave +39% and -84% tokens on one eval | [3][15] |
| Runaway loops or spend | Budget counters; repeated-action detection | Task budgets; max turns; CI round caps; denial counters (3 consecutive / 20 total) | [9][18][27] |
| Wrong tool or bad arguments | Schema validation; error rates per tool | Actionable errors; consolidated, namespaced tools; tool-use examples (72% to 90% on complex parameters) | [4][5] |
| Prompt injection | Input probes and classifiers on tool output | Sandbox; credentials outside the agent; confirmations; site and domain allowlists | [9][10][21][22] |
| Premature "done" | Independent evaluator; hidden tests | Failing-by-default checklists; separate skeptical grader | [6][7][12] |
| Approval fatigue | Approval-rate telemetry (93%) | Sandbox plus classifier instead of prompts | [9][10] |
| Silent quality regressions | Online monitors; LLM judges calibrated to humans | Versioned agents; canary and rainbow rollouts; regression eval suites | [2][14] |

On evaluation, Anthropic's guide gives a minimal recipe [14]:

- start with 20 to 50 tasks drawn from real failures;
- grade outcomes rather than paths;
- calibrate model graders against humans and give them an "Unknown" option;
- isolate trials;
- read transcripts to separate agent failures from grader bugs;
- move saturated capability evals into regression suites.

## 18.8 Side-by-side comparison

| Dimension | Coding agent | Support agent | Research agent | Browser agent |
|---|---|---|---|---|
| Default rung | Single agent loop (or blueprint) | Workflow-heavy procedures plus agentic steps | Orchestrator plus parallel workers | Single agent, hybrid with scripted flows |
| Latency regime | Seconds per step; minutes to hours per task [13] | Sub-second to a few seconds per turn; voice is strictest [32] | Minutes, run as a background job [49][52] | Tens of seconds to minutes |
| Verifier | Tests, lint, CI [27] | End state (ticket, refund) plus rubric [14] | LLM judge rubric plus citation checks [2] | URL, page and backend state [14] |
| Primary risk | Destructive commands, exfiltration | Wrong policy action, hallucination at scale [33] | Low-quality sources, hallucinated citations [2] | Prompt injection, unsafe submissions [21] |
| Isolation | OS sandbox or VM; credential proxy [10] | Scoped tools; deterministic auth gates [41] | Read-only tools; fetch sandbox | Isolated browser; takeover for secrets [45] |
| Key cost driver | Long-context re-reads; cache hit rate | Volume x calls per turn; supervisors | Tokens across subagents (about 15x chat) [2] | Screenshot tokens [20] |
| Key metric | Merged PRs per session [28] | Resolution and escalation accuracy; pass^k [19][43] | Rubric score; citation accuracy | Task success on live sites |
| Signature technique | ACI plus exactly-once edits [12][23] | Supervisors plus failover [31][33] | Effort scaling plus citation pass [2] | DOM serialization plus confirmations [45][47] |

## Key takeaways for system designers

1. **Start at the lowest rung and justify each step up.** Workflows before agents, one agent before many. Earn the about 15x token multiple of multi-agent designs with a measured gain [1][2].
2. **Size in tokens, and treat the prefix cache as the main cost lever.** Input cost grows roughly quadratically with step count. Keep the prefix stable (tools, then system, then messages), mind the TTL, and expect around an order-of-magnitude difference between cached and uncached sessions [16][29].
3. **Separate brain, hands and session.** A stateless harness, disposable sandboxes and a durable event log turn crashes into resumable events and keep credentials out of reach of generated code [8].
4. **Make verification the backbone.** Tests for code, end state for support, rubric plus citation checks for research, and page or backend state for browsing. Use an independent grader where self-review flatters [6][7][14].
5. **Put deterministic gates around irreversible actions.** Use code-enforced identity checks, confirmations, allowlists and classifier-gated autonomy, and quote the classifier's false-negative rate honestly [9][41][45].
6. **Take guards and routing off the critical path.** Run supervisors in parallel with generation, route with single-token classifiers, hedge across providers, and provision lazily [8][32][33][42].
7. **Bound everything.** Cap CI rounds, turns, denials and token budgets. Unbounded agents spawn 50 subagents or loop on CI [2][18][27].
8. **Design the handoff as carefully as the automation.** Escalation accuracy and human availability decide outcomes on the hardest cases [19][40].
9. **Expect the harness to shrink as models improve.** Re-test each component's assumption and remove scaffolding one piece at a time [7][8].

## References

1. "Building Effective Agents." Erik Schluntz and Barry Zhang, Anthropic, 2024. https://www.anthropic.com/engineering/building-effective-agents
2. "How we built our multi-agent research system." Anthropic, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
3. "Effective context engineering for AI agents." Anthropic, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
4. "Writing effective tools for agents — with agents." Anthropic, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
5. "Introducing advanced tool use on the Claude Developer Platform." Anthropic, 2025. https://www.anthropic.com/engineering/advanced-tool-use
6. "Effective harnesses for long-running agents." Anthropic, 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
7. "Harness design for long-running application development." Anthropic, 2026. https://www.anthropic.com/engineering/harness-design-long-running-apps
8. "Managed Agents" engineering post (decoupling session, harness and sandbox). Anthropic, 2026. https://www.anthropic.com/engineering/managed-agents
9. "Claude Code auto mode" engineering post. Anthropic, 2026. https://www.anthropic.com/engineering/claude-code-auto-mode
10. "Beyond permission prompts: making Claude Code more secure and autonomous." David Dworken and Oliver Weller-Davies, Anthropic, 2025. https://www.anthropic.com/engineering/claude-code-sandboxing
11. "How Claude Code works." Claude Code documentation, Anthropic, 2026. https://code.claude.com/docs/en/how-claude-code-works
12. "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet." Erik Schluntz, Anthropic, 2025. https://www.anthropic.com/research/swe-bench-sonnet
13. "Measuring agent autonomy." Anthropic, 2026. https://www.anthropic.com/research/measuring-agent-autonomy
14. "Demystifying evals for AI agents." Anthropic, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
15. "Managing context on the Claude Developer Platform." Anthropic (claude.com blog), 2025. https://claude.com/blog/context-management
16. "Prompt caching." Claude Platform documentation, Anthropic, 2026. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
17. "Pricing." Claude Platform documentation, Anthropic, accessed October 2026. https://platform.claude.com/docs/en/about-claude/pricing
18. "Task budgets." Claude Platform documentation, Anthropic, 2026. https://platform.claude.com/docs/en/build-with-claude/task-budgets
19. "Customer support agent" use-case guide. Claude Platform documentation, Anthropic, 2026. https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat
20. "Best practices for computer and browser use with Claude." Anthropic (claude.com blog), 2026. https://claude.com/blog/best-practices-for-computer-and-browser-use-with-claude
21. "Piloting Claude for Chrome." Anthropic (claude.com blog), 2025. https://claude.com/blog/claude-for-chrome
22. "Mitigating the risk of prompt injections in browser use." Anthropic, 2025. https://www.anthropic.com/research/prompt-injection-defenses
23. "Agent-Computer Interface (ACI)." SWE-agent documentation, Princeton/Stanford SWE-agent team, 2024. https://github.com/SWE-agent/SWE-agent/blob/main/docs/background/aci.md
24. Agentless repository. Xia, Deng, Dunn, Zhang (OpenAutoCoder), 2024. https://github.com/OpenAutoCoder/Agentless
25. mini-SWE-agent repository. SWE-agent team, 2025. https://github.com/SWE-agent/mini-swe-agent
26. "Minions: Stripe's one-shot, end-to-end coding agents." A. Gray, Stripe, 2026. https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents
27. "Minions: Stripe's one-shot, end-to-end coding agents, Part 2." A. Gray, Stripe, 2026. https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents-part-2
28. "Why We Built Our Own Background Agent." Ramp Builders Blog, 2026. https://builders.ramp.com/post/why-we-built-our-background-agent ; adoption figures from "Why Ramp built its own in-house coding agent, Inspect." The Pragmatic Engineer, 2026. https://newsletter.pragmaticengineer.com/p/why-ramp-built-inspect
29. "Context Engineering for AI Agents: Lessons from Building Manus." Yichao "Peak" Ji, Manus, 2025. https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
30. "Don't Build Multi-Agents." Walden Yan, Cognition, 2025. https://cognition.ai/blog/dont-build-multi-agents
31. "Preserving agent behavior while serving LLMs reliably." P. Baccichet and R. Henwood, Sierra, n.d. (captured 2026). https://sierra.ai/blog/model-failover
32. "Engineering low-latency voice agents." W. Shao and D. Lee, Sierra, n.d. (captured 2026). https://sierra.ai/blog/voice-latency
33. "From LLMs to enterprise-grade agents." T. Rajan, Sierra, n.d. (captured 2026). https://sierra.ai/blog/enterprise-grade-agents
34. "Meet Linnaeus and Darwin: Search models that drive higher resolution rates." Sierra, n.d. (captured 2026). https://sierra.ai/blog/meet-linnaeus-and-darwin-the-sierra-search-team
35. "Designing layered guardrails for reliable AI agents." S. Bae, Decagon, 2025. https://decagon.ai/blog/designing-layered-guardrails-for-reliable-ai-agents
36. "Designing low-latency AI agents through reranker optimization." N. Liu, Decagon, 2026. https://decagon.ai/blog/designing-low-latency-ai-agents-through-reranker-optimization
37. "Think fast: reasoning at 3ms a token." Fin AI Research (Intercom), 2025. https://fin.ai/research/think-fast-reasoning-at-3ms-a-token/
38. "Fin AI agent inventory" (product teardown citing Intercom help center and fin.ai pricing; secondary). helpin-ai, 2026. https://github.com/helpin-ai/helpin/blob/main/docs/research/2026-09-10-fin-ai-agent-inventory.md
39. "Klarna AI assistant handles two-thirds of customer service chats in its first month." Klarna press release, 2024. https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/
40. "Automated customer service without losing the human touch." Decagon, 2026. https://decagon.ai/blog/automated-customer-service
41. "agentforce-adlc" (posture-and-determinism, architecture-patterns references). Salesforce AI Research, 2026. https://github.com/SalesforceAIResearch/agentforce-adlc
42. "Agentforce runtime rearchitecture: reducing latency." Salesforce, 2026 (read via secondary summary). https://www.salesforce.com/blog/agentforce-reducing-latency/
43. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." S. Yao, N. Shinn, P. Razavi, K. Narasimhan, Sierra, 2024. https://arxiv.org/abs/2406.12045
44. "Computer-Using Agent." OpenAI, 2025. https://openai.com/index/computer-using-agent/
45. "Introducing Operator." OpenAI, 2025. https://openai.com/index/introducing-operator/
46. "Introducing ChatGPT agent: bridging research and action." OpenAI, 2025. https://openai.com/index/introducing-chatgpt-agent/
47. Browser Use repository and agent system prompt. Browser Use, 2024-2026. https://github.com/browser-use/browser-use ; https://github.com/browser-use/browser-use/blob/main/browser_use/agent/system_prompts/system_prompt.md
48. Stagehand repository. Browserbase, 2024-2026. https://github.com/browserbase/stagehand
49. "Introducing deep research." OpenAI, 2025. https://openai.com/index/introducing-deep-research/
50. "Introduction to the Deep Research API." OpenAI Cookbook, 2025. https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api
51. "Deep Research API with the Agents SDK." OpenAI Cookbook, 2025. https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents
52. "Introducing Perplexity Deep Research." Perplexity, 2025. https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research
53. Open Deep Research repository. LangChain, 2025. https://github.com/langchain-ai/open_deep_research
54. GPT Researcher repository. Assaf Elovic et al., 2023-2026. https://github.com/assafelovic/gpt-researcher
55. "Get started with Deep Research" quickstart notebook. Google Gemini Cookbook, 2026. https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_Deep_Research.ipynb
