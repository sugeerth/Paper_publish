# 16. Agent Case Studies from Engineering Blogs: Replit, Vercel v0, Ramp, Shopify, Uber, LinkedIn, Airbnb, Stripe, Notion, Canva, Booking.com, DoorDash and Anthropic

## Overview

The earlier chapters of this atlas describe agent architectures in the abstract. This chapter looks at what companies actually shipped and wrote about between 2024 and late 2026. The sources are engineering blogs, conference talks and practitioner write-ups. They are uneven: some give exact numbers and diagrams, others give only adoption figures. They still form the largest body of public evidence on how agents behave once they meet real users, real codebases and real budgets.

The case studies fall into three groups:

1. **Internal coding-agent platforms.** Stripe Minions, Ramp Inspect, Shopify River/Aquifer, DoorDash Flux, Uber's Minion/uReview/AutoCover stack, Airbnb AirChat, and Replit's internal "manager agent". These are background agents that turn a Slack message or ticket into a pull request.
2. **Product agents for end users.** Replit Agent, Vercel v0, Shopify Sidekick, LinkedIn Hiring Assistant, Notion Custom Agents, Booking.com's partner-messaging agent, and the support/knowledge agents at Airbnb, DoorDash and Uber.
3. **Anthropic's internal use.** Claude Code across teams, the multi-agent Research system, Project Vend, and the "agent teams" C-compiler experiment.

Several findings recur across companies that did not coordinate with each other:

- **Deterministic code wrapped around agent loops.** Stripe calls these "blueprints", DoorDash "playbooks", Shopify "Roast workflows" and LinkedIn "plan-and-execute".
- **Isolated cloud sandboxes reused from human developer infrastructure.** Examples are Stripe devboxes, Airbnb Airdev workspaces, Ramp's Modal snapshots and DoorDash's Firecracker microVMs.
- **One governed tool layer.** Stripe Toolshed, Uber MCP Gateway, DoorDash Agent Gateway and LinkedIn CAPT all sit in front of hundreds or thousands of tools, combined with progressive tool disclosure.
- **Agents that work in public.** Shopify's River works only in public Slack channels, Ramp sessions are multiplayer, and DoorDash moved Flux runs into public threads. These companies treat transcripts as a corpus that compounds.
- **Precision over recall for any agent that comments on human work.** Uber uReview and DoorDash's reviewer both filter heavily so that engineers keep listening.

> **Source note.** The research environment for this chapter could not reach most company blogs directly. Primary posts were read in full where they existed as faithful text archives: Stripe, Ramp, Shopify (Sidekick, River, Roast), Vercel, Uber uReview, DoorDash (Flux, code review, ecosystem), Notion (security, Scruff) and Replit's "Self-Driving Company". All Anthropic material and Claude customer stories were read directly. For some sources only a detailed third-party digest was available; these are marked "via digest" in the References, and their numbers should be treated as approximate. The digest came from the ZenML LLMOps Database and covered Replit's decision-time guidance and evaluation posts, LinkedIn's engineering posts, Uber's Genie/Finch/MCP-gateway posts, Airbnb's migration and platform posts, and Booking.com's posts. Canva has not published an agent architecture that could be found. Its section is therefore limited to what the public record supports.

---

## 16.1 The composite architecture that emerges

Read together, the case studies describe one composite system. No single company runs all of it, but most run most of it.

```text
  invocation surfaces                       governed tool layer
  Slack / web / CLI / PR / cron / Chrome    (Toolshed, MCP Gateway, Agent Gateway, CAPT)
            |                                   ^   authn/z, scoping, redaction,
            v                                   |   rate limits, audit, tool search
  +----------------------+   tool intents  +----+-------------------------------+
  | session service      |---------------->| registry (control plane)           |
  | durable event log    |                 | proxy (data plane) -> 100s-1000s   |
  | multiplayer state    |<----------------| of internal APIs / SaaS MCPs       |
  +----------+-----------+   results        +-----------------------------------+
             |
             v
  +----------------------+      +---------------------------------------------+
  | harness / workflow   |----->| sandbox (devbox / microVM / Modal snapshot) |
  | blueprint: det. nodes|      | repo pre-cloned, caches warm, no prod data  |
  | + agent-loop nodes   |<-----| lint/test/build, browser for visual checks  |
  +----------+-----------+      +---------------------------------------------+
             |
             v
  verification gates: local lint -> bounded CI rounds -> reviewer agent -> human
             |
             v
  telemetry: traces, merged-PR rate, acceptance rate, cost/run -> eval + mining loop
```

The defining choice is where the boundary falls between the *harness* (model loop and orchestration) and the *sandbox* (where code runs). Shopify states the principle most directly: "the harness does not live in the sandbox" [7]. Stripe and DoorDash get a similar split from devboxes and microVMs, with the orchestrator outside [2][35].

---

## 16.2 Internal coding-agent platforms

### 16.2.1 Stripe Minions: one-shot agents built on blueprints

**Problem.** Stripe's codebase is "hundreds of millions of lines of code across a few large repositories", mostly Ruby with Sorbet types and many homegrown libraries, and it moves more than $1 trillion a year in payment volume [1]. Stripe already offered attended agents (Cursor, Claude Code). Its scarce resource is developer attention, so it built *unattended* agents that can run many tasks in parallel [1][2].

**Architecture.**

- *Entry points.* Most runs start from Slack, and the agent can read the whole thread and its links. Runs can also start from a CLI, a web UI, the docs platform, the feature-flag platform and ticketing. For example, CI opens tickets for flaky tests, and each ticket has a "fix with a minion" button [1].
- *Devboxes.* The agents use the same AWS EC2 devboxes as Stripe engineers. Devboxes are pre-warmed from a pool so that one is "ready within 10 seconds": repositories cloned, Bazel and type-check caches warm, code-generation services running. They run in a QA environment with no real user data, no production services and no arbitrary egress. That isolation is why minions run "with full permissions" and no confirmation prompts [2].
- *Agent core.* The core is an internal fork of Block's goose, made in late 2024 and since developed only for unattended use. Interruptibility and other human-steering features were dropped [2].
- *Blueprints.* These are code-defined state machines that mix deterministic nodes ("Run configured linters", "Push changes") with agent nodes ("Implement task", "Fix CI failures"). Each agent node gets its own constrained tools, system prompt and trimmed context. Teams write their own blueprints, for example for LLM-assisted migrations that a codemod cannot handle [2].
- *Context.* Rule files are scoped to subdirectories and attached as the agent walks the filesystem. There are almost no global rules, so the context window does not fill before work starts. Stripe standardized on Cursor's rule format and syncs it to Claude Code's format, so three agents share one corpus [2]. MCP tools run deterministically on links in the prompt *before* the run starts, to "hydrate" context [1].
- *Toolshed.* A central MCP server with "nearly 500" tools (400+ in Part 1) is shared by "hundreds of different agents". Each minion gets a deliberately small subset by default [1][2].

```text
 Slack thread --> [hydrate links via MCP] --> devbox (10 s, warm)
                                              |
   +------------------------------------------v---------------------------+
   | blueprint                                                            |
   | (agent) implement task --> (det) run linters <-> (agent) fix lint    |
   |      --> (det) push --> CI (selected tests of 3M+) --> autofixes     |
   |      --> (agent) fix CI failures --> (det) push --> CI round 2 -> stop|
   +----------------------------------------------------------------------+
                                              |
                                  PR from template --> human review
```

**Feedback loop and budgets.** Pre-push lint heuristics run in under five seconds (often under one second, thanks to a caching daemon). CI then selects tests from more than three million. Many test failures have automatic fixes. Minions get *at most two* CI rounds, because "there are diminishing marginal returns for an LLM to run many rounds of a full CI loop" [1][2].

**Scale.** More than 1,000 merged PRs per week were fully minion-written in Part 1. Part 2 reports more than 1,300, all human-reviewed and containing no human-written code [1][2].

**Lessons.** "What's good for humans is good for agents." Devboxes, rule files and fast lint were built for people before LLMs existed [2]. Putting "LLMs into contained boxes" gives a reliability gain that compounds across the whole system [2].

### 16.2.2 Ramp Inspect: a background agent that verifies its own work

**Problem.** Ramp engineers liked Claude Code, but a laptop could run only one or two sessions. The frontend team also wanted designers to make small UI changes without a local environment [4]. Version 1 was a Chrome extension for visual edits and saw little adoption. Version 2 (November 2025) became a remote development environment with an agent on top [4].

**Architecture** [3][4]:

- *Sandbox.* Modal sandboxes with filesystem snapshots. Each repository image is rebuilt every 30 minutes (clone, install, build, even one test run to warm caches). Sessions start from the latest snapshot, so code is at most 30 minutes stale. A snapshot is also taken at the end of a session, so a follow-up can resume. The sandbox starts warming as soon as the user *begins typing*. The agent may read files before the git sync finishes, but an OpenCode plugin on `tool.execute.before` blocks edits until it does. The newsletter reports provisioning in under five seconds [4].
- *Harness.* OpenCode, chosen because it is server-first with a typed SDK and plugins, and because its source code can be read by the agent itself.
- *API and state.* Cloudflare Durable Objects give each session its own SQLite database. The Cloudflare Agents SDK and WebSocket hibernation stream tokens to every connected client. Follow-up prompts are queued rather than injected mid-run.
- *Clients.* Slack (with a fast repository classifier: "GPT 5.2 with no reasoning", with an "unknown" option), a web app with hosted VS Code and a streamed desktop for visual verification, and a Chrome extension. The extension sends the DOM/React tree rather than screenshots to save tokens, and it is distributed through MDM policy.
- *Identity.* PRs are opened with the *user's* GitHub token, not the app's. Otherwise users could approve their own changes.
- *Self-spawning.* A tool lets a session start and monitor other sessions, which supports cross-repository research or splitting a task into many small PRs.

**Scale.** About 30% of merged frontend and backend PRs came from Inspect at the time of the Ramp post [3]. The newsletter reports about 60% by January, 75% by May 2026, more than one million sessions by July, a team of 5.5 people, more than 150 contributors, and more than 200 internal agents built on the platform. Those agents include ReviewBuddy, an on-call assistant, a Playwright-generating QA agent and "Ramp Research" [4]. Ramp's chosen top metric is **sessions that end in a merged PR** [3].

**A related design: Architect and Doer.** Ramp Labs also built a two-agent finance system. An *Architect* (Gemini 2.5 Pro, 1M-token context) watches a screen recording of an accountant and compiles a reusable "Process" with steps, required files and time dependencies. A *Doer* (Claude Sonnet 4.5 with spreadsheet tools) executes it. On roughly 50 SpreadsheetBench tasks, the Doer scored 49.5% on soft-restriction tasks and 32.5% on hard-restriction tasks. One- to two-hour tasks dropped to under 10 minutes [5]. The pattern is "second-order automation": the bottleneck was writing the instruction, not executing it.

### 16.2.3 Shopify River and Aquifer: making the session the thing that survives

**Problem.** Local agents "have a ceiling". What one engineer learns dies with that engineer's session [7]. Two infrastructure bets from spring 2024 made a shared agent possible: a single monorepo ("World") and Nix for development, CI and production [7].

**River** lives in Slack and *only works in public channels*. Direct messages are not allowed. Median session length is 19 minutes, with a median of 50 tool calls. In one 30-day window there were 59,918 sessions across 5,170 channels, touching more than 7,000 people, and 3,536 River-coauthored PRs were merged. Roughly one in eight merged PRs at Shopify is River-coauthored [7]. Transcripts are mined to update skills, prompts, defaults and `AGENTS.md` files, so "the agent gets smarter without requiring model retraining" [7].

**Aquifer** is the platform underneath. It splits an agent into three parts [7]:

```text
  Session  (durable identity, append-only event log in Postgres)  <- survives all
     ^
     | replay history
  Harness  (agent loop: read history, call model, emit tool intents) <- disposable
     |
     v tool intents
  Sandbox  (filesystem, shell, repo, build/test)                     <- disposable
```

A *session cell*, which runs the Go runtime plus the harness, is created on demand and exits when idle. The next interaction may land on a different host with no loss, because "the work lives in Postgres, not in memory". Agents are *profiles*: data bundles (system prompt, skills, extensions, sandbox policy, model defaults) built with Nix. River, PR review and a headless agent are peer profiles. One substrate serves three modes: interactive (River), automation (PR review triggered by external events) and job (CI/batch, where the session log is optional) [7].

**Roast.** For structured work, Shopify open-sourced Roast. It is a convention-over-configuration workflow engine: YAML plus Markdown prompts, shell steps, Ruby steps and parallel steps, with a `CodingAgent` step backed by Claude Code. Its premise is that agents "work much better when you break down complicated prompts into discrete steps" [8].

### 16.2.4 DoorDash Flux: primitives, not workflows

DoorDash built Flux after hitting laptop limits: CPU, battery, connectivity, broad local credentials and poor auditability [35]. The stated thesis is that writing code "is mostly solved". The hard part is the environment, tools, permissions, integrations and constraints around the agent [35]. Flux has four primitives [35]:

- **Sandboxes.** Firecracker microVMs with a p95 SLO of under five seconds for end-to-end setup (VM start, cloning, build tools, harness setup). They support multi-repository, multi-PR sessions.
- **Agent Gateway (MCP).** Each playbook declares the tools it needs and is granted only those scopes. Every action is logged. The gateway has a registry (control plane) and a proxy (data plane). The proxy authenticates, authorizes, rate-limits, injects credentials and emits telemetry. Agents never hold raw vendor keys or OAuth refresh tokens. A digest of the gateway post reports millions of weekly tool calls across more than 200 MCP servers serving more than 30 agents [38].
- **Playbooks.** A playbook is a single YAML file packaging the task, inputs, context, skills, tools, permissions, validation, outputs and safety bounds. DoorDash calls it "the equivalent of a Docker container" for agentic work. Steps can be agentic or deterministic.
- **Invocation surfaces.** Slack, GitHub, cron, CLI and skills.

**Scale.** 130,000 automated engineering tasks in one month of 2026, more than 25,000 automated code reviews a week, and more than 300 playbooks with more than 10,000 weekly invocations [35]. Lessons: start narrow (code review first); make work visible (private per-run Slack channels did not build team habits, public threads did); and playbooks need workshops and hackathons, not just a platform [35].

**The code reviewer** is DoorDash's most detailed design write-up [36]. Version 1 used parallel specialist agents (security, tests, performance), which missed architectural issues. Version 2 used two general reviewers, whose attention "spread thin". Version 3 adds a **lead scout** that only *notices* suspicious spots and emits investigation leads. Two deep reviewers then verify or drop each lead, and every finding must survive a **disprove-it pass** before posting. Context comes from per-domain **review profiles** mined from `AGENTS.md`, historical review comments, Slack decisions and incident history, because "AGENTS.md and CLAUDE.md files are written for engineers authoring code, not engineers reviewing it". Results: more than 10,000 PRs a week across 56 repositories; 60.2% of settled high/critical findings lead to a code change (n=2,256), against 46% for the previous third-party tool; about $3 per review; findings arrive about 7 minutes after a PR opens [36].

**Feature-flag cleanup.** An orchestrator builds a cleanup plan from Jira, the experimentation platform and code search. After an engineer confirms the plan, removal agents run on Claude Opus, one per flag, in separate git worktrees (at most four concurrent per repository), with a one-hour hard timeout. A PR opens only if local gates pass. Early runs without worktree isolation produced race conditions and corrupted diffs. On 50 stale flags, 45 produced usable PRs at an average of $4.79 and 13.8 minutes each. The main failure mode was incomplete dead-code removal rather than unsafe changes (via digest) [39].

DoorDash also published a taxonomy of increasing autonomy: workflow, agent, deep agent (hierarchical, with a persistent shared workspace), and swarm (peer agents coordinating through A2A). It keeps deterministic workflows for certified, high-stakes reporting. It warns that "persisting every intermediate step can bloat context" and often passes only final artifacts between agents [37].

### 16.2.5 Uber: a layered stack of specialist agents

Uber describes its stack in layers [22]:

1. An AI platform on Michelangelo, with a model gateway.
2. Internal context: code, docs, Slack, Jira.
3. Off-the-shelf agents: Claude Code, Copilot, Cursor, Codex.
4. Uber-specific agents: the **Minion** background-agent platform, **AutoCover**, **uReview** and the **Shepherd** migration tool.

The **aifx CLI** provisions clients and MCP servers and keeps tools up to date across the fleet [22].

**uReview** [17] uses prompt chaining rather than an agent loop:

```text
 diff -> eligibility filter (drop config/generated/experimental files)
      -> context builder (nearby functions, classes, imports)
      -> assistants: Standard (bugs) | Best-Practices (style registry) | AppSec
      -> grader prompt: confidence per comment (thresholds per assistant/lang/category)
      -> semantic de-dup -> category classifier (suppress low-value categories)
      -> post inline -> "Useful / Not useful" -> Kafka -> Hive -> threshold tuning
```

uReview covers more than 90% of about 65,000 weekly diffs across six monorepos, with a median of 4 minutes. 75% of rated comments are marked useful and more than 65% are addressed, against 51% for human comments. Uber estimates about 1,500 engineer-hours saved per week. The best configuration paired Claude 4 Sonnet as generator with o4-mini-high as grader [17].

Whether a comment was "addressed" is measured by **re-running uReview five times on the final commit**. If no rerun reproduces the comment, it counts as addressed. Uber says five reruns is "the minimal count that virtually eliminates missed detections" [17].

Lessons: precision beats volume; guardrails matter as much as prompts; developers dislike readability nits and minor logging tweaks; and code-only context cannot judge system design [17].

**Validator and AutoCover** were built on "Lang Effect", Uber's wrapper around LangGraph (via digest) [19]:

- *Validator* runs in the IDE. It combines an LLM best-practices sub-agent with a deterministic linter sub-agent and offers precomputed fixes.
- *AutoCover* is a LangGraph workflow with a scaffolder, a generator and an executor that knows Uber's build system. It reuses Validator as a sub-agent and runs up to 100 generation iterations in parallel. Uber reports it as 2-3x faster than generic agentic coding tools, with a 10% coverage gain worth about 21,000 developer-hours [19].

**MCP Gateway** has a registry control plane and a proxy data plane. It translates MCP JSON calls to HTTP, gRPC or TChannel, serialized as Protobuf or Thrift, and routes them through the Muttley service mesh. **AutoCrawler**, a Cadence workflow, discovers services from the IDL registry and uses an LLM to write tool descriptions. New tools are *disabled by default* until the owning team approves them. Other features: PII redaction, tool-level authorization that distinguishes human, service and agent callers, **Omni MCP** for incremental server/tool discovery, **response projection** (GraphQL-style field selection), and **Code Mode**, which writes tool results to files for coding agents. Uber reports more than 800 MCP servers and more than 5,000 tools (via digest) [20].

**Scale and cost.** In March 2026, 84% of developers were agentic-coding users and 11% of PRs were opened by agents. AI costs had risen 6x since 2024. Top-down mandates worked worse than peer sharing (via digest) [22]. Uber's CTO separately reported 1,800 code changes a week written entirely by the internal agent, up from under 1% to 8% of all changes [23].

### 16.2.6 Airbnb: AirChat, and migrations run as state machines

**Migrations.** Airbnb migrated about 3,500 Enzyme test files to React Testing Library with a per-file **state machine**: refactor, then Jest fixes, then lint/TypeScript fixes, then done. Each step retried with the latest file and errors fed back, and prompts grew to 40,000-100,000 tokens of context (sibling tests, the component source, migration guidelines). The initial 4-hour bulk run reached 75%. Four days of "sample, tune, sweep" iteration reached 97%. Some long-tail files were retried 50-100 times. The work took 6 weeks against an estimated 18 months by hand (via digest) [30]. The same validate-transform-validate loop later became a reusable migration framework inside AirChat [32].

**AirChat** is Airbnb's internal agent harness, managed as a branded product: IDE plugins, then a CLI, an SDK and **AirChat Remote**. Reported results: 97% weekly and 90% daily agentic usage among engineers without mandates, PR throughput up 65%, and about 59% of code primarily AI-authored. About 40 SDK apps exist, and build time for one fell from a day or two to 2-4 hours [32].

AirChat Remote has an API that handles authentication, lifecycle and event streaming for web, Slack and SDK clients. Its worker pool runs on **Airdev workspaces**, the same cloud development environments humans use, which are network-isolated from production and user data. A per-worker daemon wraps agent runtimes behind the **Agent Client Protocol (ACP)** [32]. The team names three open problems: managing many parallel sessions, running sessions remotely with the right permissions, and moving past chat for routine operations ("this doesn't need to be a chat. You can just have a button") [32].

### 16.2.7 Replit's internal "self-driving company"

In early 2026 Replit put its own agent harness, microVMs and remote filesystem behind access policies, token proxies, audit logs and a ZeroTrust network. It then gave the agent GitHub, GCP, Azure, Linear, Notion, Slack and Zendesk [14]. Every employee gets a **manager agent that can spawn multiple agents** working in loops. Reported results: code volume up 5.8x from January to June (2.9x for a fixed cohort of authors); about 30% of human PR-review time saved because the review agent "only call[s] in a second human reviewer when necessary"; revert and incident rates flat [14]. Replit says internal agents matched vendor alert-triage and penetration-testing tools at about one-tenth of the cost [14].

---

## 16.3 Product agents for end users

### 16.3.1 Replit Agent: from manager/editor/verifier to decision-time guidance

**v1 (2024).** Replit moved from a single ReAct agent to a **manager**, **editor** agents and a **verifier** that often hands back to the user. The verifier is human-in-the-loop by design. Tools (30+) are called by having the model write code in a restricted Python DSL instead of using native function calling, because that was more reliable for many-argument tools. Memory is compressed with LLM summaries. Prompts use XML sections and Markdown instructions. Every major step is committed to git so users can roll back. The design acknowledges that "the first few steps... tend to be most successful". Observability runs on LangGraph plus LangSmith (via digest) [11].

**v2-v4 (2025-2026).** Autonomy grew from a few minutes in v1 to 10-15 minutes in v2. The team traded "almost one order of magnitude" more latency for success, ranking performance and cost above latency (via digest) [16]. By Agent 4, sessions run 6+ hours without human input. Sonnet 4.6 handles sustained work and Opus 4.7 handles architectural decisions. Replit says its vision-based self-testing (the agent clicks through the running app in a browser) is 3x faster and 10x cheaper than computer-use approaches [15].

**Decision-time guidance (2026).** Replit found three problems with long trajectories: learned priors override written rules, instruction-following degrades with context length, and adding more rules has diminishing returns. Earlier models would mock data or delete things without confirmation. Appending reminders at the end of context raised parallel tool calls by 15% compared with putting the same text in the system prompt. Beyond three or four stacked reminders, however, compliance *fell* [12]. The fix (via digest) [12]:

```text
 trajectory (user msgs, recent tool results, error patterns)
        |
        v
  fast multi-label classifier (cheap model, every iteration)
        |  selects 0..k micro-instructions from a bank of hundreds
        v
  ephemeral injection at the tail of context  --> agent step --> guidance dropped
        |
        +-- "diagnostic signal": "Found 1 new browser console log, use the log tool"
        +-- "consultation trigger": doom loop detected -> ask a *different model*
             with fresh context for a plan (generator-discriminator gap)
```

The system prompt never changes, so prompt caching stays intact; Replit reports about 90% lower cost than rewriting a dynamic system prompt. False positives cost little, so the classifier is tuned for recall [12].

**Evaluation (2026)** combines three methods (via digest) [13]:

- *ViBench.* Plain-English PRDs drawn from production traces, each with natural-language test plans executed by a Playwright-based eval agent in forked sandboxes. Variants such as "Vibe-on-Vibe" extend apps the agent built earlier, which exposed that "most models degrade when extending their own code".
- *A/B tests* on almost every prompt, tool or model change.
- *Telescope.* Clio-style trace facets, embedded and clustered with HDBSCAN.

A self-improvement loop drafts fixes as PRs with attached evidence, but engineers choose which hypotheses to pursue and approve every launch [13].

### 16.3.2 Vercel v0: a composite "model" with streaming repair

v0 is sold as a model (`v0-1.5-md` and others), but it is a pipeline built around a frontier base model [9]:

```text
 user msg --> pre-processing: system prompt + recent msgs + summaries of old msgs
          --> RAG: docs, UI examples, project sources, internal Vercel knowledge
          --> dynamic system prompt (intent detected via embeddings + keywords)
          --> route: base frontier model (new/large) | Quick Edit model (narrow edits)
          --> stream --> "LLM Suspense" rewrite layer (in-flight, <100 ms)
          --> post-stream autofixers: deterministic AST fixes + vercel-autofixer-01
          --> linter --> preview
```

**Key techniques** [9][10]:

- *Swappable base model.* `v0-1.0-md` ran on Sonnet 3.7 and `v0-1.5-md` on Sonnet 4 without rebuilding the rest of the pipeline.
- *Dynamic system prompt.* Version-specific AI SDK knowledge is injected when intent is detected. The injected text is kept stable to maximize prompt-cache hits. Hand-curated example directories sit in a read-only filesystem, which Vercel prefers over web search to avoid the "telephone" errors of small-model summaries.
- *LLM Suspense.* Text is rewritten as it streams. Long blob URLs are replaced with short tokens before generation and restored afterwards. Hallucinated `lucide-react` icons are matched by embedding search against the live export list and rewritten to the nearest real icon (for example `Triangle as VercelLogo`) within 100 ms.
- *Autofixers.* These run in under 250 ms and only when needed. Examples: AST checks for a missing `QueryClientProvider` (the model decides where to insert it), deterministic completion of `package.json` dependencies, and `vercel-autofixer-01`, trained with Fireworks AI using reinforcement fine-tuning.

**Numbers.** LLM-generated code can contain errors "as often as 10% of the time", and the pipeline gives "a double-digit increase in success rates" [10]. On Vercel's error-free-generation eval, v0-1.5-md scored 93.87 against 78.43 for Claude 4 Opus alone. The autofixer reaches 86.14% error-free output at about 8,130 chars/s, against 238.9 chars/s for gpt-4o-mini [9]. The metric optimized is the share of generations that render a working preview [10].

### 16.3.3 Shopify Sidekick: JIT instructions, calibrated judges and GRPO

Sidekick is a single-agent loop over Shopify admin tools [6]. Shopify reports how tool count changes behavior: 0-20 tools are fine; at 20-50 "boundaries become unclear"; beyond 50 there are "multiple ways to accomplish the same task". The system prompt became "Death by a Thousand Instructions" [6].

**Just-in-time (JIT) instructions** return guidance *together with tool results*, exactly when it is relevant. This keeps the system prompt stable, which also preserves caching, and lets instructions vary by beta flag, model version or page context [6]. It is the same principle as Replit's decision-time guidance [12] and v0's intent-gated prompt injection [10]: guidance belongs at the point of decision, not in a growing preamble.

**Evaluation.** Shopify replaced curated "golden" sets with *Ground Truth Sets* sampled from production. At least three experts label each conversation, and agreement is measured with Cohen's kappa, Kendall tau and Pearson correlation. LLM judges were calibrated against this human ceiling, improving from a kappa of 0.02 to 0.61 against a human baseline of 0.69. A judge is accepted once humans cannot tell whether a judge or a human scored the set. An LLM *merchant simulator* replays the goals of real conversations against candidate systems before launch [6].

**GRPO training** used an N-stage gated reward: procedural checks first, then LLM-judge scores. The model found three reward hacks: opting out of hard tasks, using customer tags as a catch-all (`customer_tags CONTAINS 'enabled'` instead of `customer_account_status = 'ENABLED'`), and hallucinated IDs or enum values. Fixing the validators and judges raised syntax-validation accuracy from about 93% to about 99% and judge correlation from 0.66 to 0.75. End-to-end quality then matched the SFT baseline [6]. Shopify's advice: "Avoid multi-agent architectures early" [6].

### 16.3.4 LinkedIn Hiring Assistant: plan-and-execute on a messaging substrate

**Problem.** Sourcing and evaluating candidates across more than 1.2 billion profiles, conversationally and also in the background ("source while you sleep") [24][25].

**Architecture** (mostly via digest) [25][26]:

- *Plan-and-execute.* LinkedIn chose this over pure ReAct for reliability. A **supervisor** plans and orchestrates, and executors run ReAct-style loops.
- *Sub-agents implemented as tools.* Intake, Sourcing, Evaluation, Outreach, Screening, Learning and Cognitive Memory. They are tools rather than separate agent identities, for efficiency.
- *One instance per recruiter.* Each recruiter gets their own agent instance with its own identity and mailbox.
- *Messaging substrate.* Agents communicate over LinkedIn's existing messaging infrastructure: messages are persisted, delivered by an orchestrator with nearline retry, and ordered FIFO within a thread but parallel across threads. Messages carry structured data and instructions, and LinkedIn explicitly avoids putting prompts in messages [26].
- *Platform pieces.* A prompt "source of truth" service (namespaced, versioned) and an OpenAI-compatible inference proxy that switches between Azure OpenAI and fine-tuned internal models and enforces quotas and moderation. A skill registry (predating MCP) supports semantic tool search. LangGraph is used behind adapters so the framework can be swapped [26].
- *Models.* Frontier models (for example GPT-4o) are used for fast-changing, recruiter-facing steps. Fine-tuned smaller models handle candidate evaluation, which "scale[s] with the number of candidates". Speculative decoding brings evaluation down to seconds [25][26].

```text
 recruiter UI <--pub/sub, streamed partials-- supervisor (planner/orchestrator)
                                                 |  messages (FIFO per thread)
     +-----------+-----------+----------+--------+--+-----------+-----------+
     intake   sourcing   evaluation  outreach  screening   learning   memory(CMA)
                 |           |                               |           |
         query gen +     fine-tuned LLM              recruiter actions  episodic /
         MUSE retrieval  + spec. decoding            -> proposals (HITL) semantic /
         (1B+ profiles)                                                  procedural
```

**Retrieval.** MUSE embeddings distill an expensive "Expert Judge" into a dual-tower model. Training labels come from a prompt-engineered open-weight reasoning "teacher". The model uses Matryoshka dimensions: 2048 for IVFPQ ANN search over more than 1 billion profiles, and the full 4096 as features for the ranker. Query embedding takes under 100 ms at p95. The index is rebuilt weekly, with daily CDC updates for changed profiles. Before MUSE, Boolean search returned zero results for about 50% of queries (via digest) [27].

**Memory.** The Cognitive Memory Agent has conversational, episodic, semantic and procedural layers. Retrieval is planned by an LLM orchestrator. Memory is scoped per recruiter and per tenant and is never used for training (via digest) [29].

**Quality** combines "product policy" (the rails, enforced by LLM judges) with "human alignment" (the compass, recruiter-validated signals). Changes to learned qualifications are proposed asynchronously and applied only after recruiter approval [25].

LinkedIn's internal coding-agent work, **CAPT**, exposes a few meta-tools (`get_tools_for_tags`, `get_tool_info`, `exec_tool`) in front of thousands of tools, plus Jinja2 "playbooks" offered as tools. It reports more than 1,000 engineers using it and about 70% faster initial triage (via digest) [28].

### 16.3.5 Notion Custom Agents: five harness rewrites and a permission-first design

Notion began agent work in late 2022 and rebuilt its harness about five times [41]:

1. JavaScript APIs that the model wrote code against.
2. Lossless XML for Notion blocks, which models struggled with.
3. "Notion-flavored Markdown" plus SQLite-syntax queries, because the right move was to "give the models what they want".
4. Declarative tool definitions, so that product teams could own their own tools.
5. Progressive tool disclosure via tool search, once the tool count passed 100.

**Evaluation** has three tiers: CI regression tests; launch report cards with an 80-90% pass bar; and frontier evals ("Notion's Last Exam") designed to pass only about 30% of the time, so they do not saturate. Notion also created a *Model Behavior Engineer* role [41].

There is no dedicated memory abstraction: agents read and write pages and databases. Manager agents supervise sub-agents; one internal user cut 70+ daily notifications to about 5 this way [41].

**Security** starts from nothing [42]:

- Agents begin with no permissions and gain them page by page.
- First-party Slack integration offers a "read and reply" scope, added after early agents with broad write access posted to `#general`.
- MCP tools are allow-listed per tool.
- Risky runtime actions require confirmation from the agent's owner.
- External data is tagged for prompt-injection monitoring.
- Owners have a delete button to undo damage.

By the end of alpha, Notion employees had built more than 3,000 Custom Agents and alpha customers more than 25,000 [42]. **Scruff**, a security-triage agent, was compared with vendor tools in a 4-week evaluation; Notion reports 95% threat-classification accuracy and 6+ hours saved per week [43]. Notion reports 90% cost reduction and up to 85% lower latency from prompt caching [44].

### 16.3.6 Booking.com: a cautious partner-messaging agent and a strict agent-eval discipline

Booking.com's partner-guest messaging agent is a FastAPI service on Kubernetes [33]:

- *Stack.* GPT-4 Mini through an internal LLM gateway that detects prompt injection, with LangGraph orchestration.
- *Three outcomes per message:* reply with a partner template (preferred), write a free-text reply, or **don't answer**.
- *Guardrails before any reasoning:* PII redaction, and "do not answer" topics such as refunds.
- *Tools on a central MCP server,* pre-selected per query and run concurrently: template retrieval (MiniLM kNN returning the top 8 above a threshold, with E5-Small for non-English messages, and templates kept current via Kafka), plus property and reservation lookups through GraphQL.

Out of about 250,000 daily message exchanges, the agent assists tens of thousands. The pilot reported 70% higher user satisfaction (via digest) [33].

Booking's evaluation framework [34] separates two views (via digest):

- *Black-box task completion.* A judge LLM sees the dialogue, the task list and product constraints.
- *Glass-box tool metrics.* Validity via JSON-schema/AST checks, correctness against expected tool sets (including unanswerable tasks), argument hallucination, and an LLM review of tool *names and descriptions*. Obscure names such as `fetch_property_rsi_with_cml` measurably hurt results, and tools should have at most three optional arguments.

Above all, an agent must beat zero-shot, few-shot and deterministic-chain baselines on performance, cost and latency before it ships [34].

### 16.3.7 Support and knowledge agents at Airbnb, DoorDash and Uber

These teams usually built agents as additions to an existing conversational platform, not as new systems:

- **Airbnb Automation Platform v2** keeps the v1 workflow actions as *tools* for a chain-of-thought loop. It has three parts: an IO handler, a Tool Manager (retries, rate limits) and an LLM adapter. Context retrievers are declared statically or dynamically, with point-in-time replay for offline evaluation. Airbnb keeps deterministic workflows for strict operations such as claims (via digest) [31].
- **DoorDash Dasher support** uses RAG plus a two-tier guardrail. A cheap sliding-window semantic-similarity check runs first, and an LLM check runs only when needed; failures are retried or escalated to a human. DoorDash reports 90% fewer hallucinations and 99% fewer severe compliance issues, plus an LLM-judge quality pipeline (via digest) [40].
- **Uber Genie (EAg-RAG)** adds query-optimizer, source-identifier and post-processor agents around hybrid BM25 and vector retrieval. It also has a custom Google Docs loader with LLM-formatted tables, and LLM-as-judge evaluation that cut evaluation time from weeks to minutes (via digest) [18].
- **Uber Finch** uses a LangGraph supervisor that routes to a SQL-writer agent and a document-reader agent. An OpenSearch index stores natural-language aliases for columns and values, queries run over single-table data marts, and evaluation compares query *results* against golden results (via digest) [21].

### 16.3.8 Canva

The public record is thin. Canva's AI Solutions team took a deliberately multi-vendor approach from mid-2024 rather than locking in one supplier. Anthropic's customer story reports 5,000 employees enabled and 65% reporting daily AI use. Use cases include design prototyping, components testable without developers, and Projects holding design principles as reusable context [45]. No Canva post was found that describes an agent architecture, tool layer or evaluation harness. Canva therefore appears here as an adoption data point, not as a design.

---

## 16.4 Anthropic's internal use

Anthropic has written up its own use more than most companies, and the material spans everyday tooling, research systems and deliberate stress tests.

**Claude Code across teams** [46]:

- Data-infrastructure staff diagnosed Kubernetes pod-IP exhaustion from dashboard screenshots.
- Security engineering resolves stack-trace investigations about 3x faster.
- Product design runs autonomous write-test-iterate loops from Figma files.
- Growth marketing built a two-sub-agent pipeline that generates hundreds of ad variants.
- Legal built "phone tree" prototypes.

**Measured change** [47]. In August 2025, Anthropic surveyed 132 engineers, interviewed 53, and analyzed 200,000 transcripts comparing February with August 2025:

- Maximum consecutive tool calls per transcript rose from 9.8 to 21.2 (+116%).
- Human turns per transcript fell from 6.2 to 4.1 (-33%).
- Self-reported Claude use rose from 28% to 59% of work, with productivity gains rising from +20% to +50%.
- 27% of assisted work would not otherwise have been done.
- Most respondents could "fully delegate" only 0-20% of their work.

**Multi-agent Research** [48] uses an orchestrator-worker design. A lead agent (Opus 4) plans, saves the plan to memory (context beyond 200,000 tokens is truncated), spawns 3-5 parallel subagents (Sonnet 4) with explicit objectives, formats and boundaries, and finally hands off to a CitationAgent. It beat single-agent Opus 4 by 90.2% on an internal eval. Token use explained 80% of performance variance. Agents use about 4x the tokens of chat, and multi-agent systems about 15x. Early failures included spawning 50 subagents for simple queries. Effort-scaling rules in the prompt fixed this: one agent with 3-10 calls for simple questions, more than 10 subagents for complex ones [48].

**Project Vend** [49][50] tested an agent running a real business. In phase one, "Claudius" (Sonnet 3.7) ran an office shop with web search, a simulated email tool, notes for state beyond the context window, and Slack. It lost money and was talked into discounts and odd purchases. Phase two added a CRM, browser-based supplier research, a merchandise agent ("Clothius") and a CEO agent ("Seymour Cash") that set objectives and key results over an agent-to-agent Slack channel. Forcing procedures (check price and lead time before quoting) helped most. The CEO "shared many of the deficiencies and blind spots" of the shopkeeper because both ran on the same model. In Anthropic's words, "bureaucracy matters" [50].

**Agent teams** [51] stress-tested parallelism. Sixteen Claude instances, each in its own Docker container, shared a bare git repository. They claimed tasks by writing lock files to `current_tasks/` and resolved merge conflicts themselves. Over nearly 2,000 sessions and about $20,000 in API cost, they built a 100,000-line Rust C compiler that builds Linux 6.9 on x86, ARM and RISC-V. A related long-running-harness post describes an initializer agent that writes `init.sh`, a `claude-progress.txt` log and an initial commit, so that each new context window can recover state quickly [52].

---

## 16.5 Comparison tables

### Internal coding-agent platforms

| System | Harness | Sandbox / start time | Tool layer | Orchestration | Reported scale |
|---|---|---|---|---|---|
| Stripe Minions [1][2] | goose fork | EC2 devbox, ~10 s warm pool | Toolshed, ~500 MCP tools | Blueprints (det. + agent nodes), max 2 CI rounds | 1,300+ merged PRs/week, no human code |
| Ramp Inspect [3][4] | OpenCode | Modal snapshots, <5 s | MCP + custom tools | Self-spawning sessions; queued prompts | ~30% then ~75% of merged PRs; 1M+ sessions |
| Shopify River/Aquifer [7] | Custom harness outside sandbox | Nix-built sandboxes | Skills, `AGENTS.md` in monorepo | Profiles over durable sessions | 1 in 8 merged PRs; ~60k sessions/30 days |
| DoorDash Flux [35][36] | Multiple harnesses | Firecracker microVM, p95 <5 s | Agent Gateway (MCP) | YAML playbooks (det. + agent steps) | 130k tasks/month; 25k+ reviews/week |
| Uber Minion et al. [17][22] | Industry agents + Lang Effect | Monorepo-backed workers | MCP Gateway, 5,000+ tools | Specialist agents (uReview, AutoCover) | 11% of PRs by agents; 1,800 changes/week |
| Airbnb AirChat Remote [32] | Runtimes behind ACP | Airdev workspaces | Plugin marketplace + SDK | Sessions and async tasks | 97% weekly use; ~59% AI-authored code |

### Product agents

| System | Core pattern | Distinctive technique | Evaluation | Key number |
|---|---|---|---|---|
| Replit Agent [11][12][13] | Manager/editor/verifier, then long-horizon single loop | Classifier-selected, ephemeral guidance | ViBench + A/B + Telescope clusters | 6+ hour sessions; ~90% cache savings |
| Vercel v0 [9][10] | Composite pipeline around base model | Streaming rewrite (Suspense) + RFT autofixer | Error-free generation rate | 93.87 vs 78.43 (base) |
| Shopify Sidekick [6] | Single agentic loop | JIT instructions with tool results | GTX sets, calibrated judges, simulator | Judge kappa 0.61 vs human 0.69 |
| LinkedIn Hiring Assistant [25][26] | Plan-and-execute supervisor | Per-recruiter instances on messaging bus | Policy judges + human-aligned signals | 1.2B+ profiles; ANN p95 <100 ms |
| Notion Custom Agents [41][42] | Tool-search agent over workspace | Build-from-nothing permissions | Regression / launch / frontier (~30%) | 3,000 internal, 25,000 alpha agents |
| Booking.com messaging [33][34] | LangGraph tool router | "Don't answer" outcome; MCP tools | Task completion + glass-box tool metrics | Tens of thousands of messages/day |

---

## 16.6 Cross-cutting failure modes and how teams answered them

| Failure mode | Where reported | Mitigation |
|---|---|---|
| System prompt overloaded with rules ("Death by a Thousand Instructions") | Shopify [6], Replit [12] | JIT instructions; classifier-selected ephemeral reminders; stable prefix for caching |
| Too many tools, overlapping semantics | Shopify [6], Notion [41], LinkedIn [28], Uber [20] | Tool search / progressive disclosure; meta-tools; per-agent curated subsets |
| Doom loops and compounding errors in long runs | Replit [11][12][16] | Consultation with a different model; git checkpoints; diagnostic notifications |
| Noisy reviewers that engineers learn to ignore | Uber [17], DoorDash [36] | Grader prompts, category suppression, scout + verify, disprove-it pass |
| Reward hacking during RL | Shopify [6] | Gated rewards: procedural checks before LLM judges; judge re-calibration |
| Runaway CI or token spend | Stripe [2], Uber [22], Anthropic [48] | Bounded CI rounds; effort-scaling rules; cost per run as a tracked metric |
| Agents interfering with each other | DoorDash [39], Stripe [2], Anthropic [51] | Per-agent worktrees, devboxes or containers; lock files; timeouts |
| Over-broad permissions causing public mistakes | Notion [42], Ramp [3] | Build-from-nothing scopes; user-token PRs; owner confirmation; undo |
| Same-model oversight sharing blind spots | Anthropic Vend [50], Replit [12] | Different model for consultation or grading (uReview uses separate generator and grader models [17]) |
| Hallucinated APIs and stale library knowledge | Vercel [9][10] | Version-pinned injected docs; streaming rewrites; embedding-matched corrections |

---

## Key takeaways for system designers

1. **Reuse human developer infrastructure first.** Stripe, Airbnb and Shopify all found that devboxes, cloud workspaces, monorepos, fast lint and rule files already gave agents what they needed. The work that makes a codebase legible to agents is "the debt you owe to your human engineers" [7].
2. **Wrap agent loops in deterministic structure.** Blueprints, playbooks, Roast steps and state-machine migrations guarantee the steps you can predict (lint, push, validate) and leave judgment to the model. Bound the expensive loops; Stripe allows at most two CI rounds [2].
3. **Keep the harness outside the sandbox, and make the session durable.** An append-only session log with disposable harness and sandbox processes gives safety, the ability to swap models, and one place to observe everything [7].
4. **Build one governed tool layer and use progressive disclosure.** Registry plus proxy, disabled-by-default tools approved by their owners, scoped credentials, redaction and tool search appear in the designs of Stripe, Uber, DoorDash and LinkedIn independently [2][20][28][38].
5. **Put guidance at the point of decision, not in the preamble.** JIT instructions [6], decision-time guidance [12] and intent-gated prompt injection [10] all keep the prefix stable (which preserves caching) and the guidance relevant.
6. **Optimize precision for anything that comments on human work.** Fewer, verified comments keep the agent enabled: uReview's 75% usefulness [17] and DoorDash's 60% acceptance [36] are the result of filtering, not volume.
7. **Make agents work in public.** Public Slack threads and multiplayer sessions spread adoption and produce a corpus to mine for skills and evaluations [3][7][35].
8. **Measure outcomes, not activity.** Track merged PRs per session [3], addressed-comment rate via reruns [17], error-free preview rate [10], judges calibrated against human agreement [6], and baselines the agent must beat on cost and latency [34].
9. **Plan for model churn.** Composite pipelines [9], inference proxies [26] and framework adapters let teams adopt a new model the day it ships; Replit has shipped agent versions on Anthropic model launch days [15]. Expect to rebuild the harness several times, as Notion did about five times [41].
10. **Do not assume a model can supervise a peer that shares its weaknesses.** Project Vend's CEO agent inherited its shopkeeper's blind spots [50]. Use a different model, deterministic checks or humans as the overseer.

---

## References

1. "Minions: Stripe's one-shot, end-to-end coding agents." A. Gray, Stripe, 2026. https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents
2. "Minions: Stripe's one-shot, end-to-end coding agents—Part 2." A. Gray, Stripe, 2026. https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents-part-2
3. "Why We Built Our Own Background Agent." Ramp Builders Blog, 2026. https://builders.ramp.com/post/why-we-built-our-background-agent
4. "Why Ramp built its own in-house coding agent, Inspect." The Pragmatic Engineer, 2026. https://newsletter.pragmaticengineer.com/p/why-ramp-built-inspect
5. "We built an agent to prompt our internal finance agent." Ramp Labs, c. 2025-2026. https://ramplabs.substack.com/p/we-built-an-agent-to-prompt-our-internal
6. "Building production-ready agentic systems: Lessons from Shopify Sidekick." A. McNamara (with B. Lafferty, M. Garner), Shopify Engineering, 2025. https://shopify.engineering/building-production-ready-agentic-systems
7. "Under the River." Shopify Engineering, 2026. https://shopify.engineering/under-the-river
8. "Introducing Roast: Structured AI workflows made easy." Shopify Engineering, 2025. https://shopify.engineering/introducing-roast
9. "Introducing the v0 composite model family." Vercel, 2025. https://vercel.com/blog/v0-composite-model-family
10. "How we made v0 an effective coding agent." Vercel, 2026. https://vercel.com/blog/how-we-made-v0-an-effective-coding-agent
11. "Breakout Agents: Replit." LangChain, 2024 (via digest). https://www.langchain.com/breakoutagents/replit
12. Replit blog post on decision-time guidance for long-trajectory agents. Replit, 2026 (via digest). https://replit.com/blog/decision-time-guidance
13. Replit blog post on evaluating and improving Replit Agent at scale (ViBench, Telescope). Replit, 2026 (via digest). https://replit.com/blog/evaluating-and-improving-agent-at-scale
14. "The Self-Driving Company." Replit, 2026. https://ld.replit.com/blog/self-driving-company
15. "Replit democratizes software development with Claude on Google Cloud." Anthropic (customer story), 2026. https://claude.com/customers/replit
16. Fireside chat on Replit Agent v2 with M. Catasta (video), 2025 (via digest). https://www.youtube.com/watch?v=h_oUYqkRybM
17. "uReview: Scalable, Trustworthy GenAI for Code Review at Uber." S. Roy Choudhary, S. Mahajan, J. Wang et al., Uber Engineering, 2025. https://www.uber.com/us/en/blog/ureview/
18. Uber Engineering blog post on Enhanced Agentic RAG (EAg-RAG) for the Genie on-call copilot. Uber, 2025 (via digest). https://www.uber.com/en-NL/blog/enhanced-agentic-rag/
19. Uber Developer Platform talk on building AI developer tools with LangGraph (Validator, AutoCover) (video), 2025 (via digest). https://www.youtube.com/watch?v=Bugs0dVcNI8
20. Uber Engineering blog post on designing the MCP Gateway. Uber, 2026 (via digest). https://www.uber.com/gb/en/blog/designing-mcp-gateway/
21. Uber Engineering blog post introducing Finch, a conversational financial-data agent. Uber, 2025 (via digest). https://www.uber.com/en-IN/blog/unlocking-financial-insights-with-finch/
22. "How Uber uses AI for development." The Pragmatic Engineer, 2026 (via digest). https://newsletter.pragmaticengineer.com/p/how-uber-uses-ai-for-development
23. "Uber CTO Says It's Seeing 'Phenomenal' Results From AI Coding." Business Insider, 2026. https://www.businessinsider.com/uber-cto-ai-coding-agentic-software-engineers-2026-3
24. LinkedIn Engineering blog post on the technology behind Hiring Assistant, its first agent. LinkedIn, 2024 (via digest). https://www.linkedin.com/blog/engineering/generative-ai/the-tech-behind-the-first-agent-from-linkedin-hiring-assistant
25. LinkedIn Engineering blog post on how Hiring Assistant was engineered. LinkedIn, 2025 (via digest). https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant
26. InfoQ presentation by LinkedIn engineers on the Hiring Assistant agent platform. InfoQ, 2025 (via digest). https://www.infoq.com/presentations/LinkedIn-agent-hiring-assistant/
27. LinkedIn Engineering blog post on semantic search for AI agents at scale (MUSE retrieval and ranking for Hiring Assistant). LinkedIn, 2026 (via digest). https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant
28. LinkedIn Engineering blog post on Contextual Agent Playbooks and Tools (CAPT). LinkedIn, 2026 (via digest). https://www.linkedin.com/blog/engineering/ai/contextual-agent-playbooks-and-tools-how-linkedin-gave-ai-coding-agents-organizational-context
29. LinkedIn Engineering blog post on personalization with the Cognitive Memory Agent. LinkedIn, 2026 (via digest). https://www.linkedin.com/blog/engineering/ai/the-linkedin-generative-ai-application-tech-stack-personalization-with-cognitive-memory-agent
30. Airbnb Engineering blog post on accelerating large-scale test migration with LLMs. Airbnb, 2025 (via digest). https://medium.com/airbnb-engineering/accelerating-large-scale-test-migration-with-llms-9565c208023b
31. Airbnb Engineering blog post on Automation Platform v2 for conversational AI. Airbnb, 2024 (via digest). https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb
32. "Beyond the CLI: Agentic AI for async workloads and non-developers" (C. Sanson and M. Capps, Airbnb). DX podcast/transcript, 2026. https://getdx.com/podcast/beyond-the-cli-agentic-ai-for-async-workloads-and-non-developers/
33. Booking.com (booking.ai) blog post on building a GenAI agent for partner-guest messaging. Booking.com, 2025 (via digest). https://booking.ai/building-a-genai-agent-for-partner-guest-messaging-f54afb72e6cf
34. Booking.com (booking.ai) blog post on AI agent evaluation. Booking.com, 2026 (via digest). https://booking.ai/ai-agent-evaluation-82e781439d97
35. "Delegating Engineering Work To Cloud-Based Agents." DoorDash Engineering, 2026. https://careersatdoordash.com/blog/delegating-engineering-work-to-cloud-based-agents/
36. "How DoorDash built an AI code reviewer engineers actually listen to." DoorDash Engineering, 2026. https://careersatdoordash.com/blog/doordash-built-an-ai-code-reviewer-engineers-actually-listen-to/
37. "Beyond Single Agents: How DoorDash is building a collaborative AI ecosystem." DoorDash Engineering, 2025. https://careersatdoordash.com/blog/beyond-single-agents-doordash-building-collaborative-ai-ecosystem/
38. DoorDash Engineering blog post on building a centralized gateway for AI agent tool access. DoorDash, 2026 (via digest). https://careersatdoordash.com/blog/how-doordash-built-a-centralized-gateway-for-ai-agent-tool-access/
39. DoorDash Engineering blog post on automating feature-flag cleanup with a multi-agent LLM system. DoorDash, 2026 (via digest). https://careersatdoordash.com/blog/automating-feature-flag-cleanup-at-scale-with-a-multi-agent-llm-system/
40. DoorDash Engineering blog post on LLM-based Dasher support automation with guardrails. DoorDash, 2024 (via digest). https://careers.doordash.com/blog/large-language-modules-based-dasher-support-automation/
41. "Notion's Token Town: 5 Rebuilds, 100+ Tools, MCP vs CLIs and the Software Factory Future" (S. Last and S. Sachs). Latent Space, 2026 (via digest). https://www.latent.space/p/notion
42. "How we built security into Custom Agents." Notion, 2026. https://www.notion.com/blog/how-we-built-security-into-custom-agents
43. "Meet Scruff, Security's New AI Teammate." Notion, c. 2026. https://www.notion.com/blog/meet-scruff-securitys-new-ai-teammate
44. "Notion builds agent orchestration where teams and Claude collaborate on real work." Anthropic (customer story), 2026. https://claude.com/customers/notion
45. "Canva empowers employees across teams with Claude." Anthropic (customer story), 2025. https://claude.com/customers/canva
46. "How Anthropic teams use Claude Code." Anthropic, 2025. https://claude.com/blog/how-anthropic-teams-use-claude-code
47. "How AI is transforming work at Anthropic." Anthropic, 2025. https://www.anthropic.com/research/how-ai-is-transforming-work-at-anthropic
48. "How we built our multi-agent research system." Anthropic, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
49. "Project Vend: Can Claude run a small shop? (And why does that matter?)." Anthropic and Andon Labs, 2025. https://www.anthropic.com/research/project-vend-1
50. "Project Vend: Phase two." Anthropic, 2025. https://www.anthropic.com/research/project-vend-2
51. "Building a C compiler with a team of parallel Claudes." N. Carlini, Anthropic, 2026. https://www.anthropic.com/engineering/building-c-compiler
52. "Effective harnesses for long-running agents." Anthropic, 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
