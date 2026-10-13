# 06. Coding Agents: Architectures, Sandboxes, and Verification Loops

## Overview

Coding agents are the most heavily used and most thoroughly documented kind of LLM agent. They suit the agent pattern well for three reasons:

- **The environment is text.** Source files, shell output and diffs are all text.
- **Actions are cheap to undo.** git, checkpoints and disposable VMs make rollback easy.
- **Success can often be checked by machine.** Tests, compilers, linters and type checkers give a signal. Anthropic's "Building Effective Agents" makes this point directly: coding agents work because "solutions can be checked with automated tests", which gives the loop feedback to iterate on. Human review is still needed to judge whether a change fits the wider system [22].

This chapter covers the architectures behind the main public coding agents:

- **Research systems.** SWE-agent and its Agent-Computer Interface (ACI), mini-SWE-agent, OpenHands/CodeAct and Agentless.
- **Terminal and IDE agents.** Claude Code, OpenAI Codex CLI, Aider, Cursor and Amp.
- **Asynchronous cloud agents.** Claude Code cloud sessions, Codex cloud, Devin, Google Jules and the GitHub Copilot coding (cloud) agent.

Two concerns run through every section:

1. **Sandboxing.** How do you let a model run arbitrary shell commands without letting it destroy, exfiltrate or escalate?
2. **Verification loops.** How does the agent, or the harness around it, decide that the work is actually done?

By 2026 the designs had converged on a recognizable shape. One model runs in a tool loop with a small set of general tools, usually a shell, a file editor and search. The loop runs inside an OS-level or VM-level sandbox with an egress proxy. Persistent repository instructions (CLAUDE.md, AGENTS.md, AGENT.md) supply project context. The run ends at a "verify" phase that ranges from "run the tests" to a separate classifier or reviewer model.

> **Source note.** The research tooling could reach some vendor sites but not others. Primary documentation was read in full for Anthropic, GitHub (via the public `github/docs` repository), OpenAI's open-source Codex repositories, SWE-agent, OpenHands and Aider. For Cursor, Devin, Jules and Amp, this chapter relies on those vendors' own GitHub-hosted material and on partner case studies. It deliberately omits product claims that could not be checked in a primary source. Treat those sections as narrower in scope, not as evidence that the systems are simpler.

---

## 1. The reference architecture

Almost every system in this chapter is a variant of the loop below. The variation lies in the *tool surface* (one bash tool or a curated ACI), the *edit format* (whole file, search/replace, unified diff, patch DSL), the *context strategy* (agentic search, a repository map, embeddings, subagents, compaction), the *isolation boundary* (none, OS sandbox, container, VM) and the *stop condition*.

```text
            +--------------------------------------------------------------+
 user task  |                         HARNESS                              |
 ---------> |  system prompt + repo instructions (CLAUDE.md / AGENTS.md)    |
            |  context manager: compaction, subagents, tool-output caps    |
            |                                                              |
            |   +---------+  tool call   +-------------------------------+ |
            |   |  LLM    | -----------> | permission / policy layer     | |
            |   | (loop)  |              | allowlists, classifier, human | |
            |   +---------+              +---------------+---------------+ |
            |        ^                                   |                 |
            |        | observation                       v                 |
            |        |                   +-------------------------------+ |
            |        +------------------ | SANDBOX (OS / container / VM) | |
            |                            | shell, editor, search, tests  | |
            |                            | egress proxy, scoped creds    | |
            |                            +---------------+---------------+ |
            |                                            |                 |
            |   verify: tests | lint | build | screenshot | reviewer model |
            +--------------------------------------------+-----------------+
                                                         v
                                        diff / commit / pull request
```

Anthropic's documentation describes the Claude Code loop as three blended phases: **gather context, take action, verify results**. It calls the layer around the model, which provides the tools and manages the context, the *agentic harness* [14]. The Claude Agent SDK post generalizes the same cycle and lists three ways to verify [25]:

- **Rules**, such as linting, with explanations of why a check failed.
- **Visual feedback**, such as screenshots checked against layout requirements.
- **LLM-as-judge**, which it calls "less robust" and slower, but useful for fuzzy criteria.

---

## 2. SWE-agent and the Agent-Computer Interface (Princeton/Stanford, 2024)

### Problem

Early experiments gave LMs a raw Linux shell. They failed in mundane ways:

- `cat` on a 2,000-line file flooded the context window.
- `sed`-based edits silently corrupted indentation.
- Commands with no output left the model unsure whether anything had happened.

The SWE-agent paper (Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan, Press; NeurIPS 2024) argues that LM agents are a new kind of end user. They deserve interfaces designed for them, in the same way that HCI designs interfaces for people [1][2].

### Architecture: the ACI

The ACI is a small set of "LM-centric commands and feedback formats" for browsing a repository and for viewing, editing and executing code [4]. The documented design choices are [3]:

- **Windowed file viewer.** A custom viewer replaces `cat`. It shows **100 lines per turn** and supports scrolling and in-file search.
- **Succinct search.** Searching a whole directory returns *only the names of matching files*, with no match context, because the authors found that extra context confused the model.
- **Edit with a linter guardrail.** Every edit runs a linter. If the result is syntactically invalid, the edit is rejected, so a broken file never persists and the model gets an immediate, localized error.
- **Explicit empty-output feedback.** A command with no output returns "Your command ran successfully and did not produce any output."

```text
  LM --(thought + ONE command)--> ACI command layer --> shell in Docker
   ^                                |  open/goto/scroll (100-line window)
   |                                |  find_file / search_dir (names only)
   |                                |  edit start:end  --> lint --> accept|reject
   +----- formatted observation <---+  "ran successfully, no output"
```

Configuration lives in a single YAML file. Demonstrations, which are full trajectories converted to YAML, can be injected to teach the model the command vocabulary [1].

### Results and lessons

The project reports resolving **about 12.3% of the full SWE-bench test set** (12.29% in the project documentation), which was state of the art at publication [4]. Later releases with stronger models led SWE-bench Verified: SWE-agent 1.0 with Claude 3.7 Sonnet was state of the art there in February 2025. SWE-agent-LM-32B became the best open-weights result in May 2025 [1]. The infrastructure was then split out into **SWE-ReX**, a runtime interface for sandboxed shells. It can run against local, Docker, AWS Fargate, Modal or Daytona backends without changing agent code. It detects when a command has finished, captures the exit code, supports interactive tools such as `ipython` and `gdb`, and keeps several shell sessions open at once. The project demonstrates 30 SWE-bench instances running in parallel [6].

The **ACI lesson** has outlasted the specific commands. Anthropic's "Building Effective Agents" makes "agent-computer interface" a first-class design concern. It recommends poka-yoke tool arguments and reports that the SWE-bench team "spent more time optimizing tools than the overall prompt" [22].

### mini-SWE-agent: the minimal counterpoint (2025)

The same group's maintainers now say that **mini-SWE-agent** has superseded SWE-agent [1]. It is a deliberate reversal of the ACI's complexity [5]:

- **Bash is the only tool.** It does not even use the LM's tool-calling API, so any model works.
- **Linear history.** The trajectory *is* the prompt: every step is appended to one message list. This makes debugging and fine-tuning data trivial to produce.
- **Stateless execution.** Each action is an independent `subprocess.run`. Sandboxing means swapping in `docker exec`, Podman, Apptainer or bubblewrap.
- **About 100 lines** for the agent class. It scores **above 74% on SWE-bench Verified** with frontier models (an earlier note reports 65%) [1][5].

The lesson is that once models became strong shell users, much of the bespoke ACI scaffolding stopped paying for itself. The *principles* stayed: bounded outputs, clear error messages and explicit success feedback. They moved into the tool descriptions and output caps of general tools.

---

## 3. OpenHands (formerly OpenDevin) and CodeAct

### CodeAct: code as the action space

CodeAct (Wang et al., ICML 2024) proposes that an agent should act by emitting *executable code*, Python run in an interpreter, rather than JSON tool calls or text commands. The agent can then revise its actions over several turns based on execution results. Across 17 LLMs on API-Bank and the M3ToolEval benchmark, CodeAct reached **up to 20% higher success** than text or JSON actions [12][13]. The release included CodeActInstruct, about 7k multi-turn trajectories, and fine-tuned CodeActAgent models.

### OpenHands platform architecture

OpenHands ("An Open Platform for AI Software Developers as Generalist Agents", arXiv 2407.16741, 2024) became the main open-source home of CodeAct-style coding agents [8]. Its 2024-2025 deployment model ran an app container that launched a **separate sandbox runtime container** for agent execution (`SANDBOX_RUNTIME_CONTAINER_IMAGE`) via the mounted Docker socket. The README also stated plainly that it was a *single-user* local tool with no built-in authentication or multi-tenant isolation [7].

In late 2025 the project was refactored into the **OpenHands Software Agent SDK** (arXiv 2511.03690) [9][10]. The SDK is split into four packages [11]:

| Package | Role |
|---|---|
| `openhands.sdk` | Agent (reason-act loop), provider-agnostic LLM, Conversation (state/lifecycle), typed Events, Condenser (history compression), Skills, Security (risk assessment before execution) |
| `openhands.tools` | Prebuilt bash, file-editing and other tools on Action/Observation/Executor base classes; MCP integration |
| `openhands.workspace` | `LocalWorkspace`, `RemoteWorkspace`, Docker and remote-API workspaces |
| `openhands.agent_server` | FastAPI service: REST and WebSocket for conversations, bash, files and events; OpenAI-compatible endpoints; per-user isolation |

```text
  Agent Canvas (UI) --TS client--> Agent Server (FastAPI, REST+WS)
                                     |  conversations, events, files
                                     v
            +--------------------------------------------------+
            | Conversation --> Agent loop --> Tool executors   |
            |      ^              |   ^           |             |
            |   Events <---- Condenser |      Security analyzer |
            +--------------------------|-----------------------+
                                       v
            Workspace: Local (same process) | Docker | Remote API
```

The central design idea is that **the same agent code runs locally or sandboxed; only the workspace type changes** [11]. The current Canvas supports three deployments [7]:

- No sandbox, running directly on the host.
- A single Docker sandbox.
- **One container per conversation** (`OH_CONVERSATION_RUNTIME=docker`), with its own Agent Server, workspace mount and persisted state.

### Lessons

- *Separate the agent from the runtime by a protocol.* With a server API in between, the same loop can drive a laptop, a container fleet or a hosted cloud.
- *Typed event logs* make replay, UI streaming and evaluation share one data model.
- Security analysis of each action, plus a condenser, are first-class components, not afterthoughts.

---

## 4. Agentless: the non-agent baseline

Agentless (Xia, Deng, Dunn, Zhang, 2024) asked whether autonomy is needed at all [49][50]. It runs a fixed three-phase pipeline:

1. **Hierarchical localization**: file, then class or function, then edit location.
2. **Repair**: sample multiple candidate patches in a simple diff format.
3. **Patch validation**: select regression tests, generate a reproduction test, and re-rank candidates by test results.

v1.0 resolved 27.3% of SWE-bench Lite at **about $0.34 per issue**, the best open-source result at the time. A December 2024 update with Claude 3.5 Sonnet reported 40.7% on Lite and 50.8% on Verified [49].

Agentless matters architecturally because it isolated **test-based candidate selection** as a separable, powerful component. The same idea reappears in Aider's retry harness and in best-of-N cloud agents.

---

## 5. Claude Code (Anthropic)

### Problem and positioning

Claude Code is a terminal-native agent. It can do "anything you can do from the command line" in the user's own repository. It runs in the terminal, IDE extensions, a desktop app, a browser (claude.ai/code), Slack and CI, all over the same loop [14].

### Architecture

**Tools** fall into five categories [14]:

- File operations: read, edit, write.
- Search: glob and regex, ripgrep-backed.
- Execution: shell commands, tests, git.
- Web: search and fetch.
- Code intelligence: type errors after edits, jump-to-definition, via plugins.

There are also orchestration tools for spawning subagents and asking the user questions.

**Context management** is the stated central constraint: "Claude's context window fills up fast, and performance degrades as it fills" [15]. The mechanisms are:

- **CLAUDE.md** project memory, loaded every session. AGENTS.md is read when present.
- **Auto memory**: the first 200 lines or 25KB of MEMORY.md [14].
- **On-demand skills**: only descriptions load until a skill is used.
- **Deferred MCP tool definitions**, loaded through tool search [14].
- **Subagents** with separate context windows that return summaries [14][15].
- **Auto-compaction**, which first clears old tool outputs and then summarizes. It stops with an error instead of thrashing when a single output refills the window [14].
- **Tool-response caps**: Claude Code caps tool responses at 25,000 tokens by default [23].

**Sessions and reversibility.** Every message, tool use and result is written to a local JSONL transcript. That enables resume and fork, and files are snapshotted before each edit for checkpoint rewind. Checkpoints do *not* cover side effects of shell commands or remote systems, which "can't be checkpointed" [14].

**Edit primitive.** Anthropic's SWE-bench agent, the predecessor of Claude Code's tool design, settled on `str_replace`-style editing. The replacement applies only if `old_str` matches *exactly once*; zero or multiple matches return an error so the model retries. The edit tool also requires **absolute paths**, because the model mangled relative paths after `cd` [21]. That minimal scaffold (prompt, bash tool, edit tool) scored 49% on SWE-bench Verified with Claude 3.5 Sonnet (new) in January 2025. Some successful runs took "hundreds of turns" and more than 100k tokens [21].

### The permission and sandbox stack

Claude Code layers several independent controls:

```text
 tool call
   |
   v
 [permission rules]  allow / ask / deny patterns; org > user > project scopes
   |
   v
 [mode]  manual | accept-edits | plan (read-only) | auto (classifier)
   |
   v
 [auto mode]  input: prompt-injection probe on tool OUTPUTS
   |          output: 2-stage transcript classifier on ACTIONS
   v
 [OS sandbox for shell]  bubblewrap (Linux) / Seatbelt (macOS)
   |   fs: write = cwd + temp;   net: only via host proxy (domain allowlist)
   v
 [checkpoints]  file snapshots for rewind (not shell side effects)
```

**OS-level sandbox (October 2025).** Anthropic's engineering post introduced a sandboxed bash tool built on **bubblewrap on Linux and Seatbelt on macOS**. The restrictions apply to every subprocess a command spawns [16]:

- **Filesystem.** Writes are confined to the working directory.
- **Network.** Network traffic leaves only through a Unix domain socket to a proxy *outside* the sandbox, which enforces domain allowlists and prompts for new domains.

In internal use, sandboxing **cut permission prompts by 84%** [16]. The post argues that both layers are required: without network isolation, a compromised agent can exfiltrate SSH keys; without filesystem isolation, it can backdoor its way to the network [16].

The current documentation adds detail [17]:

- The runtime is open source (`@anthropic-ai/sandbox-runtime`). On Linux it depends on `bubblewrap` and `socat`, with an optional seccomp filter that blocks Unix domain sockets.
- **Default reads cover most of the machine, including `~/.ssh`**, unless `denyRead` is configured. Defaults are permissive for reads and strict for writes and egress.
- A **credential-masking** mode hands sandboxed commands a per-session sentinel value. The proxy swaps in the real secret only on requests to allowed hosts. This requires TLS termination at the proxy, and AWS SigV4 requests are re-signed.
- An **escape hatch** (`dangerouslyDisableSandbox`) lets the model retry a failed command unsandboxed through the normal permission flow. Administrators can disable it.

The docs are candid about the limits [17]:

- The proxy filters on client-supplied hostnames without inspecting TLS, so allowing broad domains such as `github.com` can enable exfiltration via *domain fronting*.
- Allowing `/var/run/docker.sock` is effectively host access.
- The `enableWeakerNestedSandbox` mode for Docker "considerably weakens security".
- The sandbox wraps only shell commands. Built-in file and web tools, MCP servers and hooks follow permission rules instead.

**Auto mode (March 2026).** Users approved about **93%** of permission prompts, a sign of approval fatigue. Auto mode replaces most of those prompts with a classifier [18]:

- **Tiers.** Read-only tools and in-project edits bypass the classifier. On entering auto mode, broad allow rules that grant arbitrary code execution are dropped. Everything else (shell, web fetches, integrations, subagent spawns, writes outside the project) goes to a **transcript classifier running on Sonnet 4.6**.
- **Two stages.** Stage 1 is a fast single-token yes/no filter tuned to over-block. Stage 2 runs only on flagged actions and adds chain-of-thought to reduce false positives.
- **Reasoning-blind.** The classifier sees user messages and tool calls but not assistant prose or tool outputs. The agent therefore cannot argue its way past it, and injected content in tool results has less pull.
- **Measured rates** [18]:

  | Dataset | Stage 1 | Full pipeline |
  |---|---|---|
  | Real internal traffic, n=10,000 (false-positive rate) | 8.5% | **0.4%** |
  | Real over-eager actions, n=52 (false-negative rate) | 6.6% | **17%** |
  | Synthetic exfiltration, n=1,000 (false-negative rate) | 1.8% | **5.7%** |

- **Deny-and-continue.** A denial is returned to the agent with a reason so it can look for a safer path. Three consecutive or 20 total denials escalate to the human. Headless runs terminate instead.

The main miss mode was instructive. The classifier usually *recognizes* the dangerous action but misjudges whether the user's earlier consent covered its full impact [18].

### Cloud sessions (Claude Code on the web)

Each cloud session runs in an **isolated Anthropic-managed VM**: 4 vCPUs, 16 GB RAM and 30 GB disk [19][20]. Network access is set per environment [20]:

- **None**: no outbound access.
- **Trusted** (default): package registries, GitHub and cloud SDKs.
- **Full**: any domain.
- **Custom**: your own allowlist.

**Git credentials never enter the VM.** A custom proxy handles all git traffic. The sandboxed client authenticates to it with a scoped credential, and the proxy checks the operation itself, for example that a push targets only the configured branch, before attaching the real GitHub token [16][20].

The setup script runs once per environment. If it finishes in about five minutes, the filesystem is **snapshotted and reused** as the starting point for later sessions; the cache expires after about seven days [20]. Sessions can watch a pull request and **auto-fix** CI failures and review comments. Because GitHub emits no webhook when a base branch creates a conflict, auto-fix cannot react to merge conflicts on its own [19].

### Verification loop

Anthropic's best-practice guidance is to "give Claude a check it can run," because "Claude stops when the work looks done" [15]. The documentation ranks four increasingly strict gates [15]:

1. Ask for the check in the prompt ("run the tests and iterate").
2. A `/goal` condition that a separate evaluator re-checks after every turn.
3. A **Stop hook**: a deterministic script that blocks the turn from ending until the check passes.
4. A **verification subagent** or reviewer session in a fresh context, so the agent doing the work is not the one grading it.

The guidance also warns that a reviewer prompted to find gaps "will usually report some, even when the work is sound." It recommends limiting such reviews to correctness and stated requirements [15].

### Long-running harnesses

For multi-session builds, Anthropic describes a two-agent harness [24]:

- An **initializer** writes `init.sh`, a `claude-progress.txt` log, an initial git commit and a JSON feature list. The claude.ai clone example had 200+ features, each with `passes: false`.
- Each later **coding agent** session runs `pwd`, reads git log and the progress file, picks the highest-priority failing feature, runs `init.sh` and a smoke test, implements *one* feature, verifies it end-to-end with Puppeteer, and commits.

JSON was chosen because models are less likely to overwrite it inappropriately than Markdown. The failure modes this design addresses are trying to one-shot the whole app, declaring victory too early, and marking features done without testing them [24].

---

## 6. OpenAI Codex (CLI and cloud)

### Architecture

Codex CLI is an Apache-2.0 open-source agent "that runs locally on your computer." It is distinct from the cloud agent, Codex Web [26]. The core is in Rust (`codex-rs`). The crate layout shows how the harness is decomposed [26]:

- `core`
- `exec` / `exec-server`
- `app-server` (used by IDE integrations)
- `apply-patch`
- `execpolicy`
- `sandboxing`, `linux-sandbox`, `bwrap`, `process-hardening`
- `network-proxy`
- `codex-mcp` / `rmcp-client`
- `hooks`, `skills`, `memories`
- `rollout` (session persistence)
- `otel`
- `cloud-tasks`

Model-specific system prompts ship as files in the repository (for example `gpt_5_codex_prompt.md`). They tell the agent to [29]:

- prefer `rg` for search;
- use `apply_patch` for single-file edits, but not for generated output or bulk find-and-replace;
- keep a plan tool updated for non-trivial work and skip single-step plans;
- treat the worktree as possibly dirty: never revert changes it did not make, never amend commits unless asked, and avoid destructive git commands without approval;
- in review mode, list findings first, ordered by severity with file and line references.

Repository guidance comes from **AGENTS.md**. Codex merges `~/.codex/AGENTS.md`, the repo-root file and the current-directory file, top-down [27]. Codex can also act as an MCP *server* (`codex mcp`) and runs headless through `codex exec` for CI [27].

### Sandbox and approvals

Codex separates **what the sandbox permits** (`--sandbox read-only | workspace-write | danger-full-access`) from **when to ask the human** (`--ask-for-approval`, for example `on-request` or `never`). The recommended default for version-controlled folders is `workspace-write` plus `on-request`. Codex edits and runs commands in the workspace freely, but writes outside it, network access and git updates need approval. **Network is blocked by default** and can be enabled per sandbox mode [27].

The platform mechanisms have evolved:

- **macOS.** Commands run under **Seatbelt** (`sandbox-exec`) with a profile matching the sandbox mode [27].
- **Linux, early 2025.** Releases used **Landlock + seccomp** [27].
- **Linux, current.** The Linux helper uses **bubblewrap** by default and keeps Landlock only as a legacy option, rejected for filesystem-restricted policies because it cannot isolate app-server Unix sockets [28].

The current Linux design [28]:

```text
 bwrap --ro-bind / /                     # whole FS read-only
       --bind <root> <root>              # writable roots layered on top
       --ro-bind <root>/.git ...         # .git, gitdir target, .codex re-protected
       --unshare-user --unshare-pid --proc /proc
       --unshare-net [+ TCP->UDS->TCP bridge to managed proxy]
 then: PR_SET_NO_NEW_PRIVS + seccomp
       (deny network syscalls if restricted; always deny io_uring;
        deny new AF_UNIX sockets unless explicitly allowed)
```

Three details are worth copying:

1. **Re-protecting `.git` inside a writable root.** The agent can edit code but cannot rewrite history or hooks.
2. **Blocking `io_uring`.** It could otherwise bypass the socket-family filters.
3. **Masking symlinks and missing path components with `/dev/null`**, which defeats path-swap tricks [28].

Inside Docker, Codex's guidance is to rely on the container as the boundary and run with `danger-full-access` [27].

### Cloud environments

OpenAI publishes `codex-universal`, a reference implementation of the base Docker image used in Codex environments. It carries multi-version Python, Node, Rust, Go, Swift, Ruby, PHP and Java toolchains, selected with `CODEX_ENV_*` variables. OpenAI notes that it is "not an identical environment" to production [30].

---

## 7. Aider: the pair-programmer architecture

Aider is the longest-running open-source terminal coding assistant. Its author describes it as "intentionally" having limited agentic behavior, keeping the user in control [40]. Its research contributions are mostly about **context selection** and **edit formats**, both measured on public benchmarks.

### Repository map

Instead of embeddings or RAG, Aider builds a **repo map** from static analysis of syntax trees and the call graph [35][40]:

- The map lists key classes and functions with their signatures.
- It is ranked with a graph algorithm in which files are nodes and dependencies are edges.
- It is trimmed to a token budget (`--map-tokens`, **default 1k**). The budget expands when no files are in the chat.

The model uses the map to name files it needs, and Aider asks before adding them [40].

### Edit formats as an engineering variable

Aider treats the output format as a tunable component [36]:

- **whole**: rewrite the entire file.
- **diff**: SEARCH/REPLACE blocks with merge-conflict-style markers.
- **diff-fenced**: the file path inside the fence; used for Gemini models.
- **udiff**: simplified unified diffs.
- **editor-diff / editor-whole**: for architect mode.

Its unified-diff study with GPT-4 Turbo is a key data point on "lazy coding" [37]:

- On the laziness benchmark, switching from SEARCH/REPLACE to **unified diffs raised the score from 20% to 61%**, and lazy placeholder comments fell from 12 tasks to 4.
- Disabling *flexible patching* (hunk normalization, relative indentation, splitting large hunks) caused **9x more editing errors**.
- Emotional-appeal prompts ("I'm blind", tipping) made results worse.

The rationale behind unified diffs matches Anthropic's appendix on tool formats: use formats the model has seen widely on the internet, and avoid line counting and JSON escaping [22][37].

### Architect/Editor split

In 2024, Aider separated *reasoning about the change* (an Architect model) from *emitting well-formed edits* (an Editor model). Reasoning models such as o1 solved problems well but formatted edits poorly. o1-preview as Architect with DeepSeek or o1-mini as Editor reached **85%** on Aider's editing benchmark, against a previous best of about 79.7%. Even same-model pairs (Sonnet with Sonnet, GPT-4o with GPT-4o) beat their solo baselines [38].

### Lint-and-test loop and retry harness

After every edit, Aider lints with built-in **tree-sitter** parsers, where `ERROR` nodes mark syntax faults, or with user linters. It shows each error *inside its enclosing function or class*, using elision markers, because "LLMs are quite bad at working with source code line numbers" [39].

For SWE-bench, the harness retried until a solution was **plausible**: edits applied, no lint errors, and the repository's *pre-existing* tests passing. It alternated GPT-4o and Opus for up to six attempts, which gave **26.3% on SWE-bench Lite** [40]. With two attempts on the full set it reached **18.9%**, against 17.0% for GPT-4o alone [41]. That is an early, clean example of *verification-driven best-of-N*.

---

## 8. Cursor (Anysphere)

Cursor is an IDE-first agent. The public, verifiable material from Cursor itself shows three architectural facts.

1. **Programmable agent runtime.** The Cursor SDK (TypeScript) runs "Cursor's coding agent from your own apps." The same agent runs across **local workspaces and cloud runtimes**, streams agent events, and exposes prompts, models, cancellation, artifacts and conversation state to code. An open `sdk.v1` bridge (`cursor-sdk-bridge`) allows other languages [33].
2. **Cloud agents can be self-hosted.** A documented lab runs Cursor Cloud Agent workers on customer AWS, either EC2 with Docker or ECS/Fargate. The agent loop stays a service, while the execution sandbox moves into the customer's network boundary [33].
3. **Hooks around agent events**: prompt submission, shell commands, file edits and completion. They are used for audit logging, guards on sensitive prompts and follow-up checks [33]. This is the same deterministic-gate idea as Claude Code's hooks.

Cursor's open-source **Priompt** library (`@anysphere/priompt`) shows how it approaches context: prompts are JSX components with **priorities**, and a renderer picks the minimum priority cutoff that fits the token budget [34]. Its authors note two drawbacks: priorities everywhere become an anti-pattern, and priority-based dropping makes prompts harder to cache [34]. That tension between dynamic context packing and prompt caching recurs across every agent in this chapter.

---

## 9. Devin (Cognition)

Devin launched in early 2024 as an "autonomous AI software engineer" whose work runs inside VM sandboxes [31]. Cognition published its SWE-bench methodology, harness and output diffs in a public repository [32].

The most detailed recent public account of Devin's internals comes from Cognition's case study with Anthropic [31]:

- **Model routing.** Work is routed across several models, with Claude used for long-horizon autonomous tasks.
- **Model-specific harness.** Prompts, tool definitions, context management, *trajectory monitoring* and guardrails are tuned per model. "The harness changes whenever the model mix changes."
- **Eval-gated model swaps.** Internal evals cover file-edit mechanics, file localization and end-to-end tasks, so a strictly better model can ship on day one without regressions.
- **Trajectory monitoring.** The system detects when an agent drifts and steers it back.
- **Sandbox engineering is hard.** Cognition has rebuilt VM sandboxing "many times"; getting an agent running quickly with the latest code, dependencies and repos is a hard problem.
- **Scale signals.** Claude Sonnet 3.6 brought a 3.5x increase in merged PRs per week. Usage grew at least 7x. Roughly 50-70 engineers work on the model-integration surface [31].

The lesson for designers is that the harness is a model-specific artifact maintained by a sizable team, not a static framework.

---

## 10. Google Jules

Jules is Google Labs' asynchronous coding agent. Per Google's own GitHub Action, it is "powered by Gemini 3 Pro" and "works autonomously in a **cloud VM**", analyzing code, implementing features, fixing bugs and opening pull requests for human review [42].

The Jules API can be triggered from any GitHub event: issues, PRs, cron, `workflow_dispatch` or CI failures. Optional context includes the last commit or the commit log [42]. Two design points stand out:

- The README recommends **measurable targets**, because Jules "can verify its own results against them."
- It warns that **issue-triggered workflows need an allowlist**, since anyone can open an issue and thereby inject a prompt [42].

---

## 11. GitHub Copilot coding agent (Copilot cloud agent)

### Architecture

The Copilot agent takes a task from an issue assignment, a PR comment or Chat. It then [43][45]:

- creates a branch;
- works in an **ephemeral, firewalled development environment** powered by GitHub Actions, where it can edit code, run tests and run linters;
- opens a pull request and updates its description;
- iterates on `@copilot` review comments.

The environment is customized with `.github/workflows/copilot-setup-steps.yml`. It must contain a single job named `copilot-setup-steps`, takes effect only on the default branch, and honors only these keys: `steps`, `permissions`, `runs-on`, `services`, `snapshot` and `timeout-minutes`, with a **59-minute maximum**. A failing setup step does not abort the agent; it starts with whatever environment exists at that point. Runners can be the standard hosted runner, larger runners (including Azure private networking) or self-hosted ARC scale sets. Only Ubuntu x64 and Windows 64-bit are supported [45].

### Security model

GitHub's controls are notable for being **repository-governance controls**, not just process sandboxing [43][44]:

| Control | Mechanism |
|---|---|
| Trigger authority | Responds only to users with write access; comments from others are never shown to the agent |
| Branch scope | Pushes only to a new `copilot/` branch or the existing PR branch; never the default branch; branch protections still apply |
| Git capability | Only simple pushes; cannot run `git push` or other git commands directly; cannot approve, mark ready or merge |
| Separation of duties | The requester cannot approve the agent's PR |
| CI gating | Actions workflows on agent PRs need "Approve and run workflows" by a writer (admins may relax this) |
| Secrets | No access to org or repo Actions secrets; only the `copilot` environment's secrets |
| Egress | Firewall on by default; not compatible with self-hosted runners or Windows, where you must control egress yourself |
| Prompt injection | Hidden characters and HTML comments filtered from issues and comments |
| Provenance | Commits authored by Copilot, co-authored by the requester, signed ("Verified"), each linked to session logs |

### Verification loop

Beyond the repository's own tests and linters, the agent runs **CodeQL**, **secret scanning** and **dependency checks against the Advisory Database** on its new code and tries to fix what they find. It does this without requiring a GitHub Advanced Security license, and the session logs show what was analyzed [43][44].

---

## 12. Amp (Sourcegraph)

Amp's published guides describe a thread-centric agent [46]:

- **Workflow.** A thinking model interprets the request and gathers context. The agent plans work as TODOs (`todo_read`, `todo_write`), orchestrates tools, delegates heavy work to **subagents** through a `Task` tool, and validates that output still matches the goal.
- **Tool surface.** `bash`, `create_file`, `edit_file`, `read_file`, `list_directory`, `glob`, `grep`, `codebase_search_agent` (search delegated to an agent), `web_search`, `read_web_page`, `undo_edit`, `mermaid` and `get_diagnostics`, plus GitHub, Linear and Slack connectors and MCP over stdio, Docker or SSE.
- **Context.** **AGENT.md** files at root, subdirectory and global scope (`~/.config/AGENT.md`) act as a "persistent memory layer." Threads can be **compacted** near the context limit or **forked** to start fresh while keeping history.
- **Toolboxes.** Users drop executable tool files into a directory named by `AMP_TOOLBOX`, and the CLI exposes them to the agent [47].

The community workshop "how to build a coding agent" reproduces the core of this class of agent in Go [48]:

- a REPL;
- a tool loop that runs until the model returns text only;
- five tools: `read_file`, `list_files`, `bash`, `edit_file` and a ripgrep-backed `code_search`.

Like mini-SWE-agent, it shows how small the essential loop is. The engineering effort lies in the tools, context and safety around the loop.

---

## 13. Cross-system comparison

| System | Loop locus | Primary tools / action space | Edit primitive | Context strategy | Isolation boundary | Built-in verification |
|---|---|---|---|---|---|---|
| SWE-agent [1][3] | Local/Docker via SWE-ReX | Curated ACI commands | Line-range edit + lint gate | 100-line viewer, names-only search | Docker container | Linter rejects invalid edits |
| mini-SWE-agent [5] | Any | Bash only | Shell (sed, heredocs) | Linear full history | `docker exec`/bwrap/etc. | Whatever the model runs |
| OpenHands SDK [11] | Agent Server | CodeAct-style tools, MCP | File-edit tool | Condenser, skills | Per-conversation container | Security analyzer before actions |
| Agentless [49] | Pipeline | None (fixed phases) | Sampled diffs | Hierarchical localization | Test harness | Repro + regression test re-rank |
| Claude Code [14][17][18] | Local, cloud VM, CI | Read/Edit/Bash/Search/Web, subagents, MCP | Exact-match `str_replace` | CLAUDE.md, skills, subagents, compaction, 25k output cap | bwrap/Seatbelt + egress proxy; VM in cloud | Stop hooks, `/goal`, reviewer subagents, auto-mode classifier |
| Codex CLI [27][28] | Local; cloud | Shell, `apply_patch`, plan, MCP | Patch DSL | AGENTS.md hierarchy | Seatbelt; bwrap+seccomp; network off | Prompted tests; review mode |
| Aider [35][40] | Local | Chat + edit formats | whole/diff/udiff, architect | Graph-ranked repo map (1k tokens) | None (git as undo) | Auto lint + test-cmd loop |
| Cursor [33][34] | IDE, cloud, SDK | Agent tools, hooks | (not documented here) | Priority-packed prompts | Cloud workers, self-hostable on AWS | Hooks on edit/shell/stop |
| Devin [31] | Cloud VM | Agent tools + third-party MCPs (logs, incidents) | (not public) | Model-specific harness | Full VM | Trajectory monitoring, evals |
| Jules [42] | Cloud VM | (not public) | (not public) | Commit log / last commit | Cloud VM | Verifies against stated targets; PR review |
| Copilot agent [43][45] | GitHub Actions | Edit, test, lint, MCP | (not public) | Custom instructions, code search | Ephemeral runner + firewall | Tests, CodeQL, secret and dependency scanning |
| Amp [46] | Local CLI/IDE | Bash, file, grep, search agent, Task | `edit_file`, `undo_edit` | AGENT.md, subagents, fork/compact | (not documented here) | `get_diagnostics`, validation step |

---

## 14. Cross-cutting techniques

### 14.1 Sandboxing patterns

There are four tiers, each with a different threat model:

1. **No sandbox, human approval.** This is Aider's default and Claude Code's Manual mode. It is safe only to the extent humans actually read the prompts, and Anthropic's 93% approval-rate figure shows they mostly stop reading [18].
2. **OS-level process sandbox.** Seatbelt on macOS; bubblewrap, seccomp and namespaces on Linux; a localhost egress proxy with a domain allowlist. Both Anthropic and OpenAI converged here. Claude Code and Codex use near-identical primitives with different defaults: Codex makes the network off by default, while Claude Code routes it through an allowlisting proxy. The hard parts are (a) protecting paths inside writable roots (`.git`, settings files, shell rc files), (b) Unix sockets and `io_uring` as bypasses, and (c) TLS-blind domain filtering [16][17][28].
3. **Container per task or conversation.** OpenHands, SWE-ReX and the Codex guidance inside Docker. Containers bound the filesystem but need separate egress control.
4. **Ephemeral VM with a credential-brokering proxy.** Claude Code cloud, Devin, Jules, Copilot (Actions runners) and Cursor cloud agents. The strongest pattern keeps **long-lived credentials outside the sandbox entirely** and gives the agent a scoped token that a proxy validates *per operation*. Examples are Anthropic's git proxy, which checks the push target branch, and GitHub's rule that the agent cannot run git commands directly at all [16][44].

### 14.2 Permission policies vs. classifiers

The two approaches compare as follows:

- **Static rules** (allow/deny patterns, Codex `execpolicy`, Copilot's trigger and branch rules) are predictable but coarse.
- **Classifier gating** (Claude Code auto mode) adapts to context. It has a measurable **17% false-negative rate on real over-eager actions**, so it complements the OS sandbox rather than replacing it [18].
- Anthropic's design keeps the classifier *reasoning-blind*: it never sees assistant text or tool outputs. This is a reusable principle: a policy model should not read the content an attacker controls.

### 14.3 Edit formats

Exact-match string replacement, unified diffs and patch DSLs all beat line-number-based edits. Whole-file rewrites are the most robust and the most expensive. The best format depends on the model and should be measured [21][36][37]. Splitting *planning the change* from *formatting the edit* (Aider's architect mode) works around models that reason well but format poorly [38].

### 14.4 Context engineering

There are three schools:

1. **Agentic search** with grep and glob, preferred by the Claude Agent SDK as "more accurate and transparent" [25].
2. **Static structural maps**: Aider's graph-ranked repo map [35].
3. **Priority packing**: Cursor's Priompt [34].

All modern agents add **hierarchical instruction files** (CLAUDE.md, AGENTS.md, AGENT.md), **subagents** for context isolation, and **compaction**. Tool-output caps (25k tokens in Claude Code) and names-only search results (SWE-agent) are the same idea at different layers: never let a single observation flood the window [3][23].

### 14.5 Verification loops, from weak to strong

```text
 weakest                                                               strongest
 "looks done" -> prompted test run -> lint/syntax gate -> repo tests pass
   -> generated repro test (Agentless) -> security scanners (Copilot)
   -> deterministic Stop hook / goal evaluator -> independent reviewer
   -> human PR review with signed, logged provenance
```

The two most important lessons are about who grades the work:

- **Separate the grader from the worker.** Use a fresh-context reviewer or a deterministic hook [15].
- **Do not let the agent edit the oracle.** Anthropic's long-running harness lets coding agents change only the `passes` field of the JSON feature list [24]. Copilot's agent cannot approve its own PRs, and the requester cannot approve them either [44].

---

## 15. Failure modes catalog

| Failure | Where documented | Mitigation |
|---|---|---|
| Context flooding from large file or tool output | SWE-agent [3]; Claude Code [14][23] | Windowed viewers, output caps, subagents, compaction with thrash detection |
| Corrupted or invalid edits | SWE-agent [3]; Aider [39] | Lint gate on edit; exact-match replace; flexible patching |
| "Lazy" placeholder code | Aider [37] | Unified-diff format; high-level hunks |
| Relative-path confusion after `cd` | Anthropic SWE-bench [21][22] | Require absolute paths (poka-yoke) |
| Premature "done" | Claude Code [15]; long-running harness [24] | Runnable checks, Stop hooks, feature lists, one feature per session |
| Over-eager destructive actions | Auto mode [18]; Codex prompt [29] | Classifier gating; "never revert changes you didn't make" |
| Prompt injection via issues and comments | Copilot [44]; Jules [42]; auto mode [18] | Writer-only triggers, hidden-text filtering, injection probes on tool outputs |
| Exfiltration over allowed domains | Claude Code sandbox [17] | Narrow allowlists, TLS-terminating proxy, credential masking |
| Sandbox incompatibility leading to blanket bypass | Codex [27]; Claude Code [17]; Copilot [45] | Container-level isolation; strict mode disables unsandboxed retry |
| Slow environment startup | Devin [31]; Claude Code [20] | Snapshot and cached setup scripts; prebuilt images (codex-universal) |
| Model-specific harness drift | Devin [31] | Per-model evals gating every model swap |

---

## Key takeaways for system designers

1. **The loop is trivial; the harness is the product.** mini-SWE-agent (about 100 lines, above 74% on SWE-bench Verified) and a five-tool Go workshop agent show the core loop's size [5][48]. The differences between products lie in tool ergonomics, context policy, isolation, and verification.
2. **Design the ACI deliberately, then shrink it as models improve.** Bounded outputs, explicit success messages, lint-gated edits and absolute paths matter more than clever command sets [3][21][22]. Re-test scaffolding with each model generation and delete what no longer helps.
3. **Pick an edit format empirically per model.** Exact-match replace, unified diff or patch DSL. Consider splitting reasoning from edit formatting [36][37][38].
4. **Use both filesystem and network isolation, and put credentials outside the box.** Prefer a brokering proxy that validates each operation (branch-scoped pushes) over handing the agent a token [16][44]. Re-protect `.git` and configuration paths inside writable roots [28].
5. **Treat permission prompts as a depleting resource.** Sandboxes cut prompts by 84%. Classifiers can replace most of the rest, but they have measurable false-negative rates, so layer them; never substitute them for isolation [16][18].
6. **Keep policy models away from attacker-controlled text.** Strip tool outputs and agent reasoning from what the safety classifier sees [18].
7. **Make "done" machine-checkable, and make the checker independent.** Use tests, Stop hooks, reproduction tests and security scanners. Do not let the worker grade its own work or edit its own oracle [15][24][43][49].
8. **Persist state outside the context window.** Use git commits, progress files, JSON feature lists and instruction files. Fresh sessions should reconstruct state from artifacts, not from memory [24].
9. **Govern at the repository level for autonomous agents.** Use writer-only triggers, dedicated branches, separation of duties, CI approval gates, and signed and logged commits [44].
10. **Budget for harness maintenance per model.** Devin's experience shows that prompts, tools and guardrails are model-specific and need an eval suite to swap models safely [31].

---

## References

1. SWE-agent repository and citation for "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (NeurIPS 2024). Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan, Press; Princeton/Stanford. 2024-2025. https://github.com/SWE-agent/SWE-agent
2. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (arXiv:2405.15793). Yang et al. 2024. https://arxiv.org/abs/2405.15793
3. SWE-agent documentation: Agent-Computer Interface (ACI). SWE-agent team. 2024. https://github.com/SWE-agent/SWE-agent/blob/main/docs/background/aci.md
4. SWE-agent documentation: Background. SWE-agent team. 2024. https://github.com/SWE-agent/SWE-agent/blob/main/docs/background/index.md
5. mini-SWE-agent repository. SWE-agent team. 2025. https://github.com/SWE-agent/mini-swe-agent
6. SWE-ReX: Sandboxed code execution for AI agents. SWE-agent team. 2025. https://github.com/SWE-agent/SWE-ReX
7. OpenHands repository (Agent Canvas). OpenHands / All Hands AI. 2024-2026. https://github.com/OpenHands/OpenHands
8. OpenHands: An Open Platform for AI Software Developers as Generalist Agents (arXiv:2407.16741). Wang et al. 2024. https://arxiv.org/abs/2407.16741
9. OpenHands Software Agent SDK repository. OpenHands. 2025. https://github.com/OpenHands/software-agent-sdk
10. The OpenHands Software Agent SDK: A Composable and Extensible Foundation for Production Agents (arXiv:2511.03690). Wang, Rosenberg, Michelini, et al. 2025. https://arxiv.org/abs/2511.03690
11. OpenHands SDK Architecture Overview. OpenHands. 2025. https://github.com/OpenHands/docs/blob/main/sdk/arch/overview.mdx
12. CodeAct repository: Executable Code Actions Elicit Better LLM Agents. Xingyao Wang et al. 2024. https://github.com/xingyaoww/code-act
13. Executable Code Actions Elicit Better LLM Agents (arXiv:2402.01030, ICML 2024). Wang et al. 2024. https://arxiv.org/abs/2402.01030
14. How Claude Code works. Anthropic. 2026. https://code.claude.com/docs/en/how-claude-code-works
15. Best practices for Claude Code. Anthropic. 2026. https://code.claude.com/docs/en/best-practices
16. Beyond permission prompts: making Claude Code more secure and autonomous. David Dworken, Oliver Weller-Davies; Anthropic. 2025. https://www.anthropic.com/engineering/claude-code-sandboxing
17. Configure the sandboxed Bash tool (Claude Code docs). Anthropic. 2026. https://code.claude.com/docs/en/sandboxing
18. How we built Claude Code auto mode: a safer way to skip permissions. John Hughes; Anthropic. 2026. https://www.anthropic.com/engineering/claude-code-auto-mode
19. Use Claude Code in the cloud. Anthropic. 2026. https://code.claude.com/docs/en/claude-code-on-the-web
20. Configure cloud environments (Claude Code docs). Anthropic. 2026. https://code.claude.com/docs/en/cloud-environments
21. Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet. Erik Schluntz; Anthropic. 2025. https://www.anthropic.com/research/swe-bench-sonnet
22. Building effective agents. Erik Schluntz, Barry Zhang; Anthropic. 2024. https://www.anthropic.com/engineering/building-effective-agents
23. Writing effective tools for agents, with agents. Ken Aizawa; Anthropic. 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
24. Effective harnesses for long-running agents. Justin Young; Anthropic. 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
25. Building agents with the Claude Agent SDK. Thariq Shihipar; Anthropic. 2025. https://claude.com/blog/building-agents-with-the-claude-agent-sdk
26. Codex CLI repository. OpenAI. 2025-2026. https://github.com/openai/codex
27. Codex CLI README (release rust-v0.20.0): sandboxing, approvals, AGENTS.md, MCP, exec. OpenAI. 2025. https://github.com/openai/codex/blob/rust-v0.20.0/README.md
28. Codex Linux sandbox README. OpenAI. 2026. https://github.com/openai/codex/blob/main/codex-rs/linux-sandbox/README.md
29. Codex GPT-5-Codex system prompt. OpenAI. 2025. https://github.com/openai/codex/blob/main/codex-rs/core/gpt_5_codex_prompt.md
30. codex-universal: reference base image for Codex environments. OpenAI. 2025-2026. https://github.com/openai/codex-universal
31. Cognition (Devin) customer story. Anthropic / Cognition. 2025. https://claude.com/customers/cognition
32. Devin SWE-bench results and methodology repository. Cognition AI. 2024. https://github.com/CognitionAI/devin-swebench-results
33. Cursor cookbook (SDK, cloud agents, hooks). Cursor / Anysphere. 2026. https://github.com/cursor/cookbook
34. Priompt: priority-based prompt design library. Anysphere. 2023-2025. https://github.com/anysphere/priompt
35. Aider documentation: Repository map. Paul Gauthier / Aider. 2023-2025. https://github.com/Aider-AI/aider/blob/main/aider/website/docs/repomap.md
36. Aider documentation: Edit formats. Aider. 2024-2025. https://github.com/Aider-AI/aider/blob/main/aider/website/docs/more/edit-formats.md
37. Unified diffs make GPT-4 Turbo 3X less lazy. Paul Gauthier / Aider. 2023. https://github.com/Aider-AI/aider/blob/main/aider/website/docs/unified-diffs.md
38. Separating code reasoning and editing (Architect/Editor). Paul Gauthier / Aider. 2024. https://github.com/Aider-AI/aider/blob/main/aider/website/_posts/2024-09-26-architect.md
39. Linting code for LLMs with tree-sitter. Paul Gauthier / Aider. 2024. https://github.com/Aider-AI/aider/blob/main/aider/website/_posts/2024-05-22-linting.md
40. How aider scored SOTA 26.3% on SWE Bench Lite. Paul Gauthier / Aider. 2024. https://github.com/Aider-AI/aider/blob/main/aider/website/_posts/2024-05-22-swe-bench-lite.md
41. Aider is SOTA for both SWE Bench and SWE Bench Lite. Paul Gauthier / Aider. 2024. https://github.com/Aider-AI/aider/blob/main/aider/website/_posts/2024-06-02-main-swe-bench.md
42. Jules Actions (jules-action) README. Google Labs. 2025-2026. https://github.com/google-labs-code/jules-action
43. Responsible use of Copilot cloud agent. GitHub. 2026. https://github.com/github/docs/blob/main/content/copilot/responsible-use/agents.md
44. Risks and mitigations for Copilot cloud agent. GitHub. 2026. https://github.com/github/docs/blob/main/content/copilot/concepts/security-governance-and-network-settings/risks-and-mitigations.md
45. Customizing the development environment for Copilot cloud agent. GitHub. 2026. https://github.com/github/docs/blob/main/content/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-environment.md
46. Context Engineering - Amp (guide). Sourcegraph / Amp. 2025. https://github.com/sourcegraph/amp-examples-and-guides/blob/main/guides/context-management/Context%20Engineering%20-%20Amp.md
47. Amp toolbox guide (amp-usage). Sourcegraph / Amp. 2025. https://github.com/sourcegraph/amp-examples-and-guides/blob/main/guides/toolbox/amp-usage.md
48. How to build a coding agent (workshop). Geoffrey Huntley. 2025. https://github.com/ghuntley/how-to-build-a-coding-agent
49. Agentless repository. OpenAutoCoder (Xia, Deng, Dunn, Zhang). 2024. https://github.com/OpenAutoCoder/Agentless
50. Agentless: Demystifying LLM-based Software Engineering Agents (arXiv:2407.01489). Xia, Deng, Dunn, Zhang. 2024. https://arxiv.org/abs/2407.01489
