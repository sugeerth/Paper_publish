# 13. Agent Evaluation: Benchmarks, Trajectory Grading, pass^k Reliability, and Eval Harness Design

Note on sourcing: the web search budget for this run was exhausted, and arxiv.org, openai.com, sierra.ai, tbench.ai, huggingface.co and hal.cs.princeton.edu could not be reached from the research environment. Primary sources were read through anthropic.com (fetched directly) and through the benchmarks' own GitHub repositories, cloned and read in full. These include task files, evaluator source code, READMEs, changelogs and documentation. A small number of headline figures come from the original papers or blog posts (the SWE-bench Verified annotation statistics, GAIA's human baseline, the τ-bench pass^8 figure). They are marked "as reported" and were not re-fetched here.

## Overview

Evaluating an agent differs from evaluating a model in four ways.

1. **The agent changes the world.** Success is a property of the final environment state, not of a string. A flight-booking agent has succeeded only if the reservation exists in the database, whatever it says in the chat [1].
2. **Trajectories are long and stochastic.** Two runs of the same agent on the same task can take different paths and reach different outcomes. One sample per task tells you little about reliability.
3. **The environment is part of the measurement.** The container image, CPU and memory limits, network policy, time budget and even time of day all move scores [2].
4. **The scaffold is part of the system under test.** An eval measures the model and its harness together [1]. On the same 99 tasks and the same model, Harbor parity runs scored 68.7 with Codex CLI and 32.3 with the Terminus-2 scaffold [36].

This chapter covers the main public benchmarks and the design ideas they introduced: τ-bench and its successors, SWE-bench and its descendants, GAIA and Gaia2, WebArena and the computer-use suites, Terminal-Bench, and research-engineering benchmarks. It then turns to grading (outcome versus trajectory, code versus LLM judges, partial credit), to reliability statistics (pass@k, pass^k, error bars), and to how to build an evaluation harness. A recurring theme is that every major benchmark has needed a "verified" or v2 release. Most of the fixes were to tasks and graders, not to the agents.

### Shared vocabulary

Anthropic's January 2026 guide gives a vocabulary that the rest of this chapter uses [1]:

| Term | Meaning |
|---|---|
| Task | One test with defined inputs and success criteria |
| Trial | One attempt at a task; several trials are run because outputs vary |
| Grader | Logic that scores some aspect of a trial; a task may have several |
| Transcript / trajectory | Full record of a trial: messages, tool calls, reasoning, results |
| Outcome | The final environment state (e.g., the DB row), not the agent's claim |
| Evaluation harness | Runs tasks concurrently, records steps, grades, aggregates |
| Agent harness / scaffold | The system that lets a model act as an agent |
| Suite | A collection of tasks measuring a related capability |

Harbor, the Terminal-Bench team's framework, uses almost the same terms. It adds the *job*, a set of trials run in parallel, and the *sandbox*, an isolated runtime [37].

```text
          +---------------------- Evaluation harness -----------------------+
 task     |  +-----------+   +-------------+   +-----------+   +---------+  |
 registry -->| env image |-->|  sandbox    |<->|  agent    |-->|trajectory| |
 (versioned) | + seed DB |   | (container/ |   | scaffold  |   |  store   | |
          |  +-----------+   |  VM, limits)|   | + model   |   +----+----+  |
          |        ^         +------+------+   +-----+-----+        |       |
          |        |                |   user simulator (opt.)       |       |
          |        |                v                               v       |
          |        |         +-------------+                 +-----------+  |
          |        +---------|  verifier   |---- reward ---->| aggregate |  |
          |   (fresh copy)   | (tests, DB  |                 | pass^k,CI,|  |
          |                  |  diff, judge)|                | cost, time|  |
          |                  +-------------+                 +-----------+  |
          +-----------------------------------------------------------------+
```

## 13.1 Reliability metrics: pass@k, pass^k and error bars

### pass@k versus pass^k

*pass@k* is the probability that at least one of k attempts succeeds. It comes from code generation, where an unbiased estimator from n samples with c successes is 1 - C(n-c, k)/C(n, k) [46]. It rises with k. It fits settings where a verifier can pick the one good attempt.

*pass^k* ("pass hat k") was introduced by τ-bench [7]. It is the probability that **all** k independent trials succeed. It falls with k and measures consistency, which is what a customer-facing agent needs. The τ²/τ³ code computes it per task with a combinatorial estimator and averages over tasks [10]:

```python
def pass_hat_k(num_trials, success_count, k):
    return math.comb(success_count, k) / math.comb(num_trials, k)
```

For a per-trial success rate of 75%, pass^3 is about 0.75^3, or 42%. By k = 10, pass@k approaches 100% while pass^k approaches 0% [1]. The τ-bench leaderboard shows the decay on real agents [8]:

| Agent (tool-calling) | Domain | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---|---|---|---|---|
| claude-3-5-sonnet-20241022 | Retail | 0.692 | 0.576 | 0.509 | 0.462 |
| gpt-4o | Retail | 0.604 | 0.491 | 0.430 | 0.383 |
| claude-3-5-sonnet-20241022 | Airline | 0.460 | 0.326 | 0.263 | 0.225 |
| gpt-4o | Airline | 0.420 | 0.273 | 0.220 | 0.200 |
| gpt-4o-mini | Airline | 0.225 | 0.140 | 0.110 | 0.100 |

As reported in the τ-bench paper, gpt-4o's pass^8 on retail was below 25% [7]. An agent that resolves 60% of conversations on a typical try resolves fewer than half of tasks reliably.

Harnesses have built these metrics in. Inspect runs *epochs* per sample and reduces them with `mean`, `mode` or `pass_at_k` reducers. For example, `Epochs(5, ["mean", "pass_at_5"])` reports both [38]. MLE-bench asks for at least three seeds, with "Any Medal (%)" reported as mean ± one standard error. Its README notes that grading is deterministic but agents are "quite high-variance" [43].

### Error bars and comparing agents

Anthropic's statistical guidance, based on Evan Miller's "Adding Error Bars to Evals", gives five recommendations [3][4]:

1. Report the standard error of the mean (SEM) under the central limit theorem; the 95% CI is mean ± 1.96 × SEM.
2. Cluster standard errors when tasks are not independent, for example when several tasks share one repository, website or document. Clustered SEs on popular evals can be **more than three times** the naive ones.
3. Reduce within-task variance by resampling each task several times and averaging (the Inspect `epochs` mechanism).
4. Use paired differences when two models answer the same items. Frontier models' per-question scores correlate at about 0.3-0.7, so pairing narrows the interval.
5. Run a power analysis before the eval to choose the number of tasks and resamples.

Inspect implements clustered standard errors and Wilson intervals for binary scores (`ci_wilson()`) [38]. For agent suites built from a few repositories or sites (SWE-bench, WebArena, τ-bench domains), clustering is the realistic default.

### Infrastructure as a source of variance

Anthropic's February 2026 study treats the environment itself as a source of error [2]. Running Terminal-Bench 2.0 on Google Kubernetes Engine, the team saw scores that did not match the leaderboard. Up to 6% of tasks failed from pod errors. Kubernetes treated each task's resource spec as both a guaranteed floor and a hard kill ceiling, so a brief memory spike killed containers that would have succeeded. Across six resource settings with the same model, harness and tasks:

| Resource setting | Infra error rate | Score effect |
|---|---|---|
| 1x (strict) | 5.8% | baseline |
| 3x ceiling | 2.1% | within noise of 1x (p = 0.40) |
| Uncapped | 0.5% | about +6 points over 1x (p < 0.01) |

Between 1x and 3x, extra headroom mostly fixed crashes on tasks that would have failed anyway. Above 3x, it let agents try memory-heavy approaches such as installing the full Python data-science stack. On SWE-bench (227 problems × 10 samples), raising RAM to 5x lifted scores by only 1.54 points, because those tasks need fewer resources. The authors recommend specifying a guaranteed allocation and a separate kill threshold, reporting the multiplier, and running evals on several days. They advise readers to be skeptical of leaderboard gaps under about 3 points unless the setups match: "A few-point lead might signal a real capability gap—or it might just be a bigger VM" [2]. Harbor now exposes independent CPU and memory *enforcement policies* (`limit`, `request`, `guarantee`, `ignore`) for this reason [37].

## 13.2 τ-bench, τ²-bench and τ³-bench (Sierra)

**Problem.** Customer-service agents must talk to a person, follow a written policy, and call APIs that change state, all in one conversation. Single-turn function-calling benchmarks test none of this together.

**Architecture.** Each domain has four parts [10]:

- a **policy** document (markdown) the agent must follow;
- **agent tools** over a domain database (`FlightDB`, orders, accounts);
- **tasks**, each holding a user scenario (persona, goal, hidden information) and evaluation criteria;
- optionally, **user tools** that let the simulated user act on their own device.

An LLM plays the user. The original τ-bench used gpt-4o by default and offered several user strategies: plain `llm`, `react` (think, then reply), `verify` (a second LLM call checks the user turn and regenerates it if needed) and `reflection` [8].

```text
  +-------------------+  natural language   +------------------------+
  | User simulator LLM|<------------------->|  Agent under test      |
  | persona + goal +  |                     |  policy.md in prompt   |
  | hidden facts      |                     +-----------+------------+
  +---------+---------+                                 | tool calls
            | user tools (τ² telecom only)              v
            v                                 +------------------------+
  +-------------------+                       |  Domain DB (agent side)|
  | Device state      |<---- shared world --->|  orders, flights, plans|
  | (SIM, APN, data)  |                       +-----------+------------+
  +-------------------+                                 |
                                                        v   end of episode
   gold env  = fresh DB + replay(reference actions) --> hash(DB_gold)
   reward    = [hash(DB_pred) == hash(DB_gold)] x [all communicate_info said]
```

**Grading on outcomes.** The final reward is the *product* of the components listed in `reward_basis`. For airline, retail and telecom, the default is `["DB", "COMMUNICATE"]` [11]. The task's `actions` list is one reference trajectory. It is replayed on a fresh environment to produce the target database, and the grader compares the hash of that database with the hash of the agent's. Any tool sequence that reaches an equivalent end state passes, and the agent's path is never compared to the reference. An `ACTION` reward type does enforce the exact path. It is used only in about 9 of roughly 100 `banking_knowledge` tasks, where the retrieval path is what is being tested. The maintainers' documentation warns that this "promotes `actions` from 'one reference trajectory' to 'the only acceptable trajectory'" [11]. One consequence is that in airline task 1, where the right behavior is to refuse a cancellation, "an agent that does nothing but politely refuse will receive full reward" [11]. The authors accept this. Benchmark designers should note it, because any agent that never writes to the database passes every refusal task.

**Diagnostics.** The runner still computes a `partial_action_reward` against the reference path (broken down by read and write tools) for debugging. It does not affect the score [11]. τ-bench also ships an LLM-based *auto error identification* tool. It assigns each failure to the user, agent or environment and classifies it as `goal_partially_completed`, `used_wrong_tool`, `used_wrong_tool_argument` or `took_unintended_action`. The maintainers note that these LLM labels can be inaccurate [8].

**τ²-bench: dual control.** τ²-bench (Barres et al., 2025) adds a `telecom` domain in which the user also has to act [9]. The user simulator gets device-side tools such as `check_status_bar`, `toggle_airplane_mode`, `reseat_sim_card`, `set_apn_settings`, `toggle_roaming` and `run_speed_test`. The agent must diagnose the problem and then guide the user through fixing it [10]. Telecom tasks are generated compositionally from issue types (service, mobile data, MMS). The `full` split has 2,285 generated tasks, and the `base` split has 114 [10]. Airline has 50 tasks (30 train / 20 test) and retail 114 (74 / 40) in the current release [10]. A Gymnasium interface exposes the domains as RL environments.

**τ³-bench (2026).** The repository was renamed τ³-bench and added three things [10]:

- a `banking_knowledge` domain with a knowledge base of 700+ documents and configurable retrieval (embeddings, BM25, grep, or sandboxed shell search) [13];
- full-duplex **voice** evaluation through realtime audio APIs [14];
- **75+ task fixes** across airline, retail and banking, based on the SABER analysis (Cuadron et al., 2025). The fixes removed incorrect expected actions, clarified ambiguous instructions and fixed impossible constraints [10][12].

The July 2026 v1.0.1 release fixed grading bugs in `banking_knowledge`. Re-grading existing trajectories moved scores only upward, for example GPT-5.5 (xhigh) from 37.37 to 46.39. The changelog declares results from versions before 1.0.1 not comparable, and a `tau2 evaluate-trajs --fresh-tasks` command re-grades stored trajectories against fixed tasks [10]. A 9-point swing from grader repairs alone is larger than most gaps between models on the leaderboard.

**Failure modes and lessons.**

- *The user simulator is noisy.* A simulated user can leak hidden facts, give up, or invent requirements. The verify and reflection strategies exist to reduce this [8].
- *Static evals miss creative solutions.* Anthropic reports that Claude Opus 4.5 "failed" a τ²-bench airline task as written by finding a policy loophole that served the user better [1].
- *Store trajectories so they can be re-graded.* Task fixes are routine, so trajectories must be stored and re-scorable without re-running the agent.

## 13.3 SWE-bench and its descendants

### SWE-bench (Princeton, ICLR 2024)

**Problem and task format.** Given a repository at a base commit and a GitHub issue, the agent must produce a patch that resolves the issue [15]. Each instance records `base_commit`, `problem_statement`, the gold `patch`, a hidden `test_patch`, and two test lists [16]:

- `FAIL_TO_PASS`: tests that must flip from failing to passing;
- `PASS_TO_PASS`: tests that must keep passing.

An instance counts as resolved only if both hold. The full set has 2,294 instances, and Lite has 534 [16].

**Harness architecture.** Since June 2024 the harness has been fully containerized, a change made with OpenAI Preparedness [16]. It uses three image layers:

```text
  +--------------------------+  per-instance: repo @ base_commit, deps installed
  |  Instance images         |
  +--------------------------+  per-repo/version: interpreter + package set
  |  Environment images      |
  +--------------------------+  per-language: OS, toolchains
  |  Base images             |
  +--------------------------+
  run: apply model patch -> apply hidden test_patch -> run tests -> parse log
       -> FAIL_TO_PASS all pass AND PASS_TO_PASS all pass  => resolved
```

The docs recommend an x86_64 machine with at least 120 GB of free disk, 16 GB of RAM and 8 cores, and no more than `min(0.75 × cpu_count, 24)` workers. The `--cache_level` setting trades disk space for rebuild time. One pitfall is that results are cached by `run_id` and `instance_id` only. Re-running with a different patch under the same `run_id` silently reuses the old verdict [16]. `swebench report` re-grades saved logs without starting containers. Runs can be offloaded to Modal or to `sb-cli` on AWS [16].

### SWE-bench Verified (OpenAI Preparedness, August 2024)

Some original instances could not be solved fairly: the issue was underspecified, or the hidden tests rejected valid fixes. As reported by OpenAI, professional developers screened 1,699 random instances and filtered out about 68%. The result was a 500-instance subset that engineers confirmed solvable, labeled with difficulty buckets [17][16]. (The full annotation statistics are in OpenAI's post and were not re-fetched here.)

Verified became the standard coding-agent metric. Anthropic's January 2025 write-up shows how much a minimal scaffold could do. It gave Claude only a Bash tool and a `str_replace_editor`. The edit applies only when `old_str` matches exactly once, and paths must be absolute to make the tool "error-proof". With this setup Claude 3.5 Sonnet (new) scored 49%, against a previous best of 45%, 33% for the older Sonnet and 22% for Claude 3 Opus [5]. Successful runs often took hundreds of turns and more than 100k tokens. Some failures came from the harness itself: environment setup problems, and install patches applied twice. The model also could not see the hidden tests, so it sometimes believed it had succeeded when it had not [5].

Within about a year, Verified scores went from roughly 40% to over 80%. Anthropic describes the benchmark as nearing saturation [1]. mini-swe-agent, a loop of about 100 lines using only bash, scores above 74% and anchors a "bash only" leaderboard that holds the scaffold fixed to isolate the model [20]. Harbor's parity experiments show that the eval transfers across harnesses when the setup is matched. o4-mini with Codex scored 53.11 in both the original and the adapted harness, and gpt-5-mini with mini-swe-agent scored 56.3 on the leaderboard and 54.5 ± 0.7 over three Daytona runs [36].

### Variants

| Variant | Size | Purpose |
|---|---|---|
| SWE-bench (full) | 2,294 | Original, 12 Python repos (as reported) |
| Lite | 534 | Cheaper iteration |
| Verified | 500 | Human-screened solvable subset |
| Multimodal | 480 test tasks open-sourced (v2, Sep 2026) | Visual/front-end issues |
| Multilingual | 300 (9 languages, 42 repos) | Cross-language generalization |

Sources: [16][18].

### SWE-bench Pro (Scale AI) and the "locked protocol"

SWE-bench Pro targets long-horizon tasks. Version 1 had 731 public tasks. V2, released in September 2026, has 642 validated tasks across 11 repositories, shipped as Harbor task directories with public container images [19]. Its release notes amount to a checklist for hardening an agent benchmark:

- **The agent phase runs offline.** `network_mode = "no-network"` is set in every task. Only the model endpoint is whitelisted, and web tools are disabled.
- **The verifier never runs in the agent's sandbox.** The agent's `git diff` is captured and replayed onto a pristine image by a `PatchReplayAgent`, which then runs the unchanged verifier. Maintainers ask for both numbers to be published, with the re-graded one treated as authoritative.
- **Each task has a 50-minute budget,** enforced with an in-sandbox `timeout` so that a partial patch is still graded.
- **Probes check the sandbox.** Probe tasks print what the sandbox can reach or see, such as the network or git history.
- **Git history is sanitized.** The fixing commit, stray refs, stashes and hooks are removed. Anthropic separately found that agents can gain "an unfair edge by inspecting git history from previous trials" [1].
- **A release gate is enforced.** The reference patch resolves 642/642 tasks, an empty patch resolves 0/642, and SHA-256 checksums cover every file.
- **Tasks were repaired between versions.** From v1 to v2, 89 tasks were dropped and 529 problem statements were rewritten so that every graded assertion traces back to the text. Each rewrite was implemented blind by a second engineer. The team revised 214 test patches and 38 gold patches. Verifiers now revert agent edits to fixtures and snapshots and purge stale bytecode.

A **HARD-51** subset (tasks failed by at least two of five model families) separates frontier models better than the full set does [19].

```text
 phase 1 (agent, offline)           phase 2 (grading, pristine)
 +------------------------+         +-----------------------------+
 | image @ base commit    |  diff   | same image, fresh container |
 | sanitized git history  |-------->| apply model.patch           |
 | net: model API only    |         | restore hidden tests        |
 | 50 min timeout         |         | revert fixture edits        |
 +------------------------+         | run verifier -> reward      |
                                    +-----------------------------+
```

### Related coding and engineering benchmarks

- **SWE-Lancer** (OpenAI) scores freelance software tasks by their payout. The public repository now holds 198 of the original 237 tasks, the subset adjusted to run offline [22].
- **SWE-smith and SWE-Gym** turn repositories into SWE-bench-style training environments. Harnesses now often serve as RL rollout generators as well as evaluators, and Harbor lists rollout generation as a primary use [21][36].

## 13.4 GAIA and Gaia2

**GAIA (Meta, Hugging Face and others, 2023)** poses real-world assistant questions in three difficulty levels. The questions need browsing, file parsing (xls, mp3, pdf), multimodality and multi-step reasoning, and have short, unambiguous answers [23]. As reported in the paper, humans scored about 92% while GPT-4 with plugins scored about 15%. Inspect's port uses the 165-question validation set (53 Level 1, 86 Level 2, 26 Level 3). Test-set answers are held privately by the leaderboard [27].

**Grading** is a deterministic quasi-exact match copied from the official scorer [27]. If the ground truth is a number, `$`, `%` and `,` are stripped from the answer and the values are compared as floats. Lists are split on `,` or `;` and compared element by element. Strings are normalized before comparison. This makes GAIA cheap and reproducible. It also means a correct answer in the wrong format fails, which is why scaffolds add strict final-answer formatting.

**Progress and saturation.** Hugging Face reported GPT-4-Turbo below 7% on the leaderboard in mid-2024, and an AutoGen multi-agent system on top at 40% [25]. Their code agent then reached 44.2% on validation and 33.3% on test. It used a manager agent with sub-agents to keep noisy web context out of the planner [25]. In early 2025 OpenAI Deep Research reported 67.36% on validation, and Hugging Face's open reproduction reached 55.15% [26]. By September 2025, Hugging Face described GAIA's easier levels as "too easy" [24].

**Gaia2 and ARE (Meta, September 2025)** changed the design. GAIA was read-only; Gaia2 is read-write and runs in Meta's Agents Research Environments (ARE). ARE simulates a smartphone with email, calendar, contacts, shopping, a file system and messaging, populated with a persona's history [24][47]. About 1,000 new human-written scenarios cover execution, search, ambiguity handling, adaptability to events that occur during the episode, temporal reasoning (for example, ordering a cab after a 3-minute delay), agent-to-agent collaboration, and tolerance of injected API failures. Reference evaluation used a uniform ReAct loop at temperature 0.5 with a 16K-token generation limit and all 101 tools in the system prompt. Grading combined exact match with a Llama 3.3 70B judge [24]. Execution and search were close to solved. Ambiguity, adaptability, noise and especially the time split remained hard. The authors also reported scores normalized by cost (LLM calls and output tokens) as a Pareto frontier, on the grounds that a correct answer reached after hours of computation is "not as good" as a fast one [24]. When ported to Harbor, GAIA reproduced within noise: 51.31 ± 0.73 originally against 50.71 ± 0.53 in the adapter, over 165 tasks and 3 runs [36].

## 13.5 Web and computer-use environments

### WebArena (CMU, 2023) and WebArena-Verified (ServiceNow, 2025-26)

**Architecture.** WebArena self-hosts working websites in Docker: a shopping site, a shopping-admin CMS, a Reddit clone, GitLab, OpenStreetMap and a Wikipedia mirror. Agents act on them through a Gym-style Playwright environment, observing the accessibility tree by default [28]. The benchmark has 812 tasks. Annotators achieved 78.24% [28]. As reported, the paper's best GPT-4 agent reached about 14%. The environment has to be **reset after every full run** because tasks mutate site state [28].

**Grader types** from `evaluators.py` [28]:

- `StringEvaluator`: `exact_match`, `must_include`, and `fuzzy_match`. The last calls an LLM (gpt-4-1106-preview) to judge semantic equivalence. For unachievable tasks the expected answer is "N/A", checked by exact match to prevent false positives.
- `URLEvaluator`: checks that the final URL and its query parameters match.
- `HTMLContentEvaluator` (`program_html`): navigates to a page after the episode and checks its content with locators, which verifies side effects such as a posted comment.

**WebArena-Verified** audits every task, reference answer and evaluator by hand. It removes LLM-as-judge and substring matching in favor of "type-aware normalization and structural comparison" [29]. It also adds **offline evaluation by replaying network traces**, so runs can be graded without live sites, and a 258-task **Hard subset** for cheaper runs. Optimized Docker images are up to 92% smaller [29]. The direction is the same as SWE-bench Pro's: make the grader deterministic and separable from the live environment.

**BrowserGym and AgentLab** (ServiceNow) put MiniWoB, WebArena, WebArena-Verified, VisualWebArena, WorkArena, AssistantBench, WebLINX and others behind one interface. New benchmarks implement `AbstractBrowserTask` [30].

### OSWorld

OSWorld runs agents in full Ubuntu, Windows or macOS virtual machines and checks results with per-task scripts that inspect files, application configuration and system state. OSWorld-Verified (July 2025) fixed community-reported task problems and added AWS parallelism that cuts a full run to under an hour. For the verified leaderboard, maintainers run the submitter's agent code themselves [31]. Chapter 7 covers OSWorld's evolution in detail.

### TheAgentCompany (CMU, December 2024)

TheAgentCompany simulates a small software company. Self-hosted GitLab, Plane (project management), ownCloud and RocketChat run with pre-loaded data, and LLM-driven "NPC" coworkers can be messaged [32]. The tasks were drawn from occupational categories in O*NET, and there are 175 of them. Its main grading idea is **checkpoints**, intermediate milestones checked by code or by an LLM, with partial credit [32]:

```text
score = 0.5 * (checkpoints_passed / checkpoints_total)
      + 0.5 * floor(checkpoints_passed / checkpoints_total)   # full-completion bonus
```

Evaluator code ships encrypted inside each task image. Benchmark users may inspect rubrics, but "examinees (e.g. agents) are not allowed to read checkpoint rubrics or evaluation code." The trajectory file is optional input that the graders use to award partial credit [32].

## 13.6 Terminal-Bench and Harbor

**Terminal-Bench** (Laude Institute and Stanford, ICLR 2026) tests agents on end-to-end work in a sandboxed terminal: compiling software, training models, configuring servers and fixing security problems [33][35]. Every task has three parts: an English instruction, a test script that checks the final container state, and an oracle solution that proves the task is solvable [33]. The beta, Terminal-Bench-Core v0.1.1, had about 100 tasks. Terminal-Bench 2.0 has **89 tasks**. Each one received "several hours of human and LM-assisted validation" to make sure it is solvable, realistic and well specified [34].

Reading the 89 `task.toml` files directly shows how TB 2.0 is configured [34]:

| Property | Distribution in TB 2.0 |
|---|---|
| Difficulty | 4 easy, 55 medium, 30 hard |
| Top categories | software-engineering 26, system-administration 9, security 8, scientific-computing 8, data-science 8 |
| Human time estimates | expert and junior minutes per task; mean expert estimate about 207 min, max 2,400 |
| Agent/verifier timeouts | most commonly 900 s; others up to 3,600 s |
| Resources | 84 tasks use 1 CPU; 69 tasks use 2,048 MB RAM |
| Network | `allow_internet = true` on all 89 |

The verifier is typically a pytest file run by `tests/test.sh`. The script writes `1` or `0` to `/logs/verifier/reward.txt` [34]. These small resource caps explain why Anthropic's infrastructure-noise study used TB 2.0. With 2 GB of memory, enforcing the limit as a hard ceiling rather than a floor changes which strategies an agent can try [2].

**Harbor** rewrites the Terminal-Bench harness "from the ground up for improved reliability, observability, scalability, and performance" [34][36]. Its design:

- **The task is a portable package**: `instruction.md`, `task.toml`, `environment/` (a Dockerfile, docker-compose or Apptainer definition), `solution/solve.sh` and `tests/test.sh`. Tasks do not depend on the framework [37].
- **Rewards are files**: `reward.txt`, or `reward.json` for multi-dimensional rewards.
- **Multi-step tasks** have per-step instructions, tests and solutions, for long-horizon or continual-learning setups [37].
- **Sandboxes are pluggable**: local Docker, Daytona, Modal and others, running thousands of trials in parallel. A separate verifier sandbox and per-phase network policies are supported [36][37].
- **Agents are installed as-is**: Claude Code, Codex CLI, OpenHands and others are installed into the sandbox, so evals measure the product agent rather than a reimplementation [36].
- **ATIF** (Agent Trajectory Interchange Format, v1.7) gives every agent one JSON trajectory schema that the results viewer, SFT pipelines and trajectory replay can all read [37].
- **Rewardkit** turns verifiers into directories of reusable criteria: files, commands, JSON, HTTP, trajectories, and LLM or agent judges. Criteria can run in isolated filesystem snapshots. It was built after the team found that verifiers copied between benchmarks caused "subtle errors in verifiers during evals" [37].
- **Adapters with parity experiments**: each ported benchmark (SWE-bench, GAIA, OSWorld-Verified, Aider Polyglot and dozens more) must show score parity with the original harness before it is merged. The results are kept in a public CSV [36].

```text
 harbor run -d terminal-bench@2.0 -a claude-code -m <model> --env daytona -n 100
     |
     +--> Job ──> Trials (task x agent x model x attempt), run in parallel
              each: build env -> install agent -> run instruction (timeout)
                    -> collect ATIF trajectory + artifacts
                    -> run tests/test.sh (same or separate sandbox)
                    -> /logs/verifier/reward.{txt,json}
     +--> aggregate metrics, viewer, regrade, export rollouts for RL/SFT
```

## 13.7 Research and engineering benchmarks with rubric and partial credit

- **MLE-bench (OpenAI)** uses 75 Kaggle competitions with locally reconstructed test splits and per-competition grading scripts. The headline metric is the share of competitions where the agent earns any medal. The Low-complexity "Lite" split needs 158 GB of data against 3.3 TB for the full set. Top 24-hour agents moved from 17.1% (AIDE with o1-preview, 2024) to about 64% (early 2026) [43]. In April 2026 the leaderboard stopped accepting submissions "while we develop an improved process for ensuring submissions are fair and comparable." It now separates entries that used test-set feedback [43].
- **PaperBench (OpenAI)** asks agents to replicate 20 ICML 2024 papers. Grading uses three containers: rollout, reproduction (the agent's code is re-executed on a fresh GPU machine), and judging against a per-paper hierarchical rubric. A companion **JudgeEval** measures the LLM judge against human grading. The Code-Dev variant grades only code-development requirements and cut judge costs by about 85% [44]. The top reported score in April 2025 was 26.0 ± 0.3% [44].
- **METR time horizons** fit a logistic curve of agent success against log2 of human completion time. The "time horizon" is the human task length at which the agent succeeds 50% of the time. METR finds that horizons have doubled roughly every 7 months [45].
- **AppWorld** has 9 simulated apps, 457 APIs and about 100 simulated people. Tasks are graded by evaluation programs over the database state. Test-set evaluation code and data are released only as encrypted bundles to reduce training contamination [41].
- **AgentBoard** reports a fine-grained *progress rate* alongside binary success, for partially observable multi-round tasks [42].
- **HAL** (Princeton) made cost a first-class metric. Every result reports accuracy and dollar cost on a Pareto frontier, following "AI Agents That Matter" [39][40]. The harness ran SWE-bench Verified, USACO, AppWorld, CORE-Bench and τ-bench through one CLI. It encrypted uploaded traces to prevent contamination. The project has since paused leaderboard updates to focus on a reliability dashboard [39].

## 13.8 Trajectory grading: graders, judges and transcript analysis

### Grader taxonomy

Anthropic's comparison of grader types [1]:

| Grader | Examples | Strengths | Weaknesses |
|---|---|---|---|
| Code-based | unit tests, DB-state hash, URL/HTML checks, tool-call assertions, transcript metrics | fast, cheap, objective, reproducible | brittle to valid variation; poor for subjective quality |
| Model-based | rubric scoring, NL assertions, pairwise, reference-based, multi-judge | flexible, handles open-ended output | non-deterministic, costly, must be calibrated |
| Human | SME review, spot checks, A/B tests | gold standard; calibrates judges | slow, expensive |

Tasks can combine graders by weighted sum, by requiring all to pass (τ-bench's product of rewards), or in hybrids [1][11].

### Outcome versus path

The benchmarks surveyed here agree on one point: **grade the outcome, not the path**, unless the path is what you are testing. τ-bench hashes the database rather than matching tool calls [11]. SWE-bench runs tests rather than diffing against the gold patch [16]. WebArena-Verified compares structured results [29]. Anthropic's advice is similar: avoid rigid step checks, because agents find valid alternative paths [1]. The same guidance names the cost of outcome-only grading. The agent never sees the hidden tests and may "believe it succeeded" when it has not [5]. A refusal can also pass a DB-only check without any work being done [11].

Path checks still have uses. Transcript-level assertions can cover turn limits, forbidden tools, whether personal data was confirmed before a write, and whether Claude for Chrome chose DOM or screenshots appropriately [1]. τ-bench's read/write split of partial action reward can catch a database that passes because nothing was written [11].

### Partial credit and rubrics

Long tasks rarely succeed or fail outright. TheAgentCompany's checkpoints [32], PaperBench's rubric trees [44], Rewardkit's multi-criterion directories [37] and AgentBoard's progress rate [42] all give partial credit. That provides a training and debugging signal well before binary success becomes common. The risk is that partial credit rewards plausible-looking work. TheAgentCompany therefore keeps half of the score as a full-completion bonus [32].

### LLM-as-judge discipline

LLM judges are used where code cannot check the result: GAIA-style answers outside quasi-exact match, WebArena's `fuzzy_match`, research reports, and tone in conversations. Recommended practice [1][6][44]:

- Calibrate against human labels, and keep a JudgeEval-style meta-benchmark for the judge itself.
- Let the judge answer "Unknown" rather than forcing a verdict [1].
- Avoid verifiers that are too strict and reject correct answers over formatting or punctuation [6].
- Prefer replacing a judge with deterministic structure where possible. WebArena-Verified removed LLM judging entirely [29].
- Record the judge model and version as part of the eval configuration. Gaia2 fixes Llama 3.3 70B as its judge [24].

### Reading transcripts and scanning them automatically

Anthropic lists "read transcripts" as a required step. It is the only way to confirm that failures are fair [1]. Examples of what transcripts revealed:

- **CORE-Bench grading bugs.** Rigid numeric matching rejected a truncated but correct value, specs were ambiguous, and some stochastic tasks could not be reproduced. After fixes and a less constrained scaffold, Opus 4.5 went from 42% to 95% [1].
- **METR task misconfiguration.** Some tasks required *exceeding* a threshold while their instructions asked agents only to *reach* it, which penalized agents that followed instructions [1].
- **Tool description bias.** Claude was adding "2025" to search queries until the tool description was fixed [6].

Inspect now formalizes this as **scanners** (Inspect Scout). Scanners run online or offline over transcripts to flag refusals, evaluation awareness, environment misconfiguration, runtime errors and reward hacking. Their sparse findings are stored separately from scores [38].

## 13.9 Designing an agent eval harness

Combining the systems above gives a reference architecture and a set of defaults.

```text
+--------------------------- Task layer -----------------------------------+
| versioned task packages (instruction, env spec, oracle, verifier)        |
| checksums; release gate: oracle = 100%, no-op = 0%; splits; canaries     |
+---------------------------------+----------------------------------------+
                                  v
+--------------------------- Execution layer ------------------------------+
| sandbox provider (Docker / VM / cloud) | resource policy (floor+ceiling) |
| network policy per phase (agent offline, verifier online if needed)      |
| agent adapter: installed product agent or scaffold; user simulator       |
| limits: wall-clock, working time, tokens, cost, turns                    |
+---------------------------------+----------------------------------------+
                                  v
+--------------------------- Evidence layer -------------------------------+
| trajectory (ATIF-like JSON), artifacts, final env snapshot / patch       |
| infra-error classification separate from task failure                    |
+---------------------------------+----------------------------------------+
                                  v
+--------------------------- Grading layer --------------------------------+
| verifier in pristine sandbox (replay patch / DB hash / trace replay)     |
| code graders > rubric judges (calibrated) > human spot checks            |
| scanners over transcripts (hacking, eval awareness, misconfig)           |
| re-grade stored trajectories when tasks/graders change                   |
+---------------------------------+----------------------------------------+
                                  v
+--------------------------- Reporting layer ------------------------------+
| pass@1, pass^k (k trials), clustered SE / Wilson CI, paired deltas       |
| cost, tokens, latency, tool errors; config manifest (resources, version) |
+--------------------------------------------------------------------------+
```

**Task layer.**

- Every task needs an oracle solution, and the release gate is "oracle passes, empty agent fails". TB 2.0 runs `--agent oracle` [34], and SWE-bench Pro also runs a `nop` agent [19].
- Anthropic notes that a 0% pass rate across many trials usually means the task is broken, not that the agent is weak [1]. A good task is one on which two domain experts would reach the same verdict.
- Start with 20-50 tasks drawn from real failures, bug trackers and support tickets [1].
- Build balanced sets that test both when a behavior should occur and when it should not. Claude.ai's web-search evals cover both over-triggering and under-triggering [1].

**Execution layer.**

- Give every trial a clean, isolated environment. Shared caches, leftover files or shared CPU and memory make trials correlated [1].
- Treat resources as an experimental variable with a separate floor and ceiling [2].
- Cap each command, not only each episode. Inspect's example combines a 15-minute sample `time_limit` with a 3-minute timeout on each `bash()` call. Samples that hit a limit are still scored [38].
- Run the agent as it runs in production. Harbor installs real CLIs, and Anthropic stresses that the eval agent should behave "roughly like the production agent" [1][36].

**Grading layer.**

- Run the verifier in a sandbox the agent never touched (patch replay [19], network-trace replay [29], gold-DB replay [11]).
- Keep rubrics and evaluator code out of the agent's reach, for example by encrypting them as TheAgentCompany does [32].
- Make every result re-gradable, as with `swebench report`, `tau2 evaluate-trajs --fresh-tasks` and Harbor regrade [10][16][37].

**Reporting layer.**

- Report pass^k alongside pass@1 for anything customer-facing.
- Report cost and latency, as HAL and Gaia2 do [24][39].
- Publish the configuration manifest: harness version, resource multiplier, timeouts, scaffold, judge model.
- Treat sub-3-point differences as noise unless the setups are matched [2].

**Lifecycle.**

- Capability evals should start with low pass rates. Once they pass reliably, move them into regression suites that should stay near 100% and run continuously [1].
- Watch for saturation. Qodo built a new agentic eval when one-shot evals stopped showing Opus 4.5's gains [1].
- Hold out a test split when evals drive tool or prompt optimization [6].
- Layer automated evals with production monitoring, A/B tests and human review. Anthropic's "Swiss cheese" model treats each method as one layer that catches failures the others miss [1].

### Comparison of major agent benchmarks

| Benchmark | Domain | Size (current) | Environment | Primary grader | Headline metric | Known pitfalls |
|---|---|---|---|---|---|---|
| τ-bench / τ³ [7][10] | Customer service, dual control, voice, RAG | airline 50, retail 114, telecom 114 base / 2,285 full | Python DB + LLM user sim | DB-hash x required-info substrings | pass^k | user-sim noise; refusal tasks pass with no work; task fixes change scores |
| SWE-bench Verified [16][17] | Python repo issues | 500 | Docker, 3-layer images | FAIL_TO_PASS + PASS_TO_PASS tests | % resolved | hidden tests; saturation; run_id cache |
| SWE-bench Pro v2 [19] | Long-horizon multi-repo SWE | 642 (+HARD-51) | Harbor tasks, offline agent | patch replay on pristine image | % resolved | git-history leaks (mitigated); 50-min budget |
| GAIA [23][27] | Web/file QA | 165 val (+ private test) | open web + tools | quasi-exact match | accuracy by level | format brittleness; saturating |
| Gaia2 / ARE [24] | Read-write phone assistant | ~1,000 scenarios | simulated apps, timed events | exact match + LLM judge | success by capability; cost-normalized | judge dependence |
| WebArena(-Verified) [28][29] | Self-hosted websites | 812 (+258 hard) | Docker sites, Playwright | URL/HTML/string checks; Verified: deterministic + trace replay | success rate | state resets; LLM fuzzy match (original) |
| OSWorld-Verified [31] | Desktop OS apps | 369-task lineage | VMs (VMware/Docker/AWS) | per-task state scripts | success rate | setup-dependent tasks (e.g., Google accounts) |
| TheAgentCompany [32] | Simulated company work | 175 | GitLab, Plane, ownCloud, RocketChat, NPCs | checkpoints (code/LLM) | partial + full completion | LLM-graded checkpoints |
| Terminal-Bench 2.0 [34] | Terminal tasks | 89 | Docker via Harbor | pytest -> reward.txt | % solved | resource enforcement noise |
| MLE-bench [43] | Kaggle ML engineering | 75 | GPU containers, 24 h | competition metric vs medal thresholds | any-medal % | test-set feedback; seed variance |
| PaperBench [44] | Paper replication | 20 papers | rollout, reproduce, judge containers | LLM judge on rubric tree | replication score % | judge accuracy (JudgeEval) |

## Key takeaways for system designers

1. **Measure reliability, not just capability.** Report pass^k from k ≥ 3 trials per task for any user-facing agent. On τ-bench, pass^4 is often half of pass^1 or less [8].
2. **Grade end states with code, and keep the grader away from the agent.** Hash databases, replay patches on pristine images, and replay recorded traces. Use LLM judges only where structure cannot capture success, and calibrate them [11][19][29].
3. **Expect your benchmark to be wrong, and make re-grading cheap.** SWE-bench, WebArena, OSWorld, τ-bench and SWE-bench Pro all needed verified or v2 releases. A grader fix moved one τ³ score by 9 points [10]. Store trajectories in a standard format, and version tasks with checksums.
4. **Use oracle and no-op agents as gates.** Every task should pass with the reference solution and fail with an empty one. Tasks that fail every trial are usually broken [1][19].
5. **The environment is part of the measurement.** Document resource floors and ceilings, network policy and timeouts. Separate infrastructure errors from task failures, and do not trust gaps under about 3 points unless the setups match [2].
6. **Evaluate the agent you ship.** Scaffold choice can halve or double a score on the same model and tasks [36]. Run the production harness, or hold the scaffold fixed when comparing models [20].
7. **Read transcripts, then automate the reading.** Most grader bugs (CORE-Bench, METR thresholds, git-history leaks) were found by people reading trajectories. Scanners for reward hacking, refusals and misconfiguration scale this up [1][38].
8. **Report cost and latency alongside accuracy.** A small accuracy gain at 10x the cost is rarely a win [24][39][40].
9. **Plan the eval lifecycle.** Capability suites should start hard, graduate into near-100% regression suites, and be retired or replaced once saturated [1].
10. **Guard against contamination at every layer.** Keep test answers private (GAIA), encrypt evaluators and code (TheAgentCompany, AppWorld), encrypt shared traces (HAL), keep the agent phase offline (SWE-bench Pro), and keep held-out splits for any optimization loop [6][19][27][32][39][41].

## References

1. "Demystifying evals for AI agents." M. Grace, J. Hadfield, R. Olivares, J. De Jonghe; Anthropic. 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
2. "Quantifying infrastructure noise in agentic coding evals." G. Segato; Anthropic. 2026. https://www.anthropic.com/engineering/infrastructure-noise
3. "A statistical approach to model evaluations." Anthropic. 2024. https://www.anthropic.com/research/statistical-approach-to-model-evals
4. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." E. Miller; Anthropic. 2024. https://arxiv.org/abs/2411.00640
5. "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet." E. Schluntz; Anthropic. 2025. https://www.anthropic.com/research/swe-bench-sonnet
6. "Writing effective tools for agents — with agents." K. Aizawa; Anthropic. 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
7. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." S. Yao, N. Shinn, P. Razavi, K. Narasimhan; Sierra. 2024. https://arxiv.org/abs/2406.12045
8. τ-bench repository (leaderboard, user strategies, auto error identification). Sierra Research. 2024. https://github.com/sierra-research/tau-bench
9. "τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment." V. Barres, H. Dong, S. Ray, X. Si, K. Narasimhan; Sierra. 2025. https://arxiv.org/abs/2506.07982
10. τ²/τ³-bench repository (domains, metrics code, CHANGELOG). Sierra Research. 2025-2026. https://github.com/sierra-research/tau2-bench
11. "Task Schema and Evaluation" (docs/evaluation.md). Sierra Research. 2026. https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md
12. SABER task-quality analysis used for τ³ task fixes. A. Cuadron et al. 2025. https://arxiv.org/abs/2512.07850
13. "τ-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge." Q. Shi, A. Zytek, P. Razavi, K. Narasimhan, V. Barres. 2026. https://arxiv.org/abs/2603.04370
14. "τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains." S. Ray, K. Dhandhania, V. Barres, K. Narasimhan. 2026. https://arxiv.org/abs/2603.13686
15. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" C. Jimenez, J. Yang et al.; Princeton. ICLR 2024. https://arxiv.org/abs/2310.06770
16. SWE-bench repository and documentation (harness, datasets, FAQ). SWE-bench team. 2024-2026. https://github.com/SWE-bench/SWE-bench
17. "Introducing SWE-bench Verified." OpenAI Preparedness. 2024. https://openai.com/index/introducing-swe-bench-verified/
18. "SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?" J. Yang et al. ICLR 2025. https://arxiv.org/abs/2410.03859
19. SWE-bench Pro repository, including V2 README (locked protocol, HARD-51). Scale AI. 2025-2026. https://github.com/scaleapi/SWE-bench_Pro-os
20. mini-swe-agent repository. SWE-agent team. 2025-2026. https://github.com/SWE-agent/mini-swe-agent
21. SWE-smith repository ("SWE-smith: Scaling Data for Software Engineering Agents"). SWE-bench team. 2025. https://github.com/SWE-bench/SWE-smith
22. SWE-Lancer (preparedness repository, project/swelancer). OpenAI. 2025. https://github.com/openai/preparedness
23. "GAIA: a benchmark for General AI Assistants." G. Mialon et al. 2023. https://arxiv.org/abs/2311.12983
24. "Gaia2 and ARE: Empowering the community to study agents." C. Fourrier et al.; Hugging Face and Meta. 2025. https://huggingface.co/blog/gaia2
25. "Our Transformers Code Agent beats the GAIA benchmark." A. Roucher, S. Petrov; Hugging Face. 2024. https://huggingface.co/blog/beating-gaia
26. "Open-source DeepResearch – Freeing our search agents." Hugging Face. 2025. https://huggingface.co/blog/open-deep-research
27. Inspect Evals: GAIA implementation and scorer. UK AI Security Institute. 2024-2026. https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/gaia
28. WebArena repository and paper ("WebArena: A Realistic Web Environment for Building Autonomous Agents"). S. Zhou et al.; CMU. 2023. https://github.com/web-arena-x/webarena
29. WebArena-Verified repository. ServiceNow. 2025-2026. https://github.com/ServiceNow/webarena-verified
30. BrowserGym repository. ServiceNow. 2024-2026. https://github.com/ServiceNow/BrowserGym
31. OSWorld repository (OSWorld-Verified). XLANG Lab, HKU. 2024-2026. https://github.com/xlang-ai/OSWorld
32. "TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks." F. Xu et al.; CMU. 2024. https://github.com/TheAgentCompany/TheAgentCompany
33. Terminal-Bench repository. Laude Institute. 2025. https://github.com/laude-institute/terminal-bench
34. Terminal-Bench 2.0 task repository. Laude Institute. 2025. https://github.com/laude-institute/terminal-bench-2
35. "Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces." M. Merrill, A. Shaw, N. Carlini et al. ICLR 2026. https://openreview.net/forum?id=a7Qa4CcHak
36. Harbor repository (adapters, parity_summary.csv). Harbor Framework Team. 2025-2026. https://github.com/laude-institute/harbor
37. Harbor documentation (core concepts, tasks, resources, ATIF, Rewardkit). Harbor Framework Team. 2026. https://harborframework.com/docs
38. Inspect AI repository and documentation (epochs, metrics, limits, sandboxing, scanners). UK AI Security Institute. 2024-2026. https://github.com/UKGovernmentBEIS/inspect_ai
39. "HAL: A Holistic Agent Leaderboard for Centralized and Reproducible Agent Evaluation" (harness repository). B. Stroebl, S. Kapoor, A. Narayanan; Princeton. 2026. https://github.com/princeton-pli/hal-harness
40. "AI Agents That Matter." S. Kapoor, B. Stroebl, Z. Siegel, N. Nadgir, A. Narayanan. 2024. https://arxiv.org/abs/2407.01502
41. AppWorld repository. H. Trivedi et al.; Stony Brook NLP. 2024-2026. https://github.com/StonyBrookNLP/appworld
42. AgentBoard repository. HKUST NLP. 2024. https://github.com/hkust-nlp/AgentBoard
43. MLE-bench repository ("MLE-Bench: Evaluating Machine Learning Agents on Machine Learning Engineering"). OpenAI. 2024-2026. https://github.com/openai/mle-bench
44. PaperBench ("PaperBench: Evaluating AI's Ability to Replicate AI Research"). OpenAI. 2025. https://github.com/openai/preparedness/tree/main/project/paperbench
45. METR time-horizon analysis ("Measuring AI Ability to Complete Long Tasks"). METR. 2025-2026. https://github.com/METR/eval-analysis-public
46. "Evaluating Large Language Models Trained on Code" (pass@k estimator). M. Chen et al.; OpenAI. 2021. https://arxiv.org/abs/2107.03374
47. Meta Agents Research Environments (ARE) repository. Meta. 2025. https://github.com/facebookresearch/meta-agents-research-environments
