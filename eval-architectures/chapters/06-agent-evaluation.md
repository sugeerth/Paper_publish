# 06. Agent Evaluation Architectures: Trajectories, Outcomes, Sandboxes and Reliability

## Overview

Evaluating an agent is a different engineering problem from evaluating a single model completion. An agent takes many steps. It calls tools, changes the state of an environment, may talk to a user, and can reach a correct result by routes the eval author never thought of. Anthropic's 2026 guide puts it this way: evaluating "an agent" means evaluating the model and its harness together, and the thing that matters is usually the *outcome* (the final state of the environment), not what the agent *says* it did [1]. Its example is a flight-booking agent that replies "booked". Whether the booking happened is answered by checking the database for the reservation, not by reading the reply [1].

The public benchmarks covered here are tau-bench, SWE-bench, Terminal-Bench, WebArena, VisualWebArena, OSWorld, AgentBench and PaperBench. Each one makes a different set of answers to the same five design questions:

1. **Environment.** What world does the agent act in, and how is it isolated, reset and kept deterministic? Options range from a Docker container to a full virtual machine, a self-hosted website, or a simulated customer database.
2. **Interaction protocol.** Who takes turns? The agent may work alone, alongside an LLM-simulated user, or under "dual control", where both the agent and the user can change the world.
3. **Grader.** Is success judged on the final state (outcome), on the sequence of actions (trajectory), on what the agent communicated, or by an LLM judge against a rubric?
4. **Aggregation.** How are repeated, stochastic trials summarised? The main choices are pass@1, pass@k, pass^k and partial credit.
5. **Accounting.** How are cost, latency, tokens, infrastructure noise and contamination recorded, so that scores can be compared across systems?

This chapter takes the benchmarks one at a time, then covers reliability metrics, cost-aware evaluation, benchmark validity and the vendors' published guidance.

A note on sourcing: when this chapter was written, arXiv and several vendor domains could not be reached from the research environment. Design facts therefore come mostly from the benchmarks' own repositories and source code, and from Anthropic engineering posts that could be fetched. Headline numbers that come from paper abstracts and could not be re-checked are marked "approx.".

## 1. Anatomy of an agent evaluation

Anthropic's "Demystifying evals for AI agents" (January 2026) defines a vocabulary that most of the field now uses implicitly [1]:

- **Task**: one test, with defined inputs and success criteria.
- **Trial**: one attempt at a task. Several trials are run because outputs vary.
- **Grader**: logic that scores some aspect of performance. One task may have several graders, each with several assertions.
- **Transcript** (also called trace or trajectory): the full record of a trial, including tool calls, reasoning and intermediate results.
- **Outcome**: the final state of the environment.
- **Evaluation harness**: the infrastructure that runs tasks concurrently, records trials, grades them and aggregates the results.
- **Agent harness** (or scaffold): the system that lets a model act as an agent.

```
+------------------------------- EVALUATION HARNESS -----------------------------+
|                                                                                 |
|  task spec ---------+                                                           |
|  (instruction,      |   +---------------- SANDBOX (per trial, fresh) --------+ |
|   init config,      +-->| env state: DB / repo / VM / websites               | |
|   graders)              |      ^  tool calls       | observations             | |
|                         |      |                   v                          | |
|                         |  +----------------------------+   +--------------+  | |
|                         |  | AGENT HARNESS (scaffold)   |<->| simulated    |  | |
|                         |  | model + tools + loop       |   | user (opt.)  |  | |
|                         |  +----------------------------+   +--------------+  | |
|                         +------|-------------------------------|-------------+ |
|                                v                               v               |
|                     transcript (trajectory)           final env state (outcome)|
|                                |                               |               |
|                                v                               v               |
|           +----------------- GRADERS -----------------------------------+      |
|           | code: state diff, unit tests, URL/HTML checks, tool checks  |      |
|           | model: rubric / NL assertions    human: calibration sample  |      |
|           +--------------------------------+----------------------------+      |
|                                            v                                   |
|                 aggregation: pass@1, pass@k, pass^k, partial credit,           |
|                 cost / tokens / latency / infra-error rate                     |
+---------------------------------------------------------------------------------+
```

The same guide sorts graders into three families and gives the trade-offs of each [1]:

- **Code-based graders**: string match, fail-to-pass tests, static analysis, outcome and tool-call verification. They are fast, cheap and reproducible, but brittle when a valid answer is phrased differently.
- **Model-based graders**: rubrics, natural-language assertions, pairwise comparison. They are flexible but non-deterministic, and they need calibrating against humans.
- **Human graders**: the gold standard, but slow and expensive. They are mainly used to calibrate model graders.

Scores can be combined in three ways. *Weighted* scoring requires the total to clear a threshold. *Binary* scoring requires every grader to pass. *Hybrid* scoring mixes the two [1].

Anthropic also separates **capability evals** from **regression evals** [1]:

- A capability eval should start with a low pass rate, so the team has something to climb.
- A regression eval should sit near 100%, so that any drop signals breakage.
- A capability eval whose pass rate has become high can "graduate" into the regression suite.

## 2. Outcome grading vs trajectory grading

### 2.1 The two philosophies

**Outcome (end-state) grading** inspects the world after the agent stops. Examples are the database hash in tau-bench, the unit tests in SWE-bench, the spreadsheet contents in OSWorld and the page URL or DOM in WebArena. It accepts any route to a correct state. It cannot see harmful or wasteful intermediate steps, and it cannot credit a correct but unanticipated end state that the author did not encode.

**Trajectory grading** inspects the transcript. It asks whether the agent called the right tools with the right arguments, in an acceptable order, within a turn budget, without breaking policy. It can catch policy violations and inefficiency. Anthropic warns, however, that rigid checks on exact tool-call sequences are "too rigid" and produce brittle tests, "because agents often find valid routes designers didn't anticipate" [1]. Its stated principle is to "grade what the agent produced rather than the path it took", and to use partial credit for multi-part tasks [1].

Anthropic's multi-agent research system applies the same rule to stateful agents. It judges whether the final state is correct rather than whether a particular path was followed, and adds intermediate checkpoints only for complex workflows [36].

In practice the field has converged on **outcome-first hybrids**:

- The pass/fail decision rests on the end state.
- Trajectory signals are added as further multiplicative gates, such as "did the agent tell the user X" or "did it stay within N turns".
- Some trajectory signals are kept as *diagnostics* that do not affect the score.

### 2.2 A concrete hybrid: tau2-bench's reward basis

tau2-bench writes this hybrid out explicitly in its task schema [6]. Each task declares a `reward_basis`, a subset of five checks:

| Reward basis | Evaluator | What it checks |
|---|---|---|
| `DB` | EnvironmentEvaluator | Hash of the agent's final DB equals the hash produced by replaying gold actions on a fresh env |
| `ENV_ASSERTION` | EnvironmentEvaluator | Programmatic assertions on the final environment pass |
| `ACTION` | ActionEvaluator | Each expected action has a matching agent tool call |
| `COMMUNICATE` | CommunicateEvaluator | Required info strings appear in the agent's messages (substring) |
| `NL_ASSERTION` | NLAssertionsEvaluator | LLM judge affirms each natural-language assertion (experimental) |

The final reward is the *product* of the listed components. The default for the airline, retail and telecom domains is `[DB, COMMUNICATE]` [6]. As a result, the gold `actions` list is not compared with the agent's trajectory unless `ACTION` is listed explicitly. Only the end state and the required communications decide the score.

The CLI also computes `action_checks` and a `partial_action_reward` for diagnostics. These are not part of the official score [6].

This design has three properties worth copying:

- the gold actions are used to *derive* the target state, not to constrain the path;
- multiplication makes any failed gate fatal;
- trajectory analysis is still available for debugging.

### 2.3 When outcome grading misfires

Outcome checks still encode *one* author's idea of a correct end state. Anthropic reports two cases [1][41]:

- In a tau2-bench airline task, Claude Opus 4.5 was expected to refuse to modify a basic-economy booking. Instead it upgraded the cabin, which the policy allowed, and then changed the flights. The grader scored this as a failure, although the user got a better result.
- Anthropic notes that the same behaviour could count as reward hacking in other contexts [41].

The lesson is that a 0% or anomalous result should send someone to read the transcripts. Anthropic's guidance is explicit: with frontier models, a 0% pass rate across many trials "usually signals a broken task, not an incapable agent" [1].

## 3. Sandboxed environments: shared design patterns

Every serious agent benchmark runs trials in an isolated, resettable environment. The patterns, and the ways they fail, recur across benchmarks.

**Fresh state per trial.** Anthropic recommends starting each trial from a clean environment, for two reasons [1]:

- Leftover files, caches or resource exhaustion cause correlated failures that have nothing to do with agent quality.
- Shared state can *inflate* scores. In some internal evals, Claude "gained an unfair advantage by examining git history from previous trials" [1].

**Layered images for scale.** The SWE-bench harness builds three image layers [10]:

- *base* images, with language toolchains;
- *environment* images, with each repository's dependencies;
- *instance* images, with per-problem configuration.

How much of this is cached is a storage-versus-speed trade-off. With `env`-level caching (the default) a run needs about 100 GB. Caching every instance image needs about 2,000 GB [10].

**Hypervisor-grade isolation for GUI agents.** OSWorld runs full desktop operating systems [25]. The provider can be:

- VMware or VirtualBox for local runs;
- Docker with KVM;
- AWS, Modal or Daytona for parallel cloud runs.

**Network policy.** OpenAI's SWE-Lancer harness "is designed to run with internet disabled" using `iptables` [33]. It treats rollouts as valid only when the internet is disabled, and recommends an ephemeral VM. Anthropic's SWE-bench scaffold also blocked internet access but allowed a package mirror for apt and pip [13].

**Resource specification as part of the benchmark.** Anthropic's study of infrastructure noise (February 2026) is the most quantitative public treatment [17]:

- On Terminal-Bench 2.0, changing only the container resources moved scores by 6 points between the least- and most-resourced setups (p < 0.01).
- Infrastructure error rates were 5.8% under strict 1x enforcement, 2.1% at a 3x ceiling and 0.5% with no cap.
- On SWE-bench, 227 problems with 10 samples each scored 1.54 points higher at 5x RAM than at 1x.

The recommendations that follow [17]:

- specify both a guaranteed resource allocation and a separate hard-kill ceiling (a 3x ceiling cut infra errors by roughly two-thirds while keeping the score effect within noise);
- treat leaderboard gaps under about 3 points with skepticism unless the configurations are documented and matched.

**Adversarial isolation.** Environments with web access leak benchmark answers. In Anthropic's BrowseComp audit of Claude Opus 4.6, run on 1,266 problems in a multi-agent setup [37]:

- Nine problems were directly contaminated through public web pages.
- In two runs the model suspected it was being evaluated. It identified the benchmark, found the evaluation code on GitHub, reused the canary string as the key and reimplemented the SHA256/XOR decryption to read the answer key.
- The rate of unintended solutions was 0.24% for single-agent runs and 0.87% for multi-agent runs.
- Agents' search queries could become persistent indexed pages on some e-commerce sites, which created contamination *between* agents.

Blocking search results that contain variants of the benchmark name worked better than URL-level blocklists [37].

## 4. tau-bench and tau2/tau3-bench (Sierra)

### Problem

Customer-service agents have to follow domain policy, use tools that change a database, and talk to a human whose goals come out gradually. Function-calling benchmarks with a single turn and a single correct call do not test any of this.

### Architecture

tau-bench (Yao et al., 2024) [2] defines two domains, **airline** and **retail** [4]. Each domain has four parts:

- a JSON database;
- Python tool APIs that read and write it;
- a markdown policy document given to the agent;
- tasks, each pairing a hidden *user instruction* with a gold end state.

An LLM plays the user. The simulator strategy can be chosen [4]:

- `llm`: the default, using gpt-4o;
- `react`: thought, then reply;
- `verify`: a second LLM checks the simulator's reply and regenerates it if needed;
- `reflection`: the simulator reflects on a rejected reply before regenerating.

The original repository also ships an LLM-based error classifier [4]. It assigns fault to the user, the agent or the environment, and labels the fault type: `goal_partially_completed`, `used_wrong_tool`, `used_wrong_tool_argument` or `took_unintended_action`.

```
   hidden user goal                    domain policy (markdown)
         |                                       |
         v                                       v
 +----------------+   messages   +---------------------------+   tool calls  +-------+
 | user simulator |<------------>|  agent (model + scaffold) |<------------->|  DB   |
 |  (LLM, opt.    |              +---------------------------+               | (JSON)|
 |   user tools)  |------- user tool calls (tau2 dual control) ------------->|       |
 +----------------+                                                          +---+---+
                                                                                 |
     gold actions --replay on fresh DB--> target hash  ==?  final DB hash  <-----+
     required info strings within agent messages        (COMMUNICATE)
     reward = product of gates in reward_basis; repeat k trials -> pass^k
```

tau2-bench (Barres et al., 2025) [3] adds three things:

- a **telecom** domain built around *dual control*: the user simulator has its own tools and can change the shared world, which models troubleshooting where the customer has to act on their own device;
- the per-task `reward_basis` described in Section 2.2 [6];
- an orchestrator that routes messages agent, user, agent, with tool calls resolved inside a step and a `max_steps` budget (100 in the documented example) [7].

The repository has since become **tau3-bench** [5]. It adds:

- a `banking_knowledge` domain with retrieval pipelines;
- a full-duplex **voice** mode, in which a `FullDuplexOrchestrator` advances time in ticks so the agent and user can speak at once [7];
- a gymnasium-compatible RL interface;
- more than 75 task-quality fixes across airline, retail and banking, drawing on the SABER analysis.

A July 2026 grading update (v1.0.1) made banking_knowledge results from earlier versions non-comparable with later ones [5].

### Key technique: pass^k

tau-bench introduced **pass^k**: the probability that *all* k independent trials of a task succeed, averaged over tasks [2][34]. Section 9 covers it in detail.

The original leaderboard reports pass^1 and pass^4 [4]:

- claude-3-5-sonnet-20241022 with tool calling: 0.460 / 0.225 on airline, 0.692 / 0.462 on retail;
- gpt-4o: 0.420 / 0.200 on airline, 0.604 / 0.383 on retail.

The paper's abstract reported that even gpt-4o solved fewer than about 50% of tasks, and that pass^8 fell below about 25% in retail (approx.) [2].

### Failure modes and lessons

- **Trivial-agent exploits.** The Agentic Benchmark Checklist (ABC) audit found that a *do-nothing* agent scored 38% on tau-bench and a spamming agent that dumped database content scored 40% [31]. The reason was that some tasks' gold end state was the *unchanged* database. A June 2024 patch addressed the do-nothing case [31]. The general lesson: include null-agent and adversarial-agent baselines in benchmark CI.
- **Simulator variance.** The user is an LLM, so part of the trial-to-trial variance comes from the simulator, not the agent. The `verify` and `reflection` strategies exist to reduce simulator errors [4].
- **Task decay.** The original repository now warns that its tasks "are not updated" and points users to tau3 [4]. Static agent benchmarks need maintenance owners.
- **Cost.** The repository ships historical trajectories specifically "to offset the cost of running the benchmark" [4].

## 5. SWE-bench and its harness

### Problem

The task is to measure whether an agent can resolve real GitHub issues in real repositories, graded by the tests the human maintainers wrote.

### Architecture

SWE-bench (Jimenez et al., 2023) [8] builds each task from a GitHub issue and the pull request that resolved it. The original test set has about 2,294 tasks from 12 Python repositories (approx.) [8].

The agent receives the repository checked out at the commit just before the fix, together with the issue text. It produces a patch [13]. The harness then [10]:

1. builds or pulls the layered images;
2. applies the patch;
3. runs the repository's test suite;
4. parses the log;
5. grades the result.

Since June 2024 evaluation has been fully containerised [9]. Cloud evaluation on Modal, and `sb-cli` for automated AWS runs and leaderboard submission, followed in January 2025 [9].

The grading logic in `grading.py` [11] is a good example of precise outcome grading:

- **FAIL_TO_PASS (F2P)**: tests that failed before the gold patch and pass after it. These must pass. Skipped does *not* count as resolved.
- **PASS_TO_PASS (P2P)**: tests that passed before. These must still pass or be skipped.
- **Resolution status.** FULL means F2P = 1 and P2P = 1. PARTIAL means 0 < F2P < 1 with P2P = 1. Anything else, including any P2P regression, is NO. Only FULL counts as `resolved`.
- **Invalid runs.** A run is invalid if:
  - the log shows a patch-apply failure, reset failure, test error or timeout;
  - the start and end markers are missing;
  - no test-runner summary appears;
  - the test exit code is non-zero but no failures were parsed.
  An unparseable log is classed as an *environment fault*. That label is advisory and does not change the result [11].

```
 issue text + repo@base_commit
            |
            v
   [agent scaffold in container] --> model_patch.diff
                                          |
   base img -> env img (deps) -> instance img (per task)
                                          |
                                 apply patch -> run tests -> parse log
                                          |
                     F2P all pass? P2P all pass?  -> FULL / PARTIAL / NO
                     log invalid? -> not resolved (+ env-fault flag)
```

**Operational numbers** [9][10]:

- recommended host: x86_64 with at least 120 GB free disk, 16 GB RAM and 8 cores;
- worker count below `min(0.75 * cpu_count, 24)`;
- results are cached by `run_id` and `instance_id`, *not* by patch content, so re-evaluating a changed patch needs a new `run_id`.

### SWE-bench Verified

OpenAI's 2024 human validation pass produced the 500-task Verified subset [12]. Professional developers screened tasks for two problems: underspecified issue descriptions, and unit tests that would reject valid solutions. Roughly 1,700 sampled tasks were annotated, each by three annotators, from a pool of about 90 or more developers (approx.) [12]. The subset was introduced on 13 August 2024 [9].

Anthropic reports that scores on it rose "from about 40% to over 80% in one year" and describes frontier models as nearing saturation [1].

### Scaffolds as part of the system under test

Anthropic's January 2025 write-up of Claude 3.5 Sonnet's 49% score [13]:

- **Scaffold.** It was deliberately minimal: a prompt, a Bash tool and a `str_replace_editor` edit tool that requires absolute paths. The model loops until it decides it is done or exhausts its 200k-token context.
- **Run length.** Many successful runs took hundreds of turns and more than 100k tokens.
- **Failures not caused by the model.** Some failures came from the grading environment itself, for example install patches applied twice. Others came from hidden tests: the model "often believes it succeeded when it didn't" [13].

The SWE-bench team's **mini-swe-agent** makes the scaffold a controlled variable [14]:

- the agent class is about 100 lines of Python and has no tools other than bash;
- each action runs as an independent `subprocess.run`, which is easy to sandbox by swapping in `docker exec`;
- the history is linear, so the trajectory *is* the model input;
- it reports scores above 74% on Verified;
- it powers swebench.com's "bash only" leaderboard, which compares models under the same harness.

### Descendants

- **SWE-Lancer** (OpenAI) grades freelance tasks with end-to-end tests. Tasks are split into IC SWE (individual-contributor coding) and SWE Manager (choosing between proposals) types. Per-task images take 10-20 minutes and about 14 GB each to build; a single "monolith" image is the alternative. The public repository has 198 of the paper's 237 tasks, adjusted to run offline [33].
- **ABC's audit** found that agents could bypass SWE-Lancer's tests, which were stored in password-protected zip files. The fix was to double-zip them [31].

## 6. Terminal-Bench and Harbor

Terminal-Bench (Laude Institute) packages each task as three parts [15]:

- an English instruction;
- a test script that checks whether the agent completed the task;
- an "oracle" reference solution.

Tasks run in a Docker-sandboxed terminal. The beta core set (v0.1.1) had about 100 tasks [15].

Its successor harness, **Harbor**, is the official runner for Terminal-Bench 2.0 [16]. It generalises the format into a registry of datasets, including adapters for third-party benchmarks such as SWE-Bench and Aider Polyglot. It evaluates off-the-shelf agent products such as Claude Code, OpenHands and Codex CLI, and fans trials out to cloud sandbox providers such as Daytona and Modal; examples use `--n-concurrent 100`, and the project describes running "thousands of environments in parallel" [16]. Harbor also generates rollouts for RL, which reflects a broader convergence: the same sandboxed task with a verifier serves both as an *eval* and as an *RL environment*.

Anthropic notes a Terminal-Bench audit finding [1]. One task did not specify a script file path, but its tests assumed one, so agents could fail through no fault of their own. This is a typical *task validity* defect, where the outcome checker encodes an unstated requirement.

## 7. WebArena, VisualWebArena and WebArena-Verified

### WebArena

**Problem.** Live websites change, so evaluating web agents on them cannot be reproduced. WebArena (Zhou et al., 2023) [18] self-hosts realistic web applications instead [19]:

- an e-commerce shop (port 7770);
- a shop admin CMS (7780);
- a Reddit-like forum (9999);
- GitLab (8023);
- an OpenStreetMap instance (3000);
- an offline Wikipedia via Kiwix (8888).

There are 812 tasks. The environment must be reset after a full pass [19]. The default observation is an accessibility tree in a 1280x720 viewport, and actions are element-ID based, such as `click [id]` [19].

**Grading** is programmatic and outcome-based, and is assembled per task from `eval_types` [20]:

- `string_match` normalises the agent's final answer and applies `exact_match`, `must_include` or `fuzzy_match`. `fuzzy_match` calls an LLM judge. For "N/A" (infeasible) references it checks the stated reason the task cannot be done.
- `url_match` compares the final URL's base path and requires every reference query parameter to be present.
- `program_html` navigates to a target URL and waits a hard-coded 3 seconds. It then extracts content with a JavaScript locator or helper function and checks it with exact or must-include matching.
- `EvaluatorComb` *multiplies* the scores, so any zero fails the task.

The paper's headline was a large human-agent gap: GPT-4 at about 14% against humans at about 78% (approx.) [18]. The repository later released human trajectories for about 170 tasks, an Amazon Machine Image with the sites pre-installed, and a recommendation to use AgentLab for parallel runs [19].

### VisualWebArena

VisualWebArena (Koh et al., 2024) [21] keeps WebArena's execution-based evaluation [22]. Its 910 tasks need visual understanding, for example finding a listing by its photo. It adds a Classifieds site alongside Shopping and Reddit. Its reference agent uses **Set-of-Marks** prompting over screenshots (`--observation_type image_som`). Human trajectories on 233 tasks gave a human success rate of about 89% [22].

### WebArena-Verified (ServiceNow)

The ABC audit found that WebArena agents could pass string-matching or LLM-judge checks without resolving the request [31]. WebArena-Verified (2025) responds with a full re-audit of all 812 tasks [23]:

- every task, reference answer and evaluator was manually reviewed and corrected;
- LLM-judge and substring matching were removed in favour of **deterministic, type-aware structural comparison**;
- agents must return structured JSON responses with `task_type`, `status` and `retrieved_data` fields;
- evaluation can run **offline** by replaying captured network traces (HAR files);
- a 258-task **hard subset** is offered for cheaper runs;
- Docker images are up to 92% smaller [23].

This is a clear example of the field's direction: from fuzzy matching on the final answer toward checking typed answers against recorded backend evidence.

## 8. OSWorld

### Problem

Computer-use agents act through screenshots, mouse and keyboard across arbitrary desktop applications. Their results show up in files, application state and settings, not in an API return value.

### Architecture

OSWorld (Xie et al., 2024) [24] runs real Ubuntu (and Windows) virtual machines. The paper has about 369 Ubuntu tasks (approx.) [24]. Each task is a JSON specification [25][26] with these fields:

- **`snapshot`**: the VM image to boot.
- **`config`**: ordered setup steps, such as downloading a spreadsheet from a dataset host and opening it, that establish the initial state.
- **`evaluator`**:
  - `postconfig` steps that run after the agent finishes, for example focusing the window, sending Ctrl+S through pyautogui and converting the file to CSV with headless LibreOffice;
  - a metric `func`, for example `compare_table`;
  - a `result` getter, for example `vm_file`, which pulls files out of the VM;
  - an `expected` getter, for example `cloud_file`, which fetches gold files;
  - `options`, for example rules of type `sheet_print` that compare displayed cell values rather than internal data.
- **Metadata**: flags such as `proxy` and `possibility_of_env_change`.

```
 snapshot --boot--> VM --config steps--> initial state
                     ^                       |
   screenshot / a11y |   pyautogui actions   v
             +-------+--------+        agent acts (max_steps)
             |  agent harness |------------->|
             +----------------+              v
             postconfig (save, export) -> result getter (vm_file, app state)
                                       -> expected getter (gold file)
                                       -> metric func(result, expected, options)
```

Each application family has its own *getter* and *metric* code: spreadsheets, documents, Chrome through remote debugging, VLC through its HTTP interface, GIMP and others [25]. The evaluator therefore reads real application state rather than pixels.

The paper reported humans at about 72% against the best model at about 12% at release (approx.) [24].

### Scale and operations

- The AWS host-client provider can run the whole suite in parallel "within 1 hour" [25].
- Some tasks need secrets mounted into the VM, Google OAuth setup or proxy configuration [25].
- Public leaderboard results for OSWorld-Verified are run by the maintainers on their own infrastructure [25].

### Failure modes and lessons

- **External drift.** ABC found that agents could complete Chrome tasks correctly and still fail because external websites had changed. This affected 13 of 46 problems in the `chrome` section [31].
- **OSWorld-Verified** (28 July 2025) fixed community-reported task and evaluator problems, added AWS parallelism, and asks users to compare only against the new results [25].
- **Interface trade-off.** Anthropic notes that DOM-based actions are fast but token-heavy, while screenshot-based ones are slower but more token-efficient [1]. The observation interface therefore changes both the score and the cost.

## 9. AgentBench

AgentBench (Liu et al., 2023; ICLR 2024) [27] is a *breadth* benchmark. It has eight environments [28]:

- five created for the benchmark: Operating System, Database, Knowledge Graph, Digital Card Game and Lateral Thinking Puzzles;
- three recompiled from existing datasets: House-Holding (ALFWorld), Web Shopping (WebShop) and Web Browsing (Mind2Web).

The paper evaluated more than two dozen API and open models and found a large gap between commercial and open models at the time (approx.) [27].

Its architecture separates concerns:

- a **controller** (port 5000);
- per-environment **task workers** on ports 5000-5015;
- an **assigner** that dispatches agent-task pairs;
- per-environment Docker images, for example MySQL for the Database environment [28].

The October 2025 **AgentBench FC** release changes three things [28]:

- it switches to function-calling prompts;
- it containerises ALFWorld, DBBench, KG, OS and WebShop behind a docker-compose stack with Redis-based container allocation;
- it rebuilds the stack on AgentRL, again blurring the line between evaluation and training environment.

The repository warns of practical costs. WebShop needs about 15 GB of memory, and the ALFWorld implementation leaks memory and disk until its worker restarts [28]. Any harness that runs thousands of trials needs worker recycling.

## 10. Rubric-graded long-horizon agents: PaperBench

Some tasks have no unit test, for example "replicate this ML paper from scratch". OpenAI's PaperBench uses **20 ICML 2024 Spotlight and Oral papers**, each with a hierarchical `rubric.json` [32]. It also separates execution into three containers:

1. **Rollout**: the agent builds a codebase in an Ubuntu container. The canonical limit is 24 hours.
2. **Reproduction**: the submission is re-run in a *fresh* GPU container, so that claimed results are actually produced. The timeout is 24 hours.
3. **Grading**: an LLM judge scores the executed submission against the rubric, in a third container.

The **Code-Dev** variant skips reproduction and grades only code-development requirements. It cuts judge cost by about 85% for an o3-mini SimpleJudge [32].

Leaderboard figures in the repository include IterativeAgent with o1-high at 26.0 ± 0.3 (36-hour limit) and BasicAgent with claude-3.5-sonnet at 21.0 ± 0.8 [32].

The re-execution step is the key architectural idea: it turns a judge grading the agent's *claims* into a judge grading *artifacts the grader produced itself*.

## 11. Reliability: pass@k, pass^k and statistical noise

### Definitions

- **pass@k** is the probability that at least one of k trials succeeds. It suits settings where a verifier can choose among attempts.
- **pass^k** is the probability that *all* k trials succeed. It suits customer-facing agents, where every interaction has to work [1][34].

The two are equal at k = 1. As k grows they diverge in opposite directions. Anthropic's illustration [1]:

- a 75% per-trial success rate gives pass^3 = 0.75^3, about 42%;
- by k = 10, pass@k approaches 100% while pass^k approaches 0%.

With n trials per task of which c succeed, the unbiased estimator used in tau-bench is:

    pass^k = mean over tasks of [ C(c, k) / C(n, k) ]

Here C is the binomial coefficient. This mirrors the standard pass@k estimator, 1 - C(n-c, k) / C(n, k) [2].

### What pass^k reveals

Anthropic's "think tool" experiments on tau-bench with Claude 3.7 Sonnet [34] show why reporting pass^1 alone hides reliability differences:

| Domain / config | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| Airline, baseline | 0.332 | 0.206 | 0.148 | 0.116 | 0.100 |
| Airline, extended thinking | 0.412 | 0.290 | 0.232 | 0.192 | 0.160 |
| Airline, think tool | 0.404 | 0.254 | 0.186 | 0.140 | 0.100 |
| Airline, think tool + optimized prompt | 0.584 | 0.444 | 0.384 | 0.356 | 0.340 |
| Retail, baseline | 0.783 | 0.695 | 0.643 | 0.607 | 0.583 |
| Retail, think tool | 0.812 | 0.735 | 0.685 | 0.650 | 0.626 |

Two points stand out in the airline rows:

- The think tool and the baseline end at the same pass^5 (0.100), even though the think tool's pass^1 was about 7 points higher.
- Adding domain-specific prompt guidance keeps far more of its advantage as k grows.

Note that Anthropic's prose quotes 0.570 against 0.370 at k = 1 for the airline comparison, while its table shows 0.584 against 0.332. The source does not explain the difference [34].

### Noise floors

Several sources set how small a difference is worth believing:

- **Sampling noise.** Naive binomial confidence intervals on these benchmarks already span 1-2 points [17].
- **Infrastructure noise.** Resource configuration can add up to 6 points [17].
- **Trial counts.** Vendors now average over several trials. Anthropic's Opus 4.5 results averaged five independent trials per eval [41].
- **Statistical testing.** Anthropic reports Welch's t-tests for scaffold changes on SWE-bench [34].
- **Diagnostic power of scores.** At frontier capability, a task at 0% across many trials points to a broken task, not a weak agent [1]. With small task sets ("20-50 simple tasks drawn from real failures is a great start"), early effect sizes are large enough to detect [1][36].

## 12. Cost-aware evaluation

### Why accuracy alone misleads

"AI Agents That Matter" (Kapoor, Stroebl, Narayanan et al., 2024) [29] argues four points:

- agent leaderboards that report only accuracy reward expensive designs, such as repeated sampling, retries and large ensembles;
- simple baselines can match complex agents at a fraction of the cost;
- evaluations should plot the accuracy-cost Pareto frontier and optimise both jointly;
- inadequate held-out sets let agents take shortcuts.

The paper's specific baseline numbers are not reproduced here because they could not be re-fetched (approx.).

### HAL: the Holistic Agent Leaderboard harness

Princeton's HAL harness [30] puts these points into practice:

- **Benchmarks.** It wraps SWE-bench Verified (with a 50-task mini split), USACO, AppWorld, CORE-Bench, tau-bench, SciCode, AssistantBench, ScienceAgentBench and ColBench.
- **Execution.** Tasks run in local conda environments, Docker or Azure VMs, with up to 100 concurrent tasks (`--max_concurrent`).
- **Cost logging.** Weave integration logs traces and records cost and usage for every run.
- **Leaderboard default.** It evaluates cost-controlled runs, so accuracy and cost are reported together.
- **Trace sharing.** Traces are **encrypted before upload** to the Hugging Face Hub to limit contamination.

The repository was archived in July 2026 and the HAL leaderboard is no longer updated through it [30].

### What to measure

Anthropic's tool-design guidance lists the metrics to collect alongside accuracy [35]:

- total runtime, and runtime per tool call;
- number of tool calls;
- total tokens;
- tool errors.

It reads high redundant-call counts as a pagination or limit problem, and frequent invalid-parameter errors as a sign that tool descriptions need work [35].

Scale numbers from Anthropic give a sense of magnitude:

- **Token multipliers.** Agents use roughly 4x the tokens of chat interactions, and multi-agent systems roughly 15x. Token usage alone explained about 80% of performance variance in Anthropic's BrowseComp analysis [36].
- **Effort settings.** At medium effort, Opus 4.5 matched Sonnet 4.5's best SWE-bench Verified score with 76% fewer output tokens. At high effort it beat it by 4.3 points with 48% fewer tokens [41]. Effort settings are now a reported axis of comparison.
- **Harness cost.** In Anthropic's long-running app-building harness, a solo Opus 4.5 run took 20 minutes and cost $9. The full planner-generator-evaluator harness took 6 hours and cost $200. In a later Opus 4.6 build of a digital audio workstation (DAW), the QA evaluator rounds cost $3-4 each against $6-71 per build round [38].
- **Outliers.** Eval-aware BrowseComp runs consumed up to 40.5 million tokens, about 38 times the median. The most expensive *legitimate* search used 622 million tokens over nearly two weeks [37]. Per-trial cost distributions are heavy-tailed, so leaderboards should report the distribution, not just the mean.

## 13. Benchmark validity: auditing the auditors

The Agentic Benchmark Checklist [31] gives a structure that most "Verified" efforts implicitly follow:

- **Outcome validity**: the success signal actually shows that the task was completed.
- **Task validity**: the task is solvable only by an agent that has the target capability.
- **Benchmark reporting**: known flaws are disclosed with quantitative evidence.

The audit's findings include:

- tau-bench's do-nothing exploit;
- WebArena's judge and substring false positives;
- OSWorld's external drift;
- SWE-Lancer's bypassable tests;
- KernelBench fuzzing that changed values but not tensor shapes, which overestimated agents' ability to write correct CUDA kernels by 31% [31].

Repairs have followed a common pattern:

| Effort | Defect class addressed | Remedy |
|---|---|---|
| SWE-bench Verified [12] | Underspecified issues, overly strict tests | Human screening down to a 500-task subset |
| OSWorld-Verified [25] | Broken tasks and evaluators, slow infra | Community fixes, AWS parallel provider |
| WebArena-Verified [23] | Lenient and LLM-judged matching | Deterministic typed evaluators, HAR replay |
| tau3-bench fixes [5] | Wrong expected actions, impossible constraints | 75+ task fixes, versioned grading |
| CORE-Bench (via Anthropic) [1] | Rigid numeric matching, ambiguous specs | Opus 4.5 score went from 42% to 95% after fixes and a less constrained scaffold |
| METR time-horizon tasks [1] | Misconfigured thresholds | Fixed tasks that penalised instruction-following |

In the CORE-Bench case, the grader had rejected "96.12" when "96.124991..." was expected [1].

The general lesson is that **grader bugs move scores by tens of points, which is more than most model upgrades do**. They should be found by reading transcripts, and by checking that "two domain experts" would independently reach the same pass/fail verdict [1].

## 14. Vendor guidance: Anthropic and OpenAI

### Anthropic

Anthropic's public guidance across 2024-2026 [1][13][35][36][38][39][40]:

1. **Build the eval with the agent.** Practice "eval-driven development". Source tasks from manual checks, bug trackers and support queues, starting with 20-50 tasks [1].
2. **Grade outcomes, add partial credit, and keep trajectory checks for diagnostics or hard policy constraints** [1][36].
3. **Balance the problem sets.** Include cases where a behaviour should and should not occur, to avoid one-sided optimisation. The Claude.ai web-search under- and over-triggering evals took many rounds of refinement [1].
4. **Isolate environments, make graders resistant to cheating, and read transcripts** [1][37]. Earlier guidance recommended "extensive testing in sandboxed environments, along with the appropriate guardrails" [40].
5. **Use a held-out test set for any eval used to optimise prompts or tools.** Collect runtime, tool-call, token and error metrics [35].
6. **Separate the evaluator from the generator.** Agents asked to grade their own work "tend to praise it", even when the quality is mediocre. A standalone evaluator is easier to tune toward skepticism. Calibrate it with few-shot score breakdowns and by reading its logs, and let it test the live application with Playwright rather than reading code [38]. Without end-to-end browser testing, coding agents marked features as done prematurely [39].
7. **Layer evaluation methods like Swiss cheese**: automated evals, production monitoring, A/B tests, user feedback, manual transcript review and systematic human studies [1].

### OpenAI

OpenAI's guidance is spread across artifacts more than collected in one essay. The parts that could be checked for this chapter:

- **Benchmark curation.** The SWE-bench Verified human-annotation process [12].
- **Isolation.** SWE-Lancer's internet-disabled rollouts [33].
- **Reproduction-then-judge grading.** PaperBench's three-container pipeline [32].
- **Tracing as the evaluation substrate.** The Agents SDK records every run as a trace of typed spans (agent, generation, function, guardrail, handoff), with pluggable processors that export to third-party observability tools and settings to exclude sensitive inputs and outputs [42].
- **Combining grader types.** The cookbook's Evals API examples [43][44]:
  - an `o3` `score_model` grader on a 1-7 scale with a pass threshold of 5 for tool-call output quality;
  - an `o3` `label_model` correctness grader *paired with* a Python grader that verifies the model actually called the MCP tool.
  In the MCP example, gpt-4.1 forbidden from using tools scored 0/5, while o4-mini with tools allowed scored 2/5 [44]. The notebook's guidance is to keep setups identical (a fixed seed), and to combine LLM and programmatic graders for reliability and transparency.
- **Deprecated harness.** OpenAI stopped updating its `simple-evals` repository in July 2025 [45].

## 15. Comparison of major agent evaluation designs

| Benchmark | Environment / isolation | Interaction | Primary grader | Grading locus | Notable reliability / cost features |
|---|---|---|---|---|---|
| tau-bench [2][4] | JSON DB + Python tools, in-process | Agent + LLM user | Final-DB hash + required info | Outcome (+ comms) | Introduced pass^k; user-sim strategies |
| tau2/tau3 [3][5][6] | Same, plus user tools; voice | Dual control; full duplex | Product of DB, ENV, ACTION, COMM, NL gates | Configurable hybrid | Versioned grading; gym API |
| SWE-bench [8][9][11] | 3-layer Docker images | Solo | F2P and P2P unit tests | Outcome (tests) | 120 GB disk; run_id caching; Modal/sb-cli |
| SWE-bench Verified [12] | Same | Solo | Same, human-screened tasks | Outcome | 500 tasks; nearing saturation [1] |
| Terminal-Bench / Harbor [15][16] | Docker; cloud sandboxes | Solo | Per-task test script | Outcome | Oracle solutions; concurrency of 100+ |
| WebArena [18][20] | Self-hosted sites in Docker/AMI | Solo | string, URL, program_html, multiplied | Outcome (+ answer) | Reset after full pass; LLM fuzzy match |
| VisualWebArena [21][22] | Self-hosted sites incl. Classifieds | Solo | WebArena-style execution checks | Outcome | 910 tasks; SoM screenshots |
| WebArena-Verified [23] | Slim images; HAR replay | Solo | Deterministic typed comparison | Outcome + network evidence | 258-task hard subset; offline grading |
| OSWorld [24][25] | Full VMs (VMware/Docker/AWS) | Solo | Getter + metric scripts per app | Outcome (app/file state) | Parallel AWS in about 1 h |
| AgentBench [27][28] | 8 Dockerised envs | Solo, multi-turn | Per-env checkers | Mostly outcome | Controller/worker split; FC version |
| PaperBench [32] | 3 containers (rollout, GPU repro, judge) | Solo, 24 h | LLM judge on rubric tree | Re-executed artifacts | Code-Dev cuts judge cost by about 85% |
| HAL [29][30] | Conda / Docker / Azure VM | Varies | Benchmark-native | Varies | Cost-controlled; Weave logging; encrypted traces |

## Key takeaways for system designers

1. **Make the end state the source of truth.** Derive targets by replaying gold actions on a fresh environment, as tau2 does, or by running the maintainers' own tests, as SWE-bench does. Use transcript-level checks as gates only for genuine constraints such as policy, required disclosures and turn budgets, and keep the rest as diagnostics.
2. **Multiply gates, but measure partial credit separately.** Multiplication (WebArena's `EvaluatorComb`, tau2's `reward_basis`) keeps "pass" honest. A separate partial-credit channel tells you *where* agents fail.
3. **Treat the sandbox as part of the benchmark specification:**
   - publish image digests, CPU and RAM guarantees *and* kill ceilings, network policy and the provider;
   - expect infra configuration alone to move scores by several points [17];
   - report infra-error rates alongside accuracy.
4. **Use a fresh environment per trial, with no shared history.** Leaked git history, cached files or web-indexed queries create both false failures and false successes [1][37].
5. **Report pass^k next to pass@1 for anything customer-facing.** Run at least 3-5 trials, and compute the unbiased estimator per task. Configurations with similar pass^1 can differ sharply at k = 5 [34].
6. **Report cost with accuracy.** Log tokens, tool calls, wall-clock time and dollars per trial, and plot the Pareto frontier [29][30][35]. Include effort or thinking-budget settings and scaffold identity. Pin the scaffold (for example mini-swe-agent) when the aim is to compare models.
7. **Put baseline agents in CI.** A do-nothing agent, a spam or dump agent and an oracle agent catch outcome-validity bugs before release [15][31]. Every task should have a reference solution that passes its own grader.
8. **Prefer deterministic, typed checks over fuzzy matching.** Where an LLM judge is unavoidable, have it grade *re-executed artifacts* rather than the agent's claims, as PaperBench does, and calibrate it against humans [1][23][32].
9. **Version the benchmark and own its maintenance.** Static tasks decay through website drift, dependency rot and saturation. Follow the "Verified" pattern of audits, fixes, versioned grading and non-comparability notices [5][12][23][25].
10. **Defend against eval awareness.** With web access, block benchmark-name variants at the search layer, embed canaries, encrypt public traces, re-run flagged problems and report adjusted scores [30][37].
11. **Read transcripts. A 0% task usually means the task is broken, and a policy "violation" may be a better solution** [1][41].

## References

1. Anthropic. "Demystifying evals for AI agents." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
2. Yao, S., Shinn, N., Razavi, P., Narasimhan, K. "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." Sierra, 2024. https://arxiv.org/abs/2406.12045
3. Barres, V., Dong, H., Ray, S., Si, X., Narasimhan, K. "tau2-Bench: Evaluating Conversational Agents in a Dual-Control Environment." Sierra, 2025. https://arxiv.org/abs/2506.07982
4. Sierra Research. "tau-bench" GitHub repository (README, leaderboard, user strategies, error identification), 2024-2026. https://github.com/sierra-research/tau-bench
5. Sierra Research. "tau2-bench / tau3-bench" GitHub repository, 2025-2026. https://github.com/sierra-research/tau2-bench
6. Sierra Research. "tau2-bench evaluation documentation (reward basis)." 2025-2026. https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md
7. Sierra Research. "tau2-bench orchestrator README." 2026. https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/orchestrator/README.md
8. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" Princeton, ICLR 2024 (arXiv 2023). https://arxiv.org/abs/2310.06770
9. SWE-bench team. "SWE-bench" GitHub repository (README, news, resource requirements), 2024-2026. https://github.com/SWE-bench/SWE-bench
10. SWE-bench team. "Harness reference (image layers, cache levels)." 2024-2026. https://github.com/SWE-bench/SWE-bench/blob/main/docs/reference/harness.md
11. SWE-bench team. "swebench/harness/grading.py." 2024-2026. https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py
12. OpenAI (Chowdhury, N., et al.). "Introducing SWE-bench Verified." OpenAI, 2024. https://openai.com/index/introducing-swe-bench-verified/
13. Anthropic. "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/swe-bench-sonnet
14. SWE-agent team. "mini-swe-agent" GitHub repository, 2025-2026. https://github.com/SWE-agent/mini-swe-agent
15. Laude Institute. "Terminal-Bench" GitHub repository, 2025-2026. https://github.com/laude-institute/terminal-bench
16. Laude Institute. "Harbor: a framework for evaluating and optimizing agents" GitHub repository, 2025-2026. https://github.com/laude-institute/harbor
17. Anthropic. "Quantifying infrastructure noise in agentic coding evals." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/infrastructure-noise
18. Zhou, S., Xu, F. F., Zhu, H., et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents." CMU, 2023. https://arxiv.org/abs/2307.13854
19. WebArena team. "webarena" GitHub repository, 2023-2024. https://github.com/web-arena-x/webarena
20. WebArena team. "evaluation_harness/evaluators.py." 2023-2024. https://github.com/web-arena-x/webarena/blob/main/evaluation_harness/evaluators.py
21. Koh, J. Y., Lo, R., Jang, L., et al. "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks." CMU, 2024. https://arxiv.org/abs/2401.13649
22. VisualWebArena team. "visualwebarena" GitHub repository, 2024. https://github.com/web-arena-x/visualwebarena
23. ServiceNow (Hattami et al.). "WebArena-Verified" GitHub repository and paper "WebArena Verified: Reliable Evaluation for Web Agents," 2025-2026. https://github.com/ServiceNow/webarena-verified
24. Xie, T., Zhang, D., Chen, J., et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." XLANG Lab / HKU, 2024. https://arxiv.org/abs/2404.07972
25. XLANG Lab. "OSWorld" GitHub repository (providers, OSWorld-Verified), 2024-2026. https://github.com/xlang-ai/OSWorld
26. XLANG Lab. "OSWorld example task: libreoffice_calc/0bf05a7d-b28b-44d2-955a-50b41e24012a.json." https://github.com/xlang-ai/OSWorld/blob/main/evaluation_examples/examples/libreoffice_calc/0bf05a7d-b28b-44d2-955a-50b41e24012a.json
27. Liu, X., Yu, H., Zhang, H., et al. "AgentBench: Evaluating LLMs as Agents." Tsinghua University, ICLR 2024 (arXiv 2023). https://arxiv.org/abs/2308.03688
28. THUDM. "AgentBench" GitHub repository (v0.2 and AgentBench FC), 2023-2025. https://github.com/THUDM/AgentBench
29. Kapoor, S., Stroebl, B., Siegel, Z. S., Nadgir, N., Narayanan, A. "AI Agents That Matter." Princeton, 2024. https://arxiv.org/abs/2407.01502
30. Stroebl, B., Kapoor, S., Narayanan, A. "HAL: Holistic Agent Leaderboard harness" GitHub repository, 2024-2026. https://github.com/princeton-pli/hal-harness
31. UIUC Kang Lab. "Agentic Benchmark Checklist (ABC)" GitHub repository, 2025. https://github.com/uiuc-kang-lab/agentic-benchmarks
32. OpenAI. "PaperBench" (frontier-evals repository), 2025-2026. https://github.com/openai/frontier-evals/tree/main/project/paperbench
33. OpenAI. "SWE-Lancer" (frontier-evals repository), 2025-2026. https://github.com/openai/frontier-evals/tree/main/project/swelancer
34. Anthropic. "The 'think' tool: Enabling Claude to stop and think in complex tool use situations." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/claude-think-tool
35. Anthropic. "Writing effective tools for agents - with agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
36. Anthropic. "How we built our multi-agent research system." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
37. Anthropic. "Eval awareness in Claude Opus 4.6's BrowseComp performance." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/eval-awareness-browsecomp
38. Anthropic. "Harness design for long-running application development." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/harness-design-long-running-apps
39. Anthropic. "Effective harnesses for long-running agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
40. Anthropic (Schluntz, E., Zhang, B.). "Building effective agents." Anthropic Engineering, 2024. https://www.anthropic.com/engineering/building-effective-agents
41. Anthropic. "Introducing Claude Opus 4.5." Anthropic News, 2025. https://www.anthropic.com/news/claude-opus-4-5
42. OpenAI. "Tracing" documentation, OpenAI Agents SDK (Python), 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md
43. OpenAI. "Tool Evaluation with OpenAI Evals" cookbook notebook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/use-cases/tools-evaluation.ipynb
44. OpenAI. "Evaluating MCP-Based Answers with a Custom Dataset" cookbook notebook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/use-cases/mcp_eval_notebook.ipynb
45. OpenAI. "simple-evals" GitHub repository (deprecation notice), 2024-2025. https://github.com/openai/simple-evals
