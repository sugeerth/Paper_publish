# 12. Code Evaluation in Depth: From Unit-Tested Functions to Repository Agents and In-IDE Assistants

## Overview

Code is the domain where LLM evaluation looks easiest, because a program can be run and its output checked. That advantage is real. It is why code benchmarks moved quickly from BLEU-style text similarity to **execution-based grading**, and why coding is now the main test bed for agent evaluation. The advantage is also why code evaluation keeps failing in quiet ways. A test suite is a partial specification. A sandbox is a piece of infrastructure with its own failure modes. A public repository is almost certainly training data. Each generation of code benchmark has had to rediscover these facts at a larger scale.

This chapter follows four generations of design:

```
 GEN 1  function synthesis      HumanEval (164), MBPP (~1k), APPS (~10k), CodeContests
        prompt -> function -> hidden unit tests -> pass@k
            |  problem: weak tests, saturation, contamination, Python-only
            v
 GEN 2  hardened / broadened    EvalPlus (80x / 35x tests), MultiPL-E (18 languages),
        function-level          BigCodeBench (1,140 library-heavy tasks), LiveCodeBench
                                (dated contest problems), LiveCodeBench Pro
            |  problem: still single-file, still "toy" relative to real work
            v
 GEN 3  repository context      RepoBench, CrossCodeEval (cross-file completion);
                                SWE-bench (issue -> patch -> FAIL_TO_PASS/PASS_TO_PASS)
                                and its family: Verified, Multimodal, Multilingual,
                                Multi-SWE-bench, SWE-PolyBench, SWE-bench-Live, Pro
            |  problem: scaffold, infra and test validity dominate the score
            v
 GEN 4  end-to-end / in-product Aider polyglot (edit format + test feedback),
                                Terminal-Bench 2.0, SWE-Lancer; in-IDE evals:
                                Copilot simulation tests, Copilot Arena, acceptance
                                telemetry, A/B tests
```

The pattern across generations is consistent. Each step adds realism (more files, more languages, more of the real workflow), and each step moves more of the score's variance away from the model into things around it: the tests, the container, the scaffold, the time budget and the evaluator's resource limits. By 2026 the critical question about a coding number is less "which benchmark?" than "which benchmark version, which harness, which scaffold, how many trials, which resource envelope, and which contamination controls?"

The chapter is organized as follows. Section 12.1 defines the execution-grading contract shared by all designs. Sections 12.2 to 12.4 cover function-level, contest-level and repository-completion benchmarks. Section 12.5 is a long section on the SWE-bench family. Section 12.6 covers Aider's polyglot editing benchmark. Section 12.7 covers execution infrastructure, 12.8 contamination, and 12.9 product evaluation of code assistants. A comparison table, a reference pipeline, key takeaways and references close the chapter. Chapter 2 introduced several of these benchmarks briefly; this chapter goes into their internals.

Source note: this chapter relies mainly on primary sources that could be fetched directly at the time of writing: repository READMEs and source code, Anthropic engineering and model-launch pages, and Microsoft Research pages. Where a number comes from a source summarized in earlier chapters of this Atlas, or from secondary reporting, the text says so. Several vendor sites, including Cursor's, could not be retrieved. Section 12.9 therefore describes Cursor-style product evaluation only through comparable, documented systems and makes no claims about Cursor's internal methodology.

## 12.1 The execution-grading contract

Every design in this chapter instantiates the same contract:

```
  +-------------+    +-------------+    +-------------------+    +----------------+
  | task spec   |--->| model /     |--->| candidate         |--->| sandboxed      |
  | (prompt,    |    | agent +     |    | (function, patch, |    | executor       |
  |  repo@sha,  |    | scaffold    |    |  edited files)    |    | (process, VM,  |
  |  issue)     |    +-------------+    +-------------------+    |  container)    |
  +-------------+                                                +-------+--------+
                                                                         |
            hidden tests / verifier (unit tests, F2P+P2P, e2e, output diff)
                                                                         |
                                          +------------------------------v--------+
                                          | per-attempt verdict: pass | fail |    |
                                          | timeout | apply-fail | infra-error     |
                                          +------------------------------+--------+
                                                                         |
                                 aggregate: pass@k, pass^k, resolve rate, cost, CI
```

Three properties of this contract matter for system design.

**The verdict must have more than two values.** "Timed out", "patch did not apply" and "infrastructure error" are different from "wrong answer", and a harness that collapses them silently turns infrastructure noise into model error. The SWE-bench grader is the most careful public example. It rejects a log as invalid when it contains apply-failure, reset-failure, test-error or timeout markers, when the start and end test-output markers are missing, when no evidence exists that the suite ran ("Executed 0 of 0" does not count), or when the test exit code is non-zero but no failing test was parsed [19]. The original HumanEval harness, by contrast, records only "passed", "timed out" or "failed" [2].

**pass@k needs an unbiased estimator.** HumanEval introduced the estimator that is now standard. Draw n samples per problem, count c correct, and compute 1 - C(n-c, k) / C(n, k). The reference implementation evaluates this as a running product to avoid huge binomials [3]:

```
def estimator(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

The harness refuses to report pass@k when fewer than k samples exist, because no unbiased estimate is possible [2]. Its defaults are k in {1, 10, 100}, 4 worker processes and a 3.0-second timeout per program [3]. The BigCode harness uses n=200 samples at temperature 0.2 and top-p 0.95 for HumanEval [8].

**Agentic evals need a consistency metric as well.** Anthropic's guidance for agent evals contrasts pass@k (at least one of k trials succeeds) with **pass^k** (all k trials succeed). A 75% per-trial success rate gives about 42% pass^3. As k grows, pass@k approaches 100% and pass^k falls toward 0% [44]. A code-review bot or CI fixer that users rely on every time should be judged by pass^k. A best-of-n generation pipeline with a reliable verifier can be judged by pass@k.

## 12.2 Function-level benchmarks: HumanEval, MBPP, APPS and their repairs

### 12.2.1 HumanEval (OpenAI, 2021)

**Problem.** Measure functional correctness of code generated from a docstring, rather than textual similarity to a reference [1].

**Design.** 164 hand-written Python problems. Each has a `task_id`, a `prompt` (signature plus docstring), a canonical solution and a `test` function [2][8]. In the released file, tests average about 8 `assert` statements per problem (median 7; this chapter's count over `HumanEval.jsonl.gz`). Completions are submitted as JSONL lines containing only the generated body, and `evaluate_functional_correctness` reports pass@1, @10 and @100 [2].

**Execution.** `check_correctness` runs each program in a child `multiprocessing.Process` inside a temporary directory. A `SIGALRM` timer enforces the time limit, and the parent kills the child after `timeout + 1` seconds [4]. A `reliability_guard` sets dangerous functions such as `os.kill`, `os.system`, `os.fork`, `shutil.rmtree` and `subprocess.Popen` to `None`, and blocks imports such as `psutil` and `resource`. Its docstring is explicit that "This function is NOT a security sandbox" [4]. Memory limits are only applied when a maximum is passed, and the default call passes none [4]. The README asks users to read a disclaimer and run the suite only in a robust sandbox [2].

**Failure modes.**
- *Infrastructure masquerading as model error.* On low-RAM hosts the harness can print "malloc: can't allocate region" and fail correct programs [2].
- *Weak tests.* With only a handful of asserts per problem, incorrect programs can pass. EvalPlus (below) quantifies this.
- *Saturation and contamination.* The problems have been public since 2021 and are small, so they are now a sanity check, not a frontier measure.

### 12.2.2 MBPP (Google, 2021)

MBPP ("Mostly Basic Python Problems") contains about 1,000 crowd-sourced entry-level problems [5]. The released `mbpp.jsonl` has 974 rows, and the hand-verified `sanitized-mbpp.json` has 427 (this chapter's count). Each problem has a description, a reference solution and three test cases [5]. Task IDs define the splits: 1-10 for few-shot prompts, 11-510 for test (500 problems), 511-600 for validation and 601-974 for training [5]. The original paper used three-shot prompts with tasks 2, 3 and 4 [5].

A design detail that matters: MBPP prompts conventionally **include a test case**. The BigCode harness builds an InCoder-style docstring with the description and one test example, samples n=15 at temperature 0.1, and reports pass@1 [8]. A visible test leaks the function name and signature, and can leak the expected I/O format. MBPP numbers from different harnesses are therefore often not comparable, because the prompt construction differs.

### 12.2.3 APPS (Berkeley et al., NeurIPS 2021) and CodeContests (DeepMind, 2022)

APPS has about 10,000 Python problems, split 5,000 for training and 5,000 for test, at three difficulty levels: introductory, interview and competition [6][8]. It is graded by standard-input/standard-output tests. Two metrics are used with one sample: **average accuracy** (fraction of a problem's tests passed) and **strict accuracy** (a problem counts only if every test passes). pass@k is used with many samples [8]. Average accuracy rewards partial solutions and is easy to inflate. Strict accuracy is the one to trust.

CodeContests, released with AlphaCode, draws from Aizu and AtCoder (via CodeNet), CodeChef and HackerEarth (via description2code) and Codeforces. It includes paired I/O tests plus correct and **incorrect** human solutions [12]. Incorrect solutions are useful for a property that most benchmarks ignore: whether a test suite rejects known-wrong programs. The repository also warns that attached solutions are "not guaranteed to compile and execute in the exact same way" as on the original judges [12]. Judge semantics (compiler flags, time limits, checker tolerance) are part of the benchmark.

### 12.2.4 Repairing the tests: EvalPlus

**Problem.** If the tests are weak, pass@k overstates correctness.

**Design.** EvalPlus generates many more inputs for each problem and checks the candidate against the ground-truth solution as a differential oracle. HumanEval+ has "80x more tests than the original HumanEval" and MBPP+ "35x more tests than the original MBPP" [9]. The work was published at NeurIPS 2023 under the title "Is Your Code Generated by ChatGPT Really Correct?" [9]. EvalPlus also revealed a second-order problem: the ground truth itself can be wrong. MBPP+ v0.2.0 removed broken tasks, going from 399 to 378, which the maintainers expected to raise pass@1 by about 4 points. Further ground-truth fixes to individual tasks followed in June 2024 [9]. The project adds **EvalPerf** for code efficiency, which needs `perf` counters and a container started with `--cap-add PERFMON` [9].

**Lesson.** Every function-level benchmark should be treated as two artifacts with separate error rates: the task set and the oracle. EvalPlus fixed both, and because it changed scores, a "HumanEval" number must say whether it is base or plus.

### 12.2.5 Breadth: MultiPL-E and BigCodeBench

**MultiPL-E** translates HumanEval and MBPP into 18 more languages. Each language has a translator (`humaneval_to_L.py`) that converts the prompt, wraps the unit tests and maps Python values to target-language values. Statically typed targets are the hard case [10]. Execution happens in a single container with every toolchain, `ghcr.io/nuprl/multipl-e-evaluation`, run **with networking disabled** [10]. The design lesson is that the oracle can be compiled from one language to many, so test quality, good or bad, carries over.

**BigCodeBench** (ICLR 2025) targets practical, library-heavy code: 1,140 software-engineering-oriented tasks in `Complete` (docstring-driven) and `Instruct` (natural-language) splits, plus a 148-task Hard subset [11]. Execution backends are selectable (`e2b`, `gradio` on a Hugging Face Space, or `local`), and the repository ships a decontamination folder and "calibrated" pass@1 outputs [11]. Remote execution backends make function-level evaluation possible without running untrusted code locally. They also add a network dependency that can fail.

**Harness consolidation.** The BigCode evaluation harness runs HumanEval, HumanEval+, MBPP, MBPP+, APPS, DS-1000, MultiPL-E, HumanEvalPack, Recode and others [7]. It separates generation, which needs GPUs (`--generation_only`), from execution, which needs only CPUs (`--load_generations_path`). It also keeps execution off unless `--allow_code_execution` is passed [7]. Splitting the two stages is the right default: generation and execution can run on different machines, and execution can be re-run when the oracle changes.

## 12.3 Live and competitive programming: LiveCodeBench and LiveCodeBench Pro

**Problem.** Static problem sets leak into pretraining. Contest problems carry a publication date, so leakage can be controlled by time.

**LiveCodeBench architecture.** Problems are continuously collected from LeetCode, AtCoder and Codeforces and tagged with their release date [13][14]. Releases grow over time: release_v1 had 400 problems (May 2023 to March 2024) and release_v6 had 1,055 (to April 2025) [14]. There are four scenarios: code generation, self-repair, test-output prediction and code execution (with optional chain of thought) [14]. Scoring uses a modified APPS checker. Generation uses n=10 at temperature 0.2 and reports pass@1 and pass@5. `compute_scores.py` accepts `--start_date` and `--end_date` so a model is scored only on problems published after its training cutoff [14].

```
  contests ----> scrape statement + tests + release_date + difficulty
                     |
       release_vN = all problems with date <= T_N   (v1: 400 ... v6: 1,055)
                     |
   model M, cutoff C ---> score window (C, T_N]   <-- only comparable window
                     |
   "lite" pruned tests (fast) | --not_fast (full tests)  ; timeout sensitivity < 0.5 pt
```

**Evidence.** The LiveCodeBench authors report that some models performed markedly worse on LeetCode problems released after their cutoff dates. They used this as evidence of contamination on older problems, and for DeepSeek models they report results only on problems after August 2023 [13][14].

**Operational details that move scores.** The default `code_generation_lite` set prunes test cases for speed. Time limits cause under 0.5 points of variation in pass@1 and pass@5, and the README advises lowering parallel evaluation processes or raising the timeout when results vary [14]. Only code generation is accepted on the leaderboard [14].

**LiveCodeBench Pro** extends the idea to harder competitive programming, with C++ solutions judged locally by a `LightCPVerifierJudge`. The README recommends no more judge workers than physical CPU cores [15]. The CPU guidance matters because competitive judges enforce wall-clock limits. Oversubscribed graders turn correct, efficient solutions into "time limit exceeded".

**Lesson.** A dated benchmark is only as good as the discipline of reporting the window. "LiveCodeBench 74%" without a date range and release version is underspecified.

## 12.4 Repository-level completion: RepoBench and CrossCodeEval

Code completion in an IDE is mostly about context: the right line depends on definitions in other files. Two benchmarks isolate this.

**RepoBench** (ICLR 2024) evaluates repository-level autocompletion in Python and Java [40]. The ICLR version (v0) defines three tasks, `retrieval`, `completion` and `pipeline`, each in three settings. `cross_file_first` masks the first line that uses a module from another file, `cross_file_random` masks a later cross-file usage, and `in_file` masks a line with no cross-file dependency [40]. v0's test data was code created between February 9 and August 3, 2023, and the README itself warns that this window may overlap with training data [40]. v1.1 (February 2024) refreshed the data, scores Exact Match, Edit Similarity and CodeBLEU per setting, and offers context-length levels from 2k to 16k tokens [40]. The cross-file settings measure whether the model uses the retrieved context, which is the actual job of a completion system.

**CrossCodeEval** (Amazon, NeurIPS 2023 D&B) takes the same approach. It ships "baseline", "retrieval" and "retrieval with reference" variants crossed with three retrievers (BM25, UniXCoder and OpenAI Ada embeddings), uses tree-sitter parsers in its tooling, and computes metrics with a separate evaluation script [41]. Crossing retrievers with models lets a team attribute a completion failure to retrieval or to generation, which is the key diagnostic for a production completion stack.

**Limitation.** Both benchmarks grade by string similarity against the original line, not by execution. They suit completion latency and context-use research. They are weak as correctness evidence, because many correct completions differ textually from the original.

## 12.5 The SWE-bench family

### 12.5.1 SWE-bench: issues to patches, graded by the maintainers' tests

**Problem.** Function-level benchmarks do not exercise navigation, multi-file edits, environment setup or regression avoidance. SWE-bench (Princeton, ICLR 2024 oral) turns resolved GitHub issues into tasks: given the repository at the pre-fix commit and the issue text, produce a patch [16][17].

**Grading semantics.** The tests come from the human fix PR. **FAIL_TO_PASS (F2P)** tests fail before the fix and pass after it. **PASS_TO_PASS (P2P)** tests pass both before and after [16]. The grader's exact rules are worth knowing [19]:

- F2P succeeds only if the test passed or xfailed. Failed, errored, skipped or *missing* tests are failures.
- P2P succeeds if the test passed, xfailed or was **skipped**, since a skip is not a regression.
- Each category's score is successes / (successes + failures). An empty category scores 1.
- **FULL** resolution means F2P = 1 and P2P = 1. **PARTIAL** means 0 < F2P < 1 with P2P = 1. Everything else, including any P2P failure, is **NO**. Only FULL counts as resolved.
- FAIL_TO_FAIL and PASS_TO_FAIL are computed only when `calculate_to_fail` is set and do not affect resolution.
- Per-repository log parsers are chosen from a registry. About 676 truncated pytest IDs in Verified are matched by prefix, but only when all candidates agree on pass versus fail.

**Harness architecture.** Since June 2024 the harness has been fully containerized [17]. Images are layered so expensive dependency installs are shared [18]:

```
  base image  (OS + language toolchain)            shared by all tasks
      |
  environment image (repo-specific deps @ version)  shared by tasks of same repo/version
      |
  instance image (task-specific setup @ base_commit)
      |
  run: git apply model_patch -> apply test_patch -> run tests -> log markers
      |
  grading.py: parse log -> F2P / P2P -> FULL | PARTIAL | NO -> results.json
```

Cache levels trade disk for speed: `none` and `base` need about 120 GB during a run, `env` (the default) about 100 GB, and `instance` about 2,000 GB [18]. The recommended host is x86_64 with at least 120 GB free disk, 16 GB RAM and 8 cores, with workers capped at `min(0.75 * cpu_count, 24)`. arm64 is experimental [17][18]. On SWE-bench Lite, a 16-core machine with 12 workers takes about 30 minutes with the env cache and about 15 minutes with the instance cache [18]. Cloud evaluation is available through Modal (January 2025) and the `sb-cli` submission tool [17].

**Dataset variants** (per the repository's dataset guide) include the full set (2,294 instances), Lite, Verified (500), Multimodal (100 dev and 500 test instances with screenshots and UI elements; a fully open 480-task Multimodal v2 was released in September 2026) and Multilingual (300 instances covering 9 languages and 42 repositories) [17][20].

### 12.5.2 SWE-bench Verified: human filtering, then retirement

OpenAI built Verified in 2024 by having 93 professional Python developers screen 1,699 random SWE-bench samples. Annotators rated how well specified each issue was and how appropriately scoped its tests were, and the 500 samples that passed became Verified (as summarized in Chapter 2) [21]. For about 18 months, Verified was the headline coding number in model launches.

In February 2026 OpenAI stopped reporting it [22]. As summarized in Chapter 2, it gave two reasons. First, in an audit of problems that models often failed, at least 59.4% had flawed tests that rejected functionally correct submissions, either by enforcing implementation details or by testing behavior the issue never described. Second, frontier models could reproduce gold patches and problem details, which is consistent with training exposure [22]. Progress had also slowed: state of the art moved only from 74.9% to 80.9% in six months [22].

Independent work supports the memorization concern. Microsoft researchers asked models to identify the buggy file path from the issue description alone. Accuracy reached up to 76% on SWE-bench Verified but only up to 53% on repositories outside SWE-bench. Verbatim function reproduction reached up to 35% consecutive 5-gram accuracy on SWE-bench Verified and Full, against up to 18% on other similar benchmarks [23]. The authors conclude that gains "may be partially driven by memorization" [23].

**Lesson.** Near the ceiling, a test-based benchmark measures the false-negative rate of its tests as much as the model. Human filtering at construction time does not protect against public tasks entering training corpora afterwards.

### 12.5.3 Reporting practices around Verified: scaffolds, trials, excluded tasks

Lab disclosures show how much configuration sits behind a single Verified number.

- **Minimal scaffolds.** Anthropic's 2024 write-up used a prompt, a Bash tool and a string-replace edit tool. Edits apply only when `old_str` matches exactly one location, and paths must be absolute, a constraint that stopped the model losing track of the working directory [43]. The upgraded Claude 3.5 Sonnet reached 49%, against 45% for the previous state of the art. Many successful runs took hundreds of turns and over 100k tokens. Some failures traced to environment setup or to install patches being applied twice, not to the model [43].
- **Infrastructure exclusions.** For Claude 3.7 Sonnet, only 489 of 500 problems ran on Anthropic's infrastructure. The 11 excluded instance IDs were published, and those problems were counted as failures for leaderboard parity [45].
- **Parallel test-time compute.** The "high compute" configuration samples several attempts, discards patches that break the repository's *visible* regression tests, and ranks the rest with a scoring model, without using hidden tests. For 3.7 Sonnet it reported 70.3% against 63.7% on the same subset [45]; for Sonnet 4.5, 82.0% against 77.2% [46].
- **Trial averaging and prompt changes.** Sonnet 4.5's 77.2% is an average over 10 trials on the full 500 problems, with a prompt addendum encouraging heavy tool use ("ideally more than 100 times"). The 1M-context configuration scored 78.2% but was not reported as primary because it was implicated in inference issues [46]. Opus 4.6's Verified score was averaged over 25 trials, and a prompt-modified variant scored 81.42% [48].
- **Bash-only control.** To separate model from scaffold, the SWE-bench team's `mini-swe-agent` is about 100 lines of Python. It uses only bash, runs each action through `subprocess.run` with no persistent shell, and keeps a linear message history. It scores above 74% on Verified and powers the "bash only" leaderboard [33].

**Lesson.** A Verified number is a tuple: (model, scaffold, prompt, tools, trials, test-time compute policy, excluded tasks, context configuration). A comparison that holds one element fixed and lets the others vary is not a model comparison.

### 12.5.4 SWE-Bench Pro (Scale AI, 2025) and V2 (2026)

**Problem.** Build long-horizon, multi-file tasks that are harder to contaminate.

**Design.** As detailed in Chapter 2, SWE-Bench Pro has 1,865 human-verified tasks drawn from 41 repositories in three partitions. The public set has 731 tasks from 11 copyleft (GPL) repositories. The held-out set has 858 tasks from 12 private repositories, kept back to detect overfitting. The commercial set has 276 tasks from 18 proprietary startup codebases; results on it are published but the data is not [24][25]. Tasks are augmented with requirements and interface specifications so that a correct solution is not rejected for naming a function differently. Top models scored about 23% on the public set at launch, against 70%+ on Verified [25].

**V2.** The repository's V2 release (dated 9/22 in its news log) ships 642 validated public tasks across 11 repositories. Each is a self-contained **Harbor** task directory with a verifier, a reference solution and a public container image (`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`). V2 adds a HARD-51 subset and "locked-protocol" tooling: an offline agent phase followed by re-grading in fresh sandboxes [27]. Scale's write-up reports that 89 of the 731 V1 tasks were invalid and that 69 more had instructions contradicting their tests; these were corrected and re-solved by experts from the instructions alone [26]. The oracle agent resolves 642/642 V2 tasks [27]. Earlier maintenance included removing unit tests that "required the year 2025" [27], which is a reminder that time-dependent tests rot.

**Caveat.** The Hugging Face `default` configuration now returns V2, and V1 lives under `v1` [27]. A score labelled only "SWE-Bench Pro" can refer to either, and results also depend on the scaffold: the repository includes SWE-agent and mini-swe-agent configurations and notes when the two give comparable results [27].

### 12.5.5 Broadening the family: languages, platforms and freshness

| Benchmark | Size and scope | Key design choice |
|---|---|---|
| SWE-bench Multilingual | 300 tasks, 9 languages, 42 repos [20] | SWE-bench grading semantics outside Python |
| Multi-SWE-bench (ByteDance) | 1,632 instances in Java, TS, JS, Go, Rust, C, C++, from 2,456 candidates screened by 68 expert annotators; Mini (400) and Flash (300) subsets [28] | Expert curation; Multi-SWE-RL companion with 4,723 instances for RL |
| SWE-PolyBench (Amazon) | 2,110 issues in Java, JS, TS, Python; PB500 stratified 40/40/20 bug fix/feature/refactor; Verified 382 [29] | File-level and node-level retrieval metrics beyond pass rate |
| SWE-bench-Live (Microsoft) | MultiLang: 1,077 instances, 431 repos, 8 languages; Windows: 66 instances [30] | Monthly additions (50 verified issues per month); frozen lite/verified splits; RepoLaunch agent builds environments |
| SWE-bench Multimodal | 100 dev and 500 test instances; v2 with 480 open tasks [17][20] | Visual issues (screenshots, UI) |

Two architectural ideas stand out. **SWE-PolyBench's retrieval metrics** grade whether the agent found the right files and syntax nodes, not only whether tests passed. That gives a diagnostic signal when the final patch fails [29]. **SWE-bench-Live's RepoLaunch** uses an LLM agent to build a testable container for an arbitrary repository. Its reported efficiency update set up 856 issues from 93 repositories with at least 98% success, with 82% lower LLM API cost and 78% less Docker storage [30]. Automated environment construction is what makes a continuously refreshed benchmark affordable, and it is the same capability that limits how quickly new SWE-style benchmarks can be built at all.

### 12.5.6 The eval/RL convergence: SWE-Gym and SWE-smith

The same artifact, a container with a repository, a task and a verifier, now serves as both an evaluation item and a reinforcement-learning environment.

- **SWE-Gym** (ICML 2025) provides about 2.4k real tasks from 11 Python repositories with pre-built per-instance images. Fine-tuning on fewer than 500 trajectories gave a 32B OpenHands agent a 14-point absolute gain on Verified. Verifiers trained on trajectories for best-of-n selection reached 32% on Verified [32].
- **SWE-smith** (NeurIPS 2025 D&B) synthesizes about 52k task instances across 250+ repositories. It builds one environment per repository, synthesizes candidate tasks, keeps only those that break at least one unit test, and generates issue text. SWE-agent-LM-32B, trained on this data, reaches 40.2% pass@1 on Verified [31].

**Implication for evaluators.** Once training environments are built from the same repositories, the same task-construction pipeline and the same grading semantics as the benchmark, "contamination" includes **distributional overlap**, not just verbatim leakage. Held-out repositories, as in SWE-Bench Pro's design, are the structural answer.

### 12.5.7 Beyond the patch: SWE-Lancer and Terminal-Bench 2.0

**SWE-Lancer** (OpenAI) grounds tasks in real freelance software work. Its original paper had 237 problems, and the public repository now ships 198 tasks "adjusted and verified to run successfully offline" [34]. Tasks are individual-contributor implementation tasks (`ic_swe`) or manager tasks (`swe_manager`), in which the model chooses between proposals. Per-task images take 10-20 minutes to build and are about 14 GB each, and evaluations run with internet disabled via `iptables` on Linux [34].

**Terminal-Bench 2.0** packages each task as an instruction, a test script and an oracle solution in a Docker-sandboxed terminal. By this chapter's count, the repository has 89 task directories. Each task "has received several hours of human and LM-assisted validation" for solvability, realism and specification [35]. The official runner, **Harbor**, runs the oracle, reference agents and products such as Claude Code with configurable concurrency [35]. It also generates RL rollouts and provides adapters for other benchmarks [56], and it is the runner for SWE-Bench Pro V2 tasks [27].

## 12.6 Aider polyglot: an editing benchmark with feedback

**Problem.** Aider's earlier Python benchmark of 133 Exercism problems saturated. Top scores passed 80% (84.2% meant solving 112 of 133), record-setters gained only one or two problems, and models as old as GPT-3.5 Turbo solved half the problems [36]. Aider also needed to measure something other benchmarks ignore: whether a model can express edits in a **format a tool can apply** to files on disk.

**Selection by difficulty.** Seven models (Sonnet, Haiku, o1-mini, DeepSeek, GPT-4o, Qwen 32B Coder Instruct and GPT-4o mini) attempted all 697 Exercism problems in C++, Go, Java, JavaScript, Python and Rust. The 258 problems solved by all seven were discarded, and only problems solved by three or fewer were kept: 66 solved by none, 61 by one, 50 by two and 48 by three [36]. That gives **225 problems**: C++ 26, Go 39, Java 47, JavaScript 49, Python 34 and Rust 30 [36]. At launch, o1 with high reasoning effort led with 62% [36].

**Protocol.** Reading the benchmark code shows the loop [38]:

```
  for each exercise (random order, in a Docker container):
    files shown   = solution stubs only  (tests, examples, .meta hidden)
    instructions  = .docs/introduction.md + instructions.md + instructions.append.md
                    + "Don't change the names of existing functions or classes ...
                       Only use standard libraries ..."
    for try in 1..tries (default 2):
        model edits files via edit format (whole | diff | diff-fenced | architect)
        copy original tests in; run per-language command, 180 s timeout
          (.py pytest | .rs cargo test | .go go test | .js npm | .cpp | .java gradlew)
        pass -> stop
        fail -> next instructions = test error output +
                "The tests are correct, don't try and change them. Fix the code ..."
    report pass_rate_1, pass_rate_2, percent_cases_well_formed, cost, timeouts ...
```

Tests are hidden on the first attempt and their failure output is shown on the second, so `pass_rate_2` measures **repair from test feedback**. Test files are copied in fresh from the original directory before every run, so a model cannot "pass" by editing the tests [38]. Beyond pass rates, the report counts malformed responses, "lazy comments" (lines like `# ... rest of code`), syntax and indentation errors, exhausted context windows, test timeouts, seconds per case and total cost. It also records the aider commit hash, flagged "(dirty)" if the tree has uncommitted changes, for reproducibility [37].

**Leaderboard snapshot.** The leaderboard data file held 69 entries when fetched. The highest `pass_rate_2` was gpt-5 (high) at 88.0%, with 52.0% on the first try, 91.6% well-formed edits and a total cost of about $29 (August 2025). o3-pro (high) scored 84.9% at about $146, and DeepSeek-V3.2-Exp (Reasoner) scored 74.2% at about $1.30 (October 2025) [39]. The roughly 36-point gap between first and second try for the leader shows how much of "coding ability" on this benchmark is test-driven repair. The 100x cost range for similar scores shows why cost belongs on the leaderboard.

**Lessons.** (1) Choosing problems by difficulty against a panel of models buys headroom, but it ties the benchmark to that panel's weaknesses. (2) Edit-format conformance is a real, separately measurable failure mode for tool-integrated coding. (3) Hiding tests and then feeding back failures is a cheap way to measure the inner loop of agentic coding without a full agent scaffold.

## 12.7 Execution-based grading infrastructure

### 12.7.1 Isolation levels

```
  weakest                                                                   strongest
  in-process guard   child process +    container,          container per    full VM     
  (HumanEval         rlimits/timeouts   network none        sample + infra   (EC2,Proxmox,
  reliability_guard) (APPS-style)       (MultiPL-E)         orchestration    Vagrant)
                                                            (SWE-bench,
                                                            Inspect, Harbor)
```

Public harnesses span this range. HumanEval's guard is explicitly not a security boundary [4]. MultiPL-E runs everything in one multi-toolchain container with networking disabled [10]. BigCode and HumanEval disable execution by default [2][7]. UK AISI's Inspect binds sandboxes per sample, task or eval. It supports Docker, Kubernetes, Daytona, Modal, EC2, Proxmox, Vagrant and NVIDIA OpenShell providers, generates compose configs with `network_mode: none`, and truncates `exec()` output by default at 10 MB, sometimes silently [58]. Two Inspect caveats generalize. Network isolation applies only inside the container, so tools and scorers running on the host can still reach the internet. And a custom compose file *replaces* the generated one, so it silently drops `network_mode: none` unless the author adds it back [58].

### 12.7.2 Infrastructure as a hidden variable

Anthropic quantified what many evaluators suspected [42]. On Terminal-Bench 2.0, run on Google Kubernetes Engine, the same model, harness and tasks were run under six resource configurations:

| Resource configuration | Infra error rate | Score effect |
|---|---|---|
| 1x (floor = hard ceiling) | 5.8% | baseline |
| 3x ceiling | 2.1% | within noise (p = 0.40) |
| uncapped | 0.5% | +6 pp vs 1x (p < 0.01) |

Up to 6% of tasks failed from pod errors unrelated to the task. Under tight memory, simply installing the Python data-science stack could cause an out-of-memory kill before any solution code was written. On SWE-bench (227 problems, 10 samples each), scores rose monotonically with RAM but were only 1.54 pp higher at 5x than at 1x [42]. The recommendations are concrete. Specify a guaranteed allocation and a separate kill threshold. Calibrate the gap so that scores at floor and ceiling fall within noise of each other. Report the multiplier. Run on multiple days. Treat leaderboard gaps under about 3 pp with skepticism until configurations match [42]. Anthropic's later launch pages apply this: Opus 4.6's Terminal-Bench 2.0 runs used a 1x guaranteed / 3x ceiling allocation and 5-15 samples per task in staggered batches [48]. The Opus 4.5 page reports that improving the hosting environment raised Gemini 3 and GPT-5.1 above the values their own developers had reported [47].

### 12.7.3 A failure taxonomy for execution graders

| Failure | Example | Mitigation |
|---|---|---|
| Resource starvation fails correct code | HumanEval malloc failures [2]; OOM during pip install [42] | Floor/ceiling allocations; separate infra-error verdict |
| Timeouts tied to host load | LiveCodeBench timeouts (<0.5 pt) [14]; judge workers above physical cores [15] | Fixed CPU budgets; re-run timeouts; report timeout rate |
| Empty or missing test output counted as pass | Guarded by SWE-bench `SUITE_RAN` and exit-code checks [19] | Positive evidence that tests ran |
| Patch-apply drift | `git apply` failures; Multi-SWE-bench `patch --batch` fallback [28] | Normalize patches; log apply failures separately |
| Environment rot | Tests requiring "the year 2025" [27]; unsolvable tasks on internal infra [45] | Oracle runs on every harness change; pin dates and dependencies |
| Shared state across trials | Agent reads git history from earlier trials [44] | Fresh environment per trial |
| Grader rejects valid answers | Over-specified tests in Verified [22]; rigid numeric matching in CORE-Bench [44] | Instruction-only expert re-solves [26]; read transcripts |
| Silent output truncation | Inspect `exec()` limits [58] | Write large outputs to files |

The most valuable single practice is the **oracle run**: execute the reference solution through the full harness after every harness or image change and require 100%, as SWE-Bench Pro V2 does (642/642) and as Terminal-Bench's `--agent oracle` mode supports [27][35]. A task that the oracle cannot pass measures nothing.

## 12.8 Contamination in code

Code is unusually exposed to contamination. Benchmarks are built from public GitHub repositories and contest sites that pretraining crawls ingest, gold patches are public commits, and issue threads often discuss the fix. Contamination also takes code-specific forms:

1. **Verbatim leakage** of problems and solutions: the HumanEval and MBPP sets, contest editorials, merged PRs.
2. **Gold-patch recall**: reproducing the exact fix from the issue alone, as found for Verified [22][23].
3. **Repository familiarity**: knowing which file to edit without reading the code (76% versus 53% file-path accuracy in [23]).
4. **Distributional overlap from training environments** built from the same repositories and task pipelines (Section 12.5.6) [31][32].
5. **In-run leakage**: the agent finds the answer during evaluation, for example in git history from earlier trials [44], or, in a non-code benchmark Anthropic reported, by recognizing the benchmark and decrypting its answer key [57].

| Defense | Used by | What it protects against | Residual risk |
|---|---|---|---|
| Release-date windows | LiveCodeBench, LiveCodeBench Pro [13][15] | Verbatim leakage of new problems | Cutoffs are self-reported; windows shrink comparability |
| Continuous refresh | SWE-bench-Live (50/month) [30] | Pretraining exposure to new tasks | Live sets are not frozen; frozen splits age |
| Licensing barrier | SWE-Bench Pro public set (GPL repos) [25] | Corpora that exclude copyleft code | Depends on trainers' filtering choices |
| Private or held-out data | SWE-Bench Pro held-out and commercial sets [24][25] | All public leakage | Trust in the operator; less reproducible |
| Memorization probes | File-path and 5-gram probes [23]; lab screening reported in Chapter 2 | Detecting recall | Probes need a non-benchmark control |
| Fresh sandbox per trial | Anthropic guidance [44]; SWE-Bench Pro V2 re-grading [27] | In-run leakage | Must also scrub `.git` and caches |
| Stronger oracles | EvalPlus [9] | Memorized but brittle solutions passing weak tests | Does not fix leakage itself |

**Design rule.** Treat any public code benchmark that is more than one training cycle old as a *regression* test, not a *capability* test. Capability claims should come from dated, held-out or private items, scored with a stated window.

## 12.9 Real-world code-assistant evaluation

Benchmarks answer "can the model do the task?". Product teams also have to answer "does this model, prompt and context pipeline, in this UI and at this latency, make developers more effective?". Public documentation of how products answer that is uneven. This section covers what is documented in primary sources.

### 12.9.1 GitHub Copilot: offline simulation, online telemetry and A/B tests

**Offline: the Copilot Chat simulation harness.** The VS Code Copilot Chat extension, open source and folded into the main VS Code repository in May 2026, documents three test tiers. Unit tests run in Node, integration tests run inside VS Code, and **simulation tests ("stest")** call real model endpoints [51]. Simulation tests are costly, so the harness is built around three mechanisms [51]:

- **Repetition.** Each scenario runs 10 times "to accommodate for the stochastic nature of LLMs".
- **A committed baseline.** Results are snapshotted in `test/simulation/baseline.json`. A PR fails if it changes the baseline without committing the change, so a quality shift becomes a reviewed diff.
- **A layered response cache.** LLM responses are cached in `test/simulation/cache`, which makes re-runs fast and deterministic. Only VS Code team members can populate the cache, and community PRs that add cache layers fail.

The runner's options show its scope [51]. There are separate chat, "smart" and "fast" models, an `nRuns` setting (the baseline is used only at the baseline run count), external scenario directories with their own baselines, next-edit-suggestion (NES) test suites, a `swebenchPrompt` flag, and a CSV scorecard per run. The design is effectively **snapshot testing for stochastic systems**: repeated sampling to estimate the pass rate, a version-controlled baseline as the regression oracle, and caching so the test suite stays affordable in CI.

GitHub has also written about running offline evaluations before any model or prompt change reaches production [55]. Chapter 1 notes that the details beyond this come from secondary write-ups.

**Online: acceptance and persistence.** Ziegler et al. compared Copilot telemetry with developers' self-reported productivity. Their widely cited finding is that the acceptance rate of shown suggestions predicted perceived productivity better than how long accepted code persisted (as summarized in Chapter 8) [52]. The CUPS study (CHI 2024) had 21 programmers label their own Copilot sessions with a taxonomy of activities. It found inefficiencies and time costs, including time spent verifying suggestions, that "motivate new interface designs and metrics" [53]. Acceptance is cheap and real-time but blind to verification cost.

**Online: experimentation.** The Visual Studio C++ team describes offline evaluation on clean datasets, progressive rollouts of "a few thousand samples" to catch harmful output, sample-ratio mismatch and latency regressions, then A/B tests and qualitative studies [54]. Their Quick Info case found that frequently invoked, in-flow features benefit from faster models, while features that interrupt the flow may favor accuracy (Chapter 8) [54]. Model choice depends on the feature.

### 12.9.2 Copilot Arena: in-the-wild pairwise preference

Copilot Arena is a VS Code extension that shows two completions from different models, one above the other. Tab accepts the top one, Shift-Tab the bottom, and continuing to type accepts neither. There is also an inline-edit mode that shows two diffs [50]. Fairness controls include randomized position and pairing, simultaneous display so the faster model waits, matched generation parameters, and prompting chat models to re-emit prefix and suffix so their output can be post-processed into fill-in-the-middle completions (Chapter 5) [49]. By November 2024 it had logged over 10K battles from 833 voters. Mean context was 1,002 tokens, far longer than HumanEval prompts. 82% of accepted completions were the top one, a strong position bias that the randomization has to absorb (Chapter 5) [49]. The README notes that individuals have different model preferences and offers a personal leaderboard after 20 votes [50].

**Design lesson.** In-IDE preference is a distinct construct from benchmark correctness. It includes latency, length and style, and it is heavily position-biased. It is valuable precisely because benchmarks cannot see those factors, but it needs randomization and bias modeling, as with every arena (Chapter 5).

### 12.9.3 Cursor and other agentic IDEs: what can and cannot be said

Agentic IDEs such as Cursor evaluate models for agent mode, tab completion and next-edit prediction, and they publicly cite results on the benchmarks in this chapter. This chapter could not retrieve Cursor's own engineering pages and so makes no claims about its internal benchmarks or online metrics. The documented building blocks a product team in that position would combine are these:

```
  OFFLINE (pre-merge / pre-launch)                       ONLINE (post-launch)
  +------------------------------------+                 +---------------------------+
  | public: SWE-bench Pro / Verified   |                 | acceptance & persistence  |
  |   (bash-only control), TB 2.0,     |  ship gate      |   of completions [52]     |
  |   Aider polyglot (edit format)     |---------------->| A/B tests, SRM checks,    |
  | private: scenario suite from own   |                 |   latency/crash guards [54]|
  |   product traces, run N times,     |                 | pairwise in-IDE prefs [49]|
  |   committed baseline, cached LLM   |<----------------| failure mining -> new     |
  |   responses [51]                   |  new scenarios  |   offline scenarios       |
  | infra: oracle runs, resource caps  |                 | qualitative / CUPS-style  |
  |   documented [42]                  |                 |   session studies [53]    |
  +------------------------------------+                 +---------------------------+
```

The Copilot Chat harness supplies the offline half and the Visual Studio and Copilot telemetry work supplies the online half. Anthropic's guidance supplies the loop: start with 20-50 tasks drawn from real failures, isolate trials, read transcripts, and graduate saturated capability evals into regression suites [44].

## 12.10 Comparison table

| Benchmark / system | Unit of work | Grader | Size (public) | Contamination control | Main failure mode |
|---|---|---|---|---|---|
| HumanEval [1][2] | Python function | Hidden asserts, pass@k | 164 | None (2021, public) | Weak tests; saturated |
| MBPP [5] | Python function | 3 asserts | 974 (500 test; 427 sanitized) | None | Prompt includes a test; harness variance |
| APPS [6][8] | Stdin/stdout program | I/O tests; strict / avg accuracy | ~10k (5k test) | None | Average accuracy inflates |
| EvalPlus [9] | Function | 80x / 35x augmented tests | 164 / 378 | Stronger oracle | Ground-truth errors (fixed) |
| MultiPL-E [10] | Function, 18+ languages | Translated tests | HumanEval/MBPP translations | None | Translation fidelity |
| BigCodeBench [11] | Library-heavy function | Unit tests | 1,140 (Hard 148) | Decontamination tooling | Remote execution dependency |
| LiveCodeBench [13][14] | Contest problem | Hidden I/O tests | 1,055 (v6) | Release-date window | Window misreporting; timeouts |
| RepoBench / CCEval [40][41] | Next line in repo | EM / ES / CodeBLEU | Python, Java (+) | Dated snapshots | Not execution-based |
| SWE-bench (full) [16][20] | Repo patch | F2P + P2P | 2,294 | None | Infra cost; weak specs |
| SWE-bench Verified [21][22] | Repo patch | F2P + P2P | 500 | Human filter only | Flawed tests; memorization; retired |
| SWE-Bench Pro V2 [24][27] | Long-horizon patch | Harbor verifier, locked protocol | 642 public (+ held-out, commercial) | GPL, private sets | Version confusion (V1 / V2) |
| Multi-SWE-bench [28] | Repo patch, 7 languages | Docker harness | 1,632 | Expert curation | Patch-apply variance |
| SWE-bench-Live [30] | Repo patch, multi-OS | Auto-built env | 1,077 MultiLang + 66 Windows | Monthly refresh | Env-build reliability |
| Aider polyglot [36][38] | Exercism edit, 6 languages | Unit tests, 2 tries | 225 | Hardness-selected | Edit format; Exercism public |
| Terminal-Bench 2.0 [35] | Terminal task | Test script + oracle | 89 | Manual validation | Resource sensitivity |
| SWE-Lancer [34] | Freelance task | E2E tests / manager choice | 198 runnable | Offline, no internet | Heavy images (~14 GB) |
| Copilot simulation tests [51] | Product scenario | Scenario checks x10, baseline | Private | Private | Cache staleness |
| Copilot Arena [49][50] | In-IDE completion | Pairwise acceptance | 10K+ battles (2024) | Live user code | Position bias |

## 12.11 A reference architecture for a code-evaluation platform

```
                          +-----------------------------+
   task sources --------->| TASK REGISTRY (versioned)   |  task = {spec, repo@sha,
   (public, dated, private,| oracle solution, verifier,  |   image digest, verifier,
    product traces)       | release_date, license tag   |   oracle, created_at}
                          +--------------+--------------+
                                         |
         +-------------------------------v-------------------------------+
         | ORACLE GATE: run reference through full harness -> must = 100% |
         +-------------------------------+-------------------------------+
                                         |
   model x scaffold x budget --->+-------v--------+     +----------------------------+
   (pinned prompt, tools,        | ROLLOUT POOL   |---->| SANDBOX FLEET (per trial)   |
    trials N, thinking budget)   | (agent phase,  |     | network none by default,    |
                                 |  offline)      |     | floor/ceiling resources,    |
                                 +-------+--------+     | fresh fs, no .git history   |
                                         |              +----------------------------+
                                         v
                          +-----------------------------+
                          | RE-GRADE IN FRESH SANDBOX   |  verdicts: pass | fail |
                          | (locked protocol)           |  timeout | apply-fail |
                          +--------------+--------------+  infra-error (excluded+logged)
                                         |
          +------------------------------v-------------------------------+
          | ANALYSIS: pass@1, pass^k, CI over tasks AND trials, cost, tokens,|
          | time window, per-language/repo slices, memorization probe, diffs |
          | vs committed baseline; transcripts sampled for human review      |
          +-----------------------------------------------------------------+
```

Each box corresponds to a documented practice. The versioned registry with image digests follows SWE-Bench Pro V2's Harbor tasks [27]. The oracle gate follows SWE-Bench Pro V2 and Terminal-Bench [27][35]. Separate agent and re-grading phases follow V2's locked protocol [27]. Resource floors and ceilings follow [42]. Isolation and network defaults follow Inspect [58]. Baselines and repeated runs follow the Copilot Chat harness [51]. Transcript review follows [44].

## Key takeaways for system designers

1. **Specify the whole tuple.** Report model, scaffold, tools, prompt addenda, trials, test-time-compute policy, excluded tasks, benchmark version and resource envelope. Lab disclosures show each of these moving Verified scores by several points [45][46][48].
2. **Make the verdict multi-valued.** Separate wrong, timeout, apply-failure and infra-error. Require positive evidence that tests ran, and never count empty output as a pass [19].
3. **Gate on oracle runs.** A harness change that stops the reference solution from passing is a benchmark bug, not a model regression [27][35].
4. **Treat resources as an experimental variable.** Use a guaranteed floor with a separate ceiling, report the multiplier, and distrust gaps under about 3 pp between differently configured runs [42].
5. **Audit the oracle as hard as the model.** EvalPlus's 80x tests, MBPP+'s task removals, Verified's flawed-test rate and Pro's 89 invalid tasks all show that the test suite is a second model with its own error rate [9][22][26].
6. **Assume public code benchmarks are contaminated after one training cycle.** Use dated windows, held-out or private repositories and memorization probes for capability claims, and keep the old sets as regression tests [13][23][25].
7. **Use pass^k for reliability claims.** Products need consistency; pass@k with a verifier is a different, legitimate claim about best-of-n systems [44].
8. **Measure editing and repair, not just generation.** Aider's two-try protocol and well-formedness rate measure the inner loop of real tool-integrated coding [37][38][39].
9. **Snapshot-test the product.** Repeated runs, a committed baseline and cached responses turn stochastic product quality into a reviewable diff in CI [51].
10. **Close the loop with users.** Acceptance, persistence, A/B tests and session studies measure what benchmarks cannot see: latency, verification cost and flow. Each has known biases that need correcting [49][52][53][54].

## References

1. Chen, M., et al. (OpenAI). "Evaluating Large Language Models Trained on Code." arXiv, 2021. https://arxiv.org/abs/2107.03374
2. OpenAI. "human-eval" repository (README). GitHub, 2021. https://github.com/openai/human-eval
3. OpenAI. "human_eval/evaluation.py" (pass@k estimator and defaults). GitHub, 2021. https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
4. OpenAI. "human_eval/execution.py" (execution guard and timeouts). GitHub, 2021. https://github.com/openai/human-eval/blob/master/human_eval/execution.py
5. Austin, J., et al. (Google Research). "MBPP: Mostly Basic Python Problems" dataset README ("Program Synthesis with Large Language Models," 2021). GitHub. https://github.com/google-research/google-research/tree/master/mbpp
6. Hendrycks, D., Basart, S., Kadavath, S., et al. "Measuring Coding Challenge Competence With APPS." NeurIPS 2021; repository. https://github.com/hendrycks/apps
7. BigCode Project. "bigcode-evaluation-harness" repository README. GitHub, 2022-2025. https://github.com/bigcode-project/bigcode-evaluation-harness
8. BigCode Project. "bigcode-evaluation-harness: task documentation" (APPS, HumanEval, MBPP, DS-1000 settings). GitHub. https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/docs/README.md
9. Liu, J., Xia, C. S., Wang, Y., Zhang, L. "Is Your Code Generated by ChatGPT Really Correct?" (EvalPlus), NeurIPS 2023; repository and release notes. https://github.com/evalplus/evalplus
10. Northeastern University Programming Research Lab (nuprl). "MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation"; repository. https://github.com/nuprl/MultiPL-E
11. BigCode Project. "BigCodeBench" (ICLR 2025; arXiv:2406.15877); repository. https://github.com/bigcode-project/bigcodebench
12. Li, Y., et al. (DeepMind). "Competition-level code generation with AlphaCode." Science, 2022; CodeContests repository. https://github.com/google-deepmind/code_contests
13. Jain, N., et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." ICLR 2025 (arXiv 2024). https://arxiv.org/abs/2403.07974
14. LiveCodeBench team. Repository README (release versions, scenarios, date filtering, timeouts). GitHub, 2024-2025. https://github.com/LiveCodeBench/LiveCodeBench
15. LiveCodeBench Pro team. "LiveCodeBench-Pro" toolkit repository. GitHub, 2025. https://github.com/GavinZhengOI/LiveCodeBench-Pro
16. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024. https://arxiv.org/abs/2310.06770
17. SWE-bench team. "SWE-bench" repository README (news, Docker harness, resources). GitHub, 2023-2026. https://github.com/SWE-bench/SWE-bench
18. SWE-bench team. "Harness reference" (image layers, cache levels, timing). GitHub. https://github.com/SWE-bench/SWE-bench/blob/main/docs/reference/harness.md
19. SWE-bench team. "swebench/harness/grading.py." GitHub, 2024-2026. https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py
20. SWE-bench team. "Datasets guide" (full, Lite, Verified, Multimodal, Multilingual). GitHub. https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md
21. OpenAI. "Introducing SWE-bench Verified." OpenAI, August 2024. https://openai.com/index/introducing-swe-bench-verified/
22. OpenAI. "Why SWE-bench Verified no longer measures frontier coding capabilities." OpenAI, February 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
23. Liang, S., Garg, S., Zilouchian Moghaddam, R. (Microsoft). "The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason." 2025. https://www.microsoft.com/en-us/research/publication/the-swe-bench-illusion-when-state-of-the-art-llms-remember-instead-of-reason/
24. Deng, X., et al. (Scale AI). "SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?" arXiv, 2025. https://arxiv.org/pdf/2509.16941
25. Scale AI. "SWE-Bench Pro" (launch blog post). Scale AI, 2025. https://scale.com/blog/swe-bench-pro
26. Scale AI. "SWE-Bench Pro V2: A Cleaner, Harder-to-Game Leaderboard." Scale Labs, 2026. https://labs.scale.com/blog/swe-bench-pro-v2
27. Scale AI. "SWE-bench_Pro-os" repository README (V2, HARD-51, Harbor tasks, locked protocol). GitHub, 2025-2026. https://github.com/scaleapi/SWE-bench_Pro-os
28. Zan, D., et al. (ByteDance). "Multi-SWE-bench" (NeurIPS 2025 D&B; arXiv:2504.02605); repository. https://github.com/multi-swe-bench/multi-swe-bench
29. Rashid, M. S., et al. (Amazon). "SWE-PolyBench: A multi-language benchmark for repository level evaluation of coding agents" (arXiv:2504.08703); repository. https://github.com/amazon-science/SWE-PolyBench
30. Microsoft. "SWE-bench-Live" (NeurIPS 2025 D&B; RepoLaunch); repository. https://github.com/microsoft/SWE-bench-Live
31. Yang, J., Lieret, K., Jimenez, C. E., et al. "SWE-smith: Scaling Data for Software Engineering Agents." NeurIPS 2025 D&B; repository. https://github.com/SWE-bench/SWE-smith
32. Pan, J., Wang, X., Neubig, G., Jaitly, N., Ji, H., Suhr, A., Zhang, Y. "Training Software Engineering Agents and Verifiers with SWE-Gym." ICML 2025; repository. https://github.com/SWE-Gym/SWE-Gym
33. SWE-agent team. "mini-swe-agent" repository. GitHub, 2025-2026. https://github.com/SWE-agent/mini-swe-agent
34. OpenAI. "SWE-Lancer" (frontier-evals repository). GitHub, 2025-2026. https://github.com/openai/frontier-evals/tree/main/project/swelancer
35. Laude Institute. "terminal-bench-2" repository README and FAQ. GitHub, 2025-2026. https://github.com/laude-institute/terminal-bench-2
36. Aider (P. Gauthier). "Aider polyglot benchmark" announcement post, December 2024. https://aider.chat/2024/12/21/polyglot.html ; source: https://github.com/Aider-AI/aider/blob/main/aider/website/_posts/2024-12-21-polyglot.md
37. Aider. "Benchmark harness README." GitHub, 2024-2026. https://github.com/Aider-AI/aider/blob/main/benchmark/README.md
38. Aider. "benchmark/benchmark.py" and "benchmark/prompts.py" (retry loop, test commands, prompts). GitHub. https://github.com/Aider-AI/aider/blob/main/benchmark/benchmark.py
39. Aider. "Polyglot leaderboard data" (polyglot_leaderboard.yml). GitHub, accessed October 2026. https://github.com/Aider-AI/aider/blob/main/aider/website/_data/polyglot_leaderboard.yml
40. Liu, T., Xu, C., McAuley, J. "RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems." ICLR 2024; repository (main and archive/v0). https://github.com/Leolty/repobench
41. Ding, Y., Wang, Z., Ahmad, W. U., et al. (Amazon). "CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion." NeurIPS 2023 D&B; repository. https://github.com/amazon-science/cceval
42. Segato, G. (Anthropic). "Quantifying infrastructure noise in agentic coding evals." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/infrastructure-noise
43. Anthropic. "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/swe-bench-sonnet
44. Anthropic. "Demystifying evals for AI agents." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
45. Anthropic. "Claude 3.7 Sonnet and Claude Code" (SWE-bench methodology notes). Anthropic, 2025. https://www.anthropic.com/news/claude-3-7-sonnet
46. Anthropic. "Introducing Claude Sonnet 4.5" (methodology notes). Anthropic, 2025. https://www.anthropic.com/news/claude-sonnet-4-5
47. Anthropic. "Introducing Claude Opus 4.5" (methodology notes). Anthropic, 2025. https://www.anthropic.com/news/claude-opus-4-5
48. Anthropic. "Introducing Claude Opus 4.6" (methodology notes). Anthropic, 2026. https://www.anthropic.com/news/claude-opus-4-6
49. Chi, W., Chen, V., Angelopoulos, A. N., Chiang, W.-L., Mittal, A., Jain, N., Zhang, T., Stoica, I., Donahue, C., Talwalkar, A. "Copilot Arena." LMArena Blog, 2024. https://blog.lmarena.ai/blog/2024/copilot-arena/
50. LMArena. "copilot-arena" repository README (paper: arXiv:2502.09328). GitHub, 2024-2025. https://github.com/lmarena/copilot-arena
51. Microsoft. "vscode-copilot-chat" CONTRIBUTING (simulation tests, baseline, cache) and test/simulationMain.ts. GitHub, 2025-2026. https://github.com/microsoft/vscode-copilot-chat/blob/main/CONTRIBUTING.md ; https://github.com/microsoft/vscode-copilot-chat/blob/main/test/simulationMain.ts
52. Ziegler, A., et al. (GitHub). "Productivity Assessment of Neural Code Completion." MAPS 2022. https://arxiv.org/abs/2205.06537
53. Mozannar, H., Bansal, G., Fourney, A., Horvitz, E. (Microsoft). "Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming." CHI 2024. https://www.microsoft.com/en-us/research/publication/reading-between-the-lines-modeling-user-behavior-and-costs-in-ai-assisted-programming/
54. Akinci, S., Chiu, C. (Microsoft). "Experimentation in Generative AI: C++ Team's Practices for Continuous Improvement." 2024. https://www.microsoft.com/en-us/research/articles/experimentation-in-genai-c-teams-practices-for-continuous-improvement/
55. Adams, C. (GitHub). "How we evaluate AI models and LLMs for GitHub Copilot." GitHub Blog, 2025. https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/
56. Laude Institute. "Harbor: a framework for evaluating and optimizing agents." GitHub, 2025-2026. https://github.com/laude-institute/harbor
57. Anthropic. "Eval awareness in Claude Opus 4.6's BrowseComp performance." Anthropic Engineering, 2026. https://www.anthropic.com/engineering/eval-awareness-browsecomp
58. UK AI Security Institute. "Inspect documentation: Sandboxing." GitHub. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/docs/sandboxing.qmd
