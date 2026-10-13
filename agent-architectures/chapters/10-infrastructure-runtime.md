# 10. Agent Infrastructure and Runtime: Sandboxes, Durable Execution, State, Observability, Deployment, and Cost

## Overview

An agent is a loop: call a model, run whatever tool the model asked for, append the result, repeat. In a notebook this is about twenty lines of code. In production the same loop has to survive things a notebook never sees. It runs code nobody reviewed, often written by the model in response to text an attacker may control. It runs for minutes or hours, across process restarts and deploys. It holds state that must outlive any one container. It makes dozens to hundreds of paid model calls per task. And when it fails, someone has to work out which of those calls went wrong.

Anthropic's production notes put numbers on the pressure. Agents typically use about 4x the tokens of a chat interaction, and multi-agent systems about 15x [42]. Because errors compound across a long run, "minor system failures can be catastrophic for agents" [42]. A deploy can land while thousands of agents are mid-task, so a fleet-wide restart is not an option [42].

This chapter covers the infrastructure layer underneath the agent loop. It is organized around six concerns:

1. **Isolation (Section 1).** Where untrusted, model-generated code runs: microVMs (Firecracker), application kernels (gVisor), OS-level process sandboxes (bubblewrap, Seatbelt), and the hosted sandbox products built on them (E2B, Daytona, Modal, Vercel, Cloudflare).
2. **Durable execution (Section 2).** How a long-running loop survives crashes without repeating side effects or paying twice for model calls: Temporal, Restate, Inngest, and DBOS.
3. **Agent state and checkpointing (Section 3).** Framework checkpointers (LangGraph), append-only session logs (Anthropic Managed Agents), per-agent actors (Cloudflare Durable Objects), and filesystem snapshots.
4. **Observability (Section 4).** Trace models, the OpenTelemetry GenAI conventions, and the ingestion architectures of LangSmith, Langfuse, AgentOps, Phoenix, and Helicone.
5. **Deployment and scaling (Section 5).** Separating the harness from the sandbox, rolling out new versions while agents are mid-task, and placing sandboxes on nodes.
6. **Cost control (Section 6).** Prompt caching, model routing and cascades, gateways with spend tracking, and token-efficient tool design.

Context-window management (compaction, memory, session logs as context stores) is covered in Chapter 05. Coding-agent sandboxes from the product side are covered in Chapter 06. This chapter is about the machinery.

Note on sourcing: during research, several vendor documentation hosts (modal.com, e2b.dev, daytona.io, temporal.io, docs.temporal.io, restate.dev, inngest.com, docs.langchain.com, gvisor.dev, arxiv.org, usenix.org) could not be reached from the research environment. Claims about those systems come from their public GitHub repositories, which hold the source and, in most cases, architecture documents. Coverage of Modal is limited for this reason. Numbers taken from vendor READMEs are vendor claims, not independent benchmarks, and are labelled as such.

---

## 1. Isolation: where model-generated code runs

### 1.1 The problem

An agent with a code-execution or shell tool runs whatever the model emits. That code can be wrong (an `rm -rf` in the wrong directory), resource-hungry (a fork bomb, a 50 GB download), or adversarial (prompt injection from a web page that tells the agent to exfiltrate credentials). gVisor's documentation states the core problem plainly: containers alone are not a sandbox [8]. With plain Linux primitives such as seccomp, enforcement still happens inside the monolithic host kernel, so a workload is "only one system call away from host compromise" [9].

Anthropic's work on Claude Code sandboxing names the two boundaries that matter [19]:

- **Filesystem isolation.** Without it, a compromised process can read SSH keys and other secrets.
- **Network isolation.** Without it, the process can send whatever it reads to an attacker.

Either boundary alone is insufficient. Anthropic's internal usage found that sandboxing cut permission prompts by 84%, because the agent can act freely inside a boundary that is safe by construction [19].

The design space has three main tiers, ordered by isolation strength and cost:

```
  weaker / cheaper                                             stronger / costlier
  +--------------------+   +-------------------------+   +--------------------------+
  | OS process sandbox |   | Application kernel      |   | MicroVM                  |
  | bubblewrap, Seat-  |   | gVisor (Sentry + Gofer) |   | Firecracker, Cloud Hyp., |
  | belt, seccomp,     |   | syscalls served in a    |   | Kata; guest kernel,      |
  | namespaces         |   | userspace Go kernel     |   | KVM hardware boundary    |
  +--------------------+   +-------------------------+   +--------------------------+
   shares host kernel       two kernels, no shared code   two kernels, hardware VT
   ~zero startup cost       small fixed per-sandbox cost  <=125 ms boot, <=5 MiB VMM
```

### 1.2 Firecracker microVMs

**Problem.** AWS built Firecracker to run multi-tenant serverless workloads (AWS Lambda and Fargate) with VM-grade isolation at container-like density and startup speed [5].

**Architecture.** Firecracker is a virtual machine monitor that runs on Linux KVM [1]. Each Firecracker process hosts exactly one microVM and runs three kinds of threads: an API thread, a VMM thread, and one or more vCPU threads [3]. The device model is deliberately minimal [3]:

- VirtIO net, block, and vsock devices
- a serial console (not exposed in production builds)
- a partial i8042 controller, used mainly to signal guest reboots
- the PIC, IOAPIC, and PIT provided by KVM
- MMDS, a guest-facing metadata service run on the VMM thread

Everything else is left out to shrink memory footprint and attack surface [1].

**Security techniques.** The threat model treats guest vCPU threads as hostile from the moment they start [3]. A **jailer** process sets up cgroups, namespaces, and a chroot, drops privileges, and then execs the unprivileged Firecracker binary [1][3]. **Per-thread seccomp filters** are installed before any guest code runs and allow only the host syscalls each thread needs [3]. **Token-bucket rate limiters**, with separate buckets for operations per second and bandwidth, apply to every block and network device [3]. Firecracker does not filter network traffic, so egress policy must be enforced on the host [3]. This is why every agent sandbox built on Firecracker adds its own firewall layer (see E2B below).

**Numbers.** From the specification [2]:

- Up to 125 ms from the `InstanceStart` API call to the guest's user-space init.
- VMM overhead of at most 5 MiB for a 1 vCPU / 128 MiB microVM.
- About 12 ms wall clock (range 6-60 ms) until the API socket is available.
- Compute-only guest performance above 95% of bare metal.
- Network up to 14.5 Gbps at no more than 80% of a host core, with about 0.06 ms added latency.

Demand-fault paging and CPU oversubscription are on by default [1].

**Snapshots.** Snapshots matter more for agents than cold boot does. Firecracker supports two kinds [4]:

- **Full snapshots** save device state and all guest memory.
- **Diff snapshots** save only the pages dirtied since the last snapshot. A diff must be rebased onto a base memory file to be resumed.

On restore, the memory file is mapped privately, pages load on demand, and writes go to anonymous copy-on-write memory. A **userfaultfd (UFFD) backend** lets a separate user-space process serve page faults, which is the hook that lets sandbox providers stream memory lazily from object storage [4].

**Failure modes.** The snapshot documentation carries a warning that applies directly to agent "fork" features. Resuming the same snapshot more than once duplicates unique IDs, random seeds, and cryptographic tokens [4]. Firecracker exposes a VMGenID device that changes on every resume, and Linux 5.18+ reseeds its kernel PRNG when it changes. Application-level secrets and cached random values are not reseeded, so applications must de-duplicate that state themselves [4]. Firecracker also trusts snapshot files, so operators must add their own authentication and encryption for snapshot storage [4].

**Ecosystem.** Firecracker integrates with containerd through firecracker-containerd. That project has a host-side runtime shim talking ttrpc to an in-VM agent, which runs ordinary runc containers inside the microVM [6]. Firecracker is also integrated with Kata Containers, which packages lightweight VMs behind a containerd shimv2 runtime [5][7].

### 1.3 gVisor: an application kernel

**Architecture.** gVisor takes what its documentation calls a "distinct third approach" between containers and VMs [8]. The **Sentry** is a userspace application kernel written in Go. It reimplements Linux syscalls, memory management, filesystems, networking, and signals, and "never passes through any system call to the host" [9]. A more privileged sidecar, the **Gofer**, handles requests the restricted Sentry cannot serve itself [9]. The Sentry is itself confined with seccomp, namespaces, cgroups, and `pivot_root`. To escape, an attacker must exploit both the Sentry and the host kernel, which share no code [9]. gVisor ships as `runsc`, an OCI runtime that works with Docker, containerd, and Kubernetes [8].

**Platforms.** Syscall interception is pluggable [11]:

- **systrap** uses seccomp's trap feature. It works inside VMs and on hosts without virtualization support, and has been the default since mid-2023.
- **KVM** uses virtualization extensions for address-space switching. It is fastest on bare metal but slower under nested virtualization.
- **ptrace** is deprecated.

**Performance tradeoffs.** The cost profile is specific [10]. CPU-bound work runs at native speed, and memory access is native once mappings exist. Overhead concentrates in workloads that make many small syscalls or small file operations, such as key-value stores, static file servers, and lightweight HTTP apps. Each sandbox carries a small, mostly fixed Sentry memory overhead that grows with open files and sockets [10]. Agent code-execution workloads (Python data analysis, test runs) are mostly CPU- and import-bound, which suits this profile. Package installs and `git` operations with many small files sit at the expensive end.

**Limits.** gVisor does not protect against Spectre-class side channels or attacks above the runtime. Its documentation advises putting untrusted customers in separate sandboxes [9].

### 1.4 OS-level process sandboxes: Anthropic sandbox-runtime

Not every agent needs a VM. A local coding agent runs on the developer's own machine, where the threat is mainly a confused or injected agent, not a hostile tenant. Anthropic's open-source **sandbox-runtime** (`srt`), developed for Claude Code, enforces filesystem and network restrictions on a process tree without a container [18]:

| Platform | Mechanism |
|---|---|
| macOS | `sandbox-exec` with dynamically generated Seatbelt profiles |
| Linux | bubblewrap with network-namespace isolation, plus seccomp filters for Unix sockets |
| Windows (alpha) | dedicated local user, Windows Filtering Platform egress fence, NTFS ACEs |

**Policy model.** Reads are allowed by default, with deny-then-allow overrides. Writes are denied by default and allowed only on listed paths. Network access is allow-list only [18]. All traffic goes through host-side HTTP and SOCKS5 proxies. On Linux, the sandbox's network namespace is removed entirely and the proxies are reached over bind-mounted Unix sockets. A resolved-address check stops allowed hostnames from resolving to loopback, link-local, or cloud-metadata addresses, which blocks DNS-rebinding bypasses [18]. Claude Code's sandboxed bash tool is built on these primitives. When the agent first contacts a new domain, the proxy asks the user [19].

### 1.5 Credential isolation: keep secrets out of the box

The most consistent lesson across providers is that **the sandbox should never hold long-lived credentials**. In Claude Code on the web, each session runs in an isolated cloud sandbox. Git credentials and signing keys stay outside it. The sandboxed git client holds only a scoped credential. A git proxy validates each request (for example, that pushes target only the configured branch) and then attaches the real token before forwarding to GitHub [19].

Anthropic's Managed Agents applies the same rule more broadly. Generated code runs without access to tokens. MCP calls go through a proxy that fetches credentials from a vault using a session-scoped token, and the harness itself never sees credentials [20]. Cloudflare's Sandbox SDK has the Worker sign each storage request for mounted S3 buckets, so bucket credentials never enter the container [17].

```
   +---------------- sandbox (untrusted) ----------------+
   |  model-generated code   scoped token only           |
   |        |                                            |
   +--------|--------------------------------------------+
            v  (all egress)
   +-----------------------+      +----------------------+
   | egress proxy / git    |----->| vault: real tokens   |
   | proxy: allow-list,    |<-----| (never in sandbox)   |
   | policy check, inject  |      +----------------------+
   +-----------------------+
            |
            v   GitHub / MCP servers / package registries
```

### 1.6 E2B: snapshot-restore sandboxes on Firecracker

**Problem.** E2B provides "secure isolated sandboxes in the cloud" for running AI-generated code, with Python and JavaScript SDKs, a code-interpreter SDK, and a desktop SDK for computer use [14]. The runtime is open source and self-hostable via Terraform on AWS, GCP, Azure, or generic Linux [14].

**Architecture.** E2B's infrastructure repository and architecture document describe a strict split between control plane and data plane [12][13]:

```
  SDK ---- POST /sandboxes ----> API (control plane: auth, quotas, placement)
                                   | Postgres (entities), Redis (running state)
                                   | gRPC Create
                                   v
  user/agent traffic         Orchestrator (one per node, Go, runs as root)
        |                     - fetch template: local cache -> NFS -> object store
        v                     - network slot: netns + veth + tap + nftables
  client-proxy (edge) ---->   - rootfs: CoW overlay via in-process NBD
  Redis lookup                - memory: lazy, userfaultfd from memfile
  sandbox:routing:{id}        - resume Firecracker VM, call envd /init
  miss -> API Resume              |
                                  v
                         [ Firecracker microVM ]  <- envd agent inside (process,
                                                     filesystem, port APIs)
  Object storage: templates and snapshot diffs   ClickHouse: events, metrics
```

**Key techniques.**

- **Create is a resume.** Templates are pre-booted VMs (memory, disk, and machine state) stored in object storage. Creating a sandbox restores the template's base snapshot instead of booting a kernel. Cold boots happen only for filesystem-only templates or on explicit request [13].
- **Lazy everything.** Memory pages load on fault via userfaultfd. The root filesystem is a copy-on-write overlay over a read-only image, exposed through an in-process NBD server, so only touched data is fetched [12][13].
- **Diff snapshots for pause.** Pause captures dirty memory and the rootfs cache as diffs against the source template. These are uploaded asynchronously, and diff chains are resolved through header index files. Resume prefers the origin node, which may still have the data cached [13].
- **Fork.** A live sandbox can be checkpointed and new sandboxes started from that snapshot while the original keeps running [12]. This is the primitive behind parallel exploration ("try three fixes at once"). The clone-uniqueness caveat from Firecracker [4] applies.
- **Placement.** The API uses best-of-K sampling. It samples K ready nodes, scores each by the higher of CPU load and hugepage-pool load, and picks the lowest. Nodes that refuse with `ResourceExhausted` stay eligible, and retries use capped full-jitter exponential backoff [13].
- **Wake on traffic.** The orchestrator writes a Redis routing record when a sandbox starts running. On a miss, the client-proxy asks the API to resume the paused sandbox and retries, so paused sandboxes wake transparently when a request arrives [13].
- **Egress control.** Each sandbox gets a pooled network slot with its own nftables firewall. A TCP-level filter inspects SNI and Host headers to enforce domain allow and deny lists [13]. Firecracker itself does no filtering [3]. The edge proxy also refuses envd control routes, so a sandbox URL cannot reach `/init` or the freeze/thaw hooks [13].

**Lessons.** The E2B design treats a sandbox as a **cheap, resumable, forkable snapshot** rather than a long-lived server. The control plane is off the data path: "Sandbox traffic goes straight from the edge to the node and never passes through the API" [12]. The architecture document describes startup only as "almost instant" and publishes no latency figure [13].

### 1.7 Daytona

Daytona positions its sandboxes as "composable computers": each has a dedicated kernel, filesystem, network stack, and allocated vCPU, RAM, and disk [15]. The project claims sandboxes spin up "in under 90ms from code to execution" (a vendor claim). The architecture is split into an interface plane, a control plane, and a compute plane [15].

Daytona emphasizes **statefulness** more than ephemerality. It offers stateful snapshots for persistent agent work across sessions, OCI/Docker image compatibility, and built-in developer services: LSP, git operations, a web terminal, SSH, VNC, computer use, volumes, and webhooks [15]. Its GitHub repository states that, as of June 2026, core development moved to a private codebase and the public repository is no longer maintained [15]. Teams that chose it for self-hostability should take note.

### 1.8 Modal, Vercel, Cloudflare, and the provider-agnostic layer

- **Modal.** Modal's SDKs (Python, JavaScript/TypeScript, Go) expose Modal Sandboxes alongside its serverless functions; the OpenAI Agents SDK lists Modal as a hosted sandbox client [22]. Modal's own documentation on isolation and snapshot internals could not be reached for this chapter, so no internals are stated here.
- **Vercel Sandbox.** Each sandbox is a Firecracker microVM on the same infrastructure that runs Vercel builds [16]. Limits: default timeout 5 minutes, maximum 24 hours on Pro/Enterprise and 45 minutes on Hobby; up to 8 vCPUs (32 on Enterprise) at 2,048 MB per vCPU. Any OCI image can be used via Vercel's container registry [16].
- **Cloudflare Sandbox SDK.** Each sandbox is a Durable Object plus the Linux container it starts. Directories can be backed up to R2 and restored into any container, including one built from a newer image. Network access is configurable per container [17]. In version 1.0, ownership of the container moved from an SDK class to the developer's own Durable Object, leaving the SDK to handle files and mounts only [17].

**The provider-agnostic layer.** By 2026 agent frameworks treat the sandbox as a pluggable backend. OpenAI's Agents SDK "Sandbox Agents" give the model a persistent workspace described by a **manifest** (files, repos, mounts). The SDK owns the harness (file staging, tools, lifecycle, snapshots), while a **sandbox client** decides where commands run [21]. Supported clients are Unix-local, Docker, and hosted backends from Blaxel, Cloudflare, Daytona, E2B, Modal, Runloop, and Vercel [22]. Mounted cloud buckets are treated as ephemeral: snapshot flows skip or detach mounted paths rather than copying remote storage into the saved workspace [22]. The SDK warns that the Unix-local client adds no OS-level confinement on Linux and only filesystem restrictions on macOS, so untrusted input requires Docker or a hosted sandbox [21].

### 1.9 Comparison

| System | Isolation primitive | Start path | Persistence model | Egress control | Self-host |
|---|---|---|---|---|---|
| Firecracker (raw) | KVM microVM, jailer, seccomp | boot <=125 ms; snapshot restore | full/diff snapshots, UFFD lazy load | none built in (host must filter) | yes (Apache-2.0) |
| gVisor (runsc) | userspace app kernel (Sentry) | OCI container start | container FS | host/netstack policies | yes |
| Anthropic srt | bubblewrap / Seatbelt / WFP | process spawn | host FS (write allow-list) | proxy allow-list, DNS checks | yes (local tool) |
| E2B | Firecracker | restore pre-booted template | pause = diff vs template; fork | nftables + SNI/Host filter | yes (Terraform) |
| Daytona | dedicated-kernel sandbox | "<90 ms" (vendor) | stateful snapshots, volumes | not specified in sources | repo unmaintained 2026 |
| Vercel Sandbox | Firecracker | OCI image | not stated in README | not stated in README | no |
| Cloudflare Sandbox | container + Durable Object | DO starts container | R2 directory backups | per-container network flag | no |

---

## 2. Durable execution: making the loop survive

### 2.1 The problem

An agent loop interleaves three kinds of work:

1. Expensive, non-deterministic calls (LLM inference).
2. Side effects (sending an email, opening a PR, charging a card).
3. Waits that can last hours (human approval, a CI run).

Suppose a worker crashes at step 37 of 60. A naive retry restarts from step 1. That re-pays for 36 model calls and may repeat side effects. Model outputs are non-deterministic, so the replay may also take a different path. Anthropic's research system addresses this by building systems that resume from where the agent was when the error happened, combining "the adaptability of AI agents" with deterministic safeguards such as retry logic and regular checkpoints [42].

Durable execution engines solve this generically. Each completed step's result is recorded in a durable log. After a failure, the code re-runs but takes recorded results from the log instead of re-executing them. For agents, the key rule follows directly: **every LLM call and every tool call is a step**. Recorded model outputs are replayed, not regenerated, so recovery is both cheap and consistent with the original run.

```
   attempt 1:  [llm#1]->[tool#1]->[llm#2]->[tool#2]->[llm#3]-X crash
                  |        |         |        |
                  v        v         v        v
   durable log:  r1       r2        r3       r4          (append-only)
                  |        |         |        |
   attempt 2:  [llm#1]  [tool#1]  [llm#2]  [tool#2]  ->[llm#3]->[tool#3]-> done
               replayed replayed  replayed replayed    executed for real
               (no token cost, no repeated side effect)
```

### 2.2 Temporal

**Lineage.** Temporal is a fork of Uber's Cadence, built by Cadence's creators. It is MIT-licensed [23].

**Architecture.** The server has four roles [24]:

- **Frontend** handles client requests.
- **History** shards own individual workflow executions.
- **Matching** manages task queues that workers long-poll.
- **Internal workers** run system workflows.

Temporal is event-sourced: "an append-only history of events is stored for each workflow execution," and state can be rebuilt from it [24]. The number of history shards is fixed when the cluster is created and cannot be changed later. Each shard carries a `RangeID` fencing token [25]. Each state transition does three things: it appends history events, updates a cached **mutable state** summary, and writes transfer and timer tasks in one transaction. The transfer queue implements the **transactional outbox** pattern toward Matching [25]. Persistence can be Cassandra, MySQL, Postgres, or SQLite [25]. Task queues are split into partitions (four by default) arranged in a forwarding tree, so idle pollers and unmatched tasks can move toward the root [27].

**Execution model.** Workers run user code. A **Workflow Task** advances the deterministic workflow function until it blocks. The worker returns commands such as `ScheduleActivityTask` or `StartTimer`. **Activity Tasks** run the non-deterministic, side-effecting functions [24][26]. Each step in the lifecycle is a durable write: `WorkflowExecutionStarted`, `WorkflowTaskScheduled`, `ActivityTaskScheduled`, `ActivityTaskCompleted`, and so on. A failed activity appends `ActivityTaskFailed` and sets a retry timer [26]. Because state is rebuilt by replaying history, **workflow code must be deterministic** [24].

**Agent integrations.** Temporal maintains an `ai-integrations` repository. It lists packages for the OpenAI Agents SDK, plus Google ADK, LangGraph, Strands Agents, MCP, LangSmith, and Deep Agents, with some marked public preview and others pre-release [28]. The OpenAI Agents integration shows the canonical mapping [29]:

- **Model calls become activities.** The `OpenAIAgentsPlugin` routes every model invocation through a Temporal activity. Agent coordination logic stays in the deterministic workflow. A `start_to_close_timeout` bounds each model call.
- **Tools are split by determinism.** `activity_as_tool()` wraps an activity as a tool for external I/O. Plain `@function_tool` functions run inside the workflow and must be deterministic, with no network, database, or filesystem access. Provider-hosted tools run inside the model call.
- **Recovery is by replay.** A crashed worker resumes from recorded progress. An activity that fails partway restarts from its beginning, so activities must be idempotent.
- **Limitations are explicit.** Realtime agents, `LocalShellTool`, and `ComputerTool` are unsupported in a distributed setting. `SQLiteSession` must be replaced by `WorkflowSession`. Activity-based tools receive a read-only copy of the context. Streaming requires a configured streaming topic and cannot be combined with local activities [29].

**Failure modes and lessons.** Determinism is the main sharp edge. Any change to workflow code that alters the command sequence can break replay of in-flight executions. Agent coordination code must therefore be versioned, or kept thin, with the logic pushed into activities. Long agent sessions also grow history, because every model response is recorded as an activity result. Large prompts and responses should be stored by reference rather than inline.

### 2.3 Restate

**Model.** Restate presents itself as a platform for applications that "tolerate all infrastructure faults" [30]. Its SDKs (TypeScript, Java/Kotlin, Python, Go, Rust) send execution progress to the Restate server, which stores it in a durable **journal** and triggers retries. On retry, the server invokes the handler again with the latest journal. The SDK replays it, skipping actions that already have recorded results [31]. The building blocks are journaled actions, durable timers, durable promises, persisted key-value state, and **suspension**: a handler awaiting a promise can suspend and be re-invoked when the result arrives, which suits function-as-a-service deployments [33].

**Service types** [32]:

- **Services.** Stateless durable handlers with no concurrency limits, suitable for wide scale-out (for example, on AWS Lambda).
- **Virtual objects.** Keyed entities with isolated key-value state. Only one exclusive (state-writing) handler runs per key at a time, with requests queued per key. Shared handlers can read concurrently. The documentation lists **AI chat sessions** as a use case.
- **Workflows.** A virtual object whose `run` handler executes exactly once per workflow ID, with concurrent handlers for queries and signals.

**Relevance to agents.** The virtual-object model maps cleanly onto a conversational agent. The session ID is the key, the conversation and memory live in per-key state, and single-writer semantics prevent two concurrent user messages from interleaving turns. Suspension means an agent waiting on a human approval consumes no compute. Restate also emits OpenTelemetry traces automatically [30], and its repository links "Durable AI Agents" examples [30]. The server source appears to be written in Rust (inferred from the repository's Cargo files); the repository README does not describe the internal log architecture.

### 2.4 Inngest

**Architecture.** Inngest is event-driven [34]. The pieces are:

- An **Event API** authenticates and ingests events.
- An **event stream** buffers them.
- A **Runner** schedules new runs, resumes runs paused in `waitForEvent`, and cancels runs matching `cancelOn` expressions.
- A **multi-tenant, multi-tier queue** enforces flow control.
- An **Executor** runs steps and writes incremental state.
- A **state store** holds triggering events, step outputs, and step errors for in-flight runs [34].

Flow control is first-class: concurrency, throttling, debouncing, rate limiting, prioritization, and batching [34]. For agent fleets that share a provider rate limit, this matters as much as durability does.

**Protocol.** The SDK specification is notable because Inngest drives user code **over HTTP** rather than through long-running workers [35]:

- Inngest POSTs to a single SDK endpoint. The body carries the event and a `steps` map of memoized results keyed by the SHA-1 of each step's identifier. Repeated identifiers get a `:n` suffix.
- The SDK runs the function. Memoized steps return their recorded `data`, or rethrow their recorded `error`.
- When the code reaches a new step, the SDK responds `206 Partial Content` with an opcode: `StepRun`, `Sleep`, `WaitForEvent`, `InvokeFunction`, `AiGateway`, or `Gateway`.
- Inngest executes or schedules the step, records the result, and calls the function again.
- Each step defaults to 4 attempts. If the expected next step is missing from the recorded stack, the SDK warns that the function may have changed or is non-deterministic [35].
- When several steps are reported at once, all later calls use planned execution, so each parallel step runs in its own request [35].

The **`AiGateway`** opcode lets the Inngest platform itself make the model HTTP request (`url`, `headers`, `body`) and memoize the response [35]. During a long inference call, the user's serverless function does not have to stay alive.

**AgentKit.** Inngest's TypeScript AgentKit builds multi-agent **networks** with shared key-value **state** and **routers**, either code-based or agent-based (the latter capped with `maxIter`). It also supports MCP servers as tools. Tool handlers can wrap work in `step.run` so it inherits durable execution [36].

### 2.5 DBOS and the library-only approach

DBOS Transact takes the opposite deployment stance from Temporal: **no separate orchestration server** [37]. Functions annotated `@DBOS.workflow()` and `@DBOS.step()` checkpoint their progress as rows in Postgres. After a crash, workflows resume from the last completed step. Queues with concurrency limits, rate limits, timeouts, and priorities need no separate broker [37]. For teams that already run Postgres, this removes one piece of infrastructure from the agent stack. The tradeoff is that the application database carries the orchestration load.

### 2.6 Comparison

| Engine | Durable record | Code driver | State model | Agent-specific support | Ops footprint |
|---|---|---|---|---|---|
| Temporal | per-workflow event history, sharded | long-polling workers | workflow vars rebuilt by replay | ai-integrations: OpenAI Agents, ADK, LangGraph, Strands, MCP | server cluster + DB |
| Restate | journal per invocation | server pushes; handlers can suspend | per-key K/V (virtual objects) | AI chat sessions as virtual objects; agent examples | single binary/cluster |
| Inngest | memoized step map per run | HTTP calls into SDK endpoint | step outputs + shared network state | AgentKit; `AiGateway` opcode offloads inference | hosted or self-host server |
| DBOS | Postgres rows per step | in-process library | Postgres | general (non-deterministic API calls) | Postgres only |
| LangGraph | checkpoint per super-step | in-process | typed graph state channels | native (see Section 3) | checkpointer DB |

### 2.7 Design rules that recur across engines

1. **Wrap every model call as a step or activity.** Replay must not regenerate tokens. This saves money and keeps the replayed trajectory identical to the original.
2. **Make side-effecting tools idempotent.** Activities and steps are at-least-once at their boundary: a step that fails partway runs again from its start [29][35]. Use idempotency keys for external writes.
3. **Keep orchestration logic deterministic and thin.** Branching on model output is fine because the output is recorded. Reading the clock, generating random numbers, or doing direct I/O inside the workflow body is not [29].
4. **Externalize large payloads.** Store prompts, transcripts, and files by reference in blob storage, not inline in history.
5. **Use durable timers and signals for humans in the loop.** Waiting on `waitForEvent`, a Temporal signal, or a Restate promise costs no compute while suspended [33][35].

---

## 3. Agent state and checkpointing

Durable execution protects the control flow. Agents also have **semantic state**: the message history, scratch files, and working directory. Systems take four approaches.

### 3.1 Framework checkpointers (LangGraph)

LangGraph saves graph state at every **super-step** through a pluggable checkpointer. This enables human-in-the-loop pauses, memory across interactions, and durable execution [40]. The `BaseCheckpointSaver` interface has these methods [40]:

- `put` stores a checkpoint with its config and metadata.
- `put_writes` stores pending writes.
- `get_tuple` fetches by `thread_id` and an optional `checkpoint_id`.
- `list` enumerates checkpoints.
- `delete_thread` removes a thread's checkpoints and writes.
- Async variants of each are needed for async graphs.

A checkpoint contains `channel_values`, `channel_versions`, and `versions_seen`, keyed by a **thread** ID [40]. Starting a run from an older `checkpoint_id` gives time travel and forking.

The most important detail for agent reliability is **pending writes**. If one node fails mid-super-step, the writes from nodes that succeeded in that super-step are saved, and those nodes are not re-run on resume [40]. In parallel tool fan-out, this means one failed branch does not re-execute the successful ones.

Implementations include SQLite and Postgres (`PostgresSaver`, `AsyncPostgresSaver`), plus a conformance test suite for third-party savers [40][41]. Two operational warnings appear in the READMEs. First, user-supplied Postgres connections need `autocommit=True` and `row_factory=dict_row` [41]. Second, the default serializer accepts any Python type during deserialization, so `LANGGRAPH_STRICT_MSGPACK=true` or an explicit module allow-list is recommended [40]. Treat this as a deserialization-of-untrusted-data risk whenever checkpoint storage could be tampered with.

### 3.2 The append-only session log (Anthropic Managed Agents)

Anthropic's 2026 account of **Managed Agents** is the clearest public description of separating state from compute [20]. The first design put the harness, the state, and the sandbox in one container. That made the container a "pet": when it died the session was lost, and a harness bug, a dropped event, and a dead container all looked the same during debugging [20]. The redesign splits the agent into three parts, each with a stable interface:

```
   +--------------------+      append / read slices      +----------------------+
   | Harness (stateless)|------------------------------->| Session log          |
   | loop: call Claude, |<-------------------------------| append-only, durable |
   | route tool calls   |                                | outside harness and  |
   +---------+----------+                                | sandbox              |
             | execute(name, input) -> string            +----------------------+
             v
   +--------------------+    credentials via vault proxy (session-scoped token)
   | Sandbox (cattle)   |    provisioned lazily, only when a task needs one
   +--------------------+
```

Because the harness is stateless, it can crash and restart, then recover by replaying the session log. The sandbox is just a tool with an `execute(name, input) -> string` interface, so it can be replaced or provisioned lazily [20]. The session also serves as a durable context store outside the model's window. The harness queries it by slice and decides what to pass to the model, so context policy can change without losing recoverable history [20]. Chapter 05 covers the context side of this design.

### 3.3 Per-agent actors (Cloudflare Agents)

Cloudflare's Agents SDK gives each agent its own **Durable Object**: an addressable single-instance actor with its own storage and lifecycle [38]. State persists across restarts and syncs to connected clients. Agents can run SQLite queries directly. Scheduling supports one-time, recurring, and cron tasks. WebSockets give bidirectional real-time communication. Durable workflows support human-in-the-loop approval [38]. The key economic property is that agents "hibernate when idle and wake on demand," so inactive agents incur no cost. The README claims this supports millions of agents, one per user, session, or room (a vendor claim without published benchmarks) [38]. Restate's virtual objects (Section 2.3) are the same idea delivered as a durable-execution feature rather than an edge platform.

### 3.4 Filesystem state: snapshots, git, and progress files

For coding and computer-use agents, much of the state is the workspace itself. Three techniques appear:

- **VM or container snapshots.** E2B's pause stores memory and disk diffs against the template, and fork clones a live sandbox [12][13]. Daytona offers stateful snapshots [15]. Cloudflare backs directories up to R2 [17]. OpenAI's sandbox agents reconnect through `session`, `session_state`, or `snapshot` [21].
- **Git as a checkpoint log.** In Anthropic's long-running-agent harness, an initializer agent creates an `init.sh`, a progress log, a JSON feature list with every feature marked failing, and an initial commit. Each later session reads the git log and progress file, runs a smoke test, implements one feature, commits, and records a summary [43]. Git gives cheap rollback when a session leaves the code broken. JSON was less likely than Markdown to be inappropriately overwritten by the model [43].
- **Files as working memory.** Anthropic's code-execution-with-MCP pattern has agents write intermediate results (for example, a CSV) to the workspace so later runs can resume, and save working code as reusable skills [61].

### 3.5 Choosing a state strategy

| Need | Mechanism | Example |
|---|---|---|
| Resume control flow after crash | durable execution journal | Temporal, Restate, Inngest, DBOS |
| Resume / fork conversation at any step | per-step checkpoints keyed by thread | LangGraph checkpointers |
| Replaceable harness, recoverable context | append-only session log outside compute | Anthropic Managed Agents |
| Many idle long-lived agents | hibernating actor per agent | Cloudflare Durable Objects, Restate virtual objects |
| Resume a workspace (files, processes) | VM/FS snapshots, object-store backups | E2B pause/fork, Cloudflare R2 backups |
| Human-auditable progress across sessions | git commits + progress files | Anthropic long-running harness |

---

## 4. Observability

### 4.1 What is different about agent telemetry

Request-level APM metrics such as latency, error rate, and throughput say little about why an agent failed. Agent failures are usually semantic: a wrong tool was chosen, a search was too broad, a loop never terminated. They are visible only in the full trajectory. Anthropic reports that full production tracing let them diagnose why agents failed and fix issues systematically. They monitor decision patterns and interaction structures while avoiding monitoring the contents of individual conversations, to protect privacy [42]. Observability has to capture nested spans for agent, model call, and tool call, along with token and cost accounting, inputs and outputs (subject to redaction), and links to evaluations.

### 4.2 Trace models

**OpenAI Agents SDK.** A trace is one end-to-end workflow with a `workflow_name`, `trace_id`, and optional `group_id` for linking conversations [45]. Built-in span types cover runner tasks, turns, agent runs, LLM generations, function tools, guardrails, handoffs, and audio, plus custom spans. Tracing is **on by default** and is exported by a background `BatchTraceProcessor` [45]. Processors can be added alongside the OpenAI exporter or replace it entirely. The documentation lists more than two dozen third-party destinations, including LangSmith, Langfuse, AgentOps, Arize Phoenix, Braintrust, Datadog, and MLflow [45]. `trace_include_sensitive_data` controls whether model and tool inputs and outputs are stored. The docs advise putting redaction inside the same exporter that delivers data, so that a failed redactor cannot leak to other processors [45]. Tracing is unavailable for organizations under Zero Data Retention [45].

**OpenTelemetry GenAI semantic conventions.** These now live in a dedicated repository. It covers spans, metrics, and events for GenAI client inference, agents, tool execution, evaluation, and MCP, with attribute namespaces `gen_ai.*`, `mcp.*`, and `openai.*` [46]. The agent-span document defines operation names including `create_agent`, `invoke_agent`, `invoke_workflow`, `execute_tool`, and `plan`, plus memory operations such as `create_memory` and `search_memory` [47]. For a `create_agent` span, `gen_ai.operation.name` and `gen_ai.provider.name` are required. `gen_ai.agent.name`, `gen_ai.agent.id`, and `gen_ai.request.model` are conditionally required. System instructions are opt-in [47]. Most of these attributes are still marked **Development**; only generic ones such as `error.type` and `server.address` are Stable [47]. Teams should expect attribute names to change.

**Replay-safe tracing under durable execution.** Durable engines replay code, which would re-emit spans. Temporal's OpenAI Agents integration requires a `ReplaySafeTracerProvider` for its OpenTelemetry mode, and exports spans only when workflows complete [29]. Any tracing layered on a replaying runtime must deduplicate by step identity.

### 4.3 Platforms and their ingestion architectures

**Langfuse** is an open-source (MIT, except `ee` folders) LLM engineering platform. It covers tracing, prompt management with versioning and caching, evaluations (LLM-as-judge, code evaluators, user feedback, manual labelling), datasets, and a playground [48]. Its self-hosting documentation describes an ingestion path designed for bursty agent traffic [49]:

```
  SDKs --batches--> Web container --write--> S3 / blob (raw events, durable)
                         |                        |
                         +--ref--> Redis queue    |
                                       |          v
                                    Worker ---read---> ClickHouse
                                                     (wide observations table,
                                                      no read-time joins)
  Redis also caches API keys (cheap auth rejection) and hot prompts
```

All traces are "immediately written to S3", and only a reference is queued in Redis. Two consequences follow: request spikes do not cause timeouts limited by database capacity, and a database outage does not lose events, which can be processed later [49]. ClickHouse's wide observations table avoids read-time joins and deduplication [49].

**LangSmith** is LangChain's hosted platform for debugging, evaluating, and monitoring LLM apps and agents. Tracing is enabled with environment variables and client wrappers such as `wrap_openai` [50]. LangGraph's documentation positions LangSmith as the place for execution-path traces, state-transition capture, runtime metrics, agent evals, and deployment of long-running stateful agents [39].

**AgentOps** is an MIT-licensed SDK and dashboard aimed specifically at agents. It offers session replays, step-by-step execution graphs, and LLM cost tracking. `@session`, `@agent`, `@operation`/`@task`, and `@workflow` decorators create nested spans [51]. Its self-hostable app shows the stack that has become typical by 2026 [52]:

- an OpenTelemetry collector receiving OTLP on port 4318
- ClickHouse for traces and metrics
- Supabase/Postgres for application data and auth
- a FastAPI backend and a Next.js dashboard

**Arize Phoenix** is self-hostable observability and evaluation built on OpenTelemetry. Instrumentation comes through the OpenInference project, and it is licensed under the Elastic License 2.0 [53].

**Helicone** takes a **proxy** approach. Applications point their SDK's base URL at the Helicone gateway, which logs requests (via a Cloudflare Workers component), routes across 100+ models with fallbacks, and tracks cost and latency. Its backend uses ClickHouse for analytics and object storage for logs [54].

### 4.4 Comparison

| Platform | Capture method | Storage (per sources) | Agent-specific features | License |
|---|---|---|---|---|
| OpenAI Agents SDK tracing | in-process spans, batch exporter | OpenAI Traces or any processor | handoff/guardrail/tool spans, sensitive-data flag | MIT (SDK) |
| LangSmith | SDK wrappers, env-var tracing | hosted | graph state transitions, evals, deployment | SDK MIT; platform commercial |
| Langfuse | SDKs + integrations | S3 -> Redis queue -> ClickHouse | sessions, prompt mgmt, evals, datasets | MIT (+ee) |
| AgentOps | decorators + OTLP collector | ClickHouse + Postgres | session replay, cost tracking | MIT |
| Arize Phoenix | OpenTelemetry / OpenInference | self-hosted | evals, experiments | ELv2 |
| Helicone | HTTP proxy / gateway | ClickHouse + object storage | sessions, routing, fallbacks | Apache-2.0 |

### 4.5 Failure modes

- **Sensitive-data leakage.** Default-on tracing that captures inputs and outputs [45] will copy user data into a third system. Redact at the exporter.
- **Trace fan-out cost.** Multi-agent runs produce very large traces. Batch exporters and sampling are required at volume.
- **Replay duplication** under durable execution (Section 4.2).
- **Convention churn.** GenAI semantic conventions are still in Development status [47].

---

## 5. Deployment and scaling

### 5.1 Decouple the brain from the hands

The dominant 2026 deployment pattern separates three tiers that scale independently:

1. **Stateless harnesses**, which run the model loop.
2. **A durable state tier**: a session log, a checkpointer DB, or a durable-execution server.
3. **A sandbox fleet**, used only when a task needs compute.

Anthropic's Managed Agents measured the payoff. Once containers were provisioned only when needed, inference could begin before any sandbox existed. **p50 time-to-first-token dropped about 60% and p95 by over 90%** [20]. The OpenAI Agents SDK draws the same line between harness and sandbox client [21], and Temporal's integration draws it between workflow (coordination) and activities (I/O) [29].

```
            +---------------- load balancer / queue ----------------+
            v                         v                              v
     [harness pod]             [harness pod]                  [harness pod]   stateless
            \                         |                              /         autoscale
             +----------- session log / checkpoint DB --------------+          durable
                                      |
               +----------------------+----------------------+
               v                      v                      v
          [sandbox]              [sandbox]              (none yet)           lazy,
          microVM                microVM                                     disposable
```

### 5.2 Deploying while agents are running

Long-running agents make conventional rolling deploys dangerous, because an agent can be mid-trajectory whenever a deploy lands. Anthropic uses **rainbow deployments**: traffic shifts gradually from old to new versions while both run, so in-flight agents finish on the version they started with [42]. Durable-execution engines face the same issue as a code-compatibility problem: replaying old history against new workflow code. The Inngest SDK warns when the recorded step stack no longer matches the code [35]. Temporal requires determinism across replays [24]. In both cases, version the orchestration code or drain old executions before removing old workers.

### 5.3 Harness assumptions go stale

The Managed Agents write-up warns that harness assumptions decay as models improve. Context resets that helped Sonnet 4.5 became unnecessary on Opus 4.5 [20]. Infrastructure should therefore expose **stable, minimal interfaces**, such as `execute(name, input) -> string` for sandboxes and an append-only log for sessions. Implementations behind those interfaces can then change without migrating state [20].

### 5.4 Sandbox fleet scheduling

E2B's placement algorithm is a compact reference design [13]:

- **Best-of-K scoring** on max(CPU load, hugepage-pool load).
- **Soft refusals** with `ResourceExhausted` rather than global admission control.
- **Full-jitter capped backoff** on retries.
- **Origin-node preference on resume**, so cached snapshot data avoids an object-storage read.
- **Pooled network slots**, so namespace and firewall setup stays off the critical path.
- **Lifecycle timeouts**, including an auto-pause overstay budget (120 s by default) so idle sandboxes do not hold capacity indefinitely.

Vercel exposes the same idea as user-facing limits: a 5-minute default timeout and per-plan maximums [16].

### 5.5 Flow control against provider limits

Agent fleets share model-provider rate limits. Inngest's queue offers concurrency, throttling, rate limiting, debouncing, and priority as declarative function settings [34]. DBOS provides concurrency- and rate-limited queues in Postgres [37]. Without such controls, a burst of agent starts turns into a storm of 429 errors and retries. With durable execution, those retries are at least cheap, because completed steps are not re-run.

---

## 6. Cost control

### 6.1 Where the money goes

Agent cost is dominated by **input tokens re-sent on every step**. Usage grows across turns because prior messages are re-sent as input; the OpenAI Agents SDK documentation calls this out [56]. Token consumption is also strongly tied to outcome quality. In Anthropic's BrowseComp analysis, token usage alone explained 80% of performance variance [42]. Cost control is therefore about **spending tokens deliberately**, not simply minimizing them.

### 6.2 Measure first

The OpenAI Agents SDK exposes per-run usage: request count, input and output tokens, cached input tokens, cache-write tokens, and reasoning tokens. `request_usage_entries` breaks this down per model call for exact cost attribution [56]. Observability platforms (AgentOps, Helicone, Langfuse) and gateways (LiteLLM) aggregate this per user, project, or session [48][51][54][59]. Anthropic recommends tracking total tokens and tool-call counts during evaluation, because many redundant calls signal a tool-design problem [62].

### 6.3 Prompt caching

Agents re-send a stable prefix (tools, system prompt, prior turns), so prefix caching is the largest single lever. Anthropic's pricing illustrates the economics [55]:

- **Cache writes** cost 1.25x base input for a 5-minute TTL and 2x for a 1-hour TTL.
- **Cache reads** cost 0.1x base input on most models. The newest models are lower: 0.05x for Opus 5.5 and Sonnet 5.5, and 0.025x for Fable 5.1 and Mythos 5.1.
- The default TTL is 5 minutes and is refreshed free on each hit. The lifetime is measured from the start of the request, so a long streamed response uses up part of it.
- Cache hits are not deducted from rate limits. This is a stated reason to use the 1-hour cache for agents whose steps are more than 5 minutes apart.
- Minimum cacheable lengths range from 512 to 4,096 tokens depending on the model. Prompts below the minimum are silently not cached; check `cache_creation_input_tokens` and `cache_read_input_tokens` to confirm.

The infrastructure implication: **anything that perturbs the prefix**, such as timestamps in system prompts, reordered tool lists, or edited history, multiplies cost. Chapter 05 covers cache-friendly context layout in depth.

### 6.4 Model routing and cascades

- **Routing.** Anthropic's guidance suggests sending easy queries to smaller models (Haiku-class) and hard ones to larger models [60]. RouteLLM learns this decision from preference data. Its routers include matrix factorization, similarity-weighted Elo, a BERT classifier, and a causal-LLM classifier. The project reports cost reductions of up to 85% while keeping 95% of GPT-4 performance on MT-Bench [57].
- **Cascades and approximation.** FrugalGPT (Chen, Zaharia, Zou; TMLR 2024) collects techniques for building LLM applications under budget constraints, including LLM cascades [58].
- **Within agents.** Orchestrator-worker systems can run a strong model as the lead and cheaper models as workers. Sub-agents return condensed summaries, often 1,000-2,000 tokens after exploring with tens of thousands [44]. This both controls context and moves token volume to cheaper calls.

### 6.5 Gateways and budgets

An LLM gateway centralizes keys, routing, fallbacks, and spend accounting. LiteLLM's proxy provides virtual keys, per-project and per-user spend tracking, load balancing, and retry/fallback routing across deployments. Its README claims 8 ms P95 latency overhead at 1k RPS (vendor benchmark) [59]. Helicone offers similar gateway routing with cost tracking [54]. Gateways are also the natural place to enforce hard per-tenant budgets before a runaway agent loop spends them. At the agent level, the equivalent controls are turn limits, such as AgentKit's `maxIter` [36], and timeouts on each model activity [29].

### 6.6 Token-efficient tools and code execution

Tool design is a cost control:

- Claude Code caps tool responses at 25,000 tokens by default [62].
- Anthropic recommends pagination, filtering, truncation with steering hints, and a `response_format` switch between concise and detailed output. In one Slack example, the concise form used about a third of the tokens [62].
- Code execution with MCP goes further. The agent writes code that calls tools inside the sandbox and returns only results. In Anthropic's Google Drive-to-Salesforce example, token use fell from about 150,000 to about 2,000, a 98.7% reduction [61]. The tradeoff is a new dependency on secure sandboxing with resource limits and monitoring [61], which links cost control back to Section 1.

### 6.7 Durable execution as a cost control

Durable execution is usually presented as reliability, but it is also a cost control. A crash without it re-pays for every completed model call. With it, completed calls are replayed from the log at zero token cost (Section 2.1). Inngest's `AiGateway` opcode lets the platform make the inference call, so the user's serverless function does not have to stay alive, and be billed, during long model calls [35]. Hibernating actors [38] and auto-paused sandboxes [13] apply the same principle to idle time.

---

## 7. Failure modes across the stack

| Failure | Where it bites | Mitigation (source) |
|---|---|---|
| Sandbox escape via kernel bug | shared-kernel containers | microVM or application kernel [3][9] |
| Credential exfiltration by injected code | any sandbox holding tokens | vault + proxy injection, scoped tokens [19][20] |
| Exfiltration via allowed domain DNS tricks | egress allow-lists | resolved-address checks [18]; SNI/Host filtering [13] |
| Duplicate secrets after snapshot fork | Firecracker clones | VMGenID reseed + app-level de-dup [4] |
| Repeated side effects on retry | activities/steps | idempotency keys; steps restart from beginning [29] |
| Replay breaks after deploy | Temporal/Inngest code changes | versioning, rainbow deploys [24][35][42] |
| Lost session when container dies | "pet" containers | session log outside compute [20] |
| Re-executing successful parallel nodes | graph super-steps | pending writes [40] |
| Checkpoint deserialization attacks | pickle-like serde | strict msgpack allow-lists [40] |
| Telemetry outage drops traces | synchronous DB ingestion | blob-first ingestion with queue [49] |
| Sensitive data in traces | default-on tracing | exporter-side redaction [45] |
| Runaway cost | unbounded loops, cache misses | turn caps, gateway budgets, cache-stable prefixes [36][55][59] |

---

## 8. Key takeaways for system designers

1. **Pick isolation by threat, not by habit.** Use OS-level sandboxes (bubblewrap/Seatbelt) for a developer's own machine, gVisor where syscall overhead is acceptable and density matters, and microVMs for multi-tenant, untrusted, long-lived workloads. Always pair filesystem isolation with network isolation [18][19].
2. **Never put durable credentials in the sandbox.** Inject them at an egress or git proxy using scoped, session-bound tokens [19][20].
3. **Treat sandboxes as snapshots, not servers.** Snapshot-restore, lazy memory and disk loading, diff-based pause, and fork make sandboxes cheap enough to create per task and to branch for parallel exploration [4][12][13]. Handle clone uniqueness explicitly [4].
4. **Make the harness stateless and the state external.** An append-only session log or checkpoint store outside compute turns crashes into replays and makes containers disposable. Lazy sandbox provisioning gave Anthropic about 60% lower p50 and over 90% lower p95 time-to-first-token [20].
5. **Model every LLM call and tool call as a durable step.** Replay should never regenerate tokens or repeat side effects. Keep coordination code deterministic and thin, and make tools idempotent [24][29][35].
6. **Plan deploys for in-flight agents.** Use rainbow deployments or worker versioning. Replaying old histories against new code is a correctness problem, not just an operational one [24][35][42].
7. **Instrument with nested agent/LLM/tool spans from day one, aligned with OpenTelemetry GenAI conventions.** Expect those conventions to change [46][47]. Redact at the exporter and make tracing replay-safe [29][45].
8. **Ingest telemetry blob-first.** Durable object storage in front of the analytics database absorbs agent burstiness and database outages [49].
9. **Cost is mostly re-sent input.** Keep prefixes cache-stable, pick cache TTLs that match the step cadence, route by difficulty, cap tool output size, and push data processing into the sandbox through code execution [55][57][61][62].
10. **Budget with flow control.** Concurrency limits, rate limits, turn caps, and gateway spend limits are the brakes on a fleet of autonomous loops [34][36][59].

---

## References

1. Firecracker project (AWS). "Firecracker" GitHub repository README. 2018-2026. https://github.com/firecracker-microvm/firecracker
2. Firecracker project (AWS). "Firecracker Specification" (SPECIFICATION.md). 2019-2026. https://github.com/firecracker-microvm/firecracker/blob/main/SPECIFICATION.md
3. Firecracker project (AWS). "Firecracker Design" (docs/design.md). 2018-2026. https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md
4. Firecracker project (AWS). "Snapshot support" (docs/snapshotting/snapshot-support.md). 2020-2026. https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md
5. Firecracker project (AWS). "Frequently Asked Questions" (FAQ.md). 2018-2026. https://github.com/firecracker-microvm/firecracker/blob/main/FAQ.md
6. Firecracker project (AWS). "firecracker-containerd" GitHub repository. 2019-2026. https://github.com/firecracker-microvm/firecracker-containerd
7. Kata Containers project (OpenInfra Foundation). "Kata Containers" GitHub repository. 2018-2026. https://github.com/kata-containers/kata-containers
8. Google. "gVisor" GitHub repository README. 2018-2026. https://github.com/google/gvisor
9. Google. "Introduction to gVisor security" (g3doc/architecture_guide/intro_to_gvisor.md). 2018-2026. https://github.com/google/gvisor/blob/master/g3doc/architecture_guide/intro_to_gvisor.md
10. Google. "Performance Guide" (g3doc/architecture_guide/performance.md). 2019-2026. https://github.com/google/gvisor/blob/master/g3doc/architecture_guide/performance.md
11. Google. "Platform Guide" (g3doc/architecture_guide/platforms.md). 2019-2026. https://github.com/google/gvisor/blob/master/g3doc/architecture_guide/platforms.md
12. E2B. "E2B Infrastructure" GitHub repository README. 2023-2026. https://github.com/e2b-dev/infra
13. E2B. "Architecture" (docs/ARCHITECTURE.md). 2025-2026. https://github.com/e2b-dev/infra/blob/main/docs/ARCHITECTURE.md
14. E2B. "E2B" SDK GitHub repository README. 2023-2026. https://github.com/e2b-dev/E2B
15. Daytona. "Daytona" GitHub repository README. 2024-2026. https://github.com/daytonaio/daytona
16. Vercel. "Vercel Sandbox" GitHub repository README. 2025-2026. https://github.com/vercel/sandbox
17. Cloudflare. "Cloudflare Sandbox SDK" GitHub repository README. 2025-2026. https://github.com/cloudflare/sandbox-sdk
18. Anthropic. "Anthropic Sandbox Runtime (srt)" GitHub repository README. 2025-2026. https://github.com/anthropic-experimental/sandbox-runtime
19. Anthropic. "Beyond permission prompts: making Claude Code more secure and autonomous." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/claude-code-sandboxing
20. Martin, L., Cemaj, G., Cohen, M. (Anthropic). "Scaling Managed Agents: Decoupling the brain from the hands." Anthropic Engineering, April 8, 2026. https://www.anthropic.com/engineering/managed-agents
21. OpenAI. "Sandbox Agents" (docs/sandbox_agents.md), OpenAI Agents SDK for Python. 2026. https://github.com/openai/openai-agents-python/blob/main/docs/sandbox_agents.md
22. OpenAI. "Sandbox clients" (docs/sandbox/clients.md), OpenAI Agents SDK for Python. 2026. https://github.com/openai/openai-agents-python/blob/main/docs/sandbox/clients.md
23. Temporal Technologies. "Temporal" server GitHub repository README. 2019-2026. https://github.com/temporalio/temporal
24. Temporal Technologies. "Temporal server architecture" (docs/architecture/README.md). 2023-2026. https://github.com/temporalio/temporal/blob/main/docs/architecture/README.md
25. Temporal Technologies. "History Service" (docs/architecture/history-service.md). 2023-2026. https://github.com/temporalio/temporal/blob/main/docs/architecture/history-service.md
26. Temporal Technologies. "Workflow Lifecycle" (docs/architecture/workflow-lifecycle.md). 2023-2026. https://github.com/temporalio/temporal/blob/main/docs/architecture/workflow-lifecycle.md
27. Temporal Technologies. "Matching Service" (docs/architecture/matching-service.md). 2023-2026. https://github.com/temporalio/temporal/blob/main/docs/architecture/matching-service.md
28. Temporal Technologies. "Temporal AI integrations" GitHub repository. 2025-2026. https://github.com/temporalio/ai-integrations
29. Temporal Technologies. "OpenAI Agents SDK integration (temporalio-openai-agents)" README. 2025-2026. https://github.com/temporalio/ai-integrations/tree/main/python/openai_agents
30. Restate. "Restate" GitHub repository README. 2023-2026. https://github.com/restatedev/restate
31. Restate. "Durable Execution" concept documentation (docs/concepts/durable_execution.mdx). 2024-2025. https://github.com/restatedev/documentation/blob/main/docs/concepts/durable_execution.mdx
32. Restate. "Services" concept documentation (docs/concepts/services.mdx). 2024-2025. https://github.com/restatedev/documentation/blob/main/docs/concepts/services.mdx
33. Restate. "Durable Building Blocks" concept documentation (docs/concepts/durable_building_blocks.mdx). 2024-2025. https://github.com/restatedev/documentation/blob/main/docs/concepts/durable_building_blocks.mdx
34. Inngest. "Inngest" GitHub repository README (architecture section). 2021-2026. https://github.com/inngest/inngest
35. Inngest. "SDK Specification" (docs/SDK_SPEC.md). 2023-2026. https://github.com/inngest/inngest/blob/main/docs/SDK_SPEC.md
36. Inngest. "AgentKit" GitHub repository README. 2024-2026. https://github.com/inngest/agent-kit
37. DBOS, Inc. "DBOS Transact (Python)" GitHub repository README. 2024-2026. https://github.com/dbos-inc/dbos-transact-py
38. Cloudflare. "Cloudflare Agents" GitHub repository README. 2025-2026. https://github.com/cloudflare/agents
39. LangChain. "LangGraph" GitHub repository README. 2023-2026. https://github.com/langchain-ai/langgraph
40. LangChain. "LangGraph Checkpoint" library README (libs/checkpoint). 2024-2026. https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint
41. LangChain. "LangGraph Checkpoint Postgres" library README (libs/checkpoint-postgres). 2024-2026. https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres
42. Anthropic. "How we built our multi-agent research system." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/multi-agent-research-system
43. Anthropic. "Effective harnesses for long-running agents." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
44. Anthropic. "Effective context engineering for AI agents." Anthropic Engineering, September 29, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
45. OpenAI. "Tracing" (docs/tracing.md), OpenAI Agents SDK for Python. 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md
46. OpenTelemetry. "Semantic Conventions for Generative AI" GitHub repository. 2026. https://github.com/open-telemetry/semantic-conventions-genai
47. OpenTelemetry. "Semantic conventions for GenAI agent and framework spans" (docs/gen-ai/gen-ai-agent-spans.md). 2025-2026. https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md
48. Langfuse. "Langfuse" GitHub repository README. 2023-2026. https://github.com/langfuse/langfuse
49. Langfuse. "Self-hosting" documentation, architecture section (content/self-hosting/index.mdx). 2024-2026. https://github.com/langfuse/langfuse-docs/blob/main/content/self-hosting/index.mdx
50. LangChain. "LangSmith SDK" GitHub repository README. 2023-2026. https://github.com/langchain-ai/langsmith-sdk
51. AgentOps. "AgentOps" GitHub repository README. 2023-2026. https://github.com/AgentOps-AI/agentops
52. AgentOps. "AgentOps app" README (app/). 2025-2026. https://github.com/AgentOps-AI/agentops/tree/main/app
53. Arize AI. "Arize Phoenix" GitHub repository README. 2023-2026. https://github.com/Arize-ai/phoenix
54. Helicone. "Helicone" GitHub repository README. 2023-2026. https://github.com/Helicone/helicone
55. Anthropic. "Prompt caching." Claude Platform documentation, 2024-2026. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
56. OpenAI. "Usage" (docs/usage.md), OpenAI Agents SDK for Python. 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/usage.md
57. Ong, I., et al. (LMSYS). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv 2406.18665, 2024; and RouteLLM GitHub repository. https://github.com/lm-sys/RouteLLM
58. Chen, L., Zaharia, M., Zou, J. "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." arXiv 2305.05176, 2023 (TMLR 2024); and FrugalGPT GitHub repository. https://github.com/stanford-futuredata/FrugalGPT
59. BerriAI. "LiteLLM" GitHub repository README. 2023-2026. https://github.com/BerriAI/litellm
60. Schluntz, E., Zhang, B. (Anthropic). "Building effective agents." Anthropic Engineering, December 2024. https://www.anthropic.com/engineering/building-effective-agents
61. Anthropic. "Code execution with MCP: Building more efficient agents." Anthropic Engineering, November 4, 2025. https://www.anthropic.com/engineering/code-execution-with-mcp
62. Anthropic. "Writing effective tools for agents — with agents." Anthropic Engineering, September 11, 2025. https://www.anthropic.com/engineering/writing-tools-for-agents
