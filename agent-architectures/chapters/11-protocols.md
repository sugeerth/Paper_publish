# 11. Agent Protocols: MCP, A2A, ACP, AGNTCY, Agent Payments, Tool Registries, and Interop Patterns

## Overview

Between November 2024 and October 2026, agent interoperability went from bespoke glue code to a small set of open, foundation-governed protocols. Each protocol occupies a distinct layer:

- **Model Context Protocol (MCP)** connects an agent (the *host*) to tools, data, and prompts exposed by *servers*. Anthropic open-sourced it in November 2024 [1] and donated it to the Linux Foundation's Agentic AI Foundation (AAIF) in December 2025 [14][15]. In July 2026 it was rebuilt around a stateless core [3][10].
- **Agent2Agent (A2A)** lets opaque, independently built agents discover each other, delegate tasks, and stream results. Google announced it in April 2025 and contributed it to the Linux Foundation in June 2025 [30]. v1.0 shipped in March 2026 [28], and A2A joined AAIF in August 2026 [29].
- **IBM's Agent Communication Protocol (ACP)** was a REST-first agent-to-agent protocol behind the BeeAI platform. It was folded into A2A in 2025 [32][33].
- **AGNTCY**, an Internet-of-Agents stack hosted at the Linux Foundation [39], contributes the infrastructure around the protocols rather than a competitor to them: a schema (OASF), a federated directory, a secure messaging overlay (SLIM), and identity [34]-[38].
- **Agent payments** add a cryptographic authorization layer for money movement. Google's **Agent Payments Protocol (AP2)** [40]-[42] uses signed *mandates*. Coinbase-originated **x402** [43] revives HTTP 402 for internet-native payments and binds to HTTP, MCP, and A2A.
- **Registries and discovery** let clients find servers and agents before they connect: the MCP Registry [16][23], MCP Server Cards [21], A2A Agent Cards [31], the cross-protocol AI Catalog [45], the AGNTCY Directory [37], and ANP's DID-based discovery [46].

This chapter treats each protocol as a system design. For each one it covers the problem it solves, its components and data flow, its key techniques, the public scale numbers, its failure modes, and the lessons its designers have published. Unless stated otherwise, technical claims come from each project's normative specification as published in its repository in October 2026.

```
              THE 2026 AGENT PROTOCOL STACK (layered view)

 +--------------------------------------------------------------------------+
 | Commerce / authorization  AP2 mandates (SD-JWT VCs) | x402 (HTTP 402)    |
 +--------------------------------------------------------------------------+
 | Agent <-> agent           A2A v1.0 (JSON-RPC | gRPC | HTTP+JSON)         |
 |                           [IBM ACP merged in] [AGNTCY Agent Connect]     |
 +--------------------------------------------------------------------------+
 | Agent <-> tools/context   MCP 2026-07-28 (stdio | Streamable HTTP)       |
 |                           + extensions: Tasks, Apps, EMA, Server Cards   |
 +--------------------------------------------------------------------------+
 | Discovery / registries    MCP Registry | Agent Cards | AI Catalog |      |
 |                           AGNTCY Dir (OASF records, DHT) | ANP discovery |
 +--------------------------------------------------------------------------+
 | Identity / authz          OAuth 2.1 + RFC 9728/8707/9207, CIMD, ID-JAG,  |
 |                           signed Agent Cards (JWS), DIDs, SPIFFE/SPIRE   |
 +--------------------------------------------------------------------------+
 | Transport                 HTTP/1.1-2 + SSE, gRPC, stdio, SLIM (MLS E2E)  |
 +--------------------------------------------------------------------------+
```

---

## 11.1 Model Context Protocol (MCP)

### 11.1.1 Problem and origins

Anthropic introduced MCP on 25 November 2024 as "a new standard for connecting AI assistants to the systems where data lives." Its target was the N-by-M integration problem: "every new data source requires its own custom implementation" [1]. The launch included a specification and SDKs, local-server support in Claude Desktop, and reference servers for Google Drive, Slack, GitHub, Git, Postgres, and Puppeteer. Early adopters included Block, Apollo, Zed, Replit, Codeium, and Sourcegraph. MCP was created by David Soria Parra and Justin Spahr-Summers [1]. The specification credits the Language Server Protocol as its inspiration: MCP aims to do for AI-application integrations what LSP did for programming-language tooling [4].

Adoption numbers reported by the project itself:

- **December 2025:** more than 10,000 active public servers and 97M+ monthly SDK downloads across Python and TypeScript. First-class client support in ChatGPT, Claude, Cursor, Gemini, Microsoft Copilot, and VS Code [14][15].
- **November 2025:** the MCP Registry had "close to two thousand entries," a 407% increase since its September launch. The community counted 58 maintainers, 9 core/lead maintainers, and 2,900+ Discord contributors [19].
- **July 2026:** "close to half-a-billion downloads a month" across Tier 1 SDKs, with both the TypeScript and Python SDKs past 1 billion total downloads [10].

### 11.1.2 Spec revision timeline

| Revision | Headline changes |
|---|---|
| 2024-11-05 | Initial spec: JSON-RPC 2.0; stdio and HTTP+SSE (separate SSE and POST endpoints); resources, prompts, tools, sampling, roots [1][49] |
| 2025-03-26 | OAuth 2.1 authorization framework; **Streamable HTTP** replaces HTTP+SSE; JSON-RPC batching; tool annotations; audio content [48] |
| 2025-06-18 | Batching removed; structured tool output; servers classified as OAuth Resource Servers (RFC 9728 metadata); RFC 8707 resource indicators required; elicitation; `MCP-Protocol-Version` header [47] |
| 2025-11-25 | OIDC discovery; incremental scope consent; URL-mode elicitation; sampling with tools; **Client ID Metadata Documents** (SEP-991); experimental **tasks** (SEP-1686); SSE polling; JSON Schema 2020-12 default; formal governance [2] |
| 2026-07-28 | **Stateless core**: no `initialize` handshake, no `Mcp-Session-Id`; `server/discover`; **Multi Round-Trip Requests**; `subscriptions/listen`; `Mcp-Method`/`Mcp-Name` headers; cacheable list results; Tasks moved to an extension; Roots/Sampling/Logging deprecated; DCR deprecated in favor of CIMD; RFC 9207 issuer validation [3] |

### 11.1.3 Architecture: hosts, clients, servers, primitives

MCP uses a **client-host-server** architecture [4]:

- **Host.** The LLM application (an IDE, a chat app, an agent runtime). It creates and manages multiple clients, enforces consent and security policy, coordinates LLM integration, and aggregates context.
- **Client.** A connector inside the host. Each client talks to exactly one server, so the host-to-server relationship is 1:N through 1:1 clients.
- **Server.** A local process or remote service. It exposes *resources* (context and data), *prompts* (templated workflows that users invoke), and *tools* (functions that the model invokes).
- **Client-side features.** These are capabilities the client offers to servers: *elicitation* (asking the user for input), plus *sampling* (asking the host's LLM for a completion) and *roots* (filesystem scope). As of 2026-07-28, sampling and roots are deprecated [3].

Four stated design principles shape everything else [4]:

1. Servers should be extremely easy to build.
2. Servers should be highly composable.
3. Servers should not be able to read the whole conversation or "see into" other servers.
4. Features are added progressively through capability negotiation.

The third principle is the security core of MCP. The host holds the full conversation, and each server receives only what the host chooses to send.

```
           MCP 2026-07-28: HOST / CLIENT / SERVER TOPOLOGY

  +----------------------- Host process (e.g. IDE) ----------------------+
  |  LLM + conversation state + consent UI + policy                      |
  |   |               |                  |                               |
  | [Client 1]     [Client 2]        [Client 3]   (one client per server)|
  +---|---------------|------------------|-------------------------------+
      | stdio         | stdio            | Streamable HTTP (POST /mcp)
      v               v                  v
  [Server: files]  [Server: DB]     [Remote server] --- OAuth RS ---> [AS]
  tools/resources  tools            tools/resources/prompts
                                    _meta: protocolVersion,
                                    clientCapabilities, clientInfo
```

All messages are JSON-RPC 2.0. In the 2026-07-28 revision, every request carries `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` in `_meta`, and clients should also send `clientInfo`. Servers advertise capabilities through `server/discover`, which every server must implement and which clients may call before anything else [3][4]. There is no negotiation handshake. A server that does not support the requested version returns `UnsupportedProtocolVersionError` (code -32022) listing the versions it does support, and the client retries [3].

### 11.1.4 Transport evolution: stdio, HTTP+SSE, Streamable HTTP, stateless HTTP

**stdio** remains the local transport: the host launches the server as a subprocess and exchanges JSON-RPC over stdin/stdout. Servers may use stderr for logging [2]. stdio implementations should not follow the HTTP authorization spec and should take credentials from the environment instead [7].

**HTTP+SSE (2024-11-05)** required two endpoints: an SSE endpoint for the server-to-client stream and an HTTP POST endpoint for client messages [49]. It was replaced in 2025-03-26 by **Streamable HTTP**, which exposes a single MCP endpoint [48]. The 2025-era Streamable HTTP added several stateful features:

- A server-assigned `Mcp-Session-Id` header.
- An optional GET stream for server-initiated messages.
- SSE resumability through `Last-Event-ID`.
- Server-to-client JSON-RPC requests (sampling, elicitation, roots) sent over those SSE streams [3][11].

The Transport Working Group's December 2025 post describes what went wrong as enterprise deployments reached "millions of daily requests" [11]:

- **Infrastructure complexity.** Load balancers and API gateways had to parse full JSON-RPC bodies to route traffic.
- **Scaling friction.** Stateful connections forced sticky routing that blocked effective auto-scaling.
- **High barrier for simple tools.** Basic multi-turn interactions required backend session storage.
- **Ambiguous session scope.** There was no predictable boundary for where a session starts and ends.

The guiding idea was that "agentic applications are stateful, but the protocol itself doesn't need to be" [11].

The **2026-07-28 Streamable HTTP** transport is the result [5]:

- **One POST per message.** Every client message is its own HTTP POST to a single endpoint. The client must accept both `application/json` and `text/event-stream`. The server answers each request either with a single JSON object or with an SSE stream *scoped to that request*, which carries request-related notifications such as progress and ends with the final response.
- **No server-initiated requests on streams.** Servers must not send independent JSON-RPC requests on a stream. Sampling, elicitation, and roots travel inside results under MRTR (below).
- **Cancellation by disconnect.** Closing the SSE response stream *is* the cancellation signal on HTTP. `notifications/cancelled` is used only on stdio.
- **No resumability.** `Last-Event-ID` resumption is gone. "A broken response stream loses the in-flight request"; the client must re-issue it with a new request ID [3].
- **Routing headers.** The required headers `MCP-Protocol-Version`, `Mcp-Method`, and `Mcp-Name` (the tool, resource, or prompt name) mirror body fields. Gateways, rate limiters, and WAFs can route and meter without parsing JSON. A header/body mismatch returns 400 with a `HeaderMismatch` error [5][10].
- **Proxy and security hygiene.** Servers should send `X-Accel-Buffering: no` so reverse proxies do not buffer SSE, and should emit SSE comment keep-alives on long-lived streams. Servers must validate `Origin`, returning 403 on mismatch, to block DNS-rebinding attacks on local servers, and should bind to 127.0.0.1 when running locally [5].

```
   STREAMABLE HTTP, 2026-07-28 (stateless; any request -> any replica)

 Client                         L7 LB / gateway                Server replicas
   | POST /mcp                      |                               |
   | MCP-Protocol-Version: 2026-07-28                               |
   | Mcp-Method: tools/call         |-- route/authorize on headers->| R1 | R2 | R3
   | Mcp-Name: search               |   (no body parse, no sticky)  |
   | {jsonrpc, id:1, method, params:{..., _meta:{version, caps}}}   |
   |<------------- 200 application/json  (simple result) -----------|
   |<------------- 200 text/event-stream (progress..., result) -----|
   |   [close stream] ==> cancellation of that request              |
```

**State without sessions.** SEP-2567 removes protocol-level sessions entirely. List endpoints must no longer vary per connection. A server that needs cross-call state mints an explicit handle (a cart ID, a workflow ID) and the model passes it back as an ordinary tool argument [3][10]. The maintainers report that this "works better than session state hidden in the transport - the model can see the handle and thread it between tools" [10]. The new attack surface this creates, *state-handle hijacking*, has its own mitigation guidance (Section 11.1.8).

### 11.1.5 Multi Round-Trip Requests (MRTR)

The hardest problem in stateless MCP is a server that needs something from the client mid-call: user confirmation, a missing parameter, an LLM completion. Before 2026, the server suspended execution and sent a JSON-RPC request back over a held-open stream, so it had to track every outstanding request [11].

MRTR (SEP-2322) changes the server's reply. The server returns a result with `resultType: "input_required"`. The result contains `inputRequests`, a map from server-chosen keys to request objects such as `elicitation/create`, `sampling/createMessage`, or `roots/list`. It may also contain `requestState`, an opaque string the client must not inspect. The client gathers the answers and **retries the original request with a new JSON-RPC ID**, attaching `inputResponses` keyed the same way [6]. Every result now carries a required `resultType`, either `"complete"` or `"input_required"`. Clients treat a missing field from a legacy server as `"complete"` [3].

The model here is chat-completion APIs: the conversation state lives in the messages, not on the server. The server "reconstruct[s] the necessary state purely from the returned message, avoiding long-running state management between nodes" [11]. One production consequence: Supabase, whose MCP server runs statelessly, said elicitation had been on its roadmap but was impractical until MRTR made it possible, for example to confirm a project's cost before creating it [10]. URL-mode elicitation's `elicitationId` and the `notifications/elicitation/complete` notification were removed in the same revision. A server that needs to correlate an out-of-band interaction across retries now encodes its own identifier in `requestState` [3].

### 11.1.6 Notifications, caching, and determinism

- **Subscriptions.** `subscriptions/listen` replaces both the GET endpoint and `resources/subscribe`. It is a single long-lived POST response stream on which the client opts in to specific notification types (`toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions`). The server acknowledges the subscription and tags each notification with a `subscriptionId`. Request-scoped notifications such as progress and log messages flow only on their own request's stream [3][5].
- **Cacheable lists.** Results from `tools/list`, `prompts/list`, `resources/list`, `resources/read`, and `resources/templates/list` must carry `ttlMs`, a freshness hint, and `cacheScope` (`public` or `private`), which controls whether shared intermediaries may cache them. Notifications become an optimization rather than a correctness requirement [3][11].
- **Deterministic tool order.** Servers should return `tools/list` in a deterministic order "to enable client-side caching and improve LLM prompt cache hit rates" [3]. This is a small change with real cost impact, because a reordered tool list invalidates the model provider's prompt cache.
- **Observability.** `_meta` conventions now carry W3C trace context (`traceparent`, `tracestate`, `baggage`) for OpenTelemetry propagation (SEP-414) [3].

### 11.1.7 Authorization

MCP authorization is optional. For HTTP transports it profiles a selected subset of OAuth [7]. The roles are:

- **MCP server:** an OAuth 2.1 *resource server*.
- **MCP client:** an OAuth 2.1 *client*.
- **Authorization server (AS):** may be co-hosted with the MCP server or separate.

The normative building blocks are OAuth 2.1 (draft-ietf-oauth-v2-1-13), RFC 6750 bearer tokens, RFC 8414 AS metadata, RFC 9728 Protected Resource Metadata (PRM), RFC 8707 resource indicators, RFC 9207 issuer identification, OAuth Client ID Metadata Documents, OIDC Discovery 1.0, and (deprecated) RFC 7591 Dynamic Client Registration [7].

```
   MCP AUTHORIZATION FLOW (2026-07-28)

 Client                 MCP server (RS)              Authorization server
   |-- request, no token ->|                                 |
   |<- 401 WWW-Authenticate: Bearer resource_metadata=..., scope="files:read"
   |-- GET /.well-known/oauth-protected-resource ->|         |
   |<- PRM {authorization_servers:[...]} ----------|         |
   |-- GET AS metadata (RFC 8414 or OIDC discovery) -------->|
   |   (refuse if code_challenge_methods_supported absent)   |
   |   client_id = https://app.example/oauth.json (CIMD) --->|-- fetch metadata
   |-- browser: authorize?code_challenge(S256)&resource=<MCP server URI>
   |<- redirect: code + iss  (client MUST validate iss == recorded AS)
   |-- token: code_verifier + resource ---------------------->|
   |<- access token (audience = this MCP server) -------------|
   |-- request + Bearer token ->| validate audience; never pass token through
```

Key techniques and the reasons behind them:

- **Discovery by challenge.** A 401 with `WWW-Authenticate` points to PRM. Since SEP-985 the header is optional, with fallback to the `.well-known` path. The challenged `scope` is authoritative for the current operation, which enables **step-up** (incremental) consent: on re-authorization, clients add the newly challenged scopes to the ones already granted [2][7].
- **PKCE is mandatory, and clients must verify AS support for it.** If AS metadata lacks `code_challenge_methods_supported`, the client must refuse to proceed. S256 is required when technically possible [8].
- **Audience binding.** Clients must send RFC 8707 `resource` on authorization and token requests. Servers must reject tokens not issued for them, and must not pass a received token through to upstream APIs. An upstream call requires a separate token from the upstream AS [8][9].
- **Mix-up defense.** Since 2026-07-28, an AS should return `iss` (RFC 9207), and clients must check it against the recorded issuer before redeeming the code. Client credentials are bound to the issuer that minted them and must not be reused across authorization servers [3][10].

**Client registration** is the part most specific to MCP. Users pick the *resource server* by pasting a URL, so an unbounded set of clients meets an unbounded set of authorization servers [12]. The core maintainers set out the problems with DCR in August 2025 [12]:

- For authorization servers: unbounded database growth (one registration per client instance per machine), a client-expiry "black hole," per-instance audit noise, and an unauthenticated `/register` write endpoint that invites denial of service.
- For clients: extra state to manage and no way to check whether an ID is still valid.

The fix is **Client ID Metadata Documents (CIMD)**, whose design is credited to Bluesky's implementation. The `client_id` *is* an HTTPS URL, and the AS fetches the client's metadata (name, redirect URIs) from it at authorization time. This gives one ID per application, no expiry, and no registration endpoint. The cost is that clients must host a document, and authorization servers must defend against SSRF when fetching arbitrary URLs [12][8]. CIMD became the recommended mechanism in 2025-11-25, and DCR was formally deprecated in 2026-07-28 [2][3].

CIMD does *not* solve impersonation of localhost redirect URIs. Authorization servers must clearly display the redirect host, should warn on localhost-only redirect URIs, and may require attestation. The maintainers propose signed **software statements** (backend-issued JWTs verified against a JWKS), and eventually OS-level attestation, as complementary layers [8][12].

**Enterprise-Managed Authorization (EMA).** EMA became a stable extension in June 2026. It targets the "per-user authorization tax": every employee consenting to every server, no central policy, and work and personal accounts mixed together. During SSO, the client obtains an **Identity Assertion JWT Authorization Grant (ID-JAG)** from the corporate IdP and exchanges it at the MCP server's AS for an access token, with no per-server consent screen. Launch participants included Okta (through Cross App Access), Anthropic (across Claude, Claude Code, and Cowork), VS Code, and servers from Asana, Atlassian, Canva, Figma, Granola, Linear, and Supabase [13].

The August 2026 roadmap names **agent identity** as the next frontier, because callers are increasingly cloud workloads acting for absent users or delegating to sub-agents. Planned work includes DPoP (RFC 9449), workload identity federation, and token exchange, coordinated with the IETF OAuth and WIMSE working groups [17].

### 11.1.8 Extensions

SEP-2133 formalized **extensions**: optional, namespaced (`io.modelcontextprotocol/...`) features negotiated through an `extensions` field in client and server capabilities. Extensions can iterate outside the core release cadence; breaking changes require a new identifier [3][21].

- **Tasks** (`io.modelcontextprotocol/tasks`, SEP-2663; AWS describes Tasks as contributed by AWS). A server may answer `tools/call` with a durable *task handle* instead of a result, and the client polls with `tasks/get`, provides mid-flight input with `tasks/update`, or stops it with `tasks/cancel` [20][10]. SEP-2663 states three lessons from the 2025-11-25 experimental design:
  - The per-method and per-tool opt-in handshake was "fragile," because clients had to call `tools/list` before any task-augmented request.
  - The blocking `tasks/result` was "a blocking trap" that forced long-lived connections.
  - `tasks/list` could not be scoped safely once sessions were gone, because a task ID was the only line of defense against cross-tenant contamination.

  Under the redesign, servers may return tasks unsolicited, and `tasks/list` and client-hosted tasks are removed [20].
- **MCP Apps** (SEP-1865). Servers declare interactive HTML UI resources under the `ui://` scheme (`text/html;profile=mcp-app`), reference them from tool metadata, and render them in sandboxed iframes that talk to the host over MCP JSON-RPC. The design consolidates the community MCP-UI project and OpenAI's Apps SDK (launched November 2025, built on MCP) [22].
- **Server Cards** (SEP-2127) are static, cacheable metadata documents for discovery before connecting (Section 11.6).
- **Enterprise-Managed Authorization**, described above [13].

### 11.1.9 Failure modes and security lessons

The MCP security best-practices document lists the attacks implementers actually hit [9]:

- **Confused deputy.** An MCP proxy server that uses a static client ID toward a third-party AS can be exploited with stolen authorization codes to skip user consent. Mitigation: per-client consent before forwarding.
- **Token passthrough.** Accepting tokens without audience validation and forwarding them downstream. This is explicitly forbidden [8][9].
- **SSRF.** Through PRM and AS metadata URLs and CIMD fetches, which may target internal networks.
- **State-handle hijacking.** New with stateless MCP: guessing or stealing a server-minted handle. Servers must not treat possession of a handle as authentication. They should use non-deterministic handles bound server-side to the token-derived user ID, for example by keying state as `<user_id>:<handle>` [9].
- **Local server compromise, stdio proxy abuse, malicious authorization URLs, mix-up attacks, localhost redirect impersonation, and over-broad scopes** [9].

**Tool metadata is untrusted input.** The spec says annotations "should be considered untrusted, unless obtained from a trusted server" [4]. A March 2026 maintainer post (with GitHub and AWS co-authors) describes what the four `ToolAnnotations` hints (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`, with pessimistic defaults) can and cannot do [18]:

- They can drive confirmation prompts and feed policy engines.
- They cannot stop prompt injection, and an untrusted server can simply lie.
- Risk is "a property of the session, not of any single server." The post frames this with Simon Willison's *lethal trifecta*: private-data access plus untrusted content plus an exfiltration channel. MCP makes it easy to assemble all three from several servers.

One deployment data point: GitHub's read-only mode is enabled by about 17% of users [18]. Five SEPs proposing trust and sensitivity annotations were open as of that post.

### 11.1.10 Context cost at scale: tool search and code execution

At scale, MCP's dominant cost is tokens rather than protocol overhead. Anthropic engineering reported these figures [24][25]:

- **Definition bloat.** A five-server setup (GitHub, Slack, Sentry, Grafana, Splunk; 58 tools) consumed about 55K tokens of definitions "before the conversation even starts." Jira alone adds about 17K. Anthropic saw definitions consume 134K tokens internally before optimization [25].
- **Tool Search Tool.** Tools are marked `defer_loading: true` and discovered on demand by regex, BM25, or custom search. Context consumption fell from about 77K to about 8.7K tokens, roughly an 85% reduction. Accuracy on MCP evaluations rose from 49% to 74% for Opus 4 and from 79.5% to 88.1% for Opus 4.5. Deferred tools do not break prompt caching because they are absent from the initial prompt [25].
- **Code execution with MCP.** MCP servers are presented as a filesystem of typed code modules that the agent imports, so intermediate results stay in the sandbox instead of passing through the model. In Anthropic's example, a workflow fell from 150,000 to 2,000 tokens, a 98.7% reduction. A two-hour meeting transcript copied between tools would otherwise cost about 50,000 extra tokens. The client can also tokenize PII so that real values flow between tools without entering the model [24].
- **Programmatic tool calling** reduced average usage from 43,588 to 27,297 tokens (37%) on complex research tasks [25].

The trade-off is that code execution requires a sandbox, resource limits, and monitoring [24]. MCP's 2026 roadmap adopts the same lesson at protocol level: a **progressive discovery** effort so that a server exposes a small entry point and reveals more of its catalog as the conversation narrows. The roadmap also aims to standardize which form of a `tools/call` result is placed in front of the model [17].

### 11.1.11 Lessons from MCP's evolution

1. **Shipping a stateful bidirectional protocol and then retrofitting statelessness was expensive.** It took a breaking revision, migration notes across four Tier 1 SDKs, and a twelve-month deprecation policy [3][10]. Production feedback converged on web-native properties: stateless, cacheable, header-routable.
2. **Put state where the model can see it.** Explicit handles and request-carried state (MRTR `requestState`) replaced hidden transport sessions [10].
3. **Authorization dominates integration time** [10]. The ecosystem's distinctive topology, user-chosen resource servers meeting unknown clients, forced new OAuth practice (CIMD, issuer-bound credentials) rather than simple adoption of existing OAuth.
4. **Keep the core small and let extensions incubate.** Tasks were moved *out* of the core after one release of real-world feedback [20].

---

## 11.2 Agent2Agent Protocol (A2A)

### 11.2.1 Problem and positioning

A2A addresses "communication and interoperability between independent, potentially opaque AI agent systems." Agents collaborate "as agents, not just as tools," without sharing "internal state, memory, or tools" [26]. Its guiding principles are [26]:

- Simple (reuse HTTP, JSON-RPC 2.0, and SSE).
- Enterprise-ready.
- Async-first (very long-running and human-in-the-loop tasks).
- Modality-agnostic.
- Opaque execution.

The project describes MCP as the *vertical* integration layer (agent to tools) and A2A as the *horizontal* layer (agent to agent): "MCP inside agents, A2A between agents" [28][29].

Timeline:

- **April 2025:** announced by Google.
- **June 2025:** contributed to the Linux Foundation.
- **July 2025:** v0.3 added gRPC, signed Agent Cards, and extended Python SDK client support, with "over 150 organizations" in the ecosystem [30].
- **March 2026:** v1.0. The Technical Steering Committee includes AWS, Cisco, Google, IBM Research, Microsoft, Salesforce, SAP, and ServiceNow [28].
- **August 2026:** accepted as a Growth Stage project at AAIF. The project reports native support in Google Cloud, AWS Bedrock AgentCore Runtime, and Azure AI Foundry, and framework support in LangGraph, CrewAI, Pydantic AI, AG2, and IBM BeeAI [29].

### 11.2.2 Data model

The core objects are defined in `a2a.proto`. Since v1.0 this file is the single normative source for all bindings [27].

- **Agent Card.** A JSON self-description containing identity, `supportedInterfaces[]` (each with `url`, `protocolBinding`, and `protocolVersion`), capabilities (`streaming`, `pushNotifications`, `extendedAgentCard`, `extensions`), `securitySchemes`, default input and output modes, and `skills[]` [26][27].
- **Message.** A turn with a `role` (user or agent) and one or more **Parts** (text, file reference, or structured data).
- **Task.** A server-generated, stateful unit of work. Clients cannot create task IDs [26].
- **Artifact.** A task output made of Parts.
- **contextId.** Groups related tasks and messages into one conversation. Agents may use it to hold LLM context and must reject a message whose `contextId` and `taskId` disagree [26].

The task states are `SUBMITTED`, `WORKING`, `INPUT_REQUIRED`, `AUTH_REQUIRED`, `COMPLETED`, `FAILED`, `CANCELED`, and `REJECTED` [27].

```
            A2A v1.0 TASK LIFECYCLE (TaskState enum)

   SendMessage --> SUBMITTED --> WORKING --+--> COMPLETED
                      |             ^  |   +--> FAILED
                      v             |  |   +--> CANCELED (CancelTask)
                   REJECTED         |  +--> INPUT_REQUIRED --(client msg, same taskId)
                                    |  +--> AUTH_REQUIRED  --(credential out-of-band)
                                    +-------------------------------------+
```

### 11.2.3 Operations, bindings, and update delivery

v1.0 separates the abstract operations from three *functionally equivalent* bindings: JSON-RPC, gRPC, and HTTP+JSON/REST. Agents declare every binding they support, and clients choose among them and should implement fallback [26].

| Operation | JSON-RPC / gRPC | REST |
|---|---|---|
| Send message | `SendMessage` | `POST /message:send` |
| Stream | `SendStreamingMessage` | `POST /message:stream` |
| Get / list tasks | `GetTask` / `ListTasks` | `GET /tasks/{id}` / `GET /tasks` |
| Cancel | `CancelTask` | `POST /tasks/{id}:cancel` |
| Resubscribe | `SubscribeToTask` | `POST /tasks/{id}:subscribe` |
| Webhooks | `Create/Get/List/DeleteTaskPushNotificationConfig(s)` | `/tasks/{id}/pushNotificationConfigs` |
| Extended card | `GetExtendedAgentCard` | `GET /extendedAgentCard` |

A2A offers three ways to receive updates [26]:

- **Polling** with `GetTask`.
- **Streaming**, which requires `capabilities.streaming`. Events must be delivered in order. Multiple concurrent streams per task all receive the same ordered events, and closing one stream does not affect the task.
- **Push notifications:** HTTP POST webhooks, which require `capabilities.pushNotifications`. These suit server-to-server and disconnected clients.

v1.0 removed the `final` flag on status events in favor of binding-level stream closure. It also added `ListTasks` with cursor pagination and native **multi-tenancy** through a `tenant` field on every request, so one endpoint can host many agents [27].

### 11.2.4 Security and trust

A2A deliberately keeps identity "at the protocol layer, not within A2A semantics" [26]:

- Production deployments must use TLS, with TLS 1.3+ recommended.
- Clients learn the server's requirements from `securitySchemes`: API key, HTTP auth, OAuth2, OIDC, or mutual TLS.
- Clients obtain credentials out of band and present them on every request, and servers must authenticate every request.
- v1.0 removed the OAuth implicit and password flows and added the Device Code flow (RFC 8628) and a `pkce_required` flag [27].

**Signed Agent Cards.** A card may carry JWS signatures (RFC 7515) over a canonical form produced with the JSON Canonicalization Scheme (RFC 8785). Canonicalization respects protobuf field-presence rules, so a card rebuilt from protobuf verifies byte-for-byte, and the `signatures` field is excluded from the signed content [26].

**In-task authorization.** When an agent needs a credential mid-task (an OAuth token for a downstream API, or human approval of a destructive action), it moves the task to `AUTH_REQUIRED` and receives the credential out of band. A client that is itself an A2A agent may push the request upstream by moving its *own* task to `AUTH_REQUIRED`, which forms a chain. The spec warns that in-band credential passing exposes credentials to every agent in that chain [26].

### 11.2.5 Discovery

The Agent Card lives at `https://{domain}/.well-known/agent-card.json` (RFC 8615). Curated registries and direct configuration are documented alternatives, but "the current A2A specification does not prescribe a standard API for curated registries" [31]. Sensitive skills or internal URLs belong in the authenticated **extended Agent Card**. Card endpoints should send `Cache-Control` and `ETag` headers so clients can make conditional requests [31].

### 11.2.6 Failure modes and lessons

- **Versioning churn.** v1.0 was a breaking cleanup:
  - Methods were renamed (`message/send` became `SendMessage`).
  - Enums moved from kebab-case to SCREAMING_SNAKE_CASE to comply with ProtoJSON.
  - `kind` discriminators were removed in favor of member-based polymorphism.
  - Compound IDs (`tasks/{id}`) gave way to plain UUIDs.
  - `protocolVersion` moved onto each interface [27].

  The mitigation is that an Agent Card can advertise v0.3 and v1.0 interfaces at the same time, so clients can migrate progressively [28].
- **Push-delivery reachability.** Webhooks require the client to be reachable over HTTP. Streaming requires persistent connections. Polling adds latency and wasted requests [26].
- **Opaqueness cuts both ways.** Delegating to an agent whose plan and tools are hidden makes the Agent Card's claims, signatures, and the server's authorization policy the only levers on trust.

---

## 11.3 IBM Agent Communication Protocol (ACP)

IBM Research's ACP, which powered the BeeAI Platform, was "an open protocol for communication between AI agents, applications, and humans" [32]. Unlike JSON-RPC-based protocols, it was **REST-first**, defined by an OpenAPI document. Its main features:

- **Agent Manifest.** Name, description, metadata, and status, served at `/agents`. Discovery could be online (querying a server), open (a `.well-known/agent.yml` file), registry-based, or **embedded**, with the manifest baked into container image labels so that advertised capabilities stay in sync with the build [32].
- **Runs.** A run is a single execution, created with `POST /runs` in `sync`, `async`, or `stream` mode. Its states are `created`, `in-progress`, `awaiting`, `completed`, `cancelling`, `cancelled`, and `failed`. The **Await** mechanism pauses a run for client input, which arrives through `POST /runs/{run_id}`. An awaiting run times out to `failed` [32].
- **Distributed sessions.** A session descriptor holds URLs for history messages and state stored on arbitrary resource servers such as S3. Moving a session between ACP servers then means forwarding only the descriptor. This was an early answer to the problem MCP solved in 2026 with explicit handles [32].
- **Trajectory and citation metadata** on message parts, and high-availability deployment with Redis or PostgreSQL [32].

**Outcome.** By August 2025 the repository announced that "ACP is now part of A2A under the Linux Foundation," with a migration guide [32][33]. ACP's ideas (REST-friendly bindings, await/input-required semantics, embedded discovery) line up with features A2A formalized in v1.0, though the A2A sources do not attribute specific features to ACP. ACP should not be confused with AGNTCY's *Agent Connect Protocol*, which shares the acronym.

---

## 11.4 AGNTCY: Infrastructure for an "Internet of Agents"

AGNTCY is an open-source collective, part of the Linux Foundation [39], that builds the *infrastructure* around agent protocols rather than a single wire protocol. Its components have different maturity levels.

**Agent Connect Protocol (AGNTCY ACP).** An OpenAPI specification (v0.2.3) "to invoke and configure remote agents over an API" [34]. Its surface includes agent search and descriptors (`/agents/search`, `/agents/{id}/descriptor`), stateless runs (`/runs`, `/runs/wait`, `/runs/stream`), and stateful **threads** with history, copy, and per-thread runs (`/threads/{id}/runs/...`). The repository's last activity was in May 2025. The later AGNTCY components carry A2A and MCP rather than replacing them, which suggests the ecosystem converged on those protocols.

**OASF (Open Agentic Schema Framework).** A schema system modeled on the Open Cybersecurity Schema Framework (OCSF). Its core *record* object is annotated with hierarchical **skills** and **domains** taxonomies and extended with composable **modules**. A discovery extension lets A2A Agent Cards and AI Catalog entries carry the same skills and domains. Released schema versions are immutable [36].

**Agent Directory Service (dir).** A federated registry for MCP servers, A2A agents, and Agent Skills [37]:

- Records are OASF documents stored **content-addressed**, by CID, in an OCI registry such as Zot.
- A **distributed hash table (DHT)** maps skills, domains, modules, and locators to record IDs, so discovery is capability-based and needs no central index.
- Records carry cryptographic signatures and provenance ("verifiable claims").
- A runtime-discovery component watches Docker and Kubernetes and resolves Agent Cards or OASF records from running workloads.
- The directory is described as implementing the ARD (Agentic Resource Discovery) specification [37].

**SLIM (Secure Low-Latency Interactive Messaging).** "The secure transport layer for A2A, MCP, and SLIMRPC" [35]. Its premise is that message queues are one-directional and broker-readable, while HTTP and gRPC lack group communication and end-to-end encryption and require agents to be internet-exposed. SLIM's design [35]:

- **gRPC over HTTP/2** for multiplexed, NAT-friendly transport.
- **MLS (RFC 9420)** for end-to-end encryption at the session layer, so a compromised routing node cannot read content.
- **Hierarchical DID-based names** (`organization/namespace/service/<hash-of-public-key>`).
- Native **group and multicast** channels.
- **SPIRE-based** cross-organization identity federation.
- **SLIMRPC**, which provides protobuf RPC including multicast fan-out RPC.
- A split between data plane, session layer, and control plane, so routing nodes run only the data plane.

```
   SLIM OVERLAY CARRYING A2A / MCP (per AGNTCY docs)

  [Agent A: A2A client]                         [Agent B: A2A server]
   SLIM SDK: session layer (MLS E2E) + SLIMRPC   SLIM SDK (MLS E2E)
        |  gRPC/HTTP2                                 ^
        v                                             |
   [SLIM data-plane node] --routes by hierarchical name--> [SLIM node]
        ^ control plane: routes, node registration, group membership
        | identities: DID-style names; SPIRE federation across orgs
```

**AGNTCY Identity.** Assigns agents, MCP servers, and multi-agent systems universally unique IDs backed by verifiable credentials. It supports "bring your own ID": IdP-issued identities such as Okta, A2A Agent Cards, or W3C DIDs. Each ID is linked to ResolverMetadata for automated verification [38].

**Lesson.** AGNTCY's most durable contributions are the layers MCP and A2A leave open: a capability taxonomy, decentralized discovery, E2E-encrypted multi-party transport, and verifiable identity.

---

## 11.5 Agent Payments: AP2, x402, and Mandates

### 11.5.1 Problem

Existing payment rails assume "a human user directly interacting with a trusted interface." When an agent pays, merchants, issuers, and networks cannot answer basic questions [40][41]:

- Did the user authorize *this* purchase?
- Does the request reflect true intent, or a hallucination?
- Who is liable?

AP2 answers with "verifiable intent, not inferred action": every transaction is anchored to deterministic, non-repudiable proof [41].

### 11.5.2 AP2 v0.1 (September 2025)

Google announced AP2 on 16 September 2025 with more than 60 organizations, including Adyen, American Express, Coinbase, Mastercard, PayPal, Revolut, Salesforce, ServiceNow, UnionPay International, and Worldpay [40]. v0.1 defined three verifiable-credential *mandates*:

- **Intent Mandate.** Signed by the user. It captures shopping intent, price limits, a TTL, and "prompt playback," and authorizes human-not-present purchases.
- **Cart Mandate.** Signed by the user, typically with a hardware-backed key. It records the exact items, price, and payment method in human-present purchases.
- **Payment Mandate.** Shared with networks and issuers, it signals AI-agent presence and the transaction modality [40][41].

At the same launch, Google, Coinbase, the Ethereum Foundation, MetaMask, and others released the **A2A x402 extension** for agent-based crypto payments [40][44].

### 11.5.3 AP2 v0.2 (April 2026): Checkout and Payment Mandates

The v0.2 specification (released 28 April 2026) generalizes the design into an **agent authorization framework** and narrows the mandates to two [41][42]:

- **Five roles:**
  - Shopping Agent (SA), which is assumed to be agentic.
  - Credential Provider (CP).
  - Merchant (M).
  - Merchant Payment Processor (MPP).
  - Trusted Surface (TS), which *must* be non-agentic.

  When either side of an exchange is agentic, "the Agent itself is a potential attacker," so tamper-evident mechanisms are needed. Every verification must run in deterministic code even inside an agentic role [41].
- **Checkout Mandate.** Bound by hash to a *merchant-signed* Checkout JWT and verified by the merchant. It returns a Checkout Receipt.
- **Payment Mandate.** Bound to the same Checkout hash and verified by the CP, the network, and the MPP. It returns a signed Payment Receipt. The Checkout JWT must use a non-deterministic signature scheme such as ECDSA (not Ed25519) to resist rainbow-table attacks on the hash binding [41].
- **Credential format.** Mandates are **SD-JWT VCs** (`vct` values such as `mandate.payment.1`, with versioned suffixes) presented through OpenID4VP, with **key binding** for proof of possession. Delegation chains follow a "Delegate SD-JWT" draft, and selective disclosure means verifiers see only the constraints relevant to them [41][42].
- **Two modes:**
  - *Human present:* the user signs the closed mandates on the Trusted Surface.
  - *Human not present:* the user signs **open** mandates with constraints and the agent's public key in a `cnf` claim. The agent later signs closed mandates with that key, and verifiers receive both.

  An agent must not present a new open mandate until it has received a rejection receipt for the previous one, which prevents double-spending one authorization across checkouts [41].
- **Dispute evidence.** Mandates and receipts together form a non-repudiable record for network adjudication [41].
- **Scope.** AP2 is explicitly a security layer *within* a commerce protocol and is designed to be compatible with the Universal Commerce Protocol (UCP). Catalog and checkout APIs are out of scope [41].

```
   AP2 v0.2 HUMAN-NOT-PRESENT FLOW (simplified)

 User --(consent on Trusted Surface)--> open Checkout+Payment Mandates
                                         {constraints, cnf: agent pubkey, exp}
 Shopping Agent -- builds checkout --> Merchant -- signs Checkout JWT --> SA
 SA signs CLOSED mandates (agent key, hash(Checkout JWT))
 SA --> Credential Provider/Network: open+closed Payment Mandate -> credential
 SA --> Merchant: credential + open+closed Checkout Mandate -> Checkout Receipt
 Merchant --> MPP: charge; MPP verifies Payment Mandate -> Payment Receipt
```

### 11.5.4 x402

x402 is "an open standard for internet native payments" for crypto and fiat. Its principles include being HTTP- and transport-native, network-agnostic, and **trust-minimizing**: a facilitator or server can never move funds except "in accordance with client intentions" [43]. The repository has moved to the x402 Foundation organization [43]. v2 of the spec, dated 9 December 2025, defines:

- **Types, logic, and representation.** Transport- and scheme-independent types (`PaymentRequired`, `PaymentPayload`, `SettlementResponse`). Scheme logic: the core spec defines `exact`, and the repository adds `upto` and `batch-settlement` scheme specs. Networks are identified with CAIP-2 IDs [43].
- **The HTTP flow.** The server answers `402 Payment Required` with a base64 `PAYMENT-REQUIRED` header. The client retries with a signed `PAYMENT-SIGNATURE` header, for example an EIP-3009 transfer authorization in the `exact` scheme. The server verifies, settles, and returns `PAYMENT-RESPONSE` [43].
- **Facilitator API.** `POST /verify`, `POST /settle`, and a listing of supported schemes. The facilitator absorbs chain details such as gas and RPC so that the integration is "1 line for the server, 1 function for the client" [43].
- **MCP binding.** A paid tool returns `isError: true` with `PaymentRequired` in both `structuredContent` and `content[0].text`. The client retries with `_meta["x402/payment"]`, and settlement comes back in `_meta["x402/payment-response"]` [43].
- **A2A binding.** The a2a-x402 extension uses `payment-required`, `payment-submitted`, and `payment-completed` messages [44].

**AP2 and x402 compared.** AP2 proves *authority and intent* across card, bank, and crypto rails. x402 *moves value* inline with an HTTP, MCP, or A2A request. Google's FAQ positions x402 as composable with AP2 for stablecoin payments [41]. ANP has an AP2 adaptation draft that is explicitly not yet stable [46].

---

## 11.6 Tool and Agent Registries, Catalogs, and Discovery

Discovery has split into three tiers: **package and server registries**, **static well-known cards**, and **capability-search networks**.

**MCP Registry** (preview September 2025; API frozen at v0.1 in October 2025) [16][23]:

- **Architecture.** A Go REST API with PostgreSQL behind a CDN, built on the assumption that consumers poll daily and cache. The design tolerates "24h acceptable" downtime [23].
- **Metadata only.** Code is distributed through npm, PyPI, NuGet, Docker Hub, and similar registries, and `server.json` records point to those packages or to remote endpoints.
- **Namespaces tied to authentication.** GitHub OAuth grants `io.github.<user>/*`, and publishing under `io.github.<org>/*` requires the org *Owner* role. Domain verification uses a signed timestamp checked against a `v=MCPv1; k=...; p=...` DNS TXT record and grants `com.example.*/*` [23].
- **Neutrality.** No ranking or curation. Public and private **sub-registries** (client marketplaces, enterprise allowlists) ingest the upstream data and add their own curation. Moderation is limited to denylisting spam, malware, and impersonation [16][23].

**MCP Server Cards** (SEP-2127). Static, public, cacheable documents that describe a remote server's identity, transport endpoints, and supported protocol versions *before* connection. They are recommended at `<streamable-http-url>/server-card`. They enable IDE autoconfiguration, crawling, and fast UI hydration. If a card disagrees with the live `server/discover` result, clients should trust `server/discover` [21].

**A2A Agent Cards.** Published at `/.well-known/agent-card.json`, with optional JWS signatures and an authenticated extended variant [26][31].

**AI Catalog.** A cross-protocol effort by members of the MCP, A2A, and other communities. It defines a typed, nestable JSON container at `/.well-known/ai-catalog.json`. Each entry identifies an artifact by media type (an MCP server card, an A2A card, a Claude Code plugin, a dataset, a model card, or a nested catalog), and the catalog adds an optional Trust Manifest and a `did:web` publisher identity. It "does not replace or redefine protocol-specific artifact formats" [45]. Server Cards and the AGNTCY Directory both reference it [21][37].

**AGNTCY Directory.** OASF records, content addressing, and DHT capability routing (Section 11.4) [37].

**ANP.** Pursues a DID-native approach: `did:wba` web-based DIDs, Agent Description documents, `.well-known` active discovery, and passive registration with search agents [46].

**Client-side registries.** At runtime the "registry" problem reappears inside the host as tool selection. Deferred loading plus search (Section 11.1.10) is the in-context counterpart to these network registries [25].

```
   DISCOVERY TIERS

  Publish time                     Pre-connect                  Runtime
  ------------                     -----------                  -------
  MCP Registry (server.json) --->  Server Card / Agent Card --> server/discover
  AGNTCY Dir (OASF, CID, DHT)      /.well-known/ai-catalog.json  GetExtendedAgentCard
  Sub-registries / allowlists      (signed, cacheable)          Tool search (deferred)
```

---

## 11.7 Interoperability Patterns

1. **MCP inside, A2A between.** An A2A server agent fulfills delegated tasks by calling tools over MCP. Both specifications endorse this split [26][28].
2. **Agents as tools, and tools behind agents.** Long-running MCP tool calls can now return task handles through the Tasks extension [20]. A2A tasks already carry lifecycle state [26]. Bridges therefore map MCP `tasks/get` and `tasks/update` onto A2A `GetTask` and an `INPUT_REQUIRED` follow-up message. This mapping is our synthesis; neither spec defines a bridge.
3. **Payments as protocol extensions.** x402 defines transport bindings for HTTP, MCP, and A2A [43]. AP2 describes itself as an extension for A2A, MCP, and UCP [41]. Payment logic stays out of the core protocols.
4. **Gateway-friendly envelopes.** MCP's `Mcp-Method` and `Mcp-Name` headers and A2A's REST binding allow L7 routing, rate limiting, WAF rules, and per-tool authorization without parsing bodies [5][26]. Microsoft Foundry describes a single MCP endpoint that unifies tools under centralized governance [10].
5. **Explicit state over sessions.** MCP handles plus MRTR `requestState`, A2A `contextId`/`taskId`, and ACP distributed-session descriptors all move continuity into data that any replica can process [6][26][32].
6. **Identity federation instead of per-app consent.** MCP EMA (ID-JAG) [13], A2A `securitySchemes` with mTLS and OAuth [27], AGNTCY's DIDs, VCs, and SPIRE [35][38], and AP2's key-bound SD-JWT mandates [42] all lead toward agents that carry verifiable, delegable identity.
7. **Overlay transports.** SLIM carries A2A and MCP over an MLS-encrypted, multicast-capable overlay when HTTP's point-to-point, internet-exposed model does not fit [35].
8. **Progressive disclosure everywhere.** This covers Server Cards before `server/discover` before full listings [21]; Agent Card before extended card [31]; and tool search before full definitions [25].

---

## 11.8 Comparison

| Dimension | MCP (2026-07-28) | A2A (v1.0) | IBM ACP | AGNTCY Agent Connect | AP2 (v0.2) | x402 (v2) |
|---|---|---|---|---|---|---|
| Primary purpose | Agent to tools/context | Agent to agent delegation | Agent/app/human messaging | Invoke/configure remote agents | Authorize agent payments | Inline internet payments |
| Steward | AAIF / Linux Foundation [14] | AAIF / Linux Foundation [29] | Merged into A2A [32] | AGNTCY / LF [34][39] | Google (open source) [41] | x402 Foundation [43] |
| Wire format | JSON-RPC 2.0 | JSON-RPC, gRPC, HTTP+JSON (proto-normative) | REST (OpenAPI) | REST (OpenAPI) | SD-JWT VCs, JWT | JSON in HTTP headers / `_meta` |
| Transports | stdio, Streamable HTTP | HTTP(S), gRPC/TLS; SSE; webhooks | HTTP; streaming | HTTP | Carried by commerce protocol | HTTP, MCP, A2A |
| State model | Stateless; explicit handles; MRTR | Server-side Task + contextId | Runs + distributed sessions | Runs + threads | Mandate chains + receipts | Per-request settlement |
| Long-running work | Tasks extension (poll/update) | Native tasks; stream, poll, push | Async runs, Await | `/runs/wait`, stream | n/a | n/a |
| Human in loop | Elicitation via MRTR | `INPUT_REQUIRED`, `AUTH_REQUIRED` | `awaiting` | n/a | Trusted Surface signing | n/a |
| Discovery | Registry, Server Cards, `server/discover` | Agent Card at well-known URI | Manifest, `.well-known/agent.yml`, image labels | `/agents/search`, descriptors | n/a | Facilitator resource discovery |
| AuthN/Z | OAuth 2.1 RS, PRM, CIMD, RFC 8707/9207, EMA | securitySchemes (OAuth2, OIDC, mTLS, API key), signed cards | Platform-defined | Spec-defined API | Key binding, VCs, OpenID4VP | Wallet signatures; facilitator verify |

---

## 11.9 Key Takeaways for System Designers

1. **Pick protocols by boundary.** Use MCP where your agent consumes capabilities and A2A where you delegate to an opaque peer that owns its own plan. Payment and identity protocols ride on top as extensions.
2. **Design for stateless, web-native operation from day one.** MCP's 2026 rewrite shows the cost of retrofitting. Put routing keys in headers, make every request self-describing, make list results cacheable with deterministic order, and treat stream breaks as retry-with-new-ID rather than resume.
3. **Keep conversational state in data, not connections.** Use opaque server-minted handles bound to the authenticated principal, MRTR-style `requestState`, or A2A `contextId`. Never treat possession of a handle as authentication.
4. **Treat authorization as a primary workstream.** Audience-bound tokens (RFC 8707), no token passthrough, PKCE verification, issuer validation (RFC 9207), CIMD over DCR, and SSRF-hardened metadata fetching are baseline requirements. For enterprises, IdP-driven grants (EMA/ID-JAG) remove per-server consent fatigue.
5. **Treat all tool and agent metadata as untrusted.** Annotations, descriptions, and Agent Cards drive UX and policy only when they come from a trusted or signed source. Evaluate risk per session: watch for the combination of private data, untrusted content, and an exfiltration channel.
6. **Budget context like a resource.** Use deferred tool loading and search, code-execution-style orchestration, and progressive discovery. The published reductions range from 37% to 98.7% of tokens, with accuracy gains reported alongside.
7. **Use cryptographic intent for money.** Bind every agent payment to signed, hash-linked artifacts (mandates and receipts) verified in deterministic code, with short-lived, constraint-scoped, key-bound delegations for autonomous purchases.
8. **Layer discovery.** Combine publish-time registries, pre-connect signed cards and catalogs, and runtime capability discovery. Keep registries neutral and let curation happen in downstream sub-registries.
9. **Plan for deprecation windows.** Advertise multiple protocol versions at once (A2A interfaces; MCP dual-era support) and follow published lifecycle policies. MCP guarantees at least twelve months.

---

## References

1. Anthropic. "Introducing the Model Context Protocol." 2024. https://www.anthropic.com/news/model-context-protocol
2. MCP Maintainers. "Key Changes (2025-11-25)." Model Context Protocol Specification, 2025. https://modelcontextprotocol.io/specification/2025-11-25/changelog
3. MCP Maintainers. "Key Changes (2026-07-28)." Model Context Protocol Specification, 2026. https://modelcontextprotocol.io/specification/2026-07-28/changelog
4. MCP Maintainers. "Architecture" and "Specification overview (2026-07-28)." 2026. https://modelcontextprotocol.io/specification/2026-07-28/architecture
5. MCP Maintainers. "Streamable HTTP (2026-07-28)." 2026. https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
6. MCP Maintainers. "Multi Round-Trip Requests (2026-07-28)." 2026. https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr
7. MCP Maintainers. "Authorization (2026-07-28)." 2026. https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization
8. MCP Maintainers. "Authorization Security Considerations (2026-07-28)." 2026. https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/security-considerations
9. MCP Maintainers. "Security Best Practices (2026-07-28)." 2026. https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices
10. D. Soria Parra, D. Delimarsky (MCP Lead Maintainers). "The 2026-07-28 Specification." MCP Blog, 2026. https://blog.modelcontextprotocol.io/posts/2026-07-28/
11. K. Van Gent, S. Smith (Transport WG). "Exploring the Future of MCP Transports." MCP Blog, 2025. https://blog.modelcontextprotocol.io/posts/2025-12-19-mcp-transport-future/
12. P. Carleton. "Evolving OAuth Client Registration in the Model Context Protocol." MCP Blog, 2025. https://blog.modelcontextprotocol.io/posts/client_registration/
13. P. Carleton. "Enterprise-Managed Authorization: Zero-touch OAuth for MCP." MCP Blog, 2026. https://blog.modelcontextprotocol.io/posts/enterprise-managed-auth/
14. D. Soria Parra. "MCP joins the Agentic AI Foundation." MCP Blog, 2025. https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/
15. Anthropic. "Donating the Model Context Protocol and establishing the Agentic AI Foundation." 2025. https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation
16. D. Soria Parra, A. Jones, T. Antanavicius, T. Padilla, T. Chu. "Introducing the MCP Registry." MCP Blog, 2025. https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/
17. D. Soria Parra, D. Delimarsky. "The New MCP Roadmap." MCP Blog, 2026. https://blog.modelcontextprotocol.io/posts/mcp-roadmap/
18. O. Hungerford, S. Morrow, L. Chang. "Tool Annotations as Risk Vocabulary: What Hints Can and Can't Do." MCP Blog, 2026. https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/
19. MCP Maintainers. "One Year of MCP: November 2025 Spec Release." MCP Blog, 2025. https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/
20. L. Chang, C. McCaffrey (Agents WG). "SEP-2663: Tasks Extension." 2026. https://modelcontextprotocol.io/seps/2663-tasks-extension
21. D. Soria Parra, S. Morrow, T. Antanavicius. "SEP-2127: MCP Server Cards - HTTP Server Discovery." 2026. https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2127
22. I. Salomon, L. Yosef, et al. "SEP-1865: MCP Apps - Interactive User Interfaces for MCP." 2025. https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1865
23. MCP Registry Working Group. "MCP Registry" (repository, design principles, authentication and architecture docs). 2025-2026. https://github.com/modelcontextprotocol/registry
24. Anthropic Engineering. "Code execution with MCP: Building more efficient agents." 2025. https://www.anthropic.com/engineering/code-execution-with-mcp
25. Anthropic Engineering. "Introducing advanced tool use on the Claude Developer Platform." 2025. https://www.anthropic.com/engineering/advanced-tool-use
26. A2A Project. "Agent2Agent (A2A) Protocol Specification." 2026. https://github.com/a2aproject/A2A/blob/main/docs/specification.md
27. A2A Project. "What's New in A2A Protocol v1.0" and CHANGELOG. 2026. https://github.com/a2aproject/A2A/blob/main/docs/whats-new-v1.md
28. A2A Protocol Community. "A2A Protocol Ships v1.0: Production-Ready Standard for Agent-to-Agent Communication." 2026. https://github.com/a2aproject/A2A/blob/main/docs/blog/posts/announcing-1.0.md
29. A2A Protocol Community. "A New Chapter for A2A: Joining the Agentic AI Foundation." 2026. https://github.com/a2aproject/A2A/blob/main/docs/blog/posts/a2a-joins-aaif.md
30. Google Cloud. "Agent2Agent protocol (A2A) is getting an upgrade." 2025. https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade
31. A2A Project. "Agent Discovery in A2A." 2026. https://github.com/a2aproject/A2A/blob/main/docs/topics/agent-discovery.md
32. IBM / BeeAI. "Agent Communication Protocol (ACP)" (repository and docs). 2025. https://github.com/i-am-bee/acp
33. BeeAI. "ACP is now part of A2A under the Linux Foundation" (discussion). 2025. https://github.com/orgs/i-am-bee/discussions/5
34. AGNTCY. "Agent Connect Protocol Specification." 2025. https://github.com/agntcy/acp-spec
35. AGNTCY. "SLIM: Secure Low-Latency Interactive Messaging" (repository and docs). 2026. https://github.com/agntcy/slim
36. AGNTCY. "Open Agentic Schema Framework (OASF)." 2026. https://github.com/agntcy/oasf
37. AGNTCY. "Agent Directory Service (dir)" (repository and architecture docs). 2026. https://github.com/agntcy/dir
38. AGNTCY. "Identity." 2026. https://github.com/agntcy/identity
39. Linux Foundation. "Linux Foundation Welcomes the AGNTCY Project to Standardize Open Multi-Agent System Infrastructure and Break Down AI Agent Silos." 2025. https://www.linuxfoundation.org/press/linux-foundation-welcomes-the-agntcy-project-to-standardize-open-multi-agent-system-infrastructure-and-break-down-ai-agent-silos
40. Google Cloud. "Announcing Agent Payments Protocol (AP2)." 2025. https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol
41. Google Agentic Commerce. "Agentic Payment Protocol (v0.2) Specification" and FAQ. 2026. https://github.com/google-agentic-commerce/AP2/blob/main/docs/ap2/specification.md
42. Google Agentic Commerce. "AP2 Agent Authorization." 2026. https://github.com/google-agentic-commerce/AP2/blob/main/docs/ap2/agent_authorization.md
43. x402 Foundation / Coinbase. "x402 Protocol Specification v2" and transport specs (HTTP, MCP, A2A). 2025. https://github.com/coinbase/x402/blob/main/specs/x402-specification-v2.md
44. Google Agentic Commerce. "A2A x402 Extension." 2025. https://github.com/google-agentic-commerce/a2a-x402
45. AI Catalog contributors. "Common AI Catalog and Registry Standard." 2026. https://github.com/Agent-Card/ai-catalog
46. ANP Community. "Agent Network Protocol (ANP)." 2026. https://github.com/agent-network-protocol/AgentNetworkProtocol
47. MCP Maintainers. "Key Changes (2025-06-18)." 2025. https://modelcontextprotocol.io/specification/2025-06-18/changelog
48. MCP Maintainers. "Key Changes (2025-03-26)." 2025. https://modelcontextprotocol.io/specification/2025-03-26/changelog
49. MCP Maintainers. "Transports (2024-11-05)." Model Context Protocol Specification, 2024. https://modelcontextprotocol.io/specification/2024-11-05/basic/transports
