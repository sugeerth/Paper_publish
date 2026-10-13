# 07. Computer and Browser Use Agents: Pixels, Accessibility Trees, and the Race from 15% to Human-Level on OSWorld

## Overview

A computer-use agent (CUA) operates software through the same interface a person uses: the
screen, the mouse and the keyboard. A browser-use agent does the same inside a web browser. It
may also read the page's DOM or accessibility tree. These agents matter because most enterprise
software has no API. They also matter because the GUI is the most general action space ever
built.

In October 2024, Anthropic shipped the first frontier-model computer use API. It scored 14.9%
on OSWorld, against a human baseline of roughly 72% [1][2]. In December 2025, an open agentic
framework (Agent S3 with Behavior Best-of-N) claimed 72.6% on OSWorld, which it described as
the first result above human level [34]. By September 2026, Anthropic reported 81.8% for Claude
Opus 5.5 on the harder, long-horizon OSWorld 2.1, using partial-credit scoring [11]. The
benchmark had to be rebuilt twice along the way: as OSWorld-Verified in July 2025, then as
OSWorld 2.0/2.1 in 2026 [30][31].

This chapter covers the main designs in that period:

- **Frontier-model, pixel-native CUAs**: OpenAI's Computer-Using Agent (Operator, ChatGPT agent
  and the later GA `computer` tool), Anthropic's Claude computer use and browser toolsets, and
  Google's Gemini Computer Use and Project Mariner. Each takes screenshots in and returns
  coordinate-level actions.
- **Open-source browser frameworks**: Browser Use, Stagehand and Skyvern. These wrap a general
  LLM in a structured view of the DOM or accessibility tree and a deterministic browser driver.
- **Research agents and models**: OSWorld, WebArena, WebVoyager, SeeAct, UI-TARS, Agent S,
  OpenCUA, OmniParser, UFO, Fara and others. These set the benchmarks and the architectural
  vocabulary: grounding, set-of-marks, planner/actor splits and best-of-N trajectory selection.
- **The cross-cutting question**: should an agent perceive the interface as pixels, as an
  accessibility (a11y) tree or DOM, or as both?
- **The other constant**: prompt injection. Every system in this chapter treats it as an open
  security problem.

A note on sources. This chapter relies on primary documentation, model cards, release notes
and source code. During research, some vendor pages (openai.com, deepmind.google, blog.google)
could not be retrieved directly. Facts attributed to them are those reported in their
announcements, cross-checked against independent primary sources where possible, such as
Google's model card, which tabulates OpenAI's CUA results [18]. Figures that could not be
cross-checked are marked approximate.

---

## 1. The canonical CUA loop

Every system in this chapter is a variation on one loop. Chapter 02 describes the generic
agent loop. The CUA version adds three components:

1. **An environment** that can render a screen and accept input events: a VM, a container, a
   headless browser, or the user's own browser.
2. **An observation encoder**: a screenshot, an a11y tree, serialized DOM, or a mix.
3. **An action decoder**: mapping model output to coordinates, element IDs or code.

```
   +-------------------------------------------------------------------------+
   |                         Agent harness (client side)                      |
   |                                                                          |
   |  task --> [ LLM / VLM ] --tool_use--> [ action executor ] --> environment |
   |              ^   ^                     (xdotool, PyAutoGUI,   (VM / X11 / |
   |              |   |                      Playwright, CDP)       browser)   |
   |              |   +-- history (pruned screenshots, memory, todo)           |
   |              |                                                          |
   |              +------ observation <-- [ encoder ] <-- screen / DOM / AX ---+
   |                      (image, a11y text, SoM overlay, URL, tabs)           |
   |                                                                          |
   |  guards: domain allowlist, confirmation gates, injection classifier,     |
   |          step budget, human takeover                                     |
   +-------------------------------------------------------------------------+
```

Anthropic's documentation states the loop plainly [3]:

1. Send a request with the computer tool.
2. If `stop_reason` is `tool_use`, execute each tool call in order.
3. Return one `tool_result` per call: an image for `screenshot` or `zoom`, `"OK"` for other
   actions.
4. Repeat until the model emits no tool calls or an iteration cap is hit.

Gemini's reference agent uses the same structure. It retries model calls up to five times with
exponential backoff. It treats "no function call" as completion. It returns the current URL
alongside each screenshot [19]. OpenAI's Agents SDK runs each batch of actions from a
`computer_call` and replies with a `computer_call_output` that contains a fresh screenshot
[16].

Across vendors, these loops have converged on five design points:

- **Actions are batched.** Early APIs emitted one action per turn. By 2026, Anthropic's
  `computer_toolset_20260801` and OpenAI's GA `computer` tool both return several actions per
  turn. The harness must run them in order and halt at the first failure [3][16].
- **Coordinates are model-native and must be rescaled.** Claude predicts pixel coordinates in
  the screenshot it was shown. Gemini predicts on a normalized 0-1000 grid that the client maps
  back to pixels [3][19].
- **Screenshot history is aggressively truncated.** Gemini's reference agent keeps images only
  for the three most recent turns [19]. Anthropic recommends keeping about three screenshots
  and pruning in batches so the prompt cache prefix stays stable [5].
- **Safety checks travel in-band.** OpenAI's `pending_safety_checks` and Gemini's
  `safety_decision: require_confirmation` both come back as fields on a proposed action. The
  harness must acknowledge them before executing [16][19].
- **Execution stays on the client.** Anthropic's computer and browser toolsets are explicitly
  "client toolsets". Screenshots and inputs stay in the developer's environment, which makes
  computer use eligible for zero data retention [3][4].

---

## 2. Benchmarks: how the field measures itself

Benchmarks shaped CUA architecture more than in most agent subfields, so they are covered
before the systems.

### 2.1 OSWorld and its successors

**OSWorld** (Xie et al., NeurIPS 2024) put agents in real Ubuntu, Windows and macOS VMs. It
used 369 tasks spanning office apps, browsers, IDEs, file managers and multi-app workflows.
Each task had an execution-based evaluation script [29][30].

- Humans completed about 72% of tasks. The best 2024 model baseline completed about 12%.
- The authors identified GUI grounding and operational knowledge as the main bottlenecks [29].
- The harness supported several observation types (screenshot, a11y tree, screenshot plus a11y
  tree, and set-of-marks) and a `pyautogui` action space. It became the standard testbed for
  the pixels-versus-structure debate (Section 9).

The benchmark changed four times:

| Date | Version | What changed |
|---|---|---|
| 2024-04 | OSWorld | Original 369-task release; VMware and later Docker, AWS and Azure providers [30] |
| 2025-07-28 | OSWorld-Verified | Community-reported task fixes, AWS parallelism, re-run official results [30] |
| 2026-06-26 | OSWorld 2.0 | Long-horizon, real-world tasks; mocked websites and self-hosted GitLab [31] |
| 2026-09-16 | OSWorld 2.1 | Bug-fix release with pinned task, asset, website and image references; recommended [31] |

Three design choices make OSWorld scores hard to game:

- **Step limits.** Results are reported at 15, 50 or 100 steps. More steps materially raise
  scores. Claude 3.5 Sonnet went from 14.9% to 22.0% with more steps [1].
- **Maintainer-run verification.** Leaderboard entries are verified only after the maintainers
  run the agent themselves. Submitters must agree to disclose the agent implementation and file
  a report [30][31].
- **Version boundaries.** Anthropic notes that its scores before Sonnet 4.5 used the original
  OSWorld and its later scores use OSWorld-Verified, so the two are not directly comparable
  [10].

### 2.2 Web benchmarks

- **WebArena** (Zhou et al., 2023) self-hosts realistic sites: shopping, a shopping admin CMS,
  Reddit, GitLab, maps and Wikipedia. It has 812 tasks with functional checkers. Its default
  observation is the accessibility tree [43]. The original paper reported about 14% success
  for GPT-4 versus about 78% for humans.
- **WebVoyager** (He et al., ACL 2024) runs 643 tasks on 15 live websites. It uses screenshots
  overlaid with bounding boxes around interactive elements (a set-of-marks style) and a GPT-4V
  judge that reads the last *k* screenshots [40]. Most 2025 vendor claims use this benchmark:
  Browser Use, Skyvern 2.0, Mariner, OpenAI CUA and Gemini.
- **Online-Mind2Web** (Xue et al., COLM 2025) responded to WebVoyager saturation. It has 300
  tasks on 136 live sites, with stale tasks periodically replaced. Its judge, WebJudge,
  identifies key task points, selects key screenshots, and then judges. It agrees with human
  labels 85.7% of the time with o4-mini, against 78.7% for the WebVoyager-style judge on the
  same backbone [41]. The paper's title, "An Illusion of Progress?", captures its argument:
  live-web benchmarks with lenient automatic judges had overstated agent progress.
- **BrowserGym/AgentLab** (ServiceNow) unifies MiniWoB, WebArena, VisualWebArena, WorkArena,
  AssistantBench and others behind one Gym interface. The WebArena authors themselves
  recommend it for parallel, reproducible runs [43][44].

**Lesson.** CUA benchmarks degrade quickly. Live sites change, CAPTCHAs appear, and judges are
lenient. Every major benchmark has needed a "verified" or v2 release within about a year.
Treat any single number as specific to its harness. Google's own model card lists separate
"official leaderboard" and "measured by Browserbase" columns for the same models, and the
numbers differ by up to roughly 9 points [18].

---

## 3. OpenAI: Computer-Using Agent, Operator, ChatGPT agent

### 3.1 Problem and architecture

OpenAI introduced the Computer-Using Agent (CUA) model and the Operator product in January
2025 [13][14]. CUA combined GPT-4o's vision with reasoning trained through reinforcement
learning. It operated purely on raw screenshots and emitted mouse and keyboard actions, with
no DOM or OS APIs. OpenAI framed the loop as perception (a screenshot added to context),
reasoning (chain of thought over current and past screenshots) and action (click, scroll,
type), repeated until the task is done or user input is needed [13].

Reported launch results were [13][18]:

- OSWorld: 38.1%
- WebArena: about 58% (approximate)
- WebVoyager: 87%

Google's later model card independently tabulates CUA at 38.1% on OSWorld and 87.0% on
WebVoyager, and Operator at 61.3% on the official Online-Mind2Web leaderboard [18]. ByteDance's
UI-TARS-1.5 comparison reports 36.4% for OpenAI CUA on OSWorld at 100 steps, and 23.4% on the
ScreenSpot-Pro grounding benchmark [35]. On the stricter OSWorld-Verified, OpenCUA lists
OpenAI CUA at 31.4% at 100 steps [32].

### 3.2 Operator: product-level safety design

Operator ran CUA in a cloud-hosted browser that the user could watch. OpenAI's announcements
and system card described a layered safety design [14]:

- **Takeover mode**: the user types credentials and payment details directly, so the model
  never sees them.
- **User confirmations** before consequential actions such as submitting orders or sending
  email.
- **Task refusals** for disallowed categories.
- **Watch mode** on sensitive sites, where the agent pauses if the user is inactive.
- **A prompt-injection monitor** that halts the agent when on-screen content looks like an
  attempt to redirect it.

The same layers (handoff of secrets, confirmation gates, category blocks, injection monitors)
reappear at Anthropic and Google.

### 3.3 ChatGPT agent (mid-2025)

In July 2025, OpenAI folded Operator into **ChatGPT agent** [15]. One agent ran on a virtual
computer and combined three things:

- Operator's visual browser.
- Deep research's text-based browser, for fast reading and synthesis.
- A terminal, plus API and connector access.

The architectural point is that pixel-level browsing is expensive and slow. A production agent
should choose per step between a text browser, a visual browser, code, and direct API calls.
That idea recurs throughout this chapter. Treat any benchmark figure from that announcement
not reproduced here as approximate.

### 3.4 The API surface and its evolution

The developer API began as the `computer-use-preview` model and `computer_use_preview` tool in
the Responses API. The tool required:

- an `environment` and a display size;
- one `action` per `computer_call`;
- `truncation="auto"` [16].

By 2026, the OpenAI Agents SDK documents a GA path [16]:

- `computer-use-preview` maps to `gpt-5.6-sol`.
- The selector becomes `computer`.
- `computer_call` carries batched `actions[]`.
- Truncation no longer needs to be set.
- The SDK default model (`gpt-5.6-luna`) supports computer use.

Safety checks remain in-band. A `computer_call` may carry `pending_safety_checks`, and the
harness's `on_safety_check` callback must acknowledge each one. If no callback is set, the SDK
proceeds without prompting a person. The documentation therefore tells developers to enforce
any must-apply policy inside their own harness [16].

The `AsyncComputer` interface shows the minimal action set a harness must implement [16]:

- `screenshot`
- `click`, `double_click`, `move`, `drag(path)`
- `scroll`
- `type`, `keypress`
- `wait`

OpenAI's current sample app shows a different design. Instead of discrete click actions, the
model calls a single `exec_py` function [17]:

- It writes Python that uses PyAutoGUI against a persistent interpreter.
- Screenshots come back as `input_image` items.
- Each code call has a 60-second deadline.
- The runner refuses to start new runs if held input cannot be released.

This **code-as-action CUA** pattern trades a small, safe action space for expressiveness and
fewer round trips. The README warns that generated code runs with the user's permissions and
no OS sandbox [17].

Simular reports that OpenAI's GPT-5.6 Sol scores 62.57% on the 108-task OSWorld 2.0, citing
OpenAI's own report [34]. This is approximate and secondhand.

---

## 4. Anthropic: Claude computer use, the browser toolset, and Claude in Chrome

### 4.1 Origin and training lessons

Anthropic released computer use in public beta with the upgraded Claude 3.5 Sonnet on October
22, 2024. It scored 14.9% on OSWorld's screenshot-only category, against 7.7-7.8% for the next
best system, and 22.0% with more steps [1][2]. Early adopters included Asana, Canva, Cognition,
DoorDash, Replit and The Browser Company. Replit used the feature to evaluate apps as Replit
Agent built them [1].

The engineering post "Developing a computer use model" gives rare detail on training [2]:

- **Pixel counting was the key skill.** The model had to count pixels vertically and
  horizontally to place the cursor accurately. Anthropic called this "critical".
- **Generalization from small worlds.** Training used only a few simple programs, such as a
  calculator and a text editor, with internet access disabled for safety. The model
  nonetheless generalized and began to self-correct.
- **Known weaknesses at launch.** The model was slow and error-prone. Dragging and zooming were
  hard. The "flipbook" of screenshots could miss short-lived notifications.
- **Prompt injection was flagged on day one** as the central risk, because instructions can
  hide in screenshots of web content. Anthropic added classifiers and steered the model away
  from social media posting, domain registration and government websites [2].

### 4.2 The reference architecture

Anthropic's `computer-use-demo` is the canonical self-hosted setup [6]:

```
  +---------------------------- Docker container ---------------------------+
  |                                                                          |
  |   X11 virtual display (desktop apps, Firefox)                            |
  |        ^ xdotool / screenshot                                            |
  |        |                                                                 |
  |   tools: computer | bash | str_replace_based_edit_tool                   |
  |        ^                                                                 |
  |        | tool_use / tool_result                                          |
  |   agent loop  <---- Claude API (or Bedrock / Vertex) ---->               |
  |        |                                                                 |
  |   Streamlit chat UI :8501   noVNC :6080   VNC :5900   combined :8080     |
  +--------------------------------------------------------------------------+
```

Two design features stand out:

- **Shell and editor tools sit next to the GUI tool.** Many steps are cheaper as shell
  commands or file edits than as clicks.
- **The weak separation is called out explicitly.** The agent loop runs inside the container
  it controls, supports one session at a time, and must be reset between sessions [6].

The demo recommends XGA (1024x768). It scales larger displays down and pads smaller ones,
because relying on API-side resizing hurts accuracy and speed [6].

### 4.3 The 2026 computer toolset

By August 2026, `computer_toolset_20260801` replaced the single `computer` tool, which took an
`action` parameter, with **17 member tools** [3]:

- **Capture**: `screenshot`; `zoom`, which takes a region and returns it at full resolution
  while coordinates stay in full-screen space.
- **Clicks**: left, right, middle, double and triple click.
- **Drags**: `left_click_drag`, plus `left_mouse_down` and `left_mouse_up` for fine control.
- **Pointer**: `mouse_move`, `cursor_position`.
- **Scrolling**: `scroll`.
- **Keyboard**: `type`, `key` (with a `repeat` count), `hold_key`.
- **Timing**: `wait`.

The toolset changes the harness contract in several ways:

- Batched tool calls are standard. Harnesses run them in order and answer any blocks after a
  failure with a fixed "Not executed" error [3].
- The API no longer downscales images; oversized screenshots are rejected. Opus 4.7 and later
  models accept up to 2576 px on the long edge and about 3.75 MP (4,784 visual tokens). Earlier
  models accept 1568 px and about 1.15 MP. Clients must resize screenshots and rescale the
  returned coordinates [3][5].
- Declaring the toolset costs about 4,500 input tokens; disabling zoom saves about 410. Each
  screenshot costs about 1,000-1,800 tokens [3].
- Earlier versions remain available behind beta headers: `computer_20250124` and
  `computer_20251124` [3].

Anthropic's published "best practices" (May 2026) give benchmarked guidance that generalizes
beyond Claude [5]:

- **Resolution budget.** 1280x720 uses about 80% of the 4.6-family pixel budget. A 16 px
  checkbox on a 4K display shrinks to about 5 px at 1280x720. Avoid going below 960x540. Start
  Opus 4.7 at 1080p.
- **Thinking effort.** For Opus 4.7, `high` gets close to peak task success at about half the
  output tokens of `max`. `max` adds no accuracy for computer use. Opus 4.7 at `low` roughly
  matches Sonnet 4.6 at `max` with about one-tenth the tokens per task.
- **Context.** A 200k window fills in under 100 screenshots. Keep the last 3, prune in batches
  of 25, place four cache breakpoints (one on the stable prefix, three on recent tool
  results), and compact at about 150k tokens. Cache hits cost 10% of the input price.
- **Click precision.** Sonnet 4.6 had fewer near-misses than Opus 4.6, so a larger model is not
  always a better clicker.
- **Injection classifiers** run in parallel with inference, adding "approximately zero"
  latency and no cost [5].

### 4.4 The browser toolset: a hybrid a11y-and-pixel design

For tasks that stay in the browser, Anthropic ships `browser_toolset_20260801`, with **31
member tools** (27 on by default) [4]:

- **Navigation and capture**: `navigate`, `screenshot`, `zoom`.
- **Pointer and keyboard**: the same family as the computer toolset, plus `hover` and
  `scroll_to`.
- **Page reading**: `read_page`, `find`, `get_page_text`.
- **Forms**: `form_input`; `file_upload` is opt-in.
- **Diagnostics and scripting** (opt-in): `read_console`, `read_network`, `javascript_exec`.
- **Tabs**: `new_tab`, `list_tabs`, `switch_tab`, `close_tab`.

The key design decision is that **the accessibility tree is the primary observation**.
`read_page` returns a11y text with element references such as `[ref_2]`, capped at 50,000
characters. Click, hover and `form_input` can target a reference instead of a coordinate, "so
it doesn't have to locate elements in a screenshot" [4]. `find` takes a natural-language query
and returns up to 20 matching references. Screenshots and zoom remain available for visual
verification.

The documentation's security list doubles as a threat model [4]:

- Run in a fresh, low-privilege profile.
- Enforce domain allowlists at the network layer and re-check them after redirects.
- Accept only http(s) URLs, and parse them rather than prefix-matching.
- Treat titles, URLs and download names as untrusted.
- Keep `javascript_exec` and `file_upload` off unless the task needs them.

Browser Use reports implementing all 31 actions against local, cloud or remote CDP browsers
[22].

### 4.5 Claude in Chrome and injection defense

The Claude in Chrome research preview (August 25, 2025) gave unusually concrete red-team
numbers [7]:

- 123 adversarial test cases across 29 attack scenarios.
- Attack success rate in autonomous mode: 23.6% without mitigations, 11.2% with them.
- On four browser-specific attack types, such as hidden DOM fields and URL or tab-title
  injection: 35.7% without mitigations, 0% with them.

The product mitigations were:

- per-site permissions;
- confirmations before publishing, purchasing or sharing personal data;
- default blocks on financial services, adult and pirated-content sites.

The extension reached Max users in November 2025 and Pro, Team and Enterprise users in
December 2025 [7].

Anthropic's November 2025 research post describes three defense layers [8]:

1. Reinforcement learning on simulated injected web content, rewarding the model for spotting
   and refusing injections.
2. Classifiers that scan all untrusted content entering the context: hidden text, manipulated
   images and deceptive UI.
3. Continuous human red teaming, including external arena-style challenges run by Gray Swan.

Robustness was measured against an adaptive "Best-of-N" attacker with 100 attempts per
environment. The post states that even 1% attack success is "meaningful risk" and that the
problem is "far from a solved problem" [8].

### 4.6 Capability trajectory

| Model (date) | OSWorld variant | Reported score |
|---|---|---|
| Claude 3.5 Sonnet (Oct 2024) | OSWorld, screenshot-only | 14.9% (22.0% with more steps) [1] |
| Claude 3.7 Sonnet (Feb 2025) | OSWorld-Verified, 100 steps (third party) | 35.9% [32] |
| Claude Sonnet 4 (May 2025) | OSWorld | 42.2% [9] |
| Claude Sonnet 4.5 (Sep 2025) | OSWorld-Verified | 61.4% [9] |
| Claude Sonnet 5.5 (Sep 2026) | OSWorld 2.1, partial scoring | 80.1% [12] |
| Claude Opus 5.5 (Sep 2026) | OSWorld 2.1, partial scoring | 81.8% [11] |

Sonnet 4.5 was also reported to stay focused for "more than 30 hours" on multi-step tasks [9].
Sonnet 4.6 (February 2026) reported "human-level capability" on tasks such as complex
spreadsheets and multi-step forms. It also improved prompt-injection resistance over Sonnet
4.5, though Anthropic said it still trails the most skilled humans [10]. Note that the 2026
scores use a different benchmark (OSWorld 2.1) and partial-credit scoring, so they cannot be
compared directly with the 2024-2025 numbers.

---

## 5. Google: Project Mariner and the Gemini Computer Use model

### 5.1 Project Mariner

Google DeepMind announced Project Mariner in December 2024 as a Gemini 2.0 research prototype
running as a Chrome extension [20][21]:

- It reasoned over the active tab's pixels and web elements: text, code, images and forms.
- It reported 83.5% on WebVoyager "as a single agent setup".
- It acted only in the active tab.
- It asked for confirmation before sensitive actions such as purchases.

Later updates moved Mariner to cloud-hosted VMs that run several tasks in parallel. They also
added a "teach and repeat" mode, in which the agent learns a workflow from one demonstration,
and fed these capabilities into the Gemini API and the Gemini app's agent mode [21]. The
details come from Google's announcements and are not independently verified here.

### 5.2 Gemini 2.5 Computer Use

The **Gemini 2.5 Computer Use** model card (October 7, 2025) is the most detailed primary
source on Google's design [18]:

- **Base model.** Gemini 2.5 Pro (06-2025) with additional post-training for UI control.
- **Scope.** Optimized for web browsers. It is not yet optimized for OS-level control, but it
  "demonstrates promise" on mobile.
- **Interface.** Inputs are the task, screenshots and recent action history. Output is a
  single function call proposing the next action.

Reported results [18]:

| Benchmark (source) | Gemini 2.5 CU | Comparators (as tabulated) |
|---|---|---|
| Online-Mind2Web (official leaderboard) | 69.0% | OpenAI Operator 61.3% |
| Online-Mind2Web (measured by Browserbase) | 65.7% | Claude Sonnet 4.5 and 4 and OpenAI CUA lower (approx.) |
| WebVoyager (official / self-reported) | 88.9% | OpenAI CUA 87.0% |
| WebVoyager (measured by Browserbase) | 79.9% | others about 61-71% (approx.) |
| AndroidWorld (Google DeepMind) | 69.7% | Claude Sonnet 4.5 56.0%, Sonnet 4 62.1% |
| OSWorld | not supported | Sonnet 4.5 61.4%, Sonnet 4 42.2%, OpenAI CUA 38.1% |

Safety is split between the model and an inference-time service [18]:

- **Post-training.** The model learns to recognize high-stakes actions (purchases, downloads,
  sending messages) and to follow developer system instructions that require confirmation.
- **Per-step monitor.** A safety service inspects every proposed action. It blocks actions
  prohibited by policy, forces user confirmation for high-stakes actions, and blocks known
  dangerous sites.
- **Terms of service** forbid developers from bypassing required confirmations.

### 5.3 The reference harness

Google's `computer-use-preview` repository adds client-side details [19]:

- The `ComputerUse` tool is configured with `environment=ENVIRONMENT_BROWSER` and an optional
  list of excluded actions. Developer function declarations can sit next to it.
- Older models use a legacy action vocabulary: `click_at`, `type_text_at`, `scroll_document`,
  `wait_5_seconds`, `key_combination`. Newer Gemini 3.x models use `click`, `type`, `scroll`,
  `wait`, `hotkey`, `press_key`.
- Coordinates arrive on a 0-1000 grid and are mapped to pixels as `x/1000*width`.
- A `safety_decision` with `require_confirmation` prompts the user. Approval is echoed back as
  `safety_acknowledgement: "true"` in the function response.
- Only the three most recent turns keep screenshots.

By late 2026 the repository's default model was `gemini-3.6-flash`, and computer use was also
supported on `gemini-3.5-flash-lite`. This is a sign that computer use has moved from a
specialized model into general Flash-tier models [19].

A known failure mode: Playwright cannot capture OS-rendered `<select>` dropdowns on some
platforms, so the model sees a wrong screenshot. The suggested fixes are a hosted browser such
as Browserbase or injecting a custom-select shim [19]. This is a concrete example of the
renderer and screenshot mismatches that pixel-only agents must cope with.

---

## 6. Browser Use: DOM-first, LLM-agnostic

### 6.1 Problem and positioning

Browser Use is an MIT-licensed Python (and TypeScript) library. It lets any LLM drive
Chromium. It is among the most-starred agent projects, with about 118k GitHub stars by late
2026 [22]. The project now spans three products [22]:

- the open library;
- a hosted cloud with stealth browsers at about $0.02 per browser-hour, CAPTCHA solving and
  residential proxies;
- its own browser-optimized model, BU 2.0 (`bu-2-0`), plus an internal "BU Bench V2" of hard
  tasks.

### 6.2 Observation pipeline (from source)

Browser Use's core contribution is a carefully engineered DOM serialization. It does not ask
the model to ground actions in pixels. The `DomService` builds an enhanced tree from four
concurrent Chrome DevTools Protocol (CDP) calls, with a 10 s timeout and a 2 s retry [23]:

```
  DOMSnapshot.captureSnapshot ---(styles, paint order, rects)---+
  DOM.getDocument(depth=-1, pierce=true) --(shadow roots, iframes)+--> EnhancedDOMTreeNode
  Accessibility.getFullAXTree (per frame) --(roles, names, values)+       |
  Page.getLayoutMetrics ------------------(device pixel ratio)---+       v
                                                         DOMTreeSerializer
     + JS click-listener scan (capped at 100, skipped above 10k elements)  |
                                                                         v
        "[12]<button aria-label=Search />   *[13]<input placeholder=... />"
```

The serializer applies a series of compression and safety heuristics [24]:

- **Indexing.** Only visible, interactive elements get `[index]` tags. Elements new since the
  last step are marked with `*`.
- **Occlusion.** Paint-order filtering drops elements covered by others, such as modals and
  sticky headers.
- **Deduplication.** Bounding-box propagation removes descendants that sit 99% or more inside
  a clickable parent such as an `<a>` or `<button>`.
- **Containers.** Shadow DOM, iframes and scroll containers are labeled.
- **Compact controls.** Selects, sliders, file inputs and media get one-line "compound
  component" summaries.
- **Size limits.** SVGs are collapsed and attribute values are capped at 100 characters.
- **Privacy.** Password values are always stripped so they "must not leak into DOM snapshots
  sent to the LLM".

### 6.3 Agent loop and prompt contract

The system prompt defines a structured state block [25]:

- `<user_request>`
- `<agent_history>`
- `<agent_state>`, with file-system and todo summaries
- `<browser_state>`: URL, tabs and indexed elements
- `<browser_vision>`: a screenshot with bounding boxes, treated as ground truth when present
- `<read_state>`

The model returns JSON with `thinking`, `evaluation_previous_goal`, `memory`, `next_goal`, an
optional plan update, and a non-empty `action` list.

Two rules carry most of the reliability engineering:

- **Multi-action steps with page-change barriers.** Several actions can be chained per step,
  but page-changing actions such as `navigate`, `go_back` or `switch` must come last. The
  runtime also skips the remaining actions if a click unexpectedly navigates [25].
- **A strict `done` contract.** `done` must be the only action in its step. `success=true` is
  allowed only after the agent has verified every requirement against tool output. Long tasks
  keep a `todo.md` and `results.md` on a persistent file system; tasks under about 10 steps
  should not [25].

**Lesson.** Browser Use shows that a general LLM with a high-quality, compact,
change-annotated DOM view, plus pixels as a secondary signal, can compete with pixel-native
CUAs on web tasks. Most of the engineering lives in the serializer, not the model.

---

## 7. Stagehand: AI primitives inside deterministic automation

Stagehand (Browserbase) takes the opposite stance to autonomous agents. It is an SDK
(TypeScript, Python and Go; v4 by 2026) that adds a few **AI primitives** to Playwright-style
code [26]:

- `act("click the sign-in button")` performs a natural-language action.
- `observe(...)` returns *real selectors* for described elements. These can be passed to
  ordinary locators, so the model never sees the credentials typed into them.
- `extract(schema)` returns schema-validated data, using Zod in TypeScript or Pydantic in
  Python.

The design principle is **use AI only where the page is unknown or changing; use code
everywhere else.** Supporting mechanisms include [26]:

- *hybrid accessibility-tree trimming*, so each call sends only the needed page context;
- *self-healing*, where `act`, `observe` and `extract` re-derive how an action happens when
  the site changes;
- *server-side caching* (`cache: true` on Browserbase), so repeated identical calls cost no
  tokens;
- a Model Gateway that picks a model per action;
- a hosted MCP server exposing `navigate`, `act`, `observe` and `extract`;
- WebMCP support, so the agent can call tools that pages expose directly.

Browserbase claims 2x faster execution than Playwright-cloud-equivalent browsers but gives no
method [26].

Browserbase also matters as **infrastructure**. Google's reference CUA uses it as a backend,
and Google's model card cites Browserbase as an independent evaluator of CUA models [18][19].
That shows a split between the agent "brain" and the remote browser "hands".

---

## 8. Skyvern: planner, actor, validator, then compiled scripts

Skyvern (AGPL-3.0 core plus a managed cloud) targets form-heavy business workflows. Its
README says it uses "a swarm of agents to comprehend a website, and plan and execute its
actions", relying on vision LLMs instead of brittle XPaths [27]. Its reported results [27]:

- 64.4% on WebBench, where it is strongest on WRITE tasks (forms, logins, downloads);
- 85.8% on WebVoyager, the figure in the Skyvern 2.0 report title.

Skyvern's open prompt library shows how the swarm is structured [28]:

```
  user goal
     |
     v
  task_v2 planner ---> mini-goal {navigate | extract | loop | compute}
     ^                     |
     |                     v
  check-user-goal     extract-action actor: screenshot + element tree with IDs
  (validator)              -> [CLICK | INPUT_TEXT | SELECT_OPTION | UPLOAD_FILE |
     ^                         SOLVE_CAPTCHA | COMPLETE | TERMINATE ...] + confidence
     |                     |
     +---- results <-------+
                           |
          script-generation / script-reviewer --> reusable Playwright-style code
          script-failure-triage, cua-fallback-action (fall back to a CUA model)
```

- The **planner** (`task_v2`) breaks the goal into self-contained mini-goals. Each ends with a
  success condition and fits the step budget. Loops are preferred for repeated work. A
  `required_subgoals` checklist with evidence controls completion. Termination is reserved for
  explicit "impossible" signals, not CAPTCHAs or empty results [28].
- The **actor** (`extract-action`) sees the clickable element tree and, optionally, a
  screenshot. It returns JSON actions with per-action reasoning and a `confidence_float`. It
  must never invent element IDs and must treat page content as untrusted data even when it
  claims to be a system message. It should solve CAPTCHAs only after filling the form, because
  tokens expire [28].
- **Validators** (`check-user-goal`, `decisive-criterion-validate`, `quality-audit`) confirm
  completion.
- **Script generation and review** prompts, plus a `cua-fallback-action` and a UI-TARS system
  prompt, point to a hybrid strategy. An LLM explores a workflow once, the result is compiled
  into deterministic code, and an agent (or a CUA model) is called again only on failure [28].

Skyvern also ships the operational features enterprise automation needs [27]:

- 2FA via TOTP, email and SMS;
- Bitwarden and custom credential services;
- livestreamed sessions for human intervention;
- CDP connection to a local Chrome;
- workflow blocks (loops, file parsing, HTTP, email);
- MCP support.

---

## 9. Pixels vs. accessibility tree vs. DOM: the central design choice

| Dimension | Pixels only | A11y tree / DOM only | Hybrid (both) |
|---|---|---|---|
| Representative systems | OpenAI CUA, Gemini CU, Claude computer tool, UI-TARS, Fara | WebArena baselines, Agent-E, early Browser Use, Stagehand `observe` | Claude browser toolset, Browser Use (DOM + vision), Skyvern, UFO2, SeeAct-SoM |
| Generality | Any app, OS, canvas, game or remote desktop | Web pages, or native apps with good a11y APIs | Web and instrumented desktops |
| Grounding | Model must predict coordinates; needs pixel-level skill [2] | Exact element IDs; no coordinate error | IDs where available, pixels as fallback |
| Token cost per step | About 1,000-1,800 tokens per screenshot [3] | Can be large; capped (e.g. 50k chars) [4] | Both, unless trimmed |
| Failure modes | Small targets, high-DPI scaling, OS-drawn widgets [5][19] | Missing or wrong ARIA, canvas UIs, shadow DOM, iframes, hidden elements | Mismatch between the two views |
| Injection surface | Rendered text and images | Hidden DOM text, attributes, tab titles [7] | Both |
| Privacy controls | Hard to redact pixels | Can strip passwords pre-LLM [24] | Strip in text; mask in images |

The evidence over time:

1. **2023-2024: structure won.** WebArena's default was the accessibility tree [43]. OSWorld's
   early baselines found that text a11y input beat raw screenshots for general VLMs, and named
   grounding as the bottleneck [29]. SeeAct separated action generation from grounding and made
   set-of-marks overlays its default grounding mode [42]. WebVoyager drew boxes around
   interactive elements on the screenshot [40]. OmniParser parsed screenshots into labeled
   elements: a YOLO-based detector plus a Florence captioner. That turned pixels into
   structure for agents that could not ground directly. Its v2 reached 39.5% on ScreenSpot-Pro
   [36].
2. **2025: pixel-native models caught up.** RL-trained CUAs (OpenAI CUA, Claude, UI-TARS)
   grounded directly in pixels. UI-TARS-1.5 reached 61.6% on ScreenSpot-Pro, where OpenAI CUA
   scored 23.4% and Claude 3.7 scored 27.7% [35]. Dedicated grounding data scaled: OSWorld-G
   and the Jedi dataset (about 4M examples), and OpenCUA's AgentNet with 22.6k human
   desktop-task demonstrations across three OSes and more than 200 applications [32][33].
3. **2026: hybrid by default for the browser, pixels for the desktop.** Anthropic's browser
   toolset makes the a11y tree primary and keeps screenshot and zoom for verification [4].
   Browser Use treats the screenshot as "ground truth" next to its DOM index [25]. Microsoft's
   UFO2 uses hybrid UIA-plus-vision control detection on Windows [37]. Whole-desktop agents
   (Claude's computer toolset, Agent S3, OSWorld entrants) remain mostly pixel-based, with
   zoom to recover detail [3][34].

The practical rule:

- Use structure wherever a trustworthy structure exists. It is cheaper, more precise and easier
  to redact.
- Use pixels for coverage, for verification ("did the click actually work?"), and for
  everything structure misses: canvas, remote desktops, games and badly labeled apps.

---

## 10. Research architectures that shaped production

### 10.1 Native agent models: UI-TARS and Fara

**UI-TARS** (ByteDance, 2025) is an end-to-end "native" GUI agent model. Perception, reasoning,
grounding and action all happen in one VLM, with prompt templates for desktop, mobile and
grounding-only use. UI-TARS-1.5 adds RL-trained "think before act" reasoning [35]. Reported
results for UI-TARS-1.5 [35]:

- OSWorld (100 steps): 42.5%
- Windows Agent Arena: 42.1%
- WebVoyager: 84.8%
- Online-Mind2Web: 75.8%
- AndroidWorld: 64.2%

The open 7B release reached 27.5% on OSWorld. UI-TARS-2 (September 2025) extends the approach
to games, code and tool use [35].

**Fara-7B** (Microsoft, 2025) is a small screenshot-only web agent [39]:

- WebVoyager: 73.5%
- Online-Mind2Web: 34.1%

Its successor Fara1.5 explicitly avoids accessibility trees and separate parsing models [39].
Microsoft's Magentic-UI 0.2 pairs an on-device orchestrator (MagenticBrain) with Fara as the
browser model. Browser sessions run in a lightweight VM sandbox (Quicksand), and users can
steer, approve or take over at any point [38].

### 10.2 Compositional frameworks: Agent S and best-of-N

Simular's **Agent S** line takes a compositional approach [34]:

- a generalist planner model (for example, GPT-5);
- a specialist grounding model (UI-TARS-1.5-7B recommended) that maps described targets to
  coordinates through an agent-computer interface;
- an optional reflection agent;
- a code agent that runs Python and Bash, with a 30 s timeout per Bash script;
- a trajectory window that keeps 8 image turns by default.

The key result is **Behavior Best-of-N (bBoN)**: run several full rollouts and have a judge
pick the best trajectory. Reported gains [34]:

| Benchmark | Single rollout | With bBoN |
|---|---|---|
| OSWorld (100 steps) | 66% | 72.6% (Dec 2025) |
| WindowsAgentArena (3 rollouts) | 50.2% | 56.6% |
| AndroidWorld | 68.1% | 71.6% |

Simular's hosted agent, Sai, reportedly scored 73% on the 108-task OSWorld 2.0 (August 2026)
[34].

**Lesson.** For GUI tasks, test-time compute spent on whole trajectories plus a selector is as
effective as spending it inside a single trajectory. This only works when environments can be
snapshotted or replicated, so rollouts do not interfere with each other.

### 10.3 Open data and models: OpenCUA

**OpenCUA** (XLANG Lab, NeurIPS 2025 spotlight) open-sourced the full stack [32]:

- an annotation tool that records screen video, input events and a11y trees;
- the 22.6k-task AgentNet dataset;
- an offline benchmark;
- models trained with "reflective long chain-of-thought" that comments on the previous action,
  the alternatives and the expected next state.

OpenCUA-72B reached 45.0% on OSWorld-Verified at 100 steps. On the same table, Claude 4 Sonnet
scored 41.5% and OpenAI CUA 31.4% [32]. XLANG has since released CUA-Gym, a pipeline for
synthesizing verifiable training tasks, and CUA-Gym-Hub, a set of mock web apps packaged as RL
environments. Synthetic verifiable environments have become the training substrate for the
field [31].

### 10.4 Windows-native: UFO

Microsoft's **UFO2** ("Desktop AgentOS", April 2025) uses [37]:

- a HostAgent that coordinates per-application AppAgents;
- hybrid UIA-plus-vision control detection;
- mixed GUI and API actions with deep Win32 and COM integration;
- RAG over documents and past traces;
- speculative multi-action execution, which batches predicted actions and reports 51% fewer
  LLM calls.

**UFO3 Galaxy** (November 2025) generalizes this to multi-device DAG orchestration [37].

### 10.5 Hierarchical web agents: Agent-E

**Agent-E** (Emergence AI, 2024) used a planner over a browser-navigation agent. It introduced
**DOM distillation modes**: `text_only` for reading, `input_fields` for acting and `all_fields`
for both, so each step sees the smallest view sufficient for its purpose. It also used mutation
observers to verify that actions took effect [45]. Browser Use and Stagehand later adopted the
same ideas in their own forms: task-dependent views and change-aware observation.

---

## 11. Cross-system comparison

| System | Perception | Action space | Planner split | Runs where | Notable public numbers |
|---|---|---|---|---|---|
| OpenAI CUA / Operator | Pixels | Coordinates; later batched; code-as-action sample | Single model | OpenAI cloud browser; client harness via API | OSWorld 38.1%; WebVoyager 87% [13][18] |
| ChatGPT agent | Pixels + text browser + terminal | GUI, text browse, shell, APIs | Single agent, tool choice per step | OpenAI virtual computer | See [15] |
| Claude computer toolset | Pixels + zoom | 17 member tools, batched | Single model | Client VM or container | 14.9% to 61.4% (Verified) to 81.8% (OSWorld 2.1) [1][9][11] |
| Claude browser toolset | A11y refs primary + pixels | 31 member tools | Single model | Client browser | 50k-char page cap [4] |
| Gemini 2.5 CU / Mariner | Pixels | Normalized 0-1000 grid | Single model + safety service | Client browser; Mariner cloud VMs | Online-Mind2Web 69.0%; WebVoyager 88.9% [18] |
| Browser Use | Indexed DOM/AX + screenshot | Element-index actions, multi-action | Single LLM + todo file | Local, cloud or CDP | ~118k stars; $0.02/browser-hour [22] |
| Stagehand | Trimmed a11y tree | `act` / `observe` / `extract` in code | Developer code is the planner | Local or Browserbase | 2x speed claim [26] |
| Skyvern | Element tree + screenshot | Typed JSON actions + compiled scripts | Planner / actor / validator | Self-host or cloud | WebBench 64.4%; WebVoyager 85.8% [27] |
| Agent S3 | Pixels | Grounding model + code agent | Planner + grounder + reflection, bBoN | Local or VM | OSWorld 72.6% (bBoN) [34] |
| UI-TARS-1.5 | Pixels | Unified GUI actions | Native single model | Local | OSWorld 42.5%; ScreenSpot-Pro 61.6% [35] |
| OpenCUA-72B | Pixels | pyautogui-style | Single model with reflective CoT | Local | OSWorld-Verified 45.0% [32] |
| UFO2 | UIA + vision | GUI + API, speculative batching | Host/App agents | Windows | 51% fewer LLM calls [37] |

---

## 12. Failure modes observed across systems

1. **Grounding misses.** These are clicks a few pixels off, especially on small targets, at
   high DPI, or after unaccounted rescaling. The fixes are client-side resizing with coordinate
   rescaling, zoom actions, specialized grounding models, or a11y references [3][5][34].
2. **Stale or mismatched observations.** Examples include OS-rendered dropdowns missing from
   headless screenshots, content changing between screenshot and click, and brief
   notifications missed by "flipbook" observation [2][19]. Fixes include `*` new-element
   markers, page-change barriers that cut action batches short, and waits [24][25].
3. **Premature or false success.** The agent declares the task done without verifying it. This
   is why Browser Use's `done` requires evidence [25], Skyvern has validator prompts and
   evidence-backed subgoals [28], and Online-Mind2Web built a stricter judge [41].
4. **Overly early termination.** Agents give up on CAPTCHAs, loading states or empty results.
   Skyvern's planner explicitly forbids treating these as impossibility [28].
5. **Context exhaustion.** Screenshots fill 200k tokens in under 100 steps. The fix is
   screenshot pruning aligned with cache breakpoints, compaction, and file-based memory
   (`todo.md`) [5][25].
6. **Prompt injection.** Hidden DOM text, tab titles, URLs and images can redirect the agent.
   Unmitigated browser attacks succeeded 23.6% of the time in Anthropic's tests. Even with
   defenses, no vendor claims immunity [7][8].
7. **Irreversible actions.** These include purchases, messages, file deletion and accepting
   terms. Every vendor routes them through confirmations: in-band safety checks [16][19],
   product gates [7][14], or developer-enforced harness policy [3][4].
8. **Bot detection and authentication friction.** CAPTCHAs, 2FA and fingerprinting push
   production systems toward managed browsers with proxies and solvers, credential-vault
   integrations, and human takeover [22][27].
9. **Harness side effects.** A code-driven CUA can leave keys held down. An in-container loop
   can be damaged by the agent it controls. Sample apps add failsafes and refuse to start new
   runs until held input is released [6][17].

---

## 13. Cost and latency

Public numbers are sparse, but some anchor points exist:

- **Token cost per step.** A Claude screenshot costs about 1,000-1,800 input tokens. Declaring
  the toolset costs about 4,500 tokens once per request, mostly cacheable [3]. Older tool
  versions added 466-499 system-prompt tokens plus about 735 tokens per tool definition [3].
- **Thinking budget.** Lower effort settings can cut output tokens by 2-10x with little loss in
  accuracy (Section 4.3) [5].
- **Infrastructure.** Browser Use Cloud lists about $0.02 per browser-hour [22]. Skyvern and
  Browserbase sell anti-bot proxies and solvers as managed add-ons [26][27].
- **Calls per task.** UFO2's speculative multi-action cut LLM calls by 51% [37]. Batched
  actions in the 2026 OpenAI and Anthropic APIs have the same goal [3][16].
- **Latency.** Anthropic still lists latency as limitation number one. It recommends computer
  use where speed is not critical, such as background research and automated testing [3].

---

## Key takeaways for system designers

1. **Pick the cheapest faithful observation.** Use DOM or a11y references for web pages with
   reliable structure. Keep screenshots and zoom for verification and for surfaces structure
   cannot see. Pure pixels are for whole-desktop generality, not a default for web automation.
2. **Own the coordinate pipeline.** Resize screenshots yourself to the model's pixel budget,
   keep a scale factor, and rescale every returned coordinate. Choose resolution so the
   smallest target you care about stays several pixels wide [3][5].
3. **Batch actions, but put barriers in.** Allow multiple actions per turn. Stop at the first
   failure. Treat navigation as a barrier that ends the batch [3][25].
4. **Engineer the serializer, not just the prompt.** Paint-order occlusion, containment
   deduplication, new-element markers, compact compound controls and password stripping are
   where DOM agents win or lose [24].
5. **Make completion a verified claim.** Require evidence-backed subgoals or explicit
   verification before `done`. Use separate validator calls for high-value workflows [25][28].
6. **Compile what you have learned.** Explore with an agent, then freeze working paths into
   deterministic code, cached `act` calls or selectors. Call the model again only when the
   site changes or a step fails [26][28].
7. **Spend test-time compute on trajectories.** Best-of-N rollouts with a judge raised
   OSWorld scores by about 6 points for Agent S3. This needs environments you can snapshot and
   fork [34].
8. **Manage context like a cache.** Keep a few recent screenshots, prune in batches aligned
   with cache breakpoints, compact at a threshold, and keep long-horizon state in files rather
   than in images [5][25].
9. **Assume prompt injection will happen.** Use a fresh low-privilege profile, network-level
   domain allowlists that are re-checked after redirects, disabled `javascript_exec` and file
   upload by default, classifiers on untrusted content, and human confirmation for
   irreversible actions. Never pass secrets to the model; use takeover, `observe`-style
   selectors, or credential vaults [4][7][8][14][26][27].
10. **Distrust single benchmark numbers.** Check the benchmark version (OSWorld vs. Verified
    vs. 2.x), the step limit, the scoring rule (binary vs. partial), the judge, and who ran
    the evaluation. The same model can differ by 9 or more points between harnesses
    [10][18][31][41].
11. **Mix modalities per step.** The strongest products route each step to the cheapest
    capable channel: an API or connector, a shell, a text browser, an a11y-tree browser, or
    pixel control. ChatGPT agent's design and Claude's computer-plus-bash setup both reflect
    this [6][15].

---

## References

1. Anthropic. "Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku." 2024. https://www.anthropic.com/news/3-5-models-and-computer-use
2. Anthropic. "Developing a computer use model." 2024. https://www.anthropic.com/news/developing-computer-use
3. Anthropic. "Computer use tool" (Claude Platform documentation). 2026. https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
4. Anthropic. "Browser use toolset" (Claude Platform documentation). 2026. https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-tool
5. Anthropic. "Best practices for computer and browser use with Claude." 2026. https://claude.com/blog/best-practices-for-computer-and-browser-use-with-claude
6. Anthropic. "computer-use-demo" reference implementation (claude-quickstarts). 2024-2026. https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo
7. Anthropic. "Piloting Claude in Chrome." 2025. https://claude.com/blog/claude-for-chrome
8. Anthropic. "Mitigating the risk of prompt injections in browser use." 2025. https://www.anthropic.com/research/prompt-injection-defenses
9. Anthropic. "Introducing Claude Sonnet 4.5." 2025. https://www.anthropic.com/news/claude-sonnet-4-5
10. Anthropic. "Introducing Claude Sonnet 4.6." 2026. https://www.anthropic.com/news/claude-sonnet-4-6
11. Anthropic. "Claude Opus 5.5." 2026. https://www.anthropic.com/claude-opus-5-5
12. Anthropic. "Claude Sonnet 5.5." 2026. https://www.anthropic.com/claude-sonnet-5-5
13. OpenAI. "Computer-Using Agent." 2025. https://openai.com/index/computer-using-agent/
14. OpenAI. "Introducing Operator." 2025. https://openai.com/index/introducing-operator/
15. OpenAI. "Introducing ChatGPT agent: bridging research and action." 2025. https://openai.com/index/introducing-chatgpt-agent/
16. OpenAI. "Tools" (OpenAI Agents SDK documentation, ComputerTool section). 2026. https://github.com/openai/openai-agents-python/blob/main/docs/tools.md
17. OpenAI. "openai-cua-sample-app" (incl. python-app README). 2026. https://github.com/openai/openai-cua-sample-app
18. Google DeepMind. "Gemini 2.5 Computer Use Model Card." 2025. https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Computer-Use-Model-Card.pdf
19. Google. "computer-use-preview" (Gemini computer use reference agent; README and agent.py). 2025-2026. https://github.com/google-gemini/computer-use-preview
20. Google. "Introducing Gemini 2.0: our new AI model for the agentic era." 2024. https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/
21. Google DeepMind. "Project Mariner." 2024-2025. https://deepmind.google/technologies/project-mariner/
22. Browser Use. "browser-use" repository README. 2024-2026. https://github.com/browser-use/browser-use
23. Browser Use. "browser_use/dom/service.py" (DomService source). 2026. https://github.com/browser-use/browser-use/blob/main/browser_use/dom/service.py
24. Browser Use. "browser_use/dom/serializer/serializer.py" (DOMTreeSerializer source). 2026. https://github.com/browser-use/browser-use/blob/main/browser_use/dom/serializer/serializer.py
25. Browser Use. "browser_use/agent/system_prompts/system_prompt.md." 2026. https://github.com/browser-use/browser-use/blob/main/browser_use/agent/system_prompts/system_prompt.md
26. Browserbase. "Stagehand" repository README. 2024-2026. https://github.com/browserbase/stagehand
27. Skyvern AI. "Skyvern" repository README. 2024-2026. https://github.com/Skyvern-AI/skyvern
28. Skyvern AI. Prompt library (task_v2.j2, extract-action.j2, and others). 2026. https://github.com/Skyvern-AI/skyvern/tree/main/skyvern/forge/prompts/skyvern
29. Xie, T. et al. (XLANG Lab, HKU). "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." 2024. https://arxiv.org/abs/2404.07972
30. XLANG Lab. "OSWorld" repository (incl. OSWorld-Verified news). 2024-2026. https://github.com/xlang-ai/OSWorld
31. XLANG Lab. "OSWorld-V2" repository (OSWorld 2.0 / 2.1). 2026. https://github.com/xlang-ai/OSWorld-V2
32. XLANG Lab. "OpenCUA: Open Foundations for Computer-Use Agents" repository. 2025-2026. https://github.com/xlang-ai/OpenCUA
33. XLANG Lab. "OSWorld-G / Jedi" repository. 2025. https://github.com/xlang-ai/OSWorld-G
34. Simular AI. "Agent S" repository (Agent S2/S3, bBoN, Sai news). 2024-2026. https://github.com/simular-ai/Agent-S
35. ByteDance Seed. "UI-TARS" repository (UI-TARS-1.5, UI-TARS-2). 2025. https://github.com/bytedance/UI-TARS
36. Microsoft. "OmniParser" repository. 2024-2025. https://github.com/microsoft/OmniParser
37. Microsoft. "UFO / UFO2 / UFO3" repository. 2024-2025. https://github.com/microsoft/UFO
38. Microsoft. "Magentic-UI" repository. 2025-2026. https://github.com/microsoft/magentic-ui
39. Microsoft. "Fara" repository (Fara-7B, Fara1.5). 2025-2026. https://github.com/microsoft/fara
40. He, H. et al. "WebVoyager" repository. 2024. https://github.com/MinorJerry/WebVoyager
41. Xue, T. et al. (OSU NLP Group). "Online-Mind2Web / An Illusion of Progress? Assessing the Current State of Web Agents." 2025. https://github.com/OSU-NLP-Group/Online-Mind2Web
42. Zheng, B. et al. (OSU NLP Group). "SeeAct" repository. 2024. https://github.com/OSU-NLP-Group/SeeAct
43. Zhou, S. et al. "WebArena" repository. 2023-2024. https://github.com/web-arena-x/webarena
44. ServiceNow. "BrowserGym" repository. 2024-2026. https://github.com/ServiceNow/BrowserGym
45. Emergence AI. "Agent-E" repository. 2024. https://github.com/EmergenceAI/Agent-E
