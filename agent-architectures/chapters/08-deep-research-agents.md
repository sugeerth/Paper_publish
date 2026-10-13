# 08. Deep Research Agents: Architectures, Products, and Training

## Overview

A "deep research" agent takes an open-ended question and spends minutes, not seconds, on it. It plans, runs many searches, reads full documents, often runs code, and returns a long report with citations. The category became public in December 2024, when Google added Deep Research to Gemini. OpenAI followed on February 2, 2025, Perplexity on February 14, 2025, and Anthropic (Claude Research) on April 15, 2025 [2][7][13][19]. A large open-source and academic ecosystem grew alongside these products. It includes prompt-orchestrated pipelines such as STORM, GPT Researcher, LangChain Open Deep Research and Hugging Face's smolagents replication. It also includes models trained with reinforcement learning (RL) to search: Search-R1, DeepResearcher, WebThinker, the Tongyi WebAgent family, ASearcher and MiroThinker [21][24][26][29][33][34][35][36][38][39].

Deep research systems differ from classic retrieval-augmented generation (RAG) in three ways:

1. **Adaptive multi-step retrieval.** RAG fetches the most similar chunks once. A deep research agent runs a search loop that changes direction as it learns. Anthropic puts the contrast directly: static retrieval returns similar chunks, while its system "uses a multi-step search that dynamically finds relevant information, adapts to new findings" [1].
2. **Long horizons.** Runs span tens to hundreds of tool calls. MiroThinker v1.0 advertises up to 600 tool calls per task, and ASearcher saw trajectories of more than 100 tool turns and more than 400k generated tokens during RL training [38][39]. Context management therefore becomes a primary design concern.
3. **Synthesis output.** The deliverable is usually a structured, cited report rather than a short answer. Systems that target short answers, which Jina calls "DeepSearch", deliberately separate the two problems [31].

This chapter covers the commercial systems first, from most to least publicly documented. It then covers the open-source orchestration frameworks, the RL-trained "agentic model" line of work, and evaluation. It ends with a cross-system comparison and design lessons.

A note on sourcing: this chapter was compiled with restricted network access. Anthropic, OpenAI-cookbook, Google-cookbook and GitHub primary sources were read directly. Claims taken from the OpenAI, Google and Perplexity launch posts that could not be re-fetched are marked "per launch post".

## A generic reference architecture

Almost every documented system is a variant of the following loop:

```
  user query
      |
      v
 +-----------+   clarifying Qs   +------+
 | Scoping / |<----------------->| user |
 | clarifier |                   +------+
 +-----+-----+
       | research brief / plan
       v
 +------------------+   spawn / call    +---------------------------+
 | Planner /        |------------------>| Research worker(s)        |
 | orchestrator     |<------------------| search -> fetch -> read   |
 | (plan in memory) | condensed findings| -> reflect (ReAct loop)   |
 +--------+---------+                   +-------------+-------------+
          | enough evidence?                          | tools
          v                                           v
 +------------------+                     web search, URL fetch, code,
 | Writer /         |                     file search, MCP connectors
 | synthesizer      |
 +--------+---------+
          v
 +------------------+
 | Citation pass /  | --> final report (markdown, inline citations)
 | verifier         |
 +------------------+
```

The systems differ along five axes:

- **Who decides control flow.** Either a fixed graph decides (STORM, early GPT Researcher, the Gemini quickstart) or the model decides (OpenAI deep research, Tongyi DeepResearch, Claude Research).
- **Single context or multiple.** One long trajectory may hold everything (OpenAI, WebThinker, Tongyi ReAct mode), or work may be split into isolated worker contexts that return compressed summaries (Anthropic, LangChain ODR, GPT Researcher).
- **Prompting or training.** Behavior comes from prompts plus a frontier model, or from RL that teaches the model the loop itself (OpenAI, Kimi-Researcher, Search-R1 and its descendants).
- **Scoping and planning UX.** Options include clarifying questions (ChatGPT), an editable plan (Gemini), or none at all (most APIs).
- **Grounding and citation.** Citations are either produced inline during writing or attached afterwards by a dedicated citation pass (Anthropic).

## OpenAI deep research

### Problem and positioning

OpenAI launched deep research in ChatGPT on February 2, 2025 [7][42]. Per the launch post, it runs on a version of the o3 reasoning model optimized for web browsing and data analysis. It was trained with end-to-end reinforcement learning on hard browsing and reasoning tasks, so that it learns to plan and run multi-step search trajectories and to backtrack when it finds new information. Tasks take roughly 5 to 30 minutes [7]. The headline numbers were 26.6% on Humanity's Last Exam (per launch post) and an average of about 67% on the GAIA validation set [7][29]. The survey by Huang et al. tabulates GAIA validation scores of 78.7 / 73.2 / 58.0 on levels 1 to 3, averaging 67.4, and lists the base model as "GPT-o3-customized" [42]. OpenAI also published a system card for the product [10].

### Architecture

OpenAI has not published the internal agent graph. The design is a single agentic model that holds the loop itself, not a hand-written multi-agent pipeline. The Deep Research API, released with dated model snapshots `o3-deep-research-2025-06-26` and `o4-mini-deep-research-2025-06-26`, exposes the trajectory [8]:

```
 Responses API request (background=True)
   tools: web_search_preview (required), code_interpreter (opt), MCP (opt)
        |
        v
 +---------------------------------------------------------------+
 | o3-deep-research / o4-mini-deep-research (single agent)        |
 |  reasoning step -> web_search_call -> reasoning -> ...         |
 |  -> code_interpreter_call -> mcp_tool_call -> ... -> message   |
 +---------------------------------------------------------------+
        |
        v
 response.output: [reasoning summaries, search calls, code calls,
                   MCP calls, final message w/ url_citation annotations
                   (start_index, end_index, title, url)]
```

The cookbook documents the following details [8][9]:

- **Background mode.** Tasks take several minutes, so callers are expected to set `background=True` and poll or stream, which avoids timeouts.
- **Trajectory transparency.** Each intermediate step appears in `response.output`, typed as a reasoning summary, a web search call (including the query issued), a code interpreter call or an MCP call. Citations are character-span annotations on the final text.
- **Clarification lives outside the research model.** In ChatGPT, a lighter model ("like gpt-4.1") asks the user follow-up questions and rewrites the prompt before the research model starts. The API skips this step: "the model expects fully-formed prompts up front and will not ask for additional context". The cookbook recommends that developers rebuild that front end with a prompt rewriter.
- **Private data via MCP.** Deep research models can call MCP servers. The cookbook's example is a proxy over a vector store of internal documents.
- **Reference multi-agent wrapper.** The "Deep Research Agents" cookbook builds a four-agent pipeline on the Agents SDK: a Triage agent, a Clarifier agent, an Instruction Builder that turns enriched input into a precise research brief, and a Research agent with web search and MCP [9]. It warns against using deep research for "trivial fact lookups, simple Q&A, or short-form chat", which a plain Responses call handles faster and more cheaply.

The simpler `research_bot` example in the Agents SDK shows the "workflow" alternative. A planner agent emits a list of search terms, each with a reason. Search agents run them in parallel and summarize. A writer agent composes the report. The listed improvements include an evaluation step that lets the model "choose to improve its results, search for more stuff" [12].

### Failure modes and lessons

Per the launch post, OpenAI acknowledged that the system can hallucinate facts or make wrong inferences. It can also struggle to tell authoritative information from rumor and show weak confidence calibration [7]. The API design makes one lesson concrete: when a research run costs minutes and many tool calls, scoping the query precisely pays off. The cookbook notes that under-specified prompts lead to "verbosity, higher latency, and increased token usage" because the model hedges across interpretations [8].

## Anthropic: Claude Research (orchestrator-worker)

### Problem

Anthropic's Research feature launched in beta on April 15, 2025. Claude "operates agentically, conducting multiple searches that build on each other while determining exactly what to investigate next", and returns cited answers "in minutes". It can also search Gmail, Calendar and Google Docs through the Google Workspace integration [2]. In June 2025, Anthropic published the most detailed public engineering account of any commercial deep research system [1].

### Architecture

The design is orchestrator-worker. A **LeadResearcher** analyzes the query, writes its plan to an external **Memory**, and spawns parallel **Subagents**. Each subagent has its own context window, prompt and tools, and searches iteratively. The lead synthesizes the results and decides whether to spawn more subagents. When the lead decides it has enough, the draft and the source documents go to a separate **CitationAgent**, which places the citations [1].

```
 user --> LeadResearcher (extended thinking; plan saved to Memory)
             |  spawn 3-5 in parallel (more for hard queries)
     +-------+---------+----------------+
     v                 v                v
  Subagent A        Subagent B       Subagent C     each: own context,
  search/fetch      search/fetch     search/fetch   interleaved thinking,
  (3+ tools in      ...              ...            3+ parallel tool calls
   parallel)
     |                 |                |
     +---- condensed findings (or refs to files/artifacts) ----+
                         v
             LeadResearcher: synthesize; more subagents? --yes--> loop
                         | no
                         v
             CitationAgent (docs + draft -> claims mapped to sources)
                         v
                  final cited report
```

The production prompts are published in Anthropic's cookbook and add detail beyond the blog post [4][5][6]:

- **Query typing.** The lead classifies each query as *depth-first* (one question seen from many perspectives), *breadth-first* (independent sub-questions) or *straightforward*, and builds a different plan for each type [4].
- **Subagent count rules.** Use 1 subagent for simple queries, 2 to 3 for standard ones, 3 to 5 for medium and 5 to 10 for high complexity, with a ceiling of 20: "Prefer fewer, more capable subagents over many overly narrow ones" [4].
- **Delegation contract.** Each subagent prompt must state an objective, an output format, background context, key questions, suggested sources and sources to avoid, tools to use, and scope boundaries [4].
- **Subagent budget and loop.** A subagent sets its own "research budget": fewer than 5 tool calls for simple tasks, about 10 for hard ones, up to 15 for very difficult ones. It then runs an OODA (observe-orient-decide-act) loop. Queries should be short ("under 5 words") and broad at first. Full pages are fetched with `web_fetch` after `web_search`. Internal tools such as Drive, Gmail and Calendar take priority when they are present [5].
- **Citation agent constraints.** The citation agent must leave the text unchanged except for added citations, including whitespace. It must cite only directly supported claims and avoid fragmenting sentences [6].

### Scale, cost and latency (published numbers)

- A multi-agent system with an Opus 4 lead and Sonnet 4 subagents outperformed single-agent Opus 4 by **90.2%** on Anthropic's internal research eval [1].
- On BrowseComp, three factors explained **95%** of performance variance. **Token usage alone explained 80%**. The number of tool calls and the model choice accounted for the rest [1].
- Agents use about **4x** the tokens of a chat interaction. Multi-agent systems use about **15x** [1].
- Parallel subagents plus parallel tool calls cut research time "by up to 90%" for complex queries [1].
- Context beyond 200,000 tokens gets truncated, which is why the plan is persisted to memory [1].
- A tool-testing agent that rewrote a flawed MCP tool's description cut task completion time by **40%** for later agents [1].

### Prompting and evaluation lessons

Anthropic lists eight prompting principles [1]:

1. Simulate agents step by step to see how they fail.
2. Teach the orchestrator to delegate. Vague instructions caused duplicated work: two subagents researched the 2025 supply chain while a third covered the 2021 chip crisis.
3. Scale effort to query complexity. Early versions spawned 50 subagents for simple queries.
4. Design tools carefully and make agents inspect them before use.
5. Let Claude improve its own prompts and tool descriptions.
6. Search wide first, then narrow.
7. Guide thinking: extended thinking for the lead's planning, interleaved thinking after each tool result.
8. Parallelize tool calls.

For evaluation, they started with about 20 queries, because early prompt changes had effects large enough to show on a small set. They used a single LLM-as-judge call with a rubric covering factual accuracy, citation accuracy, completeness, source quality and tool efficiency. They judged end state rather than path. Human testers caught a bias that automated evals missed: agents preferred SEO content farms over authoritative but less highly ranked sources [1].

### Production engineering

Agents are long-running and stateful, so errors compound. The system resumes from checkpoints instead of restarting, and lets the model adapt when tools fail. Updates ship as **rainbow deployments**, which shift traffic gradually so that agents mid-run are not broken. Tracing monitors decision patterns without monitoring conversation contents. The main architectural limit is synchronous execution. The lead waits for each batch of subagents, cannot steer them mid-flight, and one slow subagent blocks the batch. Asynchronous execution is named as future work [1].

Two appendix patterns are widely reused. Subagents write large outputs to a **filesystem** and return lightweight references, which avoids a "game of telephone" through the lead. Near context limits, agents **summarize completed phases** and spawn fresh subagents with clean contexts [1]. Anthropic's later context-engineering guide generalizes the idea: subagents may consume tens of thousands of tokens but return condensed summaries of "often 1,000-2,000 tokens" [3]. Academic work has since pushed the filesystem idea further. FS-Researcher (2026) splits deep research into a Context Builder agent and a Report Writer agent that share a persistent file-system knowledge base, so a run can scale beyond one context window [43].

### Where it does not fit

Multi-agent research suits "breadth-first queries that involve pursuing multiple independent directions simultaneously". It suits tasks needing shared context or many inter-agent dependencies poorly, which covers most coding work [1].

## Google: Gemini Deep Research

### Product evolution

Gemini Deep Research launched on December 11, 2024. It was the first mainstream product in this category [13][42]. Per the launch post, the agent first proposes a multi-step research plan that the user can review and edit, then browses the web iteratively and compiles a report [14]. The product was later moved to Gemini 2.5 Pro [15]. The survey lists the original base model as Gemini-2.0-Flash and marks its training as undisclosed [42].

### API architecture (2026)

The Gemini API now exposes Deep Research as an **agent**, not a model. It is reachable only through the asynchronous **Interactions API**, not through `generate_content` [16][17]. The cookbook documents the following [16]:

- **Two tiers.** `deep-research-preview-04-2026` is tuned for speed and lower latency in interactive use. `deep-research-max-preview-04-2026` is tuned for "maximum search exhaustiveness and report comprehensiveness", suited to automated reporting. A single task "can consult over 100 sources".
- **Background execution.** Calls run with `background=True` and are polled by interaction ID. Streaming emits `interaction.start`, `content.delta` (text, `thought_summary` or image) and `interaction.complete` events. The stream can be resumed after a disconnect using the last event ID.
- **Collaborative planning as a protocol.** With `agent_config.collaborative_planning=True`, the agent returns a plan instead of executing. The user refines it over several turns linked by `previous_interaction_id`. Execution starts only when the flag is set back to false. The cookbook stresses that typing "go ahead" without flipping the flag does nothing.
- **Tools.** Defaults are `google_search`, `url_context` and `code_execution`. `mcp_server` (no-auth, bearer or OAuth, with `allowed_tools` filtering) and `file_search` over uploaded corpora are optional.
- **Visualizations.** `visualization="auto"` lets the agent produce charts and infographics, but only when the prompt asks for them.
- **Multimodal grounding.** Inputs can be images and PDFs, which the agent researches against the web.
- **Cheap follow-ups.** Follow-up questions on a finished report are answered by a standard Gemini model, not the research agent, so they return immediately.

```
 create(agent=deep-research[-max], background=True, agent_config={...})
     |
     +-- collaborative_planning=True --> plan <--> user refinements
     |                                   (previous_interaction_id chain)
     v  collaborative_planning=False
 [agent loop: google_search / url_context / code_execution / MCP / file_search]
     |  stream: thought_summary deltas, text deltas, image deltas
     v
 report (+ charts)  --> follow-ups served by a standard Gemini model
```

### Reference implementation and research

Google's open **gemini-fullstack-langgraph-quickstart** shows a minimal reflective loop. It generates initial queries, runs Gemini with Google Search grounding for each, then reflects and analyzes knowledge gaps. If gaps remain, it generates follow-up queries, up to a configured maximum number of loops. Finally it writes a cited answer. In production it uses Redis as a pub-sub broker for streaming background runs, and Postgres for thread state and an "exactly once" task queue [18]. The survey's GAIA table also lists Google's "Deep Researcher with Test-Time Diffusion" (Gemini-2.5-Pro, 69.1 average on the GAIA test set). That work treats report writing as iterative denoising of a draft [42][45].

### Lessons

Google's design puts the human-in-the-loop step in the plan, not in clarifying questions. In the API it is an explicit state machine (plan mode versus execute mode), not free-text intent detection. Separating a fast tier from an exhaustive tier, and routing follow-ups to a cheaper model, are both cost-latency controls exposed directly to developers.

## Perplexity Deep Research

Perplexity launched Deep Research on February 14, 2025 [19][42]. Per the launch post, it runs an iterative loop: it searches, reads documents and reasons about what to do next, refining its plan as it learns. It then synthesizes a report, completing most tasks in under three minutes. Perplexity reported 21.1% on Humanity's Last Exam and 93.9% on SimpleQA (per launch post) [19]. The capability is also offered to developers as a Sonar research model (`sonar-deep-research`) [20]. Perplexity has not published an internal architecture comparable to Anthropic's. Its distinctive design point is latency: it targets minutes rather than tens of minutes, which suits a search-engine product where users expect fast answers.

## Other commercial systems (brief)

The survey by Huang et al. catalogs several more closed systems [42]:

- **Grok DeepSearch** (Grok 3), with RL undisclosed.
- **Microsoft Copilot Researcher** (o3-mini).
- **Kimi-Researcher** (Moonshot, June 2025), trained with REINFORCE and rule-based outcome rewards [44].
- **Genspark Super Agent**, a mixture of agents; the survey reports 73.1 average on GAIA validation.
- **Manus** (71.4 on GAIA validation).
- **H2O.ai**'s deep research (79.73 average on the GAIA test set with Claude-3.7-Sonnet).

These figures come from the survey's tables, which compile vendor-reported results. Treat them as approximate and not directly comparable across splits.

## Open-source orchestration frameworks

### STORM and Co-STORM (Stanford OVAL)

**Problem.** STORM writes Wikipedia-style long articles from scratch using internet search. It was presented at NAACL 2024 [21][23]. The authors identify the core of the task as "automatically coming up with good questions to ask", and found that directly prompting an LLM to ask questions works poorly [23].

**Architecture.** STORM has two stages [23]:

1. **Pre-writing.** The system collects references and builds an outline, using two techniques:
   - **Perspective-guided question asking.** It surveys articles on similar topics to discover perspectives, then uses those perspectives to drive question generation.
   - **Simulated conversation.** A "Wikipedia writer" agent interviews a "topic expert" agent whose answers are grounded in retrieved sources, so follow-up questions build on what has been learned.
2. **Writing.** The system writes the full article from the outline and references, with citations, then polishes it by adding a summary section and removing duplicates.

The implementation in DSPy assigns a different model to each role. The default pairs a cheap model (`gpt-3.5-turbo`) for conversation simulation and question asking with a stronger model (`gpt-4o`) for outline generation, article generation and polishing. Retrieval is pluggable: You.com, Bing, Serper, Brave, SearXNG, DuckDuckGo, Tavily, Google, Azure AI Search, or a `VectorRM` over the user's own documents [23].

```
 topic --> find related articles --> perspectives P1..Pk
              |
   for each Pi: writer(Pi) <-> expert(grounded in search)  [simulated dialogue]
              |
       collected Q/A + URLs --> outline LM --> outline
                                   |
                     article LM (per section, cite refs) --> polish LM --> article
```

**Co-STORM** (EMNLP 2024) turns STORM into a human-AI system. It has several LLM "expert" agents, a **Moderator** that raises questions inspired by retrieved but unused information, and the human user, who can observe or steer. A dynamically updated **mind map** organizes collected information into a concept hierarchy and reduces cognitive load in long sessions [22][23].

**Results and limits.** The live preview has had more than 70,000 users. The authors say the output "cannot produce publication-ready articles", but experienced Wikipedia editors found it helpful for pre-writing [23]. STORM's lasting contribution is the idea that question generation from multiple perspectives determines research breadth. Later outline-driven systems such as WebWeaver adopt similar dynamic-outline ideas [36].

### GPT Researcher

**Architecture.** GPT Researcher is one of the earliest open deep research agents. It uses "planner" and "execution" agents [24]:

1. Create a task-specific agent for the query.
2. Generate research questions that together form an objective view.
3. Run a crawler agent per question, in parallel.
4. Summarize and source-track each resource.
5. Filter and aggregate everything into a report of more than 2,000 words drawn from more than 20 sources.

A context filter (Jev by default, with keyword fallback) selects passages before they reach the LLM. The **deep research mode** performs recursive, tree-like exploration with configurable depth and breadth. It takes about 5 minutes and costs about $0.4 per run using o3-mini at high reasoning effort. A standard report costs about $0.115 [24]. GPT Researcher works as an MCP client (`RETRIEVER=tavily,mcp`) and as an MCP server (gptr-mcp) [24].

**Multi-agent variant.** A LangGraph version uses a team of eight [25]:

- **Human**, who oversees the process and gives feedback.
- **Chief Editor**, which coordinates the team.
- **Researcher**, which is GPT Researcher itself.
- **Editor**, which plans the outline.
- **Reviewer** and **Revisor**, which loop over each draft per outline section in parallel.
- **Writer**, which compiles the final report.
- **Publisher**, which exports PDF, Docx and Markdown.

An average run yields a 5 to 6 page report [25]. This is an explicit editorial workflow, the opposite end of the spectrum from a single RL-trained agent.

### LangChain Open Deep Research

LangChain's **open_deep_research** is the best-documented example of a workflow evolving into an agent. Its legacy folder keeps two earlier designs [26]:

- A plan-and-execute **workflow** with human approval of the plan and sequential section writing with reflection.
- A **supervisor-researcher multi-agent** version.

The current graph has four stages [26]:

```
 clarify_with_user --> write_research_brief --> research_supervisor
                                                   |  tools: think_tool,
                                                   |  ConductResearch(topic),
                                                   |  ResearchComplete
                                    +--------------+--------------+
                                    v              v              v
                               researcher     researcher     researcher
                               (ReAct: search/MCP/think)  ...
                                    |              |              |
                              compress_research (per researcher)
                                    +--------------+--------------+
                                                   v
                                       final_report_generation
```

Key techniques:

- The supervisor delegates through tool calls, using `ConductResearch` and `ResearchComplete`. It stops on `ResearchComplete`, when no tool calls remain, or when it exceeds `max_researcher_iterations` (default 6).
- Concurrency is capped by `max_concurrent_research_units` (default 5), and each researcher is limited to `max_react_tool_calls` (default 10).
- Each researcher's findings are **compressed** before returning to the supervisor.
- Different models serve different roles. Defaults are `gpt-4.1-mini` for summarizing search results and `gpt-4.1` for research, compression and final writing [26].

**Numbers.** On DeepResearch Bench, the default configuration scored a RACE of 0.4309 at a cost of $45.98 and about 58M tokens for the 100 tasks. With Claude Sonnet 4 as the research model it scored 0.4401 at $187.09 and about 139M tokens. With GPT-5 it scored 0.4943 using about 205M tokens. The submitted configuration ranked 6th on the leaderboard (0.4344) in August 2025 [26]. These numbers show how quality and token spend move together, the same trend Anthropic reports for BrowseComp. LangChain's write-up on the evolution of the design is framed around the "bitter lesson": structure that helps weaker models becomes a bottleneck as models improve [27][28].

### Hugging Face smolagents Open Deep Research

Within about a day of OpenAI's launch, Hugging Face published an open replication built on smolagents [29]. Its key choices are a **CodeAgent**, which expresses actions as Python code rather than JSON tool calls, and a text-based web browser tool. The agent reached **55% pass@1** on the GAIA validation set, compared with **67%** for OpenAI's deep research, using `o1` as the default model [29]. The result showed that much of the GAIA gap could be closed by orchestration alone, and that the rest likely comes from training.

### Minimal and "DeepSearch" variants

- **dzhng/deep-research** is deliberately under 500 lines of code. It takes `breadth` and `depth` parameters. Each level issues SERP queries through Firecrawl, extracts "learnings" and new "directions", and recurses with prior goals and learnings as context until depth reaches 0. It then writes a Markdown report. Searches run concurrently [30].
- **Jina node-DeepResearch** loops search, read and reason until it finds an answer or exhausts a token budget. It explicitly does *not* optimize for long reports: "that's a completely different problem" [31]. Its demo traces range from 2 steps for simple lookups to 42 steps for speculative questions. The hosted API is OpenAI-compatible (`jina-deepsearch-v1`) and returns footnote citations [31].

## Training deep research models with reinforcement learning

The second lineage moves the loop into the model's weights. It follows the shift from prompted reasoning to RL-trained reasoning (o1, DeepSeek-R1) and applies the same idea to tool use.

### Precursors: Search-o1 and agentic RAG

**Search-o1** (EMNLP 2025) adds agentic RAG to large reasoning models. The model emits search queries mid-reasoning whenever it is uncertain. A **Reason-in-Documents** module refines the retrieved documents before injecting them back into the chain, which reduces noise. Inference batches generation across sequences and interleaves retrieval [32].

### Search-R1 and DeepResearcher: outcome-reward RL with search

**Search-R1** extends DeepSeek-R1(-Zero)-style RL to interleaved reasoning and search, built on veRL. It supports PPO, GRPO and REINFORCE, local sparse or dense retrievers, and online search engines [33]. The survey classifies its reward as rule-based outcome reward trained on NQ and HotpotQA [42]. It describes itself as "an alternative and open solution to OpenAI DeepResearch" [33].

**DeepResearcher** scales RL into *real-world* web environments rather than a local Wikipedia index. It runs GRPO with live Serper or Bing search and a distributed handler for browsing. It reports gains of up to 28.9 points over prompt-engineering baselines and up to 7.2 points over RAG-based RL agents. The authors observed emergent behaviors: forming plans, cross-validating across sources, self-reflection to redirect research, and "maintain[ing] honesty when unable to find definitive answers". They argue that training on the real web is "a fundamental requirement" rather than an implementation detail [34].

### WebThinker: research inside the thinking process

**WebThinker** (NeurIPS 2025) lets a reasoning model, QwQ-32B in the main experiments, act *during* its chain of thought [35]:

- A **Deep Web Explorer** can search, click links and buttons, and extract information.
- An **Autonomous Think-Search-and-Draft** strategy provides three report tools: draft a section, check the current report, and edit the report.

So writing happens alongside search rather than after it. Training uses iterative online DPO on preference pairs built from correctness, tool use and output quality. An auxiliary instruct model handles page reading and report editing [35]. The survey lists its GAIA validation average as 44.7 [42].

### Tongyi DeepResearch and the WebAgent family (Alibaba)

**Tongyi DeepResearch** (September 2025) is a 30.5B-parameter mixture-of-experts model with 3.3B active parameters per token and a 128K context [36]. The project is the most complete open recipe for training a deep research model:

- **Fully automated synthetic data.** Synthesis feeds agentic continual pre-training, SFT and RL.
- **Agentic continual pre-training** on agent interaction data [36].
- **End-to-end on-policy RL** using a customized GRPO with token-level policy gradients and leave-one-out advantage estimation. Negative samples are selectively filtered to keep training stable "in a non-stationary environment" [36].
- **Two inference modes.** Plain **ReAct** evaluates intrinsic ability. A **"Heavy" mode based on IterResearch** adds test-time scaling [36].
- **Tools in the reference stack.** Serper handles web search and Google Scholar, Jina Reader handles page reading, an LLM summarizes pages, Dashscope parses files, and SandboxFusion runs Python [36].

The surrounding paper family breaks the problem into parts [36][37]:

- **WebWalker** (ACL 2025): a benchmark for web traversal.
- **WebDancer** (NeurIPS 2025): a four-stage pipeline of browsing data construction, trajectory sampling, SFT cold start, and DAPO RL. It reports 64.1% Pass@3 on GAIA [37].
- **WebSailor**: SailorFog-QA, built with graph sampling and information obfuscation to create high-uncertainty questions, plus an RFT cold start followed by **DUPO** (Duplicating Sampling Policy Optimization). WebSailor-72B reports 12.0% on BrowseComp-en, 30.1% on BrowseComp-zh and 55.4% on GAIA [37].
- **WebShaper**: formalization-driven data synthesis with an agentic Expander. It reports 60.19 on GAIA and 52.50 on WebWalkerQA [37].
- **Long-horizon context work**:
  - **ReSum** summarizes context to unlock longer searches.
  - **WebResearcher** targets "unbounded reasoning" in long-horizon agents.
  - **AgentFold** adds proactive context folding.
  - **WebWeaver** structures evidence with dynamic outlines for open-ended reports.
  - **ParallelMuse** explores agentic parallel thinking.
  - **BrowseConf** applies confidence-guided test-time scaling [36].

### ASearcher: asynchronous RL for very long horizons

**ASearcher** (inclusionAI, built on AReaL) identifies a systems bottleneck. Trajectory lengths vary enormously during agentic RL. Long trajectories need dozens more turns than short ones, and up to two orders of magnitude more tokens. Batch-synchronous RL therefore leaves GPUs idle while it waits for the longest rollout. **Fully asynchronous RL** decouples rollout from training. This allows relaxed turn limits of 128 turns per trajectory and produced trajectories with more than 100 tool calls and more than 400k tokens [38].

A **data synthesis agent** creates hard questions by:

- **fuzzing**, which obscures key details, and
- **context injection**, which adds externally retrieved facts.

Each question is then validated for quality, difficulty (an LRM must fail it) and answer uniqueness [38].

ASearcher-Web-QwQ reports Avg@4 scores of 58.7 on GAIA, 51.1 on xBench-DeepSearch and 74.5 on Frames. RL contributed gains of +15.0, +22.4 and +14.6 respectively [38].

### MiroThinker: "interactive scaling"

**MiroThinker** (MiroMind) frames *interaction depth* as a third scaling axis alongside model size and context length. The agent is trained to handle "deeper and more frequent agent-environment interactions" [39]. Published configurations:

| Version | Context | Max tool calls / task | Reported results |
|---|---|---|---|
| v1.0 (8B / 30B / 72B), Nov 2025 | 256K | 600 | 72B: HLE-Text 37.7%, BrowseComp 47.1%, BrowseComp-ZH 55.6%, GAIA-Text-103 81.9% |
| v1.5 (30B / 235B), Jan 2026 | 256K | 400 | 235B: HLE-Text 39.2%, BrowseComp 69.8%, BrowseComp-ZH 71.5%, GAIA-Val-165 80.8% |
| 1.7 (30B "mini" / 235B), Mar 2026 | 256K | 300 | 235B: BrowseComp 74.0%, BrowseComp-ZH 75.3%, GAIA-Val-165 82.7%, HLE-Text 42.9% |

The maximum number of tool calls *decreases* across versions while scores rise. One plausible reading, offered as interpretation rather than vendor claim, is that later training made each interaction more efficient. The repository headline also claims 88.2 on BrowseComp for the proprietary MiroThinker-H1 agent [39].

### Kimi-Researcher

Moonshot's **Kimi-Researcher** (June 2025) belongs to the same lineage of end-to-end agentic RL. The survey records REINFORCE with rule-based outcome rewards on Kimi k1.5/k2 [42][44].

### What the RL line has established

The papers above converge on the following points [33][34][36][38][39]:

1. **Outcome rewards on verifiable QA suffice to teach the search loop.** Behaviors such as planning, cross-checking, reflection and abstention emerge without being specified.
2. **Data is the bottleneck.** Benchmarks such as NQ and HotpotQA are too easy. Every serious recipe synthesizes high-uncertainty, multi-hop, obfuscated questions and filters them by difficulty against a strong model.
3. **Training infrastructure matters.** Long-tailed rollout lengths need asynchronous RL, and live-web training needs robust tool servers and rate-limit handling.
4. **Context management becomes a learned skill.** Summarization, folding and outline-structured memory replace the external orchestrator's compression step.

## Evaluation of deep research agents

### Short-answer browsing benchmarks

- **BrowseComp** (OpenAI): hard-to-find facts. A reference implementation lives in `simple-evals`, which is no longer updated for new models but still hosts BrowseComp [11].
- **GAIA**: general assistant tasks at three difficulty levels.
- **Humanity's Last Exam (HLE)**, **xBench-DeepSearch**, **Frames**, **WebWalkerQA** and **BrowseComp-ZH**.

These benchmarks grade a final answer and so suit RL rewards. They do not measure report quality.

### Report-quality benchmarks

**DeepResearch Bench** has 100 PhD-level tasks, 50 in English and 50 in Chinese, across 22 fields. Expert-written tasks follow the topic distribution of 96,147 anonymized real queries [40]. It has two scoring frameworks:

- **RACE** (Reference-based Adaptive Criteria-driven Evaluation) generates task-specific criteria and weights across comprehensiveness, insight/depth, instruction following and readability. It scores reports *relative to* expert reference reports.
- **FACT** extracts statement-URL pairs, deduplicates them, and checks support by scraping the source and asking an LLM judge. It reports citation accuracy and the number of effective citations [40].

The judge itself must be maintained. In May 2026, after Gemini-2.5-Pro was deprecated, the maintainers benchmarked candidate judges against human annotations on 200 articles. The human inter-annotator agreement baseline was 68.78%, and GPT-5.5 reached 71.82% overall agreement. Scores from the two judges are kept on separate leaderboards rather than mixed [40]. **DeepResearch Bench II** (February 2026) replaces holistic judging with 9,430 fine-grained binary rubrics covering information recall, analysis and presentation, derived from expert articles [41].

### Production evaluation

Anthropic's approach is the most detailed public account: a small eval set early on, an LLM judge with a rubric, end-state grading, and human review that catches biases such as source-quality problems [1]. LangChain's numbers show that running a 100-task report benchmark costs about $20 to $100 or more depending on models, so evaluation cost is itself a design constraint [26].

## Cross-system comparison

| System | Control flow | Contexts | Behavior source | Scoping UX | Citation mechanism | Public scale / perf numbers |
|---|---|---|---|---|---|---|
| OpenAI deep research | Model-driven single agent | One long trajectory | End-to-end RL on o3 variant (per launch post) | Clarifier model in ChatGPT; none in API | Inline span annotations | ~5-30 min/task; HLE 26.6%; GAIA val ~67% [7][29][42] |
| Claude Research | Lead + parallel subagents | Isolated per subagent; plan in memory | Prompted frontier models (Opus/Sonnet) | Lead plans internally | Separate CitationAgent pass | +90.2% vs single agent; ~15x chat tokens; up to 90% faster [1] |
| Gemini Deep Research | Agent with plan/execute modes | Not disclosed | Not disclosed | Editable plan (collaborative_planning) | Report citations | >100 sources/task; standard vs Max tiers [16] |
| Perplexity Deep Research | Iterative search-read-reason | Not disclosed | Not disclosed | None documented | Inline citations | Most tasks <3 min; HLE 21.1%, SimpleQA 93.9% (per launch post) [19] |
| STORM / Co-STORM | Fixed pipeline (DSPy) | Per-role LMs | Prompting | Topic only; Co-STORM human steering | Per-section citation | 70k+ preview users [23] |
| GPT Researcher | Planner -> parallel crawlers -> publisher | Per-question workers | Prompting | Optional human (multi-agent) | Source tracking | ~$0.115/report; deep ~5 min, ~$0.4 [24] |
| LangChain ODR | Clarify -> brief -> supervisor -> researchers | Per-researcher, compressed | Prompting | Clarification node | Final writer | RACE 0.43-0.49; $46-$187 per 100 tasks [26] |
| HF smolagents ODR | CodeAgent loop | Single | Prompting (o1) | None | Inline | GAIA val 55% pass@1 [29] |
| Tongyi DeepResearch | ReAct or IterResearch "Heavy" | Single, 128K | CPT + SFT + on-policy RL | None | Answer-focused | 30B-A3B MoE [36] |
| ASearcher | Single ReAct agent | Single, very long | Async RL (AReaL) | None | Answer-focused | GAIA 58.7 Avg@4; >100 tool turns [38] |
| MiroThinker 1.7 | Single agent | 256K | SFT + RL ("interactive scaling") | None | Reports (online product) | BrowseComp 74.0% (235B) [39] |

## Key takeaways for system designers

1. **Token budget is the main lever, so make it explicit.** On BrowseComp, token usage explained 80% of variance in Anthropic's system, and LangChain's leaderboard runs show the same pattern [1][26]. Expose effort tiers (Gemini's standard and Max agents, OpenAI's o3 and o4-mini snapshots) and encode effort-scaling rules in prompts. Unbounded agents over-spend: early Claude Research spawned 50 subagents for simple queries [1][8][16].
2. **Separate scoping from researching.** OpenAI puts a clarifier model in front of the research model, Gemini offers a plan-approval protocol, and LangChain ODR has a clarify node and a research brief. A minute spent scoping saves many minutes of hedged, verbose research [8][16][26].
3. **Use isolated worker contexts and return compressed results.** Return 1,000 to 2,000 token summaries or file references, not raw transcripts. Persist the plan outside the context window. This is the main reason multi-agent setups beat single agents on breadth-first tasks [1][3][26].
4. **Choose topology by query shape.** Breadth-first questions parallelize well. Depth-first questions benefit from several perspectives. Straightforward questions need only one worker. Tightly coupled tasks, such as most coding, gain little from multi-agent research [1][4].
5. **Write delegation contracts.** Each worker needs an objective, an output format, sources to use and avoid, tools and boundaries. Vague delegation causes duplicated and missing coverage [1][4].
6. **Make citation a separate concern.** A dedicated citation pass over a fixed draft (Anthropic), span-level annotations (OpenAI) and post-hoc support verification (DeepResearch Bench's FACT) all treat attribution as verifiable structure, not decoration [1][8][40].
7. **Rank sources for authority, not just relevance.** Search ranking rewards SEO. Human evaluation is still needed to catch content-farm bias, and the lesson should be fed back into source-quality heuristics [1][7].
8. **Engineer for long-running, stateful jobs.** Use background execution with resumable streams, checkpoint-and-resume instead of restart, and rainbow deployments for agents mid-run. Trace decision patterns rather than content [1][8][16][18].
9. **Expect structure to be absorbed by models.** Workflows that helped weaker models, such as fixed section-by-section pipelines, become bottlenecks. The field has moved from STORM-style pipelines toward model-driven loops and RL-trained agents [27][28][33][36].
10. **If you train, invest in data and asynchronous infrastructure.** Synthetic, obfuscated, difficulty-filtered questions plus fully asynchronous RL are what made long-horizon search trainable in open models [36][37][38].
11. **Evaluate both answers and reports.** Use BrowseComp- and GAIA-style benchmarks for search competence, rubric or reference-based judges (RACE, DRB II) for report quality, and keep the judge itself versioned and calibrated against humans [11][40][41].

## References

1. "How we built our multi-agent research system." Anthropic Engineering, 2025. https://www.anthropic.com/engineering/built-multi-agent-research-system
2. "Claude takes research to new places" (Research and Google Workspace launch). Anthropic, April 15, 2025. https://claude.com/blog/research
3. "Effective context engineering for AI agents." Anthropic Engineering, September 29, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
4. Research lead agent prompt (research_lead_agent.md). Anthropic Cookbook, 2025. https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/prompts/research_lead_agent.md
5. Research subagent prompt (research_subagent.md). Anthropic Cookbook, 2025. https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/prompts/research_subagent.md
6. Citations agent prompt (citations_agent.md). Anthropic Cookbook, 2025. https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/prompts/citations_agent.md
7. "Introducing deep research." OpenAI, February 2, 2025 (launch post; could not be re-fetched for this chapter). https://openai.com/index/introducing-deep-research/
8. "Introduction to the Deep Research API." OpenAI Cookbook, 2025. https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api
9. "Deep Research Agents Cookbook." OpenAI Cookbook, 2025. https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api_agents
10. "Deep research system card." OpenAI, 2025. https://cdn.openai.com/deep-research-system-card.pdf
11. simple-evals (BrowseComp reference implementation) and "BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents." OpenAI, 2025. https://github.com/openai/simple-evals and https://openai.com/index/browsecomp
12. Research bot example. OpenAI Agents SDK (openai-agents-python), 2025. https://github.com/openai/openai-agents-python/tree/main/examples/research_bot
13. "Gemini Deep Research" overview. Google, December 2024. https://gemini.google/overview/deep-research/
14. "Try Deep Research and our new experimental model in Gemini." Google Blog, December 11, 2024. https://blog.google/products/gemini/google-gemini-deep-research/
15. "Deep Research now available on Gemini 2.5 Pro Experimental." Google Blog, 2025. https://blog.google/products/gemini/deep-research-gemini-2-5-pro-experimental/
16. "Gemini API: Deep Research Agent" quickstart notebook. Google Gemini Cookbook, 2026. https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_Deep_Research.ipynb
17. Deep Research documentation. Gemini API docs (Google), 2026. https://ai.google.dev/gemini-api/docs/deep-research
18. gemini-fullstack-langgraph-quickstart. Google Gemini, 2025. https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
19. "Introducing Perplexity Deep Research." Perplexity, February 14, 2025 (launch post; could not be re-fetched for this chapter). https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research
20. Sonar model cards (research models). Perplexity API docs, 2025. https://docs.perplexity.ai/guides/model-cards#research-models
21. Shao, Y. et al. "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models" (STORM). NAACL 2024. https://arxiv.org/abs/2402.14207
22. Jiang, Y., Shao, Y. et al. "Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations" (Co-STORM). EMNLP 2024. https://www.arxiv.org/abs/2408.15232
23. STORM / Co-STORM repository (knowledge-storm). Stanford OVAL, 2024-2025. https://github.com/stanford-oval/storm
24. GPT Researcher repository. Assaf Elovic et al., 2023-2026. https://github.com/assafelovic/gpt-researcher
25. GPT Researcher multi-agent (LangGraph) README. Assaf Elovic et al., 2024-2026. https://github.com/assafelovic/gpt-researcher/tree/main/multi_agents
26. Open Deep Research repository. LangChain, 2025. https://github.com/langchain-ai/open_deep_research
27. "Open Deep Research." LangChain Blog, July 2025. https://blog.langchain.com/open-deep-research/
28. Martin, L. Post on the evolution of Open Deep Research and the bitter lesson. July 30, 2025. https://rlancemartin.github.io/2025/07/30/bitter_lesson/
29. "Open-source DeepResearch: Freeing our search agents" and smolagents open_deep_research example. Hugging Face, February 2025. https://huggingface.co/blog/open-deep-research and https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research
30. deep-research (minimal open implementation). David Zhang (dzhng) / Duet, 2025. https://github.com/dzhng/deep-research
31. node-DeepResearch and "A Practical Guide to Implementing DeepSearch/DeepResearch." Jina AI, 2025. https://github.com/jina-ai/node-DeepResearch and https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch
32. Li, X. et al. "Search-o1: Agentic Search-Enhanced Large Reasoning Models." EMNLP 2025. https://arxiv.org/abs/2501.05366
33. Jin, B. et al. "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning," 2025, and repository. https://arxiv.org/pdf/2503.09516 and https://github.com/PeterGriffinJin/Search-R1
34. Zheng, Y. et al. (GAIR-NLP). "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments," 2025. https://arxiv.org/abs/2504.03160
35. Li, X., Jin, J., Dong, G. et al. "WebThinker: Empowering Large Reasoning Models with Deep Research Capability." NeurIPS 2025. https://arxiv.org/abs/2504.21776
36. Tongyi DeepResearch Team. "Tongyi DeepResearch Technical Report," 2025, and repository (including the WebAgent paper family: ReSum, WebResearcher, WebWeaver, AgentFold, ParallelMuse, BrowseConf). Alibaba Tongyi Lab. https://arxiv.org/pdf/2510.24701 and https://github.com/Alibaba-NLP/DeepResearch
37. WebAgent (WebWalker, WebDancer, WebSailor, WebShaper, WebWatcher) README. Alibaba Tongyi Lab, 2025. https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent
38. ASearcher: large-scale asynchronous RL for search agents. inclusionAI (Ant Group), 2025. https://github.com/inclusionAI/ASearcher
39. MiroThinker repository and v1.0 technical report. MiroMind AI, 2025-2026. https://github.com/MiroMindAI/MiroThinker and https://arxiv.org/abs/2511.11793
40. Du, M. et al. "DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents," 2025, and repository. https://arxiv.org/abs/2506.11763 and https://github.com/Ayanami0730/deep_research_bench
41. "DeepResearch Bench II." USTC Agent Research Lab, 2026. https://arxiv.org/abs/2601.08536
42. Huang, Y. et al. "Deep Research Agents: A Systematic Examination And Roadmap," 2025, and companion awesome-list. https://arxiv.org/abs/2506.18096 and https://github.com/ai-agents-2030/awesome-deep-research-agent
43. FS-Researcher: file-system-based dual-agent deep research. USTC Agent Research Lab, 2026. https://arxiv.org/abs/2602.01566
44. Kimi-Researcher. Moonshot AI, June 2025. https://moonshotai.github.io/Kimi-Researcher/
45. "Deep Researcher with Test-Time Diffusion," 2025. https://arxiv.org/abs/2507.16075
