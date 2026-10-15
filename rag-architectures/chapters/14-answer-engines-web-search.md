# 14. Answer Engines and Web-Grounded LLMs: Perplexity, Google AI Overviews and AI Mode, Bing/Copilot, ChatGPT Search, Search APIs for Agents, Deep Research and Citation Systems

## Overview

An *answer engine* is RAG with the open web as its corpus. The user asks a question. The
system decides whether to search, writes one or more queries, retrieves pages from a web
index, extracts the useful spans, and writes an answer that cites them. The same loop
underlies Perplexity, Google's AI Overviews and AI Mode, Microsoft Copilot, ChatGPT search,
Claude's web search, and the "deep research" agents that run that loop for minutes instead
of seconds.

The pipeline looks like the enterprise RAG of earlier chapters, but five things make web
grounding a different engineering problem.

1. **Someone else usually owns the index.** Only a few companies crawl and rank the web
   themselves: Google, Microsoft (Bing), and a handful of independent indexes. Everyone
   else rents retrieval through an API. The API's ranking, snippet length and freshness
   then set the quality ceiling, and its licence terms decide what you may cache, show
   or train on [10][11].
2. **The corpus is adversarial.** Web pages are written to rank. Content farms, SEO pages,
   satire, forum jokes and pages optimized for generative engines all end up in the
   retrieved set [6][17][42].
3. **The corpus changes under you.** A correct answer last week can be wrong today. Fresh
   facts are the main reason to search at all [3][36].
4. **Attribution is part of the product, and sometimes a legal duty.** Citations let users
   check the answer. They also send traffic back to publishers. Some providers make
   showing them a licence condition [9][10].
5. **The latency budget varies 1,000x.** An AI Overview has to appear on a results page.
   A deep research report may take tens of minutes [29][30][22].

This chapter is organized around those constraints. Section 1 traces the research lineage
(WebGPT, GopherCite, FreshLLMs, and the first audits of generative search engines). Section
2 gives a reference architecture. Sections 3 to 5 cover the deployed systems: search-engine
owners (Bing/Copilot, Google), answer-engine natives (Perplexity, You.com, open-source
clones) and chat assistants with a search tool (ChatGPT, Claude, Gemini API). Section 6
compares the search APIs built for agents: Perplexity Search, Exa, Tavily, Brave, Grounding
with Bing and You.com. Section 7 covers deep research agents. Section 3.3 covers publisher economics and
crawler controls, and Section 5.4 agentic browsers. Section 8 is about citation systems:
how they are built, how they are measured, how they fail, and the content-side signals
(llms.txt) that sites offer. Section 9 covers evaluation. Chapter 08 covers agentic and RL-trained search loops in general. Chapter 11
covers citation metrics such as ALCE in more depth.

**A note on sources.** Most claims in this chapter were checked against primary sources
retrieved while writing it: vendor documentation, SDK source code, official GitHub
repositories, engineering blogs and technical reports. Network restrictions blocked some
vendor blogs, for example Google's Search blog, the OpenAI and Perplexity websites, and
the Bing blog. Reference entries marked with a dagger (†) could not be re-fetched for
this edition. The claims attributed to them describe those published documents as they
were originally announced. They are kept few, and they are worth re-checking before
anyone quotes them.

```
                    THE ANSWER-ENGINE DESIGN SPACE
  latency      ~1 s            ~5-30 s                 minutes to tens of minutes
              |----------------|-----------------------|-------------------------|
  pattern     single-shot      agentic search          deep research
              grounding        (search inside the      (plan, fan out, read
              (1 query,        reasoning loop,         hundreds of pages,
              top results)     a few to ~10 calls)     write a cited report)
  examples    AI Overviews,    ChatGPT search,         OpenAI / Gemini / Claude
              Bing grounding,  Claude web search,      Research, Perplexity
              Sonar-style      AI Mode, Perplexity     deep research, Exa /
              answers          Pro search              Parallel / Tavily research
```

## 1. Lineage: from WebGPT to verifiability audits

**WebGPT (OpenAI, 2021).** WebGPT fine-tuned GPT-3 to use a text-based browser. Its
actions were commands such as search, click, scroll and *quote*. Quoting saved a passage
as a reference to cite in the final answer. Training used behaviour cloning on human
demonstrations, then a reward model trained on human comparisons, combined with rejection
sampling (best-of-n). On ELI5 questions, WebGPT answers were preferred to the human
demonstrators' answers 56% of the time [1†]. WebGPT set three conventions that answer
engines still follow: the model drives the search, evidence is collected as quotes, and
the answer must cite them.

**GopherCite (DeepMind, 2022).** GopherCite trained a model to support each answer with a
verbatim quote from a retrieved page ("self-supported question answering"). Training used
reinforcement learning from human preferences, and the model was allowed to abstain when
unsure [2†]. Two of its ideas recur in production citation systems in Section 8: verbatim
quotes as a verifiable unit of evidence, and declining to answer when evidence is weak.

**FreshLLMs / FreshPrompt (Google and collaborators, 2023).** FreshLLMs introduced FreshQA,
a question set built around fast-changing and false-premise questions. It also introduced
FreshPrompt, which puts search-engine results, including snippets and answer boxes, into
the prompt in date order [3]. The maintainers still update FreshQA weekly or on request,
with dated snapshots through April 2026. They report that Google's Gemini, Perplexity's
online LLMs and You.com used the data or method. The follow-up benchmark, SealQA, focuses
more on reasoning over noisy search results [3].

**Auditing generative search engines (Stanford, 2023).** Liu, Zhang and Liang audited
four commercial generative search engines of early 2023. On average, only about half of
the generated sentences were fully supported by their citations (51.5%), and about a
quarter of the citations did not support the sentence they were attached to (74.5%
citation precision) [4†]. The released dataset has 8,834 / 1,106 / 1,097
train/dev/test verifiability judgments. Each judgment labels a (statement, cited page)
pair as complete, partial or no support. Annotators also marked the minimal supporting
evidence [4]. The paper's text-extraction pipeline is a reusable recipe: save the page
with SingleFile, apply Chrome's DOM Distiller ("reader mode"), then run Trafilatura [4].
**ALCE** (Princeton, 2023) turned the same concern into a benchmark with citation recall
and precision. The best models of the time still lacked complete citation support on
about half of ELI5 answers [5†]. Chapter 11 covers ALCE in depth.

**Generative Engine Optimization (GEO, 2023).** Aggarwal et al. reversed the question and
asked how content creators can increase their visibility *inside* generative answers.
They found that page-side rewrites can raise a source's visibility in answers by up to
40% [6]. Any answer engine therefore has to treat "content written for LLMs" as a new
kind of SEO pressure.

## 2. Reference architecture

The deployed systems below differ in detail, but they share one pipeline. Every stage is
documented by at least one vendor.

```
 user query (+ conversation, location, locale)
      |
      v
 [1] TRIGGER / CLASSIFY ---- no search needed --> answer from parametric knowledge
      |  (is the answer current? niche? risky?)
      v
 [2] QUERY GENERATION: rewrite to standalone; generate 1..N queries ("fan-out")
      |
      v
 [3] RETRIEVAL: own index | search API | SERP provider     (filters: domain, recency,
      |                                                     locale, safe-search)
      v
 [4] CONTENT EXTRACTION: snippet | query-biased highlights | full page | PDF | live crawl
      |
      v
 [5] CONTEXT ASSEMBLY under a token budget (dedupe, per-URL caps, threshold filtering,
      |   code-based filtering)
      |
      +--> [6] AGENTIC LOOP? read results -> reflect on gaps -> new queries -> [3]
      v
 [7] GENERATION with inline citations (markers, spans, or post-hoc citation pass)
      |
      v
 [8] CITATION RENDERING + compliance (show sources, show queries, provider widgets)
```

**[1] Triggering.** Searching costs latency and money, so systems decide first whether to
search. Anthropic documents its policy. Claude searches for recent events, current
prices, facts about organizations, people or products that may have changed, and explicit
requests to look something up. It answers directly for stable knowledge, creative
writing, analysis of content already in the conversation, and greetings. Developers can
steer this through the system prompt and cap it with `max_uses` [36]. The Gemini API has a
dynamic-retrieval configuration with a mode and a numeric `dynamic_threshold`, which lets
a predictor decide when to ground [20]. The open-source Vane (formerly Perplexica) starts
every turn with a classification step. That step decides whether to research, whether to
show widgets, and how to rewrite the question into a standalone form [52].

**[2] Query generation.** The search query is rarely the user's text. Microsoft documents
this for Microsoft 365 Copilot. Copilot parses the prompt, picks out the terms where web
data would help, and generates "a few words" as a Bing query. It sends no user or tenant
identifiers and does not send the whole of any attached document [8]. Google's AI Mode
issues many related queries at once across subtopics, a technique it calls *query
fan-out* (Section 3.2) [18†].

**[3] Retrieval.** Section 6 compares the retrieval APIs. The basic choice is between your
own index, a third-party index with its own ranking (Bing, Brave, Exa, Perplexity), and
scraped results pages. Perplexity's evaluation repository calls the Tavily integration a
"representative Google SERP scraping-based API" and notes that Google and Bing do not
offer first-party web search APIs [26].

**[4]-[5] Extraction and context assembly.** This is where the "10 blue links" of classic
search turns into evidence an LLM can use. Search APIs increasingly do this step on the
server and accept a token budget. Examples are Brave's `maximum_number_of_tokens`
(1,024-32,768), Perplexity's `max_tokens_per_page`, Exa's relevance-sized highlights and
Tavily's `chunks_per_source` (Section 6). Anthropic's tool versions from February 2026 onward go further.
Claude writes and runs code that filters search or fetch results *before* they enter the
context window ("dynamic filtering") [36][37].

**[6] The loop.** Microsoft's Azure documentation of the OpenAI `web_search` tool
describes three modes. They map onto the latency bands in the overview diagram [12]:

| Mode | Behaviour | Typical use |
|---|---|---|
| Web search without reasoning | Query is forwarded to the search tool; top-ranked sources ground the answer; no multi-step planning | fast lookups, timely facts |
| Agentic search with reasoning models | Model searches inside its chain of thought, analyses results, decides whether to keep searching | complex questions; slower |
| Deep research | Multi-step reasoning, opens and reads many pages, synthesizes a citation-rich report; runs for minutes | background workloads |

**[7]-[8] Citation.** Section 8 covers this stage.

## 3. Search-engine-owned answer layers

### 3.1 Bing and Microsoft Copilot

**Prometheus (2023).** The new Bing launched in February 2023. Microsoft described a
proprietary system called *Prometheus* that combined Bing's index, ranking and answers
with an OpenAI GPT model. A component called the *Bing Orchestrator* generated internal
search queries iteratively, so that answers were grounded in fresh results and cited
them. Microsoft also reported that applying the same model to Bing's core ranking gave
the largest relevance improvement in two decades [7†]. In architectural terms, the search
engine became a tool that a planner LLM calls repeatedly, and the LLM became a component
of the search engine's ranking stack.

**LLM relevance labels.** Bing's evaluation side changed as well. Microsoft researchers
reported that an LLM, chosen and prompted to agree with careful feedback from real
searchers, can produce relevance labels at scale "as accurate as human labellers". They
also found that its accuracy changes unpredictably with simple paraphrases of the prompt,
so high-quality gold labels remain necessary [14]. Chapter 11 discusses UMBRELA, an
open-source reproduction of Bing's relevance assessor.

**Microsoft 365 Copilot web grounding: queries as citations.** Microsoft's privacy
documentation is the most detailed public description of how an enterprise assistant
mixes private and web data [8]:

- Copilot derives a short query from the prompt. The query can be informed by an open
  document, such as a Word file the user is editing, but the document itself is never
  sent. In Microsoft's example, the prompt "summarize [internal strategy doc] and tell me
  if Fabrikam announced a similar approach" becomes the query *"Fabrikam clean energy
  policy announcements"*.
- Some prompts generate no web query at all. "Find the document [coworker] wrote last
  week" is answered entirely from Microsoft 365 data.
- The exact queries sent to Bing are shown to the user as *web search query citations*
  next to the site citations. They stay visible in the thread for 24 hours, and admins
  can search, audit and run eDiscovery on them.
- Queries are sent without user or tenant identifiers. Microsoft commits that they are
  not used to improve Bing or to train foundation models.

Showing the generated query is a cheap transparency measure that other systems rarely
offer. It lets a user see at once why the answer drifted.

**Grounding with Bing for developers.** Microsoft licenses Bing to developers only as a
grounding tool inside Azure AI Foundry. Its documentation describes four steps: query
formulation, search execution, information synthesis and source attribution [9]. The
design rules are unusually strict:

- Only the generated query, tool parameters and the resource key go to Bing [9][13].
- *"Developers and end users don't have access to raw content returned from Grounding
  with Bing Search."* The model sees the results. The application gets only the model's
  answer, with citations and a link to the Bing query [9].
- Both kinds of reference must be "retained and displayed in the exact form provided by
  Microsoft". The documentation advises using the citation annotations the tool returns
  and *not* asking the model to generate citation links [9][10].
- The terms forbid training any AI model on Bing output. They forbid caching or storing
  output except as an integrated part of the customer's own work product, and they
  forbid building a database or index from it. They also restrict use to "internet
  search experiences" that visibly indicate web results [10].
- Price at the time of writing: $14 per 1,000 transactions, with a limit of 150
  transactions per second and 1 million transactions per day, for both standard and
  Custom Search [11]. Custom Search restricts grounding to configured domains and can
  block others [9].

OpenAI's `web_search` tool on Azure runs on the same Grounding with Bing service, and its
charges follow the number of Bing requests [12].

### 3.2 Google: AI Overviews, AI Mode and Deep Search

**Models trained to call Search.** The Gemini 2.5 technical report states the design
intent directly. Gemini 2.0 was "our first model family trained to natively call tools
like Google Search", so the model could write precise queries and combine fresh
information with sources. Gemini 2.5 interleaves search with its internal reasoning to
answer multi-hop queries: it issues follow-up queries both to widen the information it
has and to check the answer's factual accuracy [15]. The report links the factuality work
to the deployment of AI Overviews and AI Mode. It states that Gemini models power AI
Overviews for over 1.5 billion monthly active users. It reports SimpleQA (parametric
factuality, no search) of 54.0% for Gemini 2.5 Pro, against 24.9% for Gemini 1.5 Pro,
and FACTS Grounding (faithfulness to supplied documents) of 87.8% [15].

**AI Overviews (2024) and the "glue on pizza" lesson.** AI Overviews launched broadly in
the US in May 2024. They used a Gemini model customized for Search and placed a
generated summary with links above the results [16†]. Within days, screenshots spread of
answers that recommended adding glue to pizza or eating rocks. Google's post-mortem named
three root causes [17†]:

- **Data voids.** Nonsensical questions had no good answer on the web, so the system
  grounded on the little content that existed, which was often satire.
- **Satirical and humorous content** taken literally.
- **User-generated forum content** offering misleading advice.

Google's fixes acted on *triggering* and *source selection* rather than generation: better
detection of nonsensical queries, limits on satire and user-generated content, and no
overviews where they did not help. It also reported that content-policy violations
occurred on fewer than one in every 7 million unique queries [17†]. The general lesson
for designers is that grounding passes source quality straight through to the answer. A
model that faithfully summarizes a joke is grounded and wrong.

**AI Mode and query fan-out (2025).** AI Mode was introduced in March 2025, first on a
custom Gemini 2.0. It uses query fan-out: the system issues multiple related searches at
once, across subtopics and several data sources, and combines the results into one
answer [18†]. In May 2025 Google added *Deep Search*, an AI Mode option that can issue
hundreds of searches and produce a fully cited report in minutes [19†].

```
   "best e-bike for a steep 10 km commute in the rain, under $2k"
                                |
                     Gemini (custom for Search)
                     decomposes the information need
        +-----------+-----------+-----------+------------+-------------+
        v           v           v           v            v             v
   "e-bike hill  "waterproof "e-bike     "e-bike      shopping      reviews /
    torque"       rating      range 10km  under 2000"  graph /       forums
                  IPX"        hills"                   prices        (filtered)
        +-----------+-----------+-----+-----+------------+-------------+
                                      v
                 merge, dedupe, rank evidence; reason over gaps
                                      v
               answer with inline links  (Deep Search: same pattern,
                                          hundreds of queries, a report)
```

**Gemini API grounding: the developer view of the same machinery.** The Gemini API
exposes Google Search, Google Maps, URL context and YouTube grounding as tools [21]. The
SDK's `GroundingMetadata` type shows how Google represents attribution [20]:

- `web_search_queries`: the queries the model actually issued.
- `grounding_chunks`: the retrieved sources, such as web URIs and titles.
- `grounding_supports`: for each `segment` of the answer text, a list of
  `grounding_chunk_indices` with matching `confidence_scores` between 0 and 1. This is a
  claim-to-evidence alignment with a confidence per link.
- `search_entry_point.rendered_content`: an HTML snippet of Google Search suggestions
  that applications embed next to the answer. Google's own cookbook renders it
  [20][21].

The cookbook's Maps-grounding section states that "all grounded outputs require sources
to be displayed after the response text" [21]. Like Bing, Google ties attribution to the
licence.

**Lessons from the search-engine owners.**
(1) Owning the index lets you treat retrieval as a fan-out primitive: dozens to hundreds
of internal queries per answer, which renting would make uneconomic.
(2) Failures cluster where retrieval is thin or adversarial, not where the model is weak.
(3) Both owners make attribution a licence condition for third-party developers.

### 3.3 Publisher economics: clicks, crawlers and pay-per-crawl

Answer layers change the bargain that made the web crawlable: publishers allowed crawling
in exchange for referral traffic. Three 2025 developments show the pressure.

- **Fewer clicks.** A Pew Research Center study of the browsing data of about 900 US
  adults (March 2025) found that users clicked a traditional search result on 8% of
  Google visits that showed an AI summary, against 15% of visits without one, and
  clicked a link *inside* the summary on only about 1% of visits [61†].
- **Crawler controls move to the network edge.** In July 2025 Cloudflare began blocking
  known AI crawlers by default for new domains and launched a private beta of *pay per
  crawl*, which answers crawler requests with HTTP 402 and a price, so that a site can
  allow, charge or block each AI crawler [62†]. In August 2025 Cloudflare alleged that
  Perplexity used undeclared crawlers with generic user agents to fetch sites that had
  blocked its declared bots; Perplexity disputed the characterization, arguing that
  user-initiated fetches are not crawling [63†].
- **Design consequence.** The line between a *crawler* (building an index) and a
  *user-triggered fetch* (reading a page for one answer) is now contested commercially
  and technically. Systems should identify their fetchers honestly, honour robots and
  edge-level signals, and log which pages were fetched live for which request. Licence
  terms such as Bing's (Section 3.1) and edge-level payment schemes both push towards
  explicit, auditable retrieval.

## 4. Answer-engine natives

### 4.1 Perplexity

Perplexity built its product around the answer engine: the answer comes first and the
sources sit next to it. Its public developer surface shows how the architecture has
changed.

**From Sonar to an agent API.** Perplexity's cookbook says the Agent API is "the new
primary API for building hosted agents with native web access, citations, code execution,
subagents, and durable, long-running work". The older Sonar chat-completions API is
deprecated [24]. In the official Python SDK, a request to the Agent API names either a
`preset` (for example `sonar-pro` or `sonar-reasoning`: a model with its system prompt
and search parameters already set) or a model in `provider/model` form. A `models`
fallback chain of up to five models is tried in order. `max_steps` caps the research loop
at 100 steps. Built-in tools include `web_search`, `fetch_url`, `finance_search`,
`people_search`, and a sandbox with file read, write, grep and glob tools [25]. Perplexity
has therefore moved from "one search, one answer" towards a hosted, model-agnostic agent
runtime with its own search underneath.

**The Search API.** Perplexity sells its index directly. Requests accept one query or a
list of queries, `max_results`, total and per-page token budgets (`max_tokens`,
`max_tokens_per_page`), and a `search_context_size` of low, medium or high (default high;
larger sizes return more page content per result at higher cost). They also accept
recency filters (`hour` to `year`), date filters on publication and last update, domain
and language filters, a `search_mode` of web, academic or SEC filings, and a
`search_type` of web or people [25]. Each result carries `title`, `url`, `snippet`,
`date` and `last_updated` [25]. The last-updated field matters: it lets the downstream
model reason about staleness.

**Index architecture.** Perplexity Research described the system behind this API in
"Architecting and Evaluating an AI-First Search API" (September 2025). The article is
linked from the evaluation repository [26][27†]. As announced, the design centres on
retrieval units smaller than a document, so that the API returns the most relevant spans
of a page rather than whole pages. It combines lexical and semantic retrieval and ranks
in several stages under tight latency targets [27†]. Treat these descriptions as
approximate. Precise scale and latency figures should come from the article itself.

**search_evals: an open harness, run by a vendor.** Perplexity released `search_evals`,
an MIT-licensed framework that plugs different search APIs into identical agent harnesses
[26]. Its history is a small case study in how such comparisons change over time:

| Version (date) | What is compared | Example result |
|---|---|---|
| v1 (Sep 2025) | 4 search APIs, one agent (single-step for SimpleQA/FRAMES, deep research for BrowseComp/HLE) | SimpleQA: Perplexity 0.930, SERP-via-Tavily 0.890, Brave 0.822, Exa 0.781 |
| Feb 2026 | 5 engine configs x 3 models (Claude Opus 4.5, Claude Sonnet 4.5, GPT-5 medium) x 7 suites | BrowseComp with Opus 4.5: 0.116-0.207 across engines; with GPT-5 medium: 0.332-0.413 |
| v0.2 (Jun 2026) | whole hosted research agents (Perplexity Agent API, OpenAI Responses, Anthropic Managed Agents, Exa Agent API, Parallel Task API) | BrowseComp: Perplexity 0.805, OpenAI 0.720, Anthropic 0.598, Parallel 0.56, Exa 0.38 |

Three readings follow from the published configuration files.

- **The model can matter more than the index.** In February 2026 the BrowseComp spread
  *between models* on the same engine (for Brave, 0.207 with Opus 4.5 against 0.413 with
  GPT-5 medium) was larger than the spread between engines for one model [26].
- **Configuration is part of the result.** The harness gave Perplexity 3,000 tokens per
  query (3,000 per page). A `perplexity-long` variant used 10,000 tokens (4,000 per page)
  and scored best on most suites. Tavily ran at `search_depth="basic"`, chosen in the code
  comment because it "provides sub-1s latency". Exa ran in fast mode with highlights of a
  few sentences per URL. Brave combined descriptions with `extra_snippets` [26]. A search
  API benchmark partly measures how many evidence tokens each engine was allowed to
  return.
- **The unit of comparison moved up a level.** By mid-2026 the harness compared whole
  hosted research agents. The agents themselves were partly configurable: the Perplexity
  agent ran `openai/gpt-5.5`, the OpenAI agent ran `gpt-5.5` at high reasoning effort with
  up to 100 tool calls, and the Anthropic agent ran Claude Opus 4.7 on Managed Agents
  [26]. The question changed from "which index?" to "which search-plus-agent stack?".

All three tables come from a vendor-run harness. They are useful because the code,
prompts and graders are public, so others can rerun them. They are not neutral.

### 4.2 You.com

You.com was an early consumer answer engine. Today it presents itself to developers as a
hosted MCP server. The default tools are `you-search`, `you-contents` (page content),
`you-balance` and `you-discover`. `you-finance` and `you-research` are available on
request, and a keyless free profile exposes search and discovery [51]. The FreshLLMs
authors list You.com among the systems that used their method or data [3]. Few
architectural details are public.

### 4.3 Open-source answer engines

Open implementations show the minimal viable architecture without vendor abstractions.

**Vane (formerly Perplexica)** is a self-hostable answer engine [52]. Each turn works in
three phases:

1. A classification step decides whether to research, whether to show widgets (weather,
   stocks, calculations), and how to rewrite the question into a standalone form.
2. Research and widgets run in parallel. Research uses a SearxNG meta-search backend and,
   optionally, embedding search over uploaded files.
3. The chat model writes the answer.

Citations are produced by prompting: "We prompt the model to cite the references it
used", and the UI renders them next to the links. Widget output serves as context but
is explicitly *not* citable. Users choose a `speed`, `balanced` or `quality` mode [52].

**Google's Gemini LangGraph quickstart** is a reference deep-research loop. It generates
initial queries, runs web research for each through Gemini with Google Search, reflects
on whether there are knowledge gaps, generates follow-up queries, repeats up to a
configured maximum number of loops, and then writes an answer with citations [23].

**WikiChat (Stanford)** grounds on Wikipedia, not the open web. Its 7-stage pipeline,
where each stage is one or more LLM calls, includes extracting claims from a draft so
they can be checked against retrieved evidence. A later version merges the generate and
claim-extraction stages to cut cost [53]. It shows a *verify-then-answer* design: claims
are extracted and checked before the final answer is written, which the single-pass
designs above do not do.

**OpenDeepSearch (Sentient)** packages search as a tool for agents and offers two modes.
*Default mode* uses SERP results with minimal processing for single-hop questions. *Pro
mode* scrapes pages (via Crawl4AI), reranks them semantically, and post-processes the
results for multi-hop questions. The authors report parity with closed search
alternatives on SimpleQA and better results on FRAMES [54]. The same two tiers appear in
commercial APIs (Section 6).

## 5. Chat assistants with a search tool

### 5.1 ChatGPT search and OpenAI's web_search tool

**ChatGPT search (October 2024).** As announced, the search model was a fine-tuned GPT-4o,
post-trained with synthetic data generation techniques that included distilling outputs
from o1-preview. It drew on third-party search providers and on content from news
partners [28†]. This is another way to make a model good at search: rather than training
it end to end with a search tool, as Google describes for Gemini [15], distil
search-and-cite behaviour from a stronger reasoning model.

**The API tool.** The Responses API `web_search` tool, documented in detail through its
Azure deployment, shows the mechanics [12]:

- A `web_search_call` output item records actions of type `search`, `open_page` and
  `find_in_page`. This is a modern form of WebGPT's command set [1†][12]. The last two
  are available only with reasoning models.
- Citations are `url_citation` annotations on the output text. Each carries the URL,
  title and *character range* it supports, so the client can render inline links
  without parsing markup.
- `include: ["web_search_call.action.sources"]` returns every URL consulted, not only
  the ones cited. This separates "read" from "relied on".
- Domain allow-lists take up to 100 entries and include subdomains. The
  `external_web_access` parameter separates live fetching from cached content. Azure
  always treats it as false and notes that "the model can still access cached content"
  [12].

### 5.2 Claude: web search, web fetch and citations as server tools

Anthropic released web search in Claude.ai in March 2025, on the API in May 2025, and
added a web *fetch* tool in September 2025 [34][35]. Both are *server tools*. Anthropic's
API runs the search loop inside a single request, and the model may search several times,
"using earlier results to inform subsequent queries" [35][36]. Several design details
stand out [36][37]:

- **Effort is visible and capped.** Anthropic's documentation says simple factual queries
  typically need 1-3 searches, while comparative or multi-entity research can need 10 or
  more. `max_uses` is a hard cap. Exceeding it returns a `max_uses_exceeded` error inside
  a successful (HTTP 200) response, so the model can recover and still answer.
- **Encrypted provenance.** Each search result carries `encrypted_content`, and each
  citation carries an `encrypted_index`. Both must be passed back unchanged in later
  turns, and the API rejects modified values. The client cannot forge or edit the
  evidence behind a multi-turn conversation.
- **Cheap quotes.** A web citation includes up to 150 characters of `cited_text`.
  `cited_text`, `title` and `url` do not count as input or output tokens.
- **Dynamic filtering (2026).** From tool version `web_search_20260209`, search runs by
  default from inside code execution. Claude writes code that filters results before they
  enter the context window, which reduces token use on search-heavy requests. A later
  version, `web_search_20260318`, adds `response_inclusion: "excluded"`, which drops the
  raw result blocks consumed by a completed code-execution call from the API response
  [36]. Web fetch gained the same filtering (from `web_fetch_20260209`), cache bypass
  with `use_cache: false` (from `web_fetch_20260309`) and response inclusion (from
  `web_fetch_20260318`); every version accepts a `max_content_tokens` truncation limit
  [37].
- **Exfiltration guard.** Web fetch can only retrieve URLs that already appeared in the
  conversation: in user messages, in client tool results, or in earlier search or fetch
  results. It cannot fetch URLs that appear only in Claude's own output, or URLs that seem
  to contain credentials. This blocks a common prompt-injection pattern in which a
  malicious page instructs the model to encode private data into a URL and fetch it. (A
  credential-bearing URL is allowed only if that credential appears in the system prompt
  or a user message.) Web fetch does not render JavaScript [37].
- **Price.** $10 per 1,000 searches plus tokens. Failed searches are not billed. Web
  fetch has no charge beyond the tokens of the fetched content [35][36][37].

### 5.3 Gemini app and API

Section 3.2 covers the Gemini API's grounding tools. The architectural point is that one
trained behaviour, calling Search from inside reasoning, serves AI Overviews, AI Mode,
the Gemini app and the developer API [15][20][21].

### 5.4 Agentic browsers: the answer engine becomes the browser

In 2025 the answer engine moved into the browser itself. Perplexity launched **Comet** in
July 2025 [64†], and OpenAI launched **ChatGPT Atlas** in October 2025 with ChatGPT
built into the browser and an agent mode that can act on pages [65†]. Architecturally,
the retrieval set is no longer only search results: it includes the page the user is on,
open tabs and, with consent, logged-in sessions. That widens the attack surface described
in Section 5.2. Brave's security team showed in August 2025 that Comet could be steered by
instructions hidden in a web page that the user asked it to summarize (indirect prompt
injection), with the agent acting with the user's own authenticated privileges [66†].
The defences carry over from server-side tools and need to be stronger here: treat page
content strictly as data, require user confirmation for sensitive actions, separate
browsing in authenticated sessions from browsing of untrusted pages, and restrict which
URLs the agent may visit or submit data to [37][66†].

## 6. Search APIs built for agents

Agents now consume search more than people do in many products, and a market of search
APIs has formed to serve them. They have converged on returning **token-budgeted
evidence** rather than ranked links.

| API | Retrieval basis (public description) | Evidence unit returned | Depth / latency knobs | Freshness / filters | Grounding output |
|---|---|---|---|---|---|
| Perplexity Search [25][27†] | Own web index; hybrid retrieval, sub-document units | ranked snippets; total and per-page token budgets | `search_context_size` low/med/high | recency, publish and update dates, domain, language, academic/SEC modes | `date`, `last_updated` per result |
| Exa [45] | Own index; `neural` (embedding-based), keyword, `auto`, `fast`, `instant`, deep variants | text, query-biased *highlights* (relevance-sized), summaries | type: fast ... deep-reasoning | `livecrawl` always/fallback/never/auto/preferred; historical snapshots | deep search: per-field citations + confidence |
| Tavily [47] | Characterized by Perplexity as SERP-based [26] | snippet `content`, `chunks_per_source`, raw markdown/text | `search_depth` ultra-fast/fast/basic/advanced; `auto_parameters` | time range, dates, `max_age_hours`, domains restrict/prefer, topic news/finance | optional LLM `answer`; research reports with citation formats |
| Brave [49][50†] | Own index (per Brave) | web results + `extra_snippets`; LLM Context endpoint | token budget 1,024-32,768; per-URL caps; threshold modes | `freshness` pd/pw/pm/py or range; Goggles re-ranking | summarizer with inline references |
| Grounding with Bing [9][11] | Bing index | not exposed to the developer; model sees it | n/a | Custom Search domain scoping | citations + Bing query link, must be displayed |
| You.com [51] | not documented | search, contents | research tool on request | n/a | n/a |

### 6.1 Exa

The Exa SDK documents a retrieval stack built for machine consumers [45]:

- **Search types.** `auto` (the default) chooses the approach. `neural` uses
  embedding-based semantic search, `instant` is low-latency neural search, `fast`
  prioritizes speed, and `deep-lite`, `deep` and `deep-reasoning` are deep-search variants
  that accept `additional_queries`. In auto mode the response reports whether it resolved
  to `neural` or `keyword`, and cost is broken down the same way.
- **Highlights instead of pages.** `contents={"highlights": True}` returns query-biased
  extracts. Their length is chosen per document by relevance, and the SDK advises against
  fixed character limits. A 2026 research preview, *Dynamic Highlights*, assigns a
  **single shared context budget across all results**: the model sizes each excerpt, so
  a weak page gets nothing and a strong one gets more. This is context assembly (stage
  [5]) done by the search provider.
- **Freshness control.** The `livecrawl` options are `always`, `fallback`, `never`,
  `auto` and `preferred`. `snapshot_as_of` returns the newest stored version of a page at
  or before a given time, which supports reproducible or historical research.
- **Grounded structured output.** Deep search can return content that follows a JSON
  schema (at most 2 levels of nesting and 10 properties). Each output field carries its
  own grounding: a list of citations (URL and title) and a confidence of low, medium or
  high. This is per-field attribution for structured extraction, a citation form the
  chat-oriented APIs lack.
- **Higher-level products.** An Agent API (with an "ultra" effort level), Websets, and
  *Agent Monitors* (in beta as of August 2026). Monitors keep a table of entities by fields up to date: static fields
  are answered once, and dynamic fields are tracked from news on every refresh.
- **Agent-facing packaging.** The official MCP server exposes `web_search_exa`,
  `web_fetch_exa` (pages as clean markdown), `web_search_advanced_exa` (filters, domains,
  dates, highlights, summaries, subpage crawling) and `agent_run` [46]. The search, read
  and delegate split mirrors the search and fetch tools of Claude and OpenAI.

### 6.2 Tavily

Tavily's SDK shows a "search plus extraction" API designed to sit inside RAG and agent
frameworks [47][48]:

- `search_depth` takes `ultra-fast`, `fast`, `basic` or `advanced`. Perplexity's harness
  chose `basic` because, per its code comment, `advanced` does not give sub-second latency
  [26]. `topic` is general, news or finance. Further options are `time_range` and
  explicit dates, `max_age_hours`, `include_domains_mode` of `restrict` or `prefer` (a
  soft domain boost, not only a hard filter), `exact_match` for names and phrases,
  `include_raw_content` as markdown or text, `chunks_per_source`, and `auto_parameters`
  (let the service choose).
- `include_answer` (basic or advanced) returns a short LLM-generated answer. The older
  `get_search_context` helper, which packed result contents into a JSON string up to
  `max_tokens` (4,000 by default), is deprecated. The packing has moved to the server.
- Endpoints beyond search: `extract` (pages by URL), `crawl` and `map` (site traversal
  and structure), and `research`. Research produces reports with model `mini`, `pro` or
  `auto`, an optional JSON output schema, and citation formats `numbered`, `mla`, `apa`
  or `chicago`.
- A **Feedback API** accepts per-result scores from agents or end users "so Tavily can
  improve ranking and relevance over time". The search provider learns from the
  consuming agent's judgments.
- A keyless mode (search and extract only, rate-limited), and 1,000 free credits per
  month.

### 6.3 Brave Search API

Brave's official MCP server documents the API surface [49]. Endpoints cover web, news,
images, video, local POIs and descriptions, place search, a summarizer, and an
**LLM Context** endpoint (`/res/v1/llm/context`). The LLM Context endpoint returns
"pre-extracted web content optimized for AI agents, LLM grounding, and RAG pipelines",
with explicit budgets: total tokens 1,024-32,768, URLs 1-50, snippets 1-256, tokens per
URL 512-8,192, snippets per URL 1-100. It also has a `context_threshold_mode` of
disabled, strict, lenient or balanced, which decides how aggressively marginal content is
dropped. **Goggles** let the caller supply custom re-ranking definitions, a programmable
ranking layer exposed to the client. The summarizer works in two steps: a web search with
`summary: true` returns a key, and the summarizer turns that key into a summary,
optionally with inline references. Brave describes its API as served from its own
independent index [50†].

### 6.4 What the APIs teach

- **The context window is now the product boundary.** Every modern API takes a token
  budget and returns pre-cut evidence, so the old distinction between ranking and
  snippeting has disappeared. When you compare APIs, hold the evidence budget fixed
  (Section 4.1).
- **Freshness is a parameter, not a property.** Live crawl, cache bypass, `max_age_hours`
  and `last_updated` let the agent pay latency for freshness only when it has to
  [25][37][45][47].
- **Licences shape architecture.** Bing's terms (no raw content to the developer, no
  caching, no training, mandatory display) push designs towards a model-sees-it,
  user-sees-citations pattern [9][10]. APIs that return raw text allow local reranking,
  caching and offline evaluation.

## 7. Deep research agents

Deep research turns the agentic loop into a background job. A minimal composite of the
documented systems looks like this:

```
  user request
      |
      v
  [clarify / rewrite]  -- ChatGPT: intermediate model asks questions; API: skipped;
      |                   Gemini: optional collaborative planning; Azure: gpt-4o scoper
      v
  [plan]  lead agent / research model writes a plan (persisted to memory)
      |
      +------------+------------+------------+     parallel workers / search threads
      v            v            v            v
   search ->    search ->    search ->    MCP / file   each: search, fetch, read,
   read ->      read ->      read ->      search ->    reflect, refine query
   compress     compress     compress     compress     (own context window)
      +------------+------------+------------+
      |
      v
  [synthesize] lead agent decides: enough? -> else spawn more work
      |
      v
  [cite]  inline citations from the model, or a separate citation pass
      |
      v
  report (+ charts, structured output); background mode + webhooks / polling
```

### 7.1 OpenAI deep research

As announced in February 2025, deep research in ChatGPT was powered by a version of the
upcoming o3 model optimized for web browsing and data analysis. It was trained with
end-to-end reinforcement learning on hard browsing and reasoning tasks, took roughly 5 to
30 minutes per task, and scored 26.6% on Humanity's Last Exam. OpenAI listed limitations:
it could hallucinate, had trouble telling authoritative information from rumour, was
poorly calibrated about its own confidence, and made formatting errors in reports and
citations [29†]. Hugging Face, reproducing it, quoted a GAIA validation score of about
67% (67.36%), with 47.6% on the hardest level-3 questions [55].

The API, documented in OpenAI's cookbook [30][31], exposes the mechanics:

- Models `o3-deep-research-2025-06-26` and `o4-mini-deep-research-2025-06-26` (the
  smaller one for latency-sensitive use) run through the Responses API with
  `background=True`, because tasks take minutes.
- Data sources are `web_search_preview` (required unless an MCP source is configured),
  an optional `code_interpreter`, and MCP servers for private data. The output lists every
  intermediate step: reasoning summaries, `web_search_call`, code runs and MCP calls.
  The final text carries citation annotations with start and end indices, title and URL.
- **Clarification is a separate stage.** In ChatGPT, an intermediate model (for example
  gpt-4.1) asks follow-up questions before research begins. The API skips this step and
  "expects fully-formed prompts up front", so the cookbook adds a rewriter or clarifier.
  Without one, vague prompts produce broad research that is verbose, slow and costly
  [30].
- The agents cookbook builds a four-agent pipeline (triage, clarifier, instruction
  builder, research agent) with the Agents SDK. It warns: "Do not use Deep Research for
  trivial fact lookups" [31].

On Azure the same architecture is visible from the outside. The (now deprecated) Deep
Research tool used `gpt-4o` to clarify and scope the task, then `o3-deep-research`
grounded on Grounding with Bing. Deployments were limited to two regions, and the documentation
lists a retirement date for the model (19 November 2026 when last checked; re-verify). Microsoft now recommends general reasoning models with
`web_search` at high reasoning effort [12][13]. Deep research is being absorbed into
general-purpose reasoning models plus a search tool.

### 7.2 Gemini Deep Research

Google's technical report describes Gemini Deep Research as an agent on Gemini 2.5 Pro
that is optimized for task prioritization and for "identify[ing] when it reaches a
dead-end when browsing". Its HLE score rose from 7.95% at launch in December 2024 to
26.9%, and to 32.4% with more compute, by June 2025 [15]. The 2026 API [22]:

- Two agents: `deep-research-preview-04-2026` (lower latency, interactive) and
  `deep-research-max-preview-04-2026` (maximum search exhaustiveness, for automated
  reporting). They are available only through the asynchronous Interactions API, and a
  task "can consult over 100 sources".
- Default tools are Google Search, URL Context and Code Execution. MCP servers and File
  Search over uploaded corpora are optional, and either can replace the web entirely.
- **Collaborative planning.** The agent returns a plan instead of a report, the user
  iterates on it, and the user then sets `collaborative_planning=False` to run it. This
  is clarification as a reviewable artifact.
- Streaming of thought summaries, built-in charts (`visualization="auto"`), multimodal
  inputs, and follow-up questions answered by a standard Gemini model rather than the
  research agent.

### 7.3 Anthropic Research

Chapter 08 covers Anthropic's multi-agent Research system as an agent architecture. Here
the focus is on what it shows about web grounding [41][42][43]:

- **Orchestrator and workers.** A lead agent plans, saves the plan to memory (contexts
  over 200K tokens are truncated), spawns subagents that search in parallel with their
  own context windows, and finally hands everything to a separate **CitationAgent**.
  The CitationAgent maps claims in the report to locations in the source documents [42].
- **Effort scaling in the prompt.** Simple fact-finding uses 1 agent and 3-10 tool calls.
  Comparisons use 2-4 subagents with 10-15 calls each. Complex research uses more than 10
  subagents [42]. The published prompts cap the lead at 20 subagents. They give each
  subagent a budget: under 5 calls for simple tasks, about 10 for hard ones, an absolute
  limit of 20 tool calls and about 100 sources [43].
- **Economics.** A multi-agent configuration (Opus 4 lead, Sonnet 4 subagents)
  outperformed single-agent Opus 4 by 90.2% on an internal research evaluation. On
  BrowseComp, token usage alone explained 80% of performance variance, and token usage,
  number of tool calls and model choice together explained 95%. Agents use about
  4x the tokens of chat, and multi-agent systems about 15x. Parallel subagents (3-5) and
  parallel tool calls (3 or more) cut research time by up to 90% on complex queries [42].
- **Source quality is a prompt problem.** Human testers found that early agents preferred
  "SEO-optimized content farms over authoritative but less highly-ranked sources like
  academic PDFs or personal blogs". Source-quality heuristics in the prompts fixed it
  [42]. The subagent prompt tells the model to watch for speculation in future tense,
  news aggregators instead of original sources, passive voice with unnamed sources, and
  marketing language, and to flag conflicts to the lead instead of resolving them
  silently [43].
- **Evaluation.** A single LLM-judge call scores factual accuracy, *citation accuracy*,
  completeness, source quality and tool efficiency. Work started from about 20 real
  queries [42].
- **Context management.** For Claude Opus 4.5, Anthropic reported that combining context
  management, memory and subagent orchestration improved a deep-research evaluation by
  almost 15 points: from 70.48% to 85.30% on a fetch-enabled BrowseComp-Plus [44].

### 7.4 Research APIs from search vendors

Search providers have moved up into agents: Perplexity's Agent API, Exa's Agent API,
Tavily's `research` endpoint and Parallel's Task API [24][26][45][47]. Section 4.1 has the
June 2026 `search_evals` comparison of five of these hosted agents. Their main
architectural advantage over a general model calling a third-party search tool is
**co-location**: the loop runs next to the index, so per-query latency and per-query cost
are lower. That makes wide fan-out affordable.

### 7.5 Open replications

Within about a day of OpenAI's launch, Hugging Face's open Deep Research reached 55.15%
on the GAIA validation set, against about 46% for the previous open state of the art
(Magentic-One). Most of the gain came from letting the agent write its actions as code
instead of JSON: with JSON actions the same setup scored 33% [55]. Alibaba's Tongyi
DeepResearch is a 30.5B-parameter mixture-of-experts model with 3.3B parameters active
per token and a 128K context. It was trained specifically for long-horizon information
seeking and is evaluated on HLE, BrowseComp, BrowseComp-ZH, WebWalkerQA, xbench-DeepSearch,
FRAMES and SimpleQA [56]. Chapter 08 covers RL-trained search agents (Search-R1,
DeepResearcher and others) in detail.

## 8. Citation systems

A citation makes a checkable claim: *this span of the answer is supported by that span of
that source*. Systems differ in who creates the link, at what granularity, and whether
the system can guarantee that the link points to something real.

### 8.1 Seven implementation patterns

| Pattern | How the link is made | Granularity | Guarantees | Examples |
|---|---|---|---|---|
| Prompted markers | model writes `[n]`; UI maps n to the source list | source | none: n may be wrong or invented | Vane [52], most early answer engines |
| Span annotations | API returns char ranges + URL for answer text | answer span to URL | link targets a retrieved source | OpenAI `url_citation` [12][30] |
| Constrained source pointers | model emits indices into pre-chunked docs; API extracts quoted text | answer span to sentence, page or block | pointers are valid by construction | Anthropic Citations, web search citations [36][38][39] |
| Supports with confidence | each answer segment maps to chunk indices with scores | segment to chunk | scored, not guaranteed | Gemini `grounding_supports` [20] |
| Post-hoc citation pass | separate agent adds citations to a finished report without changing text | claim to source location | text identity checked; support judged by model | Anthropic CitationAgent [42][43] |
| Provider-rendered references | grounding service returns references the app must display verbatim | answer to URL + query | references come from provider, not model | Grounding with Bing [9][10] |
| Per-field grounding | each field of structured output carries citations + confidence | field to URLs | scored | Exa deep search output [45] |

**Constrained pointers in practice.** Anthropic's Citations feature chunks plain-text and
PDF documents into sentences, or uses caller-supplied blocks as-is ("custom content").
The model cites by pointing at those chunks, and the API turns the pointers into
`cited_text` with character indices (text), page ranges (PDF) or block indices (custom
content). Because the API parses the citations, they "are guaranteed to contain valid
pointers to the provided documents". The quoted text does not count toward output tokens
[39]. Anthropic reported recall accuracy up to 15% higher than custom implementations.
One customer, Endex, reported source hallucinations and formatting issues falling from
10% to 0%, with 20% more references per response [38]. *Search result* content blocks
apply the same mechanism to a developer's own retrieval results. Any RAG system can then
get web-search-style citations by returning `search_result` blocks (source, title, text
content) from its tools [40]. The design principle is to **make invalid citations
impossible to emit rather than detecting them later**.

**Post-hoc citation as a separate pass.** Anthropic's published `citations_agent` prompt
gives a second agent the finished report and the sources. The agent must output the same
text "100% identical" (whitespace included) with citations added. Text without the
citations is compared to the original, and the result is rejected if they differ [43].
The prompt also sets citation style: cite claims readers would want to verify rather
than common knowledge; cite whole semantic units, preferably at sentence end; do not
fragment sentences with several citations; do not repeat a source within a sentence.
Separating writing from attribution lets each stage be optimized and evaluated on its
own. The cost is an extra model pass over a long document.

**Showing queries as well as sources.** Microsoft 365 Copilot cites the *queries* it
sent to Bing, not only the pages [8]. Grounding with Bing requires the link to the Bing
query [9][10]. Gemini returns `web_search_queries` [20]. Seeing the query lets users find
the most common failure, which is searching for the wrong thing.

### 8.2 Measuring citation quality

- **Support and precision.** Liu et al. separated *citation recall* (are generated
  statements supported?) from *citation precision* (does each citation support its
  statement?) and found both well below what users assume [4†][4]. ALCE automated both
  with NLI models [5]. DeepResearch Bench's **FACT** framework does this for long reports.
  It extracts statement-URL pairs, removes duplicates, scrapes each URL, has an LLM judge
  support, and reports **citation accuracy** and **effective citations per task** over
  100 PhD-level tasks in 22 fields [59].
- **What users reward.** Search Arena collected about 24,000 conversations with
  search-augmented models (ICLR 2026). Users prefer longer responses, more citations, and
  specific sources: they favour coding sources and community or blog posts and disfavour
  Wikipedia. However, "models do not always cite retrieved sources accurately and humans
  do not always check citations" [57]. Preference optimization on such data can therefore
  reward citation *volume* over citation *validity*. Evaluation should include an
  automated support check that users cannot game.
- **External audits.** A March 2025 audit by the Tow Center at Columbia Journalism Review
  tested eight AI search tools on identifying the source of news excerpts. Collectively
  they gave incorrect answers to more than 60% of queries, often confidently, and
  sometimes cited syndicated copies instead of original publishers [60†].

### 8.3 Citation and grounding failure modes

| Failure | Mechanism | Mitigations seen in sources |
|---|---|---|
| Unsupported citation | model attaches a real source that does not say the claim | constrained pointers [39]; post-hoc pass + judge [42][43]; FACT-style audits [59] |
| Invented link | model generates a URL | do not let the model write links; use API annotations [9][12] |
| Faithful to a bad source | satire, forum jokes, data voids | trigger suppression, source-type limits [17†]; source-quality prompts [43] |
| Content-farm bias | SEO pages outrank primary sources | source heuristics in prompts [42]; Goggles / domain preferences [47][49] |
| Generative-engine optimization | pages rewritten to win LLM citations | treat as adversarial ranking; GEO shows up to +40% visibility [6] |
| Staleness | cached or outdated pages | `last_updated`, `page_age`, live crawl, cache bypass [25][36][37][45] |
| Prompt injection / exfiltration | fetched page instructs agent to leak data via URLs | fetch only URLs already in context; block credential-like URLs [37] |
| Evidence tampering across turns | client edits prior results | encrypted result and citation indices [36] |
| Over-citation | many weak citations look rigorous | citation style rules [43]; precision metrics [4][5] |

### 8.4 Content-side signals: llms.txt and markdown mirrors

Answer engines also depend on what the page side offers. The **/llms.txt** proposal
(Jeremy Howard, September 2024; v2 in 2026) asks sites to publish a markdown file at the
root or any path that gives brief background, guidance and links to detailed markdown
files. Pages are encouraged to provide a clean markdown version at the same URL with
`.md` appended, discoverable through `rel="alternate" type="text/markdown"` links [67].
The v2 text reports that thousands of sites publish one, that documentation platforms
generate it automatically, and that the AI labs publish llms.txt files for their own
developer docs [67]. For a retrieval system, such files are cheap, clean extraction
targets (stage [4]). They are also self-descriptions written by the page owner, so they
deserve the same scepticism as any content optimized for generative engines (GEO,
Section 1).

## 9. Evaluating web-grounded systems

Answer engines need evaluation along three axes: answer correctness, retrieval and
browsing difficulty, and attribution. The current benchmark set:

| Benchmark | What it stresses | Size / notes |
|---|---|---|
| SimpleQA [32] | short factual answers; saturates quickly once search is on (0.93-0.95 in [26]) | OpenAI reference implementation |
| FreshQA / SealQA [3] | fast-changing facts, false premises; noisy search results | FreshQA updated weekly |
| FRAMES [26] | multi-hop retrieval and reasoning | single-step agent in [26] |
| BrowseComp [32][33†] | hard-to-find facts needing persistent, creative browsing | 1,266 problems [26] |
| BrowseComp-Plus [58] | the same queries over a *fixed* ~100K-document human-verified corpus | separates retriever from agent; reproducible |
| HLE (text-only IR subset) [26] | frontier academic questions | 2,158 of 2,500 questions [26] |
| DeepSearchQA [26] | multi-step seeking, collation, exhaustive answers | 900 tasks |
| WideSearch [26] | breadth: collecting many independent facts into tables | 200 tasks; row-level F1 |
| DeepResearch Bench [59] | report quality (RACE) and citation trustworthiness (FACT) | 100 tasks, 22 fields, based on 96,147 real queries |
| Search Arena [57] | human pairwise preference on real prompts | ~24K conversations |
| Mind2Web 2 [68] | long-horizon, time-varying search tasks with citation-backed answers | 130 tasks; agent-as-a-judge with task-specific rubrics |

BrowseComp at launch showed the gap between browsing and plain search. As reported,
GPT-4o with browsing solved under 2% of problems, while OpenAI's deep research agent
solved about half [33†]. A year later the hosted research agents in Section 4.1 scored
0.56-0.81 [26].

**Agent-as-a-judge for live answers.** Mind2Web 2 (NeurIPS 2025 Datasets and
Benchmarks) addresses answers that change over time and cannot be graded against a fixed
string. Each task has a tree-structured rubric, and judge agents check each required
fact *and* verify that the cited URL supports it. Submitters are asked to cache the
cited pages so that evaluation sees what the agent saw [68]. This turns citation
verification (Section 8.2) into part of correctness grading.

**Methodological cautions.**
(1) The live web makes runs non-reproducible. BrowseComp-Plus fixes the corpus so that
retrievers can be compared under one agent [58].
(2) Grading is done by LLM judges whose prompts come from the original benchmarks [26].
DeepResearch Bench moved its judges to Gemini-2.5-Pro (RACE) and Gemini-2.5-Flash (FACT)
in July 2025, then, after Google announced the deprecation of Gemini-2.5-Pro, switched in
May 2026 to GPT-5.5 (RACE) and GPT-5.4-mini (FACT). The replacement candidates were
validated on 200 human-annotated reports against a human inter-annotator agreement
baseline of 68.78% [59]. Scores from different judge generations are not comparable.
(3) Vendor harnesses set evidence budgets, models and step caps (Section 4.1). Report
them with every number.
(4) Separate *failed-as-zero* from *failed-excluded* scores. `search_evals` publishes both
[26].

## 10. Master comparison

| System | Owner of index | Loop style | Query strategy | Citation mechanism | Notable public numbers |
|---|---|---|---|---|---|
| Bing Copilot / Prometheus | Microsoft | orchestrated, iterative internal queries [7†] | generated short queries; shown to users in M365 [8] | provider references, must display [9][10] | grounding $14 / 1K transactions [11] |
| Google AI Overviews | Google | single-shot, custom Gemini [16†] | n/a public | inline links | >1.5B MAU [15] |
| Google AI Mode / Deep Search | Google | query fan-out; hundreds of searches in Deep Search [18†][19†] | parallel subtopic queries | inline links | n/a |
| Gemini API grounding | Google | model-driven tool call [15][21] | `web_search_queries` returned [20] | segment-to-chunk supports + confidence [20] | SimpleQA 54.0% (2.5 Pro, no tools) [15] |
| Perplexity (consumer + Agent API) | Perplexity | single-shot to 100-step agent [25] | presets; multi-query search [25] | numbered sources; API citations | BrowseComp 0.805 (agent, [26]) |
| ChatGPT search / `web_search` | third-party providers / Bing on Azure [12][28†] | 3 modes [12] | search, open_page, find_in_page [12] | `url_citation` char spans [12] | n/a |
| Claude web search / fetch | n/a in docs | server-side loop, `max_uses`, dynamic filtering [36] | multiple progressive searches [35] | constrained pointers, `cited_text` [36][39] | $10 / 1K searches [36] |
| OpenAI deep research | as above | RL-trained o3 variant, background [29†][30] | autonomous plan + search | annotations | HLE 26.6% at launch [29†] |
| Gemini Deep Research | Google | async agent, collaborative planning [22] | plan, then search, URL context | inline citations | HLE 26.9% / 32.4% [15] |
| Anthropic Research | n/a | orchestrator + parallel subagents [42] | effort-scaled subagents | separate CitationAgent [42][43] | +90.2% vs single agent [42] |
| Exa / Tavily / Brave APIs | own (Exa, Brave) / SERP-based (Tavily, per [26]) | API + research endpoints | n/a | per-field grounding (Exa), citation formats (Tavily) | see Section 4.1 |

## Key takeaways for system designers

1. **Decide whether to search before deciding how.** A trigger classifier or a model
   policy for when to ground saves latency and money, and it removes a class of errors.
   Google's AI Overviews fixes started with *not* answering nonsensical queries and
   data-void queries [17†][36][52].
2. **Fan out, then compress.** Google's query fan-out, Anthropic's parallel subagents and
   Gemini's hundreds of Deep Search queries all share a shape: issue many narrow queries
   in parallel, compress each thread separately, and synthesize once
   [18†][19†][42]. Budget this explicitly. Token usage explained 80% of BrowseComp
   variance in Anthropic's analysis, and multi-agent research costs about 15x a chat
   turn [42].
3. **Treat evidence budget as a first-class parameter.** Modern search APIs take token
   budgets, per-URL caps and thresholds [25][45][47][49]. In vendor benchmarks the budget
   moved results as much as the choice of engine [26]. When you compare providers, hold
   the budget constant.
4. **Make invalid citations unrepresentable.** Prefer pointer-based or annotation-based
   citations that the API resolves (Anthropic Citations, OpenAI `url_citation`, Gemini
   supports) over model-written `[n]` markers or URLs [12][20][39]. Never let the model
   generate links [9].
5. **Validate support separately from generation.** A post-hoc citation pass with a
   text-identity check [43], or FACT-style statement-URL verification [59], catches
   citations that are real but irrelevant. Users reward citation volume and seldom click
   through [57], so preference data alone will not catch them.
6. **Source quality is where grounded systems fail.** Content farms, satire, forums,
   syndicated copies and GEO-optimized pages pass straight through faithful
   summarization [6][17†][42][60†]. Encode source-quality heuristics, prefer primary
   sources, and expose ranking controls such as domain preferences and Goggles.
7. **Show your queries.** Showing the generated search queries next to the citations, as
   Microsoft 365 Copilot does, is cheap. It helps users diagnose failures and helps
   auditors with compliance [8][20].
8. **Freshness is a per-request decision.** Expose `last_updated` or `page_age` to the
   model, and use live crawl or cache bypass only when the question is time-sensitive
   [25][36][37][45].
9. **Web access is an attack surface.** Restrict fetches to URLs that already appeared
   in context, block credential-like URLs, encrypt provenance across turns, and run
   domain allow-lists in regulated settings [36][37][12].
10. **Read the licence before designing the cache.** Terms on raw-content access,
    caching, training and mandatory display can rule out offline reranking, evaluation
    replay or index-building [9][10]. Pick the retrieval provider to fit the architecture
    you need.
11. **Clarify before researching.** Every deep research system adds a scoping stage: a
    clarifier model, collaborative planning, or a rewriter. Under-specified prompts lead
    to broad, slow and expensive research [22][30][31][13].
12. **Benchmark reproducibly.** Use a fixed corpus (BrowseComp-Plus) to compare
    retrievers. Report harness configuration, judge model, and both failed-as-zero and
    failed-excluded scores. Treat any vendor-run leaderboard as a starting point
    [26][58][59].

## References

Entries marked † could not be re-fetched during this edition because of network
restrictions. Claims attributed to them describe the documents as published and should
be re-verified before anyone quotes them. All other entries were retrieved and read while
writing this chapter.

1. † Nakano, R., Hilton, J., Balaji, S., et al. "WebGPT: Browser-assisted question-answering
   with human feedback." OpenAI, 2021. https://arxiv.org/abs/2112.09332
2. † Menick, J., Trebacz, M., Mikulik, V., et al. "Teaching language models to support
   answers with verified quotes" (GopherCite). DeepMind, 2022.
   https://arxiv.org/abs/2203.11147
3. Vu, T., Iyyer, M., Wang, X., et al. "FreshLLMs: Refreshing Large Language Models with
   Search Engine Augmentation" (FreshQA, FreshPrompt; SealQA follow-up). 2023-2026.
   https://arxiv.org/abs/2310.03214 ; repository: https://github.com/freshllms/freshqa
4. Liu, N. F., Zhang, T., Liang, P. "Evaluating Verifiability in Generative Search
   Engines." Stanford, Findings of EMNLP 2023. https://arxiv.org/abs/2304.09848 ; data and
   annotations: https://github.com/nelson-liu/evaluating-verifiability-in-generative-search-engines
   († applies to the headline percentages quoted from the paper; dataset facts are from
   the repository.)
5. Gao, T., Yen, H., Yu, J., Chen, D. "Enabling Large Language Models to Generate Text with
   Citations" (ALCE). Princeton, EMNLP 2023. https://arxiv.org/abs/2305.14627 ; code:
   https://github.com/princeton-nlp/ALCE († applies to the ELI5 citation-support figure.)
6. Aggarwal, P., Murahari, V., Rajpurohit, T., Kalyan, A., Narasimhan, K., Deshpande, A.
   "GEO: Generative Engine Optimization." 2023. https://arxiv.org/abs/2311.09735 ; code:
   https://github.com/GEO-optim/GEO
7. † Ribas, J. "Building the New Bing." Microsoft Bing Search Quality Insights blog,
   February 2023.
   https://blogs.bing.com/search-quality-insights/february-2023/Building-the-New-Bing
8. Microsoft. "Data, privacy, and security for web search in Microsoft 365 Copilot and
   Microsoft 365 Copilot Chat." Microsoft Learn source (MicrosoftDocs), updated February
   2026. https://github.com/MicrosoftDocs/microsoft-365-docs/blob/public/copilot/manage-public-web-access.md
9. Microsoft. "Grounding agents with Bing Search tools" (Foundry Agent Service). Microsoft
   Learn source (MicrosoftDocs), 2026.
   https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/foundry/agents/how-to/tools/bing-tools.md
10. Microsoft. "Grounding with Bing Search and Grounding with Bing Custom Search: Terms of
    Use, including Use and Display Requirements." 2025-2026.
    https://www.microsoft.com/en-us/bing/apis/grounding-legal-enterprise
11. Microsoft. "Grounding with Bing" pricing page. 2026.
    https://www.microsoft.com/en-us/bing/apis/grounding-pricing
12. Microsoft. "Web search" for Azure OpenAI Responses API (three modes, `web_search_call`
    actions, `url_citation`, limitations). Microsoft Learn source (MicrosoftDocs), 2026.
    https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/foundry/openai/includes/how-to-web-search-content.md
13. Microsoft. "Deep Research tool (preview) (classic)" and "o3-deep-research" how-to.
    Microsoft Learn source (MicrosoftDocs), 2025-2026.
    https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/foundry-classic/agents/how-to/tools-classic/deep-research.md
    ; https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/foundry/openai/includes/how-to-deep-research-content.md
14. Thomas, P., Spielman, S., Craswell, N., Mitra, B. "Large Language Models Can
    Accurately Predict Searcher Preferences." Microsoft, SIGIR 2024.
    https://www.microsoft.com/en-us/research/publication/large-language-models-can-accurately-predict-searcher-preferences/
15. Gemini Team, Google. "Gemini 2.5: Pushing the Frontier with Advanced Reasoning,
    Multimodality, Long Context, and Next Generation Agentic Capabilities." Technical
    report, 2025.
    https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
16. † Reid, L. "Generative AI in Search: Let Google do the searching for you" (AI Overviews
    launch). Google, May 2024.
    https://blog.google/products/search/generative-ai-google-search-may-2024/
17. † Reid, L. "AI Overviews: About last week." Google, May 2024.
    https://blog.google/products/search/ai-overviews-update-may-2024/
18. † Stein, R. "Expanding AI Overviews and introducing AI Mode." Google, March 2025.
    https://blog.google/products/search/ai-mode-search/ (URL as cited in [15]; content
    not re-fetched.)
19. † Google. "AI Mode in Google Search: Updates from Google I/O 2025" (Deep Search).
    Google, May 2025. https://blog.google/products/search/google-search-ai-mode-update/
20. Google. `google-genai` Python SDK, `types.py` (`GroundingMetadata`, `GroundingSupport`,
    `SearchEntryPoint`, `DynamicRetrievalConfig`). 2026.
    https://github.com/googleapis/python-genai/blob/main/google/genai/types.py
21. Google. Gemini API Cookbook: "Getting started with information grounding" and
    "Grounding using Search as a tool." 2026.
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/Grounding.ipynb ;
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/Search_Grounding.ipynb
22. Google. Gemini API Cookbook: "Gemini API: Deep Research Agent." 2026.
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_Deep_Research.ipynb
23. Google. "Gemini Fullstack LangGraph Quickstart." 2025.
    https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart
24. Perplexity. "Perplexity API Cookbook" (Agent API is the primary API; Sonar
    deprecated). 2026. https://github.com/ppl-ai/api-cookbook
25. Perplexity. `perplexity-py` official Python SDK (Search API and Agent API parameter
    and response types). 2026. https://github.com/perplexityai/perplexity-py
26. Perplexity Research. "search_evals: An Evaluation Framework for AI-First Web Search"
    (README versions of 25-26 Sep 2025, 24 Feb 2026, and June 2026; `systems.toml`;
    search-engine adapters). https://github.com/perplexityai/search_evals
27. † Perplexity Research. "Architecting and Evaluating an AI-First Search API."
    September 2025.
    https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api
    (URL as linked from [26].)
28. † OpenAI. "Introducing ChatGPT search." October 2024.
    https://openai.com/index/introducing-chatgpt-search/
29. † OpenAI. "Introducing deep research." February 2025.
    https://openai.com/index/introducing-deep-research/
30. OpenAI Cookbook. "Introduction to the Deep Research API." 2025-2026.
    https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb
31. OpenAI Cookbook. "Deep Research Agents Cookbook" (Agents SDK, four-agent pipeline).
    2025-2026.
    https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api_agents.ipynb
32. OpenAI. `simple-evals` (reference implementations of SimpleQA and BrowseComp).
    2024-2025. https://github.com/openai/simple-evals
33. † Wei, J., et al. "BrowseComp: A Simple Yet Challenging Benchmark for Browsing
    Agents." OpenAI, April 2025. https://openai.com/index/browsecomp/ ;
    https://arxiv.org/abs/2504.12516 (URLs as linked from [26][32].)
34. Anthropic. "Claude can now search the web." March 2025 (global availability update May
    2025). https://www.anthropic.com/news/web-search (now redirects to
    https://claude.com/blog/web-search)
35. Anthropic. "Introducing web search on the Anthropic API." May 2025 (web fetch update
    September 2025). https://claude.com/blog/web-search-api (formerly
    https://www.anthropic.com/news/web-search-api)
36. Anthropic. "Web search tool." Claude Platform documentation, 2026.
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
37. Anthropic. "Web fetch tool." Claude Platform documentation, 2026.
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
38. Anthropic. "Introducing Citations on the Anthropic API." January 2025.
    https://claude.com/blog/introducing-citations-api (formerly
    https://www.anthropic.com/news/introducing-citations-api)
39. Anthropic. "Citations." Claude Platform documentation, 2026.
    https://platform.claude.com/docs/en/build-with-claude/citations
40. Anthropic. "Search results." Claude Platform documentation, 2026.
    https://platform.claude.com/docs/en/build-with-claude/search-results
41. Anthropic. "Claude takes research to new places" (Research and Google Workspace
    launch). April 2025. https://claude.com/blog/research (formerly
    https://www.anthropic.com/news/research)
42. Anthropic Engineering. "How we built our multi-agent research system." June 2025.
    https://www.anthropic.com/engineering/multi-agent-research-system
43. Anthropic. Agent prompts: `research_lead_agent.md`, `research_subagent.md`,
    `citations_agent.md`. Anthropic Cookbook, 2025.
    https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents/prompts
44. Anthropic. "Introducing Claude Opus 4.5" (context management and BrowseComp-Plus
    result). November 2025. https://www.anthropic.com/news/claude-opus-4-5
45. Exa. `exa-py` official Python SDK (search types, highlights, livecrawl, snapshots,
    deep-search output grounding, Agent API, Monitors). 2026.
    https://github.com/exa-labs/exa-py
46. Exa. `exa-mcp-server`. 2026. https://github.com/exa-labs/exa-mcp-server
47. Tavily. `tavily-python` official SDK (search, extract, crawl, map, research, feedback).
    2026. https://github.com/tavily-ai/tavily-python
48. Tavily. `tavily-mcp`. 2026. https://github.com/tavily-ai/tavily-mcp
49. Brave. `brave-search-mcp-server` (Brave Search API endpoints, LLM Context parameters,
    Goggles, summarizer). 2026. https://github.com/brave/brave-search-mcp-server
50. † Brave. "Brave Search API." Product page. https://brave.com/search/api/
51. You.com. `@youdotcom-oss/mcp` (hosted You.com MCP server tools). 2026.
    https://github.com/youdotcom-oss/mcp-server
52. ItzCrazyKns. Vane (formerly Perplexica): README and architecture docs ("How Vane
    Works"). 2026. https://github.com/ItzCrazyKns/Vane/tree/master/docs/architecture
53. Semnani, S. J., Yao, V. Z., Zhang, H. C., Lam, M. S. "WikiChat: Stopping the
    Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia."
    Stanford, Findings of EMNLP 2023. https://arxiv.org/abs/2305.14292 ; code:
    https://github.com/stanford-oval/WikiChat
54. Sentient. "OpenDeepSearch: Democratizing Search with Open-source Reasoning Models and
    Reasoning Agents." 2025. https://arxiv.org/abs/2503.20201 ; code:
    https://github.com/sentient-agi/OpenDeepSearch
55. Roucher, A., Villanova del Moral, A., Noyan, M., Wolf, T., Fourrier, C. "Open-source
    DeepResearch: Freeing our search agents." Hugging Face, 4 February 2025.
    https://huggingface.co/blog/open-deep-research
56. Tongyi Lab, Alibaba. "Tongyi DeepResearch" (30B-A3B agentic model). 2025.
    https://github.com/Alibaba-NLP/DeepResearch ; paper: https://arxiv.org/abs/2510.24701
57. Miroyan, M., Wu, T.-H., King, L., et al. "Search Arena: Analyzing Search-Augmented
    LLMs." UC Berkeley / LMArena, ICLR 2026. https://arxiv.org/abs/2506.05334 ; code and
    data: https://github.com/lmarena/search-arena
58. Chen, Z., et al. "BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of
    Deep-Research Agent." University of Waterloo et al., 2025.
    https://arxiv.org/abs/2508.06600 ; code: https://github.com/texttron/BrowseComp-Plus
59. Du, M., Xu, B., Zhu, C., Wang, X., Mao, Z. "DeepResearch Bench: A Comprehensive
    Benchmark for Deep Research Agents" (RACE and FACT). 2025-2026.
    https://arxiv.org/abs/2506.11763 ; code: https://github.com/Ayanami0730/deep_research_bench
60. † Jaźwińska, K., Chandrasekar, A. "AI Search Has A Citation Problem." Tow Center for
    Digital Journalism, Columbia Journalism Review, March 2025.
    https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php
61. † Chapekis, A., Lieb, A. "Google users are less likely to click on links when an AI
    summary appears in the results." Pew Research Center, 22 July 2025.
    https://www.pewresearch.org/short-reads/2025/07/22/google-users-are-less-likely-to-click-on-links-when-an-ai-summary-appears-in-the-results/
62. † Cloudflare. "Introducing pay per crawl: enabling content owners to charge AI
    crawlers for access." 1 July 2025. https://blog.cloudflare.com/introducing-pay-per-crawl/
63. † Cloudflare. "Perplexity is using stealth, undeclared crawlers to evade website
    no-crawl directives." 4 August 2025.
    https://blog.cloudflare.com/perplexity-is-using-stealth-undeclared-crawlers-to-evade-website-no-crawl-directives/
64. † Perplexity. "Introducing Comet: Browse at the speed of thought." July 2025.
    https://www.perplexity.ai/hub/blog/introducing-comet
65. † OpenAI. "Introducing ChatGPT Atlas." 21 October 2025.
    https://openai.com/index/introducing-chatgpt-atlas/
66. † Brave. "Agentic Browser Security: Indirect Prompt Injection in Perplexity Comet."
    August 2025. https://brave.com/blog/comet-prompt-injection/
67. Howard, J. "The /llms.txt file" (v1 September 2024; v2 2026). https://llmstxt.org/ ;
    source: https://github.com/AnswerDotAI/llms-txt
68. Gou, B., et al. "Mind2Web 2: Evaluating Agentic Search with
    Agent-as-a-Judge." Ohio State University NLP Group, NeurIPS 2025 Datasets and
    Benchmarks. https://arxiv.org/abs/2506.21506 ; code:
    https://github.com/OSU-NLP-Group/Mind2Web-2
