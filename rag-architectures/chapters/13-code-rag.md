# 13. Code Retrieval and RAG: Codebase Indexes, Repo Maps, and Agentic Search in Coding Assistants

## Overview

Code is the domain where retrieval-augmented generation has been argued over most publicly and measured most often. A coding assistant must answer questions about a corpus that the model has never seen: a private repository. That corpus changes with every keystroke. It is full of exact identifiers that must not be paraphrased. Its access rules come from source control and are strict. And it has structure (definitions, references, imports, call graphs) that plain text chunks lose.

Between 2023 and 2026, four architectural families emerged. Most production systems now combine at least two of them:

1. **Index-first semantic retrieval.** Chunk the repository, embed the chunks, keep the index in sync, and retrieve top-k chunks for a query. Cursor, GitHub Copilot and Augment Code are the leading public examples. Each has added mechanisms for freshness: Merkle-tree sync, diffing against the indexed commit, real-time per-branch indexes.
2. **Search-engine (lexical) retrieval.** Reuse an existing code search engine with BM25-style ranking instead of embeddings. Sourcegraph Cody dropped embeddings in early 2024 and took this route.
3. **Structural context.** Parse the code (tree-sitter, language servers, dependency graphs) and give the model a compressed structural view. Aider's PageRank-ranked repository map and graph localizers such as LocAgent and RepoGraph belong here.
4. **Agentic search.** Give the model `grep`, `find`/`glob` and file reads, and let it search iteratively with no index. Claude Code is the reference design. SWE-agent's agent-computer interface and Anthropic's SWE-bench scaffold showed the approach on benchmarks. Cognition's SWE-grep then trained small models with reinforcement learning to do this kind of search quickly.

A 2025 talk by Augment's Colin Flaherty describes how retrieval demands grew: simple, low-latency retrieval for completions in 2023, multi-file retrieval for chatbots in 2024, and highly complex retrieval across many parts of a codebase for agents in 2025 [18]. This chapter follows that arc. It covers the production systems first, then repository maps and graphs, code embedding models, and finally what SWE-bench-style evaluations show about retrieval.

```
                      CODE CONTEXT ARCHITECTURES (2023-2026)

  latency budget:   < 1 s (completion)      seconds (chat)        minutes (agent)
                    |                       |                     |
  INDEX-FIRST       Copilot/Cursor/Augment embeddings + sync ---> exposed as a tool
  LEXICAL ENGINE    Cody: Jaccard windows   Cody: BM25 over Sourcegraph search
  STRUCTURAL        tree-sitter intent      Aider repo map        LocAgent / RepoGraph
  AGENTIC SEARCH                                                  Claude Code, SWE-agent,
                                                                  SWE-grep subagents
                    <------------- convergence: hybrid toolboxes ------------->
```

## 13.1 Why code retrieval differs from document RAG

The founding benchmark paper made the core difficulty concrete. SWE-bench (ICLR 2024) built its retrieval baseline with BM25. The authors judged dense retrieval "ill-suited" because keys and queries are very long, and because the setting of retrieving code documents with natural-language queries is unusual [31]. Results were poor, and a larger context window made them worse:

| Max context (BM25) | BM25 recall of gold-patch files (avg) | Claude 2 resolved |
|---|---|---|
| 13k tokens | 29.58% | 1.96% |
| 27k tokens | 44.41% | 1.87% |
| 50k tokens | 51.06% | 1.22% |

At 27k tokens, BM25 retrieved a superset of the oracle files in about 40% of instances. In almost half of instances it retrieved none of them [31]. With oracle retrieval, which supplies the files the reference patch edited, Claude 2 reached 4.8%. An "oracle-collapsed" setting showed only the edited lines plus or minus 15 lines. It reached 5.93% for Claude 2 and 9.39% for Claude 3 Opus [31]. The authors conclude that models "become distracted by additional context". Even when a larger BM25 budget raises recall, performance drops [31]. That trade-off between recall and distraction runs through everything that follows.

Several properties make code different from document RAG:

- **Lexical anchors dominate.** CodeRAG-Bench measured NDCG@10 for many retrievers. On RepoEval, where the query is code that shares identifiers with its target, BM25 scored 93.2, almost tied with the best dense model (Voyage-code, 94.3). On SWE-bench, where the query is natural-language issue text, BM25 fell to 43.0, while the best dense retrievers reached about 62.7 [39]. Whether lexical search is enough depends on how much vocabulary the query and target share.
- **Freshness is part of correctness.** A function created a minute ago, or one that exists only on the developer's branch, must be retrievable. Augment argued that a main-branch index refreshed every ten minutes or so "does not cut it" for this reason [12].
- **Access control.** Embedding a repository typically sends its code to an embedding service and creates a derived artifact that must be permissioned. Sourcegraph [7] and Anthropic [20] both cite this as a reason to avoid or limit embeddings.
- **Retrieved context can hurt.** Repoformer found "a large proportion of the retrieved contexts proving unhelpful or harmful" and trained models to decide whether to retrieve at all [38]. CodeRAG-Bench saw models copy functions from the retrieved context, sometimes overwriting the function being queried [39].

## 13.2 Cursor: Merkle-tree sync, chunk embeddings, and teammate index reuse

**Problem.** Cursor's agent uses semantic search alongside grep. Answering "where do we handle authentication?" requires an embedding index of every repository a user opens. Large repositories with tens of thousands of files "can take hours to process if indexed naively". Semantic search is not available until at least 80% of that work is finished [1].

**Architecture.** Cursor's January 2026 engineering post describes the pipeline [1]:

- **Merkle tree of the workspace.** Each file has a cryptographic hash. Each folder's hash is derived from its children's hashes. A local edit changes only the hashes on the path from the file to the root. The client compares its tree with the server's. Mismatched entries are synced, matching subtrees are skipped, entries missing on the client are deleted from the server, and entries missing on the server are added. The sync never modifies client files. For a 50,000-file workspace, filenames plus SHA-256 hashes total about 3.2 MB. Without the tree, that would be transferred on every update [1].
- **Syntactic chunking and an embedding cache keyed by chunk content.** Changed files are split into syntactic chunks and embedded asynchronously in the background. Unchanged chunks hit the cache, so most edits re-embed very little [1].
- **Storage and privacy.** According to Cursor's security documentation as summarized by Engineer's Codex, embeddings are stored in a remote vector database (Turbopuffer) along with line ranges and an obfuscated relative path. The path is split on `/` and `.`, and each segment is encrypted with a client-held key. At query time the server returns obfuscated paths and line ranges, and the client reads the chunk text from local disk. The same summary reports a ten-minute periodic hash check [3]. These operational details come from a secondary source, so treat them as approximate.

```
  CLIENT (IDE)                                   CURSOR BACKEND
  files --> Merkle tree (SHA-256 per file,
            folder = H(children))  -- root/subtree hashes -->  server Merkle tree
                    | diverging branches only                      |
  changed files --> syntactic chunks --> embed (async) -----> chunk-embedding cache
                                                            (keyed by chunk content)
                                                                    |
  query --> embed ----------------------------------------> vector DB (obfuscated
                                                            paths + line ranges)
  local read of returned line ranges <------ top-k ids ----------+
  (code text is read from the client's own disk)
```

**Index reuse with proof of possession.** Cursor found that clones of the same codebase are, on average, 92% similar across users within an organization. It therefore copies a teammate's index instead of building a new one [1]:

1. The client computes its Merkle tree and derives a **similarity hash (simhash)** that summarizes the file-content hashes.
2. The server treats that simhash as a vector. It searches a vector database of the simhashes of the team's (or the user's) existing indexes, and accepts a match above a threshold.
3. The matched index is copied in the background. Meanwhile the client can already query the original index.
4. **Content proofs** stop data leaking between users. The client uploads its full Merkle tree, which associates a hash with each encrypted path. During search, the server drops any result whose file hash the client cannot prove it has. Once the client and server roots match, the proofs are deleted.

The reported effect on time-to-first-query was: median 7.87 s to 525 ms, 90th percentile 2.82 min to 1.87 s, and 99th percentile 4.03 h to 21 s [1].

**Why bother: measured value of semantic search.** Cursor's November 2025 post measured its agent with and without semantic search [2]. On Cursor Context Bench, an internal dataset with known answers, semantic search gave on average 12.5% higher answer accuracy, ranging from 6.5% to 23.5% depending on the model. It improved accuracy for every model tested. In an online A/B test, code retention rose 0.3% overall and 2.6% on codebases with 1,000 or more files. Without semantic search, dissatisfied follow-up requests rose 2.2% [2]. Cursor trains its own embedding model on agent traces. An LLM looks at a completed session and ranks what would have been most helpful to retrieve at each step, and the embedding model is trained to align its similarities with those rankings [2]. The post concludes that "the combination of these two [grep and semantic search] leads to the best outcomes" [2].

**Failure modes and lessons.** Indexing cost grows with repository size and team churn, and Cursor's fix is to deduplicate across users rather than build a faster embedder. Cross-user sharing raises a real confidentiality risk, which the content-proof filter addresses at query time. The gains are concentrated in large codebases, which is consistent with the agentic-search findings below.

## 13.3 GitHub Copilot: a remote code-search index reconciled with the local diff

**Problem.** Copilot's chat and agent modes need repository context in VS Code. The repository may already be indexed on GitHub at some commit, while the developer's working tree has drifted.

**Index build time.** In March 2025, GitHub made "instant semantic code search indexing" generally available. Indexing had taken about five minutes. It now finishes in a few seconds in most cases and up to 60 seconds at worst [4].

**Embedding model.** In September 2025, GitHub announced a new code embedding model for Copilot in VS Code. Coverage of the announcement reports a 37.6% improvement in retrieval quality, about twice the throughput and an 8x smaller index. It also reports code acceptance-ratio gains of 110.7% for C# and 113.1% for Java, and training with contrastive learning on hard negatives (code that looks right but is not) [5]. These are vendor numbers and were not independently reproduced.

**Client architecture (open source).** The VS Code Copilot Chat extension is open source. Its `workspaceChunkSearch` module shows how remote and local retrieval are combined. The figures below come from the May 2026 source snapshot [6]:

- **Remote first.** `CodeSearchChunkSearch` "first calls the Github code search API to get a context window of files that are similar to the query. Then it uses the embeddings index to find the most similar chunks in the context window." A pre-check reports whether semantic search is available for the repository and at which `semantic_commit_sha` it was indexed. The semantic endpoint accepts prompts of up to 8k bytes.
- **Diff reconciliation.** The client diffs the workspace against the indexed commit. If more than 2,000 files differ (`maxDiffSize`), or more than 70% of files (`maxDiffPercentage = 0.70`), remote code search is not used. Otherwise, remote results for unchanged files are merged with results for the changed files. Changed files are searched with local embeddings when their count is under an experiment-controlled threshold, and the local diff search has a 15-second timeout.
- **Chunk budget.** The local naive chunker caps chunks at 250 tokens (`MAX_CHUNK_SIZE_TOKENS`). The number of chunks requested is the prompt's token budget divided by 250.
- **Embedding types.** The client knows two types: `text-embedding-3-small-512` and `metis-1024-I16-Binary`. The latter is a 1024-dimension embedding with binary quantization.
- **Post-processing.** Chunks matched by `.copilotignore` rules are filtered out. Results are re-sorted by embedding distance, and an optional remote reranker endpoint is gated by an experiment flag.

```
  query --+--> embed query (started eagerly)
          |
          +--> GitHub semantic code search @ indexed commit --> candidate chunks
          |                                                    (exclude diffed files)
          +--> local diff vs indexed commit
                  |-- > 2,000 files or > 70% changed --> skip remote, warn
                  |-- small diff --> local embeddings over changed files (15 s cap)
          merge --> ignore-filter --> sort by distance --> [optional reranker]
               --> top (token_budget / 250) chunks into the prompt
```

**Lessons.** Copilot treats the server index as a cache that is correct only up to a known commit. The working-tree diff is patched in locally. This is a different freshness strategy from Cursor's sync-everything Merkle tree and Augment's real-time per-branch index (Section 13.5).

## 13.4 Sourcegraph Cody: from embeddings to search-engine retrieval and layered long context

**Autocomplete retrieval (2023).** Philipp Spiess's October 2023 account of Cody's completion pipeline sets the constraint: the time from keystroke to visible completion should be "definitely under one second", which leaves little room for expensive retrieval [8]. The design has two steps:

- **Planning.** Rule-based heuristics, guided by tree-sitter, classify the request: single-line or multi-line, function body, docstring, method call.
- **Retrieval from editor context.** Open tabs and recently viewed files are searched with a sliding-window **Jaccard similarity**. A few lines above the cursor are matched against windows over the candidate files.

Cody experimented with reusing its embeddings index for completions, but began moving away from it to avoid extensive caching. Internal experiments showed that "adding irrelevant context can make the response quality worse" [8]. Later autocomplete work focused on latency. In August 2024 Sourcegraph reported that P75 latency for single-line completions fell from 900 ms to 690 ms, and accepted characters per user rose 58% [9].

**Chat retrieval without embeddings (2024).** In February 2024, Sourcegraph explained that Cody Enterprise was "leaving [embeddings] behind (for now)". It had used OpenAI `text-embedding-ada-002` [7]. Three reasons were given: code had to be sent to a third party, keeping embeddings fresh created admin complexity, and vector search over codebases with more than 100,000 repositories "is complex and resource-intensive", which held back multi-repository context [7]. A June 2024 follow-up frames the target as context at three scopes (the local file, the open repository and other remote repositories) and positions the Sourcegraph platform as the way to reach the third [11]. The replacement uses the Sourcegraph search platform:

```
  user query --> tokenize + cleanse --> Sourcegraph search over selected repos
                                         (up to 10 repos; adapted BM25 + learned signals)
                                                  | ranked file snippets
  IDE local context (open files) ---------------+ |
                                                v v
                                   global ranking across context providers
                                                |
                              take first N snippets (N depends on snippet length)
                                                |
                         prompt = prefix + user input + context --> LLM
```

**Long context as a retrieval layer (late 2024).** Sourcegraph and Google tested Gemini 1.5 Flash with a 1M-token context in Cody. They reported higher "essential recall" and a lower hallucination rate than the production Cody of September 2024 [10]. Time to first token was 30 to 40 seconds for about 1 MB of context, and fell to about 5 seconds with prefetching and model-state caching [10]. To keep that cache valid, they proposed a **layered context** design. A *permalayer* of the "logical codebase" takes about 90% of the context budget. An append-only *action history* follows, and a small *global retrieval* layer holds far-away definitions found through code search and navigation [10]. Retrieval becomes a matter of choosing what to cache, while long-range retrieval remains for distant code.

**Lessons.** Cody shows an embedding-free design working at enterprise scale. It reuses a mature lexical engine, adds tuned ranking signals, and merges local and remote context with a global rank. The trade-off is weaker conceptual matching when queries share few tokens with the code, which is the SWE-bench-like regime in CodeRAG-Bench's measurements [39].

## 13.5 Augment Code: a real-time personal index and custom context models

Augment's engineering blog is not directly accessible from this research environment. The facts below come from a research digest of that blog and of Augment's documentation [12][13][14][15]. They are vendor claims, and the numbers should be treated as approximate.

**Problem.** Retrieval must match the exact version of the code the developer is working on, including their branch and unsaved changes. Augment argued that indexes refreshed from the main branch on a delay of around ten minutes miss functions that exist only on other branches [12].

**Architecture.**

- **Per-developer, real-time index.** Updates land within seconds of a code change. The indexer processes "many thousands of files per second", so a branch switch is handled almost instantly [12]. The pipeline uses Google Cloud Pub/Sub for change events, Bigtable for storage and GPU embedding workers on a custom inference stack. Overlapping indexes between users of the same tenant are shared in RAM to keep serving costs down [12].
- **Custom context models.** Augment trains its own retrieval models rather than using generic embedding APIs. It argues that generic models confuse call sites with definitions and retrieve content that is "relevant" but unhelpful. For example, implementation code from PyTorch that the LLM already knows does not improve its output. The training goal is helpfulness over relevance [12].
- **Quantized ANN for 100M+ line codebases.** Augment estimated about 20 bytes of embedding per line of code (2 GB for 100M lines) and about 2 seconds for a brute-force search. A quantized first pass followed by full-precision re-scoring of the candidates was reported to cut memory to about 250 MB and latency to under 200 ms, while keeping about 99.9% of exact-search accuracy [13].
- **Commit history ("Context Lineage").** Each commit diff is summarized by a lightweight LLM (Gemini 2.0 Flash is named) into its goal, the functions and files it touches, and retrieval terms. The summaries are embedded alongside the file chunks, so the agent can find earlier, similar changes [14].
- **Edit events.** For completions, Augment feeds a stream of recent edits. It reported this as its single largest improvement on internal benchmarks (+2.6%), ahead of a better base model (+1.5%), RL tuning (+1.3%) and smarter chunking (+0.4%) [15].
- **Context Engine as MCP (2026).** The retrieval engine is now offered as an MCP server, either through the local Auggie CLI or a hosted HTTP endpoint, so other agents can use it [16].

**The SWE-bench counterpoint.** Augment's open-source SWE-bench Verified agent scored 65.4%. It used Claude 3.7 Sonnet as the driver and o1 as the ensembler, with only bash, file view/edit and sequential-thinking tools [17]. Its builder, Colin Flaherty, said: "We explored adding various embedding-based retrieval tools, but found that for SWE-bench tasks this was not the bottleneck - grep and find were sufficient" [18]. He named where embeddings still matter: large codebases, unstructured content such as Slack and docs, third-party code that models have not memorized, and non-text media. He also noted that SWE-bench repositories are smaller than real ones and its tasks mostly take under an hour [18]. His recommendation was to expose existing retrieval systems as tools inside an agent loop, add rerankers, and evaluate end to end, because a persistent agent can mask a weak retriever [18].

## 13.6 Windsurf and Cognition: SWE-grep, an RL-trained fast-context subagent

**Problem.** Cognition observed that agent trajectories in Windsurf and Devin "were often spending >60% of their first turn just retrieving context" [19]. Embedding search is fast but inaccurate for multi-hop questions. Agentic search is flexible but slow, because it needs dozens of sequential round trips and floods the context with irrelevant tokens [19].

**Design (October 2025).** SWE-grep and SWE-grep-mini are small models trained for one job: retrieval [19].

- **Bounded parallelism.** The models issue up to 8 parallel tool calls (grep, glob, read and similar) per turn, for at most 4 turns: 3 of exploration and 1 for the answer. In Cognition's ablations, raising parallelism from 4 to 8 searches per turn cut the turns needed from 6 to 4 at the same performance.
- **A verifiable output.** The subagent returns a list of files with line ranges, not a summary. A free-form summary from a fast model can mislead the main model and is hard to grade. A list of files and lines can be scored against ground truth.
- **Reward.** The reward is a weighted F1 over file and line retrieval, using an F-beta with beta = 0.5 that favours precision. The reasoning is that "polluting the context of the main agent was more detrimental than leaving some context out" [19]. Training used multi-turn RL with per-sequence importance sampling and a leave-one-out baseline. Overlong trajectories and trajectories with extreme importance ratios were masked out, and advantages were scaled by tool calls per turn so the model did not learn to emit duplicate calls [19].
- **Serving.** On Cerebras, SWE-grep-mini runs at over 2,800 tokens/s and SWE-grep at over 650 tokens/s, compared with 140 tokens/s for Haiku 4.5, the next-fastest model Cognition evaluated [19].

```
  main agent (frontier model)            Fast Context subagent (SWE-grep)
  "where is X handled?" ------------->  turn 1: 8 parallel {grep, glob, read}
                                         turn 2: 8 parallel ...
                                         turn 3: 8 parallel ...
                                         turn 4: answer = [(file, line ranges)]
  reads only cited ranges <-----------  reward in training: F0.5(files, lines)
```

Cognition evaluated on its internal Cognition CodeSearch Eval, built from real repositories with labelled relevant files and line ranges. It reports that the models match frontier models at retrieval while taking an order of magnitude less time [19]. With Sonnet 4.5 as the main model, Fast Context solved the same number of difficult SWE-bench Verified tasks in less end-to-end time [19]. Cognition set itself a 5-second "flow window" as a latency target [19]. The design point is notable. It accepts that agentic search beats static embedding retrieval on quality, and attacks its latency through parallelism, a restricted tool set and a model trained specifically for retrieval.

## 13.7 Claude Code: agentic search with no index

**History.** In a March 2026 interview, Claude Code's creator Boris Cherny described the early experiments [20]. The first version used RAG with a local vector database written in TypeScript and embeddings computed by a cloud model. "That worked like pretty good", but the code "drifted out of sync". A newly written local function was not yet indexed, and it was unclear how the index should be permissioned. The team also tried "using the model to ... index everything recursively". In the end, "agentic search just outperformed everything", and "agentic search ... is a fancy word for glob and grep" [20]. Cherny traced the idea to Instagram: when click-to-definition broke, engineers grepped a global index for `foo(` [20].

**Current design.** Claude Code runs an agentic loop of gather context, act and verify [22]. The retrieval tools are [23][24]:

- `Grep`, built on ripgrep, which respects `.gitignore`, and `Glob` for filename patterns. On macOS, Linux and WSL these are now left out of the default tool set. Claude instead runs `find` and `grep` through Bash, which in Claude's shell are embedded versions of `bfs` and `ugrep` [23].
- An optional `LSP` tool for definitions, references and type errors through language servers [23].
- The built-in **Explore** subagent, which is read-only (Write and Edit are denied). It is called with a thoroughness level of *quick*, *medium* or *very thorough* and returns only its findings, which "keeps exploration results out of your main conversation context" [24].
- `CLAUDE.md` project memory loaded at session start. MCP tool definitions are deferred and loaded on demand through tool search [22].

Anthropic's context-engineering guidance frames this as "just in time" retrieval: the agent keeps lightweight references (paths, queries) and loads data through tools, and Claude Code is described as a hybrid of CLAUDE.md loaded up front with glob/grep retrieval on demand. This sidesteps "stale indexing and complex syntax trees" [21]. The same guidance names the cost: runtime exploration is slower than retrieving precomputed data, and agents without guidance waste context on dead ends [21].

**Benchmark lineage.** Anthropic's SWE-bench Verified scaffold for the upgraded Claude 3.5 Sonnet (49%) had only a bash tool and a string-replace editor. Navigation was directory listing and `sed` line ranges, with no retrieval index. Many successful runs took hundreds of turns and more than 100k tokens [25].

**Counter-evidence.** Zilliz's `claude-context` MCP server adds hybrid BM25 plus dense code search with AST chunking and Merkle-tree incremental indexing. In its evaluation, a GPT-4o-mini ReAct agent on 30 SWE-bench Verified instances (3 runs each) reached the same retrieval F1 as a grep-only agent (0.40 for both). It used 39.4% fewer tokens (73,373 vs 44,449) and 36.3% fewer tool calls [26]. This is a vendor study on a small sample, but its result matches the trade-off Anthropic describes: an index buys speed and tokens, not necessarily accuracy.

## 13.8 Aider: the PageRank-ranked repository map

**Problem.** Aider pairs a user with an LLM in their git repository. In 2023 it set out to give the model "code context": how the code to be edited relates to the rest of the repository, delivered efficiently within a limited context window [27]. Sending whole files is "a bulky way to send code context" [27].

**Architecture.** Aider sends a **repository map** with every request. It lists files with their key symbols and the critical definition lines, such as class headers and function signatures, with elisions in between [28]. The map is built as follows:

1. **Tags via tree-sitter.** Modified `tags.scm` queries extract definitions and references for each language. This replaced an earlier map built with universal-ctags [27]. Tags are cached on disk per file and keyed by modification time [30].
2. **Graph.** The nodes are files. For each identifier that is both defined and referenced, an edge runs from each referencing file to each defining file [30].
3. **Edge weights** (from `repomap.py`) [30]:
   - x10 if the identifier was mentioned in the chat.
   - x10 if it is a long (8 or more characters) snake, kebab or camel-case name.
   - x0.1 if it starts with `_` (private).
   - x0.1 if it is defined in more than 5 files (generic names).
   - x50 if the referencing file is already in the chat.
   - The reference count is dampened with a square root.
4. **Personalized PageRank.** Files in the chat and files or identifiers mentioned in it get personalization mass. Each file's rank is then spread over its out-edges, which yields ranked (file, identifier) definitions [30].
5. **Fit to budget.** A binary search over the number of ranked tags finds the largest map that fits the token budget. `--map-tokens` defaults to 1,024, and the budget is expanded (by a default multiplier of 8) when no files are in the chat yet [28][30].

```
  repo files --tree-sitter tags--> defs/refs per file (disk cache by mtime)
        |
        v
  file graph: referencer --(ident, w = mul * sqrt(#refs))--> definer
        |      mul: x10 mentioned, x10 long-name, x0.1 private, x0.1 common, x50 in-chat
        v
  personalized PageRank (chat files / mentioned names get personalization)
        v
  ranked (file, symbol) defs --binary search on count--> map within --map-tokens
        v
  prompt: repo map + files the user or LLM added to the chat
```

**Results and lessons.** In its May 2024 SWE-bench Lite run, Aider described its approach as "Repository map, not RAG". The LLM reads the map, names the files it needs, and Aider adds them to the chat. Aider picked the gold-patch file in 70.3% of tasks and resolved 26.3% overall, a state-of-the-art result at the time [29]. The repo map is cheap, deterministic, needs no embedding service and stays fresh through mtime caching. It is weaker for questions phrased in natural language that share no identifiers with the code, which is where the dense retrievers and agents in Section 13.10 do better.

## 13.9 Code embedding models and benchmarks

**Benchmarks.**

- **CoIR** (ACL 2025): 10 curated datasets covering 8 retrieval tasks across 7 domains, with about two million documents. It is integrated into MTEB [40].
- **CodeRAG-Bench**: retrieval NDCG@10 and end-to-end generation across basic programming, open-domain (library documentation), and repository-level tasks [39].

The CodeRAG-Bench retrieval numbers below were transcribed from the paper by a third-party reading note [39]:

| Retriever (NDCG@10) | HumanEval | DS-1000 | RepoEval | SWE-bench |
|---|---|---|---|---|
| BM25 | 100.0 | 5.2 | 93.2 | 43.0 |
| OpenAI embeddings (text-embedding-3) | 100.0 | 18.2 | 93.0 | 43.3 |
| Jina-v2-code | 100.0 | 26.2 | 90.5 | 58.3 |
| SFR-Mistral (7B) | 100.0 | 19.3 | 83.8 | 62.7 |
| Voyage-code | 100.0 | 33.1 | 94.3 | uncertain (29.1 or 62.7) |

Two third-party transcriptions disagree on Voyage-code's SWE-bench cell (29.1 vs 62.7), so it is marked uncertain. The broader pattern is consistent across sources [39]. Library-documentation retrieval (DS-1000, ODEX) is hard for every retriever. Repository-level retrieval with code queries suits BM25. Issue-to-code retrieval favours strong dense models. On generation, GPT-4o on SWE-bench Lite resolved 2.3% with no retrieval, 30.7% with gold documents and 21.7% with the best retriever [39].

**Representative models.**

- **voyage-code-3** (December 2024). Voyage reports it beats OpenAI-v3-large by 13.80% and CodeSage-large by 16.81% on average across 32 code retrieval datasets. It has a 32K-token context, Matryoshka dimensions (2048, 1024, 512, 256) and int8/binary quantization [42].
- **CoRNStack / CodeRankEmbed.** CoRNStack is a contrastive dataset built from deduplicated The Stack v2 using consistency filtering and curriculum hard-negative mining. CodeRankEmbed is a 137M bi-encoder trained on it. CodeRankLLM, a 7B listwise reranker, is reported as the first LLM fine-tuned as a code reranker [41]. Nomic's 7B `nomic-embed-code` was also trained on CoRNStack (see Chapter 3).
- **Product-internal models.** Cursor trains on rankings of agent traces [2]. Copilot uses contrastive training with hard negatives and a binary-quantized 1024-dimension type [5][6]. Augment's models aim at helpfulness rather than relevance [12]. All three are trained on how retrieval is used in their own product, not on generic code similarity.

**Do better embeddings help agents?** LocAgent's evaluation on SWE-bench Lite compares retrievers with agents on file localization [34]:

| Method | File Acc@5 | Function Acc@10 |
|---|---|---|
| BM25 | 61.68 | ~36 |
| CodeSage-large-v2 | 78.10 | ~44 |
| CodeRankEmbed (137M) | 84.67 | ~58 |
| Agentless (Claude 3.5) | 79.56 | ~58 |
| SWE-agent (Claude 3.5) | 90.15 | ~64 |
| LocAgent (Claude 3.5) | 94.16 | ~77 |

Function-level values are approximate because of extraction artefacts in the source text. A good small code embedder beats BM25 by more than 20 points and matches the pipeline approach. Agents with tools still lead, and graph-guided agents lead the most.

## 13.10 SWE-bench retrieval findings: what the evaluations say

The SWE-bench literature functions as a set of controlled experiments on code retrieval. The main results:

**1. Static RAG baselines were weak.** BM25 RAG resolved 1.96% (Claude 2) on SWE-bench in 2023, and its performance fell as context grew [31]. Agents with purpose-built navigation tools changed the picture. SWE-agent's ACI reached 12.47% on full SWE-bench and 18.00% on Lite with GPT-4 Turbo. The RAG baseline scored 2.67% on Lite and a shell-only agent 11.00% [32]. Moatless Tools took a middle path: its author argues that rather than relying on an agent to reason its way to a solution, it is crucial to build good tools that insert the right context into the prompt. It appears as an agent baseline in later localization studies [34][44].

**2. Search tool design matters more than search power.** SWE-agent's ablation on Lite found [32]:

| Search interface | Resolved |
|---|---|
| Summarized search (lists matching files; capped at 50 results) | 18.0% |
| No search commands | 15.7% |
| Iterative search (one match at a time) | 12.0% |

With iterative search, agents inspected every match exhaustively and wasted their budget. The file viewer showed the same effect: a 100-line window scored 18.0%, a 30-line window 14.3% and full files 12.7% [32].

**3. Hybrid localization beats either method alone.** Agentless localizes hierarchically: repository tree, then files, then classes and functions (shown as signature-only "skeletons"), then edit lines. It uses no agent loop [33]. Its file-level ablation on Lite (the share of issues where the localized set contains the gold file) found [33]:

| File localization method | Contains gold file | Cost |
|---|---|---|
| LLM prompting over the repository tree | 78.67% | $0.02 |
| Embeddings (text-embedding-3-small, 512-token chunks) | 67.67% | $0.05 |
| Embeddings after LLM filters irrelevant folders | 70.33% | $0.04 |
| Combined (most common top-N of both) | 81.67% | $0.06 |

Skeletons also beat full files for element localization: 58.33% vs 53.67%, at a fraction of the tokens [33]. Agentless resolved 32.00% of Lite at $0.70 per issue with GPT-4o, and later reached 40.7% (Lite) and 50.8% (Verified) with Claude 3.5 Sonnet [33].

**4. Graphs help, mostly through precise entity lookup.** LocAgent builds a heterogeneous graph of directories, files, classes and functions with *contain*, *import*, *invoke* and *inherit* edges, plus a BM25 entity index [34]. The agent gets three tools: `SearchEntity` (hierarchical keyword index; returns fold, preview or full code), `TraverseGraph` (type-aware multi-hop BFS) and `RetrieveEntity`. It reports up to 92.7% file-level accuracy with a fine-tuned Qwen2.5-Coder-32B, about 86% lower cost than proprietary models, and a 12% gain in downstream Pass@10 resolution [34]. RepoGraph adds a repository graph to existing frameworks. A third-party review of these papers reports gains of about +2 points in resolve rate (for example, Agentless 27.33% to 29.67% with GPT-4o), at a cost of several thousand extra tokens per issue [36].

**5. Trained retrieve-and-rerank localizers can beat agents at a fraction of the cost.** SweRank uses a bi-encoder (SweRankEmbed) and an LLM reranker (SweRankLLM) trained on issue-to-modified-code pairs mined from GitHub. As reported in a third-party review of the paper, the best configuration reaches 83.21% file Acc@1 on Lite, against 77.74% for LocAgent with Claude 3.5. Reranking costs about $0.011 to $0.015 per instance, against about $0.66 for Claude-based agents [35]. Better localization carried through only weakly to resolution. Repair Pass@1 was 24.5% with SweRank localization, 22.6% with LocAgent's, and 25.9% with perfect localization [35].

**6. Completion benefits from retrieval, but selectively.** RepoCoder's iterative retrieve-generate loop beats in-file completion by more than 10% in every setting [37]. Repoformer learns when to retrieve and reports up to 70% faster online inference with no loss in quality [38].

**7. Practitioner evidence clusters by codebase size.** Augment [18] and Anthropic [20][25] found grep/find sufficient for SWE-bench-sized repositories. Cursor measured real gains from semantic search, concentrated in repositories with 1,000 or more files [2]. Cognition found agentic search better than embeddings but too slow, and trained it to be fast [19]. On SWE-bench-Live, whose issues postdate model training cutoffs, a third-party review reports that pipeline methods lose their edge with stronger models. Flexible agents lead there [43].

```
  Evidence map (who found what)
  repo size / query type      | finding                              | source
  ----------------------------+--------------------------------------+-----------
  small repo, issue text      | grep/find sufficient; agent persists | [18][25]
  large repo (>1k files)      | +semantic search: +2.6% retention    | [2]
  any repo, latency-critical  | RL fast parallel grep subagent       | [19]
  issue -> function           | trained embed + rerank > agents/cost | [35]
  identifiers shared w/ query | BM25 ~ best dense (RepoEval)         | [39]
  NL issue -> code            | dense >> BM25 (SWE-bench NDCG)       | [39]
```

## 13.11 Comparison of production and research designs

| System | Primary retrieval | Index & freshness | Structure use | Where it runs | Public numbers |
|---|---|---|---|---|---|
| Cursor [1][2][3] | Agent: grep + semantic search tool | Merkle-tree sync; chunk-content embedding cache; simhash index reuse with content proofs | Syntactic chunks | Server vectors (obfuscated paths); code read locally | TTFQ p50 7.87 s to 525 ms; +12.5% QA accuracy |
| GitHub Copilot [4][5][6] | GitHub semantic code search + embeddings | Server index at known commit; local diff patched with local embeddings | 250-token chunks | GitHub service + VS Code client | Index build ~5 min to seconds; +37.6% retrieval quality (reported) |
| Sourcegraph Cody [7][8][10] | BM25-style search engine; Jaccard windows for completion | Existing code-search index; no embeddings | Tree-sitter intent planning | Sourcegraph instance + IDE | P75 single-line 900 to 690 ms |
| Augment [12][13][14][15] | Custom context models; also MCP | Real-time per-developer index (seconds) | Commit summaries; edit events | Google Cloud | ~250 MB / <200 ms ANN for 100M LOC (reported) |
| Windsurf / Cognition [19] | RL-trained parallel grep subagent | None needed; fast tools | Returns files + line ranges | Cerebras-served small models | 8 calls x 4 turns; >2,800 tok/s (mini) |
| Claude Code [20][21][23][24] | Agentic grep/find/read; optional LSP | None (live filesystem) | LSP plugin; Explore subagent | Local tools, remote model | Qualitative ("outperformed everything") |
| Aider [27][29][30] | Repo map + LLM-requested files | Tag cache by file mtime | Tree-sitter defs/refs + PageRank | Local | 70.3% gold-file pick; 26.3% Lite |
| Agentless [33] | LLM over tree + embeddings, hierarchical | Built per issue | Skeletons | Pipeline | 81.67% file containment; 32% Lite at $0.70 |
| LocAgent [34] | Agent over code graph + BM25 entity index | Built per repo snapshot | Heterogeneous graph, multi-hop BFS | Pipeline | 92.7% file accuracy (fine-tuned 32B) |
| SweRank [35] | Bi-encoder + LLM listwise reranker | Embedding index | Function-level units | Pipeline | 83.21% file Acc@1 (reported) |

## 13.12 Failure modes catalogue

- **Staleness.** The index lags the working tree, so newly written or branch-only code is missing. Cherny cited this as the reason Claude Code dropped vector RAG [20]. The fixes are Merkle-tree sync [1], diffing against the indexed commit [6], and real-time per-branch indexes [12].
- **Leakage between users or tenants.** Shared or copied indexes can expose code a user should not see. Cursor drops results whose file hash the client cannot prove it holds [1]. Augment's documentation describes proof-of-possession for file access (reported) [16].
- **Distraction from extra context.** Larger BM25 windows lowered SWE-bench resolution [31]. Full-file views and iterative search lowered SWE-agent's score [32]. Irrelevant context degraded Cody's completions [8]. SWE-grep's precision-weighted reward exists for this reason [19].
- **Conceptual queries miss with lexical search.** BM25 NDCG on SWE-bench issue text is about 43 [39]. Semantic search helped Cursor most on large repositories [2].
- **Slowness and token cost of agentic search.** Agentic search can take dozens of sequential turns [19], and successful runs can take more than 100k tokens [25]. An index can cut tokens by about 40% at equal F1 [26].
- **Weak models gain nothing.** In CodeRAG-Bench, DeepSeekCoder-7B gained nothing on SWE-bench Lite even with gold documents [39].
- **Summaries that mislead.** A retrieval subagent's free-form summary can draw wrong conclusions. Cognition returns file and line ranges instead [19]. Claude Code's Explore returns findings to a main agent that can re-read the files [24].

## Key takeaways for system designers

1. **Make retrieval a toolbox, not a pipeline stage.** Every leading system now exposes retrieval as tools to an agent loop: grep, semantic search, LSP and graph traversal. The best measured outcomes come from combinations: grep plus semantic search at Cursor [2], prompting plus embeddings in Agentless [33], and entity index plus graph in LocAgent [34].
2. **Treat freshness as a correctness property.** With an index, design the sync protocol before choosing the embedder. Options include a Merkle tree with a content-keyed embedding cache [1], an indexed commit plus local diff with explicit cut-offs [6], or real-time per-branch updates [12]. Without an index, the live filesystem is always fresh [20].
3. **Match the retriever to vocabulary overlap.** Identifier-heavy queries suit BM25 and grep, which are near-optimal there [39]. Natural-language issue text needs strong dense or trained retrievers [34][35][39]. Route by query shape, or let the agent choose.
4. **Optimize for precision of what enters the main context.** Distraction is the most consistent negative result in code RAG [31][32][38]. Use skeletons or repo maps instead of whole files [27][33], capped and summarized search results [32], and subagents that return line ranges [19].
5. **Evaluate end to end, with ground truth for localization.** Agents can mask weak retrievers [18]. Build an internal benchmark like Cursor Context Bench [2] or Cognition CodeSearch Eval [19] with known files and lines, and run online A/B tests on retention and follow-ups [2].
6. **Train on your own usage.** The most distinctive embedding work in this space learns from product traces: agent sessions ranked by an LLM [2], hard negatives from near-miss code [5], and helpfulness rather than relevance [12]. Open models trained on CoRNStack show that data curation is the main lever [41].
7. **Scale changes the answer.** For repositories the size of SWE-bench's, grep and a persistent agent are enough [18][25]. For monorepos and multi-repository context, an index (lexical [7] or vector [12]) pays for itself in latency and tokens [2][26]. Budget for indexing costs that grow with team size, and deduplicate across users [1].
8. **Security shapes architecture.** Sending code to embedding providers and permissioning derived indexes were decisive reasons for Sourcegraph's and Anthropic's choices [7][20]. Cursor's obfuscated paths and content proofs [1][3] show what it takes to keep an index while limiting exposure.

## References

Note on access: the research environment could not reach several vendor and arXiv domains directly. Content for [1], [2], [3], [5], [12]-[16], [19], [35], [36] and [43] was read from verbatim or near-verbatim copies, or from third-party reading notes, hosted on GitHub. The URLs below are the original sources named in those copies. The Sourcegraph posts [7]-[10] were read from Sourcegraph's own website source repository (github.com/sourcegraph/about). The Aider [27]-[30], Copilot [6] and benchmark repositories were read directly.

1. Stribling, J. (Cursor). "Securely indexing large codebases." Cursor Blog, January 27, 2026. https://cursor.com/blog/secure-codebase-indexing
2. Heule, S., Jia, E., Jain, N. (Cursor). "Improving agent with semantic search." Cursor Blog, November 6, 2025. https://cursor.com/blog/semsearch
3. Engineer's Codex. "How Cursor Indexes Codebases Fast" (summarizing Cursor's security documentation and forum posts). 2025. https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast
4. GitHub. "Instant semantic code search indexing now generally available for GitHub Copilot." GitHub Changelog, March 12, 2025. https://github.blog/changelog/2025-03-12-instant-semantic-code-search-indexing-now-generally-available-for-github-copilot
5. GitHub. "GitHub Copilot gets smarter at finding your code: Inside our new embedding model." GitHub Blog, September 2025. https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/
6. Microsoft. vscode-copilot-chat source code, `src/platform/workspaceChunkSearch/` (CodeSearchChunkSearch, WorkspaceChunkSearchService), `src/platform/embeddings/common/embeddingsComputer.ts`, `src/platform/chunking/node/naiveChunker.ts`. GitHub, snapshot of May 2026. https://github.com/microsoft/vscode-copilot-chat
7. Isken, A., Hill, C. (Sourcegraph). "How Cody understands your codebase." Sourcegraph Blog, February 15, 2024. https://sourcegraph.com/blog/how-cody-understands-your-codebase
8. Spiess, P. (Sourcegraph). "The lifecycle of a code AI completion." Sourcegraph Blog, October 31, 2023. https://sourcegraph.com/blog/the-lifecycle-of-a-code-ai-completion
9. Sev, C. (Sourcegraph). "Improving Cody Autocomplete: Faster and Smarter." Sourcegraph Blog, August 21, 2024. https://sourcegraph.com/blog/improving-cody-autocomplete-faster-smarter
10. Liu, B. (Sourcegraph). "Toward infinite context for code." Sourcegraph Blog, November 7, 2024. https://sourcegraph.com/blog/towards-infinite-context-for-code
11. Isken, A. (Sourcegraph). "How Cody provides remote repository awareness for codebases of every size." Sourcegraph Blog, June 25, 2024. https://sourcegraph.com/blog/how-cody-provides-remote-repository-context
12. Augment Code. "A real-time index for your codebase: secure, personal, scalable." Augment Blog, 2024. https://www.augmentcode.com/blog/a-real-time-index-for-your-codebase-secure-personal-scalable (read via research digest: https://github.com/Lin-Guanguo/llm-memory-research/blob/main/augmentcode.research.md)
13. Augment Code. "How we made code search 40% faster" (quantized vector search for 100M+ line codebases). Augment Blog, 2025. https://www.augmentcode.com/blog/repo-scale-100M-line-codebase-quantized-vector-search
14. Augment Code. "Context Engine: Now with full commit history" (Context Lineage). Augment Blog, 2025. https://www.augmentcode.com/blog/announcing-context-lineage
15. Augment Code. "Context beats modeling." Augment Blog, 2025. https://www.augmentcode.com/blog/context-modeling
16. Augment Code. "Context Engine MCP overview." Augment Documentation, 2026. https://docs.augmentcode.com/context-services/mcp/overview
17. Augment Code. "Augment SWE-bench Verified Agent" (65.4% with Claude 3.7 Sonnet + o1 ensembler). GitHub, 2025. https://github.com/augmentcode/augment-swebench-agent
18. Liu, J. (567 Labs), with Flaherty, C. "Why Grep Beat Embeddings in Our SWE-Bench Agent (Lessons from Augment)." Systematically Improving RAG talks, 2025. https://567-labs.github.io/systematically-improving-rag/talks/colin-rag-agents/
19. Pan, B., Baronio, C., Tam, A., Marsella, P., Jain, M., Chiu, D., Swyx, Alberti, S. (Cognition). "Introducing SWE-grep and SWE-grep-mini: RL for Multi-Turn, Fast Context Retrieval." Cognition Blog, October 16, 2025. https://cognition.ai/blog/swe-grep
20. Orosz, G., with Cherny, B. "Building Claude Code with Boris Cherny." The Pragmatic Engineer podcast, March 4, 2026 (transcript, 49:42-52:07). https://youtu.be/julbw1JuAz0
21. Rajasekaran, P., Dixon, E., Ryan, C., Hadfield, J. (Anthropic). "Effective context engineering for AI agents." Anthropic Engineering, September 29, 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
22. Anthropic. "How Claude Code works." Claude Code documentation, 2026. https://code.claude.com/docs/en/how-claude-code-works
23. Anthropic. "Tools reference" (Grep, Glob, LSP behaviour). Claude Code documentation, 2026. https://code.claude.com/docs/en/tools-reference
24. Anthropic. "Subagents" (built-in Explore subagent). Claude Code documentation, 2026. https://code.claude.com/docs/en/sub-agents
25. Anthropic. "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet." Anthropic Research, 2024-2025. https://www.anthropic.com/research/swe-bench-sonnet
26. Zilliz. "claude-context" MCP server and evaluation (grep-only vs. hybrid semantic search). GitHub, 2025. https://github.com/zilliztech/claude-context (evaluation: https://github.com/zilliztech/claude-context/tree/master/evaluation)
27. Gauthier, P. (Aider). "Building a better repository map with tree sitter." Aider Blog, October 22, 2023. https://aider.chat/2023/10/22/repomap.html
28. Aider. "Repository map." Aider documentation, 2024-2026. https://aider.chat/docs/repomap.html
29. Aider. "How aider scored SOTA 26.3% on SWE Bench Lite." Aider Blog, May 22, 2024. https://aider.chat/2024/05/22/swe-bench-lite.html
30. Aider-AI. `aider/repomap.py` (graph construction, edge weights, PageRank, token-budget search). GitHub, 2026. https://github.com/Aider-AI/aider/blob/main/aider/repomap.py
31. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024 (arXiv 2310.06770). https://arxiv.org/abs/2310.06770
32. Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., Press, O. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." NeurIPS 2024 (arXiv 2405.15793). https://arxiv.org/abs/2405.15793
33. Xia, C. S., Deng, Y., Dunn, S., Zhang, L. "Agentless: Demystifying LLM-based Software Engineering Agents." arXiv 2407.01489, 2024; code: https://github.com/OpenAutoCoder/Agentless . https://arxiv.org/abs/2407.01489
34. Chen, Z., Tang, X., Deng, G., et al. "LocAgent: Graph-Guided LLM Agents for Code Localization." ACL 2025 (arXiv 2503.09089); code: https://github.com/gersteinlab/LocAgent . https://arxiv.org/abs/2503.09089
35. Reddy, R. G., et al. (Salesforce AI Research). "SweRank: Software Issue Localization with Code Ranking." arXiv 2505.07849, 2025; code: https://github.com/SalesforceAIResearch/SweRank . https://arxiv.org/abs/2505.07849
36. Ouyang, S., et al. "RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph." ICLR 2025 (arXiv 2410.14684). https://arxiv.org/abs/2410.14684
37. Zhang, F., Chen, B., Zhang, Y., et al. (Microsoft). "RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation." EMNLP 2023 (arXiv 2303.12570); code: https://github.com/microsoft/CodeT/tree/main/RepoCoder . https://arxiv.org/abs/2303.12570
38. Wu, D., Ahmad, W. U., Zhang, D., Ramanathan, M. K., Ma, X. "Repoformer: Selective Retrieval for Repository-Level Code Completion." ICML 2024 (arXiv 2403.10059). https://arxiv.org/abs/2403.10059
39. Wang, Z. Z., Asai, A., Yu, X. V., Xu, F. F., Xie, Y., Neubig, G., Fried, D. "CodeRAG-Bench: Can Retrieval Augment Code Generation?" arXiv 2406.14497, 2024 (NAACL 2025 Findings); code: https://github.com/code-rag-bench/code-rag-bench . https://arxiv.org/abs/2406.14497
40. Li, X., et al. "CoIR: A Comprehensive Benchmark for Code Information Retrieval Models." ACL 2025 (arXiv 2407.02883); code: https://github.com/CoIR-team/coir . https://arxiv.org/abs/2407.02883
41. Suresh, T., Reddy, R. G., et al. "CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking." arXiv 2412.01007, 2024; code: https://github.com/gangiswag/cornstack . https://arxiv.org/abs/2412.01007
42. Voyage AI. "voyage-code-3: More Accurate Code Retrieval." Voyage AI Blog, December 4, 2024. https://blog.voyageai.com/2024/12/04/voyage-code-3/
43. Zhang, L., et al. (Microsoft). "SWE-bench Goes Live!" (SWE-bench-Live). arXiv 2505.23419, 2025. https://arxiv.org/abs/2505.23419
44. Örwall, A. "Moatless Tools" (context-insertion tools for LLM code editing; SWE-bench agent). GitHub, 2024-2025. https://github.com/aorwall/moatless-tools
