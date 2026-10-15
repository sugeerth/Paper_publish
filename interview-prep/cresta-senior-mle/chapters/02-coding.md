# 02. Coding: Reported Cresta Problems and Practical ML-Adjacent Python

## Format notes

- **Platform and setting.** Cresta runs its technical hiring on CoderPad, including an asynchronous CoderPad Screen step for engineering roles [11]. Candidates describe live rounds as about an hour on CoderPad and Zoom with a software engineer, with the camera on. The coding is live Python, and conceptual AI questions are mixed in [6].
- **MLE phone screen.** An August 2025 1point3acres report on the MLE (AI Agent) phone screen describes one problem in two parts: sequence decoding over a dictionary of next-token probabilities, first with greedy search and then with beam search, written in Python [1][2]. PracHub lists the same problem under two Cresta-tagged titles [4].
- **Loop shape.** A Glassdoor report for a Senior ML Engineer lists a recruiter call, a LeetCode-medium technical interview, a "vibe coding" interview (building something with an LLM coding tool) and a system design round [6]. SWE loops are described as one technical screen, three coding rounds and three behavioral rounds, mixing LeetCode-style and object-oriented problems [8]. A Senior/Staff loop is described as one hour of coding, one hour of system design and a hiring-manager chat [9].
- **Difficulty and bar.** A 2026 Toronto Senior SWE report describes a phone screen with three LeetCode-medium-level questions in 70 minutes [5]. One reviewer felt the interviewer wanted a specific approach, not just any working solution [5]. Other reported practical prompts include a rate limiter [6], a job dispatcher/executor system (SWE phone screen, 2026) [3], autocomplete over a word list, and turning a flat list into a tree [7]. FDE candidates report a 1.5-hour practical round where AI tools are allowed [10].
- **Senior-level emphasis (inferred).** Expect clean, testable Python, explicit complexity trade-offs, and extensions toward streaming and production concerns such as partial transcripts, latency and concurrency. These map to Cresta's real-time voice and agent stack [13][14][15] and its Senior MLE postings (Python/Go, PyTorch, Transformers, LLMs, model optimization) [16][17].

Tag legend: [Reported] means a source reports the question or a very close variant for Cresta (cited). [Likely] means it is inferred from the role, the team and Cresta's products; it is not reported.

## A. Sequence decoding (the reported MLE screen problem)

### A1. Core problem

1. [Reported] You are given a Python dict that maps each token to a dict of candidate next tokens and their probabilities. Starting from a start token, implement greedy decoding that produces the most likely sequence until an end token or a maximum length [1][4].
   - What do you do when a token has no outgoing edges, or when the graph has a cycle?
   - How do you break ties between equal probabilities so the output is deterministic?
   - What is the time complexity per step, and overall?
2. [Reported] Extend your solution to beam search with beam width k over the same next-token dictionary, and return the top sequence or the top-k sequences with their scores [1][4].
   - Why do you accumulate log-probabilities instead of multiplying probabilities?
   - How do you handle beams that reach the end token before the others do?
   - What is the complexity in terms of k, the vocabulary branching factor and the maximum length?
3. [Reported] Design the decoder as a reusable component (PracHub title: "Design sequence decoding with greedy and beam search"). What interface would you expose so that greedy and beam share code [4]?
   - How would you plug in a real model's next-token function instead of a static dict?
   - How would you unit-test the decoder?

### A2. Likely extensions of the decoding problem

4. [Likely] Add length normalization to beam search (for example, dividing by length^alpha). Show a case where the unnormalized beam prefers a short sequence.
   - Does normalization change which hypotheses get pruned mid-search, or only the final ranking?
5. [Likely] Implement top-k sampling and nucleus (top-p) sampling with temperature over the same structure.
   - How do you make the sampler reproducible with seeds in tests?
   - What happens when p is so small that only one token survives?
6. [Likely] Add a constraint so that a banned set of tokens (for example, PII placeholders or forbidden phrases) never appears in decoded output.
   - How would you enforce "no repeated n-gram of size n"?
7. [Likely] Return the k best complete sequences without duplicates when different paths produce the same surface string.
8. [Likely] Implement early stopping: terminate beam search once no live beam can beat the best finished hypothesis.
   - Under what scoring assumption is this stopping rule correct?
9. [Likely] Batch the decoder so it decodes many prompts in parallel with a vectorized next-token function, using NumPy.
   - How do you handle sequences in the batch that finish at different steps (masking or padding)?
10. [Likely] Implement constrained decoding to one of a fixed set of allowed outputs (for example, intent labels) using a trie of allowed token sequences.
11. [Likely] Make the decoder stream tokens to a caller as a generator while beam search is still running.
    - Can you safely emit tokens before the beam has converged? Which prefix is stable?

## B. Streaming text and transcripts

12. [Likely] You receive ASR results as a stream of events `(utterance_id, is_final, text, start_ms, end_ms)`. Partial transcripts can be revised until a final arrives. Maintain the current best transcript of the conversation. (Grounded in Cresta's partial-vs-final transcript pipeline [13][14].)
    - How do you handle a final that arrives after a newer partial for the same utterance?
    - What is your memory footprint for a 2-hour call?
13. [Likely] Given the transcript stream, fire a callback only when a partial transcript is "stable", for example when a prefix has not changed across N consecutive partials.
    - How do you avoid re-firing for the same stable prefix?
14. [Likely] Merge two ordered streams of utterances (agent channel and customer channel) into a single time-ordered conversation, interleaving by start time.
    - How would you generalize this to k speakers? What is the complexity with a heap?
    - How do you handle overlapping speech?
15. [Likely] Implement a sliding-window keyword detector over a stream of tokens: emit an event when any of M keyword phrases appears. (Cresta runs cheap keyword detection on partials [14].)
    - How does Aho-Corasick compare to naive matching as M grows?
    - What if keywords can span two transcript chunks?
16. [Likely] Implement a rolling context window that keeps the last N tokens (or last T seconds) of a conversation to feed an LLM prompt.
    - How do you always keep the system prompt and the first customer statement?
17. [Likely] Write a function that deduplicates overlapping chunks of transcript text from a streaming source (chunk i+1 may repeat the tail of chunk i).
    - What is the complexity of your overlap detection? Can you use KMP or rolling hashes?
18. [Likely] Compute talk-time, silence and interruption statistics per speaker from a list of timestamped segments.
    - How do you define and count an interruption?
    - How do you handle segments with missing end times?
19. [Likely] Implement real-time redaction of credit-card numbers and SSNs in a token stream where a number can be split across chunks. (Cresta's voice stack includes audio and transcript redaction [13].)
    - How much lookahead buffer do you need, and what does it cost in latency?
    - How would you validate card numbers (Luhn) without false positives on phone numbers?
20. [Likely] Given word-level ASR timestamps and a list of detected "moments" (character spans), map each moment back to audio start and end times.
21. [Likely] Implement a debouncer/throttler so that a downstream model is called at most once every X ms per conversation while still processing the latest partial.
22. [Likely] Implement an online running mean and variance (Welford) of per-turn sentiment scores per conversation, and flag turns that deviate by more than k standard deviations.

## C. Practical systems coding (reported practical prompts and close variants)

23. [Reported] Implement a rate limiter in your language of choice [6].
    - Token bucket versus sliding-window log versus fixed window: which did you pick and why?
    - How would you make it thread-safe?
    - How do you make it per-customer (tenant) and per-endpoint?
24. [Reported] Implement a job dispatcher/executor system: a dispatcher accepts jobs and hands them to a pool of executors (reported for a 2026 SWE phone screen) [3].
    - How do you handle job failures and retries with backoff?
    - How do you support priorities, or cancellation of a queued job?
    - How do you guarantee at-least-once versus exactly-once execution?
25. [Reported] Implement an API client exercise: call a public API (reported as a Wikipedia API task in an SWE phone screen) and process the results [2].
    - How do you handle pagination, rate limits and transient errors?
    - How would you parallelize requests and cap concurrency?
26. [Likely] Implement an inference request batcher (dynamic batching): collect requests until either batch size B or wait time T, then call the model once and route results back to callers.
    - How do you choose B and T against a latency SLO? (Cresta's voice-agent target is about 1.5 s end to end [15].)
    - What happens when the model call raises for one item in the batch?
27. [Likely] Implement an LRU cache with TTL for embedding or LLM responses keyed by normalized prompt.
    - How do you make get and put O(1)?
    - How would you add a maximum size in bytes instead of in entries?
28. [Likely] Write a retry decorator with exponential backoff, jitter and a timeout, usable on both sync and async functions.
29. [Likely] Implement a simple inference-graph executor: given a DAG of model "nodes" with dependencies, run them in topological order and run independent nodes concurrently. (Cresta describes inference graphs that orchestrate ML services [14].)
    - How do you detect cycles?
    - How do you skip nodes whose conditions are not met, for example models that run only on final transcripts?
30. [Likely] Implement a circuit breaker around an external LLM or ASR provider call.
    - What state transitions exist, and how do you test them deterministically?
31. [Likely] Write an asyncio producer/consumer pipeline: ingest transcript events, run a classifier and write annotations, with bounded queues for backpressure.
    - What happens when the consumer is slower than the producer?
32. [Likely] Implement a top-K heavy-hitters tracker (for example, most frequent customer intents in the last hour) over a stream.
    - Exact (heap plus counter with time buckets) versus approximate (Count-Min Sketch): when would you choose each?
33. [Likely] Implement consistent hashing to route conversations to workers so that a conversation's events always land on the same worker.
    - What happens when a worker is added or removed?

## D. Data structures and LeetCode-medium style (reported bar: medium)

34. [Reported] Expect LeetCode-medium algorithm problems in the technical/phone screen, sometimes three in 70 minutes [5][6].
    - Can you state complexity before coding and test edge cases out loud?
35. [Reported] Implement autocomplete: given a list of words and a prefix, return matching suggestions (reported for a SWE role) [7].
    - How do you return the top-k by frequency instead of alphabetically?
    - How do you support updates to frequencies as users type?
    - Implement the trie yourself with insert, prefix search and delete; how would you extend it to wildcard queries?
36. [Reported] Convert a flat list of records with parent references into a tree (reported for a SWE role) [7].
    - How do you detect orphans and cycles?
    - Can you do it in a single pass?
37. [Reported] A matrix or grid problem disguised as a puzzle game appeared in an online coding challenge (SWE) [7].
    - BFS or DFS: which and why? What is the memory cost?
38. [Likely] Given a conversation as a list of turns, find the longest contiguous segment where the customer's sentiment never drops below a threshold.
39. [Likely] Merge overlapping time intervals (speaker segments), then compute the total non-overlapping speech duration.
40. [Likely] Find all pairs of conversations whose embedding cosine similarity exceeds a threshold, efficiently for N = 100k.
    - What changes if you use approximate nearest-neighbor search?
41. [Likely] Implement word-level edit distance and use it to compute word error rate (WER) between a reference and an ASR hypothesis.
    - How would you return the alignment (substitutions, insertions, deletions)?
    - How do you normalize text (casing, punctuation, numbers) before scoring?
42. [Likely] Longest common subsequence between two transcripts of the same call from two different ASR vendors.
43. [Likely] Group near-duplicate customer utterances using MinHash/LSH or normalized string keys.
44. [Likely] Given a log of `(conversation_id, timestamp, event)` events, find conversations where event B happened within T seconds after event A.
45. [Likely] Design a data structure supporting `add(word)` and `most_frequent_in_last(seconds)` over a stream.
46. [Likely] Validate balanced spans: given nested annotation spans over text (entities inside sentences), check that they are properly nested.
47. [Likely] Tokenize text with a given byte-pair-encoding (BPE) merge table: apply the merges in order to a word.
    - What is the complexity of your naive merge loop, and how could you speed it up?

## E. Object-oriented and low-level design coding

48. [Reported] Expect at least one object-oriented coding problem alongside LeetCode-style rounds (reported for SWE loops) [8].
    - How do you decide class boundaries, and what would you make an interface or abstract base class?
49. [Likely] Design the classes for a conversation-annotation system: Conversation, Turn, Annotation (moment versus action), and a registry of detectors that run on new turns. (Moment/action annotations come from [14].)
    - How do you add a new detector without modifying existing code?
50. [Likely] Implement a minimal tool-calling agent loop: given an LLM function that returns either text or a tool call, execute tools from a registry and feed results back until a final answer or a step limit.
    - How do you validate tool arguments against a schema?
    - How do you prevent infinite loops or repeated identical tool calls?
51. [Likely] Implement a conversation state machine for a booking or rebooking flow (states, transitions, guards) that an AI agent follows. (Cresta's agent material uses flight rebooking as an example scenario [12].)
    - How do you persist and resume state mid-conversation?
52. [Likely] Implement a plugin-style policy engine: rules such as "if intent X is detected and the agent has not said Y within 60 s, emit hint Z".
    - How do you represent time-based conditions efficiently across thousands of live calls?
53. [Likely] Implement a prompt template class with variable validation, truncation to a token budget and escaping of user content.
    - How do you guard against prompt injection from transcript text?

## F. ML-from-scratch coding in Python/NumPy

54. [Likely] Implement logistic regression training with batch gradient descent in NumPy, then predict probabilities. (A third-party guide lists from-scratch ML algorithms as screen material for Cresta ML roles; it is not verified [12].)
    - How do you add L2 regularization and verify the gradient numerically?
    - How do you make it numerically stable for large logits?
55. [Likely] Implement numerically stable softmax and log-softmax, and cross-entropy loss.
56. [Likely] Implement k-means with k-means++ initialization [12].
    - How do you detect convergence, and what do you do about empty clusters?
57. [Likely] Implement precision, recall, F1 and a confusion matrix for a multi-class intent classifier from lists of labels.
    - Macro versus micro versus weighted averaging: when does each mislead?
58. [Likely] Compute ROC-AUC from scores and labels without sklearn.
    - How do you handle tied scores?
59. [Likely] Choose a decision threshold for a "send hint" classifier that maximizes recall subject to precision >= P, given scores and labels.
60. [Likely] Implement scaled dot-product attention (with a causal mask) in NumPy or PyTorch.
    - What shapes do Q, K and V have with batching and multiple heads?
    - Why scale by sqrt(d_k)?
61. [Likely] Implement a key-value cache for autoregressive decoding and show how it changes the per-step cost.
62. [Likely] Implement cosine-similarity top-k retrieval over a matrix of embeddings for a RAG knowledge-assist feature, in vectorized NumPy.
    - How do you handle a corpus that does not fit in memory?
63. [Likely] Implement a text chunker for RAG that splits documents into overlapping chunks on sentence boundaries under a token limit.
64. [Likely] Implement BM25 scoring over a small corpus and combine it with embedding scores (hybrid retrieval).
    - How do you normalize the two score distributions before combining?
65. [Likely] Implement stratified train/validation/test splitting by conversation, so that turns from one conversation never leak across splits.
66. [Likely] Implement a simple bootstrap confidence interval for the difference in an evaluation metric between two models.
67. [Likely] Write the training loop for a PyTorch text classifier: data loader, padding/collation, forward, loss, backward, gradient clipping and evaluation.
    - Where do you put `model.eval()` and `torch.no_grad()`, and why?
    - How would you add mixed precision?
68. [Likely] Implement Expected Calibration Error (ECE) for classifier probabilities.
69. [Likely] Implement an LLM-as-judge evaluation harness: run prompts against N test cases, parse structured verdicts, retry on parse failure and aggregate scores with per-category breakdowns. (Cresta's Senior MLE quality track focuses on evaluation frameworks and failure modes such as hallucination and tool misuse [17].)
    - How do you detect and handle judge position bias?

## G. Data processing and text normalization

70. [Likely] Parse a JSONL file of conversations (possibly malformed lines) into turn-level training examples with labels, and report the counts of bad rows.
    - How do you process a 50 GB file without loading it into memory?
71. [Likely] Normalize ASR text for downstream NLP: number words to digits, collapse fillers ("um", "uh"), fix casing.
    - Which normalization steps are reversible, and why might you need reversibility?
72. [Likely] Given turn-level labels from multiple annotators, compute majority labels and Cohen's/Fleiss' kappa.
73. [Likely] Implement a group-by aggregation without pandas: average handle time per agent per day from a list of call records.
    - Then do it in pandas. Which operations dominate cost at scale?
74. [Likely] SQL: given `conversations` and `annotations` tables, find agents whose hint-acceptance rate dropped week-over-week. (SQL joins appeared in a Cresta online challenge [7].)
75. [Likely] Write a function to detect language-switching turns in a bilingual conversation using per-token language-ID scores and a smoothing window.
76. [Likely] Implement reservoir sampling to pick a uniform sample of k conversations from a stream of unknown length.
    - How do you make it weighted?
77. [Likely] Implement a sessionizer: from raw event logs with gaps, split events into sessions where a gap of more than G minutes starts a new session.

## H. AI-assisted ("vibe") coding round

78. [Reported] Use an LLM coding tool to build a small working application within the interview time (reported for a Senior ML Engineer loop) [6]. FDE candidates report a similar 1.5-hour practical where AI is allowed [10].
    - How do you verify that LLM-generated code is correct? What tests do you write first?
    - When do you stop prompting and write the code yourself?
    - How do you explain the design choices the tool made?
79. [Likely] Build a small conversation summarizer service end to end with an LLM API: input a transcript, output a structured summary (reason for call, resolution, follow-ups) validated against a JSON schema.
    - How do you handle transcripts that exceed the context window?
80. [Likely] Build a toy customer-service agent for a scenario such as order status or flight rebooking with two tools (lookup and update), including guardrails.
    - How would you write automated tests for multi-turn behavior?
81. [Likely] Given an existing buggy repository (for example, a classifier service), use AI assistance to find and fix the bug, and add a regression test.

## I. Concurrency, performance and production hygiene

82. [Likely] Make a Python preprocessing step that is too slow (pure-Python loops over tokens) 10x faster.
    - When do you vectorize, when do you use multiprocessing, and when does the GIL matter?
83. [Likely] Implement a thread-safe counter and per-key metrics registry (counts, latency histograms with percentiles).
    - How do you compute p50/p95/p99 over a stream without storing every value?
84. [Likely] Write an async function that calls three model services concurrently with an overall deadline and returns partial results if some time out.
85. [Likely] Write a timing harness that measures time-to-first-token and tokens/sec for a streaming LLM client.
86. [Likely] Implement graceful shutdown for a streaming worker: drain in-flight conversations, flush state and stop accepting new work.
87. [Likely] Given Go-like requirements (the Senior MLE role lists Python and Go [16]), sketch how you would implement your rate limiter or batcher with goroutines and channels.
    - What would change compared with your Python asyncio version?
88. [Likely] Add structured logging and tracing IDs that follow a conversation across ASR, orchestrator and model calls in your pipeline code.
89. [Likely] Write unit tests for the streaming transcript aggregator from question 12, including out-of-order and duplicate events.
    - How would you property-test it?

## References

1. "Cresta MLE面经" / "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam search decoding). 1point3acres, 2025. https://www.1point3acres.com/interview/thread/1140481 (also https://www.1point3acres.com/bbs/thread-1140481-1-1.html)
2. "cresta interview questions (2026)": company page with Cresta SWE/MLE phone screen and online test reports, including an API-implementation phone screen. 1point3acres, 2024-2026. https://www.1point3acres.com/interview/company/cresta
3. "Cresta Fulltime Software Engineer Tech Phone Screen: Job Dispatcher-Executor System Implementation" (listing seen in the site's related-threads sidebar; full text not accessed). 1point3acres, 2026. https://www.1point3acres.com/interview/thread/1172433
4. "Cresta Interview Questions": Cresta-tagged "Design sequence decoding with greedy and beam search" and "Implement greedy and beam decoding". PracHub, 2025. https://prachub.com/companies/cresta
5. "Cresta (CA) Senior Software Engineer Interview Questions". Glassdoor, 2026. https://www.glassdoor.ca/Interview/Cresta-CA-Senior-Software-Engineer-Interview-Questions-EI_IE3180909.0,9_KO10,34.htm
6. "Cresta (CA) Interview Questions & Answers" (CoderPad+Zoom Python round, rate limiter, Senior ML Engineer loop with vibe-coding round). Glassdoor, 2024-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
7. "Cresta (CA) Software Engineer Interview Questions" (autocomplete, list-to-tree, DS/algo plus SQL joins challenge). Glassdoor, 2024-2026. https://www.glassdoor.sg/Interview/Cresta-CA-Software-Engineer-Interview-Questions-EI_IE3180909.0,9_KO10,27.htm
8. "Cresta interview process". Blind (teamblind.com). https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
9. "Cresta senior backend SWE interview". Blind (teamblind.com). https://www.teamblind.com/post/cresta-senior-backend-swe-interview-6v1tmjld
10. "Interview for forward deployment engineer at Cresta". Blind (teamblind.com). https://www.teamblind.com/post/interview-for-forward-deployment-engineer-at-cresta-ngswbj44
11. "How Cresta Consolidated All Technical Hiring on CoderPad to Scale Eng Hiring". CoderPad case study. https://coderpad.io/case-study/how-cresta-consolidated-all-technical-hiring-on-coderpad-to-scale-eng-hiring/
12. "Cresta Interview Questions" (third-party guide; generic, unverified). Scoutify, 2026. https://scoutify.com/companies/cresta/interview/ ; Cresta's agent-design example (flight rebooking): "Cresta Conductor: The Agent for AI Agent Development". Cresta blog. https://cresta.com/blog/cresta-conductor-the-agent-for-ai-agent-development
13. "Understanding Cresta's Voice Platform: The Voice Stack". Cresta blog, 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
14. "Understanding Cresta's Voice Platform: ML Services, Inference Graphs, and Real-Time Intelligence". Cresta blog, 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
15. "Engineering for Real-Time Voice Agent Latency". Cresta blog. https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
16. "Senior Machine Learning Engineer (Chat Agent)". Cresta job posting, Lever. https://jobs.lever.co/cresta/172fe276-9415-4bae-9b0b-af72cf7cf014
17. "Senior Machine Learning Engineer" (agentic assistance and agent-quality tracks). Cresta posting via ZeroG Talent. https://zerogtalent.com/ai-jobs/cresta/senior-machine-learning-engineer-4249943008
