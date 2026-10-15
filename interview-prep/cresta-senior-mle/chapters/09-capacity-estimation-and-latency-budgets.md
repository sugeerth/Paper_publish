# 09. Capacity Estimation and Latency Budgets: Back-of-Envelope Numbers for Real-Time Voice, LLM Serving, Knowledge Retrieval and Conversation Storage

> **Tagging system.**
> - **[Reported] [n]**: a candidate report or question bank attributes this question, or a close variant, to Cresta. [n] points to the References list.
> - **[Likely]**: inferred from Cresta's products, engineering blog and ML job postings. No candidate has reported it.
>
> **How to use this chapter.** Every question asks you to produce a number, a budget or a trade-off. Where a question supplies assumptions ("Assume ..."), they are inputs for you to calculate from. They are not Cresta's real figures, and they are not answers. Say your assumptions out loud, round hard (powers of ten, 1 day is about 10^5 seconds), and check each result against a sanity bound before moving on.
>
> **Diagram prompts.** Some questions include a "Sketch" follow-up. It names the latency timeline or capacity diagram to draw on the whiteboard, without giving the numbers.
>
> **Research limits.** In the October 2026 review pass, every cited domain was again unreachable from this environment (DNS `ENOTFOUND` for cresta.com, fireworks.ai and prachub.com in WebFetch; the egress proxy answered 403 for cresta.com, prachub.com and job-boards.greenhouse.io), and the shared web-search budget was already used up. No page could be opened and no new candidate report could be found, so the 2024-2026 search for new reported questions came back empty. Instead, each [Reported] tag was checked against the excerpt log in chapter 10 (Part 2, tables R1-R7 and C1-C6). Both original [Reported] items match a logged excerpt and keep their tags. The new [Reported] items in section A scale up prompts logged there (a rate limiter, a worker pool, a job dispatcher, RAG and decoding). In every case the base prompt is reported and the sizing probes are inferred. No candidate has reported a capacity-estimation question for Cresta.

## Format notes for this round

- **No separate estimation round is reported.** Capacity and latency estimates are follow-ups inside the roughly one-hour system design round. A 2025-26 Senior MLE report lists that round after a recruiter call, an LC-medium round and a "vibe coding" round [1]. Senior/Staff reports describe one hour of system design next to one hour of coding [2].
- **Reported design prompts that invite sizing.** "How would you deploy a model in production?" (Senior MLE) [1], and "Design an evaluation platform for AI agents across prompt and model changes" [4][5]. Both lead naturally to fleet-size, throughput and cost-per-run questions. The PracHub experience post for the second prompt is labeled as a SWE system design round, not an MLE one [5].
- **Reported coding and screen prompts that a design round can scale up.** A rate limiter and a worker pool with round-robin assignment [1], a job dispatcher and executor (SWE, 2026, title only) [24], calling a public API [25], "What is RAG, and how do you prevent hallucination?" (FDE screen) [23], and step-by-step sequence generation (MLE, AI Agent) [22]. Section A turns each into a sizing exercise. The base prompts are reported, and the numeric follow-ups are inferred.
- **Interviewer (inferred).** Most likely a senior or staff MLE from Agentic Assist, AI Agent, Voice/ASR or Insights [6][7][8][9][10].
- **Senior bar (from postings).** Postings ask candidates to design for latency, cost efficiency, scalability, security and maintainability, and the Voice posting names real-time inference and streaming pipelines [6][7][8]. Postings also list latency and cost next to accuracy and faithfulness as quality metrics [7]. Recruiters have flagged LLM, RAG and agent fundamentals as in scope, so expect serving-cost and latency questions about LLMs specifically [3]. At senior level, expect to be pushed past "it depends" to a concrete number, a stated bottleneck and the lever you would pull first.
- **Product facts you can anchor on.** Cresta publishes both partial and final transcripts, about once per second [12]. The voice stack includes audio and transcript redaction [12]. An inference graph runs several models per conversation [13]. Cresta moved Knowledge Assist, summarization and chat suggestions to fine-tuned models served with multi-LoRA [14][15][16].

## Questions

### A. Reported prompts, with sizing probes

1. How would you deploy a model in production? [Reported] [1]
    - For a real-time assist model, what is the peak QPS you would size for, and how did you derive it from concurrent conversations?
    - How many GPU replicas do you need at peak, and how much headroom do you keep for canary plus rollback running side by side?
    - What does shadow traffic cost in GPU-hours if you mirror 100 percent of traffic for one week?
    - Which latency percentile do you put in the rollout gate, and how many requests do you need before p95 or p99 is stable?
2. Design an evaluation platform for AI agents that catches regressions across prompt and model changes. [Reported] [4][5]
    - Assume 2,000 scenarios, 5 reruns each, 12 simulated turns per scenario and 3 judge calls per turn. How many LLM calls does one full run make, and how many tokens?
    - At your assumed price per million tokens, what does one run cost? How many runs per day can the team afford?
    - How long does a run take at a given rate limit or GPU budget? What parallelism do you need to finish under 30 minutes for a CI gate?
    - Which cut (fewer reruns, smaller judge, sampled scenarios, cached unchanged turns) saves the most cost for the least loss in detection power?
    - For voice-agent scenarios that run simulated audio through TTS, then ASR, then the agent, how much do cost and wall-clock time per scenario grow compared with text-only simulation [21]?
3. Implement a rate limiter, then scale it to guard calls from real-time features to a shared LLM fleet or a third-party API. [Reported] [1] (coding round; sizing probes inferred)
    - With 500 tenants and 10 features per tenant, each with its own token bucket, how much state does the limiter hold, and can it live in one Redis shard?
    - At 50,000 limiter checks per second at peak, what does a network round-trip per check add to hint latency? When would you switch to local buckets with periodic sync?
    - Do you limit on requests per second or on tokens per minute for an LLM backend, and why does the choice matter for long prompts?
4. Implement a worker pool with an orchestrator that assigns tasks round-robin, then use it for post-call jobs. [Reported] [1] (FDE AI Agent coding round; sizing probes inferred)
    - Assume 1 million calls a day, and post-call jobs (summary, QA, tagging) that take 2-20 seconds of GPU time each. How many workers do you need at steady state and at the evening peak?
    - Why does round-robin break down when job durations vary 10x, and how much does least-loaded or work-stealing assignment cut p95 completion time?
5. Design a job dispatcher and executor, then size it for re-scoring every conversation of a tenant after a scorecard change. [Reported] [24] (SWE phone-screen title, 2026; sizing probes inferred)
    - A tenant with 50 million historical conversations changes one QA criterion. How long does a full backfill take on your batch fleet, and how do you keep it from starving daily jobs?
    - What checkpointing granularity bounds the work lost when an executor dies mid-batch?
6. What is RAG, and how do you prevent hallucination? Then put numbers on the RAG step inside a live call. [Reported] [23] (FDE technical screen, 2026; numeric probes inferred)
    - How many retrieved chunks of 300 tokens can you add to a hint prompt before prefill pushes p95 past 300 ms?
    - What does one extra groundedness-check model call add in latency and cost per hint, and would you run it on every hint or on a sample?
7. Walk through how an AI agent generates a sequence step by step, then estimate per-token decode latency from first principles. [Reported] [22] (MLE AI Agent phone-screen follow-up; numeric probes inferred)
    - For an 8B model in 16-bit weights on a GPU with about 3 TB/s of memory bandwidth, what is the lower bound on time per output token at batch 1? How far above that bound does a real server run?
    - How does that bound change at batch 32, and at what batch size does decode stop being memory-bound?
8. Call a public API and process the results, then size the integration for a voice agent's tool calls. [Reported] [25] (SWE phone screen; sizing probes inferred)
    - If 2,000 concurrent AI Agent calls each make one CRM lookup every 30 seconds, what QPS does the customer's backend see, and what happens when its rate limit is 20 QPS?
    - How large a response cache, with what TTL, would keep you under that limit without serving stale account data?

### B. Traffic model: tenants, concurrency and peaks

9. Estimate the peak number of concurrent calls for one large enterprise contact-center tenant. [Likely]
    - Assume 5,000 agents, 70 percent occupancy at peak, and an average handle time of 6 minutes. How many calls are live at the peak hour?
    - How many new calls per second start at peak? How did you use Little's Law?
    - What is the peak-to-average ratio across a day, and across a week (Monday morning, billing cycle, outage day)?
10. Estimate the platform-wide peak concurrency across all tenants. [Likely]
    - Assume a few hundred tenants with a long-tail size distribution. How much do time zones flatten the global peak?
    - Which tenants dominate, and how does that change capacity planning compared with a uniform distribution?
11. How does concurrency change when the AI Agent (autonomous voice agent) handles calls instead of human agents? [Likely] [11]
    - Without an agent-headcount ceiling, what limits concurrency: trunk capacity, the customer's queue or your GPU fleet?
    - How would you size for a marketing campaign that triples inbound volume for two hours?
12. Estimate chat (digital) concurrency for the same tenant when one human agent handles three chats at once. [Likely]
    - How does the request pattern for suggestions differ between chat (one per message) and voice (one per partial)?
13. Convert concurrency into model QPS for each feature in the inference graph. [Likely] [13]
    - If partial transcripts arrive about once per second per call [12], what is the raw trigger rate for 10,000 concurrent calls?
    - Which models run on every partial, which on every final utterance, and which once per call? What QPS does each tier see?
    - How much does debouncing (at most one call per conversation every X ms) cut QPS, and what latency does it add?
14. How would you build a capacity model the team can update each quarter? [Likely]
    - Which inputs (agents, occupancy, handle time, features enabled per tenant) drive the GPU forecast?
    - How do you check the model against real telemetry after launch?
15. Estimate the start-of-shift burst for a tenant whose 3,000 agents log in within 10 minutes. [Likely]
    - How many new streaming sessions per second does that create (WebSocket connects, ASR session setup, adapter and prompt-cache warm-up), and which component fails first?
16. How many hints per minute can a human agent actually absorb, and how does that cap the useful QPS of the hint service? [Likely] [13]
    - If agents act on at most 3-4 hints per minute, how much of the trigger rate from question 13 is wasted compute, and where would you filter it?

### C. Audio ingest and streaming ASR

17. Estimate the audio frames per second entering streaming ASR at platform peak. [Likely] [12]
    - Assume telephony audio at 8 kHz, 16-bit PCM after decoding, 20 ms frames, and two channels (agent and customer) per call. How many frames per second per call, and in total for 10,000 calls?
    - What is the ingest bandwidth in Mbit/s, before and after compression (G.711 at 64 kbit/s per channel compared with Opus)?
    - Does resampling to 16 kHz for the ASR model change your network numbers or only your compute numbers?
18. How many audio streams can one GPU serve with a streaming ASR model? [Likely] [9]
    - Assume a real-time factor of 0.02 per stream at batch 1. Why does that not mean 50 streams per GPU in practice?
    - How does chunked (cache-aware) streaming change the batching math compared with offline transcription?
    - How do you size the ASR fleet for 20,000 concurrent channels with N+1 redundancy per zone?
19. Do you transcribe both channels, or only the customer channel for some features? [Likely]
    - What fraction of the ASR fleet does that save, and which downstream features break?
20. What is the cost per call-minute of streaming ASR on your own GPUs compared with a third-party API? [Likely] [9]
    - At what monthly volume does self-hosting break even, once you count on-call and model-maintenance engineers?
21. Estimate the ASR finalization delay: from the end of speech to a final transcript. [Likely] [11][12]
    - How do the voice-activity-detection silence threshold, the chunk size and the right-context lookahead each add to it?
    - What happens to word error rate if you shorten the endpointing silence from 700 ms to 300 ms? How would you measure it?
22. How much memory does a streaming ASR session hold (encoder cache, decoder state, audio buffer)? [Likely]
    - How many sessions fit in one GPU's memory, and is memory or compute the binding limit?
23. How would you handle a reconnect or a pod restart mid-call without losing more than one second of transcript? [Likely] [12]
    - How much audio do you buffer upstream, and what does that buffer cost in memory at peak?
24. Size the transcript event bus between ASR and the downstream models. [Likely] [12]
    - At 20,000 channels emitting a partial about once per second plus finals, how many messages and MB per second does the bus carry? How many partitions do you need, and what do you key them on to keep each call in order?
    - What retention do you set so a downstream model can replay the last 60 seconds after a crash?
25. What does adding diarization to single-channel (mono) recordings cost, compared with stereo telephony where each speaker has a channel? [Likely] [9]
    - How much extra GPU time per call-minute, and what diarization error rate would make QA scoring unreliable?
26. How do the media-server jitter buffer and codec framing add to latency before audio even reaches ASR? [Likely] [11]
    - Put a number on each (packetization, jitter buffer, transcoding), and say which ones you can shrink without more packet-loss artifacts.

### D. LLM hint generation for real-time Agent Assist

27. Size the GPU fleet for LLM hint generation at p95 under about 300 ms. [Likely] [13][14]
    - Assume 10,000 concurrent calls, one hint request per final customer utterance (about one every 8 seconds per call), 1,500 prompt tokens and 40 output tokens. What is the peak request rate and token rate?
    - What prefill and decode throughput per GPU do you assume for a 7-8B model, and where does each number come from?
    - How many GPUs does that give you at 50 percent target utilization? Why not 90 percent?
    - What batch size keeps p95 under 300 ms, and how does that batch size cap per-GPU throughput?
28. Break the 300 ms hint budget into stages. [Likely] [13]
    - Sketch: a timeline from "final utterance emitted" through the transcript bus, trigger/moment detection, retrieval, prompt assembly, queue wait, prefill, decode, post-filter and UI push. Which stages are fixed and which grow with load?
    - How much of the budget does queueing take at 50 percent compared with 80 percent utilization?
29. How does model size change the GPU count for the same SLO? [Likely] [14][19]
    - Compare a 70B general model, a fine-tuned 7-8B model and a distilled 1-3B model on GPUs needed, p95 and quality risk.
    - What evidence would convince you a smaller fine-tuned model is good enough? (Cresta reports fine-tuned models matching a larger general model on its tasks [14][15].)
30. How many per-tenant LoRA adapters can one GPU hold, and what does multi-LoRA serving do to throughput? [Likely] [16]
    - What is the memory footprint of one rank-16 adapter on a 7-8B model compared with the base weights?
    - What happens to p95 when a request needs an adapter that is not loaded? How often can you afford that?
    - When is a per-tenant full fine-tune worth the extra GPUs?
31. Would you trigger hint generation on partial transcripts to save latency? [Likely] [12]
    - How many extra LLM calls does that create, and how many of them are wasted when the partial changes?
    - What hit rate on "partial equals final" would make speculative triggering pay for itself?
32. How do you set the autoscaling policy for the hint service? [Likely]
    - What metric do you scale on (queue depth, in-flight tokens, p95), and why not GPU utilization?
    - How long does a new GPU replica take to become ready (node, image, weights load), and how much warm headroom does that force you to keep?
33. What is the cost per agent-hour of real-time assist? [Likely]
    - Using your GPU count and an assumed GPU-hour price, what does one live agent cost per hour, and is that defensible against a contact-center seat price?
34. How would you degrade gracefully when GPU capacity is short during a spike? [Likely]
    - Which features do you shed first, and what does each shed save in GPU-seconds?
    - Would you serve a smaller fallback model, lengthen the debounce, or drop hints for low-value moments?
35. Estimate prefill time for a 1,500-token hint prompt from FLOPs. [Likely]
    - Using about 2 x parameters x tokens FLOPs, an 8B model, and a GPU at an assumed peak of about 1,000 dense TFLOPS with 40 percent utilization, how long does prefill take? How does that compare with your 300 ms budget?
    - At what prompt length does prefill, not decode, dominate the hint latency?
36. Apply Little's Law to the hint service: how many requests are in flight at peak, and how much KV-cache memory do they need? [Likely]
    - With your request rate from question 27 and a mean service time of 200 ms, how many sequences are in flight across the fleet, and how many per GPU?
37. How long does a new LLM replica take to become ready, and what warm pool does that force? [Likely]
    - Assume 16 GB of weights pulled from object storage at 1-2 GB/s, plus container start and CUDA graph capture. How many minutes until the replica serves traffic, and how much headroom covers a 3x spike that arrives faster than that?

### E. Voice-agent turn latency (AI Agent)

38. Draw the latency breakdown of one voice-agent turn, from the customer's last word to the first audio the customer hears. [Likely] [11]
    - Sketch: endpointing/ASR finalization, network hop, orchestration, retrieval or tool call, LLM time to first token, the first sentence chunk, TTS time to first audio, and playout through telephony. Put a number on each.
    - What total target would you set (for example under 800 ms at p50 and under 1.5 s at p95), and why those percentiles?
39. Which single stage would you attack first to cut 200 ms, and how? [Likely] [11]
    - What does each option cost in quality: a shorter endpointing threshold, a smaller LLM, speculative decoding, streaming TTS on partial sentences?
40. How much LLM time to first token can you afford, and what drives it? [Likely]
    - How does TTFT scale with prompt length on a 7-8B model, and with queue depth?
    - How much does prefix caching of the system prompt, tool schemas and conversation history save per turn?
41. How does speculative decoding change latency and GPU count for the voice agent? [Likely]
    - What draft-model acceptance rate do you need for a real speed-up, and how does it vary with temperature and domain?
    - Does speculative decoding help TTFT, time per output token, or both? Why does that matter for voice when TTS starts on the first sentence?
    - With per-token acceptance rate a and draft length k, the expected tokens per target step is (1 - a^(k+1)) / (1 - a). Compute it for a = 0.6 and 0.8 at k = 4, then subtract the draft model's cost. Is it still a win at batch 32?
42. How do you start TTS before the LLM finishes? [Likely] [11]
    - At what chunk boundary (clause, sentence, N tokens) do you hand text to TTS, and what does a bad boundary do to prosody?
    - How many output tokens per second must the LLM sustain so TTS never starves?
43. What does a tool call (for example a CRM lookup) do to the turn budget? [Likely] [11]
    - If the backend API has a p95 of 600 ms, how do you hide it: filler speech, parallel prefetch, or predicting the call from the partial transcript?
    - How many speculative prefetches per call can you afford, and what do wasted ones cost?
44. How do you handle barge-in (the customer interrupts) within the budget? [Likely]
    - How fast must you detect speech and stop TTS playout, and which component owns that timer?
    - What happens to in-flight LLM tokens and the KV cache when the turn is cancelled?
45. Estimate GPU and TTS capacity for 2,000 concurrent voice-agent calls. [Likely]
    - Assume the agent speaks 40 percent of the time and each LLM turn uses 3,000 prompt and 60 output tokens every 10 seconds. What are the token rates and GPU counts?
    - How many TTS streams fit per GPU at real-time factor r, and does TTS or the LLM dominate cost?
46. How do you measure end-to-end turn latency on real calls? [Likely] [8][11]
    - Server timestamps compared with measuring silence gaps in the stereo recording: which do you trust, and why do they disagree?
    - Which percentile do customers feel, and how do you report it per tenant?
47. How do region and network topology affect the budget? [Likely]
    - What round-trip time do you add if the telephony edge, ASR and LLM sit in different regions? When would you co-locate them?
48. A safety or policy guardrail check must run on each voice-agent response. Where does it go in the turn budget? [Likely] [20]
    - Serial check on the full response, streaming check per sentence, or parallel check with the ability to cut TTS: what does each add to time-to-first-audio, and what does each risk?
49. How does prompt length grow over a 30-minute voice-agent call, and what does that do to TTFT and cost per turn? [Likely]
    - Assume 150 turns of about 40 tokens each on top of a 2,500-token system prompt. What is the prompt at the last turn, and the total prefill tokens over the call with and without prefix caching?
    - At what length would you compact history into a running summary, and what does compaction cost?
50. Would you cache synthesized audio for common agent phrases (greetings, disclosures, hold messages)? [Likely] [11]
    - What hit rate is realistic, how much storage does a per-tenant, per-voice cache need, and how many ms does a hit save?
51. Estimate the cost per resolved conversation for the AI Agent, compared with a human agent. [Likely] [11]
    - Add ASR, LLM, TTS and telephony per minute, then divide by the containment rate. At what containment rate does automation stop being cheaper than an assumed fully loaded human cost per hour?

### F. Token cost: summarization, QA scoring and insights

52. Estimate the token cost per conversation for post-call summarization. [Likely] [14]
    - Assume a 6-minute call at about 150 spoken words per minute per active speaker and about 1.3 tokens per word. How many transcript tokens is that?
    - Add the system prompt and output. What is the cost per call at a self-hosted cost per million tokens compared with a frontier API price?
    - Multiply out to 1 million calls per day. What is the monthly bill under each option?
53. Estimate the token cost of automated QA scoring with an LLM judge. [Likely] [20]
    - Assume a scorecard with 25 criteria. Do you make one call per criterion or one call for all criteria? How many tokens does each design use per call?
    - How does prompt caching of the shared transcript prefix change the per-criterion design's cost?
    - What accuracy difference would justify the more expensive design?
54. How would you cut QA scoring cost by 10x without losing auditor trust? [Likely] [20]
    - Which criteria can a classifier or rule handle, and which need an LLM?
    - What fraction of calls do you route to the large model, and how do you pick them?
55. Estimate the cost of topic and driver discovery (Insights) across a month of calls. [Likely] [18]
    - Embedding every utterance compared with summarizing every call and then clustering: what are the token and compute costs of each?
    - How often must you recompute, and can you do it incrementally?
56. Should batch post-call work run on the same GPUs as real-time inference? [Likely]
    - How many GPU-hours per day does off-peak batch work need, and how much of it fits into the real-time fleet's night-time trough?
    - How do you keep a batch backlog from hurting real-time p95?
57. What is your total LLM cost per conversation across all features (assist, summary, QA, insights)? [Likely]
    - Which feature dominates, and is that what the business would expect?
    - How would you show cost per tenant so pricing and contracts reflect it?
    - For which features does output length dominate cost, and for which does input length? How do you cap summary length without losing required fields?
58. How do token counts and costs change for non-English calls? [Likely]
    - If Spanish or Japanese transcripts take 1.3-2x more tokens per word than English with your tokenizer, what happens to cost per call, context limits and prefill latency? Would you change the tokenizer or the model?
59. How many human labels do you need to validate an LLM QA judge, and what does that cost in annotator hours? [Likely] [20]
    - To estimate judge-human agreement to within plus or minus 3 points per criterion at 95 percent confidence, how many calls do you label? Multiply by criteria and minutes per call to get the labeling budget.
60. How would you budget tokens for the evaluation of a new summarization model before launch? [Likely] [20][21]
    - How many held-out calls and how many judge calls per call do you need for a confident comparison?

### G. Knowledge base and vector index sizing

61. Estimate the vector-index size for one tenant's knowledge base. [Likely] [15][17]
    - Assume 20,000 articles of about 1,500 words each, chunked at 300 tokens with 15 percent overlap. How many chunks?
    - At 768 or 1,024 dimensions in float32, what is the raw vector size? Add HNSW graph overhead and metadata. What is the total?
    - How does that change with int8 or binary quantization, and what recall loss do you accept?
62. Estimate the index size for all tenants, and decide between one shared index with tenant filters and one index per tenant. [Likely] [17]
    - How do noisy-neighbor latency, isolation guarantees and per-tenant deletion requests affect the choice?
    - How much memory does the whole fleet need if every index is kept in RAM?
63. What retrieval latency can Knowledge Assist afford inside the real-time budget? [Likely] [15]
    - Split it into query embedding, approximate nearest-neighbor search, re-ranking and fetching the text. Which step is the largest, and how do you cut it?
    - How many candidates can a cross-encoder re-rank within 50 ms on one GPU?
64. Estimate the cost of re-embedding all knowledge bases when you change embedding models. [Likely]
    - How many tokens, how many GPU-hours, and how do you run it without downtime (dual indexes, backfill, cut-over)?
65. How fresh must the index be, and what update rate must ingestion sustain? [Likely] [17]
    - If a tenant edits 500 articles a day, what is the embedding load, and how quickly must a change be searchable?
66. Would you index past conversations (resolved call transcripts) as well as articles? [Likely]
    - How does that change index size by orders of magnitude, and what retention or PII limits apply?
67. What load does real-time query embedding put on the embedding service? [Likely] [15]
    - If Knowledge Assist embeds a query on each detected customer question (about one every 30 seconds per call) at 10,000 concurrent calls, what QPS and how many GPUs or CPU cores do you need for a 50 ms p95?

### H. Storage, retention and PII redaction

68. Estimate transcript storage per year for the platform. [Likely] [12]
    - Assume 1 million calls a day, 6 minutes each, about 1,000 words per call, and word-level timestamps and speaker labels stored as JSON. What is the size per call and per year, raw and compressed?
    - How much do partial-transcript revisions add if you keep them, and should you?
69. Estimate audio storage per year if recordings are kept. [Likely]
    - At 64 kbit/s per channel for two channels, what is the size per call-minute, and per year at the volume above?
    - How do compression choice and retention period (30 days, 1 year, 7 years) move the total?
70. Where in the pipeline does PII redaction run, and what latency does it add? [Likely] [12]
    - For real-time redaction of a token stream, how much lookahead do you need to catch a card number split across partials, and what does that lookahead add to hint latency?
    - What throughput must the redaction model sustain at platform peak, in tokens per second?
71. Do you store both redacted and unredacted transcripts? [Likely]
    - How does each choice multiply storage, encryption-key management and access-audit volume?
    - How do you handle a customer's deletion request across transcripts, audio, embeddings and training sets?
72. Estimate the storage and compute cost of audio redaction (bleeping or silencing PII spans). [Likely] [12]
    - How do you align ASR word timestamps to audio, and what timing error forces you to pad each redacted span?
73. Size the event and analytics store that powers dashboards over conversations. [Likely] [18]
    - How many rows per call (utterances, model outputs, scorecard results), and how many rows per day?
    - Which queries must be interactive, and how does that drive partitioning and pre-aggregation?
74. What does data residency (EU, US, regulated industries) do to your capacity plan? [Likely]
    - How much extra headroom do you hold when each region must absorb its own peak alone?
75. Size the keyword-search index over one year of transcripts for Insights. [Likely] [18]
    - How large is an inverted index over the transcript volume from question 68 compared with the raw text? How does adding a vector index per utterance change the total?
76. Estimate the tracing and logging volume of the real-time pipeline. [Likely]
    - With about 15 spans per hint request and 1 KB per span, how many GB per day at peak QPS? What sampling rate keeps the bill sane while still giving p99 per tenant?

### I. Serving trade-offs that move the numbers

77. Walk through how continuous batching changes throughput and latency compared with static batching. [Likely]
    - At what point does adding requests to the batch raise p95 more than it saves cost?
78. When do you separate prefill and decode onto different GPUs? [Likely]
    - For a long-prompt, short-output workload like hint generation, which phase dominates, and does disaggregation pay off?
79. What does weight quantization (FP8, INT8, INT4) buy for each workload? [Likely]
    - How do you measure the quality cost on Cresta-like tasks, not on public benchmarks?
    - Does quantization help a memory-bound decode or a compute-bound prefill more?
80. How do you size the KV cache for long conversations? [Likely]
    - Assume a 7-8B model with grouped-query attention. What is the KV cache per token and per 8,000-token conversation? How many concurrent sequences fit on one 80 GB GPU?
    - When do you truncate history, summarize it, or offload KV to CPU memory?
81. Where does caching help, and what is the hit rate? [Likely]
    - Prefix (prompt) caching, semantic response caching for repeated knowledge questions, and retrieval caching: which is safe for which feature?
    - How do you prevent a cache from serving one tenant's content to another?
82. Distillation: how small can the hint or QA model get before it fails the quality bar? [Likely] [14][19]
    - How many teacher-labelled examples do you need, and what does generating them cost in tokens?
83. Compare a self-hosted fleet with a provider API for each workload. [Likely] [15]
    - Which workloads move to self-hosting first, given latency SLOs, steady volume and data-control needs?
    - How do provider rate limits and tail latency enter your capacity plan when you use an API?
84. What GPU type would you choose for each workload (real-time hints, voice agent, batch QA, embeddings)? [Likely]
    - How do memory size, memory bandwidth and price per hour change the answer?
85. How do you trade beam width against latency and memory in a real-time decoder? [Likely] [22]
    - What does beam width 4 do to KV-cache memory and per-step latency compared with greedy decoding?
86. When does a non-LLM model (classifier, small encoder) beat an LLM on the cost-latency-quality frontier for a real-time trigger? [Likely] [13]
    - How many classifier calls cost the same as one LLM call, and does that let you run on every partial?
87. Estimate the GPU-hours to fine-tune one tenant-specific LoRA adapter, and how many tenants the training cluster can serve per week. [Likely] [14][16]
    - Assume 50,000 conversations of 2,000 tokens, 2 epochs, an 8B base model and about 6 x parameters x tokens training FLOPs. How many GPU-hours is one run, and how many runs fit in a 64-GPU cluster per week?
88. Estimate the token and GPU budget for continued pretraining of a contact-center foundation model on transcripts. [Likely] [19]
    - How many tokens does a corpus of 100 million conversations give you after deduplication and PII redaction? How many GPU-days for one epoch on a 7-8B model, and how would you decide whether more epochs pay off?

### J. Reliability, headroom and SLO math

89. Set SLOs for each real-time feature, and derive the error budget. [Likely]
    - How do a 99.9 percent and a 99.95 percent monthly availability target translate into minutes of allowed downtime?
    - Is a hint that arrives after 2 seconds an error under the SLO?
90. How much redundancy do you need to survive the loss of a zone at peak? [Likely]
    - With three zones, what utilization must each run at in normal times so two zones can carry the peak?
91. How do you combine per-stage p95s into an end-to-end p95? [Likely]
    - Why is the sum of per-stage p95s a poor estimate, and how would you estimate the real tail?
    - How does fan-out to several models in the inference graph raise the tail [13]?
92. How do you load-test the voice pipeline before a large tenant goes live? [Likely] [21]
    - How do you generate realistic audio traffic at 2x the expected peak, and what do you watch first?
93. What does a retry policy cost in capacity during an incident? [Likely]
    - How do you stop retries from turning a slowdown into an overload (retry budgets, back-off, load shedding)?
94. How do you schedule maintenance and model rollouts when global traffic never truly reaches zero? [Likely]
    - How much spare capacity does a rolling restart of the LLM fleet need?
95. How fast can you move 10,000 live calls out of a failing region? [Likely] [12]
    - How much per-call state (ASR decoder state, conversation history, pending hints) must move, and can a call survive re-establishing ASR from scratch? What gap in transcript do you accept?

### K. Applying the estimates in a full design

96. Design Agent Assist for a new 5,000-agent tenant, and present a one-page capacity and cost plan. [Likely] [13][14]
    - Sketch: per-stage throughput, replicas per stage and the end-to-end latency budget on one diagram.
    - Which number in your plan are you least sure of, and how would you measure it in the first week?
97. Design the AI Agent for a tenant moving 30 percent of inbound calls from humans to automation. [Likely] [11]
    - How do GPU, TTS and telephony capacity change, and what is the cost per automated minute compared with a human agent minute?
98. Design Knowledge Assist for a tenant with 100,000 articles in 12 languages. [Likely] [15][17]
    - How do index size, embedding cost and retrieval latency grow, and would you use one multilingual index or one per language?
99. Design post-call summarization and QA scoring for 1 million calls per day with a 15-minute freshness target. [Likely] [14][20]
    - What is the steady-state and catch-up throughput you need, and how large a backlog can you clear after a two-hour outage?
100. Design Insights for a tenant who wants to ask open questions over two years of calls. [Likely] [18]
    - Which pre-computation (summaries, embeddings, tags) do you pay for up front, and which per query?
101. The CFO asks you to cut inference spend by 40 percent without hurting customer-visible quality. What is your plan? [Likely]
    - Rank your levers by savings and by risk, and say how you would prove quality held.
102. An enterprise customer asks for a contractual p95 of 200 ms for hints. What do you need to change, and what would you price differently? [Likely]
    - Which stages must shrink, and would you dedicate GPUs to that tenant?
103. You doubled the context window to improve hint quality, and p95 rose by 40 percent. How do you find out why and fix it? [Likely]
    - Which metrics separate queueing, prefill and decode growth?
104. Explain to a product manager why a 2 percent quality gain from a larger model is or is not worth 3x the serving cost. [Likely] [7]
    - How do you express the cost in their units (per seat, per call, per month)?

## References

1. Glassdoor, "Cresta (CA) Interview Questions" (Senior Machine Learning Engineer review: recruiter call, LeetCode-medium, vibe-coding and system design rounds; "How would you deploy a model in production?"; rate-limiter coding question; FDE AI Agent worker-pool and round-robin orchestrator question), 2024-2026. https://www.glassdoor.co.in/Interview/Cresta-CA-Interview-Questions-E3180909.htm
2. Blind, "Cresta interview process" (one hour of system design next to one hour of coding), 2024-2025. https://www.teamblind.com/post/cresta-interview-process-jmfnzjoa
3. Blind, "Anyone interview with Cresta lately?" (recruiter flagged LLM, GenAI, RAG and AI-agent fundamentals), 2025. https://www.teamblind.com/post/anyone-interview-with-cresta-lately-zvj0uhq2
4. PracHub, "Cresta ML System Design Interview Questions" ("Design an Evaluation Platform for AI Agents Across Prompt and Model Changes"), 2026. https://prachub.com/companies/cresta/categories/ml-system-design
5. PracHub, "Cresta Software Engineer Interview Experience: System Design Round: Build an Eval Platform for AI Agents". https://prachub.com/interview-experiences/cresta-software-engineer-interview-experience-system-design-round-build-an-eval-platform-for-ai-agents
6. Cresta on Greenhouse, "Senior Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4249943008
7. Cresta on Greenhouse, "Staff Machine Learning Engineer". https://job-boards.greenhouse.io/cresta/jobs/4741283008
8. Cresta on Greenhouse, "Senior Machine Learning Engineer, Voice Experience". https://job-boards.greenhouse.io/cresta/jobs/5155675008
9. Cresta on Greenhouse, "Senior Machine Learning Engineer (ASR)". https://job-boards.greenhouse.io/cresta/jobs/5101011008
10. Built In Vancouver, "Senior Machine Learning Engineer, Cresta" (Agentic Assist track). https://builtinvancouver.org/job/senior-machine-learning-engineer/8619914
11. Cresta Blog, "Engineering for Real-Time Voice Agent Latency". https://cresta.com/blog/engineering-for-real-time-voice-agent-latency
12. Cresta Blog, "Understanding Cresta's Voice Platform: The Voice Stack", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-the-voice-stack
13. Cresta Blog, "Understanding Cresta's Voice Platform: ML Services, Inference Graphs and Real-Time Intelligence", 2025. https://cresta.com/blog/understanding-crestas-voice-platform-ml-services-inference-graphs-and-real-time-intelligence
14. Cresta Blog, "How Fine-Tuned LLMs Power Knowledge Assist, Summarization and Chat Suggestions", 2024. https://cresta.com/blog/how-fine-tuned-llms-power-knowledge-assist-summarization-and-chat-suggestions
15. Fireworks AI Blog, "Story: Cresta Knowledge Assist", 2024. https://www.fireworks.ai/blog/story-cresta-knowledge-assist
16. Fireworks AI Blog, "Multi-LoRA", 2024. https://fireworks.ai/blog/multi-lora
17. Cresta Blog, "AI-Ready Knowledge for Contact Centers: Closing the Gap Between KBs and AI", 2025. https://cresta.com/blog/ai-ready-knowledge-for-contact-centers-closing-the-gap-between-kbs-and-ai
18. Cresta Blog, "Introducing the Next Generation of Cresta Insights", 2026. https://cresta.com/blog/introducing-the-next-generation-of-cresta-insights-a-complete-system-for-enterprises-to-understand-their-cx
19. Cresta Blog, "Introducing Ocean-1: World's First Contact Center Foundation Model", 2023. https://cresta.com/blog/introducing-ocean-1-worlds-first-contact-center-foundation-model
20. Cresta Blog, "How Cresta Grades Its Agents: Evaluators in an LLM World". https://cresta.com/blog/how-cresta-grades-its-agents-evaluators-in-an-llm-world
21. Cresta Blog, "Cresta Launches Automated AI Agent Testing: Confidence in Every Conversation". https://cresta.com/blog/cresta-launches-automated-ai-agent-testing-confidence-in-every-conversation
22. 1point3acres, "Cresta Machine Learning Engineer Tech Phone Interview Experience" (MLE, AI Agent direction; greedy and beam-search decoding), 2025. https://www.1point3acres.com/interview/thread/1140481

23. Glassdoor, "Cresta (CA) Interview Experience and Questions" (Forward Deployed Engineer technical screen: "What is RAG, and how do you prevent hallucination?"), 2026. https://www.glassdoor.com/Interview/Cresta-CA-Interview-Questions-E3180909.htm
24. 1point3acres, "Cresta Fulltime Software Engineer Tech Phone Screen: Job Dispatcher-Executor System Implementation" (title seen in a listing; full text not read), 2026. https://www.1point3acres.com/interview/thread/1172433
25. 1point3acres, "cresta interview questions" (company page; SWE and MLE phone screens, including calling a public API such as Wikipedia and processing the results), 2024-2026. https://www.1point3acres.com/interview/company/cresta

*Source note:* No page could be opened in either research pass. In the October 2026 review, WebFetch failed DNS for cresta.com, fireworks.ai and prachub.com, curl through the egress proxy got 403 for cresta.com, prachub.com and job-boards.greenhouse.io, and the shared web-search budget was used up. So no URL could be confirmed live or dead here, and none was removed for that reason. References 1-22 are reused from chapters 01, 04 and 07, where their excerpts were reviewed. References 23-25 are reused from chapter 10 (entries 3, 15 and 16), with the same attributions and strength notes. Reference 24 is weak: only the thread title was seen. Glassdoor mirrors [1] and [23] show one review pool, so they are not independent reports. The product facts in the format notes (partial and final transcripts about once per second, redaction in the voice stack, inference graphs, fine-tuned models with multi-LoRA) come from those reviewed excerpts. All numeric assumptions inside the questions (hardware throughput, prices, volumes) are illustrative inputs, not Cresta figures.
