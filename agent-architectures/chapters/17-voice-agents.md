# 17. Voice and Realtime Agents: Speech-to-Speech vs Cascaded Architectures, Turn-Taking and Latency Budgets

## Overview

A voice agent is an agent loop with a hard real-time edge. Earlier chapters covered tool use, planning, multi-agent handoff and context engineering. All of those still apply. Voice adds four things text agents can ignore:

1. **A latency budget measured in hundreds of milliseconds.** Someone on a call notices a slow reply long before they would notice it in a chat window.
2. **Turn-taking.** The system must decide when the user has finished, whether an overlapping sound is an interruption or a backchannel ("uh-huh"), and what to do with a reply it has half spoken.
3. **Streaming everywhere.** Audio arrives as a stream. Transcripts are partial, model tokens stream out, and synthesized audio streams back. Every stage must overlap with the next.
4. **Transport and telephony.** WebRTC, WebSockets, SIP, G.711 codecs, echo cancellation, session time limits and reconnection.

By late 2026 the public designs fall into three families plus one hybrid that is gaining ground.

- **Cascaded (STT -> LLM -> TTS).** Separate streaming models for recognition, reasoning and synthesis, coordinated by an orchestrator. Pipecat, LiveKit Agents, Vapi, Retell, Kyutai's Unmute and Hugging Face's speech-to-speech stack all follow this pattern.
- **Turn-based native speech-to-speech (S2S).** One multimodal model takes audio in and emits audio out. The interaction is still organized into turns that a VAD or semantic detector closes. OpenAI's Realtime API (gpt-realtime family), Google's Gemini Live API and AWS Nova Sonic are examples.
- **Full-duplex S2S.** The model listens and speaks at the same time and decides for itself when to talk. Kyutai's Moshi and NVIDIA's PersonaPlex are open-weight examples. OpenAI's GPT-Live (gpt-live-1) is a commercial one.
- **Speaker/reasoner hybrids.** A fast conversational voice model fronts a slower, smarter text model that does tool calls and reasoning. Examples are OpenAI's "chat-supervisor" pattern, GPT-Live "delegation" and the Vapi GPT-Live integration.

This chapter covers each family through its primary sources. Those sources are vendor documentation, SDK source code, open-source framework docs, model repositories and benchmarks. It then compares turn-detection techniques and latency budgets, lists the recurring failure modes and closes with lessons for designers.

```
            THE VOICE AGENT DESIGN SPACE (late 2026)

  user audio ──► [transport: WebRTC / WebSocket / SIP / PSTN]
                        │
      ┌─────────────────┼──────────────────────────┬────────────────────┐
      ▼                 ▼                          ▼                    ▼
  CASCADED         TURN-BASED S2S             FULL-DUPLEX S2S      SPEAKER/REASONER
  VAD+EOT          server VAD / semantic      model listens while  voice model talks,
  STT ─► LLM       VAD closes the turn;       speaking; decides    delegates tools and
  ─► TTS           one model hears + speaks   when to talk         reasoning to text LLM
  (Pipecat,        (gpt-realtime, Gemini      (Moshi, PersonaPlex, (chat-supervisor,
  LiveKit, Vapi,   Live, Nova Sonic)          gpt-live-1)          GPT-Live delegation)
  Retell, Unmute)
```

## 17.1 Why voice is a distinct architecture problem

### The latency chain

Vapi's documentation breaks voice-to-voice latency, "the gap between when a caller finishes speaking and when your assistant starts responding", into five stages that run in sequence: transport, endpointing, transcription, model response (time to first token) and speech synthesis (time to first audio) [32]. Vapi's dashboard shows the median of the transcriber, model and voice stages. It leaves out endpointing and transport, and the documentation warns that endpointing "can account for a meaningful share of that delay" [32]. The same page notes that time to first token is often the largest single component, and that more capable models are usually slower [32].

Practitioner frameworks give targets of the same order. Pipecat's overview states that "typical voice interactions complete in 500-800ms" [22]. Its Cerebrium deployment guide reports "~300ms voice-to-voice responses" when STT, LLM and TTS all run locally on the same infrastructure [23]. At the far end, Moshi's full-duplex design has a theoretical latency of 160 ms (80 ms Mimi frame plus 80 ms acoustic delay) and a practical latency of about 200 ms on an L4 GPU [37]. LLaMA-Omni reports interaction latency "as low as 226 ms" [44].

Medians do not capture the whole experience. The Hugging Face and Cerebras Gemma 4 voice write-up (July 2026) notes that "some production systems see a reasonable median latency while still experiencing frustrating multi-second delays at the P95", and that tool calls make it worse [49]. Designers should budget for tail latency, not just typical latency.

### Turn-taking is the hidden budget item

The most counter-intuitive finding across sources is that **waiting for the user to finish** often costs more than any model. Pipecat's tuning guide defines **Time To Final Segment (TTFS)**: the time from the user stopping to the final transcript arriving. It ships measured P99 TTFS values per STT provider, because "getting TTFS right is one of the most impactful tuning knobs for perceived conversation responsiveness" [20]. The built-in table, all measured with a 0.2 s VAD stop window, ranges from 0.35 s (Deepgram, Soniox) through 0.42 s (AssemblyAI) and 0.60 s (Gemini) to around 2 s for several batch-style endpoints such as OpenAI's non-realtime transcription and ElevenLabs' non-realtime API [21]. Turn-based STT services such as Deepgram Flux and Cartesia's turn API have no TTFS figure, because "the server defines the turn boundary directly" [20][21].

### A composed latency budget

The table below adds up a cascaded turn from sourced defaults. LLM and TTS first-byte times depend on the vendor and deployment and must be measured. Pipecat and Vapi both emit them as TTFB or TTFA metrics.

| Stage | Typical sourced value | Source |
|---|---|---|
| VAD stop confirmation | 0.2 s (Pipecat `stop_secs` default) | [19] |
| Final transcript after stop (P99 TTFS) | 0.35-0.6 s for fast streaming STT | [21] |
| Semantic end-of-turn inference | ~10-100 ms on CPU (Smart Turn v3) | [24] |
| Endpointing floor in LiveKit | 0.5 s min delay (0.3 s for streaming-EOT STT) | [28] |
| LLM time to first token | measure; often the largest single term | [32] |
| TTS time to first audio | measure; Unmute TTS ~750 ms -> ~450 ms after GPU split | [40] |
| Transport (WebRTC/PSTN) | not included in Vapi's displayed totals | [32] |

## 17.2 Cascaded vs speech-to-speech: the core trade-off

### Cascaded pipelines

A cascaded pipeline streams audio through VAD and STT, aggregates the transcript into an LLM context, streams LLM tokens into a TTS and streams the audio back out. Pipecat's documentation stresses that "everything happens in parallel": while the LLM is still generating, earlier sentences are already being synthesized and played [22].

Advantages documented in the sources:

- **Best instruction following and function calling.** Pipecat's OpenAI guide recommends the cascaded approach "if you are implementing a complex workflow and need the best possible instruction following performance and function calling reliability" [7].
- **Reasoning parity with text.** Artificial Analysis' Big Bench Audio (December 2024) found that GPT-4o scored 92% on the text version. The native speech-to-speech GPT-4o Realtime Preview (Oct '24) scored 66%, and the text-to-speech configuration 74%. A Whisper -> GPT-4o -> tts-1 pipeline "show[ed] minimal performance degradation compared to pure text processing", which led the authors to conclude that pipelines "currently outperform native audio for reasoning" [48]. Later realtime model generations narrow the gap, but no updated figure was available in the sources reviewed here.
- **Swappability and observability.** Each component is a replaceable service with its own metrics. Pipecat lists about 20 STT, 25 LLM and 30 TTS integrations [16]. LiveKit's plugin directory has dozens [27].
- **Open stacks.** Hugging Face's speech-to-speech pipeline (Gemma 4 31B on Cerebras plus Qwen TTS) powers more than 9,000 Reachy Mini robots [49]. Kyutai's Unmute is a fully open cascade [40].

Disadvantages: prosody and emotion in the user's voice are reduced to text, latency adds up across hops, and the turn detector must work from VAD plus text or plus separate audio models.

### Native speech-to-speech

In S2S models one network hears and speaks. Pipecat describes this as "conceptually much simpler" but cautions that "most applications also need to implement things like function calling, retrieval-augmented search, context management, and integration with existing systems. So the core pipeline is only part of an app's complexity." It recommends S2S when "better audio understanding and human-like voice output" matter more than instruction following, for example in free-form conversation [7].

### Choosing between them

| Dimension | Cascaded | Turn-based S2S | Full-duplex S2S |
|---|---|---|---|
| Turn decision | Client: VAD + EOT model | Server VAD / semantic VAD, or client-driven | Learned inside the model |
| Paralinguistics in/out | Lost on input; TTS-controlled out | Preserved | Preserved; overlap possible |
| Tool calling and instruction following | Strongest (text LLM) [7] | Good, improving per generation | Usually delegated to a text model [8][9] |
| Vendor lock-in | Low; per-component swaps | One model vendor | One model vendor or self-hosted weights |
| Exact or pre-recorded speech | Yes (TTS of fixed text) | Mostly generated | Not supported in GPT-Live on Vapi [11] |
| Representative systems | Pipecat, LiveKit, Vapi, Retell, Unmute | gpt-realtime, Gemini Live, Nova Sonic | Moshi, PersonaPlex, gpt-live-1 |

## 17.3 OpenAI Realtime API

### Problem and architecture

The Realtime API keeps a stateful, event-driven session with a speech-to-speech model. The OpenAI Agents SDK's realtime guide splits client architecture into `RealtimeAgent` (instructions, tools, handoffs, output guardrails), `RealtimeRunner` (session factory), `RealtimeSession` (live I/O, history, tool execution) and `RealtimeModel` (transport abstraction) [1]. Three transports are documented:

- **WebSocket** for server-side connections. This is the SDK default (`OpenAIRealtimeWebSocketModel`).
- **WebRTC** for browsers. OpenAI's reference console uses WebRTC with events on the data channel. It notes that client WebSockets are "not recommended in browsers" [4].
- **SIP** through a Realtime Calls API `call_id` (`OpenAIRealtimeSIPModel`) [1].

```
  Browser ──WebRTC (audio + data channel)──┐
  Phone ──PSTN──► SIP trunk ──SIP call_id──┼──► Realtime session (gpt-realtime-*)
  Server ──WebSocket (JSON events)─────────┘     │  input_audio_buffer (VAD/commit)
                                                 │  conversation items (history)
                                                 │  response.create / cancel
                                                 ▼
              events: audio deltas, transcripts, function_call, response.done
              client:  run tools ─► function_call_output ─► response.create
```

Session configuration is nested under `audio.input` and `audio.output`. It covers format (examples use `pcm16`, and Pipecat exposes PCM, PCMU and PCMA, the last two being G.711 for telephony), transcription, noise reduction, turn detection, voice and speed [1][14-adjacent: 7]. A response is single-modality, either `["text"]` or `["audio"]` [7]. Voice is fixed once audio has been produced, and model choice is per session, not per agent [1].

Pipecat's integration docs list the model line as `gpt-realtime`, `gpt-realtime-1.5`, `gpt-realtime-2` and `gpt-realtime-2.1`. The Agents SDK recommends `gpt-realtime-2.1` for new agents, and reasoning configuration is available on reasoning-capable models such as `gpt-realtime-2` [1][7]. The input transcription side channel has its own models: `gpt-live-transcribe` (incremental), `gpt-transcribe` (after commit) and `gpt-realtime-whisper`, which has a `delay` knob from `minimal` to `xhigh` that trades latency for accuracy [1].

### Turn detection

The Python SDK types document two server-side modes [2]:

- **`server_vad`**: energy/VAD-based. `threshold` (default 0.5), `prefix_padding_ms` (default 300 ms of audio kept before detected speech), `silence_duration_ms` (default 500 ms), an optional `idle_timeout_ms` that triggers a response after inactivity, and the flags `create_response` and `interrupt_response`.
- **`semantic_vad`**: a model judges whether the user has finished. Its main knob is `eagerness` ("`auto` is the default and is equivalent to `medium`").

Turn detection can also be set to `null`. The application then owns the turn: it sends `input_audio_buffer.commit` and `response.create` itself [1]. That is how frameworks plug in their own end-of-turn models. When Pipecat drives turns locally against the Realtime API, it replays a short **pre-roll** of recent audio after an interruption clears the input buffer, "so the speech onset isn't lost". The pre-roll defaults to the VAD's `start_secs` plus a margin, or 0.5 s [14-adjacent: 7].

### Interruption and truncation

When the user barges in, the SDK emits `audio_interrupted` and "updates history to match what the user heard". For telephony or other remote playback, where the server cannot know what has actually been played, the SDK provides a `RealtimePlaybackTracker` that truncates at the real playback position [1]. This is the S2S version of a problem every architecture has to solve: the model's context must hold what the user **heard**, not what the model **generated**.

### Tools, approvals and guardrails

Function tools can require approval. The session emits `tool_approval_required` and pauses until the call is approved or rejected [1]. Output guardrails run on accumulated transcript text, debounced by `debounce_text_length`. If a guardrail trips on audio, the SDK interrupts the response, sends `response.cancel` and injects a follow-up user message [1]. This is an important limit: guardrails on streamed speech are always slightly late, because some audio may already have played.

### Multi-agent patterns for realtime

OpenAI's `openai-realtime-agents` reference app documents two patterns [3]:

- **Chat-supervisor.** A realtime model handles greetings, small talk and information gathering within an explicit allow-list. A text model (`gpt-4.1` in the example) handles tool calls and harder turns. The README presents this as cheaper and as a way to reuse existing text agents, at the cost of extra latency on supervisor turns (roughly 2 s, approximate). More replies start with filler such as "Let me think."
- **Sequential handoff.** Specialist agents hand the user over through tool calls, in the style of Swarm. Each handoff triggers a `session.update` that swaps instructions and tools, so no single agent has to carry too many tools.

### Prompting

OpenAI's Realtime prompting guide recommends labeled sections: Role & Objective, Personality & Tone, Context, Reference Pronunciations, Tools, Instructions/Rules, Conversation Flow, Safety & Escalation. It also recommends short bullets, sample phrases (the model "closely follows" them), a variety rule against robotic repetition, and pinning the output language when the model drifts. One documented change was replacing "inaudible" with "unintelligible", which improved handling of noisy input [5].

### Failure modes and lessons

- Turn-detection defaults (500 ms of silence) cut off slow speakers. Semantic VAD or a client-side detector is the usual fix [2].
- Context drift after interruptions, if truncation is not tied to playback [1].
- Guardrails trip after unsafe audio has already started [1].
- Voice cannot change mid-session, and structured outputs are not supported in realtime agents [1].

## 17.4 OpenAI GPT-Live: full duplex with delegation

GPT-Live (`gpt-live-1`) is OpenAI's full-duplex speech-to-speech API, served over `wss://api.openai.com/v1/live/sessions` [8]. Pipecat's reference describes the core behavior: the model "listens and speaks at the same time: it decides on its own when to answer, stops when the user talks over it, and delegates work that needs search, reasoning, or tools to a backend text model while the conversation continues." There is "no client-side turn detection or response triggering" [8].

### Architecture

```
            ┌──────────────── GPT-Live session (gpt-live-1) ─────────────────┐
  caller ◄──┤ SPEAKER: hears + speaks continuously; decides when to delegate │
  audio  ──►│        │ delegate(request + transcript snapshot)    ▲ result   │
            └────────┼────────────────────────────────────────────┼──────────┘
                     ▼                                            │
            REASONER (text model): procedures, tools, skills ─────┘
            Responses delegation: hosted by OpenAI
            Client delegation: any Pipecat LLM in a BackendLLMWorker
```

Two delegation modes exist. In **Responses delegation**, OpenAI hosts the backend through a Responses API model configuration. In **client delegation**, the backend is any LLM service the developer runs [8]. The API fixes model, voice, store and system instruction when the session starts. With `store`, a session can be kept for 30 days as a stereo recording, and later sessions can fork from it [8].

Vapi's integration (beta) calls the two parts **speaker** and **reasoner**. The reasoner defaults to `gpt-5.6-terra` with `reasoningEffort` defaulting to `low`, and "higher effort can increase response time" [11]. The design guide explains the context split:

- The speaker hears audio and has no tools.
- The reasoner "doesn't hear audio. Works from the conversation transcript". It receives "the transcript available when the request starts" [10].

As a result, "the reasoner starts from a snapshot". If the caller changes the request while a lookup runs, the running lookup does not change, and the next delegation carries the correction [10]. Vapi recommends ordering work by dependency: delegate as soon as a lookup's inputs are known, gather other details while it runs, and design tools to return enough data to answer likely follow-ups without another round trip [10]. An interruption does not cancel a tool call [11].

### Constraints documented by an integrator

Vapi's compatibility table is a useful picture of what full duplex currently gives up [11]:

- All speech, including greetings, is generated. Exact or pre-recorded speech is not possible.
- Only preset OpenAI voices (22 of them) are available, and there are no model or voice fallbacks.
- Squads and handoffs are not supported. They are replaced by "reasoner skills".
- Knowledge bases are not available to the reasoner.
- Only blind transfers on certain carriers are supported.
- A session cannot be reconnected.
- Raw WebSocket audio must be mono 16-bit PCM at 24 kHz.

The lesson is that full duplex moves turn-taking into the model, and with it control over exact wording. Teams trade determinism for naturalness.

## 17.5 Google Gemini Live API

### Architecture

The Live API is a bidirectional streaming session (`BidiGenerateContent`) over WebSockets. Google's reference web console uses "an Event-emitting websocket-client" with separate audio I/O modules for microphone, webcam and screen capture. Tools such as Google Search grounding and function declarations go in the session config [13]. Model names in the public SDK span `gemini-2.0-flash-live-preview-04-09` through `gemini-3.1-flash-live-preview` [12][47]. Pipecat's current default points to a newer "live" model [14]. Pipecat's Gemini integration keeps a persistent connection to Gemini Live and talks to clients over WebRTC (web and mobile) or WebSockets (telephony) [14].

```
 client ─WebRTC/WS─► orchestrator (e.g., Pipecat) ─WebSocket BidiGenerateContent─► Gemini Live
                         │  setup: model, system_instruction, tools,               │
                         │         realtime_input_config, session_resumption,      │
                         │         context_window_compression, proactivity,        │
                         │         enable_affective_dialog, thinking_config        │
                         ◄── audio chunks, transcripts, toolCall, GoAway, resumption handles
```

### Turn-taking controls

The Gen AI SDK exposes unusually explicit turn semantics [12]:

- **Automatic activity detection.** This is server-side and on by default. If it is disabled, "the client must send activity signals". It has `start_of_speech_sensitivity` and `end_of_speech_sensitivity`. In the Gemini API both default to HIGH, while the enterprise Agent Platform defaults to LOW. It also has a prefix padding duration (the speech needed before start is committed) and a silence duration ("the larger this value, the longer speech gaps can be without interrupting the user's activity but this will increase the model's latency").
- **Activity handling.** `START_OF_ACTIVITY_INTERRUPTS` (the default barge-in behavior, where "the model's current response will be cut-off in the moment of the interruption") or `NO_INTERRUPTION`.
- **Turn coverage.** Controls whether a user turn includes only activity (the default), all input including silence, or audio activity plus all video.
- **Proactivity.** `proactive_audio` lets the model "reject responding to the last prompt", for example to "ignore out of context speech or to stay silent if the user did not make a request, yet."
- **Affective dialog.** "The model will detect emotions and adapt its responses accordingly."

### Long sessions

Live sessions are long-lived connections, and the API has three mechanisms that matter for production [12]:

- **Session resumption.** The server sends resumption handles. With transparent mode it also sends `last_consumed_client_message_index`, so clients can replay unacknowledged messages after a reconnect.
- **GoAway.** The server warns that the connection will soon end, with the time remaining.
- **Context window compression.** A sliding window activates at `trigger_tokens` and trims to `target_tokens` (default trigger/2), always cutting at the start of a user turn. System instructions and prefix turns are preserved. The SDK warns that "window shortening operation has some latency costs, so we should avoid running it on every turn."

For browser clients, **ephemeral auth tokens** keep long-lived keys off the device. By default a token can be used once, must open a new session within 60 seconds, and expires for messages 30 minutes later. Session resumption does not count as a use [12].

### Cross-vendor note: AWS Nova Sonic

Nova Sonic follows the same turn-based S2S pattern (16 kHz in, 24 kHz out, `endpointing_sensitivity` LOW/MEDIUM/HIGH). It has an AWS-imposed session limit of about 8 minutes. Pipecat works around it with **session continuation**: from 360 s into a session it waits for the assistant to start speaking, opens a new session in the background, replays a 3 s rolling buffer of user audio and hands off with context preserved [15]. Session rotation is a general pattern for any S2S API with connection or context limits.

## 17.6 Pipecat (Daily)

### Problem

Pipecat is an open-source Python framework for the orchestration problem. Its documentation describes coordinating streaming STT, LLM and TTS services over network transports as "complex timing, buffering, error handling, and service coordination" that "most developers end up rebuilding" [22].

### Architecture: frames, processors, pipelines, workers

Everything in Pipecat is a **frame** moving through **frame processors** connected in a **pipeline**. A **worker** runs a pipeline, and "a worker that owns a pipeline is an agent". Several workers can coordinate over a shared bus through handoff, parallel fan-out or sidecar workers [16][22].

```
transport.input() ─► STT ─► user_aggregator ─► LLM ─► TTS ─► transport.output()
      │ InputAudioRawFrame   TranscriptionFrame  LLMTextFrame  TTSAudioRawFrame │
      │                                                                         ▼
      │                                               assistant_aggregator (spoken
      │                                               text only, via TTSTextFrame)
 SystemFrames (InterruptionFrame, UserStartedSpeakingFrame) bypass queues: high priority
```

The key design choice is two **priority lanes** [17]:

- **SystemFrames** (user audio input, speaking events, `InterruptionFrame`, errors) are high priority and never discarded by interruptions.
- **DataFrames and ControlFrames** (output audio, text, `EndFrame`, TTS boundaries) are queued and processed in order.

Ordering is guaranteed within a lane, so `TTSSpeakFrame("Goodbye!")` followed by `EndFrame()` reliably speaks and then ends [17].

### Interruption semantics

When a turn-start strategy fires with interruptions enabled (the default), the user aggregator broadcasts an `InterruptionFrame` upstream and downstream [18]. Each processor then:

- cancels its task and discards queued data and control frames, except frames marked `interruptible=False` such as function results;
- cancels the in-flight LLM stream, along with functions registered with `cancel_on_interruption=True`;
- stops the TTS and clears its aggregation;
- drains the output transport's audio queue, so "the bot goes silent within roughly one audio write" [18].

The context gets **only the words actually spoken**. TTS text frames move downstream in sync with playback, and on interruption the assistant aggregator commits only the text played so far [18]. For noise-triggered interruptions with no transcript, the default runs the LLM once so the bot recovers instead of stalling [18]. Mitigations for false barge-in include mute strategies (`AlwaysUserMuteStrategy`, `FirstSpeechUserMuteStrategy`), `MinWordsUserTurnStartStrategy(min_words=3)`, and Krisp VIVA's interruption-prediction model that separates real interruptions from "uh-huh" [18]. With speech-to-speech services, the provider detects barge-in and Pipecat follows through external turn strategies while still flushing local audio [18].

### Turn detection

Turn **start** uses VAD. Silero VAD processes 30+ ms chunks in under 1 ms on one CPU thread, with defaults `confidence=0.7`, `start_secs=0.2`, `stop_secs=0.2` and `min_volume=0.6`. Transcription fallback and minimum words are alternatives. Turn **end** defaults to the Smart Turn model (the default since v0.0.102), with a plain speech-timeout strategy as the alternative [19][24-overview]. The short 0.2 s stop window is deliberate: it lets STT finalize early, and the semantic model then decides whether the user is actually done [19]. The docs recommend removing noise upstream (Krisp VIVA, ai-coustics, RNNoise) rather than raising VAD thresholds [19].

### Lessons

Pipecat's documentation turns several practices into defaults: measured per-provider P99 TTFS values fed to turn strategies [20][21], spoken-only context commits, priority lanes for control signals, and S2S services as drop-in pipeline elements. That last point means one framework can run cascaded, S2S and GPT-Live full-duplex agents, including with client delegation [7][8].

## 17.7 LiveKit Agents

### Architecture

LiveKit Agents (Apache-2.0) runs on LiveKit's WebRTC SFU and SIP stack. An **AgentServer** process schedules jobs and launches a per-session **entrypoint** when users connect through dispatch APIs [27]. An **AgentSession** holds STT, LLM, TTS, VAD and typed `userdata`. **Agents** carry instructions, tools and optional model overrides, and a `function_tool` can return a new Agent to perform a handoff while passing `chat_ctx` along [27]. Any combination of STT/LLM/TTS or a realtime model can be used, and a turn detector can be paired even with OpenAI's Realtime model [27][30].

```
  LiveKit SFU / SIP ◄──► AgentServer ──dispatch──► job: entrypoint(ctx)
                                                   └─ AgentSession
                                                       ├─ VAD (Silero) ─┐
                                                       ├─ STT ──────────┼─► audio EOT
                                                       │                │   TurnDetector
                                                       ├─ LLM / RealtimeModel ◄─ endpointing
                                                       ├─ TTS              (min/max delay)
                                                       └─ Agent A ──tool──► Agent B (handoff)
```

### Turn handling as a first-class subsystem

`TurnHandlingOptions` in the source groups four concerns [28]:

- **Endpointing.** `min_delay` defaults to 0.5 s ("minimum time since last detected speech before declaring the user's turn complete") and `max_delay` to 3.0 s. Streaming-EOT STT uses tighter defaults of 0.3 s and 2.5 s. A `dynamic` mode adapts the delay with an exponential moving average (`alpha=0.9`).
- **Interruption.** Mode `adaptive` ("ML-based detection") or `vad`. Defaults: `min_duration` 0.5 s, `min_words` 0, `resume_false_interruption=True` with `false_interruption_timeout` 2.0 s (the agent resumes speaking if the "interruption" turns out to be silence), and a `backchannel_boundary` of (1.0, 1.0) s near the start and end of agent turns, during which overlapping speech classified as a backchannel is suppressed.
- **Preemptive generation** (enabled by default). The LLM starts on a provisional transcript before the turn is confirmed. TTS is not preemptive by default. Limits: user speech up to 10 s and 3 retries per turn.
- **User turn limits**, for example `max_words`.

The session also has `aec_warmup_duration` (default 3.0 s, `None` for outbound SIP), which ignores interruptions while echo cancellation converges. Without it the agent can be interrupted by its own echo. There is also `user_away_timeout` (15 s), a `transcription_timeout` for when VAD heard speech but STT returned nothing, and IVR detection [29].

### The turn-detector model lineage

LiveKit's first end-of-utterance models were **text-based** open-weight transformers (`livekit/turn-detector`, English revision `v1.2.2-en` and multilingual `v0.4.1-intl`, int8 ONNX, under 500 MB RAM, CPU-only, shared across sessions, licensed under the LiveKit Model License) [30]. That plugin is now deprecated in favor of `livekit.agents.inference.TurnDetector`, "a unified audio end-of-turn detector" that replaces both the English and multilingual text models. It can run in the cloud with automatic local fallback and has a 1.0 s prediction timeout [30][27]. The shift from text to audio EOT matches Pipecat's audio-native Smart Turn and Kyutai's semantic VAD: prosody carries turn-yielding cues that a transcript loses.

## 17.8 Vapi

### Architecture

Vapi is a hosted orchestration platform. Its data-flow documentation describes a transport layer (SIP, telephony, WebSocket, WebRTC), a transcriber, an **orchestration layer** containing endpointing, interruption detection, emotion detection and backchanneling, a language model, a voice, and artifacts (recordings, transcripts, logs, usage metrics). Each component can use Vapi's providers, the customer's keys or the customer's own servers [33].

### Start/stop speaking plans

Vapi exposes turn-taking as two declarative plans [31]:

- **startSpeakingPlan** decides when to respond. Precedence is: transcriber-native end-of-turn (for example Deepgram Flux or AssemblyAI) when no smart plan is set, then custom rules, then a **smart endpointing plan**. Providers are `livekit` (English) and `vapi`, or a `custom-endpointing-model` server that returns a timeout. The LiveKit provider maps model probability to wait time with a configurable `waitFunction`, recommended as `2000 / (1 + exp(-10 * (x - 0.5)))` ms, a logistic curve. A text-only fallback, `transcriptionEndpointingPlan`, waits 0.1 s after punctuation, 1.5 s with no punctuation and 0.5 s after numbers. `waitSeconds` (0.4 s recommended) adds a final delay before speaking.
- **stopSpeakingPlan** decides when to yield. `numWords` (0 means VAD-based interruption, "50-100ms response"), `voiceSeconds` (0.2 s), `backoffSeconds` (1.0 s), `interruptionPhrases` (clear the pipeline immediately) and `acknowledgementPhrases` (ignore, which treats them as backchannels).

```
User audio → VAD → STT → start-speaking decision → LLM → TTS → waitSeconds → audio
                          ▲ transcriber EOT | custom rules | smart endpointing (sigmoid wait)
during bot speech: VAD start → interruptionPhrases? clear : acknowledgementPhrases? ignore
                              : numWords threshold → clear pipeline → backoffSeconds
```

### Multi-agent and full duplex

**Squads** split a large assistant into specialists with handoff tools, which Vapi says reduces hallucination, token cost and latency compared with one long prompt [34]. Vapi also offers **GPT-Live** as an alternative architecture (Section 17.4), where squads become reasoner skills and the start and stop speaking plans "don't configure a GPT-Live conversation" [11]. That is clear evidence that full-duplex models make the classic endpointing stack unnecessary.

## 17.9 Retell AI

Retell is a hosted voice-agent platform. Its public engineering detail is thinner than Vapi's or LiveKit's, but its **custom LLM WebSocket protocol** shows a clean separation between orchestration and reasoning [35]. When a call starts, Retell connects to the developer's WebSocket server (`/llm-websocket`). Messages from Retell carry an `interaction_type`:

- `call_details` and `ping_pong` for setup and keepalive;
- `update_only`, a transcript update that needs no reply;
- `response_required`, where the orchestrator has decided it is the agent's turn;
- `reminder_required`, where the user has been silent and a nudge is due.

The last two carry a monotonically increasing `response_id` and the full transcript as role/content utterances. The server streams back `response` messages with the same `response_id`, `content`, a `content_complete` flag and optional `end_call` or `transfer_number`. It can also send a `config` message enabling `auto_reconnect` and call details [35].

The `response_id` is how stale replies are dropped. If the user speaks again, Retell issues a new id, and text streamed under the old id is obsolete. This is the platform equivalent of Pipecat's `InterruptionFrame`. Retell keeps endpointing, barge-in, STT and TTS on its side, and the developer's server sees only text turns. Retell's other public repositories cover SIP tooling (forks of a Go SIP library and digest auth) and an outbound-calling simulator that shows how concurrency and calls-per-second limits shape campaigns.

## 17.10 Turn-detection techniques compared

Turn detection is where the most specialized model engineering in voice agents has happened.

| Detector | Input | Model | Size and latency | Notes |
|---|---|---|---|---|
| Silero VAD | audio | small VAD | <1 ms per 30 ms chunk, 1 CPU thread [19] | speech/silence only |
| OpenAI `server_vad` | audio | server VAD | 500 ms silence default [2] | threshold 0.5, 300 ms prefix |
| OpenAI `semantic_vad` | audio (server) | proprietary | `eagerness` (auto=medium) [2] | semantic completion |
| Gemini AAD | audio (server) | proprietary | start/end sensitivity, silence ms [12] | proactivity can stay silent |
| Pipecat Smart Turn v3 | audio (<=8 s, 16 kHz) | Whisper Tiny + linear head | ~8M params; ~10 ms to <100 ms CPU [24] | 23 languages; BSD-2; open data |
| turn-1-mini (P99Lab) | audio, streaming | causal model | 5.3M params, 1 CPU core [26] | community Pipecat plugin |
| Krisp VIVA Turn | audio, frame-by-frame | proprietary streaming | uses external VAD flag [25] | also interruption prediction [18] |
| LiveKit turn-detector | text transcript | transformer, int8 ONNX | <500 MB RAM, CPU [30] | deprecated for audio EOT |
| LiveKit TurnDetector | audio | audio EOT | cloud with local fallback, 1 s timeout [27][30] | replaces text models |
| TEN Turn Detection | text | Qwen2.5-7B based | large; GPU class [36] | finished / unfinished / wait |
| Kyutai STT semantic VAD | audio (in STT) | part of stt-1b-en_fr | 0.5 s model delay [39] | built into streaming STT |
| Transcriber-native EOT | audio (in STT) | Deepgram Flux, Cartesia | no separate TTFS [20] | server defines boundaries |

Key techniques in the sources:

- **Two-stage gating.** A cheap VAD detects a pause, and only then does a heavier semantic model decide whether the turn is over. Smart Turn is "designed to pair with a lightweight VAD such as Silero, so it runs only after silence is detected". If the user keeps talking, it reruns on the whole turn [24].
- **Audio-native input.** Smart Turn works on raw PCM "so it can use prosody cues" [24]. LiveKit moved from text to audio EOT [30].
- **Probability-to-delay mapping.** Rather than a binary decision, Vapi's smart endpointing feeds EOT probability through a sigmoid wait function [31]. LiveKit's dynamic endpointing adapts delay with an EMA [28].
- **Three-way classification.** TEN adds a `wait` class for "the user has told the AI to stop speaking", alongside finished and unfinished. On its own test set it reports 90.64% finished and 98.44% unfinished accuracy in English, and 98.90% and 92.74% in Chinese [36].
- **Backchannel awareness.** LiveKit's adaptive interruption with backchannel boundaries [28], Vapi's acknowledgement phrases [31], and Krisp's interruption prediction in Pipecat [18].

## 17.11 Open speech-native and full-duplex models

### Moshi and Delayed Streams Modeling (Kyutai)

Moshi is a full-duplex spoken dialogue framework. It models two audio streams, one for itself and one for the user, so it can listen while it speaks [37][38]. Its components:

- **Mimi**, a streaming codec that compresses 24 kHz audio into a 12.5 Hz representation at about 1.1 kbps with 80 ms frames.
- A 7B **Temporal Transformer** across time and a small **Depth Transformer** across codebooks at each time step.
- An **inner monologue**: Moshi predicts text tokens aligned with its own speech, which improves generation quality [37].

The result is 160 ms theoretical and about 200 ms practical latency on an L4 GPU [37].

```
  user audio ─► Mimi enc (12.5 Hz) ─┐
                                    ├─► Temporal Transformer (7B) ─► Depth Transformer
  Moshi audio tokens (own stream) ──┘          │ text "inner monologue"   │ per-codebook
                                               ▼                          ▼
                                       aligned text tokens        Mimi dec ─► speech
```

Kyutai generalized the approach as **Delayed Streams Modeling (DSM)**, a technique for streaming X-to-Y tasks with X and Y in {speech, text} [39][51]. Its STT models include `stt-1b-en_fr` (about 1B parameters, 0.5 s delay, built-in semantic VAD) and `stt-2.6b-en` (2.6 s delay). An H100 handles 400 real-time streams, and the Rust server handles 64 connections on an L40S at 3x real time [39]. Kyutai's **Unmute** wraps DSM STT and TTS around any OpenAI-compatible LLM with a WebSocket protocol "based on the OpenAI Realtime API". Moving STT, TTS and LLM onto separate GPUs cut TTS latency from about 750 ms on one L40S to about 450 ms [40]. This is a modular, cascaded counterpoint to Moshi from the same lab.

### PersonaPlex (NVIDIA)

PersonaPlex (2026) is a 7B full-duplex model built on Moshi's architecture and weights. It adds **text role prompts** for persona and **audio voice conditioning** through pre-packaged voice embeddings (8 "natural" and 10 "variety"). It is trained on synthetic customer-service and assistant conversations plus Fisher English corpus calls with LLM-labeled prompts [41]. It shows how full-duplex models are being adapted for agent use: role and voice control are what a contact-center deployment needs.

### Other speech-native designs

- **Sesame CSM-1B**: a Llama-family backbone plus a small audio decoder that emits Mimi RVQ codes. It conditions on conversational context for prosody, but "can't generate text" and needs a separate LLM, so it is a context-aware TTS for cascades [42].
- **Mini-Omni**: a Qwen2 backbone with a Whisper encoder and SNAC decoding that generates text and audio together ("talking while thinking"), with no separate ASR or TTS [43].
- **LLaMA-Omni**: a Whisper-large-v3 encoder and adaptor, Llama-3.1-8B-Instruct, and a unit-based vocoder with streaming synthesis. It reports 226 ms latency and was trained in under 3 days on 4 GPUs [44].
- **Qwen2.5-Omni**: a **Thinker-Talker** split (the Thinker produces text, the Talker streams speech) with TMRoPE time-aligned positions for audio and video. 3B and 7B models run on-device through MNN (Talker decoding at about 38.65 tok/s for the 7B model on Snapdragon 8 Elite) [45].
- **Ultravox**: a frozen audio encoder plus a trained projector into an LLM (default Llama 3.3 70B, with an 8B variant). It skips ASR but "currently takes in audio and emits streaming text", making it a speech-in, text-out middle ground that still needs TTS [46].

The Thinker-Talker split, Moshi's inner monologue and GPT-Live's speaker/reasoner are versions of one idea: **separate what to say (text-level reasoning) from how to say it (streaming acoustic generation)**, while keeping them time-aligned.

## 17.12 Evaluating realtime voice agents

- **Big Bench Audio** (Artificial Analysis, 2024) measures audio reasoning across speech-to-speech, speech-to-text, text-to-speech and text-to-text. It exposed the "speech reasoning gap" described in Section 17.2 [48].
- **Full-Duplex-Bench** is a four-generation benchmark [47]:
  - v1 (arXiv 2503.04721): pause handling, backchanneling, smooth turn-taking and user interruption;
  - v1.5 (2507.23159): overlap scenarios such as listener backchannels, side conversations and ambient speech;
  - v2 (2510.07838): real-time multi-turn evaluation over WebRTC or WebSocket with an automated AI examiner and LLM-as-judge;
  - v3 (2604.04847, code May 2026): multi-step tool calling under real disfluent speech (fillers, pauses, hesitations, false starts, self-corrections).

  Recent updates added Gemini 2.5 Native Audio, Gemini 3.1 Flash Live Preview and PersonaPlex [47].
- **Framework-level testing.** LiveKit ships a pytest framework that asserts on function calls and uses LLM judges [27]. Vapi offers Voice Simulations, including for GPT-Live [11]. Pipecat exposes TTFB and TTFA metrics, a user-to-bot latency observer and OpenTelemetry tracing [16][20].

## 17.13 Platform comparison

| System | Type | Transport | Turn detection | Interruption handling | Multi-agent | Notable numbers |
|---|---|---|---|---|---|---|
| OpenAI Realtime | Turn-based S2S API | WebRTC, WebSocket, SIP | server_vad / semantic_vad / manual | auto-cancel; truncate to playback | handoffs via session.update; chat-supervisor | VAD 500 ms silence, 300 ms prefix [2] |
| OpenAI GPT-Live | Full-duplex S2S API | WebSocket (+ Vapi SIP/Twilio) | in-model | in-model; tools not cancelled | speaker + reasoner delegation | reasoner effort default low (Vapi) [11] |
| Gemini Live | Turn-based S2S API | WebSocket | server AAD or client activity | START_OF_ACTIVITY_INTERRUPTS | tools; proactivity | token: 1 use, 60 s start, 30 min [12] |
| AWS Nova Sonic | Turn-based S2S API | bidirectional stream | endpointing sensitivity | provider-side | tools | ~8 min sessions; rotate at 360 s [15] |
| Pipecat | OSS framework | Daily, LiveKit, SmallWebRTC, WS, telephony | VAD + Smart Turn (default) | InterruptionFrame; spoken-only context | workers + bus; Flows | 500-800 ms typical; ~300 ms local [22][23] |
| LiveKit Agents | OSS framework + cloud | WebRTC SFU, SIP | audio EOT TurnDetector | adaptive; false-interruption resume | Agent handoff via tools | endpointing 0.5/3.0 s [28] |
| Vapi | Hosted platform | WebRTC, WS, SIP, Twilio | start plan: smart/transcriber/rules | stop plan: words, phrases, backoff | Squads; GPT-Live skills | VAD interrupt 50-100 ms [31] |
| Retell | Hosted platform | web, phone (SIP) | platform-side | response_id supersession | custom LLM server | n/a (protocol only) [35] |
| Moshi / PersonaPlex | Open full-duplex models | self-hosted WS | in-model | in-model | persona prompt | 160 ms theor., ~200 ms L4 [37] |
| Unmute | Open cascade | WebSocket | Kyutai STT semantic VAD | n/a in docs | n/a | TTS ~750 -> ~450 ms [40] |

## 17.14 Recurring failure modes

1. **Premature endpointing.** Fixed silence timers (500 ms in `server_vad`, short VAD windows) cut off users who pause mid-thought, read numbers aloud or think. Mitigations include semantic or audio EOT models [24][30], longer waits after digits (Vapi's `onNumberSeconds`) [31] and `max_delay` caps [28].
2. **Slow endpointing.** Conservative timers add hundreds of milliseconds to every turn. Pipecat's TTFS work shows that a mis-set value either cuts users off or adds dead air [20].
3. **False barge-in.** Coughs, background voices, backchannels and the agent's own echo all trigger interruptions. Mitigations include AEC warmup [29], minimum duration and word counts [28][18], backchannel classifiers [18][28], acknowledgement phrases [31] and upstream noise suppression [19].
4. **Context drift after interruption.** The model "remembers" saying things the user never heard. The fix is to commit only played text or truncate at playback position [1][18].
5. **Lost speech onset.** Clearing the input buffer on interruption drops the user's first syllables. The fix is to replay a pre-roll [7].
6. **Session limits and disconnects.** S2S sessions expire (Nova Sonic at about 8 minutes) or receive GoAway. Session rotation with audio buffering [15] and resumption handles [12] address this.
7. **Tool latency silence.** Slow tools leave dead air. Mitigations are filler preambles, async delegation that lets the speaker keep talking, and tools that return broad results [3][10].
8. **Late guardrails.** Streaming audio cannot be recalled once played. Debounced guardrails cancel the response and inject a correction [1].
9. **Reasoning gap in native audio.** S2S models have scored below text pipelines on reasoning [48]. Supervisor or delegation patterns route hard turns to text models [3][8].
10. **Tail latency.** P95 spikes, especially on multi-step tool turns, hurt perceived quality more than the median suggests [49].

## Key takeaways for system designers

- **Budget latency end to end, and count endpointing.** Vendor dashboards often show only STT + LLM + TTS medians [32]. The silence wait and STT finalization (TTFS) are often the largest controllable terms [20][21]. Track P95 and P99, not only medians [49].
- **Use two-stage turn detection.** Run a cheap VAD continuously with a short stop window (around 0.2 s) and a semantic, preferably audio-native, end-of-turn model only at pauses [19][24]. Map EOT probability to a wait time instead of a hard threshold [31][28].
- **Treat interruptions as a transaction.** On barge-in, cancel the LLM, TTS and queued audio atomically, keep non-interruptible results, and commit to context only what was played [18][1]. Add false-interruption recovery and backchannel suppression [28].
- **Pick the architecture by workload.** Cascaded pipelines win on instruction following, tool reliability, exact wording and vendor flexibility [7][48]. Turn-based S2S wins on paralinguistics and simplicity. Full duplex wins on naturalness but gives up exact speech, handoffs and some telephony features [11].
- **Split talking from thinking.** Chat-supervisor, Thinker-Talker, inner monologue and speaker/reasoner delegation all keep a low-latency voice loop in front of slower reasoning [3][8][10][37][45]. Design tools and prompts for concurrency: delegate early, return broad results, and say nothing is done until a tool confirms it [10].
- **Design for long-lived sessions.** Plan for session caps, GoAway, resumption, context compression at user-turn boundaries, and ephemeral client tokens [12][15].
- **Keep multi-agent handoff cheap.** Handoffs that swap instructions and tools inside one session (`session.update`, LiveKit Agent handoff, Vapi Squads) cut prompt size and latency without reconnecting the call [3][27][34].
- **Evaluate interactively.** Static WER and text benchmarks miss turn-taking. Use duplex benchmarks (pauses, backchannels, overlap, disfluent tool use) and simulated callers [47][27][11].

## References

1. "Realtime agents guide." OpenAI, OpenAI Agents SDK (Python) documentation, 2026. https://github.com/openai/openai-agents-python/blob/main/docs/realtime/guide.md
2. "Realtime audio input turn detection types (ServerVad, SemanticVad)." OpenAI, openai-python SDK, 2026. https://github.com/openai/openai-python/blob/main/src/openai/types/realtime/realtime_audio_input_turn_detection_param.py
3. "OpenAI Realtime Agents (chat-supervisor and sequential handoff patterns)." OpenAI, 2025. https://github.com/openai/openai-realtime-agents
4. "OpenAI Realtime Console." OpenAI, 2024-2025. https://github.com/openai/openai-realtime-console
5. "Realtime Prompting Guide." OpenAI Cookbook, 2025. https://github.com/openai/openai-cookbook/blob/main/examples/Realtime_prompting_guide.ipynb
6. "Voice pipeline." OpenAI, OpenAI Agents SDK (Python) documentation, 2025-2026. https://github.com/openai/openai-agents-python/blob/main/docs/voice/pipeline.md
7. "Building With OpenAI Audio Models and APIs" and "OpenAI Realtime" service reference. Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/features/openai-audio-models-and-apis.mdx and https://github.com/pipecat-ai/docs/blob/main/api-reference/server/services/s2s/openai.mdx
8. "OpenAI GPT-Live (OpenAILiveLLMService)." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/api-reference/server/services/s2s/openai-live.mdx
9. "GPT-Live overview." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/gpt-live/overview.mdx
10. "Design GPT-Live conversations." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/gpt-live/design.mdx
11. "GPT-Live settings and compatibility." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/gpt-live/configuration.mdx
12. "Google Gen AI Python SDK: Live API types (AutomaticActivityDetection, ActivityHandling, TurnCoverage, SessionResumptionConfig, ContextWindowCompressionConfig, ProactivityConfig, AuthToken)." Google, 2026. https://github.com/googleapis/python-genai/blob/main/google/genai/types.py
13. "Live API Web Console." Google Gemini, 2024-2025. https://github.com/google-gemini/live-api-web-console
14. "Gemini Live service reference" and "Building with Gemini Live." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/api-reference/server/services/s2s/gemini-live.mdx and https://github.com/pipecat-ai/docs/blob/main/pipecat/features/gemini-live.mdx
15. "AWS Nova Sonic service reference (session continuation)." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/api-reference/server/services/s2s/aws.mdx
16. "Pipecat: open source framework for voice and multimodal conversational AI." Daily, 2024-2026. https://github.com/pipecat-ai/pipecat
17. "Pipeline & Frame Processing." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/learn/pipeline.mdx
18. "Interruptions." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/fundamentals/interruptions.mdx
19. "Speech Input & Turn Detection." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/learn/speech-input.mdx
20. "STT Latency Tuning (TTFS)." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/fundamentals/stt-latency-tuning.mdx
21. "stt_latency.py: measured P99 TTFS per STT service." Pipecat (Daily), 2026. https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/services/stt_latency.py
22. "Overview of Pipecat." Pipecat (Daily), 2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/learn/overview.mdx
23. "Deploying Pipecat on Cerebrium." Pipecat (Daily), 2025-2026. https://github.com/pipecat-ai/docs/blob/main/pipecat/deployment/platforms/cerebrium.mdx
24. "Smart Turn (v3.x) open turn-detection model" and "Smart Turn Overview." Pipecat (Daily), 2025-2026. https://github.com/pipecat-ai/smart-turn and https://github.com/pipecat-ai/docs/blob/main/api-reference/server/utilities/turn-detection/smart-turn-overview.mdx
25. "Krisp VIVA Turn." Pipecat (Daily) / Krisp, 2026. https://github.com/pipecat-ai/docs/blob/main/api-reference/server/utilities/turn-detection/krisp-viva-turn.mdx
26. "turn-1-mini (LocalTurn1MiniAnalyzer)." P99Lab via Pipecat docs, 2026. https://github.com/pipecat-ai/docs/blob/main/api-reference/server/utilities/turn-detection/turn-1-mini.mdx
27. "LiveKit Agents: framework for realtime voice AI agents." LiveKit, 2024-2026. https://github.com/livekit/agents
28. "voice/turn.py: TurnHandlingOptions, EndpointingOptions, InterruptionOptions, PreemptiveGenerationOptions." LiveKit, 2026. https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/voice/turn.py
29. "voice/agent_session.py: AgentSession options." LiveKit, 2026. https://github.com/livekit/agents/blob/main/livekit-agents/livekit/agents/voice/agent_session.py
30. "Turn detector plugin for LiveKit Agents (README and models.py)." LiveKit, 2025-2026. https://github.com/livekit/agents/blob/main/livekit-plugins/livekit-plugins-turn-detector/README.md
31. "Voice pipeline configuration (start/stop speaking plans)." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/customization/voice-pipeline-configuration.mdx
32. "How voice-to-voice latency works in voice agents." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/assistants/model-intelligence/understanding-latency.mdx
33. "Data Flow (voice pipeline architecture)." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/security-and-privacy/data-flow.mdx
34. "Introduction to Squads." Vapi, 2026. https://github.com/VapiAI/docs/blob/main/fern/squads.mdx
35. "Retell custom LLM Python demo (WebSocket protocol types)." Retell AI, 2024-2025. https://github.com/RetellAI/retell-custom-llm-python-demo and https://github.com/RetellAI/retell-custom-llm-python-demo/blob/main/app/custom_types.py
36. "TEN Turn Detection." TEN Framework (Agora), 2025. https://github.com/TEN-framework/ten-turn-detection
37. "Moshi: a speech-text foundation model for real-time dialogue (repository)." Kyutai, 2024. https://github.com/kyutai-labs/moshi
38. Défossez et al. "Moshi: a speech-text foundation model for real-time dialogue." Kyutai, arXiv 2410.00037, 2024. https://arxiv.org/abs/2410.00037
39. "Delayed Streams Modeling: Kyutai STT & TTS." Kyutai, 2025-2026. https://github.com/kyutai-labs/delayed-streams-modeling
40. "Unmute." Kyutai, 2025. https://github.com/kyutai-labs/unmute
41. "PersonaPlex: full-duplex conversational speech model" (arXiv 2602.06053). NVIDIA, 2026. https://github.com/NVIDIA/personaplex
42. "CSM: Conversational Speech Model." Sesame AI Labs, 2025. https://github.com/SesameAILabs/csm
43. "Mini-Omni." gpt-omni, 2024. https://github.com/gpt-omni/mini-omni
44. "LLaMA-Omni." ICTNLP, 2024. https://github.com/ictnlp/LLaMA-Omni
45. "Qwen2.5-Omni." Alibaba Qwen, 2025. https://github.com/QwenLM/Qwen2.5-Omni
46. "Ultravox." Fixie.ai / Ultravox, 2024-2025. https://github.com/fixie-ai/ultravox
47. Lin et al. "Full-Duplex-Bench v1/v1.5/v2/v3." 2025-2026. https://github.com/DanielLin94144/Full-Duplex-Bench
48. "Evaluating Audio Reasoning with Big Bench Audio." Artificial Analysis (Hugging Face blog), December 2024. https://huggingface.co/blog/big-bench-audio-release
49. "Hugging Face and Cerebras bring Gemma 4 to real-time voice AI." Hugging Face, July 2026. https://huggingface.co/blog/cerebras-gemma4-voice-ai
50. "FastRTC: The Real-Time Communication Library for Python." Hugging Face, February 2025. https://huggingface.co/blog/fastrtc
51. "Delayed Streams Modeling" preprint. Kyutai, arXiv 2509.08753, 2025. https://arxiv.org/abs/2509.08753
