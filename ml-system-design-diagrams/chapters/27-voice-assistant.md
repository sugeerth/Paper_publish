# 27. Design a Voice Assistant / Speech Recognition Pipeline (Alexa, Siri)

> Source note: pages re-fetched on 2026-10-14 are marked "(read)" in References. Apple,
> Google, arXiv, Meta and Netflix domains failed DNS lookup in this run, and the search
> budget was spent; those primary sources are marked "(not re-fetched)". Rows marked
> "(assumption)" are typical interview defaults.

## Company tags

| Company / context | What links it to this problem | Source |
|---|---|---|
| Amazon (Alexa, Echo) | Product owner of the canonical "Design Alexa" system: wake word on device, cloud ASR -> NLU -> skills -> TTS | (assumption; interview reports unverified) |
| Apple (Siri, "Hey Siri") | Two-pass on-device DNN voice trigger plus speaker check, then server ASR | [1] [2] |
| Google (Assistant, Gboard voice typing) | All-neural on-device streaming RNN-T recognizer | [3] [4] |
| Walmart (voice shopping) | Unified multi-task model for several virtual assistants; voice reorder flow | [12] [19] [20] |
| Meta (SeamlessM4T) | Adjacent: multimodal speech translation model, not an assistant | [12] [18] |
| OpenAI / NVIDIA | Whisper and NeMo (Parakeet, Canary, Nemotron streaming ASR) are the usual open baselines | [8] [10] |

**Portals that cover it**

- alirezadir's MLSD guide lists "Speech Recognition Systems" as a topic (MFCC features,
  acoustic modeling, HMMs, CTC) and "Chatbot system" / "Language identification system"
  as sample questions. No full speech case study [5].
- khangich/machine-learning-interview has no speech case study [6].
- Engineer1999's case-study list has the production references (Walmart virtual
  assistants and voice reorder, Netflix speech/music detection, Meta SeamlessM4T). No
  Alexa, Siri or wake-word entry [12].
- ByteByteGo / Aminian, Exponent, Educative, Hello Interview: no dedicated speech chapter
  confirmed. Candidates apply the generic framework (see Chapter 1). (assumption)
- Open-source production toolkits act as the "solution key": WeNet [9], NeMo [10],
  openWakeWord [11], Whisper [8].

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Wake word -> transcribe -> understand intent and slots -> execute skill -> spoken reply | [1] (assumption) |
| Secondary | Multi-language, far-field (3-5 m), barge-in, follow-up mode, speaker ID / personalization | [2] (assumption) |
| Devices | 100M-500M devices. About 1B utterances/day (assumption) | (assumption) |
| QPS | About 10k average and 30-50k peak ASR streams. Each stream is long-lived over 2-10 s | (assumption) |
| Latency | Wake detect under 200 ms. Partial transcripts every 80-200 ms. End-of-speech to first TTS audio under 1 s | [10] (assumption) |
| Accuracy | WER 5-8% on assistant traffic. Wake word: FRR under 5%, FA under 0.5/hour (openWakeWord targets) | [11] |
| Power | Always-on detector runs on a DSP or low-power core at a few mW | [1] |
| Privacy | Audio stays on device until the trigger fires. Opt-in audio retention for training | [1] (assumption) |
| Vocabulary | Open vocabulary plus 100k+ contact, song and skill entities that need contextual biasing | [4] (assumption) |

## Diagram 1: End-to-end online serving path

**Figure 27.1: Online path from microphone to spoken response (cascade architecture)**

```
 DEVICE (always-on)                                   CLOUD
+----------------------+
| Mic array (6-7 mics) |
| AEC + beamforming    |
| noise suppression    |
+----------+-----------+
           | 16 kHz PCM frames (10 ms)
           v
+----------------------+   no   +---------+
| Stage-1 wake word    |------->| discard |  (audio never leaves device)
| tiny DNN on DSP      |        +---------+
+----------+-----------+
           | score > T1
           v
+----------------------+   reject
| Stage-2 checker      |----------> back to listening
| bigger DNN + speaker |
| embedding (optional) |
+----------+-----------+
           | accept: stream audio + 0.5 s pre-roll
           v
+----------------------+  gRPC/WebSocket   +--------------------------------+
| Device client        |==================>| Streaming gateway / LB         |
| VAD / endpointer     |<==================| (sticky session per stream)    |
+----------------------+  partials, TTS    | multi-device arbitration       |
                                           +---------------+----------------+
                                                           |
                       +-----------------------------------+
                       v
          +--------------------------+   +-----------------------------+
          | Cloud wake verifier      |-->| reject -> close stream      |
          | (false-accept filter)    |   +-----------------------------+
          +------------+-------------+
                       | audio frames
                       v
          +--------------------------+   +-----------------------------+
          | Streaming ASR            |<--| Context: contacts, playlist,|
          | Conformer RNN-T / CTC    |   | device state, region LM     |
          | beam search + bias FST   |   | (feature store, low-lat KV) |
          +------------+-------------+   +-----------------------------+
                       | n-best + confidence  (2nd pass: attention rescore)
                       v
          +--------------------------+
          | NLU: domain -> intent -> |
          | slots, entity resolution |
          +------------+-------------+
                       | ranked interpretations
                       v
          +--------------------------+   +-----------------------------+
          | Dialog manager / policy  |-->| Skill / action router       |
          | (context, confirmations, |   | music, timers, smart home,  |
          |  LLM fallback, safety)   |<--| search, 3rd-party skills    |
          +------------+-------------+   +-----------------------------+
                       | response text + SSML
                       v
          +--------------------------+
          | TTS (neural, streaming)  |---> audio back to device speaker
          +------------+-------------+
                       |
                       v  every stage logs (opt-in)
          +--------------------------+   +-----------------------------+
          | Event log: audio ptr,    |-->| Model store / registry      |
          | n-best, intent, latency, |   | (wake, ASR, NLU, TTS models |
          | barge-in, rephrase       |   |  pushed to cloud + device)  |
          +--------------------------+   +-----------------------------+
```

- Cascade = wake -> ASR -> NLU -> DM -> TTS. Each box can be swapped and A/B tested on its own.
- Two-pass on-device trigger with a server check follows the Apple "Hey Siri" design [1] [2].

## Diagram 2: Offline training & data pipeline

**Figure 27.2: Data flywheel for wake word, ASR and NLU models**

```
+------------------+     +-------------------+     +-------------------------+
| Opt-in audio +   |---->| De-identify, PII  |---->| Sampling: uncertain,    |
| interaction logs |     | scrub, retention  |     | rare domains, new words |
+------------------+     +-------------------+     +-----------+-------------+
                                                               |
          +----------------------------------------------------+
          v                              v                      v
+-------------------+   +---------------------------+   +--------------------+
| Human transcribe  |   | Weak / implicit labels:   |   | Synthetic data:    |
| + intent annotate |   | rephrase = ASR/NLU error, |   | TTS for new wake   |
| (gold eval sets)  |   | barge-in, task success    |   | words + entities   |
+---------+---------+   +-------------+-------------+   +---------+----------+
          |                           |                           |
          +-------------+-------------+---------------------------+
                        v
          +----------------------------+    +----------------------------+
          | Augment: RIR reverb, noise |    | Unlabeled audio pool       |
          | at 5-20 dB SNR, SpecAugment|    | -> self-supervised pretrain|
          | speed perturb              |    | (wav2vec2 / BEST-RQ style) |
          +-------------+--------------+    +-------------+--------------+
                        |                                 |
                        +----------------+----------------+
                                         v
          +---------------------------------------------------------------+
          | Training (GPU/TPU): teacher ASR -> pseudo-label -> student;   |
          | distill + quantize (int8) for device; NLU multi-task finetune |
          +-------------------------------+-------------------------------+
                                          v
          +---------------------------------------------------------------+
          | Offline eval: WER by slice (accent, far-field, noise, entity) |
          | wake FRR @ FA/hr, intent acc, slot F1, RTF, size, power       |
          +-------------------------------+-------------------------------+
                                          | pass gates
                                          v
                            +---------------------------+
                            | Model registry (versioned)|
                            +-------------+-------------+
                                          v
                            +---------------------------+
                            | Shadow -> canary -> A/B   |
                            | on live traffic (cloud)   |
                            +-------------+-------------+
                                          v
                            +---------------------------+
                            | Device OTA rollout        |
                            | (staged, kill switch)     |
                            +-------------+-------------+
                                          |
            feedback loop: new logs, errors, rephrases -> back to top (opt-in logs)
```

- Federated learning or on-device personalization can replace raw-audio upload for privacy. (assumption)
- openWakeWord trains wake models on 100% synthetic TTS positives plus about 30k hours of negatives [11].

## Diagram 3: Model architectures

**Figure 27.3a: Streaming ASR (Conformer transducer, two-pass)**

```
 audio 16 kHz
     |
     v
+--------------------+   80-dim log-mel, 25 ms window / 10 ms hop
| Feature frontend   |   stack + subsample x4 (conv)
+---------+----------+
          v
+-------------------------------+       +-----------------------------+
| Causal Conformer encoder      |       | Prediction net (LSTM/       |
| N x [FFN-MHSA(limited left    |       | stateless) over previous    |
|  context)-Conv-FFN]           |       | non-blank tokens y<u        |
+---------------+---------------+       +--------------+--------------+
                | h_t (acoustic)                       | g_u (label)
                +-------------------+------------------+
                                    v
                       +--------------------------+
                       | Joint net: tanh(W h+V g) |
                       | -> softmax over wordpiece|
                       |    vocab + <blank>       |
                       +------------+-------------+
                                    | streaming partials (RNN-T loss)
                                    v
                       +--------------------------+  bias FST: contacts,
                       | Beam search + shallow    |<- songs, app names
                       | fusion / contextual bias |
                       +------------+-------------+
                                    | n-best at endpoint
                                    v
                       +--------------------------+
                       | 2nd pass: full-context   |
                       | attention / LAS rescorer |
                       +------------+-------------+
                                    v
                       final transcript + confidence
```

**Figure 27.3b: Wake word detector and NLU heads**

```
 WAKE WORD (on device)                     NLU (cloud)
+---------------------+             +------------------------------+
| log-mel frames      |             | ASR 1-best / n-best tokens   |
| (~1 s sliding win)  |             | + device context             |
+----------+----------+             +--------------+---------------+
           v                                       v
+---------------------+             +------------------------------+
| small CNN/DS-CNN or |             | Transformer encoder (BERT-   |
| frozen embedding    |             | style, distilled)            |
| backbone            |             +------+---------------+-------+
+----------+----------+                    |               |
           v                               v               v
+---------------------+             +------------+  +----------------+
| classifier head:    |             | [CLS] head |  | token head:    |
| P(wake) per frame   |             | domain +   |  | BIO slot tags  |
| + smoothing window  |             | intent     |  | (CRF optional) |
+----------+----------+             +-----+------+  +-------+--------+
           v                              +--------+--------+
  threshold T1 -> stage 2                          v
                                    entity resolution -> catalog IDs
```

## Diagram 4: Alternative - end-to-end / LLM speech assistant

**Figure 27.4: Cascade vs. speech-LLM design**

```
 CASCADE (today's production default)
 audio -> [ASR] -> text -> [NLU] -> intent -> [DM + skills] -> text -> [TTS] -> audio
          errors compound; each stage tunable, debuggable, cheap

 SPEECH-LLM / E2E (emerging)
+---------+   +-----------------+   +-------------------------+   +-----------+
| audio   |-->| speech encoder  |-->| adapter -> LLM decoder  |-->| tool call |
| frames  |   | (Whisper-style  |   | (text or speech tokens) |   | or speech |
+---------+   |  or Conformer)  |   | + RAG / tools / memory  |   | tokens    |
              +-----------------+   +------------+------------+   +-----+-----+
                                                 |                      |
                                                 v                      v
                                     safety filter + grounding    streaming TTS /
                                                                  codec vocoder
 Tradeoffs: keeps prosody and context, fewer handoffs  <->  higher latency and cost,
            harder to bias to entities, harder to debug which stage failed
```

- Whisper is an encoder-decoder Transformer that handles transcription, translation,
  language ID and VAD as token tasks. Model sizes run from 39M to 1.55B parameters [8].
- A common hybrid: keep the on-device wake word and streaming ASR, and route long-tail
  queries that NLU rejects to an LLM. (assumption)

## Data & labels

| Model | Positive / label | Negative / hard negatives | Source of labels |
|---|---|---|---|
| Wake word | Utterances of "Alexa" / "Hey Siri" across accents and distances | TV, music, similar words ("election", "Alexis") | Accepted triggers + human audit, synthetic TTS [11] |
| ASR | Audio -> verbatim transcript | n/a | Human transcription, pseudo-labels from a teacher, weak labels |
| NLU | Utterance -> domain, intent, slot spans | Out-of-domain utterances | Annotators, grammar-generated data, user corrections |
| Endpointer | End-of-query frame | Mid-sentence pauses | Forced alignment of transcripts |
| Implicit | Task success, no rephrase within 30 s | Barge-in, "no, I said..." | Interaction logs (assumption) |

## Features

| Stage | Features |
|---|---|
| Frontend | Log-mel filterbanks (80-dim), beamformed channel, AEC residual |
| Wake word | Frame posteriors, speaker embedding (d-vector), device playback state |
| ASR context | Contact list, recent media, installed skills, locale, device type (biasing) |
| NLU | Tokens, ASR confidence, n-best, previous turn intent, device capabilities, time |
| Skill ranking | Skill popularity, user history, entity match score, latency |

## Model choices

| Component | Baseline | Advanced | Tradeoff |
|---|---|---|---|
| Wake word | GMM-HMM keyword / small DNN | DS-CNN two-stage + speaker verification [1] [2] | Size and power vs. false accepts |
| ASR | Hybrid DNN-HMM + n-gram LM + WFST | Streaming Conformer RNN-T + 2nd-pass rescorer [3] [4] [7] [13] | E2E is smaller and fits on device but is harder to bias |
| ASR (offline) | CTC encoder | Attention enc-dec (Whisper) [8] | Accuracy vs. no streaming |
| Unified | Separate streaming / full models | U2 / WeNet: one model, CTC + attention rescoring [9] | One model to maintain |
| NLU | Per-domain CRF + logistic regression | Joint BERT intent + slot, multi-task [16] [19] | Latency vs. accuracy |
| DM | Rules + state machine | Learned policy / LLM fallback | Control vs. coverage |
| TTS | Concatenative | Neural (Tacotron/FastSpeech + vocoder) | Naturalness vs. compute |

## Loss functions

| Model | Loss |
|---|---|
| Augmentation | SpecAugment [14]; SSL pretraining (wav2vec 2.0 [15]) |
| Wake word | Frame-level cross-entropy, or max-pooling loss over the keyword window |
| ASR | RNN-T loss (marginalizes over alignments). CTC as auxiliary. CE for the attention decoder. MWER fine-tuning |
| Distillation | KL(teacher logits, student logits) + RNN-T |
| NLU | CE(intent) + token CE or CRF NLL(slots), weighted sum |
| Speaker ID | GE2E / triplet loss on embeddings |

## Offline metrics

| Component | Metric |
|---|---|
| Wake word | FRR at a fixed FA/hour (DET curve), detection latency |
| ASR | WER, entity/named-WER, sliced by accent, noise, far-field, domain. Real-time factor |
| Endpointer | EP50/EP90 latency, early-cutoff rate |
| NLU | Intent accuracy, slot F1, semantic error rate (SemER) |
| End-to-end | Task completion on replayed golden set, latency p50/p90/p99 |

## Online metrics / A-B

| Metric | Meaning |
|---|---|
| Task success / user satisfaction (predicted) | North star. Model-estimated from implicit signals |
| Rephrase / repeat rate | Proxy for ASR or NLU error |
| Barge-in / abandonment | Wrong or slow response |
| False wake rate per device-day | Privacy and annoyance guardrail |
| Latency (user-perceived) | End-of-speech -> first audio |
| Engagement | Daily active devices, queries/device |

## Serving & latency budget

| Step | Budget (assumption) | Where |
|---|---|---|
| Wake word stage 1 + 2 | 50-200 ms after end of keyword | Device DSP / AP |
| Network uplink | 50-100 ms | Edge PoP |
| Streaming ASR | Partials in about 150 ms. Final 100-200 ms after endpoint | GPU/CPU pool. Batched streams |
| Endpointing | 200-500 ms of trailing silence (adaptive) | Device or cloud |
| NLU + entity resolution | 30-50 ms | CPU |
| DM + skill call | 100-300 ms (third-party skill timeout) | Microservices |
| TTS first chunk | 50-150 ms | GPU, streaming |
| Total | Under 1 s from end-of-speech to first audio | |

- On-device ASR removes the network hop and works offline. Google shipped an all-neural
  on-device RNN-T recognizer for Gboard (2019, about 80 MB after quantization) [3] [4].
- NVIDIA Nemotron-3.5-ASR-Streaming (cache-aware FastConformer) exposes a tunable latency
  of 80 ms to 1 s and 240-2400 concurrent streams per H100 [10].

## Monitoring

| Signal | Alert on |
|---|---|
| WER on daily human-transcribed sample | Regression by locale or device |
| False wake rate | Spikes. A TV ad saying "Alexa" causes mass false wakes |
| Rephrase rate per intent | New intent failures or a catalog change |
| Latency p99 per stage | Drift caused by a skill or GPU saturation |
| Input drift | SNR distribution, language mix, new device firmware |
| Out-of-vocabulary entities | New song or celebrity names (biasing lists need refresh) |

## Failure modes

| Failure | Mitigation |
|---|---|
| False accepts from media / ads | Cloud verifier, audio fingerprint blocklist of broadcast ads, speaker ID |
| Accent / dialect bias in WER | Slice metrics, targeted data collection, per-locale models |
| Rare entity misrecognition | Contextual biasing FST, personal LM, NLU fuzzy entity resolution |
| Early endpointing cuts the user off | Learned endpointer with a semantic completeness signal |
| ASR error cascades into wrong action | Pass n-best + confidence to NLU. Confirm destructive or purchase actions |
| Privacy incidents (accidental recordings) | On-device processing, retention limits, opt-in review |
| Cold start for new language | Multilingual pretrain + SSL, synthetic TTS data |

## Interviewer follow-ups

- How do you keep false wakes under X per day without hurting recall? Where does each stage run?
- Why use two-stage wake word detection instead of one larger model? What is the power budget?
- What happens when a TV ad says the wake word to millions of devices at once?
- Hybrid HMM vs. CTC vs. RNN-T vs. attention: which works for streaming, and why?
- How do you recognize a user's contact names or a song released yesterday (contextual biasing)?
- How do you decide the user has finished speaking (endpointing), and what does it trade off?
- What runs on device and what in the cloud? Offline mode? Model compression (int8, pruning, distillation)?
- How do you get labels at scale without transcribing everything? (Pseudo-labels, implicit feedback, SSL)
- How do you evaluate fairness across accents and dialects?
- How do you propagate ASR uncertainty into NLU?
- How would you add a new language with only 100 hours of labeled data?
- How do you A/B test an ASR model when the north-star metric is task success?
- How do you personalize to a household's voices while protecting privacy (federated learning)?
- Would you replace the cascade with a speech-LLM? Latency, cost and controllability?
- How do you scale GPU serving for long-lived streaming connections (stickiness, batching)?

## Components often missed

| Component | Why it matters | Source |
|---|---|---|
| Device arbitration | Several Echo devices hear one wake word; only the best-SNR device answers | (assumption) |
| Device-directed speech detection | Follow-up mode without the wake word needs a "talking to me?" classifier | (assumption) |
| Language ID | Picks the ASR model in multilingual homes | [5] [8] |
| Voice activity detection | Gates ASR compute and drives endpointing | [8] |
| Punctuation, casing, inverse text normalization | "set alarm for seven thirty" -> "7:30" before NLU and display | [10] (assumption) |
| Acoustic features and HMM/CTC basics | Portal syllabus asks for MFCCs, acoustic models, CTC | [5] |
| Streaming GPU capacity | Concurrent streams per GPU sets fleet size | [10] |

Extra follow-ups: How do you pick which device answers when three Echos hear "Alexa"?
How do you support follow-up questions without the wake word and without false accepts?

## Portal comparison

| Source | Framing | Distinctive emphasis |
|---|---|---|
| Apple ML Research (Hey Siri) [1] [2] | Voice trigger as the core problem | Two-pass on-device DNN, thresholds, speaker vector, false-trigger mitigation |
| Google Research / He et al. [3] [4] | On-device streaming ASR | RNN-T, quantization, latency vs. server models |
| OpenAI Whisper [8] | Single multitask model | Weak supervision at scale, robust offline transcription, not streaming |
| WeNet / NeMo [9] [10] | Production toolkits | Unified streaming/non-streaming, CTC + attention rescoring, latency knobs |
| openWakeWord [11] | Open wake word recipe | Frozen embedding backbone, synthetic positives, FA/hr and FRR targets |
| alirezadir / generic MLSD portals [5] | NLP topic list (speech recognition, chatbot, language ID) | Feature extraction, acoustic modeling, HMM, CTC; no end-to-end assistant design |

## References

1. "Hey Siri: An On-device DNN-powered Voice Trigger for Apple's Personal Assistant", Apple Machine Learning Research, 2017. https://machinelearning.apple.com/research/hey-siri (not re-fetched)
2. "Personalized Hey Siri", Apple Machine Learning Research, 2018. https://machinelearning.apple.com/research/personalized-hey-siri (not re-fetched)
3. "An All-Neural On-Device Speech Recognizer", Google AI Blog, 2019. https://ai.googleblog.com/2019/03/an-all-neural-on-device-speech.html (not re-fetched)
4. He et al., "Streaming End-to-end Speech Recognition For Mobile Devices", arXiv / ICASSP, 2018-2019. https://arxiv.org/abs/1811.06621 (not re-fetched)
5. alirezadir, "Machine-Learning-Interviews: ML System Design", GitHub, 2024. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (read 2026-10-14)
6. khangich, "machine-learning-interview", GitHub, 2023. https://github.com/khangich/machine-learning-interview (read 2026-10-14)
7. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition", arXiv, 2020. https://arxiv.org/abs/2005.08100 (not re-fetched)
8. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), OpenAI / GitHub, 2022. https://github.com/openai/whisper (read 2026-10-14) and https://arxiv.org/abs/2212.04356 (not re-fetched)
9. WeNet, "Production First and Production Ready End-to-End Speech Recognition Toolkit", GitHub, 2021-2022. https://github.com/wenet-e2e/wenet (read 2026-10-14). WeNet 2.0 paper: https://arxiv.org/abs/2203.15455 (not re-fetched)
10. NVIDIA, "NeMo" (ASR: Parakeet, Canary, Nemotron-3.5-ASR-Streaming cache-aware FastConformer), GitHub, 2025-2026. https://github.com/NVIDIA/NeMo (read 2026-10-14)
11. dscripka, "openWakeWord", GitHub, 2023. https://github.com/dscripka/openWakeWord (read 2026-10-14)
12. Engineer1999, "A Curated List of ML System Design Case Studies" (Netflix speech/music detection, Meta SeamlessM4T, Walmart unified multi-task virtual-assistant model), GitHub, 2024. https://github.com/Engineer1999/A-Curated-List-of-ML-System-Design-Case-Studies (read 2026-10-14)
13. Graves, "Sequence Transduction with Recurrent Neural Networks" (RNN-T), arXiv, 2012. https://arxiv.org/abs/1211.3711 (not re-fetched)
14. Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", arXiv, 2019. https://arxiv.org/abs/1904.08779 (not re-fetched)
15. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", arXiv, 2020. https://arxiv.org/abs/2006.11477 (not re-fetched)
16. Chen et al., "BERT for Joint Intent Classification and Slot Filling", arXiv, 2019. https://arxiv.org/abs/1902.10909 (not re-fetched)
17. Netflix Tech Blog, "Detecting Speech and Music in Audio Content", 2023. https://netflixtechblog.com/detecting-speech-and-music-in-audio-content-afd64e6a5bf8 (listed in [12]; not re-fetched)
18. Meta AI, "SeamlessM4T: foundational multimodal model for speech translation", 2023. https://ai.meta.com/blog/seamless-m4t/ (listed in [12]; not re-fetched)
19. Walmart Global Tech, "A Unified Multi-task Model for Supporting Multiple Virtual Assistants in Walmart", Medium, 2022. https://medium.com/walmartglobaltech/a-unified-multi-task-model-for-supporting-multiple-virtual-assistants-in-walmart-2b077c2c96e (listed in [12]; not re-fetched)
20. Walmart Global Tech, "Voice Reorder Experience: add Multiple Product Items to your shopping cart", Medium, 2022. https://medium.com/walmartglobaltech/voice-reorder-experience-add-multiple-product-items-to-your-shopping-cart-59d20fc61797 (listed in [12]; not re-fetched)
