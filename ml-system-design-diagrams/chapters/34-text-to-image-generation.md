# 34. Design a Text-to-Image Generation Service (Midjourney, DALL-E) with Safety and Serving

> Turn a text prompt (plus optional reference image, style, or aspect ratio) into
> 1-4 high-quality images in seconds. The service must stay safe at both the prompt and
> the pixel level, and cost cannot run away. In this design the "candidates" are N sampled
> images. "Ranking" means choosing among them by aesthetic and alignment scores, and
> "policy" covers safety classifiers, watermarking and provenance.

> Research note: pages re-fetched in review: [1] [2] [3] [31] [32] [36] [37]. arXiv,
> openai.com, bytebytego.com and c2pa.org were unreachable from the sandbox; those entries
> are well-known papers or pages cited from prior knowledge and marked as such.

## Company tags

| Company / product | Why it is tagged | Source |
|---|---|---|
| OpenAI (DALL-E 2/3, GPT image) | Builds and documents this exact system (system cards) | [1] [8] |
| Midjourney | Product named in the common framing of this problem ("design Midjourney") | [29] |
| Google (Imagen), Meta (Emu) | Ship production text-to-image models. Product-based tag, not a verified question report | [6] [41] |
| Stability AI | Open reference stack: SD 1.x, SDXL, Turbo | [2] [3] [7] |
| Generic GenAI / MLE loops (FAANG, AI labs) | Repo prep guides list "Image-generation pipeline" as a role-specific GenAI design topic | [32] |

Verified company reports (LeetCode Discuss, Blind) could not be gathered in this run. Treat
the rows above as product-based tags.

**Portals that cover it**

| Portal | Coverage | Link |
|---|---|---|
| ByteByteGo, Aminian & Sheng, "Generative AI System Design Interview" | Dedicated "Text-to-Image Generation" chapter, plus neighbouring chapters on high-res synthesis and personalized headshots | [29] |
| Educative, "Grokking the Generative AI System Design" | Text, image, speech and video generation case studies | [30] [31] |
| alirezadir / Agentic-AI-Systems interview prep | Lists "Image-generation pipeline" (Tier 3, role-specific) | [32] |
| Chip Huyen, "Building a GenAI Platform" | Generic GenAI serving stack (guardrails, caching, routing); not text-to-image specific | [35] |

## Requirements & scale

| Dimension | Requirement | Note / source |
|---|---|---|
| Functional | Text prompt -> 1-4 images. Options: aspect ratio, style, seed, negative prompt, variations, upscale, inpaint / outpaint with a mask, image prompt | DALL-E 2 offered inpainting and variations [1]. Others are typical |
| Resolution | 1024x1024 native, 2-4x upscale optional | SDXL trained at 1024^2, multi-aspect [3] [7] |
| Latency | Interactive: p50 ~5-10 s for 4 images, preview at ~1 s. Turbo mode: under 1 s | Assumption. 1-4 step distilled models [12] [13] |
| Throughput | 10M images/day ~ 115 img/s average, ~3-5x at peak | Typical assumption |
| Users | 10-20M MAU, heavy-tailed usage, paid tiers | Assumption |
| Safety | Block CSAM, sexual, violent, hateful, public-figure and deceptive content at prompt and output | [1] |
| Provenance | Invisible watermark plus C2PA metadata on every image | [2] [3] [28] [40] |
| Cost | GPU-seconds per image is the main cost lever. Sell GPU time by tier (fast / relax / turbo style queues) | Assumption (Midjourney-style tiers) |
| Availability | 99.9%. Degrade to fewer steps or images under load | Assumption |
| Abuse | Per-user rate limits, anomaly detection, account enforcement | [1] |

## DIAGRAM 1: Online serving path

**Figure 34.1: End-to-end request path (prompt to safe, watermarked images)**

```
 +--------+ prompt, opts +-------------+ authn, quota +---------------------+
 | Client |------------->| API Gateway |------------->| Rate limiter / tier |
 +--------+              +-------------+              +----------+----------+
    ^                                                            |
    | signed URLs / refusals                                     |
    |                                            +---------------+
    |                                            v
    |        +------------------------------------------------------------+
    |        | 1 PROMPT SAFETY: blocklist -> multilingual text classifier |
    |        |   -> LLM policy judge (public figures, minors, CSAM terms) |
    |        +---------------------+-------------+------------------------+
    |<-- refusal (block) ----------+             | allow
    |                                            v
    |        +------------------------------------------------------------+
    |        | 2 PROMPT REWRITE (LLM): expand short prompt, add style,    |
    |        |   strip living-artist / real-person names                  |
    |        +------------------------------------------------------------+
    |                                            | enriched prompt
    |                                            v
    |        +------------------------------------------------------------+
    |        | 3 RESULT CACHE lookup: hash(prompt, seed, opts, model ver) |
    |        +---------------------+-------------+------------------------+
    |<-- cached URLs (hit) --------+             | miss
    |                                            v
    |        +------------------------------------------------------------+
    |        | 4 GPU QUEUE + SCHEDULER: priority by tier (fast / relax),  |
    |        |   batch by resolution + steps, preempt free tier           |
    |        +------------------------------------------------------------+
    |                                            |
    |                                            v
    |        +------------------------------------------------------------+
    |        | 5 GENERATION WORKER (GPU, weights from model registry:     |
    |        |   base / LoRA / VAE): text encode -> N noise latents ->    |
    |        |   denoise T steps with CFG -> VAE decode -> N candidates   |
    |        +------------------------------------------------------------+
    |                                            | N = 4..8 candidates
    |                                            v
    |        +------------------------------------------------------------+
    |        | 6 OUTPUT SAFETY: NSFW / violence / gore classifiers,       |
    |        |   CSAM hash match, face / public-figure check;             |
    |        |   drop unsafe candidates (all unsafe -> refusal)           |
    |        +------------------------------------------------------------+
    |                                            |
    |                                            v
    |        +------------------------------------------------------------+
    |        | 7 RE-RANK: aesthetic + CLIP / PickScore alignment,         |
    |        |   near-dup removal, keep top-k (k = 4)                     |
    |        +------------------------------------------------------------+
    |                                            |
    |                                            v
    |        +------------------------------------------------------------+
    |        | 8 PROVENANCE: invisible watermark + signed C2PA manifest;  |
    |        |   optional 2x-4x upscaler                                  |
    |        +------------------------------------------------------------+
    |                                            |
    |                                            v
    |        +------------------------------------------------------------+
    |        | 9 OBJECT STORE + CDN: write images, issue signed URLs      |
    |        +---------------------+-------------+------------------------+
    |<-----------------------------+             | events
                                                 v
             +------------------------------------------------------------+
             | LOGGING: prompt, rewrite, seed, model ver, safety scores,  |
             | latency, user actions (upscale / download / reroll /       |
             | report) -> data lake -> Figure 34.2 feedback loop          |
             +------------------------------------------------------------+
```

- Steps 1 and 6 follow the DALL-E pattern: input filters on prompts and uploads, then output filters, with independent filters as a known gap [1]. The SD reference script also ships a post-hoc safety checker [2]
- Step 2 (LLM prompt rewrite) is the DALL-E 3 pattern: descriptive captions at training time, upsampled prompts at inference [8]
- Step 3 only hits for an identical prompt + seed + options + model version; diffusion is deterministic given the seed
- Step 8: the SD reference stack applies an invisible watermark to every output [2] [3]; C2PA adds a signed manifest [28]

## DIAGRAM 2: Offline data and training pipeline

**Figure 34.2: Data curation, training, alignment, eval and deployment loop**

```
 +------------------------+    +------------------------+    +------------------------+
 | Web image-alt-text     |--->| FILTERS: NSFW / gore   |--->| RECAPTION: VLM writes  |
 | pairs (LAION-scale)    |    | classifiers, hate-sym. |    | dense captions; blend  |
 | + licensed stock       |    | captions, dedupe,      |    | with alt-text (~95%    |
 |                        |    | watermark prob., size, |    | synthetic / 5% orig.)  |
 |                        |    | aesthetic score > 5    |    |                        |
 +------------------------+    +------------------------+    +------------+-----------+
                                                                          |
                                                                          v
 +------------------------+    +------------------------+    +------------------------+
 | PRE-COMPUTE text       |<---| SHARDED DATASET        |<---| BUCKET by aspect ratio |
 | embeddings + VAE       |    | (webdataset /          |    | and resolution         |
 | latents (cache)        |    | parquet)               |    | (256 -> 512 -> 1024)   |
 +------------+-----------+    +------------------------+    +------------------------+
              |
              v
 +------------------------+    +------------------------+    +------------------------+
 | PRETRAIN diffusion /   |--->| QUALITY FINE-TUNE on   |--->| PREFERENCE ALIGN:      |
 | flow model; drop text  |    | a few thousand curated |    | reward model (Pick-    |
 | 10% of the time (CFG)  |    | photogenic images      |    | Score / ImageReward)   |
 |                        |    |                        |    | + Diffusion-DPO / RL   |
 +------------------------+    +------------------------+    +------------+-----------+
                                                                          |
                                                                          v
 +------------------------+    +------------------------+    +------------------------+
 | MODEL REGISTRY         |<---| EVAL GATE (pass/fail): |<---| DISTILL turbo variant  |
 | (versioned weights,    |    | FID, CLIPScore, T2I-   |    | (ADD / LCM, 1-4 steps) |
 | model + system card)   |    | CompBench, human SxS,  |    | + OFFLINE EVAL suites  |
 |                        |    | red-team ASR, bias     |    | (auto, human, safety)  |
 +------------+-----------+    +------------------------+    +------------------------+
              |
              v
 +------------------------+    +------------------------+    +------------------------+
 | CANARY / A-B on 1-5%   |--->| PROD FLEET             |--->| FEEDBACK LOGS: picks,  |
 | of traffic, guardrail  |    | (Figure 34.1)          |    | upscales, rerolls,     |
 | metrics                |    |                        |    | reports, appeals       |
 +------------------------+    +------------------------+    +------------+-----------+
                                                                          |
                                                                          v
               new preference pairs + safety labels -> FILTERS, RECAPTION, ALIGN
```

- Data filtering: NSFW/violence classifiers plus hate-symbol caption filters. Filtering is tuned to avoid demographic skew (DALL-E 2 found the filter reduced images of women) [1]
- SD v1.2-1.4 trained on LAION-Aesthetics (aesthetic > 5.0, size >= 512, low watermark prob.) and dropped the text 10% of the time to enable CFG [2] [16]
- Recaptioning: DALL-E 3 trained on synthetic descriptive captions [8]. Quality tuning on a small curated set: Meta Emu [41]

## DIAGRAM 3: Model architecture (latent diffusion with cross-attention)

**Figure 34.3: Latent diffusion / DiT generator with classifier-free guidance**

```
 prompt "a corgi astronaut, oil painting"       negative prompt, or "" (null)
            |                                            |
            v                                            v
 +-------------------------+                  +-------------------------+
 | TEXT ENCODER (frozen)   |                  | same text encoder       |
 | CLIP ViT-L / OpenCLIP   |                  +------------+------------+
 | ViT-G, and/or T5-XXL    |                               | c_null
 +------------+------------+                               |
              | c_text (L tokens x d)                      |
              v                                            v
 +--------------------------------------------------------------------------+
 | DENOISER eps_theta(z_t, t, c)                 (repeat for t = T ... 1)   |
 |   U-Net (~860M in SD1) or DiT / MM-DiT transformer                       |
 |   z_t (4 x 128 x 128) -> ResBlocks + self-attn                           |
 |                       -> CROSS-ATTN (Q = image latents, K/V = text)      |
 |   CFG: eps = eps(z, c_null) + w * (eps(z, c_text) - eps(z, c_null))      |
 |        w ~ 5-8 ; sampler = DDIM / DPM-Solver / flow ODE                  |
 +-------+------------------------------------+-----------------------------+
         ^                                    |
         | z_T ~ N(0, I) from seed            | z_0 (clean latent)
         |                                    v
 +-------+--------+              +--------------------------+   +-------------+
 | noise sampler  |              | VAE DECODER (8x upsample)|-->| image       |
 | (seeded RNG)   |              | 4x128x128 -> 3x1024x1024 |   | 1024 x 1024 |
 +----------------+              +--------------------------+   +------+------+
                                                                       |
                                                                       v
                          optional: refiner (img2img, low noise), super-res
                          diffusion, ControlNet branch, LoRA style adapters

 TRAINING: x -> VAE encoder -> z_0 ;  z_t = a_t * z_0 + s_t * eps ;
           loss = || eps - eps_theta(z_t, t, c) ||^2   (or v / flow target)
```

- LDM: diffusion in an 8x-downsampled VAE latent with cross-attention conditioning [2] [4]. SD1 uses an ~860M U-Net and a ~123M frozen CLIP ViT-L/14 [2]
- SDXL: two text encoders (OpenCLIP ViT-G + CLIP ViT-L), 1024^2 multi-aspect training, plus a refiner used only as img2img [3] [7]
- CFG [9]; DDPM/DDIM [10] [11]; DiT [14]; MM-DiT with rectified flow (SD3) [15]

## DIAGRAM 4: Alternatives (pixel cascade vs latent vs distilled)

**Figure 34.4: Three production generator families and where each one fits**

```
 A) PIXEL CASCADE (Imagen; DALL-E 2 decoder)
 +------------+   +----------------+   +-----------------+   +------------------+
 | T5-XXL or  |-->| base diffusion |-->| SR diffusion    |-->| SR diffusion     |
 | CLIP prior |   | 64 x 64        |   | 64 -> 256       |   | 256 -> 1024      |
 +------------+   +----------------+   +-----------------+   +------------------+
   + strong text understanding (big LM encoder)   - 3 models, high latency/cost

 B) LATENT (Stable Diffusion / SDXL / SD3)
 +------------+   +-------------------------------+   +----------------+
 | CLIP / T5  |-->| U-Net / MM-DiT in VAE latent  |-->| VAE decode     |--> 1024^2
 +------------+   | 20-50 steps                   |   | (+ refiner)    |
                  +-------------------------------+   +----------------+
   + ~48x fewer values per step, one model   - VAE artifacts (small text, faces)

 C) DISTILLED / TURBO (SDXL-Turbo via ADD, LCM)
 +------------+   +-------------------------------+
 | teacher    |-->| student: 1-4 steps, no CFG    |--> real-time preview, free tier
 | (B)        |   | adversarial / consistency     |
 +------------+   +-------------------------------+
   + order-of-magnitude fewer steps   - lower diversity, weaker prompt following
```

- Imagen: frozen T5-XXL plus a 64 -> 256 -> 1024 cascade [6]. DALL-E 2: CLIP prior plus diffusion decoder plus 2 upsamplers [5]
- SDXL-Turbo uses adversarial diffusion distillation [3] [12]. LCM [13]. Guided distillation [39]

## Data & labels

| Data | Source | Label / signal | Use | Ref |
|---|---|---|---|---|
| Image-alt-text pairs | Web crawl (LAION-5B scale), licensed stock | Weak caption | Pretraining | [16] |
| Synthetic dense captions | VLM captioner | Descriptive caption | Better prompt following | [8] |
| Curated high-aesthetic set | Human-selected | "Photogenic" yes/no | Quality fine-tune | [41] |
| Pairwise preferences | Users choose 1 of N (upscale/download) | Win/loss pair | Reward model, DPO | [19] [27] |
| Expert ratings | Annotators | 1-7 alignment/fidelity | ImageReward-style RM | [20] |
| Safety labels | Trust & safety raters, red teams | Policy category | Prompt/image classifiers | [1] [22] |
| Abuse reports | User flags | Violation | Enforcement, classifier retraining | [1] |

## Features / conditioning inputs

| Input | Encoding | Notes |
|---|---|---|
| Prompt text | CLIP / OpenCLIP / T5 token embeddings | CLIP [23]; T5 improves spelling and composition [6] [15] |
| Negative prompt | Same encoder, replaces null in CFG | User control |
| Size / aspect / crop coords | Micro-conditioning embeddings | SDXL conditioning trick [7] |
| Seed | Initial noise z_T | Reproducibility, variations |
| Reference image / pose / edges | VAE latent, ControlNet branch | [25] |
| Style / subject | LoRA or DreamBooth adapters | Personalization [24] |
| Safety context | User tier, region, history | Gate thresholds, not fed to the generator |

## Model choices (baseline -> advanced)

| Stage | Option | Pros | Cons |
|---|---|---|---|
| Baseline | GAN (StyleGAN-T) | 1-step, fast | Mode collapse, weak text alignment |
| Baseline 2 | Autoregressive over VQ tokens (DALL-E 1, Parti) | Reuses LLM infra | Slow token-by-token decode |
| Standard | Latent diffusion U-Net + CLIP [4] | Open, efficient, mature | Weak at text and counting |
| Advanced | MM-DiT + rectified flow + T5 [15] | Scales well, better typography | Bigger, needs more GPU |
| Pixel cascade | GLIDE / Imagen / unCLIP [38] [6] [5] | Strong alignment | 3 models, cost |
| Fast path | ADD / LCM distillation [12] [13] | 1-4 steps | Lower diversity |
| Alignment | Diffusion-DPO / reward fine-tune [27] | Matches user taste | Reward hacking |
| Safety-in-model | Safe Latent Diffusion guidance [21], concept erasure fine-tune [43] | Survives removal of external filters | Over-suppresses, collateral concept damage |

## Loss

| Component | Loss | Ref |
|---|---|---|
| Denoiser | MSE on epsilon (or v / flow velocity) at random t | [10] [15] |
| CFG enabler | Drop text 10% of the time (unconditional branch) | [2] [9] |
| VAE | L1/perceptual + KL + patch-GAN | [4] |
| Reward model | Bradley-Terry pairwise logistic on preferences | [19] [20] |
| Preference tuning | Diffusion-DPO objective | [27] |
| Distillation | Adversarial + score distillation (ADD) / consistency | [12] [13] |
| Safety classifiers | Multi-label BCE, recall-weighted | Assumption |

## Offline metrics

| Metric | Measures | Caveat | Ref |
|---|---|---|---|
| FID (COCO-30k zero-shot) | Realism / distribution match | Poor match with human preference | [17] |
| Inception Score | Quality and diversity | ImageNet-biased | [33] |
| CLIPScore | Prompt-image alignment | Saturates, bag-of-words | [18] |
| PickScore / ImageReward | Predicted human preference | Can be gamed | [19] [20] |
| T2I-CompBench, GenEval | Attributes, counting, position, colour binding (GenEval uses object detectors) | Narrow prompts | [26] [42] |
| Prompt suites for human eval | DrawBench (Imagen), PartiPrompts (Parti) | Fixed prompts can be overfit | [6] [34] |
| Memorization rate | Near-duplicate outputs vs training set for sampled prompts | Needs dedup index over training data | [45] |
| Human side-by-side | Win rate on aesthetics, alignment, artifacts | Expensive. The final gate | [7] [8] |
| Safety eval | Attack success rate on red-team prompt suites, NSFW leak rate | Needs adversarial refresh | [1] [22] |
| Bias audit | Demographic distribution for neutral prompts | Requires a labelled audit set | [1] |

## Online metrics / A-B

| Metric | Type |
|---|---|
| Download / upscale / share rate per generation | Primary engagement |
| Reroll rate, prompt edit rate | Dissatisfaction (lower is better) |
| Pairwise pick rate (new vs old model shown side by side) | Model preference |
| p50/p95 time-to-first-image, queue wait | Latency guardrail |
| GPU-seconds per kept image | Cost guardrail |
| Prompt block rate, false-block appeals, user reports per 1k images | Safety guardrails |
| Paid conversion, retention D7/D30 | Business |

## Serving & latency budget (4 images, 1024^2, standard mode)

| Stage | Budget | Technique |
|---|---|---|
| Gateway, auth, quota | 20 ms | Edge |
| Prompt safety (blocklist + classifier + LLM judge) | 100-300 ms | Small models, cached verdicts |
| LLM prompt rewrite | 300-800 ms | Small LLM, run in parallel with safety, cancel on block |
| Queue wait | 0-several s | Priority queues by tier, autoscale on queue depth |
| Text encode | 10-30 ms | Cache by prompt hash |
| Denoise (25-30 steps, batch 4, CFG batched x2) | 2-6 s on A100/H100 class GPUs | fp16/bf16, FlashAttention, TensorRT/torch.compile, DPM-Solver |
| VAE decode | 100-300 ms | Tiled decode |
| Output safety + re-rank | 50-150 ms | Batched CLIP-based heads |
| Watermark + C2PA + upload | 50-100 ms | Async upload to CDN |
| Streaming previews | Every k steps | Latent preview decoder for perceived latency |

- Cost levers: fewer steps (distilled model for free tier), shared-batch CFG, feature caching across steps (DeepCache [44]), scale-to-zero LoRA adapters, result cache for repeated prompt+seed
- Fleet: separate pools for base, upscale and inpaint. Preemptible GPUs for the "relax" tier

## Monitoring

| Signal | Alert on |
|---|---|
| Safety filter block rate by category and locale | Spikes (new jailbreak) or drops (classifier broken) |
| User reports per 1k images | Above SLO |
| Output-classifier vs human-review disagreement | Drift in safety models |
| Prompt distribution (embedding clusters) | New trend or attack campaign |
| Aesthetic / CLIPScore of served images | Quality regression after deploy |
| GPU utilization, queue depth, OOMs | Capacity |
| Watermark detection rate on sampled outputs | Provenance pipeline broken |

## Failure modes

| Failure | Mitigation | Ref |
|---|---|---|
| Jailbreak via "visual synonyms" (ketchup for blood), misspellings, other languages | Output classifiers, LLM judge, adversarial red-teaming loop | [1] [22] |
| Benign prompt + benign upload = harmful combination | Joint multimodal safety check | [1] |
| Safety filter bypass on open weights | In-model safety guidance, concept erasure | [21] [22] [43] |
| Over-filtering silences groups / demographic skew | Bias audits, calibrated thresholds | [1] |
| Deepfakes of public figures, non-consensual imagery | Face-match block, name stripping in rewrite, provenance | [1] [28] |
| Copyright / living-artist style mimicry | Artist-name filters, training opt-outs, data licensing | [8] |
| Watermark removal (cropping a visible signature) | Invisible watermark plus C2PA plus server-side lookup | [1] [3] [40] |
| Poor text rendering, hands, counting | T5 encoder, DiT, recaption, reward fine-tuning | [6] [8] [15] |
| Low diversity after distillation or DPO | Mix of teacher outputs, diversity metrics | [12] [27] |
| GPU queue blowup at launch | Tiered queues, degrade steps, waitlist | [1] |
| Memorization / training-image extraction | Dedupe training data, near-dup check on outputs | [45] |

## Interviewer follow-ups

- Why diffusion over GANs or autoregressive token models? What does each trade off in quality, diversity and latency?
- Why run diffusion in latent space? What do you lose (VAE artifacts on text and faces)?
- Explain classifier-free guidance. What happens as w goes up, and how do you train for it (null-text dropout)?
- How would you get ~10 s generation time for 4 images down to under 1 s? (Distillation, schedulers, batching, caching, hardware.)
- Web captions are noisy. How do you improve prompt following? (Recaptioning, LLM prompt rewrite.)
- Where do safety filters go, and why is a prompt filter alone not enough?
- How do you measure "quality" offline when FID disagrees with humans?
- Design the human evaluation: side-by-side protocol, rater agreement, prompt set (PartiPrompts/DrawBench-style).
- How would you use implicit user feedback (upscale and download picks) to improve the model? What are the bias risks?
- How do you stop deepfakes of public figures and generation of copyrighted characters?
- How do you watermark outputs, and how robust is the watermark to crops, compression and screenshots?
- How do you schedule GPUs across free and paid tiers and handle a 10x launch spike?
- How do you support personalization (DreamBooth/LoRA) at scale without one GPU per user?
- How would you detect a new jailbreak campaign in production?
- How do you check that the model is not reproducing memorized training images?
- How do you design inpainting / outpainting (mask channel, masked-latent conditioning) and image prompts?
- Which offline benchmark would you gate on (GenEval, T2I-CompBench, PartiPrompts), and why not FID alone?

## Portal comparison

| Portal | Framing | Model emphasis | Safety / serving depth | Unique angle |
|---|---|---|---|---|
| ByteByteGo GenAI SDI [29] | "Text-to-image like Midjourney" | Diffusion vs autoregressive, U-Net/DiT, CFG, recaptioning | Prompt safety, prompt enhancement, harm detection, super-resolution in the serving pipeline | Explicit metrics section (IS, FID, CLIPScore, human eval) |
| Educative Grokking GenAI SD [30] | Image generation as one of text/image/speech/video modalities | Training, inference, scaling | Evaluation and scaling chapters | Cross-modality comparison |
| alirezadir Agentic-AI-Systems [32] | "Image-generation pipeline" as a Tier-3 role-specific topic | Not detailed | Not detailed | Positions it as a specialist question |
| Chip Huyen GenAI platform [35] | Generic GenAI app stack | Model-agnostic | Guardrails, cache, router, observability | Platform view reusable here |
| OpenAI system cards [1] [8] | Real deployment | unCLIP / DALL-E 3 recaptioning | Data filters, input/output filters, rate limits, red teaming | Ground truth on the production safety stack |

## References

1. OpenAI, "DALL-E 2 Preview - Risks and Limitations" (system card), GitHub, 2022. https://github.com/openai/dalle-2-preview/blob/main/system-card.md (fetched)
2. CompVis, "Stable Diffusion" repository README, GitHub, 2022. https://github.com/CompVis/stable-diffusion (fetched)
3. Stability AI, "Generative Models" repository (SDXL, SDXL-Turbo), GitHub, 2023. https://github.com/Stability-AI/generative-models (fetched)
4. R. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", arXiv, 2021. https://arxiv.org/abs/2112.10752 (prior knowledge)
5. A. Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2), arXiv, 2022. https://arxiv.org/abs/2204.06125 (prior knowledge)
6. C. Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen), arXiv, 2022. https://arxiv.org/abs/2205.11487 (prior knowledge)
7. D. Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", arXiv, 2023. https://arxiv.org/abs/2307.01952 (prior knowledge)
8. J. Betker et al., "Improving Image Generation with Better Captions" (DALL-E 3), OpenAI, 2023. https://cdn.openai.com/papers/dall-e-3.pdf (prior knowledge)
9. J. Ho, T. Salimans, "Classifier-Free Diffusion Guidance", arXiv, 2022. https://arxiv.org/abs/2207.12598 (prior knowledge)
10. J. Ho et al., "Denoising Diffusion Probabilistic Models", arXiv, 2020. https://arxiv.org/abs/2006.11239 (prior knowledge)
11. J. Song et al., "Denoising Diffusion Implicit Models", arXiv, 2020. https://arxiv.org/abs/2010.02502 (prior knowledge)
12. A. Sauer et al., "Adversarial Diffusion Distillation" (SDXL-Turbo), arXiv, 2023. https://arxiv.org/abs/2311.17042 (prior knowledge)
13. S. Luo et al., "Latent Consistency Models", arXiv, 2023. https://arxiv.org/abs/2310.04378 (prior knowledge)
14. W. Peebles, S. Xie, "Scalable Diffusion Models with Transformers" (DiT), arXiv, 2022. https://arxiv.org/abs/2212.09748 (prior knowledge)
15. P. Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), arXiv, 2024. https://arxiv.org/abs/2403.03206 (prior knowledge)
16. C. Schuhmann et al., "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models", arXiv, 2022. https://arxiv.org/abs/2210.08402 (prior knowledge)
17. M. Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID), arXiv, 2017. https://arxiv.org/abs/1706.08500 (prior knowledge)
18. J. Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image Captioning", arXiv, 2021. https://arxiv.org/abs/2104.08718 (prior knowledge)
19. Y. Kirstain et al., "Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation", arXiv, 2023. https://arxiv.org/abs/2305.01569 (prior knowledge)
20. J. Xu et al., "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation", arXiv, 2023. https://arxiv.org/abs/2304.05977 (prior knowledge)
21. P. Schramowski et al., "Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models", arXiv, 2022. https://arxiv.org/abs/2211.05105 (prior knowledge)
22. J. Rando et al., "Red-Teaming the Stable Diffusion Safety Filter", arXiv, 2022. https://arxiv.org/abs/2210.04610 (prior knowledge)
23. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), arXiv, 2021. https://arxiv.org/abs/2103.00020 (prior knowledge)
24. N. Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation", arXiv, 2022. https://arxiv.org/abs/2208.12242 (prior knowledge)
25. L. Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet), arXiv, 2023. https://arxiv.org/abs/2302.05543 (prior knowledge)
26. K. Huang et al., "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation", arXiv, 2023. https://arxiv.org/abs/2307.06350 (prior knowledge)
27. B. Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization", arXiv, 2023. https://arxiv.org/abs/2311.12908 (prior knowledge)
28. C2PA, "Coalition for Content Provenance and Authenticity" (content credentials spec), c2pa.org, 2024. https://c2pa.org/ (prior knowledge)
29. A. Aminian, H. Sheng, "Generative AI System Design Interview" (incl. "Text-to-Image Generation" chapter), ByteByteGo, 2024. https://bytebytego.com/ (site root; chapter content from prior knowledge, host unreachable in review)
30. Educative, "Grokking the Generative AI System Design", educative.io, 2025. https://www.educative.io/courses/generative-ai-system-design (description confirmed via [31])
31. A. Dirafzoon, "GenAI Learning Resources", GitHub (Machine-Learning-Interviews), 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/genai-resources.md (fetched)
32. A. Dirafzoon, "GenAI & Agentic System Design" interview prep, GitHub (Agentic-AI-Systems), 2025. https://github.com/alirezadir/Agentic-AI-Systems/blob/main/06_interview_prep/genai-agentic-system-design.md (fetched)
33. T. Salimans et al., "Improved Techniques for Training GANs" (Inception Score), arXiv, 2016. https://arxiv.org/abs/1606.03498 (prior knowledge)
34. J. Yu et al., "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation" (Parti, PartiPrompts), arXiv, 2022. https://arxiv.org/abs/2206.10789 (prior knowledge)
35. C. Huyen, "Building A Generative AI Platform", huyenchip.com, 2024. https://huyenchip.com/2024/07/25/genai-platform.html (prior knowledge; host unreachable in review)
36. Hugging Face, "Diffusers" library, GitHub, 2024. https://github.com/huggingface/diffusers (fetched)
37. A. Dirafzoon, "Machine-Learning-Interviews" repository, GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews (fetched)
38. A. Nichol et al., "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models", arXiv, 2021. https://arxiv.org/abs/2112.10741 (prior knowledge)
39. C. Meng et al., "On Distillation of Guided Diffusion Models", arXiv, 2022. https://arxiv.org/abs/2210.03142 (prior knowledge)
40. Google DeepMind, "SynthID" (watermarking AI-generated content), deepmind.google, 2023. https://deepmind.google/technologies/synthid/ (prior knowledge)
41. X. Dai et al., "Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack", arXiv, 2023. https://arxiv.org/abs/2309.15807 (prior knowledge)
42. D. Ghosh et al., "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment", arXiv, 2023. https://arxiv.org/abs/2310.11513 (prior knowledge)
43. R. Gandikota et al., "Erasing Concepts from Diffusion Models", arXiv, 2023. https://arxiv.org/abs/2303.07345 (prior knowledge)
44. X. Ma et al., "DeepCache: Accelerating Diffusion Models for Free", arXiv, 2023. https://arxiv.org/abs/2312.00858 (prior knowledge)
45. N. Carlini et al., "Extracting Training Data from Diffusion Models", arXiv, 2023. https://arxiv.org/abs/2301.13188 (prior knowledge)
