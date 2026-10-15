# 02. Data Engines and Flywheels: Fleet Learning Architectures from Tesla to NVIDIA

## Overview

A "data engine" (also called a data flywheel, a continuous learning machine or a data closed loop) is the outer control loop of a deployed machine-learning system. The deployed model acts as a sensor of its own weaknesses. Production traffic, whether that is a fleet of cars or a stream of LLM requests, is filtered for informative examples. Those examples are labeled cheaply, often by larger offline models, and added to training and evaluation sets. A new model is trained, gated against regression suites, and redeployed. Then the cycle repeats. Autonomous driving is where the pattern was first engineered at industrial scale, and most of the vocabulary used today (shadow mode, triggers, auto-labeling, unit-test sets, closed-loop simulation) comes from there.

This chapter surveys the publicly documented designs:

- **Tesla's Autopilot/FSD data engine**: shadow mode, fleet-deployed trigger classifiers, offline multi-trip auto-labeling, and, from 2025, a learned neural world simulator for evaluation [1][2][5][8][15].
- **Waymo**: fleet log search (Content Search), offboard auto-labeling, large-scale simulation, the EMMA end-to-end research model, and the 2025 to 2026 "Driver / Simulator / Critic" architecture built on the Waymo Foundation Model, with the Genie-3-based Waymo World Model [17][18][19][20][65][67].
- **Wayve**: an end-to-end "fleet learning loop" fed by diverse fleets, plus generative (GAIA-1, GAIA-2) and reconstructive (PRISM-1, Ghost Gym) simulation [26][27][28][29][30].
- **Cruise**: the Continuous Learning Machine (CLM), a self-supervised loop for motion prediction that mined errors and auto-labeled them from future observations [32][33][36].
- **Scale AI**: the data engine as a vendor product (Nucleus, model-assisted debugging, SiaSearch, and the 2025 "Data Engine for Physical AI") [37][38][39][41].
- **NVIDIA**: the AI Blueprint for data flywheels (LLM-agent distillation on NeMo microservices), and the physical-AI stack (Cosmos Curator, Cosmos Transfer/Predict, Cosmos-Drive-Dreams, AlpaSim, Alpamayo, Physical AI Data Factory Blueprint), now closed with AlpaGym closed-loop RL [43][45][47][48][50][51][52][60][61].
- **Other fleet-learning architectures**: Mobileye REM crowdsourced mapping, comma.ai's world-model-trained openpilot, robot fleets learning from deployment (Physical Intelligence's π\*0.6), and OEM programs [53][54][55][59][68].
- **Open evaluation benchmarks** for the gate stage of the loop: NAVSIM (pseudo-simulation) and Bench2Drive (closed loop) [63][64].

A note on sourcing. Many of these companies publish only through keynote talks, blog posts and patents. Where a number comes from a secondary recap and not from the company itself, the text says so and marks it approximate.

The generic loop that all of the designs below put into practice looks like this:

```
           +-------------------------------------------------------------+
           |                    OUTER LOOP (weeks)                       |
           v                                                             |
+-------------------+   triggers /    +------------------+   labels    +-+--------------+
| Deployed model(s) |---------------->| Data selection & |------------>| Train / distill|
| (fleet / prod)    |  shadow-mode    | storage (lake,   |  auto-label |  new candidate |
+---------+---------+  divergences    | search, curation)|  + human QA +-------+--------+
          ^                           +------------------+                     |
          |                                                                    v
          |      +-----------------+     pass gates      +---------------------+--+
          +------| Staged rollout  |<--------------------| Evaluation: regression |
                 | (shadow, canary)|                     | sets, sim, critic      |
                 +-----------------+                     +------------------------+
                         INNER LOOP (hours to days): train <-> simulate <-> evaluate
```

## 1. Tesla Autopilot / FSD data engine

### 1.1 Problem

Tesla built a camera-centric driving stack. Its training signal had to come mostly from a very large consumer fleet, not from a small fleet of instrumented test vehicles. The fleet produces far more raw sensor data than can be uploaded or labeled, and the failures that matter live in a long tail of rare scenarios. The data engine solves a selection problem: pick the few clips worth uploading. It also solves a labeling problem: label those clips at a cost and speed that human annotation cannot match. Andrej Karpathy, then head of Autopilot vision, framed the needed ingredients at CVPR 2021 as a large car-fleet data engine, a strong AI team and a supercomputer [6][7].

### 1.2 Architecture and data flow

The loop as described in Karpathy's talks, summarized by several annotators of his CVPR 2021 keynote, is [5][6][7]:

1. Deploy models in "shadow" (also called "ghost") mode.
2. Observe their predictions in the field.
3. Tune triggers for collecting new training data.
4. Turn wrong predictions into new unit tests.
5. Add similar examples to the dataset.
6. Retrain, and repeat.

Ullrich et al.'s paper on extending the V-model for AI systems (IEEE Transactions on Intelligent Vehicles, 2024; arXiv 2025) is cited in secondary write-ups as using Tesla as a worked example of the same cycle: collection drives auto-labeling and unit-test updates, and then the network is retrained and redeployed in shadow mode [2]. This chapter could not retrieve the full text to confirm the Tesla passage, so treat the attribution as secondary.

```
  FLEET (on-vehicle)                                CLOUD (offline)
  +---------------------------------------+         +-------------------------------+
  | Production net drives / or human      |         | Upload queue -> data lake     |
  | drives while candidate net runs in    |  clips  | Offline "auto-labeler":       |
  | SHADOW mode (no actuation)            |-------->|  multi-trip 3D reconstruction |
  |                                       |         |  large offline nets, hindsight|
  | Trigger classifiers (200+), selected  |         |  human QA on residual errors  |
  | by context (e.g. map: tunnel ahead)   |         +---------------+---------------+
  |  - prediction vs driver divergence    |                         |
  |  - detector hits for a target concept |                         v
  |  - crashes / disengagements           |         +-------------------------------+
  +-------------------^-------------------+         | Train (HydraNet / occupancy / |
                      |                             | end-to-end)                   |
                      | OTA: new triggers,          | Unit-test sets: never regress |
                      | new shadow candidate        | Sim / neural world simulator  |
                      +-----------------------------+-------------------------------+
```

**Shadow mode.** IEEE Spectrum describes shadow mode as running on Tesla vehicles since 2016. When the Autopilot computer is not controlling the car, it simulates the driving process in parallel with the human driver. When its predictions diverge from what the driver does, the car can save a short clip of camera and vehicle data for upload. Crashes also trigger snapshots [1]. A Chinese industry analysis warns that the term has been overstated in popular coverage. In its account, shadow mode is one part of Tesla's data closed loop, not a self-contained learning system [4].

**Trigger classifiers.** The fleet-side selector is a library of triggers: small detectors or rules run alongside the driving stack that decide when a clip is worth uploading. Secondary accounts report that Tesla hand-crafted roughly 200 triggers (recaps of Karpathy's CVPR 2021 talk often give 221) to detect discrepancies in predicted object parameters such as position, velocity and acceleration, for example during the move to vision-only depth and velocity [2][6][7]. The exact count comes from talk slides and recaps, not from a Tesla publication. A practitioner analysis adds that triggers are chosen by context. Map data showing a tunnel ahead, for example, can activate tunnel-specific triggers [3]. Karpathy described the fleet as a large, distributed, mobile data center. In a 2020 talk quoted by Spectrum he said engineers can "beam it down to the fleet, and we can ask the fleet to please apply this detector on top of everything else you're doing" [1]. In his canonical example, a request for stop signs occluded by foliage came back with thousands of examples to iterate on [5][1].

**Unit tests as regression gates.** Mispredictions become curated evaluation sets. One analysis describes the policy as follows: performance on these "unit tests" must never regress, and a new model is accepted for production only if it improves on them [5]. One review of the CVPR talk reports that Tesla ran about seven rounds of the shadow-mode loop on one feature it was migrating to vision-only. That figure is from a secondary source [7].

### 1.3 Auto-labeling

At AI Day 2021, Ashok Elluswamy presented the auto-labeling pipeline. Tesla's in-house manual labeling team had about 1,000 people, and that was not enough on its own [9][8]. Two shifts made labeling scale:

- **Label in 4D vector space instead of per-image.** Labeling reconstructed 3D scenes over time, across all cameras at once, was reported to raise throughput about 100-fold over per-image 2D labeling [9].
- **Offline auto-labeling with hindsight.** Large offline networks and reconstruction, which can look at the past and the future of a clip, produce labels for the online network. Tesla reported collecting and auto-labeling 10,000 clips in about a week, which Elluswamy said would have taken months by hand [9][11].

Tesla had already shown fusing several clips of one intersection at AI Day 2021. At AI Day 2022 it described **multi-trip reconstruction** as the main auto-labeling engine for lanes and static structure. Clips from many vehicles passing the same location are fused into a single 3D reconstruction, and offline networks then label lanes, road edges and other structure as vectors for the online stack [13]. Secondary recaps of AI Day 2022 report figures that should be treated as approximate and unverified: auto-labeling 10k trips in about 12 hours of cluster time, against an estimated about 5 million hours of manual labeling, and a training infrastructure of about 14,000 GPUs [12]. Auto-labels covered road texture, walls, barriers, static objects, objects moving through occlusion, pedestrians, motorcycles and parked cars [12].

Tesla's patent family "Generating ground truth for machine learning from time series elements" puts the hindsight principle in legal terms. It covers receiving sensor data as a time series, deriving ground truth from portions of the series, and training a model to predict that ground truth from a single element. A dependent claim ties this to lane lines. The family includes US 10,997,461 (filed 2019, granted 2021) and US 12,223,428 (granted February 2025). Elluswamy and Karpathy are among the inventors [14].

### 1.4 Simulation

Elluswamy described Tesla's simulator as "a video game with Autopilot as the player". Because scenes are generated from vector space, their labels are exact. Simulation targets scenarios that are hard to source, hard to label, or that need a closed loop [8][10].

### 1.5 The end-to-end era (2024 to 2026)

With the move to end-to-end neural networks, the data engine's job changed. It no longer curates perception labels. It curates driving demonstrations and evaluation scenarios. At ICCV 2025 (October, Honolulu), Elluswamy presented Tesla's approach on behalf of the Tesla AI team [15]. Tesla described a **neural world simulator**, trained in-house, that generates the vehicle's camera and other sensor data. It is causal, it responds to commands from the driving-policy model, and it can be used to validate new driving models against historical data [15]. Coverage of the talk reports Elluswamy describing a "Niagara Falls of data" (hundreds of years of collective fleet driving) and continued use of smart triggers to capture rare cases [16]. This closes the evaluation loop that hand-built simulators left open. Instead of replaying logs open-loop, a candidate policy can be rolled out against a learned model of the world that is conditioned on real fleet clips.

### 1.6 Failure modes and lessons

- **Selection bias by trigger design.** The fleet only uploads what the triggers ask for, so blind spots in trigger design become blind spots in the dataset. Context-gated triggers [3] reduce bandwidth but compound this risk.
- **Hype versus mechanism.** Analysts caution that "shadow mode" is often described as if it were the whole system. In practice it is one sensor feeding a curation and labeling pipeline [4]. Spectrum also notes that the data advantage had not, as of its reporting, been shown to translate into human-level safety [1].
- **Regression gates are the backbone.** The "unit tests never regress" rule [5] turns an open-ended data stream into a monotone release process.
- **Labeling cost drives architecture.** The 4D-labeling and multi-trip auto-labeling shifts [9][13] show that the data engine shaped the model design, toward vector-space outputs that offline reconstruction could label, as much as the model shaped the data engine.

## 2. Waymo

### 2.1 Problem

Waymo operates a driverless, lidar-camera-radar robotaxi service. Its fleet is far smaller than a consumer fleet, but every mile is logged with a full sensor suite and there is no human backup. Its data-engine problems are (a) finding rare examples in a large sensor archive, (b) evaluating candidate drivers safely before deployment, and (c) turning fully autonomous experience into training signal.

### 2.2 Content Search: retrieval over the fleet archive

In February 2020 Waymo described **Content Search**, built with Google Research using technology similar to Google Photos and Google Image Search [17]. Before it existed, engineers mined rare objects with heuristics such as an object's estimated speed and height, which Waymo called too broad. Content Search instead converts every object in the database into an embedding, so objects can be ranked by similarity. It supports query-by-example image, category search, and text search over OCR'd content such as road signs. Queries return in seconds. Waymo cited about 20 million real-world miles of driving history at the time [17]. A 2022 Scale AI write-up of a Dmitri Dolgov talk described a fleet driving a combined 100,000 miles or so each week, feeding and being fed by Waymo's ML systems (approximate, secondary) [24]. The retrieved examples feed model training directly. This is the "find more like this" half of the data engine. Tesla runs the same step on the vehicle with triggers. Waymo runs it in the data center over a complete log archive.

### 2.3 Simulation at scale

Waymo's 2020 simulation post describes running its virtual fleet around the clock on large amounts of compute, with tooling that lets engineers work on simulation from anywhere [23]. Waymo recruiting material from 2025 to 2026, a weaker source, describes tens of billions of simulated miles. It also describes simulation agents built with large foundation models and diffusion, and a "virtual driver's test" whose central question is whether simulated results predict real-world behavior [23].

### 2.4 Driver, Simulator, Critic (December 2025)

Waymo's December 2025 post "Demonstrably Safe AI for Autonomous Driving" is the most explicit public description of its flywheel [18]:

- **One foundation, three roles.** The Driver (on-road behavior), the Simulator (closed-loop testing) and the Critic (judges behavior and points to improvements) are all built on the **Waymo Foundation Model**.
- **Teacher to student distillation.** Waymo trains large Teacher models for driving, simulation and evaluation, then distills them into smaller Student models for deployment. Critic teachers analyze driving behavior and generate high-quality training signals for students, and they automatically build evaluation datasets.
- **"Think Fast and Think Slow".** The Foundation Model uses a System-1 / System-2 design. A fast sensor-fusion path handles reactive control, and a driving vision-language model reasons more slowly about long-tail semantics.
- **Two loops.** An inner loop couples the Simulator and the Critic with reinforcement learning. An outer loop starts from real-world fully autonomous driving: the Critic automatically flags suboptimal behavior, and the flagged events are used to generate better alternative behaviors as training data for the Driver. New software ships only after safety checks pass.

```
                         Waymo Foundation Model (Teachers)
              +--------------------+---------------------+-------------------+
              | Driver teacher     | Simulator teacher   | Critic teacher    |
              +---------+----------+----------+----------+---------+---------+
                        | distill             | distill            | distill / labels
                        v                     v                    v
   onboard Driver  <--- RL inner loop ---> Simulator <---------> Critic
   (student, fast        (closed-loop rollouts scored by Critic)    |
    + slow paths)                                                    |
        |                                                            |
        | fully autonomous miles                                     |
        +------> Critic flags suboptimal events --> better behaviors -+--> training data
                 (outer loop)                                         + eval sets
```

### 2.5 The Waymo World Model (February 2026)

In February 2026 Waymo announced the **Waymo World Model**, built on Google DeepMind's Genie 3 and adapted to driving [19][25]. Its notable properties:

- **Multi-sensor generation.** It produces camera and lidar output. Post-training converts Genie's 2D video knowledge into 3D lidar for Waymo's hardware.
- **Controllability.** Scenes can be steered with language prompts, driving inputs and scene layouts, which supports "what-if" variation of weather, time of day and road layout.
- **Long-tail synthesis.** Genie's world knowledge lets it synthesize events Waymo's fleet has never seen, such as tornadoes, flooded streets, or an elephant on the road.

Press coverage noted that the announcement included no benchmark results [25].

### 2.6 Lessons from 200+ million autonomous miles (August 2026)

Waymo's August 2026 post "10 AI Lessons from Driving 200+ Million Fully Autonomous Miles" restates the flywheel. Automated systems, including the Critic and feedback from riders and communities, surface opportunities for refinement [20]. Lessons reported in coverage include [20][21][22]:

- Multimodal sensors are indispensable.
- HD maps act as a "mental memory".
- Fewer, high-capacity, specialized foundation models are preferred.
- The Driver should combine fast control with higher-level reasoning, alongside separate onboard verification and independent evaluation.
- Neither billions of simulated miles nor millions of human-supervised miles substitute for what a system learns when it operates with no human backup. Electrek read this as a direct critique of the L2-to-L4 path [21].

### 2.7 Offboard auto-labeling, EMMA and open end-to-end data

Three public Waymo results fill in parts of the loop that the blog posts only name.

- **Offboard auto-labeling (2021).** Qi et al.'s "Offboard 3D Object Detection from Point Cloud Sequences" (CVPR 2021) is Waymo's published version of hindsight labeling. An offboard detector sees the whole lidar sequence, past and future, and tracks objects across it. It then refines each object's box using every frame in which the object appears, producing labels that approach human quality for training onboard models [67]. It is the Waymo counterpart to Tesla's offline auto-labeler (section 1.3).
- **EMMA (October 2024).** "EMMA: End-to-End Multimodal Model for Autonomous Driving" fine-tunes a Gemini-based multimodal model to map raw camera input directly to planner trajectories, perception objects and road-graph elements, all written as text [65]. It is a research model, not the deployed Driver. It shows the large teacher-model direction that the 2025 Driver / Simulator / Critic architecture later made explicit. The paper also lists its own limits: camera-only input and too few frames for long temporal context.
- **Open end-to-end data.** The Waymo Open Dataset now ships an End-to-End Driving dataset of camera data with high-level routing commands, alongside the Perception dataset and the Motion dataset (103,354 scenes) [66]. Waymo's 2025 release of this set (WOD-E2E) was presented as focused on rare, long-tail scenarios. The repository README does not state that framing, so treat it as Waymo's description. This lets outside groups test end-to-end policies on the kind of rare events Waymo's own data engine mines.

### 2.8 Lessons

Waymo's design shows the flywheel moving from *data selection* to *signal generation*. Content Search optimized retrieval. The 2025 to 2026 architecture instead uses a learned Critic to label the system's *own behavior*, and learned simulators to generate counterfactuals. The scarce resource becomes a trustworthy evaluator rather than raw data. The main open risk, which Waymo acknowledges through its "virtual driver's test" framing [23], is whether simulator and critic judgments transfer to real-world outcomes.

## 3. Wayve

### 3.1 Problem

Wayve builds an end-to-end "embodied AI" driver that it intends to license to automakers across many vehicle platforms, sensor sets and countries. Its data engine has to generalize across fleets it does not fully control. It also has to produce rare, safety-critical data that fleets rarely see.

### 3.2 Fleet learning loop

Wayve describes a **Fleet Learning Loop**: gather real-world driving data from diverse fleets, process it in cloud training infrastructure, train, simulate, evaluate, inspect, deploy, and repeat [30][31]. Its 2025 "Global Learning, Local Driving" post on Japan argues that one globally trained model can adapt to new regions [30]. Nissan has signed definitive agreements to put Wayve's AI into next-generation ProPILOT, starting with a model in Japan in fiscal 2027. A prototype Ariya drove in central Tokyo in September 2025. Both companies say deployment in mass-produced vehicles will let them learn from diverse real-world conditions, which is the consumer-fleet flywheel again, this time via an OEM partner [31].

### 3.3 Generative and reconstructive simulation

Wayve uses two kinds of offboard world models:

- **GAIA-1 (2023).** A generative world model previewed in June 2023, with a technical report in October 2023. It takes video, text and action inputs. Coverage reports about 9B parameters (about 6.5B for the autoregressive world model and about 2.6B for the video diffusion decoder), trained on about 4,700 hours of proprietary London driving data collected from 2019 to 2023 [28].
- **GAIA-2 (March 2025).** A latent-diffusion world model made of a video tokenizer plus a latent world model [27]. It is conditioned on ego-vehicle dynamics, agent configurations, environmental factors and road semantics, and it generates spatially and temporally consistent surround-view video. Its training data was curated across the UK, US and Germany, across cars and vans, and across varied sensor configurations and frame rates [26][27]. Wayve positions it as an offboard tool for synthesizing diverse and safety-critical scenarios for training and validation, especially events that are hard or impossible to collect from a real fleet [26]. A secondary wiki reports about 8.4B parameters and up to five cameras. This is not confirmed in the sources retrieved here.
- **PRISM-1 and Ghost Gym (2023 to 2024).** Ghost Gym is a closed-loop neural simulator for testing driving models under consistent conditions. PRISM-1 is a camera-only 4D (3D plus time) scene reconstruction model that feeds it, without relying on lidar, 3D boxes or scene graphs. Wayve contrasts the two approaches: GAIA generates new scenarios, while PRISM-1 *re-simulates* real logged scenes with changes to viewpoint, position or speed, or with removed actors. Wayve released WayveScenes101, a set of 101 scenes for novel-view-synthesis research [29].

```
   diverse fleets (own + OEM partners, UK/US/DE/JP)
        |  logs
        v
   +------------+    curate     +-----------------+     train      +-------------+
   | data lake  |-------------->| training sets   |--------------->| end-to-end  |
   +-----+------+               +-----------------+                | driver      |
         |                               ^                         +------+------+
         | real scenes                   | synthetic rare events          |
         v                               |                                v
   PRISM-1 4D recon --> Ghost Gym   GAIA-2 generative video  ---->  closed-loop eval
   (re-simulate, edit actors)       (condition on agents, weather)  -> inspect -> deploy
```

### 3.4 Lessons

Wayve splits world modeling into *reconstruction*, which gives high fidelity close to logged data and is suited to regression testing, and *generation*, which gives coverage beyond the data and is suited to the long tail. That split is a useful template. The weak point that Wayve's own coverage does not quantify is the gap between neural-simulator rollouts and on-road outcomes. No independent closed-loop benchmark was found for PRISM-1 [29].

## 4. Cruise: the Continuous Learning Machine

### 4.1 Problem

Cruise ran driverless robotaxis in San Francisco, a dense urban setting where motion prediction of other agents dominates the long tail. Cruise noted that U-turns made up under 0.1% of observed data, and that its cars saw fewer than 100 U-turns a day on average [32][36].

### 4.2 Architecture

Cruise's 2020 post describes the **Continuous Learning Machine (CLM)** for prediction [32][33]:

1. **Error mining (active learning).** The system automatically finds cases where the onboard prediction diverged significantly from what then happened. Only those cases enter the training set [33].
2. **Self-supervised auto-labeling.** The future is the label. The agent's trajectory as later observed by perception becomes ground truth for the earlier prediction. If a car was predicted to turn left but went straight, that yields a label with no human in the loop [32][56].
3. **Rebalancing.** Cruise up-sampled rare maneuvers to teach the predictor about them [36]. A secondary analysis notes the self-correcting dynamic: an early model that under-predicts U-turns generates many U-turn errors, so the mined dataset fills with U-turns until the model handles them [56].
4. **Train, test, deploy.** A new model must outperform the previous one before it is deployed [33].

```
  onboard prediction(t) ----+                      +--> train candidate --> eval gate
                            |  |pred - obs| > tau  |        (must beat prod)
  perception track(t+k) ----+---> mined error ---->+                |
   (future = free label)          (no human)                         v
                                                              deploy to fleet
```

Cruise said the loop let it update its models as often as about twice a week, compared with roughly quarterly before [34]. Cruise engineers later presented the CLM's ML infrastructure for the long tail at MLconf, covering distributed data processing and training [35].

### 4.3 Lessons and status

The CLM is the purest example of a **self-supervised data engine**. When the target is "what happens next", the fleet labels itself and the only cost is compute. It does not extend to tasks whose ground truth is never observed, such as the counterfactual "what *should* the AV have done". That gap is what learned critics such as Waymo's are now meant to fill. Cruise suspended driverless operations after an October 2023 pedestrian-dragging incident. In December 2024 GM announced it would stop funding Cruise's robotaxi work and fold the team into its personal-vehicle driver-assistance program [69]. The CLM is therefore a historical reference design. Its error-mining idea survives in the ADAS work that absorbed the team.

## 5. Scale AI: the data engine as a product

### 5.1 Problem

Most AV and robotics teams cannot build Tesla-scale tooling in-house. Scale AI started by selling labeled data to AV companies [37][38]. It then tried to package the rest of the loop (curation, model debugging, search) as a product.

### 5.2 Components

- **Nucleus.** Alexandr Wang described Nucleus as "Google Photos for machine learning data sets", for organizing and curating datasets and measuring model performance on them [37][38]. Its debugging workflow is a canonical data-engine step [39]:
  - Load model predictions next to ground truth.
  - Open a confusion matrix and click a cell to filter to those examples. One Scale example surfaced 2,277 objects predicted "person" but labeled "car".
  - Sort by model confidence. High-confidence false positives with IoU between 0.5 and 1 are frequently annotation errors.
  - Send them to Scale Studio for relabeling. Corrected labels import back automatically.

  **Autotag** uses model embeddings to find visually similar images or objects and turns the result into a queryable metadata tag [39]. Nucleus added 3D "Scenes": sequences of lidar point clouds with their time-aligned camera images [40].
- **SiaSearch acquisition (2021).** Scale bought SiaSearch, a search platform for driving-sensor data, to fold into Nucleus and offer a "full data engine". TechCrunch framed this as giving other companies "the same data engine superpowers that Tesla has" [37].
- **Data Engine for Physical AI (September 2025).** Scale extended its AV-era engine to robotics data collection and annotation for foundation-model training. It reported more than 100,000 production hours at its San Francisco prototyping lab, and named Physical Intelligence, Generalist AI and Cobot as customers [41].
- **Corporate context.** In June 2025 Meta took a non-voting 49% stake in Scale for about $14.3 to $14.8 billion (reports differ), and CEO Alexandr Wang moved to Meta [42].

### 5.3 Lessons

Scale's tooling shows how the data engine broke apart into separable services: search and retrieval, error analysis against predictions, human relabeling, and re-import. The design lesson is that **prediction-versus-label disagreement, sorted by confidence**, is one of the cheapest signals available for both label noise and model error. The same signal drives Tesla's unit tests and Cruise's CLM.

## 6. NVIDIA: flywheel blueprints

NVIDIA publishes reference implementations rather than running a fleet. It has two distinct flywheel lines.

### 6.1 AI Blueprint for Building Data Flywheels (LLM agents)

**Problem.** Enterprise agents often call a large LLM for every node, even when some nodes, such as tool-calling or routing, are simple enough for a much smaller model. Production logs already hold the input distribution, so the flywheel distills from it.

**Architecture** [43][44]:

```
  agent app --(prompt/completion logs, workload_id, client_id)--> Elasticsearch
                                                                     |
            FastAPI  POST /jobs  -->  Celery (Redis queue)           v
                                     parent_queue (concurrency 1) -> pull logs,
                                                                     group by workload,
                                                                     dedupe,
                                                                     stratified split
                                                                     |
           NeMo Datastore <------------- eval + fine-tune datasets --+
                 |
     +-----------+------------+----------------------+
     v                        v                      v
   base eval              ICL eval               customized eval
  (replay prompts)   (few-shot, random or      (LoRA SFT via NeMo
                      semantic selection)       Customizer)
     +-----------+------------+----------------------+
                 v
   NeMo Evaluator: LLM-as-judge similarity in [0,1] vs production model
   NeMo Deployment Manager spins NIMs up/down; MongoDB stores results
                 v
   human reviews candidates -> promote smaller model (manual decision)
```

**Key techniques.**

- Every LLM call site in an agent gets its own `workload_id`, which is the primary key for slicing, de-duplication and dataset creation. A multi-node agent therefore yields one flywheel per node [43].
- There are three experiment types per candidate NIM: **base** (replayed production prompts), **icl** (few-shot prompts, chosen randomly or by semantic similarity) and **customized** (LoRA fine-tune). All are scored by an LLM judge for similarity to the incumbent model's outputs [43].
- Default LoRA settings are 2 epochs, batch size 16, learning rate 1e-4, adapter dimension 32 and dropout 0.1 [43].

**Numbers.** In an internal HR-chatbot tool-calling case, a fine-tuned Llama-3.2-1B reached about 98% of Llama-3.1-70B's accuracy, for up to 98.6% inference-cost reduction. In another case Qwen-2.5-32B-Coder matched the 70B model without fine-tuning, cutting cost and time-to-first-token by more than 50% [43]. The minimum footprint is 6x H100/A100 with a self-hosted judge, or 2x with a remote judge. Runs are serialized because GPU availability is not discovered automatically [43].

**Failure modes, as the repo itself states them** [43]:

- Production model outputs serve as ground truth, so the flywheel can at best *match* the incumbent, not exceed it.
- No PII scrubbing happens before fine-tuning.
- No hand labels are used.
- Promotion stays a human decision. The docs describe the system as "a flashlight, not an autopilot".

As of April 2026 the repo is marked deprecated and no longer maintained [43]. Partners such as Iguazio published derived orchestration blueprints on MLRun [57].

### 6.2 Physical-AI and AV data flywheel

For autonomous vehicles and robots, NVIDIA's stack covers the curate, augment, simulate and evaluate stages of the loop:

- **Curation.** Cosmos Curator (on the Ray-based Cosmos-Xenna streaming framework, deployable locally, on Slurm or on NVIDIA Cloud Functions) does video splitting, annotation, filtering, de-duplication and dataset generation [48]. NVIDIA reported that its NeMo Curator video pipeline could process 20M hours of video in days instead of years, an 89x speedup on 1K GPUs over unoptimized CPU pipelines at equal power. Launch coverage cited 20M hours in 14 days on Blackwell (approximate, configuration-dependent) [49][45].
- **Augmentation and generation.** The Omniverse Blueprint for AV simulation uses **Cosmos Transfer** to multiply physically based sensor data across weather and lighting. Foretellix is one user. **Cosmos Predict** expands multi-sensor views and is used by Nexar and Oxa [45][46]. NVIDIA's developer blog describes the goal as an AV data flywheel that generates synthetic data, augments sensor datasets, fills coverage gaps, and builds a validation toolchain [46].
- **Cosmos-Drive-Dreams.** This open pipeline renders HD-map, bounding-box or lidar-depth condition videos from labeled clips. A VLM (Qwen3) rewrites captions into many variations. Cosmos-Transfer1-7B-Sample-AV generates 121-frame front-view videos, which a Single2MultiView model expands to multi-camera. The final VLM-based filtering step is still marked "coming soon" in the repository. NVIDIA released labels for 5,843 ten-second clips and 81,802 synthetic videos, about 3 TB in total, of which the synthetic video is about 700 GB [47].
- **Closed-loop evaluation: AlpaSim.** An open-source microservice simulator. Its services talk over gRPC, scale independently and can run multi-node on Slurm. Its renderer is pluggable: NuRec neural reconstruction is the default, and a video-model renderer (OmniDreams via FlashDreams) is supported. Bundled policies include Alpamayo-R1, Alpamayo 1.5, VaVAM and a provisional TransFuser [52].
- **Policy: Alpamayo.** Alpamayo-R1 is a 10B-parameter reasoning VLA: a Cosmos-Reason backbone plus an action expert. It was renamed Alpamayo 1 after NVIDIA's CES 2026 announcement. Its "Chain-of-Causation" (CoC) reasoning traces were produced by hybrid auto-labeling with humans in the loop. Alpamayo 1.5 (March 2026, also 10B) adds RL post-training, navigation conditioning, visual question answering and flexible camera counts [51][62].
- **Physical AI Data Factory Blueprint (GTC, March 2026).** An open reference architecture that automates creation, augmentation and evaluation of datasets using Cosmos Curator, Cosmos Transfer and **Cosmos Evaluator**, a Cosmos-Reason-based scorer that filters generated data for physical accuracy and training readiness. NVIDIA said it uses the blueprint to train and evaluate Alpamayo. Uber is listed as an early user [50].

```
 fleet logs --> Cosmos Curator --> labeled clips --(HDMap/bbox/depth)--> Cosmos Transfer
 (any OEM)      split/filter/      + captions                            / Predict
                dedupe/annotate         |                                    |
                                        v                                    v
                                 NuRec reconstruction              synthetic variants
                                        |                                    |
                                        v                                    v
                       AlpaSim closed loop  <---- Cosmos Evaluator filters -----+
                       (policy e.g. Alpamayo)        (physical plausibility)
                                        |
                                        +--> metrics --> retrain policy --> deploy
```

### 6.3 Closing the policy loop: CoC auto-labeling, AlpaGym and Alpamayo 2 (2026)

By mid-2026 NVIDIA had linked the pieces above into a loop for driving policies, not only for data. The Alpamayo Recipes hub describes three parts [60][61]:

- **Reasoning auto-labels.** A CoC auto-labeling pipeline writes reasoning labels for driving clips. The labels go into NVIDIA's Physical AI open dataset and are described as the basis for training larger reasoning models [60].
- **Closed-loop RL.** **AlpaGym** "runs a policy in closed loop inside a simulator, scores the resulting drives, and trains on them". A host control plane submits Slurm jobs. A runtime inside the GPU container drives AlpaSim episodes over gRPC, computes reward terms (the example run uses a `progress_safety` reward) and takes GRPO steps through Cosmos-RL. At the time of writing it supports only Alpamayo 1.5 and is labeled early-stage [61]. Open-loop RL and SFT recipes for Alpamayo 1 and 1.5 live in the recipes repository [60].
- **Scale-up.** The hub lists **Alpamayo 2 Super**, a 34B model on a Cosmos 3 backbone with 360-degree surround-view input and meta-action outputs, trained on more than 110,000 hours of driving data [60].

The open AV loop now has every stage in the generic diagram as a separate component: curation (Cosmos Curator), labeling (CoC auto-labeler), generation (Cosmos Transfer/Predict), closed-loop simulation (AlpaSim), RL (AlpaGym) and the policy (Alpamayo). NVIDIA's documents do not themselves call this a flywheel. Treating it as one is this chapter's reading. The repositories also do not report how well AlpaSim scores predict real-road outcomes.

### 6.4 Lessons

NVIDIA's two blueprints share one structural idea. **The flywheel is a DAG of independently deployable services** (log store, curator, generator, trainer, evaluator, deployment manager) joined by a dataset registry. Swapping any one stage, for example the renderer in AlpaSim or the judge in the LLM flywheel, does not disturb the others. They also share a weakness: **automated judges** (LLM-as-judge, Cosmos Evaluator) become the de facto spec, and their errors propagate silently unless humans audit samples.

## 7. Other fleet-learning architectures

### 7.1 Mobileye REM: a fleet as a map sensor

Mobileye's Road Experience Management turns millions of ADAS-equipped cars into a mapping fleet. On-camera processing (EyeQ) extracts landmarks and road geometry and uploads only about 10 KB per km. Mobileye says this is about 200 MB per vehicle-year of typical North American driving [53]. To protect privacy, data within a random range of a trip's origin and destination is discarded [53]. A January 2024 trade article quotes Amnon Shashua on data deals with nine automakers and reports about 95 million km mapped per day. That second figure is not verified against a primary source [53]. REM is the extreme of **edge compression**: the fleet ships semantic abstractions instead of raw clips, which makes harvesting nearly free but limits reuse for training perception.

### 7.2 comma.ai: training the policy inside a learned simulator

comma.ai's openpilot ships to real users and follows a world-model-centric design:

- **commaVQ (2023).** A VQ-VAE tokenizer (128 tokens of 10 bits per frame) and a GPT world model trained on 3 million minutes of driving video. comma released 100,000 minutes of tokenized video [54].
- **openpilot 0.9.9 (May 2025).** A new training architecture that uses parts of comma's MLSIM simulator [59].
- **openpilot 0.10 (August 2025).** Training followed the architecture in "Learning to Drive from a World Model", which the release notes call comma's CVPR paper. The longitudinal MPC in Experimental mode was replaced by end-to-end planning from the world model. Release 0.10.1 (September 2025) removed global-localization inputs from the world model, doubled its parameters and trained it on four times as many segments. Release 0.10.3 (December 2025) added a new temporal policy architecture and an on-policy training physics-noise model [59].
- **openpilot 0.11 (March 2026).** The release notes say the driving model was "fully trained using a learned simulator" [59]. comma's blog calls it the first real-world robotics agent shipped to users that was fully trained in a learned simulation [54]. That is the company's own claim. 0.11.2 (August 2026) shipped an 880M-parameter driving model, with support for big models on an external GPU [59].

The paper (arXiv 2504.19077) compares a *reprojective* simulator with a learned world-model simulator for on-policy training from real driving data [54]. The cadence above (a new world-model-trained driving model every few months, shipped to a user fleet that then produces the next round of training video) is the clearest open example of a consumer-fleet flywheel whose evaluation and training both run inside a learned simulator.

### 7.3 Robot fleets learning from deployment

The same loop is moving from cars to general-purpose robots. Scale's physical-AI data engine (section 5.2) sells the collection side. In November 2025 Physical Intelligence described π\*0.6, a VLA improved with a method it calls Recap (RL with Experience and Corrections via Advantage-conditioned Policies). The model is first trained by imitation. It is then deployed on real tasks. Autonomous rollouts, human teleoperated corrections and a learned value function are fed back to train an advantage-conditioned policy, so the robot improves from its own deployment experience [68]. Its open-source base models (π0, π0-FAST, π0.5) are pre-trained on more than 10,000 hours of robot data [70]. Structurally this is the Waymo outer loop at robot scale: deployment data, a learned critic that scores behavior, and retraining toward behavior the critic rates better. It also has the same open risk, which is whether the critic's judgments stay calibrated as the policy drifts. The π\*0.6 figures come from the company's own report and were not independently checked for this chapter.

### 7.4 OEM programs and the generalized loop

The flywheel framing has spread to automakers. Hyundai Motor Group, for example, describes an AI-powered data flywheel of continuous collection, training, validation and deployment [55]. Motional published a "Learning every mile driven" explainer in the same genre [58].

## 8. Open benchmarks for the evaluation half of the loop

Every flywheel above depends on an evaluation gate, and most companies keep theirs private. Two open benchmarks from 2024 to 2026 show how the research community closes that part of the loop, and they bear directly on gate design.

- **NAVSIM v1 and v2.** NAVSIM v1 (NeurIPS 2024 Datasets and Benchmarks) is *non-reactive* simulation on real logs. A planner's trajectory is scored by simulation-based metrics (the PDM score) rather than by L2 distance to the logged path [63]. NAVSIM v2 (CoRL 2025) adds **pseudo-simulation**: real data is augmented with synthetic observations, generated by 3D Gaussian splatting near the planned trajectory, so error recovery can be scored without running a sequential closed loop. The authors report strong correlation with closed-loop simulation at about 6x less compute. It was the main evaluation framework for the 2025 NAVSIM end-to-end driving challenge [63]. This is a practical middle point for a release gate: cheaper than full closed-loop runs, and far more predictive than open-loop error.
- **Bench2Drive.** A CARLA closed-loop benchmark (NeurIPS 2024 Datasets and Benchmarks) with 220 short evaluation routes covering 44 interactive scenarios, plus an official training set (Base 1,000 clips; Full 13,638 clips after a July 2024 cleanup) [64]. Its maintainers argue that open-loop L2 error on nuScenes "is not a meaningful indicator" and that reviewers should stop asking for it [64]. The August 2026 v0.0.4 update tackles a data-engine problem directly: the new training set is **strictly uniform over scenarios** (44 x 25 = 1,100 routes), because the old Base set's uneven scenario mix skewed results. A separate 220-route validation set was added. Spin-offs test deployment issues (Bench2Drive-Robust: camera-stream failures, ego-state errors, compute latency) and vision-language QA in the loop [64].

The lesson for flywheel builders is that **the evaluation gate needs the same care as the training set.** That means a balanced scenario mix, held-out validation that is separate from the gate set, and metrics that reward recovery and not imitation alone. The closed-loop-versus-open-loop debate in academia is the public form of the "virtual driver's test" question that Waymo raises in section 2.3.

## 9. Comparison

| System | Selection signal | Labeling | Evaluation / sim | Cadence / scale (public) |
|---|---|---|---|---|
| Tesla data engine [1][2][9][15] | On-car triggers (200+), shadow-mode divergence, crashes | Offline auto-labeler; multi-trip 3D recon; human QA | Unit-test sets that must not regress; game-engine sim; neural world simulator (2025) | 10k clips auto-labeled in ~1 week (2021); "hundreds of years" of fleet driving (2025) |
| Waymo [17][18][19] | Embedding search over full logs; Critic flags suboptimal driving | Teacher models / Critic generate signals and eval sets | RL inner loop with Simulator + Critic; Genie-3-based World Model (camera + lidar) | ~20M miles (2020); 200M+ autonomous miles (2026) |
| Wayve [26][27][29][30] | Diverse fleets incl. OEM partners | Not detailed publicly | Ghost Gym + PRISM-1 re-sim; GAIA-2 generative | GAIA-1 ~9B params, ~4,700 h data |
| Cruise CLM [32][33][34] | Prediction-vs-outcome error mining | Self-supervised (future observation is label) | Must beat previous model | ~2 updates/week (from ~quarterly) |
| Scale AI [37][39][41] | Customer-driven; confusion matrix, Autotag | Human + model-assisted (Studio) | Model-vs-label metrics in Nucleus | 100k+ production hours (physical AI lab, 2025) |
| NVIDIA LLM flywheel [43] | All production logs per workload_id | Incumbent model output as label | LLM-as-judge; base/ICL/LoRA experiments | Up to 98.6% cost cut (one internal case) |
| NVIDIA AV stack [47][50][52] | Curator filters over any log corpus | Auto + human-in-loop (CoC traces) | AlpaSim (NuRec), Cosmos Evaluator | 81,802 synthetic clips released; 20M h curation |
| Mobileye REM [53] | All driving (map features) | On-edge semantic extraction | Map consistency across fleet | ~10 KB/km; millions of cars |
| comma.ai [54][59] | User fleet driving video | World-model tokens / plans | On-policy in learned simulator | World model trained on 3M min video; 880M-param driving model (0.11.2, 2026) |
| NVIDIA AlpaGym [60][61] | Closed-loop rollouts in AlpaSim | CoC auto-labels (reasoning traces) | GRPO on simulator rewards | Alpamayo 2 Super: 34B, 110k+ h data |
| Physical Intelligence π\*0.6 [68] | Robot deployment rollouts + human corrections | Learned value function (advantage) | Real-world task success | Company-reported |

## 10. Cross-cutting techniques

**Edge versus cloud selection.** Tesla and Mobileye put selection on the vehicle (triggers, semantic compression), trading flexibility for bandwidth. Waymo, with full logging from a smaller fleet, selects in the cloud with embeddings [1][17][53]. A hybrid is common: coarse on-device triggers, then fine cloud-side retrieval and de-duplication.

**Hindsight as supervision.** Tesla's offline auto-labeler and patent [9][14], Waymo's offboard 3D detector [67], Cruise's future-as-label CLM [32] and Waymo's Critic [18] all exploit the same asymmetry. An offline system that can see the future and use far more compute labels the past for an online system that cannot.

**Regression sets as contracts.** Tesla's never-regress unit tests [5], Cruise's "must beat prior model" gate [33] and NVIDIA's human-gated promotion [43] all put a monotone release gate after a noisy data stream.

**Counterfactual generation.** Logged data cannot show what would have happened under a different policy. The field converged on learned simulators for this: Tesla's neural world simulator, the Waymo World Model, GAIA-2/PRISM-1, Cosmos/AlpaSim and comma's world model [15][19][26][29][52][54]. The new failure mode is **simulator exploitation and sim-to-real gap**. Waymo addresses it with a "virtual driver's test" for predictive validity [23]. Few public benchmarks exist for generative simulators themselves [25][29]. Planner-level open benchmarks such as NAVSIM pseudo-simulation and Bench2Drive are the nearest public proxies [63][64].

**Teacher-student distillation.** Large offline models generate labels or behaviors that are distilled into deployable students. Waymo does this explicitly [18]. NVIDIA's LLM blueprint does the same for agent nodes [43]. Tesla's offline-to-online auto-labeling is an early perception-only version [9].

## Key takeaways for system designers

1. **Design the selection function first.** Whether it is triggers on the device or embedding search in the cloud, what you choose to look at bounds what the model can learn. Version, audit and context-gate selectors the way you would models [2][3][17].
2. **Exploit hindsight wherever ground truth arrives later.** Future observations (Cruise), multi-trip reconstruction (Tesla) or a larger offline judge (Waymo, NVIDIA) can replace most human labels. Reserve humans for QA, ontology and disputed cases [9][32][18][43].
3. **Make every mined failure a permanent regression test, and balance the gate set.** Never-regress gates turn an open-ended data stream into a controllable release process. Scenario-uniform, held-out gate sets with recovery-aware metrics keep the gate from simply reflecting the data mix [5][33][63][64].
4. **Separate inner and outer loops.** Iterate fast in simulation with an automated critic (inner loop). Ground truth in real deployment (outer loop) and measure how well each predicts the other [18][23].
5. **Use reconstruction for fidelity and generation for coverage.** Re-simulating real logs (PRISM-1, NuRec) gives trustworthy regression tests. Generative world models (GAIA-2, the Waymo World Model, Cosmos) give long-tail breadth. You need both, plus a plausibility filter (Cosmos Evaluator) [29][19][50].
6. **Treat automated judges as specifications.** LLM-as-judge and learned critics encode your definition of "good". Audit them with humans, version them, and remember that a flywheel labeled by the incumbent model can at best match it [43][18].
7. **Key data by decision point.** NVIDIA's per-node `workload_id` and Tesla's per-task triggers both show that flywheels work per decision, not per product [43][2].
8. **Budget for compute and bandwidth explicitly.** Edge compression (REM's ~10 KB/km), curation throughput (~20M video hours per weeks) and GPU-serialized experiments are first-order constraints on loop latency [53][49][43].
9. **Do not over-claim the loop.** Analysts and Waymo itself caution that data volume and supervised miles do not substitute for validated, unsupervised performance [4][1][21].

## References

1. "Tesla's Autopilot Depends on a Deluge of Data". IEEE Spectrum (publication date not verified). https://spectrum.ieee.org/tesla-autopilot-data-deluge
2. "Expanding the Classical V-Model for the Development of Complex Systems Incorporating AI". arXiv:2502.13184, 2025. https://arxiv.org/pdf/2502.13184
3. "How Tesla Continuously and Automatically Improves Autopilot and Full Self-Driving Capability On 5M+ Cars". Code Compass (Substack), n.d. https://codecompass00.substack.com/p/tesla-data-engine-trigger-classifiers
4. Industry analysis of Tesla shadow mode and data closed loop. 42how (en.42how.com), n.d. https://en.42how.com/?p=6897
5. "Tesla AI chief explains why self-driving cars don't need lidar". TechTalks (B. Dickson), 2021. https://bdtechtalks.com/2021/06/28/tesla-computer-vision-autonomous-driving/
6. "Andrej Karpathy's CVPR Talk, Annotated". David Silver, 2021. https://davidsilver.blog/2021/07/05/andrej-karpathys-cvpr-talk-annotated/
7. "Karpathy on Tesla Autopilot at CVPR'21". Dynamically Typed (L. Ploeg), 2021. https://dynamicallytyped.com/stories/2021/karpathy-autopilot-cvpr/
8. "How Tesla's Autopilot Team Refines An Unfathomable Amount Of Data Is Pretty Cool". CleanTechnica, 2021. https://cleantechnica.com/2021/08/27/how-teslas-autopilot-team-refines-an-unfathomable-amount-of-data-is-pretty-cool/
9. "Tesla full self-driving release date: AI Day reveals a crucial step forward". Inverse, 2021. https://www.inverse.com/innovation/tesla-full-self-driving-release-date-ai-day
10. "LIVE Blog: Tesla AI Day". Teslarati, 2021. https://www.teslarati.com/tesla-ai-day-live-blog/
11. "Tesla's layoffs hit Autopilot team as AI develops". CNN Business, 2022. https://amp.cnn.com/cnn/2022/06/29/cars/tesla-autopilot-data-label-layoffs/index.html
12. "Recap Tesla AI Day 2022". nocode.ai, 2022. https://www.nocode.ai/recap-tesla-ai-day-2022/
13. "AI Day 2022: FSD Simplified". Drive Tesla Canada, 2022. https://driveteslacanada.ca/news/ai-day-2022-fsd-simplified/
14. "Generating ground truth for machine learning from time series elements" (US 10,997,461; US 12,223,428). Tesla, Inc. (Elluswamy, Bauch, Payne, Karpathy, Polin), 2021/2025. https://patents.google.com/patent/US12223428
15. Ashok Elluswamy, "Tesla's approach to Autonomy" (ICCV 2025 talk post). X, 2025. https://x.com/aelluswamy/status/1981644831790379245
16. "Tesla VP Ashok Describes Technology of FSD". NextBigFuture, 2025. https://www.nextbigfuture.com/2025/10/tesla-vp-ashok-describes-technology-of-fsd.html
17. "Content Search". Waymo, 2020. https://waymo.com/blog/2020/02/content-search/
18. "Demonstrably Safe AI For Autonomous Driving". Waymo, 2025. https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/
19. "The Waymo World Model: A New Frontier For Autonomous Driving Simulation". Waymo, 2026. https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/
20. "10 AI Lessons from Driving 200+ Million Fully Autonomous Miles". Waymo (S. Thirumalai), 2026. https://waymo.com/blog/2026/08/10ailessons/
21. "Waymo takes a shot at Tesla's self-driving: it's a 'false summit'". Electrek, 2026. https://electrek.co/2026/08/27/waymo-tesla-self-driving-false-summit/
22. "Exclusive: Waymo says there's no AI shortcut to self-driving". Axios, 2026. https://www.axios.com/2026/08/26/waymo-ai-shortcut-self-driving
23. "Off road, but not offline: How simulation helps advance our Waymo Driver". Waymo, 2020 (plus Waymo simulation job listings, 2025-2026). https://waymo.com/blog/2020/04/off-road-but-not-offline--simulation27/
24. "How ML at Waymo is building a scalable autonomous driver (Dmitri Dolgov)". Scale AI Exchange, 2022. https://exchange.scale.com/public/blogs/how-ml-waymo-building-scalable-autonomous-driver-dmitri-dolgov
25. "Waymo taps Google DeepMind's Genie 3 to simulate driving scenarios its cars have never seen". The Decoder, 2026. https://the-decoder.com/waymo-taps-google-deepminds-genie-3-to-simulate-driving-scenarios-its-cars-have-never-seen/
26. "Wayve Unveils GAIA-2: Cutting-Edge Scalable Video Generation for Assisted and Automated Driving". Wayve, 2025. https://wayve.ai/press/wayve-unveils-gaia2/
27. L. Russell, A. Hu, et al., "GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving". arXiv:2503.20523, Wayve, 2025. https://arxiv.org/abs/2503.20523
28. "Wayve's GAIA-1 9B generates synthetic video to train autonomous vehicles". The Decoder, 2023. https://the-decoder.com/wayves-gaia-1-9b-generates-synthetic-video-to-train-autonomous-vehicles/
29. "Wayve launches PRISM-1 for advanced autonomous driving simulations". ADAS & Autonomous Vehicle International, 2024. https://www.autonomousvehicleinternational.com/news/simulation-software/wayve-launches-prism-1-for-advanced-autonomous-driving-simulations.html
30. "Global Learning, Local Driving: Lessons From Japan". Wayve, 2025. https://wayve.ai/thinking/japan-generalization/
31. "Nissan to integrate Wayve AI tech in next-gen ProPilot ADAS". ADAS & Autonomous Vehicle International, 2025. https://www.autonomousvehicleinternational.com/news/adas/nissan-to-integrate-wayve-ai-tech-in-next-gen-propilot-adas.html
32. "Cruise's Continuous Learning Machine Predicts the Unpredictable on San Francisco Roads". Cruise (Medium), 2020. https://medium.com/cruise/cruise-continuous-learning-machine-30d60f4c691b
33. "Cruise's AI enables driverless vehicles to navigate uncommon scenarios". VentureBeat, 2020. https://venturebeat.com/ai/cruises-ai-enables-driverless-vehicles-to-navigate-uncommon-scenarios
34. "Cruise updates the AI brains behind its autonomous cars twice weekly". VentureBeat, n.d. https://venturebeat.com/ai/cruise-updates-the-ai-brains-behind-its-autonomous-cars-twice-weekly
35. "Predicting the Unpredictable With Cruise's Continuous Learning Machine". MLconf session, n.d. https://mlconf.com/sessions/predicting-the-unpredictable-with-cruises-continuous-learning-machine/
36. "How GM Cruise Leverages Machine Learning". GM Authority, 2020. https://gmauthority.com/blog/2020/09/how-cruise-uses-machine-learning-to-predict-the-unpredictable/
37. "Scale AI acquires SiaSearch to give companies the same data engine superpowers that Tesla has". TechCrunch, 2021. https://techcrunch.com/2021/11/03/scale-ai-acquires-siasearch-to-give-companies-the-same-data-engine-superpowers-that-tesla-has/
38. "Introducing Scale Nucleus". Scale AI, n.d. https://scale.com/blog/introducing-scale-nucleus
39. "Find Inaccurate Annotations" / "Find False Positive Predictions" (Nucleus docs) and "Nucleus: new Autotag". Scale AI, c. 2022. https://nucleus.scale.com/docs/find-inaccurate-annotations
40. "Nucleus now supports 3D Scenes". Scale AI, n.d. https://scale.com/blog/nucleus-3d
41. B. Levin, "Expanding Our Data Engine for Physical AI". Scale AI, 2025. https://scale.com/blog/physical-ai
42. "Meta invests $14B in Scale AI, acquires 49% stake and superintelligence lab leader". The Decoder, 2025. https://the-decoder.com/meta-invests-14b-in-scale-ai-acquires-49-stake-and-superintelligence-lab-leader/
43. "Data Flywheel Blueprint" (README and docs). NVIDIA AI Blueprints (GitHub), 2025-2026. https://github.com/NVIDIA-AI-Blueprints/data-flywheel
44. "Now Available: The NVIDIA AI Blueprint for Building Data Flywheels". NVIDIA Developer Forums, 2025. https://forums.developer.nvidia.com/t/now-available-the-nvidia-ai-blueprint-for-building-data-flywheels/335894
45. "NVIDIA Announces Major Release of Cosmos World Foundation Models and Physical AI Data Tools". NVIDIA Newsroom, 2025. https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools
46. "Accelerating AV Simulation with Neural Reconstruction and World Foundation Models". NVIDIA Technical Blog, 2025. https://developer.nvidia.com/blog/accelerating-av-simulation-with-neural-reconstruction-and-world-foundation-models/
47. "Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models". NVIDIA (arXiv:2506.09042 and GitHub), 2025. https://github.com/nv-tlabs/Cosmos-Drive-Dreams
48. "Cosmos Curator". NVIDIA (GitHub), 2025-2026. https://github.com/nvidia-cosmos/cosmos-curate
49. "Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities". NVIDIA Technical Blog, 2025. https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities
50. "Into the Omniverse: NVIDIA GTC Showcases Virtual Worlds Powering the Physical AI Era" (Physical AI Data Factory Blueprint). NVIDIA Blog, 2026. https://blogs.nvidia.com/blog/gtc-2026-virtual-worlds-physical-ai/
51. "Alpamayo 1 (formerly Alpamayo-R1)". NVIDIA Research / NVlabs (GitHub), 2025-2026. https://github.com/NVlabs/alpamayo
52. "AlpaSim". NVIDIA NVlabs (GitHub), 2025-2026. https://github.com/NVlabs/alpasim
53. "REM: Road Experience Management". Mobileye, n.d.; plus ADAS & AV International, January 2024. https://mobileye.com/technology/rem
54. "openpilot 0.11" (comma.ai blog), "Learning to Drive from a World Model" (arXiv:2504.19077), and commaVQ (GitHub). comma.ai, 2023-2025. https://blog.comma.ai/011release/
55. "Hyundai Motor Group Accelerates Autonomous Driving Innovation with AI-Powered Data Flywheel". Hyundai Motor Group, n.d. https://www.hyundaimotorgroup.com/en/news/hyundai-motor-group-accelerates-autonomous-driving-innovation-with-ai-powered-data-flywheel
56. "Active Learning, Data Selection, Data Auto-Labeling, and Simulation in Autonomous Driving, Part 5". I. Kargar (Medium), n.d. https://kargarisaac.medium.com/active-learning-data-selection-data-auto-labeling-and-simulation-in-autonomous-driving-part-5-82d7de92e82f
57. "AI Orchestration for Data Flywheel". Iguazio (build.nvidia.com), 2025. https://build.nvidia.com/iguazio/ai-orchestration-for-data-flywheel
58. "Technically Speaking: Learning Every Mile Driven". Motional, n.d. https://motional.com/news/technically-speaking-learning-every-mile-driven
59. "openpilot release notes" (RELEASES.md, versions 0.9.9 to 0.11.2). comma.ai (GitHub), 2025-2026. https://github.com/commaai/openpilot/blob/master/RELEASES.md
60. "Alpamayo Recipes" (hub for Alpamayo 1 / 1.5 Nano, Alpamayo 2 Super, SFT and RL recipes, CoC auto-labeling pipeline). NVIDIA NVlabs (GitHub), 2026. https://github.com/NVlabs/alpamayo-recipes
61. "AlpaGym: reinforcement learning for end-to-end driving policies in AlpaSim". NVIDIA NVlabs (GitHub), 2026. https://github.com/NVlabs/alpagym
62. "Alpamayo 1.5". NVIDIA NVlabs (GitHub), 2026. https://github.com/NVlabs/alpamayo1.5
63. D. Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking" (NeurIPS 2024 D&B, arXiv:2406.15349); W. Cao et al., "Pseudo-Simulation for Autonomous Driving" (CoRL 2025, arXiv:2506.04218). https://github.com/autonomousvision/navsim
64. X. Jia et al., "Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving" (NeurIPS 2024 D&B, arXiv:2406.03877), plus v0.0.4 update notes (August 2026). Thinklab-SJTU (GitHub). https://github.com/Thinklab-SJTU/Bench2Drive
65. J.-J. Hwang et al., "EMMA: End-to-End Multimodal Model for Autonomous Driving". arXiv:2410.23262, Waymo, 2024. https://arxiv.org/abs/2410.23262
66. "Waymo Open Dataset" (Perception, Motion and End-to-End Driving datasets). Waymo Research (GitHub), 2020-2026. https://github.com/waymo-research/waymo-open-dataset
67. C. R. Qi, Y. Zhou, M. Najibi, P. Sun, K. Vo, B. Deng, C. Fowlkes, D. Anguelov, "Offboard 3D Object Detection from Point Cloud Sequences". CVPR 2021 (arXiv:2103.05073). https://arxiv.org/abs/2103.05073
68. "π*0.6: a VLA That Learns From Experience" (Recap method). Physical Intelligence, November 2025. https://www.pi.website/blog/pistar06
69. "GM to refocus autonomous driving development as Cruise robotaxi funding ends". General Motors press release, 10 December 2024 (widely reported; primary URL not retrieved here). https://news.gm.com/
70. "openpi" (π0, π0-FAST, π0.5 open models). Physical Intelligence (GitHub), 2025. https://github.com/Physical-Intelligence/openpi
