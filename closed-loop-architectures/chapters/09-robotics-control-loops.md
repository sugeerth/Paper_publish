# 09. Closed-Loop Robotics and Control: Sim-to-Real Loops, Robot Foundation Model Data Flywheels, Teleoperation, DAgger, and World Models

## Overview

Robotics is where "closed loop" stops being a metaphor. In the other chapters of this atlas the loop runs through logs, labels and retraining jobs. Here, the innermost loop is literal: a policy reads a camera frame and a joint state, emits a motor command, and the world answers within milliseconds. If the policy is wrong, the next observation is already off the training distribution. Errors compound. Supervised learning on fixed data does not handle this well, and every design in this chapter is, in one way or another, a response to it.

Modern robot-learning systems stack at least five loops, each running at a very different timescale:

1. **The control loop (milliseconds).** A learned policy, often a vision-language-action model (VLA), produces actions at 3 Hz [1] up to 200 Hz [32]. It usually emits *chunks* of future actions that a lower-level controller then tracks.
2. **The correction loop (seconds to minutes).** A human or an automated supervisor intervenes when the policy drifts. Each correction is labeled supervision collected *on the policy's own state distribution*. This is the DAgger idea [25] and its human-gated descendants [26][27], which RL-from-intervention systems such as HIL-SERL [29] and Physical Intelligence's Recap [16] now extend.
3. **The fleet data flywheel (days to months).** Teleoperators and autonomous robots collect episodes. The episodes are curated, pooled across embodiments, and used to train a generalist model. The model is redeployed, which in turn collects better data. RT-1 [1], Open X-Embodiment [3], DROID [20], AutoRT [17], the π family [8][12][16] and NVIDIA GR00T [33] are all instances.
4. **The simulation loop (hours).** Policies train in massively parallel simulators with randomized physics and rendering. They transfer to hardware, and real-world failures feed back into how the simulator and its randomization ranges are set up [44][46][47][49]. Simulators are also used *in reverse*, as cheap evaluators of real-robot policies [52].
5. **The world-model loop.** A learned model of the environment stands in for reality. It is used to train behavior "in imagination" [40][41], to plan [42], to synthesize training trajectories [35], or to score policies before they touch hardware [38].

This chapter covers each loop with concrete published systems, gives their numbers where they are public, and closes with design lessons.

**Sourcing note.** This chapter was assembled under restricted network access. Most facts were checked against primary sources fetched during writing: official GitHub repositories, project pages and their source, and documentation from Google DeepMind, Physical Intelligence, NVIDIA, Meta, Hugging Face, Stanford, Berkeley and 1X. These are references [1]-[10], [12]-[17], [19]-[24], [28]-[31], [33]-[36], [38]-[42], [49]-[55]. The following could not be re-fetched in this revision: the company announcement pages for Figure Helix [32] and Physical Intelligence π*0.6 [16]; Tesla's Optimus page [37]; and several classic papers [11][18][25]-[27][43]-[48]. Claims drawn from them reflect those publications as the author knows them, and the numbers are marked approximate. Readers should confirm them against the linked originals.

---

## 1. Reference architecture: nested loops around a robot policy

```
 +-----------------------------------------------------------------------------------+
 | L4  WORLD-MODEL / SIM LOOP (hours-days)                                           |
 |   physics sim (Isaac Lab, MuJoCo)   learned video/latent world model (Cosmos,     |
 |   domain randomization, real2sim      V-JEPA 2-AC, Genie-style)                   |
 |       |  synthetic demos / RL rollouts / policy evaluation scores        ^        |
 +-------|------------------------------------------------------------------|--------+
         v                                                                  |
 +-----------------------------------------------------------------------------------+
 | L3  FLEET DATA FLYWHEEL (days-months)                                             |
 |   teleop rigs ---> episode store ---> curation/filtering ---> pretrain/post-train |
 |   (ALOHA, DROID,   (RLDS, LeRobot)    (idle filter, success   generalist VLA      |
 |    UMI, VR)              ^             labels, relabeling)          |             |
 |                          |                                          v             |
 |                autonomous rollouts <---- deploy checkpoint to fleet <---          |
 +--------------------------^--------------------------------------------------------+
                            |
 +--------------------------|--------------------------------------------------------+
 | L2  CORRECTION LOOP (seconds-minutes)                                             |
 |   human / supervisor watches rollout -> takes over (gated) -> correction labels   |
 |   reward classifier / value function -> RL update (HIL-SERL, Recap)               |
 +--------------------------^--------------------------------------------------------+
                            |
 +--------------------------|--------------------------------------------------------+
 | L1  POLICY LOOP (3-50 Hz VLA; up to 200 Hz fast head)                             |
 |   images + proprio + language --> VLA --> action chunk (H steps)                  |
 | L0  SERVO LOOP (hundreds of Hz - kHz): joint PD / whole-body controller tracks it |
 +-----------------------------------------------------------------------------------+
```

The figure is a synthesis of the systems below, not any one company's diagram. Three properties recur:

- **Coupling across loops.** A choice of action representation at L1 determines what data L3 can pool, and how readily L2 corrections can be applied. Examples are delta end-effector actions versus joint positions, and chunk length.
- **Supervision on the policy's own distribution.** L2 and the autonomous half of L3 exist to collect it, because offline demonstrations alone produce compounding errors.
- **Evaluation is the bottleneck.** Real-robot evaluation is slow and noisy. L4 exists partly to make evaluation cheap (SIMPLER [52], 1X's world-model evaluation challenge [38], RoboArena [55]).

---

## 2. The innermost loop: control rates, action chunking and latency

### 2.1 Problem

Large models are slow, and robots need smooth, reactive commands. A 7B-parameter VLA that decodes one action token at a time cannot close a 50 Hz loop. A policy that predicts one step at a time also tends to suffer from compounding error and jittery, multimodal behavior.

### 2.2 Designs and numbers

**RT-1 (Google, 2022).** An EfficientNet image encoder, FiLM-conditioned on the instruction, feeds a TokenLearner and then a Transformer that outputs discretized action tokens. The action space has 7 arm dimensions, 3 base dimensions and a mode switch. RT-1 "performs closed-loop control and commands actions at 3Hz" [1]. RT-1-X keeps the same interface: "assuming the model runs at three hertz, every 333 milliseconds, we feed the latest RGB image" [4].

**RT-2 (2023).** RT-2 represents actions as text tokens, such as the string "1 128 91 241 5 101 127 217", and *co-fine-tunes* a web-scale VLM (PaLI-X 55B or PaLM-E 12B) on robot data mixed with vision-language data [2]. It reports roughly 3x improvement on emergent-skill evaluations over baselines and roughly 2x on generalization axes [2]. Token-by-token decoding of a 55B model limits the control rate. The project page reports evaluation, not throughput.

**ACT / ALOHA (Stanford, 2023).** Action Chunking with Transformers predicts a *chunk* of future joint targets instead of one step. This reduces the effective horizon over which errors compound [18]. The README's main practical tip is a lesson in its own right: "if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus" [18].

**Diffusion Policy (Columbia/Stanford/TRI, 2023)** models the action-chunk distribution with denoising diffusion. This handles multimodal demonstrations, for example two equally valid ways around an obstacle [22].

**π0 (Physical Intelligence, 2024)** attaches a separate *action expert* to a PaliGemma VLM backbone and trains it with *flow matching*. It produces continuous action chunks. PI describes high-frequency chunked control (chunks of about 50 steps at up to about 50 Hz; approximate) [8]. The open-source openpi release serves a policy whose `infer()` call returns an `action_chunk`. It needs more than 8 GB of GPU memory for inference, more than 22.5 GB for LoRA fine-tuning and more than 70 GB for full fine-tuning [9].

**OpenVLA → OpenVLA-OFT (Stanford, 2024-2025).** OpenVLA is a 7B VLA built on Prismatic (DINOv2 and SigLIP vision encoders with a Llama-2 language model) and trained on 970K Open X-Embodiment trajectories [6]. The Optimized Fine-Tuning recipe combines four things: parallel decoding, action chunking, continuous actions and an L1 regression loss. With it, the authors report "25-50x" faster inference, "26x faster action generation speed and 3x lower latency", 97.1% average success on LIBERO, and bimanual ALOHA control at 25 Hz with a 7B model [7].

**Figure Helix (2025)** splits the model in two. "System 2" is a roughly 7B VLM that runs at about 7-9 Hz and produces a latent. "System 1" is a roughly 80M-parameter visuomotor transformer that runs at about 200 Hz, consumes that latent, and outputs continuous upper-body control: about 35 degrees of freedom including fingers. Both run on embedded GPUs on the robot (figures approximate, per Figure's announcement) [32].

**NVIDIA GR00T N1.x** uses the same split between a "slow" VLM and a "fast" action head. The current N1.7 pairs a Cosmos-Reason2-2B VLM (Qwen3-VL architecture) with a flow-matching diffusion-transformer action head. In N1.7 the action head has 16 layers, the action horizon is 40 steps (up from 16), and state/action dimensions were expanded to 132. The base model is 3B parameters and runs on one GPU with at least 16 GB, including Jetson AGX Thor/Orin. The deployment path exports to ONNX and TensorRT [33].

### 2.3 Latency engineering inside the loop

Chunking creates a new problem: what should the robot do while the next chunk is being computed? Three published answers stand out:

- **Latency matching (UMI).** The policy interface for the Universal Manipulation Interface uses "inference-time latency matching and a relative-trajectory action representation". The ablation shows that without observation/action latency matching "the policy exhibits significantly more jittery movements, due to the misalignment between observations and action executions" [21].
- **Real-time chunking (RTC, PI 2025).** The next chunk is generated while the current one is still executing. Generation is treated as *inpainting*: actions that will be executed during the inference delay are frozen, and the remainder is generated consistently with them [14]. A follow-up moves this conditioning into training time to make it cheaper [15]. The released simulation benchmark trains expert policies on 12 Kinetix levels, at 8 seeds and 65M environment steps per level, and then distills them into flow policies [15].
- **Remote inference.** openpi supports running the model on a separate GPU server and streaming actions to the robot over a websocket. Robot and policy environments stay decoupled, and images are resized client-side "to minimize bandwidth / latency" [9].

### 2.4 Lessons

- Chunk length trades reactivity against smoothness and compute amortization. GR00T's rename of `--action-horizon` to `--execution-horizon` [33] makes the distinction explicit: the model predicts 40 steps, but you choose how many to execute before replanning.
- A dual-rate design (slow semantic model, fast motor head) is now the common humanoid pattern [32][33]. It isolates the expensive model from the control-rate constraint.

---

## 3. Teleoperation and human data collection: the supply side of the flywheel

### 3.1 Problem

Robot data has no internet-scale corpus. Every trajectory costs operator time, robot time and a physical scene. Collection design determines four things: data cost, data diversity, and both the action-space and sensor compatibility of what gets pooled.

### 3.2 Systems

**ALOHA** is a low-cost bimanual leader-follower rig built from four Interbotix arms (two leaders, two followers) and four cameras, running on ROS 1. The operator back-drives the leader arms and the followers mirror them [18]. Even the setup notes reflect loop latency. The README tells builders to disable a forward-kinematics computation that "delays teleoperation", and warns that USB hubs can destabilize four-arm connections.

**Mobile ALOHA** adds a mobile base and whole-body teleoperation. The key data-flywheel result is that *co-training* with existing static ALOHA datasets boosts mobile-manipulation performance: "With 50 demonstrations for each task, co-training can increase success rates by up to 90%" [19]. Old data from a related embodiment substitutes for new data.

**DROID** is the clearest public example of *standardized, distributed* collection. The hardware is identical at every site: a Franka Panda arm, two Zed 2 stereo cameras, a wrist-mounted Zed Mini, and an Oculus Quest 2 for teleoperation, all mounted on a portable desk. With it, 50 collectors at 13 institutions across North America, Asia and Europe gathered **76k demonstration trajectories (350 h) in 564 scenes and 86 tasks over 12 months**, covering 1,417 camera viewpoints [20]. Eighteen platforms were built using the same guide [20]. Follow-up releases added three language annotations each for about 75k successful episodes (December 2024) and improved camera calibrations for 36k episodes (April 2025) [20]. This is post-hoc enrichment of an existing corpus.

**DROID's hidden curation cost.** Physical Intelligence's open reproduction of π0.5-DROID training shows that the raw corpus is not training-ready. "The DROID dataset contains many *idle* timesteps in which the robot does not move (in part due to the VR teleoperation interface...)". The recipe precomputes sample indices that skip time steps "for which the next chunk of actions would be largely idle". The full RLDS dataset needs 1.8 TB. Training takes about 2 days on 8x H100 (100k iterations, batch 256, about 1 epoch) from a π0 initialization, versus about 5 days from raw PaliGemma [10]. The teleop interface's ergonomics left a statistical artifact that every downstream consumer must filter.

**UMI** removes the robot from collection altogether. Humans carry 3D-printed hand-held grippers with a wrist camera, and SLAM recovers the trajectories. Policies trained on this data "zero-shot generalize to novel environments and objects" and are "deployable across multiple robot platforms" [21]. Hardware independence comes from relative-trajectory actions plus latency matching.

**Whole-body humanoid teleoperation.** NVIDIA's GR00T-WholeBodyControl releases VR whole-body teleoperation built on the SONIC learned controller. One 2026 checkpoint adds a "4-frame SMPL reference lookahead for more responsive whole-body teleoperation". There is an end-to-end workflow to "collect teleop data, fine-tune Isaac-GR00T N1.7, and deploy with SONIC whole-body control" on a Unitree G1 [34]. In GR00T N1.7 the VLA predicts "compact latent action tokens that a learned whole-body controller decodes into full-body joint commands" [33]. The same learned controller serves both as the teleop retargeter and as the policy's actuator, so data and deployment share one action interface.

**Amplifying human demos.** MimicGen generates "over 50,000 demonstrations from less than 200 human demonstrations across 18 tasks". The released datasets contain over 48,000 demos across 12 tasks [23]. It does this by segmenting demonstrations into object-centric pieces and transforming them to new object poses. DexMimicGen extends the approach to bimanual dexterous humanoids [24]. The related HITL-TAMP system lets one operator manage "a fleet of parallel data collection sessions" through a queue. In a 15-user study it gathered more than 3x the demos of conventional teleoperation in the same time budget [23].

**Formats as infrastructure.** Open X-Embodiment standardizes on RLDS episodes [4]. Hugging Face LeRobot defines the LeRobotDataset (Parquet plus MP4) hosted on the Hub, with tools to delete episodes, split and merge datasets [30]. openpi notes that LeRobot was not yet "scalable enough" for full DROID training and uses RLDS there instead [10].

### 3.3 Failure modes and lessons

- **Interface artifacts become data artifacts.** Examples are idle frames from VR clutching [10] and inconsistent action frames. SERL's April 2024 release fixed "a major issue in the intervention action frame" [28].
- **Standardized hardware buys poolability.** DROID's single bill of materials [20] is the reason its 350 hours are useful as a single dataset.
- **Human-video pretraining is the new frontier for cheap supply.** GR00T N1.7 attributes improved generalization and language-following to "20K hours of EgoScale human video data". It works because a relative end-effector action space is "consistent across both human and robot data" [33].

---

## 4. Cross-embodiment pooling: from RT-1 to Open X-Embodiment

### 4.1 RT-1: the single-fleet flywheel

RT-1's data came from **"over 130k episodes, which contain over 700 tasks, ... collected with a fleet of 13 robots over 17 months"** [1]. RT-1 performed 97% of the more than 700 training instructions, 76% of unseen instructions, and 83% / 59% on distractor and background robustness. These were large margins over Gato and BC-Z [1]. The project page states the flywheel rationale outright: since no skill-specific assumptions are made, "we can continuously provide more diverse data to improve its capabilities" [1]. The released checkpoints include RT-1 co-trained with Kuka data and with sim and real data [1].

### 4.2 Open X-Embodiment and RT-X

The Open X-Embodiment collaboration pooled **60 existing datasets from 34 labs into 1M+ trajectories across 22 embodiments**: 527 skills and 160,266 tasks from 21 institutions [3]. Its central tool for heterogeneity is a *lowest-common-denominator action space*. "The robot action is a 7-dimensional vector consisting of x, y, z, roll, pitch, yaw, and gripper opening or the rates of these quantities." Unused dimensions are zeroed [3]. Results:

- **RT-1-X** outperforms RT-1 or the original per-dataset methods by about 50% on average in the small-data domain [3].
- **RT-2-X (55B)** outperforms RT-2 by about 3x on emergent-skill evaluations. Its behavior is modulated by prepositions ("on" vs "near") [3].

The open repo invites new dataset contributions through an enrollment form [3][4]. The dataset is designed as a commons that grows.

### 4.3 Open generalists built on the pool

- **Octo (2024)** is a transformer-based diffusion policy pretrained on 800k episodes from 25 OXE datasets. It comes in 27M and 93M variants, running at 17 and 13 it/s on one RTX 4090. It handles heterogeneous sensors and labels and is designed for fine-tuning to new observation and action spaces [5].
- **OpenVLA (2024)** is a 7B VLA trained on 970K OXE trajectories, fine-tunable with LoRA [6]. OFT then revises the fine-tuning recipe (section 2) [7].

### 4.4 Lessons

Pooling across embodiments helps most where each lab's data is small (RT-1-X). Larger models (RT-2-X) absorb pooled data better. Both results depend on careful action-space normalization. Later systems move away from the 7-DoF lowest common denominator toward embodiment-specific heads or relative end-effector spaces [5][33].

---

## 5. Physical Intelligence: the π model family as a closed loop

PI's public record shows the clearest end-to-end commercial flywheel among the robotics labs: pretraining on a large mixed corpus, post-training per task, and since late 2025, RL from the model's own deployments.

```
   cross-embodiment robot data (10k+ h)     web / VLM data     verbal & subtask labels
                 \                               |                    /
                  v                              v                   v
          +-----------------------------------------------------------------+
          | PRETRAIN  VLM backbone (PaliGemma) + action expert              |
          |           pi0: flow matching;  pi0-FAST: FAST tokens;           |
          |           pi0.5: co-training + knowledge insulation             |
          +-------------------------------+---------------------------------+
                                          | post-train (task demos)
                                          v
          +-----------------------------------------------------------------+
          | DEPLOY  -> autonomous episodes + expert teleop corrections      |
          +-------------------------------+---------------------------------+
                                          | pi*0.6 "Recap"
                                          v
          +-----------------------------------------------------------------+
          | value function fit on all data -> advantage per step            |
          | policy re-trained conditioned on advantage indicator            |
          | (demos + autonomous rollouts + interventions) -> redeploy       |
          +-----------------------------------------------------------------+
```

**π0 (Oct 2024).** A flow-matching VLA: a PaliGemma VLM plus an action expert that generates continuous action chunks [8]. PI's pretraining mixture spans multiple robot configurations, including single-arm, bimanual and mobile platforms. openpi's base checkpoints are "pre-trained on 10k+ hours of robot data" [9]. PI cautions that π0 "was developed for our own robots ... we do not expect every such attempt to be successful" on other platforms [9]. This is a candid statement that cross-embodiment transfer is not free.

**π0-FAST (Jan 2025)** replaces continuous decoding with an autoregressive model over FAST tokens. FAST is a compression-based tokenizer for action sequences [11]. openpi notes the trade-off. π0-FAST-DROID follows language better. π0-DROID has "faster inference ... but may not follow language commands as well" [9].

**π0.5 (Apr/Sep 2025)** is "an upgraded version of π0 with better open-world generalization trained with knowledge insulation" [9][12]. Knowledge insulation trains the VLM backbone with discrete action tokens while insulating it from gradients of the continuous action expert. The goal is to protect the backbone's pretrained semantics and to train fast while running fast [13]. The π0.5-DROID checkpoint offers "fast inference and good language-following" [9].

**π*0.6 and Recap (Nov 2025)** closes the loop with reinforcement learning. Recap stands for "RL with Experience and Corrections via Advantage-conditioned Policies". It pools three data sources: (1) demonstrations, (2) the policy's own autonomous rollouts, and (3) expert teleoperated *corrections* during deployment. A learned value function scores progress. The policy is then trained conditioned on an advantage indicator, and at deployment it is asked for "high-advantage" behavior [16]. PI applies this to long, real tasks: making espresso drinks, folding diverse laundry, and assembling boxes. It reports roughly 2x throughput and roughly halved failure rates on some of the hardest tasks (approximate; see [16]). Architecturally, Recap turns a supervised VLA into a policy-improvement operator *without* switching to on-policy policy-gradient training. That matters for a multi-billion-parameter flow model.

**Open recipe details that matter in practice** [9][10]:

| Item | Published value |
|---|---|
| Base pretraining corpus | 10k+ hours robot data |
| Inference GPU memory | > 8 GB (e.g., RTX 4090) |
| LoRA / full fine-tune memory | > 22.5 GB / > 70 GB |
| Full-DROID fine-tune | about 2 days on 8x H100 from π0 init; about 5 days from PaliGemma |
| DROID RLDS on disk | 1.8 TB |
| Action space caveat | original π0.5-DROID used joint velocity; open recipe uses joint position because velocity is "much harder to simulate" |

The last row is a sim/real loop decision. The action space is chosen so that policies remain evaluable in simulation.

**Failure modes acknowledged.** These include platform mismatch [9], LoRA under-performing for full-DROID fine-tuning ("haven't found the policies to perform well so far") [10], and idle-data contamination [10].

---

## 6. Autonomous fleet data collection: AutoRT

**Problem.** Teleoperation scales linearly with human hours. Can foundation models direct a fleet to collect useful data with less supervision?

**Architecture.** Google DeepMind's AutoRT (2024) runs a five-stage loop on each robot [17]:

```
 explore map -> pick point of interest -> drive there
        |
        v
 VLM describes scene + objects ---> LLM proposes tasks
        |
        v
 LLM critic filters tasks with a "robot constitution"
 (foundational / safety / embodiment rules) and routes each to:
     [autonomous policy]  or  [human teleop]  or  [reject]
        |
        v
 execute -> score episode diversity vs prior data -> repeat
```

**Scale.** Up to 20 robots at once across multiple buildings collected **77,000+ episodes over 6,650 unique instructions**. A configurable fraction was human-demonstrated, which sets how autonomous the fleet is [17].

**Safety as a prompt-level control.** With the constitution, 88% of generated tasks were valid, rising to 93% after one round of filtering. In adversarial scenes (for example, sharp objects), valid-task rates were 83% with the constitution and 18% without [17]. Language-embedding diversity was higher than RT-1, BC-Z and Language Table data, and episodes with human assistance were the most visually diverse [17].

**Lessons.** The LLM's job here is *curriculum and routing*, not control. AutoRT's authors call it "a research proof-of-concept". The collected data was used to fine-tune RT-1 as a sanity check [17]. The main limit of the design is that the downstream policy quality bounds what autonomous episodes can teach.

---

## 7. Learning from corrections: DAgger to human-in-the-loop RL

### 7.1 DAgger and human-gated variants

**DAgger** (Ross, Gordon, Bagnell, 2011) reduces imitation learning to no-regret online learning. Roll out the current policy, have the expert label the states it *visits*, aggregate those labels into the dataset, and retrain. This removes the covariate shift that makes behavior cloning's error compound with horizon [25].

On physical robots, asking an expert to label every visited state is impractical and unsafe. Two practical descendants address this:

- **HG-DAgger** (2019): the human *gates* control. They take over only when the robot is heading somewhere unsafe, and only the intervention segments become labels [26].
- **ThriftyDAgger** (2021): the robot itself requests help based on estimated novelty and risk, under a human attention budget. This allows one human to supervise several robots [27].

These ideas reappear directly in modern stacks. Intervention data in HIL-SERL and Recap, and AutoRT's routing of tasks to humans, are all gated DAgger variants at fleet scale.

### 7.2 SERL and HIL-SERL: on-robot RL with humans in the loop

**SERL** (Berkeley et al., 2024) is a software suite for real-robot RL. It combines sample-efficient off-policy RL, reward-specification tools and robot controllers. On PCB insertion, cable routing and reset-free object relocation it trains "in just 25 to 50 minutes" [28]. Operational advice from the maintainers: add a small penalty on gripper action changes, and "providing interventions online during training in addition to loading the offline demos" by "just touching the SpaceMouse" [28].

**HIL-SERL** (2024) formalizes the loop [29]:

```
 1. teleop positive/negative examples -> train binary REWARD CLASSIFIER
 2. collect small demo set ------------> seed DEMO BUFFER
 3. online:  ACTOR (robot) --transitions--> LEARNER (off-policy RL, SAC-style)
                ^   human intervention        |
                |   (frequent early,          | periodic weight sync
                |    tapering later)          v
                +------------------------- updated policy
```

The actor and learner run asynchronously and communicate over the network. "The learner will periodically synchronize the policy with the actor" [29]. Interventions start frequent, to show solutions from various states and block unsafe behavior, and are "gradually reduce[d] ... as the policy reaches higher success rate and faster cycle times" [29]. Hugging Face LeRobot ships a HIL-SERL workflow with a distributed SAC learner, gamepad interventions and safety tools: joint/end-effector bounds and region-of-interest cropping. It advises users to keep tasks to "5-10 seconds" [30].

### 7.3 RL on top of generalist VLAs

By late 2025 the correction loop had moved onto foundation models. PI's Recap (section 5) [16] is one example. ByteDance's GR-RL applies RL to make a generalist VLA "dexterous and precise for long-horizon robotic manipulation" [31]. GR-3 is the base VLA from the same group [54].

### 7.4 Failure modes

- **Reward hacking of learned classifiers.** HIL-SERL depends on a vision success classifier trained from a few examples [29]. False positives become rewarded behavior. This is why LeRobot's guidance stresses simple, short tasks and ROI cropping [30].
- **Intervention-frame and timing bugs** silently corrupt labels [28].
- **Human attention is the scarce resource.** ThriftyDAgger-style budgets and AutoRT-style routing both exist to ration it [17][27].

---

## 8. Humanoid and company flywheels

### 8.1 Figure: Helix

Figure's Helix (Feb 2025) is a single set of weights for full upper-body humanoid control with the dual-rate architecture of section 2. A roughly 7B VLM ("System 2") at about 7-9 Hz is paired with a roughly 80M visuomotor policy ("System 1") at about 200 Hz. Training used roughly 500 hours of teleoperated data, with instructions generated in hindsight by a VLM that labels the clips. Inference runs entirely on embedded GPUs in the robot (all figures approximate, per [32]). The design lesson is to decouple semantic latency from control latency and to automate the language labeling of teleop data. The cost is that the S1/S2 interface (a latent vector) is learned end-to-end, so failures are hard to attribute to either system.

### 8.2 NVIDIA: GR00T N1 → N1.7, Dreams and SONIC

NVIDIA's public stack is the most complete open "flywheel kit".

- **Model.** GR00T N1 [33] introduced a VLM plus diffusion-transformer action head for humanoids. N1.7 (GA, Apache 2.0) uses Cosmos-Reason2-2B, a 3B base model, relative end-effector actions shared across human and robot embodiments, and pretraining that includes 20K hours of EgoScale human video [33].
- **Workflow.** The workflow is explicitly a loop: prepare data in GR00T-LeRobot format, run zero-shot inference, fine-tune, run open-loop evaluation, then sim benchmarks (RoboCasa, SimplerEnv), then real hardware, then deploy with TensorRT [33].
- **Synthetic data from world models (DreamGen / GR00T-Dreams).** A video world model (Cosmos-Predict2) is fine-tuned on the target robot. It "dreams" videos of new tasks from an image and an instruction. An *inverse dynamics model* labels those videos with actions, and the resulting pseudo-trajectories fine-tune GR00T N1. The aim is to let robots learn "new tasks in unfamiliar environments without needing specific teleoperation data" [35]. Cosmos-Predict2.5 adds action-conditioned robot models and, as of February 2026, robot *policy* checkpoints for RoboCasa and LIBERO [36].
- **Whole-body control.** SONIC is a learned whole-body controller. It is used both for VR teleoperation and as the low-level executor of VLA latent actions. Releases include training code, checkpoints and a human-motion dataset of about 288 hours [34].

### 8.3 Tesla Optimus

Tesla has published far less technical detail on Optimus's learning stack than the other organizations here. Its public AI page describes the humanoid program at a high level [37]. Public statements and press coverage describe teleoperation and human-video data collection, but no peer-reviewed architecture, dataset size or control-rate figures were available to this chapter. Designers should treat any specific Optimus numbers seen in secondary sources as unverified.

### 8.4 1X: world models as the evaluator

1X released over 100 hours of tokenized first-person data from its EVE humanoid, along with a Genie-style baseline and a MAGVIT2 tokenizer (16x16 tokens per frame). It framed three challenges: compression (next-frame token likelihood), sampling, and an *evaluation* challenge that ranks policies by rolling them out inside a learned world model [38]. The stated motivation is that "an end-to-end learned simulator of millions of robot environments would greatly accelerate progress" [38].

### 8.5 Google DeepMind: Gemini Robotics

Gemini Robotics models are offered through a trusted-tester program. The Safari SDK provides "full lifecycle toolings": checkpoint access, serving, evaluation on robots and in simulation, data upload, fine-tuning, and download of fine-tuned checkpoints [53]. In other words, the flywheel is productized as a hosted service, where customer data uploads feed fine-tunes.

---

## 9. Sim-to-real loops

### 9.1 Problem

Simulation offers unlimited, safe, parallel experience, but it is wrong in many places: contact, friction, actuator dynamics, rendering. The sim-to-real loop is the practice of making policies robust to those errors and of feeding real-world discrepancies back into the simulator.

### 9.2 Techniques

- **Domain randomization.** Randomize textures, lighting and camera poses so that reality looks like one more sample [44]. OpenAI's dexterous hand work applied this to physics as well as vision [45]. **Automatic Domain Randomization (ADR)** widens randomization ranges automatically as the policy succeeds, producing a curriculum. It was used for solving a Rubik's cube with a robot hand [46].
- **Learned actuator models.** On ANYmal, an *actuator network* trained on real motor data replaces idealized actuator models in simulation. This closes the gap where it is largest [47]. legged_gym packages "all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes" [49].
- **Online adaptation.** RMA trains a base policy with privileged environment parameters plus an adaptation module that infers those parameters from recent state-action history at deployment [48].
- **Massive parallelism.** The legged_gym line trains thousands of simulated robots on one GPU in minutes [49]. It has since migrated to Isaac Lab, a GPU-accelerated framework with more than 16 robot models and more than 30 ready environments, compatible with RSL-RL, SKRL, RL Games and Stable-Baselines [50]. MuJoCo Playground offers GPU-accelerated MJX/MuJoCo Warp environments "for robot learning research and sim-to-real", spanning locomotion and dexterous manipulation with batch rendering for vision [51].

### 9.3 Real-to-sim evaluation: closing the loop backwards

SIMPLER builds simulated replicas of common real setups (Google robot, WidowX/Bridge). It uses two approaches. **Visual matching** overlays real images onto sim backgrounds. **Variant aggregation** averages over randomized backgrounds, lighting and distractors. Paired sim/real evaluations show strong correlation, so simulation can serve for checkpoint selection and for detecting failure modes [52]. The ManiSkill3 port runs "10-15x faster" with GPU parallelization [52]. GR00T N1.7 [33] and openpi's choice of action space [10] both treat simulated evaluation as a first-class step in the loop.

### 9.4 Failure modes

- **Exploiting simulator artifacts.** Policies find physics bugs, which randomization and actuator models are designed to blunt [47][49].
- **Action spaces that cannot be simulated** break the real-to-sim path, as with joint velocity in [10].
- **Over-randomization** produces conservative policies. ADR's curriculum [46] is one answer.

---

## 10. World models as the loop

World models stand in for some or all of the real environment. Four roles appear in the literature.

| Role | Representative system | What the model predicts | How it closes the loop |
|---|---|---|---|
| Train behavior in imagination | DreamerV3 [40], DayDreamer [41] | latent states and rewards given actions | actor-critic learns on imagined rollouts; real data refits the model |
| Plan at test time | V-JEPA 2-AC [42] | future latent embeddings given actions | model-predictive planning toward an image goal |
| Generate training data | DreamGen / GR00T-Dreams [35], Cosmos [36] | future video given an image and text | inverse-dynamics model labels dreamed video with actions; VLA trains on it |
| Evaluate policies | 1X challenge [38], UniSim [43], Genie [39] | next frames given actions | rank policies inside the learned simulator before hardware |

**DreamerV3** "learns a world model from experiences and uses it to train an actor critic policy from imagined trajectories". It uses categorical latent representations and fixed hyperparameters across domains (Nature, 2025) [40]. **DayDreamer** brought this to physical robots "without using simulators". An actor process fills a replay buffer while a learner trains the world model and the behavior on imagined predictions [41]. This is the same asynchronous actor/learner split as HIL-SERL.

**V-JEPA 2** pretrains a video encoder self-supervised on internet-scale video. **V-JEPA 2-AC** is an action-conditioned latent world model post-trained "using a small amount of robot trajectory interaction data" (its training config is DROID-based). It plans pick-and-place from image goals on a Franka arm "without environment-specific data collection or task-specific training" [42]. The released comparison is below [42]:

| Method | Reach | Grasp Cup | Grasp Box | Pick-Place Cup | Pick-Place Box |
|---|---|---|---|---|---|
| Octo | 100% | 10% | 0% | 10% | 10% |
| Cosmos | 80% | 0% | 20% | 0% | 0% |
| V-JEPA 2-AC | 100% | 60% | 20% | 80% | 50% |

**Generative video world models** (Cosmos, Genie-style) predict pixels. They are expensive at inference. Cosmos-Predict2 offers 2B and 14B Video2World models and an action-conditioned variant post-trained on Bridge [36]. Their main closed-loop uses so far are *data generation* [35] and *evaluation* [38]. Pixel-space models struggle with precise contact physics, and data generated from them inherits the inverse-dynamics model's labeling errors [35].

---

## 11. Comparison of major designs

| System (org, year) | Loop(s) closed | Data source and scale | Model | Control rate / chunk | Notable lesson |
|---|---|---|---|---|---|
| RT-1 (Google, 2022) [1] | L3 single fleet | 130k episodes, 700+ tasks, 13 robots, 17 months | EfficientNet+FiLM, TokenLearner, Transformer | 3 Hz, discrete tokens | data diversity beats model tweaks |
| RT-2 (Google DeepMind, 2023) [2] | L3 + web co-training | robot data + web VQA | PaLI-X 55B / PaLM-E 12B VLA | text-token actions | web semantics transfer to control |
| OXE / RT-X (34 labs, 2023) [3] | L3 cross-lab | 1M+ trajectories, 22 embodiments, 60 datasets | RT-1-X, RT-2-X | 3 Hz | pooled data helps small-data labs (~+50%) |
| Octo (Berkeley et al., 2024) [5] | L3 open | 800k OXE episodes | 27M/93M diffusion transformer | chunked diffusion | modular fine-tuning to new sensors |
| OpenVLA / OFT (Stanford, 2024-25) [6][7] | L1 + L3 | 970K OXE trajectories | 7B VLA | 25 Hz bimanual with OFT | parallel decoding + chunks: 25-50x speedup |
| DROID (13 institutions, 2024) [20] | L3 supply | 76k trajectories, 350 h, 564 scenes | dataset + platform | n/a | standardized hardware; idle filtering needed |
| AutoRT (Google DeepMind, 2024) [17] | L3 autonomous | 77k+ episodes, up to 20 robots | VLM + LLM orchestrator, RT-1 policies | n/a | constitution raises valid tasks 18% to 83% (adversarial) |
| HIL-SERL (Berkeley, 2024) [29] | L2 RL | few demos + online interventions | off-policy RL, reward classifier | task-specific | near-perfect success within hours |
| π0 / π0.5 / π*0.6 (PI, 2024-25) [8][12][16] | L1, L2, L3 | 10k+ h robot data + web | PaliGemma VLM + flow action expert | chunks (~50 Hz) + RTC | Recap: advantage-conditioned RL on deployment data |
| Helix (Figure, 2025) [32] | L1 dual-rate | ~500 h teleop, auto-labeled | ~7B S2 + ~80M S1 | ~7-9 Hz / ~200 Hz | onboard dual-rate VLA |
| GR00T N1.7 (NVIDIA, 2026) [33][35] | L1, L3, L4 | robot demos + 20K h human video + dreamed data | 3B; Cosmos-Reason2-2B + flow DiT | 40-step horizon | shared relative end-effector space for human and robot data |
| V-JEPA 2-AC (Meta, 2025) [42] | L4 planning | internet video + small robot set | latent action-conditioned predictor | MPC planning | latent world models plan zero-shot |

---

## 12. Cross-cutting failure modes

1. **Compounding error and distribution shift.** This is the original reason for DAgger [25]. Chunking [18] and on-policy corrections [16][29] mitigate it but do not remove it.
2. **Data contamination from collection interfaces.** Idle frames [10] and action-frame bugs [28] are examples.
3. **Embodiment mismatch.** Generalist checkpoints "may or may not work" on new robots [9]. Lowest-common-denominator action spaces lose information [3].
4. **Evaluation noise and cost.** Real-world evaluation "is operationally expensive and inefficient" [52]. This motivates simulation proxies [52], world-model evaluators [38] and shared real-world arenas [55].
5. **Learned rewards and values can be exploited.** Reward classifiers [29] and value functions [16] are imperfect critics.
6. **Latency misalignment.** Unmatched observation-to-action latency causes jitter [21]. Naive chunk switching causes discontinuities, which RTC addresses [14].
7. **Safety of autonomous data collection.** Unconstrained LLM task proposals were valid only 18% of the time in adversarial scenes [17].

---

## Key takeaways for system designers

1. **Design the action interface first.** It decides what data you can pool (relative end-effector actions let human video count [33]), what you can simulate (joint position over velocity [10]), and how corrections are applied. Changing it later invalidates the corpus.
2. **Separate semantic latency from control latency.** Dual-rate stacks (Helix [32], GR00T [33]) and chunked flow/diffusion heads with real-time chunking [14] are the current answer. Measure end-to-end observation-to-actuation latency and match it during training [21].
3. **Treat collection hardware as part of the model.** Standardized rigs (DROID [20]) and robot-free collectors (UMI [21]) are what make data poolable and cheap. Budget for the curation that every rig will still need [10].
4. **Collect on-policy data deliberately.** DAgger's lesson [25] scales up as gated interventions (HG-DAgger [26], ThriftyDAgger [27]), intervention-augmented RL (HIL-SERL [29]) and advantage-conditioned retraining on deployment data (Recap [16]). Instrument the fleet to record interventions as first-class labels.
5. **Ration human attention with routing.** Let the system decide when a human is needed: risk and novelty gating [27], constitution-based task routing [17], or operator queues across parallel sessions [23].
6. **Make evaluation cheap before making training bigger.** Use correlated simulators [52], learned evaluators [38] and open-loop evaluation gates [33] to triage checkpoints before hardware time.
7. **Use simulation and world models where they are strong.** Physics simulation with randomization and actuator models works for locomotion and in-hand control [46][47][49]. Video world models are best used for data generation and evaluation [35][38], and latent world models for goal-conditioned planning [42]. Expect contact-rich precision to still need real data and real corrections [16][29].
8. **Pool across embodiments, but verify transfer per platform.** Cross-embodiment data reliably helps small-data settings [3][19]. Fine-tuned behavior still needs per-robot validation [9].

---

## References

1. RT-1: Robotics Transformer for Real-World Control at Scale. A. Brohan et al., Google (Robotics at Google / Everyday Robots), 2022. https://robotics-transformer1.github.io/ (paper: https://arxiv.org/abs/2212.06817)
2. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. A. Brohan et al., Google DeepMind, 2023. https://robotics-transformer2.github.io/ (paper: https://arxiv.org/abs/2307.15818)
3. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. Open X-Embodiment Collaboration, 2023. https://robotics-transformer-x.github.io/ (paper: https://arxiv.org/abs/2310.08864)
4. Open X-Embodiment repository (RLDS datasets, RT-1-X checkpoints). Google DeepMind, 2023. https://github.com/google-deepmind/open_x_embodiment
5. Octo: An Open-Source Generalist Robot Policy. Octo Model Team (UC Berkeley, Stanford, CMU, Google DeepMind), 2024. https://octo-models.github.io/ (report: https://arxiv.org/abs/2405.12213; code: https://github.com/octo-models/octo)
6. OpenVLA: An Open-Source Vision-Language-Action Model. M. J. Kim, K. Pertsch et al., Stanford et al., 2024. https://github.com/openvla/openvla (paper: https://arxiv.org/abs/2406.09246)
7. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT). M. J. Kim, C. Finn, P. Liang, Stanford, 2025. https://openvla-oft.github.io/ (paper: https://arxiv.org/abs/2502.19645)
8. π0: A Vision-Language-Action Flow Model for General Robot Control. K. Black et al., Physical Intelligence, 2024. https://www.physicalintelligence.company/blog/pi0 (paper: https://www.physicalintelligence.company/download/pi0.pdf)
9. openpi: open-source π0, π0-FAST and π0.5 models and code. Physical Intelligence, 2025. https://github.com/Physical-Intelligence/openpi
10. openpi: Training on DROID (full-DROID recipe, idle filtering, compute requirements). Physical Intelligence, 2025. https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md
11. FAST: Efficient Action Tokenization for Vision-Language-Action Models. K. Pertsch et al., Physical Intelligence, 2025. https://arxiv.org/abs/2501.09747 (also https://www.physicalintelligence.company/research/fast)
12. π0.5: a VLA with Open-World Generalization. Physical Intelligence, 2025. https://www.physicalintelligence.company/blog/pi05
13. VLAs that Train Fast, Run Fast, and Generalize Better (Knowledge Insulation). D. Driess et al., Physical Intelligence, 2025. https://www.physicalintelligence.company/research/knowledge_insulation
14. Real-Time Execution of Action Chunking Flow Policies. K. Black, M. Y. Galliker, S. Levine, Physical Intelligence, 2025. https://arxiv.org/abs/2506.07339
15. Training-Time Action Conditioning for Efficient Real-Time Chunking. K. Black, A. Z. Ren, M. Equi, S. Levine, Physical Intelligence, 2025. https://arxiv.org/abs/2512.05964 (code: https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
16. π*0.6: a VLA That Learns From Experience (Recap). Physical Intelligence, 2025. https://arxiv.org/abs/2511.14759
17. AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents. M. Ahn et al., Google DeepMind, 2024. https://auto-rt.github.io/ (paper: https://arxiv.org/abs/2401.12963)
18. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA / ACT). T. Z. Zhao, V. Kumar, S. Levine, C. Finn, Stanford et al., 2023. https://tonyzhaozh.github.io/aloha/ (paper: https://arxiv.org/abs/2304.13705; code: https://github.com/tonyzhaozh/act and https://github.com/tonyzhaozh/aloha)
19. Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation. Z. Fu, T. Z. Zhao, C. Finn, Stanford, 2024. https://mobile-aloha.github.io/ (paper: https://arxiv.org/abs/2401.02117)
20. DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset. A. Khazatsky, K. Pertsch et al., 2024. https://droid-dataset.github.io/ (paper: https://arxiv.org/abs/2403.12945; platform code: https://github.com/droid-dataset/droid)
21. Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. C. Chi, Z. Xu et al., Stanford / Columbia / TRI, 2024. https://umi-gripper.github.io/ (paper: https://arxiv.org/abs/2402.10329)
22. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. C. Chi et al., Columbia / TRI / MIT, 2023. https://github.com/real-stanford/diffusion_policy (paper: https://arxiv.org/abs/2303.04137)
23. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations. A. Mandlekar et al., NVIDIA / UT Austin, 2023. https://mimicgen.github.io/ (paper: https://arxiv.org/abs/2310.17596; code: https://github.com/NVlabs/mimicgen)
24. DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning. Z. Jiang et al., NVIDIA et al., 2024. https://github.com/NVlabs/DexMimicGen (paper: https://arxiv.org/abs/2410.24185)
25. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). S. Ross, G. Gordon, J. A. Bagnell, CMU, 2011. https://arxiv.org/abs/1011.0686
26. HG-DAgger: Interactive Imitation Learning with Human Experts. M. Kelly, C. Sidrane, K. Driggs-Campbell, M. J. Kochenderfer, Stanford, 2019. https://arxiv.org/abs/1810.02890
27. ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning. R. Hoque et al., UC Berkeley, 2021. https://arxiv.org/abs/2109.08273
28. SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning. J. Luo, Z. Hu et al., UC Berkeley et al., 2024. https://serl-robot.github.io/ (paper: https://arxiv.org/abs/2401.16013; code: https://github.com/rail-berkeley/serl)
29. Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning (HIL-SERL). J. Luo, C. Xu, J. Wu, S. Levine, UC Berkeley, 2024. https://hil-serl.github.io/ (paper: https://arxiv.org/abs/2410.21845; code: https://github.com/rail-berkeley/hil-serl)
30. LeRobot (library, LeRobotDataset format, HIL-SERL workflow guide). Hugging Face, 2024-2026. https://github.com/huggingface/lerobot (HIL-SERL guide: https://github.com/huggingface/lerobot/blob/main/docs/source/hilserl.mdx; paper: https://arxiv.org/abs/2602.22818)
31. GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation. Y. Li et al., ByteDance Seed, 2025. https://arxiv.org/abs/2512.01801
32. Helix: A Vision-Language-Action Model for Generalist Humanoid Control. Figure AI, 2025. https://www.figure.ai/news/helix
33. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots, and Isaac GR00T N1.7 repository. NVIDIA, 2025-2026. https://arxiv.org/abs/2503.14734 (code and N1.7 notes: https://github.com/NVIDIA/Isaac-GR00T)
34. GR00T-WholeBodyControl (Decoupled WBC, GEAR-SONIC, VR teleoperation). NVIDIA (NVlabs), 2025-2026. https://github.com/NVlabs/GR00T-WholeBodyControl (SONIC paper: https://arxiv.org/abs/2511.07820)
35. GR00T-Dreams / DreamGen: Unlocking Generalization in Robot Learning through Video World Models. NVIDIA, 2025. https://github.com/NVIDIA/GR00T-Dreams (paper: https://arxiv.org/abs/2505.12705)
36. NVIDIA Cosmos World Foundation Models (Cosmos-Predict2, Cosmos-Predict2.5). NVIDIA, 2025-2026. https://github.com/NVIDIA/Cosmos (also https://github.com/nvidia-cosmos/cosmos-predict2 and https://github.com/nvidia-cosmos/cosmos-predict2.5)
37. Tesla AI and Robotics (Optimus). Tesla, accessed 2026. https://www.tesla.com/AI
38. 1X World Model Challenge (dataset, Genie-style baseline, evaluation challenge). 1X Technologies, 2024. https://github.com/1x-technologies/1xgpt
39. Genie: Generative Interactive Environments. J. Bruce et al., Google DeepMind, 2024. https://arxiv.org/abs/2402.15391
40. Mastering Diverse Domains through World Models (DreamerV3). D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, 2023 (Nature 2025). https://arxiv.org/abs/2301.04104 (code: https://github.com/danijar/dreamerv3)
41. DayDreamer: World Models for Physical Robot Learning. P. Wu, A. Escontrela, D. Hafner, K. Goldberg, P. Abbeel, UC Berkeley, 2022. https://arxiv.org/abs/2206.14176 (code: https://github.com/danijar/daydreamer)
42. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. M. Assran et al., Meta FAIR, 2025. https://arxiv.org/abs/2506.09985 (code: https://github.com/facebookresearch/vjepa2)
43. UniSim: Learning Interactive Real-World Simulators. M. Yang et al., Google DeepMind / UC Berkeley / MIT, 2023. https://arxiv.org/abs/2310.06114
44. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. J. Tobin et al., OpenAI / UC Berkeley, 2017. https://arxiv.org/abs/1703.06907
45. Learning Dexterous In-Hand Manipulation. OpenAI et al., 2018. https://arxiv.org/abs/1808.00177
46. Solving Rubik's Cube with a Robot Hand (Automatic Domain Randomization). OpenAI et al., 2019. https://arxiv.org/abs/1910.07113
47. Learning Agile and Dynamic Motor Skills for Legged Robots (actuator network). J. Hwangbo et al., ETH Zurich, 2019. https://arxiv.org/abs/1901.08652
48. RMA: Rapid Motor Adaptation for Legged Robots. A. Kumar, Z. Fu, D. Pathak, J. Malik, UC Berkeley / CMU / Meta, 2021. https://arxiv.org/abs/2107.04034
49. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning (legged_gym). N. Rudin et al., ETH Zurich / NVIDIA, 2021. https://arxiv.org/abs/2109.11978 (code: https://github.com/leggedrobotics/legged_gym)
50. Isaac Lab: GPU-accelerated robot learning framework. NVIDIA et al., 2025. https://github.com/isaac-sim/IsaacLab (paper: https://arxiv.org/abs/2511.04831)
51. MuJoCo Playground. Google DeepMind et al., 2025. https://github.com/google-deepmind/mujoco_playground
52. Evaluating Real-World Robot Manipulation Policies in Simulation (SIMPLER). X. Li et al., UC San Diego / Stanford / UC Berkeley / Google DeepMind, 2024. https://simpler-env.github.io/ (paper: https://arxiv.org/abs/2405.05941; code: https://github.com/simpler-env/SimplerEnv)
53. Safari SDK for Gemini Robotics models. Google DeepMind, 2025. https://github.com/google-deepmind/gemini-robotics-sdk (models: https://deepmind.google/models/gemini-robotics/)
54. GR-3 Technical Report. C. Cheang et al., ByteDance Seed, 2025. https://arxiv.org/abs/2507.15493
55. RoboArena: distributed real-world evaluation of generalist robot policies. RoboArena team, 2025. https://robo-arena.github.io/
