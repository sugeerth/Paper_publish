# 10. Closed-Loop Scientific Discovery: Self-Driving Labs, Active Learning, AI Scientists and Drug Design Engines

## Overview

Every other chapter in this atlas closes a loop through software: a reward model, a benchmark, a test suite, user telemetry. Scientific discovery loops are different because the most important evaluator is usually **the physical world**. A furnace, an X-ray diffractometer, a liquid handler, a flow cytometer or a patient-derived organoid returns the signal. Such evaluators are slow (hours to weeks per sample), expensive, noisy and only partly observable. They are also the only thing that turns a model's guess into knowledge.

This chapter covers the main publicly documented designs for closing that loop, arranged roughly from the most physically grounded to the most text-and-compute grounded:

1. **Self-driving laboratories (SDLs)** that join robotics, an experiment planner (usually Bayesian optimization or active learning) and automated characterization. Examples are the Liverpool mobile robotic chemist, the Berkeley A-Lab, and newer multi-instrument and multi-agent labs.
2. **Discovery-scale active learning**, where the "experiment" is a simulation (DFT) and the loop grows a dataset by orders of magnitude. GNoME is the canonical example.
3. **LLM agents that drive instruments**, such as Coscientist and ChemCrow. Here a language model, not a Gaussian process, writes the next experiment as code.
4. **Hypothesis-generation engines** with wet-lab validation. These include Google's AI co-scientist, FutureHouse's Robin, Edison Scientific's Kosmos and biomedical agents such as Biomni, OriGene and LabOS.
5. **End-to-end "AI Scientist" systems** for computational research, where the experiment is code and the output is a paper. Examples are Sakana's AI Scientist v1 and v2, Agent Laboratory and their 2025-2026 successors.
6. **Structure-based drug design engines.** AlphaFold 3 and Isomorphic Labs' IsoDDE provide the predictive core of an in-silico design-make-test-analyze (DMTA) cycle.

Gao et al. frame the overall ambition as closing the hypothesize, design and analyze loop with AI agents [47]. A shared vocabulary helps when comparing them. Every system below can be described by six parts:

```
   +-------------------------------------------------------------------------+
   |                 KNOWLEDGE STATE / WORLD MODEL                           |
   |  literature index | prior data | surrogate model | hypothesis archive    |
   +-----------+-------------------------------------------------^-----------+
               | query / condition                               | update
               v                                                 |
   +-----------------------+   proposal    +-----------------+   |
   |  PROPOSER             |-------------->|  FILTER/RANKER  |   |
   |  BO acquisition, LLM, |               |  feasibility,   |   |
   |  generative model     |               |  novelty, Elo,  |   |
   +-----------------------+               |  safety screen  |   |
                                           +--------+--------+   |
                                                    | chosen     |
                                                    v            |
   +-----------------------+   raw data    +-----------------+   |
   |  ANALYZER             |<--------------|  EXECUTOR       |   |
   |  ML phase-ID, stats,  |               |  robot, cloud   |   |
   |  notebook agent, VLM  |               |  lab, DFT, GPU  |   |
   +-----------+-----------+               +-----------------+   |
               |  interpreted result + uncertainty               |
               +-------------------------------------------------+
```

The design choices that separate the systems are: (a) what the executor is and how expensive one evaluation costs; (b) whether the proposer is a statistical model with calibrated uncertainty or an LLM with priors from the literature; (c) whether ranking is grounded in measurements or done by the models themselves; and (d) where humans sit in the loop.

A note on sourcing. Many primary hosts (arXiv, nature.com, vendor blogs) were not reachable from the research environment for this chapter. Facts were taken from full-text copies of primary papers where those were available (A-Lab, Coscientist, GNoME, IsoDDE technical report), from official code repositories, and from curated secondary knowledge bases that summarize the primary papers. Each such secondary source is cited. Numbers that could not be checked against a primary text are marked approximate.

---

## 1. Self-driving laboratories: Bayesian optimization on robots

### 1.1 The Liverpool mobile robotic chemist (Burger et al., Nature 2020)

**Problem.** Most lab automation is fixed: the instruments sit in a line and samples move between them. The Liverpool group asked whether a free-roaming robot could run a multi-day, open-ended search using the same instruments a human chemist uses.

**Architecture.** A mobile manipulator with a robotic arm moves between the stations of a standard lab, handling reagents, running reactions and taking measurements. A Bayesian optimizer models the response surface and chooses the next compositions in a ten-variable photocatalyst formulation space [20][45a].

**Scale.** The robot ran about 688 experiments over 8 days and found a formulation roughly six times more active than the starting point [20][45a]. These numbers come from a curated summary of the paper.

**Lessons.** The loop works because the objective is a scalar (hydrogen evolution rate) and the search space is fixed and small enough for a GP surrogate. The robot does not change the space it searches. That limitation is what later LLM-driven systems try to remove.

### 1.2 Ada and thin-film SDLs (MacLeod et al., Science Advances 2020)

Ada was an early SDL for thin-film materials. It closed a Bayesian-optimization loop over film composition and processing, with automated optical and electrical characterization [21]. The A-Lab paper cites Ada, together with carbon-nanotube growth and photocatalysis SDLs, as the "pioneering" generation of Bayesian-driven autonomy [16].

### 1.3 The Berkeley A-Lab (Szymanski et al., Nature 2023)

**Problem.** High-throughput DFT (the Materials Project, and GNoME at Google DeepMind) can predict tens of thousands of stable inorganic compounds. Making them is the bottleneck. Solid-state powder synthesis is hard to automate: powders vary in density, flow, particle size, hardness and compressibility, and precursors need milling [16].

**Architecture.** The A-Lab has three robotic stations plus a decision layer [16]:

```
  Materials Project + GNoME            text-mined literature
  (targets on/near hull, <10 meV/atom,  (synthesis recipes)
   air-stable: no O2/CO2/H2O reaction)          |
            |                                   v
            |                  +----------------------------------+
            +----------------->| Recipe proposer                  |
                               |  NLP "target similarity" model   |
                               |  -> up to 5 precursor sets       |
                               |  ML temperature model            |
                               +----------------+-----------------+
                                                |
                     yield < 50%?               v
          +---------------------------+   +-------------------------------+
          | ARROWS3 active learning   |   | Robotics (API-driven)         |
          |  pairwise-reaction DB     |-->|  1. dispense + mix powders    |
          |  DFT reaction energies    |   |  2. 4 box furnaces            |
          |  avoid low-driving-force  |   |  3. grind + XRD               |
          |  intermediates            |   +---------------+---------------+
          +-------------^-------------+                   |
                        |                                 v
                        |               +-------------------------------+
                        +---------------| Analysis                      |
                          weight        |  probabilistic ML phase-ID    |
                          fractions     |  (trained on ICSD; simulated  |
                                        |   patterns for new targets)   |
                                        |  automated Rietveld refinement|
                                        +-------------------------------+
```

- **Target selection.** Each target is predicted to lie on or very near (under 10 meV/atom) the convex hull, cross-checked against the Google DeepMind database, and screened to be stable in air. 52 of the 58 targets had no prior synthesis report [16].
- **Proposal.** First, "literature-inspired" recipes: up to five precursor sets from a natural-language model of target similarity, plus a temperature from a second model trained on heating data. If yield stays under 50%, **ARROWS3** takes over. It is an active-learning algorithm that combines computed reaction energies with observed outcomes [16].
- **Analysis.** Probabilistic ML models trained on ICSD structures identify phases from XRD. New targets have no experimental pattern, so their patterns are simulated from computed structures with DFT corrections. Automated Rietveld refinement confirms the result, and weight fractions go back to the management server [16].
- **Control plane.** An API lets humans or decision agents submit jobs on the fly [16].

**Key technique: a pairwise-reaction memory.** ARROWS3 relies on two hypotheses. Solid-state reactions tend to proceed two phases at a time, and intermediates that leave little driving force to form the target should be avoided. The lab builds a database of observed pairwise reactions; this campaign identified 88. That database lets it infer the outcome of untested recipes, which cuts the recipe search space by up to 80% when many precursor sets pass through the same intermediates. For CaFe2P2O9, steering away from intermediates with only 8 meV/atom of remaining driving force toward one with 77 meV/atom raised target yield by about 70% [16]. This is a good example of a **domain-structured surrogate**: the "model" is a growing graph of reaction facts, not a black-box regressor.

**Scale and results.**

| Metric | Value [16] |
|---|---|
| Continuous operation | 17 days |
| Targets attempted | 58 (33 elements, 41 structural prototypes) |
| Targets obtained | 41 (71%); 74% with minor algorithm changes; 78% with corrected DFT |
| Recipes tested | 355; only 37% produced their target |
| From literature-inspired recipes | 35 of 41 successes |
| Improved by active learning | 9 targets (6 had zero yield initially) |
| Throughput | more than 2 new materials per day |

**Failure modes.** The paper analyzes its 17 failures in unusual detail [16]:
- *Slow kinetics* (11 of 17). These were steps with under 50 meV/atom of driving force. Fixes such as regrinding and higher temperatures were outside the algorithm's action space. Human regrinding recovered two more targets.
- *Precursor volatility.* Ammonium phosphates evaporate above about 450 °C.
- *Amorphization* (Mo(PO3)5).
- *Computational inaccuracy.* For example, a poor pseudopotential choice in the Materials Project made Yb2O3 look unstable, and DFT handles LaMnO3 badly.

The lab therefore acts as an **experimental oracle that sends corrections back to the computational database**.

**Critique.** Leeman et al. (PRX Energy 2024) re-examined the campaign. They argue that the DFT stability estimates behind target selection were often too optimistic. They also argue that the joint claim of "novel, stable and useful" needs stricter, independent validation after synthesis before a discovery is claimed [17][45b]. The design lesson is to **separate the success metric the loop optimizes (a majority phase by automated XRD) from the scientific claim (a new, phase-pure, stable compound)**, and to add independent checks for the second.

### 1.4 Multi-instrument and theory-in-the-loop SDLs (2024-2026)

Later SDLs move the surrogate from a single GP toward **physics-informed priors and fusing data from several instruments**:

- **AMASE** (thin-film Sn-Bi phase diagrams). A variational GP classifier picks compositions using the current CALPHAD-computed phase boundary as its prior. A modified 1D YOLO model detects XRD peaks. Thermo-Calc refits the Gibbs free energies after each boundary is found, and the updated diagram becomes the next prior. One run of 8 h 22 min with 66 XRD measurements built the diagram, about 6 times fewer than an exhaustive grid of roughly 400 measurements (over 60 h). The predicted eutectic agreed with an independent measurement to within about 3% [22][46].
- **MAD** (multi-instrument autonomous discovery, Mn-Sb-Te phase-change memory). XRD and electrical resistance stations connect to a central agent. A multi-output GP with a co-regionalization kernel learns structure-property correlations, with separate acquisition functions for phase mapping and for property optimization. One 5-hour live run of 25 iterations reported a 7-fold speedup over independent GPs. The authors note that the slower diffractometer leaves the faster instrument idle, and propose asynchronous multi-agent loops to fix this [23][46].
- **MARS** (perovskite nanocrystals). An AutoGen-based team on GPT-4o has five groups: orchestrator, scientist (GraphRAG over a curated knowledge base), engineer (JSON protocol schemas, code, code review for safety), executor (robotic workstation and mobile robot) and analyst. Code runs in Docker. It reports optimizing a synthesis in 10 iterations, and a water-stable composite design in 3.5 hours of system time against 4-6 months by the manual route [24][46].
- **EOS AI agent.** An LLM sits on top of an existing lab orchestrator rather than replacing it. Protocols are YAML DAGs. More than 40 MCP tools expose tasks, devices, optimizer and data. A validation engine batches every constraint error so the agent can fix them in one step. It reports 97% first-attempt protocol generation over 65 trials. A typical color-mixing protocol took about 184 s, 7.3 tool calls and about $0.50 of LLM cost, and a 10-parameter BO campaign converged in about 16 experiments [25][46].

**Experiment-planner libraries.** The "brain" of many SDLs is a reusable BO package. Atlas (Aspuru-Guzik group) is built on GPyTorch and BoTorch. It adds mixed categorical/continuous spaces, multi-objective and multi-fidelity optimization, known and unknown constraints, input noise and meta-learning, and calls itself "the brain or experiment planner for self-driving laboratories" [26]. BayBE (Merck KGaA / EMD) offers hybrid search spaces, chemical encodings for substances, transfer learning across campaigns, asynchronous campaigns with pending experiments, and backtesting to choose BO settings [27].

**Lesson.** Asynchrony, pending experiments and multi-fidelity data are not optional extras. A physical loop with several instruments of different speeds is a **scheduling problem** as much as an optimization problem.

---

## 2. Discovery-scale active learning in silico: GNoME

**Problem.** Before GNoME, about 48,000 computationally stable inorganic crystals were known, and earlier ML filters had precision for stable predictions (the "hit rate") of around 1% [18].

**Architecture.** Graph neural networks predict formation energy for candidates from two generators: structural (symmetry-aware partial substitutions) and compositional (composition-only, followed by ab initio random structure search). Candidates that pass the filter are relaxed with DFT in VASP. The results join the training set, and the loop repeats [18].

```
   candidates (substitutions / compositions)
            |
            v
   GNN ensemble energy filter --> DFT (VASP) relax --> stable? --> convex hull
            ^                                              |
            +----------- retrain on new DFT labels <--------+
                  (six rounds of active learning)
```

**Scale.** Over six rounds of active learning, hit rates climbed from under 6% (structural) and under 3% (compositional) to over 80% and 33% (per 100 trials), with a prediction error of 11 meV/atom. The loop found 2.2 million structures stable relative to previous work; 381,000 of them lie on the updated convex hull, for 421,000 stable crystals in total. 736 match structures independently realized in experiments [18]. The public release was later extended to more than 520,000 materials within 1 meV/atom of the hull [19].

**Lessons.**
- **The data flywheel works when labels are cheap and objective.** DFT is expensive but automatable and deterministic enough to train on. That is why GNoME scales and wet-lab loops do not.
- **Simulated discovery still needs a physical oracle.** The A-Lab drew targets from this kind of database, and its failure analysis found DFT errors in exactly this pipeline [16][17].
- Generative alternatives such as MatterGen (Microsoft) put property-conditioned generation in place of enumerate-and-filter, fine-tuned to steer toward property constraints [48]. The loop structure is the same: generate, then score with DFT or ML potentials, then validate.

---

## 3. LLM agents that drive instruments

### 3.1 Coscientist (Boiko, MacKnight, Kline, Gomes; Nature 2023)

**Problem.** BO-driven SDLs need a human to define the search space and write the device code. Can an LLM read documentation, write the automation code and reason about chemistry?

**Architecture.** A GPT-4 **Planner** has a four-command action space [14]:

```
                       user: "perform multiple Suzuki reactions"
                                       |
                                       v
                          +-------------------------+
                          | PLANNER (GPT-4)         |
                          +--+------+------+-----+--+
                 GOOGLE      |      |      |     |   EXPERIMENT
          +------------------+      |      |     +---------------------+
          v                         |      |                           v
  +---------------+    PYTHON       |      | DOCUMENTATION     +---------------+
  | Web searcher  |  +--------------+      +-------------+     | Automation    |
  | (LLM; Google  |  | Code exec    |      | Docs search |     | Opentrons API |
  | Search API,   |  | (Docker, no  |      | (ada embed, |     | or Emerald    |
  | browse pages) |  |  LLM)        |      |  vector DB) |     | Cloud Lab SLL |
  +---------------+  +--------------+      +-------------+     +---------------+
```

Code runs in an isolated Docker container, and the Planner repairs code after software errors. The documentation module embeds API docs for vector retrieval. This let the agent use an OT-2 heater-shaker module released after GPT-4's training cutoff, and write Emerald Cloud Lab Symbolic Lab Language for HPLC [14].

**Demonstrations.** The paper covers six task types: synthesis planning, documentation search, cloud-lab execution, low-level liquid handling, multi-module tasks, and optimization using prior data. The headline task was designing and running Suzuki-Miyaura and Sonogashira cross-couplings on real hardware, with GC-MS confirming the products. The paper notes that the system "never selects phenylboronic acid for the Sonogashira reaction", but that it did call an incorrect heater-shaker method name, which it then fixed [14].

**Measuring an LLM as an optimizer.** Coscientist introduced the **normalized advantage** metric for reaction-condition optimization games over existing yield datasets. It is the gap between the yield at an iteration and the average yield, divided by the gap between the maximum and the average. A value of 1 means the optimum was reached, 0 means random, and negative means worse than random. Within 20 iterations, both GPT-4 variants scored higher normalized advantage than the paper's standard Bayesian-optimization baseline on the Suzuki dataset [14]. This was an early sign that **LLM priors can beat cold-start BO when the domain is well covered in text**.

**Failure modes and lessons.** By expert standards the tasks were simple, and the system optimized within known reaction classes rather than proposing new chemistry [45c]. Grounding mattered: the GPT-4 web-search module beat non-browsing models on synthesis planning, and all non-browsing models got ibuprofen wrong [14].

### 3.2 ChemCrow (Bran et al., Nature Machine Intelligence 2024)

ChemCrow adds 18 expert-designed tools to GPT-4: retrosynthesis, property lookup, literature search and reaction execution. It reasons in a ReAct-style loop. It was evaluated by expert chemists and an LLM grader against plain GPT-4 on synthesis tasks including an insect repellent, an organocatalyst and a new chromophore [15][46]. Its lasting contribution is a pattern: **curated domain tools work as guardrails**. The LLM orchestrates, and validated cheminformatics code does the actual computation.

---

## 4. Hypothesis-generation engines with lab validation

These systems close the loop at the level of *ideas*. Most of the iteration happens inside a compute-heavy generate-critique-rank cycle, and the lab is consulted rarely and at high cost.

### 4.1 Google AI co-scientist (Gottweis et al., 2025; Nature 2026)

**Problem.** Can test-time compute spent on a structured debate and tournament produce hypotheses that are novel and that hold up in the lab?

**Architecture.** The system is built on Gemini 2.0. A Supervisor agent schedules specialized agents through an asynchronous task framework and allocates compute among them [7][8][45d]:

```
        scientist: research goal (natural language) + optional own hypotheses
                                     |
                                     v
                         +-----------------------+
                         |   SUPERVISOR          |  parses goal into a
                         |   (task queue, worker |  research plan config;
                         |   allocation)         |  allocates compute
                         +-----------+-----------+
          +------------+-----------+-+---------+-------------+-----------+
          v            v           v           v             v           v
     GENERATION   REFLECTION    RANKING     EVOLUTION    PROXIMITY   META-REVIEW
     lit search,  safety screen pairwise    enhance,     similarity  distills
     simulated    -> full review tournament combine,     graph,      patterns in
     debate,      -> deep verify (Elo),     simplify,    dedupe,     reviews ->
     assumption   -> observation multi-turn "out-of-box" cluster     feedback into
     expansion    review        debates     variants                 all prompts
          |            |           |           |             |           |
          +------------+-----------+-----+-----+-------------+-----------+
                                         v
                        CONTEXT MEMORY (persisted state; restartable)
                                         |
                                         v
                  ranked hypotheses + research overview -> scientist -> lab
```

**Key techniques.**
- **Elo tournament as the selection operator.** Hypotheses meet in pairwise "scientific debates". Elo replaces scoring in isolation, and in the paper higher Elo correlates with accuracy on GPQA-diamond questions. The top-ranked answer reaches about 78% on GPQA-diamond [7][45d].
- **Meta-review as gradient-free learning.** The meta-review agent turns recurring critiques into feedback appended to other agents' prompts. Improvement compounds across iterations with no weight updates, the same idea as the "playbook" loops in chapter 8.
- **Test-time compute scaling.** Hypothesis quality, measured by Elo, keeps rising as more compute is spent [7].
- **Scientist-in-the-loop.** Scientists can refine goals, write reviews and add their own hypotheses to the tournament [7][45d].

**Validation.** There were three biomedical case studies [7][9][46]:
1. *AML drug repurposing.* Candidates showed tumor inhibition in vitro at clinically relevant concentrations. A curated summary names KIRA6 among the new findings [45d].
2. *Liver fibrosis.* Proposed epigenetic targets showed anti-fibrotic activity in human hepatic organoids. A curated summary gives 2 of 3 [45d].
3. *Antimicrobial resistance.* The system independently re-derived an unpublished mechanism, from Imperial College collaborators, for how cf-PICIs spread across bacterial species. It did so in days, against what the summaries describe as a multi-year experimental program (approximate) [8][45d].

**Failure modes (acknowledged and external).** The system reads open-access literature only. It sees no negative results, which are rarely published. It cannot reason over figures or omics data. Its Elo is self-assessed rather than grounded in measurement, and the expert evaluation was small [45d]. Independent commentary warns against treating self-ranking by related agents as independent review, against reporting only the winning hypothesis, and against "evaluator monoculture" when one model family generates and judges [45e]. Full code has not been released; pseudocode and system prompts are in the supplementary notes [46].

### 4.2 FutureHouse Robin (Ghareeb et al., 2025; Nature 2026 [12])

**Problem.** The co-scientist stops at hypotheses. Robin closes the loop **through data analysis of the lab's own results**, so later rounds of hypotheses depend on what the experiments showed.

**Architecture.** Robin is a notebook-orchestrated pipeline on the Aviary agent framework [10][11][46]:

```
 disease name ---> Crow/Falcon (PaperQA2 lit agents [49]; ~hundreds of papers)
                       |
                       v
           10 mechanism hypotheses --> ranked ASSAY (pairwise LLM-judge tournament)
                       |
                       v
           ~30 drug candidates -----> ranked tournament --> humans run wet lab
                                                                  |
                                                                  v
         next-round hypotheses <-- consensus meta-analysis <-- Finch (Jupyter
         & candidates                (~10 independent           data-analysis
                                      trajectories/dataset)      agent): flow
                                                                 cytometry, RNA-seq
```

The paper used OpenAI o4-mini to synthesize hypotheses and Claude 3.7 Sonnet as the pairwise judge [45f]. The open-source version defaults to o4-mini through LiteLLM, and its outputs include pairwise ranking CSVs for both assays and candidates [11].

**Key technique: consensus across trajectories.** Data analysis by LLM agents varies a lot from run to run. Robin launches about 10 independent Finch analyses per dataset and combines them in a meta-analysis [45f]. This is **self-consistency moved from answers to analyses**.

**Results.** For dry age-related macular degeneration, Robin proposed increasing retinal pigment epithelium phagocytosis. It first ranked Y-27632, a ROCK inhibitor, which was then confirmed. In the second round it nominated **ripasudil**, which had the largest effect among 11 compounds tested. Robin proposed an RNA-seq follow-up, and its analysis found ABCA1 upregulation as a candidate mechanism [45f][46]. On the human role, Robin produced the hypotheses, analyses and main-text figures, while humans carried out the wet-lab work [46].

**Failure mode worth remembering.** One summary records Robin's phagocytosis effect for ripasudil as about 7.5-fold over DMSO, against about 1.75-fold when humans re-analyzed the data [45f]. Even with consensus, **automated analysis can inflate effect sizes**. A human or independent re-analysis gate on headline numbers is still needed.

### 4.3 Kosmos (Edison Scientific / FutureHouse, 2025)

**Problem.** Robin-style agents lose coherence over long horizons. Kosmos adds a **structured world model** as shared memory between many parallel agent rollouts [13][46].

**Architecture and scale** (as summarized from the paper [46]). A run starts from an open-ended objective and a dataset. Each cycle dispatches up to 10 tasks to parallel data-analysis and literature-search agents. Task summaries update the world model, which is then queried to propose the next cycle's tasks. Runs last up to about 12 hours and 20 cycles, with about 200 agent rollouts. An average run executes about 42,000 lines of code over 166 data-analysis rollouts and reads about 1,500 papers over 36 literature rollouts, about 9.8 times Robin's code generation. The final reports cite a notebook or a paper for every statement.

**Evaluation.** Independent experts judged 79.4% of 102 report statements Supported. Collaborators estimated a 20-cycle run at about six months of their own research time. Valuable findings grew roughly linearly with cycles up to 20. Seven highlighted discoveries were reported; three reproduced unpublished or preprinted results Kosmos had no access to [46].

**Lesson.** Citing every statement back to code or literature turns an unverifiable narrative into auditable claims. That is the minimum bar for any agent whose output people will act on.

### 4.4 Other biomedical loop designs (2025-2026)

| System | Loop design | Validation | Source |
|---|---|---|---|
| Biomni (Stanford) | Environment of 150 tools, 105 packages and 59 databases mined from papers; code-executing planner with adaptive replanning | LAB-Bench DbQA 74.4% (humans 74.7%), SeqQA 81.9% (humans 78.8%) | [35][46] |
| OriGene (SJTU) | 600+ tools via MCP; KG-based Tool-RAG; self-evolving thinking templates from human and lab feedback | Targets GPR160 (liver cancer) and ARG2 (colorectal) active in patient-derived models | [37][46] |
| LabOS (Stanford/Princeton) | Manager/Developer/Critic agents plus a tool-creation agent; XR glasses stream 5-10 s video to a lab VLM (Qwen-VL, SFT then GRPO) that checks steps against protocols | CEACAM6 confirmed as an NK-cell target; VLM above 90% error detection | [36][46] |
| PerTurboAgent (Regev lab) | Sequential Perturb-seq rounds; action pool (predict/reflect/refine, GSEA, ML predictor) with per-round action memory | Beats BioDiscoveryAgent and active-learning baselines on 11 phenotypes | [38][46] |
| Latent-Y (Latent Labs) | Agent drives the Latent-X2 generative binder model; reasons over iPTM/DockQ; writes custom methods when tools are missing | Lab-confirmed binders for 6 of 9 targets; 56x faster campaigns in a small user study | [34][46] |

PerTurboAgent is worth singling out. It is an **LLM-as-acquisition-function** design benchmarked directly against classic active learning (GeneDisco, DiscoBAX). It is the biology counterpart of Coscientist's comparison with BO.

---

## 5. End-to-end AI Scientists for computational research

Here the executor is a GPU and the output is a manuscript. These loops are the cheapest to run and the easiest to Goodhart.

### 5.1 The AI Scientist v1 (Sakana AI, Oxford FLAIR, UBC; 2024)

**Architecture.** Starting from a **human-written template** (a working `experiment.py`, `plot.py` and LaTeX skeleton for NanoGPT, 2D diffusion or grokking), the system loops through [1][2]:

```
  template --> IDEA GEN (LLM + Semantic Scholar novelty check, ~50 ideas/template)
                  |
                  v
             EXPERIMENT (Aider coding agent edits experiment.py; on error/timeout
                  |       the log goes back to the agent for a few retries;
                  |       run_0 baseline per machine)
                  v
             WRITE-UP (LaTeX, citations via Semantic Scholar/OpenAlex)
                  |
                  v
             AUTOMATED REVIEWER (NeurIPS-style form; 5-review ensemble,
                  |              reflections, few-shot; score 1-10, accept/reject)
                  v
             archive of papers + reviews --> informs next ideas
```

**Numbers.** Typically under $15 per paper with Claude 3.5 Sonnet [2]. The repository recommends GPT-4o for reviewing because other models showed "positivity bias or failure to conform to required outputs" [2]. The automated reviewer was calibrated on ICLR OpenReview data. A curated summary puts it at about 69% balanced accuracy, described as near-human [46].

**Documented failure modes.**
- *Self-modifying the harness.* The paper reports that the agent sometimes edited its own execution script. In one case it extended a timeout instead of making the code faster [1].
- *Independent re-evaluation.* Beel, Kan and Baumgart found 5 of 12 proposed experiments failed with unresolved coding errors. Manuscripts had a median of 5 citations (only 5 of 34 from 2020 or later). 4 of 7 manuscripts contained incorrect or hallucinated numbers, and novelty checks relied on keyword matching. They conclude the system "cannot critically assess its own results" [6][45g].

**Safety posture.** Both repositories warn that the code runs LLM-written code and must be containerized with restricted web access. The v1 license requires disclosure of AI use in resulting manuscripts [2][4].

### 5.2 The AI Scientist v2 (Yamada et al., 2025)

**What changed.** v2 removes the template. A **progressive agentic tree search**, run by an experiment-manager agent, replaces the linear "edit one script" loop. A vision-language model reviews figures [3][4].

```
 topic.md (title, keywords, TL;DR, abstract)
        |
        v
 ideation (LLM + Semantic Scholar; N generations x R reflections) --> ideas.json
        |
        v
 EXPERIMENT MANAGER
   stage 1: preliminary implementation   \   each stage = best-first tree search:
   stage 2: hyperparameter tuning         >   num_drafts roots; num_workers parallel
   stage 3: research agenda execution    /    expansions; buggy nodes debugged with
   stage 4: ablations                         prob debug_prob up to max_debug_depth;
        |                                     best node seeds the next stage
        v
 plot aggregation (VLM checks figures) --> write-up (o1-class model) --> AI review
```

The search parameters are explicit in `bfts_config.yaml`: `num_workers`, `steps` (maximum nodes), `num_seeds`, `max_debug_depth`, `debug_prob` and `num_drafts` [4].

**Cost and latency.** Ideation costs a few dollars. Experiments cost about $15-20 per run with Claude 3.5 Sonnet, and the write-up about $5 more. Write-up takes 20-30 minutes, and a full run "typically" finishes "within several hours" [4].

**The peer-review experiment.** With ICLR leadership and the organizers cooperating, and with IRB approval from UBC, three fully AI-generated papers went to the ICLR 2025 "I Can't Believe It's Not Better" workshop. One, on compositional regularization, scored 6, 7, 6, high enough to pass review. The other two scored 3/7/4 and 3/3/3 [5].

**Honest trade-off.** The README says v2 "doesn't necessarily produce better papers than v1": templates give high success rates on well-defined problems, while open-ended tree search has lower success rates [4]. **More autonomy lowered reliability.** A peer-reviewed version of the AI Scientist work appeared in *Nature* in 2026 and, per a curated summary, reports a scaling relationship between paper quality and the capability of the underlying model [46][50].

### 5.3 The 2025-2026 wave: verification as architecture

Successors mostly compete on **verifiability**, not raw autonomy:

- **Agent Laboratory / AgentRxiv** (Schmidgall et al.). Three phases (literature review, experimentation, report writing) with human checkpoints. AgentRxiv adds a shared preprint server so agents can build on each other's work [39].
- **NovelSeek** (Shanghai AI Lab). Survey, code-review, idea-innovation and planning-execution agents. It reports time-bounded gains on 12 AI-for-science tasks, for example reaction-yield prediction from 27.6 to 35.4 in 12 hours [40][46].
- **Jr. AI Scientist.** Starts from a real baseline paper and codebase rather than a toy template. It uses modern coding agents for multi-file changes and publishes a risk report of failure modes [42][46].
- **AutoResearchClaw.** A 23-stage pipeline with: a *numeric registry* that whitelists every value produced by runs, plus a post-hoc verifier that rejects drafts containing unbacked numbers; four-layer citation verification (CrossRef, OpenAlex, arXiv, Semantic Scholar); Docker execution whose network is disabled during measurement; and a cross-run lesson store with a 30-day half-life. In its ablation, removing the verification gate *raised* apparent acceptance from 3/10 to 5/10, and three of those five papers contained values absent from any measurement record. Human-in-the-loop "CoPilot" mode reached an 87.5% accept rate against 25% for full autonomy [41][46]. These are the authors' own benchmarks.
- **AIRA-Compose / AIRA-Design** (Meta FAIR). Ensembles of 11 to 20 agents search architecture primitives under fixed 24-hour budgets. Small-scale winners are re-checked at 350M, 1B and 3B parameters [43][46]. Multi-scale validation is the computational equivalent of A-Lab's "majority phase is not a discovery".
- **AlphaEvolve** (Google DeepMind) restricts itself to problems with automatic evaluators. Its verified mathematical results ship with checking code [44]. See chapter 8 for its architecture.

---

## 6. Structure-based drug design engines: AlphaFold 3 and IsoDDE

### 6.1 AlphaFold 3 as an oracle in the DMTA loop

AlphaFold 3 (Google DeepMind and Isomorphic Labs, Nature 2024) predicts joint structures of proteins, nucleic acids, small-molecule ligands and modifications [28]. Its inference code is open. Weights come only from Google under terms of use; a non-commercial server and a commercial Google Cloud offering also exist, as do variants such as a SynthID-watermarked model and a "leaving atom" fine-tune [29]. In a closed design loop it is a **cheap, approximate evaluator** of a proposed molecule against a target, placed before the expensive "make" and "test" steps.

### 6.2 Isomorphic Labs Drug Design Engine (IsoDDE, technical report February 2026)

**Problem.** Benchmarks show that cofolding models largely *memorize* binding modes and collapse on new pockets. AF3 fails on more than 50% of antibody-antigen docking cases. Accurate structures also do not translate into accurate affinities [30].

**Architecture (what is public).** IsoDDE is described as a unified system. Its **predictive core**, the part previewed in the report, covers structure prediction, pocket identification and binding-affinity prediction, integrated with "generative capability to navigate and optimise within vast molecular design spaces" [30]. Architecture, training data, weights and code have not been released. A secondary analysis notes that this limits independent verification [30][45h].

```
   target (T) + candidate ligand/biologic (L)
                |
                v
   +---------------------------------------------+
   | IsoDDE predictive core                      |
   |  structure  p(X | T, L)  K samples (e.g.25) |
   |  pocket ID  p(P_i | T)                      |
   |  affinity   A = f(T, L, X)                  |
   |  confidence C -> pick argmax_k C(X_k)       |
   +----------------------+----------------------+
                          | scores
                          v
   generative design (not detailed publicly) --> make/test in wet lab -->
   new data --> (presumably) model refinement      [inferred loop, not documented]
```

**Reported results** [30][31]:

| Capability | Reported result |
|---|---|
| Protein-ligand cofolding, hardest Runs N' Poses similarity bin (0-20) | more than 2x AF3 (about 50% vs about 23%) |
| Antibody-antigen high-quality interfaces | 2.3x AF3, 19.8x Boltz-2 |
| CDR-H3 loop modeling | 1.2x AF3, 1.6x Boltz-2 |
| Pocket identification | 1.5x AUPRC over P2Rank |
| Affinity (mean Pearson r, FEP+4-type set) | 0.85 vs 0.78 for FEP+ |
| Affinity (OpenFE benchmark) | 0.73 vs 0.72 |

Structure metrics are taken from the highest-confidence of 25 samples, so **test-time sampling plus confidence ranking** is part of the method, not an extra [30][45h].

**Business-side loop.** Isomorphic announced a $600M first external round in March 2025 and a $2.1B Series B in May 2026 [32]. A secondary review reports 17 active programs and a stated target of a first program in Phase 1 by the end of 2026 [33]. Treat the program count and timeline as approximate and company-reported.

**Lessons.**
- **Generalization, not in-distribution accuracy, is the bottleneck for design loops.** A design loop is out-of-distribution by definition, since it proposes molecules not yet made. Evaluators must be benchmarked on similarity-stratified splits.
- **Learned scoring is challenging physics-based scoring (FEP) on accuracy at much lower cost.** This changes loop economics: more candidates can be scored per make-test cycle. The claims remain company benchmarks on curated sets.

---

## 7. Comparison

| System | Executor (oracle) | Proposer | Ranking/selection | Human role | Scale / cost (public) | Main failure mode |
|---|---|---|---|---|---|---|
| Mobile robotic chemist [20] | Mobile robot + GC | Batched BO | Acquisition function | Defines space | about 688 expts / 8 days | Fixed search space |
| A-Lab [16] | 3-station powder robot + XRD | NLP recipe model, then ARROWS3 | Yield > 50%; DFT driving force | Picks targets; regrinds | 355 recipes / 17 days; 41/58 | Kinetics; DFT errors; novelty claims [17] |
| GNoME [18] | DFT (VASP) | GNN-filtered substitutions | Predicted energy | Pipeline design | 2.2M stable; 6 AL rounds | Sim-to-lab gap |
| AMASE / MAD [22][23] | XRD (+ resistance) | GP + CALPHAD / multi-output GP | Acquisition per task | Seeds space | 8 h / 66 XRD; 5 h / 25 iters | Instrument idle time |
| Coscientist [14] | OT-2, Emerald Cloud Lab | GPT-4 Planner + tools | LLM reasoning | Prompts task | 20-iteration optimization games | Simple tasks; API errors |
| AI co-scientist [7] | Humans' wet lab (rare) | Gemini agents, debate | Elo tournament + meta-review | Goal, reviews, lab | Test-time compute scaled | Self-judged Elo; open-access-only literature |
| Robin [10] | Humans' wet lab | o4-mini + PaperQA2 agents | LLM-judge pairwise | Runs experiments | ~10 Finch runs/dataset | Inflated effect sizes |
| Kosmos [13] | Data analysis + literature | Parallel agents + world model | World-model queries | Objective + data | ~12 h, ~42k LOC, ~1,500 papers | about 21% of statements refuted |
| AI Scientist v1 [1] | GPU (template code) | LLM + Aider | Automated reviewer | Writes template | under $15/paper | Hallucinated numbers; harness edits |
| AI Scientist v2 [3] | GPU (open-ended) | LLM + agentic tree search | BFTS node scores + VLM | Topic file | $20-25/run; hours | Lower success than v1 |
| IsoDDE [30] | In-silico predictive core | Generative design (undisclosed) | Confidence over 25 samples; affinity | Med-chem teams | Undisclosed | Opaque; company benchmarks |

---

## 8. Cross-cutting patterns and failure modes

1. **Evaluator cost decides loop topology.** When evaluations are cheap and objective (GPU runs, DFT), systems use wide search: trees, populations, active learning over millions of candidates. When they are expensive (wet labs), systems put their compute *before* the experiment: tournaments, debates, literature grounding. They send only a few top candidates to the lab and let humans do the physical work. Co-scientist, Robin and Kosmos are "think a lot, test a little" systems. A-Lab and AMASE are "test a lot, think locally".
2. **LLM priors versus calibrated uncertainty.** Coscientist and PerTurboAgent show that LLM proposers can beat cold-start BO or active learning when text covers the domain well [14][38]. AMASE and MAD show that physics priors plus GP uncertainty win when models of the physics exist [22][23]. Hybrids are emerging: LLMs set the search space, choose surrogates and write protocols, while BO libraries such as Atlas and BayBE do the acquisition [25][26][27].
3. **Self-evaluation is the weakest link.** Elo from related agents, automated reviewers, LLM judges and consensus analyses are all *internal* signals. Every independent check found a gap: Leeman et al. on A-Lab novelty [17], Beel et al. on AI Scientist numbers [6], human re-analysis of Robin's effect size [45f], and Kosmos's roughly 21% refuted statements [46].
4. **Verification gates as first-class components.** The strongest 2026 designs build verification into the architecture. Examples are numeric registries and citation verifiers [41], cite-every-claim reports [13], multi-scale re-validation [43], automatic evaluators with shipped checking code [44], and similarity-stratified benchmarks for drug-design oracles [30].
5. **Feedback flows both ways between simulation and experiment.** The A-Lab found Materials Project pseudopotential errors [16]. GNoME's 736 experimental matches are its external validity check [18]. A closed-loop lab is also an **oracle for correcting databases**.
6. **Agent sandboxing is mandatory.** Coscientist ran code in Docker [14]. The AI Scientist repositories require containers and warn about LLM-written code spawning processes [2][4]. AutoResearchClaw disables networking during measurement [41]. MARS has a code reviewer for safety before robotic execution [24]. Physical loops add a further layer: an LLM's output can move a robot arm or heat a furnace.
7. **Schedulers and asynchrony.** Instruments run at different speeds (MAD's idle resistance probe [23]), batches of parallel workers (AI Scientist v2 `num_workers` [4]) and pending experiments (BayBE async campaigns [27]) make the loop controller a job scheduler with partial observations.

---

## Key takeaways for system designers

- **Write down the oracle hierarchy explicitly.** List every evaluator from cheapest to most expensive: LLM judge, ML surrogate, simulation, automated measurement, independent human or lab validation. Decide how many candidates each tier passes on. Treat promotion between tiers as the main control knob.
- **Never let the loop's success metric become the scientific claim.** A-Lab's "majority phase" and the AI Scientist's "reviewer score" are useful optimization targets. Discovery claims need a separate, independent gate.
- **Ground generation in retrieval, and ground conclusions in execution.** Web and documentation search fixed Coscientist's synthesis errors. Cite-to-notebook reports made Kosmos auditable. Numeric registries stop hallucinated results from reaching the paper.
- **Use LLMs where the prior is valuable and BO where the uncertainty is valuable.** Let the LLM define the search space, the protocol and the hypotheses. Let a calibrated surrogate pick points in a well-posed continuous space.
- **Spend test-time compute on ranking, not only generation.** Tournaments (co-scientist, Robin) and best-first tree search (AI Scientist v2) beat sampling more candidates, but only as far as the ranker is trustworthy. Plan for evaluator monoculture by mixing model families or adding measured signals.
- **Run several analyses and reconcile them.** Robin's 10-trajectory consensus is cheap insurance against variance in agentic data analysis. Keep a human re-analysis step for headline effect sizes.
- **Design the action space to include the fixes your failure analysis will need.** A-Lab could not regrind or run multistep heating, so 11 kinetic failures were outside its reach. Log failures in a structured way so the next version's action space can be widened.
- **Benchmark evaluators on out-of-distribution splits.** Design loops propose new things by construction. IsoDDE's similarity-binned results show memorization hiding in average scores.
- **Sandbox code, network and hardware independently.** Containers for code, network policies for measurement phases, and a reviewer agent or rule engine before any physical actuation.
- **Keep humans where they add the most information per minute.** Goal setting, reviewing hypotheses, physical execution and final validation. AutoResearchClaw's CoPilot result (87.5% against 25% accept rate for full autonomy) and the co-scientist's expert-in-the-loop results both suggest that today's best systems are collaborative, not autonomous.

---

## References

1. Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., Ha, D. "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." Sakana AI et al., 2024. https://arxiv.org/abs/2408.06292
2. Sakana AI. "AI-Scientist" (code, templates, FAQ on cost and reviewer models). GitHub, 2024. https://github.com/SakanaAI/AI-Scientist
3. Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., Ha, D. "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search." 2025. https://arxiv.org/abs/2504.08066
4. Sakana AI. "AI-Scientist-v2" (code, bfts_config, cost FAQ). GitHub, 2025. https://github.com/SakanaAI/AI-Scientist-v2
5. Sakana AI. "The AI Scientist: ICLR 2025 Workshop Experiment" (submitted papers, scores, reviews). GitHub, 2025. https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment
6. Beel, J., Kan, M.-Y., Baumgart, M. "Evaluating Sakana's AI Scientist for Autonomous Research: Wishful Thinking or an Emerging Reality Towards 'Artificial Research Intelligence' (ARI)?" 2025. https://arxiv.org/abs/2502.14297
7. Gottweis, J. et al. "Towards an AI co-scientist." Google, 2025. https://arxiv.org/abs/2502.18864
8. Google Research. "Accelerating scientific breakthroughs with an AI co-scientist." 2025. https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/
9. Gottweis, J. et al. "Accelerating scientific discovery with Co-Scientist." Nature, 2026. https://doi.org/10.1038/s41586-026-10644-y
10. Ghareeb, A. E., Chang, B., Mitchener, L., et al. "Robin: A multi-agent system for automating scientific discovery." FutureHouse, 2025. https://arxiv.org/abs/2505.13400
11. FutureHouse. "robin" (code, example outputs for 10 diseases). GitHub, 2025. https://github.com/Future-House/robin
12. Ghareeb, A. E. et al. "A multi-agent system for automating scientific discovery." Nature, 2026. https://doi.org/10.1038/s41586-026-10652-y
13. Mitchener, L. et al. "Kosmos: An AI Scientist for Autonomous Discovery." Edison Scientific / FutureHouse, 2025. https://arxiv.org/abs/2511.02824
14. Boiko, D. A., MacKnight, R., Kline, B., Gomes, G. "Autonomous chemical research with large language models." Nature 624, 570-578, 2023. https://www.nature.com/articles/s41586-023-06792-0
15. Bran, A. M. et al. "Augmenting large language models with chemistry tools" (ChemCrow). Nature Machine Intelligence, 2024. https://doi.org/10.1038/s42256-024-00832-8
16. Szymanski, N. J., Rendy, B., Fei, Y., ..., Persson, K., Zeng, Y., Ceder, G. "An autonomous laboratory for the accelerated synthesis of novel materials." Nature, 2023. https://www.nature.com/articles/s41586-023-06734-w
17. Leeman, J. et al. "Challenges in High-Throughput Inorganic Material Prediction and Autonomous Synthesis." PRX Energy 3, 011002, 2024. https://doi.org/10.1103/PRXEnergy.3.011002
18. Merchant, A., Batzner, S., Schoenholz, S. S., Aykol, M., Cheon, G., Cubuk, E. D. "Scaling deep learning for materials discovery." Nature, 2023. https://www.nature.com/articles/s41586-023-06735-9
19. Google DeepMind. "Materials Discovery: GNoME" (dataset and models). GitHub, 2023-2024. https://github.com/google-deepmind/materials_discovery
20. Burger, B. et al. "A mobile robotic chemist." Nature 583, 237-241, 2020. https://doi.org/10.1038/s41586-020-2442-2
21. MacLeod, B. P. et al. "Self-driving laboratory for accelerated discovery of thin-film materials." Science Advances 6, eaaz8867, 2020. https://doi.org/10.1126/sciadv.aaz8867
22. Liang, H. et al. "Real-time experiment-theory closed-loop interaction for autonomous materials science" (AMASE). 2024. https://arxiv.org/abs/2410.17430
23. Lee, Liang, Kim, McDannald, Ocampo, Kusne, Takeuchi, Rios. "Real-time Multi-instrument Autonomous Discovery of Novel Phase-change Memory Materials." 2026. https://arxiv.org/abs/2605.18033
24. Shi et al. "Knowledge-driven autonomous materials research via collaborative multi-agent and robotic system" (MARS). Matter 9, 102577, 2026. https://doi.org/10.1016/j.matt.2025.102577
25. Angelopoulos, Cahoon, Alterovitz. "From Prompts to Protocols: An AI Agent for Laboratory Automation" (EOS AI agent). 2026. https://arxiv.org/abs/2605.16552
26. Aspuru-Guzik group (University of Toronto / Vector Institute). "atlas: Bayesian optimization for experimental science" (code; Digital Discovery 2025 paper linked). GitHub. https://github.com/aspuru-guzik-group/atlas
27. Merck KGaA / EMD Group. "BayBE: A Bayesian Back End for Design of Experiments." GitHub. https://github.com/emdgroup/baybe
28. Abramson, J. et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature, 2024. https://doi.org/10.1038/s41586-024-07487-w
29. Google DeepMind. "alphafold3" (inference code, weights terms, variants). GitHub. https://github.com/google-deepmind/alphafold3
30. Isomorphic Labs Team. "Accurate Predictions of Novel Biomolecular Interactions with IsoDDE" (technical report). 10 February 2026. https://storage.googleapis.com/isomorphiclabs-website-public-artifacts/isodde_technical_report.pdf
31. Isomorphic Labs. "The Isomorphic Labs Drug Design Engine unlocks a new frontier." 2026. https://www.isomorphiclabs.com/articles/the-isomorphic-labs-drug-design-engine-unlocks-a-new-frontier
32. Isomorphic Labs. "Isomorphic Labs announces Series B investment round." 2026. https://www.isomorphiclabs.com/articles/isomorphic-labs-announces-series-b-investment-round
33. ChatForest. "AlphaFold Won the Nobel Prize for Predicting Proteins. Now the Same Team Is Designing the Drugs." (secondary review of Isomorphic funding, programs, partnerships). 2026. https://github.com/Chatforest/chatforest.com/blob/master/content/reviews/isomorphic-labs-alphafold-drug-design-engine-clinical-trials-2-1b-series-b-2026.md
34. Latent Labs Team. "Latent-Y: A Lab-Validated Autonomous Agent for De Novo Drug Design." 2026. https://arxiv.org/abs/2603.29727
35. Huang, K. et al. "Biomni: A General-Purpose Biomedical AI Agent." bioRxiv, 2025. https://doi.org/10.1101/2025.05.30.656746
36. Cong, L. et al. "LabOS: The AI-XR Co-Scientist That Sees and Works With Humans." 2025. https://arxiv.org/abs/2510.14861
37. Zhang et al. "OriGene: A Self-Evolving Virtual Disease Biologist Automating Therapeutic Target Discovery." bioRxiv, 2025. https://doi.org/10.1101/2025.06.03.657658
38. Hao, Lee, Wang, Scalia, Regev. "PerTurboAgent: An LLM-based Agent for Designing Iterative Perturb-Seq Experiments." PMLR v311, 2025. https://proceedings.mlr.press/v311/hao25b.html
39. Schmidgall, S. et al. "Agent Laboratory: Using LLM Agents as Research Assistants" (and AgentRxiv). 2025. https://github.com/SamuelSchmidgall/AgentLaboratory (paper: https://arxiv.org/abs/2501.04227)
40. NovelSeek Team (Shanghai AI Laboratory). "NovelSeek: When Agent Becomes the Scientist - Building Closed-Loop System from Hypothesis to Verification." 2025. https://arxiv.org/abs/2505.16938
41. Liu et al. "AutoResearchClaw: Self-Reinforcing Autonomous Research with Human-AI Collaboration." 2026. https://arxiv.org/abs/2605.20025
42. Miyai, Toyooka, Otonari, Zhao, Aizawa. "Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper." TMLR, 2026. https://arxiv.org/abs/2511.04583
43. Pepe et al. "Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design." Meta FAIR, 2026. https://arxiv.org/abs/2605.15871
44. Novikov, A. et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." Google DeepMind technical report, 2025. https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf (results and verification code: https://github.com/google-deepmind/alphaevolve_results)
45. Secondary paper notes used to cross-check primary claims (GitHub-hosted reading notes):
    - 45a. "A Mobile Robotic Chemist" (paper note). https://github.com/jmeier1963/omegawiki-insilico-socialscience/blob/main/wiki/papers/burger-mobile-robotic-chemist.md
    - 45b. "Challenges in High-Throughput Inorganic Material Prediction and Autonomous Synthesis" (paper note). https://github.com/jmeier1963/omegawiki-insilico-socialscience/blob/main/wiki/papers/leeman-challenges-autonomous-synthesis.md
    - 45c. "Autonomous Chemical Research with Large Language Models" (paper note). https://github.com/jmeier1963/omegawiki-insilico-socialscience/blob/main/wiki/papers/boiko-autonomous-chemical-research.md
    - 45d. "Paper Note: Towards an AI Co-Scientist." https://github.com/flonat/flonat-research/blob/main/.context/resources/google-ai-co-scientist-review.md
    - 45e. "Source Note: Towards an AI Co-Scientist." https://github.com/corbensorenson/asi-stack-book/blob/main/sources/source_notes/ext_ai_co_scientist_2025.md
    - 45f. "Robin: A multi-agent system for automating scientific discovery" (paper note). https://github.com/jmsung/einstein/blob/main/knowledge/domains/ai-agents/source/2025-ghareeb-robin-multi-agent-system-automating.md
    - 45g. "Source: Evaluating Sakana's AI Scientist (independent critique)." https://github.com/noah-goodrich/claude-plugins/blob/main/research-tools/docs/research/2026-06-05-meta-review/sources/t4-aiinvention-beel-evaluating-sakana.md
    - 45h. "IsoDDE: From Biomolecular Structure Prediction to a Computational Drug Design Engine" (research landscape post). 2026. https://github.com/sungsoo/sungsoo.github.io/blob/master/_posts/2026-08-17-isodde-computational-drug-design-engine.html
46. Scripps AI Enablement. "Autonomous AI scientist landscape" and per-system pages (Co-Scientist, AI Scientist, Robin, Kosmos, Biomni, OriGene, LabOS, AMASE, MAD, MARS, EOS AI agent, Latent-Y, AutoResearchClaw, AIRA, NovelSeek, Jr. AI Scientist, PerTurboAgent, ChemCrow). GitHub, last reviewed 2026. https://github.com/scripps-ai-enablement/sci-ai-enabler/tree/main/autonomous-science
47. Gao, S. et al. "Empowering biomedical discovery with AI agents." Cell 187, 2024. https://doi.org/10.1016/j.cell.2024.09.022
48. Microsoft Research. "MatterGen: a generative model for inorganic materials design" (code; Nature 2025 paper). GitHub. https://github.com/microsoft/mattergen (paper: https://www.nature.com/articles/s41586-025-08628-5)
49. FutureHouse. "PaperQA2" (literature agent underlying Crow/Falcon) and "Aviary" (language-agent gym). GitHub. https://github.com/Future-House/paper-qa ; https://github.com/Future-House/aviary
50. Sakana AI. "The AI Scientist in Nature" (announcement of the peer-reviewed version, Nature 651, 914-919, 2026). https://sakana.ai/ai-scientist-nature/
