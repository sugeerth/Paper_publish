# 17. Design Google Street View Face and License-Plate Blurring (Object Detection Pipeline)

> A privacy pipeline that finds every face and license plate in billions of panoramic
> street images and blurs them before publication. Unlike most chapters, there is no
> user request in the loop. The "serving path" is a high-throughput **batch inference**
> pipeline, the model is an **object detector**, and the business metric favours
> **recall** (a missed face is a privacy incident; an extra blur costs little).
>
> Research note: in the review pass only github.com URLs could be fetched (other hosts
> failed DNS resolution, and the search budget was used up). Tags: (verified) = fetched
> in the review; (ch-ref) = verified in an earlier chapter of this atlas; (known) = a
> stable arXiv or paper URL not re-fetched. Numbers in tables are typical assumptions
> unless they carry a citation.

## Company tags

| Company | What is reported | Source |
|---|---|---|
| Google | Owns the real system: automatic face and plate blurring in Street View since 2008 (ICCV 2009 paper) | [2] [3] |
| Google (interview) | ByteByteGo ch. 3 is titled "Google Street View Blurring System" | [1] [6] |
| Meta | Real comparable system: EgoBlur, open-source face and plate blurring for Project Aria egocentric data | [25] |
| Generic (any company) | alirezadir lists "Image blurring system" as a sample CV design question, with no company named | [4] |

**Portals that cover it**

- ByteByteGo, Aminian & Xu, *Machine Learning System Design Interview*, ch. 3, "Google Street View Blurring System". This is the canonical write-up [1]. Its 27-item reference list on GitHub covers R-CNN family, YOLO, SSD, DETR, NMS, COCO eval, distributed training, GDPR, face-detection bias, active learning, human-in-the-loop [6].
- alirezadir / Machine-Learning-Interviews: CV section with two-stage (R-CNN, Fast/Faster R-CNN) and one-stage (YOLO, SSD) detectors, mAP, and "Image blurring system" as a sample question [4].
- Educative "Grokking the ML Interview" covers the related self-driving image segmentation case [7].
- Exponent, Hello Interview, Patrick Halina: generic delivery frameworks that this chapter maps onto a CV pipeline [8] [9] [10].
- Not covered: Chip Huyen's MLSD exercises have image retrieval and image classification but no detection or blurring exercise [12]; khangich/machine-learning-interview has no CV design case [13].

## Requirements & scale

| Dimension | Value | Source |
|---|---|---|
| Functional | Input: street-level image or panorama. Output: the same image with every face and plate blurred | [1] |
| Objects | 2 classes (human face, license plate). Optional extras: house numbers, people in private settings | [1] [2] |
| Primary objective | High recall at a fixed precision floor. A missed face is a privacy breach | [1] [2] |
| Reported quality (Google 2009) | About 89% of faces and 94-96% of plates blurred automatically | [2] |
| Latency | Not user-facing: batch, hours to days before publication | [1] |
| Throughput | About 1M+ new images a day; every panorama is about 75-100 MP (stitched from 8-15 cameras) | assumption |
| Total corpus | Tens of billions of images, re-run whenever the model is upgraded | assumption |
| Labelled data | About 1M images with bounding-box annotations (ByteByteGo's stated assumption) | [1] |
| Compliance | GDPR / local privacy law: raw imagery retention limits, deletion on request | [6] |
| User feedback | "Report a problem" button: users ask for extra blurring (whole house, person, car) | [3] |
| Fairness | Recall must hold across skin tones, ages, poses, plate formats from 100+ countries | assumption |
| Cost | GPU-hours per image dominate; tile and batch to keep accelerator use above 70% | assumption |

## Diagram 1 - End-to-end batch "serving" path

**Diagram 1. Ingestion to publication: tiling, detection, NMS, policy, blurring, serving**

```text
+-------------+  raw frames  +---------------+  panorama  +----------------------+
| Street View |------------->| Ingest + QA   |----------->| Image store (GCS/    |
| car / trek  |  + GPS, pose | (dedupe, blur |  + meta    | Colossus) raw, ACL'd |
| cameras     |              |  check, exif) |            +----------+-----------+
+-------------+              +---------------+                       |
                                                                     | job queue
                                                                     v
+--------------------+   tiles (e.g. 1024x1024, 20% overlap)  +----------------------+
| Preprocess         |<---------------------------------------| Batch scheduler      |
| - equirect -> cube |                                        | (Dataflow / Beam,    |
| - multi-scale pyr. |                                        |  shard by panorama)  |
| - normalise        |                                        +----------------------+
+---------+----------+
          | tensor batches
          v
+-------------------------+  load model vN   +----------------------+
| Detector fleet (GPU/TPU)|<-----------------| Model registry       |
| face + plate detector   |                  | (versioned weights,  |
| -> boxes, class, score  |                  |  thresholds per cls) |
+-----------+-------------+                  +----------------------+
            | raw boxes per tile
            v
+-------------------------+  merged boxes  +----------------------------------+
| Tile merge + NMS        |--------------->| Policy / post-process            |
| - map tile->pano coords |                | - low threshold (recall-first)   |
| - cross-tile dedupe     |                | - optional 2nd-stage verifier    |
| - (soft-)NMS, IoU 0.5   |                | - expand box margin 10-20%       |
+-------------------------+                | - uncertain band -> human review |
                                           +--------+------------------+------+
                                                    | final boxes      | low conf.
                                                    v                  v
+-------------------------+  boxes + pano  +----------------+  +------------------+
| Blur renderer           |<---------------| Box store      |  | Human review     |
| irreversible Gaussian / |                | (pano_id,boxes |  | queue (sampled + |
| mosaic, no raw copy out |                |  model_ver)    |  | uncertain)       |
+-----------+-------------+                +-------+--------+  +--------+---------+
            | publish                              | log               | labels
            v                                      v                   v
+-------------------------+  "Report a     +---------------------------------------+
| Serving CDN / Maps tiles|  problem"      | Logging & feedback store              |
| (only blurred imagery)  |--------------->| predictions, reviews, user reports    |
+-------------------------+  user reports  +---------------------------------------+
```

- No feature store: the "features" are pixels plus metadata (camera id, GPS, time).
- The raw store is tightly access-controlled. Only blurred derivatives reach serving [2] [3].
- If the model is upgraded, the corpus is backfilled. Blurs are additive (new boxes are
  unioned with old ones), so the new model never un-blurs anything.

## Diagram 2 - Offline training & data pipeline

**Diagram 2. Labels, augmentation, training, recall-gated evaluation, registry, backfill**

```text
+-------------------+   +--------------------+   +---------------------------+
| Sampled raw panos |   | Public datasets    |   | Feedback store            |
| (stratified by    |   | WIDER FACE, plate  |   | user reports, reviewer    |
| country, weather, |   | sets, COCO person  |   | fixes, missed-face cases  |
| time of day)      |   +---------+----------+   +-------------+-------------+
+---------+---------+             |                            |
          | to annotate           | pretrain / mix             | hard examples
          v                       v                            v
+-------------------+   +---------------------------------------------------+
| Annotation tool   |-->| Labelled set (about 1M images [1])                |
| boxes + class +   |   | split by geography/time, never by random tile     |
| "ignore" regions  |   | (prevents leakage of the same scene across splits)|
| 2x label + adjud. |   +-------------------------+-------------------------+
+-------------------+                             |
                                                  v
+----------------------------------+    +----------------------------------+
| Augmentation (offline + online)  |<---| Hard-negative / hard-positive    |
| flip, crop, scale jitter, blur,  |    | mining (posters, statues,        |
| color, rain/night synth, mosaic  |    | mannequins, tiny far faces)      |
+----------------+-----------------+    +----------------------------------+
                 | tf.data / tfrecords
                 v
+----------------------------------+  checkpoints  +---------------------------+
| Distributed training (GPU/TPU)   |-------------->| Offline evaluation        |
| detector, ImageNet/COCO init,    |               | mAP@0.5, AP per class,    |
| cls loss + box regression loss   |               | recall @ fixed precision, |
+----------------------------------+               | slices: size, country,    |
                 ^                                 | skin tone, night, plates  |
                 |                                 +-------------+-------------+
                 | retrain (monthly / on drift)                  | pass gates?
                 |                                               v
+----------------+-----------------+  shadow run   +---------------------------+
| Monitoring & drift               |<--------------| Model registry            |
| score histograms, boxes/pano,    |               | weights + per-class       |
| report rate, review miss rate    |               | thresholds + eval report  |
+----------------------------------+               +-------------+-------------+
                                                                 | canary 1% ->
                                                                 v  full backfill
                                                   +---------------------------+
                                                   | Batch pipeline (Diagram 1)|
                                                   +---------------------------+
```

## Diagram 3 - Model architecture

**Diagram 3. Two-stage detector (Faster R-CNN + FPN), the ByteByteGo choice [1] [14] [16]**

```text
 input tile 3 x H x W (e.g. 1024 x 1024)
          |
          v
+-------------------------------+
| Backbone CNN (ResNet-50 /     |  C2..C5 feature maps
| EfficientNet / ViT)           |------------------+
+-------------------------------+                  |
                                                   v
                                  +---------------------------------+
                                  | FPN: top-down + lateral fusion  |
                                  | P2..P6 multi-scale features     |
                                  | (tiny far faces live in P2/P3)  |
                                  +-------+-----------------+-------+
                                          |                 |
                    anchors (scales x     v                 |
                    aspect ratios:  +-------------------+   |
                    faces ~1:1,     | Region Proposal   |   |
                    plates ~1:4)    | Network (RPN)     |   |
                                    | objectness + bbox |   |
                                    +---------+---------+   |
                                              | ~1-2k proposals
                                              v             v
                                  +---------------------------------+
                                  | RoIAlign: 7x7 pooled features   |
                                  +----------------+----------------+
                                                   |
                                                   v
                                  +---------------------------------+
                                  | Detection head (2 FC layers)    |
                                  |  +-> cls: {bg, face, plate}     |
                                  |  +-> box deltas (dx,dy,dw,dh)   |
                                  +----------------+----------------+
                                                   |
                                                   v
                                  +---------------------------------+
                                  | Per-class NMS (IoU 0.5) +       |
                                  | score threshold (recall-tuned)  |
                                  +----------------+----------------+
                                                   |
                                                   v
                              output: [(x1,y1,x2,y2, class, score), ...]

 Loss = L_rpn_cls (BCE) + L_rpn_box (smooth L1)
      + L_cls (cross-entropy or focal) + L_box (smooth L1 / GIoU)
```

## Diagram 4 - Variant: one-stage detector plus the 2009 Google cascade

**Diagram 4. Alternative designs: one-stage dense detector vs the original production
cascade [2] [15] [17]**

```text
 (A) One-stage (RetinaNet / YOLO / SSD-style)    (B) Google Street View 2009 [2]
 --------------------------------------------    -------------------------------
 tile --> backbone + FPN                          panorama
           |                                         |
           v                                         v
 +---------------------+ +---------------------+  +-------------------------+
 | class subnet        | | box subnet          |  | Sliding-window face     |
 | (per anchor/point,  | | (per anchor/point,  |  | detectors (frontal +    |
 |  focal loss)        | |  L1 / GIoU)         |  | profile), high-recall   |
 +----------+----------+ +----------+----------+  +------------+------------+
            |                       |                          | many candidates
            +-----------+-----------+                          v
                        v                         +-------------------------+
             +---------------------+              | Post-processing         |
             | NMS -> boxes        |              | classifier (features:   |
             +---------------------+              | detector scores, skin   |
                                                  | color, 3D/pose cues)    |
 + faster, simpler to batch                       +------------+------------+
 + good for edge / on-car inference                            | kept boxes
 - weaker on tiny objects without                              v
   careful anchors / high-res input               +-------------------------+
                                                  | Plate detector (separate|
                                                  | cascade) + blur         |
                                                  +-------------------------+
                                                  89% faces, 94-96% plates
```

## Data & labels

| Source | Label type | Size / volume | Notes | Source |
|---|---|---|---|---|
| Human-annotated Street View images | Bounding boxes, 2 classes | About 1M images | ByteByteGo assumption. Annotate tiles, keep "ignore" regions for crowds | [1] |
| Public face sets (WIDER FACE) | Boxes, with blur, pose and occlusion attributes | 32k images, 394k faces | Pretraining and hard tiny-face cases | [18] |
| Public plate sets (country specific) | Boxes / corners | Varies | Plate shapes and scripts differ by country | assumption |
| User "Report a problem" | Image-level request plus a region | Ongoing | Noisy, and covers more than faces and plates (houses, cars) | [3] |
| Reviewer corrections | Missed or extra boxes | Sampled 0.1-1% of output | Gold set for monitoring | assumption |
| Synthetic | Pasted faces and plates, rendered weather | Unlimited | Fills rare slices | assumption |
| Active learning | Uncertain-band boxes sent to annotators | Ongoing | Cuts labelling cost; human-in-the-loop loop in ByteByteGo's refs | [6] |

## Features

| Feature | Type | Used by | Notes |
|---|---|---|---|
| RGB pixels (tile) | Tensor | Backbone | Main signal, multi-scale |
| Image pyramid level / tile coords | Integer | Merge step | Maps boxes back to the panorama |
| Camera id, lens, projection | Categorical | Preprocess | Corrects equirectangular distortion |
| GPS / country | Categorical | Thresholds, plate model | Per-region plate formats |
| Time of day, weather | Categorical | Eval slices | Night recall drops |
| Neighbour-frame boxes | Boxes | Temporal consistency | The same face seen from the next capture point |

## Model choices (baseline -> advanced)

| Stage | Model | Pros | Cons | Source |
|---|---|---|---|---|
| Baseline | Haar / HOG sliding window + classifier (Viola-Jones style) | Cheap, explainable | Low recall on profile and small faces | [2] |
| Two-stage | Faster R-CNN + FPN | High accuracy, good on small objects. ByteByteGo's pick for offline batch | Slower, two-step training | [1] [14] [16] |
| Cascade | Cascade R-CNN | Better localisation at high IoU | More compute | [19] |
| One-stage | RetinaNet / YOLO / SSD | Fast, simple. Focal loss handles class imbalance | Historically weaker on tiny objects | [15] [17] |
| Specialist | RetinaFace for faces + plate detector | Best per-class recall | Two models to run | [20] |
| Transformer | DETR / ViTDet | No anchors or NMS (DETR); strong scaling | Slow convergence, compute-heavy | [21] [22] |
| Ensemble | Union of detectors -> verifier | Maximises recall | Cost, more false positives | assumption |
| Open-source reference | Meta EgoBlur face + plate detectors | Pretrained, released for privacy use | Tuned to egocentric Aria imagery | [25] |

## Loss

| Component | Loss | Why |
|---|---|---|
| Classification | Cross-entropy over {bg, face, plate}, or focal loss | Background dominates, and focal loss down-weights easy negatives [15] |
| Box regression | Smooth L1 on (dx, dy, dw, dh), or GIoU / CIoU | GIoU is aligned with the IoU metric [23] |
| RPN (two-stage) | BCE objectness + smooth L1 | Proposal quality |
| Recall bias | Class weights / lower positive IoU threshold for faces | A missed face costs more than an extra blur |

## Offline metrics

| Metric | Definition | Target / use |
|---|---|---|
| IoU | Area of overlap / area of union | Match threshold, usually 0.5 [1] |
| Precision, recall | At IoU >= 0.5 | Report **recall at a fixed precision** (for example 0.98 recall at P >= 0.8) |
| AP / mAP | Area under the PR curve, averaged over classes (and IoU 0.5:0.95 COCO-style) | Main model-selection metric [1] |
| Recall by object size | Small (<16 px), medium, large | Tiny far faces are the hardest slice |
| COCO-style AP | AP@[0.5:0.95], AP_small/medium/large | Standard detection eval protocol [6] |
| Slice recall | Country, skin tone, night, occlusion, plate format | Fairness and coverage gates |
| Pixel coverage | Fraction of face pixels covered by any blur | Partial blur still counts as a leak |

## Online metrics / A-B

| Metric | Definition | Notes |
|---|---|---|
| User privacy reports per 1M images | "Report a problem" blur requests | Main online signal [1] [3] |
| Reviewer-measured miss rate | Missed faces or plates in an audited sample | Gold metric, not delayed |
| Over-blur rate | Blurred non-targets (signs, posters) | Hurts imagery usefulness |
| Throughput / cost | Images per GPU-hour, backlog age | Publication SLA |
| A-B method | Not user-randomised. Use a shadow run on a holdout region plus a human audit of disagreements between old and new models | No traffic split needed |

## Serving & latency budget (per panorama, batch)

| Step | Budget | Hardware / note |
|---|---|---|
| Fetch + decode + reproject | 0.3-0.5 s | CPU workers |
| Tiling + pyramid (about 40-80 tiles) | 0.1 s | CPU |
| Detection (batched tiles) | 0.5-2 s | GPU/TPU, FP16/INT8, batch 16-64 |
| Merge + NMS + policy | <0.1 s | CPU |
| Blur render + write | 0.2 s | CPU |
| End-to-end | About 1-3 s compute; wall-clock SLA hours | Throughput matters, latency does not |
| Scale check | 1M panos/day x 2 s = about 23 GPU-days/day, so about 25-50 GPUs | assumption |

## Monitoring

| Signal | Alert on |
|---|---|
| Boxes per panorama (by class, region) | Sudden drop means a model or camera regression |
| Score distribution | Shift (new camera hardware, new country) |
| Reviewer miss rate on the gold sample | Above the SLA, for example >1% missed faces |
| User report rate | Spike by region or capture batch |
| Pipeline health | Backlog age, failed shards, GPU utilisation |
| Data drift | New plate formats, snow or night share, new lens |

## Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| Missed tiny or distant faces | Low resolution | High-res tiles, P2 FPN level, multi-scale test-time augmentation |
| Faces split across tile seams | Tiling | Overlap tiles, merge with NMS in panorama coordinates |
| Equirectangular distortion at poles | Projection | Cube-map reprojection |
| False positives on posters, statues, billboards | Face-like patterns | Hard-negative mining. Over-blur is acceptable, so bias toward blurring |
| Blurring signs or business names | Plate-like text | Verifier, context features |
| Foreign plate formats | Domain shift | Per-country data, region thresholds |
| Demographic recall gaps | Biased training data | Stratified labelling, slice gates |
| Partial blur leaks identity | Tight box | Expand box margin, pixel-coverage metric |
| De-blurring attack | Weak blur | Irreversible blur (pixelate + noise), never ship the raw image |
| Model upgrade un-blurs | Replace instead of union | Union old and new boxes; boxes only ever get added |

## Interviewer follow-ups

- Why prioritise recall over precision, and how do you pick the operating threshold?
- Two-stage vs one-stage detector for an offline batch job: which and why? [1]
- How does NMS work, what goes wrong with crowds, and when would you use soft-NMS? [24]
- How do you handle 100-megapixel panoramas: downscale, tile, or multi-scale?
- How do you avoid duplicate or split boxes at tile boundaries?
- How do you get 1M labelled images cheaply? Active learning, pseudo-labels, user reports.
- How do you split train and test to avoid leakage between neighbouring frames of one street?
- mAP vs recall-at-precision: which do you report to the privacy team?
- How do you measure and guarantee recall across skin tones and countries?
- How would you A-B test a new model when there is no click signal?
- How do you backfill tens of billions of images when the model improves, and at what cost?
- How do you handle user requests to blur a whole house or car (out of model scope)?
- How would you run this on the capture vehicle (edge) instead of in the data centre?
- How would you extend it to new classes (house numbers, screens, ID cards)?
- How do GDPR and "right to be forgotten" requests change raw-data retention and training-set reuse? [6]
- How do you train on about 1M high-res images: data parallel vs model parallel? [6]
- Which samples go to annotators first (active learning on the uncertain band)? [6]
- How do you detect drift when new camera hardware ships?

## Portal comparison

| Portal | Framing | Model | Metrics | Distinctive points |
|---|---|---|---|---|
| ByteByteGo ch. 4 [1] | Batch privacy pipeline, 2 classes, about 1M labelled images | Two-stage (Faster R-CNN-like), compared with one-stage | IoU, precision, AP, mAP offline; user reports online | Full walk-through of NMS, augmentation, hard negatives, batch vs online prediction |
| Google ICCV 2009 (real system) [2] | Production privacy at Street View scale | Sliding-window detectors + post-processing classifier; separate plate detector | Recall of about 89% faces, 94-96% plates | Shows that recall-first operating points plus a manual-report path are the norm |
| Educative Grokking [7] | Self-driving perception (segmentation / detection) | CNN encoder-decoder, detectors | IoU, pixel accuracy | Same CV building blocks, real-time edge setting |
| alirezadir MLSD [4] | Generic template plus a CV detection cheat sheet | R-CNN family, YOLO, SSD | mAP, IoU | Concise checklists, no full case |
| Exponent / Hello Interview [8] [9] | Delivery framework applied to any prompt | Model-agnostic | Business vs model metrics | Interview pacing and communication |
| Meta EgoBlur (real system) [25] | Privacy blurring for egocentric AR data | Separate face and plate detectors | Not stated on repo page | Open-source weights; shows the same 2-class design in production |

## References

1. A. Aminian, A. Xu, "Machine Learning System Design Interview", ch. 3 "Google Street View Blurring System", ByteByteGo, 2023. Book overview: https://opencourser.com/book/xop33f/machine-learning-system-design-interview (ch-ref; ch. 3 title verified via [6])
2. A. Frome, G. Cheung, A. Abdulkader, M. Zennaro, B. Wu, A. Bissacco, H. Adam, H. Neven, L. Vincent, "Large-scale Privacy Protection in Google Street View", ICCV, 2009. https://research.google/pubs/pub35481/ (known; not re-verified in this run)
3. Google, "Street View: Privacy and policies / Report a problem", google.com, ongoing. https://www.google.com/streetview/policy/ (known; not re-verified in this run)
4. alirezadir, "ML system design" (ml-system-design.md), GitHub, 2025. https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md (verified)
5. (removed: IGotAnOffer Google MLE page did not support any claim in this chapter)
6. ByteByteGo, "ml-bytebytego: references for Machine Learning System Design Interview", GitHub, 2023. https://github.com/ByteByteGoHq/ml-bytebytego (verified)
7. Educative, "Grokking the Machine Learning Interview", educative.io, 2024. https://www.educative.io/courses/grokking-the-machine-learning-interview (ch-ref)
8. Exponent, "Introduction to the ML System Design Interview", tryexponent.com, 2025. https://www.tryexponent.com/courses/ml-system-design/mlsd-intro (ch-ref)
9. Hello Interview, "ML System Design in a Hurry: Delivery Framework", hellointerview.com, 2025. https://www.hellointerview.com/learn/ml-system-design/in-a-hurry/delivery (ch-ref)
10. P. Halina, "ML Systems Design Interview Guide", patrickhalina.com, 2021. http://patrickhalina.com/posts/ml-systems-design-interview-guide/ (ch-ref)
11. C. Huyen, "Introduction to Machine Learning Interviews" (ml-interviews-book), huyenchip.com, 2021. https://huyenchip.com/ml-interviews-book/ (ch-ref)
12. C. Huyen, "Machine Learning Systems Design: exercises", GitHub, 2019. https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/exercises.md (verified)
13. K. Pham, "machine-learning-interview", GitHub, 2021. https://github.com/khangich/machine-learning-interview (verified)
14. S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", arXiv, 2015. https://arxiv.org/abs/1506.01497 (known)
15. T.-Y. Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), arXiv, 2017. https://arxiv.org/abs/1708.02002 (ch-ref)
16. T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection", arXiv, 2016. https://arxiv.org/abs/1612.03144 (known)
17. J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", arXiv, 2015. https://arxiv.org/abs/1506.02640 (known)
18. S. Yang, P. Luo, C. C. Loy, X. Tang, "WIDER FACE: A Face Detection Benchmark", arXiv, 2015. https://arxiv.org/abs/1511.06523 (known)
19. Z. Cai, N. Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection", arXiv, 2017. https://arxiv.org/abs/1712.00726 (known)
20. J. Deng et al., "RetinaFace: Single-stage Dense Face Localisation in the Wild", arXiv, 2019. https://arxiv.org/abs/1905.00641 (known)
21. N. Carion et al., "End-to-End Object Detection with Transformers" (DETR), arXiv, 2020. https://arxiv.org/abs/2005.12872 (known)
22. Y. Li, H. Mao, R. Girshick, K. He, "Exploring Plain Vision Transformer Backbones for Object Detection" (ViTDet), arXiv, 2022. https://arxiv.org/abs/2203.16527 (known)
23. H. Rezatofighi et al., "Generalized Intersection over Union", arXiv, 2019. https://arxiv.org/abs/1902.09630 (known)
24. N. Bodla et al., "Soft-NMS: Improving Object Detection With One Line of Code", arXiv, 2017. https://arxiv.org/abs/1704.04503 (known)
25. Meta / facebookresearch, "EgoBlur: Responsible Innovation in Aria", GitHub + arXiv, 2023. https://github.com/facebookresearch/EgoBlur ; https://arxiv.org/abs/2308.13093 (verified via GitHub)
