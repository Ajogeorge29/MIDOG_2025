# MitoGuard: Context-Aware, Morphology-Preserving Joint Mitosis Detection and Atypia Classification

**Final research methodology and preregistration-ready experimental plan**  
**Version:** 2.0 | **Date:** 8 November 2026  
**Dataset ecosystem:** MIDOG 2025, MIDOG++, MIDOG 2025 Atypical Training Set, MITOS WSI collections, and provenance-verified external datasets  
**Target research outputs:** Reproducible algorithm paper, domain-generalization benchmark, and research-grade digital pathology inference prototype.

> **Status:** Proposed methodology, not a trained or validated model. Numerical improvements and performance targets are hypotheses, not achieved results. Clinical deployment or prognostic interpretation requires independent prospective validation.

## Executive summary

MitoGuard is a proposed hierarchical computer-vision system that (1) locates mitotic figures across hotspot, randomly sampled, and adverse histological regions; (2) distinguishes true mitoses from mitotic mimics; and (3) classifies accepted mitoses as normal or atypical. The principal innovation is **provenance-aware joint learning from partially observed labels**, combined with **morphology-preserving, context-gated verification**. Candidate generation uses a conventional high-recall detector; an adapted pretrained encoder supplies local morphology features and contextual information for joint verification and subtyping. A separate uncertainty calibration stage flags difficult examples for expert review.

The MIDOG 2025 challenge reports that false positive detections increased by 208% outside curated hotspots; its best reported detection F1 was 0.740 and its best subtype balanced accuracy was 0.908, for separate tracks rather than a unified pipeline [1,2]. These results motivate *end-to-end* evaluation on difficult regions and unseen biological/acquisition domains. The challenge includes 365 test cases across 12 tumor types [1,2]. None of these official results should be presented as a performance estimate for MitoGuard.

**Core questions:**

1. Does hierarchical, masked joint supervision improve end-to-end atypical-mitosis detection over an equally optimized detector-to-classifier cascade?
2. Does local-plus-context gated verification suppress false positives in necrotic, inflammatory, and randomly selected tissue without suppressing genuine mitoses in those regions?
3. Do morphology-aware contrastive training and modest stain perturbations improve subtype generalization on withheld domains?
4. Can separately calibrated verification and subtype probabilities reduce automated errors at an explicitly measured expert-review workload?

**Research design:** 28-week primary schedule, with a 24–32-week feasible range, staged compute allocation, specimen-level dataset controls, controlled ablations, and a locked final evaluation. The initial implementation prioritizes LoRA and limited activation memory; 4 × A100 80 GB is a **high-capacity planning scenario**, not a proven minimum requirement.

---

## 1. Evidence base and research gap

MIDOG 2025 introduced two evaluation tracks: detection in arbitrary tissue, including non-hotspot and difficult tissue regions, and normal/atypical mitotic figure classification [1,3]. The MIDOG++ data descriptor demonstrated generalization differences by tumor type, laboratory, and scanner and provided annotated mitoses and non-exhaustive hard-negative objects [4]. The subtype task is imbalanced, and its annotations cover a partially overlapping population of specimens, requiring careful mapping rather than naive concatenation [5,6]. A DINOv3-H+ LoRA submission provides a direct foundation-model baseline for atypia classification [7]. Earlier work established supervised contrastive objectives, parameter-efficient adaptation, group-robust optimization, multitask weighting, and calibration as plausible mechanisms [8–13], but **none of these mechanisms alone is a novel contribution to mitosis analysis**.

### Novelty boundaries

- **Not novel by itself:** YOLO-style detection, DINOv3 adaptation, LoRA, a two-stage detector/classifier, generic gating, task masking, or selective prediction.
- **Testable methodological contribution A:** a formally specified *hierarchical partial-label objective*, with detection supervision on valid regions, verification supervision on verified candidates, and subtype supervision only when labels are present; comparison to ordinary cascade and masked three-class alternatives is obligatory.
- **Testable methodological contribution B:** *morphology-safe contextual verification* using a residual gate, deliberate context dropout, and a measurement of false-positive reduction at matched sensitivity; specifically test whether context creates harmful shortcuts.
- **Potential contribution C:** rigorously measured end-to-end atypical detection across adverse regions and unseen domains, including audit of incomplete annotation and specimen overlap.
- **Optional contributions:** supervised contrastive learning, domain-robust optimization, and selective review. Retain only if preregistered ablations show incremental benefit.

No claim of first-ever combined detection and subtyping should be made without a current, targeted prior-art review before submission.

## 2. Datasets, annotation semantics, and provenance registry

| Source | Verified description | Proposed role | Crucial caveat |
|---|---|---|---|
| MIDOG++ | 503 specimens, 11,937 annotated mitoses, seven tumor types [4] | Detection, candidate verification, domain splits | Hard negatives are non-exhaustively labeled [4] |
| MIDOG 2025 Atypical Training Set | 11,939 classified mitoses, derived from MIDOG++; current version corrects one duplicated and one omitted CSV entry [5,6] | Supervised atypia classification | Reconcile object identities and class labels before joining |
| MIDOG 2025 Track 2 challenge page | 10,191 normal and 1,748 atypical objects; 454 labeled images in nine defined domains [5] | Class balance and domain descriptions | Do not assume every detection annotation has an exact subtype match |
| AMi-Br | 3,720 labeled figures, including 832 atypical [3] | Potential additional subtype training | Overlaps with MIDOG 2021 source cases [3,5] |
| MITOS WSI CMC | 21 whole slides, 13,907 mitosis annotations [3] | Whole-slide-context detection | Distinct annotation conventions, species, and acquisition conditions |
| MITOS WSI CCMCT | 32 whole slides, 44,880 mitosis annotations [3] | Challenging WSI detection | Annotation-validity masks and domain effects must be audited |
| Other external labeled datasets | Eligible after verification | Final external validation or supplementary training | Must document case overlap, licenses, and label ontology |

**Registry schema (one row per object/candidate, with an accompanying image/region table):** `source_dataset`, `version`, `license`, `specimen_id`, `slide_id`, `roi_id`, `image_hash`, `perceptual_hash`, `domain_id`, `species`, `tumor_type`, `laboratory`, `scanner`, `mpp_x`, `mpp_y`, `coordinate_reference`, `x`, `y`, `object_id`, `detection_label`, `subtype_label`, `detection_label_mask`, `subtype_label_mask`, `region_completely_reviewed`, `reviewer_consensus`, `annotation_protocol`, `source_annotation_id`, `partition`.

**Mandatory processing and release checks:**

1. Download and pin exact dataset releases, especially the corrected subtype CSV [6]; record checksums.
2. Reconcile 11,937 detection versus 11,939 subtype totals at the **object ID level**. The numerical discrepancy cannot be resolved by guessing.
3. Normalize coordinate conventions (image-level versus WSI-level, magnification, offsets, XY orientation, pixel and micrometer units); visually inspect random mapped objects in every domain.
4. Deduplicate by source identities, exact checksums, and perceptual/embedding similarity with manual adjudication. Ensure the known AMi-Br/MIDOG 2021 overlap never crosses train/test boundaries [3,5].
5. Build binary *region validity masks* identifying fully reviewed versus incomplete/unreviewed regions. An unmatched prediction in an incompletely annotated region **must not** automatically become a negative training label.
6. Assign splits at specimen level, and where applicable at patient/slide level; all derived crops and duplicated releases inherit the parent assignment.
7. Maintain a frozen data manifest and separate audit report with included/excluded samples and exact label mappings.

### Split hierarchy

- **Development split:** specimen-grouped training, validation, and calibration, avoiding cross-specimen leakage.
- **Biological holdouts:** leave-one-tumor-type-out where sufficient training domains remain; additionally report held-out species if design permits.
- **Acquisition holdouts:** scanner/laboratory combinations only when identifiable independently from tumor type and species; otherwise label findings as *compound-domain generalization*.
- **Final test:** untouched external cases or an accessible official evaluation, with model and operating point locked in advance. Original MIDOG challenge scores are contextual references, **not** necessarily directly comparable to a new test split.

## 3. System formulation and architecture

For a tile or WSI region, a high-recall proposal detector $D_\phi$ returns candidate centers $c_i=(x_i,y_i)$ and proposal scores $s_i$. For each proposal, extract (a) a local morphology crop $x_i^{L}$ and (b) a wider aligned tissue-context crop $x_i^{C}$. Physical field of view is harmonized using microns-per-pixel metadata, not only resized pixel dimensions.

**Recommended prototype components:**

1. **Proposal generator:** YOLO-family or another competitive detection architecture, trained on verified detection regions. Benchmark an alternative detector if compute permits. The proposal stage maximizes sensitivity without relying on subtype labels.
2. **Shared image representation:** pretrained DINOv3 adapted with LoRA [7,9], with either shared encoder passes for local/context crops or a lightweight contextual branch. Compare with a pathology-pretrained alternative using identical training/evaluation data.
3. **Context fusion:** residual gate to protect local morphology features.
4. **Verification head:** $p_M=P(M=1\mid x)$, where $M$ denotes a true mitotic figure.
5. **Subtype head:** $p_A=P(A=1\mid M=1,x)$, where $A$ denotes atypical mitosis. The subtype head should preferentially use local morphology; contextual information must be constrained and ablated.
6. **Post hoc calibration and selective review:** calibrated $p_M$ and $p_A$, plus an explicit uncertain/review state.

Let $z_L=f_\theta(x^L)$ and $z_C=f_\theta(x^C)$, with projected dimensions aligned. Define:

$$g=\sigma(W_g[z_L;z_C]+b_g), \qquad z_V=z_L+g\odot W_Cz_C.$$

Here $z_V$ feeds verification. For subtyping, use $z_A=z_L$ as the **default morphology-safe choice**; a contextual subtype variant is an ablation, not a presumed improvement. The end-to-end class probabilities are:

$$P(B)=1-p_M,\qquad P(N)=p_M(1-p_A),\qquad P(A)=p_Mp_A,$$

where $B$ is background/imposter, $N$ is normal mitosis, and $A$ is atypical mitosis. These form a valid distribution if $p_M,p_A\in[0,1]$.

**Critical clarification on gradient flow:** The *product used at inference* does **not** imply that subtype training gradients flow through $p_M$. If the optimization uses $\mathcal L_{\rm subtype}=-\log P(A\mid M=1,x)$ on subtype-labeled true mitoses, and the subtype head does not consume $p_M$, then the subtype term does not directly differentiate through the verification logit. A naïve end-to-end three-class loss $-\log[p_Mp_A]$ on atypical objects *does* include a verification gradient; this is not inherently incorrect, but its contribution must be controlled. Avoid applying a hard $p_M$ confidence threshold to training-time subtype annotations, since it can systematically discard hard true mitoses and reinforce errors.

## 4. Partial-label hierarchical learning

Define object indicators $m_i^{D},m_i^{V},m_i^{A}\in\{0,1\}$, indicating that valid detector, verification, and subtype supervision respectively exists. For known positive mitotic objects with subtype labels, the subtype loss is conditional on true mitosis, not conditioned on whether the current detector is confident.

$$\mathcal L_V=\frac{\sum_i m_i^{V}\,\mathrm{BCE}(p_{M,i},y_{M,i})}{\max(1,\sum_i m_i^{V})}.$$

$$\mathcal L_A=\frac{\sum_i m_i^{A}\,\mathrm{CE}(p_{A,i},y_{A,i})}{\max(1,\sum_i m_i^{A})}.$$

The detector loss $\mathcal L_D$ is computed only on valid annotated regions with a consistent localization protocol. For truly verified subtype-labeled positives, a hierarchical joint negative log likelihood could equivalently include $-\log p_M -\log P(y_A\mid M=1,x)$, but this coupled alternative must be explicitly compared to the separately normalized loss formulation.

The primary objective is:

$$\mathcal L_{\rm core}=\lambda_D\mathcal L_D+\lambda_V\mathcal L_V+\lambda_A\mathcal L_A.$$

Additional optional terms are activated only after baselines are established:

$$\mathcal L_{\rm extended}=\mathcal L_{\rm core}+\lambda_C\mathcal L_{\rm SupCon}+\lambda_S\mathcal L_{\rm stain-consistency}+\lambda_G\mathcal L_{\rm group-robust}.$$

**Optimization safeguards:**

- Warm up detector, verifier, and subtype head on their own valid labels before shared joint updates.
- Alternate detection/candidate and subtype minibatches, balancing **supervision counts**, not simply dataset file counts.
- Normalize losses by valid-label counts; monitor per-task gradient norm, loss scale, class recall, and cross-task interference.
- Compare (a) shared gradient throughout, (b) stop-gradient from subtype to the shared trunk for a warm-up interval, and (c) separate task adapters. Do **not** assume stop-gradient is intrinsically best: full isolation can remove the principal benefit of multitask representation learning.
- If gradients conflict, test an explicit gradient-balancing approach only after a documented baseline; task uncertainty weighting is an optional comparator [10].
- When positive subtype labels exist but $p_M$ is small, retain conditional subtype supervision. Do not mask subtype gradients based solely on predicted confidence.

**Primary scientific comparison:** masked conditional hierarchy versus naive cascade versus masked flat three-class head, with equal encoder capacity, data, tuning budget, and compute where feasible.

## 5. Context-aware verification and hard-negative policy

**Hypothesis H1:** A residual local/context verifier with context dropout improves difficult-region false-positive control at matched genuine-mitosis sensitivity.

Start with approximately 128-pixel local crops only when physical scale is compatible with official source conventions; test physically aligned wider crops (e.g., 256 or 384 pixels after scale normalization). Context may help distinguish apoptotic cells, inflammatory cells, necrotic debris, and stain artifacts, but could also produce an erroneous rule that mitoses cannot occur in those tissues [1,2,4].

**Context controls:**

- Local-only, context-only, concatenation, residual gating, and residual gating with context dropout.
- Randomly suppress the context branch on a preset fraction of training objects; sweep this fraction on validation only.
- Evaluate real positive mitoses within adverse regions separately from imposter rejection.
- Evaluate swaps or perturbations of context around a fixed local crop as a **diagnostic shortcut test**, not automatically as biological ground truth.

**Hard-negative mining:** run the frozen candidate detector over training regions with reliable annotation coverage, match proposals to ground truth, review or protocol-filter unmatched candidates, and then sample a controlled ratio of difficult and ordinary negatives. Do not label unmatched predictions in unreviewed tissue as false positives. Track mining source, iteration, candidate confidence, tissue context, and expert adjudication. Compare no mining versus ordinary mining versus context-stratified mining using equal budgets.

**Success target (hypothesis, not established outcome):** at least 15% relative decrease in false positives in adverse regions at the same overall recall, without a material decrease in adverse-region mitosis recall. Predefine the clinically acceptable sensitivity margin before test evaluation.

## 6. Morphology learning and robustness

**Hypothesis H2:** Local morphology embeddings with class-aware sampling and supervised contrastive learning outperform an equally tuned weighted-cross-entropy subtype model on held-out domains [8].

Atypical figures are a minority class in the official Track 2 training set [5]. Begin with weighted cross entropy, then compare focal loss and supervised contrastive loss. For memory-efficient contrastive learning, use (1) gradient accumulation for effective classification batch size and (2) a **detached, momentum-updated feature queue or class prototype bank** as an optional contrastive alternative; note that accumulation does not by itself create cross-microbatch in-batch contrastive pairs. Validate that stored negatives and positives do not leak held-out specimens, and assess queue staleness.

$$\mathcal L_{\rm morph}=\mathcal L_A+\lambda_C\mathcal L_{\rm SupCon}.$$

**Hypothesis H3:** Physically plausible stain perturbations and source-balanced sampling improve the worst held-out domain performance [4,11].

Use rotations and flips, constrained optical-density/stain-matrix perturbation, and low-severity acquisition noise. Aggressive geometric deformation, arbitrary cut-and-paste, and generative atypical figures should be excluded from the primary protocol because they can change the target morphology. Compare ERM, balanced source sampling, bounded stain augmentation, and regularized group DRO independently. Treat domain as a *documented* grouping, not a synthetic label inferred from image color.

**Key failure test:** improvements in overall F1 or balanced accuracy must not mask losses of atypical sensitivity or genuine mitosis detection in challenging/rare domains.

## 7. Confidence calibration and pathologist review

Use a separate specimen-disjoint calibration partition. Fit simple temperature scaling first [12]; inspect reliability diagrams, Brier score, expected calibration error, and subgroup calibration. Route candidates to review when verification/subtype confidence or risk-control criteria warrant it.

Report selective risk versus coverage, **recall of all ground truth atypical figures after abstentions**, review referrals per square millimeter, and accuracy of accepted predictions. Rejection must never be presented as a correct classification. Conformal risk control [13] is optional and its nominal guarantees must be qualified under unseen domain shift.

## 8. Hardware and memory-efficient execution plan

Compute feasibility depends strongly on selected DINOv3 variant, image resolution, microbatch size, whether both local/context crops backpropagate through the encoder, precision, and the detector's training schedule. No fixed GPU count guarantees success.

| Tier | Planned resources | Workloads | Limitations |
|---|---|---|---|
| Minimum exploratory | One 24 GB GPU; LoRA; mixed precision; checkpointing; microbatch 1–4 | Dataset audit, compact detector, subtype baseline, local-only head | Large joint dual-crop model and extensive search may be impractical |
| Recommended research | One or two 48–80 GB GPUs, or scheduled access to equivalent | Residual context fusion, controlled joint training, three seeds for selected variants | Model choice and batch size still must be profiled |
| High-capacity accelerated | 4 × A100 80 GB or comparable multi-GPU allocation | Parallel baselines, domain folds, multiple backbones, longer ablation suite | Higher cost and distributed-training overhead; not proven necessary |

**Memory-control commitments from day one:** LoRA, bf16 where supported, mixed precision for detector, frozen base weights where possible, activation checkpointing, gradient accumulation, bounded crop resolution, and separate proposal generation from joint verifier training. Cache candidate coordinates and optionally frozen features only when the experiment does not require gradient propagation through those features. Reuse the same local encoding across heads; investigate lightweight context encoding rather than requiring two large encoder passes. Measure real peak memory and examples/second with a representative batch before selecting the final configuration.

**Compute budgeting protocol:** record GPU model, memory, software environment, hours per training stage, number of candidate proposals, per-fold epochs, trials/seeds, and storage/network costs. Establish a pilot-run cost table before full sweeps. Do not invent a dollar cost or assert a required GPU fleet without benchmarking.

## 9. Training procedure and acceptance gates

1. **Gate A — data audit:** label reconciliation, no known split leakage, verified coordinate overlays, valid-area masks, frozen split manifest.
2. **Gate B — independent baselines:** detector-only, subtype-only, and conventional detector→subtype cascade with confidence thresholds chosen on validation.
3. **Gate C — joint baseline:** masked hierarchical verification/subtype system, comparing full shared versus partially isolated subtype gradient updates.
4. **Gate D — context:** local/context gating plus context dropout and hard-negative mining; demonstrate unchanged/sufficient recall on true mitoses within challenging tissue.
5. **Gate E — optional improvements:** contrastive morphology, domain-balanced optimization, and stain robustness, each retained only if matched-budget ablations show benefit.
6. **Gate F — calibration and locked evaluation:** separate calibration, threshold freeze, all-domain end-to-end evaluation, uncertainty intervals, and error audit.

**Suggested initial tuning ranges (not verified optima):** AdamW; pretrained encoder LoRA LR $10^{-5}$–$5\times10^{-5}$, new heads LR $10^{-4}$–$3\times10^{-4}$; LoRA rank 8/16; 50–100 epochs maximum with early stopping; three random seeds for key comparisons. Choose crop sizes in physical units and tune the detector's threshold for high-recall candidate generation using development data.

## 10. Evaluation design, metrics, and statistical analysis

**Detection:** F1, precision, recall, FROC and average precision under official or explicitly stated spatial matching tolerance; report challenging/hotspot/random regions independently, including FPs per reviewed area (mm²). Evaluation on partially annotated regions requires separate verification or exclusion from specificity calculations.

**Subtype:** balanced accuracy, atypical sensitivity/specificity, confusion matrix, ROC AUC and PR AUC on *ground-truth-positive crops*. These are **oracle localization** subtype metrics, separate from pipeline scores.

**End-to-end atypical detection:** count a correct AMF only if it spatially matches a ground truth AMF **and** is predicted atypical:

$$R_A^{\rm joint}=\frac{TP_{\rm matched,atypical}}{\#\mathrm{ground\ truth\ atypical\ figures}},\qquad P_A^{\rm joint}=\frac{TP_{\rm matched,atypical}}{\#\mathrm{all\ predicted\ atypical\ candidates}}.$$

Use $F1_A^{\rm joint}=2P_A^{\rm joint}R_A^{\rm joint}/(P_A^{\rm joint}+R_A^{\rm joint})$ when denominator is positive. Also report corresponding normal-mitosis joint metrics, subtype-conditioned detector recall, unmatched detections and unmatched truth, and a spatially matched three-class error breakdown.

**Robustness metrics:** worst observed domain F1; worst-domain atypical recall; average and median domain scores; detection false-positive gap from hotspot to challenging tissue; source-to-target performance change. Always include sample sizes and intervals.

**Statistics:** specimen-level paired bootstrap confidence intervals (e.g., 2,000 resamples where feasible); repeated seeds; paired contrasts on identical test specimens; predeclared primary hypothesis family with multiplicity control; sensitivity analyses that exclude low-confidence/disputed annotations. Report performance by species, tumor type, scanner/laboratory, and region where sample sizes permit. Never treat thousands of patches from the same WSI as independent subjects.

**Preregistered, aspirational targets:** (i) ≥15% relative reduction in difficult-region FPs at matched recall; (ii) ≥2 percentage points subtype balanced accuracy on held-out domains; (iii) ≥3 percentage points joint AMF F1 versus equal-capacity cascade. These are planning targets, not promises or venue acceptance criteria.

## 11. Required ablations

| ID | Comparison | Essential inference |
|---|---|---|
| A0 | Detector-only and subtype-only oracle baselines | Independent task capability |
| A1 | Conventional cascade | Strong integrated reference |
| A2 | Masked hierarchical joint versus cascade | Actual value of joint representation learning |
| A3 | Masked hierarchical versus flat three-class learning | Value of label semantics and hierarchy |
| A4 | Full gradients versus warm-up isolation versus task adapters | Gradient interference and stability |
| A5 | Local-only versus context-only versus concatenation versus residual gate | Value and risks of context |
| A6 | Residual gate with and without context dropout | Shortcut suppression |
| A7 | No/ordinary/context-stratified hard-negative mining | Specificity gains |
| A8 | Weighted cross entropy versus contrastive auxiliary loss | Morphology embedding benefit |
| A9 | ERM versus stain augmentation versus balanced sampling versus group DRO | Domain robustness effect |
| A10 | DINOv3 versus pathology-pretrained encoder at equal budget | Backbone dependence |
| A11 | No versus post hoc calibration and review | Selective-risk benefit |
| A12 | Matched-capacity and matched-compute comparison | Rules out benefits due purely to increased model size/training |

Predeclare a small primary ablation set (A1–A6, A12) to protect the paper from uncontrolled exploratory optimization; the rest are secondary. Include tests for dependence between improvements where budget allows.

## 12. Timeline: 28 weeks (adjustable within 24–32 weeks)

| Weeks | Primary work | Exit deliverable |
|---|---|---|
| 1–3 | Download, licenses, checksums, version pinning, annotation schema | Dataset manifest v1 |
| 4–6 | Coordinate reconciliation, deduplication, domain metadata, specimen splits | Audited canonical registry; leakage report |
| 7–8 | Physical scale QC, baseline data loaders and sample overlays | Frozen development protocol |
| 9–12 | Detector-only and subtype-only baselines; conventional cascade | Baseline table and compute benchmark |
| 13–16 | Hierarchical masked objective; shared encoder; gradient stability comparisons | Verified joint baseline |
| 17–20 | Gated context, context dropout, hard negative training | Core contribution ablations |
| 21–23 | Morphology/robustness experiments, selected backbone check | Domain generalization results |
| 24–25 | Calibration, threshold locking, final independent evaluation | Locked evaluation report |
| 26–28 | Error analysis, figure preparation, reproducibility package, manuscript | Submission-ready research package |

**Contingency:** reserve up to four additional weeks for label inconsistencies, expert review, external-data access or GPU availability. A small team with limited infrastructure should prioritize baseline, masked hierarchy, and context experiments rather than attempting every optional addon.

## 13. Reproducibility and research software

Repository components: `data_manifest/`, `annotation_harmonization/`, `splits/`, `detector/`, `joint_model/`, `training/`, `calibration/`, `evaluation/`, `ablation_configs/`, `viewer_export/`, `environment/`, and `reports/`. Track code commit, dataset version, config checksum, random seed, GPU identifier, training budget, optimizer state, and evaluation scripts. Export prediction coordinates, normal/atypical probabilities, unmatched-candidate flags, calibration scores, and tissue-area coverage into a viewer-compatible JSON/CSV representation. An offline Docker container is desirable for reproducibility, based on the official MIDOG guidance [14].

**Clinical boundary:** this is a *research aid*, not autonomous tumor grading or prognostic prediction. AMF frequency and clinical significance must be validated within each applicable tumor type and grading protocol before any clinical claim.

## 14. Decision rules for the eventual manuscript

- Claim **algorithmic novelty** only for components shown by matched comparisons to exceed established alternatives.
- Report if context-gating reduces FPs but also misses true mitoses; do not suppress negative findings.
- If hierarchical masked training does not outperform the equal-capacity cascade, pivot to a rigorously characterized robustness and annotation methodology paper rather than claiming novel superior multitask learning.
- Present positive performance under held-out domains separately from in-domain performance.
- Avoid numerical comparisons to original official challenge leaderboard unless data split, evaluation unit, matching and access are demonstrably comparable.
- Prioritize one main scientific thesis: **reliable joint localization and normal/atypical typing under sparse labels and adverse tissue context**.

## 15. References and source verification

References below link to publicly identifiable publisher pages, official challenge materials or arXiv. Some are preprints and should be labeled accordingly in manuscript references. Review and update the literature search again at submission.

1. Aubreville M., et al. **Mitosis Detection in the Wild: Multi-Tumor and Context-Aware Generalization in the MIDOG 2025 Challenge**. arXiv, 2026. https://arxiv.org/abs/2606.07368
2. DeepMicroscopy. **Insights from the MIDOG 2025 Challenge Paper**. 8 June 2026. https://deepmicroscopy.org/insights-from-the-midog-2025-challenge-paper/
3. MIDOG 2025 organizers. **Official datasets and challenge resources**. https://midog.deepmicroscopy.org/2025/datasets
4. Aubreville M., Wilm F., Stathonikos N., et al. **A comprehensive multi-domain dataset for mitotic figure detection**. *Scientific Data*. 2023;10:484. https://doi.org/10.1038/s41597-023-02327-4
5. MIDOG 2025 organizers. **Track 2: Atypical Classification**, dataset and evaluation details. https://midog2025.deepmicroscopy.org/track-2-atypical-classification/
6. Weiss V., et al. **A dataset of atypical vs normal mitoses classification for MIDOG – 2025**, corrected Zenodo release. https://zenodo.org/records/16044804
7. Balezo G., Bourgade R., Walter T. **Efficient Fine-Tuning of DINOv3 Pretrained on Natural Images for Atypical Mitotic Figure Classification in MIDOG 2025**. arXiv, 2025. https://arxiv.org/abs/2508.21041
8. Khosla P., et al. **Supervised Contrastive Learning**. *NeurIPS*. 2020. https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
9. Hu E.J., et al. **LoRA: Low-Rank Adaptation of Large Language Models**. *ICLR*. 2022. https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/
10. Kendall A., Gal Y., Cipolla R. **Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics**. *CVPR*. 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
11. Sagawa S., Koh P.W., Hashimoto T.B., Liang P. **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**. *ICLR*. 2020. https://arxiv.org/abs/1911.08731
12. Guo C., Pleiss G., Sun Y., Weinberger K.Q. **On Calibration of Modern Neural Networks**. *ICML*. 2017. https://proceedings.mlr.press/v70/guo17a.html
13. Angelopoulos A.N., et al. **Conformal Risk Control**. *ICLR*. 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
14. DeepMicroscopy. **MIDOG 2025 Guide: official notebooks and baseline implementations**. https://github.com/DeepMicroscopy/MIDOG_2025_Guide
15. MIDOG 2025 organizers. **Challenge overview and track description**. https://midog2025.deepmicroscopy.org/challenge-goals/

---

### Amendment log from external review

1. **Compute:** replaced the unsupported 24 GB large-model feasibility assumption with a tiered hardware plan, mandatory LoRA/checkpointing, profiling gates, and memory-efficient contrastive options.
2. **Gradient dynamics:** separated conditional subtype supervision from hierarchical inference probabilities; clarified that subtype losses do not intrinsically backpropagate through $p_M$, and made warm-up/stop-gradient a comparison rather than a mandatory threshold gate.
3. **Schedule:** expanded 16 weeks to a baseline 28-week schedule with 24–32-week planning range and six weeks for provenance harmonization.
4. **Novelty:** shifted emphasis from backbone selection to partial-label hierarchy, context-safe false-positive rejection, and end-to-end robustness.
5. **Dataset quality:** required the corrected Zenodo subtype release and explicit object-ID reconciliation, plus specimen-level deduplication and validity masks.
6. **Evidence standard:** separated published findings, methodological precedents, and unvalidated research hypotheses; retained no acceptance or guaranteed performance claims.
